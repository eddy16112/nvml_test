/*
 * MPI GPU + CPU Topology Collector
 *
 * Phase 1A: GPU collection (NVML + CUDA)
 * Phase 1B: CPU collection (hwloc, NUMA granularity, follows process mapping)
 * Phase 2:  MPI_Allgather exchange
 * Phase 3:  Build per-rank topology maps
 * Phase 4:  Print global topology
 *
 * Two modes controlled by PHASE3_USE_NVML:
 *   - Not defined (default): Phase 3 uses only pre-collected data
 *   - Defined:               Phase 3 may call NVML for cross-rank PCIe topology
 *
 * Build (with cmake):
 *   cmake -B build && cmake --build build
 *
 * Run:
 *   mpirun -np <N> ./build/gpu_topo
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <unistd.h>

#include <mpi.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <hwloc.h>

static constexpr int MAX_GPUS       = 16;
static constexpr int MAX_TOPO_NODES = 32;
static constexpr int MAX_LINKS      = 18;
static constexpr int BUSID_SZ       = 32;
static constexpr int UUID_SZ        = 96;
static constexpr int NAME_SZ        = 256;
static constexpr int HOST_SZ        = 256;

static int gRank = 0;

#define CHK_CUDA(call) do {                                        \
    cudaError_t e_ = (call);                                       \
    if (e_ != cudaSuccess) {                                       \
        fprintf(stderr, "[R%d] CUDA %s:%d – %s\n",                \
                gRank, __FILE__, __LINE__, cudaGetErrorString(e_));\
        MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                              \
} while (0)

#define CHK_NVML(call) do {                                        \
    nvmlReturn_t r_ = (call);                                      \
    if (r_ != NVML_SUCCESS) {                                      \
        fprintf(stderr, "[R%d] NVML %s:%d – %s\n",                \
                gRank, __FILE__, __LINE__, nvmlErrorString(r_));   \
        MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                              \
} while (0)

//#define PHASE3_USE_NVML

/* ==================================================================
 *  POD structures — safe for MPI_Allgather as MPI_BYTE
 * ================================================================== */

struct NvLinkPeer {
    char remoteBusId[BUSID_SZ];
    int  active;
};

#ifndef PHASE3_USE_NVML
struct PeerTopo {
    char busId[BUSID_SZ];
    int  pcieTopo;
};
#endif

struct GpuInfo {
    int      deviceId;
    char     uuid[UUID_SZ];
    char     busId[BUSID_SZ];
    char     name[NAME_SZ];
    int      ccMajor, ccMinor;
    uint64_t memMB;
    int      pcieGen, pcieWidth;
    int      numaId;
    int      nNvLinks;
    NvLinkPeer nvLinks[MAX_LINKS];
#ifndef PHASE3_USE_NVML
    int      nPeerTopos;
    PeerTopo peerTopos[MAX_GPUS];
#endif
};

struct CpuInfo {
    int numaId;
    int nCores;
};

enum HandleType {
    GPU_HANDLE = 0,
    CPU_HANDLE = 1,
};

struct Handle {
    int rank;
    union {
        struct { int deviceId; } gpu;
        struct { int numaId;  } cpu;
    };
    HandleType type;
};

inline bool operator<(const Handle& a, const Handle& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    if (a.type != b.type) return a.type < b.type;
    if (a.type == GPU_HANDLE) return a.gpu.deviceId < b.gpu.deviceId;
    return a.cpu.numaId < b.cpu.numaId;
}

inline bool operator==(const Handle& a, const Handle& b) {
    if (a.rank != b.rank || a.type != b.type) return false;
    if (a.type == GPU_HANDLE) return a.gpu.deviceId == b.gpu.deviceId;
    return a.cpu.numaId == b.cpu.numaId;
}

struct TopologyNode {
    Handle handle;
    union {
        GpuInfo gpu;
        CpuInfo cpu;
    };
};

struct RankData {
    char           hostname[HOST_SZ];
    int            rank;
    int            nTopologyNodes;
    TopologyNode   nodes[MAX_TOPO_NODES];
};

/* ==================================================================
 *  RankManager – owns RankData + per-rank topology map
 * ================================================================== */

typedef std::pair<Handle, Handle> HandlePair;

static HandlePair canonicalPair(const Handle& a, const Handle& b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

struct RankManager {
    RankData data;
    std::map<HandlePair, std::string> topology;

    void buildTopology(const std::vector<RankData>& allRanks);

    std::string query(const Handle& a, const Handle& b) const {
        auto it = topology.find(canonicalPair(a, b));
        return (it != topology.end()) ? it->second : "";
    }
};

/* ==================================================================
 *  Helpers
 * ================================================================== */

static std::string handleStr(const Handle& h) {
    if (h.type == GPU_HANDLE)
        return "GPU(" + std::to_string(h.rank) + "," + std::to_string(h.gpu.deviceId) + ")";
    return "CPU(" + std::to_string(h.rank) + "," + std::to_string(h.cpu.numaId) + ")";
}

static std::string busKey(const char* id) {
    std::string s(id);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    auto p = s.find(':');
    return (p != std::string::npos) ? s.substr(p + 1) : s;
}

static bool sameBus(const char* a, const char* b) {
    return busKey(a) == busKey(b);
}

static const char* topoTag(int t) {
    switch (t) {
        case 0:  return "X";
        case 10: return "PIX";
        case 20: return "PXB";
        case 30: return "PHB";
        case 40: return "NODE";
        case 50: return "SYS";
        default: return "?";
    }
}

static int countNvLinksByBusId(const GpuInfo& gi, const char* peerBusId) {
    int n = 0;
    for (int k = 0; k < gi.nNvLinks; k++)
        if (gi.nvLinks[k].active && sameBus(gi.nvLinks[k].remoteBusId, peerBusId))
            n++;
    return n;
}

static int countNvSwitchLinks(const GpuInfo& gi, const GpuInfo& gj,
                              const std::vector<RankData>& allRanks,
                              const std::string& host) {
    auto isKnownGpu = [&](const char* bid) -> bool {
        for (auto& R : allRanks) {
            if (std::string(R.hostname) != host) continue;
            for (int g = 0; g < R.nTopologyNodes; g++)
                if (R.nodes[g].handle.type == GPU_HANDLE &&
                    sameBus(bid, R.nodes[g].gpu.busId))
                    return true;
        }
        return false;
    };

    std::map<std::string, int> swI, swJ;
    for (int k = 0; k < gi.nNvLinks; k++)
        if (gi.nvLinks[k].active && !isKnownGpu(gi.nvLinks[k].remoteBusId))
            swI[busKey(gi.nvLinks[k].remoteBusId)]++;
    for (int k = 0; k < gj.nNvLinks; k++)
        if (gj.nvLinks[k].active && !isKnownGpu(gj.nvLinks[k].remoteBusId))
            swJ[busKey(gj.nvLinks[k].remoteBusId)]++;

    int n = 0;
    for (auto& kv : swI)
        if (swJ.count(kv.first)) n += kv.second;
    return n;
}

#ifdef PHASE3_USE_NVML
static int queryPcieTopo(const char* busIdA, const char* busIdB) {
    nvmlDevice_t a, b;
    if (nvmlDeviceGetHandleByPciBusId_v2(busIdA, &a) != NVML_SUCCESS) return -1;
    if (nvmlDeviceGetHandleByPciBusId_v2(busIdB, &b) != NVML_SUCCESS) return -1;
    nvmlGpuTopologyLevel_t lvl;
    if (nvmlDeviceGetTopologyCommonAncestor(a, b, &lvl) != NVML_SUCCESS) return -1;
    return (int)lvl;
}
#endif

static std::string resolvePcie(const GpuInfo& gi, const GpuInfo& gj) {
#ifdef PHASE3_USE_NVML
    return topoTag(queryPcieTopo(gi.busId, gj.busId));
#else
    int pt = -1;
    for (int k = 0; k < gi.nPeerTopos && pt < 0; k++)
        if (sameBus(gi.peerTopos[k].busId, gj.busId))
            pt = gi.peerTopos[k].pcieTopo;
    for (int k = 0; k < gj.nPeerTopos && pt < 0; k++)
        if (sameBus(gj.peerTopos[k].busId, gi.busId))
            pt = gj.peerTopos[k].pcieTopo;
    return topoTag(pt);
#endif
}

/* --- GPU <-> GPU connection --- */
static std::string resolveGpuGpu(
        const GpuInfo& gi, const GpuInfo& gj,
        bool sameNode,
        const std::vector<RankData>& allRanks,
        const std::string& hostname) {

    if (sameBus(gi.busId, gj.busId))
        return "X";

    if (!sameNode)
        return "NET";

    int nvl = countNvLinksByBusId(gi, gj.busId);
    if (nvl > 0)
        return "NV" + std::to_string(nvl);

    int nvs = countNvSwitchLinks(gi, gj, allRanks, hostname);
    if (nvs > 0)
        return "NV" + std::to_string(nvs);

    return resolvePcie(gi, gj);
}

/* --- GPU <-> CPU connection --- */
static std::string resolveGpuCpu(const GpuInfo& g, const CpuInfo& c,
                                 bool sameNode) {
    if (!sameNode)
        return "NET";
    if (g.numaId >= 0 && g.numaId == c.numaId)
        return "NODE";
    return "SYS";
}

/* --- CPU <-> CPU connection --- */
static std::string resolveCpuCpu(const CpuInfo& ci, const CpuInfo& cj,
                                 bool sameNode) {
    if (ci.numaId == cj.numaId && sameNode)
        return "X";
    if (!sameNode)
        return "NET";
    return "SYS";
}

/* --- Dispatcher: resolve connection between any two TopologyNodes --- */
static std::string resolveNodeConnection(
        const TopologyNode& src, const TopologyNode& dst,
        bool sameNode,
        const std::vector<RankData>& allRanks,
        const std::string& hostname) {

    HandleType st = src.handle.type;
    HandleType dt = dst.handle.type;

    if (st == GPU_HANDLE && dt == GPU_HANDLE)
        return resolveGpuGpu(src.gpu, dst.gpu, sameNode, allRanks, hostname);

    if (st == GPU_HANDLE && dt == CPU_HANDLE)
        return resolveGpuCpu(src.gpu, dst.cpu, sameNode);

    if (st == CPU_HANDLE && dt == GPU_HANDLE)
        return resolveGpuCpu(dst.gpu, src.cpu, sameNode);

    return resolveCpuCpu(src.cpu, dst.cpu, sameNode);
}

/* ==================================================================
 *  Phase 1A – GPU collection  (NVML + CUDA)
 * ================================================================== */

static void collectLocalGpus(RankData& D) {
    CHK_NVML(nvmlInit_v2());

    unsigned int nAllDev = 0;
    CHK_NVML(nvmlDeviceGetCount_v2(&nAllDev));
    int nAll = std::min((int)nAllDev, MAX_GPUS);

    nvmlDevice_t hAll[MAX_GPUS];
    char allBusIds[MAX_GPUS][BUSID_SZ] = {};
    for (int i = 0; i < nAll; i++) {
        CHK_NVML(nvmlDeviceGetHandleByIndex_v2(i, &hAll[i]));
        nvmlPciInfo_t pci;
        CHK_NVML(nvmlDeviceGetPciInfo_v3(hAll[i], &pci));
        strncpy(allBusIds[i], pci.busId, BUSID_SZ - 1);
    }

    int nCuda = 0;
    CHK_CUDA(cudaGetDeviceCount(&nCuda));
    int nGpus = std::min(nCuda, MAX_GPUS);

    for (int ci = 0; ci < nGpus; ci++) {
        int idx = D.nTopologyNodes;
        if (idx >= MAX_TOPO_NODES) break;

        D.nodes[idx].handle.rank = D.rank;
        D.nodes[idx].handle.type = GPU_HANDLE;
        D.nodes[idx].handle.gpu.deviceId = ci;
        GpuInfo& G = D.nodes[idx].gpu;
        memset(&G, 0, sizeof(G));
        G.numaId = -1;

        char cbid[BUSID_SZ] = {};
        CHK_CUDA(cudaDeviceGetPCIBusId(cbid, BUSID_SZ, ci));
        cudaDeviceProp prop;
        CHK_CUDA(cudaGetDeviceProperties(&prop, ci));
        G.deviceId = ci;
        G.ccMajor = prop.major;
        G.ccMinor = prop.minor;

        for (int k = 0; k < nAll; k++) {
            if (!sameBus(cbid, allBusIds[k])) continue;
            nvmlDevice_t hDev = hAll[k];

            CHK_NVML(nvmlDeviceGetUUID(hDev, G.uuid, UUID_SZ));
            strncpy(G.busId, allBusIds[k], BUSID_SZ - 1);
            CHK_NVML(nvmlDeviceGetName(hDev, G.name, NAME_SZ));

            nvmlMemory_t mem;
            CHK_NVML(nvmlDeviceGetMemoryInfo(hDev, &mem));
            G.memMB = mem.total / (1024ULL * 1024);

            unsigned int v = 0;
            if (nvmlDeviceGetCurrPcieLinkGeneration(hDev, &v) == NVML_SUCCESS)
                G.pcieGen = (int)v;
            v = 0;
            if (nvmlDeviceGetCurrPcieLinkWidth(hDev, &v) == NVML_SUCCESS)
                G.pcieWidth = (int)v;

            int lcnt = 0;
            for (unsigned l = 0; l < (unsigned)MAX_LINKS; l++) {
                nvmlEnableState_t st;
                nvmlReturn_t r = nvmlDeviceGetNvLinkState(hDev, l, &st);
                if (r != NVML_SUCCESS) break;
                if (st != NVML_FEATURE_ENABLED) continue;
                nvmlPciInfo_t rp;
                r = nvmlDeviceGetNvLinkRemotePciInfo_v2(hDev, l, &rp);
                if (r != NVML_SUCCESS) continue;
                if (lcnt < MAX_LINKS) {
                    strncpy(G.nvLinks[lcnt].remoteBusId, rp.busId, BUSID_SZ - 1);
                    G.nvLinks[lcnt].active = 1;
                    lcnt++;
                }
            }
            G.nNvLinks = lcnt;

#ifndef PHASE3_USE_NVML
            G.nPeerTopos = 0;
            for (int p = 0; p < nAll; p++) {
                if (p == k) continue;
                PeerTopo& pt = G.peerTopos[G.nPeerTopos];
                strncpy(pt.busId, allBusIds[p], BUSID_SZ - 1);
                nvmlGpuTopologyLevel_t lvl;
                nvmlReturn_t r2 = nvmlDeviceGetTopologyCommonAncestor(hDev, hAll[p], &lvl);
                pt.pcieTopo = (r2 == NVML_SUCCESS) ? (int)lvl : -1;
                G.nPeerTopos++;
            }
#endif
            break;
        }

        D.nTopologyNodes++;
    }

    CHK_NVML(nvmlShutdown());
}

/* ==================================================================
 *  Phase 1B – CPU collection  (hwloc, NUMA granularity)
 *             + GPU NUMA affinity mapping
 * ================================================================== */

static void collectLocalCpus(RankData& D) {
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topo);

    /* ---- process CPU binding ---- */
    hwloc_cpuset_t binding = hwloc_bitmap_alloc();
    if (hwloc_get_cpubind(topo, binding, HWLOC_CPUBIND_PROCESS) != 0)
        hwloc_bitmap_fill(binding);

    /* ---- enumerate bound NUMA nodes ---- */
    int nNuma = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    for (int i = 0; i < nNuma; i++) {
        if (D.nTopologyNodes >= MAX_TOPO_NODES) break;

        hwloc_obj_t numaObj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
        if (!numaObj) continue;

        hwloc_cpuset_t numaCpuset = numaObj->cpuset;
        if (!numaCpuset) continue;
        if (!hwloc_bitmap_intersects(binding, numaCpuset)) continue;

        hwloc_cpuset_t overlap = hwloc_bitmap_alloc();
        hwloc_bitmap_and(overlap, binding, numaCpuset);
        int nCores = 0;
        int nAllCores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
        for (int c = 0; c < nAllCores; c++) {
            hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, c);
            if (core && core->cpuset &&
                hwloc_bitmap_intersects(overlap, core->cpuset))
                nCores++;
        }
        hwloc_bitmap_free(overlap);

        int idx = D.nTopologyNodes;
        D.nodes[idx].handle.rank = D.rank;
        D.nodes[idx].handle.type = CPU_HANDLE;
        D.nodes[idx].handle.cpu.numaId = (int)numaObj->os_index;
        memset(&D.nodes[idx].cpu, 0, sizeof(CpuInfo));
        D.nodes[idx].cpu.numaId = (int)numaObj->os_index;
        D.nodes[idx].cpu.nCores = nCores;
        D.nTopologyNodes++;
    }

    /* ---- map each GPU to its closest NUMA node ---- */
    for (int i = 0; i < D.nTopologyNodes; i++) {
        if (D.nodes[i].handle.type != GPU_HANDLE) continue;
        GpuInfo& G = D.nodes[i].gpu;

        unsigned domain, bus, dev, func;
        if (sscanf(G.busId, "%x:%x:%x.%x", &domain, &bus, &dev, &func) == 4) {
            hwloc_obj_t pcidev = hwloc_get_pcidev_by_busid(topo, domain, bus, dev, func);
            if (pcidev) {
                hwloc_obj_t nonIO = hwloc_get_non_io_ancestor_obj(topo, pcidev);
                if (nonIO && nonIO->nodeset)
                    G.numaId = hwloc_bitmap_first(nonIO->nodeset);
            }
        }
    }

    hwloc_bitmap_free(binding);
    hwloc_topology_destroy(topo);
}

/* ==================================================================
 *  Phase 2 – MPI exchange
 * ================================================================== */

static void exchange(const RankData& local, std::vector<RankData>& all) {
    int ws;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
    all.resize(ws);
    MPI_Allgather(&local, sizeof(RankData), MPI_BYTE,
                  all.data(), sizeof(RankData), MPI_BYTE,
                  MPI_COMM_WORLD);
}

static std::string queryConnection(const std::vector<RankManager>& managers,
                                   const Handle& a, const Handle& b) {
    HandlePair cp = canonicalPair(a, b);
    int owner = cp.first.rank;
    if (owner < 0 || owner >= (int)managers.size())
        return "";
    return managers[owner].query(a, b);
}

/* ==================================================================
 *  Phase 3 – build topology maps (one per RankManager)
 * ================================================================== */

void RankManager::buildTopology(const std::vector<RankData>& allRanks) {
    topology.clear();

#ifdef PHASE3_USE_NVML
    CHK_NVML(nvmlInit_v2());
#endif

    const std::string myHost(data.hostname);

    for (int si = 0; si < data.nTopologyNodes; si++) {
        const Handle& src = data.nodes[si].handle;

        for (size_t r = 0; r < allRanks.size(); r++) {
            const RankData& dstRank = allRanks[r];
            bool sameNode = (std::string(dstRank.hostname) == myHost);

            for (int di = 0; di < dstRank.nTopologyNodes; di++) {
                const Handle& dst = dstRank.nodes[di].handle;

                HandlePair cp = canonicalPair(src, dst);
                if (cp.first.rank != data.rank)
                    continue;

                if (topology.count(cp))
                    continue;

                std::string conn = resolveNodeConnection(
                    data.nodes[si], dstRank.nodes[di],
                    sameNode, allRanks, myHost);

                topology[cp] = conn;
            }
        }
    }

#ifdef PHASE3_USE_NVML
    CHK_NVML(nvmlShutdown());
#endif
}

/* ==================================================================
 *  Phase 4 – print global topology from all RankManagers
 * ================================================================== */

struct GNode {
    Handle handle;
    std::string host;
    std::string uuid, busId, name;
    int ccMajor = 0, ccMinor = 0;
    uint64_t memMB = 0;
    int pcieGen = 0, pcieWidth = 0;
    int numaId = -1;
    int nCores = 0;
    std::vector<int> ownerRanks;

    bool isGpu() const { return handle.type == GPU_HANDLE; }

    std::string nodeKey() const {
        if (isGpu()) return uuid;
        return host + ":numa" + std::to_string(numaId);
    }
};

static void printTopology(const std::vector<RankManager>& managers) {
    const int ws = (int)managers.size();

    /* ---- 1. Deduplicated global node list ---- */
    std::vector<GNode> G;
    std::map<std::string, int> key2g;

    for (int r = 0; r < ws; r++) {
        const RankData& R = managers[r].data;
        for (int i = 0; i < R.nTopologyNodes; i++) {
            GNode gn;
            gn.handle = R.nodes[i].handle;
            gn.host   = R.hostname;

            if (gn.isGpu()) {
                const GpuInfo& gi = R.nodes[i].gpu;
                gn.uuid    = gi.uuid;
                gn.busId   = gi.busId;
                gn.name    = gi.name;
                gn.ccMajor = gi.ccMajor;
                gn.ccMinor = gi.ccMinor;
                gn.memMB   = gi.memMB;
                gn.pcieGen = gi.pcieGen;
                gn.pcieWidth = gi.pcieWidth;
                gn.numaId  = gi.numaId;
            } else {
                const CpuInfo& ci = R.nodes[i].cpu;
                gn.numaId = ci.numaId;
                gn.nCores = ci.nCores;
            }

            std::string k = gn.nodeKey();
            if (key2g.count(k) == 0) {
                key2g[k] = (int)G.size();
                G.push_back(gn);
            }
            G[key2g[k]].ownerRanks.push_back(r);
        }
    }

    std::sort(G.begin(), G.end(), [](const GNode& a, const GNode& b) {
        if (a.host != b.host) return a.host < b.host;
        if (a.isGpu() != b.isGpu()) return a.isGpu() > b.isGpu();
        if (a.isGpu())
            return busKey(a.busId.c_str()) < busKey(b.busId.c_str());
        return a.numaId < b.numaId;
    });
    key2g.clear();
    for (int i = 0; i < (int)G.size(); i++) key2g[G[i].nodeKey()] = i;

    int nGpus = 0, nCpus = 0;
    for (auto& g : G) { if (g.isGpu()) nGpus++; else nCpus++; }
    const int N = (int)G.size();

    /* ---- 2. Build printable connection matrix from topology maps ---- */
    std::vector<std::vector<std::string>> conn(N, std::vector<std::string>(N));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) { conn[i][j] = "X"; continue; }

            const Handle& hi = G[i].handle;
            const Handle& hj = G[j].handle;
            HandlePair cp = canonicalPair(hi, hj);
            int owner = cp.first.rank;

            auto it = managers[owner].topology.find(cp);
            if (it != managers[owner].topology.end()) {
                conn[i][j] = it->second;
            } else {
                conn[i][j] = "?";
            }
        }
    }

    /* ---- 3. Print ---- */
    printf("\n");
    printf("=========================================================================\n");
    printf("                    GLOBAL TOPOLOGY REPORT\n");
    printf("=========================================================================\n");
    printf("  %d rank(s),  %d GPU(s),  %d CPU NUMA node(s)\n", ws, nGpus, nCpus);
#ifdef PHASE3_USE_NVML
    printf("  (mode: NVML live query)\n");
#else
    printf("  (mode: pre-collected data only)\n");
#endif

    /* ---- Per-Rank Assignment ---- */
    printf("\n--- Per-Rank Assignment ---\n\n");
    for (int r = 0; r < ws; r++) {
        const RankData& R = managers[r].data;
        int rGpus = 0, rCpus = 0;
        for (int i = 0; i < R.nTopologyNodes; i++) {
            if (R.nodes[i].handle.type == GPU_HANDLE) rGpus++;
            else rCpus++;
        }
        printf("  Rank %-3d @ %-20s  %d GPU, %d CPU NUMA\n",
               r, R.hostname, rGpus, rCpus);
        for (int i = 0; i < R.nTopologyNodes; i++) {
            std::string hl = handleStr(R.nodes[i].handle);
            if (R.nodes[i].handle.type == GPU_HANDLE) {
                const GpuInfo& gi = R.nodes[i].gpu;
                printf("    %-12s  %s [%s] %lu MB PCIe-Gen%d x%d",
                       hl.c_str(), gi.busId, gi.name,
                       (unsigned long)gi.memMB,
                       gi.pcieGen, gi.pcieWidth);
                if (gi.ccMajor)
                    printf(" CC %d.%d", gi.ccMajor, gi.ccMinor);
                if (gi.numaId >= 0)
                    printf(" NUMA:%d", gi.numaId);
                printf("\n");
            } else {
                const CpuInfo& ci = R.nodes[i].cpu;
                printf("    %-12s  %d core(s)\n", hl.c_str(), ci.nCores);
            }
        }
    }

    /* ---- All Unique Nodes ---- */
    printf("\n--- All Unique Nodes (%d GPU, %d CPU NUMA) ---\n\n", nGpus, nCpus);
    for (int i = 0; i < N; i++) {
        const GNode& g = G[i];
        std::string label = handleStr(g.handle);
        if (g.isGpu()) {
            printf("  %-12s  %-16s  %-24s  %-16s  %5lu MB",
                   label.c_str(), g.busId.c_str(), g.name.c_str(),
                   g.host.c_str(), (unsigned long)g.memMB);
            if (g.ccMajor) printf("  CC%d.%d", g.ccMajor, g.ccMinor);
            if (g.numaId >= 0) printf("  NUMA:%d", g.numaId);
        } else {
            printf("  %-12s  %-16s  %-24s  %-16s  %d core(s)",
                   label.c_str(), "", "", g.host.c_str(), g.nCores);
        }
        printf("  Rank:");
        for (size_t k = 0; k < g.ownerRanks.size(); k++)
            printf(" %d", g.ownerRanks[k]);
        printf("\n");
    }

    /* ---- Topology Matrix ---- */
    printf("\n--- Topology Matrix ---\n");
    printf("  Legend:  NVx = NVLink (x links)      PIX = single PCIe switch\n");
    printf("          PXB = multi PCIe switch       PHB = host bridge\n");
    printf("          NODE = same NUMA              SYS = cross-NUMA\n");
    printf("          NET  = cross-node\n\n");

    std::vector<std::string> labels(N);
    int maxLabelLen = 0;
    for (int i = 0; i < N; i++) {
        labels[i] = handleStr(G[i].handle);
        maxLabelLen = std::max(maxLabelLen, (int)labels[i].size());
    }
    int cw = std::max(maxLabelLen + 2, 8);
    int rw = maxLabelLen + 2;

    printf("%-*s", rw, "");
    for (int j = 0; j < N; j++) printf("%-*s", cw, labels[j].c_str());
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("%-*s", rw, labels[i].c_str());
        for (int j = 0; j < N; j++)
            printf("%-*s", cw, conn[i][j].c_str());
        printf("  [%s]\n", G[i].host.c_str());
    }

    /* ---- NVLink Summary ---- */
    printf("\n--- NVLink Summary ---\n\n");
    bool any = false;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (conn[i][j].size() >= 2 && conn[i][j][0] == 'N' && conn[i][j][1] == 'V') {
                printf("  %s <-> %s : %-6s",
                       labels[i].c_str(), labels[j].c_str(),
                       conn[i][j].c_str());
                if (G[i].isGpu() && G[j].isGpu())
                    printf("  (%s <-> %s)", G[i].busId.c_str(), G[j].busId.c_str());
                printf("\n");
                any = true;
            }
    if (!any) printf("  (no NVLink connections detected)\n");

    printf("\n=========================================================================\n\n");
}

/* ==================================================================
 *  main
 * ================================================================== */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);

    int ws;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    /* Phase 1A – GPU */
    if (gRank == 0)
        printf("[Phase 1A] Collecting local GPU data (NVML + CUDA) ...\n");

    RankData local;
    memset(&local, 0, sizeof(local));
    MPI_Comm_rank(MPI_COMM_WORLD, &local.rank);
    gethostname(local.hostname, HOST_SZ);

    collectLocalGpus(local);

    /* Phase 1B – CPU */
    if (gRank == 0)
        printf("[Phase 1B] Collecting local CPU data (hwloc) ...\n");

    collectLocalCpus(local);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Phase 2 */
    if (gRank == 0)
        printf("[Phase 2] Exchanging data across %d rank(s) (MPI_Allgather, "
               "%zu bytes/rank) ...\n", ws, sizeof(RankData));

    std::vector<RankData> allData;
    exchange(local, allData);

    /* Phase 3 – build topology per rank */
    if (gRank == 0)
        printf("[Phase 3] Building topology maps ...\n");

    std::vector<RankManager> managers(ws);
    for (int r = 0; r < ws; r++) {
        managers[r].data = allData[r];
        managers[r].buildTopology(allData);
    }

    /* Phase 4 – print */
    if (gRank == 0) {
        printf("[Phase 4] Printing global topology ...\n");
        printTopology(managers);
    }

    /* queryConnection tests (rank 0 only) */
    if (gRank == 0 && ws >= 1) {
        printf("\n--- queryConnection Tests ---\n\n");

        auto mkGpuHandle = [](int rank, int devId) -> Handle {
            Handle h;
            h.rank = rank;
            h.type = GPU_HANDLE;
            h.gpu.deviceId = devId;
            return h;
        };

        auto mkCpuHandle = [](int rank, int numaId) -> Handle {
            Handle h;
            h.rank = rank;
            h.type = CPU_HANDLE;
            h.cpu.numaId = numaId;
            return h;
        };

        auto printTest = [&](const char* label,
                             const Handle& src, const Handle& dst) {
            std::string conn = queryConnection(managers, src, dst);
            printf("  %-20s  (%s -> %s) = %s\n",
                   label,
                   handleStr(src).c_str(),
                   handleStr(dst).c_str(),
                   conn.empty() ? "(not found)" : conn.c_str());
        };

        if (managers[0].data.nTopologyNodes >= 2) {
            printTest("GPU local->local",
                      mkGpuHandle(0, 0), mkGpuHandle(0, 1));
        }

        if (ws >= 2) {
            printTest("GPU local->remote",
                      mkGpuHandle(0, 0), mkGpuHandle(1, 0));
        }

        if (ws >= 2 && managers[1].data.nTopologyNodes >= 2) {
            printTest("GPU remote->remote",
                      mkGpuHandle(1, 0), mkGpuHandle(1, 1));
        }

        int firstCpuNuma = -1;
        for (int i = 0; i < managers[0].data.nTopologyNodes; i++) {
            if (managers[0].data.nodes[i].handle.type == CPU_HANDLE) {
                firstCpuNuma = managers[0].data.nodes[i].cpu.numaId;
                break;
            }
        }
        if (firstCpuNuma >= 0) {
            printTest("GPU->CPU",
                      mkGpuHandle(0, 0), mkCpuHandle(0, firstCpuNuma));
        }

        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
