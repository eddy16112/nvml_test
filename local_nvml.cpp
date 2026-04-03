/*
 * MPI GPU Topology Collector
 *
 * Two modes controlled by PHASE3_USE_NVML:
 *   - Not defined (default): Phase 3 uses only pre-collected data
 *   - Defined:               Phase 3 may call NVML for cross-rank PCIe topology
 *
 * Build:
 *   # default – Phase 3 offline (peerTopos cached in Phase 1)
 *   mpicxx -std=c++11 -o gpu_topo local_nvml.cpp \
 *       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
 *       -lcudart -lnvidia-ml
 *
 *   # Phase 3 can call NVML – smaller struct, no peerTopos needed
 *   mpicxx -std=c++11 -DPHASE3_USE_NVML -o gpu_topo local_nvml.cpp \
 *       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
 *       -lcudart -lnvidia-ml
 *
 * Run:
 *   mpirun -np <N> ./gpu_topo
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

static constexpr int MAX_GPUS  = 16;
static constexpr int MAX_LINKS = 18;
static constexpr int BUSID_SZ  = 32;
static constexpr int UUID_SZ   = 96;
static constexpr int NAME_SZ   = 256;
static constexpr int HOST_SZ   = 256;

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

#define PHASE3_USE_NVML

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
    int      nNvLinks;
    NvLinkPeer nvLinks[MAX_LINKS];
#ifndef PHASE3_USE_NVML
    int      nPeerTopos;
    PeerTopo peerTopos[MAX_GPUS];
#endif
};

struct GpuPairLink {
    int pcieTopo;
    int nvlinkCount;
};

struct CpuInfo {
    int numaId;
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
    TopologyNode   nodes[MAX_GPUS];
    GpuPairLink    link[MAX_GPUS][MAX_GPUS];
};

/* ==================================================================
 *  Helpers
 * ================================================================== */

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
                if (sameBus(bid, R.nodes[g].gpu.busId)) return true;
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

static std::string resolveCrossRankPcie(const GpuInfo& gi, const GpuInfo& gj) {
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

static std::string resolveConnection(
        const GpuInfo& gi, int srcIdx,
        const GpuInfo& gj, int dstIdx,
        const RankData& srcRank, const RankData& dstRank,
        bool sameRank, bool sameNode,
        const std::vector<RankData>& allRanks) {

    if (srcIdx == dstIdx && sameRank)
        return "X";

    if (!sameNode)
        return "NET";

    if (sameRank) {
        const GpuPairLink& lk = srcRank.link[srcIdx][dstIdx];
        if (lk.nvlinkCount > 0)
            return "NV" + std::to_string(lk.nvlinkCount);
        int nvs = countNvSwitchLinks(gi, gj, allRanks, std::string(srcRank.hostname));
        if (nvs > 0)
            return "NV" + std::to_string(nvs);
        return topoTag(lk.pcieTopo);
    }

    int nvl = countNvLinksByBusId(gi, gj.busId);
    if (nvl > 0)
        return "NV" + std::to_string(nvl);
    int nvs = countNvSwitchLinks(gi, gj, allRanks, std::string(srcRank.hostname));
    if (nvs > 0)
        return "NV" + std::to_string(nvs);
    return resolveCrossRankPcie(gi, gj);
}

/* ==================================================================
 *  Phase 1 – local collection  (NVML + CUDA allowed)
 * ================================================================== */

static void collectLocal(RankData& D) {
    memset(&D, 0, sizeof(D));
    MPI_Comm_rank(MPI_COMM_WORLD, &D.rank);
    gethostname(D.hostname, HOST_SZ);

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
    D.nTopologyNodes = std::min(nCuda, MAX_GPUS);

    int nvmlIdx[MAX_GPUS];
    for (int ci = 0; ci < D.nTopologyNodes; ci++) {
        D.nodes[ci].handle.rank = D.rank;
        D.nodes[ci].handle.type = GPU_HANDLE;
        D.nodes[ci].handle.gpu.deviceId = ci;
        GpuInfo& G = D.nodes[ci].gpu;
        nvmlIdx[ci] = -1;

        char cbid[BUSID_SZ] = {};
        CHK_CUDA(cudaDeviceGetPCIBusId(cbid, BUSID_SZ, ci));
        cudaDeviceProp prop;
        CHK_CUDA(cudaGetDeviceProperties(&prop, ci));
        G.deviceId = ci;
        G.ccMajor = prop.major;
        G.ccMinor = prop.minor;

        for (int k = 0; k < nAll; k++) {
            if (!sameBus(cbid, allBusIds[k])) continue;
            nvmlIdx[ci] = k;
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
    }

    for (int i = 0; i < D.nTopologyNodes; i++) {
        D.link[i][i] = {0, 0};
        for (int j = i + 1; j < D.nTopologyNodes; j++) {
            if (nvmlIdx[i] < 0 || nvmlIdx[j] < 0) {
                D.link[i][j] = D.link[j][i] = {-1, 0};
                continue;
            }
            nvmlGpuTopologyLevel_t lvl;
            nvmlReturn_t r = nvmlDeviceGetTopologyCommonAncestor(
                                 hAll[nvmlIdx[i]], hAll[nvmlIdx[j]], &lvl);
            int topo = (r == NVML_SUCCESS) ? (int)lvl : -1;

            int nvl = 0;
            for (int k = 0; k < D.nodes[i].gpu.nNvLinks; k++)
                if (sameBus(D.nodes[i].gpu.nvLinks[k].remoteBusId, D.nodes[j].gpu.busId))
                    nvl++;

            D.link[i][j] = D.link[j][i] = {topo, nvl};
        }
    }

    CHK_NVML(nvmlShutdown());
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

/* ==================================================================
 *  RankManager – owns RankData + per-rank topology map
 * ================================================================== */

typedef std::pair<Handle, Handle> HandlePair;

struct RankManager {
    RankData data;
    std::map<HandlePair, std::string> topology;

    void buildTopology(const std::vector<RankData>& allRanks);

    std::string query(const Handle& src, const Handle& dst) const {
        auto it = topology.find(std::make_pair(src, dst));
        return (it != topology.end()) ? it->second : "";
    }
};

static std::string queryConnection(const std::vector<RankManager>& managers,
                                   const Handle& src, const Handle& dst) {
    if (src.rank < 0 || src.rank >= (int)managers.size())
        return "";
    return managers[src.rank].query(src, dst);
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
        const GpuInfo& gi = data.nodes[si].gpu;

        for (size_t r = 0; r < allRanks.size(); r++) {
            const RankData& dstRank = allRanks[r];
            bool sameRank = ((int)r == data.rank);
            bool sameNode = (std::string(dstRank.hostname) == myHost);

            for (int di = 0; di < dstRank.nTopologyNodes; di++) {
                const Handle& dst = dstRank.nodes[di].handle;
                const GpuInfo& gj = dstRank.nodes[di].gpu;

                std::string conn = resolveConnection(
                    gi, si, gj, di,
                    data, dstRank,
                    sameRank, sameNode,
                    allRanks);

                topology[std::make_pair(src, dst)] = conn;
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

struct GGpu {
    std::string uuid, busId, name, host;
    int  ccMajor, ccMinor;
    uint64_t memMB;
    int  pcieGen, pcieWidth;
    Handle handle;
    std::vector<int> ownerRanks;
};

static void printTopology(const std::vector<RankManager>& managers) {
    const int ws = (int)managers.size();

    /* ---- 1. Deduplicated global GPU list ---- */
    std::vector<GGpu> G;
    std::map<std::string, int> uid2g;

    for (int r = 0; r < ws; r++) {
        const RankData& R = managers[r].data;
        for (int i = 0; i < R.nTopologyNodes; i++) {
            const GpuInfo& gi = R.nodes[i].gpu;
            std::string u(gi.uuid);
            if (uid2g.count(u) == 0) {
                GGpu g;
                g.uuid    = u;
                g.busId   = gi.busId;
                g.name    = gi.name;
                g.host    = R.hostname;
                g.ccMajor = gi.ccMajor;
                g.ccMinor = gi.ccMinor;
                g.memMB   = gi.memMB;
                g.pcieGen = gi.pcieGen;
                g.pcieWidth = gi.pcieWidth;
                g.handle  = R.nodes[i].handle;
                uid2g[u]  = (int)G.size();
                G.push_back(g);
            }
            G[uid2g[u]].ownerRanks.push_back(r);
        }
    }

    std::sort(G.begin(), G.end(), [](const GGpu& a, const GGpu& b) {
        if (a.host != b.host) return a.host < b.host;
        return busKey(a.busId.c_str()) < busKey(b.busId.c_str());
    });
    uid2g.clear();
    for (int i = 0; i < (int)G.size(); i++) uid2g[G[i].uuid] = i;

    const int N = (int)G.size();

    /* ---- 2. Build printable connection matrix from topology maps ---- */
    std::vector<std::vector<std::string>> conn(N, std::vector<std::string>(N));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) { conn[i][j] = "X"; continue; }

            const Handle& hi = G[i].handle;
            const Handle& hj = G[j].handle;
            HandlePair key = std::make_pair(hi, hj);

            auto it = managers[hi.rank].topology.find(key);
            if (it != managers[hi.rank].topology.end()) {
                conn[i][j] = it->second;
            } else {
                conn[i][j] = "?";
            }
        }
    }

    /* ---- 3. Print ---- */
    printf("\n");
    printf("=========================================================================\n");
    printf("                    GLOBAL GPU TOPOLOGY REPORT\n");
    printf("=========================================================================\n");
    printf("  %d rank(s),  %d unique GPU(s)\n", ws, N);
#ifdef PHASE3_USE_NVML
    printf("  (mode: NVML live query)\n");
#else
    printf("  (mode: pre-collected data only)\n");
#endif

    printf("\n--- Per-Rank GPU Assignment ---\n\n");
    for (int r = 0; r < ws; r++) {
        const RankData& R = managers[r].data;
        printf("  Rank %-3d @ %-20s  %d node(s)\n", r, R.hostname, R.nTopologyNodes);
        for (int i = 0; i < R.nTopologyNodes; i++) {
            const GpuInfo& gi = R.nodes[i].gpu;
            printf("    cuda:%d -> %s [%s] %lu MB PCIe-Gen%d x%d",
                   gi.deviceId, gi.busId, gi.name,
                   (unsigned long)gi.memMB,
                   gi.pcieGen, gi.pcieWidth);
            if (gi.ccMajor)
                printf(" CC %d.%d", gi.ccMajor, gi.ccMinor);
            printf("\n");
        }
    }

    printf("\n--- All Unique GPUs (%d) ---\n\n", N);
    for (int i = 0; i < N; i++) {
        const GGpu& g = G[i];
        printf("  GPU%-2d  %-16s  %-30s  %-16s  %5lu MB",
               i, g.busId.c_str(), g.name.c_str(),
               g.host.c_str(), (unsigned long)g.memMB);
        if (g.ccMajor) printf("  CC%d.%d", g.ccMajor, g.ccMinor);
        printf("  Rank:");
        for (size_t k = 0; k < g.ownerRanks.size(); k++)
            printf(" %d", g.ownerRanks[k]);
        printf("\n");
    }

    printf("\n--- Topology Matrix ---\n");
    printf("  Legend:  NVx = NVLink (x links, direct or via NVSwitch)\n");
    printf("          PIX = single PCIe switch   PXB = multi PCIe switch\n");
    printf("          PHB = host bridge           NODE = same NUMA\n");
    printf("          SYS = cross-NUMA            NET  = cross-node\n\n");

    int cw = 8;
    printf("%-8s", "");
    for (int j = 0; j < N; j++) printf("GPU%-*d", cw - 3, j);
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("GPU%-5d", i);
        for (int j = 0; j < N; j++)
            printf("%-*s", cw, conn[i][j].c_str());
        printf("  [%s]\n", G[i].host.c_str());
    }

    printf("\n--- NVLink Summary ---\n\n");
    bool any = false;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (conn[i][j].size() >= 2 && conn[i][j][0] == 'N' && conn[i][j][1] == 'V') {
                printf("  GPU%d <-> GPU%d : %-6s  (%s <-> %s)\n",
                       i, j, conn[i][j].c_str(),
                       G[i].busId.c_str(), G[j].busId.c_str());
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

    /* Phase 1 */
    if (gRank == 0)
        printf("[Phase 1] Collecting local GPU data (NVML + CUDA) ...\n");

    RankData local;
    collectLocal(local);
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

    /* Phase 5 – queryConnection tests (rank 0 only) */
    if (gRank == 0 && ws >= 1) {
        printf("\n--- queryConnection Tests ---\n\n");

        auto mkGpuHandle = [](int rank, int devId) -> Handle {
            Handle h;
            h.rank = rank;
            h.type = GPU_HANDLE;
            h.gpu.deviceId = devId;
            return h;
        };

        auto printTest = [&](const char* label,
                             const Handle& src, const Handle& dst) {
            std::string conn = queryConnection(managers, src, dst);
            printf("  %-20s  (rank%d:gpu%d -> rank%d:gpu%d) = %s\n",
                   label,
                   src.rank, src.gpu.deviceId,
                   dst.rank, dst.gpu.deviceId,
                   conn.empty() ? "(not found)" : conn.c_str());
        };

        /* Test 1: local -> local (rank 0 GPU 0 -> rank 0 GPU 1) */
        if (managers[0].data.nTopologyNodes >= 2) {
            printTest("local->local",
                      mkGpuHandle(0, 0), mkGpuHandle(0, 1));
        } else {
            printf("  local->local      : skipped (rank 0 has < 2 GPUs)\n");
        }

        /* Test 2: local -> remote (rank 0 GPU 0 -> rank 1 GPU 0) */
        if (ws >= 2) {
            printTest("local->remote",
                      mkGpuHandle(0, 0), mkGpuHandle(1, 0));
        } else {
            printf("  local->remote     : skipped (only 1 rank)\n");
        }

        /* Test 3: remote -> remote (rank 1 GPU 0 -> rank 1 GPU 1) */
        if (ws >= 2 && managers[1].data.nTopologyNodes >= 2) {
            printTest("remote->remote",
                      mkGpuHandle(1, 0), mkGpuHandle(1, 1));
        } else {
            printf("  remote->remote    : skipped (need rank 1 with >= 2 GPUs)\n");
        }

        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
