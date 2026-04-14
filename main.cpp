/*
 * MPI GPU + CPU Topology Collector
 *
 * Phase 1A: GPU collection  (CudaPAL – NVML + CUDA)
 * Phase 1B: CPU collection  (CPUPAL  – hwloc, NUMA granularity)
 * Phase 2:  MPI_Allgather exchange (processor info only)
 * Phase 3:  Build all topology (rank 0 only)
 * Phase 4:  Print global topology
 *
 * Build (with cmake):
 *   cmake -B build && cmake --build build
 *
 * Run:
 *   mpirun -np <N> ./build/gpu_topo
 */

#include "machine_manager.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <unistd.h>

static constexpr int MAX_TOPO_NODES = 32;

#include <mpi.h>

static int gRank = 0;

template<typename T>
static std::string toStr(const T& v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

static std::string handleStr(const CUIDTXprocessor& h) {
    if (h.type == CUIDTX_PROCESSOR_TYPE_GPU)
        return "GPU(" + std::to_string(h.memberId) + "," + std::to_string(h.gpu.deviceOrdinal) + ")";
    return "CPU(" + std::to_string(h.memberId) + "," + std::to_string(h.cpu.cpuOrdinal) + ")";
}


static inline std::string connInfoStr(const CUDTXprocessorConnectionInfo& c) {
    std::string s = connTypeTag(c.type);
    if (c.bandwidth >= 0)
        s += "(" + std::to_string((int)c.bandwidth) + ")";
    if (c.supportAtomics)
        s += "[A]";
    return s;
}
/* Fixed-size POD wire block: MPI_Allgather(MPI_BYTE, sizeof(RankDataWire)). */

struct RankDataWire {
    uint64_t      hostId;
    int           rank;
    int           nNodes;
    ProcessorInfo nodes[MAX_TOPO_NODES];
};

static void packToWire(const MachineManager& M, RankDataWire& w) {
    memset(&w, 0, sizeof(w));
    w.hostId = M.hostId();
    w.rank = M.memberId();
    w.nNodes = 0;
    for (auto& p : M.getProcessorsByType(CUIDTX_PROCESSOR_TYPE_GPU)) {
        if (w.nNodes >= MAX_TOPO_NODES) break;
        w.nodes[w.nNodes++] = p->info();
    }
    for (auto& p : M.getProcessorsByType(CUIDTX_PROCESSOR_TYPE_CPU)) {
        if (w.nNodes >= MAX_TOPO_NODES) break;
        w.nodes[w.nNodes++] = p->info();
    }
}

static MachineManager unpackFromWire(const RankDataWire& w) {
    MachineManager M(w.rank, w.hostId);
    for (int i = 0; i < w.nNodes; i++) {
        M.addProcessor(w.nodes[i].type,
                       std::make_unique<Processor>(w.nodes[i], w.rank));
    }
    return M;
}

static void exchangeRankData(const MachineManager& local,
                             std::vector<MachineManager>& all) {
    int ws;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    RankDataWire localWire;
    packToWire(local, localWire);

    std::vector<RankDataWire> allWire(ws);
    MPI_Allgather(&localWire, sizeof(RankDataWire), MPI_BYTE,
                  allWire.data(), sizeof(RankDataWire), MPI_BYTE,
                  MPI_COMM_WORLD);

    all.clear();
    all.reserve(ws);
    for (int r = 0; r < ws; r++)
        all.push_back(unpackFromWire(allWire[r]));
}

/* ==================================================================
 *  queryConnection / printTopology
 * ================================================================== */

static CUDTXprocessorConnectionInfo queryConnection(
        const std::vector<MachineManager>& managers,
        const CUIDTXprocessor& a, const CUIDTXprocessor& b) {
    uint32_t owner = std::min(a.memberId, b.memberId);
    if (owner >= (uint32_t)managers.size())
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
    return managers[owner].query(a, b);
}

struct GNode {
    TopologyNode tnode;
    std::string host;
    std::string uuid, busId, name;
    int numaId = -1;
    bool hasFabricInfo = false;
    unsigned char clusterUuid[FABRIC_UUID_SZ] = {};
    uint32_t cliqueId = 0;
    std::vector<int> ownerRanks;

    bool isGpu() const { return tnode.type == CUIDTX_PROCESSOR_TYPE_GPU; }

    std::string nodeKey() const {
        if (isGpu()) return uuid;
        return host + ":numa" + std::to_string(numaId);
    }
};

static void printTopology(const std::vector<MachineManager>& managers) {
    const int ws = (int)managers.size();

    /* ---- 1. Deduplicated global node list ---- */
    std::vector<GNode> G;
    std::map<std::string, int> key2g;

    for (int r = 0; r < ws; r++) {
        const MachineManager& M = managers[r];
        for (const auto& [type, pvec] : M.processors()) {
            for (auto& np : pvec) {
                GNode gn;
                gn.tnode  = np->topologyNode();
                { char buf[20]; snprintf(buf, sizeof(buf), "0x%016lx", (unsigned long)M.hostId()); gn.host = buf; }

                if (gn.isGpu()) {
                    const GPUInfo& gi = np->info().gpu;
                    gn.uuid    = cuUuidToStr(gi.uuid);
                    gn.busId   = gi.busId;
                    gn.name    = gi.name;
                    gn.hasFabricInfo = gi.hasFabricInfo;
                    memcpy(gn.clusterUuid, gi.clusterUuid, FABRIC_UUID_SZ);
                    gn.cliqueId = gi.cliqueId;
                }
                gn.numaId = np->info().numaId;

                std::string k = gn.nodeKey();
                if (key2g.count(k) == 0) {
                    key2g[k] = (int)G.size();
                    G.push_back(gn);
                }
                G[key2g[k]].ownerRanks.push_back(r);
            }
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

    /* ---- 2. Build connection info matrix from topology maps ---- */
    CUDTXprocessorConnectionInfo defaultConn = {CUDTX_PROCESSOR_CONNECTION_TYPE_SELF, -1.0f, false};
    std::vector<std::vector<CUDTXprocessorConnectionInfo>> cinfo(
        N, std::vector<CUDTXprocessorConnectionInfo>(N, defaultConn));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            TopologyNode::Pair nodePair = canonicalPair(G[i].tnode, G[j].tnode);
            int owner = nodePair.first.memberId;
            auto it = managers[owner].topologyMap().find(nodePair);
            if (it != managers[owner].topologyMap().end())
                cinfo[i][j] = it->second;
            else
                cinfo[i][j] = {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
        }
    }

    // short string matrix for display
    std::vector<std::vector<std::string>> conn(N, std::vector<std::string>(N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            conn[i][j] = connInfoStr(cinfo[i][j]);

    /* ---- 3. Print ---- */
    printf("\n");
    printf("=========================================================================\n");
    printf("                    GLOBAL TOPOLOGY REPORT\n");
    printf("=========================================================================\n");
    printf("  %d rank(s),  %d GPU(s),  %d CPU NUMA node(s)\n", ws, nGpus, nCpus);
    printf("  (mode: pre-collected data only)\n");

    /* ---- All Unique Nodes ---- */
    printf("\n--- All Unique Nodes (%d GPU, %d CPU NUMA) ---\n\n", nGpus, nCpus);
    for (int i = 0; i < N; i++) {
        const GNode& g = G[i];
        std::string label = toStr(g.tnode);
        if (g.isGpu()) {
            printf("  %-12s  %-16s  %-24s  %-16s",
                   label.c_str(), g.busId.c_str(), g.name.c_str(),
                   g.host.c_str());
            if (g.numaId >= 0) printf("  NUMA:%d", g.numaId);
            if (g.hasFabricInfo) printf("  Clique:%u", g.cliqueId);
        } else {
            printf("  %-12s  %-16s  %-24s  %-16s  NUMA %d",
                   label.c_str(), "", "", g.host.c_str(), g.numaId);
        }
        printf("  Rank:");
        for (size_t k = 0; k < g.ownerRanks.size(); k++)
            printf(" %d", g.ownerRanks[k]);
        printf("\n");
    }

    /* ---- Per-Manager Details ---- */
    printf("\n--- Per-Manager Details ---\n");
    for (int r = 0; r < ws; r++) {
        std::cout << managers[r];
    }

    /* ---- Topology Matrix ---- */
    printf("\n--- Topology Matrix ---\n");
    printf("  Legend:  NVL = NVLink                  PIX = single PCIe switch\n");
    printf("          PXB = multi PCIe switch       PHB = host bridge\n");
    printf("          C2C = NVLink-C2C (GPU-CPU)    NODE = same NUMA\n");
    printf("          SYS = cross-NUMA              NET  = cross-node\n");
    printf("          (N) = bandwidth in GB/s    [A] = atomics supported\n\n");

    std::vector<std::string> labels(N);
    int maxLabelLen = 0;
    for (int i = 0; i < N; i++) {
        labels[i] = toStr(G[i].tnode);
        maxLabelLen = std::max(maxLabelLen, (int)labels[i].size());
    }
    int maxConnLen = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            maxConnLen = std::max(maxConnLen, (int)conn[i][j].size());
        }
    }
    // Keep a larger fixed padding so long values like NVL(900)[A] do not look crowded.
    int cw = std::max(maxLabelLen, maxConnLen) + 4;
    int rw = maxLabelLen + 4;

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
            if (cinfo[i][j].type == CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK) {
                printf("  %s <-> %s : %-12s",
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);

    int ws;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    std::string hostIdStr = getHostId();
    MachineManager local(gRank, hashHostId(hostIdStr));

    /* Phase 1A – GPU */
    fprintf(stderr, "[R%d@%s] Phase 1A start\n", gRank, hostIdStr.c_str());
    fflush(stderr);
    if (gRank == 0)
        printf("[Phase 1A] Collecting local GPU data (CudaPAL) ...\n");
    {
        CudaPAL gpuPal;
        local.loadPAL(gpuPal);
    }
    fprintf(stderr, "[R%d@%s] Phase 1A done, %d GPUs\n",
            gRank, hostIdStr.c_str(), (int)local.getProcessorsByType(CUIDTX_PROCESSOR_TYPE_GPU).size());
    fflush(stderr);

    /* Phase 1B – CPU */
    if (gRank == 0)
        printf("[Phase 1B] Collecting local CPU data (CPUPAL) ...\n");
    {
        CPUPAL cpuPal;
        local.loadPAL(cpuPal);
    }
    fprintf(stderr, "[R%d@%s] Phase 1B done, %d CPUs\n",
            gRank, hostIdStr.c_str(), (int)local.getProcessorsByType(CUIDTX_PROCESSOR_TYPE_CPU).size());
    fflush(stderr);

    /* Phase 2 */
    MPI_Barrier(MPI_COMM_WORLD);
    if (gRank == 0)
        printf("[Phase 2] Exchanging data across %d rank(s) (MPI_Allgather, "
               "%zu bytes/rank) ...\n", ws, sizeof(RankDataWire));

    std::vector<MachineManager> managers;
    exchangeRankData(local, managers);

    /* Phase 3 – build all topology (rank 0 only) */
    if (gRank == 0) {
        printf("[Phase 3] Building topology maps ...\n");
        for (int r = 0; r < ws; r++)
            for (int d = 0; d < ws; d++)
                managers[r].buildTopology(managers[d]);
    }

    /* Phase 4 – print */
    if (gRank == 0) {
        printf("[Phase 4] Printing global topology ...\n");
        printTopology(managers);
    }

    /* queryConnection tests (rank 0 only) */
    if (gRank == 0 && ws >= 1) {
        printf("\n--- queryConnection Tests ---\n\n");

        auto mkGpu = [](uint32_t memberId, int devId) -> CUIDTXprocessor {
            CUIDTXprocessor h;
            memset(&h, 0, sizeof(h));
            h.memberId = memberId;
            h.type = CUIDTX_PROCESSOR_TYPE_GPU;
            h.gpu.deviceOrdinal = devId;
            return h;
        };

        auto mkCpu = [](uint32_t memberId, int cpuOrdinal) -> CUIDTXprocessor {
            CUIDTXprocessor h;
            memset(&h, 0, sizeof(h));
            h.memberId = memberId;
            h.type = CUIDTX_PROCESSOR_TYPE_CPU;
            h.cpu.cpuOrdinal = cpuOrdinal;
            return h;
        };

        auto printTest = [&](const char* label,
                             const CUIDTXprocessor& src, const CUIDTXprocessor& dst) {
            CUDTXprocessorConnectionInfo ci = queryConnection(managers, src, dst);
            std::string tag = connInfoStr(ci);
            printf("  %-20s  (%s -> %s) = %s\n",
                   label,
                   handleStr(src).c_str(),
                   handleStr(dst).c_str(),
                   tag.c_str());
        };

        if (managers[0].getProcessorsByType(CUIDTX_PROCESSOR_TYPE_GPU).size() >= 2) {
            printTest("GPU local->local",
                      mkGpu(0, 0), mkGpu(0, 1));
        }

        if (ws >= 2) {
            printTest("GPU local->remote",
                      mkGpu(0, 0), mkGpu(1, 0));
        }

        if (ws >= 2 && managers[1].getProcessorsByType(CUIDTX_PROCESSOR_TYPE_GPU).size() >= 2) {
            printTest("GPU remote->remote",
                      mkGpu(1, 0), mkGpu(1, 1));
        }

        if (!managers[0].getProcessorsByType(CUIDTX_PROCESSOR_TYPE_CPU).empty()) {
            int firstOrdinal = managers[0].getProcessorsByType(CUIDTX_PROCESSOR_TYPE_CPU)[0]->publicHandle().cpu.cpuOrdinal;
            printTest("GPU->CPU",
                      mkGpu(0, 0), mkCpu(0, firstOrdinal));
        }

        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
