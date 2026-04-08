/*
 * MPI GPU + CPU Topology Collector
 *
 * Phase 1A: GPU collection  (CudaPAL – NVML + CUDA)
 * Phase 1B: CPU collection  (CPUPAL  – hwloc, NUMA granularity)
 * Phase 1C: Build intra-rank topology (local, parallel across ranks)
 * Phase 2:  MPI_Allgather exchange (processors + local topology)
 * Phase 3:  Build cross-rank topology (rank 0 only)
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
#include <memory>
#include <unistd.h>

#include <mpi.h>

static int gRank = 0;

/* Fixed-size POD wire block: MPI_Allgather(MPI_BYTE, sizeof(RankDataWire)). */

struct TopoEntryWire {
    uint8_t srcType;
    int     srcLocalId;
    uint8_t dstType;
    int     dstLocalId;
    uint8_t connType;
    float   connBandwidth;
};

struct RankDataWire {
    char          hostname[HOST_SZ];
    int           rank;
    int           nNodes;
    ProcessorInfo nodes[MAX_TOPO_NODES];
    int           nTopoEntries;
    TopoEntryWire topoEntries[MAX_TOPO_PAIRS];
};

static void packToWire(const MachineManager& M, RankDataWire& w) {
    memset(&w, 0, sizeof(w));
    strncpy(w.hostname, M.hostname_, HOST_SZ - 1);
    w.rank = M.memberId;
    w.nNodes = 0;
    for (auto& p : M.gpus()) {
        if (w.nNodes >= MAX_TOPO_NODES) break;
        w.nodes[w.nNodes++] = p->info_;
    }
    for (auto& p : M.cpus()) {
        if (w.nNodes >= MAX_TOPO_NODES) break;
        w.nodes[w.nNodes++] = p->info_;
    }
    w.nTopoEntries = 0;
    for (auto& [pair, ci] : M.topologyMap_) {
        if (w.nTopoEntries >= MAX_TOPO_PAIRS) break;
        TopoEntryWire& e = w.topoEntries[w.nTopoEntries];
        e.srcType       = pair.first.type;
        e.srcLocalId    = pair.first.localId;
        e.dstType       = pair.second.type;
        e.dstLocalId    = pair.second.localId;
        e.connType      = ci.type;
        e.connBandwidth = ci.bandwidth;
        w.nTopoEntries++;
    }
}

static void unpackFromWire(const RankDataWire& w, MachineManager& M) {
    strncpy(M.hostname_, w.hostname, HOST_SZ - 1);
    M.hostname_[HOST_SZ - 1] = '\0';
    M.memberId = w.rank;
    M.processors_.clear();
    for (int i = 0; i < w.nNodes; i++) {
        M.processors_[w.nodes[i].type].emplace_back(
            std::make_unique<Processor>(w.nodes[i], w.rank));
    }
    M.topologyMap_.clear();
    for (int i = 0; i < w.nTopoEntries; i++) {
        const TopoEntryWire& e = w.topoEntries[i];
        TopologyNode src(w.rank, (CUIDTXprocessorType)e.srcType, e.srcLocalId);
        TopologyNode dst(w.rank, (CUIDTXprocessorType)e.dstType, e.dstLocalId);
        CUIDTXTopologyConnectionInfo ci;
        ci.type      = (CUDTXprocessorConnectionType)e.connType;
        ci.bandwidth = e.connBandwidth;
        M.topologyMap_[canonicalPair(src, dst)] = ci;
    }
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

    all.resize(ws);
    for (int r = 0; r < ws; r++)
        unpackFromWire(allWire[r], all[r]);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);

    int ws;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    MachineManager local;
    memset(local.hostname_, 0, HOST_SZ);
    local.memberId = gRank;
    gethostname(local.hostname_, HOST_SZ);

    /* Phase 1A – GPU */
    fprintf(stderr, "[R%d@%s] Phase 1A start\n", gRank, local.hostname_);
    fflush(stderr);
    if (gRank == 0)
        printf("[Phase 1A] Collecting local GPU data (CudaPAL) ...\n");
    {
        CudaPAL gpuPal;
        local.loadPAL(gpuPal);
    }
    fprintf(stderr, "[R%d@%s] Phase 1A done, %d GPUs\n",
            gRank, local.hostname_, (int)local.gpus().size());
    fflush(stderr);

    /* Phase 1B – CPU */
    if (gRank == 0)
        printf("[Phase 1B] Collecting local CPU data (CPUPAL) ...\n");
    {
        CPUPAL cpuPal;
        local.loadPAL(cpuPal);
    }
    fprintf(stderr, "[R%d@%s] Phase 1B done, %d CPUs\n",
            gRank, local.hostname_, (int)local.cpus().size());
    fflush(stderr);

    /* Phase 1C – local topology (intra-rank, parallel across all ranks) */
    if (gRank == 0)
        printf("[Phase 1C] Building local topology ...\n");
    local.buildTopology(local);
    fprintf(stderr, "[R%d@%s] Phase 1C done, %zu local topology entries. Entering barrier...\n",
            gRank, local.hostname_, local.topologyMap_.size());
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stderr, "[R%d@%s] Barrier passed\n", gRank, local.hostname_);
    fflush(stderr);

    /* Phase 2 */
    if (gRank == 0)
        printf("[Phase 2] Exchanging data across %d rank(s) (MPI_Allgather, "
               "%zu bytes/rank) ...\n", ws, sizeof(RankDataWire));

    std::vector<MachineManager> managers;
    exchangeRankData(local, managers);

    /* Phase 3 – build cross-rank topology (rank 0 only;
       intra-rank topology already populated from Phase 1C via wire exchange) */
    if (gRank == 0) {
        printf("[Phase 3] Building cross-rank topology maps ...\n");
        for (int r = 0; r < ws; r++)
            for (int d = 0; d < ws; d++)
                if (r != d)
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

        auto mkGpu = [](int rank, int devId) -> CUIDTXprocessor {
            CUIDTXprocessor h;
            memset(&h, 0, sizeof(h));
            h.rank = rank;
            h.type = CUIDTX_PROCESSOR_TYPE_GPU;
            h.gpu.deviceId = devId;
            return h;
        };

        auto mkCpu = [](int rank, int cpuOrdinal) -> CUIDTXprocessor {
            CUIDTXprocessor h;
            memset(&h, 0, sizeof(h));
            h.rank = rank;
            h.type = CUIDTX_PROCESSOR_TYPE_CPU;
            h.cpu.cpuOrdinal = cpuOrdinal;
            return h;
        };

        auto printTest = [&](const char* label,
                             const CUIDTXprocessor& src, const CUIDTXprocessor& dst) {
            CUIDTXTopologyConnectionInfo ci = queryConnection(managers, src, dst);
            std::string tag = connInfoStr(ci);
            printf("  %-20s  (%s -> %s) = %s\n",
                   label,
                   handleStr(src).c_str(),
                   handleStr(dst).c_str(),
                   tag.c_str());
        };

        if (managers[0].gpus().size() >= 2) {
            printTest("GPU local->local",
                      mkGpu(0, 0), mkGpu(0, 1));
        }

        if (ws >= 2) {
            printTest("GPU local->remote",
                      mkGpu(0, 0), mkGpu(1, 0));
        }

        if (ws >= 2 && managers[1].gpus().size() >= 2) {
            printTest("GPU remote->remote",
                      mkGpu(1, 0), mkGpu(1, 1));
        }

        if (!managers[0].cpus().empty()) {
            int firstOrdinal = managers[0].cpus()[0]->handle_.cpu.cpuOrdinal;
            printTest("GPU->CPU",
                      mkGpu(0, 0), mkCpu(0, firstOrdinal));
        }

        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
