/*
 * MPI GPU + CPU Topology Collector
 *
 * Phase 1A: GPU collection  (CudaPAL – NVML + CUDA)
 * Phase 1B: CPU collection  (CPUPAL  – hwloc, NUMA granularity)
 * Phase 2:  MPI_Allgather exchange
 * Phase 3:  Build per-rank topology maps
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
#include <cstring>
#include <vector>
#include <memory>
#include <unistd.h>

#include <mpi.h>
#include <nvml.h>

static int gRank = 0;

#ifdef PHASE3_USE_NVML
#define CHK_NVML(call) do {                                        \
    nvmlReturn_t r_ = (call);                                      \
    if (r_ != NVML_SUCCESS) {                                      \
        fprintf(stderr, "[R%d] NVML %s:%d – %s\n",                \
                gRank, __FILE__, __LINE__, nvmlErrorString(r_));   \
        MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                              \
} while (0)
#endif

/* POD wire format for MPI_Allgather */
struct RankDataWire {
    char          hostname[HOST_SZ];
    int           rank;
    int           nNodes;
    ProcessorInfo nodes[MAX_TOPO_NODES];
};

static void packToWire(const MachineManager& M, RankDataWire& w) {
    memset(&w, 0, sizeof(w));
    strncpy(w.hostname, M.hostname, HOST_SZ - 1);
    w.rank = M.rank;
    w.nNodes = 0;
    for (auto& p : M.gpus()) {
        if (w.nNodes >= MAX_TOPO_NODES) break;
        w.nodes[w.nNodes++] = p->info_;
    }
    for (auto& p : M.cpus()) {
        if (w.nNodes >= MAX_TOPO_NODES) break;
        w.nodes[w.nNodes++] = p->info_;
    }
}

static void unpackFromWire(const RankDataWire& w, MachineManager& M) {
    strncpy(M.hostname, w.hostname, HOST_SZ - 1);
    M.hostname[HOST_SZ - 1] = '\0';
    M.rank = w.rank;
    M.processors_.clear();
    for (int i = 0; i < w.nNodes; i++) {
        M.processors_[w.nodes[i].type].emplace_back(
            std::make_unique<Processor>(w.nodes[i], w.rank));
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
    memset(local.hostname, 0, HOST_SZ);
    local.rank = gRank;
    gethostname(local.hostname, HOST_SZ);

    /* Phase 1A – GPU */
    if (gRank == 0)
        printf("[Phase 1A] Collecting local GPU data (CudaPAL) ...\n");
    {
        CudaPAL gpuPal;
        local.loadPAL(gpuPal);
    }

    /* Phase 1B – CPU */
    if (gRank == 0)
        printf("[Phase 1B] Collecting local CPU data (CPUPAL) ...\n");
    {
        CPUPAL cpuPal;
        local.loadPAL(cpuPal);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Phase 2 */
    if (gRank == 0)
        printf("[Phase 2] Exchanging data across %d rank(s) (MPI_Allgather, "
               "%zu bytes/rank) ...\n", ws, sizeof(RankDataWire));

    std::vector<MachineManager> managers;
    exchangeRankData(local, managers);

    /* Phase 3 – build topology per rank */
    if (gRank == 0)
        printf("[Phase 3] Building topology maps ...\n");

#ifdef PHASE3_USE_NVML
    CHK_NVML(nvmlInit_v2());
#endif
    for (int r = 0; r < ws; r++)
        for (int d = 0; d < ws; d++)
            managers[r].buildTopology(managers[d]);
#ifdef PHASE3_USE_NVML
    CHK_NVML(nvmlShutdown());
#endif

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
            std::string conn = queryConnection(managers, src, dst);
            printf("  %-20s  (%s -> %s) = %s\n",
                   label,
                   handleStr(src).c_str(),
                   handleStr(dst).c_str(),
                   conn.empty() ? "(not found)" : conn.c_str());
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
