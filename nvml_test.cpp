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

static void exchange(const RankData& local, std::vector<RankData>& all) {
    int ws;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
    all.resize(ws);
    MPI_Allgather(&local, sizeof(RankData), MPI_BYTE,
                  all.data(), sizeof(RankData), MPI_BYTE,
                  MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);

    int ws;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    RankData local;
    memset(&local, 0, sizeof(local));
    MPI_Comm_rank(MPI_COMM_WORLD, &local.rank);
    gethostname(local.hostname, HOST_SZ);

    /* Phase 1A – GPU */
    if (gRank == 0)
        printf("[Phase 1A] Collecting local GPU data (CudaPAL) ...\n");
    {
        CudaPAL gpuPal;
        for (auto& node : gpuPal.enumerateProcessors()) {
            if (local.nGpus >= MAX_GPUS) break;
            node.handle.rank = local.rank;
            local.gpus[local.nGpus++] = node;
        }
    }

    /* Phase 1B – CPU */
    if (gRank == 0)
        printf("[Phase 1B] Collecting local CPU data (CPUPAL) ...\n");
    {
        CPUPAL cpuPal;
        for (auto& node : cpuPal.enumerateProcessors()) {
            int numaId = node.handle.cpu.numaId;
            if (numaId < 0 || numaId >= MAX_NUMAS) continue;
            node.handle.rank = local.rank;
            local.cpus[numaId] = node;
            local.nCpus++;
        }
    }
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

#ifdef PHASE3_USE_NVML
    CHK_NVML(nvmlInit_v2());
#endif
    std::vector<MachineManager> managers(ws);
    for (int r = 0; r < ws; r++)
        managers[r].data = allData[r];
    for (int r = 0; r < ws; r++)
        for (int d = 0; d < ws; d++)
            managers[r].buildTopology(allData[d]);
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
            h.rank = rank;
            h.type = CUIDTX_PROCESSOR_TYPE_GPU;
            h.gpu.deviceId = devId;
            return h;
        };

        auto mkCpu = [](int rank, int numaId) -> CUIDTXprocessor {
            CUIDTXprocessor h;
            h.rank = rank;
            h.type = CUIDTX_PROCESSOR_TYPE_CPU;
            h.cpu.numaId = numaId;
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

        if (managers[0].data.nGpus >= 2) {
            printTest("GPU local->local",
                      mkGpu(0, 0), mkGpu(0, 1));
        }

        if (ws >= 2) {
            printTest("GPU local->remote",
                      mkGpu(0, 0), mkGpu(1, 0));
        }

        if (ws >= 2 && managers[1].data.nGpus >= 2) {
            printTest("GPU remote->remote",
                      mkGpu(1, 0), mkGpu(1, 1));
        }

        for (int i = 0; i < MAX_NUMAS; i++) {
            if (managers[0].data.cpus[i].handle.type == CUIDTX_PROCESSOR_TYPE_CPU) {
                printTest("GPU->CPU",
                          mkGpu(0, 0),
                          mkCpu(0, managers[0].data.cpus[i].cpu.numaId));
                break;
            }
        }

        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
