/*
 * MPI GPU Topology Collector
 *
 * Phase 1 – each rank uses NVML + CUDA to collect local GPU info
 *           (all node GPUs, CUDA-visible mapping, PCIe topology, NVLink peers)
 * Phase 2 – MPI_Allgather exchanges RankData across all ranks
 * Phase 3 – rank 0 rebuilds the global topology using ONLY collected data
 *           (no CUDA / NVML calls)
 *
 * Build:
 *   mpicxx -std=c++11 -o gpu_topo local_nvml.cpp \
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

/* ==================================================================
 *  Fixed-size POD structures — safe for MPI_Allgather as MPI_BYTE
 * ================================================================== */

struct NvLinkPeer {
    char remoteBusId[BUSID_SZ];
    int  active;
};

struct GpuInfo {
    char     uuid[UUID_SZ];
    char     busId[BUSID_SZ];
    char     name[NAME_SZ];
    int      ccMajor, ccMinor;
    uint64_t memMB;
    int      pcieGen, pcieWidth;

    int        nNvLinks;
    NvLinkPeer nvLinks[MAX_LINKS];
};

struct RankData {
    char     hostname[HOST_SZ];
    int      rank;

    int      nNodeGpus;
    GpuInfo  gpus[MAX_GPUS];

    int      nVisGpus;
    int      visIdx[MAX_GPUS];

    int      topo[MAX_GPUS][MAX_GPUS];
    int      directNvl[MAX_GPUS][MAX_GPUS];
};

/* ==================================================================
 *  Helpers
 * ================================================================== */

// "00000000:AB:00.0" and "0000:AB:00.0" both become "ab:00.0"
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

/* ==================================================================
 *  Phase 1 – local collection  (NVML + CUDA allowed)
 * ================================================================== */

static void collectLocal(RankData& D) {
    memset(&D, 0, sizeof(D));
    MPI_Comm_rank(MPI_COMM_WORLD, &D.rank);
    gethostname(D.hostname, HOST_SZ);

    CHK_NVML(nvmlInit_v2());

    unsigned int nDev = 0;
    CHK_NVML(nvmlDeviceGetCount_v2(&nDev));
    D.nNodeGpus = std::min((int)nDev, MAX_GPUS);

    nvmlDevice_t hDev[MAX_GPUS];
    for (int i = 0; i < D.nNodeGpus; i++) {
        CHK_NVML(nvmlDeviceGetHandleByIndex_v2(i, &hDev[i]));

        CHK_NVML(nvmlDeviceGetUUID(hDev[i], D.gpus[i].uuid, UUID_SZ));

        nvmlPciInfo_t pci;
        CHK_NVML(nvmlDeviceGetPciInfo_v3(hDev[i], &pci));
        strncpy(D.gpus[i].busId, pci.busId, BUSID_SZ - 1);

        CHK_NVML(nvmlDeviceGetName(hDev[i], D.gpus[i].name, NAME_SZ));

        nvmlMemory_t mem;
        CHK_NVML(nvmlDeviceGetMemoryInfo(hDev[i], &mem));
        D.gpus[i].memMB = mem.total / (1024ULL * 1024);

        unsigned int v = 0;
        if (nvmlDeviceGetCurrPcieLinkGeneration(hDev[i], &v) == NVML_SUCCESS)
            D.gpus[i].pcieGen = (int)v;
        v = 0;
        if (nvmlDeviceGetCurrPcieLinkWidth(hDev[i], &v) == NVML_SUCCESS)
            D.gpus[i].pcieWidth = (int)v;
    }

    /* ---- CUDA-visible → node GPU index mapping ---- */
    int nCuda = 0;
    CHK_CUDA(cudaGetDeviceCount(&nCuda));
    D.nVisGpus = std::min(nCuda, MAX_GPUS);

    for (int ci = 0; ci < D.nVisGpus; ci++) {
        char cbid[BUSID_SZ] = {};
        CHK_CUDA(cudaDeviceGetPCIBusId(cbid, BUSID_SZ, ci));
        cudaDeviceProp prop;
        CHK_CUDA(cudaGetDeviceProperties(&prop, ci));

        D.visIdx[ci] = -1;
        for (int ni = 0; ni < D.nNodeGpus; ni++) {
            if (sameBus(cbid, D.gpus[ni].busId)) {
                D.visIdx[ci] = ni;
                D.gpus[ni].ccMajor = prop.major;
                D.gpus[ni].ccMinor = prop.minor;
                break;
            }
        }
    }

    /* ---- PCIe topology matrix ---- */
    for (int i = 0; i < D.nNodeGpus; i++) {
        D.topo[i][i] = 0;
        for (int j = i + 1; j < D.nNodeGpus; j++) {
            nvmlGpuTopologyLevel_t lvl;
            nvmlReturn_t r = nvmlDeviceGetTopologyCommonAncestor(
                                 hDev[i], hDev[j], &lvl);
            int val = (r == NVML_SUCCESS) ? (int)lvl : -1;
            D.topo[i][j] = D.topo[j][i] = val;
        }
    }

    /* ---- NVLink peer info (stored per-GPU) ---- */
    for (int i = 0; i < D.nNodeGpus; i++) {
        int cnt = 0;
        for (unsigned l = 0; l < (unsigned)MAX_LINKS; l++) {
            nvmlEnableState_t st;
            nvmlReturn_t r = nvmlDeviceGetNvLinkState(hDev[i], l, &st);
            if (r != NVML_SUCCESS) break;
            if (st != NVML_FEATURE_ENABLED) continue;

            nvmlPciInfo_t rp;
            r = nvmlDeviceGetNvLinkRemotePciInfo_v2(hDev[i], l, &rp);
            if (r != NVML_SUCCESS) continue;
            if (cnt < MAX_LINKS) {
                strncpy(D.gpus[i].nvLinks[cnt].remoteBusId, rp.busId, BUSID_SZ - 1);
                D.gpus[i].nvLinks[cnt].active = 1;
                cnt++;
            }
        }
        D.gpus[i].nNvLinks = cnt;

        /* direct NVLink count between GPU i and every other node GPU */
        for (int j = 0; j < D.nNodeGpus; j++) {
            if (i == j) continue;
            int n = 0;
            for (int k = 0; k < cnt; k++)
                if (sameBus(D.gpus[i].nvLinks[k].remoteBusId, D.gpus[j].busId))
                    n++;
            D.directNvl[i][j] = n;
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
 *  Phase 3 – global topology analysis & print (NO CUDA / NVML)
 * ================================================================== */

struct GGpu {
    std::string uuid, busId, name, host;
    int  ccMajor, ccMinor;
    uint64_t memMB;
    int  pcieGen, pcieWidth;
    int  srcRank, nodeIdx;
    std::vector<int> ownerRanks;
};

static void analyzeAndPrint(const std::vector<RankData>& all) {
    const int ws = (int)all.size();

    /* ---- 1. Deduplicated global GPU list ---- */
    std::vector<GGpu> G;
    std::map<std::string, int> uid2g;

    for (int r = 0; r < ws; r++) {
        const RankData& R = all[r];
        std::set<int> vis;
        for (int v = 0; v < R.nVisGpus; v++)
            if (R.visIdx[v] >= 0) vis.insert(R.visIdx[v]);

        for (int i = 0; i < R.nNodeGpus; i++) {
            if (!vis.count(i)) continue;
            std::string u(R.gpus[i].uuid);
            if (uid2g.count(u) == 0) {
                GGpu g;
                g.uuid      = u;
                g.busId     = R.gpus[i].busId;
                g.name      = R.gpus[i].name;
                g.host      = R.hostname;
                g.ccMajor   = R.gpus[i].ccMajor;
                g.ccMinor   = R.gpus[i].ccMinor;
                g.memMB     = R.gpus[i].memMB;
                g.pcieGen   = R.gpus[i].pcieGen;
                g.pcieWidth = R.gpus[i].pcieWidth;
                g.srcRank   = r;
                g.nodeIdx   = i;
                uid2g[u] = (int)G.size();
                G.push_back(g);
            }
            if (R.gpus[i].ccMajor > 0 && G[uid2g[u]].ccMajor == 0) {
                G[uid2g[u]].ccMajor = R.gpus[i].ccMajor;
                G[uid2g[u]].ccMinor = R.gpus[i].ccMinor;
            }
            if (vis.count(i))
                G[uid2g[u]].ownerRanks.push_back(r);
        }
    }

    /* sort by hostname then busId for nicer output */
    std::sort(G.begin(), G.end(), [](const GGpu& a, const GGpu& b) {
        if (a.host != b.host) return a.host < b.host;
        return busKey(a.busId.c_str()) < busKey(b.busId.c_str());
    });
    uid2g.clear();
    for (int i = 0; i < (int)G.size(); i++) uid2g[G[i].uuid] = i;

    const int N = (int)G.size();

    /* ---- 2. Build connection matrix ---- */

    auto isNodeGpu = [](const RankData& R, const char* bid) -> bool {
        for (int g = 0; g < R.nNodeGpus; g++)
            if (sameBus(bid, R.gpus[g].busId)) return true;
        return false;
    };

    std::vector<std::vector<std::string>> conn(N, std::vector<std::string>(N));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) { conn[i][j] = "X"; continue; }

            /* cross-node → network */
            if (G[i].host != G[j].host) { conn[i][j] = "NET"; continue; }

            /* same node → find a rank with data for both GPUs */
            const RankData* pR = nullptr;
            int li = -1, lj = -1;
            for (int r = 0; r < ws; r++) {
                if (std::string(all[r].hostname) != G[i].host) continue;
                int a = -1, b = -1;
                for (int g = 0; g < all[r].nNodeGpus; g++) {
                    if (sameBus(all[r].gpus[g].busId, G[i].busId.c_str())) a = g;
                    if (sameBus(all[r].gpus[g].busId, G[j].busId.c_str())) b = g;
                }
                if (a >= 0 && b >= 0) { pR = &all[r]; li = a; lj = b; break; }
            }
            if (!pR) { conn[i][j] = "?"; continue; }

            /* direct NVLink */
            int dNvl = pR->directNvl[li][lj];
            if (dNvl > 0) {
                conn[i][j] = "NV" + std::to_string(dNvl);
                continue;
            }

            /* NVSwitch detection: count links from GPU i to NVSwitch
               devices that GPU j also connects to */
            std::map<std::string, int> swI, swJ;
            for (int k = 0; k < pR->gpus[li].nNvLinks; k++)
                if (pR->gpus[li].nvLinks[k].active && !isNodeGpu(*pR, pR->gpus[li].nvLinks[k].remoteBusId))
                    swI[busKey(pR->gpus[li].nvLinks[k].remoteBusId)]++;
            for (int k = 0; k < pR->gpus[lj].nNvLinks; k++)
                if (pR->gpus[lj].nvLinks[k].active && !isNodeGpu(*pR, pR->gpus[lj].nvLinks[k].remoteBusId))
                    swJ[busKey(pR->gpus[lj].nvLinks[k].remoteBusId)]++;

            int nvsCount = 0;
            for (auto& kv : swI)
                if (swJ.count(kv.first)) nvsCount += kv.second;

            if (nvsCount > 0) {
                conn[i][j] = "NV" + std::to_string(nvsCount);
                continue;
            }

            /* fallback to PCIe topology */
            conn[i][j] = topoTag(pR->topo[li][lj]);
        }
    }

    /* ---- 3. Print ---- */
    printf("\n");
    printf("=========================================================================\n");
    printf("                    GLOBAL GPU TOPOLOGY REPORT\n");
    printf("=========================================================================\n");
    printf("  %d rank(s),  %d unique GPU(s)\n", ws, N);

    /* per-rank assignment */
    printf("\n--- Per-Rank GPU Assignment ---\n\n");
    for (int r = 0; r < ws; r++) {
        const RankData& R = all[r];
        printf("  Rank %-3d @ %-20s  %d CUDA-visible GPU(s)\n",
               r, R.hostname, R.nVisGpus);
        for (int v = 0; v < R.nVisGpus; v++) {
            int ni = R.visIdx[v];
            if (ni < 0) continue;
            printf("    cuda:%d -> %s [%s] %lu MB PCIe-Gen%d x%d",
                   v, R.gpus[ni].busId, R.gpus[ni].name,
                   (unsigned long)R.gpus[ni].memMB,
                   R.gpus[ni].pcieGen, R.gpus[ni].pcieWidth);
            if (R.gpus[ni].ccMajor)
                printf(" CC %d.%d", R.gpus[ni].ccMajor, R.gpus[ni].ccMinor);
            printf("\n");
        }
    }

    /* unique GPU table */
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
        if (g.ownerRanks.empty()) printf(" -");
        printf("\n");
    }

    /* topology matrix */
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

    /* NVLink summary */
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

    if (gRank == 0)
        printf("[Phase 1] Collecting local GPU data (NVML + CUDA) ...\n");

    RankData local;
    collectLocal(local);
    MPI_Barrier(MPI_COMM_WORLD);

    if (gRank == 0)
        printf("[Phase 2] Exchanging data across %d rank(s) (MPI_Allgather, "
               "%zu bytes/rank) ...\n", ws, sizeof(RankData));

    std::vector<RankData> all;
    exchange(local, all);

    if (gRank == 0) {
        printf("[Phase 3] Analyzing global topology (no CUDA/NVML calls) ...\n");
        analyzeAndPrint(all);
    }

    MPI_Finalize();
    return 0;
}
