#include "pal.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <nvml.h>
#include <cuda_runtime.h>
#include <hwloc.h>

#define PAL_CHK_CUDA(call) do {                                    \
    cudaError_t e_ = (call);                                       \
    if (e_ != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA %s:%d – %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(e_));       \
        std::exit(1);                                              \
    }                                                              \
} while (0)

#define PAL_CHK_NVML(call) do {                                    \
    nvmlReturn_t r_ = (call);                                      \
    if (r_ != NVML_SUCCESS) {                                      \
        fprintf(stderr, "NVML %s:%d – %s\n",                      \
                __FILE__, __LINE__, nvmlErrorString(r_));          \
        std::exit(1);                                              \
    }                                                              \
} while (0)

/* ==================================================================
 *  CudaPAL – GPU collection via NVML + CUDA
 * ================================================================== */

CudaPAL::~CudaPAL() = default;

static std::string cudaUuidToStr(const cudaUUID_t& u) {
    char buf[80];
    const unsigned char* b = (const unsigned char*)u.bytes;
    snprintf(buf, sizeof(buf),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        b[0],b[1],b[2],b[3], b[4],b[5], b[6],b[7],
        b[8],b[9], b[10],b[11],b[12],b[13],b[14],b[15]);
    return buf;
}

std::vector<TopologyNode> CudaPAL::enumerateProcessors() {
    std::vector<TopologyNode> result;

    PAL_CHK_NVML(nvmlInit_v2());

    unsigned int nAllDev = 0;
    PAL_CHK_NVML(nvmlDeviceGetCount_v2(&nAllDev));
    int nAll = std::min((int)nAllDev, MAX_GPUS);

    nvmlDevice_t hAll[MAX_GPUS];
    char allBusIds[MAX_GPUS][BUSID_SZ] = {};
    char allUuids[MAX_GPUS][UUID_SZ] = {};
    for (int i = 0; i < nAll; i++) {
        PAL_CHK_NVML(nvmlDeviceGetHandleByIndex_v2(i, &hAll[i]));
        nvmlPciInfo_t pci;
        PAL_CHK_NVML(nvmlDeviceGetPciInfo_v3(hAll[i], &pci));
        strncpy(allBusIds[i], pci.busId, BUSID_SZ - 1);
        PAL_CHK_NVML(nvmlDeviceGetUUID(hAll[i], allUuids[i], UUID_SZ));
    }

    int nCuda = 0;
    PAL_CHK_CUDA(cudaGetDeviceCount(&nCuda));
    int nGpus = std::min(nCuda, MAX_GPUS);

    for (int ci = 0; ci < nGpus; ci++) {
        TopologyNode node;
        memset(&node, 0, sizeof(node));
        node.handle.rank = 0;
        node.handle.type = GPU_HANDLE;
        node.handle.gpu.deviceId = ci;
        GpuInfo& G = node.gpu;
        G.numaId = -1;

        cudaDeviceProp prop;
        PAL_CHK_CUDA(cudaGetDeviceProperties(&prop, ci));
        G.deviceId = ci;
        G.ccMajor = prop.major;
        G.ccMinor = prop.minor;

        std::string cudaUuid = cudaUuidToStr(prop.uuid);

        for (int k = 0; k < nAll; k++) {
            if (cudaUuid != allUuids[k]) continue;
            nvmlDevice_t hDev = hAll[k];

            strncpy(G.uuid, allUuids[k], UUID_SZ - 1);
            strncpy(G.busId, allBusIds[k], BUSID_SZ - 1);
            PAL_CHK_NVML(nvmlDeviceGetName(hDev, G.name, NAME_SZ));

            nvmlMemory_t mem;
            PAL_CHK_NVML(nvmlDeviceGetMemoryInfo(hDev, &mem));
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

            int numaVal = -1;
            if (cudaDeviceGetAttribute(&numaVal, cudaDevAttrNumaId, ci) == cudaSuccess
                && numaVal >= 0)
                G.numaId = numaVal;

            break;
        }

        result.push_back(node);
    }

    PAL_CHK_NVML(nvmlShutdown());
    return result;
}

/* ==================================================================
 *  CPUPAL – CPU collection via hwloc (NUMA granularity)
 * ================================================================== */

CPUPAL::~CPUPAL() = default;

std::vector<TopologyNode> CPUPAL::enumerateProcessors() {
    std::vector<TopologyNode> result;

    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    hwloc_cpuset_t binding = hwloc_bitmap_alloc();
    if (hwloc_get_cpubind(topo, binding, HWLOC_CPUBIND_PROCESS) != 0)
        hwloc_bitmap_fill(binding);

    int nNuma = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    for (int i = 0; i < nNuma; i++) {
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

        TopologyNode node;
        memset(&node, 0, sizeof(node));
        node.handle.rank = 0;
        node.handle.type = CPU_HANDLE;
        node.handle.cpu.numaId = (int)numaObj->os_index;
        node.cpu.numaId = (int)numaObj->os_index;
        node.cpu.nCores = nCores;
        result.push_back(node);
    }

    hwloc_bitmap_free(binding);
    hwloc_topology_destroy(topo);
    return result;
}
