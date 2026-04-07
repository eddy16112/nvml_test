#include "pal.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <mpi.h>
#include <nvml.h>
#include <hwloc.h>

#define PAL_CHK_NVML(call) do {                                    \
    nvmlReturn_t r_ = (call);                                      \
    if (r_ != NVML_SUCCESS) {                                      \
        fprintf(stderr, "NVML %s:%d – %s\n",                      \
                __FILE__, __LINE__, nvmlErrorString(r_));          \
        fflush(stderr);                                            \
        MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                              \
} while (0)

#define PAL_CHK_CU(call) do {                                      \
    CUresult cu_ = (call);                                         \
    if (cu_ != CUDA_SUCCESS) {                                     \
        const char* cuMsg = nullptr;                               \
        cuGetErrorString(cu_, &cuMsg);                             \
        fprintf(stderr, "CUDA driver %s:%d – %s\n",                \
                __FILE__, __LINE__, cuMsg ? cuMsg : "(no message)"); \
        fflush(stderr);                                            \
        MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                              \
} while (0)

/* ==================================================================
 *  CudaPAL – GPU collection via NVML + CUDA
 * ================================================================== */

CudaPAL::~CudaPAL() = default;

static std::string cuUuidToStr(const CUuuid& u) {
    char buf[80];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
    snprintf(buf, sizeof(buf),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        b[0],b[1],b[2],b[3], b[4],b[5], b[6],b[7],
        b[8],b[9], b[10],b[11],b[12],b[13],b[14],b[15]);
    return buf;
}

/** NVML bitmask of NUMA nodes; use lowest set bit as canonical node id. */
static int nvmlGpuNumaId(nvmlDevice_t hDev) {
    unsigned long nodeSet[4] = {};
    const unsigned int nwords = (unsigned int)(sizeof(nodeSet) / sizeof(nodeSet[0]));
    nvmlReturn_t r = nvmlDeviceGetMemoryAffinity(hDev, nwords, nodeSet,
                                               NVML_AFFINITY_SCOPE_NODE);
    if (r != NVML_SUCCESS)
        return -1;
    for (unsigned int w = 0; w < nwords; w++) {
        unsigned long m = nodeSet[w];
        if (m) {
#if defined(__GNUC__) || defined(__clang__)
            return (int)__builtin_ctzl(m) + (int)(w * (sizeof(unsigned long) * 8));
#else
            for (int b = 0; b < (int)(sizeof(unsigned long) * 8); b++)
                if (m & (1UL << b))
                    return (int)(b + w * (sizeof(unsigned long) * 8));
#endif
        }
    }
    return -1;
}

std::vector<ProcessorInfo> CudaPAL::enumerateProcessors() {
    std::vector<ProcessorInfo> result;

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

    PAL_CHK_CU(cuInit(0));

    int devCount = 0;
    PAL_CHK_CU(cuDeviceGetCount(&devCount));
    int nGpus = std::min(devCount, MAX_GPUS);

    for (int ci = 0; ci < nGpus; ci++) {
        ProcessorInfo info{};
        info.type = CUIDTX_PROCESSOR_TYPE_GPU;
        info.numaId = -1;
        GPUInfo& gpuInfo = info.gpu;
        CUdevice cuDevice{};
        PAL_CHK_CU(cuDeviceGet(&cuDevice, ci));
        gpuInfo.deviceId = cuDevice;
        gpuInfo.nNvLinks = 0;
        gpuInfo.nPcies = 0;

        CUuuid cuUuid{};
        PAL_CHK_CU(cuDeviceGetUuid_v2(&cuUuid, cuDevice));
        std::string cudaUuid = cuUuidToStr(cuUuid);

        for (int k = 0; k < nAll; k++) {
            if (cudaUuid != allUuids[k]) continue;
            nvmlDevice_t hDev = hAll[k];

            strncpy(gpuInfo.uuid, allUuids[k], UUID_SZ - 1);
            strncpy(gpuInfo.busId, allBusIds[k], BUSID_SZ - 1);
            PAL_CHK_NVML(nvmlDeviceGetName(hDev, gpuInfo.name, NAME_SZ));

            nvmlC2cModeInfo_v1_t c2cInfo{};
            gpuInfo.hasC2C = (nvmlDeviceGetC2cModeInfoV(hDev, &c2cInfo) == NVML_SUCCESS
                              && c2cInfo.isC2cEnabled);

            // NVLink per-link unidirectional speed
            gpuInfo.nvlinkBwPerLinkGBps = -1.0f;
            {
                nvmlFieldValue_t fv{};
                fv.fieldId = NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON;
                nvmlDeviceGetFieldValues(hDev, 1, &fv);
                if (fv.nvmlReturn == NVML_SUCCESS && fv.value.uiVal > 0) {
                    gpuInfo.nvlinkBwPerLinkGBps = fv.value.uiVal / 1000.0f;
                } else {
                    // Fallback: derive from NVLink version of first active link
                    unsigned int ver = 0;
                    if (nvmlDeviceGetNvLinkVersion(hDev, 0, &ver) == NVML_SUCCESS && ver > 0) {
                        static const float kNvlinkGBps[] =
                            { 0, 20, 25, 25, 25, 50, 50 };
                        gpuInfo.nvlinkBwPerLinkGBps =
                            (ver < 7) ? kNvlinkGBps[ver] : 50.0f;
                    }
                }
            }

            // PCIe theoretical max bandwidth
            gpuInfo.pcieBwGBps = -1.0f;
            {
                unsigned int gen = 0, width = 0;
                if (nvmlDeviceGetMaxPcieLinkGeneration(hDev, &gen) == NVML_SUCCESS &&
                    nvmlDeviceGetMaxPcieLinkWidth(hDev, &width) == NVML_SUCCESS && gen > 0) {
                    static const float kGenGBpsPerLane[] =
                        { 0, 0.25f, 0.5f, 0.985f, 1.969f, 3.938f, 7.877f };
                    float perLane = (gen < 7) ? kGenGBpsPerLane[gen] : 0;
                    gpuInfo.pcieBwGBps = perLane * width;
                }
            }

            // C2C total bandwidth = link_count × per_link_speed
            gpuInfo.c2cBwGBps = -1.0f;
            if (gpuInfo.hasC2C) {
                nvmlFieldValue_t fvs[2]{};
                fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
                fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
                nvmlDeviceGetFieldValues(hDev, 2, fvs);
                unsigned int nLinks = 1;
                if (fvs[0].nvmlReturn == NVML_SUCCESS && fvs[0].value.uiVal > 0)
                    nLinks = fvs[0].value.uiVal;
                if (fvs[1].nvmlReturn == NVML_SUCCESS)
                    gpuInfo.c2cBwGBps = nLinks * fvs[1].value.uiVal / 1000.0f;
            }

            int lcnt = 0;
            for (unsigned l = 0; l < (unsigned)MAX_LINKS; l++) {
                nvmlEnableState_t st;
                nvmlReturn_t r = nvmlDeviceGetNvLinkState(hDev, l, &st);
                if (r != NVML_SUCCESS) break;
                if (st != NVML_FEATURE_ENABLED) continue;
                nvmlPciInfo_t rp;
                r = nvmlDeviceGetNvLinkRemotePciInfo_v2(hDev, l, &rp);
                if (r != NVML_SUCCESS) continue;
                if (lcnt >= MAX_LINKS)
                    break;
                strncpy(gpuInfo.nvLinks[lcnt].remoteBusId, rp.busId, BUSID_SZ - 1);
                lcnt++;
            }
            gpuInfo.nNvLinks = lcnt;

            gpuInfo.nPcies = 0;
            for (int p = 0; p < nAll; p++) {
                if (p == k) continue;
                if (gpuInfo.nPcies >= MAX_GPUS)
                    break;
                PCIEPeer& peer = gpuInfo.pcies[gpuInfo.nPcies];
                strncpy(peer.busId, allBusIds[p], BUSID_SZ - 1);
                nvmlGpuTopologyLevel_t lvl;
                nvmlReturn_t r2 = nvmlDeviceGetTopologyCommonAncestor(hDev, hAll[p], &lvl);
                peer.nvmlTopoLevel = (r2 == NVML_SUCCESS) ? (int)lvl : -1;
                gpuInfo.nPcies++;
            }

            int numaVal = -1;
            CUresult numaRes = cuDeviceGetAttribute(
                &numaVal, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cuDevice);
            if (numaRes == CUDA_SUCCESS && numaVal >= 0) {
                info.numaId = numaVal;
            } else {
                int nml = nvmlGpuNumaId(hDev);
                if (nml >= 0)
                    info.numaId = nml;
            }

            break;
        }

        result.push_back(info);
    }

    PAL_CHK_NVML(nvmlShutdown());
    return result;
}

/* ==================================================================
 *  CPUPAL – CPU collection via hwloc (NUMA granularity)
 * ================================================================== */

CPUPAL::~CPUPAL() = default;

std::vector<ProcessorInfo> CPUPAL::enumerateProcessors() {
    std::vector<ProcessorInfo> result;

    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    hwloc_cpuset_t binding = hwloc_bitmap_alloc();
    if (hwloc_get_cpubind(topo, binding, HWLOC_CPUBIND_PROCESS) != 0)
        hwloc_bitmap_fill(binding);

    const int numCores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
    int32_t cpuOrdinal = 0;

    for (int coreIdx = 0; coreIdx < numCores; coreIdx++) {
        hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, coreIdx);
        if (!core) continue;

        const hwloc_cpuset_t cpuset = core->cpuset ? core->cpuset : core->complete_cpuset;
        if (!cpuset) continue;

        const int numPus = hwloc_get_nbobjs_inside_cpuset_by_type(topo, cpuset, HWLOC_OBJ_PU);
        if (numPus <= 0) continue;

        bool withinBinding = false;
        for (int puIdx = 0; puIdx < numPus; puIdx++) {
            hwloc_obj_t pu = hwloc_get_obj_inside_cpuset_by_type(topo, cpuset, HWLOC_OBJ_PU, puIdx);
            if (pu && hwloc_bitmap_isset(binding, pu->os_index)) {
                withinBinding = true;
                break;
            }
        }
        if (!withinBinding) continue;

        ProcessorInfo info{};
        info.type = CUIDTX_PROCESSOR_TYPE_CPU;
        info.cpu.cpuOrdinal = cpuOrdinal;
        info.cpu.osIndex = static_cast<uint32_t>(core->os_index);

        const hwloc_nodeset_t nodeset = core->nodeset ? core->nodeset : core->complete_nodeset;
        info.numaId = nodeset ? hwloc_bitmap_first(nodeset) : -1;

        result.push_back(info);
        cpuOrdinal++;
    }

    hwloc_bitmap_free(binding);
    hwloc_topology_destroy(topo);
    return result;
}
