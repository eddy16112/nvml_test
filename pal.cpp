#include "pal.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <cassert>
#include <stdexcept>
#include <mpi.h>
#include <nvml.h>

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

CudaPAL::CudaPAL() {
    PAL_CHK_NVML(nvmlInit_v2());
    PAL_CHK_CU(cuInit(0));
}

CudaPAL::~CudaPAL() {
    PAL_CHK_NVML(nvmlShutdown());
}

// Enumerate all GPUs visible to the current process.
//
// Two device APIs are involved:
//   - NVML: sees *all* GPUs on the node (including those not assigned to this
//     process). Used for topology, bandwidth, and NVLink queries.
//   - CUDA driver: sees only GPUs in CUDA_VISIBLE_DEVICES. Provides the
//     device ordinal that the rest of the runtime uses.
//
// Strategy:
//   1. Collect handle/busId/uuid for every NVML device.
//   2. For each CUDA device, match its UUID against the NVML list to
//      correlate the two handles, then query detailed info via NVML.
std::vector<ProcessorInfo> CudaPAL::enumerateProcessors() {
    std::vector<ProcessorInfo> infos;

    // Step 1: snapshot all NVML-visible devices
    unsigned int nAll = 0;
    PAL_CHK_NVML(nvmlDeviceGetCount_v2(&nAll));
    assert(nAll <= MAX_GPUS);

    struct NvmlDeviceInfo {
        nvmlDevice_t handle;
        char busId[BUSID_SZ];
        char uuid[GPU_UUID_SZ];
    };
    std::vector<NvmlDeviceInfo> allDevs(nAll);
    for (unsigned int i = 0; i < nAll; i++) {
        PAL_CHK_NVML(nvmlDeviceGetHandleByIndex_v2(i, &allDevs[i].handle));
        nvmlPciInfo_t pci;
        PAL_CHK_NVML(nvmlDeviceGetPciInfo_v3(allDevs[i].handle, &pci));
        strncpy(allDevs[i].busId, pci.busId, BUSID_SZ - 1);
        PAL_CHK_NVML(nvmlDeviceGetUUID(allDevs[i].handle, allDevs[i].uuid, GPU_UUID_SZ));
    }

    // Step 2: iterate CUDA devices and correlate with NVML by UUID
    int nGpus = 0;
    PAL_CHK_CU(cuDeviceGetCount(&nGpus));

    for (int ci = 0; ci < nGpus; ci++) {
        ProcessorInfo info{};
        info.type = CUIDTX_PROCESSOR_TYPE_GPU;
        info.numaId = -1;
        GPUInfo& gpuInfo = info.gpu;
        CUdevice cuDevice{};
        PAL_CHK_CU(cuDeviceGet(&cuDevice, ci));
        gpuInfo.deviceOrdinal = cuDevice;
        gpuInfo.nNvSwitchLinks = 0;
        gpuInfo.nGPUPeers = 0;
        PAL_CHK_CU(cuDeviceGetAttribute(
            &(info.numaId), CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cuDevice));

        CUuuid cuUuid{};
        PAL_CHK_CU(cuDeviceGetUuid_v2(&cuUuid, cuDevice));
        std::string cudaUuidStr = cuUuidToStr(cuUuid);

        // Find the matching NVML device and query detailed topology info
        for (unsigned int k = 0; k < nAll; k++) {
            if (cudaUuidStr != allDevs[k].uuid) {
                continue;
            }
            nvmlDevice_t hDev = allDevs[k].handle;

            gpuInfo.uuid = cuUuid;
            strncpy(gpuInfo.busId, allDevs[k].busId, BUSID_SZ - 1);
            PAL_CHK_NVML(nvmlDeviceGetName(hDev, gpuInfo.name, NAME_SZ));

            bool hasC2C = false;
            {
                nvmlC2cModeInfo_v1_t c2cInfo{};
                hasC2C = (nvmlDeviceGetC2cModeInfoV(hDev, &c2cInfo) == NVML_SUCCESS
                          && c2cInfo.isC2cEnabled);
            }

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
            if (hasC2C) {
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

            // NVSwitch fabric cluster UUID + clique ID
            gpuInfo.hasFabricInfo = false;
            memset(gpuInfo.clusterUuid, 0, FABRIC_UUID_SZ);
            gpuInfo.cliqueId = 0;
            {
                nvmlGpuFabricInfo_t fabricInfo{};
                if (nvmlDeviceGetGpuFabricInfo(hDev, &fabricInfo) == NVML_SUCCESS &&
                    fabricInfo.state == NVML_GPU_FABRIC_STATE_COMPLETED &&
                    fabricInfo.status == NVML_SUCCESS) {
                    memcpy(gpuInfo.clusterUuid, fabricInfo.clusterUuid, FABRIC_UUID_SZ);
                    gpuInfo.cliqueId = fabricInfo.cliqueId;
                    gpuInfo.hasFabricInfo = true;
                }
            }

            // Build unified GpuPeer array: start with PCIe peers (same node),
            // then overlay NVLink direct-GPU link counts.
            gpuInfo.nGPUPeers = 0;
            for (unsigned int p = 0; p < nAll; p++) {
                if (p == k) continue;
                nvmlGpuTopologyLevel_t lvl;
                nvmlReturn_t ret = nvmlDeviceGetTopologyCommonAncestor(hDev, allDevs[p].handle, &lvl);
                if (ret != NVML_SUCCESS) continue;
                GPUPeer& peer = gpuInfo.gpuPeers[gpuInfo.nGPUPeers];
                strncpy(peer.busId, allDevs[p].busId, BUSID_SZ - 1);
                peer.busId[BUSID_SZ - 1] = '\0';
                peer.nvmlTopoLevel = lvl;
                peer.nvLinkCount = 0;

                nvmlGpuP2PStatus_t p2pStatus = NVML_P2P_STATUS_NOT_SUPPORTED;
                ret = nvmlDeviceGetP2PStatus(hDev, allDevs[p].handle,
                    NVML_P2P_CAPS_INDEX_ATOMICS, &p2pStatus);
                peer.atomicsSupported = (ret == NVML_SUCCESS &&
                                         p2pStatus == NVML_P2P_STATUS_OK);
                gpuInfo.nGPUPeers++;
            }

            // NVLink enumeration: count NVSwitch lanes and overlay
            // direct-GPU link counts onto the existing GpuPeer entries.
            unsigned int nLinks = NVML_NVLINK_MAX_LINKS;
            {
                nvmlFieldValue_t fv{};
                fv.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
                nvmlDeviceGetFieldValues(hDev, 1, &fv);
                if (fv.nvmlReturn == NVML_SUCCESS && fv.value.uiVal > 0)
                    nLinks = fv.value.uiVal;
            }
            for (unsigned l = 0; l < nLinks; l++) {
                nvmlEnableState_t st;
                nvmlReturn_t ret = nvmlDeviceGetNvLinkState(hDev, l, &st);
                if (ret != NVML_SUCCESS || st != NVML_FEATURE_ENABLED)
                    continue;

                nvmlIntNvLinkDeviceType_t devType = NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
                nvmlDeviceGetNvLinkRemoteDeviceType(hDev, l, &devType);

                if (devType == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
                    gpuInfo.nNvSwitchLinks++;
                } else if (devType == NVML_NVLINK_DEVICE_TYPE_GPU) {
                    nvmlPciInfo_t rp;
                    ret = nvmlDeviceGetNvLinkRemotePciInfo_v2(hDev, l, &rp);
                    if (ret != NVML_SUCCESS) continue;
                    for (int p = 0; p < gpuInfo.nGPUPeers; p++) {
                        if (strncmp(gpuInfo.gpuPeers[p].busId, rp.busId, BUSID_SZ) == 0) {
                            gpuInfo.gpuPeers[p].nvLinkCount++;
                            break;
                        }
                    }
                }
            }

            break;
        }

        infos.push_back(info);
    }

    return infos;
}

/* ==================================================================
 *  CPUPAL – CPU collection via hwloc (NUMA granularity)
 * ================================================================== */

 CPUPAL::CPUPAL()
 {
     if (hwloc_topology_init(&topo_) != 0) {
         throw std::runtime_error("Failed to initialize hwloc topology");
     }
     if (hwloc_topology_load(topo_) != 0) {
         hwloc_topology_destroy(topo_);
         throw std::runtime_error("Failed to load hwloc topology");
     }
 }
 
 CPUPAL::~CPUPAL()
 {
     hwloc_topology_destroy(topo_);
 }

std::vector<ProcessorInfo> CPUPAL::enumerateProcessors() 
{
    std::vector<ProcessorInfo> infos;

        // get the binding of the current process, and we need to filter out the processors that are not in the binding
        hwloc_cpuset_t bindingCpuset = hwloc_bitmap_alloc();
        if (hwloc_get_cpubind(topo_, bindingCpuset, HWLOC_CPUBIND_PROCESS) != 0) {
            hwloc_bitmap_free(bindingCpuset);
            throw std::runtime_error("Failed to get CPU binding info");
            return infos;
        }
    
        // iterate over all physical processors
        const int numCores = hwloc_get_nbobjs_by_type(topo_, HWLOC_OBJ_CORE);
        int32_t cpuOrdinal = 0;
        for (int coreIdx = 0; coreIdx < numCores; coreIdx++) {
            hwloc_obj_t core = hwloc_get_obj_by_type(topo_, HWLOC_OBJ_CORE, coreIdx);
            if (core == nullptr) {
                continue;
            }
    
            const hwloc_cpuset_t cpuset = core->cpuset ? core->cpuset : core->complete_cpuset;
            const int numPus = hwloc_get_nbobjs_inside_cpuset_by_type(topo_, cpuset, HWLOC_OBJ_PU);
            if (numPus <= 0) {
                throw std::runtime_error("No PU found for core " + std::to_string(coreIdx));
            }
    
            // check if the core is within the current process binding cpuset
            bool withinCurrentProcessBinding = false;
    
            CUIDTXCpuSet core_cpuset;
            core_cpuset.clear();
    
            for (int puIdx = 0; puIdx < numPus; puIdx++) {
                hwloc_obj_t pu = hwloc_get_obj_inside_cpuset_by_type(topo_, cpuset, HWLOC_OBJ_PU, puIdx);
                assert(pu != nullptr);
                if (hwloc_bitmap_isset(bindingCpuset, pu->os_index)) {
                    if (!core_cpuset.setBit(pu->os_index)) {
                        throw std::runtime_error("Failed to set bit for PU " + std::to_string(pu->os_index)
                                + " because it exceeds maxBits=" + std::to_string(CUIDTXCpuSet::maxBits));
                    }
                    withinCurrentProcessBinding = true;
                }
            }
            if (withinCurrentProcessBinding) {
                ProcessorInfo info;
                info.type = CUIDTX_PROCESSOR_TYPE_CPU;
                info.cpu.coreIndex = static_cast<uint32_t>(coreIdx);
                info.cpu.osIndex = static_cast<uint32_t>(core->os_index);
                info.cpu.cpuOrdinal = cpuOrdinal;
                info.cpu.cpuset = core_cpuset;
                const hwloc_nodeset_t nodeset = core->nodeset ? core->nodeset : core->complete_nodeset;
                info.numaId = static_cast<int32_t>(hwloc_bitmap_first(nodeset));
                infos.push_back(info);
                cpuOrdinal++;
            }
        }
    
        hwloc_bitmap_free(bindingCpuset);
    
        return infos;
}
