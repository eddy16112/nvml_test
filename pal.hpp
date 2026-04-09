#pragma once

#include <cuda.h>
#include <nvml.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <hwloc.h>

static constexpr int MAX_GPUS       = 8;
static constexpr int MAX_NVLINKS    = NVML_NVLINK_MAX_LINKS;
static constexpr int BUSID_SZ       = NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE;
static constexpr int GPU_UUID_SZ    = NVML_DEVICE_UUID_V2_BUFFER_SIZE;
static constexpr int FABRIC_UUID_SZ = NVML_GPU_FABRIC_UUID_LEN;
static constexpr int NAME_SZ        = 256;
static constexpr int HOST_SZ        = 256;

#define CUDTXmemberId uint32_t

struct CUIDTXCpuSet
{
    cpu_set_t cpuSet;

    void clear() noexcept
    {
        CPU_ZERO(&cpuSet);
    }

    bool setBit(int index) noexcept
    {
        if (index >= CUIDTXCpuSet::maxBits) {
            return false;
        }
        CPU_SET(index, &cpuSet);
        return true;
    }

    [[nodiscard]] int isSet(int index) const noexcept
    {
        return CPU_ISSET(index, &cpuSet);
    }

    [[nodiscard]] int numOnes() const noexcept
    {
        return CPU_COUNT(&cpuSet);
    }

    static constexpr int maxBits = CPU_SETSIZE;
};

/* ==================================================================
 *  Connection type enum + info struct
 * ================================================================== */

typedef enum CUDTXprocessorConnectionType_enum {
    CUDTX_PROCESSOR_CONNECTION_TYPE_SELF = 0x0,
    CUDTX_PROCESSOR_CONNECTION_TYPE_PIX = 0x1,
    CUDTX_PROCESSOR_CONNECTION_TYPE_PXB = 0x2,
    CUDTX_PROCESSOR_CONNECTION_TYPE_PHB = 0x3,
    CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK = 0x4,
    CUDTX_PROCESSOR_CONNECTION_TYPE_NODE = 0x5,
    CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM = 0x6,
    CUDTX_PROCESSOR_CONNECTION_TYPE_C2C = 0x7,
    CUDTX_PROCESSOR_CONNECTION_TYPE_MAX = 0x7FFFFFFF
} CUDTXprocessorConnectionType;

typedef struct CUDTXprocessorConnectionInfo_st {
    CUDTXprocessorConnectionType type;
    float bandwidth; // GB/s, -1 if unknown/not applicable
    bool supportAtomics;
} CUDTXprocessorConnectionInfo;

inline const char* connTypeTag(CUDTXprocessorConnectionType t) {
    switch (t) {
        case CUDTX_PROCESSOR_CONNECTION_TYPE_SELF:   return "X";
        case CUDTX_PROCESSOR_CONNECTION_TYPE_PIX:    return "PIX";
        case CUDTX_PROCESSOR_CONNECTION_TYPE_PXB:    return "PXB";
        case CUDTX_PROCESSOR_CONNECTION_TYPE_PHB:    return "PHB";
        case CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK: return "NVL";
        case CUDTX_PROCESSOR_CONNECTION_TYPE_NODE:   return "NODE";
        case CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM:    return "SYS";
        case CUDTX_PROCESSOR_CONNECTION_TYPE_C2C:    return "C2C";
        default:                                     return "NET";
    }
}


inline std::string cuUuidToStr(const CUuuid& u) {
    char buf[80];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
    snprintf(buf, sizeof(buf),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        b[0],b[1],b[2],b[3], b[4],b[5], b[6],b[7],
        b[8],b[9], b[10],b[11],b[12],b[13],b[14],b[15]);
    return buf;
}

/* ==================================================================
 *  POD structures — safe for MPI_Allgather as MPI_BYTE
 * ================================================================== */

struct GPUPeer {
    char busId[BUSID_SZ];
    int32_t nvLinkCount;      // direct NVLink lanes to this peer, 0 if none
    nvmlGpuTopologyLevel_t nvmlTopoLevel;    // PCIe topology level, -1 if not on same node
    bool atomicsSupported;
};

struct GPUInfo {
    CUdevice   deviceOrdinal;
    CUuuid     uuid;
    char       busId[BUSID_SZ];
    char       name[NAME_SZ];
    float      nvlinkBwPerLinkGBps; // per-link NVLink speed, -1 if no NVLink
    float      pcieBwGBps;          // theoretical max PCIe bandwidth, -1 if unknown
    float      c2cBwGBps;           // C2C bandwidth, -1 if no C2C
    unsigned char clusterUuid[FABRIC_UUID_SZ]; // NVSwitch fabric cluster UUID
    uint32_t   cliqueId;            // fabric clique within the cluster
    bool       hasFabricInfo;       // true if fabric info was successfully queried
    int32_t    nNvSwitchLinks;     // total NVLink lanes connected to NVSwitches
    int32_t    nGPUPeers;
    GPUPeer    gpuPeers[MAX_GPUS];
};

struct CPUInfo
{
    int32_t cpuOrdinal; // the index of the processor in the list of processors
    uint32_t coreIndex; // the global index of the physical core on the local physical node
    uint32_t osIndex; // the os_index of the physical core
    CUIDTXCpuSet cpuset; // the cpuset of the physical core, contains the os_index of the PUs in the core
};

enum CUIDTXprocessorType : uint8_t {
    CUIDTX_PROCESSOR_TYPE_GPU = 0,
    CUIDTX_PROCESSOR_TYPE_CPU = 1,
    CUIDTX_PROCESSOR_TYPE_MAX = 0xFF
};

struct CUIDTXprocessor {
    uint32_t memberId;
    union {
        struct { CUdevice deviceOrdinal; } gpu;
        struct { int32_t cpuOrdinal;  } cpu;
    };
    CUIDTXprocessorType type;
};

struct ProcessorInfo {
    CUIDTXprocessorType type;
    int numaId; // NUMA node for both GPU and CPU (-1 if unknown)
    union {
        GPUInfo gpu;
        CPUInfo cpu;
    };
};

/* ==================================================================
 *  Abstract interface — each device type implements this
 * ================================================================== */

class IProcessorAbstractionLayer {
public:
    virtual ~IProcessorAbstractionLayer() = default;

    [[nodiscard]] virtual CUIDTXprocessorType processorType() const = 0;
    [[nodiscard]] virtual std::vector<ProcessorInfo> enumerateProcessors() = 0;
};

/* ==================================================================
 *  Concrete PAL classes
 * ================================================================== */

class CudaPAL final : public IProcessorAbstractionLayer {
public:
    CudaPAL();
    ~CudaPAL() override;

    CudaPAL(const CudaPAL&) = delete;
    CudaPAL& operator=(const CudaPAL&) = delete;

    [[nodiscard]] CUIDTXprocessorType processorType() const override { return CUIDTX_PROCESSOR_TYPE_GPU; }
    [[nodiscard]] std::vector<ProcessorInfo> enumerateProcessors() override;
};

class CPUPAL final : public IProcessorAbstractionLayer {
public:
    CPUPAL();
    ~CPUPAL() override;

    CPUPAL(const CPUPAL&) = delete;
    CPUPAL& operator=(const CPUPAL&) = delete;

    [[nodiscard]] CUIDTXprocessorType processorType() const override { return CUIDTX_PROCESSOR_TYPE_CPU; }
    [[nodiscard]] std::vector<ProcessorInfo> enumerateProcessors() override;

    hwloc_topology_t topo_;
};
