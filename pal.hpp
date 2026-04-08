#pragma once

#include <cuda.h>
#include <nvml.h>

#include <cstdint>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <hwloc.h>

static constexpr int MAX_GPUS       = 16;
static constexpr int MAX_NUMAS      = 16;
static constexpr int MAX_TOPO_NODES = 32;
static constexpr int MAX_TOPO_PAIRS = 512;
static constexpr int MAX_LINKS      = 18;
static constexpr int BUSID_SZ       = 32;
static constexpr int UUID_SZ        = 96;
static constexpr int NAME_SZ        = 256;
static constexpr int HOST_SZ        = 256;

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

inline std::string connInfoStr(const CUDTXprocessorConnectionInfo& c) {
    std::string s = connTypeTag(c.type);
    if (c.bandwidth >= 0)
        s += "(" + std::to_string((int)c.bandwidth) + ")";
    return s;
}

/* ==================================================================
 *  POD structures — safe for MPI_Allgather as MPI_BYTE
 * ================================================================== */

struct NvLinkPeer {
    char remoteBusId[BUSID_SZ];
    nvmlIntNvLinkDeviceType_t remoteDeviceType;
};

struct PCIEPeer {
    char busId[BUSID_SZ];
    int  nvmlTopoLevel;
};

struct GPUInfo {
    CUdevice   deviceOrdinal;
    char       uuid[UUID_SZ];
    char       busId[BUSID_SZ];
    char       name[NAME_SZ];
    bool       hasC2C;
    float      nvlinkBwPerLinkGBps; // per-link NVLink speed, -1 if no NVLink
    float      pcieBwGBps;          // theoretical max PCIe bandwidth, -1 if unknown
    float      c2cBwGBps;           // C2C bandwidth, -1 if no C2C
    int32_t    nNvLinks;
    NvLinkPeer nvLinks[MAX_LINKS];
    int32_t    nPcies;
    PCIEPeer   pcies[MAX_GPUS];
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

struct TopologyNode
{
    uint32_t memberId;
    CUIDTXprocessorType type;
    int localId; // GPU: deviceId, CPU: numaId

    TopologyNode(uint32_t memberId, CUIDTXprocessorType type, const ProcessorInfo &info)
    : memberId(memberId)
    , type(type)
    {
        switch (info.type) {
            case CUIDTX_PROCESSOR_TYPE_GPU:
                localId = static_cast<int>(info.gpu.deviceOrdinal);
                break;
            case CUIDTX_PROCESSOR_TYPE_CPU:
                localId = info.numaId;
                break;
            default: localId = -1; break;
        }
    }

    TopologyNode(uint32_t memberId, CUIDTXprocessorType type, int localId)
        : memberId(memberId)
        , type(type)
        , localId(localId)
    {
    }

    TopologyNode() : memberId(UINT32_MAX), type(CUIDTX_PROCESSOR_TYPE_MAX), localId(-1) {}

    using Pair = std::pair<TopologyNode, TopologyNode>;

    bool operator<(const TopologyNode& rhs) const noexcept {
        if (memberId != rhs.memberId) return memberId < rhs.memberId;
        if (type != rhs.type) return type < rhs.type;
        return localId < rhs.localId;
    }

    bool operator==(const TopologyNode& rhs) const noexcept {
        return memberId == rhs.memberId && type == rhs.type && localId == rhs.localId;
    }

    bool operator!=(const TopologyNode& rhs) const noexcept {
        return !(*this == rhs);
    }
};

inline std::string topoNodeStr(const TopologyNode& n) {
    if (n.type == CUIDTX_PROCESSOR_TYPE_GPU)
        return "GPU(" + std::to_string(n.memberId) + "," + std::to_string(n.localId) + ")";
    return "CPU(" + std::to_string(n.memberId) + "," + std::to_string(n.localId) + ")";
}

class Processor {
public:
    Processor(const ProcessorInfo &info, uint32_t memberId)
    : info_(info)
    , topologyNode_(memberId, info.type, info)
    {
        std::memset(&handle_, 0, sizeof(handle_));
        handle_.memberId = memberId;
        handle_.type = info.type;
        switch (info.type) {
            case CUIDTX_PROCESSOR_TYPE_GPU:
                handle_.gpu.deviceOrdinal = info.gpu.deviceOrdinal;
                break;
            case CUIDTX_PROCESSOR_TYPE_CPU: handle_.cpu.cpuOrdinal = info.cpu.cpuOrdinal; break;
            default: break;
        }
    }

    const CUIDTXprocessor& publicHandle() const { return handle_; }
    const ProcessorInfo& info() const { return info_; }
    const TopologyNode& topologyNode() const { return topologyNode_; }

private:
    CUIDTXprocessor handle_ {};
    ProcessorInfo info_ {};
    TopologyNode topologyNode_;
};

/* ==================================================================
 *  Utility functions
 * ================================================================== */

inline std::string handleStr(const CUIDTXprocessor& h) {
    if (h.type == CUIDTX_PROCESSOR_TYPE_GPU)
        return "GPU(" + std::to_string(h.memberId) + "," + std::to_string(h.gpu.deviceOrdinal) + ")";
    return "CPU(" + std::to_string(h.memberId) + "," + std::to_string(h.cpu.cpuOrdinal) + ")";
}

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
