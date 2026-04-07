#pragma once

#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

static constexpr int MAX_GPUS       = 16;
static constexpr int MAX_NUMAS      = 16;
static constexpr int MAX_TOPO_NODES = 32;
static constexpr int MAX_TOPO_PAIRS = 512;
static constexpr int MAX_LINKS      = 18;
static constexpr int BUSID_SZ       = 32;
static constexpr int UUID_SZ        = 96;
static constexpr int NAME_SZ        = 256;
static constexpr int HOST_SZ        = 256;

/* ==================================================================
 *  POD structures — safe for MPI_Allgather as MPI_BYTE
 * ================================================================== */

struct NvLinkPeer {
    char remoteBusId[BUSID_SZ];
};

struct PCIEPeer {
    char busId[BUSID_SZ];
    int  nvmlTopoLevel;
};

struct GPUInfo {
    CUdevice   deviceId;
    char       uuid[UUID_SZ];
    char       busId[BUSID_SZ];
    char       name[NAME_SZ];
    bool       hasC2C;
    int32_t    nNvLinks;
    NvLinkPeer nvLinks[MAX_LINKS];
    int32_t    nPcies;
    PCIEPeer   pcies[MAX_GPUS];
};

struct CPUInfo {
    int32_t cpuOrdinal; // the index of the processor in the list of processors
    uint32_t osIndex; // the os_index of the physical core
};

enum CUIDTXprocessorType : uint8_t {
    CUIDTX_PROCESSOR_TYPE_GPU = 0,
    CUIDTX_PROCESSOR_TYPE_CPU = 1,
    CUIDTX_PROCESSOR_TYPE_MAX = 0xFF
};

struct CUIDTXprocessor {
    int rank;
    union {
        struct { int deviceId; } gpu;
        struct { int cpuOrdinal;  } cpu;
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
    int rank;
    CUIDTXprocessorType type;
    int localId; // GPU: deviceId, CPU: numaId

    TopologyNode(int rank, CUIDTXprocessorType type, const ProcessorInfo &info)
    : rank(rank)
    , type(type)
    {
        switch (info.type) {
            case CUIDTX_PROCESSOR_TYPE_GPU:
                localId = static_cast<int>(info.gpu.deviceId);
                break;
            case CUIDTX_PROCESSOR_TYPE_CPU:
                localId = info.numaId;
                break;
            default: localId = -1; break;
        }
    }

    TopologyNode(int rank, CUIDTXprocessorType type, int localId)
        : rank(rank)
        , type(type)
        , localId(localId)
    {
    }

    TopologyNode() : rank(-1), type(CUIDTX_PROCESSOR_TYPE_MAX), localId(-1) {}

    using Pair = std::pair<TopologyNode, TopologyNode>;

    bool operator<(const TopologyNode& rhs) const noexcept {
        if (rank != rhs.rank) return rank < rhs.rank;
        if (type != rhs.type) return type < rhs.type;
        return localId < rhs.localId;
    }

    bool operator==(const TopologyNode& rhs) const noexcept {
        return rank == rhs.rank && type == rhs.type && localId == rhs.localId;
    }

    bool operator!=(const TopologyNode& rhs) const noexcept {
        return !(*this == rhs);
    }
};

inline std::string topoNodeStr(const TopologyNode& n) {
    if (n.type == CUIDTX_PROCESSOR_TYPE_GPU)
        return "GPU(" + std::to_string(n.rank) + "," + std::to_string(n.localId) + ")";
    return "CPU(" + std::to_string(n.rank) + "," + std::to_string(n.localId) + ")";
}

class Processor {
public:
    Processor(const ProcessorInfo &info, int rank)
    : info_(info)
    , topologyNode_(rank, info.type, info)
    {
        std::memset(&handle_, 0, sizeof(handle_));
        handle_.rank = rank;
        handle_.type = info.type;
        switch (info.type) {
            case CUIDTX_PROCESSOR_TYPE_GPU:
                handle_.gpu.deviceId = static_cast<int>(info.gpu.deviceId);
                break;
            case CUIDTX_PROCESSOR_TYPE_CPU: handle_.cpu.cpuOrdinal = info.cpu.cpuOrdinal; break;
            default: break;
        }
    }

    CUIDTXprocessor handle_ {};
    ProcessorInfo info_ {};
    TopologyNode topologyNode_;
};

/* ==================================================================
 *  Utility functions
 * ================================================================== */

inline std::string handleStr(const CUIDTXprocessor& h) {
    if (h.type == CUIDTX_PROCESSOR_TYPE_GPU)
        return "GPU(" + std::to_string(h.rank) + "," + std::to_string(h.gpu.deviceId) + ")";
    return "CPU(" + std::to_string(h.rank) + "," + std::to_string(h.cpu.cpuOrdinal) + ")";
}

inline std::string busKey(const char* id) {
    std::string s(id);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    auto p = s.find(':');
    if (p != std::string::npos && p < 8)
        s = std::string(8 - p, '0') + s;
    return s;
}

inline bool sameBus(const char* a, const char* b) {
    return busKey(a) == busKey(b);
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
    CudaPAL() = default;
    ~CudaPAL() override;

    CudaPAL(const CudaPAL&) = delete;
    CudaPAL& operator=(const CudaPAL&) = delete;

    [[nodiscard]] CUIDTXprocessorType processorType() const override { return CUIDTX_PROCESSOR_TYPE_GPU; }
    [[nodiscard]] std::vector<ProcessorInfo> enumerateProcessors() override;
};

class CPUPAL final : public IProcessorAbstractionLayer {
public:
    CPUPAL() = default;
    ~CPUPAL() override;

    CPUPAL(const CPUPAL&) = delete;
    CPUPAL& operator=(const CPUPAL&) = delete;

    [[nodiscard]] CUIDTXprocessorType processorType() const override { return CUIDTX_PROCESSOR_TYPE_CPU; }
    [[nodiscard]] std::vector<ProcessorInfo> enumerateProcessors() override;
};
