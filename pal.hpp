#pragma once

#include <cstdint>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

#define PHASE3_USE_NVML

static constexpr int MAX_GPUS       = 16;
static constexpr int MAX_NUMAS      = 16;
static constexpr int MAX_TOPO_NODES = 32;
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
    int  active;
};

#ifndef PHASE3_USE_NVML
struct PeerTopo {
    char busId[BUSID_SZ];
    int  pcieTopo;
};
#endif

struct GpuInfo {
    int      deviceId;
    char     uuid[UUID_SZ];
    char     busId[BUSID_SZ];
    char     name[NAME_SZ];
    int      ccMajor, ccMinor;
    uint64_t memMB;
    int      pcieGen, pcieWidth;
    int      numaId;
    int      nNvLinks;
    NvLinkPeer nvLinks[MAX_LINKS];
#ifndef PHASE3_USE_NVML
    int      nPeerTopos;
    PeerTopo peerTopos[MAX_GPUS];
#endif
};

struct CpuInfo {
    int32_t cpuOrdinal; // the index of the processor in the list of processors
    uint32_t osIndex; // the os_index of the physical core
    int numaId;
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

inline bool operator<(const CUIDTXprocessor& a, const CUIDTXprocessor& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    if (a.type != b.type) return a.type < b.type;
    if (a.type == CUIDTX_PROCESSOR_TYPE_GPU) return a.gpu.deviceId < b.gpu.deviceId;
    return a.cpu.cpuOrdinal < b.cpu.cpuOrdinal;
}

inline bool operator==(const CUIDTXprocessor& a, const CUIDTXprocessor& b) {
    if (a.rank != b.rank || a.type != b.type) return false;
    if (a.type == CUIDTX_PROCESSOR_TYPE_GPU) return a.gpu.deviceId == b.gpu.deviceId;
    return a.cpu.cpuOrdinal == b.cpu.cpuOrdinal;
}

struct ProcessorInfo {
    CUIDTXprocessorType type;
    union {
        GpuInfo gpu;
        CpuInfo cpu;
    };
};

class Processor {
public:
    Processor(const ProcessorInfo &info, int rank)
    : info_(info)
    {
        std::memset(&handle_, 0, sizeof(handle_));
        handle_.rank = rank;
        handle_.type = info.type;
        switch (info.type) {
            case CUIDTX_PROCESSOR_TYPE_GPU: handle_.gpu.deviceId = info.gpu.deviceId; break;
            case CUIDTX_PROCESSOR_TYPE_CPU: handle_.cpu.cpuOrdinal = info.cpu.cpuOrdinal; break;
            default: break;
        }
    }

    CUIDTXprocessor handle_ {};
    ProcessorInfo info_ {};
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
