#pragma once

#include <cstdint>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

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
    int numaId;
    int nCores;
};

enum HandleType {
    GPU_HANDLE = 0,
    CPU_HANDLE = 1,
};

struct Handle {
    int rank;
    union {
        struct { int deviceId; } gpu;
        struct { int numaId;  } cpu;
    };
    HandleType type;
};

inline bool operator<(const Handle& a, const Handle& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    if (a.type != b.type) return a.type < b.type;
    if (a.type == GPU_HANDLE) return a.gpu.deviceId < b.gpu.deviceId;
    return a.cpu.numaId < b.cpu.numaId;
}

inline bool operator==(const Handle& a, const Handle& b) {
    if (a.rank != b.rank || a.type != b.type) return false;
    if (a.type == GPU_HANDLE) return a.gpu.deviceId == b.gpu.deviceId;
    return a.cpu.numaId == b.cpu.numaId;
}

struct TopologyNode {
    Handle handle;
    union {
        GpuInfo gpu;
        CpuInfo cpu;
    };
};

/* ==================================================================
 *  Utility functions
 * ================================================================== */

inline std::string handleStr(const Handle& h) {
    if (h.type == GPU_HANDLE)
        return "GPU(" + std::to_string(h.rank) + "," + std::to_string(h.gpu.deviceId) + ")";
    return "CPU(" + std::to_string(h.rank) + "," + std::to_string(h.cpu.numaId) + ")";
}

inline std::string busKey(const char* id) {
    std::string s(id);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    auto p = s.find(':');
    return (p != std::string::npos) ? s.substr(p + 1) : s;
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

    [[nodiscard]] virtual std::vector<TopologyNode> enumerateProcessors() = 0;
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

    [[nodiscard]] std::vector<TopologyNode> enumerateProcessors() override;
};

class CPUPAL final : public IProcessorAbstractionLayer {
public:
    CPUPAL() = default;
    ~CPUPAL() override;

    CPUPAL(const CPUPAL&) = delete;
    CPUPAL& operator=(const CPUPAL&) = delete;

    [[nodiscard]] std::vector<TopologyNode> enumerateProcessors() override;
};
