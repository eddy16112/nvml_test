#pragma once

#include "pal.hpp"

#include <cstdint>
#include <string>
#include <utility>

struct TopologyNode
{
    uint32_t memberId;
    CUIDTXprocessorType type;
    int localId; // GPU: deviceId, CPU: numaId

    TopologyNode(CUDTXmemberId memberId, const ProcessorInfo &info);

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
