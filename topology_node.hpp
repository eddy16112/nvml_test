#pragma once

#include "pal.hpp"

#include <cstdint>
#include <string>
#include <ostream>
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

    std::size_t hash() const noexcept
    {
        std::size_t h = std::hash<CUDTXmemberId> {}(memberId);
        h ^= std::hash<uint64_t> {}(
                 (static_cast<uint64_t>(type) << 32U) | static_cast<uint64_t>(static_cast<uint32_t>(localId))
             )
            + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        return h;
    }
};

inline std::ostream& operator<<(std::ostream& os, const TopologyNode& n) {
    if (n.type == CUIDTX_PROCESSOR_TYPE_GPU)
        os << "GPU(" << n.memberId << "," << n.localId << ")";
    else
        os << "CPU(" << n.memberId << "," << n.localId << ")";
    return os;
}

template <>
struct std::hash<TopologyNode>
{
    std::size_t operator()(const TopologyNode &v) const noexcept
    {
        return v.hash();
    }
};

template <>
struct std::hash<TopologyNode::Pair>
{
    std::size_t operator()(const TopologyNode::Pair &v) const noexcept
    {
        const std::size_t h1 = v.first.hash();
        const std::size_t h2 = v.second.hash();
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
};