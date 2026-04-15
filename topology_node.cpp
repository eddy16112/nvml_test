#include "topology_node.hpp"
#include <stdexcept>

TopologyNode::TopologyNode(CUDTXmemberId memberId, const ProcessorInfo &info)
    : memberId(memberId)
{
    switch (info.type) {
        case CUIDTX_PROCESSOR_TYPE_GPU:
            type = CUIDTX_PROCESSOR_TYPE_GPU;
            localId = static_cast<int32_t>(info.gpu.deviceOrdinal);
            break;
        case CUIDTX_PROCESSOR_TYPE_CPU:
            type = CUIDTX_PROCESSOR_TYPE_CPU;
            localId = static_cast<int32_t>(info.numaId);
            break;
        default: throw std::invalid_argument("Unsupported processor type"); break;
    }
}

std::ostream& operator<<(std::ostream& os, const TopologyNode& n) {
    if (n.type == CUIDTX_PROCESSOR_TYPE_GPU)
        os << "GPU(" << n.memberId << "," << n.localId << ")";
    else
        os << "CPU(" << n.memberId << "," << n.localId << ")";
    return os;
}