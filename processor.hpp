#pragma once

#include "pal.hpp"
#include "topology_node.hpp"

#include <ostream>

class Processor {
public:
    Processor(const ProcessorInfo &info, uint32_t memberId)
    : info_(info)
    , topologyNode_(memberId, info)
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

std::ostream& operator<<(std::ostream& os, const Processor& p);
