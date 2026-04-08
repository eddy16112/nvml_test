#pragma once

#include "pal.hpp"

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <memory>

inline TopologyNode::Pair canonicalPair(const TopologyNode& a,
                                        const TopologyNode& b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

struct MachineManager {
    char hostname_[HOST_SZ];
    uint32_t memberId;

    using ProcessorMap = std::map<CUIDTXprocessorType, std::vector<std::unique_ptr<Processor>>>;
    ProcessorMap processors_;

    std::map<TopologyNode::Pair, CUIDTXTopologyConnectionInfo> topologyMap_;

    // Load processors from a PAL
    void loadPAL(IProcessorAbstractionLayer &pal);

    void buildTopology(const MachineManager& dst);

    CUIDTXTopologyConnectionInfo query(const CUIDTXprocessor& a,
                                       const CUIDTXprocessor& b) const;

    const std::vector<std::unique_ptr<Processor>>& gpus() const {
        static const std::vector<std::unique_ptr<Processor>> empty;
        auto it = processors_.find(CUIDTX_PROCESSOR_TYPE_GPU);
        return (it != processors_.end()) ? it->second : empty;
    }

    const std::vector<std::unique_ptr<Processor>>& cpus() const {
        static const std::vector<std::unique_ptr<Processor>> empty;
        auto it = processors_.find(CUIDTX_PROCESSOR_TYPE_CPU);
        return (it != processors_.end()) ? it->second : empty;
    }

private:
    CUIDTXTopologyConnectionInfo resolveNodeConnection(
            const Processor& src, const Processor& dst,
            bool sameNode, const MachineManager& dstMgr) const;
};

CUIDTXTopologyConnectionInfo queryConnection(
        const std::vector<MachineManager>& managers,
        const CUIDTXprocessor& a,
        const CUIDTXprocessor& b);

void printTopology(const std::vector<MachineManager>& managers);
