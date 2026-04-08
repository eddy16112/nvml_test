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
    uint32_t memberId_;

    using ProcessorMap = std::map<CUIDTXprocessorType, std::vector<std::unique_ptr<Processor>>>;

    void loadPAL(IProcessorAbstractionLayer &pal);
    void buildTopology(const MachineManager& dst);

    CUDTXprocessorConnectionInfo query(const CUIDTXprocessor& a,
                                       const CUIDTXprocessor& b) const;

    const std::vector<std::unique_ptr<Processor>>& gpus() const {
        static const std::vector<std::unique_ptr<Processor>> empty;
        ProcessorMap::const_iterator it = processors_.find(CUIDTX_PROCESSOR_TYPE_GPU);
        return (it != processors_.end()) ? it->second : empty;
    }

    const std::vector<std::unique_ptr<Processor>>& cpus() const {
        static const std::vector<std::unique_ptr<Processor>> empty;
        ProcessorMap::const_iterator it = processors_.find(CUIDTX_PROCESSOR_TYPE_CPU);
        return (it != processors_.end()) ? it->second : empty;
    }

    const ProcessorMap& processors() const { return processors_; }
    const std::map<TopologyNode::Pair, CUDTXprocessorConnectionInfo>& topologyMap() const { return topologyMap_; }

    void addProcessor(CUIDTXprocessorType type, std::unique_ptr<Processor> p);
    void addTopologyEntry(const TopologyNode::Pair& pair, const CUDTXprocessorConnectionInfo& ci);
    void clearAll();

    void print() const;

private:
    ProcessorMap processors_;
    std::map<TopologyNode::Pair, CUDTXprocessorConnectionInfo> topologyMap_;

    CUDTXprocessorConnectionInfo resolveNodeConnection(
            const Processor& src, const Processor& dst,
            bool sameNode, const MachineManager& dstMgr) const;
};

CUDTXprocessorConnectionInfo queryConnection(
        const std::vector<MachineManager>& managers,
        const CUIDTXprocessor& a,
        const CUIDTXprocessor& b);

void printTopology(const std::vector<MachineManager>& managers);
