#pragma once

#include "pal.hpp"

#include <string>
#include <vector>
#include <map>
#include <utility>


typedef std::pair<CUIDTXprocessor, CUIDTXprocessor> CUIDTXprocessorPair;

inline CUIDTXprocessorPair canonicalPair(const CUIDTXprocessor& a,
                                         const CUIDTXprocessor& b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

struct MachineManager {
    char hostname[HOST_SZ];
    int  rank;

    using ProcessorMap = std::map<CUIDTXprocessorType,
                                    std::vector<TopologyNode>>;
    ProcessorMap processors_;

    std::map<CUIDTXprocessorPair, std::string> topology;

    void buildTopology(const MachineManager& remote);

    std::string query(const CUIDTXprocessor& a, const CUIDTXprocessor& b) const {
        auto it = topology.find(canonicalPair(a, b));
        return (it != topology.end()) ? it->second : "";
    }

    const std::vector<TopologyNode>& gpus() const {
        static const std::vector<TopologyNode> empty;
        auto it = processors_.find(CUIDTX_PROCESSOR_TYPE_GPU);
        return (it != processors_.end()) ? it->second : empty;
    }

    const std::vector<TopologyNode>& cpus() const {
        static const std::vector<TopologyNode> empty;
        auto it = processors_.find(CUIDTX_PROCESSOR_TYPE_CPU);
        return (it != processors_.end()) ? it->second : empty;
    }
};

void collectAllNodes(const MachineManager& M,
                     std::vector<const TopologyNode*>& out);

std::string queryConnection(const std::vector<MachineManager>& managers,
                            const CUIDTXprocessor& a,
                            const CUIDTXprocessor& b);

void printTopology(const std::vector<MachineManager>& managers);
