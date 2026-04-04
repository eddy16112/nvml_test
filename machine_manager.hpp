#pragma once

#include "pal.hpp"

#include <string>
#include <vector>
#include <map>
#include <utility>

struct RankData {
    char           hostname[HOST_SZ];
    int            rank;
    int            nGpus;
    TopologyNode   gpus[MAX_GPUS];
    int            nCpus;
    TopologyNode   cpus[MAX_NUMAS];
};

typedef std::pair<CUIDTXprocessor, CUIDTXprocessor> CUIDTXprocessorPair;

inline CUIDTXprocessorPair canonicalPair(const CUIDTXprocessor& a,
                                         const CUIDTXprocessor& b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

struct MachineManager {
    RankData data;
    std::map<CUIDTXprocessorPair, std::string> topology;

    void buildTopology(const RankData& remote);

    std::string query(const CUIDTXprocessor& a, const CUIDTXprocessor& b) const {
        auto it = topology.find(canonicalPair(a, b));
        return (it != topology.end()) ? it->second : "";
    }
};

void collectAllNodes(const RankData& R,
                     std::vector<const TopologyNode*>& out);

std::string queryConnection(const std::vector<MachineManager>& managers,
                            const CUIDTXprocessor& a,
                            const CUIDTXprocessor& b);

void printTopology(const std::vector<MachineManager>& managers);
