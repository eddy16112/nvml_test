#pragma once

#include "processor.hpp"

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <memory>
#include <ostream>

inline TopologyNode::Pair canonicalPair(const TopologyNode& a,
                                        const TopologyNode& b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

inline std::string busKey(const char* id) {
    std::string s(id);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    auto p = s.find(':');
    if (p != std::string::npos && p < 8)
        s = std::string(8 - p, '0') + s;
    return s;
}

struct MachineManager {
    char hostname_[HOST_SZ];
    uint32_t memberId_;

    using ProcessorMap = std::map<CUIDTXprocessorType, std::vector<std::unique_ptr<Processor>>>;

    void loadPAL(IProcessorAbstractionLayer &pal);

    [[nodiscard]] const std::vector<std::unique_ptr<Processor>> &getProcessorsByType(CUIDTXprocessorType type) const;
    
    void buildTopology(const MachineManager& dst);

    CUDTXprocessorConnectionInfo query(const CUIDTXprocessor& a,
                                       const CUIDTXprocessor& b) const;

    const ProcessorMap& processors() const { return processors_; }
    const std::map<TopologyNode::Pair, CUDTXprocessorConnectionInfo>& topologyMap() const { return topologyMap_; }

    void addProcessor(CUIDTXprocessorType type, std::unique_ptr<Processor> p);
    void addTopologyEntry(const TopologyNode::Pair& pair, const CUDTXprocessorConnectionInfo& ci);
    void clearAll();

    friend std::ostream& operator<<(std::ostream& os, const MachineManager& m);

private:
    ProcessorMap processors_;
    std::map<TopologyNode::Pair, CUDTXprocessorConnectionInfo> topologyMap_;

    CUDTXprocessorConnectionInfo resolveNodeConnection(
            const Processor& src, const Processor& dst,
            bool sameNode, const MachineManager& dstMgr) const;
};

