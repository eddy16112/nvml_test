#pragma once

#include "processor.hpp"
#include "result_or.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <memory>
#include <ostream>

inline TopologyNode::Pair canonicalPair(const TopologyNode& a,
                                        const TopologyNode& b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

inline std::string getHostId() {
    char hostbuf[256]{};
    gethostname(hostbuf, sizeof(hostbuf));
    std::string id(hostbuf);
    FILE* f = fopen("/proc/sys/kernel/random/boot_id", "r");
    if (f) {
        char bootId[64]{};
        if (fscanf(f, "%63s", bootId) == 1)
            id += bootId;
        fclose(f);
    }
    return id;
}

inline uint64_t hashHostId(const std::string& id) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : id) {
        h ^= static_cast<uint8_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
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
    using TopologyMap = std::unordered_map<TopologyNode::Pair, CUDTXprocessorConnectionInfo>;

    using ProcessorMap = std::map<CUIDTXprocessorType, std::vector<std::unique_ptr<Processor>>>;

    explicit MachineManager(uint32_t memberId, uint64_t hostId)
        : hostId_(hostId), memberId_(memberId) {}

    uint64_t hostId() const { return hostId_; }
    uint32_t memberId() const { return memberId_; }

    void loadPAL(IProcessorAbstractionLayer &pal);

    [[nodiscard]] const std::vector<std::unique_ptr<Processor>> &getProcessorsByType(CUIDTXprocessorType type) const;
    
    void buildTopology(const MachineManager& dst);

    ResultOr<CUDTXprocessorConnectionInfo>
    lookupTopology(const TopologyNode& a, const TopologyNode& b) const noexcept;

    ResultOr<CUDTXprocessorConnectionInfo>
    getTopology(const Processor& src, const Processor& dst) const noexcept;

    const ProcessorMap& processors() const { return processors_; }
    const TopologyMap& topologyMap() const { return topologyMap_; }

    void addProcessor(CUIDTXprocessorType type, std::unique_ptr<Processor> p);
    void addTopologyEntry(const TopologyNode::Pair& pair, const CUDTXprocessorConnectionInfo& ci);
    friend std::ostream& operator<<(std::ostream& os, const MachineManager& m);

private:
    uint64_t hostId_ = 0;
    uint32_t memberId_ = 0;
    ProcessorMap processors_;
    TopologyMap topologyMap_;

    CUDTXprocessorConnectionInfo resolveNodeConnection(
            const Processor& src, const Processor& dst,
            bool sameNode, const MachineManager& dstMgr) const;
};

