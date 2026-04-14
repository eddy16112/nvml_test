#include "machine_manager.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cassert>
#include <ostream>
#include <iomanip>


void MachineManager::loadPAL(IProcessorAbstractionLayer &pal) 
{
    // load processors from the PAL
    std::vector<ProcessorInfo> processorInfos(pal.enumerateProcessors());
    std::vector<std::unique_ptr<Processor>> &processors = processors_[pal.processorType()];
    assert(processors.empty());
    processors.reserve(processorInfos.size());
    for (const ProcessorInfo &info : processorInfos) {
        processors.emplace_back(std::make_unique<Processor>(info, memberId_));
    }
}

const std::vector<std::unique_ptr<Processor>> &MachineManager::getProcessorsByType(CUIDTXprocessorType type) const
{
    auto it = processors_.find(type);
    if (it != processors_.end()) {
        return it->second;
    }

    static const std::vector<std::unique_ptr<Processor>> empty;
    return empty;
}

void MachineManager::addProcessor(CUIDTXprocessorType type, std::unique_ptr<Processor> p) {
    processors_[type].emplace_back(std::move(p));
}

void MachineManager::addTopologyEntry(const TopologyNode::Pair& pair,
                                      const CUDTXprocessorConnectionInfo& ci) {
    topologyMap_[pair] = ci;
}


/* ==================================================================
 *  Helpers
 * ================================================================== */

static CUDTXprocessorConnectionType pcieTopoToConnType(int t) {
    switch (t) {
        case 0:  return CUDTX_PROCESSOR_CONNECTION_TYPE_SELF;
        case 10: return CUDTX_PROCESSOR_CONNECTION_TYPE_PIX;
        case 20: return CUDTX_PROCESSOR_CONNECTION_TYPE_PXB;
        case 30: return CUDTX_PROCESSOR_CONNECTION_TYPE_PHB;
        case 40: return CUDTX_PROCESSOR_CONNECTION_TYPE_NODE;
        case 50: return CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM;
        default: return CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM;
    }
}

static const GPUPeer* findGpuPeer(const GPUInfo& gi, const char* peerBusId) {
    auto sameBus = [](const char* a, const char* b) {
        return busKey(a) == busKey(b);
    };
    for (int k = 0; k < gi.nGPUPeers; k++)
        if (sameBus(gi.gpuPeers[k].busId, peerBusId))
            return &gi.gpuPeers[k];
    return nullptr;
}

static int countDirectNvLinks(const GPUInfo& gi, const char* peerBusId) {
    const GPUPeer* p = findGpuPeer(gi, peerBusId);
    return p ? p->nvLinkCount : 0;
}

static int countNvSwitchLinks(const GPUInfo& src, const GPUInfo& dst) {
    if (!src.hasFabricInfo || !dst.hasFabricInfo)
        return 0;
    if (memcmp(src.clusterUuid, dst.clusterUuid, FABRIC_UUID_SZ) != 0 ||
        src.cliqueId != dst.cliqueId)
        return 0;
    return src.nNvSwitchLinks;
}

static CUDTXprocessorConnectionInfo resolvePcie(const GPUInfo& gi,
                                                 const GPUInfo& gj) {
    int pt = -1;
    const GPUPeer* p = findGpuPeer(gi, gj.busId);
    if (p) pt = p->nvmlTopoLevel;
    if (pt < 0) {
        p = findGpuPeer(gj, gi.busId);
        if (p) pt = p->nvmlTopoLevel;
    }
    CUDTXprocessorConnectionType ct = pcieTopoToConnType(pt);
    float bw = -1.0f;
    {
        float a = gi.pcieBwGBps, b = gj.pcieBwGBps;
        if (a >= 0 && b >= 0)      bw = std::min(a, b);
        else if (a >= 0)            bw = a;
        else if (b >= 0)            bw = b;
    }
    bool atomics = p ? p->atomicsSupported : false;
    return {ct, bw, atomics};
}

// Resolution order matters: NVLink/NVSwitch checks must run before the
// cross-node fallback because systems like NVL72 can have NVLink and
// NVSwitch connections that span across nodes.
//
//  1. Same UUID                                      → X
//  2. Direct NVLink between the two GPUs             → NVLINK
//  3. Indirect NVLink via NVSwitch                   → NVLINK
//  4. Cross-node with no NVLink/NVSwitch             → NET
//  5. Same node, PCIe topology                       → PIX/PXB/PHB/NODE/SYS
static CUDTXprocessorConnectionInfo resolveGpuGpu(
        const GPUInfo& src, const GPUInfo& dst,
        bool sameNode) {

    printf("  [resolveGpuGpu] src=%s (uuid=%s) ↔ dst=%s (uuid=%s) sameNode=%d\n",
           src.busId, cuUuidToStr(src.uuid).c_str(),
           dst.busId, cuUuidToStr(dst.uuid).c_str(), sameNode);

    if (memcmp(&src.uuid, &dst.uuid, sizeof(CUuuid)) == 0)
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_SELF, -1.0f, false};

    const float perLinkBw = src.nvlinkBwPerLinkGBps;

    int nvl = countDirectNvLinks(src, dst.busId);
    if (nvl > 0) {
        printf("    → direct NVLink = %d\n", nvl);
        float bw = (perLinkBw >= 0) ? nvl * perLinkBw : -1.0f;
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK, bw, true};
    }

    int nvs = countNvSwitchLinks(src, dst);
    if (nvs > 0) {
        printf("    → NVSwitch = %d\n", nvs);
        float bw = (perLinkBw >= 0) ? nvs * perLinkBw : -1.0f;
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK, bw, true};
    }

    if (!sameNode) {
        printf("    → NET (cross-node, no NVLink/NVSwitch)\n");
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
    }

    CUDTXprocessorConnectionInfo pcie = resolvePcie(src, dst);
    printf("    → PCIe = %s\n", connTypeTag(pcie.type));
    return pcie;
}

static CUDTXprocessorConnectionInfo resolveGpuCpu(
        const ProcessorInfo& src, const ProcessorInfo& dst,
        bool sameNode) 
{
    const GPUInfo& gpu = src.gpu;
    if (!sameNode) {
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
    }
    if (src.numaId >= 0 && src.numaId == dst.numaId) {
        if (gpu.c2cBwGBps >= 0) {
            return {CUDTX_PROCESSOR_CONNECTION_TYPE_C2C, gpu.c2cBwGBps, false};
        }
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NODE, gpu.pcieBwGBps, false};
    }
    return {CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM, gpu.pcieBwGBps, false};
}

static CUDTXprocessorConnectionInfo resolveCpuCpu(const ProcessorInfo& src, const ProcessorInfo& dst,
                                                   bool sameNode) {
    if (!sameNode) {
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
    }
    if (src.numaId == dst.numaId) {
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_SELF, -1.0f, false};
    }
    return {CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM, -1.0f, false};
}

CUDTXprocessorConnectionInfo MachineManager::resolveNodeConnection(
        const Processor& src, const Processor& dst,
        bool sameNode, const MachineManager& dstMgr) const {

    CUIDTXprocessorType st = src.publicHandle().type;
    CUIDTXprocessorType dt = dst.publicHandle().type;

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuGpu(src.info().gpu, dst.info().gpu, sameNode);

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_CPU)
        return resolveGpuCpu(src.info(), dst.info(), sameNode);

    if (st == CUIDTX_PROCESSOR_TYPE_CPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuCpu(dst.info(), src.info(), sameNode);

    return resolveCpuCpu(src.info(), dst.info(), sameNode);
}

/* ==================================================================
 *  MachineManager::buildTopology
 * ================================================================== */

void MachineManager::buildTopology(const MachineManager& dst) {
    bool sameNode = (hostId_ == dst.hostId());

    for (auto& [srcType, srcVec] : processors_) {
        for (auto& srcProc : srcVec) {
            for (auto& [dstType, dstVec] : dst.processors_) {
                for (auto& dstProc : dstVec) {
                    TopologyNode::Pair nodePair = canonicalPair(srcProc->topologyNode(), dstProc->topologyNode());
                    if (nodePair.first.memberId != memberId_)
                        continue;
                    if (topologyMap_.count(nodePair))
                        continue;

                    CUDTXprocessorConnectionInfo ci = resolveNodeConnection(
                        *srcProc, *dstProc, sameNode, dst);
                    if (ci.type == CUDTX_PROCESSOR_CONNECTION_TYPE_MAX)
                        continue;
                    topologyMap_[nodePair] = ci;
                }
            }
        }
    }
}

/* ==================================================================
 *  queryConnection
 * ================================================================== */

ResultOr<CUDTXprocessorConnectionInfo>
MachineManager::lookupTopology(const TopologyNode &a, const TopologyNode &b) const noexcept
{
    TopologyNode::Pair key = canonicalPair(a, b);
    TopologyMap::const_iterator it = topologyMap_.find(key);
    if (it != topologyMap_.end()) {
        return it->second;
    }
    return CUDTX_ERROR;
}

ResultOr<CUDTXprocessorConnectionInfo>
MachineManager::getTopology(const Processor &src, const Processor &dst) const noexcept
{
    if (src.topologyNode().memberId != memberId_) {
        return CUDTX_ERROR;
    }
    return lookupTopology(src.topologyNode(), dst.topologyNode());
}

std::ostream& operator<<(std::ostream& os, const MachineManager& m) {
    os << "  Member " << std::left << std::setw(3) << m.memberId_
       << " @ 0x" << std::hex << m.hostId_ << std::dec
       << "  " << m.getProcessorsByType(CUIDTX_PROCESSOR_TYPE_GPU).size() << " GPU, "
       << m.getProcessorsByType(CUIDTX_PROCESSOR_TYPE_CPU).size() << " CPU core\n";

    for (const auto& [type, pvec] : m.processors()) {
        for (const auto& proc : pvec)
            os << *proc << '\n';
    }

    os << "\n  Topology (" << m.topologyMap_.size() << " entries):\n";
    std::vector<TopologyNode::Pair> sortedKeys;
    sortedKeys.reserve(m.topologyMap_.size());
    for (const auto& [pair, ci] : m.topologyMap_)
        sortedKeys.push_back(pair);
    std::sort(sortedKeys.begin(), sortedKeys.end());
    for (const auto& pair : sortedKeys) {
        const auto& ci = m.topologyMap_.at(pair);
        os << "    " << pair.first
           << " <-> " << pair.second
           << " : " << connTypeTag(ci.type)
           << "(BW=" << (int)ci.bandwidth
           << ", Atomics=" << (ci.supportAtomics ? "yes" : "no")
           << ")\n";
    }
    return os;
}
