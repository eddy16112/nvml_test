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

void MachineManager::clearAll() {
    processors_.clear();
    topologyMap_.clear();
}

/* ==================================================================
 *  Helpers
 * ================================================================== */

inline bool sameBus(const char* a, const char* b) 
{
    return busKey(a) == busKey(b);
}

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

// Direct GPU-to-GPU NVLink can only exist on the same node.
static int countDirectNvLinks(const GPUInfo& gi, const char* peerBusId,
                              bool sameNode) {
    if (!sameNode) return 0;
    int n = 0;
    for (int k = 0; k < gi.nNvLinks; k++)
        if (gi.nvLinks[k].remoteDeviceType == NVML_NVLINK_DEVICE_TYPE_GPU &&
            sameBus(gi.nvLinks[k].remoteBusId, peerBusId))
            n++;
    return n;
}

// Count NVLink connections routed through shared NVSwitches.
// Two GPUs on different nodes can share the same physical NVSwitch
// (e.g. NVL72), so cross-node NVSwitch matches are intentional.
static int countNvSwitchLinks(const GPUInfo& src, const GPUInfo& dst) {
    std::map<std::string, int> swSrc, swDst;
    for (int k = 0; k < src.nNvLinks; k++)
        if (src.nvLinks[k].remoteDeviceType == NVML_NVLINK_DEVICE_TYPE_SWITCH)
            swSrc[busKey(src.nvLinks[k].remoteBusId)]++;
    for (int k = 0; k < dst.nNvLinks; k++)
        if (dst.nvLinks[k].remoteDeviceType == NVML_NVLINK_DEVICE_TYPE_SWITCH)
            swDst[busKey(dst.nvLinks[k].remoteBusId)]++;

    int n = 0;
    for (const std::pair<const std::string, int>& kv : swSrc)
        if (swDst.count(kv.first)) n += kv.second;
    return n;
}

static bool lookupAtomics(const GPUInfo& src, const char* peerBusId) {
    for (int k = 0; k < src.nPcies; k++)
        if (sameBus(src.pcies[k].busId, peerBusId))
            return src.pcies[k].atomicsSupported;
    return false;
}

static CUDTXprocessorConnectionInfo resolvePcie(const GPUInfo& gi,
                                                 const GPUInfo& gj) {
    int pt = -1;
    for (int k = 0; k < gi.nPcies && pt < 0; k++)
        if (sameBus(gi.pcies[k].busId, gj.busId))
            pt = gi.pcies[k].nvmlTopoLevel;
    for (int k = 0; k < gj.nPcies && pt < 0; k++)
        if (sameBus(gj.pcies[k].busId, gi.busId))
            pt = gj.pcies[k].nvmlTopoLevel;
    CUDTXprocessorConnectionType ct = pcieTopoToConnType(pt);
    float bw = -1.0f;
    if (ct == CUDTX_PROCESSOR_CONNECTION_TYPE_PIX || ct == CUDTX_PROCESSOR_CONNECTION_TYPE_PXB || ct == CUDTX_PROCESSOR_CONNECTION_TYPE_PHB) {
        float a = gi.pcieBwGBps, b = gj.pcieBwGBps;
        if (a >= 0 && b >= 0)      bw = std::min(a, b);
        else if (a >= 0)            bw = a;
        else if (b >= 0)            bw = b;
    }
    bool atomics = lookupAtomics(gi, gj.busId);
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
    bool atomics = lookupAtomics(src, dst.busId);

    int nvl = countDirectNvLinks(src, dst.busId, sameNode);
    if (nvl > 0) {
        printf("    → direct NVLink = %d\n", nvl);
        float bw = (perLinkBw >= 0) ? nvl * perLinkBw : -1.0f;
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK, bw, atomics};
    }

    int nvs = countNvSwitchLinks(src, dst);
    if (nvs > 0) {
        printf("    → NVSwitch = %d\n", nvs);
        float bw = (perLinkBw >= 0) ? nvs * perLinkBw : -1.0f;
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK, bw, atomics};
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
        if (gpu.hasC2C) {
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
    bool sameNode = (std::string(dst.hostname()) == std::string(hostname()));

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

static TopologyNode toTopoNode(const CUIDTXprocessor& h) {
    int localId = (h.type == CUIDTX_PROCESSOR_TYPE_GPU)
                  ? h.gpu.deviceOrdinal
                  : h.cpu.cpuOrdinal;
    return TopologyNode(h.memberId, h.type, localId);
}

CUDTXprocessorConnectionInfo MachineManager::query(
        const CUIDTXprocessor& a, const CUIDTXprocessor& b) const {
    TopologyNode ta = toTopoNode(a);
    TopologyNode tb = toTopoNode(b);
    auto it = topologyMap_.find(canonicalPair(ta, tb));
    if (it != topologyMap_.end()) return it->second;
    return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
}

std::ostream& operator<<(std::ostream& os, const MachineManager& m) {
    os << "  Member " << std::left << std::setw(3) << m.memberId_
       << " @ " << std::setw(20) << m.hostname_
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
