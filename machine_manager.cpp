#include "machine_manager.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cassert>


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

inline std::string busKey(const char* id) 
{
    std::string s(id);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    auto p = s.find(':');
    if (p != std::string::npos && p < 8)
        s = std::string(8 - p, '0') + s;
    return s;
}

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

    printf("  [resolveGpuGpu] src=%s (uuid=%.40s) ↔ dst=%s (uuid=%.40s) sameNode=%d\n",
           src.busId, src.uuid, dst.busId, dst.uuid, sameNode);

    if (strcmp(src.uuid, dst.uuid) == 0)
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
    bool sameNode = (std::string(dst.hostname_) == std::string(hostname_));

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

CUDTXprocessorConnectionInfo queryConnection(
        const std::vector<MachineManager>& managers,
        const CUIDTXprocessor& a, const CUIDTXprocessor& b) {
    uint32_t owner = std::min(a.memberId, b.memberId);
    if (owner < 0 || owner >= (int)managers.size())
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
    return managers[owner].query(a, b);
}

/* ==================================================================
 *  printTopology
 * ================================================================== */

struct GNode {
    TopologyNode tnode;
    std::string host;
    std::string uuid, busId, name;
    int numaId = -1;
    std::vector<int> ownerRanks;

    bool isGpu() const { return tnode.type == CUIDTX_PROCESSOR_TYPE_GPU; }

    std::string nodeKey() const {
        if (isGpu()) return uuid;
        return host + ":numa" + std::to_string(numaId);
    }
};

void MachineManager::print() const {
    printf("  Member %-3u @ %-20s  %d GPU, %d CPU core\n",
           memberId_, hostname_,
           (int)gpus().size(), (int)cpus().size());
    for (const std::unique_ptr<Processor>& p : gpus()) {
        std::string hl = handleStr(p->publicHandle());
        const GPUInfo& gi = p->info().gpu;
        printf("    %-12s  %s [%s]",
               hl.c_str(), gi.busId, gi.name);
        if (p->info().numaId >= 0)
            printf(" NUMA:%d", p->info().numaId);
        printf("\n");
    }
    for (const std::unique_ptr<Processor>& p : cpus()) {
        std::string hl = handleStr(p->publicHandle());
        printf("    %-12s  NUMA %d  os_index %u\n",
               hl.c_str(), p->info().numaId, p->info().cpu.osIndex);
    }
    printf("\n  Topology (%zu entries):\n", topologyMap_.size());
    for (const auto& [pair, ci] : topologyMap_) {
        std::string tag = connInfoStr(ci);
        printf("    %s <-> %s : %s\n",
               topoNodeStr(pair.first).c_str(),
               topoNodeStr(pair.second).c_str(),
               tag.c_str());
    }
}

void printTopology(const std::vector<MachineManager>& managers) {
    const int ws = (int)managers.size();

    /* ---- 1. Deduplicated global node list ---- */
    std::vector<GNode> G;
    std::map<std::string, int> key2g;

    for (int r = 0; r < ws; r++) {
        const MachineManager& M = managers[r];
        for (const auto& [type, pvec] : M.processors()) {
            for (auto& np : pvec) {
                GNode gn;
                gn.tnode  = np->topologyNode();
                gn.host   = M.hostname_;

                if (gn.isGpu()) {
                    const GPUInfo& gi = np->info().gpu;
                    gn.uuid    = gi.uuid;
                    gn.busId   = gi.busId;
                    gn.name    = gi.name;
                }
                gn.numaId = np->info().numaId;

                std::string k = gn.nodeKey();
                if (key2g.count(k) == 0) {
                    key2g[k] = (int)G.size();
                    G.push_back(gn);
                }
                G[key2g[k]].ownerRanks.push_back(r);
            }
        }
    }

    std::sort(G.begin(), G.end(), [](const GNode& a, const GNode& b) {
        if (a.host != b.host) return a.host < b.host;
        if (a.isGpu() != b.isGpu()) return a.isGpu() > b.isGpu();
        if (a.isGpu())
            return busKey(a.busId.c_str()) < busKey(b.busId.c_str());
        return a.numaId < b.numaId;
    });
    key2g.clear();
    for (int i = 0; i < (int)G.size(); i++) key2g[G[i].nodeKey()] = i;

    int nGpus = 0, nCpus = 0;
    for (auto& g : G) { if (g.isGpu()) nGpus++; else nCpus++; }
    const int N = (int)G.size();

    /* ---- 2. Build connection info matrix from topology maps ---- */
    CUDTXprocessorConnectionInfo defaultConn = {CUDTX_PROCESSOR_CONNECTION_TYPE_SELF, -1.0f, false};
    std::vector<std::vector<CUDTXprocessorConnectionInfo>> cinfo(
        N, std::vector<CUDTXprocessorConnectionInfo>(N, defaultConn));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            TopologyNode::Pair nodePair = canonicalPair(G[i].tnode, G[j].tnode);
            int owner = nodePair.first.memberId;
            auto it = managers[owner].topologyMap().find(nodePair);
            if (it != managers[owner].topologyMap().end())
                cinfo[i][j] = it->second;
            else
                cinfo[i][j] = {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f, false};
        }
    }

    // short string matrix for display
    std::vector<std::vector<std::string>> conn(N, std::vector<std::string>(N));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            conn[i][j] = connInfoStr(cinfo[i][j]);

    /* ---- 3. Print ---- */
    printf("\n");
    printf("=========================================================================\n");
    printf("                    GLOBAL TOPOLOGY REPORT\n");
    printf("=========================================================================\n");
    printf("  %d rank(s),  %d GPU(s),  %d CPU NUMA node(s)\n", ws, nGpus, nCpus);
    printf("  (mode: pre-collected data only)\n");

    /* ---- All Unique Nodes ---- */
    printf("\n--- All Unique Nodes (%d GPU, %d CPU NUMA) ---\n\n", nGpus, nCpus);
    for (int i = 0; i < N; i++) {
        const GNode& g = G[i];
        std::string label = topoNodeStr(g.tnode);
        if (g.isGpu()) {
            printf("  %-12s  %-16s  %-24s  %-16s",
                   label.c_str(), g.busId.c_str(), g.name.c_str(),
                   g.host.c_str());
            if (g.numaId >= 0) printf("  NUMA:%d", g.numaId);
        } else {
            printf("  %-12s  %-16s  %-24s  %-16s  NUMA %d",
                   label.c_str(), "", "", g.host.c_str(), g.numaId);
        }
        printf("  Rank:");
        for (size_t k = 0; k < g.ownerRanks.size(); k++)
            printf(" %d", g.ownerRanks[k]);
        printf("\n");
    }

    /* ---- Per-Manager Details ---- */
    printf("\n--- Per-Manager Details ---\n");
    for (int r = 0; r < ws; r++) {
        managers[r].print();
    }

    /* ---- Topology Matrix ---- */
    printf("\n--- Topology Matrix ---\n");
    printf("  Legend:  NVL = NVLink                  PIX = single PCIe switch\n");
    printf("          PXB = multi PCIe switch       PHB = host bridge\n");
    printf("          C2C = NVLink-C2C (GPU-CPU)    NODE = same NUMA\n");
    printf("          SYS = cross-NUMA              NET  = cross-node\n");
    printf("          (N) = bandwidth in GB/s    [A] = atomics supported\n\n");

    std::vector<std::string> labels(N);
    int maxLabelLen = 0;
    for (int i = 0; i < N; i++) {
        labels[i] = topoNodeStr(G[i].tnode);
        maxLabelLen = std::max(maxLabelLen, (int)labels[i].size());
    }
    int cw = std::max(maxLabelLen + 2, 8);
    int rw = maxLabelLen + 2;

    printf("%-*s", rw, "");
    for (int j = 0; j < N; j++) printf("%-*s", cw, labels[j].c_str());
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("%-*s", rw, labels[i].c_str());
        for (int j = 0; j < N; j++)
            printf("%-*s", cw, conn[i][j].c_str());
        printf("  [%s]\n", G[i].host.c_str());
    }

    /* ---- NVLink Summary ---- */
    printf("\n--- NVLink Summary ---\n\n");
    bool any = false;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (cinfo[i][j].type == CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK) {
                printf("  %s <-> %s : %-12s",
                       labels[i].c_str(), labels[j].c_str(),
                       conn[i][j].c_str());
                if (G[i].isGpu() && G[j].isGpu())
                    printf("  (%s <-> %s)", G[i].busId.c_str(), G[j].busId.c_str());
                printf("\n");
                any = true;
            }
    if (!any) printf("  (no NVLink connections detected)\n");

    printf("\n=========================================================================\n\n");
}
