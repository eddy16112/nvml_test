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
        processors.emplace_back(std::make_unique<Processor>(info, rank));
    }
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

static CUIDTXTopologyConnectionInfo resolvePcie(const GPUInfo& gi,
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
    return {ct, bw};
}

// Resolution order matters: NVLink/NVSwitch checks must run before the
// cross-node fallback because systems like NVL72 can have NVLink and
// NVSwitch connections that span across nodes.
//
//  1. Same PCI bus (same GPU, same node only)        → X
//  2. Direct NVLink between the two GPUs             → NVLINK
//  3. Indirect NVLink via NVSwitch                   → NVLINK
//  4. Cross-node with no NVLink/NVSwitch             → NET
//  5. Same node, PCIe topology                       → PIX/PXB/PHB/NODE/SYS
static CUIDTXTopologyConnectionInfo resolveGpuGpu(
        const GPUInfo& src, const GPUInfo& dst,
        bool sameNode) {

    printf("  [resolveGpuGpu] src=%s (uuid=%.40s) ↔ dst=%s (uuid=%.40s) sameNode=%d\n",
           src.busId, src.uuid, dst.busId, dst.uuid, sameNode);

    if (strcmp(src.uuid, dst.uuid) == 0)
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_SELF, -1.0f};

    float perLink = src.nvlinkBwPerLinkGBps;
    auto nvlinkBw = [perLink](int linkCount) -> float {
        return (perLink >= 0) ? linkCount * perLink : -1.0f;
    };

    int nvl = countDirectNvLinks(src, dst.busId, sameNode);
    if (nvl > 0) {
        printf("    → direct NVLink = %d\n", nvl);
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK, nvlinkBw(nvl)};
    }

    int nvs = countNvSwitchLinks(src, dst);
    if (nvs > 0) {
        printf("    → NVSwitch = %d\n", nvs);
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NVLINK, nvlinkBw(nvs)};
    }

    if (!sameNode) {
        printf("    → NET (cross-node, no NVLink/NVSwitch)\n");
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f};
    }

    CUIDTXTopologyConnectionInfo pcie = resolvePcie(src, dst);
    printf("    → PCIe = %s\n", connTypeTag(pcie.type));
    return pcie;
}

static CUIDTXTopologyConnectionInfo resolveGpuCpu(
        const ProcessorInfo& src, const ProcessorInfo& dst,
        bool sameNode) 
{
    const GPUInfo& gpu = src.gpu;
    if (!sameNode) {
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f};
    }
    if (src.numaId >= 0 && src.numaId == dst.numaId) {
        if (gpu.hasC2C) {
            return {CUDTX_PROCESSOR_CONNECTION_TYPE_C2C, gpu.c2cBwGBps};
        }
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_NODE, gpu.pcieBwGBps};
    }
    return {CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM, gpu.pcieBwGBps};
}

static CUIDTXTopologyConnectionInfo resolveCpuCpu(const ProcessorInfo& src, const ProcessorInfo& dst,
                                                   bool sameNode) {
    if (!sameNode) {
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f};
    }
    if (src.numaId == dst.numaId) {
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_SELF, -1.0f};
    }
    return {CUDTX_PROCESSOR_CONNECTION_TYPE_SYSTEM, -1.0f};
}

CUIDTXTopologyConnectionInfo MachineManager::resolveNodeConnection(
        const Processor& src, const Processor& dst,
        bool sameNode, const MachineManager& dstMgr) const {

    CUIDTXprocessorType st = src.handle_.type;
    CUIDTXprocessorType dt = dst.handle_.type;

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuGpu(src.info_.gpu, dst.info_.gpu, sameNode);

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_CPU)
        return resolveGpuCpu(src.info_, dst.info_, sameNode);

    if (st == CUIDTX_PROCESSOR_TYPE_CPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuCpu(dst.info_, src.info_, sameNode);

    return resolveCpuCpu(src.info_, dst.info_, sameNode);
}

/* ==================================================================
 *  MachineManager::buildTopology
 * ================================================================== */

void MachineManager::buildTopology(const MachineManager& dst) {
    bool sameNode = (std::string(dst.hostname) == std::string(hostname));

    for (auto& [srcType, srcVec] : processors_) {
        for (auto& srcProc : srcVec) {
            for (auto& [dstType, dstVec] : dst.processors_) {
                for (auto& dstProc : dstVec) {
                    TopologyNode::Pair nodePair = canonicalPair(srcProc->topologyNode_, dstProc->topologyNode_);
                    if (nodePair.first.rank != rank)
                        continue;
                    if (topology.count(nodePair))
                        continue;

                    CUIDTXTopologyConnectionInfo ci = resolveNodeConnection(
                        *srcProc, *dstProc, sameNode, dst);
                    if (ci.type == CUDTX_PROCESSOR_CONNECTION_TYPE_MAX)
                        continue;
                    topology[nodePair] = ci;
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
                  ? h.gpu.deviceId
                  : h.cpu.cpuOrdinal;
    return TopologyNode(h.rank, h.type, localId);
}

CUIDTXTopologyConnectionInfo MachineManager::query(
        const CUIDTXprocessor& a, const CUIDTXprocessor& b) const {
    TopologyNode ta = toTopoNode(a);
    TopologyNode tb = toTopoNode(b);
    auto it = topology.find(canonicalPair(ta, tb));
    if (it != topology.end()) return it->second;
    return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f};
}

CUIDTXTopologyConnectionInfo queryConnection(
        const std::vector<MachineManager>& managers,
        const CUIDTXprocessor& a, const CUIDTXprocessor& b) {
    int owner = std::min(a.rank, b.rank);
    if (owner < 0 || owner >= (int)managers.size())
        return {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f};
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

void printTopology(const std::vector<MachineManager>& managers) {
    const int ws = (int)managers.size();

    /* ---- 1. Deduplicated global node list ---- */
    std::vector<GNode> G;
    std::map<std::string, int> key2g;

    for (int r = 0; r < ws; r++) {
        const MachineManager& M = managers[r];
        for (auto& [type, pvec] : M.processors_) {
            for (auto& np : pvec) {
                GNode gn;
                gn.tnode  = np->topologyNode_;
                gn.host   = M.hostname;

                if (gn.isGpu()) {
                    const GPUInfo& gi = np->info_.gpu;
                    gn.uuid    = gi.uuid;
                    gn.busId   = gi.busId;
                    gn.name    = gi.name;
                }
                gn.numaId = np->info_.numaId;

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
    CUIDTXTopologyConnectionInfo defaultConn = {CUDTX_PROCESSOR_CONNECTION_TYPE_SELF, -1.0f};
    std::vector<std::vector<CUIDTXTopologyConnectionInfo>> cinfo(
        N, std::vector<CUIDTXTopologyConnectionInfo>(N, defaultConn));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            TopologyNode::Pair nodePair = canonicalPair(G[i].tnode, G[j].tnode);
            int owner = nodePair.first.rank;
            auto it = managers[owner].topology.find(nodePair);
            if (it != managers[owner].topology.end())
                cinfo[i][j] = it->second;
            else
                cinfo[i][j] = {CUDTX_PROCESSOR_CONNECTION_TYPE_MAX, -1.0f};
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

    /* ---- Per-Rank Assignment ---- */
    printf("\n--- Per-Rank Assignment ---\n\n");
    for (int r = 0; r < ws; r++) {
        const MachineManager& M = managers[r];
        printf("  Rank %-3d @ %-20s  %d GPU, %d CPU core\n",
               r, M.hostname,
               (int)M.gpus().size(), (int)M.cpus().size());
        for (auto& p : M.gpus()) {
            std::string hl = handleStr(p->handle_);
            const GPUInfo& gi = p->info_.gpu;
            printf("    %-12s  %s [%s]",
                   hl.c_str(), gi.busId, gi.name);
            if (p->info_.numaId >= 0)
                printf(" NUMA:%d", p->info_.numaId);
            printf("\n");
        }
        for (auto& p : M.cpus()) {
            std::string hl = handleStr(p->handle_);
            printf("    %-12s  NUMA %d  os_index %u\n",
                   hl.c_str(), p->info_.numaId, p->info_.cpu.osIndex);
        }
    }

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

    /* ---- Per-Manager Topology Maps ---- */
    printf("\n--- Per-Manager Topology Maps ---\n");
    for (int r = 0; r < ws; r++) {
        const auto& M = managers[r];
        printf("\n  Manager[%d] @ %s  (%zu entries)\n",
               r, M.hostname, M.topology.size());
        for (auto& [pair, ci] : M.topology) {
            std::string tag = connInfoStr(ci);
            printf("    %s <-> %s : %s\n",
                   topoNodeStr(pair.first).c_str(),
                   topoNodeStr(pair.second).c_str(),
                   tag.c_str());
        }
    }

    /* ---- Topology Matrix ---- */
    printf("\n--- Topology Matrix ---\n");
    printf("  Legend:  NVL = NVLink                  PIX = single PCIe switch\n");
    printf("          PXB = multi PCIe switch       PHB = host bridge\n");
    printf("          C2C = NVLink-C2C (GPU-CPU)    NODE = same NUMA\n");
    printf("          SYS = cross-NUMA              NET  = cross-node\n");
    printf("          (N) = bandwidth in GB/s\n\n");

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
