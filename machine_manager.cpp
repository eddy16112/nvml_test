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

static const char* topoTag(int t) {
    switch (t) {
        case 0:  return "X";
        case 10: return "PIX";
        case 20: return "PXB";
        case 30: return "PHB";
        case 40: return "NODE";
        case 50: return "SYS";
        default: return "?";
    }
}

static std::vector<std::string> getNvLinkPeers(const GPUInfo& gi) {
    std::vector<std::string> peers;
    for (int k = 0; k < gi.nNvLinks; k++)
        peers.emplace_back(gi.nvLinks[k].remoteBusId);
    return peers;
}

// NVLink peers are always local devices. A direct GPU-to-GPU NVLink
// can only exist on the same node, so skip the search for cross-node pairs.
static int countNvLinksByBusId(const GPUInfo& gi, const char* peerBusId,
                               bool sameNode) {
    if (!sameNode) return 0;
    int n = 0;
    for (auto& remote : getNvLinkPeers(gi))
        if (sameBus(remote.c_str(), peerBusId)) n++;
    return n;
}

// NVLink peers are local devices. To tell GPUs from NVSwitches we only
// compare against GPUs on the SAME node (identified by hostname), avoiding
// false matches when different nodes have identical GPU bus IDs.
// NVSwitch bus ID keys are NOT prefixed with hostname because NVSwitch
// ASICs are physically shared across compute trays on systems like NVL72;
// two GPUs on different nodes that connect to the same NVSwitch will
// report the same NVSwitch bus ID, and that match is intentional.
static int countNvSwitchLinks(const GPUInfo& gi, const GPUInfo& gj,
                              const MachineManager& srcMgr,
                              const MachineManager& dstMgr) {
    auto isGpuOnMgr = [](const char* bid, const MachineManager& mgr) -> bool {
        for (auto& p : mgr.gpus())
            if (sameBus(bid, p->info_.gpu.busId)) return true;
        return false;
    };

    std::map<std::string, int> swI, swJ;
    for (auto& remote : getNvLinkPeers(gi))
        if (!isGpuOnMgr(remote.c_str(), srcMgr))
            swI[busKey(remote.c_str())]++;
    for (auto& remote : getNvLinkPeers(gj))
        if (!isGpuOnMgr(remote.c_str(), dstMgr))
            swJ[busKey(remote.c_str())]++;

    int n = 0;
    for (auto& kv : swI)
        if (swJ.count(kv.first)) n += kv.second;
    return n;
}

static std::string resolvePcie(const GPUInfo& gi, const GPUInfo& gj) {
    int pt = -1;
    for (int k = 0; k < gi.nPcies && pt < 0; k++)
        if (sameBus(gi.pcies[k].busId, gj.busId))
            pt = gi.pcies[k].nvmlTopoLevel;
    for (int k = 0; k < gj.nPcies && pt < 0; k++)
        if (sameBus(gj.pcies[k].busId, gi.busId))
            pt = gj.pcies[k].nvmlTopoLevel;
    return topoTag(pt);
}

// Resolution order matters: NVLink/NVSwitch checks must run before the
// cross-node fallback because systems like NVL72 can have NVLink and
// NVSwitch connections that span across nodes.
//
//  1. Same PCI bus (same GPU, same node only)        → X
//  2. Direct NVLink between the two GPUs             → NVx
//  3. Indirect NVLink via NVSwitch                   → NVx
//  4. Cross-node with no NVLink/NVSwitch             → NET
//  5. Same node, PCIe topology                       → PIX/PXB/PHB/NODE/SYS
static std::string resolveGpuGpu(
        const GPUInfo& gi, const GPUInfo& gj,
        bool sameNode,
        const MachineManager& srcMgr, const MachineManager& dstMgr) {

    printf("  [resolveGpuGpu] src=%s (uuid=%.40s) ↔ dst=%s (uuid=%.40s) sameNode=%d\n",
           gi.busId, gi.uuid, gj.busId, gj.uuid, sameNode);

    if (sameNode && sameBus(gi.busId, gj.busId))
        return "X";

    int nvl = countNvLinksByBusId(gi, gj.busId, sameNode);
    if (nvl > 0) {
        printf("    → direct NVLink = %d\n", nvl);
        return "NV" + std::to_string(nvl);
    }

    int nvs = countNvSwitchLinks(gi, gj, srcMgr, dstMgr);
    if (nvs > 0) {
        printf("    → NVSwitch = %d\n", nvs);
        return "NV" + std::to_string(nvs);
    }

    if (!sameNode) {
        printf("    → NET (cross-node, no NVLink/NVSwitch)\n");
        return "NET";
    }

    std::string pcie = resolvePcie(gi, gj);
    printf("    → PCIe = %s\n", pcie.c_str());
    return pcie;
}

static std::string resolveGpuCpu(int gpuNumaId, int cpuNumaId, bool sameNode) {
    if (!sameNode)
        return "NET";
    if (gpuNumaId >= 0 && gpuNumaId == cpuNumaId)
        return "NODE";
    return "SYS";
}

static std::string resolveCpuCpu(int numaA, int numaB, bool sameNode) {
    if (numaA == numaB && sameNode)
        return "X";
    if (!sameNode)
        return "NET";
    return "SYS";
}

static std::string resolveNodeConnection(
        const Processor& src, const Processor& dst,
        bool sameNode,
        const MachineManager& srcMgr, const MachineManager& dstMgr) {

    CUIDTXprocessorType st = src.handle_.type;
    CUIDTXprocessorType dt = dst.handle_.type;

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuGpu(src.info_.gpu, dst.info_.gpu, sameNode, srcMgr, dstMgr);

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_CPU)
        return resolveGpuCpu(src.info_.numaId, dst.info_.numaId, sameNode);

    if (st == CUIDTX_PROCESSOR_TYPE_CPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuCpu(dst.info_.numaId, src.info_.numaId, sameNode);

    return resolveCpuCpu(src.info_.numaId, dst.info_.numaId, sameNode);
}

/* ==================================================================
 *  MachineManager::buildTopology
 * ================================================================== */

void MachineManager::buildTopology(const MachineManager& remote) {
    bool sameNode = (std::string(remote.hostname) == std::string(hostname));

    for (auto& [stype, svec] : processors_) {
        for (auto& sp : svec) {
            for (auto& [dtype, dvec] : remote.processors_) {
                for (auto& dp : dvec) {
                    TopologyNode::Pair cp = canonicalPair(sp->topologyNode_, dp->topologyNode_);
                    if (cp.first.rank != rank)
                        continue;
                    if (topology.count(cp))
                        continue;

                    topology[cp] = resolveNodeConnection(
                        *sp, *dp, sameNode, *this, remote);
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

std::string MachineManager::query(const CUIDTXprocessor& a,
                                  const CUIDTXprocessor& b) const {
    TopologyNode ta = toTopoNode(a);
    TopologyNode tb = toTopoNode(b);
    auto it = topology.find(canonicalPair(ta, tb));
    return (it != topology.end()) ? it->second : "";
}

std::string queryConnection(const std::vector<MachineManager>& managers,
                            const CUIDTXprocessor& a,
                            const CUIDTXprocessor& b) {
    int owner = std::min(a.rank, b.rank);
    if (owner < 0 || owner >= (int)managers.size())
        return "";
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

    /* ---- 2. Build printable connection matrix from topology maps ---- */
    std::vector<std::vector<std::string>> conn(N, std::vector<std::string>(N));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) { conn[i][j] = "X"; continue; }

            TopologyNode::Pair cp = canonicalPair(G[i].tnode, G[j].tnode);
            int owner = cp.first.rank;

            auto it = managers[owner].topology.find(cp);
            if (it != managers[owner].topology.end()) {
                conn[i][j] = it->second;
            } else {
                conn[i][j] = "?";
            }
        }
    }

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
        for (auto& [pair, conn] : M.topology) {
            printf("    %s <-> %s : %s\n",
                   topoNodeStr(pair.first).c_str(),
                   topoNodeStr(pair.second).c_str(),
                   conn.c_str());
        }
    }

    /* ---- Topology Matrix ---- */
    printf("\n--- Topology Matrix ---\n");
    printf("  Legend:  NVx = NVLink (x links)      PIX = single PCIe switch\n");
    printf("          PXB = multi PCIe switch       PHB = host bridge\n");
    printf("          NODE = same NUMA              SYS = cross-NUMA\n");
    printf("          NET  = cross-node\n\n");

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
            if (conn[i][j].size() >= 2 && conn[i][j][0] == 'N' && conn[i][j][1] == 'V') {
                printf("  %s <-> %s : %-6s",
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
