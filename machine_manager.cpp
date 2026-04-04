#include "machine_manager.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#ifdef PHASE3_USE_NVML
#include <nvml.h>
#endif

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

static int countNvLinksByBusId(const GpuInfo& gi, const char* peerBusId) {
    int n = 0;
    for (int k = 0; k < gi.nNvLinks; k++)
        if (gi.nvLinks[k].active && sameBus(gi.nvLinks[k].remoteBusId, peerBusId))
            n++;
    return n;
}

static int countNvSwitchLinks(const GpuInfo& gi, const GpuInfo& gj,
                              const MachineManager& srcMgr,
                              const MachineManager& dstMgr) {
    auto isKnownGpu = [&](const char* bid) -> bool {
        for (auto& n : srcMgr.gpus())
            if (sameBus(bid, n.gpu.busId)) return true;
        for (auto& n : dstMgr.gpus())
            if (sameBus(bid, n.gpu.busId)) return true;
        return false;
    };

    std::map<std::string, int> swI, swJ;
    for (int k = 0; k < gi.nNvLinks; k++)
        if (gi.nvLinks[k].active && !isKnownGpu(gi.nvLinks[k].remoteBusId))
            swI[busKey(gi.nvLinks[k].remoteBusId)]++;
    for (int k = 0; k < gj.nNvLinks; k++)
        if (gj.nvLinks[k].active && !isKnownGpu(gj.nvLinks[k].remoteBusId))
            swJ[busKey(gj.nvLinks[k].remoteBusId)]++;

    int n = 0;
    for (auto& kv : swI)
        if (swJ.count(kv.first)) n += kv.second;
    return n;
}

#ifdef PHASE3_USE_NVML
static int queryPcieTopo(const char* busIdA, const char* busIdB) {
    nvmlDevice_t a, b;
    if (nvmlDeviceGetHandleByPciBusId_v2(busIdA, &a) != NVML_SUCCESS) return -1;
    if (nvmlDeviceGetHandleByPciBusId_v2(busIdB, &b) != NVML_SUCCESS) return -1;
    nvmlGpuTopologyLevel_t lvl;
    if (nvmlDeviceGetTopologyCommonAncestor(a, b, &lvl) != NVML_SUCCESS) return -1;
    return (int)lvl;
}
#endif

static std::string resolvePcie(const GpuInfo& gi, const GpuInfo& gj) {
#ifdef PHASE3_USE_NVML
    return topoTag(queryPcieTopo(gi.busId, gj.busId));
#else
    int pt = -1;
    for (int k = 0; k < gi.nPeerTopos && pt < 0; k++)
        if (sameBus(gi.peerTopos[k].busId, gj.busId))
            pt = gi.peerTopos[k].pcieTopo;
    for (int k = 0; k < gj.nPeerTopos && pt < 0; k++)
        if (sameBus(gj.peerTopos[k].busId, gi.busId))
            pt = gj.peerTopos[k].pcieTopo;
    return topoTag(pt);
#endif
}

static std::string resolveGpuGpu(
        const GpuInfo& gi, const GpuInfo& gj,
        bool sameNode,
        const MachineManager& srcMgr, const MachineManager& dstMgr) {

    if (sameNode && sameBus(gi.busId, gj.busId))
        return "X";

    int nvl = countNvLinksByBusId(gi, gj.busId);
    if (nvl > 0)
        return "NV" + std::to_string(nvl);

    int nvs = countNvSwitchLinks(gi, gj, srcMgr, dstMgr);
    if (nvs > 0)
        return "NV" + std::to_string(nvs);

    if (!sameNode)
        return "NET";

    return resolvePcie(gi, gj);
}

static std::string resolveGpuCpu(const GpuInfo& g, const CpuInfo& c,
                                 bool sameNode) {
    if (!sameNode)
        return "NET";
    if (g.numaId >= 0 && g.numaId == c.numaId)
        return "NODE";
    return "SYS";
}

static std::string resolveCpuCpu(const CpuInfo& ci, const CpuInfo& cj,
                                 bool sameNode) {
    if (ci.numaId == cj.numaId && sameNode)
        return "X";
    if (!sameNode)
        return "NET";
    return "SYS";
}

static std::string resolveNodeConnection(
        const TopologyNode& src, const TopologyNode& dst,
        bool sameNode,
        const MachineManager& srcMgr, const MachineManager& dstMgr) {

    CUIDTXprocessorType st = src.handle.type;
    CUIDTXprocessorType dt = dst.handle.type;

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuGpu(src.gpu, dst.gpu, sameNode, srcMgr, dstMgr);

    if (st == CUIDTX_PROCESSOR_TYPE_GPU && dt == CUIDTX_PROCESSOR_TYPE_CPU)
        return resolveGpuCpu(src.gpu, dst.cpu, sameNode);

    if (st == CUIDTX_PROCESSOR_TYPE_CPU && dt == CUIDTX_PROCESSOR_TYPE_GPU)
        return resolveGpuCpu(dst.gpu, src.cpu, sameNode);

    return resolveCpuCpu(src.cpu, dst.cpu, sameNode);
}

/* ==================================================================
 *  collectAllNodes
 * ================================================================== */

void collectAllNodes(const MachineManager& M,
                     std::vector<const TopologyNode*>& out) {
    for (auto& n : M.gpus()) out.push_back(&n);
    for (auto& n : M.cpus()) out.push_back(&n);
}

/* ==================================================================
 *  MachineManager::buildTopology
 * ================================================================== */

void MachineManager::buildTopology(const MachineManager& remote) {
    bool sameNode = (std::string(remote.hostname) == std::string(hostname));

    std::vector<const TopologyNode*> srcNodes, dstNodes;
    collectAllNodes(*this, srcNodes);
    collectAllNodes(remote, dstNodes);

    for (auto* srcNode : srcNodes) {
        for (auto* dstNode : dstNodes) {
            CUIDTXprocessorPair cp = canonicalPair(srcNode->handle, dstNode->handle);
            if (cp.first.rank != rank)
                continue;
            if (topology.count(cp))
                continue;

            std::string conn = resolveNodeConnection(
                *srcNode, *dstNode,
                sameNode, *this, remote);

            topology[cp] = conn;
        }
    }
}

/* ==================================================================
 *  queryConnection
 * ================================================================== */

std::string queryConnection(const std::vector<MachineManager>& managers,
                            const CUIDTXprocessor& a,
                            const CUIDTXprocessor& b) {
    CUIDTXprocessorPair cp = canonicalPair(a, b);
    int owner = cp.first.rank;
    if (owner < 0 || owner >= (int)managers.size())
        return "";
    return managers[owner].query(a, b);
}

/* ==================================================================
 *  printTopology
 * ================================================================== */

struct GNode {
    CUIDTXprocessor handle;
    std::string host;
    std::string uuid, busId, name;
    int ccMajor = 0, ccMinor = 0;
    uint64_t memMB = 0;
    int pcieGen = 0, pcieWidth = 0;
    int numaId = -1;
    std::vector<int> ownerRanks;

    bool isGpu() const { return handle.type == CUIDTX_PROCESSOR_TYPE_GPU; }

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
        std::vector<const TopologyNode*> rnodes;
        collectAllNodes(M, rnodes);
        for (auto* np : rnodes) {
            GNode gn;
            gn.handle = np->handle;
            gn.host   = M.hostname;

            if (gn.isGpu()) {
                const GpuInfo& gi = np->gpu;
                gn.uuid    = gi.uuid;
                gn.busId   = gi.busId;
                gn.name    = gi.name;
                gn.ccMajor = gi.ccMajor;
                gn.ccMinor = gi.ccMinor;
                gn.memMB   = gi.memMB;
                gn.pcieGen = gi.pcieGen;
                gn.pcieWidth = gi.pcieWidth;
                gn.numaId  = gi.numaId;
            } else {
                const CpuInfo& ci = np->cpu;
                gn.numaId = ci.numaId;
            }

            std::string k = gn.nodeKey();
            if (key2g.count(k) == 0) {
                key2g[k] = (int)G.size();
                G.push_back(gn);
            }
            G[key2g[k]].ownerRanks.push_back(r);
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

            const CUIDTXprocessor& hi = G[i].handle;
            const CUIDTXprocessor& hj = G[j].handle;
            CUIDTXprocessorPair cp = canonicalPair(hi, hj);
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
#ifdef PHASE3_USE_NVML
    printf("  (mode: NVML live query)\n");
#else
    printf("  (mode: pre-collected data only)\n");
#endif

    /* ---- Per-Rank Assignment ---- */
    printf("\n--- Per-Rank Assignment ---\n\n");
    for (int r = 0; r < ws; r++) {
        const MachineManager& M = managers[r];
        printf("  Rank %-3d @ %-20s  %d GPU, %d CPU NUMA\n",
               r, M.hostname,
               (int)M.gpus().size(), (int)M.cpus().size());
        for (auto& n : M.gpus()) {
            std::string hl = handleStr(n.handle);
            const GpuInfo& gi = n.gpu;
            printf("    %-12s  %s [%s] %lu MB PCIe-Gen%d x%d",
                   hl.c_str(), gi.busId, gi.name,
                   (unsigned long)gi.memMB,
                   gi.pcieGen, gi.pcieWidth);
            if (gi.ccMajor)
                printf(" CC %d.%d", gi.ccMajor, gi.ccMinor);
            if (gi.numaId >= 0)
                printf(" NUMA:%d", gi.numaId);
            printf("\n");
        }
        for (auto& n : M.cpus()) {
            std::string hl = handleStr(n.handle);
            printf("    %-12s  NUMA %d\n", hl.c_str(), n.cpu.numaId);
        }
    }

    /* ---- All Unique Nodes ---- */
    printf("\n--- All Unique Nodes (%d GPU, %d CPU NUMA) ---\n\n", nGpus, nCpus);
    for (int i = 0; i < N; i++) {
        const GNode& g = G[i];
        std::string label = handleStr(g.handle);
        if (g.isGpu()) {
            printf("  %-12s  %-16s  %-24s  %-16s  %5lu MB",
                   label.c_str(), g.busId.c_str(), g.name.c_str(),
                   g.host.c_str(), (unsigned long)g.memMB);
            if (g.ccMajor) printf("  CC%d.%d", g.ccMajor, g.ccMinor);
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

    /* ---- Topology Matrix ---- */
    printf("\n--- Topology Matrix ---\n");
    printf("  Legend:  NVx = NVLink (x links)      PIX = single PCIe switch\n");
    printf("          PXB = multi PCIe switch       PHB = host bridge\n");
    printf("          NODE = same NUMA              SYS = cross-NUMA\n");
    printf("          NET  = cross-node\n\n");

    std::vector<std::string> labels(N);
    int maxLabelLen = 0;
    for (int i = 0; i < N; i++) {
        labels[i] = handleStr(G[i].handle);
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
