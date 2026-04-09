#include "processor.hpp"

#include <iomanip>

std::ostream& operator<<(std::ostream& os, const Processor& p) {
    const CUIDTXprocessor& h = p.publicHandle();
    os << std::left;
    switch (p.info().type) {
    case CUIDTX_PROCESSOR_TYPE_GPU: {
        std::string hl = "GPU(" + std::to_string(h.memberId) + "," + std::to_string(h.gpu.deviceOrdinal) + ")";
        const GPUInfo& gi = p.info().gpu;
        os << "    " << std::setw(12) << hl
           << "  " << gi.busId << " [" << gi.name << "]";
        if (p.info().numaId >= 0)
            os << " NUMA:" << p.info().numaId;
        os << "  UUID:" << cuUuidToStr(gi.uuid);
        os << "  NVSwLinks:" << gi.nNvSwitchLinks
           << " GPUPeers:" << gi.nGPUPeers;
        if (gi.nvlinkBwPerLinkGBps >= 0)
            os << " NVLink BW:" << gi.nvlinkBwPerLinkGBps << " GB/s/link";
        if (gi.pcieBwGBps >= 0)
            os << "  PCIe BW:" << gi.pcieBwGBps << " GB/s";
        if (gi.c2cBwGBps >= 0)
            os << "  C2C BW:" << gi.c2cBwGBps << " GB/s";
        if (gi.hasFabricInfo) {
            os << "  Cluster:";
            const unsigned char* u = gi.clusterUuid;
            for (int b = 0; b < FABRIC_UUID_SZ; b++)
                os << std::hex << std::setfill('0') << std::setw(2) << (int)u[b];
            os << std::dec << std::setfill(' ');
            os << "  Clique:" << gi.cliqueId;
        }
        for (int pi = 0; pi < gi.nGPUPeers; pi++) {
            const GPUPeer& peer = gi.gpuPeers[pi];
            os << "\n      Peer[" << pi << "] " << peer.busId
               << "  NVL:" << peer.nvLinkCount
               << "  TopoLvl:" << peer.nvmlTopoLevel
               << "  Atomics:" << (peer.atomicsSupported ? "yes" : "no");
        }
        break;
    }
    case CUIDTX_PROCESSOR_TYPE_CPU: {
        std::string hl = "CPU(" + std::to_string(h.memberId) + "," + std::to_string(h.cpu.cpuOrdinal) + ")";
        os << "    " << std::setw(12) << hl
           << "  NUMA " << p.info().numaId
           << "  core " << p.info().cpu.coreIndex
           << "  os_index " << p.info().cpu.osIndex;
        break;
    }
    default:
        break;
    }
    return os;
}
