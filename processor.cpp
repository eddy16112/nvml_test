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
        os << "  NVLinks:" << gi.nNvLinks;
        if (gi.nvlinkBwPerLinkGBps >= 0)
            os << " (" << gi.nvlinkBwPerLinkGBps << " GB/s/link)";
        if (gi.pcieBwGBps >= 0)
            os << "  PCIe:" << gi.pcieBwGBps << " GB/s";
        if (gi.hasC2C) {
            os << "  C2C";
            if (gi.c2cBwGBps >= 0)
                os << ":" << gi.c2cBwGBps << " GB/s";
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
