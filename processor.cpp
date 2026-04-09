#include "processor.hpp"

#include <iomanip>

std::ostream& operator<<(std::ostream& os, const Processor& p) {
    std::string hl = handleStr(p.publicHandle());
    os << std::left;
    switch (p.info().type) {
    case CUIDTX_PROCESSOR_TYPE_GPU: {
        const GPUInfo& gi = p.info().gpu;
        os << "    " << std::setw(12) << hl
           << "  " << gi.busId << " [" << gi.name << "]";
        if (p.info().numaId >= 0)
            os << " NUMA:" << p.info().numaId;
        break;
    }
    case CUIDTX_PROCESSOR_TYPE_CPU:
        os << "    " << std::setw(12) << hl
           << "  NUMA " << p.info().numaId
           << "  os_index " << p.info().cpu.osIndex;
        break;
    default:
        os << "    " << hl;
        break;
    }
    return os;
}
