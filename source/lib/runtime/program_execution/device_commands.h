#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_DEVICE_COMMANDS_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_DEVICE_COMMANDS_H

#include "VernonRHI.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace vernon::runtime::program_execution {

struct DeviceBufferCopy {
    VernonRhiBuffer source;
    VernonRhiBuffer destination;
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

inline bool checkedDeviceBufferOffset(uint64_t base, size_t relative, size_t &result) {
    if (base > std::numeric_limits<size_t>::max())
        return false;
    const size_t converted = static_cast<size_t>(base);
    if (relative > std::numeric_limits<size_t>::max() - converted)
        return false;
    result = converted + relative;
    return true;
}

} // namespace vernon::runtime::program_execution

#endif
