#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_EXECUTION_CONTROL_PLANE_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_EXECUTION_CONTROL_PLANE_H

#include <cstdint>

namespace vernon::runtime::program_execution {

struct ExecutionControlPlaneUsage {
    uint64_t submissions{};
    uint64_t waits{};
    uint64_t readbacks{};
    uint64_t atomicPublications{};
    uint64_t temporaryAllocationBytes{};
    uint64_t deviceWaitNanoseconds{};
};

} // namespace vernon::runtime::program_execution

#endif
