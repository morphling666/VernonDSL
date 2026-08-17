#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_MEMORY_USAGE_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_MEMORY_USAGE_H

#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/autodiff/runtime_autodiff_memory.h"

namespace vernon::runtime::ad {

inline PullbackMemoryUsage pullbackMemoryUsage(const MemoryAccounting &memory) {
    return {memory.logicalPayloadBytes, memory.residentBytes, memory.allocatedBytes, memory.retainedAllocationBytes,
            memory.peakTemporaryBytes};
}

} // namespace vernon::runtime::ad

#endif
