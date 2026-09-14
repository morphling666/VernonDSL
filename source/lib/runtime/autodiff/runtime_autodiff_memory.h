#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_MEMORY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_MEMORY_H

#include <algorithm>
#include <cstddef>
#include <limits>

namespace vernon::runtime::ad {

struct MemoryAccounting {
    // Logical payload is diagnostic and is never added to a physical budget.
    size_t logicalPayloadBytes{};
    // All physical allocations that survive forward execution.
    size_t retainedAllocationBytes{};
    // Residency and allocator commitment of the logical residual payload only.
    // Retained primals and other support allocations belong only to retained.
    size_t residentBytes{};
    size_t allocatedBytes{};
    // Persistent checkpoint allocation is disjoint from retained allocations.
    size_t checkpointBytes{};
    // Peak apply-time allocation, including fixed and batch-local resources.
    size_t peakTemporaryBytes{};

    bool budgetedBytes(size_t &bytes) const {
        bytes = retainedAllocationBytes;
        if (checkpointBytes > std::numeric_limits<size_t>::max() - bytes)
            return false;
        bytes += checkpointBytes;
        if (peakTemporaryBytes > std::numeric_limits<size_t>::max() - bytes)
            return false;
        bytes += peakTemporaryBytes;
        return true;
    }

    void observeTemporary(size_t bytes) { peakTemporaryBytes = std::max(peakTemporaryBytes, bytes); }
};

} // namespace vernon::runtime::ad

#endif
