#ifndef VERNON_AUTODIFF_MEMORY_ACCOUNTING_H
#define VERNON_AUTODIFF_MEMORY_ACCOUNTING_H

#include <algorithm>
#include <limits>

namespace vernon::autodiff {

template <typename ByteCount> struct MemoryAccounting {
    // Logical payload is diagnostic and is never added to a physical budget.
    ByteCount logicalPayloadBytes{};
    // All physical allocations that survive forward execution.
    ByteCount retainedAllocationBytes{};
    // Residency and allocator commitment of the logical residual payload only.
    // Retained primals and other support allocations belong only to retained.
    ByteCount residentBytes{};
    ByteCount allocatedBytes{};
    // Persistent checkpoint allocation is disjoint from retained allocations.
    ByteCount checkpointBytes{};
    // Peak apply-time allocation, including fixed and batch-local resources.
    ByteCount peakTemporaryBytes{};

    bool budgetedBytes(ByteCount &bytes) const {
        bytes = retainedAllocationBytes;
        if (checkpointBytes > std::numeric_limits<ByteCount>::max() - bytes)
            return false;
        bytes += checkpointBytes;
        if (peakTemporaryBytes > std::numeric_limits<ByteCount>::max() - bytes)
            return false;
        bytes += peakTemporaryBytes;
        return true;
    }

    void observeTemporary(ByteCount bytes) { peakTemporaryBytes = std::max(peakTemporaryBytes, bytes); }
};

} // namespace vernon::autodiff

#endif
