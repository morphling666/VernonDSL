#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_REPLAY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_REPLAY_H

#include "VernonRuntime.h"
#include "runtime/autodiff/runtime_autodiff_memory.h"
#include "runtime/autodiff/runtime_autodiff_policy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vernon::runtime::ad::gpu {

struct BatchSummary {
    uint32_t requiredBytes{};
    uint32_t status{};
};

struct alignas(8) Segment {
    uint64_t virtualWorkgroup[3]{};
    uint64_t virtualGlobalBase[3]{};
    uint64_t tapeBase{};
    uint64_t tapeStride{};
    uint64_t capacityBytes{};
    uint64_t reserved{};
};

static_assert(sizeof(BatchSummary) == 8);
static_assert(sizeof(Segment) == 80);

struct BatchBudget {
    size_t capacity{};
    size_t tapeBytes{};
    size_t segmentBytes{};
    size_t statusBytes{};
    MemoryAccounting memory;
};

struct BatchRange {
    size_t begin{};
    size_t count{};
};

class ReverseBatchScheduler {
public:
    ReverseBatchScheduler(size_t groupCount, size_t capacity) : remaining_(groupCount), capacity_(capacity) {}

    bool valid() const { return capacity_ != 0; }
    bool empty() const { return remaining_ == 0; }
    BatchRange current() const {
        const size_t count = std::min(capacity_, remaining_);
        return {remaining_ - count, count};
    }
    bool setCapacity(size_t capacity) {
        capacity_ = capacity;
        return valid();
    }
    void commit() { remaining_ = current().begin; }

private:
    size_t remaining_{};
    size_t capacity_{};
};

bool planBatchBudget(PlanningPolicy policy, size_t fixedBytes, size_t maximumBytes, size_t groupCount,
                     size_t groupTapeBytes, size_t summaryBytes, BatchBudget &budget);
bool normalizeTapeStride(size_t requestedBytes, size_t &stride);

inline bool initializeSegment(VernonLaunchSize grid, VernonLaunchSize workgroup, size_t linearGroup, size_t tapeStride,
                              size_t workgroupVolume, Segment &segment) {
    size_t groupCount = 1;
    for (uint32_t extent : {grid.x, grid.y, grid.z}) {
        if (!extent || groupCount > std::numeric_limits<size_t>::max() / extent)
            return false;
        groupCount *= extent;
    }
    if (linearGroup >= groupCount || !tapeStride || !workgroupVolume ||
        tapeStride > std::numeric_limits<size_t>::max() / workgroupVolume ||
        tapeStride > std::numeric_limits<uint32_t>::max())
        return false;
    segment = {};
    size_t coordinate = linearGroup;
    segment.virtualWorkgroup[0] = coordinate % grid.x;
    coordinate /= grid.x;
    segment.virtualWorkgroup[1] = coordinate % grid.y;
    coordinate /= grid.y;
    segment.virtualWorkgroup[2] = coordinate;
    segment.virtualGlobalBase[0] = segment.virtualWorkgroup[0] * workgroup.x;
    segment.virtualGlobalBase[1] = segment.virtualWorkgroup[1] * workgroup.y;
    segment.virtualGlobalBase[2] = segment.virtualWorkgroup[2] * workgroup.z;
    if (linearGroup > std::numeric_limits<size_t>::max() / workgroupVolume)
        return false;
    segment.reserved = linearGroup * workgroupVolume;
    segment.tapeStride = tapeStride;
    segment.capacityBytes = tapeStride * workgroupVolume;
    for (uint64_t coordinate : segment.virtualWorkgroup)
        if (coordinate > std::numeric_limits<uint32_t>::max())
            return false;
    for (uint64_t coordinate : segment.virtualGlobalBase)
        if (coordinate > std::numeric_limits<uint32_t>::max())
            return false;
    if (segment.capacityBytes > std::numeric_limits<uint32_t>::max())
        return false;
    return true;
}

inline bool initializeBatchSegment(VernonLaunchSize grid, VernonLaunchSize workgroup, size_t linearGroup,
                                   size_t batchIndex, size_t tapeStride, size_t workgroupVolume, size_t capacityBytes,
                                   Segment &segment) {
    size_t groupBytes = 0;
    size_t tapeBase = 0;
    if (!initializeSegment(grid, workgroup, linearGroup, tapeStride, workgroupVolume, segment) ||
        tapeStride > std::numeric_limits<size_t>::max() / workgroupVolume)
        return false;
    groupBytes = tapeStride * workgroupVolume;
    if (batchIndex > std::numeric_limits<size_t>::max() / groupBytes)
        return false;
    tapeBase = batchIndex * groupBytes;
    if (tapeBase > capacityBytes || groupBytes > capacityBytes - tapeBase ||
        tapeBase > std::numeric_limits<uint32_t>::max() || capacityBytes > std::numeric_limits<uint32_t>::max())
        return false;
    segment.tapeBase = tapeBase;
    segment.capacityBytes = capacityBytes;
    return true;
}

inline bool requiredTapeBytes(const BatchSummary &summary, size_t laneCount, size_t currentStride,
                              size_t &requiredStride, size_t &requiredBytes, uint32_t &status) {
    if (!laneCount || !currentStride)
        return false;
    requiredStride = std::max(currentStride, static_cast<size_t>(summary.requiredBytes));
    status = summary.status;
    if (requiredStride > std::numeric_limits<size_t>::max() - 3)
        return false;
    requiredStride = (requiredStride + 3) & ~size_t{3};
    if (requiredStride > std::numeric_limits<size_t>::max() / laneCount)
        return false;
    requiredBytes = requiredStride * laneCount;
    return true;
}

} // namespace vernon::runtime::ad::gpu

#endif
