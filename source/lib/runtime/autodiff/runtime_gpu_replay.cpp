#include "runtime_gpu_replay.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vernon::runtime::ad::gpu {
namespace {

constexpr size_t kMaximumReplayBatches = 64;

bool multiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool add(size_t left, size_t right, size_t &result) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

} // namespace

bool planBatchBudget(PlanningPolicy policy, size_t fixedBytes, size_t maximumBytes, size_t groupCount,
                     size_t groupTapeBytes, size_t summaryBytes, BatchBudget &budget) {
    budget = {};
    if (!groupCount || !groupTapeBytes)
        return false;
    size_t segmentPairBytes = 0;
    size_t perGroupBytes = 0;
    size_t fixedManagedBytes = 0;
    if (!multiply(sizeof(Segment), 2, segmentPairBytes) || !add(groupTapeBytes, segmentPairBytes, perGroupBytes) ||
        !add(fixedBytes, summaryBytes, fixedManagedBytes) || fixedManagedBytes > maximumBytes ||
        perGroupBytes > maximumBytes - fixedManagedBytes)
        return false;
    const size_t maximumCapacity = std::min({groupCount, (maximumBytes - fixedManagedBytes) / perGroupBytes,
                                             static_cast<size_t>(std::numeric_limits<uint32_t>::max())});
    if (policy == PlanningPolicy::MinMemory) {
        // Minimize memory subject to a bounded control-plane cost. A one-group
        // batch made submissions and fixed-size summary readbacks scale with
        // the workgroup count, which is unusable for large dispatches.
        const size_t controlPlaneFloor =
            groupCount / kMaximumReplayBatches + static_cast<size_t>(groupCount % kMaximumReplayBatches != 0);
        budget.capacity = std::min(maximumCapacity, std::max(controlPlaneFloor, size_t{1}));
    } else if (policy == PlanningPolicy::Balanced) {
        budget.capacity = static_cast<size_t>(std::sqrt(static_cast<long double>(maximumCapacity)));
        if (budget.capacity && (budget.capacity < maximumCapacity / budget.capacity ||
                                budget.capacity * budget.capacity < maximumCapacity))
            ++budget.capacity;
        budget.capacity = std::max(budget.capacity, size_t{1});
    } else
        budget.capacity = maximumCapacity;
    if (!budget.capacity || !multiply(groupTapeBytes, budget.capacity, budget.tapeBytes) ||
        !multiply(sizeof(Segment), budget.capacity, budget.segmentBytes)) {
        budget = {};
        return false;
    }
    budget.statusBytes = summaryBytes;
    size_t variableBytes = 0;
    if (!add(budget.tapeBytes, budget.segmentBytes, variableBytes) ||
        !add(variableBytes, budget.segmentBytes, variableBytes) ||
        !add(variableBytes, budget.statusBytes, variableBytes) ||
        !add(fixedBytes, variableBytes, budget.memory.peakTemporaryBytes)) {
        budget = {};
        return false;
    }
    size_t budgetedBytes = 0;
    return budget.memory.budgetedBytes(budgetedBytes) && budgetedBytes <= maximumBytes;
}

bool normalizeTapeStride(size_t requestedBytes, size_t &stride) {
    requestedBytes = std::max(requestedBytes, size_t{16});
    if (requestedBytes > std::numeric_limits<size_t>::max() - 3)
        return false;
    stride = (requestedBytes + 3) & ~size_t{3};
    return true;
}

} // namespace vernon::runtime::ad::gpu
