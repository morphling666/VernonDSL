#include "runtime_gpu_replay.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vernon::runtime::ad::gpu {
namespace {

// A submission followed by a fixed-size status readback costs substantially
// more than a few MiB of transient device memory on every supported backend.
// Expressing that latency as an equivalent byte cost lets the planner trade
// memory against control-plane work without imposing a fixed batch count.
constexpr long double kControlPlaneEquivalentBytes = 256.0L * 1024.0L * 1024.0L;

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

size_t divideCeil(size_t value, size_t divisor) { return value / divisor + static_cast<size_t>(value % divisor != 0); }

size_t costModelCapacity(PlanningPolicy policy, size_t groupCount, size_t maximumCapacity, size_t perGroupBytes) {
    if (policy == PlanningPolicy::MinRuntime)
        return maximumCapacity;
    const long double memoryWeight = policy == PlanningPolicy::MinMemory ? 2.0L : 1.0L;
    const long double stationary = std::sqrt(static_cast<long double>(groupCount) * kControlPlaneEquivalentBytes /
                                             (memoryWeight * static_cast<long double>(perGroupBytes)));
    const size_t center = std::clamp(static_cast<size_t>(std::max(stationary, 1.0L)), size_t{1}, maximumCapacity);
    const size_t candidates[]{
        size_t{1},       center > 1 ? center - 1 : center, center, center < maximumCapacity ? center + 1 : center,
        maximumCapacity,
    };
    size_t selected = 1;
    long double selectedCost = std::numeric_limits<long double>::infinity();
    for (size_t capacity : candidates) {
        const long double cost =
            memoryWeight * static_cast<long double>(capacity) * static_cast<long double>(perGroupBytes) +
            static_cast<long double>(divideCeil(groupCount, capacity)) * kControlPlaneEquivalentBytes;
        if (cost < selectedCost || (cost == selectedCost && capacity > selected)) {
            selected = capacity;
            selectedCost = cost;
        }
    }
    return selected;
}

} // namespace

bool planBatchBudget(PlanningPolicy policy, size_t fixedBytes, size_t maximumBytes, size_t groupCount,
                     size_t groupTapeBytes, size_t summaryBytes, BatchBudget &budget, bool preferWholeDispatch) {
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
    if (!maximumCapacity)
        return false;
    // Dynamic validation benefits from one atomic transaction when the whole
    // pass fits. Proven-static replay can use the policy-selected capacity and
    // still chain every batch into one command plan without status readbacks.
    budget.capacity = preferWholeDispatch && maximumCapacity == groupCount
                          ? groupCount
                          : costModelCapacity(policy, groupCount, maximumCapacity, perGroupBytes);
    budget.wholeDispatch = budget.capacity == groupCount;
    budget.batchCount = divideCeil(groupCount, budget.capacity);
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
