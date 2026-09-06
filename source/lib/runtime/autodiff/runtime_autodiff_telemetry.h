#ifndef VERNON_RUNTIME_AUTODIFF_TELEMETRY_H
#define VERNON_RUNTIME_AUTODIFF_TELEMETRY_H

#include "VernonRuntime.h"
#include "autodiff_metadata.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::runtime {

struct AutodiffPullbackMemoryUsage {
    size_t logicalResidualBytes{};
    size_t residentBytes{};
    size_t allocatedBytes{};
    size_t retainedAllocationBytes{};
    size_t peakTemporaryBytes{};
};

struct AutodiffPullbackControlPlaneUsage {
    uint64_t submissions{};
    uint64_t waits{};
    uint64_t readbacks{};
    uint64_t atomicPublications{};
    uint64_t temporaryAllocationBytes{};
    uint64_t deviceWaitNanoseconds{};
};

VERNON_RUNTIME_CAPI AutodiffPullbackMemoryUsage autodiffPullbackMemoryUsage(const VernonPullback *pullback);
VERNON_RUNTIME_CAPI AutodiffPullbackControlPlaneUsage autodiffPullbackControlPlaneUsage(const VernonPullback *pullback);
VERNON_RUNTIME_CAPI size_t autodiffHostTapeContextLimit(const VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonRhiDevice autodiffRhiDevice(const VernonRuntimeContext *context);

struct AutodiffPullbackCheckpointPlan {
    bool present{};
    uint64_t peakBytes{};
    uint64_t memoryBudget{};
    uint64_t logicalResidualBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t initialStateBytes{};
    uint64_t restorationBytes{};
    uint64_t transactionBytes{};
    uint64_t persistentCheckpointBytes{};
    uint64_t backwardValueBytes{};
    uint64_t replayCost{};
    uint64_t recomputationCost{};
    std::string selectedPolicy;
};

struct AutodiffPullbackPassTelemetry {
    uint32_t scheduleOffset{};
    std::string passName;
    std::string residualSourceKind{"none"};
    std::string controlHistoryKind{"none"};
    uint64_t estimatedTapeBytes{};
    uint64_t logicalResidualBytes{};
    uint64_t residentTapeBytes{};
    uint64_t allocatedTapeBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t peakTemporaryTapeBytes{};
    uint64_t checkpointBytes{};
    uint64_t activeOperationCount{};
    uint64_t recomputationCost{};
};

AutodiffPullbackCheckpointPlan autodiffPullbackCheckpointPlan(const VernonPullback *pullback);
std::vector<AutodiffPullbackPassTelemetry> autodiffPullbackPassTelemetry(const VernonPullback *pullback);
uint64_t autodiffPullbackPeakRuntimeManagedBytes(const VernonPullback *pullback);
void autodiffSetProgramCheckpointPlan(VernonProgramExecutable *pipeline, const uint64_t *memoryBudget,
                                      std::string_view policy);

} // namespace vernon::runtime

#endif
