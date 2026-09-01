#ifndef VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H
#define VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H

#include "VernonRuntime.h"
#include "autodiff_metadata.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::runtime {

struct AutodiffDerivativeGroupView {
    AutodiffDerivativeRole role{};
    VernonStringView declaredPath;
    const VernonStringView *leafPaths{};
    size_t leafCount{};
};

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

struct AutodiffWriteFootprint {
    std::string owner;
    bool wholeView{true};
    std::vector<uint64_t> indices;
};

struct AutodiffGpuStageView {
    const void *artifact{};
    size_t artifactSize{};
    VernonStringView reflection;
    VernonStringView entry;
};

VERNON_RUNTIME_CAPI VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, const AutodiffDerivativeGroupView *derivativeGroups, size_t derivativeGroupCount,
    uint64_t staticTapeBytesHint, VernonStringView residualStorage, VernonStringView selectedPolicy,
    bool wholeDispatchRetentionPermitted);

VERNON_RUNTIME_CAPI VernonLoadedPipeline *
loadBackendGpuAutodiffPipeline(VernonRuntimeContext &context, const AutodiffGpuStageView &primal,
                               const AutodiffGpuStageView &forward, const AutodiffGpuStageView &backward,
                               const AutodiffDerivativeGroupView *derivativeGroups, size_t derivativeGroupCount,
                               uint64_t staticTapeBytesHint, VernonStringView residualStorage,
                               VernonStringView selectedPolicy);

VERNON_RUNTIME_CAPI bool hasAutodiffStorageObjectives(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonLaunchSize autodiffWorkgroupSize(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI std::vector<AutodiffWriteFootprint> autodiffReadFootprints(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI std::vector<AutodiffWriteFootprint> autodiffWriteFootprints(const VernonLoadedPipeline *pipeline);
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
void autodiffSetProgramCheckpointPlan(VernonLoadedPipeline *pipeline, const uint64_t *memoryBudget,
                                      std::string_view policy);

} // namespace vernon::runtime

#endif
