#ifndef VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H
#define VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H

#include "VernonRuntime.h"
#include "autodiff_metadata.h"

#include <string>
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

struct AutodiffWriteFootprint {
    std::string owner;
    bool wholeView{true};
    std::vector<uint64_t> indices;
};

VERNON_RUNTIME_CAPI VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, const AutodiffDerivativeGroupView *derivativeGroups, size_t derivativeGroupCount,
    uint64_t staticTapeBytesHint, VernonStringView residualStorage, VernonStringView selectedPolicy,
    bool wholeDispatchRetentionPermitted);

VERNON_RUNTIME_CAPI bool hasAutodiffStorageObjectives(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonLaunchSize autodiffWorkgroupSize(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI std::vector<AutodiffWriteFootprint> autodiffReadFootprints(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI std::vector<AutodiffWriteFootprint> autodiffWriteFootprints(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI AutodiffPullbackMemoryUsage autodiffPullbackMemoryUsage(const VernonPullback *pullback);
VERNON_RUNTIME_CAPI size_t autodiffHostTapeContextLimit(const VernonRuntimeContext *context);

} // namespace vernon::runtime

#endif
