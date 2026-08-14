#ifndef VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H
#define VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H

#include "VernonRuntime.h"
#include "autodiff_metadata.h"

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
};

VERNON_RUNTIME_CAPI VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, VernonStringView forwardProtocol, VernonStringView backwardProtocol,
    const AutodiffDerivativeGroupView *derivativeGroups, size_t derivativeGroupCount, uint64_t staticTapeBytesHint);

VERNON_RUNTIME_CAPI bool hasAutodiffStorageObjectives(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonLaunchSize autodiffWorkgroupSize(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI AutodiffPullbackMemoryUsage autodiffPullbackMemoryUsage(const VernonPullback *pullback);
VERNON_RUNTIME_CAPI size_t autodiffHostTapeContextLimit(const VernonRuntimeContext *context);

} // namespace vernon::runtime

#endif
