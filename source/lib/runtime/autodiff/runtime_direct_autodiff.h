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

VERNON_RUNTIME_CAPI VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, VernonStringView forwardProtocol, VernonStringView backwardProtocol,
    const AutodiffDerivativeGroupView *derivativeGroups, size_t derivativeGroupCount);

VERNON_RUNTIME_CAPI bool hasAutodiffStorageObjectives(const VernonLoadedPipeline *pipeline);

} // namespace vernon::runtime

#endif
