#ifndef VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H
#define VERNON_RUNTIME_RUNTIME_DIRECT_AUTODIFF_H

#include "VernonRuntime.h"

namespace vernon::runtime {

VERNON_RUNTIME_CAPI VernonLoadedPipeline *loadBackendCpuAutodiffPipeline(
    VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry, VernonStringView primalReflection,
    VernonStringView primalName, VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
    VernonStringView forwardName, VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
    VernonStringView backwardName, const VernonStringView *gradientPaths, size_t gradientPathCount);

} // namespace vernon::runtime

#endif
