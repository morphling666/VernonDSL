#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_FAILURE_INJECTION_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_FAILURE_INJECTION_H

#include <cstddef>

namespace vernon::runtime::ad::gpu {

enum class FailureBoundary {
    None,
    Allocation,
    Upload,
    Download,
    Copy,
    Encode,
    Submit,
    Wait,
    Resize,
    Publication,
};

bool injectFailure(FailureBoundary boundary);
void setFailureInjectionForTesting(FailureBoundary boundary, size_t failOnOccurrence = 1);
void clearFailureInjectionForTesting();

} // namespace vernon::runtime::ad::gpu

#endif
