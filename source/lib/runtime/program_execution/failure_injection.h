#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_FAILURE_INJECTION_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_FAILURE_INJECTION_H

#include <cstddef>

namespace vernon::runtime::program_execution {

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

} // namespace vernon::runtime::program_execution

#endif
