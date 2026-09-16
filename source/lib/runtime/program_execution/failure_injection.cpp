#include "failure_injection.h"

namespace vernon::runtime::program_execution {
namespace {

struct FailureInjection {
    FailureBoundary boundary{FailureBoundary::None};
    size_t remaining{};
};

thread_local FailureInjection injection;

} // namespace

bool injectFailure(FailureBoundary boundary) {
    if (injection.boundary != boundary || injection.remaining == 0)
        return false;
    if (--injection.remaining != 0)
        return false;
    injection = {};
    return true;
}

void setFailureInjectionForTesting(FailureBoundary boundary, size_t failOnOccurrence) {
    injection = {boundary, failOnOccurrence};
}

void clearFailureInjectionForTesting() { injection = {}; }

} // namespace vernon::runtime::program_execution
