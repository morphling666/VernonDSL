#ifndef VERNON_RUNTIME_AUTODIFF_HOST_TAPE_TEST_HOOKS_H
#define VERNON_RUNTIME_AUTODIFF_HOST_TAPE_TEST_HOOKS_H

#include "runtime/autodiff/host_tape_allocator.h"

#include <memory>

struct VernonRuntimeContext;

namespace vernon::runtime::ad {

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
void setHostTapeMemoryPolicyForTesting(VernonRuntimeContext &context, std::shared_ptr<HostTapeMemoryPolicy> policy);
size_t hostTapeMemoryPolicyChargedBytesForTesting(HostTapeMemoryPolicy &policy);
#endif

} // namespace vernon::runtime::ad

#endif
