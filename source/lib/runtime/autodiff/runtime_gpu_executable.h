#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_EXECUTABLE_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_EXECUTABLE_H

#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/autodiff/runtime_autodiff_policy.h"
#include "runtime/autodiff/runtime_gpu_bindings.h"
#include "runtime/autodiff/runtime_gpu_resources.h"

namespace vernon::runtime::ad::gpu {

std::shared_ptr<Executable> createNoTapeExecutable(VernonRuntimeContext &context,
                                                   std::shared_ptr<OwnedPipeline> forward,
                                                   std::shared_ptr<OwnedPipeline> backward,
                                                   std::shared_ptr<Signature> signature,
                                                   std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs);

std::shared_ptr<Executable>
createTapeExecutable(VernonRuntimeContext &context, std::shared_ptr<OwnedPipeline> primal,
                     std::shared_ptr<OwnedPipeline> forward, std::shared_ptr<OwnedPipeline> backward,
                     std::shared_ptr<Signature> signature, std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs,
                     std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs, size_t staticTapeBytesHint,
                     PlanningPolicy planningPolicy, bool requiresTapeStatus);

} // namespace vernon::runtime::ad::gpu

#endif
