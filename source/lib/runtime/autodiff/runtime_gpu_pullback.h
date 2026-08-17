#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_PULLBACK_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_PULLBACK_H

#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/autodiff/runtime_autodiff_policy.h"
#include "runtime/autodiff/runtime_gpu_bindings.h"
#include "runtime/autodiff/runtime_gpu_resources.h"

namespace vernon::runtime::ad::gpu {

std::unique_ptr<PullbackExecution>
createNoTapePullback(VernonRuntimeContext &context, std::shared_ptr<const Signature> signature,
                     std::shared_ptr<OwnedPipeline> backward, std::shared_ptr<const BindingSpecPlan> bindingSpecs,
                     VernonLaunchSize grid, DeviceValues retainedDevices, HostValues retainedHosts);

std::unique_ptr<PullbackExecution> createTapePullback(
    VernonRuntimeContext &context, std::shared_ptr<const Signature> signature, std::shared_ptr<OwnedPipeline> forward,
    std::shared_ptr<OwnedPipeline> backward, std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs,
    std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs, VernonLaunchSize grid, DeviceValues retainedDevices,
    HostValues retainedHosts, size_t staticTapeBytesHint, PlanningPolicy planningPolicy);

} // namespace vernon::runtime::ad::gpu

#endif
