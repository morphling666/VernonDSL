#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_FORWARD_PLAN_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_FORWARD_PLAN_H

#include "VernonRuntime.h"

struct VernonProgramExecutable;
struct VernonPullback;

namespace vernon::runtime {
struct ProgramInvocationContext;
}

namespace vernon::execution::detail {
class RhiCommandPlanSink;
struct RhiCommandExecutionPlan;
} // namespace vernon::execution::detail

namespace vernon::runtime::ad {
VernonStatus forwardProgramInvocation(VernonProgramExecutable &pipeline,
                                      const VernonProgramSubmitDescriptor &invocation, VernonPullback *&pullback,
                                      const ProgramInvocationContext *programContext = nullptr);
VernonStatus preparePipelineForwardCommandPlan(VernonProgramExecutable &pipeline,
                                               const VernonProgramSubmitDescriptor &invocation,
                                               const VernonAdValueSet &inputs,
                                               execution::detail::RhiCommandExecutionPlan &plan,
                                               VernonPullback *&pullback);
VernonStatus applyPullbackDeviceWithPlanSink(VernonPullback &pullback, const VernonAdDeviceValueSet *cotangents,
                                             VernonAdDeviceValueSet &gradients,
                                             const VernonPullbackApplyOptions *options,
                                             execution::detail::RhiCommandPlanSink &sink);
} // namespace vernon::runtime::ad

#endif
