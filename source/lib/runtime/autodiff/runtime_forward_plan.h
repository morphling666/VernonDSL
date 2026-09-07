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
} // namespace vernon::execution::detail

namespace vernon::runtime::ad {
VernonStatus applyPullbackDeviceWithPlanSink(VernonPullback &pullback, const VernonAdDeviceValueSet *cotangents,
                                             VernonAdDeviceValueSet &gradients,
                                             const VernonPullbackApplyOptions *options,
                                             execution::detail::RhiCommandPlanSink &sink);
} // namespace vernon::runtime::ad

#endif
