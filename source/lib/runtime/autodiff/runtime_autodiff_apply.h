#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_APPLY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_APPLY_H

#include "VernonRuntime.h"
#include "runtime/program_execution/invocation_outcome.h"

namespace vernon::runtime::ad {

VernonStatus applyPullback(VernonPullback *pullback, const VernonProgramArgument *arguments, size_t argumentCount,
                           const VernonPullbackApplyOptions *options,
                           program_execution::InvocationMutationOutcome &outcome);

} // namespace vernon::runtime::ad

#endif
