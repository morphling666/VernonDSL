#ifndef VERNON_RUNTIME_RESOLVED_STAGE_INVOCATION_H
#define VERNON_RUNTIME_RESOLVED_STAGE_INVOCATION_H

#include "VernonRuntime.h"
#include "resolved_stage_types.h"

namespace vernon::runtime {

extern "C" {
void destroyResolvedStage(VernonStageExecutable *stage);
VernonStatus submitResolvedStage(VernonStageExecutable *stage, const VernonStageInvocationDescriptor *invocation,
                                 VernonSubmission **output);
VernonStatus encodeResolvedStage(VernonRuntimeProviderObject encoder, VernonStageExecutable *stage,
                                 const VernonStageInvocationDescriptor *invocation);
}

} // namespace vernon::runtime

#endif
