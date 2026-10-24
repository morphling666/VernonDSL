#ifndef VERNON_RUNTIME_RESOLVED_STAGE_INVOCATION_H
#define VERNON_RUNTIME_RESOLVED_STAGE_INVOCATION_H

#include "VernonRuntime.h"
#include "resolved_stage_types.h"

namespace vernon::runtime {

struct ArgumentOwner;

extern "C" {
void destroyResolvedStage(VernonStageExecutable *stage);
VernonStatus submitResolvedStage(VernonStageExecutable *stage, const VernonStageInvocationDescriptor *invocation,
                                 VernonSubmission **output);
// `owners`, when given, holds the owner of each argument; otherwise the invocation owns them all.
VernonStatus encodeResolvedStage(VernonRuntimeProviderObject encoder, VernonStageExecutable *stage,
                                 const VernonStageInvocationDescriptor *invocation, const ArgumentOwner *owners);
}

} // namespace vernon::runtime

#endif
