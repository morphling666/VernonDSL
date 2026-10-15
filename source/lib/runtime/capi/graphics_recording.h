#ifndef VERNON_RUNTIME_CAPI_GRAPHICS_RECORDING_H
#define VERNON_RUNTIME_CAPI_GRAPHICS_RECORDING_H

#include "runtime/capi/program_state.h"

namespace vernon::runtime::capi {

// Resolves the invocation's recorded graphics commands into `passes`. Requires the frozen binding snapshot: the hazard
// checks find the resources the invocation's Nodes write through its bindings.
VernonStatus validateGraphicsRecordings(VernonProgramInvocation &invocation, RecordedGraphicsPasses &passes);

} // namespace vernon::runtime::capi

#endif
