#ifndef VERNON_RUNTIME_CAPI_GRAPHICS_RECORDING_H
#define VERNON_RUNTIME_CAPI_GRAPHICS_RECORDING_H

#include "runtime/capi/program_state.h"

namespace vernon::runtime::capi {

using PipelineStateResult = vernon::Result<std::shared_ptr<const RuntimeGraphicsPipelineState>, std::string>;

// Pipeline 0 of graphics Node `nodeId`: the Node's own Stage, cached in the executable.
PipelineStateResult passPipelineState(VernonProgramExecutable &executable, VernonProgramNodeId nodeId);

// `set` prepared for `pipeline`, cached in the set while the pipeline lives.
std::shared_ptr<const RuntimeBindingSetMatch>
matchBindingSet(RuntimeBindingSetState &set, const std::shared_ptr<const RuntimeGraphicsPipelineState> &pipeline);

// The boundary path of the Program parameter feeding per-draw endpoint `endpoint` of `pipeline`, by which a bound set
// entry stays bound across a pipeline switch.
const std::string &endpointPath(const RuntimeGraphicsPipelineState &pipeline, size_t endpoint);

// A binding set's frame as one invocation draws with it, and the owner of the arguments it supplies.
struct InvocationSetFrame {
    std::shared_ptr<const RuntimeLoweredSetFrame> lowered;
    ArgumentOwner owner;
};

// A match lowers its frame once and keeps it, unless the frame needs per-invocation copies or carriers; such a frame
// and its groups belong to the invocation.
bool lowerSetFrame(VernonRuntimeContext &context, const RecordedSetFrame &frame, InvocationSetFrame &output,
                   std::string &error);

} // namespace vernon::runtime::capi

#endif
