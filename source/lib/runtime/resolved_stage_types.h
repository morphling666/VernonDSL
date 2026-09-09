#ifndef VERNON_RUNTIME_RESOLVED_STAGE_TYPES_H
#define VERNON_RUNTIME_RESOLVED_STAGE_TYPES_H

#include "VernonRuntime.h"

struct VernonStageExecutable;

// Private physical invocation passed from a resolved Program node to its
// backend Stage implementation. It is not a public execution model.
struct VernonStageInvocationDescriptor {
    uint32_t struct_size;
    uint32_t abi_version;
    const VernonProgramArgument *arguments;
    size_t argument_count;
    VernonLaunchSize compute_grid;
    VernonRuntimeProviderObject command_encoder;
    const VernonGraphicsState *graphics_state;
    const VernonRenderPass *render_pass;
    const VernonDrawCommand *draw_command;
    const VernonDynamicState *dynamic_state;
};

#endif
