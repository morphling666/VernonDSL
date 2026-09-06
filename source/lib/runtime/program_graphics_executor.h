#ifndef VERNON_RUNTIME_PROGRAM_GRAPHICS_EXECUTOR_H
#define VERNON_RUNTIME_PROGRAM_GRAPHICS_EXECUTOR_H

#include "autodiff/program_value_arena.h"
#include "graphics_invocation_planner.h"
#include "program_execution_manifest.h"

#include <functional>
#include <string>

namespace vernon::runtime {

using ResolveProgramRenderPass = std::function<const VernonRenderPass *(uint32_t control)>;

bool bindProgramGraphicsControlResources(const program::Program &program, const program::Graph &graph,
                                         const program::ResolvedGraph &resolved, ad::LogicalValueFrame &invocation,
                                         const ResolveProgramRenderPass &resolveRenderPass, std::string &error);

/// Check the bound render target against the attachment signature the Program was compiled for.
///
/// Attachment formats and sample counts are baked into every backend's pipeline object, so a Program compiled for
/// one signature cannot draw into a target with another. The manifest records the compatible formats per attachment;
/// this rejects a target outside that set instead of leaving the mismatch to the backend.
bool checkProgramGraphicsAttachmentSignature(const program::GraphicsOperation &graphics,
                                             const PlannedGraphicsInvocation &plan, std::string &error);

} // namespace vernon::runtime

#endif
