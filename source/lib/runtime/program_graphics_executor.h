#ifndef VERNON_RUNTIME_PROGRAM_GRAPHICS_EXECUTOR_H
#define VERNON_RUNTIME_PROGRAM_GRAPHICS_EXECUTOR_H

#include "autodiff/program_value_arena.h"
#include "program_execution_manifest.h"

#include <functional>
#include <string>

namespace vernon::runtime {

using ResolveProgramRenderPass = std::function<const VernonRenderPass *(uint32_t control)>;

bool bindProgramGraphicsControlResources(const program::Program &program, const program::Graph &graph,
                                         const program::ResolvedGraph &resolved, ad::ProgramInvocationFrame &invocation,
                                         const ResolveProgramRenderPass &resolveRenderPass, std::string &error);

} // namespace vernon::runtime

#endif
