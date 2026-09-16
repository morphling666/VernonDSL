#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_SHAPE_RESOLVER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_SHAPE_RESOLVER_H

#include "runtime/program_execution/program_invocation_state.h"

#include <string>
#include <vector>

namespace vernon::runtime::program {
struct Program;
struct ResolvedExecutionPlan;
} // namespace vernon::runtime::program

namespace vernon::runtime::ad {

bool resolveProgramShapes(const program::Program &program, const program::ResolvedExecutionPlan *topology,
                          std::vector<program_execution::ProgramValueState> &values, std::string &error);

} // namespace vernon::runtime::ad

#endif
