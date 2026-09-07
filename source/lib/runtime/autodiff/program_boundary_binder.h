#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_BOUNDARY_BINDER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_BOUNDARY_BINDER_H

#include "VernonRuntime.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/program_execution/publication_transaction.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime::program {
struct Program;
struct ResolvedExecutionPlan;
} // namespace vernon::runtime::program

namespace vernon::runtime::ad {

struct ProgramBoundaryBindingRequest {
    const VernonStageInvocationDescriptor &invocation;
    const std::vector<std::pair<uint32_t, uint32_t>> &valueBySlot;
    program_execution::PublicationTransaction *publication{};
};

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const program::ResolvedExecutionPlan &plan, const ProgramBoundaryBindingRequest &request,
                           std::map<uint32_t, VernonProgramArgument> &externalValues,
                           std::map<uint32_t, program_execution::ProgramStorageBacking> &backings,
                           std::vector<char> &live, std::string &error);

} // namespace vernon::runtime::ad

#endif
