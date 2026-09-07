#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_BUILDER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_BUILDER_H

#include "runtime/autodiff/program_invocation_spec.h"
#include "runtime/program_execution/program_invocation_state.h"

struct VernonRuntimeContext;

namespace vernon::runtime::ad {

class ProgramTapeScratch;
class AutodiffMemoryPolicy;

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan,
                                  std::vector<program_execution::ProgramValueState> &values,
                                  std::map<uint32_t, program_execution::ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const ForwardInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error);

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan,
                                  std::vector<program_execution::ProgramValueState> &values,
                                  std::map<uint32_t, program_execution::ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const PullbackInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error);

} // namespace vernon::runtime::ad

#endif
