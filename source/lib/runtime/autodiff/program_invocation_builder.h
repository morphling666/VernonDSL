#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_BUILDER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_BUILDER_H

#include "runtime/autodiff/program_boundary_binder.h"
#include "runtime/autodiff/program_invocation_spec.h"
#include "runtime/program_execution/program_invocation_state.h"

struct VernonRuntimeContext;

namespace vernon::runtime::ad {

class ProgramTapeScratch;
class AutodiffMemoryPolicy;

struct ProgramInvocationPreparation {
    std::vector<char> typedControls;
    std::vector<char> typedControlStorages;
    std::vector<char> tapeValues;
    std::vector<char> dynamicShapes;
    std::vector<char> materializableStaticValues;
    std::vector<std::optional<size_t>> staticByteSizes;
    std::vector<std::optional<ValueLayout>> layouts;
    std::map<uint32_t, program_execution::ProgramStorageBacking> storageBackings;
    std::vector<char> forwardRequiredValues;
    std::vector<char> backwardRequiredValues;
};

ProgramInvocationPreparation prepareProgramInvocation(const program::Program &execution);

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan,
                                  const ProgramInvocationPreparation &preparation,
                                  ProgramBoundaryBindingScratch &bindingScratch,
                                  std::vector<program_execution::ProgramValueState> &values,
                                  std::map<uint32_t, program_execution::ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const ForwardInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error);

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan,
                                  const ProgramInvocationPreparation &preparation,
                                  ProgramBoundaryBindingScratch &bindingScratch,
                                  std::vector<program_execution::ProgramValueState> &values,
                                  std::map<uint32_t, program_execution::ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const PullbackInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error);

} // namespace vernon::runtime::ad

#endif
