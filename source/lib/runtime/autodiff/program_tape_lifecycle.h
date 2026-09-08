#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_TAPE_LIFECYCLE_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_TAPE_LIFECYCLE_H

#include "program_tape_scratch.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/runtime_state.h"

namespace vernon::runtime::ad {

struct ProgramTapeState {
    uint32_t value{};
    VernonLaunchSize grid{};
    VernonLaunchSize workgroup{};
    const program::TargetBinding *tape{};
    const program::TargetBinding *segment{};
    const program::TargetBinding *status{};
    const program::TargetBinding *launch{};
    size_t stride{16};
};

bool allocateProgramTapeState(ProgramTapeScratch &scratch, VernonRuntimeContext &context, ProgramTapeState &state,
                              std::string &error);
bool prepareProgramTapeStates(program_execution::ProgramInvocationState &frame, ProgramTapeScratch &scratch,
                              VernonRuntimeContext &context, const program::Program &execution,
                              const program::ResolvedExecutionPlan &topology, const program::Graph &forward,
                              std::vector<ProgramTapeState> &states, std::string &error);
bool validateProgramTapeStates(ProgramTapeScratch &scratch, std::vector<ProgramTapeState> &states, bool &retry,
                               std::string &error);

} // namespace vernon::runtime::ad

#endif
