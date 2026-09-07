#ifndef VERNON_RUNTIME_AUTODIFF_RETAINED_PULLBACK_STATE_H
#define VERNON_RUNTIME_AUTODIFF_RETAINED_PULLBACK_STATE_H

#include "program_residual_planner.h"
#include "program_tape_scratch.h"
#include "runtime/program_execution/program_invocation_state.h"

namespace vernon::runtime::ad {

class RetainedPullbackState {
public:
    RetainedPullbackState(ProgramResidualPlan plan, const program_execution::ProgramInvocationState &invocation,
                          ProgramTapeScratch tapeScratch);

    const ProgramResidualPlan &residualPlan() const { return residualPlan_; }
    const std::vector<std::vector<uint8_t>> &residualCaptures() const { return residualCaptures_; }
    const std::vector<std::vector<uint64_t>> &captureShapes() const { return captureShapes_; }
    const std::vector<program_execution::CanonicalValueSnapshot> &valueSnapshots() const { return values_; }
    bool importInto(program_execution::ProgramInvocationState &invocation, ProgramTapeScratch &tapeScratch,
                    bool importTape, std::string &error) const;
    const void *snapshotHostIdentity(uint32_t value) const;
    size_t residentBytes() const;
    size_t allocatedTapeBytes() const;

private:
    const ProgramResidualPlan residualPlan_;
    const std::vector<program_execution::CanonicalValueSnapshot> values_;
    const std::map<uint32_t, program_execution::ProgramStorageBacking> storages_;
    const std::vector<std::vector<uint8_t>> residualCaptures_;
    const std::vector<std::vector<uint64_t>> captureShapes_;
    const std::vector<char> tapeValues_;
    const ProgramTapeSnapshot tape_;
};

} // namespace vernon::runtime::ad

#endif
