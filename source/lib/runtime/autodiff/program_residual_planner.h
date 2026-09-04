#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_RESIDUAL_PLANNER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_RESIDUAL_PLANNER_H

#include "execution_graph/execution_graph_checkpoint_planner_internal.h"
#include "runtime/autodiff/program_value_arena.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"

namespace vernon::runtime::ad {

struct ProgramResidualPlan {
    execution::AutodiffDagCheckpointPlan checkpoint;
    std::vector<uint32_t> retainedValues;
    uint32_t replayEnd{};
};

std::vector<AutodiffPullbackPassTelemetry> collectProgramPassTelemetry(const program::Graph &forward,
                                                                       const program::Program &execution,
                                                                       const std::vector<ProgramHostValue> &storage,
                                                                       const ProgramResidualPlan &plan);

bool planProgramResiduals(const program::Program &execution, const VernonProgramTopology *topology,
                          const Variant &variant, const std::vector<ProgramHostValue> &materialized,
                          uint64_t memoryBudget, const std::string &policy, bool rematerializeTapes,
                          ProgramResidualPlan &result, std::string &error);

} // namespace vernon::runtime::ad

#endif
