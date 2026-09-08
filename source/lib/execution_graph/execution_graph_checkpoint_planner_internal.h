#ifndef VERNON_EXECUTION_GRAPH_CHECKPOINT_PLANNER_INTERNAL_H
#define VERNON_EXECUTION_GRAPH_CHECKPOINT_PLANNER_INTERNAL_H

#include "VernonExecutionGraph.h"

namespace vernon::execution::detail {

struct AutodiffDagOutput {
    AutodiffResourceVersion version;
    uint64_t byteSize{};
    uint64_t alignment{1};
    bool checkpointable{};
    std::vector<uint32_t> consumers;
};

struct AutodiffDagRequiredVersion {
    std::string path;
    AutodiffResourceVersion version;
    uint32_t producer{std::numeric_limits<uint32_t>::max()};
};

struct AutodiffDagNode {
    std::vector<uint32_t> predecessors;
    std::vector<AutodiffResourceVersion> inputs;
    std::vector<AutodiffDagOutput> outputs;
    std::vector<AutodiffDagRequiredVersion> requiredVersions;
    uint64_t residualBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t forwardPeakBytes{};
    uint64_t replayCost{};
    bool replayable{};
    uint64_t resourceReloadCost{};
    uint64_t recomputationCost{};
    bool deterministicReductionLegal{true};
};

enum class AutodiffCheckpointPolicy {
    MinMemory,
    Balanced,
    MinRuntime,
};

bool planDagAutodiffCheckpoints(const std::vector<AutodiffDagNode> &nodes, uint64_t memoryBudget,
                                AutodiffDagCheckpointPlan &output, std::string &error, uint64_t initialStateBytes = 0,
                                bool initialStateCheckpointable = true, uint64_t restorationBytes = 0,
                                bool restorationCheckpointable = true, uint64_t transactionBytes = 0,
                                bool transactionCheckpointable = true, uint64_t backwardValueBytes = 0,
                                AutodiffCheckpointPolicy policy = AutodiffCheckpointPolicy::Balanced);

} // namespace vernon::execution::detail

#endif
