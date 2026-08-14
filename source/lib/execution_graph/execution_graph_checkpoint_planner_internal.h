#ifndef VERNON_EXECUTION_GRAPH_CHECKPOINT_PLANNER_INTERNAL_H
#define VERNON_EXECUTION_GRAPH_CHECKPOINT_PLANNER_INTERNAL_H

#include "VernonExecutionGraph.h"

namespace vernon::execution::detail {

struct AutodiffDagOutput {
    uint32_t resource{};
    uint64_t byteSize{};
    uint64_t alignment{1};
    bool checkpointable{};
    std::vector<uint32_t> consumers;
};

struct AutodiffDagNode {
    std::vector<uint32_t> predecessors;
    std::vector<AutodiffDagOutput> outputs;
    uint64_t residualBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t forwardPeakBytes{};
    uint64_t replayCost{};
    bool replayable{};
};

bool planDagAutodiffCheckpoints(const std::vector<AutodiffDagNode> &nodes, uint64_t memoryBudget,
                                AutodiffDagCheckpointPlan &output, std::string &error, uint64_t initialStateBytes = 0,
                                bool initialStateCheckpointable = true, uint64_t restorationBytes = 0,
                                bool restorationCheckpointable = true, uint64_t transactionBytes = 0,
                                bool transactionCheckpointable = true);

} // namespace vernon::execution::detail

#endif
