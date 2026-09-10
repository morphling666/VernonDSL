#ifndef VERNON_EXECUTION_GRAPH_CHECKPOINT_PLANNER_INTERNAL_H
#define VERNON_EXECUTION_GRAPH_CHECKPOINT_PLANNER_INTERNAL_H

#include "execution_graph/command_graph.h"

#include <limits>

namespace vernon::execution {

struct AutodiffReplaySegment {
    uint32_t beginStep{};
    uint32_t endStep{};
    uint32_t checkpointIndex{std::numeric_limits<uint32_t>::max()};
    uint64_t logicalResidualBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t peakBytes{};
    uint64_t replayCost{};
    std::vector<uint32_t> releaseCheckpointResources;
};

struct AutodiffResourceVersion {
    uint32_t resource{};
    uint32_t epoch{};
};

enum class AutodiffVersionSource : uint8_t {
    RetainedOwner,
    InitialState,
    Checkpoint,
    Replay,
};

struct AutodiffRequiredResourceVersion {
    uint32_t consumer{};
    AutodiffResourceVersion version;
    uint32_t producer{std::numeric_limits<uint32_t>::max()};
    AutodiffVersionSource source{AutodiffVersionSource::RetainedOwner};
    std::string path;
};

struct AutodiffPassVersionState {
    std::vector<AutodiffResourceVersion> inputs;
    std::vector<AutodiffResourceVersion> outputs;
};

struct AutodiffCheckpointResource {
    uint32_t producer{};
    AutodiffResourceVersion version;
    uint64_t offset{};
    uint64_t byteSize{};
    uint64_t alignment{1};
    uint32_t firstCut{std::numeric_limits<uint32_t>::max()};
    uint32_t lastCut{};
};

struct AutodiffLivenessCut {
    uint32_t scheduleOffset{};
    std::vector<uint32_t> checkpointResources;
};

struct AutodiffDagCheckpointPlan {
    std::vector<AutodiffCheckpointResource> checkpointResources;
    std::vector<AutodiffLivenessCut> cuts;
    std::vector<AutodiffReplaySegment> replaySegments;
    std::vector<AutodiffPassVersionState> passVersions;
    std::vector<AutodiffRequiredResourceVersion> requiredVersions;
    uint64_t persistentCheckpointBytes{};
    uint64_t initialStateBytes{};
    uint64_t restorationBytes{};
    uint64_t transactionBytes{};
    uint64_t logicalResidualBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t maximumForwardPeakBytes{};
    uint64_t backwardValueBytes{};
    uint64_t memoryBudget{};
    uint64_t peakBytes{};
    uint64_t replayCost{};
    uint64_t captureStoreBytes{};
    uint64_t backwardLoadBytes{};
    uint64_t checkpointCopyBytes{};
    uint64_t resourceReloadCost{};
    uint64_t recomputationCost{};
    uint64_t weightedRuntimeCost{};
    bool deterministicReductionLegal{true};
    std::string selectedPolicy{"balanced"};
};

namespace detail {

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

} // namespace detail
} // namespace vernon::execution

#endif
