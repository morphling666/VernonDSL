#include "execution_graph_checkpoint_planner_internal.h"

#include <algorithm>
#include <queue>
#include <set>
#include <unordered_set>

namespace vernon::execution::detail {
namespace {

bool checkedAdd(uint64_t left, uint64_t right, uint64_t &result) {
    if (right > UINT64_MAX - left)
        return false;
    result = left + right;
    return true;
}

bool checkedMultiply(uint64_t left, uint64_t right, uint64_t &result) {
    if (right && left > UINT64_MAX / right)
        return false;
    result = left * right;
    return true;
}

bool checkedAlign(uint64_t value, uint64_t alignment, uint64_t &result) {
    if (!alignment || (alignment & (alignment - 1)) != 0 || !checkedAdd(value, alignment - 1, result))
        return false;
    result &= ~(alignment - 1);
    return true;
}

bool sameVersion(const AutodiffResourceVersion &left, const AutodiffResourceVersion &right) {
    return left.resource == right.resource && left.epoch == right.epoch;
}

struct OutputKey {
    uint32_t producer;
    uint32_t output;
};

bool materializeDagCheckpointPlan(const std::vector<AutodiffDagNode> &nodes, const std::vector<uint32_t> &cutOffsets,
                                  uint64_t initialStateBytes, bool initialStateCheckpointable,
                                  uint64_t restorationBytes, bool restorationCheckpointable, uint64_t transactionBytes,
                                  bool transactionCheckpointable, uint64_t backwardValueBytes,
                                  AutodiffDagCheckpointPlan &output, bool includeVersionMetadata = false) {
    if (!transactionCheckpointable ||
        (!cutOffsets.empty() && (!initialStateCheckpointable || !restorationCheckpointable)))
        return false;
    AutodiffDagCheckpointPlan plan;
    plan.initialStateBytes = cutOffsets.empty() ? 0 : initialStateBytes;
    plan.restorationBytes = cutOffsets.empty() ? 0 : restorationBytes;
    plan.transactionBytes = transactionBytes;
    plan.backwardValueBytes = backwardValueBytes;
    if (includeVersionMetadata) {
        plan.passVersions.reserve(nodes.size());
        for (const AutodiffDagNode &node : nodes) {
            AutodiffPassVersionState state;
            state.inputs = node.inputs;
            state.outputs.reserve(node.outputs.size());
            for (const AutodiffDagOutput &outputValue : node.outputs)
                state.outputs.push_back(outputValue.version);
            plan.passVersions.push_back(std::move(state));
        }
    }
    std::vector<std::vector<OutputKey>> liveOutputs;
    std::vector<OutputKey> checkpointOutputs;
    liveOutputs.reserve(cutOffsets.size());
    for (uint32_t cut : cutOffsets) {
        std::vector<OutputKey> live;
        for (uint32_t producer = 0; producer < cut; ++producer)
            for (uint32_t outputIndex = 0; outputIndex < nodes[producer].outputs.size(); ++outputIndex) {
                const AutodiffDagOutput &value = nodes[producer].outputs[outputIndex];
                if (!std::any_of(value.consumers.begin(), value.consumers.end(),
                                 [cut](uint32_t consumer) { return consumer >= cut; }))
                    continue;
                if (!value.checkpointable)
                    return false;
                live.push_back({producer, outputIndex});
                checkpointOutputs.push_back({producer, outputIndex});
            }
        liveOutputs.push_back(std::move(live));
    }
    std::sort(checkpointOutputs.begin(), checkpointOutputs.end(), [](const OutputKey &left, const OutputKey &right) {
        return left.producer < right.producer || (left.producer == right.producer && left.output < right.output);
    });
    checkpointOutputs.erase(std::unique(checkpointOutputs.begin(), checkpointOutputs.end(),
                                        [](const OutputKey &left, const OutputKey &right) {
                                            return left.producer == right.producer && left.output == right.output;
                                        }),
                            checkpointOutputs.end());

    std::vector<std::vector<uint32_t>> resourceIndices(nodes.size());
    for (uint32_t producer = 0; producer < nodes.size(); ++producer)
        resourceIndices[producer].assign(nodes[producer].outputs.size(), UINT32_MAX);
    for (const OutputKey &key : checkpointOutputs) {
        const AutodiffDagOutput &value = nodes[key.producer].outputs[key.output];
        uint64_t offset = 0;
        uint64_t end = 0;
        if (!checkedAlign(plan.persistentCheckpointBytes, value.alignment, offset) ||
            !checkedAdd(offset, value.byteSize, end))
            return false;
        resourceIndices[key.producer][key.output] = static_cast<uint32_t>(plan.checkpointResources.size());
        plan.checkpointResources.push_back({key.producer, value.version, offset, value.byteSize, value.alignment});
        plan.persistentCheckpointBytes = end;
    }
    for (size_t cutIndex = 0; cutIndex < cutOffsets.size(); ++cutIndex) {
        AutodiffLivenessCut cut;
        cut.scheduleOffset = cutOffsets[cutIndex];
        for (const OutputKey &key : liveOutputs[cutIndex]) {
            const uint32_t resourceIndex = resourceIndices[key.producer][key.output];
            if (resourceIndex >= plan.checkpointResources.size())
                return false;
            AutodiffCheckpointResource &resource = plan.checkpointResources[resourceIndex];
            resource.firstCut = std::min(resource.firstCut, static_cast<uint32_t>(cutIndex));
            resource.lastCut = static_cast<uint32_t>(cutIndex);
            cut.checkpointResources.push_back(resourceIndex);
        }
        plan.cuts.push_back(std::move(cut));
    }

    std::vector<uint32_t> boundaries;
    boundaries.reserve(cutOffsets.size() + 2);
    boundaries.push_back(0);
    boundaries.insert(boundaries.end(), cutOffsets.begin(), cutOffsets.end());
    boundaries.push_back(static_cast<uint32_t>(nodes.size()));
    uint64_t maximumSegmentPeak = 0;
    uint64_t maximumSegmentRetained = 0;
    for (size_t segmentIndex = 1; segmentIndex < boundaries.size(); ++segmentIndex) {
        AutodiffReplaySegment segment;
        segment.beginStep = boundaries[segmentIndex - 1];
        segment.endStep = boundaries[segmentIndex];
        if (segmentIndex > 1) {
            segment.checkpointIndex = static_cast<uint32_t>(segmentIndex - 2);
            const uint32_t cutIndex = segment.checkpointIndex;
            for (uint32_t resourceIndex = 0; resourceIndex < plan.checkpointResources.size(); ++resourceIndex)
                if (plan.checkpointResources[resourceIndex].firstCut == cutIndex)
                    segment.releaseCheckpointResources.push_back(resourceIndex);
        }
        uint64_t retainedBytes = 0;
        uint64_t logicalBytes = 0;
        uint64_t segmentPeak = 0;
        for (uint32_t node = segment.beginStep; node < segment.endStep; ++node) {
            uint64_t forwardPeak = 0;
            if (!checkedAdd(retainedBytes, nodes[node].forwardPeakBytes, forwardPeak) ||
                !checkedAdd(retainedBytes, nodes[node].retainedAllocationBytes, retainedBytes) ||
                !checkedAdd(logicalBytes, nodes[node].residualBytes, logicalBytes) ||
                !checkedAdd(segment.replayCost, nodes[node].replayCost, segment.replayCost))
                return false;
            segmentPeak = std::max(segmentPeak, forwardPeak);
        }
        segmentPeak = std::max(segmentPeak, retainedBytes);
        segment.logicalResidualBytes = logicalBytes;
        segment.retainedAllocationBytes = retainedBytes;
        segment.peakBytes = segmentPeak;
        if (!checkedAdd(plan.replayCost, segment.replayCost, plan.replayCost))
            return false;
        maximumSegmentPeak = std::max(maximumSegmentPeak, segmentPeak);
        maximumSegmentRetained = std::max(maximumSegmentRetained, retainedBytes);
        plan.replaySegments.push_back(segment);
    }
    if (plan.replaySegments.size() == 1)
        plan.replayCost = 0;
    uint64_t basePeak = 0;
    uint64_t forwardPeak = 0;
    uint64_t replayPeak = 0;
    uint64_t backwardPeak = 0;
    if (!checkedAdd(plan.persistentCheckpointBytes, plan.initialStateBytes, basePeak) ||
        !checkedAdd(basePeak, maximumSegmentPeak, basePeak) ||
        !checkedAdd(basePeak, plan.transactionBytes, forwardPeak) ||
        !checkedAdd(basePeak, plan.restorationBytes, replayPeak))
        return false;
    if (!checkedAdd(plan.persistentCheckpointBytes, plan.initialStateBytes, backwardPeak) ||
        !checkedAdd(backwardPeak, maximumSegmentRetained, backwardPeak) ||
        !checkedAdd(backwardPeak, plan.restorationBytes, backwardPeak) ||
        !checkedAdd(backwardPeak, plan.backwardValueBytes, backwardPeak))
        return false;
    plan.peakBytes = std::max({forwardPeak, replayPeak, backwardPeak});
    for (const AutodiffDagNode &node : nodes) {
        plan.deterministicReductionLegal &= node.deterministicReductionLegal;
        if (!checkedAdd(plan.captureStoreBytes, node.residualBytes, plan.captureStoreBytes) ||
            !checkedAdd(plan.backwardLoadBytes, node.residualBytes, plan.backwardLoadBytes) ||
            !checkedAdd(plan.logicalResidualBytes, node.residualBytes, plan.logicalResidualBytes) ||
            !checkedAdd(plan.retainedAllocationBytes, node.retainedAllocationBytes, plan.retainedAllocationBytes) ||
            !checkedAdd(plan.resourceReloadCost, node.resourceReloadCost, plan.resourceReloadCost) ||
            !checkedAdd(plan.recomputationCost, node.recomputationCost, plan.recomputationCost))
            return false;
        plan.maximumForwardPeakBytes = std::max(plan.maximumForwardPeakBytes, node.forwardPeakBytes);
    }
    if (plan.replaySegments.size() > 1 && !checkedMultiply(plan.captureStoreBytes, 2, plan.captureStoreBytes))
        return false;
    if (!checkedMultiply(plan.persistentCheckpointBytes, 2, plan.checkpointCopyBytes) ||
        !checkedAdd(plan.captureStoreBytes, plan.backwardLoadBytes, plan.weightedRuntimeCost) ||
        !checkedAdd(plan.weightedRuntimeCost, plan.checkpointCopyBytes, plan.weightedRuntimeCost)) {
        return false;
    }
    uint64_t weightedReload = 0;
    uint64_t weightedRecomputation = 0;
    if (!checkedMultiply(plan.resourceReloadCost, 4, weightedReload) ||
        !checkedMultiply(plan.recomputationCost, 8, weightedRecomputation) ||
        !checkedAdd(plan.weightedRuntimeCost, weightedReload, plan.weightedRuntimeCost) ||
        !checkedAdd(plan.weightedRuntimeCost, weightedRecomputation, plan.weightedRuntimeCost))
        return false;
    uint64_t weightedReplay = 0;
    if (!checkedMultiply(plan.replayCost, 16, weightedReplay) ||
        !checkedAdd(plan.weightedRuntimeCost, weightedReplay, plan.weightedRuntimeCost))
        return false;
    if (includeVersionMetadata) {
        for (uint32_t consumer = 0; consumer < nodes.size(); ++consumer)
            for (const AutodiffDagRequiredVersion &required : nodes[consumer].requiredVersions) {
                AutodiffRequiredResourceVersion planned;
                planned.consumer = consumer;
                planned.version = required.version;
                planned.producer = required.producer;
                planned.path = required.path;
                const auto segment = std::find_if(plan.replaySegments.begin(), plan.replaySegments.end(),
                                                  [&](const AutodiffReplaySegment &value) {
                                                      return consumer >= value.beginStep && consumer < value.endStep;
                                                  });
                if (required.producer == UINT32_MAX) {
                    planned.source = plan.initialStateBytes ? AutodiffVersionSource::InitialState
                                                            : AutodiffVersionSource::RetainedOwner;
                } else {
                    bool restoredFromStartingCut = false;
                    if (segment != plan.replaySegments.end() && segment->checkpointIndex != UINT32_MAX &&
                        segment->checkpointIndex < plan.cuts.size())
                        for (uint32_t resourceIndex : plan.cuts[segment->checkpointIndex].checkpointResources)
                            if (resourceIndex < plan.checkpointResources.size() &&
                                sameVersion(plan.checkpointResources[resourceIndex].version, required.version)) {
                                restoredFromStartingCut = true;
                                break;
                            }
                    if (restoredFromStartingCut)
                        planned.source = AutodiffVersionSource::Checkpoint;
                    else if (segment != plan.replaySegments.end() && segment->checkpointIndex != UINT32_MAX &&
                             required.producer >= segment->beginStep)
                        planned.source = AutodiffVersionSource::Replay;
                    else
                        planned.source = AutodiffVersionSource::RetainedOwner;
                }
                plan.requiredVersions.push_back(std::move(planned));
            }
    }
    output = std::move(plan);
    return true;
}

} // namespace

bool planDagAutodiffCheckpoints(const std::vector<AutodiffDagNode> &nodes, uint64_t memoryBudget,
                                AutodiffDagCheckpointPlan &output, std::string &error, uint64_t initialStateBytes,
                                bool initialStateCheckpointable, uint64_t restorationBytes,
                                bool restorationCheckpointable, uint64_t transactionBytes,
                                bool transactionCheckpointable, uint64_t backwardValueBytes,
                                AutodiffCheckpointPolicy policy) {
    output = {};
    error.clear();
    if (nodes.size() > UINT32_MAX) {
        error = "autodiff DAG exceeds the checkpoint planner representation";
        return false;
    }
    std::unordered_set<uint64_t> outputVersions;
    for (uint32_t node = 0; node < nodes.size(); ++node) {
        const AutodiffDagNode &descriptor = nodes[node];
        std::unordered_set<uint32_t> uniquePredecessors;
        for (uint32_t predecessor : descriptor.predecessors) {
            if (predecessor >= node || !uniquePredecessors.insert(predecessor).second) {
                error = predecessor >= node ? "autodiff DAG nodes are not in topological order"
                                            : "autodiff DAG node contains a duplicate predecessor";
                return false;
            }
        }
        for (const AutodiffResourceVersion &input : descriptor.inputs) {
            const uint64_t key = (static_cast<uint64_t>(input.resource) << 32u) | input.epoch;
            if (input.epoch && outputVersions.find(key) == outputVersions.end()) {
                error = "autodiff DAG input references an unavailable resource version";
                return false;
            }
        }
        std::unordered_set<uint32_t> resources;
        for (const AutodiffDagOutput &outputValue : descriptor.outputs) {
            const uint64_t versionKey =
                (static_cast<uint64_t>(outputValue.version.resource) << 32u) | outputValue.version.epoch;
            if (!outputValue.version.epoch || !resources.insert(outputValue.version.resource).second ||
                !outputVersions.insert(versionKey).second ||
                (outputValue.checkpointable &&
                 (!outputValue.alignment || (outputValue.alignment & (outputValue.alignment - 1)) != 0))) {
                error = "autodiff DAG node contains an invalid output";
                return false;
            }
            uint32_t previous = 0;
            bool first = true;
            for (uint32_t consumer : outputValue.consumers) {
                if (consumer <= node || consumer >= nodes.size() || (!first && consumer <= previous)) {
                    error = "autodiff DAG output contains an invalid consumer";
                    return false;
                }
                if (std::find(nodes[consumer].predecessors.begin(), nodes[consumer].predecessors.end(), node) ==
                    nodes[consumer].predecessors.end()) {
                    error = "autodiff DAG output consumer is not a scheduling successor";
                    return false;
                }
                first = false;
                previous = consumer;
            }
        }
        std::unordered_set<std::string> requiredPaths;
        for (const AutodiffDagRequiredVersion &required : descriptor.requiredVersions) {
            const bool initial = required.producer == UINT32_MAX;
            const bool producerMatches =
                initial ? required.version.epoch == 0
                        : required.producer < node &&
                              std::any_of(
                                  nodes[required.producer].outputs.begin(), nodes[required.producer].outputs.end(),
                                  [&](const AutodiffDagOutput &output) {
                                      return sameVersion(output.version, required.version);
                                  });
            if (required.path.empty() || !requiredPaths.insert(required.path).second || !producerMatches) {
                error = "autodiff DAG contains an invalid required resource version";
                return false;
            }
        }
    }
    if (!transactionCheckpointable) {
        error = "autodiff DAG forward transaction requires checkpointable writable resources";
        return false;
    }

    struct Candidate {
        std::vector<uint32_t> cuts;
        AutodiffDagCheckpointPlan plan;
    };
    struct WorseCandidate {
        bool operator()(const Candidate &left, const Candidate &right) const {
            if (left.plan.peakBytes != right.plan.peakBytes)
                return left.plan.peakBytes > right.plan.peakBytes;
            if (left.plan.replayCost != right.plan.replayCost)
                return left.plan.replayCost > right.plan.replayCost;
            if (left.cuts.size() != right.cuts.size())
                return left.cuts.size() > right.cuts.size();
            return left.cuts > right.cuts;
        }
    };

    Candidate initial;
    if (!materializeDagCheckpointPlan(nodes, initial.cuts, initialStateBytes, initialStateCheckpointable,
                                      restorationBytes, restorationCheckpointable, transactionBytes,
                                      transactionCheckpointable, backwardValueBytes, initial.plan)) {
        error = "autodiff DAG checkpoint memory or replay cost overflows";
        return false;
    }

    std::priority_queue<Candidate, std::vector<Candidate>, WorseCandidate> frontier;
    std::set<std::vector<uint32_t>> visited;
    visited.insert(initial.cuts);
    frontier.push(std::move(initial));
    constexpr size_t maximumCandidates = 65536;
    const bool deterministicReplay = std::all_of(
        nodes.begin(), nodes.end(), [](const AutodiffDagNode &node) { return node.deterministicReductionLegal; });
    const bool replayable =
        deterministicReplay &&
        std::all_of(nodes.begin(), nodes.end(), [](const AutodiffDagNode &node) { return node.replayable; });
    bool replayBlocked = false;
    std::optional<Candidate> bestFeasible;
    auto policyName = [&]() -> const char * {
        switch (policy) {
        case AutodiffCheckpointPolicy::MinMemory:
            return "min_memory";
        case AutodiffCheckpointPolicy::Balanced:
            return "balanced";
        case AutodiffCheckpointPolicy::MinRuntime:
            return "min_runtime";
        }
        return "balanced";
    };
    auto betterFeasible = [&](const Candidate &left, const Candidate &right) {
        if (policy == AutodiffCheckpointPolicy::MinMemory) {
            if (left.plan.peakBytes != right.plan.peakBytes)
                return left.plan.peakBytes < right.plan.peakBytes;
            if (left.plan.weightedRuntimeCost != right.plan.weightedRuntimeCost)
                return left.plan.weightedRuntimeCost < right.plan.weightedRuntimeCost;
        } else if (policy == AutodiffCheckpointPolicy::MinRuntime) {
            if (left.plan.weightedRuntimeCost != right.plan.weightedRuntimeCost)
                return left.plan.weightedRuntimeCost < right.plan.weightedRuntimeCost;
            if (left.plan.peakBytes != right.plan.peakBytes)
                return left.plan.peakBytes < right.plan.peakBytes;
        } else {
            uint64_t leftScore = 0;
            uint64_t rightScore = 0;
            if (!checkedAdd(left.plan.weightedRuntimeCost, left.plan.peakBytes, leftScore))
                leftScore = UINT64_MAX;
            if (!checkedAdd(right.plan.weightedRuntimeCost, right.plan.peakBytes, rightScore))
                rightScore = UINT64_MAX;
            if (leftScore != rightScore)
                return leftScore < rightScore;
            if (left.plan.peakBytes != right.plan.peakBytes)
                return left.plan.peakBytes < right.plan.peakBytes;
        }
        return left.cuts < right.cuts;
    };
    while (!frontier.empty()) {
        Candidate candidate = frontier.top();
        frontier.pop();
        if (candidate.plan.peakBytes <= memoryBudget) {
            candidate.plan.memoryBudget = memoryBudget;
            candidate.plan.selectedPolicy = policyName();
            if (!bestFeasible || betterFeasible(candidate, *bestFeasible))
                bestFeasible = candidate;
            if (policy == AutodiffCheckpointPolicy::MinRuntime)
                continue;
        }
        if (!replayable) {
            replayBlocked |= nodes.size() > 1;
            continue;
        }
        for (uint32_t cut = 1; cut < nodes.size(); ++cut) {
            if (std::binary_search(candidate.cuts.begin(), candidate.cuts.end(), cut))
                continue;
            std::vector<uint32_t> trialCuts = candidate.cuts;
            trialCuts.insert(std::lower_bound(trialCuts.begin(), trialCuts.end(), cut), cut);
            if (!visited.insert(trialCuts).second)
                continue;
            if (visited.size() > maximumCandidates) {
                if (bestFeasible) {
                    frontier = {};
                    break;
                }
                error = "autodiff DAG checkpoint search exceeded its bounded frontier";
                output = {};
                return false;
            }
            Candidate trial;
            trial.cuts = std::move(trialCuts);
            if (materializeDagCheckpointPlan(nodes, trial.cuts, initialStateBytes, initialStateCheckpointable,
                                             restorationBytes, restorationCheckpointable, transactionBytes,
                                             transactionCheckpointable, backwardValueBytes, trial.plan))
                frontier.push(std::move(trial));
        }
    }
    if (bestFeasible) {
        if (!materializeDagCheckpointPlan(nodes, bestFeasible->cuts, initialStateBytes, initialStateCheckpointable,
                                          restorationBytes, restorationCheckpointable, transactionBytes,
                                          transactionCheckpointable, backwardValueBytes, output, true)) {
            error = "autodiff DAG checkpoint memory or replay cost overflows";
            output = {};
            return false;
        }
        output.memoryBudget = memoryBudget;
        output.selectedPolicy = policyName();
        return true;
    }
    error = replayBlocked && !deterministicReplay
                ? "autodiff DAG checkpoint schedule violates deterministic reduction constraints"
            : replayBlocked ? "autodiff DAG checkpoint schedule requires replaying a non-replayable node"
                            : "autodiff DAG checkpoint schedule cannot satisfy the memory budget";
    output = {};
    return false;
}

} // namespace vernon::execution::detail
