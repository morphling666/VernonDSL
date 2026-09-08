#include "program_residual_planner.h"

#include "program_invocation_values.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <string_view>

namespace vernon::runtime::ad {
using program_execution::ProgramValueState;
namespace {

const Parameter *findParameter(const StageBindingPlan &stagePlan, const std::string &name) {
    const auto external = std::find_if(stagePlan.parameters.begin(), stagePlan.parameters.end(),
                                       [&](const Parameter &parameter) { return parameter.name == name; });
    if (external != stagePlan.parameters.end())
        return &*external;
    const auto internal = std::find_if(stagePlan.runtimeParameters.begin(), stagePlan.runtimeParameters.end(),
                                       [&](const Parameter &parameter) { return parameter.name == name; });
    return internal == stagePlan.runtimeParameters.end() ? nullptr : &*internal;
}

execution::detail::AutodiffCheckpointPolicy checkpointPolicy(const std::string &name) {
    if (name == "min_memory")
        return execution::detail::AutodiffCheckpointPolicy::MinMemory;
    if (name == "min_runtime")
        return execution::detail::AutodiffCheckpointPolicy::MinRuntime;
    return execution::detail::AutodiffCheckpointPolicy::Balanced;
}

std::string passName(const program::Node &node) {
    std::string name = node.name;
    const std::string_view suffixes[] = {".forward_with_tape", ".vjp"};
    for (std::string_view suffix : suffixes)
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            name.resize(name.size() - suffix.size());
            break;
        }
    if (name.find('.') == std::string::npos)
        name.insert(0, "program.");
    return name;
}

} // namespace

std::vector<AutodiffPullbackPassTelemetry> collectProgramPassTelemetry(const program::Graph &forward,
                                                                       const program::Program &execution,
                                                                       const std::vector<ProgramValueState> &storage,
                                                                       const ProgramTapeScratch &tapeScratch,
                                                                       const ProgramResidualPlan &plan) {
    std::vector<char> retained(execution.values.size());
    for (uint32_t value : plan.retainedValues)
        if (value < retained.size())
            retained[value] = 1;
    std::vector<char> captured(execution.values.size());
    for (uint32_t value : program::residualCaptures(execution))
        if (value < captured.size())
            captured[value] = 1;
    std::vector<AutodiffPullbackPassTelemetry> telemetry;
    telemetry.reserve(forward.nodes.size());
    for (const program::Node &node : forward.nodes) {
        AutodiffPullbackPassTelemetry item;
        item.scheduleOffset = node.id;
        item.passName = passName(node);
        uint64_t estimated = 0;
        uint64_t logical = 0;
        uint64_t resident = 0;
        uint64_t allocated = 0;
        uint64_t retainedBytes = 0;
        bool hasTape = false;
        bool keptTape = false;
        bool dynamicTape = false;
        bool hasControlHistory = false;
        bool hasTensorResidual = false;
        for (uint32_t value : node.results) {
            if (value >= storage.size())
                continue;
            const ProgramValueState &slot = storage[value];
            const auto batch = tapeScratch.hostBatch(value);
            if (batch) {
                hasTape = true;
                estimated += batch->logicalBytes();
                dynamicTape |= batch->hasDynamicLanes();
                hasControlHistory |= batch->hasControlHistory();
                if (value < retained.size() && retained[value]) {
                    keptTape = true;
                    logical += batch->logicalBytes();
                    resident += batch->isCompacted() ? batch->logicalBytes() : batch->residentBytes();
                    allocated += batch->allocatedBytes();
                    retainedBytes += batch->allocatedBytes();
                }
            } else if (value < captured.size() && captured[value] && value < retained.size() && retained[value]) {
                hasTensorResidual = true;
                retainedBytes += slot.argument.tensor.byte_size;
            }
        }
        item.estimatedTapeBytes = estimated;
        item.logicalResidualBytes = hasControlHistory ? logical : 0;
        item.residentTapeBytes = hasControlHistory ? resident : 0;
        item.allocatedTapeBytes = hasControlHistory ? allocated : 0;
        item.retainedAllocationBytes = retainedBytes;
        if (hasTape) {
            const char *kind = dynamicTape ? "dynamic_capture" : "static_capture";
            item.residualSourceKind = keptTape ? kind : std::string(kind) + "+pure_rematerialization";
            item.controlHistoryKind = hasControlHistory && keptTape ? "dynamic_capture" : "none";
            item.peakTemporaryTapeBytes = std::max(allocated, estimated);
            if (!item.peakTemporaryTapeBytes)
                item.peakTemporaryTapeBytes = 1;
        } else if (hasTensorResidual) {
            item.residualSourceKind = "static_capture";
            item.controlHistoryKind = "none";
        }
        telemetry.push_back(std::move(item));
    }
    return telemetry;
}

bool planProgramResiduals(const program::Program &execution, const program::ResolvedExecutionPlan *topology,
                          const StageBindingPlan &stagePlan, const std::vector<ProgramValueState> &materialized,
                          const ProgramTapeScratch &tapeScratch, uint64_t memoryBudget, const std::string &policy,
                          bool rematerializeTapes, ProgramResidualPlan &result, std::string &error) {
    const program::Graph *forward = program::findGraph(execution, "forward");
    if (!forward)
        return error = "Program autodiff topology has no forward graph", false;
    std::vector<std::optional<uint32_t>> producers(execution.values.size());
    std::vector<execution::detail::AutodiffDagNode> nodes(forward->nodes.size());
    const size_t graphIndex = static_cast<size_t>(forward - execution.graphs.data());
    if (!topology || !topology->resolvedProgram || graphIndex >= topology->resolvedProgram->graphs.size())
        return error = "Program autodiff topology has no resolved dependency graph", false;
    const program::ResolvedGraph &resolvedGraph = topology->resolvedProgram->graphs[graphIndex];
    for (const program::Node &node : forward->nodes) {
        if (node.id >= resolvedGraph.predecessors.size())
            return error = "Program autodiff node has no resolved dependency entry", false;
        execution::detail::AutodiffDagNode &planned = nodes[node.id];
        planned.predecessors = resolvedGraph.predecessors[node.id];
        planned.replayable = true;
        planned.replayCost = 1;
        planned.recomputationCost = 1;
        for (uint32_t value : node.results)
            producers[value] = node.id;
    }
    const auto materializedBytes = [&](uint32_t value, const program::Value &slot,
                                       const Parameter *parameter) -> std::optional<size_t> {
        if (const auto batch = tapeScratch.hostBatch(value))
            return std::max<size_t>(batch->logicalBytes(), 1);
        if (value < materialized.size() && materialized[value].argument.kind == VERNON_PROGRAM_TENSOR &&
            (materialized[value].argument.tensor.byte_size || materialized[value].concreteShape))
            return materialized[value].argument.tensor.byte_size;
        return program::isTapeValueType(slot.type) ? std::optional<size_t>(1) : programValueByteSize(slot, parameter);
    };
    uint64_t initialStateBytes = 0;
    for (const program::GraphInput &input : forward->inputs) {
        if (input.kind != program::GraphInputKind::UserInput)
            continue;
        const program::Value &slot = execution.values[input.value];
        const std::optional<size_t> bytes = materializedBytes(input.value, slot, findParameter(stagePlan, slot.name));
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - initialStateBytes)
            return error = "Program autodiff initial state size overflows", false;
        initialStateBytes += *bytes;
    }
    for (uint32_t value : program::residualCaptures(execution)) {
        if (!producers[value] || (rematerializeTapes && program::isTapeValueType(execution.values[value].type)))
            continue;
        const program::Value &slot = execution.values[value];
        const std::optional<size_t> bytes = materializedBytes(value, slot, findParameter(stagePlan, slot.name));
        execution::detail::AutodiffDagNode &node = nodes[*producers[value]];
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - node.residualBytes)
            return error = "Program autodiff residual size overflows", false;
        node.residualBytes += *bytes;
        node.retainedAllocationBytes += *bytes;
        node.forwardPeakBytes += *bytes;
    }
    if (!execution::detail::planDagAutodiffCheckpoints(nodes, memoryBudget, result.checkpoint, error, initialStateBytes,
                                                       true, 0, true, 0, true, 0, checkpointPolicy(policy)))
        return false;
    const uint32_t retainBegin =
        result.checkpoint.replaySegments.empty() ? 0 : result.checkpoint.replaySegments.back().beginStep;
    for (uint32_t value : program::residualCaptures(execution))
        if (producers[value] && ((rematerializeTapes && program::isTapeValueType(execution.values[value].type)) ||
                                 *producers[value] < retainBegin))
            result.replayEnd = std::max(result.replayEnd, *producers[value] + 1);
    for (uint32_t value : program::residualCaptures(execution))
        if (!producers[value] || *producers[value] >= result.replayEnd)
            result.retainedValues.push_back(value);
    // Invocation controls are ordinary canonical Values. Retain them through
    // the same dependency state as data values so pullback replay never needs
    // a parallel control cache.
    for (const program::Graph &graph : execution.graphs)
        for (const program::GraphInput &input : graph.inputs)
            if (input.kind == program::GraphInputKind::InvocationControl)
                result.retainedValues.push_back(input.value);
    std::sort(result.retainedValues.begin(), result.retainedValues.end());
    result.retainedValues.erase(std::unique(result.retainedValues.begin(), result.retainedValues.end()),
                                result.retainedValues.end());
    std::set<uint32_t> retainedStorages;
    for (uint32_t value : result.retainedValues)
        if (value < execution.values.size() && execution.values[value].storage)
            retainedStorages.insert(*execution.values[value].storage);
    for (const program::Value &value : execution.values)
        if (value.storage && retainedStorages.count(*value.storage))
            result.retainedValues.push_back(value.id);
    std::sort(result.retainedValues.begin(), result.retainedValues.end());
    result.retainedValues.erase(std::unique(result.retainedValues.begin(), result.retainedValues.end()),
                                result.retainedValues.end());
    uint64_t logicalResidualBytes = 0;
    for (uint32_t value : program::residualCaptures(execution)) {
        if (!producers[value])
            continue;
        const program::Value &slot = execution.values[value];
        const bool retained = std::binary_search(result.retainedValues.begin(), result.retainedValues.end(), value) ||
                              (slot.storage && retainedStorages.find(*slot.storage) != retainedStorages.end());
        if (!retained)
            continue;
        const std::optional<size_t> bytes = materializedBytes(value, slot, findParameter(stagePlan, slot.name));
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - logicalResidualBytes)
            return error = "Program autodiff logical residual size overflows", false;
        logicalResidualBytes += *bytes;
    }
    if (rematerializeTapes)
        result.checkpoint.logicalResidualBytes = logicalResidualBytes;
    if (result.replayEnd) {
        for (const program::GraphInput &input : forward->inputs)
            if (input.kind == program::GraphInputKind::UserInput)
                result.retainedValues.push_back(input.value);
        std::sort(result.retainedValues.begin(), result.retainedValues.end());
        result.retainedValues.erase(std::unique(result.retainedValues.begin(), result.retainedValues.end()),
                                    result.retainedValues.end());
    }
    return true;
}

} // namespace vernon::runtime::ad
