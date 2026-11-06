#include "resolved_execution_plan.h"

#include "runtime_state.h"

#include <algorithm>
#include <unordered_map>

namespace vernon::runtime::program {
namespace {

ExecutionPlanResult reject(std::string path, std::string message) {
    return ExecutionPlanResult{
        vernon::err(Diagnostic{"PROGRAM_EXECUTION_PLAN", "resolve", std::move(path), std::move(message)})};
}

} // namespace

std::optional<GraphDirection> graphDirection(std::string_view direction) {
    if (direction == "forward")
        return GraphDirection::Forward;
    if (direction == "backward")
        return GraphDirection::Backward;
    if (direction == "replay")
        return GraphDirection::Replay;
    return std::nullopt;
}

const ResolvedNodePlan *ResolvedExecutionPlan::node(GraphDirection graph, uint32_t nodeId) const {
    const auto found = nodes.find({graph, nodeId});
    return found == nodes.end() ? nullptr : &found->second;
}

ExecutionPlanResult completeResolvedExecutionPlan(ResolvedExecutionPlan &plan) {
    if (!plan.resolvedProgram)
        return reject("", "execution plan has no resolved Program");
    const Program &program = plan.resolvedProgram->program;

    plan.declaredValueShapes.resize(program.values.size());
    std::vector<std::optional<shape::ConcreteShape>> stageShapes(program.values.size());
    for (const Value &value : program.values)
        plan.declaredValueShapes[value.id] = shape::decodeRuntimeContractShape(value.shape);
    for (const auto &[key, node] : plan.nodes) {
        if (!node.stage)
            return reject("/graphs", "resolved Node has no Stage");
        if (node.key != key || node.projections.size() != node.stage->bindingProjection.parameters.size())
            return reject("/graphs", "resolved Node projection does not match its Stage endpoint ABI");
        for (const NodeEndpointProjection &projection : node.projections) {
            if (projection.value >= program.values.size())
                return reject("/graphs", "resolved Node projection references a missing Value");
            if (projection.logicalLeaf &&
                (!program.values[projection.value].layout ||
                 *projection.logicalLeaf >= program.values[projection.value].layout->leaves.size()))
                return reject("/graphs", "resolved Node projection references a missing logical leaf");
            if (projection.target.endpoint.interfaceKind.empty())
                return reject("/graphs", "resolved Node projection references a missing endpoint");
        }
        if (!node.stage->bindingProjection.vertex.empty())
            continue;
        const StageBindingPlan &bindings = node.stage->bindingProjection;
        for (size_t index = 0; index < node.projections.size(); ++index) {
            const NodeEndpointProjection &projection = node.projections[index];
            const vernon::runtime::Parameter &parameter = bindings.parameters[index];
            if (projection.logicalLeaf || parameter.invocationCarrier || projection.target.viewTransform ||
                isTapeValue(program.values[projection.value]) ||
                (program.values[projection.value].layout && program.values[projection.value].shape.empty()))
                continue;
            const std::optional<shape::ConcreteShape> concrete =
                shape::concrete(shape::decodeRuntimeContractShape(parameter.shape));
            if (!concrete)
                continue;
            if (!shape::matches(plan.declaredValueShapes[projection.value], *concrete))
                return reject("/values/" + std::to_string(projection.value),
                              "compiled Stage shape conflicts with the Program Value");
            if (stageShapes[projection.value] && *stageShapes[projection.value] != *concrete)
                return reject("/values/" + std::to_string(projection.value),
                              "compiled Stages disagree on the Program Value shape");
            stageShapes[projection.value] = concrete;
        }
    }

    for (size_t graphIndex = 0; graphIndex < program.graphs.size(); ++graphIndex) {
        const Graph &graph = program.graphs[graphIndex];
        const std::optional<GraphDirection> direction = graphDirection(graph.direction);
        if (!direction)
            return reject("/graphs/" + std::to_string(graphIndex), "unsupported graph direction");
        std::unordered_map<uint32_t, size_t> position;
        for (size_t index = 0; index < graph.nodes.size(); ++index)
            position.emplace(graph.nodes[index].id, index);
        for (size_t index = 0; index < graph.nodes.size(); ++index) {
            const Node &node = graph.nodes[index];
            const ResolvedNodePlan *resolved = plan.node(*direction, node.id);
            if (!resolved)
                return reject("/graphs/" + std::to_string(graphIndex) + "/nodes/" + std::to_string(node.id),
                              "canonical Node has no resolved plan");
            for (const NodeEndpointProjection &projection : resolved->projections) {
                const Value &value = program.values[projection.value];
                if (value.origin.kind != OriginKind::NodeResult)
                    continue;
                if (graphDirection(value.origin.graph) != direction) {
                    if (std::find(graph.captures.begin(), graph.captures.end(), projection.value) ==
                        graph.captures.end())
                        return reject("/graphs/" + std::to_string(graphIndex),
                                      "Node reads another graph's result its graph does not capture");
                    continue;
                }
                const auto producer = position.find(value.origin.node);
                if (producer == position.end() || producer->second > index)
                    return reject("/graphs/" + std::to_string(graphIndex),
                                  "Node reads a result no earlier Node of its graph produces");
            }
        }
    }
    return ExecutionPlanResult{vernon::ok()};
}

} // namespace vernon::runtime::program
