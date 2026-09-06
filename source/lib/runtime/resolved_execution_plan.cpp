#include "resolved_execution_plan.h"

#include "runtime_state.h"

#include <algorithm>
#include <set>

namespace vernon::runtime::program {
namespace {

bool reject(Diagnostic &diagnostic, std::string path, std::string message) {
    diagnostic = {"PROGRAM_EXECUTION_PLAN", "resolve", std::move(path), std::move(message)};
    return false;
}

ResidencyRequirement bindingResidency(const ResolvedNodePlan &node, const TargetBinding &binding) {
    if (!node.stage || !node.stage->context || node.stage->context->backend == VERNON_RUNTIME_CPU)
        return ResidencyRequirement::Host;
    switch (binding.carrier) {
    case TargetCarrier::StorageBuffer:
    case TargetCarrier::VertexBuffer:
    case TargetCarrier::IndexBuffer:
    case TargetCarrier::Image:
    case TargetCarrier::Sampler:
    case TargetCarrier::Attachment:
        return ResidencyRequirement::Device;
    case TargetCarrier::InlineValue:
    case TargetCarrier::UniformBuffer:
        return ResidencyRequirement::Host;
    }
    return ResidencyRequirement::Host;
}

TransferEndpoint valueProducer(const Value &value, GraphDirection graph) {
    switch (value.origin.kind) {
    case OriginKind::Argument:
        return {TransferEndpointKind::Boundary, graph, value.origin.slot};
    case OriginKind::NodeResult:
        return {TransferEndpointKind::Node, graph, value.origin.node};
    case OriginKind::Parameter:
        return {TransferEndpointKind::Boundary, graph, value.origin.parameter};
    case OriginKind::Allocation:
        return {TransferEndpointKind::Storage, graph, value.storage.value_or(value.id)};
    }
    return {TransferEndpointKind::Storage, graph, value.id};
}

bool addDependency(ResolvedHazardPlan &plan, const ResolvedDependencyEdge &edge) {
    if (edge.predecessor == edge.successor)
        return true;
    auto &predecessors = plan.predecessors[edge.successor];
    if (std::find(predecessors.begin(), predecessors.end(), edge.predecessor.node) == predecessors.end())
        predecessors.push_back(edge.predecessor.node);
    const auto duplicate = std::find_if(plan.edges.begin(), plan.edges.end(), [&](const ResolvedDependencyEdge &other) {
        return other.predecessor == edge.predecessor && other.successor == edge.successor &&
               other.hazard == edge.hazard && other.storage == edge.storage;
    });
    if (duplicate == plan.edges.end())
        plan.edges.push_back(edge);
    return true;
}

bool reaches(const ResolvedHazardPlan &plan, const NodeKey &from, const NodeKey &to, std::set<NodeKey> &visited) {
    if (from == to)
        return true;
    if (!visited.insert(to).second)
        return false;
    const auto predecessors = plan.predecessors.find(to);
    if (predecessors == plan.predecessors.end())
        return false;
    for (uint32_t predecessor : predecessors->second) {
        const NodeKey candidate{to.graph, predecessor};
        if (candidate == from || reaches(plan, from, candidate, visited))
            return true;
    }
    return false;
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

bool ResolvedExecutionPlan::requiresDevice(GraphDirection graph, uint32_t value) const {
    const auto found = residency.find({graph, value});
    return found != residency.end() && found->second == ResidencyRequirement::Device;
}

const std::vector<uint32_t> &ResolvedExecutionPlan::predecessors(GraphDirection graph, uint32_t nodeId) const {
    static const std::vector<uint32_t> empty;
    const auto found = hazards.predecessors.find({graph, nodeId});
    return found == hazards.predecessors.end() ? empty : found->second;
}

bool buildResolvedExecutionPolicies(ResolvedExecutionPlan &plan, Diagnostic &diagnostic) {
    diagnostic = {};
    if (!plan.resolvedProgram)
        return reject(diagnostic, "", "execution plan has no resolved Program");
    const Program &program = plan.resolvedProgram->program;

    plan.aliasDomains.clear();
    for (const Storage &storage : program.storages)
        plan.aliasDomains.emplace(storage.id, AliasDomainPlan{storage.id});
    for (const Value &value : program.values)
        if (value.storage) {
            const auto domain = plan.aliasDomains.find(*value.storage);
            if (domain == plan.aliasDomains.end())
                return reject(diagnostic, "/values/" + std::to_string(value.id),
                              "Value references an unknown Storage alias domain");
            domain->second.values.push_back(value.id);
        }

    plan.residency.clear();
    plan.transfers = {};
    plan.hazards = {};
    plan.graphicsScopes.clear();
    uint32_t transferOrder = 0;
    for (size_t graphIndex = 0; graphIndex < program.graphs.size(); ++graphIndex) {
        const Graph &graph = program.graphs[graphIndex];
        const std::optional<GraphDirection> direction = graphDirection(graph.direction);
        if (!direction)
            return reject(diagnostic, "/graphs/" + std::to_string(graphIndex), "unsupported graph direction");

        std::map<uint32_t, ResidencyRequirement> currentResidency;
        for (const Value &value : program.values)
            currentResidency[value.id] = ResidencyRequirement::Host;

        for (const Node &canonicalNode : graph.nodes) {
            const NodeKey key{*direction, canonicalNode.id};
            const auto resolvedNode = plan.nodes.find(key);
            if (resolvedNode == plan.nodes.end())
                return reject(diagnostic,
                              "/graphs/" + std::to_string(graphIndex) + "/nodes/" + std::to_string(canonicalNode.id),
                              "canonical Node has no resolved plan");
            for (const NodeEndpointProjection &projection : resolvedNode->second.projections) {
                if (projection.value >= program.values.size())
                    return reject(diagnostic, "/graphs/" + std::to_string(graphIndex),
                                  "Node projection references an unknown Value");
                const ResidencyRequirement required = bindingResidency(resolvedNode->second, projection.target);
                auto mark = [&](uint32_t value) {
                    const GraphValueKey valueKey{*direction, value};
                    auto existing = plan.residency.find(valueKey);
                    if (existing == plan.residency.end() || required == ResidencyRequirement::Device)
                        plan.residency[valueKey] = required;
                };
                mark(projection.value);
                const Value &value = program.values[projection.value];
                if (value.storage) {
                    AliasDomainPlan &domain = plan.aliasDomains.at(*value.storage);
                    auto domainResidency = domain.residency.find(*direction);
                    if (domainResidency == domain.residency.end() || required == ResidencyRequirement::Device)
                        domain.residency[*direction] = required;
                    if (required == ResidencyRequirement::Device)
                        for (uint32_t alias : domain.values)
                            plan.residency[{*direction, alias}] = ResidencyRequirement::Device;
                }
            }
        }

        for (const Node &canonicalNode : graph.nodes) {
            const NodeKey key{*direction, canonicalNode.id};
            const ResolvedNodePlan &resolvedNode = plan.nodes.at(key);
            for (const NodeEndpointProjection &projection : resolvedNode.projections) {
                const ResidencyRequirement required = bindingResidency(resolvedNode, projection.target);
                if (projection.target.projection.direction == ValueBindingDirection::Result) {
                    currentResidency[projection.value] = required;
                    continue;
                }
                const ResidencyRequirement current = currentResidency[projection.value];
                if (current != required) {
                    const TransferKind kind =
                        current == ResidencyRequirement::Host ? TransferKind::HostUpload : TransferKind::Readback;
                    plan.transfers.edges.push_back({kind,
                                                    valueProducer(program.values[projection.value], *direction),
                                                    {TransferEndpointKind::Node, *direction, canonicalNode.id},
                                                    projection.value,
                                                    program.values[projection.value].storage,
                                                    transferOrder++});
                    currentResidency[projection.value] = required;
                }
            }
            if (canonicalNode.name == "vernon.builtin.copy" && !canonicalNode.operands.empty() &&
                !canonicalNode.results.empty()) {
                const uint32_t source = canonicalNode.operands.front();
                const uint32_t result = canonicalNode.results.front();
                if (source < program.values.size() && result < program.values.size() &&
                    plan.requiresDevice(*direction, source) && plan.requiresDevice(*direction, result))
                    plan.transfers.edges.push_back({TransferKind::DeviceCopy,
                                                    valueProducer(program.values[source], *direction),
                                                    {TransferEndpointKind::Node, *direction, canonicalNode.id},
                                                    result,
                                                    program.values[result].storage,
                                                    transferOrder++});
            }
        }

        if (graphIndex >= plan.resolvedProgram->graphs.size())
            return reject(diagnostic, "/graphs/" + std::to_string(graphIndex),
                          "graph has no canonical dependency plan");
        const ResolvedGraph &canonicalDependencies = plan.resolvedProgram->graphs[graphIndex];
        for (const Node &node : graph.nodes) {
            const NodeKey successor{*direction, node.id};
            if (node.id >= canonicalDependencies.predecessors.size())
                return reject(diagnostic, "/graphs/" + std::to_string(graphIndex),
                              "Node has no canonical predecessor record");
            for (uint32_t predecessor : canonicalDependencies.predecessors[node.id])
                addDependency(plan.hazards, {{*direction, predecessor},
                                             successor,
                                             HazardKind::Canonical,
                                             BarrierRequirement::Execution,
                                             std::nullopt});
        }

        struct AliasState {
            std::optional<NodeKey> writer;
            std::vector<NodeKey> readers;
        };
        std::map<uint32_t, AliasState> aliases;
        for (const Node &node : graph.nodes) {
            const NodeKey current{*direction, node.id};
            for (const ResourceAccess &access : node.accesses) {
                AliasState &state = aliases[access.storage];
                const bool reads = access.kind == AccessKind::Read ||
                                   (access.kind == AccessKind::Write && access.access == "read_write") ||
                                   (access.kind == AccessKind::Attachment && access.access == "read_write");
                const bool writes = access.kind != AccessKind::Read;
                const BarrierRequirement barrier = access.kind == AccessKind::Attachment
                                                       ? BarrierRequirement::AttachmentTransition
                                                       : BarrierRequirement::Memory;
                if (reads && state.writer)
                    addDependency(plan.hazards,
                                  {*state.writer, current, HazardKind::ReadAfterWrite, barrier, access.storage});
                if (writes) {
                    if (state.writer)
                        addDependency(plan.hazards,
                                      {*state.writer, current, HazardKind::WriteAfterWrite, barrier, access.storage});
                    for (const NodeKey &reader : state.readers)
                        addDependency(plan.hazards,
                                      {reader, current, HazardKind::WriteAfterRead, barrier, access.storage});
                    state.readers.clear();
                    state.writer = current;
                } else if (std::find(state.readers.begin(), state.readers.end(), current) == state.readers.end()) {
                    state.readers.push_back(current);
                }
            }
        }

        for (const Node &node : graph.nodes) {
            if (executionKind(node) != ExecutionKind::Graphics)
                continue;
            const NodeKey key{*direction, node.id};
            const ResolvedGraphicsControls *controls =
                std::get_if<ResolvedGraphicsControls>(&plan.nodes.at(key).controls);
            if (!controls)
                return reject(diagnostic, "/graphs/" + std::to_string(graphIndex),
                              "graphics Node has no resolved graphics controls");
            ResolvedGraphicsScopePlan scope;
            scope.node = key;
            const auto append = [&](const ResolvedGraphicsAttachment &attachment) {
                const ResourceAccess &access = node.accesses[attachment.access];
                const bool initializes = access.before < program.values.size() &&
                                         program.values[access.before].origin.kind == OriginKind::Allocation &&
                                         access.storage < program.storages.size() &&
                                         program.storages[access.storage].initialValue == access.before;
                const AttachmentTransition transition = initializes ? AttachmentTransition::Initialize
                                                        : access.access == "read_write"
                                                            ? AttachmentTransition::ReadWrite
                                                            : AttachmentTransition::Preserve;
                scope.attachments.push_back({attachment.storage, attachment.location, attachment.aspects, transition});
            };
            for (const ResolvedGraphicsAttachment &attachment : controls->colorAttachments)
                append(attachment);
            if (controls->depthStencilAttachment)
                append(*controls->depthStencilAttachment);
            plan.graphicsScopes.push_back(std::move(scope));
        }
    }

    plan.autodiff = {};
    for (const TapePlan &tape : program.abi.tapePlans)
        plan.autodiff.tapes.push_back(
            {tape.value, tape.forwardProducer, tape.backwardConsumer, tape.requiredCarriers, tape.optionalCarriers});
    plan.autodiff.residualValues = residualCaptures(program);
    if (const Graph *replay = findGraph(program, "replay"))
        for (const Node &node : replay->nodes)
            plan.autodiff.replayNodes.push_back({GraphDirection::Replay, node.id});
    const bool compiledReplay =
        !plan.autodiff.replayNodes.empty() ||
        std::any_of(plan.autodiff.tapes.begin(), plan.autodiff.tapes.end(), [](const ResolvedTapeRequirement &tape) {
            return std::find(tape.requiredCarriers.begin(), tape.requiredCarriers.end(),
                             vernon::program_plan::TapeCarrier::ReplaySegment) != tape.requiredCarriers.end() ||
                   std::find(tape.optionalCarriers.begin(), tape.optionalCarriers.end(),
                             vernon::program_plan::TapeCarrier::ReplaySegment) != tape.optionalCarriers.end();
        });
    plan.autodiff.checkpointPolicy = compiledReplay ? CheckpointPolicy::Rematerialize : CheckpointPolicy::Retain;

    plan.publications = {};
    std::set<std::pair<ProgramOwnerKind, uint32_t>> publicationOwners;
    for (const BoundarySlot &slot : program.abi.boundarySlots) {
        if (slot.publication == BoundaryPublication::None)
            continue;
        const auto owner = std::make_pair(slot.aliasOwner.kind, slot.aliasOwner.id);
        if (!publicationOwners.insert(owner).second)
            return reject(diagnostic, "/abi/boundary_slots/" + std::to_string(slot.id),
                          "publication owner has more than one transaction");
        const PublicationCommitMode mode = slot.publication == BoundaryPublication::CommitAfterSuccess
                                               ? PublicationCommitMode::CommitAfterSuccess
                                               : PublicationCommitMode::InPlace;
        plan.publications.transactions.push_back({slot.id, slot.value, slot.aliasOwner, mode});
    }
    return true;
}

bool validateResolvedExecutionPlan(const ResolvedExecutionPlan &plan, Diagnostic &diagnostic) {
    diagnostic = {};
    if (!plan.resolvedProgram)
        return reject(diagnostic, "", "execution plan has no Program");
    const Program &program = plan.resolvedProgram->program;
    ResolvedExecutionPlan expected = plan;
    Diagnostic expectedDiagnostic;
    if (!buildResolvedExecutionPolicies(expected, expectedDiagnostic))
        return reject(diagnostic, expectedDiagnostic.path, expectedDiagnostic.message);
    for (const ResolvedDependencyEdge &required : expected.hazards.edges) {
        const auto found = std::find_if(
            plan.hazards.edges.begin(), plan.hazards.edges.end(), [&](const ResolvedDependencyEdge &candidate) {
                return candidate.predecessor == required.predecessor && candidate.successor == required.successor &&
                       candidate.hazard == required.hazard && candidate.barrier == required.barrier &&
                       candidate.storage == required.storage;
            });
        if (found == plan.hazards.edges.end())
            return reject(diagnostic, "/hazards", "conflicting alias accesses have no dependency or barrier");
    }
    if (plan.residency != expected.residency || plan.aliasDomains.size() != expected.aliasDomains.size())
        return reject(diagnostic, "/residency", "logical residency disagrees with resolved Node projections");
    for (const ResolvedTransferEdge &required : expected.transfers.edges) {
        const auto found = std::find_if(
            plan.transfers.edges.begin(), plan.transfers.edges.end(), [&](const ResolvedTransferEdge &candidate) {
                return candidate.kind == required.kind && candidate.producer.kind == required.producer.kind &&
                       candidate.producer.graph == required.producer.graph &&
                       candidate.producer.id == required.producer.id &&
                       candidate.consumer.kind == required.consumer.kind &&
                       candidate.consumer.graph == required.consumer.graph &&
                       candidate.consumer.id == required.consumer.id && candidate.value == required.value &&
                       candidate.storage == required.storage;
            });
        if (found == plan.transfers.edges.end())
            return reject(diagnostic, "/transfers", "physical use has no required logical transfer");
    }
    for (const auto &[key, node] : plan.nodes) {
        if (!node.stage)
            return reject(diagnostic, "/graphs", "resolved Node has no Stage");
        if (node.key != key || node.projections.size() != node.stage->bindingProjection.parameters.size())
            return reject(diagnostic, "/graphs", "resolved Node projection does not match its Stage endpoint ABI");
        for (const NodeEndpointProjection &projection : node.projections) {
            if (projection.value >= program.values.size())
                return reject(diagnostic, "/graphs", "resolved Node projection references a missing Value");
            if (projection.logicalLeaf &&
                (!program.values[projection.value].layout ||
                 *projection.logicalLeaf >= program.values[projection.value].layout->leaves.size()))
                return reject(diagnostic, "/graphs", "resolved Node projection references a missing logical leaf");
            if (projection.target.endpoint.interfaceKind.empty())
                return reject(diagnostic, "/graphs", "resolved Node projection references a missing endpoint");
            const Value &value = program.values[projection.value];
            if (value.origin.kind == OriginKind::NodeResult) {
                const std::optional<GraphDirection> producerGraph = graphDirection(value.origin.graph);
                if (!producerGraph)
                    return reject(diagnostic, "/graphs", "physical read has an invalid producer graph");
                if (*producerGraph != key.graph) {
                    const auto consumerGraph =
                        std::find_if(program.graphs.begin(), program.graphs.end(), [&](const Graph &graph) {
                            return graphDirection(graph.direction) == std::optional<GraphDirection>(key.graph);
                        });
                    const bool captured = consumerGraph != program.graphs.end() &&
                                          std::find(consumerGraph->captures.begin(), consumerGraph->captures.end(),
                                                    projection.value) != consumerGraph->captures.end();
                    if (!captured)
                        return reject(diagnostic, "/graphs", "physical read has no cross-graph capture or transfer");
                    continue;
                }
                std::set<NodeKey> visited;
                if (!reaches(plan.hazards, {*producerGraph, value.origin.node}, key, visited))
                    return reject(diagnostic, "/graphs", "physical read has no dominating producer or transfer");
            }
        }
    }
    for (size_t index = 0; index < plan.transfers.edges.size(); ++index) {
        const ResolvedTransferEdge &edge = plan.transfers.edges[index];
        if (edge.order != index || edge.value >= program.values.size() ||
            edge.consumer.kind != TransferEndpointKind::Node || !plan.node(edge.consumer.graph, edge.consumer.id))
            return reject(diagnostic, "/transfers/" + std::to_string(index),
                          "transfer has an invalid endpoint or ordering");
    }
    for (const ResolvedPublicationTransaction &publication : plan.publications.transactions) {
        if (publication.slot >= program.abi.boundarySlots.size() || publication.value >= program.values.size())
            return reject(diagnostic, "/publications", "publication has no unique owner or staging policy");
        const BoundarySlot &slot = program.abi.boundarySlots[publication.slot];
        if (slot.aliasOwner.kind != publication.stagingOwner.kind || slot.aliasOwner.id != publication.stagingOwner.id)
            return reject(diagnostic, "/publications", "publication staging owner disagrees with Program ABI");
        const PublicationTarget *target = findPublicationTarget(program.abi, publication.slot);
        if (publication.mode == PublicationCommitMode::CommitAfterSuccess) {
            if (!target || target->value != publication.value ||
                target->aliasOwner.kind != publication.stagingOwner.kind ||
                target->aliasOwner.id != publication.stagingOwner.id)
                return reject(diagnostic, "/publications",
                              "commit-after-success publication has no unique staging target");
        } else if (target) {
            return reject(diagnostic, "/publications", "in-place publication cannot own staged commit state");
        }
    }
    return true;
}

} // namespace vernon::runtime::program
