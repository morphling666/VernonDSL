#include "program_graph_linker.h"
#include "content_hash.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vernon::runtime {
namespace {

class DisjointSet {
public:
    explicit DisjointSet(size_t size) : parent_(size) {
        for (size_t index = 0; index < size; ++index)
            parent_[index] = index;
    }

    size_t root(size_t value) {
        if (parent_[value] != value)
            parent_[value] = root(parent_[value]);
        return parent_[value];
    }

    void join(size_t left, size_t right) {
        left = root(left);
        right = root(right);
        if (left != right)
            parent_[right] = left;
    }

private:
    std::vector<size_t> parent_;
};

struct NodeOffsets {
    size_t value{};
    size_t storage{};
    uint32_t parameter{};
    std::map<std::string, uint32_t> graphNodes;
    std::map<std::string, uint32_t> userSlots;
    std::map<std::string, uint32_t> controlSlots;
};

bool reject(program::Diagnostic &diagnostic, std::string path, std::string message) {
    diagnostic = {"PROGRAM_GRAPH_LINK", "link", std::move(path), std::move(message)};
    return false;
}

bool sameControl(const program::ControlComponent &left, const program::ControlComponent &right) {
    return left.kind == right.kind && left.value == right.value && left.reference == right.reference &&
           left.hasAxis == right.hasAxis && left.axis == right.axis;
}

bool sameControls(const std::vector<program::ControlComponent> &left,
                  const std::vector<program::ControlComponent> &right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](const auto &a, const auto &b) { return sameControl(a, b); });
}

const program::BoundarySlot *boundary(const ProgramGraphNodeSource &node, uint32_t slot) {
    if (!node.deployment)
        return nullptr;
    const auto &slots = node.deployment->program.abi.boundarySlots;
    return slot < slots.size() && slots[slot].id == slot ? &slots[slot] : nullptr;
}

bool sameBuffer(const program::BufferDescriptor &left, const program::BufferDescriptor &right) {
    return left.byteLength == right.byteLength && left.alignment == right.alignment && left.memory == right.memory &&
           left.usage == right.usage && sameControls(left.byteLengthExtents, right.byteLengthExtents);
}

bool sameImage(const program::ImageDescriptor &left, const program::ImageDescriptor &right) {
    return left.dimension == right.dimension && left.extent == right.extent && left.format == right.format &&
           left.sampleCount == right.sampleCount && left.mipLevels == right.mipLevels &&
           left.arrayLayers == right.arrayLayers && left.aspects == right.aspects && left.usage == right.usage &&
           sameControls(left.extentControls, right.extentControls);
}

bool sameStorage(const program::BoundarySlot &left, const program::BoundarySlot &right) {
    if (left.storage.has_value() != right.storage.has_value())
        return false;
    if (!left.storage)
        return true;
    if (left.storage->descriptorKind != right.storage->descriptorKind)
        return false;
    switch (left.storage->descriptorKind) {
    case program::StorageDescriptorKind::Buffer:
        return sameBuffer(left.storage->buffer, right.storage->buffer);
    case program::StorageDescriptorKind::Image:
        return sameImage(left.storage->image, right.storage->image);
    case program::StorageDescriptorKind::Opaque:
        return left.storage->opaqueContract == right.storage->opaqueContract &&
               left.storage->opaqueContractHash == right.storage->opaqueContractHash &&
               left.storage->opaqueUsage == right.storage->opaqueUsage;
    }
    return false;
}

bool sameLayout(const program::BoundarySlot &left, const program::BoundarySlot &right) {
    return left.layout.has_value() == right.layout.has_value() &&
           (!left.layout ||
            (left.layout->layoutHash == right.layout->layoutHash && left.layout->byteSize == right.layout->byteSize &&
             left.layout->alignment == right.layout->alignment));
}

bool compatible(const program::BoundarySlot &source, const program::BoundarySlot &destination) {
    return source.direction == program::BoundaryDirection::Output &&
           destination.direction == program::BoundaryDirection::Input && source.category == destination.category &&
           source.logicalType == destination.logicalType && source.outerShape == destination.outerShape &&
           source.aliasOwner.kind == destination.aliasOwner.kind && sameLayout(source, destination) &&
           sameStorage(source, destination);
}

bool storageConnectionCompatible(const program::BoundarySlot &left, const program::BoundarySlot &right) {
    return left.direction == program::BoundaryDirection::Output &&
           right.direction == program::BoundaryDirection::Output && left.category == right.category &&
           left.access == right.access && left.logicalType == right.logicalType &&
           left.outerShape == right.outerShape && left.aliasOwner.kind == program::ProgramOwnerKind::Storage &&
           right.aliasOwner.kind == program::ProgramOwnerKind::Storage && left.publication == right.publication &&
           sameLayout(left, right) && sameStorage(left, right);
}

template <typename T, typename Key> void appendUnique(std::vector<T> &values, T value, Key key) {
    const auto identity = key(value);
    if (std::none_of(values.begin(), values.end(), [&](const T &existing) { return key(existing) == identity; }))
        values.push_back(std::move(value));
}

template <typename T> void canonicalizeIds(std::vector<T> &ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

} // namespace

bool sameProgramGraphBoundaryContract(const program::BoundarySlot &left, const program::BoundarySlot &right) {
    return left.id == right.id && left.path == right.path && left.value == right.value && left.role == right.role &&
           left.direction == right.direction && left.category == right.category && left.access == right.access &&
           left.logicalType == right.logicalType && left.outerShape == right.outerShape &&
           left.aliasOwner.kind == right.aliasOwner.kind && left.aliasOwner.id == right.aliasOwner.id &&
           left.publication == right.publication && sameLayout(left, right) && sameStorage(left, right) &&
           (!left.storage || left.storage->id == right.storage->id);
}

bool linkProgramGraph(const std::vector<ProgramGraphNodeSource> &nodes,
                      const std::vector<ProgramGraphConnection> &connections,
                      const std::vector<ProgramGraphBoundaryKey> &retainedBoundaries,
                      const std::vector<ProgramGraphExport> &exports, LinkedProgramDeployment &linked,
                      program::Diagnostic &diagnostic) {
    diagnostic = {};
    linked = {};
    if (nodes.empty())
        return reject(diagnostic, "/nodes", "ProgramGraph requires at least one Program node");

    std::map<uint32_t, size_t> nodeIndices;
    std::vector<NodeOffsets> offsets(nodes.size());
    size_t valueCount = 0;
    size_t storageCount = 0;
    uint32_t parameterCount = 0;
    std::map<std::string, uint32_t> graphNodeCounts;
    std::map<std::string, uint32_t> graphUserSlotCounts;
    std::map<std::string, uint32_t> graphControlSlotCounts;
    const std::string target = nodes.front().deployment ? nodes.front().deployment->artifactSystem.target : "";
    for (size_t index = 0; index < nodes.size(); ++index) {
        const ProgramGraphNodeSource &node = nodes[index];
        if (!node.deployment || node.id == UINT32_MAX || node.bundleId.empty() || node.contentHash.empty() ||
            !nodeIndices.emplace(node.id, index).second)
            return reject(diagnostic, "/nodes/" + std::to_string(index), "ProgramGraph node is invalid or duplicated");
        if (node.deployment->artifactSystem.target != target)
            return reject(diagnostic, "/nodes/" + std::to_string(node.id),
                          "ProgramGraph nodes target different backends");
        offsets[index].value = valueCount;
        offsets[index].storage = storageCount;
        offsets[index].parameter = parameterCount;
        valueCount += node.deployment->program.values.size();
        storageCount += node.deployment->program.storages.size();
        parameterCount += static_cast<uint32_t>(node.deployment->program.parameters.size());
        for (const program::Graph &graph : node.deployment->program.graphs) {
            offsets[index].graphNodes[graph.direction] = graphNodeCounts[graph.direction];
            graphNodeCounts[graph.direction] += static_cast<uint32_t>(graph.nodes.size());
            offsets[index].userSlots[graph.direction] = graphUserSlotCounts[graph.direction];
            offsets[index].controlSlots[graph.direction] = graphControlSlotCounts[graph.direction];
            uint32_t userSlots = 0;
            uint32_t controlSlots = 0;
            for (const program::GraphInput &input : graph.inputs) {
                if (input.kind == program::GraphInputKind::UserInput)
                    userSlots = std::max(userSlots, input.slot + 1);
                else if (input.kind == program::GraphInputKind::InvocationControl ||
                         input.kind == program::GraphInputKind::Control)
                    controlSlots = std::max(controlSlots, input.slot + 1);
            }
            for (const program::Node &graphNode : graph.nodes)
                if (const auto *graphics = std::get_if<program::GraphicsOperation>(&graphNode.operation)) {
                    controlSlots = std::max({controlSlots, graphics->renderPassControl + 1,
                                             graphics->drawCommandControl + 1, graphics->dynamicStateControl + 1});
                }
            graphUserSlotCounts[graph.direction] += userSlots;
            graphControlSlotCounts[graph.direction] += controlSlots;
        }
    }

    DisjointSet valueSets(valueCount);
    DisjointSet storageSets(storageCount);
    std::map<ProgramGraphBoundaryKey, ProgramGraphBoundaryKey> destinations;
    std::map<std::pair<uint32_t, uint32_t>, ProgramGraphBoundaryKey> storagePredecessors;
    std::map<std::pair<uint32_t, uint32_t>, ProgramGraphBoundaryKey> storageSuccessors;
    std::map<size_t, ProgramGraphBoundaryKey> preferredValues;
    std::map<size_t, ProgramGraphBoundaryKey> preferredStorages;
    const auto locate =
        [&](ProgramGraphBoundaryKey key) -> std::pair<const ProgramGraphNodeSource *, const program::BoundarySlot *> {
        const auto found = nodeIndices.find(key.node);
        if (found == nodeIndices.end())
            return {};
        const ProgramGraphNodeSource &node = nodes[found->second];
        return {&node, boundary(node, key.slot)};
    };
    for (size_t index = 0; index < connections.size(); ++index) {
        const ProgramGraphConnection &connection = connections[index];
        const auto [sourceNode, source] = locate(connection.source);
        const auto [destinationNode, destination] = locate(connection.destination);
        if (!sourceNode || !source || !destinationNode || !destination)
            return reject(diagnostic, "/connections/" + std::to_string(index),
                          "connection references an unknown boundary");
        if (connection.storageConnection ? !storageConnectionCompatible(*source, *destination)
                                         : !compatible(*source, *destination))
            return reject(diagnostic, "/connections/" + std::to_string(index),
                          "connected Program boundaries have incompatible contracts");
        if (connection.storageConnection) {
            if (nodeIndices.at(connection.source.node) >= nodeIndices.at(connection.destination.node))
                return reject(diagnostic, "/connections/" + std::to_string(index),
                              "Program Storage connections must follow node order");
            const program::Storage &leftStorage = sourceNode->deployment->program.storages[source->aliasOwner.id];
            const program::Storage &rightStorage =
                destinationNode->deployment->program.storages[destination->aliasOwner.id];
            if (leftStorage.ownership != rightStorage.ownership || leftStorage.lifetime != rightStorage.lifetime ||
                leftStorage.mutability != rightStorage.mutability)
                return reject(diagnostic, "/connections/" + std::to_string(index),
                              "connected Program Storages have incompatible ownership");
            if (!storageSuccessors
                     .emplace(std::pair{connection.source.node, source->aliasOwner.id}, connection.destination)
                     .second)
                return reject(diagnostic, "/connections/" + std::to_string(index),
                              "a Program Storage version may have only one successor");
            if (!storagePredecessors
                     .emplace(std::pair{connection.destination.node, destination->aliasOwner.id}, connection.source)
                     .second)
                return reject(diagnostic, "/connections/" + std::to_string(index),
                              "a Program Storage version may have only one predecessor");
        }
        if (!connection.storageConnection && !destinations.emplace(connection.destination, connection.source).second)
            return reject(diagnostic, "/connections/" + std::to_string(index),
                          "a Program input boundary may have only one source");
        const size_t sourceIndex = nodeIndices.at(connection.source.node);
        const size_t destinationIndex = nodeIndices.at(connection.destination.node);
        const size_t sourceValue = offsets[sourceIndex].value + source->value;
        const size_t destinationValue = offsets[destinationIndex].value + destination->value;
        if (!connection.storageConnection) {
            valueSets.join(sourceValue, destinationValue);
        }
        if (source->aliasOwner.kind == program::ProgramOwnerKind::Storage) {
            const size_t sourceStorage = offsets[sourceIndex].storage + source->aliasOwner.id;
            const size_t destinationStorage = offsets[destinationIndex].storage + destination->aliasOwner.id;
            storageSets.join(sourceStorage, destinationStorage);
        }
    }
    for (const ProgramGraphConnection &connection : connections) {
        const size_t sourceIndex = nodeIndices.at(connection.source.node);
        const program::BoundarySlot &source = *boundary(nodes[sourceIndex], connection.source.slot);
        if (!connection.storageConnection) {
            const size_t root = valueSets.root(offsets[sourceIndex].value + source.value);
            const auto found = preferredValues.find(root);
            if (found == preferredValues.end() || connection.source < found->second)
                preferredValues[root] = connection.source;
        }
        if (source.aliasOwner.kind == program::ProgramOwnerKind::Storage) {
            const size_t root = storageSets.root(offsets[sourceIndex].storage + source.aliasOwner.id);
            const auto found = preferredStorages.find(root);
            if (found == preferredStorages.end() || connection.source < found->second)
                preferredStorages[root] = connection.source;
        }
    }

    std::set<ProgramGraphBoundaryKey> retained;
    for (size_t index = 0; index < retainedBoundaries.size(); ++index) {
        const auto [node, slot] = locate(retainedBoundaries[index]);
        const bool intermediateStorage =
            std::any_of(connections.begin(), connections.end(), [&](const ProgramGraphConnection &connection) {
                return connection.storageConnection && connection.source == retainedBoundaries[index];
            });
        if (!node || !slot || slot->direction != program::BoundaryDirection::Output ||
            slot->aliasOwner.kind != program::ProgramOwnerKind::Storage || intermediateStorage ||
            !retained.emplace(retainedBoundaries[index]).second)
            return reject(diagnostic, "/retained_boundaries/" + std::to_string(index),
                          "retained ProgramGraph boundary is invalid or duplicated");
    }

    std::map<ProgramGraphBoundaryKey, std::string> exportPaths;
    std::set<std::string> usedExportPaths;
    for (size_t index = 0; index < exports.size(); ++index) {
        const auto [node, slot] = locate(exports[index].boundary);
        const bool connectedDestination =
            std::any_of(connections.begin(), connections.end(), [&](const ProgramGraphConnection &connection) {
                return !connection.storageConnection && connection.destination == exports[index].boundary;
            });
        const bool intermediateStorage =
            std::any_of(connections.begin(), connections.end(), [&](const ProgramGraphConnection &connection) {
                return connection.storageConnection && connection.source == exports[index].boundary;
            });
        if (!node || !slot || exports[index].path.empty() ||
            !exportPaths.emplace(exports[index].boundary, exports[index].path).second ||
            !usedExportPaths.emplace(exports[index].path).second || connectedDestination || intermediateStorage)
            return reject(diagnostic, "/exports/" + std::to_string(index),
                          "ProgramGraph export is invalid or duplicated");
    }

    std::map<size_t, uint32_t> valueIds;
    std::map<size_t, uint32_t> storageIds;
    for (size_t index = 0; index < valueCount; ++index)
        if (!valueIds.count(valueSets.root(index)))
            valueIds.emplace(valueSets.root(index), static_cast<uint32_t>(valueIds.size()));
    for (size_t index = 0; index < storageCount; ++index)
        if (!storageIds.count(storageSets.root(index)))
            storageIds.emplace(storageSets.root(index), static_cast<uint32_t>(storageIds.size()));

    const auto globalValue = [&](size_t nodeIndex, uint32_t value) {
        return valueIds.at(valueSets.root(offsets[nodeIndex].value + value));
    };
    const auto globalStorage = [&](size_t nodeIndex, uint32_t storage) {
        return storageIds.at(storageSets.root(offsets[nodeIndex].storage + storage));
    };
    const auto globalNode = [&](size_t nodeIndex, std::string_view graph, uint32_t node) {
        return offsets[nodeIndex].graphNodes.at(std::string(graph)) + node;
    };
    for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        LinkedProgramDeployment::NodeMapping mapping;
        mapping.values.reserve(nodes[nodeIndex].deployment->program.values.size());
        for (const program::Value &value : nodes[nodeIndex].deployment->program.values)
            mapping.values.push_back(globalValue(nodeIndex, value.id));
        mapping.storages.reserve(nodes[nodeIndex].deployment->program.storages.size());
        for (const program::Storage &storage : nodes[nodeIndex].deployment->program.storages)
            mapping.storages.push_back(globalStorage(nodeIndex, storage.id));
        linked.nodeMappings.emplace(nodes[nodeIndex].id, std::move(mapping));
    }
    for (const ProgramGraphConnection &connection : connections) {
        if (!connection.storageConnection)
            continue;
        const size_t sourceIndex = nodeIndices.at(connection.source.node);
        const size_t destinationIndex = nodeIndices.at(connection.destination.node);
        const program::BoundarySlot &source = *boundary(nodes[sourceIndex], connection.source.slot);
        const program::BoundarySlot &destination = *boundary(nodes[destinationIndex], connection.destination.slot);
        const uint32_t destinationInitial =
            nodes[destinationIndex].deployment->program.storages[destination.aliasOwner.id].initialValue;
        linked.nodeMappings.at(connection.destination.node).values[destinationInitial] =
            globalValue(sourceIndex, source.value);
    }

    program::Program &program = linked.deployment.program;
    program::ArtifactSystem &artifacts = linked.deployment.artifactSystem;
    artifacts.target = target;

    std::vector<std::pair<size_t, uint32_t>> storageRepresentatives(storageIds.size(), {SIZE_MAX, 0});
    std::vector<std::pair<size_t, uint32_t>> valueRepresentatives(valueIds.size(), {SIZE_MAX, 0});
    for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        const program::Program &child = nodes[nodeIndex].deployment->program;
        for (const program::Storage &storage : child.storages) {
            const uint32_t id = globalStorage(nodeIndex, storage.id);
            if (storageRepresentatives[id].first == SIZE_MAX)
                storageRepresentatives[id] = {nodeIndex, storage.id};
        }
        for (const program::Value &value : child.values) {
            const uint32_t id = globalValue(nodeIndex, value.id);
            if (valueRepresentatives[id].first == SIZE_MAX)
                valueRepresentatives[id] = {nodeIndex, value.id};
        }
    }
    for (const auto &[root, key] : preferredStorages) {
        const size_t nodeIndex = nodeIndices.at(key.node);
        const program::BoundarySlot &slot = *boundary(nodes[nodeIndex], key.slot);
        storageRepresentatives[storageIds.at(storageSets.root(root))] = {nodeIndex, slot.aliasOwner.id};
    }
    for (const auto &[root, key] : preferredValues) {
        const size_t nodeIndex = nodeIndices.at(key.node);
        const program::BoundarySlot &slot = *boundary(nodes[nodeIndex], key.slot);
        valueRepresentatives[valueIds.at(valueSets.root(root))] = {nodeIndex, slot.value};
    }

    program.storages.resize(storageRepresentatives.size());
    for (uint32_t id = 0; id < storageRepresentatives.size(); ++id) {
        const auto [nodeIndex, localId] = storageRepresentatives[id];
        program::Storage storage = nodes[nodeIndex].deployment->program.storages[localId];
        storage.id = id;
        storage.name = "node/" + std::to_string(nodes[nodeIndex].id) + "/" + storage.name;
        storage.initialValue = globalValue(nodeIndex, storage.initialValue);
        program.storages[id] = std::move(storage);
    }

    program.values.resize(valueRepresentatives.size());
    for (uint32_t id = 0; id < valueRepresentatives.size(); ++id) {
        const size_t nodeIndex = valueRepresentatives[id].first;
        const uint32_t localId = valueRepresentatives[id].second;
        program::Value value = nodes[nodeIndex].deployment->program.values[localId];
        value.id = id;
        value.name = "node/" + std::to_string(nodes[nodeIndex].id) + "/" + value.name;
        if (value.storage)
            value.storage = globalStorage(nodeIndex, *value.storage);
        if (value.origin.kind == program::OriginKind::Parameter)
            value.origin.parameter += offsets[nodeIndex].parameter;
        else if (value.origin.kind == program::OriginKind::NodeResult)
            value.origin.node = globalNode(nodeIndex, value.origin.graph, value.origin.node);
        else if (value.origin.kind == program::OriginKind::Argument) {
            const program::Graph *originGraph =
                program::findGraph(nodes[nodeIndex].deployment->program, value.origin.graph);
            if (originGraph) {
                const auto input =
                    std::find_if(originGraph->inputs.begin(), originGraph->inputs.end(),
                                 [&](const program::GraphInput &candidate) { return candidate.value == localId; });
                if (input != originGraph->inputs.end()) {
                    if (input->kind == program::GraphInputKind::UserInput)
                        value.origin.slot += offsets[nodeIndex].userSlots.at(value.origin.graph);
                    else if (input->kind == program::GraphInputKind::InvocationControl ||
                             input->kind == program::GraphInputKind::Control)
                        value.origin.slot += offsets[nodeIndex].controlSlots.at(value.origin.graph);
                }
            }
        }
        program.values[id] = std::move(value);
    }

    for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        const ProgramGraphNodeSource &source = nodes[nodeIndex];
        const program::Program &child = source.deployment->program;
        const std::string prefix = "node/" + std::to_string(source.id) + "/";
        const program::Graph *childForward = program::findGraph(child, "forward");
        if (!childForward)
            return reject(diagnostic, "/nodes/" + std::to_string(source.id), "ProgramGraph child has no forward graph");
        std::set<std::string> forwardStages;
        for (const program::Node &node : childForward->nodes)
            forwardStages.insert(node.stage);
        for (const auto &[stageId, contract] : child.stages)
            if (forwardStages.count(stageId))
                program.stages.emplace(prefix + stageId, contract);
        std::set<std::string> forwardBlobs;
        for (const auto &[stageId, stage] : source.deployment->artifactSystem.stages) {
            if (!forwardStages.count(stageId))
                continue;
            program::StageArtifact linkedStage = stage;
            for (program::CodeModule &module : linkedStage.modules) {
                forwardBlobs.insert(module.blob);
                module.blob = prefix + module.blob;
            }
            artifacts.stages.emplace(prefix + stageId, std::move(linkedStage));
        }
        for (const auto &[blobId, blob] : source.deployment->artifactSystem.blobs) {
            if (!forwardBlobs.count(blobId))
                continue;
            const std::string linkedId = prefix + blobId;
            program::Blob linkedBlob = blob;
            linkedBlob.bundleRoot = source.bundleRoot;
            artifacts.blobs.emplace(linkedId, std::move(linkedBlob));
        }
        for (const program::Parameter &childParameter : child.parameters) {
            program::Parameter parameter = childParameter;
            parameter.id += offsets[nodeIndex].parameter;
            parameter.path = prefix + parameter.path;
            parameter.value = globalValue(nodeIndex, parameter.value);
            program.parameters.push_back(std::move(parameter));
        }
        for (const program::TapePlan &childPlan : child.abi.tapePlans) {
            program::TapePlan plan = childPlan;
            plan.value = globalValue(nodeIndex, plan.value);
            plan.backwardConsumer = false;
            program.abi.tapePlans.push_back(std::move(plan));
        }

        for (const program::Graph &childGraph : child.graphs) {
            if (childGraph.direction != "forward")
                continue;
            auto graph =
                std::find_if(program.graphs.begin(), program.graphs.end(), [&](const program::Graph &candidate) {
                    return candidate.direction == childGraph.direction;
                });
            if (graph == program.graphs.end()) {
                program.graphs.push_back({childGraph.name, childGraph.direction, {}, {}, {}, {}});
                graph = std::prev(program.graphs.end());
            }
            for (const program::GraphInput &childInput : childGraph.inputs) {
                const bool connected = std::any_of(destinations.begin(), destinations.end(), [&](const auto &entry) {
                    return entry.first.node == source.id &&
                           boundary(source, entry.first.slot)->value == childInput.value;
                });
                if (connected)
                    continue;
                program::GraphInput input = childInput;
                input.value = globalValue(nodeIndex, input.value);
                if (input.kind == program::GraphInputKind::UserInput)
                    input.slot += offsets[nodeIndex].userSlots.at(childGraph.direction);
                else if (input.kind == program::GraphInputKind::InvocationControl ||
                         input.kind == program::GraphInputKind::Control)
                    input.slot += offsets[nodeIndex].controlSlots.at(childGraph.direction);
                if (input.kind == program::GraphInputKind::Parameter)
                    input.parameter += offsets[nodeIndex].parameter;
                if (input.kind == program::GraphInputKind::Allocation)
                    input.storage = globalStorage(nodeIndex, input.storage);
                appendUnique(graph->inputs, std::move(input),
                             [](const program::GraphInput &value) { return std::pair{value.kind, value.value}; });
            }
            for (uint32_t capture : childGraph.captures)
                appendUnique(graph->captures, globalValue(nodeIndex, capture), [](uint32_t value) { return value; });
            for (const program::GraphOutput &childOutput : childGraph.outputs) {
                const bool internal = std::any_of(connections.begin(), connections.end(), [&](const auto &connection) {
                    return connection.source.node == source.id &&
                           boundary(source, connection.source.slot)->value == childOutput.value &&
                           !exportPaths.count(connection.source) && !retained.count(connection.source);
                });
                if (!internal)
                    appendUnique(
                        graph->outputs,
                        program::GraphOutput{globalValue(nodeIndex, childOutput.value), childOutput.disposition},
                        [](const program::GraphOutput &value) { return value.value; });
            }
            for (const program::Node &childNode : childGraph.nodes) {
                program::Node node = childNode;
                node.id = globalNode(nodeIndex, childGraph.direction, node.id);
                node.name = prefix + node.name;
                node.stage = prefix + node.stage;
                for (uint32_t &operand : node.operands)
                    operand = globalValue(nodeIndex, operand);
                for (uint32_t &result : node.results)
                    result = globalValue(nodeIndex, result);
                for (program::EndpointBinding &binding : node.bindings)
                    for (program::ValueEndpointProjection &projection : binding.projections)
                        projection.value = globalValue(nodeIndex, projection.value);
                for (program::ResourceAccess &access : node.accesses) {
                    const uint32_t localStorage = access.storage;
                    const uint32_t localBefore = access.before;
                    access.storage = globalStorage(nodeIndex, access.storage);
                    access.value = globalValue(nodeIndex, access.value);
                    const auto predecessor = storagePredecessors.find({source.id, localStorage});
                    if (predecessor != storagePredecessors.end() &&
                        localBefore == child.storages[localStorage].initialValue) {
                        const size_t predecessorNode = nodeIndices.at(predecessor->second.node);
                        const program::BoundarySlot &predecessorBoundary =
                            *boundary(nodes[predecessorNode], predecessor->second.slot);
                        access.before = globalValue(predecessorNode, predecessorBoundary.value);
                        const uint32_t disconnectedBefore = globalValue(nodeIndex, localBefore);
                        std::replace(node.operands.begin(), node.operands.end(), disconnectedBefore, access.before);
                    } else {
                        access.before = globalValue(nodeIndex, localBefore);
                    }
                    access.after = globalValue(nodeIndex, access.after);
                    if (access.view)
                        access.view = globalValue(nodeIndex, *access.view);
                }
                if (auto *compute = std::get_if<program::ComputeOperation>(&node.operation))
                    for (program::ControlComponent &control : compute->workgroups)
                        if (control.kind == program::ControlKind::Value)
                            control.reference = globalValue(nodeIndex, control.reference);
                if (auto *graphics = std::get_if<program::GraphicsOperation>(&node.operation)) {
                    const uint32_t controlOffset = offsets[nodeIndex].controlSlots.at(childGraph.direction);
                    graphics->renderPassControl += controlOffset;
                    graphics->drawCommandControl += controlOffset;
                    graphics->dynamicStateControl += controlOffset;
                }
                if (childGraph.direction == "forward" &&
                    program::executionKind(childNode) == program::ExecutionKind::Graphics)
                    linked.graphicsNodes.emplace(ProgramGraphGraphicsKey{source.id, childNode.id}, node.id);
                canonicalizeIds(node.operands);
                canonicalizeIds(node.results);
                graph->nodes.push_back(std::move(node));
            }
        }
    }

    for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        const ProgramGraphNodeSource &source = nodes[nodeIndex];
        for (const program::BoundarySlot &childSlot : source.deployment->program.abi.boundarySlots) {
            if (childSlot.role == program::BoundaryRole::Cotangent || childSlot.role == program::BoundaryRole::Gradient)
                continue;
            const ProgramGraphBoundaryKey key{source.id, childSlot.id};
            const bool connectedInput = destinations.count(key);
            const bool connectedOutput =
                std::any_of(connections.begin(), connections.end(),
                            [&](const ProgramGraphConnection &connection) { return connection.source == key; });
            const auto exported = exportPaths.find(key);
            if ((connectedInput || connectedOutput) && exported == exportPaths.end() && !retained.count(key))
                continue;
            program::BoundarySlot slot = childSlot;
            slot.id = static_cast<uint32_t>(program.abi.boundarySlots.size());
            slot.path = exported != exportPaths.end() ? exported->second
                                                      : "node/" + std::to_string(source.id) + "/" + childSlot.path;
            slot.value = globalValue(nodeIndex, slot.value);
            if (slot.aliasOwner.kind == program::ProgramOwnerKind::Value)
                slot.aliasOwner.id = globalValue(nodeIndex, slot.aliasOwner.id);
            else
                slot.aliasOwner.id = globalStorage(nodeIndex, slot.aliasOwner.id);
            if (slot.storage)
                slot.storage->id = globalStorage(nodeIndex, slot.storage->id);
            linked.boundarySlots.emplace(key, slot.id);
            program.abi.boundarySlots.push_back(std::move(slot));
        }
    }
    for (const ProgramGraphConnection &connection : connections) {
        if (!connection.storageConnection)
            continue;
        ProgramGraphBoundaryKey finalBoundary = connection.destination;
        for (size_t step = 0; step < connections.size(); ++step) {
            const auto next = std::find_if(connections.begin(), connections.end(), [&](const auto &candidate) {
                return candidate.storageConnection && candidate.source == finalBoundary;
            });
            if (next == connections.end())
                break;
            finalBoundary = next->destination;
        }
        const auto finalSlot = linked.boundarySlots.find(finalBoundary);
        if (finalSlot == linked.boundarySlots.end())
            return reject(diagnostic, "/connections", "storage connection does not terminate at an external boundary");
        linked.boundarySlots[connection.source] = finalSlot->second;
    }
    program.abi.publication = program::derivePublicationPlan(program.abi.boundarySlots);
    linked.deployment.key.clear();
    artifacts.target = target;
    std::string identity;
    const auto appendIdentity = [&](std::string_view value) {
        identity += std::to_string(value.size());
        identity += ':';
        identity.append(value.data(), value.size());
    };
    appendIdentity("vernon.program_graph.v2");
    for (const ProgramGraphNodeSource &node : nodes) {
        appendIdentity(std::to_string(node.id));
        appendIdentity(node.bundleId);
        appendIdentity(node.contentHash);
        for (const ProgramSpecialization &specialization : node.deployment->key) {
            appendIdentity(specialization.name);
            appendIdentity(std::to_string(static_cast<uint32_t>(specialization.kind)));
            std::visit(
                [&](const auto &value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, float>) {
                        uint32_t bits = 0;
                        std::memcpy(&bits, &value, sizeof(bits));
                        appendIdentity(std::to_string(bits));
                    } else if constexpr (std::is_same_v<T, double>) {
                        uint64_t bits = 0;
                        std::memcpy(&bits, &value, sizeof(bits));
                        appendIdentity(std::to_string(bits));
                    } else {
                        appendIdentity(std::to_string(value));
                    }
                },
                specialization.value);
        }
    }
    std::vector<ProgramGraphConnection> canonicalConnections = connections;
    std::sort(canonicalConnections.begin(), canonicalConnections.end(), [](const auto &left, const auto &right) {
        return std::tie(left.source.node, left.source.slot, left.destination.node, left.destination.slot,
                        left.storageConnection) < std::tie(right.source.node, right.source.slot, right.destination.node,
                                                           right.destination.slot, right.storageConnection);
    });
    for (const ProgramGraphConnection &connection : canonicalConnections) {
        appendIdentity(std::to_string(connection.source.node));
        appendIdentity(std::to_string(connection.source.slot));
        appendIdentity(std::to_string(connection.destination.node));
        appendIdentity(std::to_string(connection.destination.slot));
        appendIdentity(connection.storageConnection ? "storage" : "dataflow");
    }
    std::vector<ProgramGraphBoundaryKey> canonicalRetained(retained.begin(), retained.end());
    for (const ProgramGraphBoundaryKey boundary : canonicalRetained) {
        appendIdentity(std::to_string(boundary.node));
        appendIdentity(std::to_string(boundary.slot));
        appendIdentity("retained");
    }
    std::vector<ProgramGraphExport> canonicalExports = exports;
    std::sort(canonicalExports.begin(), canonicalExports.end(), [](const auto &left, const auto &right) {
        return std::tie(left.boundary.node, left.boundary.slot, left.path) <
               std::tie(right.boundary.node, right.boundary.slot, right.path);
    });
    for (const ProgramGraphExport &exported : canonicalExports) {
        appendIdentity(std::to_string(exported.boundary.node));
        appendIdentity(std::to_string(exported.boundary.slot));
        appendIdentity(exported.path);
    }
    linked.id = "program-graph:" + sha256Hex(identity.data(), identity.size());
    return true;
}

} // namespace vernon::runtime
