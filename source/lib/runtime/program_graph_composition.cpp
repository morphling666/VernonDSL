#include "program_graph_composition.h"
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

bool reject(program::Diagnostic &diagnostic, std::string path, std::string message) {
    diagnostic = {"PROGRAM_GRAPH_COMPOSE", "compose", std::move(path), std::move(message)};
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

std::string compositionIdentity(const std::vector<ProgramGraphNodeSource> &nodes,
                                std::vector<ProgramGraphConnection> connections,
                                const std::set<ProgramGraphBoundaryKey> &retained,
                                std::vector<ProgramGraphExport> exports) {
    std::string identity;
    const auto append = [&](std::string_view value) {
        identity += std::to_string(value.size());
        identity += ':';
        identity.append(value.data(), value.size());
    };
    append("vernon.program_graph");
    for (const ProgramGraphNodeSource &node : nodes) {
        append(std::to_string(node.id));
        append(node.bundleId);
        append(node.contentHash);
        for (const ProgramSpecialization &specialization : node.deployment->key) {
            append(specialization.name);
            append(std::to_string(static_cast<uint32_t>(specialization.kind)));
            std::visit(
                [&](const auto &value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, float>) {
                        uint32_t bits = 0;
                        std::memcpy(&bits, &value, sizeof(bits));
                        append(std::to_string(bits));
                    } else if constexpr (std::is_same_v<T, double>) {
                        uint64_t bits = 0;
                        std::memcpy(&bits, &value, sizeof(bits));
                        append(std::to_string(bits));
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        append(value);
                    } else {
                        append(std::to_string(value));
                    }
                },
                specialization.value);
        }
    }
    std::sort(connections.begin(), connections.end(), [](const auto &left, const auto &right) {
        return std::tie(left.source.node, left.source.slot, left.destination.node, left.destination.slot,
                        left.storageConnection) < std::tie(right.source.node, right.source.slot, right.destination.node,
                                                           right.destination.slot, right.storageConnection);
    });
    for (const ProgramGraphConnection &connection : connections) {
        append(std::to_string(connection.source.node));
        append(std::to_string(connection.source.slot));
        append(std::to_string(connection.destination.node));
        append(std::to_string(connection.destination.slot));
        append(connection.storageConnection ? "storage" : "dataflow");
    }
    for (const ProgramGraphBoundaryKey boundary : retained) {
        append(std::to_string(boundary.node));
        append(std::to_string(boundary.slot));
        append("retained");
    }
    std::sort(exports.begin(), exports.end(), [](const auto &left, const auto &right) {
        return std::tie(left.boundary.node, left.boundary.slot, left.path) <
               std::tie(right.boundary.node, right.boundary.slot, right.path);
    });
    for (const ProgramGraphExport &exported : exports) {
        append(std::to_string(exported.boundary.node));
        append(std::to_string(exported.boundary.slot));
        append(exported.path);
    }
    return "program-graph:" + sha256Hex(identity.data(), identity.size());
}

} // namespace

bool composeProgramGraph(const std::vector<ProgramGraphNodeSource> &nodes,
                         const std::vector<ProgramGraphConnection> &connections,
                         const std::vector<ProgramGraphBoundaryKey> &retainedBoundaries,
                         const std::vector<ProgramGraphExport> &exports, ComposedProgramAbi &composed,
                         program::Diagnostic &diagnostic) {
    diagnostic = {};
    composed = {};
    if (nodes.empty())
        return reject(diagnostic, "/nodes", "ProgramGraph requires at least one Program node");

    std::map<uint32_t, size_t> nodeIndices;
    std::vector<size_t> valueOffsets(nodes.size());
    std::vector<size_t> storageOffsets(nodes.size());
    size_t valueCount = 0;
    size_t storageCount = 0;
    const std::string target = nodes.front().deployment ? nodes.front().deployment->artifactSystem.target : "";
    for (size_t index = 0; index < nodes.size(); ++index) {
        const ProgramGraphNodeSource &node = nodes[index];
        if (!node.deployment || node.id == UINT32_MAX || node.bundleId.empty() || node.contentHash.empty() ||
            !nodeIndices.emplace(node.id, index).second)
            return reject(diagnostic, "/nodes/" + std::to_string(index), "ProgramGraph node is invalid or duplicated");
        if (node.deployment->artifactSystem.target != target)
            return reject(diagnostic, "/nodes/" + std::to_string(node.id),
                          "ProgramGraph nodes target different backends");
        if (!program::findGraph(node.deployment->program, "forward"))
            return reject(diagnostic, "/nodes/" + std::to_string(node.id), "ProgramGraph node has no forward graph");
        valueOffsets[index] = valueCount;
        storageOffsets[index] = storageCount;
        valueCount += node.deployment->program.values.size();
        storageCount += node.deployment->program.storages.size();
    }

    DisjointSet valueSets(valueCount);
    DisjointSet storageSets(storageCount);
    std::set<ProgramGraphBoundaryKey> destinations;
    std::set<std::pair<uint32_t, uint32_t>> storagePredecessors;
    std::set<std::pair<uint32_t, uint32_t>> storageSuccessors;
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
        const std::string path = "/connections/" + std::to_string(index);
        const auto [sourceNode, source] = locate(connection.source);
        const auto [destinationNode, destination] = locate(connection.destination);
        if (!sourceNode || !source || !destinationNode || !destination)
            return reject(diagnostic, path, "connection references an unknown boundary");
        if (connection.storageConnection ? !storageConnectionCompatible(*source, *destination)
                                         : !compatible(*source, *destination))
            return reject(diagnostic, path, "connected Program boundaries have incompatible contracts");
        const size_t sourceIndex = nodeIndices.at(connection.source.node);
        const size_t destinationIndex = nodeIndices.at(connection.destination.node);
        if (connection.storageConnection) {
            if (sourceIndex >= destinationIndex)
                return reject(diagnostic, path, "Program Storage connections must follow node order");
            const program::Storage &leftStorage = sourceNode->deployment->program.storages[source->aliasOwner.id];
            const program::Storage &rightStorage =
                destinationNode->deployment->program.storages[destination->aliasOwner.id];
            if (leftStorage.ownership != rightStorage.ownership || leftStorage.lifetime != rightStorage.lifetime ||
                leftStorage.mutability != rightStorage.mutability)
                return reject(diagnostic, path, "connected Program Storages have incompatible ownership");
            if (!storageSuccessors.emplace(connection.source.node, source->aliasOwner.id).second)
                return reject(diagnostic, path, "a Program Storage version may have only one successor");
            if (!storagePredecessors.emplace(connection.destination.node, destination->aliasOwner.id).second)
                return reject(diagnostic, path, "a Program Storage version may have only one predecessor");
        } else {
            if (!destinations.emplace(connection.destination).second)
                return reject(diagnostic, path, "a Program input boundary may have only one source");
            valueSets.join(valueOffsets[sourceIndex] + source->value,
                           valueOffsets[destinationIndex] + destination->value);
        }
        if (source->aliasOwner.kind == program::ProgramOwnerKind::Storage)
            storageSets.join(storageOffsets[sourceIndex] + source->aliasOwner.id,
                             storageOffsets[destinationIndex] + destination->aliasOwner.id);
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
        const bool connectedDestination = destinations.count(exports[index].boundary) != 0;
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
    const auto composedValue = [&](size_t nodeIndex, uint32_t value) {
        return valueIds.emplace(valueSets.root(valueOffsets[nodeIndex] + value), static_cast<uint32_t>(valueIds.size()))
            .first->second;
    };
    const auto composedStorage = [&](size_t nodeIndex, uint32_t storage) {
        return storageIds
            .emplace(storageSets.root(storageOffsets[nodeIndex] + storage), static_cast<uint32_t>(storageIds.size()))
            .first->second;
    };

    std::vector<program::BoundarySlot> &slots = composed.program.abi.boundarySlots;
    for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        const ProgramGraphNodeSource &source = nodes[nodeIndex];
        for (const program::BoundarySlot &childSlot : source.deployment->program.abi.boundarySlots) {
            if (childSlot.role == program::BoundaryRole::Cotangent || childSlot.role == program::BoundaryRole::Gradient)
                continue;
            const ProgramGraphBoundaryKey key{source.id, childSlot.id};
            const bool connectedOutput =
                std::any_of(connections.begin(), connections.end(),
                            [&](const ProgramGraphConnection &connection) { return connection.source == key; });
            const auto exported = exportPaths.find(key);
            if ((destinations.count(key) || connectedOutput) && exported == exportPaths.end() && !retained.count(key))
                continue;
            program::BoundarySlot slot = childSlot;
            slot.id = static_cast<uint32_t>(slots.size());
            slot.path = exported != exportPaths.end() ? exported->second
                                                      : "node/" + std::to_string(source.id) + "/" + childSlot.path;
            slot.value = composedValue(nodeIndex, slot.value);
            slot.aliasOwner.id = slot.aliasOwner.kind == program::ProgramOwnerKind::Value
                                     ? composedValue(nodeIndex, slot.aliasOwner.id)
                                     : composedStorage(nodeIndex, slot.aliasOwner.id);
            if (slot.storage)
                slot.storage->id = composedStorage(nodeIndex, slot.storage->id);
            composed.boundarySlots.emplace(key, slot.id);
            slots.push_back(std::move(slot));
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
        const auto finalSlot = composed.boundarySlots.find(finalBoundary);
        if (finalSlot == composed.boundarySlots.end())
            return reject(diagnostic, "/connections", "storage connection does not terminate at an external boundary");
        composed.boundarySlots[connection.source] = finalSlot->second;
    }
    composed.program.abi.publication = program::derivePublicationPlan(slots);
    composed.id = compositionIdentity(nodes, connections, retained, exports);
    return true;
}

} // namespace vernon::runtime
