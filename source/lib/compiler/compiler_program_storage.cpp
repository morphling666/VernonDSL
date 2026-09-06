#include "compiler_program_storage.h"

#include "compiler_json.h"

#include <limits>
#include <optional>
#include <vector>

namespace vernon::compiler {
namespace {

std::vector<std::string> quotedTypeFields(llvm::StringRef type) {
    std::vector<std::string> fields;
    while (true) {
        const size_t begin = type.find('"');
        if (begin == llvm::StringRef::npos)
            return fields;
        type = type.drop_front(begin + 1);
        const size_t end = type.find('"');
        if (end == llvm::StringRef::npos)
            return {};
        fields.push_back(type.take_front(end).str());
        type = type.drop_front(end + 1);
    }
}

llvm::json::Object manifestLayout(const llvm::json::Object &layout, llvm::StringRef scope) {
    llvm::json::Object result;
    for (const auto &[key, value] : layout)
        if (key != "logical_type" && key != "struct_name")
            result[key] = value;
    result["scope"] = scope.str();
    return result;
}

bool checkedByteLength(const llvm::json::Object &layout, const llvm::json::Array &shape, uint64_t &length) {
    const std::optional<int64_t> byteSize = layout.getInteger("byte_size");
    if (!byteSize || *byteSize <= 0)
        return false;
    length = static_cast<uint64_t>(*byteSize);
    for (const llvm::json::Value &extentValue : shape) {
        const std::optional<int64_t> extent = extentValue.getAsInteger();
        if (!extent || *extent <= 0 || length > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(*extent))
            return false;
        length *= static_cast<uint64_t>(*extent);
    }
    return length <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
}

bool identicalConcreteShape(const llvm::json::Array *left, const llvm::json::Array *right) {
    if (!left || !right || left->size() != right->size())
        return false;
    for (size_t index = 0; index < left->size(); ++index) {
        const std::optional<int64_t> expected = (*left)[index].getAsInteger();
        const std::optional<int64_t> actual = (*right)[index].getAsInteger();
        if (!expected || !actual)
            return false;
        if (*expected > 0 && *actual > 0 && *expected != *actual)
            return false;
        if ((*expected > 0) != (*actual > 0))
            return false;
    }
    return true;
}

bool entryAvailable(int64_t value, llvm::StringRef graph, bool captureLegal,
                    const std::map<int64_t, ProgramArgumentSlot> &argumentSlots, const std::set<int64_t> &captures) {
    if (isCurrentGraphArgument(value, graph, argumentSlots))
        return true;
    return captureLegal && captures.count(value) != 0;
}

int64_t walkLikeSource(const llvm::json::Array &values, int64_t root, llvm::StringRef allocationGraph,
                       bool captureLegal, const std::map<int64_t, ProgramArgumentSlot> &argumentSlots,
                       const std::set<int64_t> &captures,
                       const std::map<int64_t, ProgramValueProducer> &producerByValue,
                       const std::map<int64_t, int64_t> &parents, std::string &error) {
    std::set<int64_t> seen;
    int64_t current = root;
    while (true) {
        if (!seen.insert(current).second) {
            error = "owned Program Storage like-source cycle";
            return -1;
        }
        const llvm::json::Object *value = findJsonObjectByIntegerId(values, current);
        if (!value) {
            error = "owned Program Storage like-source is unknown";
            return -1;
        }
        if (const std::optional<int64_t> like = value->getInteger("like")) {
            current = *like;
            continue;
        }
        if (entryAvailable(current, allocationGraph, captureLegal, argumentSlots, captures))
            return current;
        if (!producerByValue.count(current)) {
            error = "owned Program Storage like-source is not a ControlValueRef";
            return -1;
        }
        const llvm::json::Object *allocated = findJsonObjectByIntegerId(values, programStorageRoot(parents, current));
        if (allocated)
            if (const std::optional<int64_t> like = allocated->getInteger("like"); like && *like != current) {
                current = *like;
                continue;
            }
        error = "owned Program Storage like-source has NodeResultOrigin";
        return -1;
    }
}

std::optional<llvm::json::Object>
controlValueReference(int64_t like, llvm::StringRef allocationGraph, bool captureLegal,
                      const std::map<int64_t, ProgramArgumentSlot> &argumentSlots, const std::set<int64_t> &captures,
                      const std::map<int64_t, ProgramValueProducer> &producerByValue, std::string &error) {
    if (isCurrentGraphArgument(like, allocationGraph, argumentSlots))
        return llvm::json::Object{{"value", like}};
    if (captureLegal && captures.count(like))
        return llvm::json::Object{{"capture", like}};
    error = producerByValue.count(like) ? "owned Program Storage like-source has NodeResultOrigin"
                                        : "owned Program Storage like-source is not a ControlValueRef";
    return std::nullopt;
}

llvm::json::Value dimensionExtent(const llvm::json::Object &reference, int64_t axis) {
    llvm::json::Object copy;
    for (const auto &[key, value] : reference)
        copy[key] = value;
    return llvm::json::Object{{"dimension", llvm::json::Object{{"control", std::move(copy)}, {"axis", axis}}}};
}

} // namespace

bool isProgramTextureType(llvm::StringRef type) { return type.starts_with("!vernon.texture<"); }

bool isProgramSamplerType(llvm::StringRef type) { return type.starts_with("!vernon.sampler"); }

bool isProgramAdTapeType(llvm::StringRef type) {
    return type == "!vernon.ad_tape" || type.starts_with("!vernon.ad_tape<");
}

bool isProgramTensorViewType(llvm::StringRef type) { return type.starts_with("!vernon.tensor_view<"); }

bool isValidProgramResourceAccess(llvm::StringRef access) {
    return access == "read" || access == "write" || access == "read_write";
}

bool indexProgramResources(const llvm::json::Array &values, const std::vector<CanonicalProgramGraph> &selectedGraphs,
                           const std::vector<CanonicalProgramStage> &compiledStages, ProgramResourceIndex &index,
                           std::string &error) {
    index = {};
    std::set<std::string> indexedStages;
    for (const CanonicalProgramStage &compiled : compiledStages)
        if (compiled.requestId.empty() || !index.compiledByRequest.emplace(compiled.requestId, &compiled).second) {
            error = "canonical Program stages have invalid or duplicate logical request ids";
            return false;
        }
    for (const CanonicalProgramGraph &view : selectedGraphs) {
        index.graphDirections[view.name] = view.direction;
        for (size_t slot = 0; slot < view.arguments.size(); ++slot) {
            const int64_t id = view.arguments[slot];
            if (!index.argumentSlots.emplace(id, ProgramArgumentSlot{view.name, static_cast<int64_t>(slot)}).second) {
                error = "canonical Program graph has invalid argument ids";
                return false;
            }
        }
        for (const llvm::json::Object *node : view.nodes) {
            const std::optional<llvm::StringRef> requestId = node ? node->getString("stage") : std::nullopt;
            const llvm::json::Array *nodeResults = node ? node->getArray("results") : nullptr;
            const llvm::json::Array *rawResources = node ? node->getArray("resources") : nullptr;
            const std::optional<llvm::StringRef> kind = node ? node->getString("kind") : std::nullopt;
            const llvm::json::Array *grid = node ? node->getArray("grid") : nullptr;
            if (!node || (kind != "compute" && kind != "render") || !requestId ||
                !index.compiledByRequest.count(requestId->str()) || !node->getArray("operands") || !nodeResults ||
                !node->getArray("bindings") || !rawResources || (kind == "compute" && (!grid || grid->size() != 3)) ||
                !indexedStages.insert(requestId->str()).second) {
                error = "canonical Program node has incomplete or duplicate logical execution metadata";
                return false;
            }
            for (const llvm::json::Value &operand : *node->getArray("operands"))
                if (const std::optional<int64_t> id = operand.getAsInteger();
                    id && !index.argumentSlots.count(*id) && !index.producerByValue.count(*id))
                    index.allocationGraph.try_emplace(*id, view.name);
            for (const llvm::json::Value &value : *nodeResults) {
                const std::optional<int64_t> id = value.getAsInteger();
                if (!id || !findJsonObjectByIntegerId(values, *id) ||
                    !index.producerByValue
                         .emplace(*id, ProgramValueProducer{view.name, node->getInteger("id").value_or(-1)})
                         .second) {
                    error = "canonical Program graph has invalid or multiply-produced result ids";
                    return false;
                }
            }
            std::map<int64_t, ProgramLogicalResource> &resources = index.resourcesByStage[requestId->str()];
            for (const llvm::json::Value &resourceValue : *rawResources) {
                const llvm::json::Object *resource = resourceValue.getAsObject();
                const std::optional<int64_t> value = resource ? resource->getInteger("value") : std::nullopt;
                const std::optional<llvm::StringRef> access = resource ? resource->getString("access") : std::nullopt;
                if (!value || !access || !isValidProgramResourceAccess(*access)) {
                    error = "canonical Program node has an invalid resource access";
                    return false;
                }
                const llvm::json::Object *logicalValue = findJsonObjectByIntegerId(values, *value);
                const llvm::StringRef type = logicalValue ? logicalValue->getString("type").value_or("") : "";
                const bool actualResource = isProgramTensorViewType(type) || isProgramTextureType(type) ||
                                            isProgramSamplerType(type) || isProgramAdTapeType(type);
                if (!actualResource)
                    continue;
                ProgramLogicalResource logical{*value, resource->getInteger("after"), access->str(), -1};
                if (const auto existing = resources.find(*value); existing != resources.end()) {
                    if (existing->second.access == "read" && logical.access == "read" && !existing->second.after &&
                        !logical.after)
                        continue;
                    error = "canonical Program node has conflicting repeated resource accesses";
                    return false;
                }
                const bool writable = logical.access == "write" || logical.access == "read_write";
                if ((!writable && logical.after) ||
                    (writable && !logical.after && !index.producerByValue.count(*value))) {
                    error = "canonical Program stage '" + requestId->str() + "' resource Value " +
                            std::to_string(*value) + " ('" +
                            (logicalValue ? logicalValue->getString("name").value_or("<unnamed>").str() : "<unknown>") +
                            "', type '" + type.str() + "') with access '" + logical.access +
                            "' violates exact before/after Storage SSA";
                    return false;
                }
                index.storageParents.emplace(*value, *value);
                index.resourceVersions.insert(*value);
                if (logical.after) {
                    index.storageParents.emplace(*logical.after, *logical.after);
                    const int64_t beforeRoot = programStorageRoot(index.storageParents, *value);
                    const int64_t afterRoot = programStorageRoot(index.storageParents, *logical.after);
                    index.storageParents[afterRoot] = beforeRoot;
                    index.resourceVersions.insert(*logical.after);
                }
                resources.emplace(*value, std::move(logical));
            }
        }
    }
    return true;
}

bool planProgramStorageAliases(const std::map<int64_t, int64_t> &parents, ProgramStorageAliasPlan &plan,
                               std::string &error) {
    plan = {};
    for (const auto &[value, unusedParent] : parents) {
        (void)unusedParent;
        std::set<int64_t> visited;
        int64_t owner = value;
        while (true) {
            const auto parent = parents.find(owner);
            if (parent == parents.end()) {
                error = "canonical Program Storage alias chain has an unknown parent";
                return false;
            }
            if (!visited.insert(owner).second) {
                error = "canonical Program Storage alias chain contains a cycle";
                return false;
            }
            if (parent->second == owner)
                break;
            owner = parent->second;
        }
        const ProgramStorageOwnerId ownerId{owner};
        plan.ownerByValue.emplace(value, ownerId);
        plan.versionsByOwner[ownerId].push_back(value);
    }
    return true;
}

bool materializeProgramStoragePlan(const llvm::json::Array &rawValues,
                                   const std::vector<CanonicalProgramGraph> &selectedGraphs,
                                   ProgramResourceIndex &index, std::set<int64_t> &capturedValues,
                                   ProgramStoragePlan &plan, std::string &error) {
    plan = {};
    if (!planProgramStorageAliases(index.storageParents, plan.aliases, error))
        return false;
    bool assignedAllocationGraph = true;
    while (assignedAllocationGraph) {
        assignedAllocationGraph = false;
        for (const llvm::json::Value &rowValue : rawValues) {
            const llvm::json::Object *value = rowValue.getAsObject();
            const std::optional<int64_t> id = value ? value->getInteger("id") : std::nullopt;
            const std::optional<int64_t> like = value ? value->getInteger("like") : std::nullopt;
            if (!id || !like)
                continue;
            const auto found = index.allocationGraph.find(*id);
            if (found == index.allocationGraph.end() || index.argumentSlots.count(*like) ||
                index.producerByValue.count(*like))
                continue;
            assignedAllocationGraph |= index.allocationGraph.try_emplace(*like, found->second).second;
        }
    }
    for (const auto &[id, graphName] : index.allocationGraph) {
        if (index.graphDirections[graphName] != "backward")
            continue;
        const llvm::json::Object *allocated = findJsonObjectByIntegerId(rawValues, id);
        const llvm::json::Array *shape = allocated ? allocated->getArray("shape") : nullptr;
        bool needsRuntimeExtents = false;
        if (shape)
            for (const llvm::json::Value &extent : *shape)
                needsRuntimeExtents |= isDynamicProgramExtent(extent);
        if (!needsRuntimeExtents)
            continue;
        std::string walkError;
        const int64_t like = resolveOwnedLikeSource(rawValues, id, graphName, true, index.argumentSlots, capturedValues,
                                                    index.producerByValue, index.storageParents, walkError);
        if (like >= 0 && !isCurrentGraphArgument(like, graphName, index.argumentSlots))
            capturedValues.insert(like);
    }

    std::map<int64_t, int64_t> storageByRoot;
    std::map<int64_t, bool> mutableByRoot;
    std::set<int64_t> colorAttachmentRoots;
    std::set<int64_t> depthAttachmentRoots;
    for (const CanonicalProgramGraph &graph : selectedGraphs)
        for (const llvm::json::Object *node : graph.nodes) {
            if (node->getString("kind") != "render")
                continue;
            const llvm::json::Array *operands = node->getArray("operands");
            const llvm::json::Array *results = node->getArray("results");
            const int64_t colorCount = node->getInteger("color_count").value_or(0);
            const bool hasDepth = results && colorCount >= 0 && results->size() == static_cast<size_t>(colorCount + 1);
            if (!operands || colorCount < 0 || operands->size() < static_cast<size_t>(colorCount + (hasDepth ? 1 : 0)))
                continue;
            for (int64_t index = 0; index < colorCount; ++index)
                if (const std::optional<int64_t> value = (*operands)[static_cast<size_t>(index)].getAsInteger())
                    colorAttachmentRoots.insert(plan.aliases.ownerByValue.at(*value).value);
            if (hasDepth)
                if (const std::optional<int64_t> value = (*operands)[static_cast<size_t>(colorCount)].getAsInteger())
                    depthAttachmentRoots.insert(plan.aliases.ownerByValue.at(*value).value);
        }
    for (const auto &[unusedStage, resources] : index.resourcesByStage) {
        (void)unusedStage;
        for (const auto &[before, resource] : resources)
            mutableByRoot[plan.aliases.ownerByValue.at(before).value] |= resource.access != "read";
    }
    for (const auto &[owner, versions] : plan.aliases.versionsByOwner) {
        (void)versions;
        const int64_t root = owner.value;
        const llvm::json::Object *value = findJsonObjectByIntegerId(rawValues, root);
        const llvm::json::Object *layout = value ? value->getObject("value_layout") : nullptr;
        const llvm::json::Array *shape = value ? value->getArray("shape") : nullptr;
        const llvm::StringRef type = value ? value->getString("type").value_or("") : "";
        if (!value || !shape || type.empty()) {
            error = "canonical Program resource has no valid static type or shape";
            return false;
        }
        const int64_t storage = static_cast<int64_t>(plan.storages.size());
        storageByRoot[root] = storage;
        const auto foundGraph = index.allocationGraph.find(root);
        const llvm::StringRef allocationGraph = foundGraph != index.allocationGraph.end()
                                                    ? llvm::StringRef(foundGraph->second)
                                                    : llvm::StringRef(selectedGraphs.front().name);
        const auto direction = index.graphDirections.find(allocationGraph.str());
        const bool captureLegal = direction != index.graphDirections.end() && direction->second == "backward";
        llvm::json::Object descriptor;
        if (isProgramTextureType(type)) {
            const std::vector<std::string> fields = quotedTypeFields(type);
            if (fields.size() < 2 || shape->size() > 3) {
                error = "canonical Program texture has no valid dimension, format, or rank";
                return false;
            }
            size_t spatialRank = shape->size();
            if (spatialRank == 0)
                spatialRank = fields[0] == "1d" ? 1 : fields[0] == "3d" ? 3 : 2;
            const bool borrowed = index.argumentSlots.count(root);
            llvm::json::Array extent;
            llvm::json::Array dynamicExtents;
            bool hasDynamic = false;
            for (size_t axis = 0; axis < spatialRank; ++axis)
                if (axis < shape->size() && isDynamicProgramExtent((*shape)[axis]))
                    hasDynamic = true;
            if (hasDynamic && !borrowed &&
                !planOwnedDynamicExtents(rawValues, root, allocationGraph, captureLegal, *shape, index.argumentSlots,
                                         capturedValues, index.producerByValue, index.storageParents, dynamicExtents,
                                         error))
                return false;
            for (size_t axis = 0; axis < 3; ++axis) {
                const size_t sourceAxis = spatialRank > axis ? spatialRank - axis - 1 : spatialRank;
                if (sourceAxis >= spatialRank) {
                    extent.emplace_back(int64_t{1});
                    continue;
                }
                const llvm::json::Value *planned = sourceAxis < shape->size() ? &(*shape)[sourceAxis] : nullptr;
                if (planned && !isDynamicProgramExtent(*planned))
                    extent.emplace_back(*planned->getAsInteger());
                else if (borrowed)
                    extent.emplace_back(int64_t{0});
                else if (sourceAxis < dynamicExtents.size())
                    extent.emplace_back(dynamicExtents[sourceAxis]);
                else {
                    error = "owned Program texture requires a concrete extent";
                    return false;
                }
            }
            const bool colorAttachment = colorAttachmentRoots.count(root);
            const bool depthAttachment = depthAttachmentRoots.count(root);
            descriptor = llvm::json::Object{{"tag", "image"},
                                            {"dimension", fields[0]},
                                            {"extent", std::move(extent)},
                                            {"format", fields[1]},
                                            {"sample_count", int64_t{1}},
                                            {"mip_levels", int64_t{1}},
                                            {"array_layers", int64_t{1}},
                                            {"aspects", llvm::json::Array{depthAttachment ? "depth" : "color"}},
                                            {"usage", llvm::json::Array{colorAttachment   ? "color_attachment"
                                                                        : depthAttachment ? "depth_stencil_attachment"
                                                                                          : "storage"}}};
        } else if (isProgramSamplerType(type) || isProgramAdTapeType(type)) {
            descriptor = llvm::json::Object{
                {"tag", "opaque"},
                {"contract_hash", canonicalJsonSha256(llvm::json::Value(llvm::json::Object{{"type", type.str()}}))}};
        } else {
            uint64_t byteLength = 0;
            const std::optional<int64_t> alignment = layout ? layout->getInteger("alignment") : std::nullopt;
            const std::optional<int64_t> byteSize = layout ? layout->getInteger("byte_size") : std::nullopt;
            const bool borrowed = index.argumentSlots.count(root);
            if (!layout || !alignment || *alignment <= 0 || !byteSize || *byteSize <= 0) {
                error = "canonical Program buffer has no valid static layout";
                return false;
            }
            llvm::json::Value byteLengthValue = nullptr;
            if (checkedByteLength(*layout, *shape, byteLength))
                byteLengthValue = static_cast<int64_t>(byteLength);
            else if (borrowed)
                byteLengthValue = int64_t{0};
            else {
                llvm::json::Array extents;
                if (!planOwnedDynamicExtents(rawValues, root, allocationGraph, captureLegal, *shape,
                                             index.argumentSlots, capturedValues, index.producerByValue,
                                             index.storageParents, extents, error))
                    return false;
                byteLengthValue = std::move(extents);
            }
            descriptor = llvm::json::Object{{"tag", "buffer"},
                                            {"byte_length", std::move(byteLengthValue)},
                                            {"alignment", *alignment},
                                            {"memory", "device"},
                                            {"usage", llvm::json::Array{"storage"}}};
        }
        plan.storages.emplace_back(llvm::json::Object{
            {"id", storage},
            {"name", value->getString("name").value_or("value").str()},
            {"initial_value", root},
            {"ownership", index.argumentSlots.count(root) ? "borrowed" : "owned"},
            {"lifetime", isProgramAdTapeType(type) ? "pullback" : "invocation"},
            {"mutability", mutableByRoot[root] ? "mutable" : "read_only"},
            {"descriptor", std::move(descriptor)},
        });
    }

    for (const auto &[owner, versions] : plan.aliases.versionsByOwner) {
        const int64_t root = owner.value;
        for (int64_t version : versions) {
            const llvm::json::Object *initial = findJsonObjectByIntegerId(rawValues, root);
            const llvm::json::Object *current = findJsonObjectByIntegerId(rawValues, version);
            const llvm::json::Object *initialLayout = initial ? initial->getObject("value_layout") : nullptr;
            const llvm::json::Object *currentLayout = current ? current->getObject("value_layout") : nullptr;
            const llvm::json::Array *initialShape = initial ? initial->getArray("shape") : nullptr;
            const llvm::json::Array *currentShape = current ? current->getArray("shape") : nullptr;
            const llvm::StringRef initialType = initial ? initial->getString("type").value_or("") : "";
            const llvm::StringRef currentType = current ? current->getString("type").value_or("") : "";
            const bool opaqueResource = isProgramTextureType(initialType) || isProgramSamplerType(initialType) ||
                                        isProgramAdTapeType(initialType);
            if (initialType.empty() || initialType != currentType ||
                (!opaqueResource &&
                 (!initialLayout || !currentLayout ||
                  initialLayout->getString("layout_hash") != currentLayout->getString("layout_hash"))) ||
                !identicalConcreteShape(initialShape, currentShape) ||
                !identicalConcreteShape(currentShape, initialShape)) {
                error = "canonical Program Storage SSA versions disagree on layout or concrete shape";
                return false;
            }
            plan.storageByValue[version] = storageByRoot[root];
        }
    }
    for (auto &[unusedStage, resources] : index.resourcesByStage) {
        (void)unusedStage;
        for (auto &[before, resource] : resources)
            resource.storage = plan.storageByValue[before];
    }
    for (size_t expectedId = 0; expectedId < rawValues.size(); ++expectedId) {
        const llvm::json::Object *value = findJsonObjectByIntegerId(rawValues, static_cast<int64_t>(expectedId));
        const llvm::json::Object *layout = value ? value->getObject("value_layout") : nullptr;
        const llvm::json::Array *shape = value ? value->getArray("shape") : nullptr;
        const std::optional<llvm::StringRef> type = value ? value->getString("type") : std::nullopt;
        if (!value || !shape || !type) {
            error = "canonical Program values must have contiguous ids, types, and shapes";
            return false;
        }
        llvm::json::Object origin;
        if (const auto argument = index.argumentSlots.find(static_cast<int64_t>(expectedId));
            argument != index.argumentSlots.end())
            origin = llvm::json::Object{
                {"tag", "argument"}, {"graph", argument->second.graph}, {"slot", argument->second.slot}};
        else if (const auto producer = index.producerByValue.find(static_cast<int64_t>(expectedId));
                 producer != index.producerByValue.end())
            origin = llvm::json::Object{
                {"tag", "node_result"}, {"graph", producer->second.graph}, {"node", producer->second.node}};
        else if (const auto found = index.allocationGraph.find(static_cast<int64_t>(expectedId));
                 found != index.allocationGraph.end())
            origin = llvm::json::Object{{"tag", "allocation"}, {"graph", found->second}};
        else {
            error = "canonical Program value has no origin";
            return false;
        }
        const bool resource = index.resourceVersions.count(static_cast<int64_t>(expectedId));
        const bool opaqueResource =
            isProgramTextureType(*type) || isProgramSamplerType(*type) || isProgramAdTapeType(*type);
        if (!opaqueResource && !layout) {
            error = "canonical Program byte values must have layouts";
            return false;
        }
        llvm::json::Object row{{"id", static_cast<int64_t>(expectedId)},
                               {"name", value->getString("name").value_or("value").str()},
                               {"type", type->str()},
                               {"origin", std::move(origin)},
                               {"shape", copyJsonArray(*shape)}};
        if (!opaqueResource)
            row["value_layout"] = manifestLayout(*layout, resource ? "element" : "value");
        if (resource)
            row["storage"] = plan.storageByValue[static_cast<int64_t>(expectedId)];
        plan.values.emplace_back(std::move(row));
    }
    return true;
}

int64_t programStorageRoot(const std::map<int64_t, int64_t> &parents, int64_t value) {
    int64_t root = value;
    while (parents.count(root) && parents.at(root) != root)
        root = parents.at(root);
    return root;
}

bool isDynamicProgramExtent(const llvm::json::Value &value) {
    const std::optional<int64_t> extent = value.getAsInteger();
    return !extent || *extent <= 0;
}

bool isCurrentGraphArgument(int64_t value, llvm::StringRef graph,
                            const std::map<int64_t, ProgramArgumentSlot> &argumentSlots) {
    const auto argument = argumentSlots.find(value);
    return argument != argumentSlots.end() && argument->second.graph == graph;
}

int64_t resolveOwnedLikeSource(const llvm::json::Array &values, int64_t root, llvm::StringRef allocationGraph,
                               bool captureLegal, const std::map<int64_t, ProgramArgumentSlot> &argumentSlots,
                               const std::set<int64_t> &captures,
                               const std::map<int64_t, ProgramValueProducer> &producerByValue,
                               const std::map<int64_t, int64_t> &parents, std::string &error) {
    return walkLikeSource(values, root, allocationGraph, captureLegal, argumentSlots, captures, producerByValue,
                          parents, error);
}

bool planOwnedDynamicExtents(const llvm::json::Array &values, int64_t root, llvm::StringRef allocationGraph,
                             bool captureLegal, const llvm::json::Array &shape,
                             const std::map<int64_t, ProgramArgumentSlot> &argumentSlots,
                             const std::set<int64_t> &captures,
                             const std::map<int64_t, ProgramValueProducer> &producerByValue,
                             const std::map<int64_t, int64_t> &parents, llvm::json::Array &extents,
                             std::string &error) {
    const int64_t like = resolveOwnedLikeSource(values, root, allocationGraph, captureLegal, argumentSlots, captures,
                                                producerByValue, parents, error);
    if (like < 0)
        return false;
    const std::optional<llvm::json::Object> reference =
        controlValueReference(like, allocationGraph, captureLegal, argumentSlots, captures, producerByValue, error);
    if (!reference)
        return false;
    const llvm::json::Object *likeValue = findJsonObjectByIntegerId(values, like);
    const llvm::json::Array *likeShape = likeValue ? likeValue->getArray("shape") : nullptr;
    if (!likeShape || likeShape->size() != shape.size()) {
        error = "owned Program Storage like-source rank does not match the allocated value";
        return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        const llvm::json::Value &extent = shape[axis];
        if (!isDynamicProgramExtent(extent)) {
            extents.emplace_back(*extent.getAsInteger());
            continue;
        }
        if (!isDynamicProgramExtent((*likeShape)[axis])) {
            error = "owned Program Storage dyn axis has a static like-source extent";
            return false;
        }
        extents.emplace_back(dimensionExtent(*reference, static_cast<int64_t>(axis)));
    }
    return true;
}

bool applyProgramStorageUsageRequirements(llvm::json::Array &storages,
                                          const ProgramStorageUsageRequirements &requirements, std::string &error) {
    for (const auto &[storageId, required] : requirements) {
        if (storageId < 0 || static_cast<size_t>(storageId) >= storages.size()) {
            error = "Program storage usage requirement references an unknown Storage";
            return false;
        }
        llvm::json::Object *storage = storages[static_cast<size_t>(storageId)].getAsObject();
        llvm::json::Object *descriptor = storage ? storage->getObject("descriptor") : nullptr;
        llvm::json::Array *usage = descriptor ? descriptor->getArray("usage") : nullptr;
        if (!usage) {
            error = "Program storage usage requirement has no canonical descriptor";
            return false;
        }
        std::set<std::string> merged = required;
        for (const llvm::json::Value &value : *usage) {
            const std::optional<llvm::StringRef> name = value.getAsString();
            if (!name || name->empty()) {
                error = "Program Storage contains an invalid usage";
                return false;
            }
            merged.insert(name->str());
        }
        llvm::json::Array canonical;
        for (const std::string &name : merged)
            canonical.emplace_back(name);
        *usage = std::move(canonical);
    }
    return true;
}

} // namespace vernon::compiler
