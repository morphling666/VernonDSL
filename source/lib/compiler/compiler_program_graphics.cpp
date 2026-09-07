#include "compiler_program_graphics.h"

#include "compiler_json.h"
#include "compiler_program_stage.h"

#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace vernon::compiler {
namespace {

llvm::json::Array programGraphicsAttributeCellShape(const llvm::json::Object &row) {
    if (const llvm::json::Array *shape = row.getArray("shape"))
        return copyJsonArray(*shape);
    const llvm::json::Object *valueLayout = row.getObject("value_layout");
    const llvm::json::Array *leaves = valueLayout ? valueLayout->getArray("leaves") : nullptr;
    if (!leaves || leaves->size() != 1)
        return {};
    const llvm::json::Object *leaf = (*leaves)[0].getAsObject();
    const llvm::json::Array *path = leaf ? leaf->getArray("path") : nullptr;
    const llvm::json::Array *leafShape = leaf ? leaf->getArray("shape") : nullptr;
    if (!leafShape || (path && !path->empty()))
        return {};
    return copyJsonArray(*leafShape);
}

bool appendProgramGraphicsLinkage(llvm::json::Array &rows, const llvm::json::Object &row) {
    llvm::json::Object linkage;
    if (std::optional<int64_t> location = row.getInteger("vernon.location"))
        linkage["location"] = *location;
    else if (std::optional<llvm::StringRef> builtin = row.getString("vernon.builtin"))
        linkage["builtin"] = builtin->str();
    else
        return false;
    linkage["type"] = row.getString("type").value_or("").str();
    if (linkage.get("location"))
        linkage["interpolation"] = row.getString("vernon.interpolation").value_or("smooth").str();
    rows.emplace_back(std::move(linkage));
    return true;
}

std::string programGraphicsVertexFormat(llvm::StringRef dtype, int64_t components) {
    const llvm::StringRef suffix = dtype == "f32" ? "float" : dtype == "u32" ? "uint" : "sint";
    if (components <= 1)
        return ("r32_" + suffix).str();
    if (components == 2)
        return ("rg32_" + suffix).str();
    if (components == 3)
        return ("rgb32_" + suffix).str();
    return ("rgba32_" + suffix).str();
}

} // namespace

bool buildCanonicalGraphicsInterfaces(const llvm::json::Object &compiledReflection, const llvm::json::Array &rawValues,
                                      const llvm::json::Array &storages, const ProgramNodeBindingIndex &boundValues,
                                      std::map<int64_t, ProgramLogicalResource> &resources,
                                      ProgramGraphicsInterfacePlan &plan, std::string &error) {
    const llvm::json::Array *entries = compiledReflection.getArray("entries");
    if (!entries || entries->empty()) {
        error = "compiled graphics pipeline has no reflected modules";
        return false;
    }
    std::map<std::string, const llvm::json::Object *> modules;
    for (const llvm::json::Value &entryValue : *entries) {
        const llvm::json::Object *entry = entryValue.getAsObject();
        const std::optional<llvm::StringRef> role = entry ? entry->getString("stage") : std::nullopt;
        if (!role || (*role != "vertex" && *role != "fragment") || !modules.emplace(role->str(), entry).second) {
            error = "compiled graphics pipeline has invalid or duplicate module roles";
            return false;
        }
    }
    if (!modules.count("vertex")) {
        error = "compiled graphics pipeline requires a vertex module";
        return false;
    }

    int64_t nextSlot = 0;
    std::set<std::string> coveredBindings;
    for (llvm::StringRef role : {llvm::StringRef("vertex"), llvm::StringRef("fragment")}) {
        auto moduleIt = modules.find(role.str());
        if (moduleIt == modules.end())
            continue;
        const llvm::json::Object &entry = *moduleIt->second;
        const auto appendRows = [&](const llvm::json::Array *rows, llvm::StringRef interfaceKind) -> bool {
            if (!rows)
                return true;
            for (size_t ordinal = 0; ordinal < rows->size(); ++ordinal) {
                const llvm::json::Object *row = (*rows)[ordinal].getAsObject();
                if (!row) {
                    error = "compiled graphics endpoint is not an object";
                    return false;
                }
                const int64_t endpointIndex = row->getInteger("index").value_or(ordinal);
                const bool moduleLinkage = (role == "vertex" && interfaceKind == "result") ||
                                           (role == "fragment" && interfaceKind == "argument" &&
                                            (row->getInteger("vernon.location") || row->getString("vernon.builtin")));
                if (moduleLinkage) {
                    if (!appendProgramGraphicsLinkage(
                            (role == "vertex" ? plan.vertexOutputs.json() : plan.fragmentInputs.json()), *row)) {
                        error = "compiled graphics linkage has no location or builtin";
                        return false;
                    }
                    continue;
                }
                if (role == "fragment" && interfaceKind == "result") {
                    const std::optional<int64_t> location = row->getInteger("vernon.location");
                    if (!location) {
                        error = "compiled fragment output has no location";
                        return false;
                    }
                    plan.fragmentOutputs.emplace_back(llvm::json::Object{
                        {"location", *location}, {"type", row->getString("type").value_or("").str()}});
                    continue;
                }
                const std::optional<llvm::StringRef> builtin =
                    row->getString("vernon.builtin") ? row->getString("vernon.builtin") : row->getString("builtin");
                const std::optional<llvm::StringRef> implicit = row->getString("vernon.implicit");
                if (implicit == "resolution") {
                    const llvm::json::Object *layout = programEndpointLayout(*row, false);
                    if (!layout) {
                        error = "compiled graphics resolution endpoint has no canonical layout";
                        return false;
                    }
                    llvm::json::Array abiBindings;
                    abiBindings.emplace_back(llvm::json::Object{
                        {"semantic", "value"}, {"carrier", programValueCarrier("value_slot", nextSlot++, *layout)}});
                    plan.endpoints.emplace_back(llvm::json::Object{
                        {"tag", "value"},
                        {"module", role.str()},
                        {"interface", "system_value"},
                        {"index", endpointIndex},
                        {"type", row->getString("type").value_or("").str()},
                        {"layout_hash", layout->getString("layout_hash").value_or("").str()},
                        {"transport", "by_value"},
                        {"access", "read"},
                        {"builtin", "resolution"},
                        {"abi", llvm::json::Object{{"bindings", std::move(abiBindings)}}},
                    });
                    plan.implementationEndpoints.emplace_back(
                        compiledProgramEndpointAbi(*row, role, "system_value", endpointIndex));
                    continue;
                }
                if (builtin || implicit) {
                    plan.endpoints.emplace_back(llvm::json::Object{
                        {"tag", "system"},
                        {"module", role.str()},
                        {"interface", interfaceKind.str()},
                        {"index", endpointIndex},
                        {"semantic", builtin ? builtin->str() : implicit->str()},
                        {"abi", llvm::json::Object{{"bindings", llvm::json::Array()}}},
                    });
                    /* An implicit sampler carries no portable ABI binding, because a caller never supplies one.
                     * The target still has to bind it, so it needs the compiled row's descriptor set, binding, and
                     * the sampled images it pairs with. */
                    if (implicit == "sampler")
                        plan.implementationEndpoints.emplace_back(
                            compiledProgramEndpointAbi(*row, role, interfaceKind, endpointIndex));
                    continue;
                }
                const std::optional<llvm::StringRef> source = row->getString("vernon.source_name");
                auto logicalBinding = source ? boundValues.find(source->str()) : boundValues.end();
                if (!source || logicalBinding == boundValues.end()) {
                    error = "compiled graphics endpoint is not present in logical bindings";
                    return false;
                }
                coveredBindings.insert(source->str());
                if (logicalBinding->second.size() != 1) {
                    error = "graphics physical endpoints require exactly one canonical projection";
                    return false;
                }
                const int64_t valueId = logicalBinding->second.front().value;
                const llvm::json::Object *logicalValue = findJsonObjectByIntegerId(rawValues, valueId);
                if (!logicalValue) {
                    error = "compiled graphics endpoint references an unknown logical value";
                    return false;
                }
                const llvm::StringRef kind = row->getString("kind").value_or("");
                const bool vertexAttribute = role == "vertex" && row->getArray("attribute_leaves");
                const llvm::StringRef logicalType = logicalValue->getString("type").value_or("");
                if ((kind == "image" && !isProgramTextureType(logicalType)) ||
                    (kind == "sampler" && !isProgramSamplerType(logicalType))) {
                    error = "compiled graphics resource kind does not match its Program Value";
                    return false;
                }
                const ProgramEndpointExpectation expectation{
                    logicalValue->getString("dtype"),
                    logicalValue->getArray("shape"),
                    nullptr,
                    logicalValue->getObject("value_layout"),
                    {},
                    {},
                    vertexAttribute,
                    kind == "tensor_value",
                };
                if (!verifyProgramEndpointAbi(expectation, *row, error))
                    return false;
                const bool resourceEndpoint =
                    vertexAttribute || kind == "image" || kind == "sampler" || resources.count(valueId);
                llvm::json::Array abiBindings;
                if (resourceEndpoint) {
                    abiBindings.emplace_back(llvm::json::Object{
                        {"semantic", kind == "sampler" ? "sampler" : "resource"},
                        {"carrier", llvm::json::Object{{"tag", "resource_slot"}, {"slot", nextSlot++}}}});
                    const std::optional<int64_t> accessIndex = ensureProgramResourceAccess(
                        valueId, false, resources, plan.accessByValue, plan.accesses.json());
                    const llvm::json::Object *layout = programEndpointLayout(*row, true);
                    if (!accessIndex || (!layout && kind != "image" && kind != "sampler")) {
                        error = "compiled graphics resource endpoint has no logical access or layout";
                        return false;
                    }
                    llvm::json::Object resourceLayout;
                    if (kind == "image") {
                        const ProgramLogicalResource &resource = resources.at(valueId);
                        const llvm::json::Object *storage =
                            resource.storage >= 0 && static_cast<size_t>(resource.storage) < storages.size()
                                ? storages[static_cast<size_t>(resource.storage)].getAsObject()
                                : nullptr;
                        const llvm::json::Object *descriptor = storage ? storage->getObject("descriptor") : nullptr;
                        const llvm::json::Array *aspects = descriptor ? descriptor->getArray("aspects") : nullptr;
                        if (!descriptor || descriptor->getString("tag") != "image" || !descriptor->getArray("usage") ||
                            !aspects) {
                            error = "compiled graphics image endpoint has no canonical image Storage";
                            return false;
                        }
                        plan.storageUsageRequirements[resource.storage].insert("sampled");
                        resourceLayout =
                            llvm::json::Object{{"tag", "image"},
                                               {"dimension", descriptor->getString("dimension").value_or("").str()},
                                               {"format", descriptor->getString("format").value_or("any").str()},
                                               {"sample_count", descriptor->getInteger("sample_count").value_or(0)},
                                               {"aspects", copyJsonArray(*aspects)}};
                    } else if (kind == "sampler")
                        resourceLayout = llvm::json::Object{{"tag", "sampler"}};
                    else if (vertexAttribute) {
                        const ProgramLogicalResource &resource = resources.at(valueId);
                        const llvm::json::Object *storage =
                            resource.storage >= 0 && static_cast<size_t>(resource.storage) < storages.size()
                                ? storages[static_cast<size_t>(resource.storage)].getAsObject()
                                : nullptr;
                        const llvm::json::Object *descriptor = storage ? storage->getObject("descriptor") : nullptr;
                        if (!descriptor || descriptor->getString("tag") != "buffer" || !descriptor->getArray("usage")) {
                            error = "compiled vertex attribute has no canonical buffer Storage";
                            return false;
                        }
                        plan.storageUsageRequirements[resource.storage].insert("vertex");
                        llvm::json::Array cellShape = programGraphicsAttributeCellShape(*row);
                        const int64_t viewRank = static_cast<int64_t>(cellShape.size());
                        resourceLayout = llvm::json::Object{
                            {"tag", "buffer"},
                            {"view_rank", viewRank},
                            {"shape", std::move(cellShape)},
                            {"descriptor", false},
                            {"element_layout_hash", layout->getString("layout_hash").value_or("").str()},
                            {"minimum_alignment", layout->getInteger("alignment").value_or(0)}};
                    } else
                        resourceLayout = llvm::json::Object{
                            {"tag", "buffer"},
                            {"view_rank", logicalValue->getArray("shape")
                                              ? static_cast<int64_t>(logicalValue->getArray("shape")->size())
                                              : int64_t{0}},
                            {"descriptor", true},
                            {"element_layout_hash", layout->getString("layout_hash").value_or("").str()},
                            {"minimum_alignment", layout->getInteger("alignment").value_or(0)}};
                    plan.endpoints.emplace_back(llvm::json::Object{
                        {"tag", "resource"},
                        {"module", role.str()},
                        {"interface", interfaceKind.str()},
                        {"index", endpointIndex},
                        {"role", vertexAttribute ? "vertex"
                                                 : row->getString("binding_role")
                                                       .value_or(kind == "sampler" ? "sampler" : "storage")
                                                       .str()},
                        {"type", logicalValue->getString("type").value_or("").str()},
                        {"layout", std::move(resourceLayout)},
                        {"address_space", row->getString("address_space").value_or("device").str()},
                        {"transport", "resource_handle"},
                        {"access", resources.at(valueId).access},
                        {"abi", llvm::json::Object{{"bindings", std::move(abiBindings)}}},
                    });
                    plan.endpointBindings.emplace_back(llvm::json::Object{{"module", role.str()},
                                                                          {"interface", interfaceKind.str()},
                                                                          {"index", endpointIndex},
                                                                          {"tag", "resource"},
                                                                          {"access", *accessIndex}});
                    plan.implementationEndpoints.emplace_back(
                        compiledProgramEndpointAbi(*row, role, interfaceKind, endpointIndex));
                    if (vertexAttribute) {
                        if (!plan.vertexCountValue)
                            plan.vertexCountValue = valueId;
                        const int64_t divisor = row->getInteger("vernon.instance_divisor").value_or(0);
                        if (divisor < 0) {
                            error = "compiled graphics vertex input has a negative instance divisor";
                            return false;
                        }
                        for (const llvm::json::Value &leafValue : *row->getArray("attribute_leaves"))
                            if (const llvm::json::Object *leaf = leafValue.getAsObject())
                                plan.vertexInputs.emplace_back(llvm::json::Object{
                                    {"location", leaf->getInteger("location").value_or(0)},
                                    {"endpoint_index", endpointIndex},
                                    {"format",
                                     programGraphicsVertexFormat(leaf->getString("dtype").value_or("f32"),
                                                                 leaf->getInteger("component_count").value_or(1))},
                                    {"byte_offset", leaf->getInteger("byte_offset").value_or(0)},
                                    {"byte_stride", layout->getInteger("byte_size").value_or(0)},
                                    {"step", divisor > 0 ? "instance" : "vertex"},
                                    {"divisor", divisor},
                                });
                    }
                } else {
                    const llvm::json::Object *layout = programEndpointLayout(*row, false);
                    if (!layout) {
                        error = "compiled graphics value endpoint has no canonical layout";
                        return false;
                    }
                    abiBindings.emplace_back(llvm::json::Object{
                        {"semantic", "value"}, {"carrier", programValueCarrier("value_slot", nextSlot++, *layout)}});
                    plan.endpoints.emplace_back(llvm::json::Object{
                        {"tag", "value"},
                        {"module", role.str()},
                        {"interface", interfaceKind.str()},
                        {"index", endpointIndex},
                        {"type", logicalValue->getString("type").value_or("").str()},
                        {"layout_hash", layout->getString("layout_hash").value_or("").str()},
                        {"transport", "by_value"},
                        {"access", "read"},
                        {"abi", llvm::json::Object{{"bindings", std::move(abiBindings)}}},
                    });
                    llvm::json::Array projections;
                    projections.emplace_back(
                        llvm::json::Object{{"value", valueId},
                                           {"physical_leaf", int64_t{0}},
                                           {"direction", interfaceKind == "result" ? "result" : "input"}});
                    plan.endpointBindings.emplace_back(llvm::json::Object{{"module", role.str()},
                                                                          {"interface", interfaceKind.str()},
                                                                          {"index", endpointIndex},
                                                                          {"tag", "value"},
                                                                          {"projections", std::move(projections)}});
                    plan.implementationEndpoints.emplace_back(
                        compiledProgramEndpointAbi(*row, role, interfaceKind, endpointIndex));
                }
            }
            return true;
        };
        if (!appendRows(entry.getArray("arguments"), "argument") || !appendRows(entry.getArray("results"), "result"))
            return false;
    }
    if (coveredBindings.size() != boundValues.size()) {
        error = "compiled graphics ABI does not exactly cover logical bindings";
        return false;
    }
    return true;
}

bool buildCanonicalGraphicsOperation(const llvm::json::Object &node, const llvm::json::Array &rawValues,
                                     const llvm::json::Array &storages, const ProgramNodeBindingIndex &boundValues,
                                     std::map<int64_t, ProgramLogicalResource> &resources,
                                     const llvm::json::Array &fragmentOutputs, std::optional<int64_t> vertexCountValue,
                                     std::map<int64_t, int64_t> &accessByValue, llvm::json::Array &accesses,
                                     llvm::json::Array &attachmentConstraints, llvm::json::Object &operation,
                                     std::string &error) {
    const llvm::json::Array *nodeOperands = node.getArray("operands");
    const llvm::json::Array *nodeResults = node.getArray("results");
    const int64_t colorCount = node.getInteger("color_count").value_or(1);
    if (!nodeOperands || !nodeResults || colorCount < 0 || static_cast<uint64_t>(colorCount) > nodeOperands->size() ||
        (nodeResults->size() != static_cast<uint64_t>(colorCount) &&
         nodeResults->size() != static_cast<uint64_t>(colorCount + 1)) ||
        nodeResults->empty()) {
        error = "canonical graphics node has an invalid color attachment count";
        return false;
    }
    const bool hasDepth = nodeResults && nodeResults->size() == static_cast<uint64_t>(colorCount) + 1;
    if (nodeOperands->size() < static_cast<size_t>(colorCount + (hasDepth ? 1 : 0))) {
        error = "canonical graphics node is missing attachment operands";
        return false;
    }
    struct AttachmentImage {
        int64_t value{};
        int64_t access{};
        const llvm::json::Object *descriptor{};
        std::string format;
    };
    const auto attachmentImage = [&](size_t operandIndex, llvm::StringRef role) -> std::optional<AttachmentImage> {
        const std::optional<int64_t> valueId = (*nodeOperands)[operandIndex].getAsInteger();
        const std::optional<int64_t> access =
            valueId ? ensureProgramResourceAccess(*valueId, true, resources, accessByValue, accesses) : std::nullopt;
        if (!valueId || !access) {
            error = ("canonical graphics " + role + " has no attachment access").str();
            return std::nullopt;
        }
        const ProgramLogicalResource &resource = resources.at(*valueId);
        const llvm::json::Object *storage =
            resource.storage >= 0 && static_cast<size_t>(resource.storage) < storages.size()
                ? storages[static_cast<size_t>(resource.storage)].getAsObject()
                : nullptr;
        const llvm::json::Object *descriptor = storage ? storage->getObject("descriptor") : nullptr;
        if (!descriptor || descriptor->getString("tag") != "image") {
            error = ("canonical graphics " + role + " requires an image Storage").str();
            return std::nullopt;
        }
        return AttachmentImage{*valueId, *access, descriptor, descriptor->getString("format").value_or("").str()};
    };
    std::vector<AttachmentImage> colorImages;
    for (int64_t index = 0; index < colorCount; ++index) {
        std::optional<AttachmentImage> image = attachmentImage(static_cast<size_t>(index), "color attachment");
        if (!image)
            return false;
        colorImages.push_back(*image);
    }
    std::optional<AttachmentImage> depthImage;
    if (hasDepth) {
        depthImage = attachmentImage(static_cast<size_t>(colorCount), "depth attachment");
        if (!depthImage)
            return false;
    }
    attachmentConstraints.clear();
    if (fragmentOutputs.size() != colorImages.size()) {
        error = "compiled fragment outputs must exactly match the color attachments";
        return false;
    }
    for (const llvm::json::Value &outputValue : fragmentOutputs) {
        const llvm::json::Object *output = outputValue.getAsObject();
        if (!output)
            continue;
        const int64_t location = output->getInteger("location").value_or(0);
        if (location < 0 || static_cast<uint64_t>(location) >= colorImages.size()) {
            error = "compiled fragment output has no matching color attachment";
            return false;
        }
        const AttachmentImage &color = colorImages[static_cast<size_t>(location)];
        attachmentConstraints.emplace_back(llvm::json::Object{
            {"location", location},
            {"formats", llvm::json::Array{color.format}},
            {"sample_counts", llvm::json::Array{color.descriptor->getInteger("sample_count").value_or(1)}},
            {"aspects", llvm::json::Array{"color"}}});
    }
    if (depthImage)
        attachmentConstraints.emplace_back(llvm::json::Object{
            {"location", int64_t{-1}},
            {"formats", llvm::json::Array{depthImage->format}},
            {"sample_counts", llvm::json::Array{depthImage->descriptor->getInteger("sample_count").value_or(1)}},
            {"aspects", llvm::json::Array{"depth"}}});
    const int64_t vertexCount = [&]() {
        if (vertexCountValue)
            if (const llvm::json::Object *value = findJsonObjectByIntegerId(rawValues, *vertexCountValue))
                if (const llvm::json::Array *shape = value->getArray("shape"); shape && !shape->empty()) {
                    const int64_t extent = (*shape)[0].getAsInteger().value_or(0);
                    return std::max(int64_t{0}, extent);
                }
        return int64_t{1};
    }();
    const llvm::json::Array *slots = node.getArray("control_slots");
    const llvm::json::Object *sourceState = node.getObject("graphics_state");
    if (!slots || slots->size() != 3 || !(*slots)[0].getAsInteger() || !(*slots)[1].getAsInteger() ||
        !(*slots)[2].getAsInteger() || !sourceState) {
        error = "canonical graphics node requires typed pipeline state and three Program control slots";
        return false;
    }
    llvm::json::Object pipelineState = copyJsonObject(*sourceState);
    pipelineState["topology"] = node.getString("topology").value_or("triangle_list").str();
    const llvm::json::Array *configuredBlends = sourceState->getArray("color_blends");
    if (!configuredBlends) {
        error = "graphics pipeline state has no color_blends array";
        return false;
    }
    std::map<int64_t, const llvm::json::Object *> blendByLocation;
    for (const llvm::json::Value &entryValue : *configuredBlends) {
        const llvm::json::Array *entry = entryValue.getAsArray();
        const std::optional<int64_t> location = entry && entry->size() == 2 ? (*entry)[0].getAsInteger() : std::nullopt;
        const llvm::json::Object *blend = entry && entry->size() == 2 ? (*entry)[1].getAsObject() : nullptr;
        if (!location || *location < 0 || static_cast<size_t>(*location) >= colorImages.size() || !blend ||
            !blendByLocation.emplace(*location, blend).second) {
            error = "graphics pipeline state has an invalid color blend location";
            return false;
        }
    }
    llvm::json::Array canonicalBlends;
    for (size_t location = 0; location < colorImages.size(); ++location) {
        const auto configured = blendByLocation.find(static_cast<int64_t>(location));
        llvm::json::Object blend =
            configured != blendByLocation.end()
                ? copyJsonObject(*configured->second)
                : llvm::json::Object{
                      {"enabled", false},         {"source_color", "ONE"},    {"destination_color", "ZERO"},
                      {"color_operation", "ADD"}, {"source_alpha", "ONE"},    {"destination_alpha", "ZERO"},
                      {"alpha_operation", "ADD"}, {"write_mask", int64_t{15}}};
        canonicalBlends.emplace_back(
            llvm::json::Array{static_cast<int64_t>(location), llvm::json::Value(std::move(blend))});
    }
    pipelineState["color_blends"] = std::move(canonicalBlends);
    llvm::json::Array colors;
    for (size_t index = 0; index < colorImages.size(); ++index)
        colors.emplace_back(llvm::json::Object{
            {"location", static_cast<int64_t>(index)},
            {"access", colorImages[index].access},
            {"formats", llvm::json::Array{colorImages[index].format}},
            {"sample_counts",
             llvm::json::Array{colorImages[index].descriptor->getInteger("sample_count").value_or(1)}}});
    llvm::json::Value depthStencil = nullptr;
    if (depthImage)
        depthStencil = llvm::json::Object{
            {"access", depthImage->access},
            {"formats", llvm::json::Array{depthImage->format}},
            {"sample_counts", llvm::json::Array{depthImage->descriptor->getInteger("sample_count").value_or(1)}},
            {"aspects", llvm::json::Array{"depth"}}};
    operation = llvm::json::Object{
        {"tag", "graphics"},
        {"pipeline_state", std::move(pipelineState)},
        {"render_pass", llvm::json::Object{{"control", *(*slots)[0].getAsInteger()},
                                           {"colors", std::move(colors)},
                                           {"depth_stencil", std::move(depthStencil)}}},
        {"draw", llvm::json::Object{{"control", *(*slots)[1].getAsInteger()},
                                    {"default", llvm::json::Object{{"tag", "direct"},
                                                                   {"vertex_count", vertexCount},
                                                                   {"instance_count", int64_t{1}}}}}},
        {"dynamic_state", llvm::json::Object{{"control", *(*slots)[2].getAsInteger()}}},
    };
    return true;
}

bool buildCanonicalGraphicsStageContract(const llvm::json::Object &node, llvm::json::Array requiredFeatures,
                                         ProgramGraphicsInterfacePlan &plan, llvm::json::Array attachmentConstraints,
                                         llvm::json::Object &stageContract, std::string &error) {
    std::vector<int64_t> slots;
    for (const llvm::json::Value &endpointValue : plan.endpoints.json()) {
        const llvm::json::Object *endpoint = endpointValue.getAsObject();
        const llvm::json::Object *abi = endpoint ? endpoint->getObject("abi") : nullptr;
        const llvm::json::Array *bindings = abi ? abi->getArray("bindings") : nullptr;
        if (!bindings)
            return error = "compiled graphics endpoint has no portable ABI bindings", false;
        for (const llvm::json::Value &bindingValue : *bindings) {
            const llvm::json::Object *binding = bindingValue.getAsObject();
            const llvm::json::Object *carrier = binding ? binding->getObject("carrier") : nullptr;
            const std::optional<int64_t> slot = carrier ? carrier->getInteger("slot") : std::nullopt;
            if (slot)
                slots.push_back(*slot);
        }
    }
    std::sort(slots.begin(), slots.end());
    for (size_t index = 0; index < slots.size(); ++index)
        if (slots[index] != static_cast<int64_t>(index))
            return error = "compiled graphics portable ABI slots must be contiguous and unique", false;

    llvm::json::Object graphics{
        {"topology", node.getString("topology").value_or("triangle_list").str()},
        {"vertex_inputs", plan.vertexInputs.take()},
        {"fragment_outputs", copyJsonArray(plan.fragmentOutputs.json())},
        {"linkage", llvm::json::Object{{"vertex_outputs", plan.vertexOutputs.take()},
                                       {"fragment_inputs", plan.fragmentInputs.take()}}},
        {"attachment_constraints", std::move(attachmentConstraints)},
        {"index_formats", llvm::json::Array{"u16", "u32"}},
        {"capabilities", llvm::json::Array{"direct_draw"}},
    };
    llvm::json::Object reflection{{"required_features", std::move(requiredFeatures)},
                                  {"endpoints", plan.endpoints.take()},
                                  {"graphics", std::move(graphics)}};
    stageContract = llvm::json::Object{{"operation", "graphics"}, {"reflection", std::move(reflection)}};
    return true;
}

} // namespace vernon::compiler
