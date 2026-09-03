#include "compiler_program_graphics.h"

#include "compiler_json.h"
#include "compiler_program_stage.h"

#include "llvm/ADT/Twine.h"

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
                                      const std::map<std::string, int64_t> &boundValues,
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
                    plan.implementationEndpoints.emplace_back(compiledProgramEndpointAbi(*row, role, endpointIndex));
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
                    continue;
                }
                const std::optional<llvm::StringRef> source = row->getString("vernon.source_name");
                auto logicalBinding = source ? boundValues.find(source->str()) : boundValues.end();
                if (!source || logicalBinding == boundValues.end()) {
                    error = "compiled graphics endpoint is not present in logical bindings";
                    return false;
                }
                coveredBindings.insert(source->str());
                const int64_t valueId = logicalBinding->second;
                const llvm::json::Object *logicalValue = findJsonObjectByIntegerId(rawValues, valueId);
                if (!logicalValue) {
                    error = "compiled graphics endpoint references an unknown logical value";
                    return false;
                }
                const llvm::StringRef kind = row->getString("kind").value_or("");
                const bool vertexAttribute = role == "vertex" && row->getArray("attribute_leaves");
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
                    if (kind == "image")
                        resourceLayout =
                            llvm::json::Object{{"tag", "image"},
                                               {"dimension", row->getString("dimension").value_or("").str()},
                                               {"format", row->getString("exact_storage_format").value_or("any").str()},
                                               {"sample_count", int64_t{0}},
                                               {"aspects", llvm::json::Array{"color"}}};
                    else if (kind == "sampler")
                        resourceLayout = llvm::json::Object{{"tag", "sampler"}};
                    else if (vertexAttribute) {
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
                    plan.implementationEndpoints.emplace_back(compiledProgramEndpointAbi(*row, role, endpointIndex));
                    if (vertexAttribute) {
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
                    plan.endpointBindings.emplace_back(llvm::json::Object{{"module", role.str()},
                                                                          {"interface", interfaceKind.str()},
                                                                          {"index", endpointIndex},
                                                                          {"tag", "value"},
                                                                          {"value", valueId}});
                    plan.implementationEndpoints.emplace_back(compiledProgramEndpointAbi(*row, role, endpointIndex));
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

bool buildCanonicalGraphicsOperation(
    const llvm::json::Object &node, const llvm::json::Array &rawValues, const llvm::json::Array &storages,
    const std::map<std::string, int64_t> &boundValues, std::map<int64_t, ProgramLogicalResource> &resources,
    const llvm::json::Array &fragmentOutputs, std::map<int64_t, int64_t> &accessByValue, llvm::json::Array &accesses,
    llvm::json::Array &attachmentConstraints, llvm::json::Object &operation, std::string &error) {
    const llvm::json::Array *nodeOperands = node.getArray("operands");
    const llvm::json::Array *nodeResults = node.getArray("results");
    const int64_t colorCount = node.getInteger("color_count").value_or(1);
    if (!nodeOperands || colorCount < 1 || static_cast<uint64_t>(colorCount) > nodeOperands->size() ||
        nodeOperands->size() < static_cast<uint64_t>(colorCount) ||
        (nodeResults && nodeResults->size() < static_cast<uint64_t>(colorCount))) {
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
        const llvm::json::Array *extent = descriptor ? descriptor->getArray("extent") : nullptr;
        if (!descriptor || descriptor->getString("tag") != "image" || !extent || extent->size() < 2) {
            error = ("canonical graphics " + role + " requires a concrete image Storage").str();
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
    const llvm::json::Array *extent = colorImages.front().descriptor->getArray("extent");
    attachmentConstraints.clear();
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
        for (const auto &[name, valueId] : boundValues) {
            (void)name;
            if (resources.count(valueId))
                if (const llvm::json::Object *value = findJsonObjectByIntegerId(rawValues, valueId))
                    if (const llvm::json::Array *shape = value->getArray("shape"); shape && !shape->empty())
                        return (*shape)[0].getAsInteger().value_or(1);
        }
        return int64_t{1};
    }();
    llvm::json::Array colors;
    for (size_t index = 0; index < colorImages.size(); ++index)
        colors.emplace_back(llvm::json::Object{{"location", static_cast<int64_t>(index)},
                                               {"access", colorImages[index].access},
                                               {"load", llvm::json::Object{{"tag", "discard"}}},
                                               {"store", "store"}});
    llvm::json::Value depthStencil = nullptr;
    if (depthImage)
        depthStencil = llvm::json::Object{{"access", depthImage->access},
                                          {"load", llvm::json::Object{{"tag", "discard"}}},
                                          {"store", "store"},
                                          {"depth", true},
                                          {"stencil", false}};
    operation = llvm::json::Object{
        {"tag", "graphics"},
        {"attachments",
         llvm::json::Object{{"colors", std::move(colors)},
                            {"depth_stencil", std::move(depthStencil)},
                            {"render_area", llvm::json::Object{{"x", int64_t{0}},
                                                               {"y", int64_t{0}},
                                                               {"width", (*extent)[0].getAsInteger().value_or(1)},
                                                               {"height", (*extent)[1].getAsInteger().value_or(1)}}},
                            {"layer_count", int64_t{1}}}},
        {"state",
         llvm::json::Object{
             {"raster",
              llvm::json::Object{{"front_face", "counter_clockwise"}, {"cull_mode", "none"}, {"fill_mode", "fill"}}},
             {"depth_stencil", llvm::json::Object{{"depth_test", hasDepth},
                                                  {"depth_write", hasDepth},
                                                  {"depth_compare", hasDepth ? "less" : "always"},
                                                  {"stencil_test", false}}},
             {"multisample", llvm::json::Object{{"sample_mask", int64_t{4294967295ULL}}, {"alpha_to_coverage", false}}},
             {"blend", llvm::json::Array()},
             {"viewport", nullptr},
             {"scissor", nullptr}}},
        {"draw", llvm::json::Object{{"tag", "direct"}, {"vertex_count", vertexCount}, {"instance_count", int64_t{1}}}},
    };
    return true;
}

} // namespace vernon::compiler
