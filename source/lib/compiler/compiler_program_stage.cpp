#include "compiler_program_stage.h"

#include "compiler_json.h"
#include "compiler_program_finalization.h"
#include "compiler_program_storage.h"

#include "llvm/ADT/STLExtras.h"

namespace vernon::compiler {
namespace {

bool compatibleAbiShape(const llvm::json::Array *logical, const llvm::json::Array *physical) {
    const size_t logicalRank = logical ? logical->size() : 0;
    const size_t physicalRank = physical ? physical->size() : 0;
    if (logicalRank != physicalRank)
        return false;
    if (!logicalRank)
        return true;
    for (size_t index = 0; index < logicalRank; ++index) {
        const std::optional<int64_t> concrete = (*logical)[index].getAsInteger();
        const std::optional<int64_t> declared = (*physical)[index].getAsInteger();
        if (!concrete || !declared)
            return false;
        if (*concrete > 0 && *declared > 0 && *concrete != *declared)
            return false;
    }
    return true;
}

const llvm::json::Object *selectedTransportPlan(const llvm::json::Object &row) {
    const llvm::json::Object *layouts = row.getObject("physical_layouts");
    if (!layouts)
        return nullptr;
    const llvm::json::Object *fallback = nullptr;
    for (const auto &[name, value] : *layouts) {
        const llvm::json::Object *plan = value.getAsObject();
        if (!plan)
            continue;
        const llvm::StringRef kind = plan->getString("kind").value_or("");
        if (kind != "byte_transport" && kind != "native_uniform" && kind != "cpu_call" && kind != "kernel_parameter")
            continue;
        fallback = plan;
        if (const std::optional<llvm::StringRef> profile = plan->getString("profile");
            profile && *profile == llvm::StringRef(name))
            return plan;
    }
    return fallback;
}

} // namespace

bool programResourceAccessSatisfies(llvm::StringRef physical, llvm::StringRef logical) {
    if (logical == "read")
        return physical == "read" || physical == "read_write";
    if (logical == "write")
        return physical == "write" || physical == "read_write";
    return logical == "read_write" && physical == "read_write";
}

bool programAbiShapesCompatible(const llvm::json::Array *logical, const llvm::json::Array *physical) {
    return compatibleAbiShape(logical, physical);
}

std::optional<std::string> reflectedProgramResourceAccess(const llvm::json::Object &row) {
    if (const std::optional<llvm::StringRef> access = row.getString("vernon.access");
        access && isValidProgramResourceAccess(*access))
        return access->str();
    if (const std::optional<llvm::StringRef> access = row.getString("access");
        access && isValidProgramResourceAccess(*access))
        return access->str();
    return std::nullopt;
}

const llvm::json::Object *programEndpointLayout(const llvm::json::Object &row, bool resource) {
    if (resource)
        if (const llvm::json::Object *layout = row.getObject("element_layout"))
            return layout;
    return row.getObject("value_layout");
}

llvm::json::Object programValueCarrier(llvm::StringRef tag, int64_t slot, const llvm::json::Object &layout) {
    return llvm::json::Object{{"tag", tag.str()},
                              {"slot", slot},
                              {"byte_offset", int64_t{0}},
                              {"byte_size", layout.getInteger("byte_size").value_or(0)},
                              {"alignment", layout.getInteger("alignment").value_or(0)}};
}

llvm::json::Object compiledProgramEndpointAbi(const llvm::json::Object &row, llvm::StringRef module, int64_t index) {
    llvm::json::Object compiled{{"module", module.str()}, {"index", index}};
    if (const std::optional<llvm::StringRef> builtin =
            row.getString("vernon.builtin") ? row.getString("vernon.builtin") : row.getString("builtin"))
        compiled["builtin"] = builtin->str();
    if (const std::optional<llvm::StringRef> transport = row.getString("value_transport"))
        compiled["value_transport"] = transport->str();
    if (const std::optional<int64_t> set = row.getInteger("vernon.set"))
        compiled["set"] = *set;
    if (const std::optional<int64_t> binding = row.getInteger("vernon.binding"))
        compiled["binding"] = *binding;
    if (const llvm::json::Array *sampled = row.getArray("sampled_image_bindings"))
        compiled["sampled_image_bindings"] = copyJsonArray(*sampled);
    if (const llvm::json::Object *element = row.getObject("element_layout"))
        compiled["element_layout"] = copyJsonObject(*element);
    if (const llvm::json::Object *plan = selectedTransportPlan(row))
        compiled["interface_plan"] = copyJsonObject(*plan);
    if (const llvm::json::Object *layouts = row.getObject("physical_layouts"))
        if (const llvm::json::Object *host = layouts->getObject("host_value"))
            if (const std::optional<int64_t> offset = host->getInteger("frame_offset"); offset && *offset >= 0)
                compiled["packed_frame_offset"] = *offset;
    return compiled;
}

bool indexProgramNodeBindings(const llvm::json::Array &rawBindings, std::map<std::string, int64_t> &boundValues,
                              std::string &error) {
    boundValues.clear();
    for (const llvm::json::Value &bindingValue : rawBindings) {
        const llvm::json::Object *binding = bindingValue.getAsObject();
        const std::optional<llvm::StringRef> name = binding ? binding->getString("parameter") : std::nullopt;
        const std::optional<int64_t> value = binding ? binding->getInteger("value") : std::nullopt;
        if (!name || name->empty() || !value || !boundValues.emplace(name->str(), *value).second) {
            error = "canonical compute node has invalid or duplicate parameter bindings";
            return false;
        }
    }
    return true;
}

std::optional<int64_t> ensureProgramResourceAccess(int64_t valueId, bool attachment,
                                                   std::map<int64_t, ProgramLogicalResource> &resources,
                                                   std::map<int64_t, int64_t> &accessByValue,
                                                   llvm::json::Array &accesses) {
    if (const auto found = accessByValue.find(valueId); found != accessByValue.end())
        return found->second;
    const auto resource = resources.find(valueId);
    if (resource == resources.end() || (attachment && !resource->second.after))
        return std::nullopt;
    const int64_t accessIndex = static_cast<int64_t>(accesses.size());
    accessByValue[valueId] = accessIndex;
    if (attachment)
        accesses.emplace_back(llvm::json::Object{{"tag", "attachment"},
                                                 {"storage", resource->second.storage},
                                                 {"before", valueId},
                                                 {"after", *resource->second.after}});
    else if (resource->second.access == "read")
        accesses.emplace_back(
            llvm::json::Object{{"tag", "read"}, {"storage", resource->second.storage}, {"value", valueId}});
    else if (!resource->second.after)
        accesses.emplace_back(
            llvm::json::Object{{"tag", "initialize"}, {"storage", resource->second.storage}, {"after", valueId}});
    else
        accesses.emplace_back(llvm::json::Object{{"tag", "write"},
                                                 {"storage", resource->second.storage},
                                                 {"before", valueId},
                                                 {"after", *resource->second.after},
                                                 {"access", resource->second.access}});
    return accessIndex;
}

bool compatibleProgramBindingShape(llvm::StringRef role, llvm::StringRef carrier, const llvm::json::Array *logical,
                                   const llvm::json::Array *physical) {
    if (programAbiShapesCompatible(logical, physical))
        return true;
    if (role != "cotangent" || carrier != "invocation_linear" || !logical || !physical ||
        physical->size() != logical->size() + 1)
        return false;
    for (size_t index = 0; index < logical->size(); ++index) {
        const std::optional<int64_t> expected = (*logical)[index].getAsInteger();
        const std::optional<int64_t> actual = (*physical)[index + 1].getAsInteger();
        if (!expected || !actual || (*expected > 0 && *actual > 0 && *expected != *actual))
            return false;
    }
    return true;
}

std::optional<size_t> resolveProgramValueLeafIndex(const llvm::json::Object &layout, llvm::StringRef source,
                                                   llvm::StringRef parameter) {
    const llvm::json::Array *leaves = layout.getArray("leaves");
    if (!leaves || source.empty())
        return std::nullopt;
    for (auto [leafIndex, leafValue] : llvm::enumerate(*leaves)) {
        const llvm::json::Object *leaf = leafValue.getAsObject();
        const llvm::json::Array *path = leaf ? leaf->getArray("path") : nullptr;
        if (!path)
            continue;
        if (path->empty()) {
            if (parameter == source)
                return leafIndex;
            continue;
        }
        std::string canonical = source.str();
        for (const llvm::json::Value &component : *path) {
            canonical.push_back('.');
            if (const std::optional<llvm::StringRef> field = component.getAsString())
                canonical += *field;
            else if (const std::optional<int64_t> index = component.getAsInteger())
                canonical += std::to_string(*index);
            else
                return std::nullopt;
        }
        if (parameter == canonical)
            return leafIndex;
    }
    return std::nullopt;
}

} // namespace vernon::compiler
