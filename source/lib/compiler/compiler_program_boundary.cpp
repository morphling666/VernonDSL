#include "compiler_program_boundary.h"

#include "compiler_json.h"

#include <optional>

namespace vernon::compiler {
namespace {

ProgramBoundaryAccess forwardResourceAccess(const llvm::json::Array &graphs, int64_t storageId) {
    bool reads = false;
    bool writes = false;
    for (const llvm::json::Value &graphRow : graphs) {
        const llvm::json::Object *graph = graphRow.getAsObject();
        if (!graph || graph->getString("direction") != "forward")
            continue;
        const llvm::json::Array *nodes = graph->getArray("nodes");
        if (!nodes)
            continue;
        for (const llvm::json::Value &nodeRow : *nodes) {
            const llvm::json::Object *node = nodeRow.getAsObject();
            const llvm::json::Array *accesses = node ? node->getArray("accesses") : nullptr;
            if (!accesses)
                continue;
            for (const llvm::json::Value &accessRow : *accesses) {
                const llvm::json::Object *access = accessRow.getAsObject();
                if (!access || access->getInteger("storage") != storageId)
                    continue;
                const llvm::StringRef mode = access->getString("access").value_or("");
                const llvm::StringRef tag = access->getString("tag").value_or("");
                reads |= mode == "read" || mode == "read_write" || tag == "read" || tag == "attachment";
                writes |= mode == "write" || mode == "read_write" || tag == "initialize" || tag == "write" ||
                          tag == "attachment";
            }
        }
    }
    return writes ? (reads ? ProgramBoundaryAccess::ReadWrite : ProgramBoundaryAccess::Write)
                  : ProgramBoundaryAccess::Read;
}

bool appendProgramBoundaries(const llvm::json::Object &signature, llvm::StringRef field, ProgramBoundaryRole role,
                             ProgramBoundaryDirection direction, const llvm::json::Array &values,
                             const llvm::json::Array &storages, const llvm::json::Array &graphs,
                             ProgramBoundaryPlan &plan, std::string &error) {
    const llvm::json::Array *bindings = signature.getArray(field);
    if (!bindings) {
        error = "canonical ProgramABI requires all signature boundary arrays";
        return false;
    }
    for (const llvm::json::Value &bindingValue : *bindings) {
        const llvm::json::Object *binding = bindingValue.getAsObject();
        const std::optional<llvm::StringRef> path = binding ? binding->getString("path") : std::nullopt;
        const std::optional<int64_t> valueId = binding ? binding->getInteger("value") : std::nullopt;
        const llvm::json::Object *value =
            valueId && *valueId >= 0 ? findJsonObjectByIntegerId(values, *valueId) : nullptr;
        const std::optional<llvm::StringRef> logicalType = value ? value->getString("type") : std::nullopt;
        if (!path || path->empty() || !valueId || !value || !logicalType || logicalType->empty()) {
            error = "canonical ProgramABI received an invalid public signature binding";
            return false;
        }
        ProgramBoundarySlotPlan slot;
        slot.identity = {static_cast<int64_t>(plan.slots.size()), path->str(), role};
        slot.value = *valueId;
        slot.direction = direction;
        slot.logicalType = logicalType->str();
        slot.owner = {false, *valueId};
        if (const llvm::json::Array *shape = value->getArray("shape"))
            slot.outerShape = copyJsonArray(*shape);
        if (const llvm::json::Object *layout = value->getObject("value_layout"))
            slot.valueLayout = copyJsonObject(*layout);
        if (const std::optional<int64_t> storageId = value->getInteger("storage")) {
            if (*storageId < 0 || static_cast<size_t>(*storageId) >= storages.size()) {
                error = "canonical ProgramABI boundary references an invalid Storage";
                return false;
            }
            const llvm::json::Object *storage = storages[*storageId].getAsObject();
            const llvm::json::Object *descriptor = storage ? storage->getObject("descriptor") : nullptr;
            const std::optional<llvm::StringRef> tag = descriptor ? descriptor->getString("tag") : std::nullopt;
            if (!storage || !descriptor || !tag || (*tag != "buffer" && *tag != "image" && *tag != "opaque")) {
                error = "canonical ProgramABI boundary Storage has no typed descriptor";
                return false;
            }
            slot.category = *tag == "buffer"                    ? ProgramBoundaryCategory::StorageView
                            : *tag == "image"                   ? ProgramBoundaryCategory::Texture
                            : *logicalType == "!vernon.sampler" ? ProgramBoundaryCategory::Sampler
                                                                : ProgramBoundaryCategory::Value;
            if (slot.category == ProgramBoundaryCategory::Value) {
                error = "canonical ProgramABI boundary has an unsupported opaque resource type";
                return false;
            }
            slot.storage = *storageId;
            slot.owner = {true, *storageId};
            slot.storageDescriptor = copyJsonObject(*descriptor);
            slot.access = direction == ProgramBoundaryDirection::Input ? forwardResourceAccess(graphs, *storageId)
                                                                       : ProgramBoundaryAccess::Write;
        } else {
            slot.access = direction == ProgramBoundaryDirection::Input ? ProgramBoundaryAccess::Read
                                                                       : ProgramBoundaryAccess::Write;
        }
        if (direction == ProgramBoundaryDirection::Output)
            slot.publication = ProgramBoundaryPublication::CommitAfterSuccess;
        plan.slots.push_back(std::move(slot));
    }
    return true;
}

} // namespace

bool planProgramBoundaries(const llvm::json::Object &signature, const llvm::json::Array &values,
                           const llvm::json::Array &storages, const llvm::json::Array &graphs,
                           ProgramBoundaryPlan &plan, std::string &error) {
    plan = {};
    return appendProgramBoundaries(signature, "inputs", ProgramBoundaryRole::Input, ProgramBoundaryDirection::Input,
                                   values, storages, graphs, plan, error) &&
           appendProgramBoundaries(signature, "outputs", ProgramBoundaryRole::Output, ProgramBoundaryDirection::Output,
                                   values, storages, graphs, plan, error) &&
           appendProgramBoundaries(signature, "cotangents", ProgramBoundaryRole::Cotangent,
                                   ProgramBoundaryDirection::Input, values, storages, graphs, plan, error) &&
           appendProgramBoundaries(signature, "gradients", ProgramBoundaryRole::Gradient,
                                   ProgramBoundaryDirection::Output, values, storages, graphs, plan, error);
}

llvm::StringRef programBoundaryRoleName(ProgramBoundaryRole role) {
    switch (role) {
    case ProgramBoundaryRole::Input:
        return "input";
    case ProgramBoundaryRole::Output:
        return "output";
    case ProgramBoundaryRole::Cotangent:
        return "cotangent";
    case ProgramBoundaryRole::Gradient:
        return "gradient";
    }
    return "";
}

llvm::StringRef programBoundaryDirectionName(ProgramBoundaryDirection direction) {
    switch (direction) {
    case ProgramBoundaryDirection::Input:
        return "input";
    case ProgramBoundaryDirection::Output:
        return "output";
    }
    return "";
}

llvm::StringRef programBoundaryCategoryName(ProgramBoundaryCategory category) {
    switch (category) {
    case ProgramBoundaryCategory::Value:
        return "value";
    case ProgramBoundaryCategory::StorageView:
        return "storage_view";
    case ProgramBoundaryCategory::Texture:
        return "texture";
    case ProgramBoundaryCategory::Sampler:
        return "sampler";
    }
    return "";
}

llvm::StringRef programBoundaryAccessName(ProgramBoundaryAccess access) {
    switch (access) {
    case ProgramBoundaryAccess::Read:
        return "read";
    case ProgramBoundaryAccess::Write:
        return "write";
    case ProgramBoundaryAccess::ReadWrite:
        return "read_write";
    }
    return "";
}

llvm::StringRef programBoundaryPublicationName(ProgramBoundaryPublication publication) {
    switch (publication) {
    case ProgramBoundaryPublication::CommitAfterSuccess:
        return "commit_after_success";
    case ProgramBoundaryPublication::InPlace:
        return "in_place";
    }
    return "";
}

} // namespace vernon::compiler
