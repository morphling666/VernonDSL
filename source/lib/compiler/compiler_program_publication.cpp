#include "compiler_program_publication.h"

#include "compiler_json.h"

#include <algorithm>
#include <string>

namespace vernon::compiler {
namespace {

llvm::StringRef roleName(ProgramBoundaryRole role) {
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

llvm::StringRef categoryName(ProgramBoundaryCategory category) {
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

llvm::StringRef accessName(ProgramBoundaryAccess access) {
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

} // namespace

ProgramPublicationPlan planProgramPublications(const ProgramBoundaryPlan &boundaries) {
    ProgramPublicationPlan plan;
    for (const ProgramBoundarySlotPlan &slot : boundaries.slots)
        if (slot.direction == ProgramBoundaryDirection::Output)
            plan.targets.push_back(
                {slot.identity.slot, slot.value, slot.owner, ProgramPublicationDisposition::CommitAfterSuccess});
    return plan;
}

llvm::json::Array serializeProgramBoundarySlots(const ProgramBoundaryPlan &boundaries,
                                                const ProgramPublicationPlan &publication) {
    llvm::json::Array result;
    for (const ProgramBoundarySlotPlan &slot : boundaries.slots) {
        llvm::json::Object row{
            {"id", slot.identity.slot},
            {"path", slot.identity.path},
            {"value", slot.value},
            {"role", roleName(slot.identity.role)},
            {"direction", slot.direction == ProgramBoundaryDirection::Input ? "input" : "output"},
            {"category", categoryName(slot.category)},
            {"access", accessName(slot.access)},
            {"logical_type", slot.logicalType},
            {"alias_owner", std::string(slot.owner.storage ? "storage:" : "value:") + std::to_string(slot.owner.id)},
            {"outer_shape", copyJsonArray(slot.outerShape)},
        };
        const auto target = std::find_if(
            publication.targets.begin(), publication.targets.end(),
            [&](const ProgramPublicationTargetPlan &candidate) { return candidate.slot == slot.identity.slot; });
        if (target != publication.targets.end())
            row["publication"] = "commit_after_success";
        if (slot.storage) {
            row["storage_id"] = *slot.storage;
            row["storage_descriptor"] = copyJsonObject(*slot.storageDescriptor);
        }
        if (slot.valueLayout)
            row["value_layout"] = copyJsonObject(*slot.valueLayout);
        result.emplace_back(std::move(row));
    }
    return result;
}

} // namespace vernon::compiler
