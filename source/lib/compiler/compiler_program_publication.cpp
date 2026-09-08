#include "compiler_program_publication.h"

#include "compiler_json.h"

namespace vernon::compiler {

llvm::json::Array serializeProgramBoundarySlots(const ProgramBoundaryPlan &boundaries) {
    llvm::json::Array result;
    for (const ProgramBoundarySlotPlan &slot : boundaries.slots) {
        llvm::json::Object serialized{
            {"id", slot.identity.slot},
            {"path", slot.identity.path},
            {"value", slot.value},
            {"role", programBoundaryRoleName(slot.identity.role)},
            {"direction", programBoundaryDirectionName(slot.direction)},
            {"category", programBoundaryCategoryName(slot.category)},
            {"access", programBoundaryAccessName(slot.access)},
            {"logical_type", slot.logicalType},
            {"outer_shape", copyJsonArray(slot.outerShape)},
            {"alias_owner", (slot.owner.storage ? "storage:" : "value:") + std::to_string(slot.owner.id)},
        };
        if (slot.valueLayout)
            serialized["value_layout"] = copyJsonObject(*slot.valueLayout);
        if (slot.storage) {
            serialized["storage_id"] = *slot.storage;
            serialized["storage_descriptor"] = copyJsonObject(*slot.storageDescriptor);
        }
        if (slot.publication)
            serialized["publication"] = programBoundaryPublicationName(*slot.publication);
        result.emplace_back(std::move(serialized));
    }
    return result;
}

} // namespace vernon::compiler
