#include "compiler_program_target_aggregation.h"

#include "compiler_json.h"

namespace vernon::compiler {
namespace {

bool validMetalResourceSlot(const llvm::json::Object &slot) {
    const std::optional<int64_t> descriptorSet = slot.getInteger("set");
    const std::optional<int64_t> binding = slot.getInteger("binding");
    const std::optional<int64_t> argumentBuffer = slot.getInteger("argument_buffer_index");
    const std::optional<int64_t> member = slot.getInteger("member_id");
    const std::optional<int64_t> directBuffer = slot.getInteger("direct_buffer_index");
    return slot.size() == 10 && slot.getString("entry_point") && slot.getString("stage") && slot.getString("kind") &&
           slot.getString("name") && descriptorSet && *descriptorSet >= 0 && binding && *binding >= 0 &&
           argumentBuffer && *argumentBuffer >= 0 && member && *member >= 0 && directBuffer && *directBuffer >= 0 &&
           slot.getInteger("count").value_or(0) > 0;
}

} // namespace

bool appendProgramTargetModuleMetadata(CanonicalProgramStage &stage, const llvm::json::Object &reflection,
                                       std::string &error) {
    const llvm::json::Object *target = reflection.getObject("target");
    const std::optional<llvm::StringRef> targetKind = target ? target->getString("kind") : std::nullopt;
    if (!targetKind || targetKind->empty()) {
        error = "compiled Program module has no target identity";
        return false;
    }
    const std::string identity = canonicalJsonSha256(llvm::json::Value(copyJsonObject(*target)));
    if (!stage.modules.empty() && stage.targetIdentity != identity) {
        error = "compiled Program modules disagree on target identity";
        return false;
    }
    stage.targetIdentity = identity;
    stage.targetImplementation.target = targetKind->str();

    const std::optional<int64_t> pipelineVersion = reflection.getInteger("program_version");
    if (!stage.modules.empty() && stage.pipelineVersion != pipelineVersion) {
        error = "compiled Program modules disagree on pipeline version";
        return false;
    }
    stage.pipelineVersion = pipelineVersion;

    const llvm::json::Object *implementation = reflection.getObject("implementation");
    const std::optional<llvm::StringRef> implementationTarget =
        implementation ? implementation->getString("target") : std::nullopt;
    const llvm::json::Object *metadata = implementation ? implementation->getObject("metadata") : nullptr;
    if (!implementation || implementation->size() != 2 || implementationTarget != targetKind || !metadata) {
        error = "compiled Program module has an invalid target implementation";
        return false;
    }
    if (*targetKind == "metal") {
        const llvm::json::Array *slots = metadata->getArray("resource_slots");
        if (metadata->size() != 1 || !slots) {
            error = "compiled Metal Program implementation must contain only resource_slots metadata";
            return false;
        }
        llvm::json::Array *merged = stage.targetImplementation.metadata.getArray("resource_slots");
        if (!merged) {
            stage.targetImplementation.metadata["resource_slots"] = llvm::json::Array();
            merged = stage.targetImplementation.metadata.getArray("resource_slots");
        }
        for (const llvm::json::Value &slotValue : *slots) {
            const llvm::json::Object *slot = slotValue.getAsObject();
            if (!slot || !validMetalResourceSlot(*slot)) {
                error = "compiled Metal Program resource slot is invalid";
                return false;
            }
            merged->emplace_back(copyJsonObject(*slot));
        }
    } else if (!metadata->empty()) {
        error = "compiled Program module contains unsupported target implementation metadata";
        return false;
    }
    return true;
}

} // namespace vernon::compiler
