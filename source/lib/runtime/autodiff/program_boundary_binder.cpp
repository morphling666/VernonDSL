#include "program_boundary_binder.h"

#include "runtime/program_execution_manifest.h"
#include "runtime/runtime_dispatch.h"

#include <algorithm>
#include <set>

namespace vernon::runtime::ad {

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const ProgramBoundaryBindingRequest &request,
                           std::map<uint32_t, VernonPipelineArgument> &externalValues,
                           std::map<uint32_t, ProgramStorageBacking> &backings, std::vector<char> &live,
                           std::string &error) {
    const VernonPipelineInvocation &invocation = request.invocation;
    if (invocation.argument_count != request.valueBySlot.size() ||
        (invocation.argument_count && !invocation.arguments)) {
        error = "Program invocation does not match its canonical boundary slots";
        return false;
    }
    std::map<uint32_t, const VernonPipelineArgument *> bySlot;
    for (size_t index = 0; index < invocation.argument_count; ++index) {
        const VernonPipelineArgument &argument = invocation.arguments[index];
        if (!bySlot.emplace(argument.slot, &argument).second) {
            error = "Program invocation binds one canonical slot more than once";
            return false;
        }
    }
    ProgramOwnerBindings ownerBindings;
    std::map<std::pair<program::ProgramOwnerKind, uint32_t>, size_t> stagedOwners;
    std::set<std::pair<program::ProgramOwnerKind, uint32_t>> borrowedOutputOwners;
    if (request.publications) {
        for (const auto &[slot, value] : request.valueBySlot) {
            (void)value;
            if (slot >= execution.abi.boundarySlots.size())
                continue;
            const program::BoundarySlot &boundary = execution.abi.boundarySlots[slot];
            const program::PublicationTarget *target = program::findPublicationTarget(execution.abi, slot);
            const auto supplied = bySlot.find(slot);
            if (!target || target->role != program::BoundaryRole::Output || supplied == bySlot.end() ||
                boundary.aliasOwner.kind != program::ProgramOwnerKind::Storage ||
                supplied->second->kind != VERNON_PIPELINE_TENSOR)
                continue;
            const auto storage =
                std::find_if(execution.storages.begin(), execution.storages.end(),
                             [&](const program::Storage &candidate) { return candidate.id == boundary.aliasOwner.id; });
            if (storage == execution.storages.end()) {
                error = "Program output boundary references an unknown Storage";
                return false;
            }
            const auto key = std::make_pair(boundary.aliasOwner.kind, boundary.aliasOwner.id);
            if (storage->ownership == program::StorageOwnership::Borrowed) {
                borrowedOutputOwners.insert(key);
                continue;
            }
            PendingProgramPublication publication{target, *supplied->second};
            if (supplied->second->tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                VernonRhiBuffer destination{};
                if (!resolveBackendRhiBufferReference(context, supplied->second->tensor.resource, destination)) {
                    error = "PublicationPlan device output is not backed by a referenced Vernon RHI buffer";
                    return false;
                }
                publication.destinationBuffer = destination;
            }
            stagedOwners.emplace(key, request.publications->size());
            request.publications->push_back(std::move(publication));
        }
    }
    for (const auto &[slot, value] : request.valueBySlot) {
        const auto supplied = bySlot.find(slot);
        if (slot >= execution.abi.boundarySlots.size() || execution.abi.boundarySlots[slot].id != slot ||
            value >= execution.values.size() || supplied == bySlot.end()) {
            error = "Program invocation is missing a canonical boundary slot";
            return false;
        }
        const program::ProgramOwnerId &owner = execution.abi.boundarySlots[slot].aliasOwner;
        if (!ownerBindings.bind(owner, *supplied->second, error))
            return false;
        const auto ownerKey = std::make_pair(owner.kind, owner.id);
        if (borrowedOutputOwners.count(ownerKey)) {
            if (owner.kind != program::ProgramOwnerKind::Storage || supplied->second->kind != VERNON_PIPELINE_TENSOR) {
                error = "borrowed Program output owner is not Tensor Storage";
                return false;
            }
            ProgramStorageBacking &backing = backings[owner.id];
            backing.external = *supplied->second;
            backing.bytes = supplied->second->tensor.byte_size;
            backing.sized = backing.bytes != 0;
            externalValues.emplace(value, *supplied->second);
            live[value] = 1;
            continue;
        }
        const auto staged = stagedOwners.find(ownerKey);
        if (staged != stagedOwners.end()) {
            if (execution.abi.boundarySlots[slot].role == program::BoundaryRole::Input) {
                error = "owned Program Storage cannot be bound as a public input";
                return false;
            }
            live[value] = 1;
            continue;
        }
        externalValues.emplace(value, *supplied->second);
        live[value] = 1;
        if (owner.kind != program::ProgramOwnerKind::Storage)
            continue;
        ProgramStorageBacking &backing = backings[owner.id];
        backing.external = *supplied->second;
        if (supplied->second->kind == VERNON_PIPELINE_TENSOR) {
            backing.bytes = supplied->second->tensor.byte_size;
            backing.sized = backing.bytes != 0;
        }
    }
    return true;
}

} // namespace vernon::runtime::ad
