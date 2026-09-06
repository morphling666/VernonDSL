#include "program_boundary_binder.h"

#include "runtime/program_execution_manifest.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/tensor_bridge.h"

#include <algorithm>
#include <set>

namespace vernon::runtime::ad {
namespace {

bool boundaryKindMatches(program::BoundaryCategory category, VernonProgramArgumentKind kind) {
    if (category == program::BoundaryCategory::Texture)
        return kind == VERNON_PROGRAM_IMAGE;
    if (category == program::BoundaryCategory::Sampler)
        return kind == VERNON_PROGRAM_SAMPLER;
    return kind == VERNON_PROGRAM_TENSOR;
}

bool boundaryAccessMatches(program::BoundaryAccess access, VernonValueAccess supplied) {
    if (supplied > VERNON_ACCESS_READ_WRITE)
        return false;
    if (access == program::BoundaryAccess::Read)
        return supplied != VERNON_ACCESS_WRITE;
    if (access == program::BoundaryAccess::Write)
        return supplied != VERNON_ACCESS_READ;
    return supplied == VERNON_ACCESS_READ_WRITE;
}

} // namespace

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const ProgramBoundaryBindingRequest &request,
                           std::map<uint32_t, VernonProgramArgument> &externalValues,
                           std::map<uint32_t, ProgramStorageState> &backings, std::vector<char> &live,
                           std::string &error) {
    const VernonProgramSubmitDescriptor &invocation = request.invocation;
    if (invocation.argument_count != request.valueBySlot.size() ||
        (invocation.argument_count && !invocation.arguments)) {
        error = "Program invocation does not match its canonical boundary slots";
        return false;
    }
    std::map<uint32_t, const VernonProgramArgument *> bySlot;
    for (size_t index = 0; index < invocation.argument_count; ++index) {
        const VernonProgramArgument &argument = invocation.arguments[index];
        if (!bySlot.emplace(argument.slot, &argument).second) {
            error = "Program invocation binds one canonical slot more than once";
            return false;
        }
    }
    ProgramOwnerBindings ownerBindings;
    std::vector<std::pair<program::ProgramOwnerId, const VernonTensorView *>> tensorOwners;
    std::map<std::pair<program::ProgramOwnerKind, uint32_t>, size_t> stagedOwners;
    std::set<std::pair<program::ProgramOwnerKind, uint32_t>> borrowedOutputOwners;
    if (request.publications) {
        for (const program::PublicationTarget &target : execution.abi.publication.targets) {
            const uint32_t slot = target.slot;
            if (slot >= execution.abi.boundarySlots.size())
                continue;
            const program::BoundarySlot &boundary = execution.abi.boundarySlots[slot];
            auto supplied = bySlot.find(slot);
            if (supplied == bySlot.end())
                for (const auto &[candidateSlot, value] : request.valueBySlot) {
                    (void)value;
                    if (candidateSlot >= execution.abi.boundarySlots.size())
                        continue;
                    const program::ProgramOwnerId &candidate = execution.abi.boundarySlots[candidateSlot].aliasOwner;
                    if (candidate.kind == target.aliasOwner.kind && candidate.id == target.aliasOwner.id) {
                        supplied = bySlot.find(candidateSlot);
                        if (supplied != bySlot.end())
                            break;
                    }
                }
            if (target.role != program::BoundaryRole::Output || supplied == bySlot.end() ||
                boundary.aliasOwner.kind != program::ProgramOwnerKind::Storage ||
                supplied->second->kind != VERNON_PROGRAM_TENSOR)
                continue;
            const auto storage =
                std::find_if(execution.storages.begin(), execution.storages.end(),
                             [&](const program::Storage &candidate) { return candidate.id == boundary.aliasOwner.id; });
            if (storage == execution.storages.end()) {
                error = "Program output boundary references an unknown Storage";
                return false;
            }
            const auto key = std::make_pair(boundary.aliasOwner.kind, boundary.aliasOwner.id);
            if (storage->ownership == program::StorageOwnership::Borrowed &&
                supplied->second->tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                borrowedOutputOwners.insert(key);
                continue;
            }
            PendingProgramPublication publication{&target, *supplied->second};
            if (supplied->second->tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                VernonRhiBuffer destination{};
                if (!resolveBackendRhiBufferReference(context, supplied->second->tensor.resource, destination)) {
                    error = "PublicationPlan device output is not backed by a referenced Vernon RHI buffer";
                    return false;
                }
                publication.destinationBuffer = destination;
            }
            request.publications->push_back(std::move(publication));
            stagedOwners.emplace(key, request.publications->size() - 1);
        }
    }
    for (const auto &[slot, value] : request.valueBySlot) {
        const auto supplied = bySlot.find(slot);
        if (slot >= execution.abi.boundarySlots.size() || execution.abi.boundarySlots[slot].id != slot ||
            value >= execution.values.size() || supplied == bySlot.end()) {
            error = "Program invocation is missing a canonical boundary slot";
            return false;
        }
        const program::BoundarySlot &boundary = execution.abi.boundarySlots[slot];
        if (!boundaryKindMatches(boundary.category, supplied->second->kind) ||
            (supplied->second->kind == VERNON_PROGRAM_TENSOR &&
             !boundaryAccessMatches(boundary.access, supplied->second->tensor.access))) {
            error = "Program invocation argument does not match its canonical boundary contract";
            return false;
        }
        const program::ProgramOwnerId &owner = boundary.aliasOwner;
        if (supplied->second->kind == VERNON_PROGRAM_TENSOR) {
            for (const auto &[boundOwner, tensor] : tensorOwners)
                if ((boundOwner.kind != owner.kind || boundOwner.id != owner.id) &&
                    tensorViewsHaveWritableOverlap(*tensor, supplied->second->tensor)) {
                    error = "Program boundary Storage owners have incompatible physical overlap";
                    return false;
                }
            tensorOwners.emplace_back(owner, &supplied->second->tensor);
        }
        if (!ownerBindings.bind(owner, *supplied->second, error))
            return false;
        const auto ownerKey = std::make_pair(owner.kind, owner.id);
        if (borrowedOutputOwners.count(ownerKey)) {
            if (owner.kind != program::ProgramOwnerKind::Storage || supplied->second->kind != VERNON_PROGRAM_TENSOR) {
                error = "borrowed Program output owner is not Tensor Storage";
                return false;
            }
            ProgramStorageState &backing = backings[owner.id];
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
                if (owner.kind != program::ProgramOwnerKind::Storage ||
                    supplied->second->kind != VERNON_PROGRAM_TENSOR ||
                    supplied->second->tensor.storage != VERNON_TENSOR_HOST || !supplied->second->tensor.host_data) {
                    error = "staged Program Storage input requires host Tensor data";
                    return false;
                }
                ProgramStorageState &backing = backings[owner.id];
                backing.initial = *supplied->second;
                backing.bytes = supplied->second->tensor.byte_size;
                backing.sized = true;
            }
            live[value] = 1;
            continue;
        }
        externalValues.emplace(value, *supplied->second);
        live[value] = 1;
        if (owner.kind != program::ProgramOwnerKind::Storage)
            continue;
        ProgramStorageState &backing = backings[owner.id];
        backing.external = *supplied->second;
        if (supplied->second->kind == VERNON_PROGRAM_TENSOR) {
            backing.bytes = supplied->second->tensor.byte_size;
            backing.sized = backing.bytes != 0;
        }
    }
    return true;
}

} // namespace vernon::runtime::ad
