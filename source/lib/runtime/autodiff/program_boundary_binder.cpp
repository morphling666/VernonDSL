#include "program_boundary_binder.h"

#include "runtime/program_execution_manifest.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/tensor_bridge.h"

#include <algorithm>
#include <set>

namespace vernon::runtime::ad {
using program_execution::ProgramStorageBacking;
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

bool sameResourceReference(const VernonRuntimeProviderResourceReference &lhs,
                           const VernonRuntimeProviderResourceReference &rhs) {
    return lhs.identity == rhs.identity && lhs.resource.value == rhs.resource.value && lhs.offset == rhs.offset &&
           lhs.size == rhs.size;
}

bool sameBoundaryBinding(const VernonProgramArgument &lhs, const VernonProgramArgument &rhs) {
    if (lhs.kind != rhs.kind)
        return false;
    if (lhs.kind == VERNON_PROGRAM_TENSOR)
        return lhs.tensor.storage == rhs.tensor.storage && lhs.tensor.byte_offset == rhs.tensor.byte_offset &&
               lhs.tensor.byte_size == rhs.tensor.byte_size &&
               (lhs.tensor.storage == VERNON_TENSOR_HOST
                    ? lhs.tensor.host_data == rhs.tensor.host_data
                    : sameResourceReference(lhs.tensor.resource, rhs.tensor.resource));
    if (lhs.kind == VERNON_PROGRAM_IMAGE)
        return sameResourceReference(lhs.image.view, rhs.image.view);
    return sameResourceReference(lhs.resource, rhs.resource);
}

class ProgramOwnerBindings {
public:
    bool bind(program::ProgramOwnerId owner, const VernonProgramArgument &argument, std::string &error) {
        const auto [binding, inserted] = bindings_.emplace(std::make_pair(owner.kind, owner.id), argument);
        if (!inserted && !sameBoundaryBinding(binding->second, argument))
            return error = "Program invocation binds one owner to different resources", false;
        return true;
    }

private:
    std::map<std::pair<program::ProgramOwnerKind, uint32_t>, VernonProgramArgument> bindings_;
};

} // namespace

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const program::ResolvedExecutionPlan &plan, const ProgramBoundaryBindingRequest &request,
                           std::map<uint32_t, VernonProgramArgument> &externalValues,
                           std::map<uint32_t, ProgramStorageBacking> &backings, std::vector<char> &live,
                           std::string &error) {
    const VernonStageInvocationDescriptor &invocation = request.invocation;
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
    std::set<std::pair<program::ProgramOwnerKind, uint32_t>> inPlaceOwners;
    if (request.publication) {
        for (const program::ResolvedPublicationTransaction &transaction : plan.publications.transactions) {
            const uint32_t slot = transaction.slot;
            if (std::none_of(request.valueBySlot.begin(), request.valueBySlot.end(),
                             [&](const auto &binding) { return binding.first == slot; }))
                continue;
            if (slot >= execution.abi.boundarySlots.size()) {
                error = "resolved publication references a missing canonical boundary slot";
                return false;
            }
            const program::BoundarySlot &boundary = execution.abi.boundarySlots[slot];
            const auto key = std::make_pair(transaction.stagingOwner.kind, transaction.stagingOwner.id);
            const auto supplied = bySlot.find(slot);
            if (supplied == bySlot.end()) {
                error = "publication output is not bound at its exact canonical slot";
                return false;
            }
            if (transaction.mode == program::PublicationCommitMode::InPlace) {
                if (boundary.aliasOwner.kind != program::ProgramOwnerKind::Storage ||
                    !request.publication->bindInPlace(slot, *supplied->second, error))
                    return false;
                inPlaceOwners.insert(key);
                continue;
            }
            const program::PublicationTarget *target = program::findPublicationTarget(execution.abi, slot);
            if (!target) {
                error = "resolved commit-after-success publication has no canonical target";
                return false;
            }
            if (target->role != program::BoundaryRole::Output ||
                boundary.aliasOwner.kind != program::ProgramOwnerKind::Storage) {
                error = "commit-after-success publication requires its exact Storage output slot";
                return false;
            }
            const auto storage =
                std::find_if(execution.storages.begin(), execution.storages.end(),
                             [&](const program::Storage &candidate) { return candidate.id == boundary.aliasOwner.id; });
            if (storage == execution.storages.end()) {
                error = "Program output boundary references an unknown Storage";
                return false;
            }
            if (supplied->second->kind == VERNON_PROGRAM_IMAGE) {
                VernonProgramArgument staging{};
                if (!request.publication->bindImageCommit(context, slot, *target, *supplied->second, staging, error))
                    return false;
                externalValues.emplace(transaction.value, staging);
                backings[transaction.stagingOwner.id].external = std::move(staging);
            } else if (supplied->second->kind != VERNON_PROGRAM_TENSOR) {
                error = "commit-after-success publication has an unsupported resource kind";
                return false;
            } else if (supplied->second->tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                VernonRhiBuffer destination{};
                if (!resolveBackendRhiBufferReference(context, supplied->second->tensor.resource, destination)) {
                    error = "PublicationPlan device output is not backed by a referenced Vernon RHI buffer";
                    return false;
                }
                if (!request.publication->bindDeviceCommit(slot, *target, *supplied->second, destination, error))
                    return false;
            } else if (!request.publication->bindHostCommit(slot, *target, *supplied->second, error)) {
                return false;
            }
            stagedOwners.emplace(key, slot);
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
        if (inPlaceOwners.count(ownerKey)) {
            if (owner.kind != program::ProgramOwnerKind::Storage || supplied->second->kind != VERNON_PROGRAM_TENSOR) {
                error = "in-place Program output owner is not Tensor Storage";
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
            if (boundary.role == program::BoundaryRole::Input && boundary.access != program::BoundaryAccess::Write) {
                if (owner.kind != program::ProgramOwnerKind::Storage) {
                    error = "staged Program input requires Storage data";
                    return false;
                }
                if (supplied->second->kind == VERNON_PROGRAM_TENSOR) {
                    ProgramStorageBacking &backing = backings[owner.id];
                    backing.initial = *supplied->second;
                    backing.bytes = supplied->second->tensor.byte_size;
                    backing.sized = true;
                } else if (supplied->second->kind != VERNON_PROGRAM_IMAGE) {
                    error = "staged Program input has an unsupported Storage resource";
                    return false;
                }
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
        if (supplied->second->kind == VERNON_PROGRAM_TENSOR) {
            backing.bytes = supplied->second->tensor.byte_size;
            backing.sized = backing.bytes != 0;
        }
    }
    return true;
}

} // namespace vernon::runtime::ad
