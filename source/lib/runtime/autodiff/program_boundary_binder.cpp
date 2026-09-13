#include "program_boundary_binder.h"

#include "runtime/program_execution/program_boundary_contract.h"
#include "runtime/program_execution/program_image_binding.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/tensor_bridge.h"

#include <algorithm>
#include <limits>

namespace vernon::runtime::ad {
using program_execution::ProgramStorageBacking;
namespace {

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

} // namespace

void ProgramBoundaryBindingScratch::reset(const program::Program &program) {
    valueCount = program.values.size();
    bySlot.resize(program.abi.boundarySlots.size());
    std::fill(bySlot.begin(), bySlot.end(), nullptr);
    externalValues.resize(program.values.size());
    for (auto &value : externalValues)
        value.reset();
    ownerBindings.resize(program.values.size() + program.storages.size());
    for (auto &binding : ownerBindings)
        binding.reset();
    tensorOwners.clear();
    stagedOwners.resize(ownerBindings.size());
    std::fill(stagedOwners.begin(), stagedOwners.end(), std::numeric_limits<size_t>::max());
    inPlaceOwners.resize(ownerBindings.size());
    std::fill(inPlaceOwners.begin(), inPlaceOwners.end(), 0);
    liveValues.resize(program.values.size());
    std::fill(liveValues.begin(), liveValues.end(), 0);
    liveStorages.resize(program.storages.size());
    std::fill(liveStorages.begin(), liveStorages.end(), 0);
}

size_t ProgramBoundaryBindingScratch::ownerIndex(program::ProgramOwnerId owner) const {
    return owner.kind == program::ProgramOwnerKind::Value ? owner.id : valueCount + owner.id;
}

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const program::ResolvedExecutionPlan &plan, const ProgramBoundaryBindingRequest &request,
                           ProgramBoundaryBindingScratch &scratch, std::map<uint32_t, ProgramStorageBacking> &backings,
                           std::vector<char> &live, std::string &error) {
    if (request.argumentCount != request.valueBySlot.size() || (request.argumentCount && !request.arguments)) {
        error = "Program invocation does not match its canonical boundary slots";
        return false;
    }
    for (size_t index = 0; index < request.argumentCount; ++index) {
        const VernonProgramArgument &argument = request.arguments[index];
        if (argument.slot >= scratch.bySlot.size() || scratch.bySlot[argument.slot]) {
            error = "Program invocation binds one canonical slot more than once";
            return false;
        }
        scratch.bySlot[argument.slot] = &argument;
    }
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
            const size_t ownerIndex = scratch.ownerIndex(transaction.stagingOwner);
            const VernonProgramArgument *supplied = scratch.bySlot[slot];
            if (!supplied || ownerIndex >= scratch.ownerBindings.size()) {
                error = "publication output is not bound at its exact canonical slot";
                return false;
            }
            if (transaction.mode == program::PublicationCommitMode::InPlace) {
                if (boundary.aliasOwner.kind != program::ProgramOwnerKind::Storage)
                    return error = "in-place publication boundary is not Storage-backed", false;
                auto bound = request.publication->bindInPlace(slot, *supplied);
                if (bound.isErr())
                    return error = program_execution::publicationErrorMessage(bound.error()), false;
                scratch.inPlaceOwners[ownerIndex] = 1;
                continue;
            }
            const program::PublicationTarget *target = program::findPublicationTarget(execution.abi, slot);
            if (!target) {
                error = "resolved commit-after-success publication has no canonical target";
                return false;
            }
            if (target->role != boundary.role ||
                (boundary.role != program::BoundaryRole::Output && boundary.role != program::BoundaryRole::Gradient)) {
                error = "commit-after-success publication requires its exact publication slot";
                return false;
            }
            if (supplied->kind == VERNON_PROGRAM_IMAGE) {
                VernonProgramArgument staging{};
                auto bound = request.publication->bindImageCommit(context, slot, *target, *supplied, staging);
                if (bound.isErr())
                    return error = program_execution::publicationErrorMessage(bound.error()), false;
                scratch.externalValues[transaction.value] = staging;
                backings[transaction.stagingOwner.id].external = std::move(staging);
            } else if (supplied->kind != VERNON_PROGRAM_TENSOR) {
                error = "commit-after-success publication has an unsupported resource kind";
                return false;
            } else if (supplied->tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                auto destination = resolveBackendRhiBufferReference(context, supplied->tensor.resource);
                if (destination.isErr()) {
                    error = "PublicationPlan device output is not backed by a referenced Vernon RHI buffer";
                    return false;
                }
                auto bound = request.publication->bindDeviceCommit(slot, *target, *supplied, destination.value());
                if (bound.isErr())
                    return error = program_execution::publicationErrorMessage(bound.error()), false;
            } else {
                auto bound = request.publication->bindHostCommit(slot, *target, *supplied);
                if (bound.isErr())
                    return error = program_execution::publicationErrorMessage(bound.error()), false;
            }
            scratch.stagedOwners[ownerIndex] = slot;
        }
    }
    for (const auto &[slot, value] : request.valueBySlot) {
        const VernonProgramArgument *supplied = slot < scratch.bySlot.size() ? scratch.bySlot[slot] : nullptr;
        if (slot >= execution.abi.boundarySlots.size() || execution.abi.boundarySlots[slot].id != slot ||
            value >= execution.values.size() || !supplied) {
            error = "Program invocation is missing a canonical boundary slot";
            return false;
        }
        const program::BoundarySlot &boundary = execution.abi.boundarySlots[slot];
        if (!program_execution::argumentMatchesBoundary(boundary, *supplied)) {
            error = "Program invocation argument does not match its canonical boundary contract";
            return false;
        }
        const program::ProgramOwnerId &owner = boundary.aliasOwner;
        if (supplied->kind == VERNON_PROGRAM_IMAGE) {
            if (owner.kind != program::ProgramOwnerKind::Storage || owner.id >= execution.storages.size()) {
                error = "Program image boundary does not reference Storage";
                return false;
            }
            const program::Storage &storage = execution.storages[owner.id];
            if (storage.ownership == program::StorageOwnership::Borrowed) {
                program_execution::BoundProgramImage resolvedImage;
                if (!program_execution::resolveBorrowedProgramImage(context, storage, supplied->image.view,
                                                                    resolvedImage, error))
                    return false;
            } else if (boundary.role != program::BoundaryRole::Output &&
                       boundary.role != program::BoundaryRole::Gradient) {
                error = "owned Program image Storage cannot be supplied as an input boundary";
                return false;
            }
        }
        if (supplied->kind == VERNON_PROGRAM_TENSOR) {
            for (const auto &[boundOwner, tensor] : scratch.tensorOwners)
                if ((boundOwner.kind != owner.kind || boundOwner.id != owner.id) &&
                    tensorViewsHaveWritableOverlap(*tensor, supplied->tensor)) {
                    error = "Program boundary Storage owners have incompatible physical overlap";
                    return false;
                }
            scratch.tensorOwners.push_back({owner, &supplied->tensor});
        }
        const size_t ownerIndex = scratch.ownerIndex(owner);
        if (ownerIndex >= scratch.ownerBindings.size())
            return error = "Program boundary owner is outside the canonical Program", false;
        auto &ownerBinding = scratch.ownerBindings[ownerIndex];
        if (ownerBinding && !sameBoundaryBinding(*ownerBinding, *supplied))
            return error = "Program invocation binds one owner to different resources", false;
        ownerBinding = *supplied;
        if (scratch.inPlaceOwners[ownerIndex]) {
            if (owner.kind != program::ProgramOwnerKind::Storage) {
                error = "in-place Program output owner is not Storage-backed";
                return false;
            }
            if (supplied->kind == VERNON_PROGRAM_IMAGE) {
                backings[owner.id].external = *supplied;
                scratch.externalValues[value] = *supplied;
                live[value] = 1;
                continue;
            }
            if (supplied->kind != VERNON_PROGRAM_TENSOR) {
                error = "in-place Program output owner has no physical Storage";
                return false;
            }
            ProgramStorageBacking &backing = backings[owner.id];
            backing.external = *supplied;
            backing.bytes = supplied->tensor.byte_size;
            backing.sized = backing.bytes != 0;
            scratch.externalValues[value] = *supplied;
            live[value] = 1;
            continue;
        }
        if (scratch.stagedOwners[ownerIndex] != std::numeric_limits<size_t>::max()) {
            if (boundary.role == program::BoundaryRole::Input) {
                if (owner.kind != program::ProgramOwnerKind::Storage) {
                    error = "staged Program input requires Storage data";
                    return false;
                }
                if (supplied->kind == VERNON_PROGRAM_TENSOR) {
                    ProgramStorageBacking &backing = backings[owner.id];
                    backing.initial = *supplied;
                    backing.bytes = std::max(backing.bytes, supplied->tensor.byte_size);
                    backing.sized = true;
                } else if (supplied->kind != VERNON_PROGRAM_IMAGE) {
                    error = "staged Program input has an unsupported Storage resource";
                    return false;
                }
            }
            live[value] = 1;
            continue;
        }
        scratch.externalValues[value] = *supplied;
        live[value] = 1;
        if (owner.kind != program::ProgramOwnerKind::Storage)
            continue;
        ProgramStorageBacking &backing = backings[owner.id];
        backing.external = *supplied;
        if (supplied->kind == VERNON_PROGRAM_TENSOR) {
            backing.bytes = supplied->tensor.byte_size;
            backing.sized = backing.bytes != 0;
        }
    }
    return true;
}

} // namespace vernon::runtime::ad
