#include "program_invocation_state.h"

#include "runtime/runtime_state.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace vernon::runtime::program_execution {

ProgramInvocationState::ProgramInvocationState(const program::ResolvedExecutionPlan &plan,
                                               std::vector<ProgramValueState> values,
                                               std::map<uint32_t, ProgramStorageBacking> storageBackings)
    : plan_(&plan), values_(std::move(values)), storageBackings_(std::move(storageBackings)),
      arguments_(values_.size()), deviceValues_(values_.size()) {
    for (size_t value = 0; value < values_.size(); ++value) {
        arguments_[value] = values_[value].argument;
        rebindDescriptor(static_cast<uint32_t>(value));
    }
}

ProgramInvocationState::~ProgramInvocationState() noexcept {
    if (imageDevice_.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return;
    for (auto image = ownedImages_.rbegin(); image != ownedImages_.rend(); ++image) {
        if (image->view.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImageView(imageDevice_, image->view);
        if (image->image.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImage(imageDevice_, image->image);
    }
}

ProgramInvocationState::ProgramInvocationState(ProgramInvocationState &&other) noexcept
    : plan_(other.plan_), values_(std::move(other.values_)), storageBackings_(std::move(other.storageBackings_)),
      arguments_(std::move(other.arguments_)), deviceValues_(std::move(other.deviceValues_)),
      deviceUploads_(std::move(other.deviceUploads_)), controlImages_(std::move(other.controlImages_)),
      controlImageDescriptors_(std::move(other.controlImageDescriptors_)), imageDevice_(other.imageDevice_),
      ownedImages_(std::move(other.ownedImages_)), invocationContext_(other.invocationContext_) {
    other.imageDevice_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    for (size_t value = 0; value < arguments_.size(); ++value)
        rebindDescriptor(static_cast<uint32_t>(value));
}

vernon::Option<CanonicalValueSnapshot> ProgramInvocationState::snapshotValue(uint32_t value) const {
    if (value >= values_.size())
        return {};
    CanonicalValueSnapshot snapshot;
    snapshot.value = value;
    snapshot.logical = values_[value];
    if (!snapshot.logical.ownedHostBytes.empty() && snapshot.logical.argument.kind == VERNON_PROGRAM_TENSOR)
        snapshot.logical.argument.tensor.host_data = snapshot.logical.ownedHostBytes.data();
    snapshot.residentArgument = arguments_[value];
    if (deviceValues_[value]) {
        snapshot.deviceOwner = deviceValues_[value];
        const VernonTensorView &resident = snapshot.residentArgument.tensor;
        snapshot.logical.stagedDeviceInitial.reset();
        if (resident.element_layout.byte_size) {
            ProgramValueState::StagedDeviceInitial initial;
            initial.source = resident.resource;
            initial.retainedSourceBuffer = deviceValues_[value]->handle();
            initial.byteOffset = resident.byte_offset;
            initial.byteSize = resident.byte_size;
            initial.elementLayout = resident.element_layout;
            if (resident.rank && resident.shape && resident.byte_strides) {
                initial.shape.assign(resident.shape, resident.shape + resident.rank);
                initial.strides.assign(resident.byte_strides, resident.byte_strides + resident.rank);
            }
            snapshot.logical.stagedDeviceInitial = std::move(initial);
        }
    }
    return vernon::Option<CanonicalValueSnapshot>{vernon::some(std::move(snapshot))};
}

ProgramInvocationResult<void> ProgramInvocationState::importSnapshot(const CanonicalValueSnapshot &snapshot) {
    if (snapshot.value >= values_.size())
        return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::SnapshotValueUnavailable)};
    const uint32_t value = snapshot.value;
    values_[value] = snapshot.logical;
    if (!values_[value].ownedHostBytes.empty() && values_[value].argument.kind == VERNON_PROGRAM_TENSOR)
        values_[value].argument.tensor.host_data = values_[value].ownedHostBytes.data();
    if (snapshot.deviceOwner) {
        arguments_[value] = values_[value].argument;
        deviceValues_[value].reset();
    } else {
        arguments_[value] = snapshot.residentArgument;
    }
    rebindDescriptor(value);
    return ProgramInvocationResult<void>{vernon::ok()};
}

ProgramInvocationResult<void>
ProgramInvocationState::importStorageSnapshots(const std::map<uint32_t, ProgramStorageBacking> &snapshots) {
    for (const auto &[storage, snapshot] : snapshots) {
        const auto current = storageBackings_.find(storage);
        if (current == storageBackings_.end())
            return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::SnapshotStorageUnavailable)};
        current->second = snapshot;
    }
    return ProgramInvocationResult<void>{vernon::ok()};
}

void ProgramInvocationState::retainOnly(const std::vector<char> &retained) {
    for (size_t value = 0; value < arguments_.size(); ++value) {
        if (value < retained.size() && retained[value]) {
            ProgramValueState &host = values_[value];
            if (!deviceValues_[value] && host.ownedHostBytes.empty() && host.argument.kind == VERNON_PROGRAM_TENSOR &&
                host.argument.tensor.storage == VERNON_TENSOR_HOST && host.argument.tensor.host_data &&
                host.argument.tensor.byte_size) {
                const auto *data = static_cast<const uint8_t *>(host.argument.tensor.host_data);
                host.ownedHostBytes.assign(data, data + host.argument.tensor.byte_size);
                host.argument.tensor.host_data = host.ownedHostBytes.data();
                arguments_[value] = host.argument;
            }
            continue;
        }
        values_[value] = {};
        arguments_[value] = {};
        deviceValues_[value].reset();
    }
    deviceUploads_.clear();
}

void ProgramInvocationState::rebindDescriptor(uint32_t value) {
    if (value >= values_.size() || value >= arguments_.size())
        return;
    ProgramValueState &host = values_[value];
    if (host.argument.kind != VERNON_PROGRAM_TENSOR || arguments_[value].kind != VERNON_PROGRAM_TENSOR)
        return;
    if (!host.ownedHostBytes.empty() && host.argument.tensor.storage == VERNON_TENSOR_HOST) {
        host.argument.tensor.host_data = host.ownedHostBytes.data();
        arguments_[value].tensor.host_data = host.ownedHostBytes.data();
    }
    if (host.concreteShape) {
        host.argument.tensor.rank = static_cast<uint32_t>(host.concreteShape->size());
        host.argument.tensor.shape = host.concreteShape->data();
        arguments_[value].tensor.rank = static_cast<uint32_t>(host.concreteShape->size());
        arguments_[value].tensor.shape = host.concreteShape->data();
    }
    if (!host.strides.empty()) {
        host.argument.tensor.byte_strides = host.strides.data();
        arguments_[value].tensor.byte_strides = host.strides.data();
    }
}

ProgramInvocationResult<uint64_t> resolveProgramControl(const program::Program &program,
                                                        const std::vector<ProgramValueState> &values,
                                                        const program::ControlComponent &control) {
    if (control.kind == program::ControlKind::Static)
        return ProgramInvocationResult<uint64_t>{vernon::ok(control.value)};
    const uint32_t valueId = control.reference;
    if (valueId >= values.size() || valueId >= program.values.size())
        return ProgramInvocationResult<uint64_t>{vernon::err(ProgramInvocationError::ControlValueUnavailable)};
    const VernonProgramArgument &argument = values[valueId].argument;
    if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
        !argument.tensor.host_data || argument.tensor.byte_offset > argument.tensor.byte_size)
        return ProgramInvocationResult<uint64_t>{vernon::err(ProgramInvocationError::ControlValueUnavailable)};
    const std::string &dtype = program.values[valueId].canonicalType.dtype;
    const bool scalar32 = dtype == "u32" || dtype == "ui32" || dtype == "i32" || dtype == "si32";
    const bool scalar64 = dtype == "u64" || dtype == "ui64" || dtype == "i64" || dtype == "si64" || dtype == "index";
    if (!scalar32 && !scalar64)
        return ProgramInvocationResult<uint64_t>{vernon::err(ProgramInvocationError::ControlValueInvalid)};
    const size_t scalarSize = scalar32 ? sizeof(uint32_t) : sizeof(uint64_t);
    if (scalarSize > argument.tensor.byte_size - argument.tensor.byte_offset)
        return ProgramInvocationResult<uint64_t>{vernon::err(ProgramInvocationError::ControlValueUnavailable)};
    const uint8_t *data = static_cast<const uint8_t *>(argument.tensor.host_data) + argument.tensor.byte_offset;
    uint64_t value{};
    if (dtype == "u32" || dtype == "ui32") {
        uint32_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        value = scalar;
    } else if (dtype == "i32" || dtype == "si32") {
        int32_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        if (scalar < 0)
            return ProgramInvocationResult<uint64_t>{vernon::err(ProgramInvocationError::ControlValueInvalid)};
        value = static_cast<uint64_t>(scalar);
    } else if (dtype == "u64" || dtype == "ui64" || dtype == "index") {
        std::memcpy(&value, data, sizeof(value));
    } else if (dtype == "i64" || dtype == "si64") {
        int64_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        if (scalar < 0)
            return ProgramInvocationResult<uint64_t>{vernon::err(ProgramInvocationError::ControlValueInvalid)};
        value = static_cast<uint64_t>(scalar);
    } else {
        return ProgramInvocationResult<uint64_t>{vernon::err(ProgramInvocationError::ControlValueInvalid)};
    }
    return ProgramInvocationResult<uint64_t>{vernon::ok(value)};
}

ProgramInvocationResult<uint64_t>
ProgramInvocationState::resolveControl(const program::Program &program,
                                       const program::ControlComponent &control) const {
    return resolveProgramControl(program, values_, control);
}

ProgramInvocationResult<void>
ProgramInvocationState::bindControlImageStorage(VernonRuntimeContext &context, const program::Program &program,
                                                uint32_t storage, VernonRuntimeProviderResourceReference view) {
    if (!view.identity || !view.resource.value)
        return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageInvalid)};
    if (storage >= program.storages.size())
        return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageInvalid)};
    const auto [binding, inserted] = controlImages_.emplace(storage, view);
    if (!inserted && binding->second.identity != view.identity)
        return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageConflict)};
    if (!inserted)
        return ProgramInvocationResult<void>{vernon::ok()};
    BoundProgramImage descriptor;
    std::string error;
    if (!resolveBorrowedProgramImage(context, program.storages[storage], view, descriptor, error)) {
        controlImages_.erase(storage);
        return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageResolutionFailed)};
    }
    controlImageDescriptors_.emplace(storage, std::move(descriptor));
    return ProgramInvocationResult<void>{vernon::ok()};
}

ProgramInvocationResult<void> ProgramInvocationState::allocateOwnedImageStorages(VernonRuntimeContext &context,
                                                                                 const program::Program &program) {
    imageDevice_ = context.rhiDevice;
    if (imageDevice_.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        if (std::any_of(program.storages.begin(), program.storages.end(), [](const program::Storage &storage) {
                return storage.ownership == program::StorageOwnership::Owned &&
                       storage.descriptorKind == program::StorageDescriptorKind::Image;
            }))
            return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageInvalid)};
        else
            return ProgramInvocationResult<void>{vernon::ok()};
    }
    for (const program::Storage &storage : program.storages) {
        if (storage.ownership != program::StorageOwnership::Owned ||
            storage.descriptorKind != program::StorageDescriptorKind::Image)
            continue;
        std::array<uint32_t, 3> extent{};
        for (size_t axis = 0; axis < extent.size(); ++axis) {
            uint64_t component = axis < storage.image.extent.size() ? storage.image.extent[axis] : 0;
            if (!storage.image.extentControls.empty()) {
                auto resolved = resolveControl(program, storage.image.extentControls[axis]);
                if (resolved.isErr())
                    return ProgramInvocationResult<void>{vernon::err(resolved.error())};
                component = resolved.value();
            }
            if (!component || component > std::numeric_limits<uint32_t>::max())
                return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ControlValueInvalid)};
            extent[axis] = static_cast<uint32_t>(component);
        }
        VernonRhiImageDescriptor imageDescriptor{};
        std::string error;
        if (!materializeOwnedProgramImageDescriptor(storage, extent, imageDescriptor, error))
            return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageResolutionFailed)};
        OwnedImageStorage owned{{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0},
                                {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}};
        if (vernonRhiDeviceCreateImage(imageDevice_, &imageDescriptor, &owned.image) != VERNON_RHI_STATUS_OK)
            return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageAllocationFailed)};
        VernonRhiImageViewDescriptor viewDescriptor{};
        viewDescriptor.struct_size = sizeof(viewDescriptor);
        viewDescriptor.image = owned.image;
        viewDescriptor.dimension = imageDescriptor.dimension;
        viewDescriptor.format = imageDescriptor.format;
        viewDescriptor.mip_level_count = imageDescriptor.mip_levels;
        viewDescriptor.array_layer_count = imageDescriptor.array_layers;
        for (const std::string &aspect : storage.image.aspects)
            viewDescriptor.aspects |= aspect == "color"     ? VERNON_RHI_IMAGE_ASPECT_COLOR
                                      : aspect == "depth"   ? VERNON_RHI_IMAGE_ASPECT_DEPTH
                                      : aspect == "stencil" ? VERNON_RHI_IMAGE_ASPECT_STENCIL
                                                            : 0;
        if (vernonRhiDeviceCreateImageView(imageDevice_, &viewDescriptor, &owned.view) != VERNON_RHI_STATUS_OK) {
            vernonRhiDeviceDestroyImage(imageDevice_, owned.image);
            return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageAllocationFailed)};
        }
        VernonRuntimeProviderResourceReference reference{};
        if (vernonRuntimeReferenceRhiImageView(&context, owned.view, &reference) != VERNON_STATUS_OK) {
            vernonRhiDeviceDestroyImageView(imageDevice_, owned.view);
            vernonRhiDeviceDestroyImage(imageDevice_, owned.image);
            return ProgramInvocationResult<void>{vernon::err(ProgramInvocationError::ImageStorageResolutionFailed)};
        }
        ownedImages_.push_back(owned);
        controlImages_.emplace(storage.id, reference);
    }
    return ProgramInvocationResult<void>{vernon::ok()};
}

const char *programInvocationErrorMessage(ProgramInvocationError error) noexcept {
    switch (error) {
    case ProgramInvocationError::SnapshotValueUnavailable:
        return "retained Value snapshot exceeds invocation state";
    case ProgramInvocationError::SnapshotStorageUnavailable:
        return "retained Storage snapshot is absent from apply invocation";
    case ProgramInvocationError::DeviceRestoreFailed:
        return "Program invocation cannot restore planned device Storage";
    case ProgramInvocationError::ControlValueUnavailable:
        return "Program dispatch control is not an available retained host scalar";
    case ProgramInvocationError::ControlValueInvalid:
        return "Program dispatch control must be a non-negative integer scalar";
    case ProgramInvocationError::ImageStorageInvalid:
        return "Program image Storage is invalid for this invocation";
    case ProgramInvocationError::ImageStorageConflict:
        return "Program Storage has multiple runtime image views";
    case ProgramInvocationError::ImageStorageResolutionFailed:
        return "Program image Storage cannot be resolved";
    case ProgramInvocationError::ImageStorageAllocationFailed:
        return "Program image Storage allocation failed";
    }
    return "unknown Program invocation error";
}

vernon::Option<std::reference_wrapper<const VernonRuntimeProviderResourceReference>>
ProgramInvocationState::controlImage(uint32_t storage) const {
    const auto found = controlImages_.find(storage);
    if (found == controlImages_.end())
        return {};
    return vernon::Option<std::reference_wrapper<const VernonRuntimeProviderResourceReference>>{
        vernon::some(std::cref(found->second))};
}

vernon::Option<std::reference_wrapper<const VernonProgramArgument>>
ProgramInvocationState::externalStorage(uint32_t storage) const {
    const auto found = storageBackings_.find(storage);
    if (found == storageBackings_.end() || !found->second.external)
        return {};
    return vernon::Option<std::reference_wrapper<const VernonProgramArgument>>{
        vernon::some(std::cref(*found->second.external))};
}

vernon::Option<std::reference_wrapper<const VernonProgramArgument>>
ProgramInvocationState::argument(uint32_t value) const {
    if (value >= arguments_.size())
        return {};
    return vernon::Option<std::reference_wrapper<const VernonProgramArgument>>{
        vernon::some(std::cref(arguments_[value]))};
}

vernon::Option<std::reference_wrapper<VernonProgramArgument>> ProgramInvocationState::argument(uint32_t value) {
    if (value >= arguments_.size())
        return {};
    return vernon::Option<std::reference_wrapper<VernonProgramArgument>>{vernon::some(std::ref(arguments_[value]))};
}

vernon::Option<VernonRhiBuffer> ProgramInvocationState::buffer(uint32_t value) const {
    if (value < deviceValues_.size() && deviceValues_[value])
        return vernon::Option<VernonRhiBuffer>{vernon::some(deviceValues_[value]->handle())};
    return {};
}

} // namespace vernon::runtime::program_execution
