#include "program_invocation_state.h"

#include <algorithm>
#include <cstring>

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

ProgramInvocationState::~ProgramInvocationState() = default;

ProgramInvocationState::ProgramInvocationState(ProgramInvocationState &&other) noexcept
    : plan_(other.plan_), values_(std::move(other.values_)), storageBackings_(std::move(other.storageBackings_)),
      arguments_(std::move(other.arguments_)), deviceValues_(std::move(other.deviceValues_)),
      deviceUploads_(std::move(other.deviceUploads_)), controlImages_(std::move(other.controlImages_)),
      invocationContext_(other.invocationContext_) {
    for (size_t value = 0; value < arguments_.size(); ++value)
        rebindDescriptor(static_cast<uint32_t>(value));
}

CanonicalValueSnapshot ProgramInvocationState::snapshotValue(uint32_t value) const {
    CanonicalValueSnapshot snapshot;
    snapshot.value = value;
    if (value >= values_.size())
        return snapshot;
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
    return snapshot;
}

bool ProgramInvocationState::importSnapshot(const CanonicalValueSnapshot &snapshot, std::string &error) {
    if (snapshot.value >= values_.size())
        return error = "retained Value snapshot exceeds invocation state", false;
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
    return true;
}

bool ProgramInvocationState::importStorageSnapshots(const std::map<uint32_t, ProgramStorageBacking> &snapshots,
                                                    std::string &error) {
    for (const auto &[storage, snapshot] : snapshots) {
        const auto current = storageBackings_.find(storage);
        if (current == storageBackings_.end())
            return error = "retained Storage snapshot is absent from apply invocation", false;
        current->second = snapshot;
    }
    return true;
}

bool ProgramInvocationState::restoreDeviceValuesFromHost(program::GraphDirection graph, std::string &error) {
    std::vector<const DeviceBuffer *> restored;
    for (size_t value = 0; value < arguments_.size(); ++value) {
        if (!plan_->requiresDevice(graph, static_cast<uint32_t>(value)) || !deviceValues_[value] ||
            std::find(restored.begin(), restored.end(), deviceValues_[value].get()) != restored.end())
            continue;
        const VernonTensorView &host = values_[value].argument.tensor;
        if (!host.host_data || host.byte_offset || host.byte_size != deviceValues_[value]->size() ||
            !deviceValues_[value]->upload(host.host_data, host.byte_size))
            return error = "Program invocation cannot restore planned device Storage", false;
        restored.push_back(deviceValues_[value].get());
    }
    return true;
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

bool resolveProgramControl(const program::Program &program, const std::vector<ProgramValueState> &values,
                           const program::ControlComponent &control, uint64_t &value, std::string &error) {
    if (control.kind == program::ControlKind::Static) {
        value = control.value;
        return true;
    }
    uint32_t valueId = UINT32_MAX;
    if (control.kind == program::ControlKind::Parameter) {
        if (control.reference < program.parameters.size())
            valueId = program.parameters[control.reference].value;
    } else {
        const auto argument =
            std::find_if(program.values.begin(), program.values.end(), [&](const program::Value &row) {
                return row.origin.kind == program::OriginKind::Argument && row.origin.slot == control.reference;
            });
        if (argument != program.values.end())
            valueId = argument->id;
    }
    if (valueId >= values.size())
        return error = "Program dispatch control references an unavailable Value", false;
    const VernonProgramArgument &argument = values[valueId].argument;
    if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
        !argument.tensor.host_data || argument.tensor.byte_offset > argument.tensor.byte_size)
        return error = "Program dispatch control is not a retained host scalar", false;
    const uint8_t *data = static_cast<const uint8_t *>(argument.tensor.host_data) + argument.tensor.byte_offset;
    const std::string &dtype = program.values[valueId].canonicalType.dtype;
    if (dtype == "u32" || dtype == "ui32") {
        uint32_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        value = scalar;
    } else if (dtype == "i32" || dtype == "si32") {
        int32_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        if (scalar < 0)
            return error = "Program dispatch control must be non-negative", false;
        value = static_cast<uint64_t>(scalar);
    } else if (dtype == "u64" || dtype == "ui64" || dtype == "index") {
        std::memcpy(&value, data, sizeof(value));
    } else if (dtype == "i64" || dtype == "si64") {
        int64_t scalar{};
        std::memcpy(&scalar, data, sizeof(scalar));
        if (scalar < 0)
            return error = "Program dispatch control must be non-negative", false;
        value = static_cast<uint64_t>(scalar);
    } else {
        return error = "Program dispatch control must be an integer scalar", false;
    }
    return true;
}

bool ProgramInvocationState::resolveControl(const program::Program &program, const program::ControlComponent &control,
                                            uint64_t &value, std::string &error) const {
    return resolveProgramControl(program, values_, control, value, error);
}

bool ProgramInvocationState::bindControlImageStorage(const program::Program &program, uint32_t storage,
                                                     VernonRuntimeProviderResourceReference view, std::string &error) {
    if (!view.identity || !view.resource.value)
        return error = "Program attachment Storage has no runtime image view", false;
    if (storage >= program.storages.size())
        return error = "Program attachment references unknown Storage", false;
    return controlImages_.emplace(storage, view).second || controlImages_.at(storage).identity == view.identity ||
           (error = "Program Storage has multiple runtime image views", false);
}

const VernonRuntimeProviderResourceReference *ProgramInvocationState::controlImage(uint32_t storage) const {
    const auto found = controlImages_.find(storage);
    return found == controlImages_.end() ? nullptr : &found->second;
}

const VernonProgramArgument *ProgramInvocationState::argument(uint32_t value) const {
    return value < arguments_.size() ? &arguments_[value] : nullptr;
}

VernonProgramArgument *ProgramInvocationState::argument(uint32_t value) {
    return const_cast<VernonProgramArgument *>(static_cast<const ProgramInvocationState &>(*this).argument(value));
}

VernonRhiBuffer ProgramInvocationState::buffer(uint32_t value) const {
    if (value < deviceValues_.size() && deviceValues_[value])
        return deviceValues_[value]->handle();
    return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

} // namespace vernon::runtime::program_execution
