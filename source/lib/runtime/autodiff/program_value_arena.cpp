#include "program_value_arena.h"

#include "runtime/runtime_state.h"
#include "runtime_gpu_argument_binding.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace vernon::runtime::ad {
namespace {

constexpr size_t invalidCarrier = std::numeric_limits<size_t>::max();

VernonValueAccess valueAccess(const std::string &access) {
    if (access == "read")
        return VERNON_ACCESS_READ;
    if (access == "write")
        return VERNON_ACCESS_WRITE;
    return VERNON_ACCESS_READ_WRITE;
}

std::optional<shape::DeclaredShape> logicalProjectionShape(const Parameter &parameter,
                                                           const program::TargetBinding *target) {
    const shape::DeclaredShape physical = shape::decodeRuntimeContractShape(parameter.shape);
    if (!target || !target->viewTransform)
        return physical;
    const std::vector<program::ViewAxisTransform> &axes = target->viewTransform->axes;
    if (axes.size() != physical.size())
        return std::nullopt;
    std::vector<std::optional<shape::Extent>> logical;
    for (size_t physicalAxis = 0; physicalAxis < axes.size(); ++physicalAxis) {
        if (axes[physicalAxis].source != program::ViewAxisSource::LogicalAxis)
            continue;
        if (logical.size() <= axes[physicalAxis].logicalAxis)
            logical.resize(axes[physicalAxis].logicalAxis + 1);
        std::optional<shape::Extent> &extent = logical[axes[physicalAxis].logicalAxis];
        if (extent && *extent != physical[physicalAxis])
            return std::nullopt;
        extent = physical[physicalAxis];
    }
    shape::DeclaredShape result;
    result.reserve(logical.size());
    for (const std::optional<shape::Extent> &extent : logical) {
        if (!extent)
            return std::nullopt;
        result.push_back(*extent);
    }
    return result;
}

struct TensorCopyRegion {
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

bool planTensorCopy(const VernonTensorView &source, const VernonTensorView &destination,
                    std::vector<TensorCopyRegion> &regions, std::string &error) {
    if (source.rank != destination.rank ||
        (source.rank && (!source.shape || !source.byte_strides || !destination.shape || !destination.byte_strides)) ||
        !source.element_layout.byte_size || source.element_layout.byte_size != destination.element_layout.byte_size) {
        return error = "node endpoint materialization has incompatible tensor layouts", false;
    }
    for (uint32_t axis = 0; axis < source.rank; ++axis)
        if (source.shape[axis] != destination.shape[axis])
            return error = "node endpoint materialization has incompatible tensor shapes", false;
    if (source.rank &&
        std::any_of(source.shape, source.shape + source.rank, [](uint64_t extent) { return extent == 0; }))
        return true;

    const size_t elementBytes = source.element_layout.byte_size;
    std::vector<uint64_t> index(source.rank);
    for (;;) {
        if (source.byte_offset > std::numeric_limits<int64_t>::max() ||
            destination.byte_offset > std::numeric_limits<int64_t>::max())
            return error = "node endpoint materialization offset exceeds the portable range", false;
        int64_t sourceOffset = static_cast<int64_t>(source.byte_offset);
        int64_t destinationOffset = static_cast<int64_t>(destination.byte_offset);
        const auto advance = [](int64_t &offset, uint64_t coordinate, int64_t stride) {
            if (stride >= 0) {
                const uint64_t positive = static_cast<uint64_t>(stride);
                if (positive &&
                    coordinate > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - offset) / positive)
                    return false;
                offset += static_cast<int64_t>(coordinate * positive);
                return true;
            }
            const uint64_t magnitude = static_cast<uint64_t>(-(stride + 1)) + 1;
            if (magnitude && coordinate > static_cast<uint64_t>(offset) / magnitude)
                return false;
            offset -= static_cast<int64_t>(coordinate * magnitude);
            return true;
        };
        for (uint32_t axis = 0; axis < source.rank; ++axis)
            if (!advance(sourceOffset, index[axis], source.byte_strides[axis]) ||
                !advance(destinationOffset, index[axis], destination.byte_strides[axis]))
                return error = "node endpoint materialization layout overflows", false;
        const auto validOffset = [&](int64_t offset, size_t capacity) {
            return offset >= 0 && static_cast<uint64_t>(offset) <= capacity &&
                   elementBytes <= capacity - static_cast<size_t>(offset);
        };
        if (!validOffset(sourceOffset, source.byte_size) || !validOffset(destinationOffset, destination.byte_size))
            return error = "node endpoint materialization exceeds its Storage backing", false;
        const size_t sourceByte = static_cast<size_t>(sourceOffset);
        const size_t destinationByte = static_cast<size_t>(destinationOffset);
        if (!regions.empty() && regions.back().sourceOffset + regions.back().size == sourceByte &&
            regions.back().destinationOffset + regions.back().size == destinationByte) {
            regions.back().size += elementBytes;
        } else {
            regions.push_back({sourceByte, destinationByte, elementBytes});
        }
        if (source.rank == 0)
            break;
        uint32_t axis = source.rank;
        while (axis) {
            --axis;
            if (++index[axis] < source.shape[axis])
                break;
            index[axis] = 0;
        }
        if (axis == 0 && index[0] == 0)
            break;
    }
    return true;
}

} // namespace

bool MaterializedNodeFrame::prepareHost(std::string &error) const {
    for (const HostCopy &copy : copiesBefore) {
        if (!copy.source || !copy.destination)
            return error = "node endpoint input transfer has no host backing", false;
        std::memcpy(copy.destination + copy.destinationOffset, copy.source + copy.sourceOffset, copy.size);
    }
    return true;
}

bool MaterializedNodeFrame::commitHost(std::string &error) const {
    for (const HostCopy &copy : copiesAfter) {
        if (!copy.source || !copy.destination)
            return error = "node endpoint result transfer has no host backing", false;
        std::memcpy(copy.destination + copy.destinationOffset, copy.source + copy.sourceOffset, copy.size);
    }
    return true;
}

LogicalValueFrame::LogicalValueFrame(const std::vector<VernonProgramArgument> &hostArguments)
    : logicalArguments_(hostArguments), logicalBuffers_(hostArguments.size()), carriers_(hostArguments.size()) {}

LogicalValueFrame::LogicalValueFrame(std::vector<LogicalProgramValue> hostValues)
    : hostValues_(std::move(hostValues)), logicalArguments_(hostValues_.size()), logicalBuffers_(hostValues_.size()),
      carriers_(hostValues_.size()) {
    for (size_t value = 0; value < hostValues_.size(); ++value) {
        logicalArguments_[value] = hostValues_[value].argument;
        rebindLogicalDescriptor(static_cast<uint32_t>(value));
    }
}

LogicalValueFrame::LogicalValueFrame(LogicalValueFrame &&other) noexcept
    : hostValues_(std::move(other.hostValues_)), logicalArguments_(std::move(other.logicalArguments_)),
      logicalBuffers_(std::move(other.logicalBuffers_)), deviceUploads_(std::move(other.deviceUploads_)),
      carriers_(std::move(other.carriers_)), controlImages_(std::move(other.controlImages_)),
      invocationContext_(other.invocationContext_), deviceResident_(other.deviceResident_) {
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        rebindLogicalDescriptor(static_cast<uint32_t>(value));
        for (Carrier &carrier : carriers_[value]) {
            if (carrier.argument.kind != VERNON_PROGRAM_TENSOR)
                continue;
            carrier.argument.tensor.shape = carrier.shape.empty() ? nullptr : carrier.shape.data();
            carrier.argument.tensor.byte_strides = carrier.strides.empty() ? nullptr : carrier.strides.data();
        }
    }
}

bool LogicalValueFrame::materializeDevice(VernonRuntimeContext &context, const std::vector<char> &required,
                                          std::string &error) {
    if (required.size() != logicalArguments_.size()) {
        error = "Program invocation frame requirement set does not match its values";
        return false;
    }
    deviceUploads_.clear();
    struct HostBacking {
        const void *data{};
        size_t bytes{};
        std::shared_ptr<gpu::DeviceBuffer> device;
    };
    std::unordered_map<uintptr_t, HostBacking> backings;
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        if (!required[value])
            continue;
        const VernonProgramArgument &argument = logicalArguments_[value];
        if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
            !argument.tensor.host_data || !argument.tensor.byte_size)
            continue;
        const uintptr_t identity = reinterpret_cast<uintptr_t>(argument.tensor.host_data);
        HostBacking &backing = backings[identity];
        backing.data = argument.tensor.host_data;
        backing.bytes = std::max(backing.bytes, argument.tensor.byte_offset + argument.tensor.byte_size);
    }
    for (auto &[identity, backing] : backings) {
        (void)identity;
        backing.device = std::make_shared<gpu::DeviceBuffer>(context, backing.bytes);
        if (!backing.device->valid()) {
            error = "Program invocation frame could not allocate a Storage backing";
            return false;
        }
        deviceUploads_.push_back({backing.device->handle(), backing.data, backing.bytes});
    }
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        if (!required[value])
            continue;
        VernonProgramArgument &argument = logicalArguments_[value];
        if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
            !argument.tensor.host_data || !argument.tensor.byte_size)
            continue;
        const auto found = backings.find(reinterpret_cast<uintptr_t>(argument.tensor.host_data));
        if (found == backings.end()) {
            error = "Program invocation frame lost a Storage backing";
            return false;
        }
        logicalBuffers_[value] = found->second.device;
        VernonRuntimeProviderResourceReference reference{};
        if (!found->second.device->reference(reference)) {
            error = "Program invocation frame could not reference a Storage backing";
            return false;
        }
        argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        argument.tensor.resource = reference;
    }
    deviceResident_ = true;
    return true;
}

bool LogicalValueFrame::restoreDeviceValuesFromHost(const std::vector<char> &required, std::string &error) {
    if (required.size() != logicalArguments_.size() || hostValues_.size() != logicalArguments_.size()) {
        error = "Program device restore set does not match its values";
        return false;
    }
    std::vector<const gpu::DeviceBuffer *> restored;
    for (size_t value = 0; value < required.size(); ++value) {
        if (!required[value] || !logicalBuffers_[value] ||
            std::find(restored.begin(), restored.end(), logicalBuffers_[value].get()) != restored.end())
            continue;
        const VernonTensorView &host = hostValues_[value].argument.tensor;
        if (!host.host_data || host.byte_offset || host.byte_size != logicalBuffers_[value]->size() ||
            !logicalBuffers_[value]->upload(host.host_data, host.byte_size)) {
            error = "Program invocation frame cannot restore a Storage backing";
            return false;
        }
        restored.push_back(logicalBuffers_[value].get());
    }
    return true;
}

bool LogicalValueFrame::adoptRetainedValue(uint32_t value, const LogicalValueFrame &retained, std::string &error) {
    if (value >= logicalArguments_.size() || value >= retained.logicalArguments_.size()) {
        error = "Program retained Value exceeds its invocation frame";
        return false;
    }
    const VernonProgramArgument &retainedArgument = retained.logicalArguments_[value];
    const bool externalResource =
        retainedArgument.kind != VERNON_PROGRAM_TENSOR || retainedArgument.tensor.storage == VERNON_TENSOR_RHI_RESOURCE;
    if (retained.logicalBuffers_[value] || externalResource) {
        if (logicalBuffers_[value]) {
            const VernonRhiBuffer replaced = logicalBuffers_[value]->handle();
            deviceUploads_.erase(std::remove_if(deviceUploads_.begin(), deviceUploads_.end(),
                                                [&](const ProgramDeviceUpload &upload) {
                                                    return upload.destination.index == replaced.index &&
                                                           upload.destination.generation == replaced.generation;
                                                }),
                                 deviceUploads_.end());
        }
        logicalBuffers_[value] = retained.logicalBuffers_[value];
        logicalArguments_[value] = retainedArgument;
    }
    for (size_t index = 0; index < carriers_[value].size(); ++index)
        if (retained.carriers_[value][index].buffer)
            carriers_[value][index] = retained.carriers_[value][index];
    if (value < hostValues_.size() && value < retained.hostValues_.size()) {
        hostValues_[value] = retained.hostValues_[value];
        if (!hostValues_[value].owned.empty() && hostValues_[value].argument.kind == VERNON_PROGRAM_TENSOR)
            hostValues_[value].argument.tensor.host_data = hostValues_[value].owned.data();
        if (!retained.logicalBuffers_[value])
            logicalArguments_[value] = hostValues_[value].argument;
    }
    rebindLogicalDescriptor(value);
    return true;
}

void LogicalValueFrame::retainOnly(const std::vector<char> &retained) {
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        if (value < retained.size() && retained[value]) {
            LogicalProgramValue &host = hostValues_[value];
            if (!logicalBuffers_[value] && host.owned.empty() && host.argument.kind == VERNON_PROGRAM_TENSOR &&
                host.argument.tensor.storage == VERNON_TENSOR_HOST && host.argument.tensor.host_data &&
                host.argument.tensor.byte_size) {
                const auto *data = static_cast<const uint8_t *>(host.argument.tensor.host_data);
                host.owned.assign(data, data + host.argument.tensor.byte_size);
                host.argument.tensor.host_data = host.owned.data();
                logicalArguments_[value] = host.argument;
            }
            continue;
        }
        hostValues_[value] = {};
        logicalArguments_[value] = {};
        logicalBuffers_[value].reset();
        carriers_[value] = {};
    }
    deviceUploads_.clear();
}

void LogicalValueFrame::rebindLogicalDescriptor(uint32_t value) {
    if (value >= hostValues_.size() || value >= logicalArguments_.size())
        return;
    LogicalProgramValue &host = hostValues_[value];
    if (host.argument.kind != VERNON_PROGRAM_TENSOR || logicalArguments_[value].kind != VERNON_PROGRAM_TENSOR)
        return;
    if (!host.owned.empty() && host.argument.tensor.storage == VERNON_TENSOR_HOST) {
        host.argument.tensor.host_data = host.owned.data();
        logicalArguments_[value].tensor.host_data = host.owned.data();
    }
    if (host.concreteShape) {
        host.argument.tensor.rank = static_cast<uint32_t>(host.concreteShape->size());
        host.argument.tensor.shape = host.concreteShape->data();
        logicalArguments_[value].tensor.rank = static_cast<uint32_t>(host.concreteShape->size());
        logicalArguments_[value].tensor.shape = host.concreteShape->data();
    }
    if (!host.strides.empty()) {
        host.argument.tensor.byte_strides = host.strides.data();
        logicalArguments_[value].tensor.byte_strides = host.strides.data();
    }
}

bool LogicalValueFrame::allocateCarrier(VernonRuntimeContext &context, uint32_t value,
                                        const program::TargetBinding &binding, size_t byteSize,
                                        std::vector<uint64_t> shape, std::vector<int64_t> strides, std::string &error) {
    if (value >= carriers_.size() || !byteSize || shape.size() != strides.size()) {
        error = "Program carrier allocation has an invalid value or layout";
        return false;
    }
    Carrier *slot = binding.tapeCarrier ? carrier(value, *binding.tapeCarrier) : nullptr;
    if (!slot) {
        error = "Program carrier allocation has an unsupported semantic";
        return false;
    }
    slot->buffer = std::make_shared<gpu::DeviceBuffer>(context, byteSize);
    if (!slot->buffer->valid()) {
        error = "Program carrier allocation exceeds available device memory";
        return false;
    }
    slot->shape = std::move(shape);
    slot->strides = std::move(strides);
    slot->argument = {};
    slot->argument.kind = VERNON_PROGRAM_TENSOR;
    slot->argument.tensor.struct_size = sizeof(VernonTensorView);
    slot->argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    if (!slot->buffer->reference(slot->argument.tensor.resource)) {
        error = "Program carrier allocation could not create a backend reference";
        return false;
    }
    slot->argument.tensor.element_layout = pipelineValueLayout(binding.elementLayout);
    slot->argument.tensor.access = valueAccess(binding.access);
    slot->argument.tensor.rank = static_cast<uint32_t>(slot->shape.size());
    slot->argument.tensor.shape = slot->shape.empty() ? nullptr : slot->shape.data();
    slot->argument.tensor.byte_strides = slot->strides.empty() ? nullptr : slot->strides.data();
    slot->argument.tensor.byte_size = byteSize;
    deviceResident_ = true;
    return true;
}

bool LogicalValueFrame::uploadCarrier(uint32_t value, program_plan::TapeCarrier carrierKind, const void *data,
                                      size_t byteSize, std::string &error) {
    Carrier *slot = carrier(value, carrierKind);
    if (!slot || !slot->buffer || byteSize > slot->buffer->size() || !slot->buffer->upload(data, byteSize)) {
        error = "Program carrier upload exceeds its device allocation";
        return false;
    }
    return true;
}

bool LogicalValueFrame::downloadCarrier(uint32_t value, program_plan::TapeCarrier carrierKind, void *data,
                                        size_t byteSize, std::string &error) const {
    const Carrier *slot = carrier(value, carrierKind);
    if (!slot || !slot->buffer || byteSize > slot->buffer->size() || !slot->buffer->download(data, byteSize)) {
        error = "Program carrier readback exceeds its device allocation";
        return false;
    }
    return true;
}

bool LogicalValueFrame::downloadLogicalToHost(const std::vector<char> &required, std::string &error) const {
    if (required.size() != logicalArguments_.size()) {
        error = "Program device readback set does not match its values";
        return false;
    }
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        if (!required[value] || !logicalBuffers_[value])
            continue;
        const LogicalProgramValue &host = hostValues_[value];
        if (host.argument.kind != VERNON_PROGRAM_TENSOR || !host.argument.tensor.host_data ||
            !logicalBuffers_[value]->download(host.argument.tensor.byte_offset,
                                              const_cast<void *>(host.argument.tensor.host_data),
                                              host.argument.tensor.byte_size)) {
            error = "Program invocation frame could not download a logical value";
            return false;
        }
    }
    return true;
}

bool resolveProgramControl(const program::Program &program, const std::vector<LogicalProgramValue> &hostValues,
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
    if (valueId >= hostValues.size())
        return error = "Program dispatch control references an unavailable value", false;
    const LogicalProgramValue &host = hostValues[valueId];
    const VernonProgramArgument &argument = host.argument;
    if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
        !argument.tensor.host_data || argument.tensor.byte_offset > argument.tensor.byte_size)
        return error = "Program dispatch control Value " + std::to_string(valueId) +
                       " is not a retained host scalar (kind " + std::to_string(argument.kind) + ", storage " +
                       std::to_string(argument.tensor.storage) + ", bytes " +
                       std::to_string(argument.tensor.byte_size) + ", owned " + std::to_string(host.owned.size()) + ")",
               false;
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

bool LogicalValueFrame::resolveControl(const program::Program &program, const program::ControlComponent &control,
                                       uint64_t &value, std::string &error) const {
    return resolveProgramControl(program, hostValues_, control, value, error);
}

bool LogicalValueFrame::bindControlImageStorage(const program::Program &program, uint32_t storage,
                                                VernonRuntimeProviderResourceReference view, std::string &error) {
    if (!view.identity || !view.resource.value)
        return error = "Program attachment Storage has no runtime image view", false;
    if (storage >= program.storages.size())
        return error = "Program attachment access references an unknown Storage", false;
    return controlImages_.emplace(storage, view).second || controlImages_.at(storage).identity == view.identity ||
           (error = "Program Storage is projected from multiple runtime image views", false);
}

bool LogicalValueFrame::materializeNodeArguments(const program::Program &program, const program::Node &node,
                                                 const VernonResolvedProgramNode &nodePlan,
                                                 MaterializedNodeFrame &output, std::string &error) const {
    if (!nodePlan.pipeline || nodePlan.bindings.size() != nodePlan.pipeline->variant.parameters.size()) {
        error = "resolved Program stage has an invalid binding plan";
        return false;
    }
    output = {};
    output.arguments.reserve(nodePlan.bindings.size());
    output.shapes.reserve(nodePlan.bindings.size());
    output.strides.reserve(nodePlan.bindings.size());
    uint64_t dispatchInvocations = 1;
    if (program::executionKind(node) == program::ExecutionKind::Compute) {
        for (const program::ControlComponent &control : program::computeOperation(node).workgroups) {
            uint64_t extent{};
            if (!resolveControl(program, control, extent, error))
                return false;
            if (!extent || dispatchInvocations > std::numeric_limits<uint64_t>::max() / extent) {
                error = "Program dispatch invocation count overflows";
                return false;
            } else {
                dispatchInvocations *= extent;
            }
        }
    }
    for (uint32_t extent :
         {nodePlan.pipeline->workgroupSize.x, nodePlan.pipeline->workgroupSize.y, nodePlan.pipeline->workgroupSize.z})
        if (!extent || dispatchInvocations > std::numeric_limits<uint64_t>::max() / extent) {
            error = "Program dispatch invocation count overflows";
            return false;
        } else {
            dispatchInvocations *= extent;
        }
    for (size_t parameterIndex = 0; parameterIndex < nodePlan.bindings.size(); ++parameterIndex) {
        const Parameter &parameter = nodePlan.pipeline->variant.parameters[parameterIndex];
        const VernonProgramStageBinding &binding = nodePlan.bindings[parameterIndex];
        if (binding.value >= logicalArguments_.size() || binding.value >= program.values.size()) {
            error = "resolved Program stage binding exceeds the invocation frame";
            return false;
        }
        VernonProgramArgument controlImage{};
        const VernonProgramArgument *source = nullptr;
        const program::Value &programValue = program.values[binding.value];
        const bool requiresHostProjection =
            std::any_of(parameter.uses.begin(), parameter.uses.end(), [](const ParameterUse &use) {
                return !use.tensorViewDescriptor && (use.interfaceKind == "value" || use.interfaceKind == "uniform" ||
                                                     use.interfaceKind == "result");
            });
        const auto image = programValue.storage ? controlImages_.find(*programValue.storage) : controlImages_.end();
        if (image != controlImages_.end()) {
            controlImage.slot = binding.value;
            controlImage.kind = VERNON_PROGRAM_IMAGE;
            controlImage.image.view = image->second;
            source = &controlImage;
        } else if (requiresHostProjection && binding.value < hostValues_.size() &&
                   hostValues_[binding.value].ownership == ProgramValueOwnership::BorrowedHost) {
            source = &hostValues_[binding.value].argument;
        } else {
            source = argument(binding.value, binding.target ? &*binding.target : nullptr);
        }
        if (!source) {
            error = "resolved Program stage binding has no physical carrier";
            return false;
        }
        VernonProgramArgument materialized = *source;
        materialized.slot = parameter.slot;
        if (requiresHostProjection && materialized.kind == VERNON_PROGRAM_TENSOR &&
            materialized.tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
            if (!logicalBuffers_[binding.value]) {
                error = "resolved Program inline value has no logical device backing";
                return false;
            }
            output.hostStorage.emplace_back(materialized.tensor.byte_size);
            if (!logicalBuffers_[binding.value]->download(materialized.tensor.byte_offset,
                                                          output.hostStorage.back().data(),
                                                          output.hostStorage.back().size())) {
                error = "resolved Program inline value could not be downloaded";
                return false;
            }
            materialized.tensor.storage = VERNON_TENSOR_HOST;
            materialized.tensor.host_data = output.hostStorage.back().data();
            materialized.tensor.byte_offset = 0;
        }
        if (binding.target && materialized.kind == VERNON_PROGRAM_TENSOR) {
            materialized.tensor.access = valueAccess(binding.target->access);
            materialized.tensor.element_layout = pipelineValueLayout(binding.target->elementLayout);
        }
        output.shapes.emplace_back();
        output.strides.emplace_back();
        if (materialized.kind == VERNON_PROGRAM_TENSOR && materialized.tensor.rank) {
            const shape::DeclaredShape declared =
                shape::decodeRuntimeContractShape(program.values[binding.value].shape);
            if (binding.value < hostValues_.size() && hostValues_[binding.value].concreteShape) {
                output.shapes.back() = *hostValues_[binding.value].concreteShape;
                output.strides.back() = hostValues_[binding.value].strides;
            } else if (!shape::isConcrete(declared)) {
                error = "resolved Program TensorView shape was not concrete before physical materialization";
                return false;
            } else if (!materialized.tensor.shape || !materialized.tensor.byte_strides) {
                error = "resolved Program TensorView has incomplete logical layout metadata";
                return false;
            } else {
                output.shapes.back().assign(materialized.tensor.shape,
                                            materialized.tensor.shape + materialized.tensor.rank);
                output.strides.back().assign(materialized.tensor.byte_strides,
                                             materialized.tensor.byte_strides + materialized.tensor.rank);
            }
            if (output.shapes.back().size() != output.strides.back().size()) {
                error = "resolved Program TensorView runtime shape has no matching strides";
                return false;
            }
            materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
            materialized.tensor.shape = output.shapes.back().data();
            materialized.tensor.byte_strides = output.strides.back().data();
        }
        if (binding.target && binding.target->semantic == program::CarrierSemantic::Tape) {
            if (binding.target->tapeCarrier &&
                (*binding.target->tapeCarrier == program_plan::TapeCarrier::ReplayStatus ||
                 *binding.target->tapeCarrier == program_plan::TapeCarrier::LaunchMetadata)) {
                if (const std::optional<shape::ConcreteShape> concrete = shape::concrete(binding.target->shape)) {
                    size_t elements = 0;
                    if (!shape::checkedElementCount(*concrete, elements) || !binding.target->elementLayout.byteSize ||
                        elements > std::numeric_limits<size_t>::max() / binding.target->elementLayout.byteSize ||
                        elements * binding.target->elementLayout.byteSize > materialized.tensor.byte_size) {
                        error = "resolved Program fixed tape carrier is smaller than its stage ABI";
                        return false;
                    }
                    materialized.tensor.byte_size = elements * binding.target->elementLayout.byteSize;
                }
            }
            gpu::InternalBufferView view;
            if (materialized.kind != VERNON_PROGRAM_TENSOR ||
                !gpu::materializeInternalBufferView(binding.target->shape, binding.target->elementLayout,
                                                    materialized.tensor.byte_size, view)) {
                error = "resolved Program carrier '" + binding.target->name + "' (role " + binding.target->role +
                        ", bytes " + std::to_string(materialized.tensor.byte_size) +
                        ") has an incompatible stage-local view";
                return false;
            }
            output.shapes.back() = std::move(view.shape);
            output.strides.back() = std::move(view.strides);
            materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
            materialized.tensor.shape = output.shapes.back().empty() ? nullptr : output.shapes.back().data();
            materialized.tensor.byte_strides = output.strides.back().empty() ? nullptr : output.strides.back().data();
        }
        if (binding.leaf) {
            const program::Value &slot = program.values[binding.value];
            if (materialized.kind != VERNON_PROGRAM_TENSOR || !slot.layout ||
                *binding.leaf >= slot.layout->leaves.size()) {
                error = "resolved Program stage leaf exceeds the canonical Value ABI";
                return false;
            }
            const ValueLayout valueLayout = program::materializeValueLayout(*slot.layout, slot.type);
            const ValueLeaf &leaf = valueLayout.leaves[*binding.leaf];
            const ValueLayout &parameterLayout =
                parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
            materialized.tensor.byte_offset += leaf.byteOffset;
            materialized.tensor.element_layout = pipelineValueLayout(parameterLayout);
            const std::optional<shape::DeclaredShape> projectionShape =
                logicalProjectionShape(parameter, binding.target ? &*binding.target : nullptr);
            std::vector<uint64_t> projectedShape;
            std::vector<int64_t> projectedStrides;
            std::vector<uint64_t> compactShape;
            std::vector<int64_t> compactStrides;
            if (!projectionShape || !parameterLayout.byteSize ||
                !shape::materializeLeafProjection(output.shapes.back(), output.strides.back(), *projectionShape,
                                                  parameterLayout.byteSize, projectedShape, projectedStrides) ||
                !shape::materializeCompactProjection(output.shapes.back(), *projectionShape, parameterLayout.byteSize,
                                                     compactShape, compactStrides)) {
                error = "Program aggregate leaf projection has an incompatible physical shape";
                return false;
            }
            if (projectedShape != compactShape || projectedStrides != compactStrides) {
                size_t elements = 0;
                size_t byteSize = 0;
                if (!shape::checkedElementCount(compactShape, elements) ||
                    elements > std::numeric_limits<size_t>::max() / parameterLayout.byteSize ||
                    !(byteSize = elements * parameterLayout.byteSize)) {
                    error = "Program aggregate leaf endpoint size overflows";
                    return false;
                }
                VernonTensorView canonical = materialized.tensor;
                canonical.rank = static_cast<uint32_t>(projectedShape.size());
                canonical.shape = projectedShape.empty() ? nullptr : projectedShape.data();
                canonical.byte_strides = projectedStrides.empty() ? nullptr : projectedStrides.data();

                output.shapes.back() = std::move(compactShape);
                output.strides.back() = std::move(compactStrides);
                materialized.tensor.byte_offset = 0;
                materialized.tensor.byte_size = byteSize;
                materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
                materialized.tensor.shape = output.shapes.back().empty() ? nullptr : output.shapes.back().data();
                materialized.tensor.byte_strides =
                    output.strides.back().empty() ? nullptr : output.strides.back().data();

                const bool reads = materialized.tensor.access != VERNON_ACCESS_WRITE;
                const bool writes = materialized.tensor.access != VERNON_ACCESS_READ;
                std::vector<TensorCopyRegion> regions;
                if (canonical.storage == VERNON_TENSOR_HOST) {
                    output.hostStorage.emplace_back(byteSize);
                    materialized.tensor.host_data = output.hostStorage.back().data();
                    VernonTensorView compact = materialized.tensor;
                    if (!planTensorCopy(canonical, compact, regions, error))
                        return false;
                    const auto *canonicalBytes = static_cast<const uint8_t *>(canonical.host_data);
                    auto *compactBytes = output.hostStorage.back().data();
                    for (const TensorCopyRegion &region : regions) {
                        if (reads)
                            output.copiesBefore.push_back({canonicalBytes, compactBytes, region.sourceOffset,
                                                           region.destinationOffset, region.size});
                        if (writes)
                            output.copiesAfter.push_back({compactBytes, const_cast<uint8_t *>(canonicalBytes),
                                                          region.destinationOffset, region.sourceOffset, region.size});
                    }
                } else if (canonical.storage == VERNON_TENSOR_RHI_RESOURCE && nodePlan.pipeline->context) {
                    auto storage = std::make_shared<gpu::DeviceBuffer>(*nodePlan.pipeline->context, byteSize);
                    if (!storage->valid() || !storage->reference(materialized.tensor.resource))
                        return error = "Program aggregate leaf endpoint allocation failed", false;
                    output.deviceStorage.push_back(storage);
                    VernonTensorView compact = materialized.tensor;
                    if (!planTensorCopy(canonical, compact, regions, error))
                        return false;
                    VernonRhiBuffer canonicalBuffer{};
                    if (!resolveBackendRhiBufferReference(*nodePlan.pipeline->context, canonical.resource,
                                                          canonicalBuffer))
                        return error = "Program aggregate leaf endpoint has no RHI Storage backing", false;
                    for (const TensorCopyRegion &region : regions) {
                        if (reads)
                            output.deviceCopiesBefore.push_back({canonicalBuffer, storage->handle(),
                                                                 canonical.resource.offset + region.sourceOffset,
                                                                 region.destinationOffset, region.size});
                        if (writes)
                            output.deviceCopiesAfter.push_back(
                                {storage->handle(), canonicalBuffer, region.destinationOffset,
                                 canonical.resource.offset + region.sourceOffset, region.size});
                    }
                } else {
                    error = "Program aggregate leaf endpoint has no materializable Storage backing";
                    return false;
                }
            } else {
                output.shapes.back() = std::move(projectedShape);
                output.strides.back() = std::move(projectedStrides);
            }
            materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
            materialized.tensor.shape = output.shapes.back().empty() ? nullptr : output.shapes.back().data();
            materialized.tensor.byte_strides = output.strides.back().empty() ? nullptr : output.strides.back().data();
        }
        if (binding.target && binding.target->viewTransform) {
            if (materialized.kind != VERNON_PROGRAM_TENSOR || !materialized.tensor.shape ||
                !materialized.tensor.byte_strides ||
                parameter.shape.size() != binding.target->viewTransform->axes.size()) {
                error = "Program view transform has an incompatible logical view";
                return false;
            }
            std::vector<uint64_t> &shape = output.shapes.back();
            std::vector<int64_t> &strides = output.strides.back();
            const std::vector<uint64_t> logicalShape = shape;
            const std::vector<int64_t> logicalStrides = strides;
            shape.clear();
            strides.clear();
            shape.reserve(binding.target->viewTransform->axes.size());
            strides.reserve(binding.target->viewTransform->axes.size());
            for (size_t physicalAxis = 0; physicalAxis < binding.target->viewTransform->axes.size(); ++physicalAxis) {
                const program::ViewAxisTransform &axis = binding.target->viewTransform->axes[physicalAxis];
                if (axis.source == program::ViewAxisSource::Constant) {
                    if (!axis.constantExtent || !axis.zeroStride) {
                        error = "Program constant view axis has no storage mapping";
                        return false;
                    }
                    shape.push_back(axis.constantExtent);
                    strides.push_back(0);
                    continue;
                }
                if (axis.source == program::ViewAxisSource::InvocationLinearCarrier) {
                    if (!axis.zeroStride) {
                        error = "Program dispatch-derived view axis must broadcast";
                        return false;
                    }
                    shape.push_back(dispatchInvocations);
                    strides.push_back(0);
                    continue;
                }
                if (axis.logicalAxis >= logicalShape.size()) {
                    error = "Program view transform references an unknown logical axis";
                    return false;
                }
                const uint64_t physicalExtent = parameter.shape[physicalAxis];
                const uint64_t logicalExtent = logicalShape[axis.logicalAxis];
                if (physicalExtent && logicalExtent && physicalExtent != logicalExtent) {
                    error = "Program view transform for '" + parameter.name + "' axis " + std::to_string(physicalAxis) +
                            " expects extent " + std::to_string(physicalExtent) + " but the logical view has " +
                            std::to_string(logicalExtent);
                    return false;
                }
                shape.push_back(logicalExtent ? logicalExtent : physicalExtent);
                strides.push_back(axis.zeroStride ? 0 : logicalStrides[axis.logicalAxis]);
            }
            materialized.tensor.rank = static_cast<uint32_t>(shape.size());
            materialized.tensor.shape = shape.data();
            materialized.tensor.byte_strides = strides.data();
        }
        output.arguments.push_back(materialized);
    }
    return true;
}

const VernonProgramArgument *LogicalValueFrame::argument(uint32_t value, const program::TargetBinding *binding) const {
    if (value >= logicalArguments_.size())
        return nullptr;
    if (binding) {
        if (const Carrier *physical = binding->tapeCarrier ? carrier(value, *binding->tapeCarrier) : nullptr)
            if (physical->buffer)
                return &physical->argument;
    }
    return &logicalArguments_[value];
}

VernonProgramArgument *LogicalValueFrame::argument(uint32_t value, const program::TargetBinding *binding) {
    return const_cast<VernonProgramArgument *>(static_cast<const LogicalValueFrame &>(*this).argument(value, binding));
}

VernonRhiBuffer LogicalValueFrame::buffer(uint32_t value, const program::TargetBinding *binding) const {
    if (binding)
        if (const Carrier *physical = binding->tapeCarrier ? carrier(value, *binding->tapeCarrier) : nullptr)
            if (physical->buffer)
                return physical->buffer->handle();
    if (value < logicalBuffers_.size() && logicalBuffers_[value])
        return logicalBuffers_[value]->handle();
    if (value < carriers_.size())
        for (const Carrier &physical : carriers_[value])
            if (physical.buffer)
                return physical.buffer->handle();
    return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

size_t LogicalValueFrame::carrierIndex(program_plan::TapeCarrier carrierKind) {
    if (carrierKind == program_plan::TapeCarrier::TapeData)
        return 0;
    if (carrierKind == program_plan::TapeCarrier::ReplaySegment)
        return 1;
    if (carrierKind == program_plan::TapeCarrier::ReplayStatus)
        return 2;
    if (carrierKind == program_plan::TapeCarrier::LaunchMetadata)
        return 3;
    return invalidCarrier;
}

LogicalValueFrame::Carrier *LogicalValueFrame::carrier(uint32_t value, program_plan::TapeCarrier carrierKind) {
    const size_t index = carrierIndex(carrierKind);
    return value < carriers_.size() && index != invalidCarrier ? &carriers_[value][index] : nullptr;
}

const LogicalValueFrame::Carrier *LogicalValueFrame::carrier(uint32_t value,
                                                             program_plan::TapeCarrier carrierKind) const {
    const size_t index = carrierIndex(carrierKind);
    return value < carriers_.size() && index != invalidCarrier ? &carriers_[value][index] : nullptr;
}

} // namespace vernon::runtime::ad
