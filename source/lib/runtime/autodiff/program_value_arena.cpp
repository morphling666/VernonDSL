#include "program_value_arena.h"

#include "runtime/runtime_state.h"
#include "runtime_gpu_argument_binding.h"

#include <algorithm>
#include <cstdint>
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

} // namespace

ProgramInvocationFrame::ProgramInvocationFrame(const std::vector<VernonPipelineArgument> &hostArguments)
    : logicalArguments_(hostArguments), logicalBuffers_(hostArguments.size()), carriers_(hostArguments.size()) {}

ProgramInvocationFrame::ProgramInvocationFrame(std::vector<ProgramHostValue> hostValues)
    : hostValues_(std::move(hostValues)), logicalArguments_(hostValues_.size()), logicalBuffers_(hostValues_.size()),
      carriers_(hostValues_.size()) {
    for (size_t value = 0; value < hostValues_.size(); ++value) {
        logicalArguments_[value] = hostValues_[value].argument;
        rebindLogicalDescriptor(static_cast<uint32_t>(value));
    }
}

bool ProgramInvocationFrame::materializeDevice(VernonRuntimeContext &context, const std::vector<char> &required,
                                               std::string &error) {
    if (required.size() != logicalArguments_.size()) {
        error = "Program invocation frame requirement set does not match its values";
        return false;
    }
    struct HostBacking {
        const void *data{};
        size_t bytes{};
        std::shared_ptr<gpu::DeviceBuffer> device;
    };
    std::unordered_map<uintptr_t, HostBacking> backings;
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        if (!required[value])
            continue;
        const VernonPipelineArgument &argument = logicalArguments_[value];
        if (value < hostValues_.size() && (hostValues_[value].ownership == ProgramValueOwnership::BorrowedHost ||
                                           hostValues_[value].ownership == ProgramValueOwnership::BorrowedDevice))
            continue;
        if (argument.kind != VERNON_PIPELINE_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
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
        if (!backing.device->valid() || !backing.device->upload(backing.data, backing.bytes)) {
            error = "Program invocation frame could not allocate or upload a Storage backing";
            return false;
        }
    }
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        if (!required[value])
            continue;
        VernonPipelineArgument &argument = logicalArguments_[value];
        if (value < hostValues_.size() && (hostValues_[value].ownership == ProgramValueOwnership::BorrowedHost ||
                                           hostValues_[value].ownership == ProgramValueOwnership::BorrowedDevice))
            continue;
        if (argument.kind != VERNON_PIPELINE_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
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

bool ProgramInvocationFrame::restoreDeviceValuesFromHost(const std::vector<char> &required, std::string &error) {
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

bool ProgramInvocationFrame::adoptRetainedValue(uint32_t value, const ProgramInvocationFrame &retained,
                                                std::string &error) {
    if (value >= logicalArguments_.size() || value >= retained.logicalArguments_.size()) {
        error = "Program retained Value exceeds its invocation frame";
        return false;
    }
    const VernonPipelineArgument &retainedArgument = retained.logicalArguments_[value];
    const bool externalResource = retainedArgument.kind != VERNON_PIPELINE_TENSOR ||
                                  retainedArgument.tensor.storage == VERNON_TENSOR_RHI_RESOURCE;
    if (retained.logicalBuffers_[value] || externalResource) {
        logicalBuffers_[value] = retained.logicalBuffers_[value];
        logicalArguments_[value] = retainedArgument;
    }
    for (size_t index = 0; index < carriers_[value].size(); ++index)
        if (retained.carriers_[value][index].buffer)
            carriers_[value][index] = retained.carriers_[value][index];
    if (value < hostValues_.size() && value < retained.hostValues_.size()) {
        hostValues_[value].concreteShape = retained.hostValues_[value].concreteShape;
        hostValues_[value].strides = retained.hostValues_[value].strides;
    }
    rebindLogicalDescriptor(value);
    return true;
}

void ProgramInvocationFrame::rebindLogicalDescriptor(uint32_t value) {
    if (value >= hostValues_.size() || value >= logicalArguments_.size())
        return;
    ProgramHostValue &host = hostValues_[value];
    if (host.argument.kind != VERNON_PIPELINE_TENSOR || logicalArguments_[value].kind != VERNON_PIPELINE_TENSOR)
        return;
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

bool ProgramInvocationFrame::allocateCarrier(VernonRuntimeContext &context, uint32_t value,
                                             const program::TargetBinding &binding, size_t byteSize,
                                             std::vector<uint64_t> shape, std::vector<int64_t> strides,
                                             std::string &error) {
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
    slot->argument.kind = VERNON_PIPELINE_TENSOR;
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

bool ProgramInvocationFrame::uploadCarrier(uint32_t value, program_plan::TapeCarrier carrierKind, const void *data,
                                           size_t byteSize, std::string &error) {
    Carrier *slot = carrier(value, carrierKind);
    if (!slot || !slot->buffer || byteSize > slot->buffer->size() || !slot->buffer->upload(data, byteSize)) {
        error = "Program carrier upload exceeds its device allocation";
        return false;
    }
    return true;
}

bool ProgramInvocationFrame::downloadCarrier(uint32_t value, program_plan::TapeCarrier carrierKind, void *data,
                                             size_t byteSize, std::string &error) const {
    const Carrier *slot = carrier(value, carrierKind);
    if (!slot || !slot->buffer || byteSize > slot->buffer->size() || !slot->buffer->download(data, byteSize)) {
        error = "Program carrier readback exceeds its device allocation";
        return false;
    }
    return true;
}

bool ProgramInvocationFrame::downloadLogicalToHost(const std::vector<char> &required, std::string &error) const {
    if (required.size() != logicalArguments_.size()) {
        error = "Program device readback set does not match its values";
        return false;
    }
    for (size_t value = 0; value < logicalArguments_.size(); ++value) {
        if (!required[value] || !logicalBuffers_[value])
            continue;
        const ProgramHostValue &host = hostValues_[value];
        if (host.argument.kind != VERNON_PIPELINE_TENSOR || !host.argument.tensor.host_data ||
            !logicalBuffers_[value]->download(host.argument.tensor.byte_offset,
                                              const_cast<void *>(host.argument.tensor.host_data),
                                              host.argument.tensor.byte_size)) {
            error = "Program invocation frame could not download a logical value";
            return false;
        }
    }
    return true;
}

bool ProgramInvocationFrame::materializeNodeArguments(const program::Program &program, const program::Node &node,
                                                      const VernonResolvedProgramStage &stage,
                                                      MaterializedProgramArguments &output, std::string &error) const {
    if (!stage.pipeline || stage.bindings.size() != stage.pipeline->variant.parameters.size()) {
        error = "resolved Program stage has an invalid binding plan";
        return false;
    }
    output = {};
    output.arguments.reserve(stage.bindings.size());
    output.shapes.reserve(stage.bindings.size());
    output.strides.reserve(stage.bindings.size());
    uint64_t dispatchInvocations = 1;
    for (const program::ControlComponent &control : node.compute.workgroups) {
        if (control.kind != program::ControlKind::Static) {
            error = "Program dispatch requires a resolved static invocation shape";
            return false;
        }
        const uint64_t extent = control.value;
        if (!extent || dispatchInvocations > std::numeric_limits<uint64_t>::max() / extent) {
            error = "Program dispatch invocation count overflows";
            return false;
        } else {
            dispatchInvocations *= extent;
        }
    }
    for (uint32_t extent :
         {stage.pipeline->workgroupSize.x, stage.pipeline->workgroupSize.y, stage.pipeline->workgroupSize.z})
        if (!extent || dispatchInvocations > std::numeric_limits<uint64_t>::max() / extent) {
            error = "Program dispatch invocation count overflows";
            return false;
        } else {
            dispatchInvocations *= extent;
        }
    for (size_t parameterIndex = 0; parameterIndex < stage.bindings.size(); ++parameterIndex) {
        const Parameter &parameter = stage.pipeline->variant.parameters[parameterIndex];
        const VernonProgramStageBinding &binding = stage.bindings[parameterIndex];
        if (binding.value >= logicalArguments_.size() || binding.value >= program.values.size()) {
            error = "resolved Program stage binding exceeds the invocation frame";
            return false;
        }
        const VernonPipelineArgument *source = argument(binding.value, binding.target ? &*binding.target : nullptr);
        if (!source) {
            error = "resolved Program stage binding has no physical carrier";
            return false;
        }
        VernonPipelineArgument materialized = *source;
        materialized.slot = parameter.slot;
        if (binding.target && materialized.kind == VERNON_PIPELINE_TENSOR) {
            materialized.tensor.access = valueAccess(binding.target->access);
            materialized.tensor.element_layout = pipelineValueLayout(binding.target->elementLayout);
        }
        output.shapes.emplace_back();
        output.strides.emplace_back();
        if (materialized.kind == VERNON_PIPELINE_TENSOR && materialized.tensor.rank) {
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
            const shape::DeclaredShape &declared = binding.target->shape;
            if (const std::optional<shape::ConcreteShape> concrete = shape::concrete(declared)) {
                size_t elements = 0;
                size_t bytes = 0;
                if (!shape::checkedElementCount(*concrete, elements) || !binding.target->elementLayout.byteSize ||
                    elements > std::numeric_limits<size_t>::max() / binding.target->elementLayout.byteSize ||
                    (bytes = elements * binding.target->elementLayout.byteSize) > materialized.tensor.resource.size) {
                    error = "resolved Program carrier stage-local view exceeds its retained allocation";
                    return false;
                }
                materialized.tensor.byte_size = bytes;
            }
            gpu::InternalBufferView view;
            if (materialized.kind != VERNON_PIPELINE_TENSOR ||
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
            if (materialized.kind != VERNON_PIPELINE_TENSOR || !slot.layout ||
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
            const bool wholeElementProjection = valueLayout.leaves.size() == 1 && leaf.path.empty();
            const bool materializedProjection =
                projectionShape && parameterLayout.byteSize &&
                (wholeElementProjection ? shape::materializeLeafProjection(output.shapes.back(), output.strides.back(),
                                                                           *projectionShape, parameterLayout.byteSize,
                                                                           output.shapes.back(), output.strides.back())
                                        : shape::materializeCompactProjection(
                                              output.shapes.back(), *projectionShape, parameterLayout.byteSize,
                                              output.shapes.back(), output.strides.back()));
            if (!materializedProjection) {
                error = "Program aggregate leaf projection has an incompatible physical shape";
                return false;
            }
            materialized.tensor.rank = static_cast<uint32_t>(output.shapes.back().size());
            materialized.tensor.shape = output.shapes.back().empty() ? nullptr : output.shapes.back().data();
            materialized.tensor.byte_strides = output.strides.back().empty() ? nullptr : output.strides.back().data();
        }
        if (binding.target && binding.target->viewTransform) {
            if (materialized.kind != VERNON_PIPELINE_TENSOR || !materialized.tensor.shape ||
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

const VernonPipelineArgument *ProgramInvocationFrame::argument(uint32_t value,
                                                               const program::TargetBinding *binding) const {
    if (value >= logicalArguments_.size())
        return nullptr;
    if (binding) {
        if (const Carrier *physical = binding->tapeCarrier ? carrier(value, *binding->tapeCarrier) : nullptr)
            if (physical->buffer)
                return &physical->argument;
    }
    return &logicalArguments_[value];
}

VernonPipelineArgument *ProgramInvocationFrame::argument(uint32_t value, const program::TargetBinding *binding) {
    return const_cast<VernonPipelineArgument *>(
        static_cast<const ProgramInvocationFrame &>(*this).argument(value, binding));
}

VernonRhiBuffer ProgramInvocationFrame::buffer(uint32_t value, const program::TargetBinding *binding) const {
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

size_t ProgramInvocationFrame::carrierIndex(program_plan::TapeCarrier carrierKind) {
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

ProgramInvocationFrame::Carrier *ProgramInvocationFrame::carrier(uint32_t value,
                                                                 program_plan::TapeCarrier carrierKind) {
    const size_t index = carrierIndex(carrierKind);
    return value < carriers_.size() && index != invalidCarrier ? &carriers_[value][index] : nullptr;
}

const ProgramInvocationFrame::Carrier *ProgramInvocationFrame::carrier(uint32_t value,
                                                                       program_plan::TapeCarrier carrierKind) const {
    const size_t index = carrierIndex(carrierKind);
    return value < carriers_.size() && index != invalidCarrier ? &carriers_[value][index] : nullptr;
}

} // namespace vernon::runtime::ad
