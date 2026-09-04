#include "runtime_gpu_preparation.h"

#include "execution_graph/execution_graph_internal.h"
#include "runtime_gpu_argument_binding.h"
#include "runtime_gpu_commands.h"
#include "runtime_gpu_failure_injection.h"

#include "rhi/rhi_internal.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime::ad::gpu {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool valueMatchesTemplate(const VernonAdValue &value, const ValueAbi &abi) {
    if (value.dtype != abi.dtype || !value.data || value.rank != abi.logicalShape.size() ||
        (value.rank && !value.shape))
        return false;
    for (size_t dimension = 0; dimension < abi.logicalShape.size(); ++dimension)
        if (abi.logicalShape[dimension] && abi.logicalShape[dimension] != value.shape[dimension])
            return false;
    return true;
}

bool materializeValueAbi(ValueAbi &abi, const VernonAdValue &value) {
    if (!valueMatchesTemplate(value, abi))
        return false;
    abi.byteSize = value.size;
    abi.logicalShape.clear();
    if (value.rank)
        abi.logicalShape.assign(value.shape, value.shape + value.rank);
    return true;
}

bool singleLeafParameter(const Parameter &parameter) {
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    return layout && layout->leaves.size() == 1;
}

bool materializePhysicalShape(const Parameter &parameter, const std::vector<uint64_t> &logicalShape,
                              std::vector<uint64_t> &shape, std::vector<int64_t> &strides) {
    const std::vector<uint64_t> logical = logicalShape;
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    if (!layout || layout->leaves.size() != 1 || !layout->byteSize)
        return false;
    const ValueLeaf &leaf = layout->leaves.front();
    size_t expectedRank = parameter.shape.size();
    bool hasDescriptorRank = false;
    for (const ParameterUse &use : parameter.uses) {
        if (!use.tensorViewDescriptor)
            continue;
        if (hasDescriptorRank && expectedRank != use.tensorViewDescriptor->rank)
            return false;
        expectedRank = use.tensorViewDescriptor->rank;
        hasDescriptorRank = true;
    }
    size_t physicalRank = logical.size();
    if (logical.size() != expectedRank) {
        if (logical.size() < leaf.shape.size())
            return false;
        physicalRank = logical.size() - leaf.shape.size();
        for (size_t dimension = 0; dimension < leaf.shape.size(); ++dimension)
            if (leaf.shape[dimension] && leaf.shape[dimension] != logical[physicalRank + dimension])
                return false;
    }
    if (physicalRank > expectedRank)
        return false;
    const auto matchesDescriptorShapes = [&](const std::vector<uint64_t> &candidate) {
        for (const ParameterUse &use : parameter.uses)
            if (use.tensorViewDescriptor)
                for (size_t dimension = 0; dimension < candidate.size(); ++dimension)
                    if (use.shape[dimension] && use.shape[dimension] != candidate[dimension])
                        return false;
        return true;
    };
    std::vector<uint64_t> physical(logical.begin(), logical.begin() + physicalRank);
    size_t physicalElements = 1;
    for (uint64_t extent : physical)
        if (!checkedMultiply(physicalElements, static_cast<size_t>(extent), physicalElements))
            return false;
    shape.clear();
    if (expectedRank > physicalRank)
        for (const ParameterUse &use : parameter.uses) {
            if (!use.tensorViewDescriptor || use.shape.size() != expectedRank)
                continue;
            std::vector<uint64_t> candidate = use.shape;
            size_t fixedElements = 1;
            size_t firstDynamic = candidate.size();
            for (size_t dimension = 0; dimension < candidate.size(); ++dimension) {
                if (!candidate[dimension]) {
                    candidate[dimension] = 1;
                    if (firstDynamic == candidate.size())
                        firstDynamic = dimension;
                }
                if (!checkedMultiply(fixedElements, static_cast<size_t>(candidate[dimension]), fixedElements))
                    return false;
            }
            if (!fixedElements || physicalElements % fixedElements)
                continue;
            const size_t quotient = physicalElements / fixedElements;
            if (firstDynamic == candidate.size()) {
                if (quotient != 1)
                    continue;
            } else {
                candidate[firstDynamic] = quotient;
            }
            if (matchesDescriptorShapes(candidate)) {
                shape = std::move(candidate);
                break;
            }
        }
    if (shape.empty()) {
        shape = physical;
        shape.resize(expectedRank, 1);
    }
    if (!matchesDescriptorShapes(shape)) {
        std::vector<uint64_t> leading(expectedRank - physicalRank, 1);
        leading.insert(leading.end(), physical.begin(), physical.end());
        if (!matchesDescriptorShapes(leading))
            return false;
        shape = std::move(leading);
    }
    if (parameter.source != "direct") {
        if (shape.size() != parameter.shape.size())
            return false;
        for (size_t dimension = 0; dimension < shape.size(); ++dimension)
            if (parameter.shape[dimension] && parameter.shape[dimension] != shape[dimension])
                return false;
    }
    strides.resize(shape.size());
    size_t stride = layout->byteSize;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return false;
        strides[dimension] = static_cast<int64_t>(stride);
        if (!checkedMultiply(stride, static_cast<size_t>(shape[dimension]), stride))
            return false;
    }
    return true;
}

const VernonAdValue *findParameterValue(const VernonAdValueSet &set, const Parameter &parameter) {
    if (!singleLeafParameter(parameter))
        return nullptr;
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    return findValue(set, canonicalValueLeafPath(parameter.name, layout->leaves.front()));
}

const VernonAdValue *findParameterValues(const VernonAdValueSet &set, const Parameter &parameter) {
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    if (!layout || layout->leaves.empty())
        return nullptr;
    const VernonAdValue *representative = nullptr;
    for (const ValueLeaf &leaf : layout->leaves) {
        const VernonAdValue *value = findValue(set, canonicalValueLeafPath(parameter.name, leaf));
        if (!value)
            return nullptr;
        if (!representative)
            representative = value;
    }
    return representative;
}

bool fillTensorView(const Parameter &parameter, DeviceValue &device, VernonTensorView &tensor,
                    VernonRuntimeProviderResourceReference resource) {
    if (!singleLeafParameter(parameter) || !resource.resource.value)
        return false;
    bool existingLayoutMatches = device.shape.size() == device.strides.size();
    for (const ParameterUse &use : parameter.uses)
        if (use.tensorViewDescriptor && (use.tensorViewDescriptor->rank != device.shape.size() ||
                                         (!use.shape.empty() && use.shape.size() != device.shape.size())))
            existingLayoutMatches = false;
        else if (use.tensorViewDescriptor && !use.shape.empty())
            for (size_t dimension = 0; dimension < use.shape.size(); ++dimension)
                if (use.shape[dimension] && use.shape[dimension] != device.shape[dimension])
                    existingLayoutMatches = false;
    if (!existingLayoutMatches) {
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
        if (!materializePhysicalShape(parameter, device.shape, shape, strides))
            return false;
        device.shape = std::move(shape);
        device.strides = std::move(strides);
    }
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    tensor = {};
    tensor.struct_size = sizeof(tensor);
    tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    tensor.resource = resource;
    tensor.element_layout = pipelineValueLayout(layout);
    const std::optional<VernonValueAccess> access = pipelineValueAccess(parameter.access);
    if (!access)
        return false;
    tensor.access = *access;
    tensor.rank = static_cast<uint32_t>(device.shape.size());
    tensor.shape = device.shape.empty() ? nullptr : device.shape.data();
    tensor.byte_strides = device.strides.empty() ? nullptr : device.strides.data();
    tensor.byte_offset = device.byteOffset;
    tensor.byte_size = device.buffer.size();
    return true;
}

bool fillHostTensorView(const Parameter &parameter, HostValue &host, VernonTensorView &tensor) {
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    if (!singleLeafParameter(parameter) || !materializePhysicalShape(parameter, host.shape, shape, strides))
        return false;
    host.shape = std::move(shape);
    host.strides = std::move(strides);
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    tensor = {};
    tensor.struct_size = sizeof(tensor);
    tensor.storage = VERNON_TENSOR_HOST;
    tensor.host_data = host.bytes.data();
    tensor.element_layout = pipelineValueLayout(layout);
    const std::optional<VernonValueAccess> access = pipelineValueAccess(parameter.access);
    if (!access)
        return false;
    tensor.access = *access;
    tensor.rank = static_cast<uint32_t>(host.shape.size());
    tensor.shape = host.shape.empty() ? nullptr : host.shape.data();
    tensor.byte_strides = host.strides.empty() ? nullptr : host.strides.data();
    tensor.byte_size = host.bytes.size();
    return true;
}

bool usesStorage(const Parameter &parameter) {
    return std::any_of(parameter.uses.begin(), parameter.uses.end(),
                       [](const ParameterUse &use) { return use.interfaceKind == "storage"; });
}

bool isAutodiffInternal(const Parameter &parameter) {
    return parameter.autodiffRole == AutodiffResourceRole::Tape ||
           parameter.autodiffRole == AutodiffResourceRole::ReplaySegment ||
           parameter.autodiffRole == AutodiffResourceRole::ReplayStatus ||
           parameter.autodiffRole == AutodiffResourceRole::LaunchMetadata;
}

bool appendDeviceArgument(const Parameter &parameter, DeviceValue &device,
                          std::vector<VernonProgramArgument> &arguments) {
    VernonRuntimeProviderResourceReference resource{};
    if (!device.buffer.reference(resource))
        return false;
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    if (!fillTensorView(parameter, device, argument.tensor, resource))
        return false;
    arguments.push_back(argument);
    return true;
}

bool buildForwardArguments(VernonRuntimeContext &context, const Variant &variant, const VernonAdValueSet &inputs,
                           DeviceValues &working, DeviceValues &retainedDevices, HostValues &retainedHosts,
                           std::vector<VernonProgramArgument> &arguments, std::vector<DeviceBufferUpload> &uploads,
                           bool retainStorageHostCopy) {
    arguments.reserve(variant.parameters.size());
    for (const Parameter &parameter : variant.parameters) {
        if (isAutodiffInternal(parameter))
            continue;
        const VernonAdValue *value = findParameterValue(inputs, parameter);
        if (!value || !value->data || !value->size)
            return false;
        if (!usesStorage(parameter)) {
            auto [host, inserted] = retainedHosts.emplace(parameter.name, HostValue(*value));
            if (!inserted)
                return false;
            VernonProgramArgument argument{};
            argument.slot = parameter.slot;
            argument.kind = VERNON_PROGRAM_TENSOR;
            if (!fillHostTensorView(parameter, host->second, argument.tensor))
                return false;
            arguments.push_back(argument);
            continue;
        }
        DeviceValue current(context, *value);
        DeviceValue original(context, *value);
        if (!current.buffer.valid() || !original.buffer.valid())
            return false;
        uploads.push_back({current.buffer.handle(), 0, value->data, value->size});
        uploads.push_back({original.buffer.handle(), 0, value->data, value->size});
        auto [workingIt, inserted] = working.emplace(parameter.name, std::move(current));
        auto [retainedIt, retainedInserted] = retainedDevices.emplace(parameter.name, std::move(original));
        const bool hostRetained =
            !retainStorageHostCopy || retainedHosts.emplace(parameter.name, HostValue(*value)).second;
        if (!inserted || !retainedInserted || !hostRetained ||
            !appendDeviceArgument(parameter, workingIt->second, arguments))
            return false;
    }
    return true;
}

template <typename Handle> bool decodeResourceHandle(uint64_t key, Handle &handle) {
    const uint32_t encodedIndex = static_cast<uint32_t>(key);
    if (!encodedIndex)
        return false;
    handle = {encodedIndex - 1, static_cast<uint32_t>(key >> 32)};
    return true;
}

const VernonProgramArgument *findInvocationArgument(const VernonProgramSubmitDescriptor &invocation, uint32_t slot) {
    for (size_t index = 0; index < invocation.argument_count; ++index)
        if (invocation.arguments[index].slot == slot)
            return &invocation.arguments[index];
    return nullptr;
}

bool stageValue(DeviceBuffer &source, void *destination, size_t size, PreparedForwardPublication &publication) {
    auto staged = std::find_if(publication.values.begin(), publication.values.end(),
                               [&](const StagedForwardValue &value) { return value.destination == destination; });
    if (staged == publication.values.end()) {
        publication.values.push_back({destination, std::vector<uint8_t>(size)});
        staged = std::prev(publication.values.end());
    } else if (staged->bytes.size() != size) {
        return false;
    }
    return source.download(staged->bytes.data(), staged->bytes.size());
}

bool stageForwardValues(const Signature &signature, const Variant &variant, DeviceValues &working,
                        const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                        PreparedForwardPublication &publication) {
    publication.values.clear();
    for (const ValueAbi &output : signature.outputs) {
        const auto parameter =
            std::find_if(variant.parameters.begin(), variant.parameters.end(), [&](const Parameter &value) {
                if (!singleLeafParameter(value))
                    return false;
                const ValueLayout &layout = value.valueLayout ? *value.valueLayout : value.elementLayout;
                return canonicalValueLeafPath(value.name, layout.leaves.front()) == output.path;
            });
        if (parameter == variant.parameters.end())
            return false;
        const auto device = working.find(parameter->name);
        if (device == working.end())
            return false;
        VernonAdValue *destination = findValue(outputs, output.path);
        if (destination && (!valueMatches(*destination, output) ||
                            !stageValue(device->second.buffer, destination->data, destination->size, publication)))
            return false;
        const VernonAdValue *input = findValue(inputs, output.path);
        if (input && valueMatches(*input, output) && (!destination || input->data != destination->data) &&
            !stageValue(device->second.buffer, const_cast<void *>(input->data), input->size, publication))
            return false;
        if (!destination && !input)
            return false;
    }
    for (const Parameter &parameter : variant.parameters) {
        if (!usesStorage(parameter) || parameter.access == "read")
            continue;
        const VernonAdValue *destination = findParameterValue(inputs, parameter);
        const auto device = working.find(parameter.name);
        if (!destination || !destination->data || device == working.end() ||
            destination->dtype != device->second.dtype || destination->size != device->second.buffer.size() ||
            !stageValue(device->second.buffer, const_cast<void *>(destination->data), destination->size, publication))
            return false;
    }
    return true;
}

bool materializeRuntimeSignature(const Signature &source, const VernonAdValueSet &inputs, Signature &result) {
    result = source;
    for (ValueAbi &abi : result.inputs) {
        const VernonAdValue *input = findValue(inputs, abi.path);
        if (!input || !materializeValueAbi(abi, *input))
            return false;
    }
    for (ValueAbi &output : result.outputs) {
        const auto input = std::find_if(result.inputs.begin(), result.inputs.end(),
                                        [&](const ValueAbi &value) { return value.path == output.path; });
        if (input == result.inputs.end())
            return false;
        output = *input;
    }
    for (ValueAbi &cotangent : result.cotangents)
        if (!materializeDerivativeValueAbi(cotangent, result.outputs))
            return false;
    for (ValueAbi &gradient : result.gradients)
        if (!materializeDerivativeValueAbi(gradient, result.inputs))
            return false;
    return true;
}

} // namespace

VernonStatus prepareForward(VernonRuntimeContext &context, OwnedPipeline &pipeline,
                            const std::shared_ptr<Signature> &signature, VernonLaunchSize computeGrid,
                            const VernonAdValueSet &inputs, const ForwardExecutionTarget &target,
                            const std::vector<std::string> &retainedNames, const char *resourceName,
                            PreparedForwardExecution &prepared) {
    prepared = {};
    prepared.signature = std::make_shared<Signature>();
    if (!materializeRuntimeSignature(*signature, inputs, *prepared.signature))
        return fail(context, "GPU autodiff values cannot be materialized from reflection");
    std::vector<VernonProgramArgument> arguments;
    std::vector<DeviceBufferUpload> uploads;
    std::vector<DeviceBufferCopy> copies;
    VernonRhiCommandEncoder nativeEncoder{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (!target.encodedInvocation()) {
        if (!buildForwardArguments(context, pipeline->variant, inputs, prepared.working, prepared.retainedDevices,
                                   prepared.retainedHosts, arguments, uploads, false))
            return fail(context, std::string("cannot prepare GPU autodiff ") + resourceName + " resources",
                        VERNON_STATUS_INTERNAL_ERROR);
    } else {
        if (!target.invocation)
            return fail(context, "encoded GPU autodiff forward has no invocation");
        const VernonProgramSubmitDescriptor &invocation = *target.invocation;
        if (target.externalEncoder() &&
            (!decodeResourceHandle(target.encoder.value, nativeEncoder) ||
             vernon::rhi::commandEncoderKey(context.rhiDevice, nativeEncoder) != target.encoder.value))
            return fail(context, "GPU autodiff command encoder belongs to another Runtime device");
        if (invocation.argument_count)
            arguments.assign(invocation.arguments, invocation.arguments + invocation.argument_count);
        arguments.reserve(arguments.size() + 1);
        for (const Parameter &parameter : pipeline->variant.parameters) {
            if (isAutodiffInternal(parameter))
                continue;
            const VernonAdValue *value = findParameterValues(inputs, parameter);
            const VernonProgramArgument *argument = findInvocationArgument(invocation, parameter.slot);
            if (!value || !argument || argument->kind != VERNON_PROGRAM_TENSOR)
                return fail(context, std::string("cannot prepare encoded GPU autodiff ") + resourceName + " resources",
                            VERNON_STATUS_INTERNAL_ERROR);
            if (std::find(retainedNames.begin(), retainedNames.end(), parameter.name) == retainedNames.end())
                continue;
            if (!usesStorage(parameter)) {
                if (!singleLeafParameter(parameter))
                    return fail(context, "encoded GPU autodiff cannot retain a non-storage aggregate Value");
                if (!prepared.retainedHosts.emplace(parameter.name, HostValue(*value)).second)
                    return fail(context, "cannot retain encoded GPU autodiff host value", VERNON_STATUS_INTERNAL_ERROR);
                continue;
            }
            const VernonTensorView &tensor = argument->tensor;
            size_t before = 0;
            size_t after = 0;
            size_t span = 0;
            if (tensor.storage != VERNON_TENSOR_RHI_RESOURCE)
                return fail(context,
                            "encoded GPU autodiff retained Value '" + parameter.name + "' is not RHI-backed storage");
            if (!tensorFitsAllocation(tensor) || !tensorRelativeByteBounds(tensor, before, after) ||
                !tensorRequiredSpan(tensor, span) || !span || before > tensor.byte_offset)
                return fail(context, "encoded GPU autodiff retained Value '" + parameter.name +
                                         "' has an invalid physical layout");
            if (tensor.rank && (!tensor.shape || !tensor.byte_strides))
                return fail(context, "encoded GPU autodiff retained Value '" + parameter.name +
                                         "' has incomplete physical layout metadata");
            VernonRhiBuffer source{};
            if (!decodeResourceHandle(tensor.resource.resource.value, source) ||
                vernon::rhi::bufferResource(context.rhiDevice, source) != tensor.resource.resource.value)
                return fail(context, "encoded GPU autodiff storage belongs to another Runtime device");
            const size_t sourceOffset = tensor.byte_offset - before;
            if (tensor.resource.offset > UINT64_MAX - sourceOffset)
                return fail(context, "encoded GPU autodiff retained Value offset overflows");
            std::vector<uint64_t> shape;
            std::vector<int64_t> strides;
            if (tensor.rank) {
                shape.assign(tensor.shape, tensor.shape + tensor.rank);
                strides.assign(tensor.byte_strides, tensor.byte_strides + tensor.rank);
            }
            DeviceValue retained(context, span, value->dtype, std::move(shape), std::move(strides), before);
            if (!retained.buffer.valid())
                return fail(context, "cannot allocate GPU autodiff retained Value", VERNON_STATUS_INTERNAL_ERROR);
            if (target.deferredCommandPlan()) {
                copies.push_back({source, retained.buffer.handle(), tensor.resource.offset + sourceOffset, 0, span});
                if (!prepared.retainedDevices.emplace(parameter.name, std::move(retained)).second)
                    return fail(context, "cannot retain GPU autodiff Value", VERNON_STATUS_INTERNAL_ERROR);
                continue;
            }
            if (vernonRhiCommandEncoderCopyBuffer(context.rhiDevice, nativeEncoder, source,
                                                  tensor.resource.offset + sourceOffset, retained.buffer.handle(), 0,
                                                  span) != VERNON_RHI_STATUS_OK)
                return fail(context, "cannot encode GPU autodiff retained Value copy", VERNON_STATUS_INTERNAL_ERROR);
            VernonRhiBarrier sourceBarrier{};
            sourceBarrier.struct_size = sizeof(sourceBarrier);
            sourceBarrier.destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
            sourceBarrier.source_access = VERNON_RHI_ACCESS_TRANSFER_READ;
            sourceBarrier.destination_access = VERNON_RHI_ACCESS_SHADER_READ | VERNON_RHI_ACCESS_SHADER_WRITE;
            sourceBarrier.old_state = VERNON_RHI_STATE_TRANSFER_SOURCE;
            sourceBarrier.new_state = VERNON_RHI_STATE_SHADER_WRITE;
            sourceBarrier.buffer = source;
            VernonRhiBarrier retainedBarrier{};
            retainedBarrier.struct_size = sizeof(retainedBarrier);
            retainedBarrier.destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
            retainedBarrier.source_access = VERNON_RHI_ACCESS_TRANSFER_WRITE;
            retainedBarrier.destination_access = VERNON_RHI_ACCESS_SHADER_READ;
            retainedBarrier.old_state = VERNON_RHI_STATE_TRANSFER_DESTINATION;
            retainedBarrier.new_state = VERNON_RHI_STATE_SHADER_READ;
            retainedBarrier.buffer = retained.buffer.handle();
            const VernonRhiBarrier barriers[]{sourceBarrier, retainedBarrier};
            if (vernonRhiCommandEncoderBarrier(context.rhiDevice, nativeEncoder, barriers, std::size(barriers)) !=
                    VERNON_RHI_STATUS_OK ||
                !prepared.retainedDevices.emplace(parameter.name, std::move(retained)).second)
                return fail(context, "cannot order GPU autodiff retained Value copy", VERNON_STATUS_INTERNAL_ERROR);
        }
    }
    const uint32_t launchData[3]{computeGrid.x, computeGrid.y, computeGrid.z};
    DeviceBuffer launchBuffer(context, sizeof(launchData));
    InternalBufferView launchView;
    for (const Parameter &parameter : pipeline->variant.parameters) {
        if (parameter.autodiffRole != AutodiffResourceRole::LaunchMetadata)
            continue;
        if (!launchBuffer.valid())
            return fail(context, std::string("cannot bind GPU autodiff ") + resourceName + " launch metadata",
                        VERNON_STATUS_INTERNAL_ERROR);
        if (target.externalEncoder()) {
            if (!launchBuffer.upload(nativeEncoder, launchData, sizeof(launchData)))
                return fail(context, std::string("cannot encode GPU autodiff ") + resourceName + " launch metadata",
                            VERNON_STATUS_INTERNAL_ERROR);
        } else {
            uploads.push_back({launchBuffer.handle(), 0, launchData, sizeof(launchData)});
        }
        if (target.externalEncoder()) {
            VernonRhiBarrier barrier{};
            barrier.struct_size = sizeof(barrier);
            barrier.destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
            barrier.source_access = VERNON_RHI_ACCESS_TRANSFER_WRITE;
            barrier.destination_access = VERNON_RHI_ACCESS_SHADER_READ;
            barrier.old_state = VERNON_RHI_STATE_TRANSFER_DESTINATION;
            barrier.new_state = VERNON_RHI_STATE_SHADER_READ;
            barrier.buffer = launchBuffer.handle();
            if (vernonRhiCommandEncoderBarrier(context.rhiDevice, nativeEncoder, &barrier, 1) != VERNON_RHI_STATUS_OK)
                return fail(context, std::string("cannot order GPU autodiff ") + resourceName + " launch metadata",
                            VERNON_STATUS_INTERNAL_ERROR);
        }
        if (!appendInternalBufferArgument(context, parameter, launchBuffer, sizeof(launchData), launchView, arguments))
            return fail(context, std::string("cannot bind GPU autodiff ") + resourceName + " launch metadata",
                        VERNON_STATUS_INTERNAL_ERROR);
    }
    if (target.deferredCommandPlan()) {
        const VernonStatus status =
            buildPipelineCommandPlan(context, copies, uploads, *pipeline, arguments, computeGrid, {},
                                     execution::detail::CommandNodeKind::Derivative, *target.commandPlan);
        if (status == VERNON_STATUS_OK)
            target.commandPlan->retainedContexts.push_back(std::make_shared<DeviceBuffer>(std::move(launchBuffer)));
        return status;
    }
    if (!target.externalEncoder())
        return executePipelineCommandDagAndWait(*pipeline, computeGrid, arguments, uploads,
                                                execution::detail::CommandNodeKind::Derivative);
    VernonProgramSubmitDescriptor encoded = *target.invocation;
    encoded.arguments = arguments.empty() ? nullptr : arguments.data();
    encoded.argument_count = arguments.size();
    encoded.command_encoder = {};
    return vernonRuntimeProgramEncode(target.encoder, pipeline.get(), &encoded);
}

bool stageForwardResults(const Signature &signature, const OwnedPipeline &pipeline, DeviceValues &working,
                         const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         PreparedForwardPublication &publication) {
    return stageForwardValues(signature, pipeline->variant, working, inputs, outputs, publication);
}

bool publishForwardResults(const PreparedForwardPublication &publication) {
    if (injectFailure(FailureBoundary::Publication))
        return false;
    for (const StagedForwardValue &value : publication.values)
        std::memcpy(value.destination, value.bytes.data(), value.bytes.size());
    return true;
}

} // namespace vernon::runtime::ad::gpu
