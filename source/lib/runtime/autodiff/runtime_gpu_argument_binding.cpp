#include "runtime_gpu_argument_binding.h"

#include "runtime_autodiff_internal.h"

#include <limits>
#include <optional>

namespace vernon::runtime::ad::gpu {
namespace {

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool singleLeafParameter(const Parameter &parameter) {
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    return layout && layout->leaves.size() == 1;
}

bool fillDeviceTensor(const Parameter &parameter, const DeviceValue &device, const std::vector<uint64_t> &shape,
                      const std::vector<int64_t> &strides, VernonTensorView &tensor) {
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    if (layout.leaves.empty() || shape.size() != strides.size())
        return false;
    VernonRuntimeProviderResourceReference resource{};
    if (!device.buffer.reference(resource))
        return false;
    const std::optional<VernonValueAccess> access = pipelineValueAccess(parameter.access);
    if (!access)
        return false;
    tensor = {};
    tensor.struct_size = sizeof(tensor);
    tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    tensor.resource = resource;
    tensor.element_layout = pipelineValueLayout(layout);
    tensor.access = *access;
    tensor.rank = static_cast<uint32_t>(shape.size());
    tensor.shape = shape.empty() ? nullptr : shape.data();
    tensor.byte_strides = strides.empty() ? nullptr : strides.data();
    tensor.byte_offset = device.byteOffset;
    tensor.byte_size = device.buffer.size();
    return true;
}

bool fillHostTensor(const Parameter &parameter, const HostValue &host, const std::vector<uint64_t> &shape,
                    const std::vector<int64_t> &strides, VernonTensorView &tensor) {
    if (!singleLeafParameter(parameter) || shape.size() != strides.size())
        return false;
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    const std::optional<VernonValueAccess> access = pipelineValueAccess(parameter.access);
    if (!access)
        return false;
    tensor = {};
    tensor.struct_size = sizeof(tensor);
    tensor.storage = VERNON_TENSOR_HOST;
    tensor.host_data = host.bytes.data();
    tensor.element_layout = pipelineValueLayout(layout);
    tensor.access = *access;
    tensor.rank = static_cast<uint32_t>(shape.size());
    tensor.shape = shape.empty() ? nullptr : shape.data();
    tensor.byte_strides = strides.empty() ? nullptr : strides.data();
    tensor.byte_size = host.bytes.size();
    return true;
}

} // namespace

bool appendInternalBufferArgument(VernonRuntimeContext &context, const Parameter &parameter, const DeviceBuffer &buffer,
                                  size_t logicalBytes, InternalBufferView &view,
                                  std::vector<VernonPipelineArgument> &arguments) {
    (void)context;
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    if (!layout || layout->leaves.size() != 1 || !logicalBytes)
        return false;
    const ValueLeaf &leaf = layout->leaves.front();
    const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
    const size_t scalarBytes = dtype ? dtypeSize(*dtype) : 0;
    size_t elementBytes = 0;
    if (!scalarBytes || !leaf.scalarCount ||
        !checkedMultiply(scalarBytes, static_cast<size_t>(leaf.scalarCount), elementBytes) || !elementBytes)
        return false;
    view.shape = parameter.shape;
    size_t dynamicDimensions = 0;
    size_t staticElements = 1;
    for (uint64_t extent : view.shape) {
        if (!extent) {
            ++dynamicDimensions;
            continue;
        }
        if (!checkedMultiply(staticElements, static_cast<size_t>(extent), staticElements))
            return false;
    }
    if (dynamicDimensions == 1) {
        if (logicalBytes % elementBytes != 0 || logicalBytes / elementBytes % staticElements != 0)
            return false;
        const uint64_t dynamicExtent = logicalBytes / elementBytes / staticElements;
        for (uint64_t &extent : view.shape)
            if (!extent) {
                extent = dynamicExtent;
                break;
            }
    } else if (dynamicDimensions != 0) {
        return false;
    }
    view.strides.resize(view.shape.size());
    size_t stride = elementBytes;
    for (size_t dimension = view.shape.size(); dimension-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return false;
        view.strides[dimension] = static_cast<int64_t>(stride);
        if (!checkedMultiply(stride, static_cast<size_t>(view.shape[dimension]), stride))
            return false;
    }
    if (stride > logicalBytes)
        return false;
    VernonRuntimeProviderResourceReference resource{};
    if (!buffer.reference(resource))
        return false;
    VernonPipelineArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(argument.tensor);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = resource;
    argument.tensor.element_layout = pipelineValueLayout(*layout);
    const std::optional<VernonValueAccess> access = pipelineValueAccess(parameter.access);
    if (!access)
        return false;
    argument.tensor.access = *access;
    argument.tensor.rank = static_cast<uint32_t>(view.shape.size());
    argument.tensor.shape = view.shape.data();
    argument.tensor.byte_strides = view.strides.data();
    argument.tensor.byte_size = logicalBytes;
    arguments.push_back(argument);
    return true;
}

bool appendBindingArgument(const Binding &binding, std::vector<VernonPipelineArgument> &arguments) {
    const Parameter &parameter = *binding.parameter;
    const std::vector<uint64_t> &shape = binding.shape();
    const std::vector<int64_t> &strides = binding.strides();
    VernonPipelineArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PIPELINE_TENSOR;
    if (binding.source == BindingSource::Device) {
        if (!binding.device || !fillDeviceTensor(parameter, *binding.device, shape, strides, argument.tensor))
            return false;
    } else if (binding.source == BindingSource::Host) {
        if (!binding.host || !fillHostTensor(parameter, *binding.host, shape, strides, argument.tensor))
            return false;
    } else {
        return false;
    }
    arguments.push_back(argument);
    return true;
}

} // namespace vernon::runtime::ad::gpu
