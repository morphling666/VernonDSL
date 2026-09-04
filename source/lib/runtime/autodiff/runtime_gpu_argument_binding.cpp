#include "runtime_gpu_argument_binding.h"

#include "runtime/shape_layout.h"
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

bool materializeInternalBufferView(const shape::DeclaredShape &declaredShape, const ValueLayout &layout,
                                   size_t logicalBytes, InternalBufferView &view) {
    if (layout.leaves.size() != 1 || !logicalBytes)
        return false;
    const ValueLeaf &leaf = layout.leaves.front();
    const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
    const size_t scalarBytes = dtype ? dtypeSize(*dtype) : 0;
    size_t elementBytes = 0;
    if (!scalarBytes || !leaf.scalarCount ||
        !checkedMultiply(scalarBytes, static_cast<size_t>(leaf.scalarCount), elementBytes) || !elementBytes)
        return false;
    size_t elements = 0;
    return shape::resolveSingleDynamicExtent(declaredShape, elementBytes, logicalBytes, view.shape) &&
           shape::rowMajorByteStrides(view.shape, elementBytes, view.strides) &&
           shape::checkedElementCount(view.shape, elements) && elements <= logicalBytes / elementBytes;
}

bool appendInternalBufferArgument(VernonRuntimeContext &context, const Parameter &parameter, const DeviceBuffer &buffer,
                                  size_t logicalBytes, InternalBufferView &view,
                                  std::vector<VernonProgramArgument> &arguments) {
    (void)context;
    const ValueLayout *layout = parameter.valueLayout ? &*parameter.valueLayout : &parameter.elementLayout;
    if (!layout ||
        !materializeInternalBufferView(shape::decodeRuntimeContractShape(parameter.shape), *layout, logicalBytes, view))
        return false;
    VernonRuntimeProviderResourceReference resource{};
    if (!buffer.reference(resource))
        return false;
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
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

bool appendBindingArgument(const Binding &binding, std::vector<VernonProgramArgument> &arguments) {
    const Parameter &parameter = *binding.parameter;
    const std::vector<uint64_t> &shape = binding.shape();
    const std::vector<int64_t> &strides = binding.strides();
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
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
