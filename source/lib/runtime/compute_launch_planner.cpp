#include "compute_launch_planner.h"

#include "pipeline_metadata.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>

namespace vernon::runtime {

namespace {

using ComputeArgumentMap = std::unordered_map<uint32_t, const VernonPipelineArgument *>;

bool fail(std::string &error, const char *message) {
    error = message;
    return false;
}

bool packTensorViewDescriptor(const VernonTensorView &tensor, std::vector<uint8_t> &storage) {
    if (!valueLayoutValid(tensor.element_layout) || !tensor.element_layout.byte_size ||
        (tensor.rank && (!tensor.shape || !tensor.byte_strides)) ||
        tensor.byte_offset % tensor.element_layout.byte_size)
        return false;
    storage.assign(8 * (2 + 2 * tensor.rank), 0);
    const uint64_t pointer = tensor.storage == VERNON_TENSOR_HOST
                                 ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.host_data))
                                 : static_cast<uint64_t>(tensor.resource.resource.value + tensor.resource.offset);
    const uint64_t elementOffset = tensor.byte_offset / tensor.element_layout.byte_size;
    std::memcpy(storage.data(), &pointer, sizeof(pointer));
    std::memcpy(storage.data() + 8, &elementOffset, sizeof(elementOffset));
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (tensor.byte_strides[dimension] % static_cast<int64_t>(tensor.element_layout.byte_size))
            return false;
        const int64_t elementStride =
            tensor.byte_strides[dimension] / static_cast<int64_t>(tensor.element_layout.byte_size);
        std::memcpy(storage.data() + 16 + 8 * dimension, &tensor.shape[dimension], sizeof(uint64_t));
        std::memcpy(storage.data() + 16 + 8 * (tensor.rank + dimension), &elementStride, sizeof(elementStride));
    }
    return true;
}

bool planComputeArguments(const Variant &variant, const ComputeArgumentMap &arguments, VernonLaunchSize workgroup,
                          const VernonPipelineInvocation &invocation, PlannedComputeLaunch &plan, std::string &error) {
    constexpr size_t kMaxComputeArgumentIndex = 4095;
    size_t computeArgumentSpan = 0;
    size_t computeArgumentCount = 0;
    for (const Parameter &parameter : variant.parameters)
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != variant.compute)
                continue;
            if (use.index > kMaxComputeArgumentIndex)
                return fail(error, "compute argument index exceeds the supported range");
            computeArgumentSpan = std::max(computeArgumentSpan, static_cast<size_t>(use.index) + 1);
            ++computeArgumentCount;
        }

    plan = {};
    plan.arguments.resize(computeArgumentSpan);
    plan.hostTensorStorage.reserve(computeArgumentCount);
    std::vector<uint8_t> assigned(computeArgumentSpan);

    for (const Parameter &parameter : variant.parameters) {
        const auto suppliedIt = arguments.find(parameter.slot);
        if (suppliedIt == arguments.end())
            return fail(error, "compute argument is missing");
        const VernonPipelineArgument &supplied = *suppliedIt->second;
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != variant.compute)
                continue;
            if (assigned[use.index])
                return fail(error, "compute argument index is duplicated");

            ComputeLaunchArgument &argument = plan.arguments[use.index];
            if (supplied.kind == VERNON_PIPELINE_IMAGE) {
                if (parameter.kind != "image" || !supplied.image.view.identity || !supplied.image.view.resource.value)
                    return fail(error, "compute image argument is invalid");
                argument = ComputeImageArgument{supplied.image.view};
                assigned[use.index] = 1;
                continue;
            }
            if (supplied.kind != VERNON_PIPELINE_TENSOR)
                return fail(error, "compute argument kind is unsupported");
            if (supplied.tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                if (!supplied.tensor.resource.identity || !supplied.tensor.resource.resource.value ||
                    supplied.tensor.byte_offset > supplied.tensor.resource.size ||
                    supplied.tensor.byte_size > supplied.tensor.resource.size || !tensorFitsAllocation(supplied.tensor))
                    return fail(error, "compute RHI Tensor view is invalid");
                argument = ComputeTensorArgument{supplied.tensor.resource};
            } else {
                if (use.interfaceKind == "storage") {
                    if (supplied.tensor.storage != VERNON_TENSOR_HOST || !supplied.tensor.host_data ||
                        !tensorFitsAllocation(supplied.tensor))
                        return fail(error, "compute host storage Tensor is invalid");
                    argument = ComputeTensorArgument{{}, supplied.tensor.host_data, supplied.tensor.byte_size};
                } else {
                    if (!use.interfacePlan || !use.interfacePlan->root)
                        return fail(error, "compute value Tensor is missing its typed interface plan");
                    if (use.interfacePlan->root->size > std::numeric_limits<size_t>::max())
                        return fail(error, "compute Tensor physical size exceeds the host size range");
                    const std::vector<uint64_t> &logicalShape = use.shape.empty() ? parameter.shape : use.shape;
                    std::optional<TensorCopyPlan> layout =
                        parameter.valueLayout
                            ? compileWholeValueCopyPlan(supplied.tensor.element_layout, *use.interfacePlan->root)
                            : compileElementStreamCopyPlan(supplied.tensor.element_layout, logicalShape,
                                                           *use.interfacePlan->root);
                    if (!layout)
                        return fail(error, "compute Tensor interface plan does not match its canonical layout");
                    VernonTensorView packedTensor = supplied.tensor;
                    if (packedTensor.rank == layout->shape.size() + 1 && packedTensor.shape &&
                        packedTensor.byte_strides && packedTensor.shape[0] == 1) {
                        --packedTensor.rank;
                        ++packedTensor.shape;
                        ++packedTensor.byte_strides;
                    }
                    auto packed = packTensor(packedTensor, *layout);
                    if (!packed)
                        return fail(error, "failed to pack reflected compute Tensor layout");
                    plan.hostTensorStorage.push_back(std::move(*packed));
                    argument = ComputeScalarArgument{plan.hostTensorStorage.back().data(),
                                                     plan.hostTensorStorage.back().size()};
                }
            }
            if (use.tensorViewDescriptor) {
                plan.hostTensorStorage.emplace_back();
                if (!packTensorViewDescriptor(supplied.tensor, plan.hostTensorStorage.back()))
                    return fail(error, "failed to pack TensorView dispatch descriptor");
                auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
                if (!tensor)
                    return fail(error, "TensorView descriptor is attached to a non-Tensor argument");
                tensor->tensorView = &supplied.tensor;
                tensor->tensorViewData = plan.hostTensorStorage.back().data();
                tensor->tensorViewSize = plan.hostTensorStorage.back().size();
            }
            assigned[use.index] = 1;
        }
    }

    plan.commandEncoder = invocation.command_encoder;
    plan.grid = invocation.compute_grid;
    if (!plan.grid.x || !plan.grid.y || !plan.grid.z) {
        for (const Parameter &parameter : variant.parameters) {
            const auto argumentIt = arguments.find(parameter.slot);
            if (argumentIt == arguments.end())
                continue;
            const VernonPipelineArgument &argument = *argumentIt->second;
            if (argument.kind != VERNON_PIPELINE_TENSOR || argument.tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                (argument.tensor.rank && !argument.tensor.shape))
                continue;
            const uint32_t rank = argument.tensor.rank;
            if (!rank) {
                plan.grid = {1, 1, 1};
                break;
            }
            const uint32_t gridDimensions = rank < 3 ? rank : 3;
            for (uint32_t dimension = 0; dimension < gridDimensions; ++dimension) {
                if (!argument.tensor.shape[rank - 1 - dimension] ||
                    argument.tensor.shape[rank - 1 - dimension] > std::numeric_limits<uint32_t>::max())
                    return fail(error, "inferred compute grid exceeds uint32 range");
            }
            if (!workgroup.x || !workgroup.y || !workgroup.z)
                return fail(error, "compute workgroup size is invalid");
            const auto groups = [](uint64_t extent, uint32_t size) {
                return static_cast<uint32_t>((extent - 1) / size + 1);
            };
            plan.grid = {1, 1, 1};
            plan.grid.x = groups(argument.tensor.shape[rank - 1], workgroup.x);
            if (rank > 1)
                plan.grid.y = groups(argument.tensor.shape[rank - 2], workgroup.y);
            if (rank > 2)
                plan.grid.z = groups(argument.tensor.shape[rank - 3], workgroup.z);
            break;
        }
    }
    if (!plan.grid.x || !plan.grid.y || !plan.grid.z)
        return fail(error, "compute grid cannot be inferred");
    return true;
}

} // namespace

std::optional<int64_t> computeBindingDescriptorValue(const ComputeLaunchArgument &argument,
                                                     const ComputeBindingSource &source) {
    const auto *tensorArgument = std::get_if<ComputeTensorArgument>(&argument);
    const VernonTensorView *tensor = tensorArgument ? tensorArgument->tensorView : nullptr;
    if (!tensor || !valueLayoutValid(tensor->element_layout))
        return std::nullopt;
    const int64_t elementSize = static_cast<int64_t>(tensor->element_layout.byte_size);
    switch (source.kind) {
    case ComputeBindingSourceKind::Argument:
        return std::nullopt;
    case ComputeBindingSourceKind::TensorOffset:
        if (tensor->byte_offset % tensor->element_layout.byte_size ||
            tensor->byte_offset / tensor->element_layout.byte_size >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return std::nullopt;
        return static_cast<int64_t>(tensor->byte_offset / tensor->element_layout.byte_size);
    case ComputeBindingSourceKind::TensorExtent:
        if (source.dimension >= tensor->rank || !tensor->shape ||
            tensor->shape[source.dimension] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return std::nullopt;
        return static_cast<int64_t>(tensor->shape[source.dimension]);
    case ComputeBindingSourceKind::TensorStride:
        if (source.dimension >= tensor->rank || !tensor->byte_strides ||
            tensor->byte_strides[source.dimension] % elementSize)
            return std::nullopt;
        return tensor->byte_strides[source.dimension] / elementSize;
    }
    return std::nullopt;
}

bool planComputeInvocation(const Variant &variant, VernonLaunchSize workgroup,
                           const VernonPipelineInvocation &invocation, PlannedComputeLaunch &plan, std::string &error) {
    ComputeArgumentMap arguments;
    for (size_t index = 0; index < invocation.argument_count; ++index)
        if (!arguments.emplace(invocation.arguments[index].slot, &invocation.arguments[index]).second)
            return fail(error, "duplicate pipeline argument slot");
    if (arguments.size() != variant.parameters.size())
        return fail(error, "pipeline argument count does not match layout");

    std::vector<const VernonTensorView *> tensors;
    tensors.reserve(variant.parameters.size());
    for (const Parameter &parameter : variant.parameters) {
        const auto found = arguments.find(parameter.slot);
        if (found == arguments.end())
            return fail(error, "pipeline argument kind does not match layout");
        if (parameter.kind == "image") {
            if (found->second->kind != VERNON_PIPELINE_IMAGE)
                return fail(error, "pipeline argument kind does not match layout");
            continue;
        }
        if (parameter.kind != "tensor" || found->second->kind != VERNON_PIPELINE_TENSOR)
            return fail(error, "pipeline argument kind does not match layout");
        const VernonTensorView &tensor = found->second->tensor;
        const ValueLayout &expectedLayout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
        const bool accessCompatible = parameter.access.empty()      ? true
                                      : parameter.access == "read"  ? tensor.access != VERNON_ACCESS_WRITE
                                      : parameter.access == "write" ? tensor.access != VERNON_ACCESS_READ
                                                                    : tensor.access == VERNON_ACCESS_READ_WRITE;
        if (tensor.struct_size < sizeof(VernonTensorView))
            return fail(error, "pipeline Tensor argument structure is incomplete");
        if (!valueLayoutsEqual(tensor.element_layout, pipelineValueLayout(expectedLayout)))
            return fail(error, "pipeline Tensor argument element layout does not match reflection");
        if (tensor.access > VERNON_ACCESS_READ_WRITE || !accessCompatible)
            return fail(error, "pipeline Tensor argument access does not match reflection");
        if (tensor.storage != VERNON_TENSOR_HOST && tensor.storage != VERNON_TENSOR_RHI_RESOURCE)
            return fail(error, "pipeline Tensor argument storage kind is invalid");
        if (!tensorFitsAllocation(tensor)) {
            size_t requiredSpan = 0;
            if (!tensorRequiredSpan(tensor, requiredSpan))
                return fail(error, "pipeline Tensor argument byte layout is invalid");
            error = "pipeline Tensor argument '" + parameter.name + "' at byte offset " +
                    std::to_string(tensor.byte_offset) + " requires " + std::to_string(requiredSpan) +
                    " bytes but its allocation has " + std::to_string(tensor.byte_size);
            return false;
        }
        if (tensor.access != VERNON_ACCESS_READ && !tensorByteLayoutInjective(tensor))
            return fail(error, "writable pipeline Tensor argument must have an injective byte layout");
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != variant.compute)
                continue;
            if (!use.tensorViewDescriptor)
                continue;
            if (tensor.rank != use.tensorViewDescriptor->rank || tensor.rank != use.shape.size() ||
                (tensor.rank && (!tensor.shape || !tensor.byte_strides)) || !tensor.element_layout.byte_size ||
                tensor.byte_offset % tensor.element_layout.byte_size)
                return fail(error, "pipeline TensorView descriptor does not match its declared rank");
            for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
                if ((use.shape[dimension] && use.shape[dimension] != tensor.shape[dimension]) ||
                    tensor.byte_strides[dimension] % static_cast<int64_t>(tensor.element_layout.byte_size))
                    return fail(error, "pipeline TensorView descriptor violates static shape or element stride");
            }
        }
        if (parameter.source != "direct") {
            const bool allowLeading =
                std::any_of(parameter.uses.begin(), parameter.uses.end(), [](const ParameterUse &use) {
                    return use.interfaceKind == "input" || use.interfaceKind == "value";
                });
            const size_t offset =
                allowLeading && tensor.rank == parameter.shape.size() + 1 && tensor.shape && tensor.shape[0] == 1 ? 1
                                                                                                                  : 0;
            if (tensor.rank != parameter.shape.size() + offset)
                return fail(error, "pipeline Tensor rank does not match layout");
            for (size_t dimension = 0; dimension < parameter.shape.size(); ++dimension)
                if (parameter.shape[dimension] && parameter.shape[dimension] != tensor.shape[dimension + offset])
                    return fail(error, "pipeline Tensor shape does not match layout");
        }
        if (tensor.storage == VERNON_TENSOR_HOST && !tensor.host_data)
            return fail(error, "pipeline Tensor argument does not match layout");
        tensors.push_back(&tensor);
    }
    for (size_t left = 0; left < tensors.size(); ++left)
        for (size_t right = left + 1; right < tensors.size(); ++right)
            if (tensorViewsHaveWritableOverlap(*tensors[left], *tensors[right]))
                return fail(error, "pipeline Tensor arguments have incompatible physical overlap");
    return planComputeArguments(variant, arguments, workgroup, invocation, plan, error);
}

} // namespace vernon::runtime
