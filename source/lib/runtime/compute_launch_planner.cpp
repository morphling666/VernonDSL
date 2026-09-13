#include "compute_launch_planner.h"

#include "pipeline_metadata.h"
#include "program_execution/program_boundary_contract.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

namespace vernon::runtime {

namespace {

bool fail(std::string &error, const char *message) {
    error = message;
    return false;
}

bool packTensorViewDescriptor(const VernonTensorView &tensor, std::vector<uint8_t> &storage) {
    if (!valueLayoutValid(tensor.element_layout) || !tensor.element_layout.byte_size ||
        (tensor.rank && (!tensor.shape || !tensor.byte_strides)) ||
        tensor.byte_offset % tensor.element_layout.byte_size)
        return false;
    constexpr size_t fieldSize = sizeof(uintptr_t);
    storage.assign(fieldSize * (2 + 2 * tensor.rank), 0);
    uintptr_t pointer = reinterpret_cast<uintptr_t>(tensor.host_data);
    if (tensor.storage != VERNON_TENSOR_HOST) {
        const uint64_t resourceAddress = tensor.resource.resource.value + tensor.resource.offset;
        if (resourceAddress < tensor.resource.resource.value || resourceAddress > std::numeric_limits<uintptr_t>::max())
            return false;
        pointer = static_cast<uintptr_t>(resourceAddress);
    }
    const uint64_t wideElementOffset = tensor.byte_offset / tensor.element_layout.byte_size;
    if (wideElementOffset > std::numeric_limits<size_t>::max())
        return false;
    const size_t elementOffset = static_cast<size_t>(wideElementOffset);
    std::memcpy(storage.data(), &pointer, sizeof(pointer));
    std::memcpy(storage.data() + fieldSize, &elementOffset, sizeof(elementOffset));
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (tensor.byte_strides[dimension] % static_cast<int64_t>(tensor.element_layout.byte_size))
            return false;
        const int64_t elementStride =
            tensor.byte_strides[dimension] / static_cast<int64_t>(tensor.element_layout.byte_size);
        if (tensor.shape[dimension] > std::numeric_limits<size_t>::max() ||
            elementStride < std::numeric_limits<intptr_t>::min() ||
            elementStride > std::numeric_limits<intptr_t>::max())
            return false;
        const size_t extent = static_cast<size_t>(tensor.shape[dimension]);
        const intptr_t stride = static_cast<intptr_t>(elementStride);
        std::memcpy(storage.data() + fieldSize * (2 + dimension), &extent, sizeof(extent));
        std::memcpy(storage.data() + fieldSize * (2 + tensor.rank + dimension), &stride, sizeof(stride));
    }
    return true;
}

const VernonProgramArgument *findArgument(const VernonStageInvocationDescriptor &invocation, uint32_t slot) {
    for (size_t index = 0; index < invocation.argument_count; ++index)
        if (invocation.arguments[index].slot == slot)
            return &invocation.arguments[index];
    return nullptr;
}

bool planComputeArguments(const StageBindingPlan &stagePlan, const VernonStageInvocationDescriptor &invocation,
                          PlannedComputeLaunch &plan, std::string &error) {
    constexpr size_t kMaxComputeArgumentIndex = 4095;
    size_t computeArgumentSpan = 0;
    size_t computeArgumentCount = 0;
    for (const Parameter &parameter : stagePlan.parameters)
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != stagePlan.compute)
                continue;
            if (use.index > kMaxComputeArgumentIndex)
                return fail(error, "compute argument index exceeds the supported range");
            computeArgumentSpan = std::max(computeArgumentSpan, static_cast<size_t>(use.index) + 1);
            ++computeArgumentCount;
        }

    plan.reset();
    plan.arguments.resize(computeArgumentSpan);
    plan.hostTensorStorage.reserve(std::max(plan.hostTensorStorage.capacity(), computeArgumentCount));
    plan.assignedArguments.resize(computeArgumentSpan);
    std::fill(plan.assignedArguments.begin(), plan.assignedArguments.end(), 0);

    for (const Parameter &parameter : stagePlan.parameters) {
        const VernonProgramArgument *suppliedArgument = findArgument(invocation, parameter.slot);
        if (!suppliedArgument)
            return fail(error, "compute argument is missing");
        const VernonProgramArgument &supplied = *suppliedArgument;
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != stagePlan.compute)
                continue;
            if (plan.assignedArguments[use.index])
                return fail(error, "compute argument index is duplicated");

            ComputeLaunchArgument &argument = plan.arguments[use.index];
            if (supplied.kind == VERNON_PROGRAM_IMAGE) {
                if (parameter.kind != "image" || !supplied.image.view.identity || !supplied.image.view.resource.value)
                    return fail(error, "compute image argument is invalid");
                argument = ComputeImageArgument{supplied.image.view};
                plan.assignedArguments[use.index] = 1;
                continue;
            }
            if (supplied.kind != VERNON_PROGRAM_TENSOR)
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
                    auto layout =
                        use.tensorPacking == TensorRepresentation::WholeValue
                            ? compileWholeValueCopyPlan(pipelineValueLayout(*use.valueLayout), *use.interfacePlan->root)
                            : compileElementStreamCopyPlan(pipelineValueLayout(*use.valueLayout), logicalShape,
                                                           *use.interfacePlan->root);
                    if (layout.isErr()) {
                        const ValueLayout &canonical = *use.valueLayout;
                        error = "compute Tensor interface plan for parameter '" + parameter.name +
                                "' does not match its canonical layout (type " + canonical.logicalType + ", bytes " +
                                std::to_string(canonical.byteSize) + ", leaves " +
                                std::to_string(canonical.leaves.size()) + ", physical bytes " +
                                std::to_string(use.interfacePlan->root->size) + ")";
                        return false;
                    }
                    VernonTensorView packedTensor = supplied.tensor;
                    if (packedTensor.rank == layout.value().shape.size() + 1 && packedTensor.shape &&
                        packedTensor.byte_strides && packedTensor.shape[0] == 1) {
                        --packedTensor.rank;
                        ++packedTensor.shape;
                        ++packedTensor.byte_strides;
                    }
                    auto packed = packTensor(packedTensor, layout.value());
                    if (packed.isErr())
                        return fail(error, tensorBridgeErrorMessage(packed.error()));
                    std::vector<uint8_t> &storage = plan.appendHostTensorStorage();
                    storage = std::move(packed).value();
                    argument = ComputeScalarArgument{storage.data(), storage.size()};
                    if (use.interfaceKind == "result")
                        plan.resultCommits.push_back(
                            {plan.hostTensorStorageCount - 1, supplied.tensor, std::move(layout).value()});
                }
            }
            if (use.tensorViewDescriptor) {
                std::vector<uint8_t> &storage = plan.appendHostTensorStorage();
                if (!packTensorViewDescriptor(supplied.tensor, storage))
                    return fail(error, "failed to pack TensorView dispatch descriptor");
                auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
                if (!tensor)
                    return fail(error, "TensorView descriptor is attached to a non-Tensor argument");
                tensor->tensorView = &supplied.tensor;
                tensor->tensorViewData = storage.data();
                tensor->tensorViewSize = storage.size();
            }
            plan.assignedArguments[use.index] = 1;
        }
    }

    plan.commandEncoder = invocation.command_encoder;
    plan.grid = invocation.compute_grid;
    if (!plan.grid.x)
        return fail(error, "compute grid axis x must be nonzero");
    if (!plan.grid.y)
        return fail(error, "compute grid axis y must be nonzero");
    if (!plan.grid.z)
        return fail(error, "compute grid axis z must be nonzero");
    return true;
}

} // namespace

void PlannedComputeLaunch::reset() {
    arguments.clear();
    for (std::vector<uint8_t> &storage : hostTensorStorage)
        storage.clear();
    hostTensorStorageCount = 0;
    resultCommits.clear();
    validationTensors.clear();
    assignedArguments.clear();
    grid = {};
    commandEncoder = {};
}

std::vector<uint8_t> &PlannedComputeLaunch::appendHostTensorStorage() {
    if (hostTensorStorageCount == hostTensorStorage.size())
        hostTensorStorage.emplace_back();
    return hostTensorStorage[hostTensorStorageCount++];
}

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
    case ComputeBindingSourceKind::TensorOffset: {
        if (tensor->byte_offset % tensor->element_layout.byte_size)
            return std::nullopt;
        const size_t offset = tensor->byte_offset / tensor->element_layout.byte_size;
        if constexpr (sizeof(size_t) >= sizeof(int64_t))
            if (offset > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                return std::nullopt;
        return static_cast<int64_t>(offset);
    }
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

bool planComputeInvocation(const StageBindingPlan &stagePlan, const VernonStageInvocationDescriptor &invocation,
                           PlannedComputeLaunch &plan, std::string &error) {
    for (size_t index = 0; index < invocation.argument_count; ++index)
        for (size_t other = index + 1; other < invocation.argument_count; ++other)
            if (invocation.arguments[index].slot == invocation.arguments[other].slot)
                return fail(error, "duplicate pipeline argument slot");
    if (invocation.argument_count != stagePlan.parameters.size())
        return fail(error, "pipeline argument count does not match layout");

    plan.validationTensors.clear();
    plan.validationTensors.reserve(stagePlan.parameters.size());
    for (const Parameter &parameter : stagePlan.parameters) {
        const VernonProgramArgument *argument = findArgument(invocation, parameter.slot);
        if (!argument)
            return fail(error, "pipeline argument kind does not match layout");
        if (parameter.kind == "image") {
            if (argument->kind != VERNON_PROGRAM_IMAGE)
                return fail(error, "pipeline argument kind does not match layout");
            continue;
        }
        if (parameter.kind != "tensor" || argument->kind != VERNON_PROGRAM_TENSOR)
            return fail(error, "pipeline argument kind does not match layout");
        const VernonTensorView &tensor = argument->tensor;
        const ValueLayout &expectedLayout =
            parameter.tensorArgument == TensorRepresentation::WholeValue && parameter.valueLayout
                ? *parameter.valueLayout
                : parameter.elementLayout;
        const VernonValueAccess requiredAccess = parameter.access == "read"    ? VERNON_ACCESS_READ
                                                 : parameter.access == "write" ? VERNON_ACCESS_WRITE
                                                                               : VERNON_ACCESS_READ_WRITE;
        const bool accessCompatible =
            parameter.access.empty() || program_execution::valueAccessSatisfies(requiredAccess, tensor.access);
        if (tensor.struct_size < sizeof(VernonTensorView)) {
            error = "pipeline Tensor argument '" + parameter.name + "' structure is incomplete";
            return false;
        }
        const VernonValueLayoutView expectedLayoutView = pipelineValueLayout(expectedLayout);
        if (!valueLayoutsEqual(tensor.element_layout, expectedLayoutView)) {
            error = "pipeline Tensor argument '" + parameter.name + "' element layout does not match reflection";
            error += " (supplied bytes " + std::to_string(tensor.element_layout.byte_size) + ", expected bytes " +
                     std::to_string(expectedLayoutView.byte_size) + ", supplied leaves " +
                     std::to_string(tensor.element_layout.leaf_count) + ", expected leaves " +
                     std::to_string(expectedLayoutView.leaf_count) + ")";
            return false;
        }
        if (tensor.access > VERNON_ACCESS_READ_WRITE || !accessCompatible)
            return fail(error, "pipeline Tensor argument access does not match reflection");
        if (tensor.storage != VERNON_TENSOR_HOST && tensor.storage != VERNON_TENSOR_RHI_RESOURCE)
            return fail(error, "pipeline Tensor argument storage kind is invalid");
        if (!tensorFitsAllocation(tensor)) {
            auto requiredSpan = tensorRequiredSpan(tensor);
            if (requiredSpan.isErr())
                return fail(error, "pipeline Tensor argument byte layout is invalid");
            error = "pipeline Tensor argument '" + parameter.name + "' at byte offset " +
                    std::to_string(tensor.byte_offset) + " requires " + std::to_string(requiredSpan.value()) +
                    " bytes but its allocation has " + std::to_string(tensor.byte_size);
            return false;
        }
        if (tensor.access != VERNON_ACCESS_READ && !tensorByteLayoutInjective(tensor))
            return fail(error, "writable pipeline Tensor argument must have an injective byte layout");
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != stagePlan.compute)
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
                    return error = "pipeline Tensor argument '" + parameter.name +
                                   "' TensorView descriptor violates static shape or element stride at axis " +
                                   std::to_string(dimension),
                           false;
            }
        }
        if (parameter.source != StageParameterSource::Direct &&
            parameter.tensorArgument != TensorRepresentation::WholeValue) {
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
        plan.validationTensors.push_back(&tensor);
    }
    for (size_t left = 0; left < plan.validationTensors.size(); ++left)
        for (size_t right = left + 1; right < plan.validationTensors.size(); ++right)
            if (tensorViewsHaveWritableOverlap(*plan.validationTensors[left], *plan.validationTensors[right]))
                return error = "pipeline Tensor arguments #" + std::to_string(left) + " and #" + std::to_string(right) +
                               " have incompatible physical overlap (offsets " +
                               std::to_string(plan.validationTensors[left]->byte_offset) + " and " +
                               std::to_string(plan.validationTensors[right]->byte_offset) + ", element bytes " +
                               std::to_string(plan.validationTensors[left]->element_layout.byte_size) + " and " +
                               std::to_string(plan.validationTensors[right]->element_layout.byte_size) + ", ranks " +
                               std::to_string(plan.validationTensors[left]->rank) + " and " +
                               std::to_string(plan.validationTensors[right]->rank) + ", first strides " +
                               std::to_string(plan.validationTensors[left]->rank
                                                  ? plan.validationTensors[left]->byte_strides[0]
                                                  : 0) +
                               " and " +
                               std::to_string(plan.validationTensors[right]->rank
                                                  ? plan.validationTensors[right]->byte_strides[0]
                                                  : 0) +
                               ")",
                       false;
    return planComputeArguments(stagePlan, invocation, plan, error);
}

bool commitComputeResults(const PlannedComputeLaunch &plan, std::string &error) {
    for (const ResultCommitPlan &publication : plan.resultCommits) {
        if (publication.storageIndex >= plan.hostTensorStorage.size()) {
            error = "compute Value result does not match its canonical host destination";
            return false;
        }
        auto unpacked =
            unpackTensor(plan.hostTensorStorage[publication.storageIndex], publication.destination, publication.layout);
        if (unpacked.isErr())
            return error = tensorBridgeErrorMessage(unpacked.error()), false;
    }
    return true;
}

} // namespace vernon::runtime
