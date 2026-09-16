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

bool checkedAdd(int64_t left, int64_t right, int64_t minimum, int64_t maximum, int64_t &result) {
    if ((right > 0 && left > maximum - right) || (right < 0 && left < minimum - right))
        return false;
    result = left + right;
    return result >= minimum && result <= maximum;
}

bool checkedProjectionProduct(uint64_t steps, int64_t stride, int64_t minimum, int64_t maximum, int64_t &result) {
    if (!steps || !stride) {
        result = 0;
        return true;
    }
    if (stride > 0) {
        if (steps > static_cast<uint64_t>(maximum / stride))
            return false;
        result = static_cast<int64_t>(steps) * stride;
        return true;
    }
    const uint64_t magnitude =
        stride == std::numeric_limits<int64_t>::min() ? uint64_t{1} << 63 : static_cast<uint64_t>(-stride);
    const uint64_t minimumMagnitude =
        minimum == std::numeric_limits<int64_t>::min() ? uint64_t{1} << 63 : static_cast<uint64_t>(-minimum);
    if (steps > minimumMagnitude / magnitude)
        return false;
    if (steps * magnitude == (uint64_t{1} << 63))
        result = std::numeric_limits<int64_t>::min();
    else
        result = -static_cast<int64_t>(steps * magnitude);
    return result >= minimum;
}

bool metadataTensorValid(const VernonTensorView &tensor, uint32_t rank, int64_t minimum, int64_t maximum,
                         std::string &error) {
    if (!valueLayoutValid(tensor.element_layout) || !tensor.element_layout.byte_size || tensor.rank != rank ||
        (rank && (!tensor.shape || !tensor.byte_strides)) || tensor.byte_offset % tensor.element_layout.byte_size ||
        tensor.byte_offset > tensor.byte_size || (tensor.storage == VERNON_TENSOR_HOST && !tensor.host_data) ||
        (tensor.storage == VERNON_TENSOR_RHI_RESOURCE &&
         (tensor.resource.offset >= tensor.resource.size ||
          tensor.byte_size > tensor.resource.size - tensor.resource.offset))) {
        error = "TensorView metadata source has an invalid rank or element layout";
        return false;
    }
    const uint64_t elementSize = tensor.element_layout.byte_size;
    if (elementSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        error = "TensorView element size exceeds logical stride conversion limits";
        return false;
    }
    const uint64_t offset = tensor.byte_offset / elementSize;
    if (offset > static_cast<uint64_t>(maximum)) {
        error = "TensorView logical offset is not representable by the metadata profile";
        return false;
    }
    for (size_t leaf = 0; leaf < tensor.element_layout.leaf_count; ++leaf) {
        const VernonValueLeafView &layoutLeaf = tensor.element_layout.leaves[leaf];
        auto scalarSize = dataTypeSize(static_cast<VernonDataType>(layoutLeaf.dtype));
        if (scalarSize.isErr() || !layoutLeaf.scalar_count || layoutLeaf.byte_offset > elementSize ||
            layoutLeaf.scalar_count > (elementSize - layoutLeaf.byte_offset) / scalarSize.value()) {
            error = "TensorView physical storage leaf exceeds its element layout";
            return false;
        }
    }
    bool empty = false;
    int64_t minimumIndex = static_cast<int64_t>(offset);
    int64_t maximumIndex = static_cast<int64_t>(offset);
    std::vector<int64_t> logicalStrides(rank);
    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
        const uint64_t extent = tensor.shape[dimension];
        if (extent > static_cast<uint64_t>(maximum) ||
            tensor.byte_strides[dimension] % static_cast<int64_t>(elementSize)) {
            error = "TensorView extent or byte stride is not exactly representable in logical elements";
            return false;
        }
        const int64_t stride = tensor.byte_strides[dimension] / static_cast<int64_t>(elementSize);
        if (stride < minimum || stride > maximum) {
            error = "TensorView logical stride is not representable by the metadata profile";
            return false;
        }
        logicalStrides[dimension] = stride;
        empty |= !extent;
    }
    if (empty)
        return true;
    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
        const uint64_t extent = tensor.shape[dimension];
        const int64_t stride = logicalStrides[dimension];
        int64_t product = 0;
        if (!checkedProjectionProduct(extent - 1, stride, minimum, maximum, product)) {
            error = "TensorView projection multiplication overflows the metadata profile";
            return false;
        }
        int64_t next = 0;
        if (product < 0) {
            if (!checkedAdd(minimumIndex, product, minimum, maximum, next)) {
                error = "TensorView projection accumulation overflows the metadata profile";
                return false;
            }
            minimumIndex = next;
        } else {
            if (!checkedAdd(maximumIndex, product, minimum, maximum, next)) {
                error = "TensorView projection accumulation overflows the metadata profile";
                return false;
            }
            maximumIndex = next;
        }
    }
    if (minimumIndex < 0) {
        error = "TensorView projection precedes its owner storage";
        return false;
    }
    for (size_t leaf = 0; leaf < tensor.element_layout.leaf_count; ++leaf) {
        const VernonValueLeafView &layoutLeaf = tensor.element_layout.leaves[leaf];
        auto scalarSize = dataTypeSize(static_cast<VernonDataType>(layoutLeaf.dtype));
        const uint64_t leafSize = scalarSize.value() * layoutLeaf.scalar_count;
        const uint64_t maximumLogical = static_cast<uint64_t>(maximumIndex);
        if (maximumLogical > std::numeric_limits<uint64_t>::max() / elementSize ||
            maximumLogical * elementSize > std::numeric_limits<uint64_t>::max() - layoutLeaf.byte_offset ||
            maximumLogical * elementSize + layoutLeaf.byte_offset > tensor.byte_size ||
            leafSize > tensor.byte_size - (maximumLogical * elementSize + layoutLeaf.byte_offset)) {
            error = "TensorView projection exceeds a physical storage leaf";
            return false;
        }
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
                    supplied.tensor.resource.offset >= supplied.tensor.resource.size ||
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
            const bool metadataArgument =
                stagePlan.metadataCarrier &&
                std::any_of(stagePlan.metadataCarrier->fields.begin(), stagePlan.metadataCarrier->fields.end(),
                            [&](const MetadataFieldIdentity &field) { return field.argument == use.index; });
            if (metadataArgument) {
                auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
                if (!tensor)
                    return fail(error, "metadata carrier field is attached to a non-storage Tensor argument");
                tensor->tensorView = &supplied.tensor;
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
    metadataPayload.clear();
    metadataAlignment = 0;
    assignedArguments.clear();
    grid = {};
    commandEncoder = {};
}

std::vector<uint8_t> &PlannedComputeLaunch::appendHostTensorStorage() {
    if (hostTensorStorageCount == hostTensorStorage.size())
        hostTensorStorage.emplace_back();
    return hostTensorStorage[hostTensorStorageCount++];
}

bool materializeMetadataCarrier(const MetadataCarrier &carrier, const std::vector<ComputeLaunchArgument> &arguments,
                                std::vector<uint8_t> &payload, std::string &error) {
    payload.clear();
    const bool i32 = carrier.representation == "i32";
    const bool i64 = carrier.representation == "i64";
    if ((!i32 && !i64) || !carrier.size || carrier.size > std::numeric_limits<size_t>::max() ||
        carrier.fields.empty() || carrier.fields.size() != carrier.members.size()) {
        error = "metadata carrier plan is incomplete";
        return false;
    }
    const int64_t minimum = i32 ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int64_t>::min();
    const int64_t maximum = i32 ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int64_t>::max();
    for (size_t ordinal = 0; ordinal < carrier.fields.size(); ++ordinal) {
        const MetadataFieldIdentity &field = carrier.fields[ordinal];
        if ((field.kind == MetadataFieldKind::Offset) == field.dimension.has_value()) {
            error = "metadata field identity has an invalid dimension";
            return false;
        }
        for (size_t prior = 0; prior < ordinal; ++prior)
            if (carrier.fields[prior].argument == field.argument && carrier.fields[prior].kind == field.kind &&
                carrier.fields[prior].dimension == field.dimension) {
                error = "metadata carrier contains a duplicate semantic field identity";
                return false;
            }
    }
    std::vector<uint8_t> validated(arguments.size());
    std::vector<uint8_t> initialized(carrier.members.size());
    payload.assign(static_cast<size_t>(carrier.size), 0);
    for (size_t memberIndex = 0; memberIndex < carrier.members.size(); ++memberIndex) {
        const PhysicalMetadataMember &member = carrier.members[memberIndex];
        if (member.semanticOrdinal >= carrier.fields.size() || initialized[member.semanticOrdinal]++ ||
            member.byteSize != (i32 ? sizeof(int32_t) : sizeof(int64_t)) || !member.alignment ||
            (member.alignment & (member.alignment - 1)) || member.byteOffset % member.alignment ||
            member.byteOffset > carrier.encodedSize || member.byteSize > carrier.encodedSize - member.byteOffset ||
            member.byteOffset > payload.size() || member.byteSize > payload.size() - member.byteOffset) {
            error = "metadata carrier member mapping is not bijective";
            payload.clear();
            return false;
        }
        for (size_t prior = 0; prior < memberIndex; ++prior) {
            const PhysicalMetadataMember &other = carrier.members[prior];
            if (member.byteOffset < other.byteOffset + other.byteSize &&
                other.byteOffset < member.byteOffset + member.byteSize) {
                error = "metadata carrier physical members overlap";
                payload.clear();
                return false;
            }
        }
        const MetadataFieldIdentity &field = carrier.fields[member.semanticOrdinal];
        if (field.argument >= arguments.size()) {
            error = "metadata field references an unknown argument";
            payload.clear();
            return false;
        }
        const auto *tensorArgument = std::get_if<ComputeTensorArgument>(&arguments[field.argument]);
        const VernonTensorView *tensor = tensorArgument ? tensorArgument->tensorView : nullptr;
        if (!tensor) {
            error = "metadata field source is not a TensorView";
            payload.clear();
            return false;
        }
        uint32_t rank = 0;
        for (const MetadataFieldIdentity &candidate : carrier.fields)
            if (candidate.argument == field.argument && candidate.dimension)
                rank = std::max(rank, *candidate.dimension + 1);
        size_t offsetCount = 0;
        size_t extentCount = 0;
        size_t strideCount = 0;
        for (const MetadataFieldIdentity &candidate : carrier.fields) {
            if (candidate.argument != field.argument)
                continue;
            offsetCount += candidate.kind == MetadataFieldKind::Offset ? 1u : 0u;
            extentCount += candidate.kind == MetadataFieldKind::Extent ? 1u : 0u;
            strideCount += candidate.kind == MetadataFieldKind::Stride ? 1u : 0u;
        }
        if (offsetCount != 1 || extentCount != rank || strideCount != rank) {
            error = "metadata carrier TensorView record is incomplete";
            payload.clear();
            return false;
        }
        if (!validated[field.argument] && !metadataTensorValid(*tensor, rank, minimum, maximum, error)) {
            payload.clear();
            return false;
        }
        validated[field.argument] = 1;
        int64_t semanticValue = 0;
        if (field.kind == MetadataFieldKind::Offset) {
            semanticValue = static_cast<int64_t>(tensor->byte_offset / tensor->element_layout.byte_size);
        } else {
            if (!field.dimension || *field.dimension >= tensor->rank) {
                error = "dimensioned metadata field is outside the TensorView rank";
                payload.clear();
                return false;
            }
            semanticValue =
                field.kind == MetadataFieldKind::Extent
                    ? static_cast<int64_t>(tensor->shape[*field.dimension])
                    : tensor->byte_strides[*field.dimension] / static_cast<int64_t>(tensor->element_layout.byte_size);
        }
        if (i32) {
            const int32_t encoded = static_cast<int32_t>(semanticValue);
            std::memcpy(payload.data() + member.byteOffset, &encoded, sizeof(encoded));
        } else {
            std::memcpy(payload.data() + member.byteOffset, &semanticValue, sizeof(semanticValue));
        }
    }
    if (std::find(initialized.begin(), initialized.end(), uint8_t{0}) != initialized.end()) {
        error = "metadata carrier did not initialize every physical member";
        payload.clear();
        return false;
    }
    return true;
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
            const bool metadataArgument =
                stagePlan.metadataCarrier &&
                std::any_of(stagePlan.metadataCarrier->fields.begin(), stagePlan.metadataCarrier->fields.end(),
                            [&](const MetadataFieldIdentity &field) { return field.argument == use.index; });
            if (!metadataArgument)
                continue;
            if (tensor.rank != use.shape.size() || (tensor.rank && (!tensor.shape || !tensor.byte_strides)) ||
                !tensor.element_layout.byte_size || tensor.byte_offset % tensor.element_layout.byte_size)
                return fail(error, "pipeline TensorView metadata does not match its declared rank");
            for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
                if ((use.shape[dimension] && use.shape[dimension] != tensor.shape[dimension]) ||
                    tensor.byte_strides[dimension] % static_cast<int64_t>(tensor.element_layout.byte_size))
                    return error = "pipeline Tensor argument '" + parameter.name +
                                   "' TensorView metadata violates static shape or element stride at axis " +
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
    if (!planComputeArguments(stagePlan, invocation, plan, error))
        return false;
    if (stagePlan.metadataCarrier) {
        if (!materializeMetadataCarrier(*stagePlan.metadataCarrier, plan.arguments, plan.metadataPayload, error))
            return false;
        plan.metadataAlignment = static_cast<size_t>(stagePlan.metadataCarrier->alignment);
    }
    return true;
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
