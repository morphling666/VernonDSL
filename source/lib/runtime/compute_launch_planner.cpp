#include "compute_launch_planner.h"

#include "pipeline_metadata.h"
#include "tensor_bridge.h"

#include <algorithm>
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

bool planComputeArguments(const Variant &variant, const ComputeArgumentMap &arguments,
                          const VernonPipelineInvocation &invocation, PlannedComputeLaunch &plan, std::string &error) {
    size_t computeArgumentCount = 0;
    for (const Parameter &parameter : variant.parameters)
        for (const ParameterUse &use : parameter.uses)
            computeArgumentCount += use.stage == "compute" || use.stage == variant.compute;

    plan = {};
    plan.arguments.resize(computeArgumentCount);
    plan.hostTensorStorage.reserve(computeArgumentCount);
    std::vector<uint8_t> assigned(computeArgumentCount);

    for (const Parameter &parameter : variant.parameters) {
        const auto suppliedIt = arguments.find(parameter.slot);
        if (suppliedIt == arguments.end())
            return fail(error, "compute argument is missing");
        const VernonPipelineArgument &supplied = *suppliedIt->second;
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != variant.compute)
                continue;
            if (use.index >= computeArgumentCount || assigned[use.index])
                return fail(error, "compute argument indices are not contiguous");

            ComputeLaunchArgument &argument = plan.arguments[use.index];
            if (supplied.kind != VERNON_PIPELINE_TENSOR)
                return fail(error, "compute argument kind is unsupported");
            if (supplied.tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                if (!supplied.tensor.resource.identity || !supplied.tensor.resource.resource.value ||
                    supplied.tensor.byte_offset > supplied.tensor.resource.size ||
                    supplied.tensor.byte_size > supplied.tensor.resource.size)
                    return fail(error, "compute RHI Tensor view is invalid");
                argument.kind = ComputeLaunchArgumentKind::Tensor;
                argument.resource = supplied.tensor.resource;
            } else {
                if (use.interfaceKind == "storage") {
                    if (supplied.tensor.storage != VERNON_TENSOR_HOST || !supplied.tensor.host_data ||
                        !tensorFitsAllocation(supplied.tensor))
                        return fail(error, "compute host storage Tensor is invalid");
                    argument.kind = ComputeLaunchArgumentKind::Tensor;
                    argument.hostData = supplied.tensor.host_data;
                    argument.hostSize = supplied.tensor.byte_size;
                    assigned[use.index] = 1;
                    continue;
                }
                argument.kind = ComputeLaunchArgumentKind::Scalar;
                if (!use.physicalValueLayout)
                    return fail(error, "compute value Tensor is missing its compiler-planned physical layout");
                if (use.physicalValueLayout->size > std::numeric_limits<size_t>::max())
                    return fail(error, "compute Tensor physical size exceeds the host size range");
                TensorPackingLayout layout;
                layout.elementSize = supplied.tensor.element_layout.byte_size;
                layout.shape = use.shape.empty() ? parameter.shape : use.shape;
                layout.byteSize = static_cast<size_t>(use.physicalValueLayout->size);
                for (uint64_t stride : use.physicalValueLayout->byteStrides) {
                    if (stride > std::numeric_limits<size_t>::max())
                        return fail(error, "compute Tensor physical stride exceeds the host size range");
                    layout.byteStrides.push_back(static_cast<size_t>(stride));
                }
                VernonTensorView packedTensor = supplied.tensor;
                if (packedTensor.rank == layout.shape.size() + 1 && packedTensor.shape && packedTensor.byte_strides &&
                    packedTensor.shape[0] == 1) {
                    --packedTensor.rank;
                    ++packedTensor.shape;
                    ++packedTensor.byte_strides;
                }
                auto packed = packTensor(packedTensor, layout);
                if (!packed)
                    return fail(error, "failed to pack reflected compute Tensor layout");
                plan.hostTensorStorage.push_back(std::move(*packed));
                argument.scalarData = plan.hostTensorStorage.back().data();
                argument.scalarSize = plan.hostTensorStorage.back().size();
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
                !argument.tensor.rank || !argument.tensor.shape)
                continue;
            const uint32_t rank = argument.tensor.rank;
            const uint32_t gridDimensions = rank < 3 ? rank : 3;
            for (uint32_t dimension = 0; dimension < gridDimensions; ++dimension) {
                if (argument.tensor.shape[rank - 1 - dimension] > std::numeric_limits<uint32_t>::max())
                    return fail(error, "inferred compute grid exceeds uint32 range");
            }
            plan.grid = {1, 1, 1};
            plan.grid.x = static_cast<uint32_t>(argument.tensor.shape[rank - 1]);
            if (rank > 1)
                plan.grid.y = static_cast<uint32_t>(argument.tensor.shape[rank - 2]);
            if (rank > 2)
                plan.grid.z = static_cast<uint32_t>(argument.tensor.shape[rank - 3]);
            break;
        }
    }
    if (!plan.grid.x || !plan.grid.y || !plan.grid.z)
        return fail(error, "compute grid cannot be inferred");
    return true;
}

} // namespace

bool planComputeInvocation(const Variant &variant, const VernonPipelineInvocation &invocation,
                           PlannedComputeLaunch &plan, std::string &error) {
    ComputeArgumentMap arguments;
    for (size_t index = 0; index < invocation.argument_count; ++index)
        if (!arguments.emplace(invocation.arguments[index].slot, &invocation.arguments[index]).second)
            return fail(error, "duplicate pipeline argument slot");
    if (arguments.size() != variant.parameters.size())
        return fail(error, "pipeline argument count does not match layout");

    for (const Parameter &parameter : variant.parameters) {
        const auto found = arguments.find(parameter.slot);
        if (found == arguments.end() || parameter.kind != "tensor" || found->second->kind != VERNON_PIPELINE_TENSOR)
            return fail(error, "pipeline argument kind does not match layout");
        const VernonTensorView &tensor = found->second->tensor;
        if (tensor.struct_size < sizeof(VernonTensorView) ||
            !valueLayoutsEqual(tensor.element_layout, pipelineValueLayout(parameter.elementLayout)) ||
            tensor.access > VERNON_ACCESS_READ_WRITE ||
            (tensor.storage != VERNON_TENSOR_HOST && tensor.storage != VERNON_TENSOR_RHI_RESOURCE) ||
            (tensor.storage == VERNON_TENSOR_HOST && !tensorFitsAllocation(tensor)))
            return fail(error, "pipeline Tensor argument does not match layout");
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
    }
    return planComputeArguments(variant, arguments, invocation, plan, error);
}

} // namespace vernon::runtime
