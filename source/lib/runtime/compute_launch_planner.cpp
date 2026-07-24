#include "compute_launch_planner.h"

#include "tensor_bridge.h"

#include <limits>
#include <optional>
#include <utility>

namespace vernon::runtime {

namespace {

bool fail(std::string &error, const char *message) {
    error = message;
    return false;
}

} // namespace

bool planComputeLaunch(const Variant &variant, const ComputeArgumentMap &arguments,
                       const VernonPipelineInvocation &invocation, const void *expectedContext,
                       const ComputePlannerCallbacks &callbacks, PlannedComputeLaunch &plan, std::string &error) {
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

            VernonLaunchArgument &argument = plan.arguments[use.index];
            if (supplied.kind != VERNON_PIPELINE_TENSOR)
                return fail(error, "compute argument kind is unsupported");
            if (supplied.tensor.storage == VERNON_TENSOR_DEVICE) {
                if (!supplied.tensor.buffer || !callbacks.bufferContext ||
                    callbacks.bufferContext(callbacks.userData, supplied.tensor.buffer) != expectedContext ||
                    !tensorFitsAllocation(supplied.tensor))
                    return fail(error, "compute device Tensor view is invalid");
                argument.kind = VERNON_LAUNCH_TENSOR;
                argument.buffer = supplied.tensor.buffer;
            } else {
                const std::optional<size_t> byteSize = tensorLogicalByteSize(supplied.tensor);
                if (!byteSize)
                    return fail(error, "compute host Tensor is invalid");
                argument.kind = VERNON_LAUNCH_SCALAR;
                argument.scalar_size = *byteSize;
                if (isRowMajorContiguous(supplied.tensor)) {
                    argument.scalar_data = hostTensorData(supplied.tensor);
                } else {
                    auto packed = packTensorRowMajor(supplied.tensor);
                    if (!packed)
                        return fail(error, "failed to pack compute host Tensor");
                    plan.hostTensorStorage.push_back(std::move(*packed));
                    argument.scalar_data = plan.hostTensorStorage.back().data();
                }
            }
            assigned[use.index] = 1;
        }
    }

    plan.grid = invocation.compute_grid;
    if (!plan.grid.x || !plan.grid.y || !plan.grid.z) {
        for (const Parameter &parameter : variant.parameters) {
            const auto argumentIt = arguments.find(parameter.slot);
            if (argumentIt == arguments.end())
                continue;
            const VernonPipelineArgument &argument = *argumentIt->second;
            if (argument.kind != VERNON_PIPELINE_TENSOR || argument.tensor.storage != VERNON_TENSOR_DEVICE ||
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

} // namespace vernon::runtime
