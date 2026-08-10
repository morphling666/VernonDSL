#ifndef VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H
#define VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H

#include "VernonRuntime.h"
#include "pipeline_manifest.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime {

enum class ComputeLaunchArgumentKind { Tensor, Scalar };

struct ComputeLaunchArgument {
    ComputeLaunchArgumentKind kind{ComputeLaunchArgumentKind::Tensor};
    VernonRuntimeProviderResourceReference resource{};
    const void *hostData{};
    size_t hostSize{};
    const void *scalarData{};
    size_t scalarSize{};
    const VernonTensorView *tensorView{};
};

enum class ComputeBindingSourceKind { Argument, TensorOffset, TensorExtent, TensorStride };

struct ComputeBindingSource {
    ComputeBindingSourceKind kind{ComputeBindingSourceKind::Argument};
    uint32_t argumentIndex{};
    uint32_t dimension{};
};

struct PlannedComputeLaunch {
    std::vector<ComputeLaunchArgument> arguments;
    std::vector<std::vector<uint8_t>> hostTensorStorage;
    VernonLaunchSize grid{};
    VernonRuntimeProviderObject commandEncoder{};
};

std::optional<int64_t> computeBindingDescriptorValue(const ComputeLaunchArgument &argument,
                                                     const ComputeBindingSource &source);

bool planComputeInvocation(const Variant &variant, VernonLaunchSize workgroup,
                           const VernonPipelineInvocation &invocation, PlannedComputeLaunch &plan, std::string &error);

} // namespace vernon::runtime

#endif
