#ifndef VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H
#define VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H

#include "VernonRuntime.h"
#include "pipeline_manifest.h"

#include <cstdint>
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
};

struct PlannedComputeLaunch {
    std::vector<ComputeLaunchArgument> arguments;
    std::vector<std::vector<uint8_t>> hostTensorStorage;
    VernonLaunchSize grid{};
};

bool planComputeInvocation(const Variant &variant, const VernonPipelineInvocation &invocation,
                           PlannedComputeLaunch &plan, std::string &error);

} // namespace vernon::runtime

#endif
