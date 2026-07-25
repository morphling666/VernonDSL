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
    VernonDeviceBuffer *buffer{};
    VernonRuntimeProviderResourceReference resource{};
    const void *scalarData{};
    size_t scalarSize{};
};

struct ComputePlannerCallbacks {
    const void *userData{};
    const void *(*bufferContext)(const void *userData, const VernonDeviceBuffer *buffer){};
};

struct PlannedComputeLaunch {
    std::vector<ComputeLaunchArgument> arguments;
    std::vector<std::vector<uint8_t>> hostTensorStorage;
    VernonLaunchSize grid{};
};

bool planComputeInvocation(const Variant &variant, const VernonPipelineInvocation &invocation,
                           const void *expectedContext, const ComputePlannerCallbacks &callbacks,
                           PlannedComputeLaunch &plan, std::string &error);

} // namespace vernon::runtime

#endif
