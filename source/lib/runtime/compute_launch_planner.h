#ifndef VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H
#define VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H

#include "VernonRuntime.h"
#include "pipeline_manifest.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::runtime {

using ComputeArgumentMap = std::unordered_map<uint32_t, const VernonPipelineArgument *>;

struct ComputePlannerCallbacks {
    const void *userData{};
    const void *(*bufferContext)(const void *userData, const VernonDeviceBuffer *buffer){};
};

struct PlannedComputeLaunch {
    std::vector<VernonLaunchArgument> arguments;
    std::vector<std::vector<uint8_t>> hostTensorStorage;
    VernonLaunchSize grid{};
};

bool planComputeLaunch(const Variant &variant, const ComputeArgumentMap &arguments,
                       const VernonPipelineInvocation &invocation, const void *expectedContext,
                       const ComputePlannerCallbacks &callbacks, PlannedComputeLaunch &plan, std::string &error);

} // namespace vernon::runtime

#endif
