#ifndef VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H
#define VERNON_RUNTIME_COMPUTE_LAUNCH_PLANNER_H

#include "VernonRuntime.h"
#include "resolved_stage_types.h"
#include "stage_binding_plan.h"
#include "tensor_bridge.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace vernon::runtime {

struct ComputeTensorArgument {
    VernonRuntimeProviderResourceReference resource{};
    const void *hostData{};
    size_t hostSize{};
    const VernonTensorView *tensorView{};
};

struct ComputeImageArgument {
    VernonRuntimeProviderResourceReference view{};
};

struct ComputeScalarArgument {
    const void *data{};
    size_t size{};
};

struct ComputeSamplerArgument {
    VernonRuntimeProviderResourceReference resource{};
};

using ComputeLaunchArgument =
    std::variant<ComputeTensorArgument, ComputeImageArgument, ComputeScalarArgument, ComputeSamplerArgument>;

struct ResultCommitPlan {
    size_t storageIndex{};
    VernonTensorView destination{};
    TensorCopyPlan layout;
};

struct PlannedComputeLaunch {
    std::vector<ComputeLaunchArgument> arguments;
    std::vector<std::vector<uint8_t>> hostTensorStorage;
    std::vector<ResultCommitPlan> resultCommits;
    std::vector<const VernonTensorView *> validationTensors;
    std::vector<uint8_t> metadataPayload;
    size_t metadataAlignment{};
    std::vector<uint8_t> assignedArguments;
    size_t hostTensorStorageCount{};
    VernonLaunchSize grid{};
    VernonRuntimeProviderObject commandEncoder{};

    void reset();
    std::vector<uint8_t> &appendHostTensorStorage();
};

bool materializeMetadataCarrier(const MetadataCarrier &carrier, const std::vector<ComputeLaunchArgument> &arguments,
                                std::vector<uint8_t> &payload, std::string &error);

bool planComputeInvocation(const StageBindingPlan &stagePlan, const VernonStageInvocationDescriptor &invocation,
                           PlannedComputeLaunch &plan, std::string &error);
bool commitComputeResults(const PlannedComputeLaunch &plan, std::string &error);

} // namespace vernon::runtime

#endif
