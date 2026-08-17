#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_PREPARATION_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_PREPARATION_H

#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/autodiff/runtime_gpu_resources.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::runtime::ad::gpu {

struct PreparedForwardExecution {
    std::shared_ptr<Signature> signature;
    DeviceValues working;
    DeviceValues retainedDevices;
    HostValues retainedHosts;
};

struct StagedForwardValue {
    void *destination{};
    std::vector<uint8_t> bytes;
};

struct PreparedForwardPublication {
    std::vector<StagedForwardValue> values;
};

VernonStatus prepareForward(VernonRuntimeContext &context, OwnedPipeline &pipeline,
                            const std::shared_ptr<Signature> &signature, VernonLaunchSize computeGrid,
                            const VernonAdValueSet &inputs, const ForwardExecutionTarget &target,
                            const std::vector<std::string> &retainedNames, const char *resourceName,
                            PreparedForwardExecution &prepared);

bool stageForwardResults(const Signature &signature, const OwnedPipeline &pipeline, DeviceValues &working,
                         const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         PreparedForwardPublication &publication);
bool publishForwardResults(const PreparedForwardPublication &publication);

} // namespace vernon::runtime::ad::gpu

#endif
