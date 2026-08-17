#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_REPLAY_EXECUTION_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_REPLAY_EXECUTION_H

#include "runtime/autodiff/runtime_gpu_argument_binding.h"
#include "runtime/autodiff/runtime_gpu_bindings.h"
#include "runtime/autodiff/runtime_gpu_commands.h"

namespace vernon::runtime::ad::gpu {

struct ReplayArgumentViews {
    InternalBufferView tape;
    InternalBufferView segment;
    InternalBufferView status;
    InternalBufferView launch;
};

bool planReplayRestoreCopies(const VernonLoadedPipeline &forward, DeviceValues &working, DeviceValues &retained,
                             std::vector<DeviceBufferCopy> &copies);

bool appendReplayArguments(VernonRuntimeContext &context, const BindingPlan &bindings, DeviceBuffer &tape,
                           size_t tapeBytes, DeviceBuffer &segment, size_t segmentBytes, DeviceBuffer &status,
                           size_t statusBytes, DeviceBuffer &launch, size_t launchBytes, ReplayArgumentViews &views,
                           std::vector<VernonPipelineArgument> &arguments, std::string &failedParameter);

} // namespace vernon::runtime::ad::gpu

#endif
