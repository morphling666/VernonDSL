#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_REPLAY_EXECUTION_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_REPLAY_EXECUTION_H

#include "runtime/autodiff/runtime_gpu_argument_binding.h"
#include "runtime/autodiff/runtime_gpu_bindings.h"
#include "runtime/program_execution/device_commands.h"

namespace vernon::runtime::ad::gpu {

struct ReplayArgumentViews {
    program_execution::PhysicalBufferView tape;
    program_execution::PhysicalBufferView segment;
    program_execution::PhysicalBufferView status;
    program_execution::PhysicalBufferView launch;
};

bool planReplayRestoreCopies(const VernonStageExecutable &forward, DeviceValues &working, DeviceValues &retained,
                             std::vector<program_execution::DeviceBufferCopy> &copies);

bool appendReplayArguments(VernonRuntimeContext &context, const BindingPlan &bindings,
                           program_execution::DeviceBuffer &tape, size_t tapeBytes,
                           program_execution::DeviceBuffer &segment, size_t segmentBytes,
                           program_execution::DeviceBuffer &status, size_t statusBytes,
                           program_execution::DeviceBuffer &launch, size_t launchBytes, ReplayArgumentViews &views,
                           std::vector<VernonProgramArgument> &arguments, std::string &failedParameter);

} // namespace vernon::runtime::ad::gpu

#endif
