#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_COMMANDS_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_COMMANDS_H

#include "VernonRuntime.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"

#include <cstddef>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime::ad::gpu {

struct DeviceBufferCopy {
    VernonRhiBuffer source;
    VernonRhiBuffer destination;
    size_t offset{};
    size_t size{};
};

VernonStatus submitAndWait(VernonLoadedPipeline &pipeline, VernonLaunchSize grid,
                           std::vector<VernonPipelineArgument> &arguments,
                           PullbackControlPlaneUsage *telemetry = nullptr);
VernonStatus submitWithCopiesAndWait(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copies,
                                     VernonLoadedPipeline &pipeline, std::vector<VernonPipelineArgument> &arguments,
                                     VernonLaunchSize grid, PullbackControlPlaneUsage *telemetry = nullptr);

} // namespace vernon::runtime::ad::gpu

#endif
