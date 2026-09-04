#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_COMMANDS_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_COMMANDS_H

#include "VernonRuntime.h"
#include "execution_graph/execution_command_model.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"

#include <cstddef>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::execution::detail {
class RhiCommandPlanSink;
struct RhiCommandNodeEncoder;
struct RhiCommandResourceBinding;
struct RhiCommandExecutionPlan;
} // namespace vernon::execution::detail

namespace vernon::runtime::ad::gpu {

struct DeviceBufferCopy {
    VernonRhiBuffer source;
    VernonRhiBuffer destination;
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

struct DeviceBufferUpload {
    VernonRhiBuffer destination;
    size_t destinationOffset{};
    const void *source{};
    size_t size{};
};

using GpuCommandCompletionCallback = VernonRhiStatus (*)(void *context);

bool encodeBufferCopies(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                        const std::vector<DeviceBufferCopy> &copies);
VernonStatus executeBufferCopiesAndWait(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copies);
VernonStatus buildBufferUploadCommandPlan(VernonRuntimeContext &context, const std::vector<DeviceBufferUpload> &uploads,
                                          execution::detail::RhiCommandExecutionPlan &plan);
VernonStatus encodePipelineCommand(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                                   VernonLoadedPipeline &pipeline, std::vector<VernonPipelineArgument> &arguments,
                                   VernonLaunchSize grid);
VernonStatus executeCommandPlanAndWait(VernonRuntimeContext &context,
                                       const execution::detail::RhiCommandExecutionPlan &plan,
                                       PullbackControlPlaneUsage *telemetry = nullptr,
                                       execution::detail::RhiCommandPlanSink *sink = nullptr, bool flush = false);
VernonStatus buildPipelineCommandPlan(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copiesBefore,
                                      const std::vector<DeviceBufferUpload> &uploadsBefore,
                                      VernonLoadedPipeline &pipeline,
                                      const std::vector<VernonPipelineArgument> &arguments, VernonLaunchSize grid,
                                      const std::vector<DeviceBufferCopy> &copiesAfter,
                                      execution::detail::CommandNodeKind kind,
                                      execution::detail::RhiCommandExecutionPlan &plan);

VernonStatus executePipelineCommandDagAndWait(VernonLoadedPipeline &pipeline, VernonLaunchSize grid,
                                              std::vector<VernonPipelineArgument> &arguments,
                                              const std::vector<DeviceBufferUpload> &uploadsBefore,
                                              execution::detail::CommandNodeKind kind,
                                              PullbackControlPlaneUsage *telemetry = nullptr,
                                              execution::detail::RhiCommandPlanSink *sink = nullptr);
VernonStatus executePipelineCommandDagAndWait(
    VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copiesBefore,
    const std::vector<DeviceBufferUpload> &uploadsBefore, VernonLoadedPipeline &pipeline,
    std::vector<VernonPipelineArgument> &arguments, VernonLaunchSize grid,
    const std::vector<DeviceBufferCopy> &copiesAfter, execution::detail::CommandNodeKind kind,
    PullbackControlPlaneUsage *telemetry = nullptr, execution::detail::RhiCommandPlanSink *sink = nullptr);
VernonStatus executePipelineStatusCommandDagAndWait(
    VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copiesBefore, VernonLoadedPipeline &pipeline,
    std::vector<VernonPipelineArgument> &arguments, VernonLaunchSize grid,
    const std::vector<DeviceBufferUpload> &uploadsBefore, VernonRhiBuffer statusBuffer, size_t statusOffset,
    size_t statusSize, GpuCommandCompletionCallback complete, void *completionContext,
    execution::detail::CommandNodeKind kind, PullbackControlPlaneUsage *telemetry = nullptr,
    execution::detail::RhiCommandPlanSink *sink = nullptr);

} // namespace vernon::runtime::ad::gpu

#endif
