#ifndef VERNON_RUNTIME_RUNTIME_DISPATCH_H
#define VERNON_RUNTIME_RUNTIME_DISPATCH_H

#include "VernonRuntimeProvider.h"
#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_groups.h"
#include "program_execution_manifest.h"
#include "resolved_stage_invocation.h"
#include "runtime_state.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct VernonProgramExecutable;
struct VernonProgramBundle;
struct VernonRuntimeContext;

namespace vernon::hal::cuda {
struct DeviceState;
} // namespace vernon::hal::cuda
namespace vernon::hal::opengl {
struct DeviceState;
} // namespace vernon::hal::opengl
namespace vernon::hal::vulkan {
struct DeviceState;
} // namespace vernon::hal::vulkan

namespace vernon::runtime {

template <typename T> using RuntimeResult = vernon::Result<T, vernon::RuntimeError>;

// A context whose backend `initialize` sets up over a device that the HAL created.
RuntimeResult<std::unique_ptr<VernonRuntimeContext>>
createRuntimeContextForHal(VernonRuntimeBackend backend,
                           const std::function<RuntimeResult<void>(VernonRuntimeContext &)> &initialize);

bool isOpenGLBackend(VernonRuntimeBackend backend);
RuntimeResult<void> probeBackend(VernonRuntimeBackend backend, std::string &diagnostic);
RuntimeResult<void> initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex);
// A CUDA context runs on `device`, which the HAL created over `state`.
RuntimeResult<void> initializeCudaBackend(VernonRuntimeContext &context, std::shared_ptr<hal::Device> device,
                                          const hal::cuda::DeviceState &state);
// An OpenGL context reaches the host's context through `state`; its HAL device is created on first use.
RuntimeResult<void> initializeOpenGLBackend(VernonRuntimeContext &context,
                                            std::shared_ptr<hal::opengl::DeviceState> state);
// A Vulkan context runs on the device of `state`; its HAL device is created on first use.
RuntimeResult<void> initializeVulkanBackend(VernonRuntimeContext &context,
                                            std::shared_ptr<hal::vulkan::DeviceState> state);
void destroyBackend(VernonRuntimeContext &context);
void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &capabilities);
RuntimeResult<void> validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements);

RuntimeResult<void> registerBackendStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count);
RuntimeResult<void> unregisterBackendStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count);
using BackendStageLoadResult = vernon::Result<std::unique_ptr<VernonStageExecutable>, BackendPipelineFailure>;
BackendStageLoadResult loadBackendTypedComputePipeline(VernonRuntimeContext &context, StageBindingPlan stagePlan,
                                                       ReflectedEntry reflection, const void *artifact,
                                                       size_t artifactSize, const std::string &entry,
                                                       const std::vector<NativeResourceSlot> &nativeSlots);
BackendPipelineResult resolveBackendPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                             VernonStageExecutable &pipeline);
void destroyBackendPipeline(VernonStageExecutable &pipeline);

// Describes a HAL image view, the only image a schedule binds.
RuntimeResult<VernonRuntimeProviderImageDescription>
describeBackendImage(VernonRuntimeContext &context, VernonRuntimeProviderResourceReference resource);

} // namespace vernon::runtime

#endif
