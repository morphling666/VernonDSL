#ifndef VERNON_RUNTIME_RUNTIME_DISPATCH_H
#define VERNON_RUNTIME_RUNTIME_DISPATCH_H

#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "program_execution/invocation_outcome.h"
#include "program_execution/materialized_node_frame.h"
#include "program_execution/program_invocation_state.h"
#include "program_execution_manifest.h"
#include "resolved_stage_invocation.h"
#include "runtime_state.h"

#include <memory>
#include <string>
#include <vector>

struct VernonProgramExecutable;
struct VernonProgramBundle;
struct VernonRuntimeContext;

namespace vernon::runtime {

template <typename T> using RuntimeResult = vernon::Result<T, vernon::RuntimeError>;

VernonStatus executePipelineProgramGraph(VernonRuntimeContext &context, const program::ResolvedExecutionPlan &execution,
                                         const program::Graph &graph, program_execution::ProgramInvocationState &frame,
                                         const program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint,
                                         program_execution::SubmissionState &submission);

bool isOpenGLBackend(VernonRuntimeBackend backend);
RuntimeResult<void> probeBackend(VernonRuntimeBackend backend, std::string &diagnostic);
RuntimeResult<void> initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex);
RuntimeResult<void> initializeBackendForRhiDevice(VernonRuntimeContext &context, VernonRhiDevice device);
void destroyBackend(VernonRuntimeContext &context);
void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &capabilities);
RuntimeResult<void> validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements);

RuntimeResult<void> registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint);
using BackendStageLoadResult = vernon::Result<std::unique_ptr<VernonStageExecutable>, BackendPipelineFailure>;
BackendStageLoadResult loadBackendTypedComputePipeline(VernonRuntimeContext &context, StageBindingPlan stagePlan,
                                                       ReflectedEntry reflection, const void *artifact,
                                                       size_t artifactSize, const std::string &entry,
                                                       VernonCpuEntryPoint cpuEntry,
                                                       const std::vector<NativeResourceSlot> &nativeSlots);
BackendPipelineResult resolveBackendPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                             VernonStageExecutable &pipeline);
void destroyBackendPipeline(VernonStageExecutable &pipeline);
RuntimeResult<void> invokeBackendPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan);
RuntimeResult<void> invokeBackendComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

RuntimeResult<VernonRuntimeProviderResourceReference>
referenceBackendRhiBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, uint64_t size);
RuntimeResult<VernonRhiBuffer>
resolveBackendRhiBufferReference(VernonRuntimeContext &context,
                                 const VernonRuntimeProviderResourceReference &reference);
RuntimeResult<VernonRuntimeProviderResourceReference> referenceBackendRhiImageView(VernonRuntimeContext &context,
                                                                                   VernonRhiImageView view);
RuntimeResult<VernonRuntimeProviderResourceReference> referenceBackendRhiSampler(VernonRuntimeContext &context,
                                                                                 VernonRhiSampler sampler);
RuntimeResult<VernonRuntimeProviderImageDescription>
describeBackendImage(VernonRuntimeContext &context, VernonRuntimeProviderResourceReference resource);
RuntimeResult<VernonRuntimeProviderObject> referenceBackendCommandEncoder(VernonRuntimeContext &context,
                                                                          VernonRhiCommandEncoder encoder);

} // namespace vernon::runtime

#endif
