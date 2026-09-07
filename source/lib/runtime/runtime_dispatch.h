#ifndef VERNON_RUNTIME_RUNTIME_DISPATCH_H
#define VERNON_RUNTIME_RUNTIME_DISPATCH_H

#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "program_execution/materialized_node_frame.h"
#include "program_execution/program_invocation_state.h"
#include "program_execution_manifest.h"
#include "runtime_state.h"

#include <string>
#include <vector>

struct VernonProgramExecutable;
struct VernonProgramBundle;
struct VernonRuntimeContext;

namespace vernon::runtime {

bool isOpenGLBackend(VernonRuntimeBackend backend);
bool probeBackend(VernonRuntimeBackend backend, std::string &diagnostic);
bool initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex);
bool initializeBackendForRhiDevice(VernonRuntimeContext &context, VernonRhiDevice device);
void destroyBackend(VernonRuntimeContext &context);
void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &capabilities);
bool validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements);

VernonStatus registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint);
VernonStageExecutable *loadBackendCpuEntryPipeline(VernonRuntimeContext &context, VernonCpuEntryPoint entryPoint,
                                                   const char *reflection, size_t reflectionSize, const char *entry,
                                                   size_t entrySize);
VernonStageExecutable *loadBackendArtifactPipeline(VernonRuntimeContext &context, const void *artifact,
                                                   size_t artifactSize, const char *reflection, size_t reflectionSize,
                                                   const char *entry, size_t entrySize);
VernonStageExecutable *loadBackendTypedComputePipeline(VernonRuntimeContext &context, Variant variant,
                                                       ReflectedEntry reflection, const void *artifact,
                                                       size_t artifactSize, const std::string &entry,
                                                       VernonCpuEntryPoint cpuEntry,
                                                       const std::vector<NativeResourceSlot> &nativeSlots);
bool buildDirectComputeStage(VernonRuntimeContext &context, const void *artifact, size_t artifactSize,
                             const char *reflection, size_t reflectionSize, const char *entry, size_t entrySize,
                             Stage &stage, Variant &variant, ReflectedEntry &reflectedEntry);
bool buildReflectedComputeVariant(const Stage &stage, VernonRuntimeBackend backend, Variant &variant,
                                  std::string &error);
bool isDirectPipelineTopology(const Variant &variant);

bool resolveBackendPipeline(BackendPipelineBundle &bundle, const Variant &variant, VernonStageExecutable &pipeline);
void destroyBackendPipeline(VernonStageExecutable &pipeline);
VernonStatus invokeBackendPipeline(VernonStageExecutable &pipeline, const VernonStageInvocationDescriptor &invocation,
                                   const PlannedGraphicsInvocation &plan);
VernonStatus invokeBackendComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);
VernonStatus executePipelineProgramGraph(VernonProgramExecutable &pipeline, const program::Graph &graph,
                                         program_execution::ProgramInvocationState &frame,
                                         const program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint);

VernonStatus referenceBackendRhiBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset,
                                       uint64_t size, VernonRuntimeProviderResourceReference &output);
bool resolveBackendRhiBufferReference(VernonRuntimeContext &context,
                                      const VernonRuntimeProviderResourceReference &reference, VernonRhiBuffer &output);
VernonStatus referenceBackendRhiImageView(VernonRuntimeContext &context, VernonRhiImageView view,
                                          VernonRuntimeProviderResourceReference &output);
VernonStatus referenceBackendRhiSampler(VernonRuntimeContext &context, VernonRhiSampler sampler,
                                        VernonRuntimeProviderResourceReference &output);
VernonStatus describeBackendImage(VernonRuntimeContext &context, VernonRuntimeProviderResourceReference resource,
                                  VernonRuntimeProviderImageDescription &description);
VernonStatus referenceBackendCommandEncoder(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                                            VernonRuntimeProviderObject &output);

} // namespace vernon::runtime

#endif
