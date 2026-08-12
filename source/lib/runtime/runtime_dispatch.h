#ifndef VERNON_RUNTIME_RUNTIME_DISPATCH_H
#define VERNON_RUNTIME_RUNTIME_DISPATCH_H

#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "runtime_state.h"

#include <string>

namespace vernon::runtime {

bool isOpenGLBackend(VernonRuntimeBackend backend);
bool probeBackend(VernonRuntimeBackend backend, std::string &diagnostic);
bool initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex);
bool initializeBackendForRhiDevice(VernonRuntimeContext &context, VernonRhiDevice device);
void destroyBackend(VernonRuntimeContext &context);
void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &capabilities);
bool validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements);

VernonStatus registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint);
VernonLoadedPipeline *loadBackendCpuEntryPipeline(VernonRuntimeContext &context, VernonCpuEntryPoint entryPoint,
                                                  const char *reflection, size_t reflectionSize, const char *entry,
                                                  size_t entrySize);
VernonLoadedPipeline *loadBackendArtifactPipeline(VernonRuntimeContext &context, const void *artifact,
                                                  size_t artifactSize, const char *reflection, size_t reflectionSize,
                                                  const char *entry, size_t entrySize);
bool buildReflectedComputeVariant(const Stage &stage, VernonRuntimeBackend backend, Variant &variant,
                                  std::string &error);

bool resolveBackendPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
void destroyBackendPipeline(VernonLoadedPipeline &pipeline);
VernonStatus invokeBackendPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                   const PlannedGraphicsInvocation &plan);
VernonStatus invokeBackendComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan);

VernonStatus referenceBackendRhiBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset,
                                       uint64_t size, VernonRuntimeProviderResourceReference &output);
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
