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
bool initializeOpenGLBackend(VernonRuntimeContext &context, const VernonOpenGLContextCallbacks &callbacks);
void destroyBackend(VernonRuntimeContext &context);
void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &capabilities);
bool validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements);

bool createBackendBuffer(VernonDeviceBuffer &buffer);
void importBackendOpenGLBuffer(VernonDeviceBuffer &buffer, uint32_t name);
VernonStatus destroyBackendBuffer(VernonDeviceBuffer &buffer);
VernonStatus copyToBackendBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromBackendBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

bool createBackendTexture(VernonDeviceTexture &texture);
void importBackendOpenGLTexture(VernonDeviceTexture &texture, uint32_t name);
void destroyBackendTexture(VernonDeviceTexture &texture);
VernonStatus copyToBackendTexture(VernonDeviceTexture &texture, const void *source, size_t size);
VernonStatus copyFromBackendTexture(const VernonDeviceTexture &texture, void *destination, size_t size);

bool createBackendSampler(VernonDeviceSampler &sampler);
void importBackendOpenGLSampler(VernonDeviceSampler &sampler, uint32_t name);
void destroyBackendSampler(VernonDeviceSampler &sampler);

VernonStatus registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint);
VernonLoadedPipeline *loadBackendCpuEntryPipeline(VernonRuntimeContext &context, VernonCpuEntryPoint entryPoint,
                                                  const char *reflection, size_t reflectionSize, const char *entry,
                                                  size_t entrySize);
VernonLoadedPipeline *loadBackendCpuNativePipeline(VernonRuntimeContext &context, const CpuNativeArtifact &artifact);
VernonLoadedPipeline *loadBackendArtifactPipeline(VernonRuntimeContext &context, const void *artifact,
                                                  size_t artifactSize, const char *reflection, size_t reflectionSize,
                                                  const char *entry, size_t entrySize);

bool resolveBackendPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
void destroyBackendPipeline(VernonLoadedPipeline &pipeline);
VernonStatus invokeBackendPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                   const PlannedGraphicsInvocation &plan);
VernonStatus invokeBackendComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan);

VernonStatus backendComputeToGraphicsBarrier(VernonRuntimeContext &context);
VernonStatus synchronizeBackend(VernonRuntimeContext &context);
VernonStatus referenceBackendRhiBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset,
                                       uint64_t size, VernonRuntimeProviderResourceReference &output);
VernonStatus referenceBackendRhiImage(VernonRuntimeContext &context, VernonRhiImage image,
                                      VernonRuntimeProviderResourceReference &output);
VernonStatus referenceBackendRhiSampler(VernonRuntimeContext &context, VernonRhiSampler sampler,
                                        VernonRuntimeProviderResourceReference &output);

} // namespace vernon::runtime

#endif
