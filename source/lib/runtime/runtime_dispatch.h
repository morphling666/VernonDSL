#ifndef VERNON_RUNTIME_RUNTIME_DISPATCH_H
#define VERNON_RUNTIME_RUNTIME_DISPATCH_H

#include "graphics_invocation_planner.h"
#include "runtime_state.h"

#include <string>

namespace vernon::runtime {

bool isOpenGLBackend(VernonRuntimeBackend backend);
bool probeBackend(VernonRuntimeBackend backend, std::string &diagnostic);
bool initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex);
bool initializeOpenGLBackend(VernonRuntimeContext &context, const VernonExternalOpenGLContext &externalContext);
void destroyBackend(VernonRuntimeContext &context);
void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &capabilities);

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

VernonLoadedKernel *loadBackendCpuEntry(VernonRuntimeContext &context, VernonCpuEntryPoint entryPoint,
                                        const char *reflection, size_t reflectionSize, const char *entry,
                                        size_t entrySize);
VernonStatus registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint);
VernonLoadedKernel *loadBackendCpuNativeArtifact(VernonRuntimeContext &context, const CpuNativeArtifact &artifact);
VernonLoadedKernel *loadBackendArtifact(VernonRuntimeContext &context, const void *artifact, size_t artifactSize,
                                        const char *reflection, size_t reflectionSize, const char *entry,
                                        size_t entrySize);
VernonStatus unloadBackendKernel(VernonLoadedKernel &kernel);
VernonStatus launchBackendKernel(VernonLoadedKernel &kernel, VernonLaunchSize globalSize,
                                 const VernonLaunchArgument *arguments, size_t argumentCount);

VernonLoadedKernel *backendPipelineComputeKernel(VernonLoadedPipeline &pipeline);
bool resolveBackendPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
void destroyBackendPipeline(VernonLoadedPipeline &pipeline);
VernonStatus invokeBackendPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                   const PlannedGraphicsInvocation &plan);

VernonStatus backendComputeToGraphicsBarrier(VernonRuntimeContext &context);
VernonStatus synchronizeBackend(VernonRuntimeContext &context);

} // namespace vernon::runtime

#endif
