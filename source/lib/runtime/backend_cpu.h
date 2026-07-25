#ifndef VERNON_RUNTIME_BACKEND_CPU_H
#define VERNON_RUNTIME_BACKEND_CPU_H

#include "../platform/platform_library.h"
#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "VernonRuntimeProvider.h"
#include "pipeline_bundle.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"

#include <string>
#include <vector>

struct VernonDeviceBuffer;
struct VernonRuntimeContext;

namespace vernon::runtime {

struct CpuKernelState {
    VernonCpuEntryPoint entry{};
    PlatformLibrary nativeLibrary;
};

struct CpuContextState {
    VernonRuntimeDeviceProvider provider{};
    std::string error;
};

struct CpuBufferState {
    std::vector<unsigned char> storage;
};

struct CpuPipelineState {
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeCoreBindings *bindings{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> layout;
    std::vector<VernonRuntimeProviderBindingValue> values;
    uint32_t workgroup[3]{1, 1, 1};
};

struct CpuProviderShaderPayload {
    CpuKernelState *kernel{};
    ReflectedEntry *reflection{};
};

bool initializeCpuContext(VernonRuntimeContext &context, uint32_t deviceIndex);
const VernonRuntimeDeviceProvider *cpuProvider(VernonRuntimeContext &context);
uint64_t cpuProviderResourceIdentity(const VernonRuntimeContext &context);
VernonStringView cpuProviderLastError(const VernonRuntimeContext &context);
bool prepareCpuComputePipeline(VernonRuntimeContext &context, CpuKernelState kernel, ReflectedEntry reflection,
                               CpuPipelineState &state);

inline CpuBufferState &cpuBufferState(VernonDeviceBuffer &buffer) {
    return runtimeBackendState<CpuBufferState>(buffer);
}

inline const CpuBufferState &cpuBufferState(const VernonDeviceBuffer &buffer) {
    return runtimeBackendState<CpuBufferState>(buffer);
}

VernonStatus registerStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entry);

bool loadCpuEntry(VernonCpuEntryPoint entry, const char *reflection, size_t reflectionSize, const char *entryName,
                  size_t entryNameSize, CpuKernelState &state, ReflectedEntry &metadata, std::string &error);

bool loadCpuNativeArtifact(const CpuNativeArtifact &artifact, CpuKernelState &state, ReflectedEntry &metadata,
                           std::string &error);

bool createCpuBuffer(VernonDeviceBuffer &buffer);
VernonStatus copyToCpuBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromCpuBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

} // namespace vernon::runtime

#endif
