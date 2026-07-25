#ifndef VERNON_RUNTIME_BACKEND_CPU_H
#define VERNON_RUNTIME_BACKEND_CPU_H

#include "VernonRuntime.h"
#include "pipeline_bundle.h"
#include "pipeline_metadata.h"
#include "platform_library.h"
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

struct CpuBufferState {
    std::vector<unsigned char> storage;
};

struct CpuPipelineState {
    VernonLoadedKernel *kernel{};
};

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

VernonStatus launchCpuKernel(VernonRuntimeContext &context, const CpuKernelState &state, const ReflectedEntry &metadata,
                             VernonLaunchSize globalSize, const VernonLaunchArgument *arguments, size_t argumentCount,
                             std::string &error);

} // namespace vernon::runtime

#endif
