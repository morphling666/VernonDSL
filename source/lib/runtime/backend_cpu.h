#ifndef VERNON_RUNTIME_BACKEND_CPU_H
#define VERNON_RUNTIME_BACKEND_CPU_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "VernonRuntimeProvider.h"
#include "cpu_workgroup_dispatch.h"
#include "pipeline_bundle.h"
#include "pipeline_metadata.h"
#include "platform/platform_library.h"
#include "runtime/autodiff/tape_allocator_abi.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime {

struct CpuKernelState {
    VernonCpuEntryPoint entry{};
    PlatformLibrary nativeLibrary;
};

struct CpuContextState {
    VernonRuntimeDeviceProvider provider{};
    std::unique_ptr<CpuWorkgroupScheduler> scheduler;
    std::string error;
};

struct CpuPipelineState {
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeCoreBindings *bindings{};
    VernonCpuEntryPoint entry{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> layout;
    std::vector<VernonRuntimeProviderBindingValue> values;
    std::vector<std::string> layoutBuiltins;
    std::vector<size_t> packedOffsets;
    std::vector<size_t> packedFieldSizes;
    size_t packedSize{};
    size_t tapeAllocatorOffset{std::numeric_limits<size_t>::max()};
    size_t tapeRootOffset{std::numeric_limits<size_t>::max()};
    VernonAdTapeAllocator *tapeAllocator{};
    VernonAdRegionHandle tapeRoot{};
    uint32_t workgroup[3]{1, 1, 1};
};

struct CpuProviderShaderPayload {
    CpuKernelState *kernel{};
    ReflectedEntry *reflection{};
};

bool initializeCpuContext(VernonRuntimeContext &context, uint32_t deviceIndex);
const VernonRuntimeDeviceProvider *cpuProvider(VernonRuntimeContext &context);
CpuWorkgroupScheduler &cpuWorkgroupScheduler(VernonRuntimeContext &context);
uint64_t cpuProviderResourceIdentity(const VernonRuntimeContext &context);
VernonStringView cpuProviderLastError(const VernonRuntimeContext &context);
bool prepareCpuComputePipeline(VernonRuntimeContext &context, CpuKernelState kernel, ReflectedEntry reflection,
                               CpuPipelineState &state);
void setCpuProgramTape(VernonLoadedPipeline &pipeline, VernonAdTapeAllocator *allocator, VernonAdRegionHandle root);

VernonStatus registerStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entry);

bool loadCpuEntry(VernonCpuEntryPoint entry, const char *reflection, size_t reflectionSize, const char *entryName,
                  size_t entryNameSize, CpuKernelState &state, ReflectedEntry &metadata, std::string &error);

bool loadCpuNativeArtifact(VernonRuntimeContext &context, const CpuNativeArtifact &artifact, CpuKernelState &state,
                           ReflectedEntry &metadata, std::string &error);
bool findRegisteredCpuEntry(VernonRuntimeContext &context, const std::string &symbol, VernonCpuEntryPoint &entry,
                            std::string &error);

} // namespace vernon::runtime

#endif
