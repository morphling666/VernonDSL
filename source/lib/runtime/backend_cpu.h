#ifndef VERNON_RUNTIME_BACKEND_CPU_H
#define VERNON_RUNTIME_BACKEND_CPU_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "VernonRuntimeProvider.h"
#include "cpu_workgroup_dispatch.h"
#include "pipeline_bundle.h"
#include "pipeline_metadata.h"
#include "platform/platform_library.h"

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
CpuWorkgroupScheduler &cpuWorkgroupScheduler(VernonRuntimeContext &context);
uint64_t cpuProviderResourceIdentity(const VernonRuntimeContext &context);
VernonStringView cpuProviderLastError(const VernonRuntimeContext &context);
bool prepareCpuComputePipeline(VernonRuntimeContext &context, CpuKernelState kernel, ReflectedEntry reflection,
                               CpuPipelineState &state);

VernonStatus registerStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entry);

bool loadCpuEntry(VernonCpuEntryPoint entry, const char *reflection, size_t reflectionSize, const char *entryName,
                  size_t entryNameSize, CpuKernelState &state, ReflectedEntry &metadata, std::string &error);

bool loadCpuNativeArtifact(VernonRuntimeContext &context, const CpuNativeArtifact &artifact, CpuKernelState &state,
                           ReflectedEntry &metadata, std::string &error);
bool findRegisteredCpuEntry(VernonRuntimeContext &context, const std::string &symbol, VernonCpuEntryPoint &entry,
                            std::string &error);

} // namespace vernon::runtime

#endif
