#ifndef VERNON_RHI_CUDA_DRIVER_H
#define VERNON_RHI_CUDA_DRIVER_H

#include "../platform/platform_library.h"
#include "VernonRHI.h"

#include <cstddef>
#include <mutex>
#include <string>

namespace vernon::rhi::cuda {

using Device = int;
using DevicePointer = unsigned long long;
using Context = struct ContextOpaque *;
using Module = struct ModuleOpaque *;
using Function = struct FunctionOpaque *;
using Stream = struct StreamOpaque *;
using Event = struct EventOpaque *;
using Result = int;

constexpr Result kSuccess = 0;

struct VERNON_RHI_CAPI Driver {
    using Init = Result (*)(unsigned);
    using DeviceGet = Result (*)(Device *, int);
    using DeviceGetAttribute = Result (*)(int *, int, Device);
    using DriverGetVersion = Result (*)(int *);
    using PrimaryContextRetain = Result (*)(Context *, Device);
    using PrimaryContextRelease = Result (*)(Device);
    using ContextSetCurrent = Result (*)(Context);
    using ContextSynchronize = Result (*)();
    using ErrorName = Result (*)(Result, const char **);
    using ErrorString = Result (*)(Result, const char **);
    using MemoryAllocate = Result (*)(DevicePointer *, size_t);
    using MemoryFree = Result (*)(DevicePointer);
    using HostAllocate = Result (*)(void **, size_t, unsigned);
    using HostFree = Result (*)(void *);
    using CopyHostToDevice = Result (*)(DevicePointer, const void *, size_t);
    using CopyDeviceToHost = Result (*)(void *, DevicePointer, size_t);
    using CopyHostToDeviceAsync = Result (*)(DevicePointer, const void *, size_t, Stream);
    using CopyDeviceToHostAsync = Result (*)(void *, DevicePointer, size_t, Stream);
    using ModuleLoadData = Result (*)(Module *, const void *, unsigned, int *, void **);
    using ModuleGetFunction = Result (*)(Function *, Module, const char *);
    using ModuleUnload = Result (*)(Module);
    using StreamCreate = Result (*)(Stream *, unsigned);
    using StreamDestroy = Result (*)(Stream);
    using StreamSynchronize = Result (*)(Stream);
    using EventCreate = Result (*)(Event *, unsigned);
    using EventDestroy = Result (*)(Event);
    using EventRecord = Result (*)(Event, Stream);
    using EventSynchronize = Result (*)(Event);
    using LaunchKernel = Result (*)(Function, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                    Stream, void **, void **);

    bool load();

    platform::PlatformLibrary library;
    std::mutex mutex;
    std::string error;
    bool attempted{};
    bool available{};
    Init init{};
    DeviceGet deviceGet{};
    DeviceGetAttribute deviceGetAttribute{};
    DriverGetVersion driverGetVersion{};
    PrimaryContextRetain primaryContextRetain{};
    PrimaryContextRelease primaryContextRelease{};
    ContextSetCurrent contextSetCurrent{};
    ContextSynchronize contextSynchronize{};
    ErrorName errorName{};
    ErrorString errorString{};
    MemoryAllocate memoryAllocate{};
    MemoryFree memoryFree{};
    HostAllocate hostAllocate{};
    HostFree hostFree{};
    CopyHostToDevice copyHostToDevice{};
    CopyDeviceToHost copyDeviceToHost{};
    CopyHostToDeviceAsync copyHostToDeviceAsync{};
    CopyDeviceToHostAsync copyDeviceToHostAsync{};
    ModuleLoadData moduleLoadData{};
    ModuleGetFunction moduleGetFunction{};
    ModuleUnload moduleUnload{};
    StreamCreate streamCreate{};
    StreamDestroy streamDestroy{};
    StreamSynchronize streamSynchronize{};
    EventCreate eventCreate{};
    EventDestroy eventDestroy{};
    EventRecord eventRecord{};
    EventSynchronize eventSynchronize{};
    LaunchKernel launchKernel{};
};

VERNON_RHI_CAPI Driver &driver();

} // namespace vernon::rhi::cuda

#endif
