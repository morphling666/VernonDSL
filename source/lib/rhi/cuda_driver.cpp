#include "cuda_driver.h"

namespace vernon::rhi::cuda {

bool Driver::load() {
    std::lock_guard<std::mutex> guard(mutex);
    if (attempted)
        return available;
    attempted = true;
#if defined(_WIN32)
    constexpr const char *libraryName = "nvcuda.dll";
#else
    constexpr const char *libraryName = "libcuda.so.1";
#endif
    if (!library.open(libraryName, error))
        return false;

#define VERNON_LOAD_CUDA(member, symbolName)                                                                           \
    member = reinterpret_cast<decltype(member)>(library.symbol(symbolName));                                           \
    if (!member) {                                                                                                     \
        error = std::string("CUDA Driver is missing symbol ") + symbolName;                                            \
        return false;                                                                                                  \
    }
#define VERNON_LOAD_CUDA_VERSIONED(member, symbolName)                                                                 \
    member = reinterpret_cast<decltype(member)>(library.symbol(symbolName "_v2"));                                     \
    if (!member)                                                                                                       \
        member = reinterpret_cast<decltype(member)>(library.symbol(symbolName));                                       \
    if (!member) {                                                                                                     \
        error = std::string("CUDA Driver is missing symbol ") + symbolName;                                            \
        return false;                                                                                                  \
    }

    VERNON_LOAD_CUDA(init, "cuInit");
    VERNON_LOAD_CUDA(deviceGet, "cuDeviceGet");
    VERNON_LOAD_CUDA(deviceGetAttribute, "cuDeviceGetAttribute");
    VERNON_LOAD_CUDA(driverGetVersion, "cuDriverGetVersion");
    VERNON_LOAD_CUDA(primaryContextRetain, "cuDevicePrimaryCtxRetain");
    VERNON_LOAD_CUDA(primaryContextRelease, "cuDevicePrimaryCtxRelease");
    VERNON_LOAD_CUDA(contextSetCurrent, "cuCtxSetCurrent");
    VERNON_LOAD_CUDA(contextSynchronize, "cuCtxSynchronize");
    VERNON_LOAD_CUDA(errorName, "cuGetErrorName");
    VERNON_LOAD_CUDA(errorString, "cuGetErrorString");
    VERNON_LOAD_CUDA_VERSIONED(memoryAllocate, "cuMemAlloc");
    VERNON_LOAD_CUDA_VERSIONED(memoryFree, "cuMemFree");
    VERNON_LOAD_CUDA(hostAllocate, "cuMemHostAlloc");
    VERNON_LOAD_CUDA(hostFree, "cuMemFreeHost");
    VERNON_LOAD_CUDA_VERSIONED(copyHostToDevice, "cuMemcpyHtoD");
    VERNON_LOAD_CUDA_VERSIONED(copyDeviceToHost, "cuMemcpyDtoH");
    VERNON_LOAD_CUDA_VERSIONED(copyHostToDeviceAsync, "cuMemcpyHtoDAsync");
    VERNON_LOAD_CUDA_VERSIONED(copyDeviceToHostAsync, "cuMemcpyDtoHAsync");
    VERNON_LOAD_CUDA(moduleLoadData, "cuModuleLoadDataEx");
    VERNON_LOAD_CUDA(moduleGetFunction, "cuModuleGetFunction");
    VERNON_LOAD_CUDA(moduleUnload, "cuModuleUnload");
    VERNON_LOAD_CUDA(streamCreate, "cuStreamCreate");
    VERNON_LOAD_CUDA_VERSIONED(streamDestroy, "cuStreamDestroy");
    VERNON_LOAD_CUDA(streamSynchronize, "cuStreamSynchronize");
    VERNON_LOAD_CUDA(eventCreate, "cuEventCreate");
    VERNON_LOAD_CUDA_VERSIONED(eventDestroy, "cuEventDestroy");
    VERNON_LOAD_CUDA(eventRecord, "cuEventRecord");
    VERNON_LOAD_CUDA(eventSynchronize, "cuEventSynchronize");
    VERNON_LOAD_CUDA(launchKernel, "cuLaunchKernel");

#undef VERNON_LOAD_CUDA_VERSIONED
#undef VERNON_LOAD_CUDA
    available = true;
    return true;
}

Driver &driver() {
    static Driver instance;
    return instance;
}

} // namespace vernon::rhi::cuda
