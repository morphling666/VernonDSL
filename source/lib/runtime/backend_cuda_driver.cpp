#include "backend_cuda_driver.h"

namespace vernon::runtime {

bool CudaDriver::load() {
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
    memoryAllocate = reinterpret_cast<MemoryAllocate>(library.symbol("cuMemAlloc_v2"));
    if (!memoryAllocate)
        memoryAllocate = reinterpret_cast<MemoryAllocate>(library.symbol("cuMemAlloc"));
    memoryFree = reinterpret_cast<MemoryFree>(library.symbol("cuMemFree_v2"));
    if (!memoryFree)
        memoryFree = reinterpret_cast<MemoryFree>(library.symbol("cuMemFree"));
    copyHostToDevice = reinterpret_cast<CopyHostToDevice>(library.symbol("cuMemcpyHtoD_v2"));
    if (!copyHostToDevice)
        copyHostToDevice = reinterpret_cast<CopyHostToDevice>(library.symbol("cuMemcpyHtoD"));
    copyDeviceToHost = reinterpret_cast<CopyDeviceToHost>(library.symbol("cuMemcpyDtoH_v2"));
    if (!copyDeviceToHost)
        copyDeviceToHost = reinterpret_cast<CopyDeviceToHost>(library.symbol("cuMemcpyDtoH"));
    if (!memoryAllocate || !memoryFree || !copyHostToDevice || !copyDeviceToHost) {
        error = "CUDA Driver is missing versioned memory operation symbols";
        return false;
    }
    VERNON_LOAD_CUDA(moduleLoadData, "cuModuleLoadDataEx");
    VERNON_LOAD_CUDA(moduleGetFunction, "cuModuleGetFunction");
    VERNON_LOAD_CUDA(moduleUnload, "cuModuleUnload");
    VERNON_LOAD_CUDA(launchKernel, "cuLaunchKernel");
#undef VERNON_LOAD_CUDA
    available = true;
    return true;
}

CudaDriver &cudaDriver() {
    static CudaDriver driver;
    return driver;
}

} // namespace vernon::runtime
