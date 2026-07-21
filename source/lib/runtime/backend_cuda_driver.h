#ifndef VERNON_RUNTIME_BACKEND_CUDA_DRIVER_H
#define VERNON_RUNTIME_BACKEND_CUDA_DRIVER_H

#include "platform_library.h"

#include <cstddef>
#include <mutex>
#include <string>

namespace vernon::runtime {

using CudaDevice = int;
using CudaDevicePointer = unsigned long long;
using CudaContext = struct CudaContextOpaque *;
using CudaModule = struct CudaModuleOpaque *;
using CudaFunction = struct CudaFunctionOpaque *;
using CudaStream = struct CudaStreamOpaque *;
using CudaResult = int;

constexpr CudaResult kCudaSuccess = 0;

struct CudaDriver {
  using Init = CudaResult (*)(unsigned);
  using DeviceGet = CudaResult (*)(CudaDevice *, int);
  using PrimaryContextRetain = CudaResult (*)(CudaContext *, CudaDevice);
  using PrimaryContextRelease = CudaResult (*)(CudaDevice);
  using ContextSetCurrent = CudaResult (*)(CudaContext);
  using ContextSynchronize = CudaResult (*)();
  using ErrorName = CudaResult (*)(CudaResult, const char **);
  using ErrorString = CudaResult (*)(CudaResult, const char **);
  using MemoryAllocate = CudaResult (*)(CudaDevicePointer *, size_t);
  using MemoryFree = CudaResult (*)(CudaDevicePointer);
  using CopyHostToDevice = CudaResult (*)(CudaDevicePointer, const void *,
                                          size_t);
  using CopyDeviceToHost = CudaResult (*)(void *, CudaDevicePointer, size_t);
  using ModuleLoadData = CudaResult (*)(CudaModule *, const void *, unsigned,
                                        int *, void **);
  using ModuleGetFunction = CudaResult (*)(CudaFunction *, CudaModule,
                                           const char *);
  using ModuleUnload = CudaResult (*)(CudaModule);
  using LaunchKernel = CudaResult (*)(CudaFunction, unsigned, unsigned,
                                      unsigned, unsigned, unsigned, unsigned,
                                      unsigned, CudaStream, void **, void **);

  bool load();

  PlatformLibrary library;
  std::mutex mutex;
  std::string error;
  bool attempted{};
  bool available{};
  Init init{};
  DeviceGet deviceGet{};
  PrimaryContextRetain primaryContextRetain{};
  PrimaryContextRelease primaryContextRelease{};
  ContextSetCurrent contextSetCurrent{};
  ContextSynchronize contextSynchronize{};
  ErrorName errorName{};
  ErrorString errorString{};
  MemoryAllocate memoryAllocate{};
  MemoryFree memoryFree{};
  CopyHostToDevice copyHostToDevice{};
  CopyDeviceToHost copyDeviceToHost{};
  ModuleLoadData moduleLoadData{};
  ModuleGetFunction moduleGetFunction{};
  ModuleUnload moduleUnload{};
  LaunchKernel launchKernel{};
};

CudaDriver &cudaDriver();

} // namespace vernon::runtime

#endif
