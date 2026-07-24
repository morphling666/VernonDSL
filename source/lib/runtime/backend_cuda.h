#ifndef VERNON_RUNTIME_BACKEND_CUDA_H
#define VERNON_RUNTIME_BACKEND_CUDA_H

#include "VernonRuntime.h"
#include "backend_cuda_driver.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"

#include <string>
#include <unordered_map>

struct VernonDeviceBuffer;
namespace vernon::runtime {

struct CudaContextState {
    CudaDevice device{};
    CudaContext context{};
};

struct CudaBufferState {
    CudaDevicePointer devicePointer{};
};

struct CudaPipelineState {
    std::unordered_map<std::string, VernonLoadedKernel *> kernels;
};

inline CudaContextState &cudaState(VernonRuntimeContext &context) {
    return runtimeBackendState<CudaContextState>(context);
}

inline const CudaContextState &cudaState(const VernonRuntimeContext &context) {
    return runtimeBackendState<CudaContextState>(context);
}

inline CudaBufferState &cudaBufferState(VernonDeviceBuffer &buffer) {
    return runtimeBackendState<CudaBufferState>(buffer);
}

inline const CudaBufferState &cudaBufferState(const VernonDeviceBuffer &buffer) {
    return runtimeBackendState<CudaBufferState>(buffer);
}

struct CudaKernelState {
    CudaModule module{};
    CudaFunction function{};
};

bool probeCuda(std::string &diagnostic);
bool initializeCudaContext(VernonRuntimeContext &context, uint32_t deviceIndex);
void destroyCudaContext(VernonRuntimeContext &context);
VernonStatus synchronizeCuda(VernonRuntimeContext &context);

bool createCudaBuffer(VernonDeviceBuffer &buffer);
VernonStatus destroyCudaBuffer(VernonDeviceBuffer &buffer);
VernonStatus copyToCudaBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromCudaBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

bool loadCudaKernel(VernonRuntimeContext &context, const void *artifact, size_t artifactSize, const char *reflection,
                    size_t reflectionSize, const char *entry, size_t entrySize, CudaKernelState &state,
                    ReflectedEntry &metadata);
VernonStatus destroyCudaKernel(VernonRuntimeContext &context, CudaKernelState &state);
VernonStatus launchCudaKernel(VernonRuntimeContext &context, const CudaKernelState &state,
                              const ReflectedEntry &metadata, VernonLaunchSize globalSize,
                              const VernonLaunchArgument *arguments, size_t argumentCount);

} // namespace vernon::runtime

#endif
