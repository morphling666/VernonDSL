#ifndef VERNON_RUNTIME_BACKEND_CUDA_H
#define VERNON_RUNTIME_BACKEND_CUDA_H

#include "../rhi/cuda_backend.h"
#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"

#include <string>
#include <vector>
struct VernonDeviceBuffer;
struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

using CudaDevicePointer = rhi::cuda::DevicePointer;
using CudaResult = rhi::cuda::Result;
constexpr CudaResult kCudaSuccess = rhi::cuda::kSuccess;

struct CudaContextState {
    rhi::cuda::DeviceState device;
    VernonRuntimeRhiAdapter *adapter{};
    uint32_t computeCapabilityMajor{};
    uint32_t computeCapabilityMinor{};
    uint32_t driverVersion{};
};

struct CudaBufferState {
    CudaDevicePointer devicePointer{};
};

struct CudaPipelineState {
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeCoreBindings *bindings{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> layout;
    std::vector<VernonRuntimeProviderBindingValue> values;
    uint32_t workgroup[3]{1, 1, 1};
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

bool probeCuda(std::string &diagnostic);
bool initializeCudaContext(VernonRuntimeContext &context, uint32_t deviceIndex);
void destroyCudaContext(VernonRuntimeContext &context);
VernonStatus synchronizeCuda(VernonRuntimeContext &context);

bool createCudaBuffer(VernonDeviceBuffer &buffer);
VernonStatus destroyCudaBuffer(VernonDeviceBuffer &buffer);
VernonStatus copyToCudaBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromCudaBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

} // namespace vernon::runtime

#endif
