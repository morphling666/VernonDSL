#ifndef VERNON_RUNTIME_BACKEND_CUDA_H
#define VERNON_RUNTIME_BACKEND_CUDA_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "rhi/cuda_backend.h"
#include "rhi_provider_pipeline.h"
#include "runtime_state.h"

struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

using CudaDevicePointer = rhi::cuda::DevicePointer;
using CudaResult = rhi::cuda::Result;
constexpr CudaResult kCudaSuccess = rhi::cuda::kSuccess;

struct CudaContextState {
    VernonRuntimeRhiAdapter *adapter{};
    uint32_t computeCapabilityMajor{};
    uint32_t computeCapabilityMinor{};
    uint32_t driverVersion{};
};

struct CudaPipelineState {
    RhiComputePipeline rhiCompute;
};

inline CudaContextState &cudaState(VernonRuntimeContext &context) {
    return runtimeBackendState<CudaContextState>(context);
}

inline const CudaContextState &cudaState(const VernonRuntimeContext &context) {
    return runtimeBackendState<CudaContextState>(context);
}

} // namespace vernon::runtime

#endif
