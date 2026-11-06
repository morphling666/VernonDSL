#ifndef VERNON_RUNTIME_BACKEND_CUDA_H
#define VERNON_RUNTIME_BACKEND_CUDA_H

#include "VernonRuntime.h"
#include "runtime_state.h"

#include <cstdint>

namespace vernon::runtime {

// What a CUDA context knows of its HAL device, for validating Program requirements.
struct CudaContextState {
    uint32_t computeCapabilityMajor{};
    uint32_t computeCapabilityMinor{};
    uint32_t driverVersion{};
};

inline CudaContextState &cudaState(VernonRuntimeContext &context) {
    return runtimeBackendState<CudaContextState>(context);
}

inline const CudaContextState &cudaState(const VernonRuntimeContext &context) {
    return runtimeBackendState<CudaContextState>(context);
}

} // namespace vernon::runtime

#endif
