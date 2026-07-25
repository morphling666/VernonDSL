#include "backend_cuda.h"

#include "rhi_adapter/adapter_internal.h"
#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vernon::runtime {
namespace {

#if defined(VERNON_HAS_CUDA_RUNTIME)
VernonStatus cudaFail(VernonRuntimeContext &context, CudaResult result, const char *operation) {
    if (result == kCudaSuccess)
        return VERNON_STATUS_OK;
    context.error = rhi::cuda::describeResult(result, operation);
    return VERNON_STATUS_INTERNAL_ERROR;
}

#endif

} // namespace

bool probeCuda(std::string &diagnostic) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    rhi::cuda::Driver &api = rhi::cuda::driver();
    if (!api.load()) {
        diagnostic = api.error;
        return false;
    }
    rhi::cuda::Device device{};
    CudaResult status = api.init(0);
    if (status == kCudaSuccess)
        status = api.deviceGet(&device, 0);
    if (status == kCudaSuccess)
        return true;
    diagnostic = "CUDA Driver loaded but no usable device was found";
    return false;
#else
    diagnostic = "VernonRuntime was built without CUDA support";
    return false;
#endif
}

bool initializeCudaContext(VernonRuntimeContext &context, uint32_t deviceIndex) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    auto state = std::make_unique<CudaContextState>();
    const CudaResult status = state->device.initialize(deviceIndex);
    if (status != kCudaSuccess) {
        context.error = rhi::cuda::describeResult(status, "CUDA RHI device initialization");
        return false;
    }
    state->computeCapabilityMajor = state->device.computeCapabilityMajor;
    state->computeCapabilityMinor = state->device.computeCapabilityMinor;
    state->driverVersion = state->device.driverVersion;
    state->adapter = createBorrowedCudaRhiAdapter(state->device);
    if (!state->adapter) {
        context.error = "failed to create CUDA Runtime RHI adapter";
        return false;
    }
    installRuntimeBackendState(context, state.release());
    return true;
#else
    (void)context;
    (void)deviceIndex;
    return false;
#endif
}

void destroyCudaContext(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaContextState &state = cudaState(context);
    vernonRuntimeRhiAdapterDestroy(state.adapter);
    state.adapter = nullptr;
    state.device.shutdown();
#else
    (void)context;
#endif
}

VernonStatus synchronizeCuda(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    return cudaFail(context, cudaState(context).device.synchronize(), "cuStreamSynchronize");
#else
    (void)context;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

bool createCudaBuffer(VernonDeviceBuffer &buffer) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    auto *state = new CudaBufferState();
    if (cudaFail(*buffer.context, cudaState(*buffer.context).device.allocate(state->devicePointer, buffer.size),
                 "cuMemAlloc") != VERNON_STATUS_OK) {
        delete state;
        return false;
    }
    installRuntimeBackendState(buffer, state);
    return true;
#else
    (void)buffer;
    return false;
#endif
}

VernonStatus destroyCudaBuffer(VernonDeviceBuffer &buffer) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaBufferState &state = cudaBufferState(buffer);
    if (!state.devicePointer)
        return VERNON_STATUS_OK;
    const VernonStatus status =
        cudaFail(*buffer.context, cudaState(*buffer.context).device.free(state.devicePointer), "cuMemFree");
    if (status == VERNON_STATUS_OK)
        state.devicePointer = 0;
    return status;
#else
    (void)buffer;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus copyToCudaBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (!source || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer upload";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    const CudaBufferState &state = cudaBufferState(buffer);
    return cudaFail(*buffer.context,
                    cudaState(*buffer.context).device.upload(state.devicePointer + offset, source, size),
                    "cuMemcpyHtoDAsync");
#else
    (void)buffer;
    (void)offset;
    (void)source;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus copyFromCudaBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (!destination || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer readback";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    const CudaBufferState &state = cudaBufferState(buffer);
    return cudaFail(*buffer.context,
                    cudaState(*buffer.context).device.download(destination, state.devicePointer + offset, size),
                    "cuMemcpyDtoHAsync");
#else
    (void)buffer;
    (void)offset;
    (void)destination;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
