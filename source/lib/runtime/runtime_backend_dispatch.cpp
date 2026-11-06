#include "runtime_dispatch.h"

#include "hal/native_images.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "hal/cuda/cuda_state.h"
#endif
#include "hal/opengl/opengl_state.h"
#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "hal/vulkan/vulkan_state.h"
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "backend_directx12.h"
#include "hal/d3d12/d3d12_state.h"
#endif
#if defined(VERNON_HAS_METAL_RUNTIME)
#include "backend_metal.h"
#include "hal/metal/metal_state.h"
#endif
#include "backend_cpu.h"
#include "backend_opengl.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "backend_cuda.h"
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan.h"
#endif

#include <new>

namespace vernon::runtime {

namespace {

RuntimeResult<void> runtimeFailure(RuntimeErrorCode code, const char *operation) noexcept {
    return RuntimeResult<void>{err(RuntimeError{code, {operation, 0, 0}})};
}

} // namespace

bool isOpenGLBackend(VernonRuntimeBackend backend) {
    return backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES;
}

RuntimeResult<void> initializeCudaBackend(VernonRuntimeContext &context, std::shared_ptr<hal::Device> device,
                                          const hal::cuda::DeviceState &state) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (context.backend != VERNON_RUNTIME_CUDA || !device)
        return runtimeFailure(RuntimeErrorCode::InvalidArgument, "initialize_cuda_backend");
    auto *backendState = new (std::nothrow) CudaContextState();
    if (!backendState)
        return runtimeFailure(RuntimeErrorCode::InternalFailure, "initialize_cuda_backend");
    backendState->computeCapabilityMajor = state.computeCapabilityMajor;
    backendState->computeCapabilityMinor = state.computeCapabilityMinor;
    backendState->driverVersion = state.driverVersion;
    installRuntimeBackendState(context, backendState);
    std::lock_guard<std::mutex> lock(context.halDeviceMutex);
    context.halDevice = std::move(device);
    return RuntimeResult<void>{ok()};
#else
    (void)device;
    (void)state;
    invocationDiagnostic(context) = "VernonRuntime was built without CUDA support";
    return runtimeFailure(RuntimeErrorCode::Unsupported, "initialize_cuda_backend");
#endif
}

RuntimeResult<void> initializeOpenGLBackend(VernonRuntimeContext &context,
                                            std::shared_ptr<hal::opengl::DeviceState> state) {
    if (!isOpenGLBackend(context.backend) || !state)
        return runtimeFailure(RuntimeErrorCode::InvalidArgument, "initialize_opengl_backend");
    auto *backendState = new (std::nothrow) OpenGLContextState();
    if (!backendState)
        return runtimeFailure(RuntimeErrorCode::InternalFailure, "initialize_opengl_backend");
    backendState->device = std::move(state);
    installRuntimeBackendState(context, backendState);
    return RuntimeResult<void>{ok()};
}

RuntimeResult<void> initializeVulkanBackend(VernonRuntimeContext &context,
                                            std::shared_ptr<hal::vulkan::DeviceState> state) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (context.backend != VERNON_RUNTIME_VULKAN || !state)
        return runtimeFailure(RuntimeErrorCode::InvalidArgument, "initialize_vulkan_backend");
    auto *backendState = new (std::nothrow) VulkanContextState();
    if (!backendState)
        return runtimeFailure(RuntimeErrorCode::InternalFailure, "initialize_vulkan_backend");
    backendState->device = std::move(state);
    installRuntimeBackendState(context, backendState);
    return RuntimeResult<void>{ok()};
#else
    (void)state;
    invocationDiagnostic(context) = "VernonRuntime was built without Vulkan support";
    return runtimeFailure(RuntimeErrorCode::Unsupported, "initialize_vulkan_backend");
#endif
}

RuntimeResult<void> initializeDirectX12Backend(VernonRuntimeContext &context,
                                               std::shared_ptr<hal::d3d12::DeviceState> state) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (context.backend != VERNON_RUNTIME_DIRECTX12 || !state)
        return runtimeFailure(RuntimeErrorCode::InvalidArgument, "initialize_directx12_backend");
    auto *backendState = new (std::nothrow) DirectX12ContextState();
    if (!backendState)
        return runtimeFailure(RuntimeErrorCode::InternalFailure, "initialize_directx12_backend");
    backendState->device = std::move(state);
    installRuntimeBackendState(context, backendState);
    return RuntimeResult<void>{ok()};
#else
    (void)state;
    invocationDiagnostic(context) = "VernonRuntime was built without DirectX 12 support";
    return runtimeFailure(RuntimeErrorCode::Unsupported, "initialize_directx12_backend");
#endif
}

RuntimeResult<void> initializeMetalBackend(VernonRuntimeContext &context,
                                           std::shared_ptr<hal::metal::DeviceState> state) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    if (context.backend != VERNON_RUNTIME_METAL || !state)
        return runtimeFailure(RuntimeErrorCode::InvalidArgument, "initialize_metal_backend");
    auto *backendState = new (std::nothrow) MetalContextState();
    if (!backendState)
        return runtimeFailure(RuntimeErrorCode::InternalFailure, "initialize_metal_backend");
    backendState->device = std::move(state);
    installRuntimeBackendState(context, backendState);
    return RuntimeResult<void>{ok()};
#else
    (void)state;
    invocationDiagnostic(context) = "VernonRuntime was built without Metal support";
    return runtimeFailure(RuntimeErrorCode::Unsupported, "initialize_metal_backend");
#endif
}

RuntimeResult<VernonRuntimeProviderImageDescription>
describeBackendImage(VernonRuntimeContext &, VernonRuntimeProviderResourceReference resource) {
    VernonRuntimeProviderImageDescription description{};
    description.struct_size = sizeof(description);
    if ((resource.identity & hal::kReferenceKindMask) != hal::kNativeViewReference ||
        !hal::describeNativeView(resource, description))
        return RuntimeResult<VernonRuntimeProviderImageDescription>{
            err(RuntimeError{RuntimeErrorCode::InvalidArgument, {"describe_backend_image", 0, 0}})};
    return RuntimeResult<VernonRuntimeProviderImageDescription>{ok(description)};
}

RuntimeResult<void> probeBackend(VernonRuntimeBackend backend, std::string &diagnostic) {
    switch (backend) {
    case VERNON_RUNTIME_CPU:
        return RuntimeResult<void>{ok()};
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
    {
        hal::cuda::DeviceState probe;
        const hal::cuda::Result status = probe.initialize(0);
        if (status == hal::cuda::kSuccess)
            return RuntimeResult<void>{ok()};
        diagnostic = hal::cuda::describeResult(status, "CUDA device creation");
        return runtimeFailure(RuntimeErrorCode::DeviceFailure, "probe_backend");
    }
#else
        diagnostic = "VernonRuntime was built without CUDA support";
        return runtimeFailure(RuntimeErrorCode::Unsupported, "probe_backend");
#endif
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    {
        hal::vulkan::DeviceState probe;
        if (probe.initialize(0, diagnostic))
            return RuntimeResult<void>{ok()};
        return runtimeFailure(RuntimeErrorCode::DeviceFailure, "probe_backend");
    }
#else
        diagnostic = "VernonRuntime was built without Vulkan support";
        return runtimeFailure(RuntimeErrorCode::Unsupported, "probe_backend");
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    {
        hal::d3d12::DeviceState probe;
        if (probe.initialize(0, 0, diagnostic))
            return RuntimeResult<void>{ok()};
        return runtimeFailure(RuntimeErrorCode::DeviceFailure, "probe_backend");
    }
#else
        diagnostic = "VernonRuntime was built without DirectX 12 support";
        return runtimeFailure(RuntimeErrorCode::Unsupported, "probe_backend");
#endif
    case VERNON_RUNTIME_METAL:
#if defined(VERNON_HAS_METAL_RUNTIME)
    {
        hal::metal::DeviceState probe;
        if (probe.initialize(0, 0, diagnostic))
            return RuntimeResult<void>{ok()};
        return runtimeFailure(RuntimeErrorCode::DeviceFailure, "probe_backend");
    }
#else
        diagnostic = "VernonRuntime was built without Metal support";
        return runtimeFailure(RuntimeErrorCode::Unsupported, "probe_backend");
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        diagnostic = "backend requires a host-owned external context";
        return runtimeFailure(RuntimeErrorCode::Unsupported, "probe_backend");
    }
    diagnostic = "VernonRuntime backend is not enabled";
    return runtimeFailure(RuntimeErrorCode::Unsupported, "probe_backend");
}

RuntimeResult<void> initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex) {
    if (context.backend == VERNON_RUNTIME_CPU)
        return initializeCpuContext(context, deviceIndex)
                   ? RuntimeResult<void>{ok()}
                   : runtimeFailure(RuntimeErrorCode::InternalFailure, "initialize_cpu_backend");
    invocationDiagnostic(context) = "GPU Runtime contexts are created over a HAL device";
    return runtimeFailure(RuntimeErrorCode::InvalidArgument, "initialize_backend");
}

void destroyBackend(VernonRuntimeContext &context) {
    // The HAL device waits for its submissions on the backend state, which is destroyed below.
    {
        std::lock_guard<std::mutex> lock(context.halDeviceMutex);
        context.halDevice.reset();
    }
    destroyRuntimeBackendState(context);
}

void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &result) {
    result.available = 1;
    if (context.backend == VERNON_RUNTIME_CPU || context.backend == VERNON_RUNTIME_CUDA) {
        result.supports_compute = 1;
        result.supports_storage_buffers = 1;
        return;
    }
    if (context.backend == VERNON_RUNTIME_VULKAN || context.backend == VERNON_RUNTIME_DIRECTX12 ||
        context.backend == VERNON_RUNTIME_METAL) {
        result.supports_compute = 1;
        result.supports_storage_buffers = 1;
        result.supports_graphics = 1;
        result.graphics_draw_abi_version = VERNON_PROGRAM_VERSION;
        if (context.backend == VERNON_RUNTIME_DIRECTX12)
            result.api_version_major = 12;
        return;
    }
    result.supports_graphics = 1;
    const VernonOpenGLContextCallbacks &external = openGLState(context).device->callbacks;
    result.api_version_major = external.api_version_major;
    result.api_version_minor = external.api_version_minor;
    result.supports_compute =
        context.backend == VERNON_RUNTIME_OPENGL_ES
            ? (result.api_version_major > 3 || (result.api_version_major == 3 && result.api_version_minor >= 1))
            : (result.api_version_major > 4 || (result.api_version_major == 4 && result.api_version_minor >= 3));
    result.supports_storage_buffers = result.supports_compute;
    result.graphics_draw_abi_version = VERNON_PROGRAM_VERSION;
}

} // namespace vernon::runtime
