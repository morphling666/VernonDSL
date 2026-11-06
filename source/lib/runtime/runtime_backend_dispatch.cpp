#include "runtime_dispatch.h"

#include "hal/native_images.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "hal/cuda/cuda_state.h"
#endif
#include "hal/opengl/opengl_state.h"
#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "hal/vulkan/vulkan_state.h"
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
    case VERNON_RUNTIME_METAL:
        diagnostic = "DirectX 12 and Metal have no HAL device yet";
        return runtimeFailure(RuntimeErrorCode::Unsupported, "probe_backend");
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
    if (context.backend == VERNON_RUNTIME_VULKAN) {
        result.supports_compute = 1;
        result.supports_storage_buffers = 1;
        result.supports_graphics = 1;
        result.graphics_draw_abi_version = VERNON_PROGRAM_VERSION;
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

RuntimeResult<void> validateRuntimeRequirements(VernonRuntimeContext &context,
                                                const RuntimeRequirements &requirements) {
    VernonRuntimeCapabilities capabilities{};
    fillBackendCapabilities(context, capabilities);
    for (const std::string &feature : requirements.features) {
        const bool supported =
            feature == "compute" || feature == "atomics" || feature == "barriers" || feature == "workgroup_storage"
                ? capabilities.supports_compute
            : feature == "tensor_views" ? capabilities.supports_storage_buffers
            : feature == "instancing" || feature == "samplers" || feature == "textures" ? capabilities.supports_graphics
                                                                                        : false;
        if (!supported) {
            invocationDiagnostic(context) = "pipeline requires unsupported runtime feature '" + feature + "'";
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_runtime_requirements");
        }
    }
    if (context.backend == VERNON_RUNTIME_CPU) {
        auto validated = validateCpuRuntimeRequirements(requirements.targetTriple, requirements.objectFormat);
        if (validated.isErr()) {
            invocationDiagnostic(context) = renderStageArtifactError(
                validated.error(), requirements.targetTriple + " / " + requirements.objectFormat);
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_cpu_runtime_requirements");
        }
        return RuntimeResult<void>{ok()};
    }
    if (isOpenGLBackend(context.backend)) {
        const VernonOpenGLContextCallbacks &actual = openGLState(context).device->callbacks;
        const RuntimeVersion actualApi{actual.api_version_major, actual.api_version_minor};
        const bool apiSatisfied = runtimeVersionAtLeast(actualApi, requirements.apiVersion);
        const uint32_t actualGlsl = glslVersionForApi(actualApi);
        if (!apiSatisfied || actualGlsl < requirements.glslVersion) {
            invocationDiagnostic(context) =
                "pipeline requires API " + std::to_string(requirements.apiVersion.major) + "." +
                std::to_string(requirements.apiVersion.minor) + " / GLSL " + std::to_string(requirements.glslVersion) +
                ", context provides " + std::to_string(actual.api_version_major) + "." +
                std::to_string(actual.api_version_minor) + " / GLSL " + std::to_string(actualGlsl);
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_opengl_runtime_requirements");
        }
        return RuntimeResult<void>{ok()};
    }
    if (context.backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        const hal::vulkan::DeviceState &actual = *vulkanState(context).device;
        const uint32_t actualMajor = VK_VERSION_MAJOR(actual.apiVersion);
        const uint32_t actualMinor = VK_VERSION_MINOR(actual.apiVersion);
        if (!runtimeVersionAtLeast({actualMajor, actualMinor}, requirements.apiVersion)) {
            invocationDiagnostic(context) = "pipeline requires Vulkan " +
                                            std::to_string(requirements.apiVersion.major) + "." +
                                            std::to_string(requirements.apiVersion.minor) + ", device provides " +
                                            std::to_string(actualMajor) + "." + std::to_string(actualMinor);
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_vulkan_runtime_requirements");
        }
        RuntimeVersion supportedSpirv{1, 0};
        if (actualMajor > 1 || actualMinor >= 3)
            supportedSpirv = {1, 6};
        else if (actualMinor >= 2)
            supportedSpirv = {1, 5};
        else if (actualMinor >= 1)
            supportedSpirv = {1, 3};
        if (requirements.shaderVersion.major > supportedSpirv.major ||
            (requirements.shaderVersion.major == supportedSpirv.major &&
             requirements.shaderVersion.minor > supportedSpirv.minor)) {
            invocationDiagnostic(context) =
                "pipeline requires SPIR-V " + std::to_string(requirements.shaderVersion.major) + "." +
                std::to_string(requirements.shaderVersion.minor) + ", device API supports " +
                std::to_string(supportedSpirv.major) + "." + std::to_string(supportedSpirv.minor);
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_vulkan_runtime_requirements");
        }
        uint64_t invocations = 1;
        for (size_t index = 0; index < 3; ++index) {
            if (requirements.computeWorkgroupSize[index] > actual.maxComputeWorkGroupSize[index]) {
                invocationDiagnostic(context) =
                    "pipeline compute workgroup dimension " + std::to_string(index) + " requires " +
                    std::to_string(requirements.computeWorkgroupSize[index]) + ", device provides " +
                    std::to_string(actual.maxComputeWorkGroupSize[index]);
                return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_vulkan_runtime_requirements");
            }
            invocations *= requirements.computeWorkgroupSize[index];
        }
        if (invocations > actual.maxComputeWorkGroupInvocations) {
            invocationDiagnostic(context) = "pipeline compute workgroup requires " + std::to_string(invocations) +
                                            " invocations, device provides " +
                                            std::to_string(actual.maxComputeWorkGroupInvocations);
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_vulkan_runtime_requirements");
        }
        return RuntimeResult<void>{ok()};
#endif
    }
    if (context.backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
        const CudaContextState &actual = cudaState(context);
        if (!runtimeVersionAtLeast({actual.computeCapabilityMajor, actual.computeCapabilityMinor},
                                   requirements.minimumComputeCapability)) {
            invocationDiagnostic(context) = "pipeline requires CUDA compute capability " +
                                            std::to_string(requirements.minimumComputeCapability.major) + "." +
                                            std::to_string(requirements.minimumComputeCapability.minor) +
                                            ", device provides " + std::to_string(actual.computeCapabilityMajor) + "." +
                                            std::to_string(actual.computeCapabilityMinor);
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_cuda_runtime_requirements");
        }
        const uint32_t actualAddressSize = static_cast<uint32_t>(sizeof(void *) * 8);
        if (requirements.addressSize != actualAddressSize) {
            invocationDiagnostic(context) = "pipeline requires PTX address size " +
                                            std::to_string(requirements.addressSize) + ", runtime provides " +
                                            std::to_string(actualAddressSize);
            return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_cuda_runtime_requirements");
        }
        return RuntimeResult<void>{ok()};
#endif
    }
    invocationDiagnostic(context) = "runtime requirements cannot be validated by selected backend";
    return runtimeFailure(RuntimeErrorCode::Unsupported, "validate_runtime_requirements");
}

} // namespace vernon::runtime
