#include "runtime_dispatch.h"

#include "../rhi/rhi_internal.h"
#include "backend_cpu.h"
#include "backend_opengl.h"
#include "rhi_adapter/adapter_internal.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "backend_cuda.h"
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan.h"
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "backend_directx12.h"
#endif

namespace vernon::runtime {

bool isOpenGLBackend(VernonRuntimeBackend backend) {
    return backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES;
}

bool initializeBackendForRhiDevice(VernonRuntimeContext &context, VernonRhiDevice device) {
    VernonRhiBackend backend;
    switch (context.backend) {
    case VERNON_RUNTIME_CUDA:
        backend = VERNON_RHI_BACKEND_CUDA;
        break;
    case VERNON_RUNTIME_VULKAN:
        backend = VERNON_RHI_BACKEND_VULKAN;
        break;
    case VERNON_RUNTIME_DIRECTX12:
        backend = VERNON_RHI_BACKEND_DIRECTX12;
        break;
    case VERNON_RUNTIME_OPENGL:
        backend = VERNON_RHI_BACKEND_OPENGL;
        break;
    case VERNON_RUNTIME_OPENGL_ES:
        backend = VERNON_RHI_BACKEND_OPENGL_ES;
        break;
    case VERNON_RUNTIME_CPU:
        context.error = "CPU Runtime does not use a Vernon RHI device";
        return false;
    }
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, backend);
    if (!adapter) {
        context.error = "cannot create Runtime adapter for Vernon RHI device";
        return false;
    }
    switch (context.backend) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    case VERNON_RUNTIME_CUDA: {
        auto *state = new CudaContextState();
        state->adapter = adapter;
        const auto *deviceState =
            static_cast<const rhi::cuda::DeviceState *>(rhi::deviceState(device, VERNON_RHI_BACKEND_CUDA));
        state->computeCapabilityMajor = deviceState->computeCapabilityMajor;
        state->computeCapabilityMinor = deviceState->computeCapabilityMinor;
        state->driverVersion = deviceState->driverVersion;
        installRuntimeBackendState(context, state);
        break;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    case VERNON_RUNTIME_VULKAN: {
        auto *state = new VulkanContextState();
        state->adapter = adapter;
        const auto *deviceState =
            static_cast<const rhi::vulkan::DeviceState *>(rhi::deviceState(device, VERNON_RHI_BACKEND_VULKAN));
        state->apiVersion = deviceState->apiVersion;
        state->maxComputeWorkGroupInvocations = deviceState->maxComputeWorkGroupInvocations;
        std::copy(std::begin(deviceState->maxComputeWorkGroupSize), std::end(deviceState->maxComputeWorkGroupSize),
                  std::begin(state->maxComputeWorkGroupSize));
        state->dynamicRendering = deviceState->dynamicRendering;
        installRuntimeBackendState(context, state);
        break;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    case VERNON_RUNTIME_DIRECTX12: {
        auto *state = new DirectX12ContextState();
        state->adapter = adapter;
        const auto *deviceState =
            static_cast<const rhi::directx12::DeviceState *>(rhi::deviceState(device, VERNON_RHI_BACKEND_DIRECTX12));
        state->featureLevel = deviceState->featureLevel;
        state->shaderModel = deviceState->shaderModel;
        state->rootSignatureVersion = deviceState->rootSignatureVersion;
        state->resourceBindingTier = deviceState->resourceBindingTier;
        state->maxComputeInvocations = deviceState->maxComputeInvocations;
        std::copy(std::begin(deviceState->maxComputeWorkGroupSize), std::end(deviceState->maxComputeWorkGroupSize),
                  std::begin(state->maxComputeWorkGroupSize));
        installRuntimeBackendState(context, state);
        break;
    }
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES: {
        auto *state = new OpenGLContextState();
        state->adapter = adapter;
        const auto rhiBackend =
            context.backend == VERNON_RUNTIME_OPENGL ? VERNON_RHI_BACKEND_OPENGL : VERNON_RHI_BACKEND_OPENGL_ES;
        const auto *deviceState = static_cast<const rhi::opengl::DeviceState *>(rhi::deviceState(device, rhiBackend));
        state->device.callbacks = deviceState->callbacks;
        installRuntimeBackendState(context, state);
        break;
    }
    default:
        vernonRuntimeRhiAdapterDestroy(adapter);
        context.error = "Runtime backend is unavailable for Vernon RHI device";
        return false;
    }
    context.borrowedRhiDevice = true;
    context.rhiDevice = device;
    return true;
}

VernonRuntimeRhiAdapter *borrowedRhiAdapter(VernonRuntimeContext &context) {
    if (!context.borrowedRhiDevice)
        return nullptr;
    VernonRuntimeRhiAdapter *adapter = nullptr;
    switch (context.backend) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    case VERNON_RUNTIME_CUDA:
        adapter = cudaState(context).adapter;
        break;
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    case VERNON_RUNTIME_VULKAN:
        adapter = vulkanState(context).adapter;
        break;
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    case VERNON_RUNTIME_DIRECTX12:
        adapter = directX12State(context).adapter;
        break;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        adapter = openGLState(context).adapter;
        break;
    default:
        return nullptr;
    }
    return adapter;
}

VernonStatus referenceBackendRhiBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset,
                                       uint64_t size, VernonRuntimeProviderResourceReference &output) {
    VernonRuntimeRhiAdapter *adapter = borrowedRhiAdapter(context);
    if (!adapter)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffer, offset, size, &output);
}

VernonStatus referenceBackendRhiImage(VernonRuntimeContext &context, VernonRhiImage image,
                                      VernonRuntimeProviderResourceReference &output) {
    VernonRuntimeRhiAdapter *adapter = borrowedRhiAdapter(context);
    return adapter ? vernonRuntimeRhiAdapterReferenceImage(adapter, image, &output) : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus referenceBackendRhiSampler(VernonRuntimeContext &context, VernonRhiSampler sampler,
                                        VernonRuntimeProviderResourceReference &output) {
    VernonRuntimeRhiAdapter *adapter = borrowedRhiAdapter(context);
    return adapter ? vernonRuntimeRhiAdapterReferenceSampler(adapter, sampler, &output)
                   : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus referenceBackendCommandEncoder(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                                            VernonRuntimeProviderObject &output) {
    VernonRuntimeRhiAdapter *adapter = borrowedRhiAdapter(context);
    return adapter ? vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, encoder, &output)
                   : VERNON_STATUS_INVALID_ARGUMENT;
}

bool probeBackend(VernonRuntimeBackend backend, std::string &diagnostic) {
    auto probeRhi = [&](VernonRhiBackend rhiBackend) {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = rhiBackend;
        VernonRhiDevice device = vernonRhiCreateDevice(&descriptor);
        if (device.index == static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX)) {
            diagnostic = "no usable Vernon RHI device was found";
            return false;
        }
        vernonRhiDestroyDevice(device);
        return true;
    };
    switch (backend) {
    case VERNON_RUNTIME_CPU:
        return true;
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return probeRhi(VERNON_RHI_BACKEND_CUDA);
#else
        diagnostic = "VernonRuntime was built without CUDA support";
        return false;
#endif
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return probeRhi(VERNON_RHI_BACKEND_VULKAN);
#else
        diagnostic = "VernonRuntime was built without Vulkan support";
        return false;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return probeRhi(VERNON_RHI_BACKEND_DIRECTX12);
#else
        diagnostic = "VernonRuntime was built without DirectX 12 support";
        return false;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        diagnostic = "backend requires a host-owned external context";
        return false;
    default:
        diagnostic = "VernonRuntime backend is not enabled";
        return false;
    }
}

bool initializeBackend(VernonRuntimeContext &context, uint32_t deviceIndex) {
    if (context.backend == VERNON_RUNTIME_CPU)
        return initializeCpuContext(context, deviceIndex);
    context.error = "GPU Runtime contexts require a Vernon RHI device";
    return false;
}

void destroyBackend(VernonRuntimeContext &context) {
    if (VernonRuntimeRhiAdapter *adapter = borrowedRhiAdapter(context))
        vernonRuntimeRhiAdapterDestroy(adapter);
    destroyRuntimeBackendState(context);
}

void fillBackendCapabilities(const VernonRuntimeContext &context, VernonRuntimeCapabilities &result) {
    result.available = 1;
    if (context.backend == VERNON_RUNTIME_CPU || context.backend == VERNON_RUNTIME_CUDA) {
        result.supports_compute = 1;
        result.supports_storage_buffers = 1;
        return;
    }
    if (context.backend == VERNON_RUNTIME_VULKAN || context.backend == VERNON_RUNTIME_DIRECTX12) {
        result.supports_compute = 1;
        result.supports_storage_buffers = 1;
        result.supports_graphics = 1;
        result.graphics_draw_abi_version = VERNON_PIPELINE_VERSION;
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        if (context.backend == VERNON_RUNTIME_DIRECTX12) {
            result.api_version_major = 12;
            result.api_version_minor = 0;
        }
#endif
        return;
    }
    result.supports_graphics = 1;
    const VernonOpenGLContextCallbacks &external = openGLState(context).device.callbacks;
    result.api_version_major = external.api_version_major;
    result.api_version_minor = external.api_version_minor;
    result.supports_compute =
        context.backend == VERNON_RUNTIME_OPENGL_ES
            ? (result.api_version_major > 3 || (result.api_version_major == 3 && result.api_version_minor >= 1))
            : (result.api_version_major > 4 || (result.api_version_major == 4 && result.api_version_minor >= 3));
    result.supports_storage_buffers = result.supports_compute;
    result.graphics_draw_abi_version = VERNON_PIPELINE_VERSION;
}

bool validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements) {
    VernonRuntimeCapabilities capabilities{};
    fillBackendCapabilities(context, capabilities);
    for (const std::string &feature : requirements.features) {
        const bool supported = feature == "compute"        ? capabilities.supports_compute
                               : feature == "tensor_views" ? capabilities.supports_storage_buffers
                               : feature == "instancing" || feature == "samplers" || feature == "textures"
                                   ? capabilities.supports_graphics
                                   : false;
        if (!supported) {
            context.error = "pipeline requires unsupported runtime feature '" + feature + "'";
            return false;
        }
    }
    if (context.backend == VERNON_RUNTIME_CPU)
        return validateCpuRuntimeRequirements(requirements.targetTriple, requirements.objectFormat, context.error);
    if (isOpenGLBackend(context.backend)) {
        const VernonOpenGLContextCallbacks &actual = openGLState(context).device.callbacks;
        const RuntimeVersion actualApi{actual.api_version_major, actual.api_version_minor};
        const bool apiSatisfied = runtimeVersionAtLeast(actualApi, requirements.apiVersion);
        const uint32_t actualGlsl = glslVersionForApi(actualApi);
        if (!apiSatisfied || actualGlsl < requirements.glslVersion) {
            context.error = "pipeline requires API " + std::to_string(requirements.apiVersion.major) + "." +
                            std::to_string(requirements.apiVersion.minor) + " / GLSL " +
                            std::to_string(requirements.glslVersion) + ", context provides " +
                            std::to_string(actual.api_version_major) + "." + std::to_string(actual.api_version_minor) +
                            " / GLSL " + std::to_string(actualGlsl);
            return false;
        }
        return true;
    }
    if (context.backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        const VulkanContextState &actual = vulkanState(context);
        const uint32_t actualMajor = VK_VERSION_MAJOR(actual.apiVersion);
        const uint32_t actualMinor = VK_VERSION_MINOR(actual.apiVersion);
        if (!runtimeVersionAtLeast({actualMajor, actualMinor}, requirements.apiVersion)) {
            context.error = "pipeline requires Vulkan " + std::to_string(requirements.apiVersion.major) + "." +
                            std::to_string(requirements.apiVersion.minor) + ", device provides " +
                            std::to_string(actualMajor) + "." + std::to_string(actualMinor);
            return false;
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
            context.error = "pipeline requires SPIR-V " + std::to_string(requirements.shaderVersion.major) + "." +
                            std::to_string(requirements.shaderVersion.minor) + ", device API supports " +
                            std::to_string(supportedSpirv.major) + "." + std::to_string(supportedSpirv.minor);
            return false;
        }
        uint64_t invocations = 1;
        for (size_t index = 0; index < 3; ++index) {
            if (requirements.computeWorkgroupSize[index] > actual.maxComputeWorkGroupSize[index]) {
                context.error = "pipeline compute workgroup dimension " + std::to_string(index) + " requires " +
                                std::to_string(requirements.computeWorkgroupSize[index]) + ", device provides " +
                                std::to_string(actual.maxComputeWorkGroupSize[index]);
                return false;
            }
            invocations *= requirements.computeWorkgroupSize[index];
        }
        if (invocations > actual.maxComputeWorkGroupInvocations) {
            context.error = "pipeline compute workgroup requires " + std::to_string(invocations) +
                            " invocations, device provides " + std::to_string(actual.maxComputeWorkGroupInvocations);
            return false;
        }
        return true;
#endif
    }
    if (context.backend == VERNON_RUNTIME_DIRECTX12) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        const DirectX12ContextState &actual = directX12State(context);
        const uint32_t featureLevel = static_cast<uint32_t>(actual.featureLevel);
        const uint32_t requiredFeatureLevel =
            requirements.minimumFeatureLevel.major * 0x1000 + requirements.minimumFeatureLevel.minor * 0x100;
        if (featureLevel < requiredFeatureLevel) {
            context.error =
                "pipeline requires D3D feature level " + std::to_string(requirements.minimumFeatureLevel.major) + "." +
                std::to_string(requirements.minimumFeatureLevel.minor) + ", device provides " +
                std::to_string((featureLevel >> 12) & 0xf) + "." + std::to_string((featureLevel >> 8) & 0xf);
            return false;
        }
        const uint32_t actualShaderModel = static_cast<uint32_t>(actual.shaderModel);
        const uint32_t requiredShaderModel = requirements.shaderVersion.major * 0x10 + requirements.shaderVersion.minor;
        if (actualShaderModel < requiredShaderModel) {
            context.error = "pipeline requires Shader Model " + std::to_string(requirements.shaderVersion.major) + "." +
                            std::to_string(requirements.shaderVersion.minor) + ", device provides " +
                            std::to_string((actualShaderModel >> 4) & 0xf) + "." +
                            std::to_string(actualShaderModel & 0xf);
            return false;
        }
        const uint32_t requiredRootSignature =
            requirements.rootSignatureVersion.major == 1 ? 1 + requirements.rootSignatureVersion.minor : UINT32_MAX;
        if (static_cast<uint32_t>(actual.rootSignatureVersion) < requiredRootSignature) {
            context.error = "pipeline requires a newer D3D12 root-signature version";
            return false;
        }
        uint64_t invocations = 1;
        for (size_t index = 0; index < 3; ++index) {
            if (requirements.computeWorkgroupSize[index] > actual.maxComputeWorkGroupSize[index]) {
                context.error = "pipeline compute workgroup dimension exceeds D3D12 device limits";
                return false;
            }
            invocations *= requirements.computeWorkgroupSize[index];
        }
        if (invocations > actual.maxComputeInvocations) {
            context.error = "pipeline compute workgroup exceeds D3D12 thread-group limit";
            return false;
        }
        return true;
#endif
    }
    if (context.backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
        const CudaContextState &actual = cudaState(context);
        if (!runtimeVersionAtLeast({actual.computeCapabilityMajor, actual.computeCapabilityMinor},
                                   requirements.minimumComputeCapability)) {
            context.error = "pipeline requires CUDA compute capability " +
                            std::to_string(requirements.minimumComputeCapability.major) + "." +
                            std::to_string(requirements.minimumComputeCapability.minor) + ", device provides " +
                            std::to_string(actual.computeCapabilityMajor) + "." +
                            std::to_string(actual.computeCapabilityMinor);
            return false;
        }
        const uint32_t actualAddressSize = static_cast<uint32_t>(sizeof(void *) * 8);
        if (requirements.addressSize != actualAddressSize) {
            context.error = "pipeline requires PTX address size " + std::to_string(requirements.addressSize) +
                            ", runtime provides " + std::to_string(actualAddressSize);
            return false;
        }
        return true;
#endif
    }
    context.error = "runtime requirements cannot be validated by selected backend";
    return false;
}

} // namespace vernon::runtime
