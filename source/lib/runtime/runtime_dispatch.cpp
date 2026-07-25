#include "runtime_dispatch.h"

#include "backend_cpu.h"
#include "backend_opengl.h"
#include "compute_launch_planner.h"
#include "graphics_opengl_encoder.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "backend_directx12.h"
#include "graphics_directx12_encoder.h"
#endif

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "backend_cuda.h"
#endif

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan.h"
#include "graphics_vulkan_encoder.h"
#endif

#if defined(VERNON_RUNTIME_TESTING)
#include "runtime_test_hooks.h"
#endif

#include <nlohmann/json.hpp>

#include <memory>

namespace vernon::runtime {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    context.error = std::move(error);
    return status;
}

} // namespace

bool isOpenGLBackend(VernonRuntimeBackend backend) {
    return backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES;
}

bool probeBackend(VernonRuntimeBackend backend, std::string &diagnostic) {
    switch (backend) {
    case VERNON_RUNTIME_CPU:
        return true;
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return probeCuda(diagnostic);
#else
        diagnostic = "VernonRuntime was built without CUDA support";
        return false;
#endif
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return probeVulkan(diagnostic);
#else
        diagnostic = "VernonRuntime was built without Vulkan support";
        return false;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return probeDirectX12(diagnostic);
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
    switch (context.backend) {
    case VERNON_RUNTIME_CPU:
        return deviceIndex == 0;
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return initializeCudaContext(context, deviceIndex);
#else
        return false;
#endif
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return initializeVulkanContext(context, deviceIndex);
#else
        return false;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return initializeDirectX12Context(context, deviceIndex);
#else
        return false;
#endif
    default:
        return false;
    }
}

bool initializeOpenGLBackend(VernonRuntimeContext &context, const VernonExternalOpenGLContext &externalContext) {
    return isOpenGLBackend(context.backend) && initializeOpenGLContext(context, externalContext);
}

void destroyBackend(VernonRuntimeContext &context) {
    switch (context.backend) {
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        destroyCudaContext(context);
#endif
        break;
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        destroyVulkanContext(context);
#endif
        break;
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        destroyDirectX12Context(context);
#endif
        break;
    default:
        break;
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
    if (context.backend == VERNON_RUNTIME_VULKAN || context.backend == VERNON_RUNTIME_DIRECTX12) {
        result.supports_compute = 1;
        result.supports_storage_buffers = 1;
        result.supports_graphics = 1;
        result.graphics_draw_abi_version = 2;
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        if (context.backend == VERNON_RUNTIME_DIRECTX12) {
            const DirectX12ContextState &state = directX12State(context);
            result.api_version_major = 12;
            result.api_version_minor = 0;
        }
#endif
        return;
    }
    result.supports_graphics = 1;
    const VernonExternalOpenGLContext &external = openGLState(context).external;
    result.api_version_major = external.api_version_major;
    result.api_version_minor = external.api_version_minor;
    result.supports_compute =
        context.backend == VERNON_RUNTIME_OPENGL_ES
            ? (result.api_version_major > 3 || (result.api_version_major == 3 && result.api_version_minor >= 1))
            : (result.api_version_major > 4 || (result.api_version_major == 4 && result.api_version_minor >= 3));
    result.supports_storage_buffers = result.supports_compute;
    result.graphics_draw_abi_version = 2;
}

bool validateRuntimeRequirements(VernonRuntimeContext &context, const RuntimeRequirements &requirements) {
    if (!requirements.present)
        return true;
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
    if (context.backend == VERNON_RUNTIME_CPU) {
        return validateCpuRuntimeRequirements(requirements.targetTriple, requirements.objectFormat,
                                              requirements.invocationAbiVersion, context.error);
    }
    if (isOpenGLBackend(context.backend)) {
        const VernonExternalOpenGLContext &actual = openGLState(context).external;
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

bool createBackendBuffer(VernonDeviceBuffer &buffer) {
    switch (buffer.context->backend) {
    case VERNON_RUNTIME_CPU:
        return createCpuBuffer(buffer);
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return createCudaBuffer(buffer);
#else
        return false;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        return createOpenGLBuffer(buffer);
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return createVulkanBuffer(buffer);
#else
        return false;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return createDirectX12Buffer(buffer);
#else
        return false;
#endif
    default:
        return false;
    }
}

void importBackendOpenGLBuffer(VernonDeviceBuffer &buffer, uint32_t name) { importOpenGLBuffer(buffer, name); }

VernonStatus destroyBackendBuffer(VernonDeviceBuffer &buffer) {
    switch (buffer.context->backend) {
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return destroyCudaBuffer(buffer);
#else
        break;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        destroyOpenGLBuffer(buffer);
        break;
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        destroyVulkanBuffer(buffer);
#endif
        break;
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        destroyDirectX12Buffer(buffer);
#endif
        break;
    default:
        break;
    }
    return VERNON_STATUS_OK;
}

VernonStatus copyToBackendBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
    switch (buffer.context->backend) {
    case VERNON_RUNTIME_CPU:
        return copyToCpuBuffer(buffer, offset, source, size);
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return copyToCudaBuffer(buffer, offset, source, size);
#else
        break;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        return copyToOpenGLBuffer(buffer, offset, source, size);
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return copyToVulkanBuffer(buffer, offset, source, size);
#else
        break;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return copyToDirectX12Buffer(buffer, offset, source, size);
#else
        break;
#endif
    default:
        break;
    }
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

VernonStatus copyFromBackendBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
    switch (buffer.context->backend) {
    case VERNON_RUNTIME_CPU:
        return copyFromCpuBuffer(buffer, offset, destination, size);
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return copyFromCudaBuffer(buffer, offset, destination, size);
#else
        break;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        return copyFromOpenGLBuffer(buffer, offset, destination, size);
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return copyFromVulkanBuffer(buffer, offset, destination, size);
#else
        break;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return copyFromDirectX12Buffer(buffer, offset, destination, size);
#else
        break;
#endif
    default:
        break;
    }
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

bool createBackendTexture(VernonDeviceTexture &texture) {
    if (isOpenGLBackend(texture.context->backend))
        return createOpenGLTexture(texture);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (texture.context->backend == VERNON_RUNTIME_VULKAN)
        return createVulkanTexture(texture);
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (texture.context->backend == VERNON_RUNTIME_DIRECTX12)
        return createDirectX12Texture(texture);
#endif
    return false;
}

void importBackendOpenGLTexture(VernonDeviceTexture &texture, uint32_t name) { importOpenGLTexture(texture, name); }

void destroyBackendTexture(VernonDeviceTexture &texture) {
    if (isOpenGLBackend(texture.context->backend))
        destroyOpenGLTexture(texture);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    else if (texture.context->backend == VERNON_RUNTIME_VULKAN)
        destroyVulkanTexture(texture);
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    else if (texture.context->backend == VERNON_RUNTIME_DIRECTX12)
        destroyDirectX12Texture(texture);
#endif
}

VernonStatus copyToBackendTexture(VernonDeviceTexture &texture, const void *source, size_t size) {
    if (isOpenGLBackend(texture.context->backend))
        return copyToOpenGLTexture(texture, source, size);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (texture.context->backend == VERNON_RUNTIME_VULKAN)
        return copyToVulkanTexture(texture, source, size);
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (texture.context->backend == VERNON_RUNTIME_DIRECTX12)
        return copyToDirectX12Texture(texture, source, size);
#endif
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

VernonStatus copyFromBackendTexture(const VernonDeviceTexture &texture, void *destination, size_t size) {
    if (isOpenGLBackend(texture.context->backend))
        return copyFromOpenGLTexture(texture, destination, size);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (texture.context->backend == VERNON_RUNTIME_VULKAN)
        return copyFromVulkanTexture(texture, destination, size);
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (texture.context->backend == VERNON_RUNTIME_DIRECTX12)
        return copyFromDirectX12Texture(texture, destination, size);
#endif
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

bool createBackendSampler(VernonDeviceSampler &sampler) {
    if (isOpenGLBackend(sampler.context->backend))
        return createOpenGLSampler(sampler);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (sampler.context->backend == VERNON_RUNTIME_VULKAN)
        return createVulkanSampler(sampler);
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (sampler.context->backend == VERNON_RUNTIME_DIRECTX12)
        return createDirectX12Sampler(sampler);
#endif
    return false;
}

void importBackendOpenGLSampler(VernonDeviceSampler &sampler, uint32_t name) { importOpenGLSampler(sampler, name); }

void destroyBackendSampler(VernonDeviceSampler &sampler) {
    if (isOpenGLBackend(sampler.context->backend))
        destroyOpenGLSampler(sampler);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    else if (sampler.context->backend == VERNON_RUNTIME_VULKAN)
        destroyVulkanSampler(sampler);
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    else if (sampler.context->backend == VERNON_RUNTIME_DIRECTX12)
        destroyDirectX12Sampler(sampler);
#endif
}

VernonLoadedKernel *loadBackendCpuNativeArtifact(VernonRuntimeContext &context, const CpuNativeArtifact &artifact) {
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = &context;
    installRuntimeBackendState(*kernel, new CpuKernelState());
    if (!loadCpuNativeArtifact(artifact, runtimeBackendState<CpuKernelState>(*kernel), kernel->reflection,
                               context.error)) {
        destroyRuntimeBackendState(*kernel);
        return nullptr;
    }
    ++context.liveKernels;
    return kernel.release();
}

VernonLoadedKernel *loadBackendCpuEntry(VernonRuntimeContext &context, VernonCpuEntryPoint entryPoint,
                                        const char *reflection, size_t reflectionSize, const char *entry,
                                        size_t entrySize) {
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = &context;
    installRuntimeBackendState(*kernel, new CpuKernelState());
    if (!loadCpuEntry(entryPoint, reflection, reflectionSize, entry, entrySize,
                      runtimeBackendState<CpuKernelState>(*kernel), kernel->reflection, context.error)) {
        destroyRuntimeBackendState(*kernel);
        return nullptr;
    }
    ++context.liveKernels;
    return kernel.release();
}

VernonStatus registerBackendStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint) {
    return registerStaticCpuEntry(symbol, entryPoint);
}

VernonLoadedKernel *loadBackendArtifact(VernonRuntimeContext &context, const void *artifact, size_t artifactSize,
                                        const char *reflection, size_t reflectionSize, const char *entry,
                                        size_t entrySize) {
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = &context;
    if (isOpenGLBackend(context.backend)) {
        try {
            const nlohmann::json parsed =
                nlohmann::json::parse(reflection, reflection + reflectionSize, nullptr, false);
            if (parsed.is_discarded() ||
                !parseReflection(parsed, std::string(entry, entrySize), kernel->reflection, context.error))
                return nullptr;
            uint32_t binding = 0;
            for (ReflectedArgument &argument : kernel->reflection.arguments) {
                if (argument.kind == "builtin")
                    continue;
                if (argument.binding == UINT32_MAX)
                    argument.binding = binding;
                ++binding;
            }
            makeCurrent(&context);
            const std::string source(static_cast<const char *>(artifact), artifactSize);
            const GlUint shader = compileShader(&context, kComputeShader, source);
            if (!shader)
                return nullptr;
            const GlUint program = linkProgram(&context, {shader});
            if (!program)
                return nullptr;
            auto *state = new OpenGLKernelState();
            state->program = program;
            installRuntimeBackendState(*kernel, state);
        } catch (const std::exception &error) {
            context.error = std::string("failed to load OpenGL artifact: ") + error.what();
            return nullptr;
        }
    } else if (context.backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
        installRuntimeBackendState(*kernel, new CudaKernelState());
        if (!loadCudaKernel(context, artifact, artifactSize, reflection, reflectionSize, entry, entrySize,
                            runtimeBackendState<CudaKernelState>(*kernel), kernel->reflection)) {
            destroyRuntimeBackendState(*kernel);
            return nullptr;
        }
#else
        return nullptr;
#endif
    } else if (context.backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        installRuntimeBackendState(*kernel, new VulkanKernelState());
        if (!loadVulkanKernel(context, artifact, artifactSize, reflection, reflectionSize, entry, entrySize,
                              runtimeBackendState<VulkanKernelState>(*kernel), kernel->reflection)) {
            destroyRuntimeBackendState(*kernel);
            return nullptr;
        }
#else
        return nullptr;
#endif
    } else if (context.backend == VERNON_RUNTIME_DIRECTX12) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        installRuntimeBackendState(*kernel, new DirectX12KernelState());
        if (!loadDirectX12Kernel(context, artifact, artifactSize, reflection, reflectionSize, entry, entrySize,
                                 runtimeBackendState<DirectX12KernelState>(*kernel), kernel->reflection)) {
            destroyRuntimeBackendState(*kernel);
            return nullptr;
        }
#else
        return nullptr;
#endif
    } else {
        return nullptr;
    }
    ++context.liveKernels;
    return kernel.release();
}

VernonStatus unloadBackendKernel(VernonLoadedKernel &kernel) {
    if (isOpenGLBackend(kernel.context->backend))
        destroyOpenGLProgram(*kernel.context, runtimeBackendState<OpenGLKernelState>(kernel).program);
#if defined(VERNON_HAS_CUDA_RUNTIME)
    else if (kernel.context->backend == VERNON_RUNTIME_CUDA) {
        const VernonStatus status = destroyCudaKernel(*kernel.context, runtimeBackendState<CudaKernelState>(kernel));
        if (status != VERNON_STATUS_OK)
            return status;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    else if (kernel.context->backend == VERNON_RUNTIME_VULKAN)
        destroyVulkanKernel(*kernel.context, runtimeBackendState<VulkanKernelState>(kernel));
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    else if (kernel.context->backend == VERNON_RUNTIME_DIRECTX12)
        destroyDirectX12Kernel(*kernel.context, runtimeBackendState<DirectX12KernelState>(kernel));
#endif
    destroyRuntimeBackendState(kernel);
    return VERNON_STATUS_OK;
}

VernonStatus launchBackendKernel(VernonLoadedKernel &kernel, VernonLaunchSize globalSize,
                                 const VernonLaunchArgument *arguments, size_t argumentCount) {
    if (isOpenGLBackend(kernel.context->backend))
        return launchOpenGLKernel(*kernel.context, runtimeBackendState<OpenGLKernelState>(kernel).program,
                                  kernel.reflection, globalSize, arguments, argumentCount);
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (kernel.context->backend == VERNON_RUNTIME_CUDA)
        return launchCudaKernel(*kernel.context, runtimeBackendState<CudaKernelState>(kernel), kernel.reflection,
                                globalSize, arguments, argumentCount);
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (kernel.context->backend == VERNON_RUNTIME_VULKAN)
        return launchVulkanKernel(*kernel.context, runtimeBackendState<VulkanKernelState>(kernel), kernel.reflection,
                                  globalSize, arguments, argumentCount);
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (kernel.context->backend == VERNON_RUNTIME_DIRECTX12)
        return launchDirectX12Kernel(*kernel.context, runtimeBackendState<DirectX12KernelState>(kernel),
                                     kernel.reflection, globalSize, arguments, argumentCount);
#endif
    if (kernel.context->backend == VERNON_RUNTIME_CPU)
        return launchCpuKernel(*kernel.context, runtimeBackendState<CpuKernelState>(kernel), kernel.reflection,
                               globalSize, arguments, argumentCount, kernel.context->error);
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

VernonLoadedKernel *backendPipelineComputeKernel(VernonLoadedPipeline &pipeline) {
    switch (pipeline.context->backend) {
    case VERNON_RUNTIME_CPU:
        return runtimeBackendState<CpuPipelineState>(pipeline).kernel;
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return runtimeBackendState<CudaPipelineState>(pipeline).kernel;
#else
        return nullptr;
#endif
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return runtimeBackendState<VulkanPipelineState>(pipeline).computeKernel;
#else
        return nullptr;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return runtimeBackendState<DirectX12PipelineState>(pipeline).kernel;
#else
        return nullptr;
#endif
    default:
        return nullptr;
    }
}

bool resolveBackendPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
    if (bundle.context->backend == VERNON_RUNTIME_CPU) {
        auto *state = new CpuPipelineState();
        state->kernel = loadBackendCpuNativeArtifact(*bundle.context, *bundle.stages.at(variant.compute).cpuArtifact);
        if (!state->kernel) {
            delete state;
            return false;
        }
        installRuntimeBackendState(pipeline, state);
        return true;
    }
    if (bundle.context->backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
        auto *state = new CudaPipelineState();
        const Stage &stage = bundle.stages.at(variant.compute);
        state->kernel =
            loadBackendArtifact(*bundle.context, stage.source.data(), stage.source.size(), stage.reflection.data(),
                                stage.reflection.size(), stage.entry.data(), stage.entry.size());
        if (!state->kernel) {
            delete state;
            return false;
        }
        installRuntimeBackendState(pipeline, state);
        return true;
#else
        return false;
#endif
    }
    if (bundle.context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        auto *state = new VulkanPipelineState();
        if (!variant.compute.empty()) {
            const Stage &stage = bundle.stages.at(variant.compute);
            state->computeKernel =
                loadBackendArtifact(*bundle.context, stage.binary.data(), stage.binary.size(), stage.reflection.data(),
                                    stage.reflection.size(), stage.entry.data(), stage.entry.size());
            if (!state->computeKernel) {
                delete state;
                return false;
            }
        }
        if (!variant.vertex.empty()) {
            const Stage &vertex = bundle.stages.at(variant.vertex);
            const Stage &fragment = bundle.stages.at(variant.fragment);
            VulkanPushConstantRanges pushConstantRanges;
            if (!planVulkanPushConstantRanges(variant, vulkanState(*bundle.context).maxPushConstantsSize,
                                              pushConstantRanges, bundle.context->error)) {
                delete state;
                return false;
            }
            if (!createVulkanShaderModule(*bundle.context, vertex.binary.data(), vertex.binary.size(), state->vertex,
                                          pushConstantRanges.vertex.size ? pushConstantRanges.vertex.offset : 0) ||
                !createVulkanShaderModule(*bundle.context, fragment.binary.data(), fragment.binary.size(),
                                          state->fragment,
                                          pushConstantRanges.fragment.size ? pushConstantRanges.fragment.offset : 0)) {
                destroyVulkanShaderModule(*bundle.context, state->vertex);
                delete state;
                return false;
            }
            state->vertexEntry = vertex.entry;
            state->fragmentEntry = fragment.entry;
        }
        installRuntimeBackendState(pipeline, state);
        return true;
#else
        return false;
#endif
    }
    if (bundle.context->backend == VERNON_RUNTIME_DIRECTX12) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        auto *pipelineState = new DirectX12PipelineState();
        if (!variant.compute.empty()) {
            auto *state = new DirectX12KernelState();
            const Stage &stage = bundle.stages.at(variant.compute);
            auto kernel = std::make_unique<VernonLoadedKernel>();
            kernel->context = bundle.context;
            installRuntimeBackendState(*kernel, state);
            if (!loadDirectX12Kernel(*bundle.context, stage.binary.data(), stage.binary.size(), stage.reflection.data(),
                                     stage.reflection.size(), stage.entry.data(), stage.entry.size(), *state,
                                     kernel->reflection)) {
                destroyRuntimeBackendState(*kernel);
                delete pipelineState;
                return false;
            }
            ++bundle.context->liveKernels;
            pipelineState->kernel = kernel.release();
        } else {
            pipelineState->vertexDxil = bundle.stages.at(variant.vertex).binary;
            pipelineState->fragmentDxil = bundle.stages.at(variant.fragment).binary;
        }
        installRuntimeBackendState(pipeline, pipelineState);
        return true;
#else
        return false;
#endif
    }
    auto *state = new OpenGLPipelineState();
    if (!createOpenGLPipeline(*bundle.context, variant, bundle.stages, state->computeProgram, state->graphicsProgram,
                              state->vertexArray, state->framebuffer, state->workgroup)) {
        delete state;
        return false;
    }
    installRuntimeBackendState(pipeline, state);
    return true;
}

void destroyBackendPipeline(VernonLoadedPipeline &pipeline) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU || pipeline.context->backend == VERNON_RUNTIME_CUDA) {
        if (VernonLoadedKernel *kernel = backendPipelineComputeKernel(pipeline))
            vernonRuntimeKernelUnload(kernel);
    } else if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
        if (auto *kernel = state.kernel)
            vernonRuntimeKernelUnload(kernel);
        for (auto &[key, graphicsPipeline] : state.graphicsPipelines) {
            (void)key;
            graphicsPipeline->Release();
        }
        state.graphicsPipelines.clear();
#endif
    } else if (pipeline.context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(pipeline);
        if (state.computeKernel)
            vernonRuntimeKernelUnload(state.computeKernel);
        else
            synchronizeVulkan(*pipeline.context);
        destroyVulkanGraphicsCache(pipeline.context, state.graphicsCache);
        destroyVulkanShaderModule(*pipeline.context, state.vertex);
        destroyVulkanShaderModule(*pipeline.context, state.fragment);
#endif
    } else {
        OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
        destroyOpenGLPipeline(*pipeline.context, state.computeProgram, state.graphicsProgram, state.vertexArray,
                              state.framebuffer);
    }
    destroyRuntimeBackendState(pipeline);
}

VernonStatus invokeBackendPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                   const PlannedGraphicsInvocation &plan) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU || pipeline.context->backend == VERNON_RUNTIME_CUDA)
        return VERNON_STATUS_OK;
    if (pipeline.context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(pipeline);
        if (!state.vertex)
            return VERNON_STATUS_OK;
        std::string error;
        const VulkanGraphicsState graphicsState{pipeline.context,   state.vertex,         state.fragment,
                                                &state.vertexEntry, &state.fragmentEntry, &state.graphicsCache};
        const VernonStatus status =
            encodeAndSubmitVulkanGraphics(graphicsState, pipeline.variant, invocation, plan, error);
        return status == VERNON_STATUS_OK ? status : fail(*pipeline.context, error, status);
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    }
    if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
        if (!state.vertexDxil.empty()) {
            std::string error;
            const VernonStatus status =
                encodeAndSubmitDirectX12Graphics(*pipeline.context, state, pipeline.variant, invocation, plan, error);
            return status == VERNON_STATUS_OK ? status : fail(*pipeline.context, error, status);
        }
        return VERNON_STATUS_OK;
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    }
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    if (!state.graphicsProgram)
        return VERNON_STATUS_OK;
    std::string error;
    const OpenGLGraphicsState graphicsState{pipeline.context, state.graphicsProgram, state.vertexArray,
                                            state.framebuffer};
    const VernonStatus status = encodeAndSubmitOpenGLGraphics(graphicsState, pipeline.variant, invocation, plan, error);
    return status == VERNON_STATUS_OK ? status : fail(*pipeline.context, error, status);
}

VernonStatus invokeBackendComputePipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                          const PlannedGraphicsInvocation &plan) {
    if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        ComputeArgumentMap arguments;
        for (const auto &[slot, argument] : plan.arguments)
            arguments.emplace(slot, argument);
        PlannedComputeLaunch launch;
        const ComputePlannerCallbacks callbacks{nullptr,
                                                [](const void *, const VernonDeviceBuffer *buffer) -> const void * {
                                                    return buffer ? buffer->context : nullptr;
                                                }};
        std::string error;
        if (!planComputeLaunch(pipeline.variant, arguments, invocation, pipeline.context, callbacks, launch, error))
            return fail(*pipeline.context, error);
        return vernonRuntimeLaunch(runtimeBackendState<DirectX12PipelineState>(pipeline).kernel, launch.grid,
                                   launch.arguments.data(), launch.arguments.size());
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    }
    if (!isOpenGLBackend(pipeline.context->backend))
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    if (!state.computeProgram)
        return fail(*pipeline.context, "OpenGL compute pipeline program is not loaded");
    std::string error;
    const VernonStatus status = encodeAndSubmitOpenGLCompute(*pipeline.context, state.computeProgram, state.workgroup,
                                                             pipeline.variant, invocation, plan, error);
    return status == VERNON_STATUS_OK ? status : fail(*pipeline.context, error, status);
}

VernonStatus backendComputeToGraphicsBarrier(VernonRuntimeContext &context) {
    return isOpenGLBackend(context.backend) ? openGLComputeToGraphicsBarrier(context)
                                            : VERNON_STATUS_UNSUPPORTED_TARGET;
}

VernonStatus synchronizeBackend(VernonRuntimeContext &context) {
    switch (context.backend) {
    case VERNON_RUNTIME_CPU:
        return VERNON_STATUS_OK;
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return synchronizeCuda(context);
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return synchronizeVulkan(context);
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return synchronizeDirectX12(context);
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        return synchronizeOpenGL(context);
    default:
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }
}

#if defined(VERNON_RUNTIME_TESTING)
VulkanGraphicsCacheStats getVulkanGraphicsCacheStats(const VernonRuntimeContext *context,
                                                     const VernonLoadedPipeline *pipeline) {
    VulkanGraphicsCacheStats result;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!context || !pipeline || pipeline->context != context || context->backend != VERNON_RUNTIME_VULKAN)
        return result;
    result.defaultImplicitSamplerCreations = vulkanState(*context).defaultImplicitSamplerCreations;
    const VulkanGraphicsCache &cache = runtimeBackendState<VulkanPipelineState>(*pipeline).graphicsCache;
    result.descriptorSetLayoutCreations = cache.descriptorSetLayoutCreations;
    result.pipelineLayoutCreations = cache.pipelineLayoutCreations;
    result.graphicsPipelineCreations = cache.graphicsPipelineCreations;
#else
    (void)context;
    (void)pipeline;
#endif
    return result;
}

size_t getDirectX12GraphicsPipelineCreationCount(const VernonLoadedPipeline *pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (pipeline && pipeline->context && pipeline->context->backend == VERNON_RUNTIME_DIRECTX12)
        return runtimeBackendState<DirectX12PipelineState>(*pipeline).graphicsPipelineCreations;
#else
    (void)pipeline;
#endif
    return 0;
}
#endif

} // namespace vernon::runtime
