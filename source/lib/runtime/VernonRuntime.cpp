#include "VernonRuntime.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#if defined(VERNON_RUNTIME_TESTING)
#include "runtime/runtime_test_hooks.h"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace vernon::runtime;

} // namespace

namespace {

GraphicsResourceSnapshot plannerBufferSnapshot(const void *, const VernonDeviceBuffer *buffer) {
    GraphicsResourceSnapshot snapshot;
    if (buffer) {
        snapshot.context = buffer->context;
        snapshot.bufferSize = buffer->size;
    }
    return snapshot;
}

const void *plannerBufferContext(const void *, const VernonDeviceBuffer *buffer) {
    return buffer ? buffer->context : nullptr;
}

GraphicsResourceSnapshot plannerTextureSnapshot(const void *, const VernonDeviceTexture *texture) {
    GraphicsResourceSnapshot snapshot;
    if (texture) {
        snapshot.context = texture->context;
        snapshot.textureDimension = texture->dimension;
        snapshot.textureFormat = texture->format;
        snapshot.textureWidth = texture->width;
        snapshot.textureHeight = texture->height;
        snapshot.textureDepth = texture->depth;
    }
    return snapshot;
}

const void *plannerSamplerContext(const void *, const VernonDeviceSampler *sampler) {
    return sampler ? sampler->context : nullptr;
}

VernonStatus fail(VernonRuntimeContext *context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    if (context)
        context->error = std::move(error);
    return status;
}

} // namespace

extern "C" {

VernonRuntimeCapabilities vernonRuntimeGetCapabilities(VernonRuntimeBackend backend) {
    static thread_local std::string diagnostic;
    diagnostic.clear();
    VernonRuntimeCapabilities result{};
    result.available = probeBackend(backend, diagnostic);
    if (backend == VERNON_RUNTIME_CPU && result.available) {
        result.supports_compute = 1;
        result.supports_storage_buffers = 1;
    } else if ((backend == VERNON_RUNTIME_CUDA || backend == VERNON_RUNTIME_VULKAN) && result.available) {
        result.supports_compute = result.available;
        result.supports_storage_buffers = result.available;
    } else if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
        result.supports_graphics = 1;
    }
    result.diagnostic = {diagnostic.data(), diagnostic.size()};
    return result;
}

VernonRuntimeContext *vernonRuntimeCreate(VernonRuntimeBackend backend, uint32_t deviceIndex) {
    VernonRuntimeCreateOptions options{};
    options.struct_size = sizeof(options);
    options.device_index = deviceIndex;
    return vernonRuntimeCreateWithOptions(backend, &options);
}

VernonRuntimeContext *vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                                                     const VernonRuntimeCreateOptions *options) {
    if (!options || options->struct_size < sizeof(VernonRuntimeCreateOptions))
        return nullptr;
    auto context = std::make_unique<VernonRuntimeContext>();
    context->backend = backend;
    if (!initializeBackend(*context, options->device_index))
        return nullptr;
    return context.release();
}

VernonRuntimeContext *vernonRuntimeCreateExternalOpenGL(const VernonExternalOpenGLContext *externalContext) {
    return vernonRuntimeCreateExternalOpenGLForBackend(VERNON_RUNTIME_OPENGL, externalContext);
}

VernonRuntimeContext *vernonRuntimeCreateExternalOpenGLForBackend(VernonRuntimeBackend backend,
                                                                  const VernonExternalOpenGLContext *externalContext) {
    if (!externalContext || externalContext->struct_size < sizeof(VernonExternalOpenGLContext) ||
        !externalContext->make_current || !externalContext->get_proc_address ||
        (backend != VERNON_RUNTIME_OPENGL && backend != VERNON_RUNTIME_OPENGL_ES) ||
        externalContext->api_version_major < 2)
        return nullptr;
    auto context = std::make_unique<VernonRuntimeContext>();
    context->backend = backend;
    if (!initializeOpenGLBackend(*context, *externalContext))
        return nullptr;
    return context.release();
}

VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context) {
    if (!context)
        return VERNON_STATUS_OK;
    if (context->liveBuffers || context->liveKernels || context->liveTextures || context->liveSamplers ||
        context->liveBundles || context->livePipelines)
        return fail(context, "runtime context still owns live handles");
    destroyBackend(*context);
    delete context;
    return VERNON_STATUS_OK;
}

VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context) {
    return context ? VernonStringView{context->error.data(), context->error.size()} : VernonStringView{nullptr, 0};
}

VernonRuntimeCapabilities vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context) {
    VernonRuntimeCapabilities result{};
    if (!context)
        return result;
    fillBackendCapabilities(*context, result);
    result.diagnostic = {context->error.data(), context->error.size()};
    return result;
}

VernonDeviceBuffer *vernonRuntimeBufferAllocate(VernonRuntimeContext *context, size_t size, size_t alignment) {
    if (!context ||
        (context->backend != VERNON_RUNTIME_CPU && context->backend != VERNON_RUNTIME_CUDA &&
         context->backend != VERNON_RUNTIME_VULKAN && context->backend != VERNON_RUNTIME_OPENGL &&
         context->backend != VERNON_RUNTIME_OPENGL_ES) ||
        !size || !alignment || (alignment & (alignment - 1))) {
        fail(context, "invalid compute buffer allocation");
        return nullptr;
    }
    auto result = std::make_unique<VernonDeviceBuffer>();
    result->context = context;
    result->size = size;
    result->alignment = alignment;
    if (!createBackendBuffer(*result))
        return nullptr;
    ++context->liveBuffers;
    return result.release();
}

VernonDeviceBuffer *vernonRuntimeImportOpenGLBuffer(VernonRuntimeContext *context, uint32_t buffer, size_t size,
                                                    size_t alignment) {
    if (!context || !buffer || !size || !alignment || (alignment & (alignment - 1)) ||
        (context->backend != VERNON_RUNTIME_OPENGL && context->backend != VERNON_RUNTIME_OPENGL_ES))
        return nullptr;
    auto result = std::make_unique<VernonDeviceBuffer>();
    result->context = context;
    result->size = size;
    result->alignment = alignment;
    importBackendOpenGLBuffer(*result, buffer);
    ++context->liveBuffers;
    return result.release();
}

VernonStatus vernonRuntimeBufferFree(VernonDeviceBuffer *buffer) {
    if (!buffer)
        return VERNON_STATUS_OK;
    const VernonStatus status = destroyBackendBuffer(*buffer);
    if (status != VERNON_STATUS_OK)
        return status;
    destroyRuntimeBackendState(*buffer);
    --buffer->context->liveBuffers;
    delete buffer;
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeCopyFromHost(VernonDeviceBuffer *buffer, size_t offset, const void *source, size_t size) {
    if (buffer)
        return copyToBackendBuffer(*buffer, offset, source, size);
    return fail(buffer ? buffer->context : nullptr, "invalid compute buffer upload");
}

VernonStatus vernonRuntimeCopyToHost(const VernonDeviceBuffer *buffer, size_t offset, void *destination, size_t size) {
    if (buffer)
        return copyFromBackendBuffer(*buffer, offset, destination, size);
    return fail(buffer ? buffer->context : nullptr, "invalid compute buffer readback");
}

VernonDeviceTexture *vernonRuntimeTextureCreate(VernonRuntimeContext *context,
                                                const VernonTextureDescriptor *descriptor) {
    if (!context || !descriptor || descriptor->struct_size < sizeof(VernonTextureDescriptor) || !descriptor->width ||
        !descriptor->height || !descriptor->depth || !descriptor->mip_levels)
        return nullptr;
    if (isOpenGLBackend(context->backend)) {
        auto texture = std::make_unique<VernonDeviceTexture>();
        texture->context = context;
        texture->width = descriptor->width;
        texture->height = descriptor->height;
        texture->depth = descriptor->depth;
        texture->mipLevels = descriptor->mip_levels;
        texture->dimension = descriptor->dimension;
        texture->format = descriptor->format;
        if (!createBackendTexture(*texture))
            return nullptr;
        ++context->liveTextures;
        return texture.release();
    }
    if (context->backend != VERNON_RUNTIME_VULKAN) {
        fail(context, "owned sampled textures require the Vulkan backend", VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    if (descriptor->dimension > VERNON_TEXTURE_CUBE ||
        (descriptor->dimension != VERNON_TEXTURE_3D && descriptor->depth != 1) ||
        (descriptor->dimension == VERNON_TEXTURE_CUBE && descriptor->width != descriptor->height)) {
        fail(context, "sampled texture descriptor dimensions are invalid");
        return nullptr;
    }
    auto texture = std::make_unique<VernonDeviceTexture>();
    texture->context = context;
    texture->width = descriptor->width;
    texture->height = descriptor->height;
    texture->depth = descriptor->depth;
    texture->mipLevels = descriptor->mip_levels;
    texture->dimension = descriptor->dimension;
    texture->format = descriptor->format;
    if (!createBackendTexture(*texture))
        return nullptr;
    ++context->liveTextures;
    return texture.release();
}

VernonDeviceTexture *vernonRuntimeTextureCreate2D(VernonRuntimeContext *context, uint32_t width, uint32_t height,
                                                  VernonTextureFormat format) {
    const VernonTextureDescriptor descriptor{
        sizeof(VernonTextureDescriptor), VERNON_TEXTURE_2D, format, width, height, 1, 1, {0, 0, 0, 0}};
    return vernonRuntimeTextureCreate(context, &descriptor);
}

VernonDeviceTexture *vernonRuntimeImportOpenGLTexture(VernonRuntimeContext *context, uint32_t texture,
                                                      const VernonTextureDescriptor *descriptor) {
    if (!context || (context->backend != VERNON_RUNTIME_OPENGL && context->backend != VERNON_RUNTIME_OPENGL_ES) ||
        !texture || !descriptor || descriptor->struct_size < sizeof(VernonTextureDescriptor) || !descriptor->width ||
        !descriptor->height || !descriptor->depth || !descriptor->mip_levels ||
        descriptor->format > VERNON_TEXTURE_R11G11B10_FLOAT || descriptor->dimension > VERNON_TEXTURE_CUBE ||
        (descriptor->dimension != VERNON_TEXTURE_3D && descriptor->depth != 1) ||
        (descriptor->dimension == VERNON_TEXTURE_CUBE && descriptor->width != descriptor->height))
        return nullptr;
    auto result = std::make_unique<VernonDeviceTexture>();
    result->context = context;
    result->width = descriptor->width;
    result->height = descriptor->height;
    result->depth = descriptor->depth;
    result->mipLevels = descriptor->mip_levels;
    result->dimension = descriptor->dimension;
    result->format = descriptor->format;
    importBackendOpenGLTexture(*result, texture);
    ++context->liveTextures;
    return result.release();
}

VernonDeviceTexture *vernonRuntimeImportOpenGLTexture2D(VernonRuntimeContext *context, uint32_t texture, uint32_t width,
                                                        uint32_t height, VernonTextureFormat format) {
    const VernonTextureDescriptor descriptor{
        sizeof(VernonTextureDescriptor), VERNON_TEXTURE_2D, format, width, height, 1, 1, {0, 0, 0, 0}};
    return vernonRuntimeImportOpenGLTexture(context, texture, &descriptor);
}

VernonStatus vernonRuntimeTextureFree(VernonDeviceTexture *texture) {
    if (!texture)
        return VERNON_STATUS_OK;
    destroyBackendTexture(*texture);
    destroyRuntimeBackendState(*texture);
    --texture->context->liveTextures;
    delete texture;
    return VERNON_STATUS_OK;
}

VernonDeviceSampler *vernonRuntimeSamplerCreate(VernonRuntimeContext *context,
                                                const VernonSamplerDescriptor *descriptor) {
    if (!context || !descriptor || descriptor->struct_size < sizeof(VernonSamplerDescriptor))
        return nullptr;
    if (isOpenGLBackend(context->backend)) {
        auto sampler = std::make_unique<VernonDeviceSampler>();
        sampler->context = context;
        sampler->descriptor = *descriptor;
        if (!createBackendSampler(*sampler))
            return nullptr;
        ++context->liveSamplers;
        return sampler.release();
    }
    if (context->backend != VERNON_RUNTIME_VULKAN) {
        fail(context, "owned sampler creation requires Vulkan; import an OpenGL sampler",
             VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    auto sampler = std::make_unique<VernonDeviceSampler>();
    sampler->context = context;
    sampler->descriptor = *descriptor;
    if (!createBackendSampler(*sampler))
        return nullptr;
    ++context->liveSamplers;
    return sampler.release();
}

VernonDeviceSampler *vernonRuntimeImportOpenGLSampler(VernonRuntimeContext *context, uint32_t sampler) {
    if (!context || (context->backend != VERNON_RUNTIME_OPENGL && context->backend != VERNON_RUNTIME_OPENGL_ES) ||
        !sampler)
        return nullptr;
    auto result = std::make_unique<VernonDeviceSampler>();
    result->context = context;
    importBackendOpenGLSampler(*result, sampler);
    ++context->liveSamplers;
    return result.release();
}

VernonStatus vernonRuntimeSamplerFree(VernonDeviceSampler *sampler) {
    if (!sampler)
        return VERNON_STATUS_OK;
    destroyBackendSampler(*sampler);
    destroyRuntimeBackendState(*sampler);
    --sampler->context->liveSamplers;
    delete sampler;
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeTextureCopyFromHost(VernonDeviceTexture *texture, const void *source, size_t size) {
    if (texture)
        return copyToBackendTexture(*texture, source, size);
    return fail(texture ? texture->context : nullptr, "texture upload backend is unsupported",
                VERNON_STATUS_UNSUPPORTED_TARGET);
}

VernonStatus vernonRuntimeTextureCopyToHost(const VernonDeviceTexture *texture, void *destination, size_t size) {
    if (texture)
        return copyFromBackendTexture(*texture, destination, size);
    return fail(texture ? texture->context : nullptr, "texture readback backend is unsupported",
                VERNON_STATUS_UNSUPPORTED_TARGET);
}

VernonLoadedKernel *vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context, VernonCpuEntryPoint entryPoint,
                                              const char *reflection, size_t reflectionSize, const char *entry,
                                              size_t entrySize) {
    if (!context || context->backend != VERNON_RUNTIME_CPU || !entryPoint || !reflection || !reflectionSize || !entry ||
        !entrySize)
        return nullptr;
    return loadBackendCpuEntry(*context, entryPoint, reflection, reflectionSize, entry, entrySize);
}

VernonStatus vernonRuntimeRegisterStaticCpuEntry(VernonStringView symbol, VernonCpuEntryPoint entryPoint) {
    return registerBackendStaticCpuEntry(symbol, entryPoint);
}

VernonLoadedKernel *vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact, size_t artifactSize,
                                              const char *reflection, size_t reflectionSize, const char *entry,
                                              size_t entrySize) {
    if (!context || !artifact || !artifactSize || !reflection || !reflectionSize || !entry || !entrySize)
        return nullptr;
    if (context->backend == VERNON_RUNTIME_CPU) {
        fail(context,
             "CPU AOT artifacts must be loaded from a compute bundle so the "
             "native library target and content hash can be validated",
             VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    return loadBackendArtifact(*context, artifact, artifactSize, reflection, reflectionSize, entry, entrySize);
}

VernonLoadedKernel *vernonRuntimeLoadComputeBundle(VernonRuntimeContext *context, const char *directory) {
    if (!context || context->backend != VERNON_RUNTIME_CPU || !directory)
        return nullptr;
    try {
        const std::filesystem::path root = std::filesystem::u8path(directory);
        CpuNativeArtifact artifact;
        if (!parseCpuComputeBundle(root, artifact, context->error))
            return nullptr;
        return loadBackendCpuNativeArtifact(*context, artifact);
    } catch (const std::exception &error) {
        fail(context, std::string("failed to load CPU AOT bundle: ") + error.what(), VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    }
}

VernonStatus vernonRuntimeKernelUnload(VernonLoadedKernel *kernel) {
    if (!kernel)
        return VERNON_STATUS_OK;
    const VernonStatus status = unloadBackendKernel(*kernel);
    if (status != VERNON_STATUS_OK)
        return status;
    --kernel->context->liveKernels;
    delete kernel;
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeLaunch(VernonLoadedKernel *kernel, VernonLaunchSize globalSize,
                                 const VernonLaunchArgument *arguments, size_t argumentCount) {
    if (!kernel || !globalSize.x || !globalSize.y || !globalSize.z)
        return fail(kernel ? kernel->context : nullptr, "compute launch grid dimensions must be positive");
    const size_t expected = static_cast<size_t>(
        std::count_if(kernel->reflection.arguments.begin(), kernel->reflection.arguments.end(),
                      [](const ReflectedArgument &argument) { return argument.kind != "builtin"; }));
    if (argumentCount != expected || (expected && !arguments))
        return fail(kernel->context, "compute launch argument count does not match reflection");

    size_t validated = 0;
    for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
        if (reflected.kind == "builtin")
            continue;
        const VernonLaunchArgument &argument = arguments[validated++];
        if (reflected.kind == "tensor") {
            if (argument.kind != VERNON_LAUNCH_TENSOR || !argument.buffer ||
                argument.buffer->context != kernel->context ||
                (reflected.tensorBytes && argument.buffer->size < reflected.tensorBytes) ||
                argument.buffer->alignment < reflected.alignment)
                return fail(kernel->context, "compute Tensor argument does not match reflection");
        } else if (argument.kind != VERNON_LAUNCH_SCALAR || !argument.scalar_data ||
                   argument.scalar_size != reflected.cpuSize) {
            return fail(kernel->context, "compute scalar argument does not match reflection");
        }
    }

    const VernonStatus status = launchBackendKernel(*kernel, globalSize, arguments, argumentCount);
    return status == VERNON_STATUS_UNSUPPORTED_TARGET ? fail(kernel->context, "compute backend is unsupported", status)
                                                      : status;
}

VernonStatus vernonRuntimePipelineBundleInspectTarget(const void *bundleData, size_t bundleSize,
                                                      VernonRuntimeBackend *target) {
    if (!bundleData || !bundleSize || !target)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        const nlohmann::json root = nlohmann::json::parse(
            static_cast<const char *>(bundleData), static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
        if (root.is_discarded() || !root.is_object())
            return VERNON_STATUS_PARSE_ERROR;
        const bool schema2 = root.value("schema_version", 0) == 2 && root.value("type", "") == "pipeline";
        std::string manifestError;
        if (!schema2 || root.value("invocation_abi_version", 0) != VERNON_PIPELINE_INVOCATION_ABI_VERSION ||
            !validateManifestHash(root, true, manifestError))
            return VERNON_STATUS_PARSE_ERROR;
        const std::string name = root.value("target", "");
        if (name == "cpu")
            *target = VERNON_RUNTIME_CPU;
        else if (name == "cuda")
            *target = VERNON_RUNTIME_CUDA;
        else if (name == "vulkan")
            *target = VERNON_RUNTIME_VULKAN;
        else if (name == "opengl")
            *target = VERNON_RUNTIME_OPENGL;
        else if (name == "opengles")
            *target = VERNON_RUNTIME_OPENGL_ES;
        else
            return VERNON_STATUS_UNSUPPORTED_TARGET;
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_PARSE_ERROR;
    }
}

VernonPipelineBundle *vernonRuntimeLoadPipelineBundle(VernonRuntimeContext *context, const void *bundleData,
                                                      size_t bundleSize) {
    return vernonRuntimeLoadPipelineBundleWithOptions(context, bundleData, bundleSize, nullptr);
}

VernonPipelineBundle *vernonRuntimeLoadPipelineBundleWithOptions(VernonRuntimeContext *context, const void *bundleData,
                                                                 size_t bundleSize,
                                                                 const VernonPipelineBundleLoadOptions *options) {
    if (!context ||
        (context->backend != VERNON_RUNTIME_CPU && context->backend != VERNON_RUNTIME_OPENGL &&
         context->backend != VERNON_RUNTIME_OPENGL_ES && context->backend != VERNON_RUNTIME_VULKAN &&
         context->backend != VERNON_RUNTIME_CUDA) ||
        !bundleData || !bundleSize || (options && options->struct_size < sizeof(VernonPipelineBundleLoadOptions)))
        return nullptr;
    try {
        std::optional<std::filesystem::path> bundleDirectory;
        if (options && options->bundle_directory && options->bundle_directory[0] != '\0')
            bundleDirectory = std::filesystem::u8path(options->bundle_directory);
        const nlohmann::json root = nlohmann::json::parse(static_cast<const char *>(bundleData),
                                                          static_cast<const char *>(bundleData) + bundleSize);
        const char *expectedTarget = context->backend == VERNON_RUNTIME_CPU    ? "cpu"
                                     : context->backend == VERNON_RUNTIME_CUDA ? "cuda"
                                     : context->backend == VERNON_RUNTIME_VULKAN
                                         ? "vulkan"
                                         : (context->backend == VERNON_RUNTIME_OPENGL_ES ? "opengles" : "opengl");
        const bool schema2 =
            root.is_object() && root.value("schema_version", 0) == 2 && root.value("type", "") == "pipeline";
        if (!schema2 || root.value("invocation_abi_version", 0) != VERNON_PIPELINE_INVOCATION_ABI_VERSION ||
            root.value("target", "") != expectedTarget || !root.contains("stage_artifacts") ||
            !root["stage_artifacts"].is_object() || !root.contains("variants") || !root["variants"].is_array()) {
            fail(context, "unsupported or invalid pipeline bundle");
            return nullptr;
        }
        if (!validateManifestHash(root, true, context->error))
            return nullptr;
        RuntimeRequirements requirements;
        if (!parseRuntimeRequirements(root, expectedTarget, requirements, context->error) ||
            !validateRuntimeRequirements(*context, requirements))
            return nullptr;
        auto bundle = std::make_unique<VernonPipelineBundle>();
        bundle->context = context;
        bundle->id = root.value("id", "");
        for (const nlohmann::json &feature : root.value("features", nlohmann::json::array()))
            bundle->features.push_back(feature.get<std::string>());
        if (schema2 &&
            (!std::is_sorted(bundle->features.begin(), bundle->features.end()) ||
             std::adjacent_find(bundle->features.begin(), bundle->features.end()) != bundle->features.end())) {
            fail(context, "pipeline feature table is not canonical");
            return nullptr;
        }
        for (const auto &[id, value] : root["stage_artifacts"].items()) {
            if (!value.is_object() ||
                (value.contains("id") && (!value["id"].is_string() || value["id"].get<std::string>() != id))) {
                fail(context, "pipeline stage id is invalid");
                return nullptr;
            }
            Stage stage;
            stage.stage = value.value("stage", "");
            stage.entry = value.value("entry", "");
            if (value.contains("reflection") && value["reflection"].contains("entries")) {
                stage.reflection = value["reflection"].dump();
                for (const nlohmann::json &entry : value["reflection"]["entries"]) {
                    if (entry.value("name", "") != stage.entry)
                        continue;
                    const auto workgroup = entry.value("workgroup_size", nlohmann::json::array());
                    if (workgroup.size() == 3)
                        for (size_t index = 0; index < 3; ++index) {
                            stage.workgroup[index] = workgroup[index].get<uint32_t>();
                            if (!stage.workgroup[index]) {
                                fail(context, "compute workgroup dimensions must be non-zero");
                                return nullptr;
                            }
                        }
                }
            }
            std::optional<ResolvedArtifact> resolved;
            if (schema2) {
                if (!value.contains("artifact")) {
                    fail(context, "pipeline stage artifact descriptor is missing");
                    return nullptr;
                }
                ResolvedArtifact artifact;
                if (!resolveArtifact(value["artifact"], bundleDirectory, artifact, context->error))
                    return nullptr;
                const char *expectedFormat = context->backend == VERNON_RUNTIME_CUDA        ? "ptx"
                                             : context->backend == VERNON_RUNTIME_VULKAN    ? "spirv"
                                             : context->backend == VERNON_RUNTIME_OPENGL_ES ? "gles"
                                                                                            : "glsl";
                const std::string encoding = value["artifact"].value("encoding", "");
                const bool validCpuFormat =
                    context->backend == VERNON_RUNTIME_CPU &&
                    (artifact.format == "native_library" || artifact.format == "relocatable_object");
                const bool validFormat =
                    context->backend == VERNON_RUNTIME_CPU ? validCpuFormat : artifact.format == expectedFormat;
                if (!validFormat || (artifact.external && value["artifact"].contains("encoding")) ||
                    (!artifact.external && ((artifact.format == "spirv" && encoding != "base64") ||
                                            (artifact.format != "spirv" && encoding != "utf8"))) ||
                    value.value("target", "") != expectedTarget) {
                    fail(context, "pipeline stage artifact format is invalid for target");
                    return nullptr;
                }
                resolved = std::move(artifact);
            }
            if (context->backend == VERNON_RUNTIME_CPU) {
                if (!bundleDirectory || (schema2 && (!resolved || !resolved->external))) {
                    fail(context, "CPU pipeline artifacts require an external bundle directory");
                    return nullptr;
                }
                const nlohmann::json &nativeArtifact =
                    value.contains("artifact") && value["artifact"].is_object() ? value["artifact"] : value;
                CpuNativeArtifact artifact;
                artifact.root = *bundleDirectory;
                artifact.relativeLibrary =
                    std::filesystem::u8path(nativeArtifact.value("path", value.value("native_library", "")));
                artifact.entry = stage.entry;
                artifact.format = schema2 ? resolved->format : value.value("format", "");
                artifact.symbol = value.value("symbol", "");
                artifact.operatingSystem = value.value("operating_system", "");
                artifact.architecture = value.value("architecture", "");
                artifact.targetTriple = value.value("target_triple", "");
                artifact.objectFormat = value.value("object_format", "");
                artifact.invocationAbiVersion = value.value("cpu_invocation_abi_version", 0u);
                artifact.size = nativeArtifact.value("size", uint64_t{0});
                artifact.sha256 = nativeArtifact.value("sha256", "");
                if (value.contains("reflection"))
                    artifact.reflection = value["reflection"];
                std::filesystem::path validatedPath;
                const std::string format = schema2 ? resolved->format : value.value("format", "");
                if ((format != "native_library" && format != "relocatable_object") ||
                    !resolveCpuNativeArtifact(artifact, validatedPath, nullptr, context->error))
                    return nullptr;
                stage.cpuArtifact = std::move(artifact);
            }
            if (schema2 && context->backend == VERNON_RUNTIME_VULKAN)
                stage.binary = std::move(resolved->bytes);
            else if (schema2 && context->backend != VERNON_RUNTIME_CPU)
                stage.source.assign(resolved->bytes.begin(), resolved->bytes.end());
            const bool hasArtifact =
                context->backend == VERNON_RUNTIME_CPU ? stage.cpuArtifact.has_value()
                : context->backend == VERNON_RUNTIME_VULKAN
                    ? !stage.binary.empty() && stage.binary.size() % sizeof(uint32_t) == 0 && !stage.reflection.empty()
                : context->backend == VERNON_RUNTIME_CUDA ? (schema2 || value.value("format", "") == "ptx") &&
                                                                !stage.source.empty() && !stage.reflection.empty()
                                                          : !stage.source.empty();
            if (stage.stage.empty() || stage.entry.empty() || !hasArtifact) {
                fail(context, "pipeline stage artifact is invalid");
                return nullptr;
            }
            bundle->stages.emplace(id, std::move(stage));
        }
        for (const nlohmann::json &value : root["variants"]) {
            Variant variant;
            if (!parseVariant(value, variant, context->error))
                return nullptr;
            auto validStage = [&](const std::string &id, const char *kind) {
                if (id.empty())
                    return true;
                const auto found = bundle->stages.find(id);
                return found != bundle->stages.end() && found->second.stage == kind;
            };
            for (const auto &[stage, id] : variant.program)
                if (!validStage(id, stage.c_str())) {
                    fail(context, "pipeline variant references an invalid program stage");
                    return nullptr;
                }
            const bool computeProgram = variant.program.size() == 1 && !variant.compute.empty();
            const bool graphicsProgram =
                variant.program.size() == 2 && !variant.vertex.empty() && !variant.fragment.empty();
            if (!computeProgram && !graphicsProgram) {
                fail(context, "pipeline program stage topology is not supported by this runtime",
                     VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            if ((context->backend == VERNON_RUNTIME_CPU || context->backend == VERNON_RUNTIME_CUDA) &&
                !computeProgram) {
                fail(context, std::string(expectedTarget) + " pipeline bundles support compute programs only",
                     VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            bundle->variants.push_back(std::move(variant));
        }
        std::sort(bundle->variants.begin(), bundle->variants.end(),
                  [](const Variant &left, const Variant &right) { return left.key < right.key; });
        if (std::adjacent_find(bundle->variants.begin(), bundle->variants.end(),
                               [](const Variant &left, const Variant &right) { return left.key == right.key; }) !=
            bundle->variants.end()) {
            fail(context, "pipeline bundle contains duplicate feature variants");
            return nullptr;
        }
        if (bundle->id.empty()) {
            fail(context, "pipeline bundle id is missing or empty");
            return nullptr;
        }
        if (bundle->variants.empty()) {
            fail(context, "pipeline bundle contains no variants");
            return nullptr;
        }
        ++context->liveBundles;
        return bundle.release();
    } catch (const nlohmann::json::exception &error) {
        fail(context, std::string("invalid pipeline bundle: ") + error.what());
        return nullptr;
    } catch (const std::exception &error) {
        fail(context, std::string("failed to load pipeline bundle: ") + error.what(), VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    } catch (...) {
        fail(context, "failed to load pipeline bundle", VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    }
}

VernonStringView vernonRuntimePipelineBundleGetId(const VernonPipelineBundle *bundle) {
    return bundle ? VernonStringView{bundle->id.data(), bundle->id.size()} : VernonStringView{nullptr, 0};
}

namespace {

bool fillParameterView(const Parameter &source, VernonPipelineParameterView &destination) {
    const auto kind = pipelineArgumentKind(source.kind);
    const auto dtype =
        source.kind == "sampler" ? std::optional<VernonDataType>(VERNON_DATA_F32) : pipelineDataType(source.dtype);
    const auto access = pipelineValueAccess(source.access);
    if (!kind || !dtype || !access)
        return false;
    destination = {source.slot,
                   {source.name.data(), source.name.size()},
                   *kind,
                   *dtype,
                   *access,
                   static_cast<uint32_t>(source.shape.size()),
                   source.shape.empty() ? nullptr : source.shape.data()};
    return true;
}

VernonStatus fillTextureConstraintView(const Parameter &source, VernonPipelineTextureConstraintView &destination) {
    if (source.kind != "texture")
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto dimension = pipelineTextureDimension(source.dimension);
    const auto format = source.textureFormat.empty() ? std::optional<VernonTextureFormat>{}
                                                     : pipelineTextureFormat(source.textureFormat);
    if (!dimension || (!source.textureFormat.empty() && !format))
        return VERNON_STATUS_PARSE_ERROR;
    destination.dimension = *dimension;
    destination.has_format_constraint = format.has_value() ? 1u : 0u;
    destination.format = format.value_or(static_cast<VernonTextureFormat>(0));
    std::fill(std::begin(destination.reserved), std::end(destination.reserved), 0);
    return VERNON_STATUS_OK;
}

bool fillOutputView(const Output &source, VernonPipelineOutputView &destination) {
    const auto kind = pipelineArgumentKind(source.kind);
    const auto dtype = pipelineDataType(source.dtype);
    const auto access = pipelineValueAccess(source.access);
    if (!kind || !dtype || !access)
        return false;
    destination = {{source.name.data(), source.name.size()},
                   *kind,
                   *dtype,
                   *access,
                   static_cast<uint32_t>(source.shape.size()),
                   source.shape.empty() ? nullptr : source.shape.data(),
                   source.location};
    return true;
}

bool stringViewEquals(VernonStringView view, const std::string &value) {
    return view.size == value.size() && (!view.size || std::memcmp(view.data, value.data(), view.size) == 0);
}

} // namespace

void vernonRuntimePipelineBundleDestroy(VernonPipelineBundle *bundle) {
    if (!bundle)
        return;
    --bundle->context->liveBundles;
    delete bundle;
}

VernonLoadedPipeline *vernonRuntimeResolvePipeline(VernonPipelineBundle *bundle, VernonFeatureSetView features) {
    if (!bundle || (features.count && !features.names))
        return nullptr;
    std::vector<std::string> key;
    for (size_t index = 0; index < features.count; ++index) {
        if (!features.names[index])
            return nullptr;
        key.emplace_back(features.names[index]);
    }
    std::sort(key.begin(), key.end());
    const auto found = std::find_if(bundle->variants.begin(), bundle->variants.end(),
                                    [&](const Variant &variant) { return variant.key == key; });
    if (found == bundle->variants.end()) {
        fail(bundle->context, "pipeline bundle has no exact feature variant");
        return nullptr;
    }
    auto pipeline = std::make_unique<VernonLoadedPipeline>();
    pipeline->context = bundle->context;
    pipeline->variant = *found;
    if (!resolveBackendPipeline(*bundle, *found, *pipeline))
        return nullptr;
    ++bundle->context->livePipelines;
    return pipeline.release();
}

size_t vernonRuntimeLoadedPipelineGetParameterCount(const VernonLoadedPipeline *pipeline) {
    return pipeline ? pipeline->variant.parameters.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetParameterByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                            VernonPipelineParameterView *parameter) {
    if (!pipeline || !parameter || index >= pipeline->variant.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillParameterView(pipeline->variant.parameters[index], *parameter) ? VERNON_STATUS_OK
                                                                              : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineFindParameter(const VernonLoadedPipeline *pipeline, VernonStringView name,
                                                      VernonPipelineParameterView *parameter) {
    if (!pipeline || !parameter || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found = std::find_if(pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
                                    [&](const Parameter &p) { return stringViewEquals(name, p.name); });
    if (found == pipeline->variant.parameters.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillParameterView(*found, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineGetTextureConstraintByParameterIndex(
    const VernonLoadedPipeline *pipeline, size_t parameterIndex, VernonPipelineTextureConstraintView *constraint) {
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonPipelineTextureConstraintView) ||
        parameterIndex >= pipeline->variant.parameters.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillTextureConstraintView(pipeline->variant.parameters[parameterIndex], *constraint);
}

VernonStatus vernonRuntimeLoadedPipelineFindTextureConstraint(const VernonLoadedPipeline *pipeline,
                                                              VernonStringView parameterName,
                                                              VernonPipelineTextureConstraintView *constraint) {
    if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonPipelineTextureConstraintView) ||
        (parameterName.size && !parameterName.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found =
        std::find_if(pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
                     [&](const Parameter &parameter) { return stringViewEquals(parameterName, parameter.name); });
    if (found == pipeline->variant.parameters.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillTextureConstraintView(*found, *constraint);
}

size_t vernonRuntimeLoadedPipelineGetOutputCount(const VernonLoadedPipeline *pipeline) {
    return pipeline ? pipeline->variant.outputs.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetOutputByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                         VernonPipelineOutputView *output) {
    if (!pipeline || !output || index >= pipeline->variant.outputs.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillOutputView(pipeline->variant.outputs[index], *output) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineFindOutput(const VernonLoadedPipeline *pipeline, VernonStringView name,
                                                   VernonPipelineOutputView *output) {
    if (!pipeline || !output || (name.size && !name.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto found = std::find_if(pipeline->variant.outputs.begin(), pipeline->variant.outputs.end(),
                                    [&](const Output &value) { return stringViewEquals(name, value.name); });
    if (found == pipeline->variant.outputs.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    return fillOutputView(*found, *output) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
}

void vernonRuntimeLoadedPipelineDestroy(VernonLoadedPipeline *pipeline) {
    if (!pipeline)
        return;
    destroyBackendPipeline(*pipeline);
    --pipeline->context->livePipelines;
    delete pipeline;
}

VernonStatus vernonRuntimePipelineInvoke(VernonLoadedPipeline *pipeline, const VernonPipelineInvocation *invocation) {
    if (!pipeline || !invocation || invocation->struct_size < sizeof(VernonPipelineInvocation) ||
        invocation->abi_version != VERNON_PIPELINE_INVOCATION_ABI_VERSION ||
        (invocation->argument_count && !invocation->arguments))
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline invocation");
    PlannedGraphicsInvocation plan;
    std::string planningError;
    const GraphicsPlannerCallbacks plannerCallbacks{nullptr, &plannerBufferSnapshot, &plannerTextureSnapshot,
                                                    &plannerSamplerContext};
    if (!planGraphicsInvocation(pipeline->variant, *invocation, pipeline->context, plannerCallbacks, plan,
                                planningError))
        return fail(pipeline->context, planningError);
    const auto &arguments = plan.arguments;

    if (pipeline->variant.compute.empty())
        return invokeBackendPipeline(*pipeline, *invocation, plan);
    if (isOpenGLBackend(pipeline->context->backend))
        return invokeBackendComputePipeline(*pipeline, *invocation, plan);
    VernonLoadedKernel *computeKernel = backendPipelineComputeKernel(*pipeline);
    if (!computeKernel)
        return fail(pipeline->context, "compute pipeline program is not loaded");
    PlannedComputeLaunch computePlan;
    std::string computePlanningError;
    const ComputePlannerCallbacks computePlannerCallbacks{nullptr, &plannerBufferContext};
    if (!planComputeLaunch(pipeline->variant, arguments, *invocation, pipeline->context, computePlannerCallbacks,
                           computePlan, computePlanningError))
        return fail(pipeline->context, computePlanningError);
    return vernonRuntimeLaunch(computeKernel, computePlan.grid, computePlan.arguments.data(),
                               computePlan.arguments.size());
}

VernonStatus vernonRuntimeComputeToGraphicsBarrier(VernonRuntimeContext *context) {
    if (!context || (context->backend != VERNON_RUNTIME_OPENGL && context->backend != VERNON_RUNTIME_OPENGL_ES))
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    return backendComputeToGraphicsBarrier(*context);
}

VernonStatus vernonRuntimeSynchronize(VernonRuntimeContext *context) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return synchronizeBackend(*context);
}

} // extern "C"
