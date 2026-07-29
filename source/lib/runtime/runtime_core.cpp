#include "VernonRuntimeCore.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

struct VernonRuntimeCorePipeline {
    struct GraphicsCacheEntry {
        uint32_t topology{};
        std::vector<uint32_t> colorFormats;
        uint32_t depthStencilFormat{};
        uint32_t sampleCount{1};
        std::vector<uint32_t> vertexStrides;
        uint64_t vertexLayoutIdentity{};
        VernonRuntimeProviderObject pipeline{};
    };

    VernonRuntimeDeviceProvider provider{};
    VernonRuntimeProviderDeviceIdentity identity{};
    VernonRuntimeProviderPipelineKind kind{};
    VernonRuntimeProviderObject layout{};
    VernonRuntimeProviderObject pipeline{};
    std::array<VernonRuntimeProviderObject, VERNON_RUNTIME_PROVIDER_MAX_SHADER_STAGES> shaders{};
    size_t shaderCount{};
    uint32_t pushConstantSize{};
    std::vector<GraphicsCacheEntry> graphicsCache;
    std::mutex graphicsCacheMutex;
    std::atomic<uint32_t> references{1};
};

struct VernonRuntimeCoreBindings {
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeProviderObject handle{};
    std::vector<VernonRuntimeProviderResourceReference> resources;
    std::vector<VernonRuntimeProviderResourceReference> pendingResources;
};

struct VernonRuntimeCoreGraphicsVariant {
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeProviderObject handle{};
};

namespace {

constexpr bool present(VernonRuntimeProviderObject object) { return object.value != 0; }

bool providerIsValid(const VernonRuntimeDeviceProvider &provider, VernonRuntimeProviderPipelineKind kind) {
    if (provider.struct_size < sizeof(VernonRuntimeDeviceProvider) ||
        provider.abi_version != VERNON_RUNTIME_DEVICE_PROVIDER_ABI_VERSION || !provider.get_capabilities ||
        !provider.get_device_identity || !provider.prepare_shader || !provider.prepare_pipeline_layout ||
        !provider.prepare_pipeline || !provider.create_binding_set || !provider.update_binding_set ||
        !provider.destroy_shader || !provider.destroy_pipeline_layout || !provider.destroy_pipeline ||
        !provider.destroy_binding_set)
        return false;
    return kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE ? provider.encode_dispatch != nullptr
                                                            : provider.encode_draw != nullptr;
}

void releasePipeline(VernonRuntimeCorePipeline *pipeline) {
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    for (auto &entry : pipeline->graphicsCache)
        if (present(entry.pipeline))
            pipeline->provider.destroy_pipeline(pipeline->provider.user_data, entry.pipeline);
    if (present(pipeline->pipeline))
        pipeline->provider.destroy_pipeline(pipeline->provider.user_data, pipeline->pipeline);
    if (present(pipeline->layout))
        pipeline->provider.destroy_pipeline_layout(pipeline->provider.user_data, pipeline->layout);
    while (pipeline->shaderCount != 0) {
        --pipeline->shaderCount;
        pipeline->provider.destroy_shader(pipeline->provider.user_data, pipeline->shaders[pipeline->shaderCount]);
    }
    delete pipeline;
}

bool layoutIsCanonical(const VernonRuntimeProviderBindingLayoutEntry *bindings, size_t count) {
    if (count != 0 && !bindings)
        return false;
    for (size_t index = 0; index < count; ++index) {
        const auto &binding = bindings[index];
        if (binding.array_count == 0 || binding.stage_mask == 0)
            return false;
        if (index != 0 && bindings[index - 1].slot >= binding.slot)
            return false;
    }
    return true;
}

bool shaderStagesAreValid(const VernonRuntimeCorePipelineDescriptor &descriptor) {
    if (!descriptor.shaders || descriptor.shader_count == 0 ||
        descriptor.shader_count > VERNON_RUNTIME_PROVIDER_MAX_SHADER_STAGES)
        return false;
    uint32_t stages = 0;
    constexpr uint32_t knownStages = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE | VERNON_RUNTIME_PROVIDER_STAGE_VERTEX |
                                     VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
    for (size_t index = 0; index < descriptor.shader_count; ++index) {
        const auto &shader = descriptor.shaders[index];
        if (shader.struct_size < sizeof(VernonRuntimeProviderShaderDescriptor) || shader.stage == 0 ||
            (shader.stage & ~knownStages) != 0 || (shader.stage & (shader.stage - 1)) != 0 ||
            (stages & shader.stage) != 0 || !shader.data || shader.size == 0 || !shader.format.data ||
            shader.format.size == 0 || !shader.entry.data || shader.entry.size == 0) {
            return false;
        }
        stages |= shader.stage;
    }
    if (descriptor.kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE)
        return descriptor.shader_count == 1 && stages == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    constexpr uint32_t graphicsStages = VERNON_RUNTIME_PROVIDER_STAGE_VERTEX | VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
    return (stages & graphicsStages) == graphicsStages && !(stages & VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE);
}

void releaseResources(const VernonRuntimeDeviceProvider &provider,
                      const std::vector<VernonRuntimeProviderResourceReference> &resources) {
    if (!provider.release_resource)
        return;
    for (auto resource = resources.rbegin(); resource != resources.rend(); ++resource)
        provider.release_resource(provider.user_data, *resource);
}

VernonStatus retainResources(const VernonRuntimeDeviceProvider &provider,
                             const VernonRuntimeProviderBindingValue *values, size_t valueCount,
                             std::vector<VernonRuntimeProviderResourceReference> &resources) {
    if (valueCount != 0 && !values)
        return VERNON_STATUS_INVALID_ARGUMENT;
    size_t resourceCount = 0;
    for (size_t index = 0; index < valueCount; ++index)
        resourceCount += values[index].kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                         values[index].kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                         (values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) == 0;
    try {
        resources.reserve(resourceCount);
    } catch (const std::bad_alloc &) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    for (size_t index = 0; index < valueCount; ++index) {
        if (values[index].kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
            values[index].kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            if (!values[index].inline_data || values[index].inline_size == 0) {
                releaseResources(provider, resources);
                resources.clear();
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            continue;
        }
        if ((values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
            continue;
        if (!present(values[index].resource.resource) || values[index].resource.identity == 0) {
            releaseResources(provider, resources);
            resources.clear();
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        if (provider.retain_resource) {
            const VernonStatus status = provider.retain_resource(provider.user_data, values[index].resource);
            if (status != VERNON_STATUS_OK) {
                releaseResources(provider, resources);
                resources.clear();
                return status;
            }
        }
        resources.push_back(values[index].resource);
    }
    return VERNON_STATUS_OK;
}

} // namespace

extern "C" VernonStatus vernonRuntimeCorePreparePipeline(const VernonRuntimeDeviceProvider *provider,
                                                         const VernonRuntimeCorePipelineDescriptor *descriptor,
                                                         VernonRuntimeCorePipeline **output) {
    if (output)
        *output = nullptr;
    if (!provider || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        !providerIsValid(*provider, descriptor->kind) ||
        !layoutIsCanonical(descriptor->bindings, descriptor->binding_count) || !shaderStagesAreValid(*descriptor))
        return VERNON_STATUS_INVALID_ARGUMENT;

    const uint32_t requiredFacet = descriptor->kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE
                                       ? VERNON_RUNTIME_PROVIDER_COMPUTE
                                       : VERNON_RUNTIME_PROVIDER_GRAPHICS;
    if ((descriptor->required_capabilities & requiredFacet) == 0 ||
        (provider->get_capabilities(provider->user_data) & descriptor->required_capabilities) !=
            descriptor->required_capabilities) {
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }

    auto pipeline = std::unique_ptr<VernonRuntimeCorePipeline>(new (std::nothrow) VernonRuntimeCorePipeline());
    if (!pipeline)
        return VERNON_STATUS_INTERNAL_ERROR;
    pipeline->provider = *provider;
    pipeline->identity = provider->get_device_identity(provider->user_data);
    pipeline->kind = descriptor->kind;
    pipeline->pushConstantSize = descriptor->push_constant_size;

    for (size_t index = 0; index < descriptor->shader_count; ++index) {
        VernonRuntimeProviderObject shader{};
        const VernonStatus status = provider->prepare_shader(provider->user_data, &descriptor->shaders[index], &shader);
        if (status != VERNON_STATUS_OK || !present(shader)) {
            releasePipeline(pipeline.release());
            return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
        }
        pipeline->shaders[pipeline->shaderCount++] = shader;
    }

    const VernonRuntimeProviderPipelineLayoutDescriptor layoutDescriptor{
        sizeof(VernonRuntimeProviderPipelineLayoutDescriptor),
        descriptor->bindings,
        descriptor->binding_count,
        descriptor->vertex_attributes,
        descriptor->vertex_attribute_count,
        descriptor->push_constant_size,
        {0, 0, 0, 0}};
    VernonStatus status = provider->prepare_pipeline_layout(provider->user_data, &layoutDescriptor, &pipeline->layout);
    if (status != VERNON_STATUS_OK || !present(pipeline->layout)) {
        releasePipeline(pipeline.release());
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }

    const VernonRuntimeProviderPipelineDescriptor providerDescriptor{
        sizeof(VernonRuntimeProviderPipelineDescriptor),
        descriptor->kind,
        descriptor->required_capabilities,
        pipeline->shaders.data(),
        pipeline->shaderCount,
        pipeline->layout,
        descriptor->topology,
        descriptor->color_formats,
        descriptor->color_format_count,
        descriptor->depth_stencil_format,
        descriptor->sample_count,
        {descriptor->workgroup_size[0], descriptor->workgroup_size[1], descriptor->workgroup_size[2]},
        nullptr,
        0,
        {0, 0, 0, 0}};
    status = provider->prepare_pipeline(provider->user_data, &providerDescriptor, &pipeline->pipeline);
    if (status != VERNON_STATUS_OK || !present(pipeline->pipeline)) {
        releasePipeline(pipeline.release());
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }

    *output = pipeline.release();
    return VERNON_STATUS_OK;
}

extern "C" void vernonRuntimeCorePipelineDestroy(VernonRuntimeCorePipeline *pipeline) { releasePipeline(pipeline); }

extern "C" VernonRuntimeProviderDeviceIdentity
vernonRuntimeCorePipelineGetDeviceIdentity(const VernonRuntimeCorePipeline *pipeline) {
    return pipeline ? pipeline->identity : VernonRuntimeProviderDeviceIdentity{};
}

extern "C" VernonStatus vernonRuntimeCoreCreateBindings(VernonRuntimeCorePipeline *pipeline,
                                                        const VernonRuntimeProviderBindingValue *values,
                                                        size_t valueCount, VernonRuntimeCoreBindings **output) {
    if (output)
        *output = nullptr;
    if (!pipeline || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto bindings = std::unique_ptr<VernonRuntimeCoreBindings>(new (std::nothrow) VernonRuntimeCoreBindings());
    if (!bindings)
        return VERNON_STATUS_INTERNAL_ERROR;
    bindings->pipeline = pipeline;
    VernonStatus status = retainResources(pipeline->provider, values, valueCount, bindings->resources);
    if (status != VERNON_STATUS_OK)
        return status;
    const VernonRuntimeProviderBindingSetDescriptor descriptor{
        sizeof(VernonRuntimeProviderBindingSetDescriptor), pipeline->layout, values, valueCount, {0, 0, 0, 0}};
    status = pipeline->provider.create_binding_set(pipeline->provider.user_data, &descriptor, &bindings->handle);
    if (status != VERNON_STATUS_OK || !present(bindings->handle)) {
        releaseResources(pipeline->provider, bindings->resources);
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }
    try {
        size_t resourceSlotCount = 0;
        for (size_t index = 0; index < valueCount; ++index)
            resourceSlotCount += values[index].kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                                 values[index].kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
        bindings->resources.reserve(resourceSlotCount);
        bindings->pendingResources.reserve(resourceSlotCount);
    } catch (const std::bad_alloc &) {
        pipeline->provider.destroy_binding_set(pipeline->provider.user_data, bindings->handle);
        releaseResources(pipeline->provider, bindings->resources);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    pipeline->references.fetch_add(1, std::memory_order_relaxed);
    *output = bindings.release();
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeCoreUpdateBindings(VernonRuntimeCoreBindings *bindings,
                                                        const VernonRuntimeProviderBindingValue *values,
                                                        size_t valueCount) {
    if (!bindings)
        return VERNON_STATUS_INVALID_ARGUMENT;
    bindings->pendingResources.clear();
    VernonStatus status = retainResources(bindings->pipeline->provider, values, valueCount, bindings->pendingResources);
    if (status != VERNON_STATUS_OK)
        return status;
    status = bindings->pipeline->provider.update_binding_set(bindings->pipeline->provider.user_data, bindings->handle,
                                                             values, valueCount);
    if (status != VERNON_STATUS_OK) {
        releaseResources(bindings->pipeline->provider, bindings->pendingResources);
        bindings->pendingResources.clear();
        return status;
    }
    releaseResources(bindings->pipeline->provider, bindings->resources);
    bindings->resources.swap(bindings->pendingResources);
    bindings->pendingResources.clear();
    return VERNON_STATUS_OK;
}

extern "C" void vernonRuntimeCoreBindingsDestroy(VernonRuntimeCoreBindings *bindings) {
    if (!bindings)
        return;
    VernonRuntimeCorePipeline *pipeline = bindings->pipeline;
    pipeline->provider.destroy_binding_set(pipeline->provider.user_data, bindings->handle);
    releaseResources(pipeline->provider, bindings->resources);
    delete bindings;
    releasePipeline(pipeline);
}

extern "C" VernonStatus
vernonRuntimeCorePrepareGraphicsVariant(VernonRuntimeCorePipeline *pipeline,
                                        const VernonRuntimeCoreGraphicsCompatibility *compatibility,
                                        VernonRuntimeCoreGraphicsVariant **output) {
    if (output)
        *output = nullptr;
    if (!pipeline || pipeline->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || !compatibility || !output ||
        compatibility->struct_size < sizeof(*compatibility) || compatibility->sample_count == 0 ||
        (compatibility->color_format_count != 0 && !compatibility->color_formats) ||
        (compatibility->vertex_stride_count != 0 && !compatibility->vertex_strides))
        return VERNON_STATUS_INVALID_ARGUMENT;

    auto variant =
        std::unique_ptr<VernonRuntimeCoreGraphicsVariant>(new (std::nothrow) VernonRuntimeCoreGraphicsVariant());
    if (!variant)
        return VERNON_STATUS_INTERNAL_ERROR;
    std::lock_guard<std::mutex> guard(pipeline->graphicsCacheMutex);
    const auto cached =
        std::find_if(pipeline->graphicsCache.begin(), pipeline->graphicsCache.end(), [&](const auto &entry) {
            return entry.topology == compatibility->topology &&
                   entry.depthStencilFormat == compatibility->depth_stencil_format &&
                   entry.sampleCount == compatibility->sample_count &&
                   entry.vertexLayoutIdentity == compatibility->vertex_layout_identity &&
                   entry.vertexStrides.size() == compatibility->vertex_stride_count &&
                   (entry.vertexStrides.empty() || std::equal(entry.vertexStrides.begin(), entry.vertexStrides.end(),
                                                              compatibility->vertex_strides)) &&
                   entry.colorFormats.size() == compatibility->color_format_count &&
                   (entry.colorFormats.empty() ||
                    std::equal(entry.colorFormats.begin(), entry.colorFormats.end(), compatibility->color_formats));
        });
    if (cached != pipeline->graphicsCache.end()) {
        variant->pipeline = pipeline;
        variant->handle = cached->pipeline;
        pipeline->references.fetch_add(1, std::memory_order_relaxed);
        *output = variant.release();
        return VERNON_STATUS_OK;
    }

    const VernonRuntimeProviderPipelineDescriptor descriptor{sizeof(VernonRuntimeProviderPipelineDescriptor),
                                                             VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE,
                                                             VERNON_RUNTIME_PROVIDER_GRAPHICS,
                                                             pipeline->shaders.data(),
                                                             pipeline->shaderCount,
                                                             pipeline->layout,
                                                             compatibility->topology,
                                                             compatibility->color_formats,
                                                             compatibility->color_format_count,
                                                             compatibility->depth_stencil_format,
                                                             compatibility->sample_count,
                                                             {1, 1, 1},
                                                             compatibility->vertex_strides,
                                                             compatibility->vertex_stride_count,
                                                             {0, 0, 0, 0}};
    VernonRuntimeProviderObject prepared{};
    const VernonStatus status =
        pipeline->provider.prepare_pipeline(pipeline->provider.user_data, &descriptor, &prepared);
    if (status != VERNON_STATUS_OK || !present(prepared))
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    try {
        VernonRuntimeCorePipeline::GraphicsCacheEntry entry;
        entry.topology = compatibility->topology;
        if (compatibility->color_format_count != 0)
            entry.colorFormats.assign(compatibility->color_formats,
                                      compatibility->color_formats + compatibility->color_format_count);
        entry.depthStencilFormat = compatibility->depth_stencil_format;
        entry.sampleCount = compatibility->sample_count;
        if (compatibility->vertex_stride_count != 0)
            entry.vertexStrides.assign(compatibility->vertex_strides,
                                       compatibility->vertex_strides + compatibility->vertex_stride_count);
        entry.vertexLayoutIdentity = compatibility->vertex_layout_identity;
        entry.pipeline = prepared;
        pipeline->graphicsCache.push_back(std::move(entry));
    } catch (const std::bad_alloc &) {
        pipeline->provider.destroy_pipeline(pipeline->provider.user_data, prepared);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    variant->pipeline = pipeline;
    variant->handle = prepared;
    pipeline->references.fetch_add(1, std::memory_order_relaxed);
    *output = variant.release();
    return VERNON_STATUS_OK;
}

extern "C" void vernonRuntimeCoreGraphicsVariantDestroy(VernonRuntimeCoreGraphicsVariant *variant) {
    if (!variant)
        return;
    VernonRuntimeCorePipeline *pipeline = variant->pipeline;
    delete variant;
    releasePipeline(pipeline);
}

extern "C" VernonStatus vernonRuntimeCoreEncodeDispatch(const VernonRuntimeCorePipeline *pipeline,
                                                        const VernonRuntimeCoreBindings *bindings,
                                                        VernonRuntimeProviderObject commandEncoder,
                                                        const uint32_t groupCount[3], const void *pushConstants,
                                                        size_t pushConstantSize) {
    if (!pipeline || pipeline->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE || !groupCount || groupCount[0] == 0 ||
        groupCount[1] == 0 || groupCount[2] == 0 || (bindings && bindings->pipeline != pipeline) ||
        pushConstantSize > pipeline->pushConstantSize || (pushConstantSize != 0 && !pushConstants))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonRuntimeProviderDispatchDescriptor descriptor{sizeof(VernonRuntimeProviderDispatchDescriptor),
                                                             pipeline->pipeline,
                                                             bindings ? bindings->handle
                                                                      : VernonRuntimeProviderObject{},
                                                             {groupCount[0], groupCount[1], groupCount[2]},
                                                             pushConstants,
                                                             pushConstantSize,
                                                             {0, 0, 0, 0}};
    return pipeline->provider.encode_dispatch(pipeline->provider.user_data, commandEncoder, &descriptor);
}

extern "C" VernonStatus vernonRuntimeCoreEncodeDraw(const VernonRuntimeCorePipeline *pipeline,
                                                    const VernonRuntimeCoreBindings *bindings,
                                                    VernonRuntimeProviderObject commandEncoder, uint32_t vertexCount,
                                                    uint32_t instanceCount, uint32_t firstVertex,
                                                    uint32_t firstInstance) {
    VernonRuntimeCoreDrawInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.command_encoder = commandEncoder;
    invocation.vertex_count = vertexCount;
    invocation.instance_count = instanceCount;
    invocation.first_vertex = firstVertex;
    invocation.first_instance = firstInstance;
    return vernonRuntimeCoreEncodeDrawInvocation(pipeline, bindings, &invocation);
}

extern "C" VernonStatus vernonRuntimeCoreEncodeDrawInvocation(const VernonRuntimeCorePipeline *pipeline,
                                                              const VernonRuntimeCoreBindings *bindings,
                                                              const VernonRuntimeCoreDrawInvocation *invocation) {
    if (!pipeline || pipeline->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || !invocation ||
        invocation->struct_size < sizeof(VernonRuntimeCoreDrawInvocation) || invocation->vertex_count == 0 ||
        invocation->instance_count == 0 || (bindings && bindings->pipeline != pipeline) ||
        (invocation->color_attachment_count != 0 && !invocation->color_attachments) ||
        ((invocation->index_count != 0) != (invocation->index_buffer.resource.value != 0)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonRuntimeProviderDrawDescriptor descriptor{
        sizeof(VernonRuntimeProviderDrawDescriptor),
        pipeline->pipeline,
        bindings ? bindings->handle : VernonRuntimeProviderObject{},
        invocation->vertex_count,
        invocation->instance_count,
        invocation->first_vertex,
        invocation->first_instance,
        invocation->color_attachments,
        invocation->color_attachment_count,
        invocation->depth_stencil_attachment,
        invocation->depth_load_operation,
        invocation->depth_store_operation,
        invocation->clear_depth,
        {invocation->viewport[0], invocation->viewport[1], invocation->viewport[2], invocation->viewport[3]},
        {invocation->scissor[0], invocation->scissor[1], invocation->scissor[2], invocation->scissor[3]},
        invocation->topology,
        invocation->index_buffer,
        invocation->index_count,
        invocation->index_type,
        {0, 0, 0, 0}};
    return pipeline->provider.encode_draw(pipeline->provider.user_data, invocation->command_encoder, &descriptor);
}

extern "C" VernonStatus
vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(const VernonRuntimeCoreGraphicsVariant *variant,
                                                     const VernonRuntimeCoreBindings *bindings,
                                                     const VernonRuntimeCoreDrawInvocation *invocation) {
    const VernonRuntimeCorePipeline *pipeline = variant ? variant->pipeline : nullptr;
    if (!variant || !pipeline || pipeline->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || !invocation ||
        invocation->struct_size < sizeof(VernonRuntimeCoreDrawInvocation) || invocation->vertex_count == 0 ||
        invocation->instance_count == 0 || (bindings && bindings->pipeline != pipeline) ||
        (invocation->color_attachment_count != 0 && !invocation->color_attachments) ||
        ((invocation->index_count != 0) != (invocation->index_buffer.resource.value != 0)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonRuntimeProviderDrawDescriptor descriptor{
        sizeof(VernonRuntimeProviderDrawDescriptor),
        variant->handle,
        bindings ? bindings->handle : VernonRuntimeProviderObject{},
        invocation->vertex_count,
        invocation->instance_count,
        invocation->first_vertex,
        invocation->first_instance,
        invocation->color_attachments,
        invocation->color_attachment_count,
        invocation->depth_stencil_attachment,
        invocation->depth_load_operation,
        invocation->depth_store_operation,
        invocation->clear_depth,
        {invocation->viewport[0], invocation->viewport[1], invocation->viewport[2], invocation->viewport[3]},
        {invocation->scissor[0], invocation->scissor[1], invocation->scissor[2], invocation->scissor[3]},
        invocation->topology,
        invocation->index_buffer,
        invocation->index_count,
        invocation->index_type,
        {0, 0, 0, 0}};
    return pipeline->provider.encode_draw(pipeline->provider.user_data, invocation->command_encoder, &descriptor);
}
