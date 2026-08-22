#include "VernonRuntimeCore.h"
#include "graphics_variant_key.h"
#include "provider_image_description.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

struct VernonRuntimeCorePipeline {
    VernonRuntimeDeviceProvider provider{};
    VernonRuntimeProviderDeviceIdentity identity{};
    VernonRuntimeProviderPipelineKind kind{};
    VernonRuntimeProviderObject layout{};
    VernonRuntimeProviderObject pipeline{};
    std::array<VernonRuntimeProviderObject, VERNON_RUNTIME_PROVIDER_MAX_SHADER_STAGES> shaders{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> bindings;
    size_t shaderCount{};
    uint32_t pushConstantSize{};
    std::unordered_map<vernon::runtime::GraphicsVariantKey, VernonRuntimeProviderObject,
                       vernon::runtime::GraphicsVariantKeyHash, vernon::runtime::GraphicsVariantKeyEqual>
        graphicsCache;
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

bool packedUniformBytes(const VernonRuntimeProviderBindingLayoutEntry &layout) {
    return layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
           layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
           (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
            layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM);
}

const VernonRuntimeProviderBindingLayoutEntry *layoutForSlot(const VernonRuntimeCorePipeline &pipeline, uint32_t slot) {
    const auto found =
        std::find_if(pipeline.bindings.begin(), pipeline.bindings.end(),
                     [slot](const VernonRuntimeProviderBindingLayoutEntry &layout) { return layout.slot == slot; });
    return found == pipeline.bindings.end() ? nullptr : &*found;
}

const VernonRuntimeProviderResourceReference *bindingResource(const VernonRuntimeProviderBindingValue &value) {
    switch (value.kind) {
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
    case VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER:
        return &value.payload.buffer.resource;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return &value.payload.image.view;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return &value.payload.sampler.resource;
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        return nullptr;
    }
    return nullptr;
}

bool providerIsValid(const VernonRuntimeDeviceProvider &provider, VernonRuntimeProviderPipelineKind kind) {
    if (provider.struct_size < sizeof(VernonRuntimeDeviceProvider) || provider.abi_version != VERNON_PIPELINE_VERSION ||
        !provider.get_capabilities || !provider.get_device_identity || !provider.prepare_shader ||
        !provider.prepare_pipeline_layout || !provider.prepare_pipeline || !provider.create_binding_set ||
        !provider.update_binding_set || !provider.retain_resource || !provider.release_resource ||
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
        if (present(entry.second))
            pipeline->provider.destroy_pipeline(pipeline->provider.user_data, entry.second);
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

bool imageSampleResultClass(VernonTextureFormat format, VernonImageSampleResultClass &result) {
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
    case VERNON_TEXTURE_RGBA8_SRGB:
    case VERNON_TEXTURE_RGBA16_FLOAT:
    case VERNON_TEXTURE_RGBA32_FLOAT:
    case VERNON_TEXTURE_R8_UNORM:
    case VERNON_TEXTURE_R16_FLOAT:
    case VERNON_TEXTURE_R32_FLOAT:
    case VERNON_TEXTURE_RG8_UNORM:
    case VERNON_TEXTURE_RGB8_UNORM:
    case VERNON_TEXTURE_R11G11B10_FLOAT:
    case VERNON_TEXTURE_D32_FLOAT:
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        result = VERNON_IMAGE_SAMPLE_FLOAT;
        return true;
    }
    return false;
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
    for (auto resource = resources.rbegin(); resource != resources.rend(); ++resource)
        provider.release_resource(provider.user_data, *resource);
}

VernonStatus retainResources(const VernonRuntimeCorePipeline &pipeline, const VernonRuntimeProviderBindingValue *values,
                             size_t valueCount, std::vector<VernonRuntimeProviderResourceReference> &resources) {
    const VernonRuntimeDeviceProvider &provider = pipeline.provider;
    if (valueCount != 0 && !values)
        return VERNON_STATUS_INVALID_ARGUMENT;
    size_t resourceCount = 0;
    for (size_t index = 0; index < valueCount; ++index) {
        const auto *layout = layoutForSlot(pipeline, values[index].slot);
        resourceCount += layout && !packedUniformBytes(*layout) &&
                         (values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) == 0;
    }
    try {
        resources.reserve(resourceCount);
    } catch (const std::bad_alloc &) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    for (size_t index = 0; index < valueCount; ++index) {
        const auto *layout = layoutForSlot(pipeline, values[index].slot);
        if (!layout)
            return VERNON_STATUS_INVALID_ARGUMENT;
        if (packedUniformBytes(*layout)) {
            if (!values[index].payload.inline_value.data || values[index].payload.inline_value.size == 0) {
                releaseResources(provider, resources);
                resources.clear();
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            continue;
        }
        if ((values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
            continue;
        const VernonRuntimeProviderResourceReference *resource = bindingResource(values[index]);
        if (!resource || !present(resource->resource) || resource->identity == 0) {
            releaseResources(provider, resources);
            resources.clear();
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        const VernonStatus status = provider.retain_resource(provider.user_data, *resource);
        if (status != VERNON_STATUS_OK) {
            releaseResources(provider, resources);
            resources.clear();
            return status;
        }
        resources.push_back(*resource);
    }
    return VERNON_STATUS_OK;
}

VernonStatus validateImageBindings(const VernonRuntimeCorePipeline &pipeline,
                                   const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != pipeline.bindings.size() || (valueCount && !values))
        return VERNON_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = std::lower_bound(
            pipeline.bindings.begin(), pipeline.bindings.end(), values[index].slot,
            [](const VernonRuntimeProviderBindingLayoutEntry &entry, uint32_t slot) { return entry.slot < slot; });
        if (found == pipeline.bindings.end() || found->slot != values[index].slot || found->kind != values[index].kind)
            return VERNON_STATUS_INVALID_ARGUMENT;
        if (found->kind != VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE &&
            found->kind != VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
            continue;
        if ((values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
            continue;
        if (!pipeline.provider.describe_image)
            return VERNON_STATUS_INVALID_ARGUMENT;
        VernonRuntimeProviderImageDescription description{};
        description.struct_size = sizeof(description);
        const VernonStatus status = pipeline.provider.describe_image(pipeline.provider.user_data,
                                                                     values[index].payload.image.view, &description);
        if (status != VERNON_STATUS_OK)
            return status;
        const uint32_t requiredUsage =
            found->kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ? VERNON_IMAGE_STORAGE : VERNON_IMAGE_SAMPLED;
        VernonImageSampleResultClass sampleResultClass{};
        if (!vernon::runtime::providerImageDescriptionIsCanonical(description) ||
            description.resource_kind != VERNON_RUNTIME_PROVIDER_IMAGE_VIEW ||
            description.view.dimension != found->image_dimension || !(description.image.usage & requiredUsage) ||
            (found->kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE &&
             (!imageSampleResultClass(description.view.format, sampleResultClass) ||
              sampleResultClass != found->sample_result_class)) ||
            (found->kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE &&
             description.view.format != found->storage_image_format))
            return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return VERNON_STATUS_OK;
}

bool drawInvocationIsValid(const VernonRuntimeCorePipeline *pipeline, const VernonRuntimeCoreBindings *bindings,
                           const VernonRuntimeCoreDrawInvocation *invocation) {
    if (!pipeline || pipeline->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || !invocation ||
        invocation->struct_size < sizeof(VernonRuntimeCoreDrawInvocation) || invocation->vertex_count == 0 ||
        invocation->instance_count == 0 || (bindings && bindings->pipeline != pipeline) ||
        (invocation->color_attachment_count != 0 && !invocation->color_attachments) ||
        invocation->color_attachment_count > VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS ||
        ((invocation->index_count != 0) != (invocation->index_buffer.resource.value != 0)) ||
        invocation->depth_load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
        invocation->depth_store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD ||
        invocation->stencil_load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
        invocation->stencil_store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD ||
        !std::isfinite(invocation->clear_depth) || invocation->clear_depth < 0.0f || invocation->clear_depth > 1.0f ||
        invocation->clear_stencil > 0xff || invocation->stencil_reference > 0xff)
        return false;
    for (size_t index = 0; index < invocation->color_attachment_count; ++index) {
        const auto &attachment = invocation->color_attachments[index];
        if (attachment.location != index || attachment.load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
            attachment.store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD ||
            !std::all_of(std::begin(attachment.clear_color), std::end(attachment.clear_color),
                         [](float value) { return std::isfinite(value); }))
            return false;
    }
    return true;
}

VernonStatus validateDrawAttachments(const VernonRuntimeCorePipeline &pipeline,
                                     const VernonRuntimeCoreDrawInvocation &invocation) {
    struct Attachment {
        VernonRuntimeProviderImageDescription description{};
        uint32_t width{};
        uint32_t height{};
    };
    std::array<Attachment, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS + 1> attachments{};
    size_t attachmentCount = 0;
    const auto describe = [&](VernonRuntimeProviderResourceReference view, uint32_t requiredUsage,
                              uint32_t requiredAspects) -> VernonStatus {
        if (!view.identity || !present(view.resource))
            return VERNON_STATUS_INVALID_ARGUMENT;
        Attachment attachment;
        attachment.description.struct_size = sizeof(attachment.description);
        const VernonStatus status =
            pipeline.provider.describe_image(pipeline.provider.user_data, view, &attachment.description);
        if (status != VERNON_STATUS_OK)
            return status;
        const auto &description = attachment.description;
        const auto &range = description.view.subresources;
        if (!vernon::runtime::providerImageDescriptionIsCanonical(description) ||
            description.resource_kind != VERNON_RUNTIME_PROVIDER_IMAGE_VIEW ||
            (description.image.usage & requiredUsage) == 0 || (range.aspects & requiredAspects) == 0 ||
            (range.aspects & ~requiredAspects) != 0 || range.mip_level_count != 1 ||
            description.view.dimension == VERNON_TEXTURE_3D)
            return VERNON_STATUS_INVALID_ARGUMENT;
        attachment.width = std::max(description.image.extent.width >> range.base_mip_level, 1u);
        attachment.height = std::max(description.image.extent.height >> range.base_mip_level, 1u);
        for (size_t index = 0; index < attachmentCount; ++index) {
            const Attachment &existing = attachments[index];
            const auto &left = existing.description.view.subresources;
            const bool mipOverlap = left.base_mip_level < range.base_mip_level + range.mip_level_count &&
                                    range.base_mip_level < left.base_mip_level + left.mip_level_count;
            const bool layerOverlap = left.base_array_layer < range.base_array_layer + range.array_layer_count &&
                                      range.base_array_layer < left.base_array_layer + left.array_layer_count;
            if (existing.description.parent_identity == description.parent_identity && mipOverlap && layerOverlap &&
                (left.aspects & range.aspects) != 0)
                return VERNON_STATUS_INVALID_ARGUMENT;
            if (existing.width != attachment.width || existing.height != attachment.height ||
                existing.description.view.subresources.array_layer_count != range.array_layer_count ||
                existing.description.image.sample_count != description.image.sample_count)
                return VERNON_STATUS_INVALID_ARGUMENT;
        }
        attachments[attachmentCount++] = std::move(attachment);
        return VERNON_STATUS_OK;
    };
    for (size_t index = 0; index < invocation.color_attachment_count; ++index) {
        const VernonStatus status = describe(invocation.color_attachments[index].view, VERNON_IMAGE_COLOR_ATTACHMENT,
                                             VERNON_IMAGE_ASPECT_COLOR);
        if (status != VERNON_STATUS_OK)
            return status;
    }
    if (present(invocation.depth_stencil_view.resource))
        return describe(invocation.depth_stencil_view, VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
                        VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
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
    const bool requiresImageDescriptions =
        descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE ||
        (descriptor->binding_count != 0 &&
         std::any_of(descriptor->bindings, descriptor->bindings + descriptor->binding_count,
                     [](const VernonRuntimeProviderBindingLayoutEntry &binding) {
                         return binding.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                binding.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
                     }));
    if (requiresImageDescriptions && !provider->describe_image)
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
    try {
        pipeline->bindings.assign(descriptor->bindings, descriptor->bindings + descriptor->binding_count);
    } catch (const std::bad_alloc &) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }

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
        descriptor->rasterization,
        descriptor->depth_stencil,
        descriptor->color_blends,
        descriptor->color_blend_count,
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
    VernonStatus status = validateImageBindings(*pipeline, values, valueCount);
    if (status != VERNON_STATUS_OK)
        return status;
    status = retainResources(*pipeline, values, valueCount, bindings->resources);
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
        for (size_t index = 0; index < valueCount; ++index) {
            const auto *layout = layoutForSlot(*pipeline, values[index].slot);
            resourceSlotCount += layout && !packedUniformBytes(*layout);
        }
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
    VernonStatus status = validateImageBindings(*bindings->pipeline, values, valueCount);
    if (status != VERNON_STATUS_OK)
        return status;
    status = retainResources(*bindings->pipeline, values, valueCount, bindings->pendingResources);
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
        (compatibility->vertex_stride_count != 0 && !compatibility->vertex_strides) ||
        (compatibility->color_blend_count != 0 && !compatibility->color_blends) ||
        compatibility->color_blend_count != compatibility->color_format_count)
        return VERNON_STATUS_INVALID_ARGUMENT;

    auto variant =
        std::unique_ptr<VernonRuntimeCoreGraphicsVariant>(new (std::nothrow) VernonRuntimeCoreGraphicsVariant());
    if (!variant)
        return VERNON_STATUS_INTERNAL_ERROR;
    vernon::runtime::GraphicsVariantKey key;
    try {
        key.topology = compatibility->topology;
        if (compatibility->color_format_count)
            key.colorFormats.assign(compatibility->color_formats,
                                    compatibility->color_formats + compatibility->color_format_count);
        key.depthStencilFormat = compatibility->depth_stencil_format;
        key.sampleCount = compatibility->sample_count;
        if (compatibility->vertex_stride_count)
            key.vertexStrides.assign(compatibility->vertex_strides,
                                     compatibility->vertex_strides + compatibility->vertex_stride_count);
        key.rasterization = compatibility->rasterization;
        key.depthStencil = compatibility->depth_stencil;
        if (compatibility->color_blend_count)
            key.colorBlends.assign(compatibility->color_blends,
                                   compatibility->color_blends + compatibility->color_blend_count);
    } catch (const std::bad_alloc &) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    {
        std::lock_guard<std::mutex> guard(pipeline->graphicsCacheMutex);
        const auto cached = pipeline->graphicsCache.find(key);
        if (cached != pipeline->graphicsCache.end()) {
            variant->pipeline = pipeline;
            variant->handle = cached->second;
            pipeline->references.fetch_add(1, std::memory_order_relaxed);
            *output = variant.release();
            return VERNON_STATUS_OK;
        }
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
                                                             compatibility->rasterization,
                                                             compatibility->depth_stencil,
                                                             compatibility->color_blends,
                                                             compatibility->color_blend_count,
                                                             {0, 0, 0, 0}};
    VernonRuntimeProviderObject prepared{};
    const VernonStatus status =
        pipeline->provider.prepare_pipeline(pipeline->provider.user_data, &descriptor, &prepared);
    if (status != VERNON_STATUS_OK || !present(prepared)) {
        if (present(prepared))
            pipeline->provider.destroy_pipeline(pipeline->provider.user_data, prepared);
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }
    {
        std::lock_guard<std::mutex> guard(pipeline->graphicsCacheMutex);
        try {
            const auto [cached, inserted] = pipeline->graphicsCache.emplace(std::move(key), prepared);
            if (!inserted) {
                pipeline->provider.destroy_pipeline(pipeline->provider.user_data, prepared);
                prepared = cached->second;
            }
        } catch (const std::bad_alloc &) {
            pipeline->provider.destroy_pipeline(pipeline->provider.user_data, prepared);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
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
    if (!drawInvocationIsValid(pipeline, bindings, invocation))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus attachmentStatus = validateDrawAttachments(*pipeline, *invocation);
    if (attachmentStatus != VERNON_STATUS_OK)
        return attachmentStatus;
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
        invocation->depth_stencil_view,
        invocation->depth_load_operation,
        invocation->depth_store_operation,
        invocation->clear_depth,
        {invocation->viewport[0], invocation->viewport[1], invocation->viewport[2], invocation->viewport[3]},
        {invocation->scissor[0], invocation->scissor[1], invocation->scissor[2], invocation->scissor[3]},
        invocation->topology,
        invocation->index_buffer,
        invocation->index_count,
        invocation->index_type,
        invocation->stencil_load_operation,
        invocation->stencil_store_operation,
        invocation->clear_stencil,
        invocation->stencil_reference,
        {invocation->render_area[0], invocation->render_area[1], invocation->render_area[2],
         invocation->render_area[3]}};
    return pipeline->provider.encode_draw(pipeline->provider.user_data, invocation->command_encoder, &descriptor);
}

extern "C" VernonStatus
vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(const VernonRuntimeCoreGraphicsVariant *variant,
                                                     const VernonRuntimeCoreBindings *bindings,
                                                     const VernonRuntimeCoreDrawInvocation *invocation) {
    const VernonRuntimeCorePipeline *pipeline = variant ? variant->pipeline : nullptr;
    if (!variant || !drawInvocationIsValid(pipeline, bindings, invocation))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus attachmentStatus = validateDrawAttachments(*pipeline, *invocation);
    if (attachmentStatus != VERNON_STATUS_OK)
        return attachmentStatus;
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
        invocation->depth_stencil_view,
        invocation->depth_load_operation,
        invocation->depth_store_operation,
        invocation->clear_depth,
        {invocation->viewport[0], invocation->viewport[1], invocation->viewport[2], invocation->viewport[3]},
        {invocation->scissor[0], invocation->scissor[1], invocation->scissor[2], invocation->scissor[3]},
        invocation->topology,
        invocation->index_buffer,
        invocation->index_count,
        invocation->index_type,
        invocation->stencil_load_operation,
        invocation->stencil_store_operation,
        invocation->clear_stencil,
        invocation->stencil_reference,
        {invocation->render_area[0], invocation->render_area[1], invocation->render_area[2],
         invocation->render_area[3]}};
    return pipeline->provider.encode_draw(pipeline->provider.user_data, invocation->command_encoder, &descriptor);
}
