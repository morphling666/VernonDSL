#include "VernonFormats.hpp"
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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

template <typename Callback> VernonStatus runtimeCoreBoundary(Callback &&callback) noexcept {
    try {
        return std::forward<Callback>(callback)();
    } catch (...) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

template <typename Value, typename Callback>
Value runtimeCoreValueBoundary(Value fallback, Callback &&callback) noexcept {
    try {
        return std::forward<Callback>(callback)();
    } catch (...) {
        return fallback;
    }
}

template <typename Callback> void runtimeCoreVoidBoundary(Callback &&callback) noexcept {
    try {
        std::forward<Callback>(callback)();
    } catch (...) {
    }
}

template <typename Callback> VernonStatus providerStatusBoundary(Callback &&callback) noexcept {
    try {
        return std::forward<Callback>(callback)();
    } catch (...) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

template <typename Value, typename Callback>
VernonStatus providerValueBoundary(Value &output, Callback &&callback) noexcept {
    try {
        output = std::forward<Callback>(callback)();
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

template <typename Callback> void providerVoidBoundary(Callback &&callback) noexcept {
    try {
        std::forward<Callback>(callback)();
    } catch (...) {
    }
}

constexpr size_t kMaxGroups = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;

} // namespace

struct VernonRuntimeCoreGroupLayout {
    VernonRuntimeDeviceProvider provider{};
    VernonRuntimeProviderObject handle{};
    // Entry names point into `names`.
    std::vector<VernonRuntimeProviderBindingLayoutEntry> entries;
    std::vector<std::string> names;
    size_t hash{};
    std::atomic<uint32_t> references{1};
};

struct VernonRuntimeCorePipeline {
    VernonRuntimeDeviceProvider provider{};
    VernonRuntimeProviderDeviceIdentity identity{};
    VernonRuntimeProviderPipelineKind kind{};
    VernonRuntimeProviderObject layout{};
    VernonRuntimeProviderObject pipeline{};
    std::array<VernonRuntimeProviderObject, VERNON_RUNTIME_PROVIDER_MAX_SHADER_STAGES> shaders{};
    size_t shaderCount{};
    std::array<VernonRuntimeCoreGroupLayout *, kMaxGroups> groupLayouts{};
    size_t groupCount{};
    // Bit g is set when group layout g has entries, so a dispatch or draw needs a group bound at g.
    uint32_t requiredGroups{};
    uint32_t pushConstantSize{};
    std::unordered_map<vernon::runtime::GraphicsVariantKey, VernonRuntimeProviderObject,
                       vernon::runtime::GraphicsVariantKeyHash, vernon::runtime::GraphicsVariantKeyEqual>
        graphicsCache;
    std::mutex graphicsCacheMutex;
    std::atomic<uint32_t> references{1};
};

struct VernonRuntimeCoreGroup {
    VernonRuntimeCoreGroupLayout *layout{};
    VernonRuntimeProviderObject handle{};
    bool transient{};
    VernonRuntimeProviderObject encoder{};
    std::atomic<uint32_t> references{1};
};

struct VernonRuntimeCoreGraphicsVariant {
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeProviderObject handle{};
};

struct VernonRuntimeCoreCommands {
    VernonRuntimeDeviceProvider provider{};
    VernonRuntimeProviderObject encoder{};
    VernonRuntimeProviderObject native{};
    // The bound pipeline's owner, referenced while bound, and the bound native pipeline or variant.
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeProviderObject bound{};
    // Referenced while bound.
    std::array<VernonRuntimeCoreGroup *, kMaxGroups> groups{};
    uint32_t boundGroups{};
    bool rendering{};
};

namespace {

constexpr bool present(VernonRuntimeProviderObject object) { return object.value != 0; }

bool packedUniformBytes(const VernonRuntimeProviderBindingLayoutEntry &layout) {
    return layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
           layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
           (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
            layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM);
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

bool providerIsValid(const VernonRuntimeDeviceProvider &provider) {
    return provider.struct_size >= sizeof(VernonRuntimeDeviceProvider) &&
           provider.abi_version == VERNON_PROGRAM_VERSION && provider.get_capabilities &&
           provider.get_device_identity && provider.prepare_shader && provider.prepare_group_layout &&
           provider.prepare_pipeline_layout && provider.prepare_pipeline && provider.create_group &&
           provider.begin_commands && provider.end_commands && provider.encode_bind_pipeline &&
           provider.encode_bind_group && provider.destroy_shader && provider.destroy_group_layout &&
           provider.destroy_pipeline_layout && provider.destroy_pipeline && provider.destroy_group;
}

bool providerSupports(const VernonRuntimeDeviceProvider &provider, VernonRuntimeProviderPipelineKind kind) {
    if (!providerIsValid(provider))
        return false;
    return kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE
               ? provider.encode_dispatch != nullptr
               : provider.encode_draw && provider.begin_rendering && provider.end_rendering && provider.describe_image;
}

// Group layouts are shared per provider, so that equal layouts are one object.
struct GroupLayoutRegistry {
    std::mutex mutex;
    std::unordered_multimap<size_t, VernonRuntimeCoreGroupLayout *> layouts;
};

GroupLayoutRegistry &groupLayoutRegistry() {
    static GroupLayoutRegistry registry;
    return registry;
}

bool sameProvider(const VernonRuntimeDeviceProvider &left, const VernonRuntimeDeviceProvider &right) {
    return left.user_data == right.user_data && left.prepare_group_layout == right.prepare_group_layout;
}

bool sameEntry(const VernonRuntimeProviderBindingLayoutEntry &left,
               const VernonRuntimeProviderBindingLayoutEntry &right) {
    return left.slot == right.slot && left.binding == right.binding && left.kind == right.kind &&
           left.stage_mask == right.stage_mask && left.access == right.access &&
           left.array_count == right.array_count && left.argument_index == right.argument_index &&
           left.element_size == right.element_size && left.interface_kind == right.interface_kind &&
           std::string_view(left.name.data ? left.name.data : "", left.name.size) ==
               std::string_view(right.name.data ? right.name.data : "", right.name.size) &&
           left.element_count == right.element_count && left.vector_count == right.vector_count &&
           left.divisor == right.divisor && left.element_alignment == right.element_alignment &&
           left.numeric_type == right.numeric_type && left.image_dimension == right.image_dimension &&
           left.storage_image_format == right.storage_image_format &&
           left.sample_result_class == right.sample_result_class;
}

size_t entriesHash(const VernonRuntimeDeviceProvider &provider, const VernonRuntimeProviderBindingLayoutEntry *entries,
                   size_t count) {
    size_t hash = std::hash<const void *>{}(provider.user_data);
    const auto mix = [&](size_t value) { hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2); };
    mix(count);
    for (size_t index = 0; index < count; ++index) {
        const auto &entry = entries[index];
        mix(entry.binding);
        mix(entry.kind);
        mix(entry.argument_index);
        mix(entry.element_size);
        mix(std::hash<std::string_view>{}(std::string_view(entry.name.data ? entry.name.data : "", entry.name.size)));
    }
    return hash;
}

bool groupEntriesAreCanonical(const VernonRuntimeProviderBindingLayoutEntry *entries, size_t count) {
    if (count != 0 && !entries)
        return false;
    for (size_t index = 0; index < count; ++index)
        if (entries[index].slot != index || entries[index].array_count == 0 || entries[index].stage_mask == 0 ||
            (entries[index].name.size != 0 && !entries[index].name.data))
            return false;
    return true;
}

void retainGroupLayout(VernonRuntimeCoreGroupLayout *layout) noexcept {
    layout->references.fetch_add(1, std::memory_order_relaxed);
}

void releaseGroupLayout(VernonRuntimeCoreGroupLayout *layout) noexcept {
    if (!layout)
        return;
    GroupLayoutRegistry &registry = groupLayoutRegistry();
    {
        // Under the registry lock, so that a lookup never revives a layout that is being destroyed.
        std::lock_guard<std::mutex> guard(registry.mutex);
        if (layout->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
            return;
        auto [begin, end] = registry.layouts.equal_range(layout->hash);
        for (auto entry = begin; entry != end; ++entry)
            if (entry->second == layout) {
                registry.layouts.erase(entry);
                break;
            }
    }
    providerVoidBoundary([&] { layout->provider.destroy_group_layout(layout->provider.user_data, layout->handle); });
    delete layout;
}

VernonStatus prepareGroupLayoutImpl(const VernonRuntimeDeviceProvider *provider,
                                    const VernonRuntimeProviderBindingLayoutEntry *entries, size_t entryCount,
                                    VernonRuntimeCoreGroupLayout **output) {
    if (output)
        *output = nullptr;
    if (!provider || !output || !providerIsValid(*provider) || !groupEntriesAreCanonical(entries, entryCount))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (std::any_of(entries, entries + entryCount,
                    [](const VernonRuntimeProviderBindingLayoutEntry &entry) {
                        return entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                               entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
                    }) &&
        !provider->describe_image)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const size_t hash = entriesHash(*provider, entries, entryCount);
    GroupLayoutRegistry &registry = groupLayoutRegistry();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto [begin, end] = registry.layouts.equal_range(hash);
    for (auto entry = begin; entry != end; ++entry) {
        VernonRuntimeCoreGroupLayout *candidate = entry->second;
        if (sameProvider(candidate->provider, *provider) && candidate->entries.size() == entryCount &&
            std::equal(candidate->entries.begin(), candidate->entries.end(), entries, sameEntry)) {
            retainGroupLayout(candidate);
            *output = candidate;
            return VERNON_STATUS_OK;
        }
    }
    auto layout = std::make_unique<VernonRuntimeCoreGroupLayout>();
    layout->provider = *provider;
    layout->hash = hash;
    layout->entries.assign(entries, entries + entryCount);
    layout->names.reserve(entryCount);
    for (VernonRuntimeProviderBindingLayoutEntry &entry : layout->entries) {
        entry.set = 0;
        layout->names.emplace_back(entry.name.data ? entry.name.data : "", entry.name.size);
        entry.name = {entry.name.size ? layout->names.back().data() : nullptr, entry.name.size};
    }
    const VernonRuntimeProviderGroupLayoutDescriptor descriptor{
        sizeof(descriptor), layout->entries.data(), layout->entries.size(), {0, 0, 0, 0}};
    const VernonStatus status = providerStatusBoundary(
        [&] { return provider->prepare_group_layout(provider->user_data, &descriptor, &layout->handle); });
    if (status != VERNON_STATUS_OK || !present(layout->handle)) {
        if (present(layout->handle))
            providerVoidBoundary([&] { provider->destroy_group_layout(provider->user_data, layout->handle); });
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }
    registry.layouts.emplace(hash, layout.get());
    *output = layout.release();
    return VERNON_STATUS_OK;
}

void retainPipeline(VernonRuntimeCorePipeline *pipeline) noexcept {
    pipeline->references.fetch_add(1, std::memory_order_relaxed);
}

void releasePipeline(VernonRuntimeCorePipeline *pipeline) noexcept {
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    for (auto &entry : pipeline->graphicsCache)
        if (present(entry.second))
            providerVoidBoundary(
                [&] { pipeline->provider.destroy_pipeline(pipeline->provider.user_data, entry.second); });
    if (present(pipeline->pipeline))
        providerVoidBoundary(
            [&] { pipeline->provider.destroy_pipeline(pipeline->provider.user_data, pipeline->pipeline); });
    if (present(pipeline->layout))
        providerVoidBoundary(
            [&] { pipeline->provider.destroy_pipeline_layout(pipeline->provider.user_data, pipeline->layout); });
    while (pipeline->shaderCount != 0) {
        --pipeline->shaderCount;
        providerVoidBoundary([&] {
            pipeline->provider.destroy_shader(pipeline->provider.user_data, pipeline->shaders[pipeline->shaderCount]);
        });
    }
    for (size_t index = 0; index < pipeline->groupCount; ++index)
        releaseGroupLayout(pipeline->groupLayouts[index]);
    delete pipeline;
}

void retainGroup(VernonRuntimeCoreGroup *group) noexcept { group->references.fetch_add(1, std::memory_order_relaxed); }

void releaseGroup(VernonRuntimeCoreGroup *group) noexcept {
    if (!group || group->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    providerVoidBoundary(
        [&] { group->layout->provider.destroy_group(group->layout->provider.user_data, group->handle); });
    releaseGroupLayout(group->layout);
    delete group;
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

bool groupLayoutsAreValid(const VernonRuntimeCorePipelineDescriptor &descriptor,
                          const VernonRuntimeDeviceProvider &provider) {
    if (descriptor.group_layout_count > kMaxGroups || (descriptor.group_layout_count != 0 && !descriptor.group_layouts))
        return false;
    return std::all_of(
        descriptor.group_layouts, descriptor.group_layouts + descriptor.group_layout_count,
        [&](const VernonRuntimeCoreGroupLayout *layout) { return layout && sameProvider(layout->provider, provider); });
}

// Values match their entries' kinds, and every value that names a resource names one.
bool groupValuesMatchLayout(const VernonRuntimeCoreGroupLayout &layout, const VernonRuntimeProviderBindingValue *values,
                            size_t valueCount) {
    if (valueCount != layout.entries.size() || (valueCount != 0 && !values))
        return false;
    for (size_t index = 0; index < valueCount; ++index) {
        const VernonRuntimeProviderBindingValue &value = values[index];
        const VernonRuntimeProviderBindingLayoutEntry &entry = layout.entries[index];
        if (value.slot != entry.slot || value.kind != entry.kind)
            return false;
        if (packedUniformBytes(entry) || (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE) != 0) {
            if (!value.payload.inline_value.data || value.payload.inline_value.size == 0)
                return false;
            continue;
        }
        if ((value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
            continue;
        const VernonRuntimeProviderResourceReference *resource = bindingResource(value);
        if (!resource || !present(resource->resource) || resource->identity == 0)
            return false;
    }
    return true;
}

VernonStatus validateImageValues(const VernonRuntimeCoreGroupLayout &layout,
                                 const VernonRuntimeProviderBindingValue *values) {
    const VernonRuntimeDeviceProvider &provider = layout.provider;
    for (size_t index = 0; index < layout.entries.size(); ++index) {
        const VernonRuntimeProviderBindingLayoutEntry &entry = layout.entries[index];
        if (entry.kind != VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE && entry.kind != VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
            continue;
        if ((values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
            continue;
        VernonRuntimeProviderImageDescription description{};
        description.struct_size = sizeof(description);
        const VernonStatus status = providerStatusBoundary([&] {
            return provider.describe_image(provider.user_data, values[index].payload.image.view, &description);
        });
        if (status != VERNON_STATUS_OK)
            return status;
        const uint32_t requiredUsage =
            entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ? VERNON_IMAGE_STORAGE : VERNON_IMAGE_SAMPLED;
        if (!vernon::runtime::providerImageDescriptionIsCanonical(description) ||
            description.resource_kind != VERNON_RUNTIME_PROVIDER_IMAGE_VIEW ||
            description.view.dimension != entry.image_dimension || !(description.image.usage & requiredUsage) ||
            (entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE &&
             !vernon::formats::providesSampleResultClass(description.view.format, entry.sample_result_class)) ||
            (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE &&
             description.view.format != entry.storage_image_format))
            return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return VERNON_STATUS_OK;
}

bool drawInvocationIsValid(const VernonRuntimeCoreDrawInvocation *invocation) {
    return invocation && invocation->struct_size >= sizeof(VernonRuntimeCoreDrawInvocation) &&
           (invocation->vertex_count != 0 || invocation->index_count != 0) && invocation->instance_count != 0 &&
           (invocation->index_count != 0) == (invocation->index_buffer.resource.value != 0) &&
           invocation->stencil_reference <= 0xff;
}

bool renderingDescriptorIsValid(const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(VernonRuntimeProviderRenderingDescriptor) ||
        (descriptor->color_attachment_count != 0 && !descriptor->color_attachments) ||
        descriptor->color_attachment_count > VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS ||
        (descriptor->color_attachment_count == 0 && !descriptor->depth_stencil_attachment) ||
        (descriptor->sampled_image_count != 0 && !descriptor->sampled_images) || descriptor->render_area[2] == 0 ||
        descriptor->render_area[3] == 0 || descriptor->layers == 0)
        return false;
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        const auto &attachment = descriptor->color_attachments[index];
        if (attachment.location != index || attachment.load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
            attachment.store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD)
            return false;
    }
    const VernonRuntimeProviderDepthStencilAttachment *depth = descriptor->depth_stencil_attachment;
    return !depth || (depth->depth_load_operation <= VERNON_RUNTIME_PROVIDER_LOAD_DISCARD &&
                      depth->depth_store_operation <= VERNON_RUNTIME_PROVIDER_STORE_DISCARD &&
                      depth->stencil_load_operation <= VERNON_RUNTIME_PROVIDER_LOAD_DISCARD &&
                      depth->stencil_store_operation <= VERNON_RUNTIME_PROVIDER_STORE_DISCARD &&
                      std::isfinite(depth->clear_depth) && depth->clear_depth >= 0.0f && depth->clear_depth <= 1.0f &&
                      depth->clear_stencil <= 0xff);
}

// Attachments are distinct single-mip views that share their extent, layer count and sample count and contain the
// render area; sampled images have sampled usage and overlap no attachment.
VernonStatus validateRenderingAttachments(const VernonRuntimeDeviceProvider &provider,
                                          const VernonRuntimeProviderRenderingDescriptor &rendering) {
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
        const VernonStatus status = providerStatusBoundary(
            [&] { return provider.describe_image(provider.user_data, view, &attachment.description); });
        if (status != VERNON_STATUS_OK)
            return status;
        const auto &description = attachment.description;
        const auto &range = description.view.subresources;
        if (!vernon::runtime::providerImageDescriptionIsCanonical(description) ||
            description.resource_kind != VERNON_RUNTIME_PROVIDER_IMAGE_VIEW ||
            (description.image.usage & requiredUsage) == 0 || (range.aspects & requiredAspects) == 0 ||
            (range.aspects & ~requiredAspects) != 0 || range.mip_level_count != 1 ||
            description.view.dimension == VERNON_TEXTURE_3D || range.array_layer_count < rendering.layers)
            return VERNON_STATUS_INVALID_ARGUMENT;
        attachment.width = std::max(description.image.extent.width >> range.base_mip_level, 1u);
        attachment.height = std::max(description.image.extent.height >> range.base_mip_level, 1u);
        if (uint64_t{rendering.render_area[0]} + rendering.render_area[2] > attachment.width ||
            uint64_t{rendering.render_area[1]} + rendering.render_area[3] > attachment.height)
            return VERNON_STATUS_INVALID_ARGUMENT;
        for (size_t index = 0; index < attachmentCount; ++index) {
            const Attachment &existing = attachments[index];
            if (vernon::runtime::providerImageSubresourcesOverlap(
                    vernon::runtime::providerImageSubresources(existing.description),
                    vernon::runtime::providerImageSubresources(description)))
                return VERNON_STATUS_INVALID_ARGUMENT;
            if (existing.width != attachment.width || existing.height != attachment.height ||
                existing.description.view.subresources.array_layer_count != range.array_layer_count ||
                existing.description.image.sample_count != description.image.sample_count)
                return VERNON_STATUS_INVALID_ARGUMENT;
        }
        attachments[attachmentCount++] = std::move(attachment);
        return VERNON_STATUS_OK;
    };
    for (size_t index = 0; index < rendering.color_attachment_count; ++index) {
        const VernonRuntimeProviderColorAttachment &color = rendering.color_attachments[index];
        const VernonStatus status = describe(color.view, VERNON_IMAGE_COLOR_ATTACHMENT, VERNON_IMAGE_ASPECT_COLOR);
        if (status != VERNON_STATUS_OK)
            return status;
        // The view's format selects the clear member: integer formats read raw bits, the others finite floats.
        if (!vernon::formats::isInteger(attachments[attachmentCount - 1].description.view.format) &&
            !std::all_of(std::begin(color.clear_color.f32), std::end(color.clear_color.f32),
                         [](float value) { return std::isfinite(value); }))
            return VERNON_STATUS_INVALID_ARGUMENT;
    }
    const VernonRuntimeProviderDepthStencilAttachment *depth = rendering.depth_stencil_attachment;
    if (depth) {
        const VernonStatus status = describe(depth->view, VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
                                             VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
        if (status != VERNON_STATUS_OK)
            return status;
        // A depth-only view has no stencil contents to clear.
        const bool stencil =
            attachments[attachmentCount - 1].description.view.subresources.aspects & VERNON_IMAGE_ASPECT_STENCIL;
        if (!stencil && depth->stencil_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR)
            return VERNON_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < rendering.sampled_image_count; ++index) {
        const VernonRuntimeProviderResourceReference image = rendering.sampled_images[index];
        if (!image.identity || !present(image.resource))
            return VERNON_STATUS_INVALID_ARGUMENT;
        VernonRuntimeProviderImageDescription description{};
        description.struct_size = sizeof(description);
        const VernonStatus status =
            providerStatusBoundary([&] { return provider.describe_image(provider.user_data, image, &description); });
        if (status != VERNON_STATUS_OK)
            return status;
        if (!vernon::runtime::providerImageDescriptionIsCanonical(description) ||
            (description.image.usage & VERNON_IMAGE_SAMPLED) == 0)
            return VERNON_STATUS_INVALID_ARGUMENT;
        for (size_t attachment = 0; attachment < attachmentCount; ++attachment)
            if (vernon::runtime::providerImageSubresourcesOverlap(
                    vernon::runtime::providerImageSubresources(attachments[attachment].description),
                    vernon::runtime::providerImageSubresources(description)))
                return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return VERNON_STATUS_OK;
}

VernonStatus preparePipelineImpl(const VernonRuntimeDeviceProvider *provider,
                                 const VernonRuntimeCorePipelineDescriptor *descriptor,
                                 VernonRuntimeCorePipeline **output) {
    if (output)
        *output = nullptr;
    if (!provider || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        !providerSupports(*provider, descriptor->kind) || !groupLayoutsAreValid(*descriptor, *provider) ||
        !shaderStagesAreValid(*descriptor))
        return VERNON_STATUS_INVALID_ARGUMENT;

    const uint32_t requiredFacet = descriptor->kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE
                                       ? VERNON_RUNTIME_PROVIDER_COMPUTE
                                       : VERNON_RUNTIME_PROVIDER_GRAPHICS;
    uint32_t capabilities{};
    if (providerValueBoundary(capabilities, [&] { return provider->get_capabilities(provider->user_data); }) !=
        VERNON_STATUS_OK)
        return VERNON_STATUS_INTERNAL_ERROR;
    if ((descriptor->required_capabilities & requiredFacet) == 0 ||
        (capabilities & descriptor->required_capabilities) != descriptor->required_capabilities) {
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }

    auto pipeline = std::unique_ptr<VernonRuntimeCorePipeline>(new (std::nothrow) VernonRuntimeCorePipeline());
    if (!pipeline)
        return VERNON_STATUS_INTERNAL_ERROR;
    pipeline->provider = *provider;
    if (providerValueBoundary(pipeline->identity, [&] { return provider->get_device_identity(provider->user_data); }) !=
        VERNON_STATUS_OK)
        return VERNON_STATUS_INTERNAL_ERROR;
    pipeline->kind = descriptor->kind;
    pipeline->pushConstantSize = descriptor->push_constant_size;
    std::array<VernonRuntimeProviderObject, kMaxGroups> groupLayouts{};
    for (size_t index = 0; index < descriptor->group_layout_count; ++index) {
        VernonRuntimeCoreGroupLayout *layout = descriptor->group_layouts[index];
        retainGroupLayout(layout);
        pipeline->groupLayouts[pipeline->groupCount++] = layout;
        groupLayouts[index] = layout->handle;
        if (!layout->entries.empty())
            pipeline->requiredGroups |= 1u << index;
    }

    for (size_t index = 0; index < descriptor->shader_count; ++index) {
        VernonRuntimeProviderObject shader{};
        const VernonStatus status = providerStatusBoundary(
            [&] { return provider->prepare_shader(provider->user_data, &descriptor->shaders[index], &shader); });
        if (status != VERNON_STATUS_OK || !present(shader)) {
            releasePipeline(pipeline.release());
            return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
        }
        pipeline->shaders[pipeline->shaderCount++] = shader;
    }

    const VernonRuntimeProviderPipelineLayoutDescriptor layoutDescriptor{
        sizeof(VernonRuntimeProviderPipelineLayoutDescriptor),
        groupLayouts.data(),
        descriptor->group_layout_count,
        descriptor->vertex_attributes,
        descriptor->vertex_attribute_count,
        descriptor->push_constant_size,
        {0, 0, 0, 0}};
    VernonStatus status = providerStatusBoundary(
        [&] { return provider->prepare_pipeline_layout(provider->user_data, &layoutDescriptor, &pipeline->layout); });
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
    status = providerStatusBoundary(
        [&] { return provider->prepare_pipeline(provider->user_data, &providerDescriptor, &pipeline->pipeline); });
    if (status != VERNON_STATUS_OK || !present(pipeline->pipeline)) {
        releasePipeline(pipeline.release());
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }

    *output = pipeline.release();
    return VERNON_STATUS_OK;
}

VernonStatus createGroupImpl(VernonRuntimeCoreGroupLayout *layout, const VernonRuntimeProviderBindingValue *values,
                             size_t valueCount, VernonRuntimeProviderGroupLifetime lifetime,
                             VernonRuntimeProviderObject encoder, VernonRuntimeCoreGroup **output) {
    if (output)
        *output = nullptr;
    if (!layout || !output || !groupValuesMatchLayout(*layout, values, valueCount) ||
        (lifetime != VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT && lifetime != VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT) ||
        (lifetime == VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT && present(encoder)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus validated = validateImageValues(*layout, values);
    if (validated != VERNON_STATUS_OK)
        return validated;
    auto group = std::unique_ptr<VernonRuntimeCoreGroup>(new (std::nothrow) VernonRuntimeCoreGroup());
    if (!group)
        return VERNON_STATUS_INTERNAL_ERROR;
    const VernonRuntimeProviderGroupDescriptor descriptor{sizeof(descriptor), layout->handle, values,      valueCount,
                                                          lifetime,           encoder,        {0, 0, 0, 0}};
    const VernonStatus status = providerStatusBoundary(
        [&] { return layout->provider.create_group(layout->provider.user_data, &descriptor, &group->handle); });
    if (status != VERNON_STATUS_OK || !present(group->handle)) {
        if (present(group->handle))
            providerVoidBoundary([&] { layout->provider.destroy_group(layout->provider.user_data, group->handle); });
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }
    retainGroupLayout(layout);
    group->layout = layout;
    group->transient = lifetime == VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT;
    group->encoder = encoder;
    *output = group.release();
    return VERNON_STATUS_OK;
}

VernonStatus prepareGraphicsVariantImpl(VernonRuntimeCorePipeline *pipeline,
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
    {
        std::lock_guard<std::mutex> guard(pipeline->graphicsCacheMutex);
        const auto cached = pipeline->graphicsCache.find(key);
        if (cached != pipeline->graphicsCache.end()) {
            variant->pipeline = pipeline;
            variant->handle = cached->second;
            retainPipeline(pipeline);
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
    const VernonStatus status = providerStatusBoundary(
        [&] { return pipeline->provider.prepare_pipeline(pipeline->provider.user_data, &descriptor, &prepared); });
    if (status != VERNON_STATUS_OK || !present(prepared)) {
        if (present(prepared))
            providerVoidBoundary([&] { pipeline->provider.destroy_pipeline(pipeline->provider.user_data, prepared); });
        return status == VERNON_STATUS_OK ? VERNON_STATUS_INTERNAL_ERROR : status;
    }
    {
        std::lock_guard<std::mutex> guard(pipeline->graphicsCacheMutex);
        const auto [cached, inserted] = pipeline->graphicsCache.emplace(std::move(key), prepared);
        if (!inserted) {
            providerVoidBoundary([&] { pipeline->provider.destroy_pipeline(pipeline->provider.user_data, prepared); });
            prepared = cached->second;
        }
    }
    variant->pipeline = pipeline;
    variant->handle = prepared;
    retainPipeline(pipeline);
    *output = variant.release();
    return VERNON_STATUS_OK;
}

VernonStatus beginCommandsImpl(const VernonRuntimeDeviceProvider *provider, VernonRuntimeProviderObject encoder,
                               VernonRuntimeCoreCommands **output) {
    if (output)
        *output = nullptr;
    if (!provider || !output || !providerIsValid(*provider))
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto commands = std::unique_ptr<VernonRuntimeCoreCommands>(new (std::nothrow) VernonRuntimeCoreCommands());
    if (!commands)
        return VERNON_STATUS_INTERNAL_ERROR;
    commands->provider = *provider;
    commands->encoder = encoder;
    const VernonStatus status = providerStatusBoundary(
        [&] { return provider->begin_commands(provider->user_data, encoder, &commands->native); });
    if (status != VERNON_STATUS_OK)
        return status;
    if (!present(commands->native))
        return VERNON_STATUS_INTERNAL_ERROR;
    *output = commands.release();
    return VERNON_STATUS_OK;
}

void unbindGroup(VernonRuntimeCoreCommands &commands, size_t index) noexcept {
    releaseGroup(std::exchange(commands.groups[index], nullptr));
    commands.boundGroups &= ~(1u << index);
}

VernonStatus endCommandsImpl(VernonRuntimeCoreCommands *commands) {
    if (!commands)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::unique_ptr<VernonRuntimeCoreCommands> owned(commands);
    VernonStatus status = VERNON_STATUS_OK;
    if (owned->rendering) {
        status = providerStatusBoundary(
            [&] { return owned->provider.end_rendering(owned->provider.user_data, owned->native); });
        if (status == VERNON_STATUS_OK)
            status = VERNON_STATUS_INVALID_ARGUMENT;
    }
    const VernonStatus ended =
        providerStatusBoundary([&] { return owned->provider.end_commands(owned->provider.user_data, owned->native); });
    for (size_t index = 0; index < kMaxGroups; ++index)
        unbindGroup(*owned, index);
    releasePipeline(owned->pipeline);
    return status != VERNON_STATUS_OK ? status : ended;
}

VernonStatus bindNativePipeline(VernonRuntimeCoreCommands &commands, VernonRuntimeCorePipeline &pipeline,
                                VernonRuntimeProviderObject handle) {
    if (commands.bound.value == handle.value && commands.pipeline == &pipeline)
        return VERNON_STATUS_OK;
    const VernonStatus status = providerStatusBoundary(
        [&] { return commands.provider.encode_bind_pipeline(commands.provider.user_data, commands.native, handle); });
    if (status != VERNON_STATUS_OK)
        return status;
    for (size_t index = 0; index < kMaxGroups; ++index)
        if (commands.groups[index] &&
            (index >= pipeline.groupCount || commands.groups[index]->layout != pipeline.groupLayouts[index]))
            unbindGroup(commands, index);
    if (commands.pipeline != &pipeline) {
        retainPipeline(&pipeline);
        releasePipeline(std::exchange(commands.pipeline, &pipeline));
    }
    commands.bound = handle;
    return VERNON_STATUS_OK;
}

VernonStatus bindPipelineImpl(VernonRuntimeCoreCommands *commands, VernonRuntimeCorePipeline *pipeline) {
    if (!commands || !pipeline || pipeline->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE ||
        !sameProvider(commands->provider, pipeline->provider) || commands->rendering)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return bindNativePipeline(*commands, *pipeline, pipeline->pipeline);
}

VernonStatus bindGraphicsVariantImpl(VernonRuntimeCoreCommands *commands, VernonRuntimeCoreGraphicsVariant *variant) {
    if (!commands || !variant || !sameProvider(commands->provider, variant->pipeline->provider))
        return VERNON_STATUS_INVALID_ARGUMENT;
    return bindNativePipeline(*commands, *variant->pipeline, variant->handle);
}

VernonStatus bindGroupImpl(VernonRuntimeCoreCommands *commands, uint32_t index, VernonRuntimeCoreGroup *group) {
    if (!commands || !group || !commands->pipeline || index >= commands->pipeline->groupCount ||
        group->layout != commands->pipeline->groupLayouts[index] ||
        (group->transient && group->encoder.value != commands->encoder.value))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (commands->groups[index] == group)
        return VERNON_STATUS_OK;
    const VernonStatus status = providerStatusBoundary([&] {
        return commands->provider.encode_bind_group(commands->provider.user_data, commands->native, index,
                                                    group->handle);
    });
    if (status != VERNON_STATUS_OK)
        return status;
    retainGroup(group);
    releaseGroup(std::exchange(commands->groups[index], group));
    commands->boundGroups |= 1u << index;
    return VERNON_STATUS_OK;
}

bool groupsAreComplete(const VernonRuntimeCoreCommands &commands) {
    return (commands.boundGroups & commands.pipeline->requiredGroups) == commands.pipeline->requiredGroups;
}

VernonStatus dispatchImpl(VernonRuntimeCoreCommands *commands, const uint32_t groupCount[3], const void *pushConstants,
                          size_t pushConstantSize) {
    if (!commands || !commands->pipeline || commands->pipeline->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE ||
        !groupCount || groupCount[0] == 0 || groupCount[1] == 0 || groupCount[2] == 0 ||
        pushConstantSize > commands->pipeline->pushConstantSize || (pushConstantSize != 0 && !pushConstants) ||
        !groupsAreComplete(*commands))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonRuntimeProviderDispatchDescriptor descriptor{sizeof(VernonRuntimeProviderDispatchDescriptor),
                                                             {groupCount[0], groupCount[1], groupCount[2]},
                                                             pushConstants,
                                                             pushConstantSize,
                                                             {0, 0, 0, 0}};
    return providerStatusBoundary([&] {
        return commands->provider.encode_dispatch(commands->provider.user_data, commands->native, &descriptor);
    });
}

VernonStatus beginRenderingImpl(VernonRuntimeCoreCommands *commands,
                                const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    if (!commands || commands->rendering ||
        !providerSupports(commands->provider, VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE) ||
        !renderingDescriptorIsValid(descriptor))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus attachmentStatus = validateRenderingAttachments(commands->provider, *descriptor);
    if (attachmentStatus != VERNON_STATUS_OK)
        return attachmentStatus;
    const VernonStatus status = providerStatusBoundary(
        [&] { return commands->provider.begin_rendering(commands->provider.user_data, commands->native, descriptor); });
    commands->rendering = status == VERNON_STATUS_OK;
    return status;
}

VernonStatus endRenderingImpl(VernonRuntimeCoreCommands *commands) {
    if (!commands || !commands->rendering)
        return VERNON_STATUS_INVALID_ARGUMENT;
    commands->rendering = false;
    return providerStatusBoundary(
        [&] { return commands->provider.end_rendering(commands->provider.user_data, commands->native); });
}

VernonStatus drawImpl(VernonRuntimeCoreCommands *commands, const VernonRuntimeCoreDrawInvocation *invocation) {
    if (!commands || !commands->rendering || !commands->pipeline ||
        commands->pipeline->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || !drawInvocationIsValid(invocation) ||
        !groupsAreComplete(*commands))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonRuntimeProviderDrawDescriptor descriptor{
        sizeof(VernonRuntimeProviderDrawDescriptor),
        invocation->vertex_count,
        invocation->instance_count,
        invocation->first_instance,
        invocation->base_vertex,
        {invocation->viewport[0], invocation->viewport[1], invocation->viewport[2], invocation->viewport[3]},
        {invocation->scissor[0], invocation->scissor[1], invocation->scissor[2], invocation->scissor[3]},
        invocation->topology,
        invocation->index_buffer,
        invocation->index_count,
        invocation->index_type,
        invocation->stencil_reference};
    return providerStatusBoundary(
        [&] { return commands->provider.encode_draw(commands->provider.user_data, commands->native, &descriptor); });
}

} // namespace

extern "C" VernonStatus vernonRuntimeCorePrepareGroupLayout(const VernonRuntimeDeviceProvider *provider,
                                                            const VernonRuntimeProviderBindingLayoutEntry *entries,
                                                            size_t entryCount, VernonRuntimeCoreGroupLayout **output) {
    return runtimeCoreBoundary([&] { return prepareGroupLayoutImpl(provider, entries, entryCount, output); });
}

extern "C" void vernonRuntimeCoreGroupLayoutDestroy(VernonRuntimeCoreGroupLayout *layout) {
    runtimeCoreVoidBoundary([&] { releaseGroupLayout(layout); });
}

extern "C" VernonStatus vernonRuntimeCorePreparePipeline(const VernonRuntimeDeviceProvider *provider,
                                                         const VernonRuntimeCorePipelineDescriptor *descriptor,
                                                         VernonRuntimeCorePipeline **output) {
    return runtimeCoreBoundary([&] { return preparePipelineImpl(provider, descriptor, output); });
}

extern "C" void vernonRuntimeCorePipelineDestroy(VernonRuntimeCorePipeline *pipeline) {
    runtimeCoreVoidBoundary([&] { releasePipeline(pipeline); });
}

extern "C" VernonRuntimeProviderDeviceIdentity
vernonRuntimeCorePipelineGetDeviceIdentity(const VernonRuntimeCorePipeline *pipeline) {
    return runtimeCoreValueBoundary(VernonRuntimeProviderDeviceIdentity{}, [&] {
        return pipeline ? pipeline->identity : VernonRuntimeProviderDeviceIdentity{};
    });
}

extern "C" VernonStatus vernonRuntimeCoreCreateGroup(VernonRuntimeCoreGroupLayout *layout,
                                                     const VernonRuntimeProviderBindingValue *values, size_t valueCount,
                                                     VernonRuntimeProviderGroupLifetime lifetime,
                                                     VernonRuntimeProviderObject encoder,
                                                     VernonRuntimeCoreGroup **output) {
    return runtimeCoreBoundary([&] { return createGroupImpl(layout, values, valueCount, lifetime, encoder, output); });
}

extern "C" void vernonRuntimeCoreGroupDestroy(VernonRuntimeCoreGroup *group) {
    runtimeCoreVoidBoundary([&] { releaseGroup(group); });
}

extern "C" VernonStatus
vernonRuntimeCorePrepareGraphicsVariant(VernonRuntimeCorePipeline *pipeline,
                                        const VernonRuntimeCoreGraphicsCompatibility *compatibility,
                                        VernonRuntimeCoreGraphicsVariant **output) {
    return runtimeCoreBoundary([&] { return prepareGraphicsVariantImpl(pipeline, compatibility, output); });
}

extern "C" void vernonRuntimeCoreGraphicsVariantDestroy(VernonRuntimeCoreGraphicsVariant *variant) {
    runtimeCoreVoidBoundary([&] {
        if (!variant)
            return;
        VernonRuntimeCorePipeline *pipeline = variant->pipeline;
        delete variant;
        releasePipeline(pipeline);
    });
}

extern "C" VernonStatus vernonRuntimeCoreBeginCommands(const VernonRuntimeDeviceProvider *provider,
                                                       VernonRuntimeProviderObject encoder,
                                                       VernonRuntimeCoreCommands **output) {
    return runtimeCoreBoundary([&] { return beginCommandsImpl(provider, encoder, output); });
}

extern "C" VernonStatus vernonRuntimeCoreEndCommands(VernonRuntimeCoreCommands *commands) {
    return runtimeCoreBoundary([&] { return endCommandsImpl(commands); });
}

extern "C" VernonStatus vernonRuntimeCoreBindPipeline(VernonRuntimeCoreCommands *commands,
                                                      VernonRuntimeCorePipeline *pipeline) {
    return runtimeCoreBoundary([&] { return bindPipelineImpl(commands, pipeline); });
}

extern "C" VernonStatus vernonRuntimeCoreBindGraphicsVariant(VernonRuntimeCoreCommands *commands,
                                                             VernonRuntimeCoreGraphicsVariant *variant) {
    return runtimeCoreBoundary([&] { return bindGraphicsVariantImpl(commands, variant); });
}

extern "C" VernonStatus vernonRuntimeCoreBindGroup(VernonRuntimeCoreCommands *commands, uint32_t index,
                                                   VernonRuntimeCoreGroup *group) {
    return runtimeCoreBoundary([&] { return bindGroupImpl(commands, index, group); });
}

extern "C" VernonStatus vernonRuntimeCoreDispatch(VernonRuntimeCoreCommands *commands, const uint32_t groupCount[3],
                                                  const void *pushConstants, size_t pushConstantSize) {
    return runtimeCoreBoundary([&] { return dispatchImpl(commands, groupCount, pushConstants, pushConstantSize); });
}

extern "C" VernonStatus vernonRuntimeCoreBeginRendering(VernonRuntimeCoreCommands *commands,
                                                        const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    return runtimeCoreBoundary([&] { return beginRenderingImpl(commands, descriptor); });
}

extern "C" VernonStatus vernonRuntimeCoreEndRendering(VernonRuntimeCoreCommands *commands) {
    return runtimeCoreBoundary([&] { return endRenderingImpl(commands); });
}

extern "C" VernonStatus vernonRuntimeCoreDraw(VernonRuntimeCoreCommands *commands,
                                              const VernonRuntimeCoreDrawInvocation *invocation) {
    return runtimeCoreBoundary([&] { return drawImpl(commands, invocation); });
}
