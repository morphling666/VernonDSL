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
constexpr size_t kMaxVertexBindings = VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS;

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
    // Bit b is set for vertex binding b, so a draw needs a vertex buffer bound at b.
    uint32_t vertexBindings{};
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
    bool updatable{};
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
};

namespace {

// Finds or inserts a table index by a non-zero 64-bit hash, with `equal(index)` telling whether the entry at an index
// with that hash is the one sought. Clearing keeps the slots' storage.
class IndexTable {
public:
    template <typename Equal> std::pair<uint32_t, bool> insert(uint64_t hash, uint32_t next, Equal &&equal) {
        if ((count_ + 1) * 2 > slots_.size())
            grow();
        const size_t mask = slots_.size() - 1;
        for (size_t probe = mix(hash) & mask;; probe = (probe + 1) & mask) {
            Slot &slot = slots_[probe];
            if (!slot.hash) {
                slot = {hash, next};
                ++count_;
                return {next, true};
            }
            if (slot.hash == hash && equal(slot.index))
                return {slot.index, false};
        }
    }

    void clear() {
        if (count_)
            std::fill(slots_.begin(), slots_.end(), Slot{});
        count_ = 0;
    }

private:
    struct Slot {
        uint64_t hash{};
        uint32_t index{};
    };

    static size_t mix(uint64_t hash) {
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdull;
        return static_cast<size_t>(hash ^ (hash >> 33));
    }

    void grow() {
        std::vector<Slot> previous(std::max<size_t>(slots_.size() * 2, 64));
        previous.swap(slots_);
        const size_t mask = slots_.size() - 1;
        for (const Slot &slot : previous) {
            if (!slot.hash)
                continue;
            size_t probe = mix(slot.hash) & mask;
            while (slots_[probe].hash)
                probe = (probe + 1) & mask;
            slots_[probe] = slot;
        }
    }

    std::vector<Slot> slots_;
    size_t count_{};
};

// A rendering scope of a recording; the descriptor's pointers are filled when a batch is created.
struct PacketRendering {
    VernonRuntimeProviderRenderingDescriptor descriptor{};
    std::array<VernonRuntimeProviderColorAttachment, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colors{};
    VernonRuntimeProviderDepthStencilAttachment depth{};
    bool hasDepth{};
    // The scope's sampled images in VernonRuntimeCorePackets::sampledImages.
    size_t firstSampledImage{};
};

enum class PayloadSiteKind : uint8_t { Draw, DrawIndexed, Dispatch, Clear };

// Payload words a command wrote, with the limits EncodePackets checks them against.
struct PayloadSite {
    PayloadSiteKind kind{};
    uint32_t payload{};
    // DrawIndexed: the index buffer's capacity in indices. Clear: the color attachment count.
    uint32_t limit{};
    // Clear: bit c is set when color attachment c has an integer format, whose clear values are raw bits.
    uint32_t integerColors{};
};

constexpr uint32_t packetHeader(VernonRuntimeProviderPacketOpcode opcode, size_t words) {
    return static_cast<uint32_t>(opcode) | static_cast<uint32_t>(words << 16);
}

constexpr size_t kDrawPayloadWords = sizeof(VernonRuntimeProviderDrawPayload) / sizeof(uint32_t);
constexpr size_t kMaxPacketWords = 0xffff;

} // namespace

struct VernonRuntimeCorePackets {
    VernonRuntimeDeviceProvider provider{};
    std::vector<uint32_t> words;
    std::vector<uint32_t> payload;
    std::vector<VernonRuntimeProviderObject> objects;
    IndexTable objectIndices;
    // Referenced by the recording: the owner of each pipeline object and each group object.
    std::vector<VernonRuntimeCorePipeline *> pipelines;
    std::vector<VernonRuntimeCoreGroup *> groups;
    std::vector<VernonRuntimeProviderResourceReference> resources;
    IndexTable resourceIndices;
    std::vector<PacketRendering> renderings;
    std::vector<VernonRuntimeProviderResourceReference> sampledImages;
    std::vector<PayloadSite> sites;
    // The command encoder of the transient groups the recording binds, which is the only one it is encodable on.
    VernonRuntimeProviderObject transientEncoder{};
    // The binding state after the last recorded command.
    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeProviderObject bound{};
    std::array<VernonRuntimeCoreGroup *, kMaxGroups> boundGroups{};
    uint32_t boundGroupMask{};
    std::array<VernonRuntimeProviderResourceReference, kMaxVertexBindings> vertexBuffers{};
    uint32_t boundVertexBuffers{};
    bool rendering{};
    uint32_t debugDepth{};
    // The depth when the rendering scope began; groups above it close before the scope ends.
    uint32_t renderingDebugDepth{};
};

struct VernonRuntimeCorePacketBatch {
    VernonRuntimeDeviceProvider provider{};
    VernonRuntimeProviderObject handle{};
    // Referenced by the batch, whose provider objects belong to them.
    std::vector<VernonRuntimeCorePipeline *> pipelines;
    std::vector<VernonRuntimeCoreGroup *> groups;
    std::vector<PayloadSite> sites;
    size_t payloadWords{};
    VernonRuntimeProviderObject transientEncoder{};
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
           provider.create_packet_batch && provider.begin_commands && provider.end_commands &&
           provider.resume_commands && provider.encode_packets && provider.destroy_shader &&
           provider.destroy_group_layout && provider.destroy_pipeline_layout && provider.destroy_pipeline &&
           provider.destroy_group && provider.destroy_packet_batch;
}

bool providerSupports(const VernonRuntimeDeviceProvider &provider, VernonRuntimeProviderPipelineKind kind) {
    if (!providerIsValid(provider))
        return false;
    return kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE || provider.describe_image != nullptr;
}

// Bindings are distinct and below the limit, and every attribute reads one of them; compute pipelines have none.
bool vertexInputsAreValid(const VernonRuntimeCorePipelineDescriptor &descriptor, uint32_t &bindings) {
    bindings = 0;
    if ((descriptor.vertex_binding_count != 0 && !descriptor.vertex_bindings) ||
        (descriptor.vertex_attribute_count != 0 && !descriptor.vertex_attributes) ||
        (descriptor.kind == VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE &&
         (descriptor.vertex_binding_count != 0 || descriptor.vertex_attribute_count != 0)))
        return false;
    for (size_t index = 0; index < descriptor.vertex_binding_count; ++index) {
        const uint32_t binding = descriptor.vertex_bindings[index].binding;
        if (binding >= kMaxVertexBindings || (bindings & (1u << binding)) != 0)
            return false;
        bindings |= 1u << binding;
    }
    for (size_t index = 0; index < descriptor.vertex_attribute_count; ++index) {
        const uint32_t binding = descriptor.vertex_attributes[index].binding;
        if (binding >= kMaxVertexBindings || (bindings & (1u << binding)) == 0)
            return false;
    }
    return true;
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
           left.element_alignment == right.element_alignment && left.numeric_type == right.numeric_type &&
           left.image_dimension == right.image_dimension && left.storage_image_format == right.storage_image_format &&
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
// render area; sampled images have sampled usage and overlap no attachment. Bit c of integerColors receives whether
// color attachment c has an integer format.
VernonStatus validateRenderingAttachments(const VernonRuntimeDeviceProvider &provider,
                                          const VernonRuntimeProviderRenderingDescriptor &rendering,
                                          uint32_t &integerColors) {
    integerColors = 0;
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
        if (vernon::formats::isInteger(attachments[attachmentCount - 1].description.view.format))
            integerColors |= 1u << index;
        else if (!std::all_of(std::begin(color.clear_color.f32), std::end(color.clear_color.f32),
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
    uint32_t vertexBindings = 0;
    if (!provider || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        !providerSupports(*provider, descriptor->kind) || !groupLayoutsAreValid(*descriptor, *provider) ||
        !shaderStagesAreValid(*descriptor) || !vertexInputsAreValid(*descriptor, vertexBindings))
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
    pipeline->vertexBindings = vertexBindings;
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
        descriptor->vertex_bindings,
        descriptor->vertex_binding_count,
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
        (lifetime != VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT && lifetime != VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT &&
         lifetime != VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE) ||
        (lifetime != VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT && present(encoder)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (lifetime == VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE && !layout->provider.update_group)
        return VERNON_STATUS_UNSUPPORTED_TARGET;
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
    group->updatable = lifetime == VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE;
    if (group->transient)
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
    for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
        if ((pipeline->vertexBindings & (1u << binding)) != 0 &&
            (binding >= compatibility->vertex_stride_count || compatibility->vertex_strides[binding] == 0))
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

VernonStatus endCommandsImpl(VernonRuntimeCoreCommands *commands) {
    if (!commands)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::unique_ptr<VernonRuntimeCoreCommands> owned(commands);
    return providerStatusBoundary(
        [&] { return owned->provider.end_commands(owned->provider.user_data, owned->native); });
}

VernonStatus resumeCommandsImpl(VernonRuntimeCoreCommands *commands) {
    if (!commands)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return providerStatusBoundary(
        [&] { return commands->provider.resume_commands(commands->provider.user_data, commands->native); });
}

VernonStatus updateGroupImpl(VernonRuntimeCoreCommands *commands, VernonRuntimeCoreGroup *group,
                             const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (!commands || !group || !group->updatable || !sameProvider(commands->provider, group->layout->provider) ||
        !groupValuesMatchLayout(*group->layout, values, valueCount))
        return VERNON_STATUS_INVALID_ARGUMENT;
    return providerStatusBoundary([&] {
        return commands->provider.update_group(commands->provider.user_data, commands->native, group->handle, values,
                                               valueCount);
    });
}

VernonStatus createPacketsImpl(const VernonRuntimeDeviceProvider *provider, VernonRuntimeCorePackets **output) {
    if (output)
        *output = nullptr;
    if (!provider || !output || !providerIsValid(*provider))
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto packets = std::unique_ptr<VernonRuntimeCorePackets>(new (std::nothrow) VernonRuntimeCorePackets());
    if (!packets)
        return VERNON_STATUS_INTERNAL_ERROR;
    packets->provider = *provider;
    *output = packets.release();
    return VERNON_STATUS_OK;
}

void resetPackets(VernonRuntimeCorePackets &packets) noexcept {
    for (VernonRuntimeCorePipeline *pipeline : packets.pipelines)
        releasePipeline(pipeline);
    for (VernonRuntimeCoreGroup *group : packets.groups)
        releaseGroup(group);
    packets.pipelines.clear();
    packets.groups.clear();
    packets.words.clear();
    packets.payload.clear();
    packets.objects.clear();
    packets.objectIndices.clear();
    packets.resources.clear();
    packets.resourceIndices.clear();
    packets.renderings.clear();
    packets.sampledImages.clear();
    packets.sites.clear();
    packets.transientEncoder = {};
    packets.pipeline = nullptr;
    packets.bound = {};
    packets.boundGroups = {};
    packets.boundGroupMask = 0;
    packets.boundVertexBuffers = 0;
    packets.rendering = false;
    packets.debugDepth = 0;
    packets.renderingDebugDepth = 0;
}

// The object table index of `handle`, whose owner the recording references from its first entry on.
template <typename Owner>
uint32_t packetObject(VernonRuntimeCorePackets &packets, VernonRuntimeProviderObject handle, Owner *owner,
                      std::vector<Owner *> &owners, void (*retain)(Owner *) noexcept) {
    const auto next = static_cast<uint32_t>(packets.objects.size());
    const auto [index, inserted] = packets.objectIndices.insert(
        handle.value, next, [&](uint32_t candidate) { return packets.objects[candidate].value == handle.value; });
    if (inserted) {
        packets.objects.push_back(handle);
        owners.push_back(owner);
        retain(owner);
    }
    return index;
}

bool sameReference(const VernonRuntimeProviderResourceReference &left,
                   const VernonRuntimeProviderResourceReference &right) {
    return left.identity == right.identity && left.resource.value == right.resource.value &&
           left.offset == right.offset && left.size == right.size;
}

uint32_t packetResource(VernonRuntimeCorePackets &packets, const VernonRuntimeProviderResourceReference &resource) {
    uint64_t hash = resource.resource.value * 0x9e3779b97f4a7c15ull;
    hash = (hash ^ resource.identity) * 0x100000001b3ull;
    hash = (hash ^ resource.offset) * 0x100000001b3ull;
    hash = ((hash ^ resource.size) * 0x100000001b3ull) | 1;
    const auto next = static_cast<uint32_t>(packets.resources.size());
    const auto [index, inserted] = packets.resourceIndices.insert(
        hash, next, [&](uint32_t candidate) { return sameReference(packets.resources[candidate], resource); });
    if (inserted)
        packets.resources.push_back(resource);
    return index;
}

void unbindGroup(VernonRuntimeCorePackets &packets, size_t index) noexcept {
    packets.boundGroups[index] = nullptr;
    packets.boundGroupMask &= ~(1u << index);
}

VernonStatus bindPacketPipeline(VernonRuntimeCorePackets &packets, VernonRuntimeCorePipeline &pipeline,
                                VernonRuntimeProviderObject handle) {
    if (packets.bound.value == handle.value && packets.pipeline == &pipeline)
        return VERNON_STATUS_OK;
    const uint32_t object = packetObject(packets, handle, &pipeline, packets.pipelines, retainPipeline);
    packets.words.insert(packets.words.end(), {packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_BIND_PIPELINE, 2), object});
    for (size_t index = 0; index < kMaxGroups; ++index)
        if (packets.boundGroups[index] &&
            (index >= pipeline.groupCount || packets.boundGroups[index]->layout != pipeline.groupLayouts[index]))
            unbindGroup(packets, index);
    packets.pipeline = &pipeline;
    packets.bound = handle;
    return VERNON_STATUS_OK;
}

VernonStatus bindPacketsPipelineImpl(VernonRuntimeCorePackets *packets, VernonRuntimeCorePipeline *pipeline) {
    if (!packets || !pipeline || pipeline->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE ||
        !sameProvider(packets->provider, pipeline->provider) || packets->rendering)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return bindPacketPipeline(*packets, *pipeline, pipeline->pipeline);
}

VernonStatus bindPacketsGraphicsVariantImpl(VernonRuntimeCorePackets *packets,
                                            VernonRuntimeCoreGraphicsVariant *variant) {
    if (!packets || !variant || !sameProvider(packets->provider, variant->pipeline->provider))
        return VERNON_STATUS_INVALID_ARGUMENT;
    return bindPacketPipeline(*packets, *variant->pipeline, variant->handle);
}

VernonStatus bindPacketsGroupImpl(VernonRuntimeCorePackets *packets, uint32_t index, VernonRuntimeCoreGroup *group) {
    if (!packets || !group || !packets->pipeline || index >= packets->pipeline->groupCount ||
        group->layout != packets->pipeline->groupLayouts[index] ||
        (group->transient && present(packets->transientEncoder) &&
         packets->transientEncoder.value != group->encoder.value))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (packets->boundGroups[index] == group)
        return VERNON_STATUS_OK;
    const uint32_t object = packetObject(*packets, group->handle, group, packets->groups, retainGroup);
    packets->words.insert(packets->words.end(),
                          {packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_BIND_GROUP, 3), index, object});
    if (group->transient)
        packets->transientEncoder = group->encoder;
    packets->boundGroups[index] = group;
    packets->boundGroupMask |= 1u << index;
    return VERNON_STATUS_OK;
}

// Records only the run between the first and the last binding that changes.
VernonStatus bindPacketsVertexBuffersImpl(VernonRuntimeCorePackets *packets, uint32_t first,
                                          const VernonRuntimeProviderResourceReference *buffers, size_t count) {
    if (!packets || (count != 0 && !buffers) || first > kMaxVertexBindings || count > kMaxVertexBindings - first)
        return VERNON_STATUS_INVALID_ARGUMENT;
    size_t begin = count;
    size_t end = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!present(buffers[index].resource))
            return VERNON_STATUS_INVALID_ARGUMENT;
        const size_t binding = first + index;
        if ((packets->boundVertexBuffers & (1u << binding)) == 0 ||
            !sameReference(packets->vertexBuffers[binding], buffers[index])) {
            begin = std::min(begin, index);
            end = index + 1;
        }
    }
    if (begin >= end)
        return VERNON_STATUS_OK;
    packets->words.push_back(packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_BIND_VERTEX_BUFFERS, 2 + end - begin));
    packets->words.push_back(static_cast<uint32_t>(first + begin));
    for (size_t index = begin; index < end; ++index) {
        packets->words.push_back(packetResource(*packets, buffers[index]));
        packets->vertexBuffers[first + index] = buffers[index];
        packets->boundVertexBuffers |= 1u << (first + index);
    }
    return VERNON_STATUS_OK;
}

bool groupsAreComplete(const VernonRuntimeCorePackets &packets) {
    return (packets.boundGroupMask & packets.pipeline->requiredGroups) == packets.pipeline->requiredGroups;
}

uint32_t appendPayload(VernonRuntimeCorePackets &packets, const uint32_t *words, size_t count) {
    const auto index = static_cast<uint32_t>(packets.payload.size());
    packets.payload.insert(packets.payload.end(), words, words + count);
    return index;
}

VernonStatus packetsDispatchImpl(VernonRuntimeCorePackets *packets, const uint32_t groupCount[3]) {
    if (!packets || !packets->pipeline || packets->pipeline->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE ||
        packets->rendering || !groupCount || groupCount[0] == 0 || groupCount[1] == 0 || groupCount[2] == 0 ||
        !groupsAreComplete(*packets))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const uint32_t payload = appendPayload(*packets, groupCount, 3);
    packets->sites.push_back({PayloadSiteKind::Dispatch, payload, 0, 0});
    packets->words.insert(packets->words.end(), {packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_DISPATCH, 2), payload});
    return VERNON_STATUS_OK;
}

uint32_t bitsOf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float floatOf(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

VernonStatus packetsBeginRenderingImpl(VernonRuntimeCorePackets *packets,
                                       const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    if (!packets || packets->rendering ||
        !providerSupports(packets->provider, VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE) ||
        !renderingDescriptorIsValid(descriptor))
        return VERNON_STATUS_INVALID_ARGUMENT;
    uint32_t integerColors = 0;
    const VernonStatus attachmentStatus = validateRenderingAttachments(packets->provider, *descriptor, integerColors);
    if (attachmentStatus != VERNON_STATUS_OK)
        return attachmentStatus;
    PacketRendering rendering;
    rendering.descriptor = *descriptor;
    std::copy_n(descriptor->color_attachments, descriptor->color_attachment_count, rendering.colors.begin());
    if (descriptor->depth_stencil_attachment) {
        rendering.depth = *descriptor->depth_stencil_attachment;
        rendering.hasDepth = true;
    }
    rendering.firstSampledImage = packets->sampledImages.size();
    packets->sampledImages.insert(packets->sampledImages.end(), descriptor->sampled_images,
                                  descriptor->sampled_images + descriptor->sampled_image_count);
    const auto renderingIndex = static_cast<uint32_t>(packets->renderings.size());
    packets->renderings.push_back(rendering);
    const auto payload = static_cast<uint32_t>(packets->payload.size());
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
        appendPayload(*packets, descriptor->color_attachments[index].clear_color.u32, 4);
    const VernonRuntimeProviderDepthStencilAttachment *depth = descriptor->depth_stencil_attachment;
    const uint32_t depthStencil[2]{bitsOf(depth ? depth->clear_depth : 0.0f), depth ? depth->clear_stencil : 0};
    appendPayload(*packets, depthStencil, 2);
    packets->sites.push_back(
        {PayloadSiteKind::Clear, payload, static_cast<uint32_t>(descriptor->color_attachment_count), integerColors});
    packets->words.insert(packets->words.end(),
                          {packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_BEGIN_RENDERING, 3), renderingIndex, payload});
    packets->rendering = true;
    packets->renderingDebugDepth = packets->debugDepth;
    return VERNON_STATUS_OK;
}

VernonStatus packetsEndRenderingImpl(VernonRuntimeCorePackets *packets) {
    if (!packets || !packets->rendering || packets->debugDepth != packets->renderingDebugDepth)
        return VERNON_STATUS_INVALID_ARGUMENT;
    packets->words.push_back(packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_END_RENDERING, 1));
    packets->rendering = false;
    return VERNON_STATUS_OK;
}

// The checks that EncodePackets repeats on payload words, against the index buffer's capacity of an indexed draw.
bool drawPayloadIsValid(const VernonRuntimeProviderDrawPayload &draw, bool indexed, uint64_t indexCapacity) {
    if (draw.vertex_count == 0 || draw.stencil_reference > 0xff)
        return false;
    return indexed ? uint64_t{draw.first_index} + draw.vertex_count <= indexCapacity
                   : draw.first_index == 0 && draw.base_vertex == 0;
}

VernonStatus drawPayloadImpl(const VernonRuntimeCoreDrawInvocation *invocation,
                             VernonRuntimeProviderDrawPayload *payload) {
    if (!invocation || !payload || invocation->struct_size < sizeof(VernonRuntimeCoreDrawInvocation) ||
        invocation->topology > 2)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const bool indexed = invocation->index_count != 0;
    if (indexed != present(invocation->index_buffer.resource) || (indexed && invocation->index_type != 0))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonRuntimeProviderDrawPayload draw{};
    draw.vertex_count = indexed ? invocation->index_count : invocation->vertex_count;
    draw.instance_count = invocation->instance_count;
    draw.first_instance = invocation->first_instance;
    draw.base_vertex = indexed ? invocation->base_vertex : 0;
    draw.first_index = indexed ? invocation->first_index : 0;
    std::copy_n(invocation->viewport, 4, draw.viewport);
    std::copy_n(invocation->scissor, 4, draw.scissor);
    draw.stencil_reference = invocation->stencil_reference;
    if (!drawPayloadIsValid(draw, indexed, invocation->index_buffer.size / sizeof(uint32_t)) ||
        (!indexed && (invocation->base_vertex != 0 || invocation->first_index != 0)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    *payload = draw;
    return VERNON_STATUS_OK;
}

VernonStatus packetsDrawImpl(VernonRuntimeCorePackets *packets, const VernonRuntimeCoreDrawInvocation *invocation) {
    if (!packets || !packets->rendering || !packets->pipeline ||
        packets->pipeline->kind != VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE || !groupsAreComplete(*packets) ||
        (packets->boundVertexBuffers & packets->pipeline->vertexBindings) != packets->pipeline->vertexBindings)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonRuntimeProviderDrawPayload draw{};
    if (const VernonStatus status = drawPayloadImpl(invocation, &draw); status != VERNON_STATUS_OK)
        return status;
    const bool indexed = invocation->index_count != 0;
    const uint64_t indexCapacity = invocation->index_buffer.size / sizeof(uint32_t);
    uint32_t words[kDrawPayloadWords];
    std::memcpy(words, &draw, sizeof(words));
    const uint32_t payload = appendPayload(*packets, words, kDrawPayloadWords);
    if (indexed) {
        const uint32_t index = packetResource(*packets, invocation->index_buffer);
        packets->sites.push_back({PayloadSiteKind::DrawIndexed, payload,
                                  static_cast<uint32_t>(std::min<uint64_t>(indexCapacity, UINT32_MAX)), 0});
        packets->words.insert(packets->words.end(), {packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_DRAW_INDEXED, 4),
                                                     invocation->topology, index, payload});
    } else {
        packets->sites.push_back({PayloadSiteKind::Draw, payload, 0, 0});
        packets->words.insert(packets->words.end(),
                              {packetHeader(VERNON_RUNTIME_PROVIDER_PACKET_DRAW, 3), invocation->topology, payload});
    }
    return VERNON_STATUS_OK;
}

VernonStatus packetsDebugImpl(VernonRuntimeCorePackets *packets, VernonRuntimeProviderPacketOpcode opcode,
                              VernonStringView label) {
    if (!packets || (label.size != 0 && !label.data))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (opcode == VERNON_RUNTIME_PROVIDER_PACKET_POP_DEBUG_GROUP) {
        if (packets->debugDepth <= (packets->rendering ? packets->renderingDebugDepth : 0))
            return VERNON_STATUS_INVALID_ARGUMENT;
        packets->words.push_back(packetHeader(opcode, 1));
        --packets->debugDepth;
        return VERNON_STATUS_OK;
    }
    const size_t labelWords = (label.size + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    if (2 + labelWords > kMaxPacketWords)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const size_t start = packets->words.size();
    packets->words.resize(start + 2 + labelWords, 0);
    packets->words[start] = packetHeader(opcode, 2 + labelWords);
    packets->words[start + 1] = static_cast<uint32_t>(label.size);
    if (label.size)
        std::memcpy(packets->words.data() + start + 2, label.data, label.size);
    if (opcode == VERNON_RUNTIME_PROVIDER_PACKET_PUSH_DEBUG_GROUP)
        ++packets->debugDepth;
    return VERNON_STATUS_OK;
}

void releasePacketBatch(VernonRuntimeCorePacketBatch *batch) noexcept {
    if (!batch)
        return;
    if (present(batch->handle))
        providerVoidBoundary([&] { batch->provider.destroy_packet_batch(batch->provider.user_data, batch->handle); });
    for (VernonRuntimeCorePipeline *pipeline : batch->pipelines)
        releasePipeline(pipeline);
    for (VernonRuntimeCoreGroup *group : batch->groups)
        releaseGroup(group);
    delete batch;
}

VernonStatus createPacketBatchImpl(const VernonRuntimeCorePackets *packets, VernonRuntimeCorePacketBatch **output) {
    if (output)
        *output = nullptr;
    if (!packets || !output || packets->rendering || packets->debugDepth != 0)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::vector<VernonRuntimeProviderRenderingDescriptor> renderings;
    renderings.reserve(packets->renderings.size());
    for (const PacketRendering &rendering : packets->renderings) {
        VernonRuntimeProviderRenderingDescriptor descriptor = rendering.descriptor;
        descriptor.color_attachments = descriptor.color_attachment_count ? rendering.colors.data() : nullptr;
        descriptor.depth_stencil_attachment = rendering.hasDepth ? &rendering.depth : nullptr;
        descriptor.sampled_images =
            descriptor.sampled_image_count ? packets->sampledImages.data() + rendering.firstSampledImage : nullptr;
        renderings.push_back(descriptor);
    }
    auto batch = std::unique_ptr<VernonRuntimeCorePacketBatch, void (*)(VernonRuntimeCorePacketBatch *)>(
        new (std::nothrow) VernonRuntimeCorePacketBatch(), releasePacketBatch);
    if (!batch)
        return VERNON_STATUS_INTERNAL_ERROR;
    batch->provider = packets->provider;
    batch->sites = packets->sites;
    batch->payloadWords = packets->payload.size();
    batch->transientEncoder = packets->transientEncoder;
    batch->pipelines.reserve(packets->pipelines.size());
    batch->groups.reserve(packets->groups.size());
    for (VernonRuntimeCorePipeline *pipeline : packets->pipelines) {
        retainPipeline(pipeline);
        batch->pipelines.push_back(pipeline);
    }
    for (VernonRuntimeCoreGroup *group : packets->groups) {
        retainGroup(group);
        batch->groups.push_back(group);
    }
    const VernonRuntimeProviderPacketBatchDescriptor descriptor{sizeof(descriptor),
                                                                packets->words.data(),
                                                                packets->words.size(),
                                                                packets->objects.data(),
                                                                packets->objects.size(),
                                                                packets->resources.data(),
                                                                packets->resources.size(),
                                                                renderings.data(),
                                                                renderings.size(),
                                                                packets->payload.size(),
                                                                {0, 0, 0, 0}};
    const VernonStatus status = providerStatusBoundary([&] {
        return packets->provider.create_packet_batch(packets->provider.user_data, &descriptor, &batch->handle);
    });
    if (status != VERNON_STATUS_OK)
        return status;
    if (!present(batch->handle))
        return VERNON_STATUS_INTERNAL_ERROR;
    *output = batch.release();
    return VERNON_STATUS_OK;
}

bool payloadSiteIsValid(const PayloadSite &site, const uint32_t *payload) {
    const uint32_t *words = payload + site.payload;
    switch (site.kind) {
    case PayloadSiteKind::Draw:
    case PayloadSiteKind::DrawIndexed: {
        VernonRuntimeProviderDrawPayload draw;
        std::memcpy(&draw, words, sizeof(draw));
        return drawPayloadIsValid(draw, site.kind == PayloadSiteKind::DrawIndexed, site.limit);
    }
    case PayloadSiteKind::Dispatch:
        return words[0] != 0 && words[1] != 0 && words[2] != 0;
    case PayloadSiteKind::Clear: {
        for (uint32_t color = 0; color < site.limit; ++color)
            if ((site.integerColors & (1u << color)) == 0)
                for (uint32_t component = 0; component < 4; ++component)
                    if (!std::isfinite(floatOf(words[color * 4 + component])))
                        return false;
        const float depth = floatOf(words[site.limit * 4]);
        return std::isfinite(depth) && depth >= 0.0f && depth <= 1.0f && words[site.limit * 4 + 1] <= 0xff;
    }
    }
    return false;
}

VernonStatus encodePacketsImpl(VernonRuntimeCoreCommands *commands, const VernonRuntimeCorePacketBatch *batch,
                               const uint32_t *payload, size_t payloadWordCount) {
    if (!commands || !batch || !sameProvider(commands->provider, batch->provider) ||
        payloadWordCount < batch->payloadWords || (payloadWordCount != 0 && !payload) ||
        (present(batch->transientEncoder) && batch->transientEncoder.value != commands->encoder.value))
        return VERNON_STATUS_INVALID_ARGUMENT;
    for (const PayloadSite &site : batch->sites)
        if (!payloadSiteIsValid(site, payload))
            return VERNON_STATUS_INVALID_ARGUMENT;
    return providerStatusBoundary([&] {
        return commands->provider.encode_packets(commands->provider.user_data, commands->native, batch->handle, payload,
                                                 payloadWordCount);
    });
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

extern "C" VernonStatus vernonRuntimeCorePacketsCreate(const VernonRuntimeDeviceProvider *provider,
                                                       VernonRuntimeCorePackets **output) {
    return runtimeCoreBoundary([&] { return createPacketsImpl(provider, output); });
}

extern "C" void vernonRuntimeCorePacketsDestroy(VernonRuntimeCorePackets *packets) {
    runtimeCoreVoidBoundary([&] {
        if (!packets)
            return;
        resetPackets(*packets);
        delete packets;
    });
}

extern "C" void vernonRuntimeCorePacketsReset(VernonRuntimeCorePackets *packets) {
    runtimeCoreVoidBoundary([&] {
        if (packets)
            resetPackets(*packets);
    });
}

extern "C" int vernonRuntimeCorePacketsIsEmpty(const VernonRuntimeCorePackets *packets) {
    return !packets || packets->words.empty();
}

extern "C" const uint32_t *vernonRuntimeCorePacketsGetPayload(const VernonRuntimeCorePackets *packets,
                                                              size_t *wordCount) {
    if (wordCount)
        *wordCount = packets ? packets->payload.size() : 0;
    return packets ? packets->payload.data() : nullptr;
}

extern "C" VernonStatus vernonRuntimeCorePacketsBindPipeline(VernonRuntimeCorePackets *packets,
                                                             VernonRuntimeCorePipeline *pipeline) {
    return runtimeCoreBoundary([&] { return bindPacketsPipelineImpl(packets, pipeline); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsBindGraphicsVariant(VernonRuntimeCorePackets *packets,
                                                                    VernonRuntimeCoreGraphicsVariant *variant) {
    return runtimeCoreBoundary([&] { return bindPacketsGraphicsVariantImpl(packets, variant); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsBindGroup(VernonRuntimeCorePackets *packets, uint32_t index,
                                                          VernonRuntimeCoreGroup *group) {
    return runtimeCoreBoundary([&] { return bindPacketsGroupImpl(packets, index, group); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsBindVertexBuffers(VernonRuntimeCorePackets *packets,
                                                                  uint32_t firstBinding,
                                                                  const VernonRuntimeProviderResourceReference *buffers,
                                                                  size_t bufferCount) {
    return runtimeCoreBoundary(
        [&] { return bindPacketsVertexBuffersImpl(packets, firstBinding, buffers, bufferCount); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsDispatch(VernonRuntimeCorePackets *packets,
                                                         const uint32_t groupCount[3]) {
    return runtimeCoreBoundary([&] { return packetsDispatchImpl(packets, groupCount); });
}

extern "C" VernonStatus
vernonRuntimeCorePacketsBeginRendering(VernonRuntimeCorePackets *packets,
                                       const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    return runtimeCoreBoundary([&] { return packetsBeginRenderingImpl(packets, descriptor); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsEndRendering(VernonRuntimeCorePackets *packets) {
    return runtimeCoreBoundary([&] { return packetsEndRenderingImpl(packets); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsDraw(VernonRuntimeCorePackets *packets,
                                                     const VernonRuntimeCoreDrawInvocation *invocation) {
    return runtimeCoreBoundary([&] { return packetsDrawImpl(packets, invocation); });
}

extern "C" VernonStatus vernonRuntimeCoreDrawPayload(const VernonRuntimeCoreDrawInvocation *invocation,
                                                     VernonRuntimeProviderDrawPayload *payload) {
    return runtimeCoreBoundary([&] { return drawPayloadImpl(invocation, payload); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsPushDebugGroup(VernonRuntimeCorePackets *packets,
                                                               VernonStringView label) {
    return runtimeCoreBoundary(
        [&] { return packetsDebugImpl(packets, VERNON_RUNTIME_PROVIDER_PACKET_PUSH_DEBUG_GROUP, label); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsPopDebugGroup(VernonRuntimeCorePackets *packets) {
    return runtimeCoreBoundary(
        [&] { return packetsDebugImpl(packets, VERNON_RUNTIME_PROVIDER_PACKET_POP_DEBUG_GROUP, {}); });
}

extern "C" VernonStatus vernonRuntimeCorePacketsInsertDebugMarker(VernonRuntimeCorePackets *packets,
                                                                  VernonStringView label) {
    return runtimeCoreBoundary(
        [&] { return packetsDebugImpl(packets, VERNON_RUNTIME_PROVIDER_PACKET_INSERT_DEBUG_MARKER, label); });
}

extern "C" VernonStatus vernonRuntimeCorePacketBatchCreate(const VernonRuntimeCorePackets *packets,
                                                           VernonRuntimeCorePacketBatch **output) {
    return runtimeCoreBoundary([&] { return createPacketBatchImpl(packets, output); });
}

extern "C" void vernonRuntimeCorePacketBatchDestroy(VernonRuntimeCorePacketBatch *batch) {
    runtimeCoreVoidBoundary([&] { releasePacketBatch(batch); });
}

extern "C" VernonStatus vernonRuntimeCoreBeginCommands(const VernonRuntimeDeviceProvider *provider,
                                                       VernonRuntimeProviderObject encoder,
                                                       VernonRuntimeCoreCommands **output) {
    return runtimeCoreBoundary([&] { return beginCommandsImpl(provider, encoder, output); });
}

extern "C" VernonStatus vernonRuntimeCoreEndCommands(VernonRuntimeCoreCommands *commands) {
    return runtimeCoreBoundary([&] { return endCommandsImpl(commands); });
}

extern "C" VernonStatus vernonRuntimeCoreResumeCommands(VernonRuntimeCoreCommands *commands) {
    return runtimeCoreBoundary([&] { return resumeCommandsImpl(commands); });
}

extern "C" VernonStatus vernonRuntimeCoreUpdateGroup(VernonRuntimeCoreCommands *commands, VernonRuntimeCoreGroup *group,
                                                     const VernonRuntimeProviderBindingValue *values,
                                                     size_t valueCount) {
    return runtimeCoreBoundary([&] { return updateGroupImpl(commands, group, values, valueCount); });
}

extern "C" VernonStatus vernonRuntimeCoreEncodePackets(VernonRuntimeCoreCommands *commands,
                                                       const VernonRuntimeCorePacketBatch *batch,
                                                       const uint32_t *payload, size_t payloadWordCount) {
    return runtimeCoreBoundary([&] { return encodePacketsImpl(commands, batch, payload, payloadWordCount); });
}
