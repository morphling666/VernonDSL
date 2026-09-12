#include "adapter_common.h"
#include "rhi/rhi_internal.h"
#include "runtime/vertex_attribute_capabilities.h"

#if defined(VERNON_HAS_VULKAN_RHI)

#include "rhi/vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

struct VulkanAdapterState {
    rhi::vulkan::DeviceState *device{};
};

VulkanAdapterState &vulkanState(VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_VULKAN);
    assert(adapter.backend.state);
    return *static_cast<VulkanAdapterState *>(adapter.backend.state);
}

const VulkanAdapterState &vulkanState(const VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_VULKAN);
    assert(adapter.backend.state);
    return *static_cast<const VulkanAdapterState *>(adapter.backend.state);
}

rhi::vulkan::DeviceState &vulkanDevice(VernonRuntimeRhiAdapter &adapter) { return *vulkanState(adapter).device; }
const rhi::vulkan::DeviceState &vulkanDevice(const VernonRuntimeRhiAdapter &adapter) {
    return *vulkanState(adapter).device;
}

void destroyBackend(void *state) noexcept { delete static_cast<VulkanAdapterState *>(state); }
RhiAdapterResult<void> synchronizeBackend(void *state, std::string &error) noexcept {
    try {
        if (static_cast<VulkanAdapterState *>(state)->device->synchronize(error))
            return RhiAdapterResult<void>{vernon::ok()};
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure, {"vulkan_synchronize", 0, 0}})};
    } catch (...) {
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure, {"vulkan_synchronize", 0, 0}})};
    }
}
uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const VulkanAdapterState *>(state)->device);
}
void invalidateBackend(void *) noexcept {}

const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity, invalidateBackend};

template <typename Handle> Handle nativeHandle(uint64_t value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(static_cast<uintptr_t>(value));
    else
        return static_cast<Handle>(value);
}

template <typename Handle> uint64_t nativeHandleBits(Handle value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<uintptr_t>(value);
    else
        return static_cast<uint64_t>(value);
}

VkAttachmentDescription attachmentDescription(VkFormat format, VkSampleCountFlagBits samples, VkAttachmentLoadOp load,
                                              VkAttachmentStoreOp store, VkAttachmentLoadOp stencilLoad,
                                              VkAttachmentStoreOp stencilStore, VkImageLayout layout) {
    VkAttachmentDescription result{};
    result.format = format;
    result.samples = samples;
    result.loadOp = load;
    result.storeOp = store;
    result.stencilLoadOp = stencilLoad;
    result.stencilStoreOp = stencilStore;
    result.initialLayout = layout;
    result.finalLayout = layout;
    return result;
}

void destroyRecordedFramebuffer(void *context, uint64_t object) {
    auto &device = *static_cast<rhi::vulkan::DeviceState *>(context);
    rhi::vulkan::driver().destroyFramebuffer(device.device, nativeHandle<VkFramebuffer>(object), nullptr);
}

void destroyRecordedRenderPass(void *context, uint64_t object) {
    auto &device = *static_cast<rhi::vulkan::DeviceState *>(context);
    rhi::vulkan::driver().destroyRenderPass(device.device, nativeHandle<VkRenderPass>(object), nullptr);
}

struct PreparedShader {
    rhi::vulkan::DeviceState *device{};
    uint32_t stage{};
    VkShaderModule module{};
    std::string entry;
    std::vector<uint32_t> words;
};
struct PreparedLayout {
    struct Entry {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        uint32_t inlineOffset{UINT32_MAX};
    };
    struct PushRange {
        uint32_t stage{};
        uint32_t offset{};
        uint32_t size{};
    };
    rhi::vulkan::DeviceState *device{};
    std::vector<Entry> entries;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    std::vector<PushRange> pushRanges;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<uint8_t> descriptorSetsUsed;
    uint32_t pushConstantSize{};

    ~PreparedLayout();
};
struct PreparedPipeline {
    struct RenderingCacheEntry {
        std::array<VernonRuntimeProviderResourceReference, 9> resources{};
        std::array<VkFormat, 9> formats{};
        std::array<VernonRhiLoadOperation, 9> loads{};
        std::array<VernonRhiStoreOperation, 9> stores{};
        uint32_t colorCount{};
        uint32_t width{};
        uint32_t height{};
        uint64_t lastEncoder{};
        VernonRhiLoadOperation stencilLoad{VERNON_RHI_LOAD_DISCARD};
        VernonRhiStoreOperation stencilStore{VERNON_RHI_STORE_DISCARD};
        bool hasDepth{};
        VkRenderPass renderPass{};
        VkFramebuffer framebuffer{};
    };
    std::atomic<uint32_t> references{1};
    VernonRuntimeRhiAdapter *adapter{};
    rhi::vulkan::DeviceState *device{};
    PreparedLayout *layout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline pipeline{};
    VkRenderPass renderPass{};
    std::vector<VkShaderModule> specializedShaderModules;
    std::vector<RenderingCacheEntry> renderingCache;
    std::mutex renderingCacheMutex;
    bool graphics{};

    ~PreparedPipeline();
};

constexpr size_t maxRenderingCacheEntries = 32;
struct PreparedBindingSet {
    std::atomic<uint32_t> references{1};
    struct Slot {
        PreparedLayout::Entry entry{};
        VernonRuntimeProviderBindingValue value{};
        std::vector<uint8_t> inlineStorage;
    };
    struct SnapshotKey {
        uint64_t revision{};

        bool operator==(const SnapshotKey &other) const { return revision == other.revision; }
    };
    struct SnapshotKeyHash {
        size_t operator()(const SnapshotKey &key) const { return std::hash<uint64_t>{}(key.revision); }
    };
    struct Snapshot {
        SnapshotKey key;
        rhi::vulkan::DeviceState *device{};
        std::vector<VkDescriptorSet> descriptorSets;
        std::vector<Slot> slots;
        rhi::vulkan::Buffer inlineBuffer;
        std::vector<VkDeviceSize> inlineOffsets;
        size_t commandReferences{};

        ~Snapshot();
    };
    rhi::vulkan::DeviceState *device{};
    PreparedLayout *layout{};
    std::vector<Slot> slots;
    std::unordered_map<uint32_t, size_t> slotIndices;
    std::unordered_map<uint64_t, size_t> samplerByBinding;
    std::vector<size_t> valueIndices;
    std::vector<uint8_t> seenSlots;
    std::vector<uint8_t> pushConstantStorage;
    std::unordered_map<SnapshotKey, std::unique_ptr<Snapshot>, SnapshotKeyHash> snapshots;
    uint64_t resourceRevision{1};
    std::mutex mutex;

    ~PreparedBindingSet();
};

uint64_t descriptorBindingKey(uint32_t set, uint32_t binding) { return (static_cast<uint64_t>(set) << 32) | binding; }

void releaseCommandBindings(void *context, uint64_t);
void releaseCommandBindingSnapshot(void *context, uint64_t object);
void releaseCommandPipeline(void *context, uint64_t);

RhiAdapterResult<void> retainCommandObjects(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            PreparedPipeline &pipeline, PreparedBindingSet *bindings) {
    pipeline.references.fetch_add(1, std::memory_order_relaxed);
    if (!deferCommandCleanup(adapter, encoder, &pipeline, 0, releaseCommandPipeline)) {
        releaseCommandPipeline(&pipeline, 0);
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_command_pipeline_cleanup_registration_failed", 0, 0}})};
    }
    if (bindings) {
        bindings->references.fetch_add(1, std::memory_order_relaxed);
        if (!deferCommandCleanup(adapter, encoder, bindings, 0, releaseCommandBindings)) {
            releaseCommandBindings(bindings, 0);
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"vulkan_command_bindings_cleanup_registration_failed", 0, 0}})};
        }
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

VkAttachmentLoadOp attachmentLoad(uint32_t operation) {
    switch (operation) {
    case VERNON_RHI_LOAD_CLEAR:
        return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case VERNON_RHI_LOAD_PRESERVE:
        return VK_ATTACHMENT_LOAD_OP_LOAD;
    default:
        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }
}

VkAttachmentStoreOp attachmentStore(uint32_t operation) {
    return operation == VERNON_RHI_STORE_PRESERVE ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

VkShaderStageFlags stages(uint32_t value) {
    VkShaderStageFlags result = 0;
    if (value & VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
        result |= VK_SHADER_STAGE_COMPUTE_BIT;
    if (value & VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
        result |= VK_SHADER_STAGE_VERTEX_BIT;
    if (value & VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
        result |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return result;
}

VkDescriptorType descriptorType(VernonRuntimeProviderBindingKind kind) {
    switch (kind) {
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
        // Compute inline values use an internally owned storage buffer;
        // graphics inline values take the push-constant path instead.
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    case VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER:
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

VkFormat vertexFormat(uint32_t dtype, uint32_t components) {
    static constexpr VkFormat formats[][4] = {
        {VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT},
        {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT},
        {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
        {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
        {VK_FORMAT_R64_SFLOAT, VK_FORMAT_R64G64_SFLOAT, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED},
    };
    const int row = dtype == VERNON_RUNTIME_PROVIDER_I32   ? 0
                    : dtype == VERNON_RUNTIME_PROVIDER_U32 ? 1
                    : dtype == VERNON_RUNTIME_PROVIDER_F16 ? 2
                    : dtype == VERNON_RUNTIME_PROVIDER_F32 ? 3
                    : dtype == VERNON_RUNTIME_PROVIDER_F64 ? 4
                                                           : -1;
    return row < 0 || components == 0 || components > 4 ? VK_FORMAT_UNDEFINED : formats[row][components - 1];
}

bool relocatePushConstantOffsets(std::vector<uint32_t> &words, uint32_t delta) {
    if (delta == 0)
        return true;
    constexpr uint16_t kOpTypePointer = 32;
    constexpr uint16_t kOpVariable = 59;
    constexpr uint16_t kOpMemberDecorate = 72;
    constexpr uint32_t kPushConstantStorageClass = 9;
    constexpr uint32_t kOffsetDecoration = 35;
    uint32_t pushConstantPointerType = 0;
    uint32_t pushConstantStruct = 0;
    for (size_t index = 5; index < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[index] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[index]);
        if (wordCount == 0 || index + wordCount > words.size())
            return false;
        if (opcode == kOpVariable && wordCount >= 4 && words[index + 3] == kPushConstantStorageClass)
            pushConstantPointerType = words[index + 1];
        index += wordCount;
    }
    if (pushConstantPointerType == 0)
        return true;
    for (size_t index = 5; index < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[index] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[index]);
        if (opcode == kOpTypePointer && wordCount >= 4 && words[index + 1] == pushConstantPointerType &&
            words[index + 2] == kPushConstantStorageClass) {
            pushConstantStruct = words[index + 3];
            break;
        }
        index += wordCount;
    }
    if (pushConstantStruct == 0)
        return false;
    for (size_t index = 5; index < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[index] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[index]);
        if (opcode == kOpMemberDecorate && wordCount >= 5 && words[index + 1] == pushConstantStruct &&
            words[index + 3] == kOffsetDecoration)
            words[index + 4] += delta;
        index += wordCount;
    }
    return true;
}

RhiAdapterResult<void> failureResult(VkResult result, const char *operation) {
    return result == VK_SUCCESS
               ? RhiAdapterResult<void>{vernon::ok()}
               : RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                     vernon::ProviderErrorCode::BackendFailure, {operation, static_cast<uint64_t>(result), 0}})};
}

uint32_t getCapabilities(void *) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    const auto &device = vulkanDevice(adapter);
    return {0x56554c4b414eu, reinterpret_cast<uintptr_t>(device.physicalDevice), device.apiVersion};
}

RhiAdapterResult<void> prepareShaderResult(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !descriptor->data ||
        descriptor->size < sizeof(uint32_t) || descriptor->size % sizeof(uint32_t) != 0 || !descriptor->entry.data ||
        descriptor->entry.size == 0 || descriptor->format.size != 5 ||
        std::memcmp(descriptor->format.data, "spirv", 5) != 0)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_adapter_received_an_invalid_spirv_shader", 0, 0}})};
    auto shader = std::unique_ptr<PreparedShader>(new (std::nothrow) PreparedShader());
    if (!shader)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_shader_preparation_ran_out_of_memory", 0, 0}})};
    shader->device = &vulkanDevice(adapter);
    shader->stage = descriptor->stage;
    shader->entry.assign(descriptor->entry.data, descriptor->entry.size);
    const auto *words = static_cast<const uint32_t *>(descriptor->data);
    const size_t wordCount = descriptor->size / 4u;
    shader->words.assign(words, words + wordCount);
    const VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, descriptor->size,
                                              static_cast<const uint32_t *>(descriptor->data)};
    const VkResult result =
        rhi::vulkan::driver().createShaderModule(shader->device->device, &createInfo, nullptr, &shader->module);
    if (result != VK_SUCCESS)
        return failureResult(result, "vkCreateShaderModule");
    *output = toHandle(shader.release());
    adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_adapter_received_an_invalid_pipeline_layout", 0, 0}})};
    auto layout = std::unique_ptr<PreparedLayout>(new (std::nothrow) PreparedLayout());
    if (!layout)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_layout_preparation_ran_out_of_memory", 0, 0}})};
    layout->device = &vulkanDevice(adapter);
    std::vector<std::vector<VkDescriptorSetLayoutBinding>> nativeBindings;
    try {
        layout->entries.reserve(descriptor->binding_count);
        for (size_t index = 0; index < descriptor->binding_count; ++index) {
            const auto &source = descriptor->bindings[index];
            const bool supported = source.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
            if (!supported || source.array_count != 1)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::Unsupported, {"vulkan_layout_contains_an_unsupported_binding", 0, 0}})};
            PreparedLayout::Entry entry;
            entry.layout = source;
            if (source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
                if (source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) {
                    if (source.set == UINT32_MAX)
                        return RhiAdapterResult<void>{
                            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                              {"vulkan_descriptor_set_index_is_invalid", 0, 0}})};
                    if (nativeBindings.size() <= source.set)
                        nativeBindings.resize(static_cast<size_t>(source.set) + 1);
                    nativeBindings[source.set].push_back(
                        {source.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
                }
            } else if (source.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                if (source.set == UINT32_MAX)
                    return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                        vernon::ProviderErrorCode::InvalidArgument, {"vulkan_descriptor_set_index_is_invalid", 0, 0}})};
                if (nativeBindings.size() <= source.set)
                    nativeBindings.resize(static_cast<size_t>(source.set) + 1);
                auto &setBindings = nativeBindings[source.set];
                const auto existing = std::find_if(setBindings.begin(), setBindings.end(), [&](const auto &binding) {
                    return binding.binding == source.binding;
                });
                if (existing == setBindings.end())
                    setBindings.push_back(
                        {source.binding, descriptorType(source.kind), 1, stages(source.stage_mask), nullptr});
                else if ((existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
                          source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) ||
                         (existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER &&
                          source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)) {
                    existing->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    existing->stageFlags |= stages(source.stage_mask);
                } else
                    return RhiAdapterResult<void>{vernon::err(
                        vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"vulkan_layout_contains_duplicate_descriptor_bindings", 0, 0}})};
            }
            layout->entries.push_back(entry);
        }
        std::vector<size_t> pushEntries;
        for (size_t index = 0; index < layout->entries.size(); ++index)
            if (layout->entries[index].layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                layout->entries[index].layout.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
                pushEntries.push_back(index);
        std::sort(pushEntries.begin(), pushEntries.end(), [&](size_t left, size_t right) {
            const auto &leftLayout = layout->entries[left].layout;
            const auto &rightLayout = layout->entries[right].layout;
            if (leftLayout.stage_mask != rightLayout.stage_mask)
                return leftLayout.stage_mask < rightLayout.stage_mask;
            // Shader push-constant members follow entry argument order, while provider slots are name-sorted.
            return leftLayout.argument_index < rightLayout.argument_index;
        });
        for (size_t index : pushEntries) {
            PreparedLayout::Entry &entry = layout->entries[index];
            const uint32_t alignment = entry.layout.element_alignment;
            if (!alignment || (alignment & (alignment - 1)) != 0 ||
                layout->pushConstantSize > UINT32_MAX - (alignment - 1))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"vulkan_inline_uniform_alignment_is_invalid", 0, 0}})};
            entry.inlineOffset = (layout->pushConstantSize + alignment - 1) & ~(alignment - 1);
            if (entry.layout.element_size > UINT32_MAX - entry.inlineOffset)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"vulkan_push_constant_layout_size_overflow", 0, 0}})};
            layout->pushConstantSize = entry.inlineOffset + entry.layout.element_size;
            const uint32_t entryEnd = entry.inlineOffset + entry.layout.element_size;
            if (layout->pushRanges.empty() || layout->pushRanges.back().stage != entry.layout.stage_mask)
                layout->pushRanges.push_back({entry.layout.stage_mask, entry.inlineOffset, entry.layout.element_size});
            else
                layout->pushRanges.back().size = entryEnd - layout->pushRanges.back().offset;
        }
        for (size_t index = 0; index < descriptor->vertex_attribute_count; ++index) {
            const auto &attribute = descriptor->vertex_attributes[index];
            const VkFormat format = vertexFormat(attribute.dtype, attribute.component_count);
            const bool bindingExists =
                std::any_of(layout->entries.begin(), layout->entries.end(), [&](const PreparedLayout::Entry &entry) {
                    return entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                           entry.layout.binding == attribute.binding;
                });
            std::string capabilityDiagnostic;
            if (!bindingExists)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument,
                    {"vulkan_validate_vertex_attribute_binding", attribute.binding, attribute.location}})};
            if (!validateVertexAttributeCapability(VertexAttributeBackend::Vulkan, attribute,
                                                   layout->device->maxVertexInputAttributes, true,
                                                   capabilityDiagnostic))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument,
                    {"vulkan_validate_vertex_attribute_capability", attribute.location, attribute.binding}})};
            VkFormatProperties properties{};
            if (format != VK_FORMAT_UNDEFINED)
                rhi::vulkan::driver().getPhysicalDeviceFormatProperties(layout->device->physicalDevice, format,
                                                                        &properties);
            if (format == VK_FORMAT_UNDEFINED || !(properties.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument,
                    {"vulkan_vertex_attribute_format_is_unsupported_by_the_selected_device", 0, 0}})};
            layout->vertexAttributes.push_back(attribute);
        }
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_layout_preparation_ran_out_of_memory", 0, 0}})};
    }
    if (layout->pushConstantSize > layout->device->maxPushConstantsSize)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::Unsupported, {"vulkan_push_constants_exceed_the_device_limit", 0, 0}})};
    layout->descriptorSetLayouts.resize(nativeBindings.size());
    layout->descriptorSetsUsed.resize(nativeBindings.size());
    for (size_t set = 0; set < nativeBindings.size(); ++set) {
        const auto &setBindings = nativeBindings[set];
        const VkDescriptorSetLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
                                                         0, static_cast<uint32_t>(setBindings.size()),
                                                         setBindings.data()};
        const VkResult result = rhi::vulkan::driver().createDescriptorSetLayout(
            layout->device->device, &createInfo, nullptr, &layout->descriptorSetLayouts[set]);
        if (result != VK_SUCCESS)
            return failureResult(result, "vkCreateDescriptorSetLayout");
        layout->descriptorSetsUsed[set] = !setBindings.empty();
    }
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createPipelineLayoutResult(VernonRuntimeRhiAdapter &adapter, PreparedPipeline &pipeline) {
    VkPushConstantRange range{VK_SHADER_STAGE_ALL, 0, pipeline.layout->pushConstantSize};
    const VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                nullptr,
                                                0,
                                                static_cast<uint32_t>(pipeline.layout->descriptorSetLayouts.size()),
                                                pipeline.layout->descriptorSetLayouts.data(),
                                                pipeline.layout->pushConstantSize ? 1u : 0u,
                                                pipeline.layout->pushConstantSize ? &range : nullptr};
    return failureResult(rhi::vulkan::driver().createPipelineLayout(pipeline.device->device, &createInfo, nullptr,
                                                                    &pipeline.pipelineLayout),
                         "vkCreatePipelineLayout");
}

RhiAdapterResult<void> preparePipelineImplResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                                 VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout || !descriptor->shaders ||
        descriptor->shader_count == 0)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_adapter_received_an_invalid_pipeline", 0, 0}})};
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    pipeline->adapter = &adapter;
    adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
    pipeline->device = &vulkanDevice(adapter);
    pipeline->layout = layout;
    pipeline->graphics = descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    pipeline->specializedShaderModules.reserve(2);
    auto pipelineLayout = createPipelineLayoutResult(adapter, *pipeline);
    if (!pipelineLayout)
        return pipelineLayout;
    if (pipeline->graphics && descriptor->color_format_count == 0 && descriptor->depth_stencil_format == 0) {
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    }
    if (!pipeline->graphics) {
        auto *shader = descriptor->shader_count == 1 ? fromHandle<PreparedShader>(descriptor->shaders[0]) : nullptr;
        if (!shader || shader->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"vulkan_compute_pipeline_requires_one_compute_shader", 0, 0}})};
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    nullptr,
                                                    0,
                                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                                    shader->module,
                                                    shader->entry.c_str(),
                                                    nullptr};
        const VkComputePipelineCreateInfo createInfo{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage, pipeline->pipelineLayout, {}, 0};
        auto created = failureResult(rhi::vulkan::driver().createComputePipelines(
                                         pipeline->device->device, {}, 1, &createInfo, nullptr, &pipeline->pipeline),
                                     "vkCreateComputePipelines");
        if (!created)
            return created;
    } else {
        PreparedShader *vertex = nullptr;
        PreparedShader *fragment = nullptr;
        for (size_t index = 0; index < descriptor->shader_count; ++index) {
            auto *shader = fromHandle<PreparedShader>(descriptor->shaders[index]);
            if (shader && shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
                vertex = shader;
            else if (shader && shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
                fragment = shader;
        }
        if (!vertex || !fragment)
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"vulkan_graphics_provider_requires_vertex_and_fragment_shaders", 0, 0}})};
        if (descriptor->sample_count != 1 || descriptor->color_format_count > 8 ||
            (descriptor->depth_stencil_format && descriptor->depth_stencil_format != VK_FORMAT_D32_SFLOAT &&
             descriptor->depth_stencil_format != VK_FORMAT_D32_SFLOAT_S8_UINT))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument,
                {"vulkan_graphics_pipeline_uses_an_unsupported_attachment_configuration", 0, 0}})};
        const auto stageOffset = [&](uint32_t stage) {
            uint32_t offset = UINT32_MAX;
            for (const auto &entry : layout->entries)
                if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE && entry.layout.stage_mask == stage &&
                    entry.inlineOffset != UINT32_MAX)
                    offset = std::min(offset, entry.inlineOffset);
            return offset == UINT32_MAX ? 0u : offset;
        };
        const auto specializeShader = [&](PreparedShader &shader, uint32_t offset, VkShaderModule &module) {
            if (offset == 0) {
                module = shader.module;
                return RhiAdapterResult<void>{vernon::ok()};
            }
            std::vector<uint32_t> words = shader.words;
            if (!relocatePushConstantOffsets(words, offset))
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"vulkan_provider_could_not_relocate_push_constant_offsets", 0, 0}})};
            const VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                                      words.size() * sizeof(uint32_t), words.data()};
            const VkResult result =
                rhi::vulkan::driver().createShaderModule(pipeline->device->device, &moduleInfo, nullptr, &module);
            if (result != VK_SUCCESS)
                return failureResult(result, "vkCreateShaderModule for relocated push constants");
            pipeline->specializedShaderModules.push_back(module);
            return RhiAdapterResult<void>{vernon::ok()};
        };
        VkShaderModule vertexModule{};
        VkShaderModule fragmentModule{};
        auto specializedVertex =
            specializeShader(*vertex, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_VERTEX), vertexModule);
        if (!specializedVertex)
            return specializedVertex;
        auto specializedFragment =
            specializeShader(*fragment, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT), fragmentModule);
        if (!specializedFragment)
            return specializedFragment;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                                   nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertexModule,
                                                                   vertex->entry.c_str(), nullptr}};
        shaderStages.push_back({VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, fragment->entry.c_str(), nullptr});
        std::vector<VkVertexInputBindingDescription> vertexBindings;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
        for (const auto &entry : layout->entries)
            if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                vertexBindings.push_back(
                    {entry.layout.binding, 0,
                     entry.layout.divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
            }
        for (const VernonRuntimeProviderVertexAttribute &attribute : layout->vertexAttributes)
            vertexAttributes.push_back({attribute.location, attribute.binding,
                                        vertexFormat(attribute.dtype, attribute.component_count),
                                        attribute.relative_offset});
        for (auto &binding : vertexBindings) {
            if (binding.binding >= descriptor->vertex_stride_count)
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                      {"vulkan_graphics_pipeline_is_missing_a_vertex_stride", 0, 0}})};
            binding.stride = descriptor->vertex_strides[binding.binding];
        }
        const VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(vertexBindings.size()),
            vertexBindings.data(),
            static_cast<uint32_t>(vertexAttributes.size()),
            vertexAttributes.data()};
        const VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
            descriptor->topology == 1   ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
            : descriptor->topology == 2 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                                        : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_FALSE};
        const VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        const auto &rasterization = descriptor->rasterization;
        if (rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
            rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE || rasterization.depth_clamp ||
            rasterization.depth_bias_enabled > 1 || !std::isfinite(rasterization.depth_bias_constant) ||
            !std::isfinite(rasterization.depth_bias_slope))
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"vulkan_graphics_pipeline_contains_an_unsupported_rasterization_state", 0, 0}})};
        const VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                            nullptr,
                                                            0,
                                                            VK_FALSE,
                                                            VK_FALSE,
                                                            VK_POLYGON_MODE_FILL,
                                                            static_cast<VkCullModeFlags>(rasterization.cull_mode),
                                                            rasterization.front_face == VERNON_RHI_FRONT_FACE_CLOCKWISE
                                                                ? VK_FRONT_FACE_CLOCKWISE
                                                                : VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                            rasterization.depth_bias_enabled ? VK_TRUE : VK_FALSE,
                                                            rasterization.depth_bias_constant,
                                                            0,
                                                            rasterization.depth_bias_slope,
                                                            1};
        const VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<VkSampleCountFlagBits>(std::max(1u, descriptor->sample_count)),
            VK_FALSE,
            0,
            nullptr,
            VK_FALSE,
            VK_FALSE};
        const bool hasDepth = descriptor->depth_stencil_format != 0;
        const bool hasStencil = descriptor->depth_stencil_format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        const auto &depthState = descriptor->depth_stencil;
        if (depthState.depth_test > 1 || depthState.depth_write > 1 || depthState.stencil_test > 1 ||
            depthState.depth_compare > VERNON_RHI_COMPARE_ALWAYS ||
            (!hasDepth && (depthState.depth_test || depthState.depth_write)) ||
            (!hasStencil && depthState.stencil_test))
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"vulkan_graphics_pipeline_contains_an_invalid_depth_stencil_state", 0, 0}})};
        const auto stencilFace = [](const VernonStencilFaceState &face) {
            return VkStencilOpState{static_cast<VkStencilOp>(face.stencil_fail),
                                    static_cast<VkStencilOp>(face.pass),
                                    static_cast<VkStencilOp>(face.depth_fail),
                                    static_cast<VkCompareOp>(face.compare),
                                    0,
                                    0,
                                    0};
        };
        if (depthState.front.stencil_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
            depthState.front.depth_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
            depthState.front.pass > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
            depthState.front.compare > VERNON_RHI_COMPARE_ALWAYS ||
            depthState.back.stencil_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
            depthState.back.depth_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
            depthState.back.pass > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
            depthState.back.compare > VERNON_RHI_COMPARE_ALWAYS || depthState.stencil_read_mask > 0xff ||
            depthState.stencil_write_mask > 0xff)
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"vulkan_graphics_pipeline_contains_an_invalid_stencil_state", 0, 0}})};
        VkPipelineDepthStencilStateCreateInfo depthStencil{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            nullptr,
            0,
            depthState.depth_test ? VK_TRUE : VK_FALSE,
            depthState.depth_write ? VK_TRUE : VK_FALSE,
            static_cast<VkCompareOp>(depthState.depth_test ? depthState.depth_compare : VERNON_RHI_COMPARE_ALWAYS),
            VK_FALSE,
            depthState.stencil_test ? VK_TRUE : VK_FALSE,
            stencilFace(depthState.front),
            stencilFace(depthState.back),
            0,
            1};
        depthStencil.front.compareMask = depthState.stencil_read_mask;
        depthStencil.front.writeMask = depthState.stencil_write_mask;
        depthStencil.back.compareMask = depthState.stencil_read_mask;
        depthStencil.back.writeMask = depthState.stencil_write_mask;
        std::array<VkPipelineColorBlendAttachmentState, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS>
            blendAttachments{};
        if (descriptor->color_blend_count != descriptor->color_format_count ||
            (descriptor->color_blend_count && !descriptor->color_blends))
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"vulkan_graphics_pipeline_blend_state_does_not_match_its_attachments", 0, 0}})};
        for (size_t index = 0; index < descriptor->color_format_count; ++index) {
            const auto &source = descriptor->color_blends[index];
            if (source.blend_enabled > 1 || source.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.color_operation > VERNON_RHI_BLEND_MAXIMUM ||
                source.alpha_operation > VERNON_RHI_BLEND_MAXIMUM || (source.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"vulkan_graphics_pipeline_contains_an_invalid_blend_state", 0, 0}})};
            auto &target = blendAttachments[index];
            target.blendEnable = source.blend_enabled;
            target.srcColorBlendFactor = static_cast<VkBlendFactor>(source.source_color_factor);
            target.dstColorBlendFactor = static_cast<VkBlendFactor>(source.destination_color_factor);
            target.colorBlendOp = static_cast<VkBlendOp>(source.color_operation);
            target.srcAlphaBlendFactor = static_cast<VkBlendFactor>(source.source_alpha_factor);
            target.dstAlphaBlendFactor = static_cast<VkBlendFactor>(source.destination_alpha_factor);
            target.alphaBlendOp = static_cast<VkBlendOp>(source.alpha_operation);
            target.colorWriteMask = source.write_mask;
        }
        const VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                        nullptr,
                                                        0,
                                                        VK_FALSE,
                                                        VK_LOGIC_OP_COPY,
                                                        static_cast<uint32_t>(descriptor->color_format_count),
                                                        blendAttachments.data(),
                                                        {}};
        const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                             VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        const VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                       3, dynamicStates};
        const VkPipelineRenderingCreateInfo rendering{
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(descriptor->color_format_count),
            reinterpret_cast<const VkFormat *>(descriptor->color_formats),
            static_cast<VkFormat>(descriptor->depth_stencil_format),
            hasStencil ? static_cast<VkFormat>(descriptor->depth_stencil_format) : VK_FORMAT_UNDEFINED};
        std::array<VkAttachmentDescription, 9> renderPassAttachments{};
        std::array<VkAttachmentReference, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> renderPassReferences{};
        VkAttachmentReference depthReference{};
        if (!pipeline->device->dynamicRendering) {
            // This render pass defines pipeline compatibility only. Draw-time load/store operations and image
            // views belong to the render pass created by encodeDraw.
            for (size_t index = 0; index < descriptor->color_format_count; ++index) {
                renderPassAttachments[index] = attachmentDescription(
                    static_cast<VkFormat>(descriptor->color_formats[index]),
                    static_cast<VkSampleCountFlagBits>(std::max(1u, descriptor->sample_count)),
                    VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                renderPassReferences[index] = {static_cast<uint32_t>(index), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            }
            if (hasDepth) {
                renderPassAttachments[descriptor->color_format_count] =
                    attachmentDescription(static_cast<VkFormat>(descriptor->depth_stencil_format),
                                          static_cast<VkSampleCountFlagBits>(std::max(1u, descriptor->sample_count)),
                                          VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
                                          hasStencil ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                          hasStencil ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                depthReference = {static_cast<uint32_t>(descriptor->color_format_count),
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            }
            const VkSubpassDescription subpass{0,
                                               VK_PIPELINE_BIND_POINT_GRAPHICS,
                                               0,
                                               nullptr,
                                               static_cast<uint32_t>(descriptor->color_format_count),
                                               renderPassReferences.data(),
                                               nullptr,
                                               hasDepth ? &depthReference : nullptr,
                                               0,
                                               nullptr};
            const VkRenderPassCreateInfo renderPassInfo{
                VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                nullptr,
                0,
                static_cast<uint32_t>(descriptor->color_format_count + (hasDepth ? 1 : 0)),
                renderPassAttachments.data(),
                1,
                &subpass,
                0,
                nullptr};
            auto createdRenderPass =
                failureResult(rhi::vulkan::driver().createRenderPass(pipeline->device->device, &renderPassInfo, nullptr,
                                                                     &pipeline->renderPass),
                              "vkCreateRenderPass");
            if (!createdRenderPass)
                return createdRenderPass;
        }
        const VkGraphicsPipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                                      pipeline->device->dynamicRendering ? &rendering : nullptr,
                                                      0,
                                                      static_cast<uint32_t>(shaderStages.size()),
                                                      shaderStages.data(),
                                                      &vertexInput,
                                                      &assembly,
                                                      nullptr,
                                                      &viewport,
                                                      &raster,
                                                      &multisample,
                                                      hasDepth ? &depthStencil : nullptr,
                                                      &blend,
                                                      &dynamic,
                                                      pipeline->pipelineLayout,
                                                      pipeline->renderPass,
                                                      0,
                                                      {},
                                                      0};
        auto createdPipeline =
            failureResult(rhi::vulkan::driver().createGraphicsPipelines(pipeline->device->device, {}, 1, &createInfo,
                                                                        nullptr, &pipeline->pipeline),
                          "vkCreateGraphicsPipelines");
        if (!createdPipeline)
            return createdPipeline;
    }
    *output = toHandle(pipeline.release());
    adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    try {
        return preparePipelineImplResult(data, descriptor, output);
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> updateBindingsImplResult(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"vulkan_binding_values_do_not_match_the_prepared_layout", 0, 0}})};
    std::fill(bindings.seenSlots.begin(), bindings.seenSlots.end(), uint8_t{0});
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = bindings.slotIndices.find(values[index].slot);
        if (found == bindings.slotIndices.end() || bindings.seenSlots[found->second])
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"vulkan_binding_slot_is_invalid_or_duplicated", 0, 0}})};
        bindings.seenSlots[found->second] = 1;
        bindings.valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        if (value->kind != slot.entry.layout.kind)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"vulkan_binding_slot_or_kind_is_invalid", 0, 0}})};
        if (packedUniformBytes(slot.entry.layout.kind, slot.entry.layout.interface_kind)) {
            if (!value->payload.inline_value.data || value->payload.inline_value.size != slot.inlineStorage.size())
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"vulkan_inline_or_uniform_buffer_binding_size_is_invalid", 0, 0}})};
        } else if ((value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) == 0) {
            const auto *resource = providerBindingResource(*value);
            if (!resource || !resource->identity || !resource->resource.value)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"vulkan_resource_binding_is_invalid", 0, 0}})};
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && value->payload.buffer.stride == 0)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"vulkan_vertex_binding_has_no_stride", 0, 0}})};
        }
    }
    bool changed = false;
    bool resourcesChanged = false;
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto &value = values[bindings.valueIndices[index]];
        const auto *oldResource = providerBindingResource(slot.value);
        const auto *newResource = providerBindingResource(value);
        bool slotChanged =
            slot.value.slot != value.slot || slot.value.kind != value.kind || slot.value.flags != value.flags;
        if (packedUniformBytes(slot.entry.layout.kind, slot.entry.layout.interface_kind)) {
            slotChanged |= slot.value.payload.inline_value.size != value.payload.inline_value.size ||
                           (value.payload.inline_value.data &&
                            std::memcmp(slot.inlineStorage.data(), value.payload.inline_value.data,
                                        value.payload.inline_value.size));
        } else {
            slotChanged |=
                !oldResource || !newResource || std::memcmp(oldResource, newResource, sizeof(*newResource)) != 0;
            if (value.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                value.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
                slotChanged |= slot.value.payload.buffer.stride != value.payload.buffer.stride;
        }
        changed |= slotChanged;
        resourcesChanged |= slotChanged && slot.entry.inlineOffset == UINT32_MAX;
    }
    if (!changed)
        return RhiAdapterResult<void>{vernon::ok()};
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        slot.value = values[bindings.valueIndices[index]];
        if (slot.value.payload.inline_value.data &&
            packedUniformBytes(slot.value.kind, slot.entry.layout.interface_kind)) {
            std::memcpy(slot.inlineStorage.data(), slot.value.payload.inline_value.data,
                        slot.value.payload.inline_value.size);
            if (slot.entry.inlineOffset != UINT32_MAX)
                std::memcpy(bindings.pushConstantStorage.data() + slot.entry.inlineOffset,
                            slot.value.payload.inline_value.data, slot.value.payload.inline_value.size);
            slot.value.payload = {};
        }
    }
    if (resourcesChanged) {
        if (++bindings.resourceRevision == 0)
            bindings.resourceRevision = 1;
        for (auto snapshot = bindings.snapshots.begin(); snapshot != bindings.snapshots.end();)
            if (snapshot->second->commandReferences == 0)
                snapshot = bindings.snapshots.erase(snapshot);
            else
                ++snapshot;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> retainBindingResources(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                              const PreparedBindingSet *bindings) {
    if (!bindings)
        return RhiAdapterResult<void>{vernon::ok()};
    for (const auto &slot : bindings->slots) {
        if (packedUniformBytes(slot.entry.layout.kind, slot.entry.layout.interface_kind))
            continue;
        const auto *resource = providerBindingResource(slot.value);
        if (!resource || !resource->resource.value)
            continue;
        auto retained = retainCommandResource(adapter, encoder, *resource);
        if (!retained)
            return retained;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

PreparedBindingSet::Snapshot::~Snapshot() {
    if (!device)
        return;
    if (inlineBuffer.buffer)
        device->destroyBuffer(inlineBuffer);
    for (VkDescriptorSet descriptorSet : descriptorSets) {
        if (!descriptorSet)
            continue;
        std::string ignored;
        (void)device->freeDescriptorSet(descriptorSet, ignored);
    }
}

PreparedBindingSet::~PreparedBindingSet() = default;

void recordPushConstants(VkCommandBuffer command, const PreparedPipeline &pipeline,
                         const PreparedBindingSet &bindings) {
    for (const auto &range : pipeline.layout->pushRanges)
        rhi::vulkan::driver().cmdPushConstants(command, pipeline.pipelineLayout, stages(range.stage), range.offset,
                                               range.size, bindings.pushConstantStorage.data() + range.offset);
}

void releaseCommandBindingSnapshot(void *context, uint64_t object) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    auto *snapshot = reinterpret_cast<PreparedBindingSet::Snapshot *>(static_cast<uintptr_t>(object));
    if (!bindings || !snapshot)
        return;
    std::lock_guard<std::mutex> guard(bindings->mutex);
    const auto found = bindings->snapshots.find(snapshot->key);
    if (found == bindings->snapshots.end() || found->second.get() != snapshot || snapshot->commandReferences == 0)
        return;
    --snapshot->commandReferences;
    if (snapshot->commandReferences == 0 && snapshot->key.revision != bindings->resourceRevision)
        bindings->snapshots.erase(found);
}

RhiAdapterResult<void> retainBindingSnapshot(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             PreparedBindingSet &bindings, PreparedBindingSet::Snapshot &snapshot) {
    ++snapshot.commandReferences;
    if (deferCommandCleanup(adapter, encoder, &bindings, reinterpret_cast<uintptr_t>(&snapshot),
                            releaseCommandBindingSnapshot))
        return RhiAdapterResult<void>{vernon::ok()};
    --snapshot.commandReferences;
    return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
        vernon::ProviderErrorCode::BackendFailure, {"vulkan_binding_snapshot_cleanup_allocation_failed", 0, 0}})};
}

RhiAdapterResult<PreparedBindingSet::Snapshot *>
snapshotBindings(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, PreparedPipeline &pipeline,
                 PreparedBindingSet &bindings, VkCommandBuffer command) {
    std::string backendDiagnostic;
    std::lock_guard<std::mutex> guard(bindings.mutex);
    if (!retainBindingResources(adapter, encoder, &bindings))
        return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_binding_snapshot_could_not_retain_resources", 0, 0}})};
    const PreparedBindingSet::SnapshotKey key{bindings.resourceRevision};
    const auto existing = bindings.snapshots.find(key);
    if (existing != bindings.snapshots.end()) {
        if (!retainBindingSnapshot(adapter, encoder, bindings, *existing->second))
            return RhiAdapterResult<PreparedBindingSet::Snapshot *>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"vulkan_binding_snapshot_cleanup_registration_failed", 0, 0}})};
        recordPushConstants(command, pipeline, bindings);
        return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::ok(existing->second.get())};
    }

    auto snapshot = std::make_unique<PreparedBindingSet::Snapshot>();
    snapshot->key = key;
    snapshot->device = bindings.device;
    snapshot->slots = bindings.slots;
    snapshot->inlineOffsets.resize(snapshot->slots.size(), VK_WHOLE_SIZE);
    VkDeviceSize inlineSize = 0;
    VkBufferUsageFlags inlineUsage = 0;
    const VkDeviceSize alignment = bindings.device->descriptorBufferOffsetAlignment;
    for (size_t index = 0; index < snapshot->slots.size(); ++index) {
        const auto &slot = snapshot->slots[index];
        if (!packedUniformBytes(slot.entry.layout.kind, slot.entry.layout.interface_kind) ||
            (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE && slot.entry.inlineOffset != UINT32_MAX))
            continue;
        inlineSize = (inlineSize + alignment - 1) & ~(alignment - 1);
        snapshot->inlineOffsets[index] = inlineSize;
        inlineSize += slot.inlineStorage.size();
        inlineUsage |= slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                           ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                           : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (inlineSize) {
        if (!bindings.device->createBuffer(snapshot->inlineBuffer, inlineSize, inlineUsage,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                           backendDiagnostic))
            return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure,
                {"vulkan_create_inline_binding_buffer", inlineSize, static_cast<uint32_t>(inlineUsage)}})};
        void *mapped = nullptr;
        const VkResult result = rhi::vulkan::driver().mapMemory(bindings.device->device, snapshot->inlineBuffer.memory,
                                                                0, inlineSize, 0, &mapped);
        if (result != VK_SUCCESS) {
            auto failed = failureResult(result, "vkMapMemory");
            return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(std::move(failed).error())};
        }
        for (size_t index = 0; index < snapshot->slots.size(); ++index)
            if (snapshot->inlineOffsets[index] != VK_WHOLE_SIZE)
                std::memcpy(static_cast<uint8_t *>(mapped) + snapshot->inlineOffsets[index],
                            snapshot->slots[index].inlineStorage.data(), snapshot->slots[index].inlineStorage.size());
        rhi::vulkan::driver().unmapMemory(bindings.device->device, snapshot->inlineBuffer.memory);
    }
    snapshot->descriptorSets.resize(bindings.layout->descriptorSetLayouts.size());
    for (size_t set = 0; set < bindings.layout->descriptorSetLayouts.size(); ++set)
        if (bindings.layout->descriptorSetsUsed[set] &&
            !bindings.device->allocateDescriptorSet(bindings.layout->descriptorSetLayouts[set],
                                                    snapshot->descriptorSets[set], backendDiagnostic))
            return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"vulkan_allocate_binding_descriptor_set", set, 0}})};

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    writes.reserve(snapshot->slots.size());
    buffers.reserve(snapshot->slots.size());
    images.reserve(snapshot->slots.size());
    for (size_t index = 0; index < snapshot->slots.size(); ++index) {
        auto &slot = snapshot->slots[index];
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER ||
            (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE && slot.entry.inlineOffset != UINT32_MAX) ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)
            continue;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = snapshot->descriptorSets[slot.entry.layout.set];
        write.dstBinding = slot.entry.layout.binding;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType(slot.entry.layout.kind);
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
            const bool internallyOwned = packedUniformBytes(slot.entry.layout.kind, slot.entry.layout.interface_kind);
            rhi::vulkan::Buffer *buffer = &snapshot->inlineBuffer;
            if (!internallyOwned) {
                auto resolved = resolveCommandRhiResource(adapter, encoder, slot.value.payload.buffer.resource);
                if (!resolved) {
                    return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(std::move(resolved).error())};
                }
                buffer = reinterpret_cast<rhi::vulkan::Buffer *>(static_cast<uintptr_t>(std::move(resolved).value()));
            }
            if (!buffer)
                return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"vulkan_buffer_binding_is_stale", 0, 0}})};
            buffers.push_back(
                {buffer->buffer,
                 internallyOwned ? snapshot->inlineOffsets[index] : slot.value.payload.buffer.resource.offset,
                 internallyOwned ? slot.inlineStorage.size() : slot.value.payload.buffer.resource.size});
            write.pBufferInfo = &buffers.back();
        } else {
            VkDescriptorImageInfo info{};
            auto resolvedImage = resolveCommandRhiResource(adapter, encoder, slot.value.payload.image.view);
            if (!resolvedImage) {
                return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(std::move(resolvedImage).error())};
            }
            auto *image =
                reinterpret_cast<rhi::vulkan::Image *>(static_cast<uintptr_t>(std::move(resolvedImage).value()));
            info.imageView = image->view;
            info.imageLayout = slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
                                   ? VK_IMAGE_LAYOUT_GENERAL
                                   : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            const auto samplerIndex =
                bindings.samplerByBinding.find(descriptorBindingKey(slot.entry.layout.set, slot.entry.layout.binding));
            if (samplerIndex != bindings.samplerByBinding.end()) {
                const auto &sampler = snapshot->slots[samplerIndex->second];
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                if ((sampler.value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
                    info.sampler = bindings.device->defaultImplicitSampler.sampler;
                else {
                    auto resolvedSampler =
                        resolveCommandRhiResource(adapter, encoder, sampler.value.payload.sampler.resource);
                    if (!resolvedSampler) {
                        return RhiAdapterResult<PreparedBindingSet::Snapshot *>{
                            vernon::err(std::move(resolvedSampler).error())};
                    }
                    auto *nativeSampler = reinterpret_cast<rhi::vulkan::Sampler *>(
                        static_cast<uintptr_t>(std::move(resolvedSampler).value()));
                    info.sampler = nativeSampler->sampler;
                }
            }
            images.push_back(info);
            write.pImageInfo = &images.back();
        }
        writes.push_back(write);
    }
    size_t bufferIndex = 0;
    size_t imageIndex = 0;
    for (auto &write : writes) {
        if (write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
            write.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            write.pBufferInfo = &buffers[bufferIndex++];
        else
            write.pImageInfo = &images[imageIndex++];
    }
    if (!writes.empty())
        rhi::vulkan::driver().updateDescriptorSets(bindings.device->device, static_cast<uint32_t>(writes.size()),
                                                   writes.data(), 0, nullptr);

    PreparedBindingSet::Snapshot *result = snapshot.get();
    try {
        bindings.snapshots.emplace(key, std::move(snapshot));
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_binding_snapshot_allocation_failed", 0, 0}})};
    }
    adapter.bindingSnapshotCreations.fetch_add(1, std::memory_order_relaxed);
    if (!retainBindingSnapshot(adapter, encoder, bindings, *result)) {
        auto owned = std::move(bindings.snapshots.find(key)->second);
        bindings.snapshots.erase(key);
        return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_binding_snapshot_cleanup_registration_failed", 0, 0}})};
    }
    recordPushConstants(command, pipeline, bindings);
    return RhiAdapterResult<PreparedBindingSet::Snapshot *>{vernon::ok(result)};
}

RhiAdapterResult<void> createBindingsResult(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || !layout)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"vulkan_adapter_received_an_invalid_binding_set_descriptor", 0, 0}})};
    auto bindings = std::unique_ptr<PreparedBindingSet>(new (std::nothrow) PreparedBindingSet());
    if (!bindings)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_binding_preparation_ran_out_of_memory", 0, 0}})};
    bindings->device = &vulkanDevice(adapter);
    bindings->layout = layout;
    bindings->slots.reserve(layout->entries.size());
    bindings->slotIndices.reserve(layout->entries.size());
    bindings->samplerByBinding.reserve(layout->entries.size());
    bindings->valueIndices.resize(layout->entries.size());
    bindings->seenSlots.resize(layout->entries.size());
    bindings->pushConstantStorage.resize(layout->pushConstantSize);
    for (size_t index = 0; index < layout->entries.size(); ++index) {
        const auto &entry = layout->entries[index];
        PreparedBindingSet::Slot slot;
        slot.entry = entry;
        if (!bindings->slotIndices.emplace(entry.layout.slot, index).second)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"vulkan_binding_layout_contains_duplicate_slots", 0, 0}})};
        if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
            !bindings->samplerByBinding.emplace(descriptorBindingKey(entry.layout.set, entry.layout.binding), index)
                 .second)
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"vulkan_binding_layout_contains_duplicate_sampler_bindings", 0, 0}})};
        if (packedUniformBytes(entry.layout.kind, entry.layout.interface_kind))
            slot.inlineStorage.resize(entry.layout.element_size);
        bindings->slots.push_back(std::move(slot));
    }
    auto status = updateBindingsImplResult(adapter, *bindings, descriptor->values, descriptor->value_count);
    if (!status)
        return status;
    *output = toHandle(bindings.release());
    adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> updateBindingsResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_adapter_received_an_invalid_binding_set", 0, 0}})};
    std::lock_guard<std::mutex> guard(bindings->mutex);
    return updateBindingsImplResult(adapter, *bindings, values, valueCount);
}

void restoreImageLayout(void *context, uint64_t layout) {
    static_cast<rhi::vulkan::Image *>(context)->layout = static_cast<VkImageLayout>(layout);
}

RhiAdapterResult<void> transitionImage(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                       VkCommandBuffer command, rhi::vulkan::Image &image, VkImageLayout newLayout) {
    if (image.layout == newLayout)
        return RhiAdapterResult<void>{vernon::ok()};
    auto deferred =
        deferCommandRollback(adapter, encoder, &image, static_cast<uint64_t>(image.layout), restoreImageLayout);
    if (!deferred)
        return deferred;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = image.layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {rhi::vulkan::imageAspectMask(image.format), 0, VK_REMAINING_MIP_LEVELS, 0,
                                VK_REMAINING_ARRAY_LAYERS};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask =
        newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        : newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : VK_ACCESS_SHADER_READ_BIT;
    rhi::vulkan::driver().cmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                             &barrier);
    image.layout = newLayout;
    return RhiAdapterResult<void>{vernon::ok()};
}

void bindDescriptorSets(VkCommandBuffer command, PreparedPipeline &pipeline,
                        const PreparedBindingSet::Snapshot *snapshot, VkPipelineBindPoint bindPoint) {
    if (!snapshot)
        return;
    for (uint32_t set = 0; set < snapshot->descriptorSets.size(); ++set)
        if (snapshot->descriptorSets[set])
            rhi::vulkan::driver().cmdBindDescriptorSets(command, bindPoint, pipeline.pipelineLayout, set, 1,
                                                        &snapshot->descriptorSets[set], 0, nullptr);
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject commandEncoder,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || !pipeline || pipeline->graphics)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_adapter_received_an_invalid_dispatch", 0, 0}})};
    auto nativeCommand = nativeCommandEncoder(adapter, commandEncoder);
    if (!nativeCommand)
        return RhiAdapterResult<void>{vernon::err(std::move(nativeCommand).error())};
    const VkCommandBuffer command =
        reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(std::move(nativeCommand).value()));
    if (commandEncoderRendering(adapter, commandEncoder))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_dispatch_command_encoder_is_invalid", 0, 0}})};
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_dispatch_could_not_retain_provider_objects", 0, 0}})};
    PreparedBindingSet::Snapshot *bindingSnapshot = nullptr;
    if (bindings) {
        auto snapshot = snapshotBindings(adapter, commandEncoder, *pipeline, *bindings, command);
        if (!snapshot)
            return RhiAdapterResult<void>{vernon::err(std::move(snapshot).error())};
        bindingSnapshot = std::move(snapshot).value();
    }
    if (bindings)
        for (const auto &slot : bindings->slots)
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                auto resolved = resolveCommandRhiResource(adapter, commandEncoder, slot.value.payload.image.view);
                if (!resolved)
                    return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
                auto *image =
                    reinterpret_cast<rhi::vulkan::Image *>(static_cast<uintptr_t>(std::move(resolved).value()));
                const VkImageLayout layout = slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
                                                 ? VK_IMAGE_LAYOUT_GENERAL
                                                 : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                if (!image || !transitionImage(adapter, commandEncoder, command, *image, layout))
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                          {"vulkan_dispatch_could_not_track_image_layout", 0, 0}})};
            }
    auto &driver = rhi::vulkan::driver();
    driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    bindDescriptorSets(command, *pipeline, bindingSnapshot, VK_PIPELINE_BIND_POINT_COMPUTE);
    driver.cmdDispatch(command, descriptor->group_count[0], descriptor->group_count[1], descriptor->group_count[2]);
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_dispatch_command_encoder_state_changed", 0, 0}})};
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDrawResult(void *data, VernonRuntimeProviderObject commandEncoder,
                                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!validCommonDrawDescriptor(descriptor))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_adapter_received_an_invalid_draw_descriptor", 0, 0}})};
    if (!pipeline)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"vulkan_adapter_received_an_invalid_prepared_pipeline", 0, 0}})};
    if (!pipeline->graphics)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"vulkan_adapter_received_a_compute_pipeline_for_a_draw", 0, 0}})};
    if (!pipeline->pipeline)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"vulkan_adapter_received_an_uninitialized_graphics_pipeline", 0, 0}})};
    auto nativeCommand = nativeCommandEncoder(adapter, commandEncoder);
    if (!nativeCommand)
        return RhiAdapterResult<void>{vernon::err(std::move(nativeCommand).error())};
    const VkCommandBuffer command =
        reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(std::move(nativeCommand).value()));
    auto renderingClaimResult =
        claimCommandRendering(adapter, commandEncoder,
                              pipeline->device->dynamicRendering ? vernon::rhi::CommandRenderingDynamic
                                                                 : vernon::rhi::CommandRenderingRenderPass);
    if (!renderingClaimResult)
        return RhiAdapterResult<void>{vernon::err(std::move(renderingClaimResult).error())};
    const int renderingClaim = std::move(renderingClaimResult).value();
    const bool beginRendering = renderingClaim != 0;
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"vulkan_draw_could_not_retain_provider_objects", 0, 0}})};
    PreparedBindingSet::Snapshot *bindingSnapshot = nullptr;
    if (bindings) {
        auto snapshot = snapshotBindings(adapter, commandEncoder, *pipeline, *bindings, command);
        if (!snapshot)
            return RhiAdapterResult<void>{vernon::err(std::move(snapshot).error())};
        bindingSnapshot = std::move(snapshot).value();
    }
    std::array<VernonRhiLoadOperation, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorLoads{};
    std::array<VernonRhiStoreOperation, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorStores{};
    std::array<std::array<float, 4>, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorClears{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        colorLoads[index] = static_cast<VernonRhiLoadOperation>(descriptor->color_attachments[index].load_operation);
        colorStores[index] = static_cast<VernonRhiStoreOperation>(descriptor->color_attachments[index].store_operation);
        std::copy(std::begin(descriptor->color_attachments[index].clear_color),
                  std::end(descriptor->color_attachments[index].clear_color), colorClears[index].begin());
        auto operations = commandColorOperations(adapter, commandEncoder, index, colorLoads[index], colorStores[index],
                                                 colorClears[index].data());
        if (!operations)
            return RhiAdapterResult<void>{vernon::err(std::move(operations).error())};
    }
    VernonRhiLoadOperation depthLoad = static_cast<VernonRhiLoadOperation>(descriptor->depth_load_operation);
    VernonRhiStoreOperation depthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
    VernonRhiLoadOperation stencilLoad = static_cast<VernonRhiLoadOperation>(descriptor->stencil_load_operation);
    VernonRhiStoreOperation stencilStore = static_cast<VernonRhiStoreOperation>(descriptor->stencil_store_operation);
    float clearDepth = descriptor->clear_depth;
    uint32_t clearStencil = descriptor->clear_stencil;
    auto depthOperations = commandDepthOperations(adapter, commandEncoder, depthLoad, depthStore, stencilLoad,
                                                  stencilStore, clearDepth, clearStencil);
    if (!depthOperations)
        return RhiAdapterResult<void>{vernon::err(std::move(depthOperations).error())};
    const bool hasRenderArea = descriptor->render_area[2] && descriptor->render_area[3];
    const VkRect2D renderArea{
        {static_cast<int32_t>(hasRenderArea ? descriptor->render_area[0] : descriptor->viewport[0]),
         static_cast<int32_t>(hasRenderArea ? descriptor->render_area[1] : descriptor->viewport[1])},
        {hasRenderArea ? descriptor->render_area[2] : descriptor->viewport[2],
         hasRenderArea ? descriptor->render_area[3] : descriptor->viewport[3]}};
    const uint32_t framebufferWidth = static_cast<uint32_t>(renderArea.offset.x) + renderArea.extent.width;
    const uint32_t framebufferHeight = static_cast<uint32_t>(renderArea.offset.y) + renderArea.extent.height;
    std::array<VkRenderingAttachmentInfo, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> attachments{};
    std::array<VkImageView, 9> imageViews{};
    std::array<VkFormat, 9> attachmentFormats{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->color_attachments[index].view))
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"vulkan_draw_could_not_retain_its_color_attachment", 0, 0}})};
        auto resolved = resolveCommandRhiResource(adapter, commandEncoder, descriptor->color_attachments[index].view);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        auto *image = reinterpret_cast<rhi::vulkan::Image *>(static_cast<uintptr_t>(std::move(resolved).value()));
        if (beginRendering &&
            !transitionImage(adapter, commandEncoder, command, *image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL))
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"vulkan_draw_could_not_track_color_attachment_layout", 0, 0}})};
        imageViews[index] = image->view;
        attachmentFormats[index] = image->format;
        attachments[index] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                              nullptr,
                              image->view,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_RESOLVE_MODE_NONE,
                              {},
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              attachmentLoad(colorLoads[index]),
                              attachmentStore(colorStores[index]),
                              {}};
        std::copy(colorClears[index].begin(), colorClears[index].end(), attachments[index].clearValue.color.float32);
    }
    VkRenderingAttachmentInfo depthAttachment{};
    VkRenderingAttachmentInfo stencilAttachment{};
    bool hasStencilAttachment = false;
    if (descriptor->depth_stencil_view.resource.value) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_view))
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"vulkan_draw_could_not_retain_its_depth_attachment", 0, 0}})};
        auto resolved = resolveCommandRhiResource(adapter, commandEncoder, descriptor->depth_stencil_view);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        auto *image = reinterpret_cast<rhi::vulkan::Image *>(static_cast<uintptr_t>(std::move(resolved).value()));
        if (image->format != VK_FORMAT_D32_SFLOAT && image->format != VK_FORMAT_D32_SFLOAT_S8_UINT)
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"vulkan_draw_contains_an_invalid_depth_attachment", 0, 0}})};
        hasStencilAttachment = image->format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        if (!hasStencilAttachment &&
            (stencilLoad != VERNON_RHI_LOAD_DISCARD || stencilStore != VERNON_RHI_STORE_DISCARD || clearStencil))
            return RhiAdapterResult<void>{vernon::err(
                vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                      {"vulkan_draw_requests_stencil_operations_for_a_depth_only_attachment", 0, 0}})};
        if (beginRendering && !transitionImage(adapter, commandEncoder, command, *image,
                                               VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL))
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"vulkan_draw_could_not_track_depth_attachment_layout", 0, 0}})};
        imageViews[descriptor->color_attachment_count] = image->view;
        attachmentFormats[descriptor->color_attachment_count] = image->format;
        depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                           nullptr,
                           image->view,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                           VK_RESOLVE_MODE_NONE,
                           {},
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           attachmentLoad(depthLoad),
                           attachmentStore(depthStore),
                           {}};
        depthAttachment.clearValue.depthStencil = {clearDepth, clearStencil};
        if (hasStencilAttachment) {
            stencilAttachment = depthAttachment;
            stencilAttachment.loadOp = attachmentLoad(stencilLoad);
            stencilAttachment.storeOp = attachmentStore(stencilStore);
        }
    }
    const VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO,
                                    nullptr,
                                    0,
                                    renderArea,
                                    1,
                                    0,
                                    static_cast<uint32_t>(descriptor->color_attachment_count),
                                    attachments.data(),
                                    descriptor->depth_stencil_view.resource.value ? &depthAttachment : nullptr,
                                    hasStencilAttachment ? &stencilAttachment : nullptr};
    auto &driver = rhi::vulkan::driver();
    if (bindingSnapshot && beginRendering)
        for (const auto &slot : bindingSnapshot->slots)
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
                auto resolved = resolveCommandRhiResource(adapter, commandEncoder, slot.value.payload.image.view);
                if (!resolved)
                    return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
                auto *image =
                    reinterpret_cast<rhi::vulkan::Image *>(static_cast<uintptr_t>(std::move(resolved).value()));
                if (!transitionImage(adapter, commandEncoder, command, *image,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                          {"vulkan_draw_could_not_track_sampled_image_layout", 0, 0}})};
            }
    VkRenderPass scopeRenderPass{};
    VkFramebuffer framebuffer{};
    if (pipeline->device->dynamicRendering && beginRendering)
        driver.cmdBeginRendering(command, &rendering);
    else if (!pipeline->device->dynamicRendering && beginRendering) {
        PreparedPipeline::RenderingCacheEntry cacheKey{};
        cacheKey.colorCount = static_cast<uint32_t>(descriptor->color_attachment_count);
        cacheKey.width = framebufferWidth;
        cacheKey.height = framebufferHeight;
        cacheKey.hasDepth = descriptor->depth_stencil_view.resource.value != 0;
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            cacheKey.resources[index] = descriptor->color_attachments[index].view;
            cacheKey.formats[index] = attachmentFormats[index];
            cacheKey.loads[index] = colorLoads[index];
            cacheKey.stores[index] = colorStores[index];
        }
        if (cacheKey.hasDepth) {
            const size_t index = descriptor->color_attachment_count;
            cacheKey.resources[index] = descriptor->depth_stencil_view;
            cacheKey.formats[index] = attachmentFormats[index];
            cacheKey.loads[index] = depthLoad;
            cacheKey.stores[index] = depthStore;
            cacheKey.stencilLoad = stencilLoad;
            cacheKey.stencilStore = stencilStore;
        }
        const auto sameReference = [](const VernonRuntimeProviderResourceReference &left,
                                      const VernonRuntimeProviderResourceReference &right) {
            return left.identity == right.identity && left.resource.value == right.resource.value;
        };
        const auto sameCacheKey = [&](const PreparedPipeline::RenderingCacheEntry &entry) {
            if (entry.colorCount != cacheKey.colorCount || entry.width != cacheKey.width ||
                entry.height != cacheKey.height || entry.hasDepth != cacheKey.hasDepth ||
                entry.stencilLoad != cacheKey.stencilLoad || entry.stencilStore != cacheKey.stencilStore)
                return false;
            const size_t count = cacheKey.colorCount + (cacheKey.hasDepth ? 1u : 0u);
            for (size_t index = 0; index < count; ++index)
                if (!sameReference(entry.resources[index], cacheKey.resources[index]) ||
                    entry.formats[index] != cacheKey.formats[index] || entry.loads[index] != cacheKey.loads[index] ||
                    entry.stores[index] != cacheKey.stores[index])
                    return false;
            return true;
        };
        std::lock_guard<std::mutex> cacheGuard(pipeline->renderingCacheMutex);
        auto cached = std::find_if(pipeline->renderingCache.begin(), pipeline->renderingCache.end(), sameCacheKey);
        const bool cacheHit = cached != pipeline->renderingCache.end();
        bool cacheNewEntry = !cacheHit;
        if (cacheHit) {
            scopeRenderPass = cached->renderPass;
            framebuffer = cached->framebuffer;
            cached->lastEncoder = commandEncoder.value;
        } else if (pipeline->renderingCache.size() >= maxRenderingCacheEntries) {
            const auto victim =
                std::find_if(pipeline->renderingCache.begin(), pipeline->renderingCache.end(),
                             [&](const auto &entry) { return entry.lastEncoder != commandEncoder.value; });
            if (victim == pipeline->renderingCache.end())
                cacheNewEntry = false;
            else {
                driver.destroyFramebuffer(pipeline->device->device, victim->framebuffer, nullptr);
                driver.destroyRenderPass(pipeline->device->device, victim->renderPass, nullptr);
                const size_t count = victim->colorCount + (victim->hasDepth ? 1u : 0u);
                for (size_t index = 0; index < count; ++index) {
                    auto released = releaseRetainedRhiResource(adapter, victim->resources[index]);
                    (void)released;
                }
                pipeline->renderingCache.erase(victim);
            }
        }
        std::array<VkAttachmentDescription, 9> renderPassAttachments{};
        std::array<VkAttachmentReference, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> renderPassReferences{};
        VkAttachmentReference depthReference{};
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            renderPassAttachments[index] = attachmentDescription(
                attachmentFormats[index], VK_SAMPLE_COUNT_1_BIT, attachmentLoad(colorLoads[index]),
                attachmentStore(colorStores[index]), VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            renderPassReferences[index] = {static_cast<uint32_t>(index), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        const bool hasDepth = descriptor->depth_stencil_view.resource.value != 0;
        if (hasDepth) {
            renderPassAttachments[descriptor->color_attachment_count] = attachmentDescription(
                attachmentFormats[descriptor->color_attachment_count], VK_SAMPLE_COUNT_1_BIT, attachmentLoad(depthLoad),
                attachmentStore(depthStore), attachmentLoad(stencilLoad), attachmentStore(stencilStore),
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            depthReference = {static_cast<uint32_t>(descriptor->color_attachment_count),
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        }
        const VkSubpassDescription subpass{0,
                                           VK_PIPELINE_BIND_POINT_GRAPHICS,
                                           0,
                                           nullptr,
                                           static_cast<uint32_t>(descriptor->color_attachment_count),
                                           renderPassReferences.data(),
                                           nullptr,
                                           hasDepth ? &depthReference : nullptr,
                                           0,
                                           nullptr};
        const VkRenderPassCreateInfo renderPassInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(descriptor->color_attachment_count + (hasDepth ? 1 : 0)),
            renderPassAttachments.data(),
            1,
            &subpass,
            0,
            nullptr};
        VkResult result = VK_SUCCESS;
        if (!cacheHit)
            result = driver.createRenderPass(pipeline->device->device, &renderPassInfo, nullptr, &scopeRenderPass);
        if (result != VK_SUCCESS)
            return failureResult(result, "vkCreateRenderPass");
        const VkFramebufferCreateInfo framebufferInfo{
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            nullptr,
            0,
            scopeRenderPass,
            static_cast<uint32_t>(descriptor->color_attachment_count +
                                  (descriptor->depth_stencil_view.resource.value ? 1 : 0)),
            imageViews.data(),
            framebufferWidth,
            framebufferHeight,
            1};
        if (!cacheHit)
            result = driver.createFramebuffer(pipeline->device->device, &framebufferInfo, nullptr, &framebuffer);
        if (result != VK_SUCCESS) {
            driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
            return failureResult(result, "vkCreateFramebuffer");
        }
        if (!cacheHit && cacheNewEntry) {
            const size_t count = cacheKey.colorCount + (cacheKey.hasDepth ? 1u : 0u);
            size_t retained = 0;
            for (; retained < count; ++retained)
                if (!retainRhiResource(adapter, cacheKey.resources[retained]))
                    break;
            if (retained != count) {
                while (retained) {
                    auto released = releaseRetainedRhiResource(adapter, cacheKey.resources[--retained]);
                    (void)released;
                }
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                      {"vulkan_rendering_cache_could_not_retain_attachments", 0, 0}})};
            }
            cacheKey.lastEncoder = commandEncoder.value;
            cacheKey.renderPass = scopeRenderPass;
            cacheKey.framebuffer = framebuffer;
            try {
                pipeline->renderingCache.push_back(cacheKey);
            } catch (const std::bad_alloc &) {
                while (retained) {
                    auto released = releaseRetainedRhiResource(adapter, cacheKey.resources[--retained]);
                    (void)released;
                }
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::BackendFailure, {"vulkan_rendering_cache_allocation_failed", 0, 0}})};
            }
        } else if (!cacheHit) {
            if (!deferCommandCleanup(adapter, commandEncoder, pipeline->device, nativeHandleBits(scopeRenderPass),
                                     destroyRecordedRenderPass)) {
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                          {"vulkan_temporary_render_pass_cleanup_allocation_failed", 0, 0}})};
            }
            if (!deferCommandCleanup(adapter, commandEncoder, pipeline->device, nativeHandleBits(framebuffer),
                                     destroyRecordedFramebuffer)) {
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                          {"vulkan_temporary_framebuffer_cleanup_allocation_failed", 0, 0}})};
            }
        }
        std::array<VkClearValue, 9> clearValues{};
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            std::copy(colorClears[index].begin(), colorClears[index].end(), clearValues[index].color.float32);
        if (descriptor->depth_stencil_view.resource.value)
            clearValues[descriptor->color_attachment_count].depthStencil = {clearDepth, clearStencil};
        const VkRenderPassBeginInfo begin{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr,
            scopeRenderPass,
            framebuffer,
            renderArea,
            static_cast<uint32_t>(descriptor->color_attachment_count +
                                  (descriptor->depth_stencil_view.resource.value ? 1 : 0)),
            clearValues.data()};
        driver.cmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    }
    driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
    bindDescriptorSets(command, *pipeline, bindingSnapshot, VK_PIPELINE_BIND_POINT_GRAPHICS);
    if (bindingSnapshot)
        for (const auto &slot : bindingSnapshot->slots) {
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
                auto resolved = resolveCommandRhiResource(adapter, commandEncoder, slot.value.payload.image.view);
                if (!resolved)
                    return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
            } else if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                auto resolved = resolveCommandRhiResource(adapter, commandEncoder, slot.value.payload.buffer.resource);
                if (!resolved)
                    return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
                auto *buffer =
                    reinterpret_cast<rhi::vulkan::Buffer *>(static_cast<uintptr_t>(std::move(resolved).value()));
                const VkDeviceSize offset = slot.value.payload.buffer.resource.offset;
                driver.cmdBindVertexBuffers(command, slot.entry.layout.binding, 1, &buffer->buffer, &offset);
            }
        }
    const VkViewport viewport{static_cast<float>(descriptor->viewport[0]),
                              static_cast<float>(descriptor->viewport[1] + descriptor->viewport[3]),
                              static_cast<float>(descriptor->viewport[2]),
                              -static_cast<float>(descriptor->viewport[3]),
                              0,
                              1};
    const VkRect2D scissor{{static_cast<int32_t>(descriptor->scissor[0]), static_cast<int32_t>(descriptor->scissor[1])},
                           {descriptor->scissor[2], descriptor->scissor[3]}};
    driver.cmdSetViewport(command, 0, 1, &viewport);
    driver.cmdSetScissor(command, 0, 1, &scissor);
    driver.cmdSetStencilReference(command, VK_STENCIL_FACE_FRONT_AND_BACK, descriptor->stencil_reference);
    adapter.lastStencilReference.store(descriptor->stencil_reference, std::memory_order_relaxed);
    adapter.lastDrawIndexed.store(descriptor->index_count != 0, std::memory_order_relaxed);
    if (descriptor->index_count) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->index_buffer))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"vulkan_draw_could_not_retain_its_index_buffer", 0, 0}})};
        auto resolved = resolveCommandRhiResource(adapter, commandEncoder, descriptor->index_buffer);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        auto *buffer = reinterpret_cast<rhi::vulkan::Buffer *>(static_cast<uintptr_t>(std::move(resolved).value()));
        driver.cmdBindIndexBuffer(command, buffer->buffer, descriptor->index_buffer.offset, VK_INDEX_TYPE_UINT32);
        driver.cmdDrawIndexed(command, descriptor->index_count, descriptor->instance_count, 0, descriptor->first_vertex,
                              descriptor->first_instance);
    } else
        driver.cmdDraw(command, descriptor->vertex_count, descriptor->instance_count, descriptor->first_vertex,
                       descriptor->first_instance);
    if (!recordProviderCommand(adapter, commandEncoder, true))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"vulkan_draw_command_encoder_state_changed", 0, 0}})};
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> destroyShaderResult(void *, VernonRuntimeProviderObject handle) {
    auto *shader = fromHandle<PreparedShader>(handle);
    if (!shader)
        return RhiAdapterResult<void>{vernon::ok()};
    rhi::vulkan::driver().destroyShaderModule(shader->device->device, shader->module, nullptr);
    delete shader;
    return RhiAdapterResult<void>{vernon::ok()};
}
RhiAdapterResult<void> destroyLayoutResult(void *, VernonRuntimeProviderObject handle) {
    auto *layout = fromHandle<PreparedLayout>(handle);
    if (!layout)
        return RhiAdapterResult<void>{vernon::ok()};
    delete layout;
    return RhiAdapterResult<void>{vernon::ok()};
}
PreparedLayout::~PreparedLayout() {
    if (!device)
        return;
    for (VkDescriptorSetLayout descriptorSetLayout : descriptorSetLayouts)
        if (descriptorSetLayout)
            rhi::vulkan::driver().destroyDescriptorSetLayout(device->device, descriptorSetLayout, nullptr);
}
PreparedPipeline::~PreparedPipeline() {
    if (adapter)
        adapter->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
    if (!device)
        return;
    for (auto &entry : renderingCache) {
        rhi::vulkan::driver().destroyFramebuffer(device->device, entry.framebuffer, nullptr);
        rhi::vulkan::driver().destroyRenderPass(device->device, entry.renderPass, nullptr);
        const size_t count = entry.colorCount + (entry.hasDepth ? 1u : 0u);
        for (size_t index = 0; index < count; ++index) {
            auto released = releaseRetainedRhiResource(*adapter, entry.resources[index]);
            (void)released;
        }
    }
    if (pipeline)
        rhi::vulkan::driver().destroyPipeline(device->device, pipeline, nullptr);
    if (renderPass)
        rhi::vulkan::driver().destroyRenderPass(device->device, renderPass, nullptr);
    for (VkShaderModule module : specializedShaderModules)
        rhi::vulkan::driver().destroyShaderModule(device->device, module, nullptr);
    if (pipelineLayout)
        rhi::vulkan::driver().destroyPipelineLayout(device->device, pipelineLayout, nullptr);
}
void releaseCommandPipeline(void *context, uint64_t) {
    auto *pipeline = static_cast<PreparedPipeline *>(context);
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete pipeline;
}
RhiAdapterResult<void> destroyPipelineResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
}
void releaseCommandBindings(void *context, uint64_t) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    if (!bindings || bindings->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete bindings;
}
RhiAdapterResult<void> destroyBindingsResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
}

void destroyShader(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyShaderResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "Vulkan shader destruction failed");
}
void destroyLayout(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyLayoutResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "Vulkan layout destruction failed");
}
void destroyPipeline(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyPipelineResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "Vulkan pipeline destruction failed");
}
void destroyBindings(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyBindingsResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "Vulkan binding-set destruction failed");
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "Vulkan shader preparation failed");
}
VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareLayoutResult(data, descriptor, output), "Vulkan layout preparation failed");
}
VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, preparePipelineResult(data, descriptor, output),
                          "Vulkan pipeline preparation failed");
}
VernonStatus createBindings(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createBindingsResult(data, descriptor, output),
                          "Vulkan binding-set creation failed");
}
VernonStatus updateBindings(void *data, VernonRuntimeProviderObject handle,
                            const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, updateBindingsResult(data, handle, values, valueCount),
                          "Vulkan binding-set update failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject encoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, encoder, descriptor), "Vulkan dispatch encoding failed");
}
VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject encoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDrawResult(data, encoder, descriptor), "Vulkan draw encoding failed");
}

} // namespace

void initializeVulkanProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainRhiResourceCallback;
    adapter.provider.release_resource = releaseRhiResourceCallback;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_binding_set = createBindings;
    adapter.provider.update_binding_set = updateBindings;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_binding_set = destroyBindings;
}

} // namespace vernon::runtime::rhi_adapter

namespace vernon::runtime {

VernonRuntimeRhiAdapter *createVulkanRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend) {
    if (backend != VERNON_RHI_BACKEND_VULKAN)
        return nullptr;
    auto *deviceState = static_cast<rhi::vulkan::DeviceState *>(rhi::deviceState(device, backend));
    if (!deviceState)
        return nullptr;
    auto state = std::unique_ptr<rhi_adapter::VulkanAdapterState>(new (std::nothrow) rhi_adapter::VulkanAdapterState());
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!state || !adapter)
        return nullptr;
    state->device = deviceState;
    adapter->rhiBackend = backend;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    [[maybe_unused]] auto *adoptedState = state.release();
    rhi_adapter::initializeVulkanProvider(*adapter);
    return adapter.release();
}

} // namespace vernon::runtime

#endif
