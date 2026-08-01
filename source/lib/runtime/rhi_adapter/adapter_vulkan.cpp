#include "../vertex_attribute_capabilities.h"
#include "adapter_common.h"

#if defined(VERNON_HAS_VULKAN_RHI)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

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
    VkDescriptorSetLayout descriptorSetLayout{};
    uint32_t pushConstantSize{};
    bool hasDescriptors{};
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
        VkDescriptorSet descriptorSet{};
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
    std::unordered_map<uint32_t, size_t> samplerByBinding;
    std::vector<size_t> valueIndices;
    std::vector<uint8_t> seenSlots;
    std::vector<uint8_t> pushConstantStorage;
    std::unordered_map<SnapshotKey, std::unique_ptr<Snapshot>, SnapshotKeyHash> snapshots;
    uint64_t resourceRevision{1};
    std::mutex mutex;

    ~PreparedBindingSet();
};

void releaseCommandBindings(void *context, uint64_t);
void releaseCommandBindingSnapshot(void *context, uint64_t object);
void releaseCommandPipeline(void *context, uint64_t);

bool retainCommandObjects(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                          PreparedPipeline &pipeline, PreparedBindingSet *bindings) {
    pipeline.references.fetch_add(1, std::memory_order_relaxed);
    if (!deferCommandCleanup(adapter, encoder, &pipeline, 0, releaseCommandPipeline)) {
        releaseCommandPipeline(&pipeline, 0);
        return false;
    }
    if (bindings) {
        bindings->references.fetch_add(1, std::memory_order_relaxed);
        if (!deferCommandCleanup(adapter, encoder, bindings, 0, releaseCommandBindings)) {
            releaseCommandBindings(bindings, 0);
            return false;
        }
    }
    return true;
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

VernonStatus failure(VernonRuntimeRhiAdapter &adapter, VkResult result, const char *operation) {
    return result == VK_SUCCESS
               ? VERNON_STATUS_OK
               : fail(adapter, std::string(operation) + " failed with VkResult " + std::to_string(result),
                      VERNON_STATUS_INTERNAL_ERROR);
}

uint32_t getCapabilities(void *) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return {0x56554c4b414eu, reinterpret_cast<uintptr_t>(adapter.vulkanDevice->physicalDevice),
            adapter.vulkanDevice->apiVersion};
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !descriptor->data ||
        descriptor->size < sizeof(uint32_t) || descriptor->size % sizeof(uint32_t) != 0 || !descriptor->entry.data ||
        descriptor->entry.size == 0 || descriptor->format.size != 5 ||
        std::memcmp(descriptor->format.data, "spirv", 5) != 0)
        return fail(adapter, "Vulkan adapter received an invalid SPIR-V shader");
    auto shader = std::unique_ptr<PreparedShader>(new (std::nothrow) PreparedShader());
    if (!shader)
        return fail(adapter, "Vulkan shader preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    shader->device = adapter.vulkanDevice;
    shader->stage = descriptor->stage;
    shader->entry.assign(descriptor->entry.data, descriptor->entry.size);
    shader->words.assign(static_cast<const uint32_t *>(descriptor->data),
                         static_cast<const uint32_t *>(descriptor->data) + descriptor->size / sizeof(uint32_t));
    const VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, descriptor->size,
                                              static_cast<const uint32_t *>(descriptor->data)};
    const VkResult result =
        rhi::vulkan::driver().createShaderModule(shader->device->device, &createInfo, nullptr, &shader->module);
    if (result != VK_SUCCESS)
        return failure(adapter, result, "vkCreateShaderModule");
    *output = toHandle(shader.release());
    adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return fail(adapter, "Vulkan adapter received an invalid pipeline layout");
    auto layout = std::unique_ptr<PreparedLayout>(new (std::nothrow) PreparedLayout());
    if (!layout)
        return fail(adapter, "Vulkan layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    layout->device = adapter.vulkanDevice;
    std::vector<VkDescriptorSetLayoutBinding> nativeBindings;
    try {
        layout->entries.reserve(descriptor->binding_count);
        for (size_t index = 0; index < descriptor->binding_count; ++index) {
            const auto &source = descriptor->bindings[index];
            const bool supported = source.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
            if (!supported || source.array_count != 1)
                return fail(adapter, "Vulkan layout contains an unsupported binding", VERNON_STATUS_UNSUPPORTED_TARGET);
            PreparedLayout::Entry entry;
            entry.layout = source;
            if (source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
                if (source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
                    nativeBindings.push_back(
                        {source.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
            } else if (source.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                const auto existing =
                    std::find_if(nativeBindings.begin(), nativeBindings.end(),
                                 [&](const auto &binding) { return binding.binding == source.binding; });
                if (existing == nativeBindings.end())
                    nativeBindings.push_back(
                        {source.binding, descriptorType(source.kind), 1, stages(source.stage_mask), nullptr});
                else if ((existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
                          source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) ||
                         (existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER &&
                          source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)) {
                    existing->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    existing->stageFlags |= stages(source.stage_mask);
                } else
                    return fail(adapter, "Vulkan layout contains duplicate descriptor bindings");
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
                return fail(adapter, "Vulkan inline uniform alignment is invalid");
            entry.inlineOffset = (layout->pushConstantSize + alignment - 1) & ~(alignment - 1);
            if (entry.layout.element_size > UINT32_MAX - entry.inlineOffset)
                return fail(adapter, "Vulkan push-constant layout size overflow");
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
                return fail(adapter, "Vulkan vertex attribute references an unknown binding");
            if (!validateVertexAttributeCapability(VertexAttributeBackend::Vulkan, attribute,
                                                   layout->device->maxVertexInputAttributes, true,
                                                   capabilityDiagnostic))
                return fail(adapter, std::move(capabilityDiagnostic));
            VkFormatProperties properties{};
            if (format != VK_FORMAT_UNDEFINED)
                rhi::vulkan::driver().getPhysicalDeviceFormatProperties(layout->device->physicalDevice, format,
                                                                        &properties);
            if (format == VK_FORMAT_UNDEFINED || !(properties.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT))
                return fail(adapter, "Vulkan vertex attribute format is unsupported by the selected device");
            layout->vertexAttributes.push_back(attribute);
        }
    } catch (const std::bad_alloc &) {
        return fail(adapter, "Vulkan layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
    if (layout->pushConstantSize > layout->device->maxPushConstantsSize)
        return fail(adapter, "Vulkan push constants exceed the device limit", VERNON_STATUS_UNSUPPORTED_TARGET);
    const VkDescriptorSetLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
                                                     static_cast<uint32_t>(nativeBindings.size()),
                                                     nativeBindings.data()};
    layout->hasDescriptors = !nativeBindings.empty();
    const VkResult result = rhi::vulkan::driver().createDescriptorSetLayout(layout->device->device, &createInfo,
                                                                            nullptr, &layout->descriptorSetLayout);
    if (result != VK_SUCCESS)
        return failure(adapter, result, "vkCreateDescriptorSetLayout");
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus createPipelineLayout(VernonRuntimeRhiAdapter &adapter, PreparedPipeline &pipeline) {
    VkPushConstantRange range{VK_SHADER_STAGE_ALL, 0, pipeline.layout->pushConstantSize};
    const VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                nullptr,
                                                0,
                                                1,
                                                &pipeline.layout->descriptorSetLayout,
                                                pipeline.layout->pushConstantSize ? 1u : 0u,
                                                pipeline.layout->pushConstantSize ? &range : nullptr};
    return failure(adapter,
                   rhi::vulkan::driver().createPipelineLayout(pipeline.device->device, &createInfo, nullptr,
                                                              &pipeline.pipelineLayout),
                   "vkCreatePipelineLayout");
}

VernonStatus preparePipelineImpl(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                 VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout || !descriptor->shaders ||
        descriptor->shader_count == 0)
        return fail(adapter, "Vulkan adapter received an invalid pipeline");
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return fail(adapter, "Vulkan pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    pipeline->adapter = &adapter;
    pipeline->device = adapter.vulkanDevice;
    pipeline->layout = layout;
    pipeline->graphics = descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    pipeline->specializedShaderModules.reserve(2);
    VernonStatus status = createPipelineLayout(adapter, *pipeline);
    if (status != VERNON_STATUS_OK)
        return status;
    if (pipeline->graphics && descriptor->color_format_count == 0) {
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    }
    if (!pipeline->graphics) {
        auto *shader = descriptor->shader_count == 1 ? fromHandle<PreparedShader>(descriptor->shaders[0]) : nullptr;
        if (!shader || shader->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
            return fail(adapter, "Vulkan compute pipeline requires one compute shader");
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    nullptr,
                                                    0,
                                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                                    shader->module,
                                                    shader->entry.c_str(),
                                                    nullptr};
        const VkComputePipelineCreateInfo createInfo{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage, pipeline->pipelineLayout, {}, 0};
        status = failure(adapter,
                         rhi::vulkan::driver().createComputePipelines(pipeline->device->device, {}, 1, &createInfo,
                                                                      nullptr, &pipeline->pipeline),
                         "vkCreateComputePipelines");
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
            return fail(adapter, "Vulkan graphics provider requires vertex and fragment shaders");
        if (descriptor->sample_count != 1 || descriptor->color_format_count > 8 ||
            (descriptor->depth_stencil_format && descriptor->depth_stencil_format != VK_FORMAT_D32_SFLOAT &&
             descriptor->depth_stencil_format != VK_FORMAT_D32_SFLOAT_S8_UINT))
            return fail(adapter, "Vulkan graphics pipeline uses an unsupported attachment configuration");
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
                return VERNON_STATUS_OK;
            }
            std::vector<uint32_t> words = shader.words;
            if (!relocatePushConstantOffsets(words, offset))
                return fail(adapter, "Vulkan provider could not relocate push-constant offsets");
            const VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                                      words.size() * sizeof(uint32_t), words.data()};
            const VkResult result =
                rhi::vulkan::driver().createShaderModule(pipeline->device->device, &moduleInfo, nullptr, &module);
            if (result != VK_SUCCESS)
                return failure(adapter, result, "vkCreateShaderModule for relocated push constants");
            pipeline->specializedShaderModules.push_back(module);
            return VERNON_STATUS_OK;
        };
        VkShaderModule vertexModule{};
        VkShaderModule fragmentModule{};
        status = specializeShader(*vertex, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_VERTEX), vertexModule);
        if (status == VERNON_STATUS_OK)
            status = specializeShader(*fragment, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT), fragmentModule);
        if (status != VERNON_STATUS_OK)
            return status;
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
                return fail(adapter, "Vulkan graphics pipeline is missing a vertex stride");
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
            return fail(adapter, "Vulkan graphics pipeline contains an unsupported rasterization state");
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
            return fail(adapter, "Vulkan graphics pipeline contains an invalid depth/stencil state");
        const auto stencilFace = [](const VernonRuntimeProviderStencilFaceState &face) {
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
            return fail(adapter, "Vulkan graphics pipeline contains an invalid stencil state");
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
        std::array<VkPipelineColorBlendAttachmentState, 8> blendAttachments{};
        if (descriptor->color_blend_count != descriptor->color_format_count ||
            (descriptor->color_blend_count && !descriptor->color_blends))
            return fail(adapter, "Vulkan graphics pipeline blend state does not match its attachments");
        for (size_t index = 0; index < descriptor->color_format_count; ++index) {
            const auto &source = descriptor->color_blends[index];
            if (source.blend_enabled > 1 || source.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
                source.color_operation > VERNON_RHI_BLEND_MAXIMUM ||
                source.alpha_operation > VERNON_RHI_BLEND_MAXIMUM || (source.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
                return fail(adapter, "Vulkan graphics pipeline contains an invalid blend state");
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
        std::array<VkAttachmentReference, 8> renderPassReferences{};
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
            status = failure(adapter,
                             rhi::vulkan::driver().createRenderPass(pipeline->device->device, &renderPassInfo, nullptr,
                                                                    &pipeline->renderPass),
                             "vkCreateRenderPass");
            if (status != VERNON_STATUS_OK)
                return status;
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
        status = failure(adapter,
                         rhi::vulkan::driver().createGraphicsPipelines(pipeline->device->device, {}, 1, &createInfo,
                                                                       nullptr, &pipeline->pipeline),
                         "vkCreateGraphicsPipelines");
    }
    if (status != VERNON_STATUS_OK)
        return status;
    *output = toHandle(pipeline.release());
    adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    try {
        return preparePipelineImpl(data, descriptor, output);
    } catch (const std::bad_alloc &) {
        return fail(*static_cast<VernonRuntimeRhiAdapter *>(data), "Vulkan pipeline preparation ran out of memory",
                    VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus retainResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return retainRhiResource(adapter, resource) ? VERNON_STATUS_OK
                                                : fail(adapter, "Vulkan adapter received a stale resource");
}

void releaseResource(void *data, VernonRuntimeProviderResourceReference resource) {
    releaseRhiResource(*static_cast<VernonRuntimeRhiAdapter *>(data), resource);
}

VernonStatus updateBindingsImpl(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return fail(adapter, "Vulkan binding values do not match the prepared layout");
    std::fill(bindings.seenSlots.begin(), bindings.seenSlots.end(), uint8_t{0});
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = bindings.slotIndices.find(values[index].slot);
        if (found == bindings.slotIndices.end() || bindings.seenSlots[found->second])
            return fail(adapter, "Vulkan binding slot is invalid or duplicated");
        bindings.seenSlots[found->second] = 1;
        bindings.valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        if (value->kind != slot.entry.layout.kind)
            return fail(adapter, "Vulkan binding slot or kind is invalid");
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            if (!value->inline_data || value->inline_size != slot.inlineStorage.size())
                return fail(adapter, "Vulkan inline or uniform-buffer binding size is invalid");
        }
    }
    bool changed = false;
    bool resourcesChanged = false;
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto &value = values[bindings.valueIndices[index]];
        const bool slotChanged =
            slot.value.slot != value.slot || slot.value.kind != value.kind ||
            std::memcmp(&slot.value.resource, &value.resource, sizeof(value.resource)) != 0 ||
            slot.value.inline_size != value.inline_size || slot.value.flags != value.flags ||
            slot.value.stride != value.stride ||
            (value.inline_data && std::memcmp(slot.inlineStorage.data(), value.inline_data, value.inline_size));
        changed |= slotChanged;
        resourcesChanged |= slotChanged && slot.entry.inlineOffset == UINT32_MAX;
    }
    if (!changed)
        return VERNON_STATUS_OK;
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        slot.value = values[bindings.valueIndices[index]];
        if (slot.value.inline_data) {
            std::memcpy(slot.inlineStorage.data(), slot.value.inline_data, slot.value.inline_size);
            if (slot.entry.inlineOffset != UINT32_MAX)
                std::memcpy(bindings.pushConstantStorage.data() + slot.entry.inlineOffset, slot.value.inline_data,
                            slot.value.inline_size);
            slot.value.inline_data = nullptr;
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
    return VERNON_STATUS_OK;
}

bool retainBindingResources(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                            const PreparedBindingSet *bindings) {
    if (!bindings)
        return true;
    for (const auto &slot : bindings->slots)
        if (slot.value.resource.resource.value && !retainCommandResource(adapter, encoder, slot.value.resource))
            return false;
    return true;
}

PreparedBindingSet::Snapshot::~Snapshot() {
    if (!device)
        return;
    if (inlineBuffer.buffer)
        device->destroyBuffer(inlineBuffer);
    if (descriptorSet) {
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

bool retainBindingSnapshot(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                           PreparedBindingSet &bindings, PreparedBindingSet::Snapshot &snapshot) {
    ++snapshot.commandReferences;
    if (deferCommandCleanup(adapter, encoder, &bindings, reinterpret_cast<uintptr_t>(&snapshot),
                            releaseCommandBindingSnapshot))
        return true;
    --snapshot.commandReferences;
    fail(adapter, "Vulkan binding snapshot cleanup allocation failed", VERNON_STATUS_INTERNAL_ERROR);
    return false;
}

PreparedBindingSet::Snapshot *snapshotBindings(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                               PreparedPipeline &pipeline, PreparedBindingSet &bindings,
                                               VkCommandBuffer command) {
    std::lock_guard<std::mutex> guard(bindings.mutex);
    if (!retainBindingResources(adapter, encoder, &bindings))
        return nullptr;
    const PreparedBindingSet::SnapshotKey key{bindings.resourceRevision};
    const auto existing = bindings.snapshots.find(key);
    if (existing != bindings.snapshots.end()) {
        if (!retainBindingSnapshot(adapter, encoder, bindings, *existing->second))
            return nullptr;
        recordPushConstants(command, pipeline, bindings);
        return existing->second.get();
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
        if (slot.entry.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
            (slot.entry.layout.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE || slot.entry.inlineOffset != UINT32_MAX))
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
                                           adapter.error))
            return nullptr;
        void *mapped = nullptr;
        const VkResult result = rhi::vulkan::driver().mapMemory(bindings.device->device, snapshot->inlineBuffer.memory,
                                                                0, inlineSize, 0, &mapped);
        if (result != VK_SUCCESS) {
            failure(adapter, result, "vkMapMemory");
            return nullptr;
        }
        for (size_t index = 0; index < snapshot->slots.size(); ++index)
            if (snapshot->inlineOffsets[index] != VK_WHOLE_SIZE)
                std::memcpy(static_cast<uint8_t *>(mapped) + snapshot->inlineOffsets[index],
                            snapshot->slots[index].inlineStorage.data(), snapshot->slots[index].inlineStorage.size());
        rhi::vulkan::driver().unmapMemory(bindings.device->device, snapshot->inlineBuffer.memory);
    }
    if (bindings.layout->hasDescriptors &&
        !bindings.device->allocateDescriptorSet(bindings.layout->descriptorSetLayout, snapshot->descriptorSet,
                                                adapter.error))
        return nullptr;

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
        write.dstSet = snapshot->descriptorSet;
        write.dstBinding = slot.entry.layout.binding;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType(slot.entry.layout.kind);
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
            const bool internallyOwned = slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                                         slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
            rhi::vulkan::Buffer *buffer =
                internallyOwned
                    ? &snapshot->inlineBuffer
                    : reinterpret_cast<rhi::vulkan::Buffer *>(resolveRhiResource(adapter, slot.value.resource));
            if (!buffer) {
                fail(adapter, "Vulkan buffer binding is stale");
                return nullptr;
            }
            buffers.push_back({buffer->buffer,
                               internallyOwned ? snapshot->inlineOffsets[index] : slot.value.resource.offset,
                               internallyOwned ? slot.inlineStorage.size() : slot.value.resource.size});
            write.pBufferInfo = &buffers.back();
        } else {
            VkDescriptorImageInfo info{};
            auto *image = reinterpret_cast<rhi::vulkan::Image *>(resolveRhiResource(adapter, slot.value.resource));
            if (!image) {
                fail(adapter, "Vulkan image binding is stale");
                return nullptr;
            }
            info.imageView = image->view;
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            const auto samplerIndex = bindings.samplerByBinding.find(slot.entry.layout.binding);
            if (samplerIndex != bindings.samplerByBinding.end()) {
                const auto &sampler = snapshot->slots[samplerIndex->second];
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                if ((sampler.value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
                    info.sampler = bindings.device->defaultImplicitSampler.sampler;
                else {
                    auto *nativeSampler =
                        reinterpret_cast<rhi::vulkan::Sampler *>(resolveRhiResource(adapter, sampler.value.resource));
                    if (!nativeSampler) {
                        fail(adapter, "Vulkan sampler binding is stale");
                        return nullptr;
                    }
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
        fail(adapter, "Vulkan binding snapshot allocation failed", VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    }
    adapter.bindingSnapshotCreations.fetch_add(1, std::memory_order_relaxed);
    if (!retainBindingSnapshot(adapter, encoder, bindings, *result)) {
        auto owned = std::move(bindings.snapshots.find(key)->second);
        bindings.snapshots.erase(key);
        return nullptr;
    }
    recordPushConstants(command, pipeline, bindings);
    return result;
}

VernonStatus createBindings(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || !layout)
        return fail(adapter, "Vulkan adapter received an invalid binding-set descriptor");
    auto bindings = std::unique_ptr<PreparedBindingSet>(new (std::nothrow) PreparedBindingSet());
    if (!bindings)
        return fail(adapter, "Vulkan binding preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    bindings->device = adapter.vulkanDevice;
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
            return fail(adapter, "Vulkan binding layout contains duplicate slots");
        if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
            !bindings->samplerByBinding.emplace(entry.layout.binding, index).second)
            return fail(adapter, "Vulkan binding layout contains duplicate sampler bindings");
        if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
            entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER)
            slot.inlineStorage.resize(entry.layout.element_size);
        bindings->slots.push_back(std::move(slot));
    }
    const VernonStatus status = updateBindingsImpl(adapter, *bindings, descriptor->values, descriptor->value_count);
    if (status != VERNON_STATUS_OK)
        return status;
    *output = toHandle(bindings.release());
    adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus updateBindings(void *data, VernonRuntimeProviderObject handle,
                            const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return fail(adapter, "Vulkan adapter received an invalid binding set");
    std::lock_guard<std::mutex> guard(bindings->mutex);
    return updateBindingsImpl(adapter, *bindings, values, valueCount);
}

void restoreImageLayout(void *context, uint64_t layout) {
    static_cast<rhi::vulkan::Image *>(context)->layout = static_cast<VkImageLayout>(layout);
}

bool transitionImage(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, VkCommandBuffer command,
                     rhi::vulkan::Image &image, VkImageLayout newLayout) {
    if (image.layout == newLayout)
        return true;
    if (!deferCommandRollback(adapter, encoder, &image, static_cast<uint64_t>(image.layout), restoreImageLayout))
        return false;
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
    return true;
}

void bindDescriptorSet(VkCommandBuffer command, PreparedPipeline &pipeline,
                       const PreparedBindingSet::Snapshot *snapshot, VkPipelineBindPoint bindPoint) {
    if (!snapshot)
        return;
    if (snapshot->descriptorSet)
        rhi::vulkan::driver().cmdBindDescriptorSets(command, bindPoint, pipeline.pipelineLayout, 0, 1,
                                                    &snapshot->descriptorSet, 0, nullptr);
}

VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commandEncoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || !pipeline || pipeline->graphics)
        return fail(adapter, "Vulkan adapter received an invalid dispatch");
    const VkCommandBuffer command = reinterpret_cast<VkCommandBuffer>(nativeCommandEncoder(adapter, commandEncoder));
    if (!command || commandEncoderRendering(adapter, commandEncoder))
        return fail(adapter, "Vulkan dispatch command encoder is invalid");
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "Vulkan dispatch could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    auto *bindingSnapshot =
        bindings ? snapshotBindings(adapter, commandEncoder, *pipeline, *bindings, command) : nullptr;
    if (bindings && !bindingSnapshot)
        return fail(adapter, "Vulkan dispatch could not snapshot its bindings", VERNON_STATUS_INTERNAL_ERROR);
    auto &driver = rhi::vulkan::driver();
    driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    bindDescriptorSet(command, *pipeline, bindingSnapshot, VK_PIPELINE_BIND_POINT_COMPUTE);
    driver.cmdDispatch(command, descriptor->group_count[0], descriptor->group_count[1], descriptor->group_count[2]);
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return fail(adapter, "Vulkan dispatch command encoder state changed");
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commandEncoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || !pipeline || !pipeline->graphics || !pipeline->pipeline ||
        descriptor->color_attachment_count == 0 || descriptor->color_attachment_count > 8)
        return fail(adapter, "Vulkan adapter received an invalid draw");
    const VkCommandBuffer command = reinterpret_cast<VkCommandBuffer>(nativeCommandEncoder(adapter, commandEncoder));
    const int renderingClaim =
        claimCommandRendering(adapter, commandEncoder,
                              pipeline->device->dynamicRendering ? vernon::rhi::CommandRenderingDynamic
                                                                 : vernon::rhi::CommandRenderingRenderPass);
    if (!command || renderingClaim < 0)
        return fail(adapter, "Vulkan draw command encoder is invalid");
    const bool beginRendering = renderingClaim != 0;
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "Vulkan draw could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    auto *bindingSnapshot =
        bindings ? snapshotBindings(adapter, commandEncoder, *pipeline, *bindings, command) : nullptr;
    if (bindings && !bindingSnapshot)
        return fail(adapter, "Vulkan draw could not snapshot its bindings", VERNON_STATUS_INTERNAL_ERROR);
    std::array<VernonRhiLoadOperation, 8> colorLoads{};
    std::array<VernonRhiStoreOperation, 8> colorStores{};
    std::array<std::array<float, 4>, 8> colorClears{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        colorLoads[index] = static_cast<VernonRhiLoadOperation>(descriptor->color_attachments[index].load_operation);
        colorStores[index] = static_cast<VernonRhiStoreOperation>(descriptor->color_attachments[index].store_operation);
        std::copy(std::begin(descriptor->color_attachments[index].clear_color),
                  std::end(descriptor->color_attachments[index].clear_color), colorClears[index].begin());
        (void)commandColorOperations(adapter, commandEncoder, index, colorLoads[index], colorStores[index],
                                     colorClears[index].data());
    }
    VernonRhiLoadOperation depthLoad = static_cast<VernonRhiLoadOperation>(descriptor->depth_load_operation);
    VernonRhiStoreOperation depthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
    VernonRhiLoadOperation stencilLoad = static_cast<VernonRhiLoadOperation>(descriptor->stencil_load_operation);
    VernonRhiStoreOperation stencilStore = static_cast<VernonRhiStoreOperation>(descriptor->stencil_store_operation);
    float clearDepth = descriptor->clear_depth;
    uint32_t clearStencil = descriptor->clear_stencil;
    (void)commandDepthOperations(adapter, commandEncoder, depthLoad, depthStore, stencilLoad, stencilStore, clearDepth,
                                 clearStencil);
    std::array<VkRenderingAttachmentInfo, 8> attachments{};
    std::array<VkImageView, 9> imageViews{};
    std::array<VkFormat, 9> attachmentFormats{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->color_attachments[index].image))
            return fail(adapter, "Vulkan draw could not retain its color attachment", VERNON_STATUS_INTERNAL_ERROR);
        auto *image = reinterpret_cast<rhi::vulkan::Image *>(
            resolveRhiResource(adapter, descriptor->color_attachments[index].image));
        if (!image)
            return fail(adapter, "Vulkan color attachment is stale");
        if (beginRendering &&
            !transitionImage(adapter, commandEncoder, command, *image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL))
            return fail(adapter, "Vulkan draw could not track color attachment layout", VERNON_STATUS_INTERNAL_ERROR);
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
    if (descriptor->depth_stencil_attachment.resource.value) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_attachment))
            return fail(adapter, "Vulkan draw could not retain its depth attachment", VERNON_STATUS_INTERNAL_ERROR);
        auto *image =
            reinterpret_cast<rhi::vulkan::Image *>(resolveRhiResource(adapter, descriptor->depth_stencil_attachment));
        if (!image || (image->format != VK_FORMAT_D32_SFLOAT && image->format != VK_FORMAT_D32_SFLOAT_S8_UINT))
            return fail(adapter, "Vulkan draw contains an invalid depth attachment");
        hasStencilAttachment = image->format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        if (!hasStencilAttachment &&
            (stencilLoad != VERNON_RHI_LOAD_DISCARD || stencilStore != VERNON_RHI_STORE_DISCARD || clearStencil))
            return fail(adapter, "Vulkan draw requests stencil operations for a depth-only attachment");
        if (beginRendering && !transitionImage(adapter, commandEncoder, command, *image,
                                               VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL))
            return fail(adapter, "Vulkan draw could not track depth attachment layout", VERNON_STATUS_INTERNAL_ERROR);
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
    const VkRenderingInfo rendering{
        VK_STRUCTURE_TYPE_RENDERING_INFO,
        nullptr,
        0,
        {{static_cast<int32_t>(descriptor->viewport[0]), static_cast<int32_t>(descriptor->viewport[1])},
         {descriptor->viewport[2], descriptor->viewport[3]}},
        1,
        0,
        static_cast<uint32_t>(descriptor->color_attachment_count),
        attachments.data(),
        descriptor->depth_stencil_attachment.resource.value ? &depthAttachment : nullptr,
        hasStencilAttachment ? &stencilAttachment : nullptr};
    auto &driver = rhi::vulkan::driver();
    if (bindingSnapshot && beginRendering)
        for (const auto &slot : bindingSnapshot->slots)
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
                auto *image = reinterpret_cast<rhi::vulkan::Image *>(resolveRhiResource(adapter, slot.value.resource));
                if (!image)
                    return fail(adapter, "Vulkan sampled image binding is stale");
                if (!transitionImage(adapter, commandEncoder, command, *image,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
                    return fail(adapter, "Vulkan draw could not track sampled image layout",
                                VERNON_STATUS_INTERNAL_ERROR);
            }
    VkRenderPass scopeRenderPass{};
    VkFramebuffer framebuffer{};
    if (pipeline->device->dynamicRendering && beginRendering)
        driver.cmdBeginRendering(command, &rendering);
    else if (!pipeline->device->dynamicRendering && beginRendering) {
        PreparedPipeline::RenderingCacheEntry cacheKey{};
        cacheKey.colorCount = static_cast<uint32_t>(descriptor->color_attachment_count);
        cacheKey.width = descriptor->viewport[2];
        cacheKey.height = descriptor->viewport[3];
        cacheKey.hasDepth = descriptor->depth_stencil_attachment.resource.value != 0;
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            cacheKey.resources[index] = descriptor->color_attachments[index].image;
            cacheKey.formats[index] = attachmentFormats[index];
            cacheKey.loads[index] = colorLoads[index];
            cacheKey.stores[index] = colorStores[index];
        }
        if (cacheKey.hasDepth) {
            const size_t index = descriptor->color_attachment_count;
            cacheKey.resources[index] = descriptor->depth_stencil_attachment;
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
                for (size_t index = 0; index < count; ++index)
                    releaseRhiResource(adapter, victim->resources[index]);
                pipeline->renderingCache.erase(victim);
            }
        }
        std::array<VkAttachmentDescription, 9> renderPassAttachments{};
        std::array<VkAttachmentReference, 8> renderPassReferences{};
        VkAttachmentReference depthReference{};
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            renderPassAttachments[index] = attachmentDescription(
                attachmentFormats[index], VK_SAMPLE_COUNT_1_BIT, attachmentLoad(colorLoads[index]),
                attachmentStore(colorStores[index]), VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            renderPassReferences[index] = {static_cast<uint32_t>(index), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        const bool hasDepth = descriptor->depth_stencil_attachment.resource.value != 0;
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
            return failure(adapter, result, "vkCreateRenderPass");
        const VkFramebufferCreateInfo framebufferInfo{
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            nullptr,
            0,
            scopeRenderPass,
            static_cast<uint32_t>(descriptor->color_attachment_count +
                                  (descriptor->depth_stencil_attachment.resource.value ? 1 : 0)),
            imageViews.data(),
            descriptor->viewport[2],
            descriptor->viewport[3],
            1};
        if (!cacheHit)
            result = driver.createFramebuffer(pipeline->device->device, &framebufferInfo, nullptr, &framebuffer);
        if (result != VK_SUCCESS) {
            driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
            return failure(adapter, result, "vkCreateFramebuffer");
        }
        if (!cacheHit && cacheNewEntry) {
            const size_t count = cacheKey.colorCount + (cacheKey.hasDepth ? 1u : 0u);
            size_t retained = 0;
            for (; retained < count; ++retained)
                if (!retainRhiResource(adapter, cacheKey.resources[retained]))
                    break;
            if (retained != count) {
                while (retained)
                    releaseRhiResource(adapter, cacheKey.resources[--retained]);
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
                return fail(adapter, "Vulkan rendering cache could not retain attachments",
                            VERNON_STATUS_INTERNAL_ERROR);
            }
            cacheKey.lastEncoder = commandEncoder.value;
            cacheKey.renderPass = scopeRenderPass;
            cacheKey.framebuffer = framebuffer;
            try {
                pipeline->renderingCache.push_back(cacheKey);
            } catch (const std::bad_alloc &) {
                while (retained)
                    releaseRhiResource(adapter, cacheKey.resources[--retained]);
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
                return fail(adapter, "Vulkan rendering cache allocation failed", VERNON_STATUS_INTERNAL_ERROR);
            }
        } else if (!cacheHit) {
            if (!deferCommandCleanup(adapter, commandEncoder, pipeline->device, nativeHandleBits(scopeRenderPass),
                                     destroyRecordedRenderPass)) {
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                driver.destroyRenderPass(pipeline->device->device, scopeRenderPass, nullptr);
                return fail(adapter, "Vulkan temporary render-pass cleanup allocation failed",
                            VERNON_STATUS_INTERNAL_ERROR);
            }
            if (!deferCommandCleanup(adapter, commandEncoder, pipeline->device, nativeHandleBits(framebuffer),
                                     destroyRecordedFramebuffer)) {
                driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
                return fail(adapter, "Vulkan temporary framebuffer cleanup allocation failed",
                            VERNON_STATUS_INTERNAL_ERROR);
            }
        }
        std::array<VkClearValue, 9> clearValues{};
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            std::copy(colorClears[index].begin(), colorClears[index].end(), clearValues[index].color.float32);
        if (descriptor->depth_stencil_attachment.resource.value)
            clearValues[descriptor->color_attachment_count].depthStencil = {clearDepth, clearStencil};
        const VkRenderPassBeginInfo begin{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr,
            scopeRenderPass,
            framebuffer,
            {{static_cast<int32_t>(descriptor->viewport[0]), static_cast<int32_t>(descriptor->viewport[1])},
             {descriptor->viewport[2], descriptor->viewport[3]}},
            static_cast<uint32_t>(descriptor->color_attachment_count +
                                  (descriptor->depth_stencil_attachment.resource.value ? 1 : 0)),
            clearValues.data()};
        driver.cmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    }
    driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
    bindDescriptorSet(command, *pipeline, bindingSnapshot, VK_PIPELINE_BIND_POINT_GRAPHICS);
    if (bindingSnapshot)
        for (const auto &slot : bindingSnapshot->slots) {
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
                auto *image = reinterpret_cast<rhi::vulkan::Image *>(resolveRhiResource(adapter, slot.value.resource));
                if (!image)
                    return fail(adapter, "Vulkan sampled image binding is stale");
            } else if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                auto *buffer =
                    reinterpret_cast<rhi::vulkan::Buffer *>(resolveRhiResource(adapter, slot.value.resource));
                if (!buffer)
                    return fail(adapter, "Vulkan vertex buffer binding is stale");
                const VkDeviceSize offset = slot.value.resource.offset;
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
    if (descriptor->index_count) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->index_buffer))
            return fail(adapter, "Vulkan draw could not retain its index buffer", VERNON_STATUS_INTERNAL_ERROR);
        auto *buffer = reinterpret_cast<rhi::vulkan::Buffer *>(resolveRhiResource(adapter, descriptor->index_buffer));
        if (!buffer)
            return fail(adapter, "Vulkan index buffer is stale");
        driver.cmdBindIndexBuffer(command, buffer->buffer, descriptor->index_buffer.offset, VK_INDEX_TYPE_UINT32);
        driver.cmdDrawIndexed(command, descriptor->index_count, descriptor->instance_count, 0, descriptor->first_vertex,
                              descriptor->first_instance);
    } else
        driver.cmdDraw(command, descriptor->vertex_count, descriptor->instance_count, descriptor->first_vertex,
                       descriptor->first_instance);
    if (!recordProviderCommand(adapter, commandEncoder, true))
        return fail(adapter, "Vulkan draw command encoder state changed");
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

void destroyShader(void *, VernonRuntimeProviderObject handle) {
    auto *shader = fromHandle<PreparedShader>(handle);
    if (!shader)
        return;
    rhi::vulkan::driver().destroyShaderModule(shader->device->device, shader->module, nullptr);
    delete shader;
}
void destroyLayout(void *, VernonRuntimeProviderObject handle) {
    auto *layout = fromHandle<PreparedLayout>(handle);
    if (!layout)
        return;
    rhi::vulkan::driver().destroyDescriptorSetLayout(layout->device->device, layout->descriptorSetLayout, nullptr);
    delete layout;
}
PreparedPipeline::~PreparedPipeline() {
    if (!device)
        return;
    for (auto &entry : renderingCache) {
        rhi::vulkan::driver().destroyFramebuffer(device->device, entry.framebuffer, nullptr);
        rhi::vulkan::driver().destroyRenderPass(device->device, entry.renderPass, nullptr);
        const size_t count = entry.colorCount + (entry.hasDepth ? 1u : 0u);
        for (size_t index = 0; index < count; ++index)
            releaseRhiResource(*adapter, entry.resources[index]);
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
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
}
void releaseCommandBindings(void *context, uint64_t) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    if (!bindings || bindings->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete bindings;
}
void destroyBindings(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
}

} // namespace

void initializeVulkanProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PIPELINE_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainResource;
    adapter.provider.release_resource = releaseResource;
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

#endif
