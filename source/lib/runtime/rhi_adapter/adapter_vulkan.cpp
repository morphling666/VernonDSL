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
#include <new>
#include <string>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

constexpr size_t kMaxGroups = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;

struct VulkanAdapterState {
    rhi::vulkan::DeviceState *device{};
    std::atomic<uint64_t> nextSerial{};
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
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "vulkan_synchronize");
    } catch (...) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "vulkan_synchronize");
    }
}
uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const VulkanAdapterState *>(state)->device);
}
const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity};

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

struct PreparedShader {
    rhi::vulkan::DeviceState *device{};
    uint32_t stage{};
    VkShaderModule module{};
    std::string entry;
    std::vector<uint32_t> words;
};

// Graphics inline values travel as push constants; the pipeline layout places them.
struct GroupPushValue {
    uint32_t entry{};
    uint32_t dataOffset{};
    uint32_t size{};
};

// Retained by the groups and pipeline layouts made from it, since commands that used those may still be recording.
struct GroupLayout : RhiRetainedObject {
    struct Entry {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        // The sampler entry that shares this sampled image's binding as a combined image sampler.
        uint32_t combinedSampler{UINT32_MAX};
        bool combinedWithImage{};
        bool inlineBuffer{};
    };
    rhi::vulkan::DeviceState *device{};
    std::vector<Entry> entries;
    std::vector<GroupPushValue> pushValues;
    uint32_t pushDataSize{};
    VkDescriptorSetLayout setLayout{};
    bool hasDescriptors{};

    ~GroupLayout() override {
        if (setLayout)
            rhi::vulkan::driver().destroyDescriptorSetLayout(device->device, setLayout, nullptr);
    }
};

struct PushPlacement {
    uint32_t dataOffset{};
    uint32_t offset{};
    uint32_t size{};
    uint32_t stage{};
};

struct PushRange {
    VkShaderStageFlags stages{};
    uint32_t offset{};
    uint32_t size{};

    bool operator==(const PushRange &other) const {
        return stages == other.stages && offset == other.offset && size == other.size;
    }
};

struct PreparedLayout : RhiRetainedObject {
    rhi::vulkan::DeviceState *device{};
    std::array<GroupLayout *, kMaxGroups> groups{};
    uint32_t groupCount{};
    // Bit g is set when the group layout at index g has entries; an index without entries is never bound.
    uint32_t groupMask{};
    // Per group index, in the order of the group layout's push values.
    std::array<std::vector<PushPlacement>, kMaxGroups> pushPlacements;
    std::vector<PushRange> pushRanges;
    uint32_t pushConstantSize{};
    uint32_t pushGroupMask{};
    std::vector<VernonRuntimeProviderVertexBinding> vertexBindings;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    VkPipelineLayout pipelineLayout{};

    ~PreparedLayout() override {
        if (pipelineLayout)
            rhi::vulkan::driver().destroyPipelineLayout(device->device, pipelineLayout, nullptr);
        for (uint32_t group = 0; group < groupCount; ++group)
            releaseRetainedObject(groups[group]);
    }
};

// The number of leading group indices at which descriptor sets bound with one layout remain valid for the other.
uint32_t compatibleGroups(const PreparedLayout *bound, const PreparedLayout &next) {
    if (!bound)
        return 0;
    if (bound == &next)
        return next.groupCount;
    if (bound->pushRanges != next.pushRanges)
        return 0;
    uint32_t count = 0;
    while (count < bound->groupCount && count < next.groupCount && bound->groups[count] == next.groups[count])
        ++count;
    return count;
}

struct PreparedPipeline : RhiRetainedObject {
    VernonRuntimeRhiAdapter *adapter{};
    rhi::vulkan::DeviceState *device{};
    PreparedLayout *layout{};
    VkPipeline pipeline{};
    VkRenderPass renderPass{};
    std::vector<VkShaderModule> specializedShaderModules;
    bool graphics{};

    ~PreparedPipeline() override {
        adapter->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
        if (pipeline)
            rhi::vulkan::driver().destroyPipeline(device->device, pipeline, nullptr);
        if (renderPass)
            rhi::vulkan::driver().destroyRenderPass(device->device, renderPass, nullptr);
        for (VkShaderModule module : specializedShaderModules)
            rhi::vulkan::driver().destroyShaderModule(device->device, module, nullptr);
        releaseRetainedObject(layout);
    }
};

struct VulkanGroup : RhiGroup {
    struct ImageUse {
        rhi::vulkan::Image *image{};
        VkImageLayout layout{};
    };
    rhi::vulkan::DeviceState *device{};
    GroupLayout *layout{};
    uint64_t serial{};
    rhi::vulkan::DescriptorSet descriptorSet;
    std::vector<uint8_t> pushData;
    std::vector<ImageUse> images;

    ~VulkanGroup() override {
        if (descriptorSet.set)
            device->freeDescriptorSet(descriptorSet);
        releaseRetainedObject(layout);
    }
};

// Descriptor sets bound at a bind point stay valid across pipelines whose layouts are compatible at their index.
struct BindPointState {
    const PreparedLayout *layout{};
    std::array<uint64_t, kMaxGroups> sets{};
};

// Applied state holds group serials rather than pointers, because addresses are reused.
struct VulkanCommands : RhiCommands {
    VkCommandBuffer command{};
    PreparedPipeline *pipeline{};
    std::array<VulkanGroup *, kMaxGroups> groups{};
    BindPointState graphicsState;
    BindPointState computeState;
    // The layout whose push constant placement `pushData` follows, and the group serial each index copied in.
    const PreparedLayout *pushLayout{};
    std::array<uint64_t, kMaxGroups> pushGroups{};
    std::vector<uint8_t> pushData;
    // Rendering scopes begun so far, and the scope in which each group index's sampled images were last checked.
    uint64_t scope{};
    std::array<uint64_t, kMaxGroups> checkedGroups{};
    std::array<uint64_t, kMaxGroups> checkedScopes{};
    bool viewportValid{};
    bool scissorValid{};
    bool stencilValid{};
    std::array<uint32_t, 4> viewport{};
    std::array<uint32_t, 4> scissor{};
    uint32_t stencilReference{};
    uint64_t indexBufferKey{};
    uint64_t indexBufferOffset{};
    VkBuffer indexBuffer{};
};

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
        // Compute inline values use a storage buffer the group owns; graphics inline values are push constants.
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

bool pushValue(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
           entry.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
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
               : providerFailure(vernon::ProviderErrorCode::BackendFailure, operation, static_cast<uint64_t>(result));
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
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "vulkan_adapter_received_an_invalid_spirv_shader");
    auto shader = std::unique_ptr<PreparedShader>(new (std::nothrow) PreparedShader());
    if (!shader)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "vulkan_shader_preparation_ran_out_of_memory");
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

RhiAdapterResult<void> prepareGroupLayoutImplResult(VernonRuntimeRhiAdapter &adapter,
                                                    const VernonRuntimeProviderGroupLayoutDescriptor &descriptor,
                                                    VernonRuntimeProviderObject *output) {
    std::unique_ptr<GroupLayout> layout(new (std::nothrow) GroupLayout());
    if (!layout)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "vulkan_group_layout_preparation_ran_out_of_memory");
    layout->device = &vulkanDevice(adapter);
    std::vector<VkDescriptorSetLayoutBinding> nativeBindings;
    layout->entries.reserve(descriptor.entry_count);
    for (size_t index = 0; index < descriptor.entry_count; ++index) {
        const auto &source = descriptor.entries[index];
        if (descriptorType(source.kind) == VK_DESCRIPTOR_TYPE_MAX_ENUM)
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "vulkan_group_layout_contains_an_unsupported_binding", index);
        if (source.array_count != 1)
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "vulkan_group_layout_contains_an_unsupported_binding", index);
        GroupLayout::Entry entry;
        entry.layout = source;
        entry.inlineBuffer = !pushValue(source) && packedUniformBytes(source.kind, source.interface_kind);
        if (pushValue(source)) {
            const uint32_t alignment = source.element_alignment;
            if (!alignment || (alignment & (alignment - 1)) != 0 || source.element_size == 0)
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "vulkan_inline_uniform_alignment_is_invalid", index);
            layout->pushValues.push_back({static_cast<uint32_t>(index), layout->pushDataSize, source.element_size});
            layout->pushDataSize += source.element_size;
        } else {
            const auto existing = std::find_if(nativeBindings.begin(), nativeBindings.end(),
                                               [&](const auto &binding) { return binding.binding == source.binding; });
            // The compiler links one descriptor per Program Value across a pipeline, so a repeated binding of the
            // same type is one value read by several stages.
            if (existing == nativeBindings.end())
                nativeBindings.push_back(
                    {source.binding, descriptorType(source.kind), 1, stages(source.stage_mask), nullptr});
            else if (existing->descriptorType == descriptorType(source.kind) ||
                     (existing->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                      (source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                       source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)))
                existing->stageFlags |= stages(source.stage_mask);
            else if ((existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
                      source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) ||
                     (existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER &&
                      source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)) {
                existing->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                existing->stageFlags |= stages(source.stage_mask);
            } else
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "vulkan_group_layout_contains_duplicate_descriptor_bindings", index);
        }
        layout->entries.push_back(entry);
    }
    for (uint32_t image = 0; image < layout->entries.size(); ++image) {
        auto &entry = layout->entries[image];
        if (entry.layout.kind != VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)
            continue;
        for (uint32_t sampler = 0; sampler < layout->entries.size(); ++sampler)
            if (layout->entries[sampler].layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                layout->entries[sampler].layout.binding == entry.layout.binding) {
                entry.combinedSampler = sampler;
                layout->entries[sampler].combinedWithImage = true;
            }
    }
    layout->hasDescriptors = !nativeBindings.empty();
    const VkDescriptorSetLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
                                                     static_cast<uint32_t>(nativeBindings.size()),
                                                     nativeBindings.data()};
    if (auto created = failureResult(rhi::vulkan::driver().createDescriptorSetLayout(
                                         layout->device->device, &createInfo, nullptr, &layout->setLayout),
                                     "vkCreateDescriptorSetLayout");
        !created)
        return created;
    *output = toHandle(layout.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareGroupLayoutResult(void *data,
                                                const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                                VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->entry_count != 0 && !descriptor->entries))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "vulkan_adapter_received_an_invalid_group_layout");
    try {
        return prepareGroupLayoutImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "vulkan_group_layout_preparation_ran_out_of_memory");
    }
}

// Places the graphics inline values of every group: sorted by stage, then by entry argument order, which shader
// push-constant members follow, while each stage's block is relocated to its offset.
RhiAdapterResult<void> placePushValues(PreparedLayout &layout) {
    struct Candidate {
        uint32_t group{};
        uint32_t value{};
        const VernonRuntimeProviderBindingLayoutEntry *entry{};
    };
    std::vector<Candidate> candidates;
    for (uint32_t group = 0; group < layout.groupCount; ++group) {
        const auto &values = layout.groups[group]->pushValues;
        layout.pushPlacements[group].resize(values.size());
        for (uint32_t value = 0; value < values.size(); ++value)
            candidates.push_back({group, value, &layout.groups[group]->entries[values[value].entry].layout});
        if (!values.empty())
            layout.pushGroupMask |= 1u << group;
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &left, const Candidate &right) {
        if (left.entry->stage_mask != right.entry->stage_mask)
            return left.entry->stage_mask < right.entry->stage_mask;
        return left.entry->argument_index < right.entry->argument_index;
    });
    for (const Candidate &candidate : candidates) {
        const uint32_t alignment = candidate.entry->element_alignment;
        if (layout.pushConstantSize > UINT32_MAX - (alignment - 1))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "vulkan_push_constant_layout_size_overflow");
        const uint32_t offset = (layout.pushConstantSize + alignment - 1) & ~(alignment - 1);
        if (candidate.entry->element_size > UINT32_MAX - offset)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "vulkan_push_constant_layout_size_overflow");
        const GroupPushValue &value = layout.groups[candidate.group]->pushValues[candidate.value];
        layout.pushPlacements[candidate.group][candidate.value] = {value.dataOffset, offset, value.size,
                                                                   candidate.entry->stage_mask};
        layout.pushConstantSize = offset + value.size;
        const VkShaderStageFlags stageFlags = stages(candidate.entry->stage_mask);
        if (layout.pushRanges.empty() || layout.pushRanges.back().stages != stageFlags)
            layout.pushRanges.push_back({stageFlags, offset, value.size});
        else
            layout.pushRanges.back().size = offset + value.size - layout.pushRanges.back().offset;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutImplResult(VernonRuntimeRhiAdapter &adapter,
                                               const VernonRuntimeProviderPipelineLayoutDescriptor &descriptor,
                                               VernonRuntimeProviderObject *output) {
    std::unique_ptr<PreparedLayout> layout(new (std::nothrow) PreparedLayout());
    if (!layout)
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "vulkan_layout_preparation_ran_out_of_memory");
    layout->device = &vulkanDevice(adapter);
    std::array<VkDescriptorSetLayout, kMaxGroups> setLayouts{};
    for (size_t group = 0; group < descriptor.group_layout_count; ++group) {
        auto *groupLayout = fromHandle<GroupLayout>(descriptor.group_layouts[group]);
        if (!groupLayout)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "vulkan_pipeline_layout_group_layout_is_invalid", group);
        groupLayout->references.fetch_add(1, std::memory_order_relaxed);
        layout->groups[group] = groupLayout;
        layout->groupCount = static_cast<uint32_t>(group + 1);
        setLayouts[group] = groupLayout->setLayout;
        if (!groupLayout->entries.empty())
            layout->groupMask |= 1u << group;
    }
    if (auto placed = placePushValues(*layout); !placed)
        return placed;
    if (descriptor.push_constant_size != 0) {
        if (layout->pushConstantSize != 0)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "vulkan_pipeline_layout_mixes_push_constants_and_inline_values");
        layout->pushConstantSize = descriptor.push_constant_size;
        layout->pushRanges.push_back({VK_SHADER_STAGE_COMPUTE_BIT, 0, descriptor.push_constant_size});
    }
    if (layout->pushConstantSize > layout->device->maxPushConstantsSize)
        return providerFailure(vernon::ProviderErrorCode::Unsupported, "vulkan_push_constants_exceed_the_device_limit");
    layout->vertexBindings.assign(descriptor.vertex_bindings,
                                  descriptor.vertex_bindings + descriptor.vertex_binding_count);
    for (size_t index = 0; index < descriptor.vertex_attribute_count; ++index) {
        const auto &attribute = descriptor.vertex_attributes[index];
        std::string capabilityDiagnostic;
        if (!validateVertexAttributeCapability(VertexAttributeBackend::Vulkan, attribute,
                                               layout->device->maxVertexInputAttributes, true, capabilityDiagnostic))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "vulkan_validate_vertex_attribute_capability", attribute.location,
                                   attribute.binding);
        const VkFormat format = vertexFormat(attribute.dtype, attribute.component_count);
        VkFormatProperties properties{};
        if (format != VK_FORMAT_UNDEFINED)
            rhi::vulkan::driver().getPhysicalDeviceFormatProperties(layout->device->physicalDevice, format,
                                                                    &properties);
        if (format == VK_FORMAT_UNDEFINED || !(properties.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "vulkan_vertex_attribute_format_is_unsupported_by_the_selected_device");
        layout->vertexAttributes.push_back(attribute);
    }
    std::vector<VkPushConstantRange> ranges;
    for (const PushRange &range : layout->pushRanges)
        ranges.push_back({range.stages, range.offset, range.size});
    const VkPipelineLayoutCreateInfo createInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr,      0, layout->groupCount, setLayouts.data(),
        static_cast<uint32_t>(ranges.size()),          ranges.data()};
    if (auto created = failureResult(rhi::vulkan::driver().createPipelineLayout(layout->device->device, &createInfo,
                                                                                nullptr, &layout->pipelineLayout),
                                     "vkCreatePipelineLayout");
        !created)
        return created;
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->group_layout_count > kMaxGroups ||
        (descriptor->group_layout_count != 0 && !descriptor->group_layouts) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "vulkan_adapter_received_an_invalid_pipeline_layout");
    try {
        return prepareLayoutImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "vulkan_layout_preparation_ran_out_of_memory");
    }
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
    layout->references.fetch_add(1, std::memory_order_relaxed);
    pipeline->layout = layout;
    pipeline->graphics = descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
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
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage, layout->pipelineLayout, {}, 0};
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
            for (uint32_t group = 0; group < layout->groupCount; ++group)
                for (const PushPlacement &placement : layout->pushPlacements[group])
                    if (placement.stage == stage)
                        offset = std::min(offset, placement.offset);
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
        for (const VernonRuntimeProviderVertexBinding &binding : layout->vertexBindings)
            vertexBindings.push_back(
                {binding.binding, 0, binding.divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
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
                                                      layout->pipelineLayout,
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
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "vulkan_pipeline_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> validateGroupValue(const GroupLayout::Entry &entry,
                                          const VernonRuntimeProviderBindingValue &value) {
    const auto &layout = entry.layout;
    if (value.kind != layout.kind)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_group_value_kind_is_invalid",
                               layout.slot);
    if (packedUniformBytes(layout.kind, layout.interface_kind)) {
        if (!value.payload.inline_value.data || value.payload.inline_value.size != layout.element_size)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "vulkan_inline_or_uniform_buffer_binding_size_is_invalid", layout.slot);
        return RhiAdapterResult<void>{vernon::ok()};
    }
    const bool defaultResource = (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
    if (defaultResource)
        return layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER
                   ? RhiAdapterResult<void>{vernon::ok()}
                   : providerFailure(vernon::ProviderErrorCode::Unsupported,
                                     "vulkan_default_resource_is_supported_only_for_samplers", layout.slot);
    const auto *resource = providerBindingResource(value);
    const bool image =
        layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    const bool sampler = layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER;
    const uint64_t expectedKind = sampler ? kRhiSamplerResource : kRhiBufferResource;
    if (!resource || !resource->resource.value ||
        (image ? !isRhiImageReference(*resource) : (resource->identity & kRhiResourceKindMask) != expectedKind) ||
        (!image && !sampler && resource->offset >= resource->size))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_resource_binding_is_invalid",
                               layout.slot);
    return RhiAdapterResult<void>{vernon::ok()};
}

// Places the inline buffer values, and `ranges` receives each one's buffer range. A transient group writes them to its
// encoder's transient memory; a persistent group fills one persistent block of the device in place.
RhiAdapterResult<void> placeInlineValues(VernonRuntimeRhiAdapter &adapter, VulkanGroup &group,
                                         const VernonRuntimeProviderGroupDescriptor &descriptor,
                                         std::vector<VkDescriptorBufferInfo> &ranges) {
    const auto &entries = group.layout->entries;
    const VernonRuntimeProviderBindingValue *values = descriptor.values;
    const VkDeviceSize alignment = group.device->descriptorBufferOffsetAlignment;
    ranges.assign(entries.size(), {});
    if (descriptor.lifetime == VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT) {
        for (size_t index = 0; index < entries.size(); ++index) {
            if (!entries[index].inlineBuffer)
                continue;
            const VkDeviceSize size = entries[index].layout.element_size;
            auto written = writeCommandTransient(adapter, descriptor.command_encoder,
                                                 values[index].payload.inline_value.data, size, alignment);
            if (!written)
                return RhiAdapterResult<void>{vernon::err(std::move(written).error())};
            ranges[index] = {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(written.value().buffer)),
                             written.value().offset, size};
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }
    VkDeviceSize size = 0;
    for (size_t index = 0; index < entries.size(); ++index) {
        if (!entries[index].inlineBuffer)
            continue;
        size = (size + alignment - 1) & ~(alignment - 1);
        ranges[index] = {VK_NULL_HANDLE, size, entries[index].layout.element_size};
        size += entries[index].layout.element_size;
    }
    if (size == 0)
        return RhiAdapterResult<void>{vernon::ok()};
    auto placed = placePersistent(adapter, group, nullptr, size, alignment);
    if (!placed)
        return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
    const vernon::rhi::PersistentBlock &block = placed.value();
    for (size_t index = 0; index < entries.size(); ++index)
        if (entries[index].inlineBuffer) {
            std::memcpy(static_cast<uint8_t *>(block.mapped) + ranges[index].offset,
                        values[index].payload.inline_value.data, values[index].payload.inline_value.size);
            ranges[index].buffer = reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(block.buffer));
            ranges[index].offset += block.offset;
        }
    return RhiAdapterResult<void>{vernon::ok()};
}

// A transient group's set belongs to its encoder's command frame, which resets it when it retires.
RhiAdapterResult<void> allocateGroupDescriptorSet(VernonRuntimeRhiAdapter &adapter, VulkanGroup &group,
                                                  const VernonRuntimeProviderGroupDescriptor &descriptor) {
    if (descriptor.lifetime != VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT) {
        std::string backendDiagnostic;
        if (!group.device->allocateDescriptorSet(group.layout->setLayout, group.descriptorSet, backendDiagnostic))
            return providerFailure(vernon::ProviderErrorCode::BackendFailure, "vulkan_allocate_group_descriptor_set");
        return RhiAdapterResult<void>{vernon::ok()};
    }
    auto native = nativeCommandEncoder(adapter, descriptor.command_encoder);
    if (!native)
        return RhiAdapterResult<void>{vernon::err(std::move(native).error())};
    auto set = rhi::vulkan::allocateCommandDescriptorSet(
        adapter.rhiDevice, reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(native.value())),
        group.layout->setLayout);
    if (!set)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(set).error()))};
    group.descriptorSet.set = set.value();
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupImplResult(VernonRuntimeRhiAdapter &adapter,
                                             const VernonRuntimeProviderGroupDescriptor &descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto *layout = fromHandle<GroupLayout>(descriptor.group_layout);
    if (!layout || descriptor.value_count != layout->entries.size() ||
        (descriptor.value_count != 0 && !descriptor.values))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "vulkan_adapter_received_an_invalid_group_descriptor");
    const VernonRuntimeProviderBindingValue *values = descriptor.values;
    for (size_t index = 0; index < layout->entries.size(); ++index)
        if (auto valid = validateGroupValue(layout->entries[index], values[index]); !valid)
            return valid;
    auto group = std::make_unique<VulkanGroup>();
    group->device = &vulkanDevice(adapter);
    layout->references.fetch_add(1, std::memory_order_relaxed);
    group->layout = layout;
    std::vector<uint64_t> natives(layout->entries.size());
    for (size_t index = 0; index < layout->entries.size(); ++index) {
        const auto &entry = layout->entries[index].layout;
        const bool defaultResource = (values[index].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
        if (packedUniformBytes(entry.kind, entry.interface_kind) || defaultResource)
            continue;
        auto acquired = acquireGroupResource(adapter, *group, *providerBindingResource(values[index]));
        if (!acquired)
            return RhiAdapterResult<void>{vernon::err(std::move(acquired).error())};
        natives[index] = acquired.value();
        if (!natives[index])
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_resource_binding_is_stale",
                                   entry.slot);
        if (writesBinding(entry))
            group->writes.push_back(*providerBindingResource(values[index]));
    }
    std::vector<VkDescriptorBufferInfo> inlineRanges;
    if (auto placed = placeInlineValues(adapter, *group, descriptor, inlineRanges); !placed)
        return placed;
    group->pushData.resize(layout->pushDataSize);
    for (const GroupPushValue &value : layout->pushValues)
        std::memcpy(group->pushData.data() + value.dataOffset, values[value.entry].payload.inline_value.data,
                    value.size);
    if (layout->hasDescriptors)
        if (auto allocated = allocateGroupDescriptorSet(adapter, *group, descriptor); !allocated)
            return allocated;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    writes.reserve(layout->entries.size());
    buffers.reserve(layout->entries.size());
    images.reserve(layout->entries.size());
    for (size_t index = 0; index < layout->entries.size(); ++index) {
        const GroupLayout::Entry &entry = layout->entries[index];
        const auto kind = entry.layout.kind;
        if (pushValue(entry.layout) || (kind == VERNON_RUNTIME_PROVIDER_SAMPLER && entry.combinedWithImage))
            continue;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = group->descriptorSet.set;
        write.dstBinding = entry.layout.binding;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType(kind);
        const auto samplerOf = [&](size_t samplerIndex) {
            return (values[samplerIndex].flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0
                       ? group->device->defaultImplicitSampler.sampler
                       : reinterpret_cast<rhi::vulkan::Sampler *>(natives[samplerIndex])->sampler;
        };
        if (kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER || kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
            kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
            if (entry.inlineBuffer) {
                buffers.push_back(inlineRanges[index]);
            } else {
                const auto &resource = values[index].payload.buffer.resource;
                buffers.push_back(
                    {reinterpret_cast<rhi::vulkan::Buffer *>(natives[index])->buffer, resource.offset, resource.size});
            }
            write.pBufferInfo = &buffers.back();
        } else if (kind == VERNON_RUNTIME_PROVIDER_SAMPLER) {
            images.push_back({samplerOf(index), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
            write.pImageInfo = &images.back();
        } else {
            auto *image = reinterpret_cast<rhi::vulkan::Image *>(natives[index]);
            const VkImageLayout imageLayout = kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
                                                  ? VK_IMAGE_LAYOUT_GENERAL
                                                  : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            group->images.push_back({image, imageLayout});
            VkDescriptorImageInfo info{VK_NULL_HANDLE, image->view, imageLayout};
            if (entry.combinedSampler != UINT32_MAX) {
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                info.sampler = samplerOf(entry.combinedSampler);
            }
            images.push_back(info);
            write.pImageInfo = &images.back();
        }
        writes.push_back(write);
    }
    if (!writes.empty())
        rhi::vulkan::driver().updateDescriptorSets(group->device->device, static_cast<uint32_t>(writes.size()),
                                                   writes.data(), 0, nullptr);
    group->serial = vulkanState(adapter).nextSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    *output = toHandle(group.release());
    adapter.groupCreations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupResult(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                                         VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "vulkan_adapter_received_an_invalid_group_descriptor");
    try {
        return createGroupImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "vulkan_group_creation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> beginCommandsResult(void *data, VernonRuntimeProviderObject encoder,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!output)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_begin_commands_output_is_null");
    std::unique_ptr<VulkanCommands> commands(new (std::nothrow) VulkanCommands());
    if (!commands)
        return providerFailure(vernon::ProviderErrorCode::ResourceExhausted, "vulkan_commands_allocation");
    if (auto begun = beginRhiCommands(adapter, encoder, *commands); !begun)
        return begun;
    commands->command = reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(commands->native));
    try {
        commands->pushData.resize(vulkanDevice(adapter).maxPushConstantsSize);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::ResourceExhausted, "vulkan_commands_allocation");
    }
    *output = toHandle(commands.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endCommandsResult(void *, VernonRuntimeProviderObject handle) {
    std::unique_ptr<VulkanCommands> commands(fromHandle<VulkanCommands>(handle));
    return commands
               ? RhiAdapterResult<void>{vernon::ok()}
               : providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_end_commands_handle_is_invalid");
}

// The RHI records no pipeline, descriptor set, vertex buffer or dynamic state binds of its own.
RhiAdapterResult<void> resumeCommandsResult(void *, VernonRuntimeProviderObject handle) {
    const auto *commands = fromHandle<VulkanCommands>(handle);
    return commands && !commands->rendering
               ? RhiAdapterResult<void>{vernon::ok()}
               : providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_resume_commands_is_invalid");
}

RhiAdapterResult<void> bindPipelineResult(void *data, VernonRuntimeProviderObject handle,
                                          VernonRuntimeProviderObject pipelineHandle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<VulkanCommands>(handle);
    auto *pipeline = fromHandle<PreparedPipeline>(pipelineHandle);
    if (!commands || !pipeline || !pipeline->pipeline)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_bind_pipeline_is_invalid");
    if (auto retained = retainForCommands(adapter, commands->encoder, *pipeline); !retained)
        return retained;
    const PreparedLayout &layout = *pipeline->layout;
    BindPointState &bindPoint = pipeline->graphics ? commands->graphicsState : commands->computeState;
    if (bindPoint.layout != &layout) {
        const uint32_t kept = compatibleGroups(bindPoint.layout, layout);
        std::fill(bindPoint.sets.begin() + kept, bindPoint.sets.end(), 0);
        bindPoint.layout = &layout;
    }
    for (uint32_t group = 0; group < kMaxGroups; ++group)
        if (commands->groups[group] &&
            (group >= layout.groupCount || commands->groups[group]->layout != layout.groups[group]))
            commands->groups[group] = nullptr;
    rhi::vulkan::driver().cmdBindPipeline(
        commands->command, pipeline->graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline->pipeline);
    commands->pipeline = pipeline;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> bindGroupResult(void *data, VernonRuntimeProviderObject handle, uint32_t index,
                                       VernonRuntimeProviderObject groupHandle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<VulkanCommands>(handle);
    auto *group = fromHandle<VulkanGroup>(groupHandle);
    if (!commands || !group || !commands->pipeline || index >= commands->pipeline->layout->groupCount ||
        group->layout != commands->pipeline->layout->groups[index])
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_bind_group_is_invalid", index);
    if (auto retained = retainForCommands(adapter, commands->encoder, *group); !retained)
        return retained;
    commands->groups[index] = group;
    return RhiAdapterResult<void>{vernon::ok()};
}

// Binds the descriptor sets and pushes the inline values of the bound groups that the native state lacks.
RhiAdapterResult<void> flushGroups(VulkanCommands &commands, VkPipelineBindPoint bindPoint, BindPointState &state) {
    const PreparedLayout &layout = *commands.pipeline->layout;
    auto &driver = rhi::vulkan::driver();
    if (commands.pushLayout != &layout) {
        commands.pushLayout = &layout;
        commands.pushGroups.fill(0);
    }
    bool pushed = false;
    for (uint32_t index = 0; index < layout.groupCount; ++index) {
        if ((layout.groupMask & (1u << index)) == 0)
            continue;
        const VulkanGroup *group = commands.groups[index];
        if (!group)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_command_group_is_unbound",
                                   index);
        if (group->layout->hasDescriptors && state.sets[index] != group->serial) {
            driver.cmdBindDescriptorSets(commands.command, bindPoint, layout.pipelineLayout, index, 1,
                                         &group->descriptorSet.set, 0, nullptr);
            state.sets[index] = group->serial;
        }
        if ((layout.pushGroupMask & (1u << index)) == 0 || commands.pushGroups[index] == group->serial)
            continue;
        for (const PushPlacement &placement : layout.pushPlacements[index])
            std::memcpy(commands.pushData.data() + placement.offset, group->pushData.data() + placement.dataOffset,
                        placement.size);
        commands.pushGroups[index] = group->serial;
        pushed = true;
    }
    if (pushed)
        for (const PushRange &range : layout.pushRanges)
            driver.cmdPushConstants(commands.command, layout.pipelineLayout, range.stages, range.offset, range.size,
                                    commands.pushData.data() + range.offset);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> recordBoundGroupWrites(VernonRuntimeRhiAdapter &adapter, const VulkanCommands &commands) {
    const PreparedLayout &layout = *commands.pipeline->layout;
    for (uint32_t index = 0; index < layout.groupCount; ++index)
        if ((layout.groupMask & (1u << index)) != 0 && !commands.groups[index]->writes.empty())
            if (auto recorded = recordGroupWrites(adapter, commands.encoder, *commands.groups[index]); !recorded)
                return recorded;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<VulkanCommands>(handle);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !commands || !commands->pipeline ||
        commands->pipeline->graphics || commands->rendering ||
        descriptor->push_constant_size > commands->pipeline->layout->pushConstantSize ||
        (descriptor->push_constant_size != 0 && !descriptor->push_constants))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "vulkan_adapter_received_an_invalid_dispatch");
    if (auto flushed = flushGroups(*commands, VK_PIPELINE_BIND_POINT_COMPUTE, commands->computeState); !flushed)
        return flushed;
    for (uint32_t index = 0; index < commands->pipeline->layout->groupCount; ++index) {
        if ((commands->pipeline->layout->groupMask & (1u << index)) == 0)
            continue;
        for (const VulkanGroup::ImageUse &use : commands->groups[index]->images) {
            auto transitioned =
                rhi::vulkan::transitionCommandImage(adapter.rhiDevice, commands->command, *use.image, use.layout);
            if (transitioned.isErr())
                return RhiAdapterResult<void>{vernon::err(providerError(std::move(transitioned).error()))};
        }
    }
    auto &driver = rhi::vulkan::driver();
    if (descriptor->push_constant_size != 0)
        driver.cmdPushConstants(commands->command, commands->pipeline->layout->pipelineLayout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(descriptor->push_constant_size),
                                descriptor->push_constants);
    driver.cmdDispatch(commands->command, descriptor->group_count[0], descriptor->group_count[1],
                       descriptor->group_count[2]);
    if (auto recorded = recordBoundGroupWrites(adapter, *commands); !recorded)
        return recorded;
    if (auto recorded = recordProviderCommand(adapter, commands->encoder, false); !recorded)
        return recorded;
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

// A render pass cannot change image layouts, so the images a scope's draws sample became readable when it began.
RhiAdapterResult<void> checkSampledImages(VernonRuntimeRhiAdapter &adapter, VulkanCommands &commands) {
    for (uint32_t index = 0; index < commands.pipeline->layout->groupCount; ++index) {
        if ((commands.pipeline->layout->groupMask & (1u << index)) == 0)
            continue;
        const VulkanGroup &group = *commands.groups[index];
        if (commands.checkedGroups[index] == group.serial && commands.checkedScopes[index] == commands.scope)
            continue;
        for (const VulkanGroup::ImageUse &use : group.images) {
            if (use.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                continue;
            auto readable = rhi::vulkan::imageInLayout(adapter.rhiDevice, commands.command, *use.image,
                                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (readable.isErr())
                return RhiAdapterResult<void>{vernon::err(providerError(std::move(readable).error()))};
            if (!readable.value())
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                       "vulkan_draw_samples_an_image_its_rendering_scope_does_not_declare");
        }
        commands.checkedGroups[index] = group.serial;
        commands.checkedScopes[index] = commands.scope;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> bindVertexBuffersResult(void *data, VernonRuntimeProviderObject handle, uint32_t firstBinding,
                                               const VernonRuntimeProviderResourceReference *references, size_t count) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<VulkanCommands>(handle);
    if (!commands || !references || count == 0 || firstBinding > VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS ||
        count > VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS - firstBinding)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_bind_vertex_buffers_is_invalid");
    std::array<VkBuffer, VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS> buffers{};
    std::array<VkDeviceSize, VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS> offsets{};
    for (size_t index = 0; index < count; ++index) {
        const VernonRuntimeProviderResourceReference &reference = references[index];
        if ((reference.identity & kRhiResourceKindMask) != kRhiBufferResource || reference.offset >= reference.size)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_vertex_buffer_is_invalid",
                                   firstBinding + index);
        if (auto retained = retainCommandResource(adapter, commands->encoder, reference); !retained)
            return retained;
        auto resolved = resolveCommandRhiResource(adapter, commands->encoder, reference);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        buffers[index] = reinterpret_cast<rhi::vulkan::Buffer *>(resolved.value())->buffer;
        offsets[index] = reference.offset;
    }
    rhi::vulkan::driver().cmdBindVertexBuffers(commands->command, firstBinding, static_cast<uint32_t>(count),
                                               buffers.data(), offsets.data());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDrawResult(void *data, VernonRuntimeProviderObject handle,
                                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<VulkanCommands>(handle);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !commands || !commands->pipeline ||
        !commands->pipeline->graphics || !commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "vulkan_adapter_received_an_invalid_draw_descriptor");
    if (auto opened = openCommandRendering(adapter, commands->encoder); !opened)
        return RhiAdapterResult<void>{vernon::err(std::move(opened).error())};
    if (auto flushed = flushGroups(*commands, VK_PIPELINE_BIND_POINT_GRAPHICS, commands->graphicsState); !flushed)
        return flushed;
    if (auto checked = checkSampledImages(adapter, *commands); !checked)
        return checked;
    auto &driver = rhi::vulkan::driver();
    const VkCommandBuffer command = commands->command;
    const std::array<uint32_t, 4> viewport{descriptor->viewport[0], descriptor->viewport[1], descriptor->viewport[2],
                                           descriptor->viewport[3]};
    if (!commands->viewportValid || commands->viewport != viewport) {
        const VkViewport native{static_cast<float>(viewport[0]),
                                static_cast<float>(viewport[1] + viewport[3]),
                                static_cast<float>(viewport[2]),
                                -static_cast<float>(viewport[3]),
                                0,
                                1};
        driver.cmdSetViewport(command, 0, 1, &native);
        commands->viewport = viewport;
        commands->viewportValid = true;
    }
    const std::array<uint32_t, 4> scissor{descriptor->scissor[0], descriptor->scissor[1], descriptor->scissor[2],
                                          descriptor->scissor[3]};
    if (!commands->scissorValid || commands->scissor != scissor) {
        const VkRect2D native{{static_cast<int32_t>(scissor[0]), static_cast<int32_t>(scissor[1])},
                              {scissor[2], scissor[3]}};
        driver.cmdSetScissor(command, 0, 1, &native);
        commands->scissor = scissor;
        commands->scissorValid = true;
    }
    if (!commands->stencilValid || commands->stencilReference != descriptor->stencil_reference) {
        driver.cmdSetStencilReference(command, VK_STENCIL_FACE_FRONT_AND_BACK, descriptor->stencil_reference);
        commands->stencilReference = descriptor->stencil_reference;
        commands->stencilValid = true;
    }
    if (descriptor->index_count) {
        const auto &indexBuffer = descriptor->index_buffer;
        if (commands->indexBufferKey != indexBuffer.resource.value) {
            if (auto retained = retainCommandResource(adapter, commands->encoder, indexBuffer); !retained)
                return retained;
            auto resolved = resolveCommandRhiResource(adapter, commands->encoder, indexBuffer);
            if (!resolved)
                return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
            commands->indexBuffer = reinterpret_cast<rhi::vulkan::Buffer *>(resolved.value())->buffer;
            commands->indexBufferKey = indexBuffer.resource.value;
            commands->indexBufferOffset = UINT64_MAX;
        }
        if (commands->indexBufferOffset != indexBuffer.offset) {
            driver.cmdBindIndexBuffer(command, commands->indexBuffer, indexBuffer.offset, VK_INDEX_TYPE_UINT32);
            commands->indexBufferOffset = indexBuffer.offset;
        }
        driver.cmdDrawIndexed(command, descriptor->index_count, descriptor->instance_count, 0, descriptor->base_vertex,
                              descriptor->first_instance);
    } else {
        driver.cmdDraw(command, descriptor->vertex_count, descriptor->instance_count, 0, descriptor->first_instance);
    }
    if (auto recorded = recordBoundGroupWrites(adapter, *commands); !recorded)
        return recorded;
    if (auto recorded = recordProviderCommand(adapter, commands->encoder, true); !recorded)
        return recorded;
    adapter.lastStencilReference.store(descriptor->stencil_reference, std::memory_order_relaxed);
    adapter.lastDrawIndexed.store(descriptor->index_count != 0, std::memory_order_relaxed);
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> beginRenderingResult(void *data, VernonRuntimeProviderObject handle,
                                            const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<VulkanCommands>(handle);
    if (!commands || !descriptor || commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_begin_rendering_is_invalid");
    if (auto begun = beginEncoderRendering(adapter, commands->encoder, *descriptor); !begun)
        return begun;
    commands->rendering = true;
    ++commands->scope;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endRenderingResult(void *data, VernonRuntimeProviderObject handle) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<VulkanCommands>(handle);
    if (!commands || !commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "vulkan_end_rendering_is_invalid");
    commands->rendering = false;
    return endEncoderRendering(adapter, commands->encoder);
}

void destroyShader(void *, VernonRuntimeProviderObject handle) {
    auto *shader = fromHandle<PreparedShader>(handle);
    if (!shader)
        return;
    rhi::vulkan::driver().destroyShaderModule(shader->device->device, shader->module, nullptr);
    delete shader;
}
void destroyGroupLayout(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<GroupLayout>(handle));
}
void destroyLayout(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<PreparedLayout>(handle));
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<PreparedPipeline>(handle));
}
void destroyGroup(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<VulkanGroup>(handle));
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "Vulkan shader preparation failed");
}
VernonStatus prepareGroupLayout(void *data, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareGroupLayoutResult(data, descriptor, output),
                          "Vulkan group layout preparation failed");
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
VernonStatus createGroup(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                         VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createGroupResult(data, descriptor, output), "Vulkan group creation failed");
}
VernonStatus beginCommands(void *data, VernonRuntimeProviderObject encoder, VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginCommandsResult(data, encoder, output), "Vulkan commands could not begin");
}
VernonStatus endCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endCommandsResult(data, commands), "Vulkan commands failed");
}
VernonStatus resumeCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, resumeCommandsResult(data, commands), "Vulkan commands could not resume");
}
VernonStatus bindPipeline(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject pipeline) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindPipelineResult(data, commands, pipeline), "Vulkan pipeline binding failed");
}
VernonStatus bindGroup(void *data, VernonRuntimeProviderObject commands, uint32_t index,
                       VernonRuntimeProviderObject group) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindGroupResult(data, commands, index, group), "Vulkan group binding failed");
}
VernonStatus bindVertexBuffers(void *data, VernonRuntimeProviderObject commands, uint32_t firstBinding,
                               const VernonRuntimeProviderResourceReference *buffers, size_t count) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, bindVertexBuffersResult(data, commands, firstBinding, buffers, count),
                          "Vulkan vertex buffer binding failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commands,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, commands, descriptor), "Vulkan dispatch encoding failed");
}
VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commands,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDrawResult(data, commands, descriptor), "Vulkan draw encoding failed");
}
VernonStatus beginRendering(void *data, VernonRuntimeProviderObject commands,
                            const VernonRuntimeProviderRenderingDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginRenderingResult(data, commands, descriptor),
                          "Vulkan rendering could not begin");
}
VernonStatus endRendering(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endRenderingResult(data, commands), "Vulkan rendering could not end");
}

} // namespace

void initializeVulkanProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_group_layout = prepareGroupLayout;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_group = createGroup;
    adapter.provider.begin_commands = beginCommands;
    adapter.provider.end_commands = endCommands;
    adapter.provider.resume_commands = resumeCommands;
    adapter.provider.encode_bind_pipeline = bindPipeline;
    adapter.provider.encode_bind_group = bindGroup;
    adapter.provider.encode_bind_vertex_buffers = bindVertexBuffers;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.begin_rendering = beginRendering;
    adapter.provider.end_rendering = endRendering;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_group_layout = destroyGroupLayout;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_group = destroyGroup;
}

} // namespace vernon::runtime::rhi_adapter

namespace vernon::runtime {

VernonRuntimeRhiAdapter *createVulkanRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend) {
    if (backend != VERNON_RHI_BACKEND_VULKAN)
        return nullptr;
    auto resolvedDeviceState = rhi::deviceState(device, backend);
    if (resolvedDeviceState.isErr())
        return nullptr;
    auto *deviceState = static_cast<rhi::vulkan::DeviceState *>(resolvedDeviceState.value());
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
