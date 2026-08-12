#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "logical_resource_record.h"
#include "rhi_test_hooks.h"
#include "sampler_filter.h"
#include "vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
struct VulkanBufferSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::vulkan::Buffer buffer;
    VernonRhiVulkanBorrowedBufferDescriptor descriptor{};
    VernonRhiBufferDescriptor ownedDescriptor{};
};

struct VulkanImageSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::vulkan::Image image;
    VernonRhiVulkanBorrowedImageDescriptor descriptor{};
    VernonRhiImageDescriptor ownedDescriptor{};
};

struct VulkanImageViewSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::vulkan::Image bindingImage;
    VernonRhiImageViewDescriptor descriptor{};
    uint64_t imageResource{};
    bool ownedNative{};
};

struct VulkanSamplerSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::vulkan::Sampler sampler;
};

struct VulkanInteropDevice {
    vernon::rhi::vulkan::DeviceState state;
    std::deque<VulkanBufferSlot> buffers;
    std::deque<VulkanImageSlot> images;
    std::deque<VulkanSamplerSlot> samplers;
    std::deque<VulkanImageViewSlot> imageViews;
    uint32_t queueCapabilities{};
    bool owned{};
    std::string error;
    std::mutex mutex;
};

struct VulkanDeviceSlot {
    std::shared_ptr<VulkanInteropDevice> device;
    uint32_t generation{1};
};

constexpr uint32_t vulkanDeviceBit = uint32_t{1} << 30;
std::vector<VulkanDeviceSlot> vulkanDevices;
std::mutex deviceMutex;

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

bool decodeResourceKey(uint64_t key, uint32_t &index, uint32_t &generation) {
    const uint64_t encodedIndex = key & UINT32_MAX;
    generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return false;
    index = static_cast<uint32_t>(encodedIndex - 1);
    return true;
}

template <typename Slots> auto *lookupResourceRecord(Slots &slots, uint64_t key) {
    uint32_t index = 0;
    uint32_t generation = 0;
    if (!decodeResourceKey(key, index, generation) || index >= slots.size())
        return static_cast<typename Slots::value_type *>(nullptr);
    auto &slot = slots[index];
    return slot.isRetained(generation) ? &slot : nullptr;
}

std::shared_ptr<VulkanInteropDevice> lookupVulkanDevice(VernonRhiDevice handle) {
    if ((handle.index & vulkanDeviceBit) == 0)
        return {};
    const uint32_t index = handle.index & ~vulkanDeviceBit;
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (index >= vulkanDevices.size())
        return {};
    VulkanDeviceSlot &slot = vulkanDevices[index];
    return slot.device && slot.generation == handle.generation ? slot.device : std::shared_ptr<VulkanInteropDevice>{};
}

template <typename Slots, typename Handle> typename Slots::value_type *lookupVulkanSlot(Slots &slots, Handle handle) {
    if (handle.index >= slots.size())
        return nullptr;
    auto &slot = slots[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}

template <typename Slots, typename Handle>
typename Slots::value_type &allocateVulkanSlot(Slots &slots, Handle &output) {
    uint32_t index = 0;
    while (index < slots.size() && slots[index].occupied)
        ++index;
    if (index == slots.size())
        slots.emplace_back();
    auto &slot = slots[index];
    slot.publish();
    output = {index, slot.generation};
    return slot;
}

template <typename Slot> void releaseVulkanSlot(Slot &slot) {
    if (slot.publicAlive)
        slot.destroyPublicOwner();
    slot.recycle();
}

VulkanImageViewSlot *lookupVulkanImageView(std::deque<VulkanImageViewSlot> &slots, VernonRhiImageView handle) {
    if (handle.index >= slots.size())
        return nullptr;
    VulkanImageViewSlot &slot = slots[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}

VulkanImageViewSlot &allocateVulkanImageView(std::deque<VulkanImageViewSlot> &slots, VernonRhiImageView &output) {
    uint32_t index = 0;
    while (index < slots.size() && slots[index].occupied)
        ++index;
    if (index == slots.size())
        slots.emplace_back();
    VulkanImageViewSlot &slot = slots[index];
    slot.publish();
    output = {index, slot.generation};
    return slot;
}

void releaseVulkanImageView(VulkanImageViewSlot &slot) {
    if (slot.publicAlive)
        slot.destroyPublicOwner();
    slot.recycle();
}

template <typename Handle> Handle vulkanHandle(uint64_t bits) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(bits);
    else
        return static_cast<Handle>(bits);
}

template <typename Handle> uint64_t vulkanHandleBits(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<uint64_t>(handle);
    else
        return static_cast<uint64_t>(handle);
}

void restoreVulkanImageLayout(void *context, uint64_t layout) {
    static_cast<vernon::rhi::vulkan::Image *>(context)->layout = static_cast<VkImageLayout>(layout);
}

VkFormat vulkanFormat(VernonRhiFormat format) {
    constexpr VkFormat formats[] = {
        VK_FORMAT_UNDEFINED,           VK_FORMAT_R8_UNORM,
        VK_FORMAT_R8G8_UNORM,          VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_SRGB,       VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_R8G8B8_UNORM,        VK_FORMAT_R32G32_SFLOAT,
        VK_FORMAT_R32G32B32_SFLOAT,    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
    };
    const uint32_t index = static_cast<uint32_t>(format);
    return index < sizeof(formats) / sizeof(formats[0]) ? formats[index] : VK_FORMAT_UNDEFINED;
}

VkImageLayout vulkanImageLayout(VernonRhiResourceState state) {
    switch (state) {
    case VERNON_RHI_STATE_UNDEFINED:
        return VK_IMAGE_LAYOUT_UNDEFINED;
    case VERNON_RHI_STATE_COMMON:
        return VK_IMAGE_LAYOUT_GENERAL;
    case VERNON_RHI_STATE_TRANSFER_SOURCE:
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case VERNON_RHI_STATE_TRANSFER_DESTINATION:
        return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case VERNON_RHI_STATE_SHADER_READ:
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case VERNON_RHI_STATE_SHADER_WRITE:
        return VK_IMAGE_LAYOUT_GENERAL;
    case VERNON_RHI_STATE_COLOR_ATTACHMENT:
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT:
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case VERNON_RHI_STATE_PRESENT:
        return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

VkImageLayout defaultVulkanImageLayout(const VernonRhiImageDescriptor &descriptor) {
    if (descriptor.usage & VERNON_RHI_IMAGE_STORAGE)
        return VK_IMAGE_LAYOUT_GENERAL;
    if (descriptor.usage & VERNON_RHI_IMAGE_SAMPLED)
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (descriptor.usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    if (descriptor.usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE)
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
}

void vulkanBufferDependency(VernonRhiResourceState state, VkPipelineStageFlags &stages, VkAccessFlags &access) {
    switch (state) {
    case VERNON_RHI_STATE_TRANSFER_SOURCE:
        stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        access = VK_ACCESS_TRANSFER_READ_BIT;
        return;
    case VERNON_RHI_STATE_TRANSFER_DESTINATION:
        stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        access = VK_ACCESS_TRANSFER_WRITE_BIT;
        return;
    case VERNON_RHI_STATE_SHADER_READ:
        stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                 VK_ACCESS_INDEX_READ_BIT;
        return;
    case VERNON_RHI_STATE_SHADER_WRITE:
        stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        return;
    case VERNON_RHI_STATE_COLOR_ATTACHMENT:
        stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        return;
    case VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT:
        stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        return;
    case VERNON_RHI_STATE_PRESENT:
        stages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        access = VK_ACCESS_MEMORY_READ_BIT;
        return;
    case VERNON_RHI_STATE_UNDEFINED:
        stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        access = 0;
        return;
    default:
        stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        return;
    }
}

VkPipelineStageFlags vulkanShaderStages(uint32_t stages) {
    VkPipelineStageFlags result = 0;
    if (stages & VERNON_RHI_STAGE_COMPUTE)
        result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (stages & VERNON_RHI_STAGE_VERTEX)
        result |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    if (stages & VERNON_RHI_STAGE_FRAGMENT)
        result |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    return result;
}

VkAccessFlags vulkanAccess(uint32_t access) {
    VkAccessFlags result = 0;
    if (access & VERNON_RHI_ACCESS_TRANSFER_READ)
        result |= VK_ACCESS_TRANSFER_READ_BIT;
    if (access & VERNON_RHI_ACCESS_TRANSFER_WRITE)
        result |= VK_ACCESS_TRANSFER_WRITE_BIT;
    if (access & VERNON_RHI_ACCESS_SHADER_READ)
        result |= VK_ACCESS_SHADER_READ_BIT;
    if (access & VERNON_RHI_ACCESS_SHADER_WRITE)
        result |= VK_ACCESS_SHADER_WRITE_BIT;
    if (access & VERNON_RHI_ACCESS_COLOR_READ)
        result |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    if (access & VERNON_RHI_ACCESS_COLOR_WRITE)
        result |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    if (access & VERNON_RHI_ACCESS_DEPTH_STENCIL_READ)
        result |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    if (access & VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE)
        result |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (access & VERNON_RHI_ACCESS_VERTEX_READ)
        result |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    if (access & VERNON_RHI_ACCESS_INDEX_READ)
        result |= VK_ACCESS_INDEX_READ_BIT;
    if (access & VERNON_RHI_ACCESS_INDIRECT_READ)
        result |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    if (access & VERNON_RHI_ACCESS_HOST_READ)
        result |= VK_ACCESS_HOST_READ_BIT;
    if (access & VERNON_RHI_ACCESS_HOST_WRITE)
        result |= VK_ACCESS_HOST_WRITE_BIT;
    return result;
}

void vulkanBarrierDependency(VernonRhiResourceState state, uint32_t stageMask, uint32_t accessMask,
                             VkPipelineStageFlags &stages, VkAccessFlags &access) {
    vulkanBufferDependency(state, stages, access);
    if ((state == VERNON_RHI_STATE_SHADER_READ || state == VERNON_RHI_STATE_SHADER_WRITE) && stageMask) {
        const VkPipelineStageFlags explicitStages = vulkanShaderStages(stageMask);
        if (explicitStages)
            stages = explicitStages;
    }
    if (accessMask)
        access = vulkanAccess(accessMask);
}

void vulkanImageDependency(VkImageLayout layout, VkPipelineStageFlags &stages, VkAccessFlags &access) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        access = 0;
        return;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        access = VK_ACCESS_TRANSFER_READ_BIT;
        return;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        access = VK_ACCESS_TRANSFER_WRITE_BIT;
        return;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        access = VK_ACCESS_SHADER_READ_BIT;
        return;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        return;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        return;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        stages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        access = VK_ACCESS_MEMORY_READ_BIT;
        return;
    default:
        stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        return;
    }
}

void restoreVulkanImageLayouts(void *context, uint64_t encoderKey) {
    auto &image = *static_cast<vernon::rhi::vulkan::Image *>(context);
    const auto found = image.layoutJournals.find(encoderKey);
    if (found == image.layoutJournals.end())
        return;
    image.layout = found->second.layout;
    image.subresourceLayouts = found->second.subresources;
}

void clearVulkanImageLayoutJournal(void *context, uint64_t encoderKey) {
    static_cast<vernon::rhi::vulkan::Image *>(context)->layoutJournals.erase(encoderKey);
}

void transitionVulkanImage(VkCommandBuffer command, VulkanImageSlot &slot, VkImageLayout target,
                           bool forceMemoryDependency = false, const VernonRhiBarrier *dependency = nullptr) {
    auto &image = slot.image;
    const VernonRhiImageDescriptor &descriptor = image.owned ? slot.ownedDescriptor : slot.descriptor.descriptor;
    const size_t planeStride = static_cast<size_t>(descriptor.mip_levels) * descriptor.array_layers;
    const size_t subresourceCount = planeStride;
    if (image.subresourceLayouts.size() != subresourceCount)
        return;
    const bool alreadyTarget = std::all_of(image.subresourceLayouts.begin(), image.subresourceLayouts.end(),
                                           [target](VkImageLayout layout) { return layout == target; });
    if (alreadyTarget && !forceMemoryDependency)
        return;
    const auto emit = [&](VkImageLayout source, uint32_t baseMip, uint32_t mipCount, uint32_t baseLayer,
                          uint32_t layerCount) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = source;
        barrier.newLayout = target;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        const VkImageAspectFlags aspect = vernon::rhi::vulkan::imageAspectMask(image.format);
        barrier.subresourceRange = {aspect, baseMip, mipCount, baseLayer, layerCount};
        VkPipelineStageFlags sourceStage{};
        VkPipelineStageFlags destinationStage{};
        vulkanImageDependency(source, sourceStage, barrier.srcAccessMask);
        vulkanImageDependency(target, destinationStage, barrier.dstAccessMask);
        if (dependency) {
            vulkanBarrierDependency(static_cast<VernonRhiResourceState>(dependency->old_state),
                                    dependency->source_stage_mask, dependency->source_access, sourceStage,
                                    barrier.srcAccessMask);
            vulkanBarrierDependency(static_cast<VernonRhiResourceState>(dependency->new_state),
                                    dependency->destination_stage_mask, dependency->destination_access,
                                    destinationStage, barrier.dstAccessMask);
        }
        vernon::rhi::vulkan::driver().cmdPipelineBarrier(command, sourceStage, destinationStage, 0, 0, nullptr, 0,
                                                         nullptr, 1, &barrier);
    };
    const bool uniform = std::all_of(image.subresourceLayouts.begin(), image.subresourceLayouts.end(),
                                     [&](VkImageLayout layout) { return layout == image.subresourceLayouts.front(); });
    if (uniform) {
        if (image.subresourceLayouts.front() != target || forceMemoryDependency)
            emit(image.subresourceLayouts.front(), 0, descriptor.mip_levels, 0, descriptor.array_layers);
    } else {
        for (uint32_t layer = 0; layer < descriptor.array_layers; ++layer)
            for (uint32_t mip = 0; mip < descriptor.mip_levels; ++mip) {
                const VkImageLayout source = image.subresourceLayouts[mip + layer * descriptor.mip_levels];
                if (source != target || forceMemoryDependency)
                    emit(source, mip, 1, layer, 1);
            }
    }
    std::fill(image.subresourceLayouts.begin(), image.subresourceLayouts.end(), target);
    image.layout = target;
}

void validate(const VulkanInteropDevice &device) {
    for (const VulkanBufferSlot &buffer : device.buffers)
        buffer.validate();
    for (const VulkanImageSlot &image : device.images)
        image.validate();
    for (const VulkanSamplerSlot &sampler : device.samplers)
        sampler.validate();
    for (const VulkanImageViewSlot &view : device.imageViews) {
        view.validate();
        if (!view.occupied)
            continue;
        assert(view.descriptor.image.index < device.images.size());
        const VulkanImageSlot &image = device.images[view.descriptor.image.index];
        assert(image.occupied && image.generation == view.descriptor.image.generation);
    }
}
} // namespace

namespace vernon::rhi::vulkan_api {

extern "C" VERNON_RHI_CAPI VernonRhiDevice
vernonRhiCreateBorrowedVulkanDevice(const VernonRhiVulkanBorrowedDeviceDescriptor *descriptor) {
    constexpr uint32_t allQueueCapabilities =
        VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->instance ||
        !descriptor->physical_device || !descriptor->device || !descriptor->queue || !descriptor->command_buffer ||
        (descriptor->queue_capabilities & ~allQueueCapabilities) != 0)
        return invalidDevice();
    auto device = std::shared_ptr<VulkanInteropDevice>(new (std::nothrow) VulkanInteropDevice());
    if (!device ||
        !device->state.initializeBorrowed(
            static_cast<VkInstance>(descriptor->instance), static_cast<VkPhysicalDevice>(descriptor->physical_device),
            static_cast<VkDevice>(descriptor->device), static_cast<VkQueue>(descriptor->queue),
            descriptor->queue_family_index, static_cast<VkCommandBuffer>(descriptor->command_buffer), device->error))
        return invalidDevice();
    uint32_t queueFamilyCount = 0;
    vernon::rhi::vulkan::driver().getPhysicalDeviceQueueFamilyProperties(device->state.physicalDevice,
                                                                         &queueFamilyCount, nullptr);
    if (descriptor->queue_family_index >= queueFamilyCount) {
        device->state.shutdown();
        return invalidDevice();
    }
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vernon::rhi::vulkan::driver().getPhysicalDeviceQueueFamilyProperties(device->state.physicalDevice,
                                                                         &queueFamilyCount, queueFamilies.data());
    VkQueueFlags required = 0;
    if ((descriptor->queue_capabilities & VERNON_RHI_QUEUE_TRANSFER) != 0)
        required |= VK_QUEUE_TRANSFER_BIT;
    if ((descriptor->queue_capabilities & VERNON_RHI_QUEUE_COMPUTE) != 0)
        required |= VK_QUEUE_COMPUTE_BIT;
    if ((descriptor->queue_capabilities & VERNON_RHI_QUEUE_GRAPHICS) != 0)
        required |= VK_QUEUE_GRAPHICS_BIT;
    if ((queueFamilies[descriptor->queue_family_index].queueFlags & required) != required) {
        device->state.shutdown();
        return invalidDevice();
    }
    device->queueCapabilities = descriptor->queue_capabilities;
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < vulkanDevices.size() && vulkanDevices[index].device)
        ++index;
    if (index == vulkanDevices.size()) {
        if (index >= vulkanDeviceBit)
            return invalidDevice();
        vulkanDevices.emplace_back();
    }
    vulkanDevices[index].device = std::move(device);
    return {index | vulkanDeviceBit, vulkanDevices[index].generation};
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedQueue(VernonRhiDevice handle,
                                                                                 void **output) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.queue;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedCommandBuffer(VernonRhiDevice handle,
                                                                                         void **output) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.borrowedCommandBuffer;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedBuffer(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedBufferDescriptor *descriptor, VernonRhiBuffer *output) {
    constexpr uint32_t allUsages = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION |
                                   VERNON_RHI_BUFFER_UNIFORM | VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX |
                                   VERNON_RHI_BUFFER_INDEX | VERNON_RHI_BUFFER_INDIRECT;
    auto device = lookupVulkanDevice(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->buffer ||
        !descriptor->size || (descriptor->usage & ~allUsages) != 0 || descriptor->state == VERNON_RHI_STATE_UNDEFINED ||
        !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanBufferSlot &slot = allocateVulkanSlot(device->buffers, *output);
    slot.buffer.buffer = vulkanHandle<VkBuffer>(descriptor->buffer);
    slot.buffer.owned = false;
    slot.descriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImage(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageDescriptor *descriptor, VernonRhiImage *output) {
    constexpr uint32_t allUsages = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION |
                                   VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_STORAGE |
                                   VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT;
    auto device = lookupVulkanDevice(handle);
    const bool validCube =
        descriptor && (descriptor->descriptor.dimension != VERNON_RHI_IMAGE_CUBE ||
                       (descriptor->descriptor.width == descriptor->descriptor.height &&
                        descriptor->descriptor.depth == 1 && descriptor->descriptor.array_layers == 6));
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->image ||
        descriptor->descriptor.struct_size < sizeof(descriptor->descriptor) ||
        vulkanFormat(descriptor->descriptor.format) == VK_FORMAT_UNDEFINED || !descriptor->descriptor.width ||
        !descriptor->descriptor.height || !descriptor->descriptor.depth || !descriptor->descriptor.mip_levels ||
        !descriptor->descriptor.array_layers || !descriptor->descriptor.sample_count ||
        (descriptor->descriptor.usage & ~allUsages) != 0 || !validCube || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot &slot = allocateVulkanSlot(device->images, *output);
    slot.image.image = vulkanHandle<VkImage>(descriptor->image);
    slot.image.format = vulkanFormat(descriptor->descriptor.format);
    slot.image.layout = vulkanImageLayout(descriptor->state);
    try {
        slot.image.subresourceLayouts.assign(static_cast<size_t>(descriptor->descriptor.mip_levels) *
                                                 descriptor->descriptor.array_layers,
                                             slot.image.layout);
    } catch (const std::bad_alloc &) {
        releaseVulkanSlot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot.image.colorAttachment = (descriptor->descriptor.usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0;
    slot.image.owned = false;
    slot.descriptor = *descriptor;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImageView(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageViewDescriptor *descriptor, VernonRhiImageView *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->image_view ||
        descriptor->descriptor.struct_size < sizeof(descriptor->descriptor) || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot *image = lookupVulkanSlot(device->images, descriptor->descriptor.image);
    if (!image || descriptor->descriptor.format != image->descriptor.descriptor.format ||
        !descriptor->descriptor.mip_level_count || !descriptor->descriptor.array_layer_count ||
        descriptor->descriptor.base_mip_level + descriptor->descriptor.mip_level_count >
            image->descriptor.descriptor.mip_levels ||
        descriptor->descriptor.base_array_layer + descriptor->descriptor.array_layer_count >
            image->descriptor.descriptor.array_layers)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!image->retain())
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    VulkanImageViewSlot &slot = allocateVulkanImageView(device->imageViews, *output);
    slot.bindingImage = image->image;
    slot.bindingImage.view = vulkanHandle<VkImageView>(descriptor->image_view);
    slot.bindingImage.owned = false;
    slot.descriptor = descriptor->descriptor;
    slot.imageResource = resourceKey(descriptor->descriptor.image);
    slot.ownedNative = false;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiVulkanDeviceGetBufferNativeHandle(VernonRhiDevice handle,
                                                                                      VernonRhiBuffer buffer,
                                                                                      uint64_t *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = vulkanHandleBits(slot->buffer.buffer);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createImageView(VernonRhiDevice handle, const VernonRhiImageViewDescriptor *descriptor,
                                VernonRhiImageView *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !output ||
        !descriptor->mip_level_count || !descriptor->array_layer_count || !descriptor->aspects)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot *image = lookupVulkanSlot(device->images, descriptor->image);
    if (!image)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiImageDescriptor &imageDescriptor =
        image->image.owned ? image->ownedDescriptor : image->descriptor.descriptor;
    if (!vernon::rhi::validImageViewDescriptor(imageDescriptor, *descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VkImageAspectFlags aspectMask = 0;
    aspectMask |= (descriptor->aspects & VERNON_RHI_IMAGE_ASPECT_COLOR) ? VK_IMAGE_ASPECT_COLOR_BIT : 0;
    aspectMask |= (descriptor->aspects & VERNON_RHI_IMAGE_ASPECT_DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0;
    aspectMask |= (descriptor->aspects & VERNON_RHI_IMAGE_ASPECT_STENCIL) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0;
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image->image.image;
    info.viewType = descriptor->dimension == VERNON_RHI_IMAGE_3D     ? VK_IMAGE_VIEW_TYPE_3D
                    : descriptor->dimension == VERNON_RHI_IMAGE_CUBE ? VK_IMAGE_VIEW_TYPE_CUBE
                    : descriptor->array_layer_count > 1              ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                                     : VK_IMAGE_VIEW_TYPE_2D;
    info.format = vulkanFormat(descriptor->format);
    info.subresourceRange = {aspectMask, descriptor->base_mip_level, descriptor->mip_level_count,
                             descriptor->base_array_layer, descriptor->array_layer_count};
    VkImageView nativeView{};
    if (vernon::rhi::vulkan::driver().createImageView(device->state.device, &info, nullptr, &nativeView) != VK_SUCCESS)
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    if (!image->retain()) {
        vernon::rhi::vulkan::driver().destroyImageView(device->state.device, nativeView, nullptr);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    VulkanImageViewSlot &slot = allocateVulkanImageView(device->imageViews, *output);
    slot.bindingImage = image->image;
    slot.bindingImage.view = nativeView;
    slot.bindingImage.owned = false;
    slot.descriptor = *descriptor;
    slot.imageResource = resourceKey(descriptor->image);
    slot.ownedNative = true;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyImageView(VernonRhiDevice handle, VernonRhiImageView imageView) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageViewSlot *slot = lookupVulkanImageView(device->imageViews, imageView);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t parentKey = slot->imageResource;
    slot->destroyPublicOwner();
    if (slot->bindingReferences != 0)
        return VERNON_RHI_STATUS_OK;
    if (slot->ownedNative)
        vernon::rhi::vulkan::driver().destroyImageView(device->state.device, slot->bindingImage.view, nullptr);
    releaseVulkanImageView(*slot);
    VulkanImageSlot *parent = lookupResourceRecord(device->images, parentKey);
    if (parent && parent->release()) {
        device->state.destroyImage(parent->image);
        releaseVulkanSlot(*parent);
    }
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView imageView, uint64_t *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageViewSlot *slot = lookupVulkanImageView(device->imageViews, imageView);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = vulkanHandleBits(slot->bindingImage.view);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiDevice createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {

    auto device = std::shared_ptr<VulkanInteropDevice>(new (std::nothrow) VulkanInteropDevice());
    if (!device) {
        vernon::rhi::setDeviceCreationError("cannot allocate Vulkan RHI device state");
        return invalidDevice();
    }
    if (!device->state.initialize(descriptor->device_index, device->error)) {
        vernon::rhi::setDeviceCreationError(std::move(device->error));
        return invalidDevice();
    }
    device->owned = true;
    device->queueCapabilities = VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < vulkanDevices.size() && vulkanDevices[index].device)
        ++index;
    if (index == vulkanDevices.size())
        vulkanDevices.emplace_back();
    vulkanDevices[index].device = std::move(device);
    return {index | vulkanDeviceBit, vulkanDevices[index].generation};
}

void destroyDevice(VernonRhiDevice handle) {

    std::shared_ptr<VulkanInteropDevice> device;
    {
        const uint32_t index = handle.index & ~vulkanDeviceBit;
        std::lock_guard<std::mutex> guard(deviceMutex);
        if (index >= vulkanDevices.size() || vulkanDevices[index].generation != handle.generation)
            return;
        VulkanDeviceSlot &slot = vulkanDevices[index];
        device = std::move(slot.device);
        ++slot.generation;
        if (slot.generation == 0)
            slot.generation = 1;
    }
    if (device) {
        std::lock_guard<std::mutex> guard(device->mutex);
        for (VulkanBufferSlot &buffer : device->buffers)
            if (buffer.occupied)
                device->state.destroyBuffer(buffer.buffer);
        for (VulkanImageSlot &image : device->images)
            if (image.occupied)
                device->state.destroyImage(image.image);
        for (VulkanSamplerSlot &sampler : device->samplers)
            if (sampler.occupied)
                device->state.destroySampler(sampler.sampler);
        device->state.shutdown();
    }
    return;
}

VernonStringView lastError(VernonRhiDevice handle) {
    auto device = lookupVulkanDevice(handle);
    return device ? VernonStringView{device->error.data(), device->error.size()} : VernonStringView{};
}

VernonRhiStatus synchronize(VernonRhiDevice handle) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return device->state.synchronize(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor,
                             VernonRhiBuffer *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanBufferSlot &slot = allocateVulkanSlot(device->buffers, *output);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_STORAGE) != 0)
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_VERTEX) != 0)
        usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_INDEX) != 0)
        usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_UNIFORM) != 0)
        usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (descriptor->size > (std::numeric_limits<VkDeviceSize>::max)() - 3) {
        releaseVulkanSlot(slot);
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    const VkDeviceSize allocationSize = (descriptor->size + 3) & ~VkDeviceSize{3};
    if (!device->state.createBuffer(slot.buffer, allocationSize, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    device->error)) {
        releaseVulkanSlot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot.ownedDescriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                   const VernonRhiBufferUploadRange *ranges, size_t rangeCount);

VernonRhiStatus uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size) {
    const VernonRhiBufferUploadRange range{offset, source, size};
    return uploadBufferRanges(handle, buffer, &range, 1);
}

VernonRhiStatus uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                   const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    if (!ranges || rangeCount == 0 || rangeCount > (std::numeric_limits<uint32_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::vector<VkBufferCopy> copies;
    copies.reserve(rangeCount);
    VkDeviceSize packedSize = 0;
    VkDeviceSize destinationBegin = (std::numeric_limits<VkDeviceSize>::max)();
    VkDeviceSize destinationEnd = 0;
    bool allCopiesAligned = true;
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot->ownedDescriptor.size || range.size > slot->ownedDescriptor.size - range.offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        if (packedSize > (std::numeric_limits<VkDeviceSize>::max)() - 15)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        packedSize = (packedSize + 15) & ~VkDeviceSize{15};
        if (range.size > (std::numeric_limits<VkDeviceSize>::max)() - packedSize)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        copies.push_back({packedSize, range.offset, range.size});
        packedSize += range.size;
        destinationBegin = std::min(destinationBegin, range.offset);
        destinationEnd = std::max(destinationEnd, range.offset + range.size);
        allCopiesAligned = allCopiesAligned && (range.offset & 3) == 0 && (range.size & 3) == 0;
    }
    auto &driver = vernon::rhi::vulkan::driver();
    if (!allCopiesAligned) {
        std::vector<std::pair<VkDeviceSize, VkDeviceSize>> alignedRanges;
        alignedRanges.reserve(rangeCount);
        for (size_t index = 0; index < rangeCount; ++index) {
            const VkDeviceSize begin = ranges[index].offset & ~VkDeviceSize{3};
            const VkDeviceSize end = (ranges[index].offset + ranges[index].size + 3) & ~VkDeviceSize{3};
            alignedRanges.emplace_back(begin, end);
        }
        std::sort(alignedRanges.begin(), alignedRanges.end());
        size_t mergedCount = 0;
        for (const auto &[begin, end] : alignedRanges) {
            if (mergedCount && begin <= alignedRanges[mergedCount - 1].second) {
                alignedRanges[mergedCount - 1].second = std::max(alignedRanges[mergedCount - 1].second, end);
            } else {
                alignedRanges[mergedCount++] = {begin, end};
            }
        }
        alignedRanges.resize(mergedCount);

        VkDeviceSize alignedSize = 0;
        std::vector<VkBufferCopy> readbackCopies;
        readbackCopies.reserve(mergedCount);
        for (const auto &[begin, end] : alignedRanges) {
            const VkDeviceSize size = end - begin;
            if (size > (std::numeric_limits<VkDeviceSize>::max)() - alignedSize)
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            readbackCopies.push_back({begin, alignedSize, size});
            alignedSize += size;
        }
        if (alignedSize > (std::numeric_limits<size_t>::max)())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::vector<uint8_t> contents;
        try {
            contents.resize(static_cast<size_t>(alignedSize));
        } catch (const std::bad_alloc &) {
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        VkBuffer readback{};
        VkDeviceSize readbackOffset{};
        uint8_t *readbackMapped = nullptr;
        if (!device->state.acquireStaging(false, alignedSize, 16, readback, readbackOffset, readbackMapped,
                                          device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        for (VkBufferCopy &copy : readbackCopies)
            copy.dstOffset += readbackOffset;
        VkCommandBuffer readbackCommand{};
        if (!device->state.beginCommands(readbackCommand, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        driver.cmdCopyBuffer(readbackCommand, slot->buffer.buffer, readback,
                             static_cast<uint32_t>(readbackCopies.size()), readbackCopies.data());
        if (!device->state.submitCommands(readbackCommand, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        for (size_t index = 0; index < alignedRanges.size(); ++index) {
            const VkDeviceSize size = alignedRanges[index].second - alignedRanges[index].first;
            std::memcpy(contents.data() + (readbackCopies[index].dstOffset - readbackOffset),
                        readbackMapped + (readbackCopies[index].dstOffset - readbackOffset), static_cast<size_t>(size));
        }
        for (size_t rangeIndex = 0; rangeIndex < rangeCount; ++rangeIndex) {
            for (size_t alignedIndex = 0; alignedIndex < alignedRanges.size(); ++alignedIndex) {
                const auto [begin, end] = alignedRanges[alignedIndex];
                if (ranges[rangeIndex].offset < begin || ranges[rangeIndex].offset + ranges[rangeIndex].size > end)
                    continue;
                const VkDeviceSize packedBegin = readbackCopies[alignedIndex].dstOffset - readbackOffset;
                std::memcpy(contents.data() + packedBegin + ranges[rangeIndex].offset - begin,
                            ranges[rangeIndex].source, static_cast<size_t>(ranges[rangeIndex].size));
                break;
            }
        }

        VkBuffer staging{};
        VkDeviceSize stagingOffset{};
        uint8_t *mapped = nullptr;
        if (!device->state.acquireStaging(true, alignedSize, 16, staging, stagingOffset, mapped, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::memcpy(mapped, contents.data(), contents.size());
        std::vector<VkBufferCopy> uploadCopies;
        uploadCopies.reserve(alignedRanges.size());
        VkDeviceSize sourceOffset = stagingOffset;
        for (const auto &[begin, end] : alignedRanges) {
            uploadCopies.push_back({sourceOffset, begin, end - begin});
            sourceOffset += end - begin;
        }
        VkCommandBuffer command{};
        if (!device->state.beginCommands(command, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        driver.cmdCopyBuffer(command, staging, slot->buffer.buffer, static_cast<uint32_t>(uploadCopies.size()),
                             uploadCopies.data());
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                                VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = slot->buffer.buffer;
        barrier.offset = destinationBegin;
        barrier.size = destinationEnd - destinationBegin;
        driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                  nullptr, 1, &barrier, 0, nullptr);
        return device->state.submitCommands(command, device->error) ? VERNON_RHI_STATUS_OK
                                                                    : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(true, packedSize, 16, staging, stagingOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    for (size_t index = 0; index < rangeCount; ++index) {
        std::memcpy(mapped + copies[index].srcOffset, ranges[index].source, static_cast<size_t>(ranges[index].size));
        copies[index].srcOffset += stagingOffset;
    }
    VkCommandBuffer command{};
    if (!device->state.beginCommands(command, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    driver.cmdCopyBuffer(command, staging, slot->buffer.buffer, static_cast<uint32_t>(copies.size()), copies.data());
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                            VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = slot->buffer.buffer;
    barrier.offset = destinationBegin;
    barrier.size = destinationEnd - destinationBegin;
    driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                              nullptr, 1, &barrier, 0, nullptr);
    return device->state.submitCommands(command, device->error) ? VERNON_RHI_STATUS_OK
                                                                : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                               uint64_t size) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!destination || size == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
    if (!slot || offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VkDeviceSize alignedBegin = offset & ~VkDeviceSize{3};
    const VkDeviceSize alignedEnd = (offset + size + 3) & ~VkDeviceSize{3};
    const VkDeviceSize alignedSize = alignedEnd - alignedBegin;
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(false, alignedSize, 16, staging, stagingOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    VkCommandBuffer command{};
    if (!device->state.beginCommands(command, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    const VkBufferCopy copy{alignedBegin, stagingOffset, alignedSize};
    vernon::rhi::vulkan::driver().cmdCopyBuffer(command, slot->buffer.buffer, staging, 1, &copy);
    if (!device->state.submitCommands(command, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    std::memcpy(destination, mapped + (offset - alignedBegin), static_cast<size_t>(size));
    return VERNON_RHI_STATUS_OK;
}

uint32_t isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupVulkanSlot(device->buffers, buffer) != nullptr;
}

VernonRhiStatus createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor,
                            VernonRhiImage *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }

    const VkFormat format = descriptor ? vulkanFormat(descriptor->format) : VK_FORMAT_UNDEFINED;
    const bool depthFormat = descriptor && (descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT ||
                                            descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || format == VK_FORMAT_UNDEFINED ||
        descriptor->width == 0 || descriptor->height == 0 || descriptor->depth == 0 || descriptor->mip_levels == 0 ||
        descriptor->sample_count != 1 || (depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)) ||
        (!depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VkFormatFeatureFlags requiredFeatures = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_SAMPLED) != 0)
        requiredFeatures |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_STORAGE) != 0)
        requiredFeatures |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0)
        requiredFeatures |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0)
        requiredFeatures |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkFormatProperties formatProperties{};
    vernon::rhi::vulkan::driver().getPhysicalDeviceFormatProperties(device->state.physicalDevice, format,
                                                                    &formatProperties);
    if ((formatProperties.optimalTilingFeatures & requiredFeatures) != requiredFeatures) {
        device->error = "Vulkan image format does not support the requested optimal-tiling usage";
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }
    VulkanImageSlot &slot = allocateVulkanSlot(device->images, *output);
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                      (descriptor->dimension == VERNON_RHI_IMAGE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0);
    imageInfo.imageType = descriptor->dimension == VERNON_RHI_IMAGE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {descriptor->width, descriptor->height, descriptor->depth};
    imageInfo.mipLevels = descriptor->mip_levels;
    imageInfo.arrayLayers = descriptor->dimension == VERNON_RHI_IMAGE_3D ? 1 : descriptor->array_layers;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_SAMPLED) != 0)
        imageInfo.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_STORAGE) != 0)
        imageInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0)
        imageInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0)
        imageInfo.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.viewType = descriptor->dimension == VERNON_RHI_IMAGE_3D     ? VK_IMAGE_VIEW_TYPE_3D
                        : descriptor->dimension == VERNON_RHI_IMAGE_CUBE ? VK_IMAGE_VIEW_TYPE_CUBE
                                                                         : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = vernon::rhi::vulkan::imageAspectMask(format);
    viewInfo.subresourceRange.levelCount = descriptor->mip_levels;
    viewInfo.subresourceRange.layerCount = imageInfo.arrayLayers;
    if (!device->state.createImage(slot.image, imageInfo, viewInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                   device->error)) {
        releaseVulkanSlot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot.ownedDescriptor = *descriptor;
    try {
        slot.image.subresourceLayouts.assign(static_cast<size_t>(descriptor->mip_levels) * descriptor->array_layers,
                                             slot.image.layout);
    } catch (const std::bad_alloc &) {
        device->state.destroyImage(slot.image);
        releaseVulkanSlot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus setImageSampler(VernonRhiDevice handle, VernonRhiImage image,
                                const VernonRhiSamplerDescriptor *descriptor) {
    return VERNON_RHI_STATUS_UNSUPPORTED;
}

VernonRhiStatus uploadImage(VernonRhiDevice handle, VernonRhiImage image, const VernonRhiImageUploadDescriptor *uploads,
                            size_t uploadCount) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!uploads || uploadCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
    if (!slot || !slot->image.owned)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiImageDescriptor &descriptor = slot->ownedDescriptor;
    if ((descriptor.dimension != VERNON_RHI_IMAGE_2D && descriptor.dimension != VERNON_RHI_IMAGE_3D &&
         descriptor.dimension != VERNON_RHI_IMAGE_CUBE) ||
        (descriptor.dimension != VERNON_RHI_IMAGE_3D && descriptor.depth != 1))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    size_t totalSize = 0;
    std::vector<size_t> uploadSizes(uploadCount);
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const bool validLayer = descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                                    ? upload.array_layer < descriptor.array_layers
                                    : upload.array_layer == 0;
        const bool validMip = upload.mip_level < descriptor.mip_levels;
        const uint32_t mipWidth = validMip ? vernon::rhi::imageMipExtent(descriptor.width, upload.mip_level) : 0;
        const uint32_t mipHeight = validMip ? vernon::rhi::imageMipExtent(descriptor.height, upload.mip_level) : 0;
        const uint32_t mipDepth = !validMip ? 0
                                  : descriptor.dimension == VERNON_RHI_IMAGE_3D
                                      ? vernon::rhi::imageMipExtent(descriptor.depth, upload.mip_level)
                                      : descriptor.depth;
        const auto uploadSize =
            vernon::rhi::imageRegionByteSize(descriptor.format, upload.width, upload.height, upload.depth);
        if (upload.struct_size < sizeof(upload) || !validMip || !validLayer || !upload.width || !upload.height ||
            !upload.depth || upload.offset_x >= mipWidth || upload.width > mipWidth - upload.offset_x ||
            upload.offset_y >= mipHeight || upload.height > mipHeight - upload.offset_y ||
            upload.offset_z >= mipDepth || upload.depth > mipDepth - upload.offset_z ||
            !vernon::rhi::uploadLayoutMatches(descriptor.format, upload.source_format, upload.source_type) ||
            !upload.data || !uploadSize)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        if (*uploadSize > (std::numeric_limits<size_t>::max)() - totalSize)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        uploadSizes[index] = *uploadSize;
        totalSize += uploadSizes[index];
    }
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(true, totalSize, 16, staging, stagingOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    std::vector<VkBufferImageCopy> regions(uploadCount);
    size_t byteOffset = 0;
    for (size_t index = 0; index < uploadCount; ++index) {
        std::memcpy(mapped + byteOffset, uploads[index].data, uploadSizes[index]);
        VkBufferImageCopy &region = regions[index];
        region.bufferOffset = stagingOffset + byteOffset;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = uploads[index].mip_level;
        region.imageSubresource.baseArrayLayer = uploads[index].array_layer;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {static_cast<int32_t>(uploads[index].offset_x),
                              static_cast<int32_t>(uploads[index].offset_y),
                              static_cast<int32_t>(uploads[index].offset_z)};
        region.imageExtent = {uploads[index].width, uploads[index].height, uploads[index].depth};
        byteOffset += uploadSizes[index];
    }
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!device->state.beginCommands(command, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vernon::rhi::vulkan::driver().cmdCopyBufferToImage(command, staging, slot->image.image,
                                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                       static_cast<uint32_t>(regions.size()), regions.data());
    transitionVulkanImage(command, *slot, defaultVulkanImageLayout(slot->ownedDescriptor));
    return device->state.submitCommands(command, device->error) ? VERNON_RHI_STATUS_OK
                                                                : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                              const VernonRhiImageDownloadDescriptor *download, void *destination, size_t size) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
    if (!slot || !slot->image.owned || !download || download->struct_size < sizeof(*download) || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const auto expected = imageDownloadByteSize(slot->ownedDescriptor, *download);
    if (!expected || size != *expected)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(false, size, 16, staging, stagingOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    const VkImageLayout restore = slot->image.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? defaultVulkanImageLayout(slot->ownedDescriptor)
                                      : slot->image.layout;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!device->state.beginCommands(command, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const bool depthStencil = slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    const size_t pixels = static_cast<size_t>(download->width) * download->height * download->depth;
    std::array<VkBufferImageCopy, 2> regions{};
    regions[0].bufferOffset = stagingOffset;
    regions[0].imageSubresource.aspectMask = vernon::rhi::vulkan::imagePrimaryCopyAspectMask(slot->image.format);
    regions[0].imageSubresource.mipLevel = download->mip_level;
    regions[0].imageSubresource.baseArrayLayer = download->array_layer;
    regions[0].imageSubresource.layerCount = 1;
    regions[0].imageOffset = {static_cast<int32_t>(download->offset_x), static_cast<int32_t>(download->offset_y),
                              static_cast<int32_t>(download->offset_z)};
    regions[0].imageExtent = {download->width, download->height, download->depth};
    uint32_t regionCount = 1;
    if (depthStencil) {
        regions[1] = regions[0];
        regions[1].bufferOffset = stagingOffset + pixels * sizeof(float);
        regions[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        regionCount = 2;
    }
    vernon::rhi::vulkan::driver().cmdCopyImageToBuffer(command, slot->image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                       staging, regionCount, regions.data());
    transitionVulkanImage(command, *slot, restore);
    if (!device->state.submitCommands(command, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    if (!depthStencil) {
        std::memcpy(destination, mapped, size);
    } else {
        auto *output = static_cast<uint8_t *>(destination);
        const uint8_t *stencil = mapped + pixels * sizeof(float);
        for (size_t index = 0; index < pixels; ++index) {
            float depth{};
            std::memcpy(&depth, mapped + index * sizeof(depth), sizeof(depth));
            vernon::rhi::storePackedDepthStencil(output + index * vernon::rhi::packedDepthStencilPixelSize, depth,
                                                 stencil[index]);
        }
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
    if (!slot || !slot->image.owned || slot->ownedDescriptor.mip_levels < 2 ||
        slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT ||
        slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT ||
        !(slot->ownedDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(slot->ownedDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VkFormatProperties properties{};
    vernon::rhi::vulkan::driver().getPhysicalDeviceFormatProperties(device->state.physicalDevice, slot->image.format,
                                                                    &properties);
    constexpr VkFormatFeatureFlags blitFeatures = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if ((properties.optimalTilingFeatures & blitFeatures) != blitFeatures)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    const VkImageLayout restore = slot->image.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? defaultVulkanImageLayout(slot->ownedDescriptor)
                                      : slot->image.layout;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!device->state.beginCommands(command, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    auto barrier = [&](uint32_t baseMip, uint32_t mipCount, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkImageMemoryBarrier value{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        value.oldLayout = oldLayout;
        value.newLayout = newLayout;
        value.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        value.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        value.image = slot->image.image;
        value.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        value.subresourceRange.baseMipLevel = baseMip;
        value.subresourceRange.levelCount = mipCount;
        value.subresourceRange.layerCount = slot->ownedDescriptor.array_layers;
        VkPipelineStageFlags sourceStage{};
        VkPipelineStageFlags destinationStage{};
        vulkanImageDependency(oldLayout, sourceStage, value.srcAccessMask);
        vulkanImageDependency(newLayout, destinationStage, value.dstAccessMask);
        vernon::rhi::vulkan::driver().cmdPipelineBarrier(command, sourceStage, destinationStage, 0, 0, nullptr, 0,
                                                         nullptr, 1, &value);
    };
    const VkFilter filter = properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
                                ? VK_FILTER_LINEAR
                                : VK_FILTER_NEAREST;
    for (uint32_t level = 1; level < slot->ownedDescriptor.mip_levels; ++level) {
        barrier(level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = level - 1;
        blit.srcSubresource.layerCount = slot->ownedDescriptor.array_layers;
        blit.srcOffsets[1] = {
            static_cast<int32_t>(vernon::rhi::imageMipExtent(slot->ownedDescriptor.width, level - 1)),
            static_cast<int32_t>(vernon::rhi::imageMipExtent(slot->ownedDescriptor.height, level - 1)),
            static_cast<int32_t>(slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D
                                     ? vernon::rhi::imageMipExtent(slot->ownedDescriptor.depth, level - 1)
                                     : 1)};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = level;
        blit.dstSubresource.layerCount = slot->ownedDescriptor.array_layers;
        blit.dstOffsets[1] = {static_cast<int32_t>(vernon::rhi::imageMipExtent(slot->ownedDescriptor.width, level)),
                              static_cast<int32_t>(vernon::rhi::imageMipExtent(slot->ownedDescriptor.height, level)),
                              static_cast<int32_t>(slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D
                                                       ? vernon::rhi::imageMipExtent(slot->ownedDescriptor.depth, level)
                                                       : 1)};
        vernon::rhi::vulkan::driver().cmdBlitImage(command, slot->image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                   slot->image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                                                   filter);
    }
    barrier(0, slot->ownedDescriptor.mip_levels - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, restore);
    barrier(slot->ownedDescriptor.mip_levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, restore);
    std::fill(slot->image.subresourceLayouts.begin(), slot->image.subresourceLayouts.end(), restore);
    slot->image.layout = restore;
    return device->state.submitCommands(command, device->error) ? VERNON_RHI_STATUS_OK
                                                                : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus bindImage(VernonRhiDevice handle, VernonRhiImage image, uint32_t textureUnit) {
    return VERNON_RHI_STATUS_UNSUPPORTED;
}

VernonRhiStatus destroyImage(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyImage(slot->image);
        releaseVulkanSlot(*slot);
    }
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

uint32_t isImageValid(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupVulkanSlot(device->images, image) != nullptr;
}

VernonRhiStatus createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor,
                              VernonRhiSampler *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    SamplerFilter filter;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        !decodeSamplerFilter(*descriptor, filter))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (filter.maxAnisotropy > 1.0f)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanSamplerSlot &slot = allocateVulkanSlot(device->samplers, *output);
    auto address = [](uint32_t mode) {
        constexpr VkSamplerAddressMode values[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                   VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                   VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
        return values[mode];
    };
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.minFilter = filter.minLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    info.magFilter = filter.magLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    info.mipmapMode = filter.mipLinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = address(descriptor->address_u);
    info.addressModeV = address(descriptor->address_v);
    info.addressModeW = address(descriptor->address_w);
    info.maxLod = VK_LOD_CLAMP_NONE;
    if (!device->state.createSampler(slot.sampler, info, device->error)) {
        releaseVulkanSlot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanSamplerSlot *slot = lookupVulkanSlot(device->samplers, sampler);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroySampler(slot->sampler);
        releaseVulkanSlot(*slot);
    }
    return VERNON_RHI_STATUS_OK;
}

uint32_t isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupVulkanSlot(device->samplers, sampler) != nullptr;
}

VernonRhiStatus getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image, uint64_t *output) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = vulkanHandleBits(slot->image.image);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyBuffer(slot->buffer);
        releaseVulkanSlot(*slot);
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer, void **output) {
    return VERNON_RHI_STATUS_UNSUPPORTED;
}

bool ownsDevice(VernonRhiDevice handle) { return static_cast<bool>(lookupVulkanDevice(handle)); }

void *deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupVulkanDevice(handle);
    return device ? &device->state : nullptr;
}

bool beginCommands(VernonRhiDevice handle, uint64_t &native, VernonRhiBackend &backend) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    VkCommandBuffer command{};
    if (!device->state.beginCommands(command, device->error))
        return false;
    backend = VERNON_RHI_BACKEND_VULKAN;
    native = vulkanHandleBits(command);
    return true;
}

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool computeWrites, bool &completed,
                    bool &externalCompletion) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    const bool submitted = command && device->state.submitCommands(command, device->error);
    completed = submitted && !device->state.nativeObjectsBorrowed;
    externalCompletion = submitted && device->state.nativeObjectsBorrowed;
    return submitted;
}

void completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) {}

void abandonCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return;

    std::lock_guard<std::mutex> guard(device->mutex);
    if (!device->state.nativeObjectsBorrowed) {
        const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
        if (command)
            vernon::rhi::vulkan::driver().endCommandBuffer(command);
    }
    return;
}

bool recordBarriers(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native, const VernonRhiBarrier *barriers,
                    size_t barrierCount) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!command)
        return false;
    try {
        std::vector<VkBufferMemoryBarrier> bufferBarriers;
        std::vector<VkImageMemoryBarrier> imageBarriers;
        bufferBarriers.reserve(barrierCount);
        imageBarriers.reserve(barrierCount);
        VkPipelineStageFlags sourceStages = 0;
        VkPipelineStageFlags destinationStages = 0;
        for (size_t index = 0; index < barrierCount; ++index) {
            if (barriers[index].is_image) {
                auto *slot = lookupResourceRecord(device->images, resourceKey(barriers[index].image));
                if (!slot)
                    return false;
                const VernonRhiImageDescriptor &descriptor =
                    slot->ownedDescriptor.struct_size ? slot->ownedDescriptor : slot->descriptor.descriptor;
                const auto &range = barriers[index].image_subresources;
                const uint32_t imageLayers = descriptor.array_layers;
                const uint64_t mipEnd = range.mip_level_count == UINT32_MAX
                                            ? descriptor.mip_levels
                                            : uint64_t{range.base_mip_level} + range.mip_level_count;
                const uint64_t layerEnd = range.array_layer_count == UINT32_MAX
                                              ? imageLayers
                                              : uint64_t{range.base_array_layer} + range.array_layer_count;
                const uint32_t availableAspects = vernon::rhi::imageFormatAspects(descriptor.format);
                if (range.base_mip_level >= descriptor.mip_levels || mipEnd > descriptor.mip_levels ||
                    range.base_array_layer >= imageLayers || layerEnd > imageLayers ||
                    (range.aspects & ~availableAspects) != 0)
                    return false;
                const VkImageLayout target = vulkanImageLayout(barriers[index].new_state);
                const size_t planeStride = static_cast<size_t>(descriptor.mip_levels) * imageLayers;
                const size_t subresourceCount = planeStride;
                if (slot->image.subresourceLayouts.size() != subresourceCount)
                    return false;
                auto journal = slot->image.layoutJournals.find(encoderKey);
                bool inserted = false;
                if (journal == slot->image.layoutJournals.end()) {
                    const auto result = slot->image.layoutJournals.emplace(
                        encoderKey,
                        vernon::rhi::vulkan::Image::LayoutJournal{slot->image.layout, slot->image.subresourceLayouts});
                    journal = result.first;
                    inserted = result.second;
                }
                if (inserted &&
                    (!deferCommandCleanup(handle, encoderKey, &slot->image, encoderKey,
                                          clearVulkanImageLayoutJournal) ||
                     !deferCommandRollback(handle, encoderKey, &slot->image, encoderKey, restoreVulkanImageLayouts))) {
                    slot->image.layoutJournals.erase(journal);
                    return false;
                }
                const VkImageAspectFlags nativeAspect = vernon::rhi::vulkan::imageAspectMask(slot->image.format);
                for (uint32_t layer = range.base_array_layer; layer < layerEnd; ++layer)
                    for (uint32_t mip = range.base_mip_level; mip < mipEnd; ++mip) {
                        const size_t subresource = mip + layer * descriptor.mip_levels;
                        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                        barrier.oldLayout = slot->image.subresourceLayouts[subresource];
                        barrier.newLayout = target;
                        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.image = slot->image.image;
                        barrier.subresourceRange = {nativeAspect, mip, 1, layer, 1};
                        VkPipelineStageFlags source{};
                        VkPipelineStageFlags destination{};
                        vulkanBarrierDependency(static_cast<VernonRhiResourceState>(barriers[index].old_state),
                                                barriers[index].source_stage_mask, barriers[index].source_access,
                                                source, barrier.srcAccessMask);
                        vulkanBarrierDependency(static_cast<VernonRhiResourceState>(barriers[index].new_state),
                                                barriers[index].destination_stage_mask,
                                                barriers[index].destination_access, destination, barrier.dstAccessMask);
                        sourceStages |= source;
                        destinationStages |= destination;
                        imageBarriers.push_back(barrier);
                        slot->image.subresourceLayouts[subresource] = target;
                    }
                if (std::all_of(slot->image.subresourceLayouts.begin(), slot->image.subresourceLayouts.end(),
                                [target](VkImageLayout layout) { return layout == target; }))
                    slot->image.layout = target;
            } else {
                auto *slot = lookupResourceRecord(device->buffers, resourceKey(barriers[index].buffer));
                if (!slot)
                    return false;
                VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                VkPipelineStageFlags source{};
                VkPipelineStageFlags destination{};
                vulkanBarrierDependency(static_cast<VernonRhiResourceState>(barriers[index].old_state),
                                        barriers[index].source_stage_mask, barriers[index].source_access, source,
                                        barrier.srcAccessMask);
                vulkanBarrierDependency(static_cast<VernonRhiResourceState>(barriers[index].new_state),
                                        barriers[index].destination_stage_mask, barriers[index].destination_access,
                                        destination, barrier.dstAccessMask);
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = slot->buffer.buffer;
                barrier.offset = 0;
                barrier.size = VK_WHOLE_SIZE;
                sourceStages |= source;
                destinationStages |= destination;
                bufferBarriers.push_back(barrier);
            }
        }
        if (!bufferBarriers.empty() || !imageBarriers.empty())
            vernon::rhi::vulkan::driver().cmdPipelineBarrier(
                command, sourceStages ? sourceStages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                destinationStages ? destinationStages : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr,
                static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
                static_cast<uint32_t>(imageBarriers.size()), imageBarriers.data());
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

bool endRendering(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                  uint32_t colorDiscardMask, uint32_t depthStencilDiscard, const uint64_t *colorResources,
                  size_t colorCount, uint64_t depthResource, uint64_t) {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || backend != VERNON_RHI_BACKEND_VULKAN)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (backendKind == CommandRenderingDynamic)
        vernon::rhi::vulkan::driver().cmdEndRendering(command);
    else if (backendKind == CommandRenderingRenderPass)
        vernon::rhi::vulkan::driver().cmdEndRenderPass(command);
    else
        return false;
    return true;
}

bool clearColor(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind, int32_t x,
                int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target, uint32_t location,
                const float color[4]) {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || backend != VERNON_RHI_BACKEND_VULKAN ||
        (backendKind != CommandRenderingDynamic && backendKind != CommandRenderingRenderPass))
        return false;
    VkClearAttachment clear{};
    clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear.colorAttachment = location;
    std::copy(color, color + 4, clear.clearValue.color.float32);
    const VkClearRect rectangle{{{x, y}, {width, height}}, 0, layers};
    std::lock_guard<std::mutex> guard(device->mutex);
    vernon::rhi::vulkan::driver().cmdClearAttachments(command, 1, &clear, 1, &rectangle);
    return true;
}

bool clearDepthStencil(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                       int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target,
                       float depth, uint32_t stencil, uint32_t aspects) {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || backend != VERNON_RHI_BACKEND_VULKAN ||
        (backendKind != CommandRenderingDynamic && backendKind != CommandRenderingRenderPass))
        return false;
    VkClearAttachment clear{};
    clear.aspectMask = ((aspects & VERNON_RHI_ATTACHMENT_DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                       ((aspects & VERNON_RHI_ATTACHMENT_STENCIL) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    clear.clearValue.depthStencil = {depth, stencil};
    const VkClearRect rectangle{{{x, y}, {width, height}}, 0, layers};
    std::lock_guard<std::mutex> guard(device->mutex);
    vernon::rhi::vulkan::driver().cmdClearAttachments(command, 1, &clear, 1, &rectangle);
    return true;
}

uint64_t bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer))
        return resourceKey(buffer);
    return 0;
}

uint64_t imageResource(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (VulkanImageSlot *slot = lookupVulkanSlot(device->images, image))
        return resourceKey(image);
    return 0;
}

uint64_t imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupVulkanImageView(device->imageViews, view) ? resourceKey(view) : 0;
}

uint64_t samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (VulkanSamplerSlot *slot = lookupVulkanSlot(device->samplers, sampler))
        return resourceKey(sampler);
    return 0;
}

bool retainResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        VulkanBufferSlot *slot = lookupResourceRecord(device->buffers, key);
        return slot && slot->retain();
    }
    if (kind == ResourceKind::Image) {
        VulkanImageSlot *slot = lookupResourceRecord(device->images, key);
        return slot && slot->retain();
    }
    if (kind == ResourceKind::ImageView) {
        VulkanImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
        return slot && slot->retain();
    }
    VulkanSamplerSlot *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->retain();
}

uint64_t resolveResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        auto *slot = lookupResourceRecord(device->buffers, key);
        return slot && slot->occupied ? reinterpret_cast<uintptr_t>(&slot->buffer) : 0;
    }
    if (kind == ResourceKind::Image) {
        auto *slot = lookupResourceRecord(device->images, key);
        return slot && slot->occupied ? reinterpret_cast<uintptr_t>(&slot->image) : 0;
    }
    if (kind == ResourceKind::ImageView) {
        auto *slot = lookupResourceRecord(device->imageViews, key);
        return slot ? reinterpret_cast<uintptr_t>(&slot->bindingImage) : 0;
    }
    auto *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->occupied ? reinterpret_cast<uintptr_t>(&slot->sampler) : 0;
}

bool describeImageResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageDescriptor *descriptor) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !descriptor)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const VulkanImageSlot *slot = lookupResourceRecord(device->images, key);
    if (!slot || !slot->occupied)
        return false;
    *descriptor = slot->image.owned ? slot->ownedDescriptor : slot->descriptor.descriptor;
    return true;
}

bool describeImageViewResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageViewDescriptor *view,
                               VernonRhiImageDescriptor *image, uint64_t *parentKey) {
    auto device = lookupVulkanDevice(handle);
    if (!device || !view || !image || !parentKey)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const VulkanImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
    if (!slot)
        return false;
    const VulkanImageSlot *parent = lookupResourceRecord(device->images, slot->imageResource);
    if (!parent)
        return false;
    *view = slot->descriptor;
    *image = parent->image.owned ? parent->ownedDescriptor : parent->descriptor.descriptor;
    *parentKey = slot->imageResource;
    return true;
}

void releaseResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return;

    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        auto *slot = lookupResourceRecord(device->buffers, key);
        if (!slot || !slot->release())
            return;
        device->state.destroyBuffer(slot->buffer);
        releaseVulkanSlot(*slot);
        return;
    }
    if (kind == ResourceKind::Image) {
        auto *slot = lookupResourceRecord(device->images, key);
        if (!slot || !slot->release())
            return;
        device->state.destroyImage(slot->image);
        releaseVulkanSlot(*slot);
        return;
    }
    if (kind == ResourceKind::ImageView) {
        auto *slot = lookupResourceRecord(device->imageViews, key);
        if (!slot || !slot->release())
            return;
        const uint64_t parentKey = slot->imageResource;
        if (slot->ownedNative)
            vernon::rhi::vulkan::driver().destroyImageView(device->state.device, slot->bindingImage.view, nullptr);
        releaseVulkanImageView(*slot);
        auto *parent = lookupResourceRecord(device->images, parentKey);
        if (parent && parent->release()) {
            device->state.destroyImage(parent->image);
            releaseVulkanSlot(*parent);
        }
        return;
    }
    auto *slot = lookupResourceRecord(device->samplers, key);
    if (!slot || !slot->release())
        return;
    device->state.destroySampler(slot->sampler);
    releaseVulkanSlot(*slot);
    return;
}

} // namespace vernon::rhi::vulkan_api

const vernon::rhi::BackendDispatch &vernon::rhi::vulkanBackendDispatch() {
    using namespace vulkan_api;
    static const BackendDispatch dispatch{
        VERNON_RHI_BACKEND_VULKAN,
        ownsDevice,
        createOwnedDevice,
        vulkan_api::destroyDevice,
        lastError,
        synchronize,
        deviceStateForBackend,
        createBuffer,
        uploadBuffer,
        uploadBufferRanges,
        downloadBuffer,
        destroyBuffer,
        isBufferValid,
        getBufferNativeHandle,
        createImage,
        setImageSampler,
        uploadImage,
        downloadImage,
        generateImageMipmaps,
        bindImage,
        destroyImage,
        isImageValid,
        getImageNativeHandle,
        createSampler,
        destroySampler,
        isSamplerValid,
        bufferResource,
        imageResource,
        samplerResource,
        retainResource,
        resolveResource,
        describeImageResource,
        releaseResource,
        beginCommands,
        submitCommands,
        completeBorrowedCommands,
        abandonCommands,
        recordBarriers,
        endRendering,
        clearColor,
        clearDepthStencil,
        nullptr,
        createImageView,
        destroyImageView,
        getImageViewNativeHandle,
        imageViewResource,
        describeImageViewResource,
    };
    return dispatch;
}

vernon::rhi::VulkanCacheStats vernon::rhi::getVulkanCacheStats(VernonRhiDevice handle) {
    VulkanCacheStats result;
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        result.defaultImplicitSamplerCreations = device->state.defaultImplicitSamplerCreations;
        result.commandBufferAllocations = device->state.commandBufferAllocations;
        result.descriptorPoolCreations = device->state.descriptorPoolCreations;
        result.stagingBufferAllocations = device->state.stagingBufferAllocations;
        result.dynamicRendering = device->state.dynamicRendering;
    }
    return result;
}
