#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "public_c_boundary.h"
#include "rhi_test_hooks.h"
#include "sampler_filter.h"
#include "vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vernon::rhi {
bool deviceHasActiveCommandEncoder(VernonRhiDevice device);
}

namespace {
using VulkanBufferLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::BufferResourceTag>;
using VulkanImageLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::ImageResourceTag>;
using VulkanImageViewLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::ImageViewResourceTag>;
using VulkanSamplerLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::SamplerResourceTag>;

struct VulkanBufferSlot {
    explicit VulkanBufferSlot(VulkanBufferLifecycle value) noexcept : lifecycle(std::move(value)) {}

    VulkanBufferLifecycle lifecycle;
    vernon::rhi::vulkan::Buffer buffer;
    VernonRhiVulkanBorrowedBufferDescriptor descriptor{};
    VernonRhiBufferDescriptor ownedDescriptor{};
};

struct VulkanImageSlot {
    explicit VulkanImageSlot(VulkanImageLifecycle value) noexcept : lifecycle(std::move(value)) {}

    VulkanImageLifecycle lifecycle;
    vernon::rhi::vulkan::Image image;
    VernonRhiVulkanBorrowedImageDescriptor descriptor{};
    VernonRhiImageDescriptor ownedDescriptor{};
};

struct VulkanImageViewSlot {
    explicit VulkanImageViewSlot(VulkanImageViewLifecycle value) noexcept : lifecycle(std::move(value)) {}

    VulkanImageViewLifecycle lifecycle;
    vernon::rhi::vulkan::Image bindingImage;
    VernonRhiImageViewDescriptor descriptor{};
    std::optional<vernon::rhi::RetainedResourceLease<vernon::rhi::ImageResourceTag>> parent;
    bool ownedNative{};
};

struct VulkanSamplerSlot {
    explicit VulkanSamplerSlot(VulkanSamplerLifecycle value) noexcept : lifecycle(std::move(value)) {}

    VulkanSamplerLifecycle lifecycle;
    vernon::rhi::vulkan::Sampler sampler;
};

struct VulkanInteropDevice {
    vernon::rhi::vulkan::DeviceState state;
    vernon::rhi::StableResourceSlotContainer<VulkanBufferSlot> buffers;
    vernon::rhi::StableResourceSlotContainer<VulkanImageSlot> images;
    vernon::rhi::StableResourceSlotContainer<VulkanSamplerSlot> samplers;
    vernon::rhi::StableResourceSlotContainer<VulkanImageViewSlot> imageViews;
    vernon::rhi::CommandDeviceStateRef commandState;
    uint32_t queueCapabilities{};
    std::string error;
    std::mutex creationMutex;
    std::mutex mutex;
};

constexpr uint32_t vulkanDeviceBit = uint32_t{1} << 30;
constexpr std::size_t vulkanDeviceCapacity = 256;
using VulkanDevices = vernon::rhi::DeviceRegistry<VulkanInteropDevice, vulkanDeviceCapacity>;
using VulkanDeviceAnchor = vernon::rhi::DeviceRegistryAnchor<VulkanInteropDevice>;

VulkanDevices &vulkanDevices() noexcept {
    // Ensure the loader outlives all registry-owned device states.
    (void)vernon::rhi::vulkan::driver();
    static VulkanDevices devices;
    return devices;
}

VernonRhiDevice invalidDevice() noexcept { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) noexcept {
    if (handle.index == UINT32_MAX || handle.generation == 0)
        return 0;
    return (uint64_t{handle.generation} << 32) | (uint64_t{handle.index} + 1);
}

template <typename Tag>
vernon::Result<vernon::rhi::ResourceHandle<Tag>, vernon::RhiError> decodeResourceKey(uint64_t key,
                                                                                     const char *operation) noexcept {
    const uint64_t encodedIndex = key & UINT32_MAX;
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return vernon::Result<vernon::rhi::ResourceHandle<Tag>, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {operation, key, 0}})};
    return vernon::Result<vernon::rhi::ResourceHandle<Tag>, vernon::RhiError>{
        vernon::ok(vernon::rhi::ResourceHandle<Tag>{static_cast<uint32_t>(encodedIndex - 1), generation})};
}

vernon::RhiError invalidResource(const char *operation, uint64_t value = 0, uint32_t detail = 0) noexcept {
    return {vernon::RhiErrorCode::InvalidArgument, {operation, value, detail}};
}

VernonRhiStatus status(vernon::RhiError error) noexcept {
    switch (error.code) {
    case vernon::RhiErrorCode::InvalidArgument:
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    case vernon::RhiErrorCode::Unsupported:
        return VERNON_RHI_STATUS_UNSUPPORTED;
    case vernon::RhiErrorCode::ResourceExhausted:
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    default:
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
}

class VulkanDeviceAccess {
public:
    explicit VulkanDeviceAccess(vernon::Result<VulkanDeviceAnchor, vernon::RhiError> result) noexcept {
        if (result.isOk())
            anchor_.emplace(std::move(result).value());
    }

    [[nodiscard]] explicit operator bool() const noexcept { return anchor_.has_value(); }
    [[nodiscard]] bool isOk() const noexcept { return anchor_.has_value(); }
    [[nodiscard]] bool isErr() const noexcept { return !anchor_.has_value(); }
    [[nodiscard]] VulkanInteropDevice *operator->() noexcept { return &anchor_->device(); }
    [[nodiscard]] const VulkanInteropDevice *operator->() const noexcept { return &anchor_->device(); }
    [[nodiscard]] VulkanDeviceAnchor &value() noexcept { return *anchor_; }

private:
    std::optional<VulkanDeviceAnchor> anchor_;
};

VulkanDeviceAccess lookupVulkanDevice(VernonRhiDevice handle) noexcept {
    if ((handle.index & vulkanDeviceBit) == 0)
        return VulkanDeviceAccess{vernon::Result<VulkanDeviceAnchor, vernon::RhiError>{
            vernon::err(invalidResource("lookup_vulkan_device", handle.generation))}};
    return VulkanDeviceAccess{vulkanDevices().lookup({handle.index & ~vulkanDeviceBit, handle.generation})};
}

template <typename Tag, typename Slots>
vernon::Result<typename Slots::value_type *, vernon::RhiError> reserveVulkanSlot(VulkanInteropDevice &device,
                                                                                 Slots &slots) noexcept {
    std::size_t slotCount = 0;
    {
        std::lock_guard<std::mutex> guard(device.mutex);
        slotCount = slots.size();
    }
    // The caller serializes creators with creationMutex, so deque structure is
    // stable while lifecycle state is inspected without the device mutex.
    for (std::size_t index = 0; index < slotCount; ++index)
        if (!slots[index].lifecycle.snapshot().occupied)
            return vernon::Result<typename Slots::value_type *, vernon::RhiError>{vernon::ok(&slots[index])};
    if (slotCount >= UINT32_MAX)
        return vernon::Result<typename Slots::value_type *, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::ResourceExhausted, {"allocate_vulkan_resource_slot", slotCount, 0}})};
    const uint32_t index = static_cast<uint32_t>(slotCount);
    auto lifecycle = vernon::rhi::ResourceLifecycleSlot<Tag>::create(index);
    if (lifecycle.isErr())
        return vernon::Result<typename Slots::value_type *, vernon::RhiError>{
            vernon::err(std::move(lifecycle).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    return slots.emplace("allocate_vulkan_resource_slot", std::move(lifecycle).value());
}

template <typename Tag, typename Slots>
vernon::Result<typename Slots::value_type *, vernon::RhiError> findVulkanSlot(VulkanInteropDevice &device, Slots &slots,
                                                                              vernon::rhi::ResourceHandle<Tag> handle,
                                                                              const char *operation) noexcept {
    std::lock_guard<std::mutex> guard(device.mutex);
    if (handle.index >= slots.size())
        return vernon::Result<typename Slots::value_type *, vernon::RhiError>{
            vernon::err(invalidResource(operation, handle.generation, handle.index))};
    return vernon::Result<typename Slots::value_type *, vernon::RhiError>{vernon::ok(&slots[handle.index])};
}

template <typename Slot> struct PinnedVulkanSlot {
    PinnedVulkanSlot(Slot *value, vernon::OperationPin operation) noexcept : slot(value), pin(std::move(operation)) {}

    Slot *slot;
    vernon::OperationPin pin;
};

template <typename Tag, typename Slots>
vernon::Result<PinnedVulkanSlot<typename Slots::value_type>, vernon::RhiError>
findAndPinVulkanSlot(VulkanInteropDevice &device, Slots &slots, vernon::rhi::ResourceHandle<Tag> handle,
                     const char *operation) noexcept {
    auto found = findVulkanSlot(device, slots, handle, operation);
    if (found.isErr())
        return vernon::Result<PinnedVulkanSlot<typename Slots::value_type>, vernon::RhiError>{
            vernon::err(std::move(found).error())};
    auto pinned = found.value()->lifecycle.pin(handle);
    if (pinned.isErr())
        return vernon::Result<PinnedVulkanSlot<typename Slots::value_type>, vernon::RhiError>{
            vernon::err(std::move(pinned).error())};
    return vernon::Result<PinnedVulkanSlot<typename Slots::value_type>, vernon::RhiError>{
        vernon::ok(PinnedVulkanSlot<typename Slots::value_type>(found.value(), std::move(pinned).value()))};
}

template <typename Tag, typename Handle> vernon::rhi::ResourceHandle<Tag> typedHandle(Handle handle) noexcept {
    return {handle.index, handle.generation};
}

template <typename Handle, typename Tag> Handle publicHandle(vernon::rhi::ResourceHandle<Tag> handle) noexcept {
    return {handle.index, handle.generation};
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

void makeVulkanBufferReadableByTransfer(VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset,
                                        VkDeviceSize size) noexcept {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;
    vernon::rhi::vulkan::driver().cmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0,
                                                     nullptr);
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

vernon::Result<void, vernon::RhiError>
teardownBuffer(void *context, vernon::rhi::ResourceHandle<vernon::rhi::BufferResourceTag> handle) noexcept {
    auto &device = *static_cast<VulkanInteropDevice *>(context);
    std::lock_guard<std::mutex> guard(device.mutex);
    if (handle.index >= device.buffers.size())
        return vernon::Result<void, vernon::RhiError>{vernon::err(invalidResource("teardown_vulkan_buffer"))};
    device.state.destroyBuffer(device.buffers[handle.index].buffer);
    device.buffers[handle.index].descriptor = {};
    device.buffers[handle.index].ownedDescriptor = {};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError>
teardownImage(void *context, vernon::rhi::ResourceHandle<vernon::rhi::ImageResourceTag> handle) noexcept {
    auto &device = *static_cast<VulkanInteropDevice *>(context);
    std::lock_guard<std::mutex> guard(device.mutex);
    if (handle.index >= device.images.size())
        return vernon::Result<void, vernon::RhiError>{vernon::err(invalidResource("teardown_vulkan_image"))};
    device.state.destroyImage(device.images[handle.index].image);
    device.images[handle.index].descriptor = {};
    device.images[handle.index].ownedDescriptor = {};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError>
teardownImageView(void *context, vernon::rhi::ResourceHandle<vernon::rhi::ImageViewResourceTag> handle) noexcept {
    auto &device = *static_cast<VulkanInteropDevice *>(context);
    if (handle.index >= device.imageViews.size())
        return vernon::Result<void, vernon::RhiError>{vernon::err(invalidResource("teardown_vulkan_image_view"))};
    VulkanImageViewSlot &slot = device.imageViews[handle.index];
    if (!slot.parent)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure,
                                         {"release_vulkan_image_view_parent", handle.generation, handle.index}})};
    auto prepared = slot.parent->prepareRelease();
    if (prepared.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(prepared).error())};
    {
        std::lock_guard<std::mutex> guard(device.mutex);
        if (slot.ownedNative && slot.bindingImage.view)
            vernon::rhi::vulkan::driver().destroyImageView(device.state.device, slot.bindingImage.view, nullptr);
        slot.bindingImage = {};
        slot.descriptor = {};
        slot.ownedNative = false;
    }
    if (prepared.value().commit().isErr())
        vernon::resultContractViolation();
    slot.parent.reset();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError>
teardownSampler(void *context, vernon::rhi::ResourceHandle<vernon::rhi::SamplerResourceTag> handle) noexcept {
    auto &device = *static_cast<VulkanInteropDevice *>(context);
    std::lock_guard<std::mutex> guard(device.mutex);
    if (handle.index >= device.samplers.size())
        return vernon::Result<void, vernon::RhiError>{vernon::err(invalidResource("teardown_vulkan_sampler"))};
    device.state.destroySampler(device.samplers[handle.index].sampler);
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownDevice(VulkanInteropDevice &device) noexcept {
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.shutdown();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}
} // namespace

namespace vernon::rhi::vulkan_api {

static VernonRhiDevice
vernonRhiCreateBorrowedVulkanDeviceBusiness(const VernonRhiVulkanBorrowedDeviceDescriptor *descriptor) {
    constexpr uint32_t allQueueCapabilities =
        VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->instance ||
        !descriptor->physical_device || !descriptor->device || !descriptor->queue || !descriptor->command_buffer ||
        (descriptor->queue_capabilities & ~allQueueCapabilities) != 0)
        return invalidDevice();
    auto reserved = vulkanDevices().reserve();
    if (reserved.isErr())
        return invalidDevice();
    auto reservation = std::move(reserved).value();
    VulkanInteropDevice &device = reservation.device();
    auto owner = reservation.retainOwner();
    if (owner.isErr()) {
        vernon::rhi::setDeviceCreationError("cannot retain Vulkan RHI device owner");
        return invalidDevice();
    }
    auto commandState = vernon::rhi::createCommandDeviceState(std::move(owner).value());
    if (commandState.isErr()) {
        vernon::rhi::setDeviceCreationError("cannot allocate Vulkan command state");
        return invalidDevice();
    }
    device.commandState = std::move(commandState).value();
    if (!device.state.initializeBorrowed(
            static_cast<VkInstance>(descriptor->instance), static_cast<VkPhysicalDevice>(descriptor->physical_device),
            static_cast<VkDevice>(descriptor->device), static_cast<VkQueue>(descriptor->queue),
            descriptor->queue_family_index, static_cast<VkCommandBuffer>(descriptor->command_buffer), device.error))
        return invalidDevice();
    uint32_t queueFamilyCount = 0;
    vernon::rhi::vulkan::driver().getPhysicalDeviceQueueFamilyProperties(device.state.physicalDevice, &queueFamilyCount,
                                                                         nullptr);
    if (descriptor->queue_family_index >= queueFamilyCount) {
        device.state.shutdown();
        return invalidDevice();
    }
    std::unique_ptr<VkQueueFamilyProperties[]> queueFamilies{new (std::nothrow)
                                                                 VkQueueFamilyProperties[queueFamilyCount]};
    if (!queueFamilies) {
        device.state.shutdown();
        return invalidDevice();
    }
    vernon::rhi::vulkan::driver().getPhysicalDeviceQueueFamilyProperties(device.state.physicalDevice, &queueFamilyCount,
                                                                         queueFamilies.get());
    VkQueueFlags required = 0;
    if ((descriptor->queue_capabilities & VERNON_RHI_QUEUE_TRANSFER) != 0)
        required |= VK_QUEUE_TRANSFER_BIT;
    if ((descriptor->queue_capabilities & VERNON_RHI_QUEUE_COMPUTE) != 0)
        required |= VK_QUEUE_COMPUTE_BIT;
    if ((descriptor->queue_capabilities & VERNON_RHI_QUEUE_GRAPHICS) != 0)
        required |= VK_QUEUE_GRAPHICS_BIT;
    if ((queueFamilies[descriptor->queue_family_index].queueFlags & required) != required) {
        device.state.shutdown();
        return invalidDevice();
    }
    device.queueCapabilities = descriptor->queue_capabilities;
    auto published = vulkanDevices().publish(std::move(reservation));
    if (published.isErr())
        return invalidDevice();
    return {published.value().index | vulkanDeviceBit, published.value().generation};
}

static VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedQueueBusiness(VernonRhiDevice handle, void **output) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device.value().device().state.queue;
    return VERNON_RHI_STATUS_OK;
}

static VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedCommandBufferBusiness(VernonRhiDevice handle, void **output) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device.value().device().state.borrowedCommandBuffer;
    return VERNON_RHI_STATUS_OK;
}

static VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedBufferBusiness(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedBufferDescriptor *descriptor, VernonRhiBuffer *output) {
    constexpr uint32_t allUsages = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION |
                                   VERNON_RHI_BUFFER_UNIFORM | VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX |
                                   VERNON_RHI_BUFFER_INDEX | VERNON_RHI_BUFFER_INDIRECT;
    auto device = lookupVulkanDevice(handle);
    if (device.isErr() || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->buffer ||
        !descriptor->size || (descriptor->usage & ~allUsages) != 0 || descriptor->state == VERNON_RHI_STATE_UNDEFINED ||
        !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VulkanInteropDevice &context = device.value().device();
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return status(std::move(owner).error());
    auto creation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return status(std::move(creation).error());
    std::lock_guard<std::mutex> creationGuard(context.creationMutex);
    auto reserved = reserveVulkanSlot<vernon::rhi::BufferResourceTag>(context, context.buffers);
    if (reserved.isErr())
        return status(std::move(reserved).error());
    VulkanBufferSlot &slot = *reserved.value();
    std::lock_guard<std::mutex> guard(context.mutex);
    slot.buffer.buffer = vulkanHandle<VkBuffer>(descriptor->buffer);
    slot.buffer.owned = false;
    slot.descriptor = *descriptor;
    auto published = slot.lifecycle.publish(std::move(creation).value(), &context, teardownBuffer);
    if (published.isErr()) {
        slot.buffer = {};
        slot.descriptor = {};
        return status(std::move(published).error());
    }
    *output = publicHandle<VernonRhiBuffer>(published.value());
    return VERNON_RHI_STATUS_OK;
}

static VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImageBusiness(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageDescriptor *descriptor, VernonRhiImage *output) {
    constexpr uint32_t allUsages = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION |
                                   VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_STORAGE |
                                   VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT;
    auto device = lookupVulkanDevice(handle);
    const bool validCube =
        descriptor && (descriptor->descriptor.dimension != VERNON_RHI_IMAGE_CUBE ||
                       (descriptor->descriptor.width == descriptor->descriptor.height &&
                        descriptor->descriptor.depth == 1 && descriptor->descriptor.array_layers == 6));
    if (device.isErr() || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->image ||
        descriptor->descriptor.struct_size < sizeof(descriptor->descriptor) ||
        vulkanFormat(descriptor->descriptor.format) == VK_FORMAT_UNDEFINED || !descriptor->descriptor.width ||
        !descriptor->descriptor.height || !descriptor->descriptor.depth || !descriptor->descriptor.mip_levels ||
        !descriptor->descriptor.array_layers || !descriptor->descriptor.sample_count ||
        (descriptor->descriptor.usage & ~allUsages) != 0 || !validCube || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VulkanInteropDevice &context = device.value().device();
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return status(std::move(owner).error());
    auto creation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return status(std::move(creation).error());
    std::lock_guard<std::mutex> creationGuard(context.creationMutex);
    auto reserved = reserveVulkanSlot<vernon::rhi::ImageResourceTag>(context, context.images);
    if (reserved.isErr())
        return status(std::move(reserved).error());
    VulkanImageSlot &slot = *reserved.value();
    std::lock_guard<std::mutex> guard(context.mutex);
    const auto layout = vulkanImageLayout(descriptor->state);
    const size_t subresourceCount =
        static_cast<size_t>(descriptor->descriptor.mip_levels) * descriptor->descriptor.array_layers;
    auto subresourceLayouts = decltype(slot.image.subresourceLayouts)(subresourceCount, layout);
    slot.image.image = vulkanHandle<VkImage>(descriptor->image);
    slot.image.format = vulkanFormat(descriptor->descriptor.format);
    slot.image.layout = layout;
    slot.image.subresourceLayouts = std::move(subresourceLayouts);
    slot.image.colorAttachment = (descriptor->descriptor.usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0;
    slot.image.owned = false;
    slot.descriptor = *descriptor;
    auto published = slot.lifecycle.publish(std::move(creation).value(), &context, teardownImage);
    if (published.isErr()) {
        slot.image = {};
        slot.descriptor = {};
        return status(std::move(published).error());
    }
    *output = publicHandle<VernonRhiImage>(published.value());
    return VERNON_RHI_STATUS_OK;
}

static VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImageViewBusiness(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageViewDescriptor *descriptor, VernonRhiImageView *output) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr() || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->image_view ||
        descriptor->descriptor.struct_size < sizeof(descriptor->descriptor) || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VulkanInteropDevice &context = device.value().device();
    auto found = findVulkanSlot(context, context.images,
                                typedHandle<vernon::rhi::ImageResourceTag>(descriptor->descriptor.image),
                                "import_vulkan_image_view");
    if (found.isErr())
        return status(std::move(found).error());
    VulkanImageSlot *image = found.value();
    auto imagePin = image->lifecycle.pin(typedHandle<vernon::rhi::ImageResourceTag>(descriptor->descriptor.image));
    if (imagePin.isErr())
        return status(std::move(imagePin).error());
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return status(std::move(owner).error());
    auto creation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return status(std::move(creation).error());
    auto retained = image->lifecycle.retain(typedHandle<vernon::rhi::ImageResourceTag>(descriptor->descriptor.image));
    if (retained.isErr())
        return status(std::move(retained).error());
    std::lock_guard<std::mutex> creationGuard(context.creationMutex);
    auto reserved = reserveVulkanSlot<vernon::rhi::ImageViewResourceTag>(context, context.imageViews);
    if (reserved.isErr())
        return status(std::move(reserved).error());
    VulkanImageViewSlot &slot = *reserved.value();
    std::lock_guard<std::mutex> guard(context.mutex);
    if (descriptor->descriptor.format != image->descriptor.descriptor.format ||
        !descriptor->descriptor.mip_level_count || !descriptor->descriptor.array_layer_count ||
        descriptor->descriptor.base_mip_level + descriptor->descriptor.mip_level_count >
            image->descriptor.descriptor.mip_levels ||
        descriptor->descriptor.base_array_layer + descriptor->descriptor.array_layer_count >
            image->descriptor.descriptor.array_layers)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot.bindingImage.image = image->image.image;
    slot.bindingImage.layout = image->image.layout;
    slot.bindingImage.format = image->image.format;
    slot.bindingImage.colorAttachment = image->image.colorAttachment;
    slot.bindingImage.view = vulkanHandle<VkImageView>(descriptor->image_view);
    slot.bindingImage.owned = false;
    slot.descriptor = descriptor->descriptor;
    slot.parent.emplace(std::move(retained).value());
    slot.ownedNative = false;
    auto published = slot.lifecycle.publish(std::move(creation).value(), &context, teardownImageView);
    if (published.isErr()) {
        slot.bindingImage = {};
        slot.descriptor = {};
        slot.parent.reset();
        return status(std::move(published).error());
    }
    *output = publicHandle<VernonRhiImageView>(published.value());
    return VERNON_RHI_STATUS_OK;
}

static VernonRhiStatus vernonRhiVulkanDeviceGetBufferNativeHandleBusiness(VernonRhiDevice handle,
                                                                          VernonRhiBuffer buffer, uint64_t *output) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VulkanInteropDevice &context = device.value().device();
    auto found = findVulkanSlot(context, context.buffers, typedHandle<vernon::rhi::BufferResourceTag>(buffer),
                                "get_vulkan_buffer_native_handle");
    if (found.isErr())
        return status(std::move(found).error());
    VulkanBufferSlot *slot = found.value();
    auto pin = slot->lifecycle.pin(typedHandle<vernon::rhi::BufferResourceTag>(buffer));
    if (pin.isErr())
        return status(std::move(pin).error());
    std::lock_guard<std::mutex> guard(context.mutex);
    *output = vulkanHandleBits(slot->buffer.buffer);
    return VERNON_RHI_STATUS_OK;
}

static vernon::Result<VernonRhiDevice, vernon::RhiError>
vernonRhiCreateBorrowedVulkanDeviceImpl(const VernonRhiVulkanBorrowedDeviceDescriptor *descriptor) {
    VernonRhiDevice device = vernonRhiCreateBorrowedVulkanDeviceBusiness(descriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX || !device.generation)
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"create_borrowed_vulkan_device", 0, 0}})};
    return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::ok(device)};
}

extern "C" VERNON_RHI_CAPI VernonRhiDevice
vernonRhiCreateBorrowedVulkanDevice(const VernonRhiVulkanBorrowedDeviceDescriptor *descriptor) {
    return vernon::rhi::publicHandleBoundary([&] { return vernonRhiCreateBorrowedVulkanDeviceImpl(descriptor); },
                                             invalidDevice());
}

#define VERNON_VULKAN_STATUS_BOUNDARY(name, parameters, arguments)                                                     \
    static vernon::Result<void, vernon::RhiError> name##Impl parameters {                                              \
        return vernon::rhi::publicStatusResult(name##Business arguments, #name);                                       \
    }                                                                                                                  \
    extern "C" VERNON_RHI_CAPI VernonRhiStatus name parameters {                                                       \
        return vernon::rhi::publicStatusBoundary([&] { return name##Impl arguments; });                                \
    }

VERNON_VULKAN_STATUS_BOUNDARY(vernonRhiVulkanDeviceGetBorrowedQueue, (VernonRhiDevice handle, void **output),
                              (handle, output))
VERNON_VULKAN_STATUS_BOUNDARY(vernonRhiVulkanDeviceGetBorrowedCommandBuffer, (VernonRhiDevice handle, void **output),
                              (handle, output))
VERNON_VULKAN_STATUS_BOUNDARY(vernonRhiVulkanDeviceImportBorrowedBuffer,
                              (VernonRhiDevice handle, const VernonRhiVulkanBorrowedBufferDescriptor *descriptor,
                               VernonRhiBuffer *output),
                              (handle, descriptor, output))
VERNON_VULKAN_STATUS_BOUNDARY(vernonRhiVulkanDeviceImportBorrowedImage,
                              (VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageDescriptor *descriptor,
                               VernonRhiImage *output),
                              (handle, descriptor, output))
VERNON_VULKAN_STATUS_BOUNDARY(vernonRhiVulkanDeviceImportBorrowedImageView,
                              (VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageViewDescriptor *descriptor,
                               VernonRhiImageView *output),
                              (handle, descriptor, output))
VERNON_VULKAN_STATUS_BOUNDARY(vernonRhiVulkanDeviceGetBufferNativeHandle,
                              (VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t *output),
                              (handle, buffer, output))

#undef VERNON_VULKAN_STATUS_BOUNDARY

vernon::Result<VernonRhiImageView, vernon::RhiError> createImageView(VernonRhiDevice handle,
                                                                     const VernonRhiImageViewDescriptor *descriptor) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr() || !descriptor || descriptor->struct_size < sizeof(*descriptor) ||
        !descriptor->mip_level_count || !descriptor->array_layer_count || !descriptor->aspects)
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{
            vernon::err(invalidResource("create_vulkan_image_view"))};
    VulkanInteropDevice &context = device.value().device();
    auto found = findVulkanSlot(context, context.images, typedHandle<vernon::rhi::ImageResourceTag>(descriptor->image),
                                "create_vulkan_image_view");
    if (found.isErr())
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{vernon::err(std::move(found).error())};
    VulkanImageSlot *image = found.value();
    auto imagePin = image->lifecycle.pin(typedHandle<vernon::rhi::ImageResourceTag>(descriptor->image));
    if (imagePin.isErr())
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{vernon::err(std::move(imagePin).error())};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{vernon::err(std::move(owner).error())};
    auto creation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{vernon::err(std::move(creation).error())};
    auto retained = image->lifecycle.retain(typedHandle<vernon::rhi::ImageResourceTag>(descriptor->image));
    if (retained.isErr())
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{vernon::err(std::move(retained).error())};
    std::lock_guard<std::mutex> creationGuard(context.creationMutex);
    auto reserved = reserveVulkanSlot<vernon::rhi::ImageViewResourceTag>(context, context.imageViews);
    if (reserved.isErr())
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{vernon::err(std::move(reserved).error())};
    VulkanImageViewSlot &slot = *reserved.value();
    std::lock_guard<std::mutex> guard(context.mutex);
    const VernonRhiImageDescriptor &imageDescriptor =
        image->image.owned ? image->ownedDescriptor : image->descriptor.descriptor;
    if (!vernon::rhi::validImageViewDescriptor(imageDescriptor, *descriptor))
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{
            vernon::err(invalidResource("create_vulkan_image_view"))};
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
    if (vernon::rhi::vulkan::driver().createImageView(context.state.device, &info, nullptr, &nativeView) != VK_SUCCESS)
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{
            vernon::err(backendFailure("create_vulkan_image_view"))};
    slot.bindingImage.image = image->image.image;
    slot.bindingImage.layout = image->image.layout;
    slot.bindingImage.format = image->image.format;
    slot.bindingImage.colorAttachment = image->image.colorAttachment;
    slot.bindingImage.view = nativeView;
    slot.bindingImage.owned = false;
    slot.descriptor = *descriptor;
    slot.parent.emplace(std::move(retained).value());
    slot.ownedNative = true;
    auto published = slot.lifecycle.publish(std::move(creation).value(), &context, teardownImageView);
    if (published.isErr()) {
        vernon::rhi::vulkan::driver().destroyImageView(context.state.device, nativeView, nullptr);
        slot.bindingImage = {};
        slot.descriptor = {};
        slot.ownedNative = false;
        slot.parent.reset();
        return vernon::Result<VernonRhiImageView, vernon::RhiError>{vernon::err(std::move(published).error())};
    }
    return vernon::Result<VernonRhiImageView, vernon::RhiError>{
        vernon::ok(publicHandle<VernonRhiImageView>(published.value()))};
}

vernon::Result<void, vernon::RhiError> destroyImageView(VernonRhiDevice handle, VernonRhiImageView imageView) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr())
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(invalidResource("destroy_vulkan_image_view", imageView.generation, imageView.index))};
    VulkanInteropDevice &context = device.value().device();
    auto found = findVulkanSlot(context, context.imageViews, typedHandle<vernon::rhi::ImageViewResourceTag>(imageView),
                                "destroy_vulkan_image_view");
    if (found.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(found).error())};
    return found.value()->lifecycle.destroyPublic(typedHandle<vernon::rhi::ImageViewResourceTag>(imageView));
}

vernon::Result<uint64_t, vernon::RhiError> getImageViewNativeHandle(VernonRhiDevice handle,
                                                                    VernonRhiImageView imageView) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr())
        return vernon::Result<uint64_t, vernon::RhiError>{
            vernon::err(invalidResource("get_vulkan_image_view_native_handle", imageView.generation, imageView.index))};
    VulkanInteropDevice &context = device.value().device();
    auto found = findVulkanSlot(context, context.imageViews, typedHandle<vernon::rhi::ImageViewResourceTag>(imageView),
                                "get_vulkan_image_view_native_handle");
    if (found.isErr())
        return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(found).error())};
    VulkanImageViewSlot *slot = found.value();
    auto pin = slot->lifecycle.pin(typedHandle<vernon::rhi::ImageViewResourceTag>(imageView));
    if (pin.isErr())
        return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(context.mutex);
    return vernon::Result<uint64_t, vernon::RhiError>{vernon::ok(vulkanHandleBits(slot->bindingImage.view))};
}

vernon::Result<VernonRhiDevice, vernon::RhiError> createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{
            vernon::err(invalidResource("create_owned_vulkan_device"))};
    auto reserved = vulkanDevices().reserve();
    if (reserved.isErr()) {
        vernon::rhi::setDeviceCreationError("cannot reserve Vulkan RHI device state");
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(reserved).error())};
    }
    auto reservation = std::move(reserved).value();
    VulkanInteropDevice &device = reservation.device();
    auto owner = reservation.retainOwner();
    if (owner.isErr()) {
        vernon::rhi::setDeviceCreationError("cannot retain Vulkan RHI device owner");
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(owner).error())};
    }
    auto commandState = vernon::rhi::createCommandDeviceState(std::move(owner).value());
    if (commandState.isErr()) {
        vernon::rhi::setDeviceCreationError("cannot allocate Vulkan command state");
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(commandState).error())};
    }
    device.commandState = std::move(commandState).value();
    if (!device.state.initialize(descriptor->device_index, device.error)) {
        vernon::rhi::setDeviceCreationError(std::move(device.error));
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{
            vernon::err(backendFailure("create_owned_vulkan_device", descriptor->device_index))};
    }
    device.queueCapabilities = VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    auto published = vulkanDevices().publish(std::move(reservation));
    if (published.isErr()) {
        vernon::rhi::setDeviceCreationError("cannot publish Vulkan RHI device state");
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(published).error())};
    }
    return vernon::Result<VernonRhiDevice, vernon::RhiError>{
        vernon::ok(VernonRhiDevice{published.value().index | vulkanDeviceBit, published.value().generation})};
}

vernon::Result<void, vernon::RhiError> destroyDevice(VernonRhiDevice handle) noexcept {
    if ((handle.index & vulkanDeviceBit) == 0)
        return vernon::Result<void, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"destroy_vulkan_device", handle.index, 0}})};
    return vulkanDevices().remove({handle.index & ~vulkanDeviceBit, handle.generation}, teardownDevice);
}

vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError> commandState(VernonRhiDevice handle) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr())
        return vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"vulkan_command_state", handle.index, 0}})};
    return device.value().device().commandState.retain();
}

vernon::Option<VernonStringView> lastError(VernonRhiDevice handle) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr())
        return vernon::none;
    VulkanInteropDevice &context = device.value().device();
    std::lock_guard<std::mutex> guard(context.mutex);
    return vernon::Option<VernonStringView>{vernon::some(VernonStringView{context.error.data(), context.error.size()})};
}

vernon::Result<void, vernon::RhiError> synchronize(VernonRhiDevice handle) {
    auto device = lookupVulkanDevice(handle);
    if (device.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(unsupported("synchronize_vulkan_device"))};
    VulkanInteropDevice &context = device.value().device();
    std::lock_guard<std::mutex> guard(context.mutex);
    if (!context.state.synchronize(context.error))
        return vernon::Result<void, vernon::RhiError>{vernon::err(backendFailure("synchronize_vulkan_device"))};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<VernonRhiBuffer, vernon::RhiError> createBuffer(VernonRhiDevice handle,
                                                               const VernonRhiBufferDescriptor *descriptor) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{vernon::err(unsupported("create_vulkan_buffer"))};

    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK)
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{vernon::err(invalidResource("create_vulkan_buffer"))};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{vernon::err(std::move(owner).error())};
    auto creation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{vernon::err(std::move(creation).error())};
    std::lock_guard<std::mutex> creationGuard(device->creationMutex);
    auto reserved = reserveVulkanSlot<vernon::rhi::BufferResourceTag>(device.value().device(), device->buffers);
    if (reserved.isErr())
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{vernon::err(std::move(reserved).error())};
    VulkanBufferSlot &slot = *reserved.value();
    std::lock_guard<std::mutex> guard(device->mutex);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_STORAGE) != 0)
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_VERTEX) != 0)
        usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_INDEX) != 0)
        usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if ((descriptor->usage & VERNON_RHI_BUFFER_UNIFORM) != 0)
        usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (descriptor->size > (std::numeric_limits<VkDeviceSize>::max)() - 3)
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{
            vernon::err(invalidResource("create_vulkan_buffer", descriptor->size))};
    const VkDeviceSize allocationSize = (descriptor->size + 3) & ~VkDeviceSize{3};
    const VkMemoryPropertyFlags requiredMemory =
        descriptor->memory_class == VERNON_RHI_MEMORY_DEVICE
            ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags preferredMemory =
        descriptor->memory_class == VERNON_RHI_MEMORY_DEVICE ? 0 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (!device->state.createBuffer(slot.buffer, allocationSize, usage, requiredMemory, device->error,
                                    preferredMemory)) {
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{
            vernon::err(backendFailure("create_vulkan_buffer", descriptor->size))};
    }
    if (descriptor->memory_class != VERNON_RHI_MEMORY_DEVICE) {
        void *mapped = nullptr;
        const VkResult mapResult = vernon::rhi::vulkan::driver().mapMemory(device->state.device, slot.buffer.memory, 0,
                                                                           allocationSize, 0, &mapped);
        if (mapResult != VK_SUCCESS) {
            device->error = "vkMapMemory failed with Vulkan error " + std::to_string(static_cast<int>(mapResult));
            device->state.destroyBuffer(slot.buffer);
            return vernon::Result<VernonRhiBuffer, vernon::RhiError>{
                vernon::err(backendFailure("map_vulkan_buffer", allocationSize))};
        }
        slot.buffer.mapped = static_cast<uint8_t *>(mapped);
    }
    slot.ownedDescriptor = *descriptor;
    auto published = slot.lifecycle.publish(std::move(creation).value(), &device.value().device(), teardownBuffer);
    if (published.isErr()) {
        device->state.destroyBuffer(slot.buffer);
        slot.ownedDescriptor = {};
        return vernon::Result<VernonRhiBuffer, vernon::RhiError>{vernon::err(std::move(published).error())};
    }
    return vernon::Result<VernonRhiBuffer, vernon::RhiError>{
        vernon::ok(publicHandle<VernonRhiBuffer>(published.value()))};
}

vernon::Result<void, vernon::RhiError> uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                                          const VernonRhiBufferUploadRange *ranges, size_t rangeCount);

vernon::Result<void, vernon::RhiError> uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset,
                                                    const void *source, uint64_t size) {
    const VernonRhiBufferUploadRange range{offset, source, size};
    return uploadBufferRanges(handle, buffer, &range, 1);
}

vernon::Result<void, vernon::RhiError> uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                                          const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return unsupportedResult("upload_vulkan_buffer");
    if (!ranges || rangeCount == 0 || rangeCount > (std::numeric_limits<uint32_t>::max)())
        return invalidResult("upload_vulkan_buffer_ranges", rangeCount);
    auto pinned = findAndPinVulkanSlot(device.value().device(), device->buffers,
                                       typedHandle<vernon::rhi::BufferResourceTag>(buffer), "upload_vulkan_buffer");
    if (pinned.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanBufferSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (slot->ownedDescriptor.memory_class == VERNON_RHI_MEMORY_UPLOAD) {
        if (!slot->buffer.mapped)
            return backendResult("upload_vulkan_buffer");
        for (size_t index = 0; index < rangeCount; ++index) {
            const VernonRhiBufferUploadRange &range = ranges[index];
            if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
                range.offset > slot->ownedDescriptor.size || range.size > slot->ownedDescriptor.size - range.offset)
                return invalidResult("upload_vulkan_buffer_ranges", index);
            std::memcpy(slot->buffer.mapped + range.offset, range.source, static_cast<size_t>(range.size));
        }
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    }
    if (vernon::rhi::deviceHasActiveCommandEncoder(handle)) {
        device->error = "device-level Vulkan upload cannot submit while a command encoder is recording";
        return invalidResult("upload_vulkan_buffer");
    }
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
            return invalidResult("upload_vulkan_buffer_ranges", index);
        if (packedSize > (std::numeric_limits<VkDeviceSize>::max)() - 15)
            return invalidResult("upload_vulkan_buffer_ranges", index);
        packedSize = (packedSize + 15) & ~VkDeviceSize{15};
        if (range.size > (std::numeric_limits<VkDeviceSize>::max)() - packedSize)
            return invalidResult("upload_vulkan_buffer_ranges", index);
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
                return invalidResult("upload_vulkan_buffer_ranges", alignedSize);
            readbackCopies.push_back({begin, alignedSize, size});
            alignedSize += size;
        }
        if (alignedSize > (std::numeric_limits<size_t>::max)())
            return invalidResult("upload_vulkan_buffer_ranges", alignedSize);
        std::vector<uint8_t> contents(static_cast<size_t>(alignedSize));
        VkBuffer readback{};
        VkDeviceSize readbackOffset{};
        uint8_t *readbackMapped = nullptr;
        if (!device->state.acquireStaging(false, alignedSize, 16, readback, readbackOffset, readbackMapped,
                                          device->error))
            return backendResult("acquire_vulkan_readback_staging", alignedSize);
        for (VkBufferCopy &copy : readbackCopies)
            copy.dstOffset += readbackOffset;
        VkCommandBuffer readbackCommand{};
        if (!device->state.beginCommands(readbackCommand, device->error))
            return backendResult("begin_vulkan_upload_readback");
        driver.cmdCopyBuffer(readbackCommand, slot->buffer.buffer, readback,
                             static_cast<uint32_t>(readbackCopies.size()), readbackCopies.data());
        if (!device->state.submitCommands(readbackCommand, device->error))
            return backendResult("submit_vulkan_upload_readback");
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
            return backendResult("acquire_vulkan_upload_staging", alignedSize);
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
            return backendResult("begin_vulkan_upload_commands");
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
        if (!device->state.submitCommands(command, device->error))
            return backendResult("submit_vulkan_upload_commands");
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    }
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(true, packedSize, 16, staging, stagingOffset, mapped, device->error))
        return backendResult("acquire_vulkan_upload_staging", packedSize);
    for (size_t index = 0; index < rangeCount; ++index) {
        std::memcpy(mapped + copies[index].srcOffset, ranges[index].source, static_cast<size_t>(ranges[index].size));
        copies[index].srcOffset += stagingOffset;
    }
    VkCommandBuffer command{};
    if (!device->state.beginCommands(command, device->error))
        return backendResult("begin_vulkan_upload_commands");
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
    if (!device->state.submitCommands(command, device->error))
        return backendResult("submit_vulkan_upload_commands");
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset,
                                                      void *destination, uint64_t size) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return unsupportedResult("download_vulkan_buffer");

    if (!destination || size == 0)
        return invalidResult("download_vulkan_buffer", size);
    auto pinned = findAndPinVulkanSlot(device.value().device(), device->buffers,
                                       typedHandle<vernon::rhi::BufferResourceTag>(buffer), "download_vulkan_buffer");
    if (pinned.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanBufferSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
        return invalidResult("download_vulkan_buffer", offset);
    const VkDeviceSize alignedBegin = offset & ~VkDeviceSize{3};
    const VkDeviceSize alignedEnd = (offset + size + 3) & ~VkDeviceSize{3};
    const VkDeviceSize alignedSize = alignedEnd - alignedBegin;
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(false, alignedSize, 16, staging, stagingOffset, mapped, device->error))
        return backendResult("acquire_vulkan_readback_staging", alignedSize);
    VkCommandBuffer command{};
    if (!device->state.beginCommands(command, device->error))
        return backendResult("begin_vulkan_download_commands");
    const VkBufferCopy copy{alignedBegin, stagingOffset, alignedSize};
    makeVulkanBufferReadableByTransfer(command, slot->buffer.buffer, alignedBegin, alignedSize);
    vernon::rhi::vulkan::driver().cmdCopyBuffer(command, slot->buffer.buffer, staging, 1, &copy);
    if (!device->state.submitCommands(command, device->error))
        return backendResult("submit_vulkan_download_commands");
    std::memcpy(destination, mapped + (offset - alignedBegin), static_cast<size_t>(size));
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> downloadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                                            const VernonRhiBufferDownloadRange *ranges,
                                                            size_t rangeCount) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return unsupportedResult("download_vulkan_buffer_ranges");
    if (!ranges || rangeCount == 0 || rangeCount > (std::numeric_limits<uint32_t>::max)())
        return invalidResult("download_vulkan_buffer_ranges", rangeCount);
    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->buffers,
                             typedHandle<vernon::rhi::BufferResourceTag>(buffer), "download_vulkan_buffer_ranges");
    if (pinned.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanBufferSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    std::vector<VkBufferCopy> copies;
    copies.reserve(rangeCount);
    VkDeviceSize stagingSize = 0;
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferDownloadRange &range = ranges[index];
        if (!range.destination || range.size == 0 || range.offset > slot->ownedDescriptor.size ||
            range.size > slot->ownedDescriptor.size - range.offset)
            return invalidResult("download_vulkan_buffer_ranges", index);
        const VkDeviceSize begin = range.offset & ~VkDeviceSize{3};
        const VkDeviceSize end = (range.offset + range.size + 3) & ~VkDeviceSize{3};
        stagingSize = (stagingSize + 3) & ~VkDeviceSize{3};
        if (end - begin > (std::numeric_limits<VkDeviceSize>::max)() - stagingSize)
            return invalidResult("download_vulkan_buffer_ranges", index);
        copies.push_back({begin, stagingSize, end - begin});
        stagingSize += end - begin;
    }
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(false, stagingSize, 16, staging, stagingOffset, mapped, device->error))
        return backendResult("acquire_vulkan_readback_staging", stagingSize);
    for (VkBufferCopy &copy : copies)
        copy.dstOffset += stagingOffset;
    VkCommandBuffer command{};
    if (!device->state.beginCommands(command, device->error))
        return backendResult("begin_vulkan_download_commands");
    makeVulkanBufferReadableByTransfer(command, slot->buffer.buffer, 0, slot->ownedDescriptor.size);
    vernon::rhi::vulkan::driver().cmdCopyBuffer(command, slot->buffer.buffer, staging,
                                                static_cast<uint32_t>(copies.size()), copies.data());
    if (!device->state.submitCommands(command, device->error))
        return backendResult("submit_vulkan_download_commands");
    for (size_t index = 0; index < rangeCount; ++index) {
        const VkDeviceSize begin = ranges[index].offset & ~VkDeviceSize{3};
        const VkDeviceSize mappedOffset = copies[index].dstOffset - stagingOffset;
        std::memcpy(ranges[index].destination, mapped + mappedOffset + ranges[index].offset - begin,
                    static_cast<size_t>(ranges[index].size));
    }
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<bool, vernon::RhiError> isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return vernon::Result<bool, vernon::RhiError>{
            vernon::err(invalidResource("validate_vulkan_buffer", buffer.generation, buffer.index))};
    }

    auto pinned = findAndPinVulkanSlot(device.value().device(), device->buffers,
                                       typedHandle<vernon::rhi::BufferResourceTag>(buffer), "validate_vulkan_buffer");
    if (pinned.isErr())
        return vernon::Result<bool, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    return vernon::Result<bool, vernon::RhiError>{vernon::ok(true)};
}

vernon::Result<VernonRhiImage, vernon::RhiError> createImage(VernonRhiDevice handle,
                                                             const VernonRhiImageDescriptor *descriptor) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<VernonRhiImage, vernon::RhiError>{vernon::err(invalidResource("create_vulkan_image"))};

    const VkFormat format = descriptor ? vulkanFormat(descriptor->format) : VK_FORMAT_UNDEFINED;
    const bool depthFormat = descriptor && (descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT ||
                                            descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT);
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || format == VK_FORMAT_UNDEFINED ||
        descriptor->width == 0 || descriptor->height == 0 || descriptor->depth == 0 || descriptor->mip_levels == 0 ||
        descriptor->sample_count != 1 || (depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)) ||
        (!depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)))
        return vernon::Result<VernonRhiImage, vernon::RhiError>{vernon::err(invalidResource("create_vulkan_image"))};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return vernon::Result<VernonRhiImage, vernon::RhiError>{vernon::err(std::move(owner).error())};
    auto creation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return vernon::Result<VernonRhiImage, vernon::RhiError>{vernon::err(std::move(creation).error())};
    std::lock_guard<std::mutex> creationGuard(device->creationMutex);
    auto reserved = reserveVulkanSlot<vernon::rhi::ImageResourceTag>(device.value().device(), device->images);
    if (reserved.isErr())
        return vernon::Result<VernonRhiImage, vernon::RhiError>{vernon::err(std::move(reserved).error())};
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
        return vernon::Result<VernonRhiImage, vernon::RhiError>{
            vernon::err(unsupported("create_vulkan_image", descriptor->format))};
    }
    VulkanImageSlot &slot = *reserved.value();
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
    const size_t subresourceCount = static_cast<size_t>(descriptor->mip_levels) * descriptor->array_layers;
    auto subresourceLayouts = decltype(slot.image.subresourceLayouts)(subresourceCount, VK_IMAGE_LAYOUT_UNDEFINED);
    if (!device->state.createImage(slot.image, imageInfo, viewInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                   device->error)) {
        return vernon::Result<VernonRhiImage, vernon::RhiError>{
            vernon::err(backendFailure("create_vulkan_image", descriptor->format))};
    }
    slot.ownedDescriptor = *descriptor;
    slot.image.subresourceLayouts = std::move(subresourceLayouts);
    auto published = slot.lifecycle.publish(std::move(creation).value(), &device.value().device(), teardownImage);
    if (published.isErr()) {
        device->state.destroyImage(slot.image);
        slot.ownedDescriptor = {};
        return vernon::Result<VernonRhiImage, vernon::RhiError>{vernon::err(std::move(published).error())};
    }
    return vernon::Result<VernonRhiImage, vernon::RhiError>{
        vernon::ok(publicHandle<VernonRhiImage>(published.value()))};
}

vernon::Result<void, vernon::RhiError> setImageSampler(VernonRhiDevice, VernonRhiImage,
                                                       const VernonRhiSamplerDescriptor *) {
    return unsupportedResult("set_vulkan_image_sampler");
}

vernon::Result<void, vernon::RhiError> uploadImage(VernonRhiDevice handle, VernonRhiImage image,
                                                   const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return unsupportedResult("upload_vulkan_image");

    if (!uploads || uploadCount == 0)
        return invalidResult("upload_vulkan_image", uploadCount);
    auto pinned = findAndPinVulkanSlot(device.value().device(), device->images,
                                       typedHandle<vernon::rhi::ImageResourceTag>(image), "upload_vulkan_image");
    if (pinned.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanImageSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (!slot || !slot->image.owned)
        return invalidResult("upload_vulkan_image");
    const VernonRhiImageDescriptor &descriptor = slot->ownedDescriptor;
    if ((descriptor.dimension != VERNON_RHI_IMAGE_2D && descriptor.dimension != VERNON_RHI_IMAGE_3D &&
         descriptor.dimension != VERNON_RHI_IMAGE_CUBE) ||
        (descriptor.dimension != VERNON_RHI_IMAGE_3D && descriptor.depth != 1))
        return invalidResult("upload_vulkan_image");
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
        const auto uploadSize = vernon::rhi::imageTransferRegionByteSize(descriptor.format, upload.aspect, upload.width,
                                                                         upload.height, upload.depth);
        if (upload.struct_size < sizeof(upload) || !validMip || !validLayer || !upload.width || !upload.height ||
            !upload.depth || upload.offset_x >= mipWidth || upload.width > mipWidth - upload.offset_x ||
            upload.offset_y >= mipHeight || upload.height > mipHeight - upload.offset_y ||
            upload.offset_z >= mipDepth || upload.depth > mipDepth - upload.offset_z ||
            !vernon::rhi::imageTransferAspectMatches(descriptor.format, upload.aspect, upload.source_format,
                                                     upload.source_type) ||
            !upload.data || !uploadSize)
            return invalidResult("upload_vulkan_image", index);
        if (*uploadSize > (std::numeric_limits<size_t>::max)() - totalSize)
            return invalidResult("upload_vulkan_image", index);
        uploadSizes[index] = *uploadSize;
        totalSize += uploadSizes[index];
    }
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(true, totalSize, 16, staging, stagingOffset, mapped, device->error))
        return backendResult("acquire_vulkan_image_upload_staging", totalSize);
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(uploadCount * 2);
    size_t byteOffset = 0;
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const bool packedDepthStencil =
            upload.aspect == (VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL);
        const size_t pixels = static_cast<size_t>(upload.width) * upload.height * upload.depth;
        if (packedDepthStencil) {
            const auto *source = static_cast<const uint8_t *>(upload.data);
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                float depth{};
                uint8_t stencil{};
                vernon::rhi::loadPackedDepthStencil(source + pixel * vernon::rhi::packedDepthStencilPixelSize, depth,
                                                    stencil);
                std::memcpy(mapped + byteOffset + pixel * sizeof(float), &depth, sizeof(depth));
                mapped[byteOffset + pixels * sizeof(float) + pixel] = stencil;
            }
        } else {
            std::memcpy(mapped + byteOffset, upload.data, uploadSizes[index]);
        }
        VkBufferImageCopy region{};
        region.bufferOffset = stagingOffset + byteOffset;
        region.imageSubresource.aspectMask = (upload.aspect & VERNON_RHI_IMAGE_ASPECT_COLOR) ? VK_IMAGE_ASPECT_COLOR_BIT
                                             : (upload.aspect & VERNON_RHI_IMAGE_ASPECT_DEPTH)
                                                 ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                 : VK_IMAGE_ASPECT_STENCIL_BIT;
        region.imageSubresource.mipLevel = upload.mip_level;
        region.imageSubresource.baseArrayLayer = upload.array_layer;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {static_cast<int32_t>(upload.offset_x), static_cast<int32_t>(upload.offset_y),
                              static_cast<int32_t>(upload.offset_z)};
        region.imageExtent = {upload.width, upload.height, upload.depth};
        regions.push_back(region);
        if (packedDepthStencil) {
            region.bufferOffset += pixels * sizeof(float);
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            regions.push_back(region);
        }
        byteOffset += uploadSizes[index];
    }
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!device->state.beginCommands(command, device->error))
        return backendResult("begin_vulkan_image_upload");
    transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vernon::rhi::vulkan::driver().cmdCopyBufferToImage(command, staging, slot->image.image,
                                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                       static_cast<uint32_t>(regions.size()), regions.data());
    transitionVulkanImage(command, *slot, defaultVulkanImageLayout(slot->ownedDescriptor));
    if (!device->state.submitCommands(command, device->error))
        return backendResult("submit_vulkan_image_upload");
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                                                     const VernonRhiImageDownloadDescriptor *download,
                                                     void *destination, size_t size) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return unsupportedResult("download_vulkan_image");

    auto pinned = findAndPinVulkanSlot(device.value().device(), device->images,
                                       typedHandle<vernon::rhi::ImageResourceTag>(image), "download_vulkan_image");
    if (pinned.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanImageSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (!slot->image.owned || !download || download->struct_size < sizeof(*download) || !destination)
        return invalidResult("download_vulkan_image");
    const auto expected = imageDownloadByteSize(slot->ownedDescriptor, *download);
    if (!expected || size != *expected)
        return invalidResult("download_vulkan_image", size);
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(false, size, 16, staging, stagingOffset, mapped, device->error))
        return backendResult("acquire_vulkan_image_readback_staging", size);
    const VkImageLayout restore = slot->image.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? defaultVulkanImageLayout(slot->ownedDescriptor)
                                      : slot->image.layout;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!device->state.beginCommands(command, device->error))
        return backendResult("begin_vulkan_image_download");
    transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const bool depthStencil = download->aspect == (VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL);
    const size_t pixels = static_cast<size_t>(download->width) * download->height * download->depth;
    std::array<VkBufferImageCopy, 2> regions{};
    regions[0].bufferOffset = stagingOffset;
    regions[0].imageSubresource.aspectMask =
        (download->aspect & VERNON_RHI_IMAGE_ASPECT_COLOR)   ? VK_IMAGE_ASPECT_COLOR_BIT
        : (download->aspect & VERNON_RHI_IMAGE_ASPECT_DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                             : VK_IMAGE_ASPECT_STENCIL_BIT;
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
        return backendResult("submit_vulkan_image_download");
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
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> downloadImageBatch(VernonRhiDevice handle, VernonRhiImage image,
                                                          const VernonRhiImageDownload *downloads,
                                                          size_t downloadCount) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return unsupportedResult("download_vulkan_image_batch");
    if (!downloads || downloadCount == 0)
        return invalidResult("download_vulkan_image_batch", downloadCount);
    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->images, typedHandle<vernon::rhi::ImageResourceTag>(image),
                             "download_vulkan_image_batch");
    if (pinned.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanImageSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (!slot->image.owned)
        return invalidResult("download_vulkan_image_batch");
    std::vector<size_t> offsets;
    offsets.reserve(downloadCount);
    size_t totalSize = 0;
    for (size_t index = 0; index < downloadCount; ++index) {
        const auto expected = imageDownloadByteSize(slot->ownedDescriptor, downloads[index].descriptor);
        if (!downloads[index].destination || !expected || downloads[index].size != *expected ||
            totalSize > (std::numeric_limits<size_t>::max)() - 15)
            return invalidResult("download_vulkan_image_batch", index);
        totalSize = (totalSize + 15) & ~size_t{15};
        offsets.push_back(totalSize);
        if (*expected > (std::numeric_limits<size_t>::max)() - totalSize)
            return invalidResult("download_vulkan_image_batch", index);
        totalSize += *expected;
    }
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(false, totalSize, 16, staging, stagingOffset, mapped, device->error))
        return backendResult("acquire_vulkan_image_readback_staging", totalSize);
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(downloadCount * 2);
    for (size_t index = 0; index < downloadCount; ++index) {
        const VernonRhiImageDownloadDescriptor &download = downloads[index].descriptor;
        VkBufferImageCopy region{};
        region.bufferOffset = stagingOffset + offsets[index];
        region.imageSubresource.aspectMask =
            (download.aspect & VERNON_RHI_IMAGE_ASPECT_COLOR)   ? VK_IMAGE_ASPECT_COLOR_BIT
            : (download.aspect & VERNON_RHI_IMAGE_ASPECT_DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                                : VK_IMAGE_ASPECT_STENCIL_BIT;
        region.imageSubresource.mipLevel = download.mip_level;
        region.imageSubresource.baseArrayLayer = download.array_layer;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {static_cast<int32_t>(download.offset_x), static_cast<int32_t>(download.offset_y),
                              static_cast<int32_t>(download.offset_z)};
        region.imageExtent = {download.width, download.height, download.depth};
        regions.push_back(region);
        if (download.aspect == (VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL)) {
            region.bufferOffset +=
                static_cast<size_t>(download.width) * download.height * download.depth * sizeof(float);
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            regions.push_back(region);
        }
    }
    const VkImageLayout restore = slot->image.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? defaultVulkanImageLayout(slot->ownedDescriptor)
                                      : slot->image.layout;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!device->state.beginCommands(command, device->error))
        return backendResult("begin_vulkan_image_download_batch");
    transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vernon::rhi::vulkan::driver().cmdCopyImageToBuffer(command, slot->image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                       staging, static_cast<uint32_t>(regions.size()), regions.data());
    transitionVulkanImage(command, *slot, restore);
    if (!device->state.submitCommands(command, device->error))
        return backendResult("submit_vulkan_image_download_batch");
    for (size_t index = 0; index < downloadCount; ++index) {
        const VernonRhiImageDownloadDescriptor &download = downloads[index].descriptor;
        if (download.aspect != (VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL)) {
            std::memcpy(downloads[index].destination, mapped + offsets[index], downloads[index].size);
            continue;
        }
        const size_t pixels = static_cast<size_t>(download.width) * download.height * download.depth;
        auto *output = static_cast<uint8_t *>(downloads[index].destination);
        const uint8_t *depth = mapped + offsets[index];
        const uint8_t *stencil = depth + pixels * sizeof(float);
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            float depthValue{};
            std::memcpy(&depthValue, depth + pixel * sizeof(float), sizeof(depthValue));
            vernon::rhi::storePackedDepthStencil(output + pixel * vernon::rhi::packedDepthStencilPixelSize, depthValue,
                                                 stencil[pixel]);
        }
    }
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return unsupportedResult("generate_vulkan_image_mipmaps");
    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->images, typedHandle<vernon::rhi::ImageResourceTag>(image),
                             "generate_vulkan_image_mipmaps");
    if (pinned.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanImageSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (!slot->image.owned || slot->ownedDescriptor.mip_levels < 2 ||
        slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT ||
        slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT ||
        !(slot->ownedDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(slot->ownedDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION))
        return invalidResult("generate_vulkan_image_mipmaps");
    VkFormatProperties properties{};
    vernon::rhi::vulkan::driver().getPhysicalDeviceFormatProperties(device->state.physicalDevice, slot->image.format,
                                                                    &properties);
    constexpr VkFormatFeatureFlags blitFeatures = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if ((properties.optimalTilingFeatures & blitFeatures) != blitFeatures)
        return unsupportedResult("generate_vulkan_image_mipmaps", slot->ownedDescriptor.format);
    const VkImageLayout restore = slot->image.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? defaultVulkanImageLayout(slot->ownedDescriptor)
                                      : slot->image.layout;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!device->state.beginCommands(command, device->error))
        return backendResult("begin_vulkan_generate_mipmaps");
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
    if (!device->state.submitCommands(command, device->error))
        return backendResult("submit_vulkan_generate_mipmaps");
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> bindImage(VernonRhiDevice, VernonRhiImage, uint32_t textureUnit) {
    return unsupportedResult("bind_vulkan_image", textureUnit);
}

vernon::Result<void, vernon::RhiError> destroyImage(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return unsupportedResult("destroy_vulkan_image");
    }
    auto found = findVulkanSlot(device.value().device(), device->images,
                                typedHandle<vernon::rhi::ImageResourceTag>(image), "destroy_vulkan_image");
    if (found.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(found).error())};
    return found.value()->lifecycle.destroyPublic(typedHandle<vernon::rhi::ImageResourceTag>(image));
}

vernon::Result<bool, vernon::RhiError> isImageValid(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return vernon::Result<bool, vernon::RhiError>{
            vernon::err(invalidResource("validate_vulkan_image", image.generation, image.index))};
    }

    auto pinned = findAndPinVulkanSlot(device.value().device(), device->images,
                                       typedHandle<vernon::rhi::ImageResourceTag>(image), "validate_vulkan_image");
    if (pinned.isErr())
        return vernon::Result<bool, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    return vernon::Result<bool, vernon::RhiError>{vernon::ok(true)};
}

vernon::Result<VernonRhiSampler, vernon::RhiError> createSampler(VernonRhiDevice handle,
                                                                 const VernonRhiSamplerDescriptor *descriptor) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{vernon::err(unsupported("create_vulkan_sampler"))};

    SamplerFilter filter;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !decodeSamplerFilter(*descriptor, filter))
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{
            vernon::err(invalidResource("create_vulkan_sampler"))};
    if (filter.maxAnisotropy > 1.0f)
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{vernon::err(unsupported("create_vulkan_sampler"))};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{vernon::err(std::move(owner).error())};
    auto creation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{vernon::err(std::move(creation).error())};
    std::lock_guard<std::mutex> creationGuard(device->creationMutex);
    auto reserved = reserveVulkanSlot<vernon::rhi::SamplerResourceTag>(device.value().device(), device->samplers);
    if (reserved.isErr())
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{vernon::err(std::move(reserved).error())};
    VulkanSamplerSlot &slot = *reserved.value();
    std::lock_guard<std::mutex> guard(device->mutex);
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
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{vernon::err(backendFailure("create_vulkan_sampler"))};
    }
    auto published = slot.lifecycle.publish(std::move(creation).value(), &device.value().device(), teardownSampler);
    if (published.isErr()) {
        device->state.destroySampler(slot.sampler);
        return vernon::Result<VernonRhiSampler, vernon::RhiError>{vernon::err(std::move(published).error())};
    }
    return vernon::Result<VernonRhiSampler, vernon::RhiError>{
        vernon::ok(publicHandle<VernonRhiSampler>(published.value()))};
}

vernon::Result<void, vernon::RhiError> destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return unsupportedResult("destroy_vulkan_sampler");
    }
    auto found = findVulkanSlot(device.value().device(), device->samplers,
                                typedHandle<vernon::rhi::SamplerResourceTag>(sampler), "destroy_vulkan_sampler");
    if (found.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(found).error())};
    return found.value()->lifecycle.destroyPublic(typedHandle<vernon::rhi::SamplerResourceTag>(sampler));
}

vernon::Result<bool, vernon::RhiError> isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return vernon::Result<bool, vernon::RhiError>{
            vernon::err(invalidResource("validate_vulkan_sampler", sampler.generation, sampler.index))};
    }

    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->samplers,
                             typedHandle<vernon::rhi::SamplerResourceTag>(sampler), "validate_vulkan_sampler");
    if (pinned.isErr())
        return vernon::Result<bool, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    return vernon::Result<bool, vernon::RhiError>{vernon::ok(true)};
}

vernon::Result<uint64_t, vernon::RhiError> getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(unsupported("get_vulkan_image_native_handle"))};
    }

    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->images, typedHandle<vernon::rhi::ImageResourceTag>(image),
                             "get_vulkan_image_native_handle");
    if (pinned.isErr())
        return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(pinned).error())};
    VulkanImageSlot *slot = pinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    return vernon::Result<uint64_t, vernon::RhiError>{vernon::ok(vulkanHandleBits(slot->image.image))};
}

vernon::Result<void, vernon::RhiError> destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return unsupportedResult("destroy_vulkan_buffer");
    }
    auto found = findVulkanSlot(device.value().device(), device->buffers,
                                typedHandle<vernon::rhi::BufferResourceTag>(buffer), "destroy_vulkan_buffer");
    if (found.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(found).error())};
    return found.value()->lifecycle.destroyPublic(typedHandle<vernon::rhi::BufferResourceTag>(buffer));
}

vernon::Result<void *, vernon::RhiError> getBufferNativeHandle(VernonRhiDevice, VernonRhiBuffer) {
    return vernon::Result<void *, vernon::RhiError>{vernon::err(unsupported("get_vulkan_buffer_native_handle"))};
}

bool ownsDevice(VernonRhiDevice handle) { return static_cast<bool>(lookupVulkanDevice(handle)); }

vernon::Result<void *, vernon::RhiError> deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<void *, vernon::RhiError>{
            vernon::err(invalidResource("get_vulkan_device_state", handle.generation, handle.index))};
    return vernon::Result<void *, vernon::RhiError>{vernon::ok(static_cast<void *>(&device->state))};
}

Result<CommandRecording, RhiError> beginCommands(VernonRhiDevice handle) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return Result<CommandRecording, RhiError>{
            err(invalidResource("begin_vulkan_commands", handle.generation, handle.index))};

    std::lock_guard<std::mutex> guard(device->mutex);
    VkCommandBuffer command{};
    if (!device->state.beginCommands(command, device->error))
        return Result<CommandRecording, RhiError>{
            err(backendFailure("begin_vulkan_commands", handle.generation, handle.index))};
    return Result<CommandRecording, RhiError>{
        ok(CommandRecording{vulkanHandleBits(command), VERNON_RHI_BACKEND_VULKAN})};
}

Result<CommandSubmission, RhiError> submitCommands(VernonRhiDevice handle, uint64_t native, bool) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return Result<CommandSubmission, RhiError>{
            err(invalidResource("submit_vulkan_commands", handle.generation, handle.index))};

    std::lock_guard<std::mutex> guard(device->mutex);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    const bool submitted = command && device->state.submitCommands(command, device->error);
    if (!submitted)
        return Result<CommandSubmission, RhiError>{err(backendFailure("submit_vulkan_commands", native))};
    return Result<CommandSubmission, RhiError>{
        ok(CommandSubmission{!device->state.nativeObjectsBorrowed, device->state.nativeObjectsBorrowed})};
}

Result<void, RhiError> completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) noexcept {
    if (!lookupVulkanDevice(handle))
        return invalidResult("complete_vulkan_commands", native);
    return Result<void, RhiError>{ok()};
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return;

    std::lock_guard<std::mutex> guard(device->mutex);
    device->state.abandonCommands(vulkanHandle<VkCommandBuffer>(native));
}

Result<void, RhiError> recordBarriers(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native,
                                      const VernonRhiBarrier *barriers, size_t barrierCount) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordBarriers_vulkan", native, 0}})};
    }

    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!command || (!barriers && barrierCount))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordBarriers_vulkan", native, 0}})};
    try {
        std::vector<vernon::OperationPin> pins;
        std::vector<void *> slots;
        pins.reserve(barrierCount);
        slots.reserve(barrierCount);
        for (size_t index = 0; index < barrierCount; ++index) {
            if (barriers[index].is_image) {
                auto pinned = findAndPinVulkanSlot(device.value().device(), device->images,
                                                   typedHandle<vernon::rhi::ImageResourceTag>(barriers[index].image),
                                                   "record_vulkan_barrier");
                if (pinned.isErr())
                    return Result<void, RhiError>{err(std::move(pinned).error())};
                auto access = std::move(pinned).value();
                slots.push_back(access.slot);
                pins.push_back(std::move(access.pin));
            } else {
                auto pinned = findAndPinVulkanSlot(device.value().device(), device->buffers,
                                                   typedHandle<vernon::rhi::BufferResourceTag>(barriers[index].buffer),
                                                   "record_vulkan_barrier");
                if (pinned.isErr())
                    return Result<void, RhiError>{err(std::move(pinned).error())};
                auto access = std::move(pinned).value();
                slots.push_back(access.slot);
                pins.push_back(std::move(access.pin));
            }
        }
        std::lock_guard<std::mutex> guard(device->mutex);
        std::vector<VkBufferMemoryBarrier> bufferBarriers;
        std::vector<VkImageMemoryBarrier> imageBarriers;
        bufferBarriers.reserve(barrierCount);
        imageBarriers.reserve(barrierCount);
        VkPipelineStageFlags sourceStages = 0;
        VkPipelineStageFlags destinationStages = 0;
        for (size_t index = 0; index < barrierCount; ++index) {
            if (barriers[index].is_image) {
                auto *slot = static_cast<VulkanImageSlot *>(slots[index]);
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
                    return Result<void, RhiError>{
                        err(RhiError{RhiErrorCode::InvalidArgument, {"recordBarriers_vulkan_range", native, 0}})};
                const VkImageLayout target = vulkanImageLayout(barriers[index].new_state);
                const size_t planeStride = static_cast<size_t>(descriptor.mip_levels) * imageLayers;
                const size_t subresourceCount = planeStride;
                if (slot->image.subresourceLayouts.size() != subresourceCount)
                    return Result<void, RhiError>{
                        err(RhiError{RhiErrorCode::LifecycleFailure, {"recordBarriers_vulkan_layout", native, 0}})};
                auto journal = slot->image.layoutJournals.find(encoderKey);
                bool inserted = false;
                if (journal == slot->image.layoutJournals.end()) {
                    const auto result = slot->image.layoutJournals.emplace(
                        encoderKey,
                        vernon::rhi::vulkan::Image::LayoutJournal{slot->image.layout, slot->image.subresourceLayouts});
                    journal = result.first;
                    inserted = result.second;
                }
                if (inserted) {
                    auto cleanup = deferCommandCleanup(handle, encoderKey, &slot->image, encoderKey,
                                                       clearVulkanImageLayoutJournal);
                    if (cleanup.isErr()) {
                        slot->image.layoutJournals.erase(journal);
                        return cleanup;
                    }
                    auto rollback =
                        deferCommandRollback(handle, encoderKey, &slot->image, encoderKey, restoreVulkanImageLayouts);
                    if (rollback.isErr()) {
                        slot->image.layoutJournals.erase(journal);
                        return rollback;
                    }
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
                auto *slot = static_cast<VulkanBufferSlot *>(slots[index]);
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
        return Result<void, RhiError>{ok()};
    } catch (const std::bad_alloc &) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::ResourceExhausted, {"recordBarriers_vulkan", native, 0}})};
    }
}

Result<void, RhiError> recordBufferCopy(VernonRhiDevice handle, uint64_t native, VernonRhiBuffer source,
                                        uint64_t sourceOffset, VernonRhiBuffer destination, uint64_t destinationOffset,
                                        uint64_t size) noexcept {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || !size)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordBufferCopy_vulkan", native, 0}})};
    auto sourcePinned =
        findAndPinVulkanSlot(device.value().device(), device->buffers,
                             typedHandle<vernon::rhi::BufferResourceTag>(source), "record_vulkan_buffer_copy");
    auto destinationPinned =
        findAndPinVulkanSlot(device.value().device(), device->buffers,
                             typedHandle<vernon::rhi::BufferResourceTag>(destination), "record_vulkan_buffer_copy");
    if (sourcePinned.isErr())
        return Result<void, RhiError>{err(std::move(sourcePinned).error())};
    if (destinationPinned.isErr())
        return Result<void, RhiError>{err(std::move(destinationPinned).error())};
    auto *sourceSlot = sourcePinned.value().slot;
    auto *destinationSlot = destinationPinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (sourceOffset > sourceSlot->ownedDescriptor.size || size > sourceSlot->ownedDescriptor.size - sourceOffset ||
        destinationOffset > destinationSlot->ownedDescriptor.size ||
        size > destinationSlot->ownedDescriptor.size - destinationOffset)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordBufferCopy_vulkan", native, 0}})};
    const VkBufferCopy copy{sourceOffset, destinationOffset, size};
    vernon::rhi::vulkan::driver().cmdCopyBuffer(command, sourceSlot->buffer.buffer, destinationSlot->buffer.buffer, 1,
                                                &copy);
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> recordImageCopy(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native,
                                       VernonRhiImage source, VernonRhiImage destination,
                                       const VernonRhiImageCopyRegion *regions, size_t regionCount) noexcept {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || !regions || !regionCount)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordImageCopy_vulkan", native, 0}})};
    auto sourcePinned =
        findAndPinVulkanSlot(device.value().device(), device->images,
                             typedHandle<vernon::rhi::ImageResourceTag>(source), "record_vulkan_image_copy");
    auto destinationPinned =
        findAndPinVulkanSlot(device.value().device(), device->images,
                             typedHandle<vernon::rhi::ImageResourceTag>(destination), "record_vulkan_image_copy");
    if (sourcePinned.isErr())
        return Result<void, RhiError>{err(std::move(sourcePinned).error())};
    if (destinationPinned.isErr())
        return Result<void, RhiError>{err(std::move(destinationPinned).error())};
    auto *sourceSlot = sourcePinned.value().slot;
    auto *destinationSlot = destinationPinned.value().slot;
    std::lock_guard<std::mutex> guard(device->mutex);
    const auto prepare = [&](VulkanImageSlot &slot, VkImageLayout layout) -> Result<void, RhiError> {
        auto journal = slot.image.layoutJournals.find(encoderKey);
        bool inserted = false;
        if (journal == slot.image.layoutJournals.end()) {
            try {
                const auto result = slot.image.layoutJournals.emplace(
                    encoderKey,
                    vernon::rhi::vulkan::Image::LayoutJournal{slot.image.layout, slot.image.subresourceLayouts});
                journal = result.first;
                inserted = result.second;
            } catch (const std::bad_alloc &) {
                return Result<void, RhiError>{
                    err(RhiError{RhiErrorCode::ResourceExhausted, {"recordImageCopy_vulkan", native, 0}})};
            }
        }
        if (inserted) {
            auto cleanup =
                deferCommandCleanup(handle, encoderKey, &slot.image, encoderKey, clearVulkanImageLayoutJournal);
            if (cleanup.isErr()) {
                slot.image.layoutJournals.erase(journal);
                return cleanup;
            }
            auto rollback =
                deferCommandRollback(handle, encoderKey, &slot.image, encoderKey, restoreVulkanImageLayouts);
            if (rollback.isErr()) {
                slot.image.layoutJournals.erase(journal);
                return rollback;
            }
        }
        transitionVulkanImage(command, slot, layout);
        return Result<void, RhiError>{ok()};
    };
    auto sourcePrepared = prepare(*sourceSlot, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if (sourcePrepared.isErr())
        return sourcePrepared;
    auto destinationPrepared = prepare(*destinationSlot, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    if (destinationPrepared.isErr())
        return destinationPrepared;
    try {
        std::vector<VkImageCopy> copies(regionCount);
        for (size_t index = 0; index < regionCount; ++index) {
            const VernonRhiImageCopyRegion &region = regions[index];
            VkImageCopy &copy = copies[index];
            copy.srcSubresource = {static_cast<VkImageAspectFlags>(region.aspects), region.source_mip_level,
                                   region.source_array_layer, 1};
            copy.srcOffset = {static_cast<int32_t>(region.source_x), static_cast<int32_t>(region.source_y),
                              static_cast<int32_t>(region.source_z)};
            copy.dstSubresource = {static_cast<VkImageAspectFlags>(region.aspects), region.destination_mip_level,
                                   region.destination_array_layer, 1};
            copy.dstOffset = {static_cast<int32_t>(region.destination_x), static_cast<int32_t>(region.destination_y),
                              static_cast<int32_t>(region.destination_z)};
            copy.extent = {region.width, region.height, region.depth};
        }
        vernon::rhi::vulkan::driver().cmdCopyImage(
            command, sourceSlot->image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destinationSlot->image.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(copies.size()), copies.data());
        return Result<void, RhiError>{ok()};
    } catch (const std::bad_alloc &) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::ResourceExhausted, {"recordImageCopy_vulkan", native, 0}})};
    }
}

Result<void, RhiError> endRendering(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend,
                                    uint32_t backendKind, uint32_t colorDiscardMask, uint32_t depthStencilDiscard,
                                    const uint64_t *colorResources, size_t colorCount, uint64_t depthResource,
                                    uint64_t) noexcept {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || backend != VERNON_RHI_BACKEND_VULKAN)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"endRendering_vulkan", native, backendKind}})};
    std::lock_guard<std::mutex> guard(device->mutex);
    if (backendKind == CommandRenderingDynamic)
        vernon::rhi::vulkan::driver().cmdEndRendering(command);
    else if (backendKind == CommandRenderingRenderPass)
        vernon::rhi::vulkan::driver().cmdEndRenderPass(command);
    else
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"endRendering_vulkan", native, backendKind}})};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> clearColor(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend,
                                  uint32_t backendKind, uint64_t, int32_t x, int32_t y, uint32_t width, uint32_t height,
                                  uint32_t layers, uint64_t target, uint32_t location, const float color[4]) noexcept {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || backend != VERNON_RHI_BACKEND_VULKAN ||
        (backendKind != CommandRenderingDynamic && backendKind != CommandRenderingRenderPass))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"clearColor_vulkan", native, backendKind}})};
    VkClearAttachment clear{};
    clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear.colorAttachment = location;
    std::copy(color, color + 4, clear.clearValue.color.float32);
    const VkClearRect rectangle{{{x, y}, {width, height}}, 0, layers};
    std::lock_guard<std::mutex> guard(device->mutex);
    vernon::rhi::vulkan::driver().cmdClearAttachments(command, 1, &clear, 1, &rectangle);
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> clearDepthStencil(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend,
                                         uint32_t backendKind, uint64_t, int32_t x, int32_t y, uint32_t width,
                                         uint32_t height, uint32_t layers, uint64_t target, float depth,
                                         uint32_t stencil, uint32_t aspects) noexcept {
    auto device = lookupVulkanDevice(handle);
    const VkCommandBuffer command = vulkanHandle<VkCommandBuffer>(native);
    if (!device || !command || backend != VERNON_RHI_BACKEND_VULKAN ||
        (backendKind != CommandRenderingDynamic && backendKind != CommandRenderingRenderPass))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"clearDepthStencil_vulkan", native, backendKind}})};
    VkClearAttachment clear{};
    clear.aspectMask = ((aspects & VERNON_RHI_ATTACHMENT_DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                       ((aspects & VERNON_RHI_ATTACHMENT_STENCIL) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    clear.clearValue.depthStencil = {depth, stencil};
    const VkClearRect rectangle{{{x, y}, {width, height}}, 0, layers};
    std::lock_guard<std::mutex> guard(device->mutex);
    vernon::rhi::vulkan::driver().cmdClearAttachments(command, 1, &clear, 1, &rectangle);
    return Result<void, RhiError>{ok()};
}

Result<uint64_t, RhiError> bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return Result<uint64_t, RhiError>{
            err(invalidResource("get_vulkan_buffer_resource", buffer.generation, buffer.index))};
    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->buffers,
                             typedHandle<vernon::rhi::BufferResourceTag>(buffer), "get_vulkan_buffer_resource");
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    return Result<uint64_t, RhiError>{ok(resourceKey(buffer))};
}

Result<uint64_t, RhiError> imageResource(VernonRhiDevice handle, VernonRhiImage image) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return Result<uint64_t, RhiError>{
            err(invalidResource("get_vulkan_image_resource", image.generation, image.index))};
    auto pinned = findAndPinVulkanSlot(device.value().device(), device->images,
                                       typedHandle<vernon::rhi::ImageResourceTag>(image), "get_vulkan_image_resource");
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    return Result<uint64_t, RhiError>{ok(resourceKey(image))};
}

Result<uint64_t, RhiError> imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return Result<uint64_t, RhiError>{
            err(invalidResource("get_vulkan_image_view_resource", view.generation, view.index))};
    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->imageViews,
                             typedHandle<vernon::rhi::ImageViewResourceTag>(view), "get_vulkan_image_view_resource");
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    return Result<uint64_t, RhiError>{ok(resourceKey(view))};
}

Result<uint64_t, RhiError> samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return Result<uint64_t, RhiError>{
            err(invalidResource("get_vulkan_sampler_resource", sampler.generation, sampler.index))};
    auto pinned =
        findAndPinVulkanSlot(device.value().device(), device->samplers,
                             typedHandle<vernon::rhi::SamplerResourceTag>(sampler), "get_vulkan_sampler_resource");
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    return Result<uint64_t, RhiError>{ok(resourceKey(sampler))};
}

vernon::Result<RetainedRhiResourceLease, vernon::RhiError> retainResource(VernonRhiDevice handle, ResourceKind kind,
                                                                          uint64_t key) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{
            vernon::err(invalidResource("retain_vulkan_resource", key, static_cast<uint32_t>(kind)))};
    if (kind == ResourceKind::Buffer) {
        auto decoded = decodeResourceKey<BufferResourceTag>(key, "retain_vulkan_buffer");
        if (decoded.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(decoded).error())};
        auto found = findVulkanSlot(device.value().device(), device->buffers, decoded.value(), "retain_vulkan_buffer");
        if (found.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(found).error())};
        auto retained = found.value()->lifecycle.retain(decoded.value());
        if (retained.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(retained).error())};
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{
            vernon::ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    if (kind == ResourceKind::Image) {
        auto decoded = decodeResourceKey<ImageResourceTag>(key, "retain_vulkan_image");
        if (decoded.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(decoded).error())};
        auto found = findVulkanSlot(device.value().device(), device->images, decoded.value(), "retain_vulkan_image");
        if (found.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(found).error())};
        auto retained = found.value()->lifecycle.retain(decoded.value());
        if (retained.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(retained).error())};
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{
            vernon::ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    if (kind == ResourceKind::ImageView) {
        auto decoded = decodeResourceKey<ImageViewResourceTag>(key, "retain_vulkan_image_view");
        if (decoded.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(decoded).error())};
        auto found =
            findVulkanSlot(device.value().device(), device->imageViews, decoded.value(), "retain_vulkan_image_view");
        if (found.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(found).error())};
        auto retained = found.value()->lifecycle.retain(decoded.value());
        if (retained.isErr())
            return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(retained).error())};
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{
            vernon::ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    if (kind != ResourceKind::Sampler)
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{
            vernon::err(invalidResource("retain_vulkan_resource", key, static_cast<uint32_t>(kind)))};
    auto decoded = decodeResourceKey<SamplerResourceTag>(key, "retain_vulkan_sampler");
    if (decoded.isErr())
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(decoded).error())};
    auto found = findVulkanSlot(device.value().device(), device->samplers, decoded.value(), "retain_vulkan_sampler");
    if (found.isErr())
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(found).error())};
    auto retained = found.value()->lifecycle.retain(decoded.value());
    if (retained.isErr())
        return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{vernon::err(std::move(retained).error())};
    return vernon::Result<RetainedRhiResourceLease, vernon::RhiError>{
        vernon::ok(RetainedRhiResourceLease{std::move(retained).value()})};
}

vernon::Result<uint64_t, vernon::RhiError> resolveResource(VernonRhiDevice handle, ResourceKind kind,
                                                           uint64_t key) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<uint64_t, vernon::RhiError>{
            vernon::err(invalidResource("resolve_vulkan_resource", key, static_cast<uint32_t>(kind)))};
    if (kind == ResourceKind::Buffer) {
        auto decoded = decodeResourceKey<BufferResourceTag>(key, "resolve_vulkan_buffer");
        if (decoded.isErr())
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(decoded).error())};
        auto found = findVulkanSlot(device.value().device(), device->buffers, decoded.value(), "resolve_vulkan_buffer");
        if (found.isErr())
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(found).error())};
        std::lock_guard<std::mutex> guard(device->mutex);
        return vernon::Result<uint64_t, vernon::RhiError>{
            vernon::ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&found.value()->buffer)))};
    }
    if (kind == ResourceKind::Image) {
        auto decoded = decodeResourceKey<ImageResourceTag>(key, "resolve_vulkan_image");
        if (decoded.isErr())
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(decoded).error())};
        auto found = findVulkanSlot(device.value().device(), device->images, decoded.value(), "resolve_vulkan_image");
        if (found.isErr())
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(found).error())};
        std::lock_guard<std::mutex> guard(device->mutex);
        return vernon::Result<uint64_t, vernon::RhiError>{
            vernon::ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&found.value()->image)))};
    }
    if (kind == ResourceKind::ImageView) {
        auto decoded = decodeResourceKey<ImageViewResourceTag>(key, "resolve_vulkan_image_view");
        if (decoded.isErr())
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(decoded).error())};
        auto found =
            findVulkanSlot(device.value().device(), device->imageViews, decoded.value(), "resolve_vulkan_image_view");
        if (found.isErr())
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(found).error())};
        std::lock_guard<std::mutex> guard(device->mutex);
        return vernon::Result<uint64_t, vernon::RhiError>{
            vernon::ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&found.value()->bindingImage)))};
    }
    if (kind != ResourceKind::Sampler)
        return vernon::Result<uint64_t, vernon::RhiError>{
            vernon::err(invalidResource("resolve_vulkan_resource", key, static_cast<uint32_t>(kind)))};
    auto decoded = decodeResourceKey<SamplerResourceTag>(key, "resolve_vulkan_sampler");
    if (decoded.isErr())
        return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(decoded).error())};
    auto found = findVulkanSlot(device.value().device(), device->samplers, decoded.value(), "resolve_vulkan_sampler");
    if (found.isErr())
        return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(found).error())};
    std::lock_guard<std::mutex> guard(device->mutex);
    return vernon::Result<uint64_t, vernon::RhiError>{
        vernon::ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&found.value()->sampler)))};
}

vernon::Result<ImageResourceDescription, vernon::RhiError> describeImageResource(VernonRhiDevice handle,
                                                                                 uint64_t key) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<ImageResourceDescription, vernon::RhiError>{
            vernon::err(invalidResource("describe_vulkan_image", key))};
    auto decoded = decodeResourceKey<ImageResourceTag>(key, "describe_vulkan_image");
    if (decoded.isErr())
        return vernon::Result<ImageResourceDescription, vernon::RhiError>{vernon::err(std::move(decoded).error())};
    auto found = findVulkanSlot(device.value().device(), device->images, decoded.value(), "describe_vulkan_image");
    if (found.isErr())
        return vernon::Result<ImageResourceDescription, vernon::RhiError>{vernon::err(std::move(found).error())};
    std::lock_guard<std::mutex> guard(device->mutex);
    const VulkanImageSlot *slot = found.value();
    return vernon::Result<ImageResourceDescription, vernon::RhiError>{
        vernon::ok(ImageResourceDescription{slot->image.owned ? slot->ownedDescriptor : slot->descriptor.descriptor})};
}

vernon::Result<ImageViewResourceDescription, vernon::RhiError> describeImageViewResource(VernonRhiDevice handle,
                                                                                         uint64_t key) noexcept {
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{
            vernon::err(invalidResource("describe_vulkan_image_view", key))};
    auto decoded = decodeResourceKey<ImageViewResourceTag>(key, "describe_vulkan_image_view");
    if (decoded.isErr())
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{vernon::err(std::move(decoded).error())};
    auto viewFound =
        findVulkanSlot(device.value().device(), device->imageViews, decoded.value(), "describe_vulkan_image_view");
    if (viewFound.isErr())
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{
            vernon::err(std::move(viewFound).error())};
    VulkanImageViewSlot *slot = viewFound.value();
    if (!slot->parent)
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{
            vernon::err(invalidResource("describe_vulkan_image_view_parent", key))};
    const auto parentHandle = slot->parent->handle();
    auto parentFound =
        findVulkanSlot(device.value().device(), device->images, parentHandle, "describe_vulkan_image_view_parent");
    if (parentFound.isErr())
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{
            vernon::err(std::move(parentFound).error())};
    const auto parentState = parentFound.value()->lifecycle.snapshot();
    if (!parentState.occupied || parentState.generation != parentHandle.generation)
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{
            vernon::err(invalidResource("describe_vulkan_image_view_parent", key))};
    std::lock_guard<std::mutex> guard(device->mutex);
    const VulkanImageSlot *parent = parentFound.value();
    return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{vernon::ok(ImageViewResourceDescription{
        slot->descriptor, parent->image.owned ? parent->ownedDescriptor : parent->descriptor.descriptor,
        resourceKey(parentHandle)})};
}

} // namespace vernon::rhi::vulkan_api

const vernon::rhi::BackendDispatch &vernon::rhi::vulkanBackendDispatch() {
    using namespace vulkan_api;
    static const BackendDispatch dispatch = [] {
        BackendDispatch result{};
        result.backend = VERNON_RHI_BACKEND_VULKAN;
        result.commandCapabilities = BackendCommandIndependentRecording;
        result.ownsDevice = ownsDevice;
        result.createOwnedDevice = createOwnedDevice;
        result.destroyDevice = vulkan_api::destroyDevice;
        result.lastError = lastError;
        result.synchronize = synchronize;
        result.deviceState = deviceStateForBackend;
        result.commandState = commandState;
        result.createBuffer = createBuffer;
        result.uploadBuffer = uploadBuffer;
        result.uploadBufferRanges = uploadBufferRanges;
        result.downloadBufferRanges = downloadBufferRanges;
        result.downloadBuffer = downloadBuffer;
        result.destroyBuffer = destroyBuffer;
        result.isBufferValid = isBufferValid;
        result.getBufferNativeHandle = getBufferNativeHandle;
        result.createImage.emplace(createImage);
        result.setImageSampler.emplace(setImageSampler);
        result.uploadImage.emplace(uploadImage);
        result.downloadImage.emplace(downloadImage);
        result.downloadImageBatch.emplace(downloadImageBatch);
        result.generateImageMipmaps.emplace(generateImageMipmaps);
        result.bindImage.emplace(bindImage);
        result.destroyImage.emplace(destroyImage);
        result.isImageValid.emplace(isImageValid);
        result.getImageNativeHandle.emplace(getImageNativeHandle);
        result.createSampler.emplace(createSampler);
        result.destroySampler.emplace(destroySampler);
        result.isSamplerValid.emplace(isSamplerValid);
        result.bufferResource = bufferResource;
        result.imageResource.emplace(imageResource);
        result.samplerResource.emplace(samplerResource);
        result.retainResource = retainResource;
        result.resolveRetainedResource = resolveResource;
        result.describeImageResource.emplace(describeImageResource);
        result.beginCommands = beginCommands;
        result.submitCommands = submitCommands;
        result.completeBorrowedCommands.emplace(completeBorrowedCommands);
        result.abandonCommands.emplace(abandonCommands);
        result.recordBarriers.emplace(recordBarriers);
        result.recordBufferCopy.emplace(recordBufferCopy);
        result.recordImageCopy.emplace(recordImageCopy);
        result.endRendering.emplace(endRendering);
        result.clearColor.emplace(clearColor);
        result.clearDepthStencil.emplace(clearDepthStencil);
        result.createImageView.emplace(createImageView);
        result.destroyImageView.emplace(destroyImageView);
        result.getImageViewNativeHandle.emplace(getImageViewNativeHandle);
        result.imageViewResource.emplace(imageViewResource);
        result.describeImageViewResource.emplace(describeImageViewResource);
        return result;
    }();
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
