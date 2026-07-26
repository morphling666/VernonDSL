#include "VernonRHI.h"

#include "opengl_backend.h"
#include "rhi_internal.h"
#if defined(VERNON_HAS_CUDA_RHI)
#include "cuda_backend.h"
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
#include "directx12_backend.h"
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
#include "vulkan_backend.h"
#endif

#include <algorithm>
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

using vernon::rhi::opengl::DeviceState;
using vernon::rhi::opengl::Enum;
using vernon::rhi::opengl::Image;
using vernon::rhi::opengl::Int;
using vernon::rhi::opengl::Size;

struct FormatInfo {
    Int internal{};
    Enum external{};
    Enum allocationType{};
};

struct ImageSlot {
    Image image;
    VernonRhiImageDescriptor descriptor{};
    Enum target{};
    uint32_t generation{1};
    bool occupied{};
};

struct BufferSlot {
    vernon::rhi::opengl::Buffer buffer;
    VernonRhiBufferDescriptor descriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct SamplerSlot {
    vernon::rhi::opengl::Sampler sampler;
    uint32_t generation{1};
    bool occupied{};
};

struct OpenGLDevice {
    DeviceState state;
    std::vector<BufferSlot> buffers;
    std::vector<ImageSlot> images;
    std::vector<SamplerSlot> samplers;
    std::string error;
    std::mutex mutex;
};

struct DeviceSlot {
    std::shared_ptr<OpenGLDevice> device;
    uint32_t generation{1};
};

#if defined(VERNON_HAS_CUDA_RHI)
struct CudaBufferSlot {
    vernon::rhi::cuda::DevicePointer pointer{};
    VernonRhiBufferDescriptor descriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct CudaDevice {
    vernon::rhi::cuda::DeviceState state;
    std::vector<CudaBufferSlot> buffers;
    std::string error;
    std::mutex mutex;
};

struct CudaDeviceSlot {
    std::shared_ptr<CudaDevice> device;
    uint32_t generation{1};
};

constexpr uint32_t cudaDeviceBit = uint32_t{1} << 29;
std::vector<CudaDeviceSlot> cudaDevices;
#endif

#if defined(VERNON_HAS_DIRECTX12_RHI)
struct DirectX12BufferSlot {
    vernon::rhi::directx12::Buffer buffer;
    VernonRhiDirectX12BorrowedBufferDescriptor descriptor{};
    VernonRhiBufferDescriptor ownedDescriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct DirectX12ImageSlot {
    vernon::rhi::directx12::Image image;
    VernonRhiDirectX12BorrowedImageDescriptor descriptor{};
    VernonRhiImageDescriptor ownedDescriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct DirectX12DescriptorRangeSlot {
    VernonRhiDirectX12BorrowedDescriptorRangeDescriptor descriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct DirectX12SamplerSlot {
    vernon::rhi::directx12::Sampler sampler;
    uint32_t generation{1};
    bool occupied{};
};

struct DirectX12InteropDevice {
    vernon::rhi::directx12::DeviceState state;
    std::deque<DirectX12BufferSlot> buffers;
    std::deque<DirectX12ImageSlot> images;
    std::deque<DirectX12SamplerSlot> samplers;
    std::deque<DirectX12DescriptorRangeSlot> descriptorRanges;
    uint32_t queueCapabilities{};
    bool owned{};
    std::string error;
    std::mutex mutex;
};

struct DirectX12DeviceSlot {
    std::shared_ptr<DirectX12InteropDevice> device;
    uint32_t generation{1};
};

constexpr uint32_t directX12DeviceBit = uint32_t{1} << 31;
std::vector<DirectX12DeviceSlot> directX12Devices;
#endif

#if defined(VERNON_HAS_VULKAN_RHI)
struct VulkanBufferSlot {
    vernon::rhi::vulkan::Buffer buffer;
    VernonRhiVulkanBorrowedBufferDescriptor descriptor{};
    VernonRhiBufferDescriptor ownedDescriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct VulkanImageSlot {
    vernon::rhi::vulkan::Image image;
    VernonRhiVulkanBorrowedImageDescriptor descriptor{};
    VernonRhiImageDescriptor ownedDescriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct VulkanImageViewSlot {
    VkImageView view{};
    VernonRhiVulkanBorrowedImageViewDescriptor descriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct VulkanSamplerSlot {
    vernon::rhi::vulkan::Sampler sampler;
    uint32_t generation{1};
    bool occupied{};
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
#endif

std::mutex deviceMutex;
std::vector<DeviceSlot> devices;

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

std::optional<size_t> rgba8Size(const VernonRhiImageDescriptor &descriptor) {
    if (descriptor.dimension != VERNON_RHI_IMAGE_2D || descriptor.format != VERNON_RHI_FORMAT_RGBA8_UNORM ||
        descriptor.depth != 1 || descriptor.mip_levels != 1 || descriptor.array_layers != 1 ||
        descriptor.width > (std::numeric_limits<size_t>::max)() / descriptor.height)
        return std::nullopt;
    const size_t pixels = static_cast<size_t>(descriptor.width) * descriptor.height;
    if (pixels > (std::numeric_limits<size_t>::max)() / 4)
        return std::nullopt;
    return pixels * 4;
}

std::shared_ptr<OpenGLDevice> lookupDevice(VernonRhiDevice handle) {
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (handle.index >= devices.size())
        return {};
    DeviceSlot &slot = devices[handle.index];
    return slot.device && slot.generation == handle.generation ? slot.device : std::shared_ptr<OpenGLDevice>{};
}

#if defined(VERNON_HAS_CUDA_RHI)
std::shared_ptr<CudaDevice> lookupCudaDevice(VernonRhiDevice handle) {
    if ((handle.index & cudaDeviceBit) == 0)
        return {};
    const uint32_t index = handle.index & ~cudaDeviceBit;
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (index >= cudaDevices.size())
        return {};
    CudaDeviceSlot &slot = cudaDevices[index];
    return slot.device && slot.generation == handle.generation ? slot.device : std::shared_ptr<CudaDevice>{};
}

CudaBufferSlot *lookupCudaBuffer(CudaDevice &device, VernonRhiBuffer handle) {
    if (handle.index >= device.buffers.size())
        return nullptr;
    CudaBufferSlot &slot = device.buffers[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot : nullptr;
}
#endif

#if defined(VERNON_HAS_DIRECTX12_RHI)
std::shared_ptr<DirectX12InteropDevice> lookupDirectX12Device(VernonRhiDevice handle) {
    if ((handle.index & directX12DeviceBit) == 0)
        return {};
    const uint32_t index = handle.index & ~directX12DeviceBit;
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (index >= directX12Devices.size())
        return {};
    DirectX12DeviceSlot &slot = directX12Devices[index];
    return slot.device && slot.generation == handle.generation ? slot.device
                                                               : std::shared_ptr<DirectX12InteropDevice>{};
}

template <typename Slots, typename Handle>
typename Slots::value_type *lookupDirectX12Slot(Slots &slots, Handle handle) {
    if (handle.index >= slots.size())
        return nullptr;
    auto &slot = slots[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot : nullptr;
}

template <typename Slots, typename Handle>
typename Slots::value_type &allocateDirectX12Slot(Slots &slots, Handle &output) {
    uint32_t index = 0;
    while (index < slots.size() && slots[index].occupied)
        ++index;
    if (index == slots.size())
        slots.emplace_back();
    auto &slot = slots[index];
    slot.occupied = true;
    output = {index, slot.generation};
    return slot;
}

template <typename Slot> void releaseDirectX12Slot(Slot &slot) {
    slot.occupied = false;
    ++slot.generation;
    if (slot.generation == 0)
        slot.generation = 1;
}

D3D12_RESOURCE_STATES directX12ResourceState(VernonRhiResourceState state) {
    switch (state) {
    case VERNON_RHI_STATE_COMMON:
        return D3D12_RESOURCE_STATE_COMMON;
    case VERNON_RHI_STATE_TRANSFER_SOURCE:
        return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case VERNON_RHI_STATE_TRANSFER_DESTINATION:
        return D3D12_RESOURCE_STATE_COPY_DEST;
    case VERNON_RHI_STATE_SHADER_READ:
        return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    case VERNON_RHI_STATE_SHADER_WRITE:
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case VERNON_RHI_STATE_COLOR_ATTACHMENT:
        return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT:
        return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case VERNON_RHI_STATE_PRESENT:
        return D3D12_RESOURCE_STATE_PRESENT;
    case VERNON_RHI_STATE_UNDEFINED:
        return D3D12_RESOURCE_STATE_COMMON;
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

DXGI_FORMAT directX12Format(VernonRhiFormat format) {
    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_UNKNOWN,
        DXGI_FORMAT_R8_UNORM,
        DXGI_FORMAT_R8G8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        DXGI_FORMAT_R16_FLOAT,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_UNKNOWN,
        DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R32G32B32_FLOAT,
        DXGI_FORMAT_R11G11B10_FLOAT,
    };
    const uint32_t index = static_cast<uint32_t>(format);
    return index < sizeof(formats) / sizeof(formats[0]) ? formats[index] : DXGI_FORMAT_UNKNOWN;
}

template <typename Object> bool belongsToDirectX12Device(Object *object, ID3D12Device *expected) {
    ID3D12Device *actual = nullptr;
    if (FAILED(object->GetDevice(IID_PPV_ARGS(&actual))))
        return false;
    const bool matches = actual == expected;
    actual->Release();
    return matches;
}
#endif

#if defined(VERNON_HAS_VULKAN_RHI)
std::shared_ptr<VulkanInteropDevice> lookupVulkanDevice(VernonRhiDevice handle) {
    if ((handle.index & vulkanDeviceBit) == 0)
        return {};
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if ((handle.index & directX12DeviceBit) != 0)
        return {};
#endif
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
    return slot.occupied && slot.generation == handle.generation ? &slot : nullptr;
}

template <typename Slots, typename Handle>
typename Slots::value_type &allocateVulkanSlot(Slots &slots, Handle &output) {
    uint32_t index = 0;
    while (index < slots.size() && slots[index].occupied)
        ++index;
    if (index == slots.size())
        slots.emplace_back();
    auto &slot = slots[index];
    slot.occupied = true;
    output = {index, slot.generation};
    return slot;
}

template <typename Slot> void releaseVulkanSlot(Slot &slot) {
    slot.occupied = false;
    ++slot.generation;
    if (slot.generation == 0)
        slot.generation = 1;
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

VkFormat vulkanFormat(VernonRhiFormat format) {
    constexpr VkFormat formats[] = {
        VK_FORMAT_UNDEFINED,           VK_FORMAT_R8_UNORM,
        VK_FORMAT_R8G8_UNORM,          VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_SRGB,       VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_R8G8B8_UNORM,        VK_FORMAT_R32G32_SFLOAT,
        VK_FORMAT_R32G32B32_SFLOAT,    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
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

void transitionVulkanImage(VkCommandBuffer command, VulkanImageSlot &slot, VkImageLayout target) {
    auto &image = slot.image;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = image.layout;
    barrier.newLayout = target;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (image.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    } else if (image.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (image.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (image.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (image.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (target == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (target == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (target == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        destinationStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    vernon::rhi::vulkan::driver().cmdPipelineBarrier(command, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr,
                                                     1, &barrier);
    image.layout = target;
}

void validate(const VulkanInteropDevice &device) {
#ifndef NDEBUG
    for (const VulkanImageViewSlot &view : device.imageViews) {
        assert(view.generation != 0);
        if (!view.occupied)
            continue;
        assert(view.descriptor.descriptor.image.index < device.images.size());
        const VulkanImageSlot &image = device.images[view.descriptor.descriptor.image.index];
        assert(image.occupied && image.generation == view.descriptor.descriptor.image.generation);
    }
#else
    (void)device;
#endif
}
#endif

ImageSlot *lookupImage(OpenGLDevice &device, VernonRhiImage handle) {
    if (handle.index >= device.images.size())
        return nullptr;
    ImageSlot &slot = device.images[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot : nullptr;
}

BufferSlot *lookupBuffer(OpenGLDevice &device, VernonRhiBuffer handle) {
    if (handle.index >= device.buffers.size())
        return nullptr;
    BufferSlot &slot = device.buffers[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot : nullptr;
}

SamplerSlot *lookupSampler(OpenGLDevice &device, VernonRhiSampler handle) {
    if (handle.index >= device.samplers.size())
        return nullptr;
    SamplerSlot &slot = device.samplers[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot : nullptr;
}

void validate(const OpenGLDevice &device) {
#ifndef NDEBUG
    for (const BufferSlot &slot : device.buffers) {
        assert(slot.generation != 0);
        assert(slot.occupied == (slot.buffer.name != 0));
    }
    for (const ImageSlot &slot : device.images) {
        assert(slot.generation != 0);
        assert(slot.occupied == (slot.image.name != 0));
    }
    for (const SamplerSlot &slot : device.samplers) {
        assert(slot.generation != 0);
        assert(slot.occupied == (slot.sampler.name != 0));
    }
#else
    (void)device;
#endif
}

bool formatInfo(VernonRhiFormat format, FormatInfo &result) {
    constexpr Enum red = 0x1903;
    constexpr Enum rg = 0x8227;
    constexpr Enum rgb = 0x1907;
    constexpr Enum rgba = 0x1908;
    constexpr Enum depth = 0x1902;
    constexpr Enum unsignedByte = 0x1401;
    constexpr Enum floating = 0x1406;
    switch (format) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        result = {0x8229, red, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RG8_UNORM:
        result = {0x822B, rg, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RGB8_UNORM:
        result = {0x8051, rgb, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
        result = {0x8058, rgba, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RGBA8_SRGB:
        result = {0x8C43, rgba, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_R16_FLOAT:
        result = {0x822D, red, floating};
        return true;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
        result = {0x881A, rgba, floating};
        return true;
    case VERNON_RHI_FORMAT_R32_FLOAT:
        result = {0x822E, red, floating};
        return true;
    case VERNON_RHI_FORMAT_RG32_FLOAT:
        result = {0x8230, rg, floating};
        return true;
    case VERNON_RHI_FORMAT_RGB32_FLOAT:
        result = {0x8815, rgb, floating};
        return true;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        result = {0x8814, rgba, floating};
        return true;
    case VERNON_RHI_FORMAT_R11G11B10_FLOAT:
        result = {0x8C3A, rgb, floating};
        return true;
    case VERNON_RHI_FORMAT_D32_FLOAT:
        result = {0x8CAC, depth, floating};
        return true;
    case VERNON_RHI_FORMAT_UNDEFINED:
        return false;
    }
    return false;
}

Enum imageTarget(VernonRhiImageDimension dimension) {
    switch (dimension) {
    case VERNON_RHI_IMAGE_2D:
        return vernon::rhi::opengl::kTexture2D;
    case VERNON_RHI_IMAGE_3D:
        return vernon::rhi::opengl::kTexture3D;
    case VERNON_RHI_IMAGE_CUBE:
        return vernon::rhi::opengl::kTextureCubeMap;
    }
    return 0;
}

Int samplerFilter(uint32_t value) {
    constexpr Int values[] = {0x2600, 0x2601, 0x2700, 0x2703, 0x2701, 0x2702};
    return value < sizeof(values) / sizeof(values[0]) ? values[value] : 0;
}

Int samplerAddress(uint32_t value) {
    constexpr Int values[] = {0x2901, 0x812F, 0x8370};
    return value < sizeof(values) / sizeof(values[0]) ? values[value] : 0;
}

Enum imageDataFormat(VernonRhiImageDataFormat format) {
    constexpr Enum values[] = {0x1903, 0x8227, 0x1907, 0x80E0, 0x1908, 0x80E1, 0x1902, 0x84F9};
    return static_cast<uint32_t>(format) < sizeof(values) / sizeof(values[0]) ? values[static_cast<uint32_t>(format)]
                                                                              : 0;
}

VernonRhiStatus fail(OpenGLDevice &device, std::string message,
                     VernonRhiStatus status = VERNON_RHI_STATUS_INVALID_ARGUMENT) {
    device.error = std::move(message);
    return status;
}

} // namespace

VernonRhiDevice vernon::rhi::createDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return invalidDevice();
    if (descriptor->backend == VERNON_RHI_BACKEND_OPENGL || descriptor->backend == VERNON_RHI_BACKEND_OPENGL_ES)
        return vernonRhiCreateOpenGLDevice(descriptor->opengl_callbacks,
                                           descriptor->backend == VERNON_RHI_BACKEND_OPENGL_ES);
#if defined(VERNON_HAS_CUDA_RHI)
    if (descriptor->backend == VERNON_RHI_BACKEND_CUDA) {
        auto device = std::shared_ptr<CudaDevice>(new (std::nothrow) CudaDevice());
        if (!device)
            return invalidDevice();
        const auto status = device->state.initialize(descriptor->device_index);
        if (status != vernon::rhi::cuda::kSuccess) {
            device->error = vernon::rhi::cuda::describeResult(status, "cuDevicePrimaryCtxRetain");
            return invalidDevice();
        }
        std::lock_guard<std::mutex> guard(deviceMutex);
        uint32_t index = 0;
        while (index < cudaDevices.size() && cudaDevices[index].device)
            ++index;
        if (index == cudaDevices.size())
            cudaDevices.emplace_back();
        cudaDevices[index].device = std::move(device);
        return {index | cudaDeviceBit, cudaDevices[index].generation};
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (descriptor->backend == VERNON_RHI_BACKEND_DIRECTX12) {
        auto device = std::shared_ptr<DirectX12InteropDevice>(new (std::nothrow) DirectX12InteropDevice());
        if (!device || !device->state.initialize(descriptor->device_index, false, device->error))
            return invalidDevice();
        device->owned = true;
        device->queueCapabilities = VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
        std::lock_guard<std::mutex> guard(deviceMutex);
        uint32_t index = 0;
        while (index < directX12Devices.size() && directX12Devices[index].device)
            ++index;
        if (index == directX12Devices.size())
            directX12Devices.emplace_back();
        directX12Devices[index].device = std::move(device);
        return {index | directX12DeviceBit, directX12Devices[index].generation};
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (descriptor->backend == VERNON_RHI_BACKEND_VULKAN) {
        auto device = std::shared_ptr<VulkanInteropDevice>(new (std::nothrow) VulkanInteropDevice());
        if (!device || !device->state.initialize(descriptor->device_index, device->error))
            return invalidDevice();
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
#endif
    return invalidDevice();
}

extern "C" VernonRhiDevice vernonRhiCreateOpenGLDevice(const VernonOpenGLContextCallbacks *callbacks,
                                                       uint32_t embeddedProfile) {
    if (!callbacks)
        return invalidDevice();
    auto device = std::shared_ptr<OpenGLDevice>(new (std::nothrow) OpenGLDevice());
    if (!device || !device->state.initialize(*callbacks, embeddedProfile != 0, device->error))
        return invalidDevice();
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < devices.size() && devices[index].device)
        ++index;
    if (index == devices.size())
        devices.emplace_back();
    devices[index].device = std::move(device);
    return {index, devices[index].generation};
}

void vernon::rhi::destroyDevice(VernonRhiDevice handle) {
#if defined(VERNON_HAS_CUDA_RHI)
    if ((handle.index & cudaDeviceBit) != 0) {
        std::shared_ptr<CudaDevice> device;
        {
            const uint32_t index = handle.index & ~cudaDeviceBit;
            std::lock_guard<std::mutex> guard(deviceMutex);
            if (index >= cudaDevices.size() || cudaDevices[index].generation != handle.generation)
                return;
            CudaDeviceSlot &slot = cudaDevices[index];
            device = std::move(slot.device);
            ++slot.generation;
            if (slot.generation == 0)
                slot.generation = 1;
        }
        if (device) {
            std::lock_guard<std::mutex> guard(device->mutex);
            for (CudaBufferSlot &buffer : device->buffers)
                if (buffer.occupied)
                    device->state.free(buffer.pointer);
            device->state.shutdown();
        }
        return;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if ((handle.index & directX12DeviceBit) != 0) {
        std::shared_ptr<DirectX12InteropDevice> device;
        {
            const uint32_t index = handle.index & ~directX12DeviceBit;
            std::lock_guard<std::mutex> guard(deviceMutex);
            if (index >= directX12Devices.size() || directX12Devices[index].generation != handle.generation)
                return;
            DirectX12DeviceSlot &slot = directX12Devices[index];
            device = std::move(slot.device);
            ++slot.generation;
            if (slot.generation == 0)
                slot.generation = 1;
        }
        if (device) {
            std::lock_guard<std::mutex> guard(device->mutex);
            for (DirectX12BufferSlot &buffer : device->buffers)
                if (buffer.occupied)
                    device->state.destroyBuffer(buffer.buffer);
            for (DirectX12ImageSlot &image : device->images)
                if (image.occupied)
                    device->state.destroyImage(image.image);
            device->state.shutdown();
        }
        return;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if ((handle.index & vulkanDeviceBit) != 0) {
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
#endif
    std::shared_ptr<OpenGLDevice> device;
    {
        std::lock_guard<std::mutex> guard(deviceMutex);
        if (handle.index >= devices.size() || devices[handle.index].generation != handle.generation)
            return;
        DeviceSlot &slot = devices[handle.index];
        device = std::move(slot.device);
        ++slot.generation;
        if (slot.generation == 0)
            slot.generation = 1;
    }
    if (!device)
        return;
    std::lock_guard<std::mutex> guard(device->mutex);
    for (BufferSlot &buffer : device->buffers)
        if (buffer.occupied)
            device->state.destroyBuffer(buffer.buffer);
    for (ImageSlot &image : device->images)
        if (image.occupied)
            device->state.destroyImage(image.image);
    for (SamplerSlot &sampler : device->samplers)
        if (sampler.occupied)
            device->state.destroySampler(sampler.sampler);
}

VernonStringView vernon::rhi::deviceLastError(VernonRhiDevice handle) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle))
        return {device->error.data(), device->error.size()};
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle))
        return {device->error.data(), device->error.size()};
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle))
        return {device->error.data(), device->error.size()};
#endif
    auto device = lookupDevice(handle);
    return device ? VernonStringView{device->error.data(), device->error.size()} : VernonStringView{};
}

VernonRhiStatus vernon::rhi::synchronizeDevice(VernonRhiDevice handle) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        const auto status = device->state.synchronize();
        if (status == vernon::rhi::cuda::kSuccess)
            return VERNON_RHI_STATUS_OK;
        device->error = vernon::rhi::cuda::describeResult(status, "cuStreamSynchronize");
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return device->state.synchronize(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return device->state.synchronize(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    device->state.makeCurrent();
    device->state.driver.finish();
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateBuffer(VernonRhiDevice handle,
                                                       const VernonRhiBufferDescriptor *descriptor,
                                                       VernonRhiBuffer *output) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle)) {
        if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
            descriptor->size > (std::numeric_limits<size_t>::max)())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        uint32_t index = 0;
        while (index < device->buffers.size() && device->buffers[index].occupied)
            ++index;
        if (index == device->buffers.size())
            device->buffers.emplace_back();
        CudaBufferSlot &slot = device->buffers[index];
        const auto status = device->state.allocate(slot.pointer, static_cast<size_t>(descriptor->size));
        if (status != vernon::rhi::cuda::kSuccess) {
            device->error = vernon::rhi::cuda::describeResult(status, "cuMemAlloc");
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        slot.descriptor = *descriptor;
        slot.occupied = true;
        *output = {index, slot.generation};
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12BufferSlot &slot = allocateDirectX12Slot(device->buffers, *output);
        const bool unorderedAccess = (descriptor->usage & VERNON_RHI_BUFFER_STORAGE) != 0;
        if (!device->state.createBuffer(slot.buffer, static_cast<size_t>(descriptor->size), unorderedAccess,
                                        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, device->error)) {
            releaseDirectX12Slot(slot);
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        slot.ownedDescriptor = *descriptor;
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
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
        if (!device->state.createBuffer(slot.buffer, descriptor->size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        device->error)) {
            releaseVulkanSlot(slot);
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        slot.ownedDescriptor = *descriptor;
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    uint32_t index = 0;
    while (index < device->buffers.size() && device->buffers[index].occupied)
        ++index;
    if (index == device->buffers.size())
        device->buffers.emplace_back();
    BufferSlot &slot = device->buffers[index];
    if (!device->state.createBuffer(slot.buffer, static_cast<size_t>(descriptor->size), device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.descriptor = *descriptor;
    slot.occupied = true;
    *output = {index, slot.generation};
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceUploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset,
                                                       const void *source, uint64_t size) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle)) {
        if (!source || size > (std::numeric_limits<size_t>::max)())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer);
        if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        const auto status = device->state.upload(slot->pointer + offset, source, static_cast<size_t>(size));
        if (status == vernon::rhi::cuda::kSuccess)
            return VERNON_RHI_STATUS_OK;
        device->error = vernon::rhi::cuda::describeResult(status, "cuMemcpyHtoDAsync");
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        if (!source)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
        if (!slot || offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        ID3D12Resource *upload = nullptr;
        size_t uploadOffset = 0;
        uint8_t *mapped = nullptr;
        if (!device->state.acquireStaging(true, static_cast<size_t>(size), 256, upload, uploadOffset, mapped,
                                          device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::memcpy(mapped, source, static_cast<size_t>(size));
        if (!device->state.beginCommands(device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        vernon::rhi::directx12::transition(device->state.commands, slot->buffer.resource, slot->buffer.state,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
        device->state.commands->CopyBufferRegion(slot->buffer.resource, offset, upload, uploadOffset, size);
        vernon::rhi::directx12::transition(device->state.commands, slot->buffer.resource, slot->buffer.state,
                                           D3D12_RESOURCE_STATE_COMMON);
        return device->state.submitCommands(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        if (!source)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
        if (!slot || offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        VkBuffer staging{};
        VkDeviceSize stagingOffset{};
        uint8_t *mapped = nullptr;
        if (!device->state.acquireStaging(true, size, 16, staging, stagingOffset, mapped, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::memcpy(mapped, source, static_cast<size_t>(size));
        VkCommandBuffer command{};
        if (!device->state.beginCommands(command, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        const VkBufferCopy copy{stagingOffset, offset, size};
        auto &driver = vernon::rhi::vulkan::driver();
        driver.cmdCopyBuffer(command, staging, slot->buffer.buffer, 1, &copy);
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                                VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = slot->buffer.buffer;
        barrier.offset = offset;
        barrier.size = size;
        driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                  nullptr, 1, &barrier, 0, nullptr);
        return device->state.submitCommands(command, device->error) ? VERNON_RHI_STATUS_OK
                                                                    : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device || !source)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return fail(*device, "OpenGL RHI buffer upload range is invalid");
    return device->state.uploadBuffer(slot->buffer, static_cast<size_t>(offset), source, static_cast<size_t>(size),
                                      device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

extern "C" VernonRhiStatus vernonRhiDeviceDownloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                                         uint64_t offset, void *destination, uint64_t size) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle)) {
        if (!destination || size > (std::numeric_limits<size_t>::max)())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer);
        if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        const auto status = device->state.download(destination, slot->pointer + offset, static_cast<size_t>(size));
        if (status == vernon::rhi::cuda::kSuccess)
            return VERNON_RHI_STATUS_OK;
        device->error = vernon::rhi::cuda::describeResult(status, "cuMemcpyDtoHAsync");
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        if (!destination)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
        if (!slot || offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        ID3D12Resource *readback = nullptr;
        size_t readbackOffset = 0;
        uint8_t *mapped = nullptr;
        if (!device->state.acquireStaging(false, static_cast<size_t>(size), 256, readback, readbackOffset, mapped,
                                          device->error) ||
            !device->state.beginCommands(device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        vernon::rhi::directx12::transition(device->state.commands, slot->buffer.resource, slot->buffer.state,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
        device->state.commands->CopyBufferRegion(readback, readbackOffset, slot->buffer.resource, offset, size);
        vernon::rhi::directx12::transition(device->state.commands, slot->buffer.resource, slot->buffer.state,
                                           D3D12_RESOURCE_STATE_COMMON);
        if (!device->state.submitCommands(device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::memcpy(destination, mapped, static_cast<size_t>(size));
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        if (!destination)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
        if (!slot || offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        VkBuffer staging{};
        VkDeviceSize stagingOffset{};
        uint8_t *mapped = nullptr;
        if (!device->state.acquireStaging(false, size, 16, staging, stagingOffset, mapped, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        VkCommandBuffer command{};
        if (!device->state.beginCommands(command, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        const VkBufferCopy copy{offset, stagingOffset, size};
        vernon::rhi::vulkan::driver().cmdCopyBuffer(command, slot->buffer.buffer, staging, 1, &copy);
        if (!device->state.submitCommands(command, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::memcpy(destination, mapped, static_cast<size_t>(size));
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return fail(*device, "OpenGL RHI buffer readback range is invalid");
    return device->state.downloadBuffer(slot->buffer, static_cast<size_t>(offset), destination,
                                        static_cast<size_t>(size), device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

extern "C" uint32_t vernonRhiDeviceIsBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return lookupCudaBuffer(*device, buffer) != nullptr;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return lookupDirectX12Slot(device->buffers, buffer) != nullptr;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return lookupVulkanSlot(device->buffers, buffer) != nullptr;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupBuffer(*device, buffer) != nullptr;
}

extern "C" VernonRhiStatus
vernonRhiDeviceCreateImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor, VernonRhiImage *output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        const DXGI_FORMAT format = descriptor ? directX12Format(descriptor->format) : DXGI_FORMAT_UNKNOWN;
        if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || format == DXGI_FORMAT_UNKNOWN ||
            descriptor->width == 0 || descriptor->height == 0 || descriptor->depth == 0 ||
            descriptor->mip_levels == 0 || descriptor->sample_count != 1)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12ImageSlot &slot = allocateDirectX12Slot(device->images, *output);
        D3D12_RESOURCE_DESC native{};
        native.Dimension = descriptor->dimension == VERNON_RHI_IMAGE_3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                                                        : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        native.Width = descriptor->width;
        native.Height = descriptor->height;
        native.DepthOrArraySize = static_cast<UINT16>(
            descriptor->dimension == VERNON_RHI_IMAGE_3D ? descriptor->depth : descriptor->array_layers);
        native.MipLevels = static_cast<UINT16>(descriptor->mip_levels);
        native.Format = format;
        native.SampleDesc.Count = 1;
        native.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        if ((descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0)
            native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if ((descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0)
            native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if ((descriptor->usage & VERNON_RHI_IMAGE_STORAGE) != 0)
            native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (!device->state.createImage(slot.image, native, format, D3D12_RESOURCE_STATE_COMMON, device->error)) {
            releaseDirectX12Slot(slot);
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        slot.ownedDescriptor = *descriptor;
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        const VkFormat format = descriptor ? vulkanFormat(descriptor->format) : VK_FORMAT_UNDEFINED;
        if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || format == VK_FORMAT_UNDEFINED ||
            descriptor->width == 0 || descriptor->height == 0 || descriptor->depth == 0 ||
            descriptor->mip_levels == 0 || descriptor->sample_count != 1)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanImageSlot &slot = allocateVulkanSlot(device->images, *output);
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.flags = descriptor->dimension == VERNON_RHI_IMAGE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
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
        viewInfo.subresourceRange.aspectMask =
            descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = descriptor->mip_levels;
        viewInfo.subresourceRange.layerCount = imageInfo.arrayLayers;
        if (!device->state.createImage(slot.image, imageInfo, viewInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                       device->error)) {
            releaseVulkanSlot(slot);
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        slot.ownedDescriptor = *descriptor;
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    FormatInfo format;
    const Enum target = imageTarget(descriptor->dimension);
    const bool validCube =
        descriptor->dimension != VERNON_RHI_IMAGE_CUBE ||
        (descriptor->width == descriptor->height && descriptor->depth == 1 && descriptor->array_layers == 6);
    if (!target || !formatInfo(descriptor->format, format) || descriptor->width == 0 || descriptor->height == 0 ||
        descriptor->depth == 0 || descriptor->mip_levels == 0 || descriptor->sample_count != 1 || !validCube)
        return fail(*device, "OpenGL RHI image descriptor is invalid");
    uint32_t index = 0;
    while (index < device->images.size() && device->images[index].occupied)
        ++index;
    if (index == device->images.size()) {
        try {
            device->images.emplace_back();
        } catch (const std::bad_alloc &) {
            return fail(*device, "OpenGL RHI image slot allocation failed", VERNON_RHI_STATUS_INTERNAL_ERROR);
        }
    }
    ImageSlot &slot = device->images[index];
    device->state.makeCurrent();
    device->state.driver.genTextures(1, &slot.image.name);
    if (!slot.image.name)
        return fail(*device, "OpenGL RHI image allocation failed", VERNON_RHI_STATUS_INTERNAL_ERROR);
    if ((descriptor->usage & (VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)) != 0) {
        auto &driver = device->state.driver;
        driver.bindTexture(target, slot.image.name);
        if (descriptor->dimension == VERNON_RHI_IMAGE_3D) {
            driver.texImage3D(target, 0, format.internal, static_cast<Size>(descriptor->width),
                              static_cast<Size>(descriptor->height), static_cast<Size>(descriptor->depth), 0,
                              format.external, format.allocationType, nullptr);
        } else if (descriptor->dimension == VERNON_RHI_IMAGE_CUBE) {
            for (uint32_t face = 0; face < 6; ++face)
                driver.texImage2D(vernon::rhi::opengl::kTextureCubeMapPositiveX + face, 0, format.internal,
                                  static_cast<Size>(descriptor->width), static_cast<Size>(descriptor->height), 0,
                                  format.external, format.allocationType, nullptr);
        } else {
            driver.texImage2D(target, 0, format.internal, static_cast<Size>(descriptor->width),
                              static_cast<Size>(descriptor->height), 0, format.external, format.allocationType,
                              nullptr);
        }
        driver.bindTexture(target, 0);
    }
    slot.descriptor = *descriptor;
    slot.target = target;
    slot.occupied = true;
    validate(*device);
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceSetImageSampler(VernonRhiDevice handle, VernonRhiImage image,
                                                          const VernonRhiSamplerDescriptor *descriptor) {
    auto device = lookupDevice(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    if (!slot || !minFilter || (descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR) || !magFilter || !addressU ||
        !addressV || !addressW)
        return fail(*device, "OpenGL RHI image sampler descriptor is invalid");
    device->state.makeCurrent();
    auto &driver = device->state.driver;
    driver.bindTexture(slot->target, slot->image.name);
    driver.texParameteri(slot->target, 0x2802, addressU);
    driver.texParameteri(slot->target, 0x2803, addressV);
    if (slot->descriptor.dimension != VERNON_RHI_IMAGE_2D)
        driver.texParameteri(slot->target, 0x8072, addressW);
    driver.texParameteri(slot->target, 0x2801, minFilter);
    driver.texParameteri(slot->target, 0x2800, magFilter);
    driver.bindTexture(slot->target, 0);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceUploadImage(VernonRhiDevice handle, VernonRhiImage image,
                                                      const VernonRhiImageUploadDescriptor *uploads,
                                                      size_t uploadCount) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        if (!uploads || uploadCount != 1)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
        if (!slot || !slot->image.owned)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        const VernonRhiImageDescriptor &descriptor = slot->ownedDescriptor;
        const auto expected = rgba8Size(descriptor);
        const VernonRhiImageUploadDescriptor &upload = uploads[0];
        if (!expected || upload.struct_size < sizeof(upload) || upload.mip_level != 0 || upload.array_layer != 0 ||
            upload.width != descriptor.width || upload.height != descriptor.height || upload.depth != 1 ||
            upload.source_format != VERNON_RHI_IMAGE_DATA_RGBA || upload.source_type != VERNON_RHI_IMAGE_DATA_UINT8 ||
            !upload.data)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        auto &state = device->state;
        const D3D12_RESOURCE_DESC native = slot->image.resource->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows = 0;
        UINT64 rowBytes = 0;
        UINT64 required = 0;
        state.device->GetCopyableFootprints(&native, 0, 1, 0, &footprint, &rows, &rowBytes, &required);
        if (static_cast<size_t>(rowBytes) * rows != *expected)
            return VERNON_RHI_STATUS_UNSUPPORTED;
        ID3D12Resource *staging = nullptr;
        size_t stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!state.acquireStaging(true, static_cast<size_t>(required), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, staging,
                                  stagingOffset, mapped, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        state.device->GetCopyableFootprints(&native, 0, 1, stagingOffset, &footprint, &rows, &rowBytes, nullptr);
        for (UINT row = 0; row < rows; ++row)
            std::memcpy(mapped + (footprint.Offset - stagingOffset) + row * footprint.Footprint.RowPitch,
                        static_cast<const uint8_t *>(upload.data) + row * rowBytes, static_cast<size_t>(rowBytes));
        if (!state.beginCommands(device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        vernon::rhi::directx12::transition(state.commands, slot->image.resource, slot->image.state,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = slot->image.resource;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = staging;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = footprint;
        state.commands->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        vernon::rhi::directx12::transition(state.commands, slot->image.resource, slot->image.state,
                                           D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        return state.submitCommands(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        if (!uploads || uploadCount != 1)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
        if (!slot || !slot->image.owned)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        const VernonRhiImageDescriptor &descriptor = slot->ownedDescriptor;
        const auto expected = rgba8Size(descriptor);
        const VernonRhiImageUploadDescriptor &upload = uploads[0];
        if (!expected || upload.struct_size < sizeof(upload) || upload.mip_level != 0 || upload.array_layer != 0 ||
            upload.width != descriptor.width || upload.height != descriptor.height || upload.depth != 1 ||
            upload.source_format != VERNON_RHI_IMAGE_DATA_RGBA || upload.source_type != VERNON_RHI_IMAGE_DATA_UINT8 ||
            !upload.data)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceSize stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!device->state.acquireStaging(true, *expected, 16, staging, stagingOffset, mapped, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::memcpy(mapped, upload.data, *expected);
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (!device->state.beginCommands(command, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = stagingOffset;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {descriptor.width, descriptor.height, 1};
        vernon::rhi::vulkan::driver().cmdCopyBufferToImage(command, staging, slot->image.image,
                                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return device->state.submitCommands(command, device->error) ? VERNON_RHI_STATUS_OK
                                                                    : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device || (uploadCount != 0 && !uploads))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    device->state.makeCurrent();
    auto &driver = device->state.driver;
    driver.bindTexture(slot->target, slot->image.name);
    driver.pixelStorei(0x0CF5, 1);
    FormatInfo storageFormat;
    if (!formatInfo(slot->descriptor.format, storageFormat)) {
        driver.bindTexture(slot->target, 0);
        return fail(*device, "OpenGL RHI image storage format is invalid");
    }
    if (uploadCount == 0 && (slot->descriptor.usage &
                             (VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)) != 0) {
        if (slot->descriptor.dimension == VERNON_RHI_IMAGE_3D) {
            driver.texImage3D(slot->target, 0, storageFormat.internal, static_cast<Size>(slot->descriptor.width),
                              static_cast<Size>(slot->descriptor.height), static_cast<Size>(slot->descriptor.depth), 0,
                              storageFormat.external, storageFormat.allocationType, nullptr);
        } else if (slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE) {
            for (uint32_t face = 0; face < 6; ++face)
                driver.texImage2D(vernon::rhi::opengl::kTextureCubeMapPositiveX + face, 0, storageFormat.internal,
                                  static_cast<Size>(slot->descriptor.width), static_cast<Size>(slot->descriptor.height),
                                  0, storageFormat.external, storageFormat.allocationType, nullptr);
        } else {
            driver.texImage2D(slot->target, 0, storageFormat.internal, static_cast<Size>(slot->descriptor.width),
                              static_cast<Size>(slot->descriptor.height), 0, storageFormat.external,
                              storageFormat.allocationType, nullptr);
        }
    }
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const Enum sourceFormat = imageDataFormat(upload.source_format);
        const bool validLayer =
            slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE ? upload.array_layer < 6 : upload.array_layer == 0;
        if (upload.struct_size < sizeof(upload) || upload.mip_level >= slot->descriptor.mip_levels ||
            upload.width == 0 || upload.height == 0 || upload.depth == 0 || !validLayer || !sourceFormat ||
            upload.source_type > VERNON_RHI_IMAGE_DATA_FLOAT32) {
            driver.bindTexture(slot->target, 0);
            return fail(*device, "OpenGL RHI image upload descriptor is invalid");
        }
        const Enum type = upload.source_type == VERNON_RHI_IMAGE_DATA_UINT8 ? 0x1401 : 0x1406;
        if (slot->descriptor.dimension == VERNON_RHI_IMAGE_3D)
            driver.texImage3D(slot->target, static_cast<Int>(upload.mip_level), storageFormat.internal,
                              static_cast<Size>(upload.width), static_cast<Size>(upload.height),
                              static_cast<Size>(upload.depth), 0, sourceFormat, type, upload.data);
        else
            driver.texImage2D(slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                                  ? vernon::rhi::opengl::kTextureCubeMapPositiveX + upload.array_layer
                                  : slot->target,
                              static_cast<Int>(upload.mip_level), storageFormat.internal,
                              static_cast<Size>(upload.width), static_cast<Size>(upload.height), 0, sourceFormat, type,
                              upload.data);
    }
    driver.bindTexture(slot->target, 0);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDownloadImage(VernonRhiDevice handle, VernonRhiImage image, void *destination,
                                                        size_t size) {
    if (!destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
        if (!slot || !slot->image.owned)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        const auto expected = rgba8Size(slot->ownedDescriptor);
        if (!expected || size != *expected)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        auto &state = device->state;
        const D3D12_RESOURCE_DESC native = slot->image.resource->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows = 0;
        UINT64 rowBytes = 0;
        UINT64 required = 0;
        state.device->GetCopyableFootprints(&native, 0, 1, 0, &footprint, &rows, &rowBytes, &required);
        ID3D12Resource *staging = nullptr;
        size_t stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!state.acquireStaging(false, static_cast<size_t>(required), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, staging,
                                  stagingOffset, mapped, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        state.device->GetCopyableFootprints(&native, 0, 1, stagingOffset, &footprint, &rows, &rowBytes, nullptr);
        if (!state.beginCommands(device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        vernon::rhi::directx12::transition(state.commands, slot->image.resource, slot->image.state,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION target{};
        target.pResource = staging;
        target.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        target.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = slot->image.resource;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        state.commands->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
        vernon::rhi::directx12::transition(state.commands, slot->image.resource, slot->image.state,
                                           D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        if (!state.submitCommands(device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        for (UINT row = 0; row < rows; ++row)
            std::memcpy(static_cast<uint8_t *>(destination) + row * rowBytes,
                        mapped + (footprint.Offset - stagingOffset) + row * footprint.Footprint.RowPitch,
                        static_cast<size_t>(rowBytes));
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
        if (!slot || !slot->image.owned)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        const auto expected = rgba8Size(slot->ownedDescriptor);
        if (!expected || size != *expected)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceSize stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!device->state.acquireStaging(false, size, 16, staging, stagingOffset, mapped, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        const VkImageLayout restore = slot->image.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                          ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                          : slot->image.layout;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (!device->state.beginCommands(command, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        transitionVulkanImage(command, *slot, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = stagingOffset;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {slot->ownedDescriptor.width, slot->ownedDescriptor.height, 1};
        vernon::rhi::vulkan::driver().cmdCopyImageToBuffer(command, slot->image.image,
                                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);
        transitionVulkanImage(command, *slot, restore);
        if (!device->state.submitCommands(command, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        std::memcpy(destination, mapped, size);
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    const auto expected = slot ? rgba8Size(slot->descriptor) : std::nullopt;
    FormatInfo format;
    if (!slot || !expected || size != *expected || !formatInfo(slot->descriptor.format, format))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device->state.downloadImage2D(slot->image, static_cast<Size>(slot->descriptor.width),
                                         static_cast<Size>(slot->descriptor.height), format.external,
                                         format.allocationType, destination, device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

extern "C" VernonRhiStatus vernonRhiDeviceGenerateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    device->state.makeCurrent();
    device->state.driver.bindTexture(slot->target, slot->image.name);
    device->state.driver.generateMipmap(slot->target);
    device->state.driver.bindTexture(slot->target, 0);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceBindImage(VernonRhiDevice handle, VernonRhiImage image,
                                                    uint32_t textureUnit) {
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    device->state.makeCurrent();
    device->state.driver.activeTexture(vernon::rhi::opengl::kTexture0 + textureUnit);
    device->state.driver.bindTexture(slot->target, slot->image.name);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyImage(VernonRhiDevice handle, VernonRhiImage image) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        device->state.destroyImage(slot->image);
        releaseDirectX12Slot(*slot);
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        if (std::any_of(device->imageViews.begin(), device->imageViews.end(), [image](const VulkanImageViewSlot &view) {
                return view.occupied && view.descriptor.descriptor.image.index == image.index &&
                       view.descriptor.descriptor.image.generation == image.generation;
            }))
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        device->state.destroyImage(slot->image);
        releaseVulkanSlot(*slot);
        validate(*device);
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    device->state.destroyImage(slot->image);
    slot->occupied = false;
    ++slot->generation;
    if (slot->generation == 0)
        slot->generation = 1;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" uint32_t vernonRhiDeviceIsImageValid(VernonRhiDevice handle, VernonRhiImage image) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return lookupDirectX12Slot(device->images, image) != nullptr;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return lookupVulkanSlot(device->images, image) != nullptr;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupImage(*device, image) != nullptr;
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateSampler(VernonRhiDevice handle,
                                                        const VernonRhiSamplerDescriptor *descriptor,
                                                        VernonRhiSampler *output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
            descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR ||
            descriptor->address_u > VERNON_RHI_ADDRESS_MIRRORED_REPEAT ||
            descriptor->address_v > VERNON_RHI_ADDRESS_MIRRORED_REPEAT ||
            descriptor->address_w > VERNON_RHI_ADDRESS_MIRRORED_REPEAT)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12SamplerSlot &slot = allocateDirectX12Slot(device->samplers, *output);
        auto address = [](uint32_t mode) {
            constexpr D3D12_TEXTURE_ADDRESS_MODE values[] = {
                D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
            return values[mode];
        };
        slot.sampler.descriptor.Filter = descriptor->mag_filter == VERNON_RHI_FILTER_NEAREST
                                             ? D3D12_FILTER_MIN_MAG_MIP_POINT
                                             : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        slot.sampler.descriptor.AddressU = address(descriptor->address_u);
        slot.sampler.descriptor.AddressV = address(descriptor->address_v);
        slot.sampler.descriptor.AddressW = address(descriptor->address_w);
        slot.sampler.descriptor.MaxAnisotropy = 1;
        slot.sampler.descriptor.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        slot.sampler.descriptor.MinLOD = 0;
        slot.sampler.descriptor.MaxLOD = D3D12_FLOAT32_MAX;
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
            descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR ||
            descriptor->address_u > VERNON_RHI_ADDRESS_MIRRORED_REPEAT ||
            descriptor->address_v > VERNON_RHI_ADDRESS_MIRRORED_REPEAT ||
            descriptor->address_w > VERNON_RHI_ADDRESS_MIRRORED_REPEAT)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanSamplerSlot &slot = allocateVulkanSlot(device->samplers, *output);
        auto address = [](uint32_t mode) {
            constexpr VkSamplerAddressMode values[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                       VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                       VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
            return values[mode];
        };
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.minFilter = descriptor->min_filter == VERNON_RHI_FILTER_NEAREST ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        info.magFilter = descriptor->mag_filter == VERNON_RHI_FILTER_NEAREST ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        info.mipmapMode = descriptor->mip_filter == VERNON_RHI_FILTER_NEAREST ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                                                                              : VK_SAMPLER_MIPMAP_MODE_LINEAR;
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
#endif
    auto device = lookupDevice(handle);
    if (!device || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    if (!minFilter || descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR || !magFilter || !addressU || !addressV ||
        !addressW)
        return fail(*device, "OpenGL RHI sampler descriptor is invalid");
    std::lock_guard<std::mutex> guard(device->mutex);
    uint32_t index = 0;
    while (index < device->samplers.size() && device->samplers[index].occupied)
        ++index;
    if (index == device->samplers.size())
        device->samplers.emplace_back();
    SamplerSlot &slot = device->samplers[index];
    if (!device->state.createSampler(slot.sampler, addressU, addressV, addressW, minFilter, magFilter, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.occupied = true;
    *output = {index, slot.generation};
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12SamplerSlot *slot = lookupDirectX12Slot(device->samplers, sampler);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        releaseDirectX12Slot(*slot);
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanSamplerSlot *slot = lookupVulkanSlot(device->samplers, sampler);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        device->state.destroySampler(slot->sampler);
        releaseVulkanSlot(*slot);
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    SamplerSlot *slot = lookupSampler(*device, sampler);
    if (!slot)
        return fail(*device, "OpenGL RHI sampler handle is stale");
    device->state.destroySampler(slot->sampler);
    slot->occupied = false;
    ++slot->generation;
    if (slot->generation == 0)
        slot->generation = 1;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" uint32_t vernonRhiDeviceIsSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return lookupDirectX12Slot(device->samplers, sampler) != nullptr;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        return lookupVulkanSlot(device->samplers, sampler) != nullptr;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupSampler(*device, sampler) != nullptr;
}

extern "C" VernonRhiStatus vernonRhiDeviceGetImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image,
                                                               uint64_t *output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        if (!output)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        *output = reinterpret_cast<uint64_t>(slot->image.resource);
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        if (!output)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanImageSlot *slot = lookupVulkanSlot(device->images, image);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        *output = vulkanHandleBits(slot->image.image);
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    *output = slot->image.name;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiDevice
vernonRhiCreateBorrowedDirectX12Device(const VernonRhiDirectX12BorrowedDeviceDescriptor *descriptor) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->device || !descriptor->queue ||
        !descriptor->command_list ||
        (descriptor->queue_capabilities &
         ~(VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS)) != 0)
        return invalidDevice();
    auto *queue = static_cast<ID3D12CommandQueue *>(descriptor->queue);
    auto *commands = static_cast<ID3D12GraphicsCommandList *>(descriptor->command_list);
    auto *nativeDevice = static_cast<ID3D12Device *>(descriptor->device);
    if (queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT ||
        commands->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT || !belongsToDirectX12Device(queue, nativeDevice) ||
        !belongsToDirectX12Device(commands, nativeDevice))
        return invalidDevice();
    auto device = std::shared_ptr<DirectX12InteropDevice>(new (std::nothrow) DirectX12InteropDevice());
    if (!device || !device->state.initializeBorrowed(nativeDevice, queue, commands, device->error))
        return invalidDevice();
    device->queueCapabilities = descriptor->queue_capabilities;
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < directX12Devices.size() && directX12Devices[index].device)
        ++index;
    if (index == directX12Devices.size()) {
        if (index >= directX12DeviceBit)
            return invalidDevice();
        directX12Devices.emplace_back();
    }
    directX12Devices[index].device = std::move(device);
    return {index | directX12DeviceBit, directX12Devices[index].generation};
#else
    (void)descriptor;
    return invalidDevice();
#endif
}

extern "C" VernonRhiStatus vernonRhiDirectX12DeviceGetBorrowedQueue(VernonRhiDevice handle, void **output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    auto device = lookupDirectX12Device(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.queue;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDirectX12DeviceGetBorrowedCommandList(VernonRhiDevice handle, void **output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    auto device = lookupDirectX12Device(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.commands;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedBuffer(
    VernonRhiDevice handle, const VernonRhiDirectX12BorrowedBufferDescriptor *descriptor, VernonRhiBuffer *output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->resource ||
        !descriptor->size || descriptor->state == VERNON_RHI_STATE_UNDEFINED || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto *resource = static_cast<ID3D12Resource *>(descriptor->resource);
    if (!belongsToDirectX12Device(resource, device->state.device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const D3D12_RESOURCE_DESC native = resource->GetDesc();
    if (native.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || native.Width < descriptor->size)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot &slot = allocateDirectX12Slot(device->buffers, *output);
    slot.buffer.resource = resource;
    slot.buffer.state = directX12ResourceState(descriptor->state);
    slot.buffer.owned = false;
    slot.descriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)descriptor;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedImage(
    VernonRhiDevice handle, const VernonRhiDirectX12BorrowedImageDescriptor *descriptor, VernonRhiImage *output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->resource ||
        descriptor->image.struct_size < sizeof(descriptor->image) || descriptor->state == VERNON_RHI_STATE_UNDEFINED ||
        !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto *resource = static_cast<ID3D12Resource *>(descriptor->resource);
    if (!belongsToDirectX12Device(resource, device->state.device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const D3D12_RESOURCE_DESC native = resource->GetDesc();
    const DXGI_FORMAT format = directX12Format(descriptor->image.format);
    const D3D12_RESOURCE_DIMENSION dimension = descriptor->image.dimension == VERNON_RHI_IMAGE_3D
                                                   ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                                   : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    const uint32_t expectedDepthOrLayers =
        descriptor->image.dimension == VERNON_RHI_IMAGE_3D ? descriptor->image.depth : descriptor->image.array_layers;
    const bool validCube = descriptor->image.dimension != VERNON_RHI_IMAGE_CUBE ||
                           (descriptor->image.width == descriptor->image.height &&
                            descriptor->image.array_layers == 6 && descriptor->image.depth == 1);
    if (format == DXGI_FORMAT_UNKNOWN || native.Dimension != dimension || native.Format != format ||
        native.Width != descriptor->image.width || native.Height != descriptor->image.height ||
        native.DepthOrArraySize != expectedDepthOrLayers || native.MipLevels != descriptor->image.mip_levels ||
        native.SampleDesc.Count != descriptor->image.sample_count || !validCube)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot &slot = allocateDirectX12Slot(device->images, *output);
    slot.image.resource = resource;
    slot.image.format = format;
    slot.image.state = directX12ResourceState(descriptor->state);
    slot.image.owned = false;
    slot.descriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)descriptor;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedDescriptorRange(
    VernonRhiDevice handle, const VernonRhiDirectX12BorrowedDescriptorRangeDescriptor *descriptor,
    VernonRhiNativeDescriptorRange *output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->heap ||
        !descriptor->cpu_handle || !descriptor->descriptor_count ||
        descriptor->heap_type > VERNON_RHI_NATIVE_DESCRIPTOR_DEPTH_STENCIL || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto *heap = static_cast<ID3D12DescriptorHeap *>(descriptor->heap);
    if (!belongsToDirectX12Device(heap, device->state.device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const D3D12_DESCRIPTOR_HEAP_DESC native = heap->GetDesc();
    const D3D12_DESCRIPTOR_HEAP_TYPE expected[] = {
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    };
    if (native.Type != expected[descriptor->heap_type] || descriptor->descriptor_count > native.NumDescriptors)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const bool shaderVisible = (native.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) != 0;
    if (shaderVisible != (descriptor->gpu_handle != 0))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t increment = device->state.device->GetDescriptorHandleIncrementSize(native.Type);
    const uint64_t cpuStart = heap->GetCPUDescriptorHandleForHeapStart().ptr;
    if (descriptor->cpu_handle < cpuStart || (descriptor->cpu_handle - cpuStart) % increment != 0 ||
        (descriptor->cpu_handle - cpuStart) / increment + descriptor->descriptor_count > native.NumDescriptors)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (shaderVisible) {
        const uint64_t gpuStart = heap->GetGPUDescriptorHandleForHeapStart().ptr;
        if (descriptor->gpu_handle < gpuStart || (descriptor->gpu_handle - gpuStart) % increment != 0 ||
            (descriptor->gpu_handle - gpuStart) / increment + descriptor->descriptor_count > native.NumDescriptors)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12DescriptorRangeSlot &slot = allocateDirectX12Slot(device->descriptorRanges, *output);
    slot.descriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)descriptor;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        const auto status = device->state.free(slot->pointer);
        if (status != vernon::rhi::cuda::kSuccess) {
            device->error = vernon::rhi::cuda::describeResult(status, "cuMemFree");
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        slot->pointer = 0;
        slot->occupied = false;
        ++slot->generation;
        if (slot->generation == 0)
            slot->generation = 1;
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        device->state.destroyBuffer(slot->buffer);
        releaseDirectX12Slot(*slot);
        return VERNON_RHI_STATUS_OK;
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        device->state.destroyBuffer(slot->buffer);
        releaseVulkanSlot(*slot);
        return VERNON_RHI_STATUS_OK;
    }
#endif
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot)
        return fail(*device, "OpenGL RHI buffer handle is stale");
    device->state.destroyBuffer(slot->buffer);
    slot->occupied = false;
    ++slot->generation;
    if (slot->generation == 0)
        slot->generation = 1;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyNativeDescriptorRange(VernonRhiDevice handle,
                                                                       VernonRhiNativeDescriptorRange range) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12DescriptorRangeSlot *slot = lookupDirectX12Slot(device->descriptorRanges, range);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    releaseDirectX12Slot(*slot);
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)range;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDeviceGetBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                                                void **output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        if (!output)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> guard(device->mutex);
        DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
        if (!slot)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        *output = slot->buffer.resource;
        return VERNON_RHI_STATUS_OK;
    }
#endif
    (void)buffer;
    (void)output;
    return lookupDevice(handle) ? VERNON_RHI_STATUS_UNSUPPORTED : VERNON_RHI_STATUS_INVALID_ARGUMENT;
}

extern "C" VernonRhiStatus
vernonRhiDeviceGetNativeDescriptorRange(VernonRhiDevice handle, VernonRhiNativeDescriptorRange range,
                                        VernonRhiDirectX12BorrowedDescriptorRangeDescriptor *output) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    auto device = lookupDirectX12Device(handle);
    if (!device || !output || output->struct_size < sizeof(*output))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12DescriptorRangeSlot *slot = lookupDirectX12Slot(device->descriptorRanges, range);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = slot->descriptor;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)range;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

void *vernon::rhi::deviceState(VernonRhiDevice handle, VernonRhiBackend backend) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (backend == VERNON_RHI_BACKEND_CUDA)
        if (auto device = lookupCudaDevice(handle))
            return &device->state;
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (backend == VERNON_RHI_BACKEND_DIRECTX12)
        if (auto device = lookupDirectX12Device(handle))
            return &device->state;
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (backend == VERNON_RHI_BACKEND_VULKAN)
        if (auto device = lookupVulkanDevice(handle))
            return &device->state;
#endif
    if (backend == VERNON_RHI_BACKEND_OPENGL || backend == VERNON_RHI_BACKEND_OPENGL_ES)
        if (auto device = lookupDevice(handle))
            return &device->state;
    return nullptr;
}

uint64_t vernon::rhi::bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {
#if defined(VERNON_HAS_CUDA_RHI)
    if (auto device = lookupCudaDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer))
            return slot->pointer;
    }
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer))
            return reinterpret_cast<uintptr_t>(&slot->buffer);
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer))
            return reinterpret_cast<uintptr_t>(&slot->buffer);
    }
#endif
    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (BufferSlot *slot = lookupBuffer(*device, buffer))
            return slot->buffer.name;
    }
    return 0;
}

uint64_t vernon::rhi::imageResource(VernonRhiDevice handle, VernonRhiImage image) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image))
            return reinterpret_cast<uintptr_t>(&slot->image);
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (VulkanImageSlot *slot = lookupVulkanSlot(device->images, image))
            return reinterpret_cast<uintptr_t>(&slot->image);
    }
#endif
    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (ImageSlot *slot = lookupImage(*device, image))
            return slot->image.name;
    }
    return 0;
}

uint64_t vernon::rhi::samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) {
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (auto device = lookupDirectX12Device(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (DirectX12SamplerSlot *slot = lookupDirectX12Slot(device->samplers, sampler))
            return reinterpret_cast<uintptr_t>(&slot->sampler);
    }
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (auto device = lookupVulkanDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (VulkanSamplerSlot *slot = lookupVulkanSlot(device->samplers, sampler))
            return reinterpret_cast<uintptr_t>(&slot->sampler);
    }
#endif
    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (SamplerSlot *slot = lookupSampler(*device, sampler))
            return slot->sampler.name;
    }
    return 0;
}

extern "C" VernonRhiDevice
vernonRhiCreateBorrowedVulkanDevice(const VernonRhiVulkanBorrowedDeviceDescriptor *descriptor) {
#if defined(VERNON_HAS_VULKAN_RHI)
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
#else
    (void)descriptor;
    return invalidDevice();
#endif
}

extern "C" VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedQueue(VernonRhiDevice handle, void **output) {
#if defined(VERNON_HAS_VULKAN_RHI)
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.queue;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiVulkanDeviceGetBorrowedCommandBuffer(VernonRhiDevice handle, void **output) {
#if defined(VERNON_HAS_VULKAN_RHI)
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.borrowedCommandBuffer;
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedBuffer(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedBufferDescriptor *descriptor, VernonRhiBuffer *output) {
#if defined(VERNON_HAS_VULKAN_RHI)
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
#else
    (void)handle;
    (void)descriptor;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImage(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageDescriptor *descriptor, VernonRhiImage *output) {
#if defined(VERNON_HAS_VULKAN_RHI)
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
    slot.image.colorAttachment = (descriptor->descriptor.usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0;
    slot.image.owned = false;
    slot.descriptor = *descriptor;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)descriptor;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiVulkanDeviceImportBorrowedImageView(
    VernonRhiDevice handle, const VernonRhiVulkanBorrowedImageViewDescriptor *descriptor, VernonRhiImageView *output) {
#if defined(VERNON_HAS_VULKAN_RHI)
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
    VulkanImageViewSlot &slot = allocateVulkanSlot(device->imageViews, *output);
    slot.view = vulkanHandle<VkImageView>(descriptor->image_view);
    slot.descriptor = *descriptor;
    validate(*device);
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)descriptor;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiVulkanDeviceGetBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                                                      uint64_t *output) {
#if defined(VERNON_HAS_VULKAN_RHI)
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanBufferSlot *slot = lookupVulkanSlot(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = vulkanHandleBits(slot->buffer.buffer);
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)buffer;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyImageView(VernonRhiDevice handle, VernonRhiImageView imageView) {
#if defined(VERNON_HAS_VULKAN_RHI)
    auto device = lookupVulkanDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageViewSlot *slot = lookupVulkanSlot(device->imageViews, imageView);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    releaseVulkanSlot(*slot);
    validate(*device);
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)imageView;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}

extern "C" VernonRhiStatus vernonRhiDeviceGetImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView imageView,
                                                                   uint64_t *output) {
#if defined(VERNON_HAS_VULKAN_RHI)
    auto device = lookupVulkanDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    VulkanImageViewSlot *slot = lookupVulkanSlot(device->imageViews, imageView);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = vulkanHandleBits(slot->view);
    return VERNON_RHI_STATUS_OK;
#else
    (void)handle;
    (void)imageView;
    (void)output;
    return VERNON_RHI_STATUS_UNSUPPORTED;
#endif
}
