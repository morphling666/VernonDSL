#include "hal/vulkan/vulkan_resources.h"

#include "VernonFormats.hpp"
#include "hal/native_buffers.h"
#include "hal/native_images.h"
#include "hal/resources.h"
#include "hal/vulkan/vulkan_state.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::hal {
namespace {

using vulkan::DeviceState;

Serials<vulkan::Buffer> &serials() {
    static Serials<vulkan::Buffer> instance;
    return instance;
}

template <typename T = void> Done<T> internal(const char *what, const std::string &detail) {
    return fail<T>(VERNON_HAL_INTERNAL_ERROR, detail.empty() ? std::string(what) : std::string(what) + ": " + detail);
}

VkFormat vulkanFormat(VernonTextureFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,       \
                      vulkan, ...)                                                                                     \
    case VERNON_TEXTURE_##NAME:                                                                                        \
        return vulkan;
#include "VernonFormats.def"
    }
    return VK_FORMAT_UNDEFINED;
}

// A shader names any other storage format through StorageImageExtendedFormats, which needs the device feature.
bool coreStorageFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return true;
    default:
        return false;
    }
}

VkImageViewType viewType(VernonTextureDimension dimension) {
    switch (dimension) {
    case VERNON_TEXTURE_3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case VERNON_TEXTURE_CUBE:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case VERNON_TEXTURE_2D_ARRAY:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case VERNON_TEXTURE_CUBE_ARRAY:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case VERNON_TEXTURE_2D:
        break;
    }
    return VK_IMAGE_VIEW_TYPE_2D;
}

VkImageAspectFlags copyAspect(uint32_t aspect) {
    return (aspect & VERNON_IMAGE_ASPECT_COLOR)   ? VK_IMAGE_ASPECT_COLOR_BIT
           : (aspect & VERNON_IMAGE_ASPECT_DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                  : VK_IMAGE_ASPECT_STENCIL_BIT;
}

// The layout an image rests in between operations, which the passes that use it transition from.
VkImageLayout restingLayout(uint32_t usage) {
    if (usage & VERNON_IMAGE_STORAGE)
        return VK_IMAGE_LAYOUT_GENERAL;
    if (usage & VERNON_IMAGE_SAMPLED)
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT)
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    if (usage & VERNON_IMAGE_COLOR_ATTACHMENT)
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (usage & VERNON_IMAGE_TRANSFER_SOURCE)
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
}

void imageDependency(VkImageLayout layout, VkPipelineStageFlags &stages, VkAccessFlags &access) {
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

vulkan::ImageSubresources everything(const vulkan::Image &image) {
    const uint32_t layers =
        image.mipLevels ? static_cast<uint32_t>(image.subresourceLayouts.size() / image.mipLevels) : 0;
    return {0, image.mipLevels, 0, layers};
}

// Transitions the subresources `range` of `image`, which tracks their layouts, within recording `command`. Runs under
// the device's lock.
bool transition(DeviceState &state, VkCommandBuffer command, vulkan::Image &image,
                const vulkan::ImageSubresources &range, VkImageLayout target, std::string &error) {
    if (!range.mipLevelCount || !range.arrayLayerCount || state.movedToLayout(command, image, range, target))
        return true;
    // A subresource whose source is `target` needs no barrier here.
    std::vector<VkImageLayout> sources;
    sources.reserve(size_t{range.mipLevelCount} * range.arrayLayerCount);
    for (uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.arrayLayerCount; ++layer)
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.mipLevelCount; ++mip) {
            std::optional<VkImageLayout> from;
            if (!state.moveImageLayout(command, image, mip + size_t{layer} * image.mipLevels, target, from, error))
                return false;
            sources.push_back(from.value_or(target));
        }
    std::vector<VkImageMemoryBarrier> barriers;
    VkPipelineStageFlags sourceStages{};
    VkPipelineStageFlags destinationStages{};
    const auto emit = [&](VkImageLayout source, uint32_t baseMip, uint32_t mipCount, uint32_t baseLayer,
                          uint32_t layerCount) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = source;
        barrier.newLayout = target;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        barrier.subresourceRange = {vulkan::imageAspectMask(image.format), baseMip, mipCount, baseLayer, layerCount};
        VkPipelineStageFlags sourceStage{};
        VkPipelineStageFlags destinationStage{};
        imageDependency(source, sourceStage, barrier.srcAccessMask);
        imageDependency(target, destinationStage, barrier.dstAccessMask);
        sourceStages |= sourceStage;
        destinationStages |= destinationStage;
        barriers.push_back(barrier);
    };
    if (std::all_of(sources.begin(), sources.end(),
                    [first = sources.front()](VkImageLayout layout) { return layout == first; })) {
        if (sources.front() != target)
            emit(sources.front(), range.baseMipLevel, range.mipLevelCount, range.baseArrayLayer, range.arrayLayerCount);
    } else {
        for (size_t index = 0; index < sources.size(); ++index)
            if (sources[index] != target)
                emit(sources[index], range.baseMipLevel + static_cast<uint32_t>(index % range.mipLevelCount), 1,
                     range.baseArrayLayer + static_cast<uint32_t>(index / range.mipLevelCount), 1);
    }
    if (!barriers.empty())
        vulkan::driver().cmdPipelineBarrier(command, sourceStages, destinationStages, 0, 0, nullptr, 0, nullptr,
                                            static_cast<uint32_t>(barriers.size()), barriers.data());
    return true;
}

// The layout that operations outside passes leave `image` in: the one it last shared, or its resting layout before
// anything wrote it.
VkImageLayout restoredLayout(const vulkan::Image &image, uint32_t usage) {
    return image.layout == VK_IMAGE_LAYOUT_UNDEFINED ? restingLayout(usage) : image.layout;
}

// One Vulkan buffer, named by its serial.
class NativeBuffer {
public:
    NativeBuffer(std::shared_ptr<DeviceState> state, vulkan::Buffer buffer, uint64_t size)
        : state_(std::move(state)), identity_(nativeBufferIdentity(state_.get())), buffer_(buffer), size_(size) {
        serial_ = serials().add(&buffer_);
    }
    NativeBuffer(const NativeBuffer &) = delete;
    NativeBuffer &operator=(const NativeBuffer &) = delete;
    ~NativeBuffer() {
        serials().remove(serial_);
        vulkan::destroyBuffer(*state_, buffer_);
    }

    vulkan::Buffer &buffer() { return buffer_; }
    uint64_t size() const { return size_; }
    VernonRuntimeProviderResourceReference reference(uint64_t offset, uint64_t size) const {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = identity_;
        reference.resource.value = serial_;
        reference.offset = offset;
        reference.size = size;
        return reference;
    }

private:
    std::shared_ptr<DeviceState> state_;
    uint64_t identity_;
    vulkan::Buffer buffer_;
    uint64_t size_;
    uint64_t serial_{};
};

struct VulkanHalBuffer final : VernonHalBuffer {
    VulkanHalBuffer(std::shared_ptr<DeviceState> state, vulkan::Buffer buffer, uint64_t size)
        : native(std::move(state), buffer, size) {}

    NativeBuffer native;
};

// Destroys native objects once the work submitted so far completes.
void destroyLater(DeviceState &state, DeviceState::DeferredDestruction destruction) {
    std::lock_guard<std::mutex> guard(state.mutex);
    state.destroyAfterSubmittedWork(std::move(destruction));
}

struct VulkanImage final : VernonHalImage {
    explicit VulkanImage(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    VulkanImage(const VulkanImage &) = delete;
    VulkanImage &operator=(const VulkanImage &) = delete;
    ~VulkanImage() override {
        forgetUndefined(&serial, 1);
        DeviceState::DeferredDestruction destruction;
        destruction.image = native;
        destroyLater(*state, std::move(destruction));
    }

    std::shared_ptr<DeviceState> state;
    // Tracks the layouts of every subresource, for the image and all its views.
    vulkan::Image native;
    uint64_t serial{nextImageSerial()};
};

struct VulkanView final : VernonHalImageView {
    explicit VulkanView(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    VulkanView(const VulkanView &) = delete;
    VulkanView &operator=(const VulkanView &) = delete;
    ~VulkanView() override {
        if (serial)
            nativeViews().remove(serial);
        DeviceState::DeferredDestruction destruction;
        destruction.view = binding.view;
        destroyLater(*state, std::move(destruction));
    }

    std::shared_ptr<DeviceState> state;
    // What passes bind: the view over its subresources of the image, whose layouts the image tracks.
    vulkan::Image binding;
    NativeView entry;
    uint64_t serial{};
};

struct VulkanSampler final : VernonHalSampler {
    explicit VulkanSampler(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    VulkanSampler(const VulkanSampler &) = delete;
    VulkanSampler &operator=(const VulkanSampler &) = delete;
    ~VulkanSampler() override {
        if (serial)
            nativeSamplers().remove(serial);
        DeviceState::DeferredDestruction destruction;
        destruction.sampler = sampler;
        destroyLater(*state, std::move(destruction));
    }

    std::shared_ptr<DeviceState> state;
    vulkan::Sampler sampler;
    uint64_t serial{};
};

struct AcquireSemaphore {
    VkSemaphore semaphore{};
    // The submission that last waited on it; zero when none is pending.
    uint64_t serial{};
};

// A window's swapchain, and the ring of images that passes render to and presents blit to it. The images are the
// surface's own rather than the swapchain's, so a frame acquires one without waiting for the presentation engine.
struct VulkanSurface final : VernonHalSurface {
    explicit VulkanSurface(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    VulkanSurface(const VulkanSurface &) = delete;
    VulkanSurface &operator=(const VulkanSurface &) = delete;
    ~VulkanSurface() override;

    std::shared_ptr<DeviceState> state;
    VkSurfaceKHR surface{};
    VernonHalSurfaceConfiguration configuration{};
    // Index-aligned; each view covers its image.
    std::vector<std::unique_ptr<VulkanImage>> images;
    std::vector<std::unique_ptr<VulkanView>> views;
    size_t next{};
    VkSwapchainKHR swapchain{};
    VkExtent2D extent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkSemaphore> presentReady;
    // One more than the swapchain's images, so the acquire that reuses a semaphore follows the present that waited
    // on it.
    std::vector<AcquireSemaphore> acquires;
    size_t nextAcquire{};
    bool stale{};
};

// Runs under the device's lock with the queue idle, so no submission or present still uses what it destroys.
void destroySwapchain(VulkanSurface &surface) {
    const vulkan::Driver &api = vulkan::driver();
    const VkDevice device = surface.state->device;
    for (VkSemaphore semaphore : surface.presentReady)
        api.destroySemaphore(device, semaphore, nullptr);
    for (const AcquireSemaphore &acquire : surface.acquires)
        api.destroySemaphore(device, acquire.semaphore, nullptr);
    if (surface.swapchain)
        api.destroySwapchain(device, surface.swapchain, nullptr);
    surface.presentReady.clear();
    surface.acquires.clear();
    surface.nextAcquire = 0;
    surface.swapchain = VK_NULL_HANDLE;
    surface.swapchainImages.clear();
    surface.extent = {};
}

VulkanSurface::~VulkanSurface() {
    std::lock_guard<std::mutex> guard(state->mutex);
    std::string ignored;
    (void)state->synchronize(ignored);
    destroySwapchain(*this);
    if (surface)
        vulkan::driver().destroySurface(state->instance, surface, nullptr);
}

VkPresentModeKHR presentMode(VernonHalPresentMode mode) {
    switch (mode) {
    case VERNON_HAL_PRESENT_MAILBOX:
        return VK_PRESENT_MODE_MAILBOX_KHR;
    case VERNON_HAL_PRESENT_IMMEDIATE:
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    case VERNON_HAL_PRESENT_FIFO:
        break;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

// Replaces the swapchain with one at the window's extent in the configured present mode. Runs under the device's
// lock. A minimized window has no extent and leaves the surface without a swapchain until it has one.
bool rebuildSwapchain(VulkanSurface &surface, std::string &error) {
    const vulkan::Driver &api = vulkan::driver();
    DeviceState &state = *surface.state;
    // The old swapchain's images and semaphores are idle once the queue is.
    if (!state.synchronize(error))
        return false;
    surface.stale = false;
    VkSurfaceCapabilitiesKHR capabilities{};
    if (api.getPhysicalDeviceSurfaceCapabilities(state.physicalDevice, surface.surface, &capabilities) != VK_SUCCESS) {
        error = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed";
        return false;
    }
    VkExtent2D extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX)
        extent = {std::clamp(surface.configuration.width, capabilities.minImageExtent.width,
                             capabilities.maxImageExtent.width),
                  std::clamp(surface.configuration.height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height)};
    if (!extent.width || !extent.height) {
        destroySwapchain(surface);
        return true;
    }
    if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        error = "Vulkan swapchain images cannot be transfer destinations";
        return false;
    }
    uint32_t formatCount = 0;
    std::vector<VkSurfaceFormatKHR> formats;
    if (api.getPhysicalDeviceSurfaceFormats(state.physicalDevice, surface.surface, &formatCount, nullptr) ==
        VK_SUCCESS) {
        formats.resize(formatCount);
        if (api.getPhysicalDeviceSurfaceFormats(state.physicalDevice, surface.surface, &formatCount, formats.data()) !=
            VK_SUCCESS)
            formats.clear();
        formats.resize(std::min<size_t>(formats.size(), formatCount));
    }
    if (formats.empty()) {
        error = "Vulkan surface reports no formats";
        return false;
    }
    // The surface images hold display-encoded RGBA8 values, which a UNORM swapchain shows unchanged.
    VkSurfaceFormatKHR format = formats.front();
    for (const VkSurfaceFormatKHR &candidate : formats)
        if ((candidate.format == VK_FORMAT_B8G8R8A8_UNORM || candidate.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format = candidate;
            break;
        }
    if (format.format == VK_FORMAT_UNDEFINED)
        format = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    uint32_t imageCount = std::max(capabilities.minImageCount + 1, surface.configuration.image_count);
    if (capabilities.maxImageCount)
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(capabilities.supportedCompositeAlpha & compositeAlpha))
        compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(capabilities.supportedCompositeAlpha &
                                                                  (~capabilities.supportedCompositeAlpha + 1));
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = surface.surface;
    info.minImageCount = imageCount;
    info.imageFormat = format.format;
    info.imageColorSpace = format.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = capabilities.currentTransform;
    info.compositeAlpha = compositeAlpha;
    info.presentMode = presentMode(surface.configuration.present_mode);
    info.clipped = VK_TRUE;
    info.oldSwapchain = surface.swapchain;
    VkSwapchainKHR swapchain{};
    const VkResult created = api.createSwapchain(state.device, &info, nullptr, &swapchain);
    destroySwapchain(surface);
    if (created != VK_SUCCESS) {
        error = "vkCreateSwapchainKHR failed with Vulkan error " + std::to_string(static_cast<int>(created));
        return false;
    }
    surface.swapchain = swapchain;
    surface.extent = extent;
    uint32_t count = 0;
    if (api.getSwapchainImages(state.device, swapchain, &count, nullptr) != VK_SUCCESS) {
        error = "vkGetSwapchainImagesKHR failed";
        return false;
    }
    surface.swapchainImages.resize(count);
    if (api.getSwapchainImages(state.device, swapchain, &count, surface.swapchainImages.data()) != VK_SUCCESS) {
        error = "vkGetSwapchainImagesKHR failed";
        return false;
    }
    surface.swapchainImages.resize(count);
    const VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t index = 0; index <= count; ++index) {
        VkSemaphore semaphore{};
        if (api.createSemaphore(state.device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
            error = "vkCreateSemaphore failed";
            return false;
        }
        surface.acquires.push_back({semaphore, 0});
        if (index == count)
            break;
        if (api.createSemaphore(state.device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
            error = "vkCreateSemaphore failed";
            return false;
        }
        surface.presentReady.push_back(semaphore);
    }
    return true;
}

class VulkanDeviceBuffers final : public DeviceBuffers {
public:
    explicit VulkanDeviceBuffers(std::shared_ptr<DeviceState> state)
        : state_(state), pool_(std::make_shared<BufferPool<NativeBuffer>>([state](uint64_t size) {
              using Created = Result<std::unique_ptr<NativeBuffer>, std::string>;
              vulkan::Buffer native{};
              auto created =
                  vulkan::createBuffer(*state, size, VERNON_HAL_BUFFER_STORAGE, VERNON_HAL_MEMORY_DEVICE, native);
              if (created.isErr())
                  return Created{err("cannot create a Vulkan buffer: " + created.error())};
              return Created{ok(std::make_unique<NativeBuffer>(state, native, size))};
          })) {}

    Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) override { return pool_->allocate(size); }

    Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) override {
        NativeBuffer &native = static_cast<BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the upload overruns the buffer"))};
        const VernonHalBufferWrite write{0, bytes, size};
        return vulkan::writeBuffer(*state_, native.buffer(), native.size(), VERNON_HAL_MEMORY_DEVICE, &write, 1);
    }

    Result<void, std::string> download(const Buffer &buffer, void *bytes, size_t size) override {
        NativeBuffer &native = static_cast<const BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the download overruns the buffer"))};
        const VernonHalBufferRead read{0, bytes, size};
        return vulkan::readBuffer(*state_, native.buffer(), native.size(), &read, 1);
    }

private:
    std::shared_ptr<DeviceState> state_;
    std::shared_ptr<BufferPool<NativeBuffer>> pool_;
};

class VulkanResources final : public Resources {
public:
    explicit VulkanResources(std::shared_ptr<DeviceState> state) : owner_(std::move(state)), state_(*owner_) {}

    Done<uint32_t> formatCapabilities(VernonTextureFormat format) override {
        if (!formats::find(format))
            return fail<uint32_t>(VERNON_HAL_INVALID_ARGUMENT, "the format is unknown");
        const VkFormat native = vulkanFormat(format);
        if (native == VK_FORMAT_UNDEFINED)
            return Done<uint32_t>{ok(0u)};
        const vulkan::Driver &api = vulkan::driver();
        VkFormatProperties properties{};
        api.getPhysicalDeviceFormatProperties(state_.physicalDevice, native, &properties);
        const VkFormatFeatureFlags features = properties.optimalTilingFeatures;
        const bool depth = formats::hasDepth(format);
        const auto has = [&](VkFormatFeatureFlags required) { return (features & required) == required; };
        uint32_t result = 0;
        if (has(depth ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
            result |= VERNON_HAL_FORMAT_RENDERABLE;
        if (has(VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT))
            result |= VERNON_HAL_FORMAT_BLENDABLE;
        if (has(VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
            result |= VERNON_HAL_FORMAT_FILTERABLE;
        if (has(VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) &&
            (state_.shaderStorageImageExtendedFormats || coreStorageFormat(native)))
            result |= VERNON_HAL_FORMAT_STORAGE;
        // Textures sample by default, so a format the device copies but cannot sample has no use as one.
        if (has(VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
            result |= VERNON_HAL_FORMAT_COPYABLE;
        // A color format renderable with more than one sample is also a resolve destination; the table baseline
        // removes resolving from formats whose class cannot average samples.
        VkImageFormatProperties image{};
        if ((result & VERNON_HAL_FORMAT_RENDERABLE) &&
            api.getPhysicalDeviceImageFormatProperties(
                state_.physicalDevice, native, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0,
                &image) == VK_SUCCESS &&
            (image.sampleCounts & ~VK_SAMPLE_COUNT_1_BIT) != 0)
            result |= VERNON_HAL_FORMAT_MULTISAMPLE | VERNON_HAL_FORMAT_RESOLVABLE;
        return Done<uint32_t>{ok(result & formats::find(format)->baseline)};
    }

    Done<std::unique_ptr<VernonHalBuffer>> createBuffer(const VernonHalBufferDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalBuffer>>;
        vulkan::Buffer native{};
        auto created = vulkan::createBuffer(state_, descriptor.size, descriptor.usage, descriptor.memory, native);
        if (created.isErr())
            return internal<std::unique_ptr<VernonHalBuffer>>("cannot create the buffer", created.error());
        auto buffer =
            std::unique_ptr<VulkanHalBuffer>(new (std::nothrow) VulkanHalBuffer(owner_, native, descriptor.size));
        if (!buffer) {
            vulkan::destroyBuffer(state_, native);
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a buffer");
        }
        return Created{ok(std::unique_ptr<VernonHalBuffer>(std::move(buffer)))};
    }

    Done<void> write(VernonHalBuffer &buffer, const VernonHalBufferWrite *writes, size_t count) override {
        NativeBuffer &native = static_cast<VulkanHalBuffer &>(buffer).native;
        auto written = vulkan::writeBuffer(state_, native.buffer(), buffer.descriptor.size, buffer.descriptor.memory,
                                           writes, count);
        if (written.isErr())
            return internal("cannot write the buffer", written.error());
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalBuffer &buffer, const VernonHalBufferRead *reads, size_t count) override {
        NativeBuffer &native = static_cast<VulkanHalBuffer &>(buffer).native;
        auto read = vulkan::readBuffer(state_, native.buffer(), buffer.descriptor.size, reads, count);
        if (read.isErr())
            return internal("cannot read the buffer", read.error());
        return Done<void>{ok()};
    }

    Done<void *> address(const VernonHalBuffer &) override {
        return fail<void *>(VERNON_HAL_UNSUPPORTED, "only CPU and CUDA buffers have an address");
    }

    Done<std::unique_ptr<VernonHalImage>> createImage(const VernonImageDescriptor &descriptor) override {
        auto made = makeImage(descriptor);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImage>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImage>>{ok(std::unique_ptr<VernonHalImage>(std::move(made).value()))};
    }

    Done<void> write(VernonHalImage &image, const VernonHalImageWrite *writes, size_t count) override {
        if (!count)
            return Done<void>{ok()};
        auto &target = static_cast<VulkanImage &>(image);
        std::vector<size_t> offsets(count);
        size_t total = 0;
        for (size_t index = 0; index < count; ++index) {
            const auto bytes = regionBytes(image.descriptor, writes[index].region);
            if (!bytes)
                return fail(VERNON_HAL_INVALID_ARGUMENT, "a write's data layout does not match the image");
            offsets[index] = (total + 15) & ~size_t{15};
            total = offsets[index] + *bytes;
        }
        std::lock_guard<std::mutex> guard(state_.mutex);
        std::string error;
        VkCommandBuffer command{};
        if (!state_.beginCommands(command, error))
            return internal("cannot write the image", error);
        VkBuffer staging{};
        VkDeviceSize stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!state_.acquireStaging(command, true, total, 16, staging, stagingOffset, mapped, error)) {
            state_.abandonCommands(command);
            return internal("cannot stage the image's write", error);
        }
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(count * 2);
        for (size_t index = 0; index < count; ++index) {
            const VernonHalImageRegion &region = writes[index].region;
            const bool packed = region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
            const size_t texels = size_t{region.width} * region.height * region.depth;
            uint8_t *destination = mapped + offsets[index];
            const auto *source = static_cast<const uint8_t *>(writes[index].source);
            if (packed) {
                // Vulkan copies depth and stencil as separate aspects: all depths, then all stencils.
                for (size_t texel = 0; texel < texels; ++texel) {
                    std::memcpy(destination + texel * sizeof(float), source + texel * kPackedDepthStencilBytes,
                                sizeof(float));
                    destination[texels * sizeof(float) + texel] = source[texel * kPackedDepthStencilBytes + 4];
                }
            } else {
                std::memcpy(destination, source, *regionBytes(image.descriptor, region));
            }
            VkBufferImageCopy copy = copyRegion(region, stagingOffset + offsets[index]);
            regions.push_back(copy);
            if (packed) {
                copy.bufferOffset += texels * sizeof(float);
                copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
                regions.push_back(copy);
            }
        }
        vulkan::Image &native = target.native;
        if (!transition(state_, command, native, everything(native), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, error)) {
            state_.abandonCommands(command);
            return internal("cannot write the image", error);
        }
        vulkan::driver().cmdCopyBufferToImage(command, staging, native.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              static_cast<uint32_t>(regions.size()), regions.data());
        if (!transition(state_, command, native, everything(native), restingLayout(image.descriptor.usage), error)) {
            state_.abandonCommands(command);
            return internal("cannot write the image", error);
        }
        // Takes effect in queue order, after the work submitted before it, without a host wait.
        uint64_t serial = 0;
        if (!state_.submitCommands(command, serial, error))
            return internal("cannot submit the image's write", error);
        forgetUndefined(&target.serial, 1);
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalImage &image, const VernonHalImageRead *reads, size_t count) override {
        if (!count)
            return Done<void>{ok()};
        auto &source = static_cast<VulkanImage &>(image);
        if (anyUndefined(&source.serial, 1))
            return fail(VERNON_HAL_INVALID_ARGUMENT, "the image's contents are undefined");
        std::vector<size_t> offsets(count);
        size_t total = 0;
        for (size_t index = 0; index < count; ++index) {
            const auto bytes = regionBytes(image.descriptor, reads[index].region);
            if (!bytes || *bytes != reads[index].size)
                return fail(VERNON_HAL_INVALID_ARGUMENT, "a read's size or data layout does not match the image");
            offsets[index] = (total + 15) & ~size_t{15};
            total = offsets[index] + *bytes;
        }
        std::lock_guard<std::mutex> guard(state_.mutex);
        vulkan::Image &native = source.native;
        const VkImageLayout restore = restoredLayout(native, image.descriptor.usage);
        std::string error;
        VkCommandBuffer command{};
        if (!state_.beginCommands(command, error))
            return internal("cannot read the image", error);
        VkBuffer staging{};
        VkDeviceSize stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!state_.acquireStaging(command, false, total, 16, staging, stagingOffset, mapped, error)) {
            state_.abandonCommands(command);
            return internal("cannot stage the image's read", error);
        }
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(count * 2);
        for (size_t index = 0; index < count; ++index) {
            const VernonHalImageRegion &region = reads[index].region;
            VkBufferImageCopy copy = copyRegion(region, stagingOffset + offsets[index]);
            regions.push_back(copy);
            if (region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL)) {
                copy.bufferOffset += size_t{region.width} * region.height * region.depth * sizeof(float);
                copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
                regions.push_back(copy);
            }
        }
        if (!transition(state_, command, native, everything(native), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, error)) {
            state_.abandonCommands(command);
            return internal("cannot read the image", error);
        }
        vulkan::driver().cmdCopyImageToBuffer(command, native.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging,
                                              static_cast<uint32_t>(regions.size()), regions.data());
        if (!transition(state_, command, native, everything(native), restore, error)) {
            state_.abandonCommands(command);
            return internal("cannot read the image", error);
        }
        // The host waits for this read only, not for later submissions.
        uint64_t serial = 0;
        if (!state_.submitCommands(command, serial, error) || !state_.waitForSerial(serial, error))
            return internal("cannot read the image", error);
        for (size_t index = 0; index < count; ++index) {
            const VernonHalImageRegion &region = reads[index].region;
            const uint8_t *staged = mapped + offsets[index];
            auto *destination = static_cast<uint8_t *>(reads[index].destination);
            if (region.aspect != (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL)) {
                std::memcpy(destination, staged, reads[index].size);
                continue;
            }
            const size_t texels = size_t{region.width} * region.height * region.depth;
            for (size_t texel = 0; texel < texels; ++texel) {
                uint8_t *packed = destination + texel * kPackedDepthStencilBytes;
                std::memcpy(packed, staged + texel * sizeof(float), sizeof(float));
                packed[4] = staged[texels * sizeof(float) + texel];
                std::memset(packed + 5, 0, kPackedDepthStencilBytes - 5);
            }
        }
        return Done<void>{ok()};
    }

    Done<void> generateMipmaps(VernonHalImage &image) override {
        const VernonImageDescriptor &descriptor = image.descriptor;
        if (descriptor.mip_level_count < 2 || formats::hasDepth(descriptor.format) ||
            !(descriptor.usage & VERNON_IMAGE_TRANSFER_SOURCE) ||
            !(descriptor.usage & VERNON_IMAGE_TRANSFER_DESTINATION))
            return fail(VERNON_HAL_INVALID_ARGUMENT,
                        "mipmaps need a color image with several mips that transfers in both directions");
        vulkan::Image &native = static_cast<VulkanImage &>(image).native;
        VkFormatProperties properties{};
        vulkan::driver().getPhysicalDeviceFormatProperties(state_.physicalDevice, native.format, &properties);
        constexpr VkFormatFeatureFlags blit = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
        if ((properties.optimalTilingFeatures & blit) != blit)
            return fail(VERNON_HAL_UNSUPPORTED, "the device cannot blit images of the format");
        const VkFilter filter = properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
                                    ? VK_FILTER_LINEAR
                                    : VK_FILTER_NEAREST;
        std::lock_guard<std::mutex> guard(state_.mutex);
        const VkImageLayout restore = restoredLayout(native, descriptor.usage);
        std::string error;
        VkCommandBuffer command{};
        if (!state_.beginCommands(command, error))
            return internal("cannot generate the image's mipmaps", error);
        const uint32_t layers = descriptor.array_layer_count;
        const uint32_t mips = descriptor.mip_level_count;
        const auto moved = [&](const vulkan::ImageSubresources &range, VkImageLayout target) {
            if (transition(state_, command, native, range, target, error))
                return true;
            state_.abandonCommands(command);
            return false;
        };
        if (!moved({0, mips, 0, layers}, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL))
            return internal("cannot generate the image's mipmaps", error);
        const auto extent = [&](uint32_t level) {
            const auto shrink = [&](uint32_t size) { return static_cast<int32_t>(std::max(size >> level, 1u)); };
            return VkOffset3D{shrink(descriptor.extent.width), shrink(descriptor.extent.height),
                              descriptor.dimension == VERNON_TEXTURE_3D ? shrink(descriptor.extent.depth) : 1};
        };
        for (uint32_t level = 1; level < mips; ++level) {
            if (!moved({level - 1, 1, 0, layers}, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL))
                return internal("cannot generate the image's mipmaps", error);
            VkImageBlit region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, layers};
            region.srcOffsets[1] = extent(level - 1);
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, layers};
            region.dstOffsets[1] = extent(level);
            vulkan::driver().cmdBlitImage(command, native.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, native.image,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, filter);
        }
        if (!moved({0, mips, 0, layers}, restore))
            return internal("cannot generate the image's mipmaps", error);
        uint64_t serial = 0;
        if (!state_.submitCommands(command, serial, error))
            return internal("cannot submit the image's mipmaps", error);
        return Done<void>{ok()};
    }

    Done<std::unique_ptr<VernonHalImageView>> createView(VernonHalImage &image,
                                                         const VernonImageViewDescriptor &descriptor) override {
        auto made = makeView(static_cast<VulkanImage &>(image), descriptor);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImageView>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImageView>>{
            ok(std::unique_ptr<VernonHalImageView>(std::move(made).value()))};
    }

    Done<std::unique_ptr<VernonHalSampler>> createSampler(const VernonHalSamplerDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSampler>>;
        if (descriptor.min_filter > VERNON_HAL_FILTER_LINEAR || descriptor.mag_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.mip_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.address_u > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_v > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_w > VERNON_HAL_ADDRESS_MIRRORED_REPEAT || !(descriptor.max_anisotropy <= 16.0f))
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_INVALID_ARGUMENT,
                                                           "the sampler descriptor is invalid");
        if (descriptor.max_anisotropy > 1.0f)
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_UNSUPPORTED,
                                                           "Vulkan samplers sample isotropically");
        constexpr VkSamplerAddressMode kAddress[] = {VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                     VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.minFilter = descriptor.min_filter == VERNON_HAL_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        info.magFilter = descriptor.mag_filter == VERNON_HAL_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        info.mipmapMode = descriptor.mip_filter == VERNON_HAL_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                                            : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = kAddress[descriptor.address_u];
        info.addressModeV = kAddress[descriptor.address_v];
        info.addressModeW = kAddress[descriptor.address_w];
        info.maxLod = VK_LOD_CLAMP_NONE;
        auto sampler = std::unique_ptr<VulkanSampler>(new (std::nothrow) VulkanSampler(owner_));
        if (!sampler)
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a sampler");
        std::string error;
        if (!state_.createSampler(sampler->sampler, info, error))
            return internal<std::unique_ptr<VernonHalSampler>>("cannot create the sampler", error);
        sampler->serial = nativeSamplers().add(&sampler->sampler);
        return Created{ok(std::unique_ptr<VernonHalSampler>(std::move(sampler)))};
    }

    Done<std::unique_ptr<VernonHalSurface>> createSurface(const VernonHalSurfaceDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSurface>>;
        if (!state_.presentation)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_UNSUPPORTED, "the device cannot present");
        auto surface = std::unique_ptr<VulkanSurface>(new (std::nothrow) VulkanSurface(owner_));
        if (!surface)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a surface");
        std::string error;
        if (!vulkan::createPlatformSurface(state_.instance, descriptor.native.win32_window, surface->surface, error))
            return internal<std::unique_ptr<VernonHalSurface>>("cannot create the surface", error);
        VkBool32 presentable = VK_FALSE;
        if (vulkan::driver().getPhysicalDeviceSurfaceSupport(state_.physicalDevice, state_.queueFamily,
                                                             surface->surface, &presentable) != VK_SUCCESS ||
            !presentable)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_UNSUPPORTED,
                                                           "the device's queue cannot present to the window");
        return Created{ok(std::unique_ptr<VernonHalSurface>(std::move(surface)))};
    }

    Done<VernonHalSurfaceCapabilities> capabilities(VernonHalSurface &surface) override {
        const vulkan::Driver &api = vulkan::driver();
        const VkSurfaceKHR native = static_cast<VulkanSurface &>(surface).surface;
        VkSurfaceCapabilitiesKHR window{};
        if (api.getPhysicalDeviceSurfaceCapabilities(state_.physicalDevice, native, &window) != VK_SUCCESS)
            return fail<VernonHalSurfaceCapabilities>(VERNON_HAL_INTERNAL_ERROR, "cannot query the surface");
        std::array<VkPresentModeKHR, 8> modes{};
        auto modeCount = static_cast<uint32_t>(modes.size());
        const VkResult listed =
            api.getPhysicalDeviceSurfacePresentModes(state_.physicalDevice, native, &modeCount, modes.data());
        if (listed != VK_SUCCESS && listed != VK_INCOMPLETE)
            modeCount = 0;
        VernonHalSurfaceCapabilities capabilities{};
        capabilities.struct_size = sizeof(capabilities);
        capabilities.formats[0] = VERNON_TEXTURE_RGBA8_UNORM;
        capabilities.format_count = 1;
        capabilities.present_modes = 1u << VERNON_HAL_PRESENT_FIFO;
        for (uint32_t index = 0; index < modeCount; ++index) {
            if (modes[index] == VK_PRESENT_MODE_MAILBOX_KHR)
                capabilities.present_modes |= 1u << VERNON_HAL_PRESENT_MAILBOX;
            if (modes[index] == VK_PRESENT_MODE_IMMEDIATE_KHR)
                capabilities.present_modes |= 1u << VERNON_HAL_PRESENT_IMMEDIATE;
        }
        // Without sampled or storage usage, an acquired image is read only by a preserving render scope or a read,
        // which a submission rejects while its contents are undefined.
        capabilities.usage = VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE;
        capabilities.min_image_count = 1;
        capabilities.current_width = window.currentExtent.width;
        capabilities.current_height = window.currentExtent.height;
        return Done<VernonHalSurfaceCapabilities>{ok(capabilities)};
    }

    // The new images replace the old ones only once all of them exist.
    Done<void> configure(VernonHalSurface &surface, const VernonHalSurfaceConfiguration &configuration) override {
        auto &target = static_cast<VulkanSurface &>(surface);
        VernonImageDescriptor descriptor{VERNON_TEXTURE_2D,
                                         {configuration.width, configuration.height, 1},
                                         configuration.format,
                                         1,
                                         1,
                                         1,
                                         configuration.usage | VERNON_IMAGE_TRANSFER_SOURCE};
        const VernonImageViewDescriptor viewDescriptor{
            VERNON_TEXTURE_2D, configuration.format, {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR}};
        std::vector<std::unique_ptr<VulkanImage>> images;
        std::vector<std::unique_ptr<VulkanView>> views;
        for (uint32_t index = 0; index < configuration.image_count; ++index) {
            auto image = makeImage(descriptor);
            if (image.isErr())
                return Done<void>{err(std::move(image).error())};
            std::unique_ptr<VulkanImage> made = std::move(image).value();
            made->device = surface.device;
            made->descriptor = descriptor;
            made->surface = &surface;
            auto view = makeView(*made, viewDescriptor);
            if (view.isErr())
                return Done<void>{err(std::move(view).error())};
            std::unique_ptr<VulkanView> madeView = std::move(view).value();
            madeView->image = made.get();
            madeView->descriptor = viewDescriptor;
            images.push_back(std::move(made));
            views.push_back(std::move(madeView));
        }
        // The old views go before their images.
        std::vector<std::unique_ptr<VulkanView>> oldViews = std::exchange(target.views, std::move(views));
        std::vector<std::unique_ptr<VulkanImage>> oldImages = std::exchange(target.images, std::move(images));
        target.configuration = configuration;
        target.next = 0;
        std::string error;
        bool rebuilt = false;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            rebuilt = rebuildSwapchain(target, error);
        }
        oldViews.clear();
        oldImages.clear();
        if (!rebuilt)
            return internal("cannot configure the surface", error);
        return Done<void>{ok()};
    }

    // A window whose extent is no longer the configured one, or a swapchain a present found out of date, reports out
    // of date here, so the host configures again.
    Done<VernonHalSurfaceImage> acquire(VernonHalSurface &surface) override {
        auto &source = static_cast<VulkanSurface &>(surface);
        if (source.images.empty())
            return fail<VernonHalSurfaceImage>(VERNON_HAL_INVALID_ARGUMENT, "the surface is not configured");
        VkSurfaceCapabilitiesKHR window{};
        if (vulkan::driver().getPhysicalDeviceSurfaceCapabilities(state_.physicalDevice, source.surface, &window) !=
            VK_SUCCESS)
            return fail<VernonHalSurfaceImage>(VERNON_HAL_INTERNAL_ERROR, "cannot query the surface");
        const bool resized =
            window.currentExtent.width != UINT32_MAX && (window.currentExtent.width != source.configuration.width ||
                                                         window.currentExtent.height != source.configuration.height);
        bool stale = false;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            stale = source.stale;
        }
        if (stale || resized)
            return fail<VernonHalSurfaceImage>(VERNON_HAL_SURFACE_OUT_OF_DATE, "the surface is out of date");
        VulkanImage &image = *source.images[source.next];
        markUndefined(image.serial);
        return Done<VernonHalSurfaceImage>{ok(VernonHalSurfaceImage{&image, source.views[source.next].get(), 0})};
    }

    // A frame whose swapchain is out of date, or whose window is minimized, is dropped, and the surface goes stale.
    Done<void> present(VernonHalSurface &surface) override {
        auto &target = static_cast<VulkanSurface &>(surface);
        VulkanImage &presented = *target.images[target.next];
        target.next = (target.next + 1) % target.images.size();
        std::lock_guard<std::mutex> guard(state_.mutex);
        const vulkan::Driver &api = vulkan::driver();
        std::string error;
        if ((!target.swapchain || target.stale) && !rebuildSwapchain(target, error))
            return internal("cannot present the surface", error);
        if (!target.swapchain)
            return Done<void>{ok()};
        AcquireSemaphore &acquire = target.acquires[target.nextAcquire];
        target.nextAcquire = (target.nextAcquire + 1) % target.acquires.size();
        if (acquire.serial && !state_.waitForSerial(acquire.serial, error))
            return internal("cannot present the surface", error);
        acquire.serial = 0;
        uint32_t index = 0;
        const VkResult acquired = api.acquireNextImage(state_.device, target.swapchain, UINT64_MAX, acquire.semaphore,
                                                       VK_NULL_HANDLE, &index);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            target.stale = true;
            return Done<void>{ok()};
        }
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
            return internal("cannot present the surface", "vkAcquireNextImageKHR failed with Vulkan error " +
                                                              std::to_string(static_cast<int>(acquired)));
        target.stale = acquired == VK_SUBOPTIMAL_KHR;
        // From here a failure leaves the acquire semaphore signaled with no wait; the rebuild that staleness causes
        // replaces it.
        VkCommandBuffer command{};
        if (!state_.beginCommands(command, error)) {
            target.stale = true;
            return internal("cannot present the surface", error);
        }
        vulkan::Image &native = presented.native;
        if (!transition(state_, command, native, everything(native), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, error)) {
            state_.abandonCommands(command);
            target.stale = true;
            return internal("cannot present the surface", error);
        }
        const VkImage swapchainImage = target.swapchainImages[index];
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = swapchainImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        // The acquire semaphore's wait covers the transfer stage, which this barrier's source scope chains from.
        api.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                               0, nullptr, 1, &barrier);
        VkImageBlit region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[1] = {static_cast<int32_t>(target.configuration.width),
                                static_cast<int32_t>(target.configuration.height), 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstOffsets[1] = {static_cast<int32_t>(target.extent.width), static_cast<int32_t>(target.extent.height),
                                1};
        const bool scaled =
            target.extent.width != target.configuration.width || target.extent.height != target.configuration.height;
        api.cmdBlitImage(command, native.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImage,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                         scaled ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        api.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                               nullptr, 0, nullptr, 1, &barrier);
        uint64_t serial = 0;
        if (!state_.submitCommands(command, serial, error,
                                   {acquire.semaphore, VK_PIPELINE_STAGE_TRANSFER_BIT, target.presentReady[index]})) {
            target.stale = true;
            return internal("cannot present the surface", error);
        }
        acquire.serial = serial;
        VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &target.presentReady[index];
        info.swapchainCount = 1;
        info.pSwapchains = &target.swapchain;
        info.pImageIndices = &index;
        const VkResult shown = api.queuePresent(state_.queue, &info);
        if (shown == VK_ERROR_OUT_OF_DATE_KHR || shown == VK_SUBOPTIMAL_KHR)
            target.stale = true;
        else if (shown != VK_SUCCESS)
            return internal("cannot present the surface",
                            "vkQueuePresentKHR failed with Vulkan error " + std::to_string(static_cast<int>(shown)));
        return Done<void>{ok()};
    }

    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalBuffer &buffer,
                                                           uint64_t offset, uint64_t size) override {
        return Done<VernonRuntimeProviderResourceReference>{
            ok(static_cast<const VulkanHalBuffer &>(buffer).native.reference(offset, size))};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalImageView &view) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeViewReference);
        reference.resource.value = static_cast<const VulkanView &>(view).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalSampler &sampler) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeSamplerReference);
        reference.resource.value = static_cast<const VulkanSampler &>(sampler).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }

private:
    static VkBufferImageCopy copyRegion(const VernonHalImageRegion &region, VkDeviceSize offset) {
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {copyAspect(region.aspect), region.mip_level, region.array_layer, 1};
        copy.imageOffset = {static_cast<int32_t>(region.offset_x), static_cast<int32_t>(region.offset_y),
                            static_cast<int32_t>(region.offset_z)};
        copy.imageExtent = {region.width, region.height, region.depth};
        return copy;
    }

    Done<std::unique_ptr<VulkanImage>> makeImage(const VernonImageDescriptor &descriptor) {
        using Made = Done<std::unique_ptr<VulkanImage>>;
        if (!validImage(descriptor))
            return fail<std::unique_ptr<VulkanImage>>(VERNON_HAL_INVALID_ARGUMENT, "the image descriptor is invalid");
        const VkFormat format = vulkanFormat(descriptor.format);
        if (format == VK_FORMAT_UNDEFINED ||
            (descriptor.dimension == VERNON_TEXTURE_CUBE_ARRAY && !state_.imageCubeArray))
            return fail<std::unique_ptr<VulkanImage>>(VERNON_HAL_UNSUPPORTED, "Vulkan cannot create the image");
        const uint32_t usage = descriptor.usage;
        VkFormatFeatureFlags features = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        const std::pair<uint32_t, std::pair<VkFormatFeatureFlags, VkImageUsageFlags>> kUsages[] = {
            {VERNON_IMAGE_SAMPLED, {VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT, VK_IMAGE_USAGE_SAMPLED_BIT}},
            {VERNON_IMAGE_STORAGE, {VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, VK_IMAGE_USAGE_STORAGE_BIT}},
            {VERNON_IMAGE_COLOR_ATTACHMENT,
             {VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}},
            {VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
             {VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}},
        };
        for (const auto &[bit, native] : kUsages)
            if (usage & bit) {
                features |= native.first;
                imageUsage |= native.second;
            }
        VkFormatProperties properties{};
        vulkan::driver().getPhysicalDeviceFormatProperties(state_.physicalDevice, format, &properties);
        if ((properties.optimalTilingFeatures & features) != features)
            return fail<std::unique_ptr<VulkanImage>>(VERNON_HAL_UNSUPPORTED,
                                                      "the image's format does not support its usage");
        const bool cube =
            descriptor.dimension == VERNON_TEXTURE_CUBE || descriptor.dimension == VERNON_TEXTURE_CUBE_ARRAY;
        const bool volume = descriptor.dimension == VERNON_TEXTURE_3D;
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0);
        imageInfo.imageType = volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = {descriptor.extent.width, descriptor.extent.height, descriptor.extent.depth};
        imageInfo.mipLevels = descriptor.mip_level_count;
        imageInfo.arrayLayers = volume ? 1 : descriptor.array_layer_count;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = imageUsage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.viewType = viewType(descriptor.dimension);
        viewInfo.format = format;
        viewInfo.subresourceRange = {vulkan::imageAspectMask(format), 0, descriptor.mip_level_count, 0,
                                     imageInfo.arrayLayers};
        auto image = std::unique_ptr<VulkanImage>(new (std::nothrow) VulkanImage(owner_));
        if (!image)
            return fail<std::unique_ptr<VulkanImage>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate an image");
        std::string error;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            if (!state_.createImage(image->native, imageInfo, viewInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, error))
                return internal<std::unique_ptr<VulkanImage>>("cannot create the image", error);
            image->native.subresourceLayouts.assign(size_t{descriptor.mip_level_count} * imageInfo.arrayLayers,
                                                    VK_IMAGE_LAYOUT_UNDEFINED);
            image->native.mipLevels = descriptor.mip_level_count;
        }
        return Made{ok(std::move(image))};
    }

    Done<std::unique_ptr<VulkanView>> makeView(VulkanImage &image, const VernonImageViewDescriptor &descriptor) {
        using Made = Done<std::unique_ptr<VulkanView>>;
        if (!validView(image.descriptor, descriptor))
            return fail<std::unique_ptr<VulkanView>>(VERNON_HAL_INVALID_ARGUMENT, "the view descriptor is invalid");
        const VkFormat format = vulkanFormat(descriptor.format);
        if (format == VK_FORMAT_UNDEFINED)
            return fail<std::unique_ptr<VulkanView>>(VERNON_HAL_UNSUPPORTED, "Vulkan cannot view the format");
        auto view = std::unique_ptr<VulkanView>(new (std::nothrow) VulkanView(owner_));
        if (!view)
            return fail<std::unique_ptr<VulkanView>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a view");
        const VernonImageSubresourceRange &range = descriptor.subresources;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image.native.image;
        info.viewType = viewType(descriptor.dimension);
        info.format = format;
        info.subresourceRange = {range.aspects, range.base_mip_level, range.mip_level_count, range.base_array_layer,
                                 range.array_layer_count};
        vulkan::Image &binding = view->binding;
        if (vulkan::driver().createImageView(state_.device, &info, nullptr, &binding.view) != VK_SUCCESS)
            return fail<std::unique_ptr<VulkanView>>(VERNON_HAL_INTERNAL_ERROR, "vkCreateImageView failed");
        binding.image = image.native.image;
        binding.layoutOwner = &image.native;
        binding.subresources = {range.base_mip_level, range.mip_level_count, range.base_array_layer,
                                range.array_layer_count};
        binding.format = image.native.format;
        binding.colorAttachment = image.native.colorAttachment;
        binding.owned = false;
        view->entry = {image.descriptor, descriptor, image.serial, &binding};
        view->serial = nativeViews().add(&view->entry);
        return Made{ok(std::move(view))};
    }

    std::shared_ptr<DeviceState> owner_;
    DeviceState &state_;
};

} // namespace

vulkan::Buffer *vulkanBuffer(uint64_t serial) { return serials().find(serial); }

std::unique_ptr<Resources> createVulkanResources(std::shared_ptr<vulkan::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<Resources>(new (std::nothrow) VulkanResources(std::move(state)));
}

std::unique_ptr<DeviceBuffers> createVulkanDeviceBuffers(std::shared_ptr<vulkan::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<DeviceBuffers>(new (std::nothrow) VulkanDeviceBuffers(std::move(state)));
}

} // namespace vernon::hal
