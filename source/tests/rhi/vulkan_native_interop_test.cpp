#include "VernonRHI.h"
#include "rhi/rhi_internal.h"
#include "rhi/vulkan_backend.h"
#include "runtime_rhi_test_utils.h"
#include "vernon_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

template <typename Handle> uint64_t handleBits(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<uint64_t>(handle);
    else
        return static_cast<uint64_t>(handle);
}

bool hasExtension(const std::vector<VkExtensionProperties> &extensions, std::string_view name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties &extension) {
        return std::string_view(extension.extensionName) == name;
    });
}

TEST(VulkanDeviceSelection, RanksHighPerformanceHardwareFirst) {
    EXPECT_LT(vernon::rhi::vulkan::physicalDeviceTypeRank(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
              vernon::rhi::vulkan::physicalDeviceTypeRank(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU));
    EXPECT_LT(vernon::rhi::vulkan::physicalDeviceTypeRank(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
              vernon::rhi::vulkan::physicalDeviceTypeRank(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU));
    EXPECT_LT(vernon::rhi::vulkan::physicalDeviceTypeRank(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU),
              vernon::rhi::vulkan::physicalDeviceTypeRank(VK_PHYSICAL_DEVICE_TYPE_CPU));
}

TEST(VulkanImageAspect, DistinguishesColorDepthAndPackedDepthStencilFormats) {
    EXPECT_EQ(vernon::rhi::vulkan::imageAspectMask(VK_FORMAT_R8G8B8A8_UNORM), VK_IMAGE_ASPECT_COLOR_BIT);
    EXPECT_EQ(vernon::rhi::vulkan::imageAspectMask(VK_FORMAT_D32_SFLOAT), VK_IMAGE_ASPECT_DEPTH_BIT);
    EXPECT_EQ(vernon::rhi::vulkan::imageAspectMask(VK_FORMAT_D32_SFLOAT_S8_UINT),
              VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    EXPECT_EQ(vernon::rhi::vulkan::imagePrimaryCopyAspectMask(VK_FORMAT_R8G8B8A8_UNORM), VK_IMAGE_ASPECT_COLOR_BIT);
    EXPECT_EQ(vernon::rhi::vulkan::imagePrimaryCopyAspectMask(VK_FORMAT_D32_SFLOAT), VK_IMAGE_ASPECT_DEPTH_BIT);
    EXPECT_EQ(vernon::rhi::vulkan::imagePrimaryCopyAspectMask(VK_FORMAT_D32_SFLOAT_S8_UINT), VK_IMAGE_ASPECT_DEPTH_BIT);
}

TEST(VulkanMemorySelection, PrefersCachedCoherentReadbackMemoryAndFallsBack) {
    vernon::rhi::vulkan::DeviceState state;
    state.memoryProperties.memoryTypeCount = 2;
    state.memoryProperties.memoryTypes[0].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    state.memoryProperties.memoryTypes[1].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    constexpr VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    EXPECT_EQ(state.findMemoryType(0b11, required, VK_MEMORY_PROPERTY_HOST_CACHED_BIT), 1u);
    EXPECT_EQ(state.findMemoryType(0b01, required, VK_MEMORY_PROPERTY_HOST_CACHED_BIT), 0u);
    EXPECT_FALSE(state.findMemoryType(0b00, required, VK_MEMORY_PROPERTY_HOST_CACHED_BIT).has_value());
}

TEST(VulkanOwnedDevice, CreatesDeviceAndNegotiatesPortabilityWhenAdvertised) {
    auto &driver = vernon::rhi::vulkan::driver();
    if (!driver.load())
        GTEST_SKIP() << driver.error;

    std::string error;
    vernon::rhi::vulkan::DeviceState state;
    ASSERT_TRUE(state.initialize(0, error)) << error;

    uint32_t instanceExtensionCount = 0;
    ASSERT_EQ(driver.enumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr), VK_SUCCESS);
    std::vector<VkExtensionProperties> instanceExtensions(instanceExtensionCount);
    ASSERT_EQ(driver.enumerateInstanceExtensionProperties(
                  nullptr, &instanceExtensionCount, instanceExtensions.empty() ? nullptr : instanceExtensions.data()),
              VK_SUCCESS);
    EXPECT_EQ(state.portabilityEnumeration, hasExtension(instanceExtensions, "VK_KHR_portability_enumeration"));

    uint32_t deviceExtensionCount = 0;
    ASSERT_EQ(driver.enumerateDeviceExtensionProperties(state.physicalDevice, nullptr, &deviceExtensionCount, nullptr),
              VK_SUCCESS);
    std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
    ASSERT_EQ(driver.enumerateDeviceExtensionProperties(state.physicalDevice, nullptr, &deviceExtensionCount,
                                                        deviceExtensions.empty() ? nullptr : deviceExtensions.data()),
              VK_SUCCESS);
    EXPECT_EQ(state.portabilitySubset, hasExtension(deviceExtensions, "VK_KHR_portability_subset"));
}

TEST(VulkanOwnedDevice, DownloadsDepthOnlyImageThroughDepthAspect) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_VULKAN;
    VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "Vulkan device is unavailable";

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_D32_FLOAT;
    imageDescriptor.width = 1;
    imageDescriptor.height = 1;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);
    float depth{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = download.height = download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    EXPECT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, &depth, sizeof(depth)), VERNON_RHI_STATUS_OK)
        << vernon::test::text(vernonRhiDeviceGetLastError(device));
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(VulkanOwnedDevice, TransitionsPackedDepthStencilImageWithBothAspects) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_VULKAN;
    VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "Vulkan device is unavailable";

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    imageDescriptor.width = 1;
    imageDescriptor.height = 1;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiBarrier barrier{};
    barrier.struct_size = sizeof(barrier);
    barrier.source_stage_mask = VERNON_RHI_STAGE_FRAGMENT;
    barrier.destination_stage_mask = VERNON_RHI_STAGE_FRAGMENT;
    barrier.destination_access = VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE;
    barrier.old_state = VERNON_RHI_STATE_UNDEFINED;
    barrier.new_state = VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT;
    barrier.image = image;
    barrier.is_image = 1;
    barrier.image_subresources = {0, 1, 0, 1, VERNON_RHI_IMAGE_ASPECT_DEPTH};
    EXPECT_EQ(vernonRhiCommandEncoderBarrier(device, encoder, &barrier, 1), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(VulkanNativeInterop, BorrowsObjectsWithoutOwningTheirLifetime) {
    std::string error;
    vernon::rhi::vulkan::DeviceState owner;
    if (!owner.initialize(0, error))
        GTEST_SKIP() << error;

    vernon::rhi::vulkan::Buffer buffer;
    ASSERT_TRUE(
        owner.createBuffer(buffer, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, error))
        << error;

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {4, 4, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = imageInfo.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    vernon::rhi::vulkan::Image image;
    ASSERT_TRUE(owner.createImage(image, imageInfo, viewInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, error)) << error;

    VernonRhiVulkanBorrowedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.instance = owner.instance;
    deviceDescriptor.physical_device = owner.physicalDevice;
    deviceDescriptor.device = owner.device;
    deviceDescriptor.queue = owner.queue;
    ASSERT_FALSE(owner.availableCommandFrames.empty());
    const VkCommandBuffer borrowedCommand = owner.availableCommandFrames.front().command;
    deviceDescriptor.command_buffer = borrowedCommand;
    deviceDescriptor.queue_family_index = owner.queueFamily;
    deviceDescriptor.queue_capabilities = VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    const VernonRhiDevice device = vernonRhiCreateBorrowedVulkanDevice(&deviceDescriptor);
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonRhiVulkanBorrowedBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.buffer = handleBits(buffer.buffer);
    bufferDescriptor.size = 256;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.state = VERNON_RHI_STATE_COMMON;
    VernonRhiBuffer importedBuffer{};
    ASSERT_EQ(vernonRhiVulkanDeviceImportBorrowedBuffer(device, &bufferDescriptor, &importedBuffer),
              VERNON_RHI_STATUS_OK);

    VernonRhiVulkanBorrowedImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.image = handleBits(image.image);
    imageDescriptor.descriptor.struct_size = sizeof(imageDescriptor.descriptor);
    imageDescriptor.descriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.descriptor.width = 4;
    imageDescriptor.descriptor.height = 4;
    imageDescriptor.descriptor.depth = 1;
    imageDescriptor.descriptor.mip_levels = 1;
    imageDescriptor.descriptor.array_layers = 1;
    imageDescriptor.descriptor.sample_count = 1;
    imageDescriptor.descriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    imageDescriptor.state = VERNON_RHI_STATE_UNDEFINED;
    VernonRhiImage importedImage{};
    ASSERT_EQ(vernonRhiVulkanDeviceImportBorrowedImage(device, &imageDescriptor, &importedImage), VERNON_RHI_STATUS_OK);

    VernonRhiVulkanBorrowedImageViewDescriptor imageViewDescriptor{};
    imageViewDescriptor.struct_size = sizeof(imageViewDescriptor);
    imageViewDescriptor.image_view = handleBits(image.view);
    imageViewDescriptor.descriptor.struct_size = sizeof(imageViewDescriptor.descriptor);
    imageViewDescriptor.descriptor.image = importedImage;
    imageViewDescriptor.descriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageViewDescriptor.descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageViewDescriptor.descriptor.mip_level_count = 1;
    imageViewDescriptor.descriptor.array_layer_count = 1;
    imageViewDescriptor.descriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView importedImageView{};
    ASSERT_EQ(vernonRhiVulkanDeviceImportBorrowedImageView(device, &imageViewDescriptor, &importedImageView),
              VERNON_RHI_STATUS_OK);

    void *native = nullptr;
    EXPECT_EQ(vernonRhiVulkanDeviceGetBorrowedQueue(device, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, owner.queue);
    EXPECT_EQ(vernonRhiVulkanDeviceGetBorrowedCommandBuffer(device, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, borrowedCommand);
    uint64_t nativeBits = 0;
    EXPECT_EQ(vernonRhiVulkanDeviceGetBufferNativeHandle(device, importedBuffer, &nativeBits), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(nativeBits, handleBits(buffer.buffer));
    EXPECT_EQ(vernonRhiDeviceGetImageNativeHandle(device, importedImage, &nativeBits), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(nativeBits, handleBits(image.image));
    EXPECT_EQ(vernonRhiDeviceGetImageViewNativeHandle(device, importedImageView, &nativeBits), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(nativeBits, handleBits(image.view));

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    VernonRhiCompletionState completionState{};
    ASSERT_EQ(vernonRhiCompletionGetState(device, completion, &completionState), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(completionState, VERNON_RHI_COMPLETION_PENDING);
    ASSERT_EQ(vernonRhiCompletionSignal(device, completion, VERNON_RHI_STATUS_OK), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, importedImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, importedImageView), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, importedImage), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, importedBuffer), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);

    VkCommandBuffer commands = VK_NULL_HANDLE;
    ASSERT_TRUE(owner.beginCommands(commands, error)) << error;
    ASSERT_TRUE(owner.submitCommands(commands, error)) << error;
    owner.destroyImage(image);
    owner.destroyBuffer(buffer);
}

TEST(VulkanNativeInterop, KeepsOwnedResourceAddressesStableAsSlotsGrow) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_VULKAN;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "Vulkan device is unavailable";

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.alignment = 16;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_VERTEX;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;

    std::vector<VernonRhiBuffer> buffers(64);
    for (VernonRhiBuffer &buffer : buffers)
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);

    const uint64_t firstResource = vernon::rhi::bufferResource(device, buffers.front());
    ASSERT_NE(firstResource, 0u);
    for (size_t index = 0; index < 64; ++index) {
        VernonRhiBuffer extra{};
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &extra), VERNON_RHI_STATUS_OK);
        buffers.push_back(extra);
    }
    EXPECT_EQ(vernon::rhi::bufferResource(device, buffers.front()), firstResource);

    for (VernonRhiBuffer buffer : buffers)
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(VulkanNativeInterop, RecyclesIndividuallyFreedDescriptorSets) {
    std::string error;
    vernon::rhi::vulkan::DeviceState owner;
    if (!owner.initialize(0, error))
        GTEST_SKIP() << error;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    VkDescriptorSetLayout layout{};
    auto &driver = vernon::rhi::vulkan::driver();
    ASSERT_EQ(driver.createDescriptorSetLayout(owner.device, &layoutInfo, nullptr, &layout), VK_SUCCESS);

    for (size_t index = 0; index < 512; ++index) {
        VkDescriptorSet set{};
        ASSERT_TRUE(owner.allocateDescriptorSet(layout, set, error)) << error;
        ASSERT_TRUE(owner.freeDescriptorSet(set, error)) << error;
    }

    driver.destroyDescriptorSetLayout(owner.device, layout, nullptr);
}

} // namespace
