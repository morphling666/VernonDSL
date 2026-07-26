#include "VernonRHI.h"
#include "rhi/rhi_internal.h"
#include "rhi/vulkan_backend.h"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename Handle> uint64_t handleBits(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<uint64_t>(handle);
    else
        return static_cast<uint64_t>(handle);
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
    deviceDescriptor.command_buffer = owner.frames[0].command;
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
    imageViewDescriptor.descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageViewDescriptor.descriptor.mip_level_count = 1;
    imageViewDescriptor.descriptor.array_layer_count = 1;
    VernonRhiImageView importedImageView{};
    ASSERT_EQ(vernonRhiVulkanDeviceImportBorrowedImageView(device, &imageViewDescriptor, &importedImageView),
              VERNON_RHI_STATUS_OK);

    void *native = nullptr;
    EXPECT_EQ(vernonRhiVulkanDeviceGetBorrowedQueue(device, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, owner.queue);
    EXPECT_EQ(vernonRhiVulkanDeviceGetBorrowedCommandBuffer(device, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, owner.frames[0].command);
    uint64_t nativeBits = 0;
    EXPECT_EQ(vernonRhiVulkanDeviceGetBufferNativeHandle(device, importedBuffer, &nativeBits), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(nativeBits, handleBits(buffer.buffer));
    EXPECT_EQ(vernonRhiDeviceGetImageNativeHandle(device, importedImage, &nativeBits), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(nativeBits, handleBits(image.image));
    EXPECT_EQ(vernonRhiDeviceGetImageViewNativeHandle(device, importedImageView, &nativeBits), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(nativeBits, handleBits(image.view));

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, importedImage), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, importedImageView), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, importedImage), VERNON_RHI_STATUS_OK);
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
