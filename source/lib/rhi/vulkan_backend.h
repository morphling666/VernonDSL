#ifndef VERNON_RHI_VULKAN_BACKEND_H
#define VERNON_RHI_VULKAN_BACKEND_H

#include "VernonRHI.h"
#include "vulkan_driver.h"

#include <array>
#include <optional>
#include <string>

namespace vernon::rhi::vulkan {

struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    bool owned{true};
};

struct Image {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    VkFormat format{VK_FORMAT_UNDEFINED};
    bool colorAttachment{};
    bool owned{true};
};

struct Sampler {
    VkSampler sampler{};
    bool owned{true};
};

struct VERNON_RHI_CAPI DeviceState {
    static constexpr uint32_t frameCount = 3;

    struct CommandFrame {
        VkCommandBuffer command{};
        VkFence fence{};
        bool submitted{};
    };

    struct StagingRing {
        Buffer buffer;
        uint8_t *mapped{};
        VkDeviceSize capacity{};
        VkDeviceSize cursor{};
    };

    ~DeviceState();

    bool initialize(uint32_t deviceIndex, std::string &error);
    bool initializeBorrowed(VkInstance borrowedInstance, VkPhysicalDevice borrowedPhysicalDevice,
                            VkDevice borrowedDevice, VkQueue borrowedQueue, uint32_t borrowedQueueFamily,
                            VkCommandBuffer borrowedCommandBuffer, std::string &error);
    void shutdown();
    bool synchronize(std::string &error);
    bool beginCommands(VkCommandBuffer &command, std::string &error);
    bool submitCommands(VkCommandBuffer command, std::string &error);
    std::optional<uint32_t> findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const;
    bool createBuffer(Buffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memoryProperties, std::string &error);
    void destroyBuffer(Buffer &buffer);
    bool createImage(Image &image, const VkImageCreateInfo &imageInfo, const VkImageViewCreateInfo &viewInfo,
                     VkMemoryPropertyFlags memoryProperties, std::string &error);
    void destroyImage(Image &image);
    bool createSampler(Sampler &sampler, const VkSamplerCreateInfo &createInfo, std::string &error);
    void destroySampler(Sampler &sampler);
    bool acquireStaging(bool upload, VkDeviceSize size, VkDeviceSize alignment, VkBuffer &buffer, VkDeviceSize &offset,
                        uint8_t *&mapped, std::string &error);
    bool allocateDescriptorSet(VkDescriptorSetLayout layout, VkDescriptorSet &set, std::string &error);
    bool freeDescriptorSet(VkDescriptorSet set, std::string &error);

    VkInstance instance{};
    VkPhysicalDevice physicalDevice{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t queueFamily{};
    uint32_t maxPushConstantsSize{};
    uint32_t maxVertexInputAttributes{};
    uint32_t apiVersion{};
    uint32_t maxComputeWorkGroupInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    bool dynamicRendering{};
    bool nativeObjectsBorrowed{};
    VkCommandBuffer borrowedCommandBuffer{};
    VkCommandPool commandPool{};
    std::array<CommandFrame, frameCount> frames{};
    uint32_t currentFrame{frameCount - 1};
    VkDescriptorPool descriptorPool{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    Sampler defaultImplicitSampler;
    size_t defaultImplicitSamplerCreations{};
    size_t commandBufferAllocations{};
    size_t descriptorPoolCreations{};
    size_t stagingBufferAllocations{};
    StagingRing uploadRing;
    StagingRing readbackRing;
};

} // namespace vernon::rhi::vulkan

#endif
