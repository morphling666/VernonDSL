#ifndef VERNON_RHI_VULKAN_BACKEND_H
#define VERNON_RHI_VULKAN_BACKEND_H

#include "VernonRHI.h"
#include "vulkan_driver.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::rhi::vulkan {

VERNON_RHI_CAPI uint32_t physicalDeviceTypeRank(VkPhysicalDeviceType type);

inline VkImageAspectFlags imageAspectMask(VkFormat format) {
    if (format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    if (format == VK_FORMAT_D32_SFLOAT)
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

inline VkImageAspectFlags imagePrimaryCopyAspectMask(VkFormat format) {
    return imageAspectMask(format) & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}

struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    uint8_t *mapped{};
    bool owned{true};
};

struct Image {
    struct LayoutJournal {
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::vector<VkImageLayout> subresources;
    };

    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    std::vector<VkImageLayout> subresourceLayouts;
    std::unordered_map<uint64_t, LayoutJournal> layoutJournals;
    VkFormat format{VK_FORMAT_UNDEFINED};
    bool colorAttachment{};
    bool owned{true};
};

struct Sampler {
    VkSampler sampler{};
    bool owned{true};
};

struct VERNON_RHI_CAPI DeviceState {
    struct CommandFrame {
        VkCommandPool pool{};
        VkCommandBuffer command{};
        VkFence fence{};
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
    void abandonCommands(VkCommandBuffer command);
    std::optional<uint32_t> findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                                           VkMemoryPropertyFlags preferred = 0) const;
    bool createBuffer(Buffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memoryProperties, std::string &error,
                      VkMemoryPropertyFlags preferredMemoryProperties = 0);
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
    VkDeviceSize descriptorBufferOffsetAlignment{1};
    bool dynamicRendering{};
    bool shaderBufferFloat32AtomicAdd{};
    bool portabilityEnumeration{};
    bool portabilitySubset{};
    bool nativeObjectsBorrowed{};
    VkCommandBuffer borrowedCommandBuffer{};
    std::vector<CommandFrame> availableCommandFrames;
    std::unordered_map<uintptr_t, CommandFrame> activeCommandFrames;
    VkDescriptorPool descriptorPool{};
    std::mutex descriptorMutex;
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
