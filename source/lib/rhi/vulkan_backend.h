#ifndef VERNON_RHI_VULKAN_BACKEND_H
#define VERNON_RHI_VULKAN_BACKEND_H

#include "VernonError.hpp"
#include "VernonFormats.hpp"
#include "VernonRHI.h"
#include "VernonResult.hpp"
#include "staging_pool.h"
#include "vulkan_driver.h"

#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::rhi::vulkan {

VERNON_RHI_CAPI uint32_t physicalDeviceTypeRank(VkPhysicalDeviceType type);

static_assert(VERNON_RHI_IMAGE_ASPECT_COLOR == VK_IMAGE_ASPECT_COLOR_BIT &&
              VERNON_RHI_IMAGE_ASPECT_DEPTH == VK_IMAGE_ASPECT_DEPTH_BIT &&
              VERNON_RHI_IMAGE_ASPECT_STENCIL == VK_IMAGE_ASPECT_STENCIL_BIT);

// Formats outside the table, which only imported images carry, are color formats.
inline VkImageAspectFlags imageAspectMask(VkFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, rhi, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,  \
                      vulkan, ...)                                                                                     \
    case vulkan:                                                                                                       \
        return formats::aspects(VERNON_RHI_FORMAT_##NAME);
#include "VernonFormats.def"
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
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

struct ImageSubresources {
    uint32_t baseMipLevel{};
    uint32_t mipLevelCount{};
    uint32_t baseArrayLayer{};
    uint32_t arrayLayerCount{};
};

struct Image {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    // An image tracks the layout of each of its subresources in `subresourceLayouts`, indexed `mip + layer *
    // mipLevels`, as the work submitted so far leaves them; `layout` is the layout they last shared. A view's image
    // tracks none: its layouts are those of the subresources `subresources` of `layoutOwner`, so every view of an
    // image observes the same layouts.
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    std::vector<VkImageLayout> subresourceLayouts;
    uint32_t mipLevels{};
    Image *layoutOwner{};
    ImageSubresources subresources{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    bool colorAttachment{};
    bool owned{true};
};

struct Sampler {
    VkSampler sampler{};
    bool owned{true};
};

// A descriptor set and the pool that frees it.
struct DescriptorSet {
    VkDescriptorSet set{};
    VkDescriptorPool pool{};
};

struct VERNON_RHI_CAPI DeviceState {
    struct StagingChunk {
        Buffer buffer;
        VkDeviceSize capacity{};
    };

    // The layouts a recording needs an image's subresources in when it begins executing and leaves them in, indexed
    // like Image::subresourceLayouts; kUnusedLayout marks the subresources it does not use.
    struct ImageLayoutUse {
        std::vector<VkImageLayout> initial;
        std::vector<VkImageLayout> current;
    };
    static constexpr VkImageLayout kUnusedLayout = VK_IMAGE_LAYOUT_MAX_ENUM;

    struct CommandFrame {
        VkCommandPool pool{};
        VkCommandBuffer command{};
        // Allocated on first need; holds the transitions submission records ahead of `command`.
        VkCommandBuffer prologue{};
        VkFence fence{};
        uint64_t serial{};
        StagingArena<StagingChunk> upload;
        StagingArena<StagingChunk> readback;
        // Keyed by the image tracking the layouts; the encoder retains every image it uses until it completes.
        std::unordered_map<Image *, ImageLayoutUse> imageLayouts;
    };

    // The staging a borrowed device's recordings acquired up to the submission numbered `serial`.
    struct BorrowedStaging {
        uint64_t serial{};
        StagingArena<StagingChunk> upload;
        StagingArena<StagingChunk> readback;
    };

    // A native object that submitted work may still use; it is destroyed once the queue reaches `serial`.
    struct DeferredDestruction {
        uint64_t serial{};
        Buffer buffer;
        Image image;
        VkImageView view{};
        Sampler sampler;
    };

    // Bounds the frames, and so the staging, that queue-ordered uploads can keep alive without GPU progress.
    static constexpr size_t maxFramesInFlight = 64;

    ~DeviceState();

    bool initialize(uint32_t deviceIndex, std::string &error);
    bool initializeBorrowed(VkInstance borrowedInstance, VkPhysicalDevice borrowedPhysicalDevice,
                            VkDevice borrowedDevice, VkQueue borrowedQueue, uint32_t borrowedQueueFamily,
                            VkCommandBuffer borrowedCommandBuffer, std::string &error);
    void shutdown();
    bool synchronize(std::string &error);
    bool beginCommands(VkCommandBuffer &command, std::string &error);
    // Submits without waiting. `serial` is the point of the device timeline at which the work completes; a borrowed
    // device submits nothing, since its host executes the command buffer, and numbers the submission instead.
    bool submitCommands(VkCommandBuffer command, uint64_t &serial, std::string &error);
    bool waitForSerial(uint64_t serial, std::string &error);
    bool pollSerial(uint64_t serial, bool &completed, std::string &error);
    // Releases the staging of borrowed submission `serial` and of everything recorded before it, once the host
    // reports that the submission executed. The host executes the borrowed command buffer in recording order, so that
    // covers abandoned recordings too, but not recordings submitted after it.
    void retireBorrowedCommands(uint64_t serial);
    void abandonCommands(VkCommandBuffer command);
    void destroyAfterSubmittedWork(DeferredDestruction destruction);
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
    // Staging owned by the frame recording `command`, valid until that frame retires.
    bool acquireStaging(VkCommandBuffer command, bool upload, VkDeviceSize size, VkDeviceSize alignment,
                        VkBuffer &buffer, VkDeviceSize &offset, uint8_t *&mapped, std::string &error);
    // Allocates from a chain of pools that grows once every pool is exhausted; sets live until they are freed.
    bool allocateDescriptorSet(VkDescriptorSetLayout layout, DescriptorSet &set, std::string &error);
    void freeDescriptorSet(DescriptorSet &set);
    // Image layouts follow submission order, since recordings may be submitted in another order than they are
    // recorded in. A recording never changes an image's layouts: submission transitions them to the layouts the
    // recording begins from in a prologue ahead of it, then sets them to the layouts it leaves. A borrowed device's
    // recordings execute in the order they are recorded, so they transition from the image's layouts directly.
    //
    // Moves subresource `subresource` of `owner`, the image tracking its layouts, to `layout` within recording
    // `command`. `from` is the layout the recording transitions it from here, or empty when the subresource's first
    // use is transitioned by the prologue.
    bool moveImageLayout(VkCommandBuffer command, Image &owner, size_t subresource, VkImageLayout layout,
                         std::optional<VkImageLayout> &from, std::string &error);
    // The layout subresource `subresource` of `owner` has at this point of recording `command`.
    VkImageLayout recordedImageLayout(VkCommandBuffer command, Image &owner, size_t subresource);

    VkInstance instance{};
    VkPhysicalDevice physicalDevice{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t queueFamily{};
    // Zero when the queue writes no timestamps.
    uint32_t timestampValidBits{};
    float timestampPeriod{};
    uint32_t maxPushConstantsSize{};
    uint32_t maxVertexInputAttributes{};
    uint32_t apiVersion{};
    uint32_t maxComputeWorkGroupInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    VkDeviceSize descriptorBufferOffsetAlignment{1};
    bool dynamicRendering{};
    bool shaderBufferFloat32AtomicAdd{};
    bool imageCubeArray{};
    bool portabilityEnumeration{};
    bool portabilitySubset{};
    bool nativeObjectsBorrowed{};
    // Null unless this state created the instance with VK_EXT_debug_utils; the loader may return entry points for an
    // extension that a borrowed instance never enabled.
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel{};
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel{};
    PFN_vkCmdInsertDebugUtilsLabelEXT cmdInsertDebugUtilsLabel{};
    VkCommandBuffer borrowedCommandBuffer{};
    std::vector<CommandFrame> availableCommandFrames;
    std::unordered_map<uintptr_t, CommandFrame> recordingCommandFrames;
    // Submitted frames in serial order. A fence signal covers every batch submitted before it on the queue, so the
    // first unsignaled frame bounds the completed prefix of the timeline.
    std::deque<CommandFrame> inFlightCommandFrames;
    CommandFrame borrowedCommandFrame;
    // In submission order; the staging recorded since the last borrowed submission stays in borrowedCommandFrame.
    std::deque<BorrowedStaging> borrowedStaging;
    uint64_t borrowedSubmissions{};
    std::vector<DeferredDestruction> deferredDestructions;
    StagingPool<StagingChunk> uploadStaging;
    StagingPool<StagingChunk> readbackStaging;
    uint64_t submittedSerial{};
    uint64_t completedSerial{};
    bool deviceLost{};
    std::vector<VkDescriptorPool> descriptorPools;
    // The pool allocations try first: the last one that had room.
    size_t currentDescriptorPool{};
    std::mutex descriptorMutex;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    Sampler defaultImplicitSampler;
    size_t defaultImplicitSamplerCreations{};
    size_t commandBufferAllocations{};
    size_t descriptorPoolCreations{};
    size_t stagingBufferAllocations{};

private:
    bool createCommandFrame(CommandFrame &frame, std::string &error);
    void destroyCommandFrame(CommandFrame &frame);
    void recycleCommandFrame(CommandFrame frame);
    CommandFrame *recordingFrame(VkCommandBuffer command);
    bool recordLayoutPrologue(CommandFrame &frame, bool &recorded, std::string &error);
    void commitImageLayouts(CommandFrame &frame);
    void retireStaging(CommandFrame &frame);
    bool createStagingChunk(bool upload, VkDeviceSize capacity, StagingChunk &chunk, std::string &error);
    bool createDescriptorPool(std::string &error);
    bool retireCommandFrames(uint64_t through, std::string &error);
    void destroyDeferred(bool all);
    void destroyNow(DeferredDestruction &destruction);
};

// Transitions the subresources of `image`, an image or a view, within recording `command`.
VERNON_RHI_CAPI Result<void, RhiError> transitionCommandImage(VernonRhiDevice device, VkCommandBuffer command,
                                                              Image &image, VkImageLayout layout) noexcept;
// Whether every subresource of `image`, an image or a view, is in `layout` at this point of recording `command`.
VERNON_RHI_CAPI Result<bool, RhiError> imageInLayout(VernonRhiDevice device, VkCommandBuffer command, Image &image,
                                                     VkImageLayout layout) noexcept;

} // namespace vernon::rhi::vulkan

#endif
