#ifndef VERNON_HAL_VULKAN_STATE_H
#define VERNON_HAL_VULKAN_STATE_H

#include "VernonFormats.hpp"
#include "VernonHal.h"
#include "VernonResult.hpp"
#include "hal/staging_pool.h"
#include "hal/vulkan/vulkan_driver.h"

#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::hal::vulkan {

uint32_t physicalDeviceTypeRank(VkPhysicalDeviceType type);

static_assert(VERNON_IMAGE_ASPECT_COLOR == VK_IMAGE_ASPECT_COLOR_BIT &&
              VERNON_IMAGE_ASPECT_DEPTH == VK_IMAGE_ASPECT_DEPTH_BIT &&
              VERNON_IMAGE_ASPECT_STENCIL == VK_IMAGE_ASPECT_STENCIL_BIT);

// Formats outside the table are color formats.
inline VkImageAspectFlags imageAspectMask(VkFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,       \
                      vulkan, ...)                                                                                     \
    case vulkan:                                                                                                       \
        return formats::aspects(VERNON_TEXTURE_##NAME);
#include "VernonFormats.def"
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    uint8_t *mapped{};
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
    // A view's image names the image and memory of the image it views, which it does not destroy.
    bool owned{true};
};

struct Sampler {
    VkSampler sampler{};
};

// The Vulkan instance and device a HAL device creates, which its device, resources and runtime context share. The
// native handles, limits and features never change once initialized; every other member changes only under `mutex`,
// which callers hold around the member functions and which the free functions below take themselves.
struct DeviceState {
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
        // Keyed by the image tracking the layouts; the recording's owner keeps every image it uses alive until it
        // completes.
        std::unordered_map<Image *, ImageLayoutUse> imageLayouts;
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

    DeviceState() = default;
    ~DeviceState();
    DeviceState(const DeviceState &) = delete;
    DeviceState &operator=(const DeviceState &) = delete;

    bool initialize(uint32_t deviceIndex, std::string &error);
    void shutdown();
    bool synchronize(std::string &error);
    bool beginCommands(VkCommandBuffer &command, std::string &error);
    // Binary semaphores a submission waits on before `waitStage` and signals once it completes; null ones are skipped.
    struct SubmitSemaphores {
        VkSemaphore wait{};
        VkPipelineStageFlags waitStage{};
        VkSemaphore signal{};
    };

    // Submits without waiting. `serial` is the point of the device timeline at which the work completes.
    bool submitCommands(VkCommandBuffer command, uint64_t &serial, std::string &error,
                        const SubmitSemaphores &semaphores = {});
    // Submits `command`, which the caller recorded, ended and keeps, ahead of nothing but the pending uploads and a
    // prologue that transitions the images of `layouts` to the layouts it begins from. Afterwards the images are in
    // the layouts it leaves.
    bool submitRecorded(VkCommandBuffer command, const std::unordered_map<Image *, ImageLayoutUse> &layouts,
                        uint64_t &serial, std::string &error);
    // Buffer uploads record into one pending recording instead of submitting one each, and every submission and
    // synchronization submits it ahead of itself, so that uploads keep queue order. `command` follows the earlier
    // uploads of the recording, after a barrier ordering their transfers.
    bool beginUploadCommands(VkCommandBuffer &command, std::string &error);
    bool flushUploads(std::string &error);
    bool waitForSerial(uint64_t serial, std::string &error);
    bool pollSerial(uint64_t serial, bool &completed, std::string &error);
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
    // Staging owned by the frame recording `command`, valid until that frame retires. Upload staging can also back
    // uniform and storage buffer descriptors.
    bool acquireStaging(VkCommandBuffer command, bool upload, VkDeviceSize size, VkDeviceSize alignment,
                        VkBuffer &buffer, VkDeviceSize &offset, uint8_t *&mapped, std::string &error);
    // Image layouts follow submission order, since recordings may be submitted in another order than they are
    // recorded in. A recording never changes an image's layouts: submission transitions them to the layouts the
    // recording begins from in a prologue ahead of it, then sets them to the layouts it leaves.
    //
    // Moves subresource `subresource` of `owner`, the image tracking its layouts, to `layout` within recording
    // `command`. `from` is the layout the recording transitions it from here, or empty when the subresource's first
    // use is transitioned by the prologue.
    bool moveImageLayout(VkCommandBuffer command, Image &owner, size_t subresource, VkImageLayout layout,
                         std::optional<VkImageLayout> &from, std::string &error);
    // Whether recording `command` already moved every subresource of `range` of `owner` to `layout`, so that moving
    // them there again changes nothing.
    bool movedToLayout(VkCommandBuffer command, Image &owner, const ImageSubresources &range, VkImageLayout layout);

    std::mutex mutex;
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
    VkDeviceSize descriptorBufferOffsetAlignment{1};
    bool dynamicRendering{};
    bool drawIndirectCount{};
    // multiDrawIndirect and drawIndirectFirstInstance are enabled, which device draw streams need.
    bool multiDrawIndirect{};
    bool shaderBufferFloat32AtomicAdd{};
    bool imageCubeArray{};
    // VK_KHR_surface, the platform's surface extension and VK_KHR_swapchain are enabled.
    bool presentation{};
    bool portabilityEnumeration{};
    bool portabilitySubset{};
    // Null unless the instance has VK_EXT_debug_utils.
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel{};
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel{};
    PFN_vkCmdInsertDebugUtilsLabelEXT cmdInsertDebugUtilsLabel{};
    std::vector<CommandFrame> availableCommandFrames;
    std::unordered_map<uintptr_t, CommandFrame> recordingCommandFrames;
    // The recording of the uploads not yet submitted, one of recordingCommandFrames; null when there are none.
    VkCommandBuffer pendingUploads{};
    // Submitted frames in serial order. A fence signal covers every batch submitted before it on the queue, so the
    // first unsignaled frame bounds the completed prefix of the timeline.
    std::deque<CommandFrame> inFlightCommandFrames;
    std::vector<DeferredDestruction> deferredDestructions;
    StagingPool<StagingChunk> uploadStaging;
    StagingPool<StagingChunk> readbackStaging;
    uint64_t submittedSerial{};
    uint64_t completedSerial{};
    bool deviceLost{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    Sampler defaultImplicitSampler;

private:
    bool createCommandFrame(CommandFrame &frame, std::string &error);
    void destroyCommandFrame(CommandFrame &frame);
    void recycleCommandFrame(CommandFrame frame);
    CommandFrame *recordingFrame(VkCommandBuffer command);
    bool recordLayoutPrologue(CommandFrame &frame, bool &recorded, std::string &error);
    bool recordPrologue(CommandFrame &frame, const std::vector<VkImageMemoryBarrier> &barriers, std::string &error);
    void commitImageLayouts(CommandFrame &frame);
    static void commitLayouts(const std::unordered_map<Image *, ImageLayoutUse> &layouts);
    void retireFrameMemory(StagingArena<StagingChunk> &upload, StagingArena<StagingChunk> &readback);
    bool createStagingChunk(bool upload, VkDeviceSize capacity, StagingChunk &chunk, std::string &error);
    bool retireCommandFrames(uint64_t through, std::string &error);
    void destroyDeferred(bool all);
    void destroyNow(DeferredDestruction &destruction);
};

// These take the state's mutex.
//
// DeviceState::submitRecorded; the result is the serial at which the submission completes.
Result<uint64_t, std::string>
submitRecordedCommands(DeviceState &state, VkCommandBuffer command,
                       const std::unordered_map<Image *, DeviceState::ImageLayoutUse> &layouts);
Result<void, std::string> waitForSubmission(DeviceState &state, uint64_t serial);
Result<bool, std::string> submissionCompleted(DeviceState &state, uint64_t serial);
// A buffer of `size` bytes, rounded up to four, that transfers in both directions; mapped unless in device memory.
// `usage` holds VernonHalBufferUsageBits.
Result<void, std::string> createBuffer(DeviceState &state, uint64_t size, uint32_t usage, VernonHalMemory memory,
                                       Buffer &buffer);
// Writes `writes` into `buffer` of `size` bytes in queue order: after the work submitted before, without a host wait.
// UPLOAD memory is written directly.
Result<void, std::string> writeBuffer(DeviceState &state, Buffer &buffer, uint64_t size, VernonHalMemory memory,
                                      const VernonHalBufferWrite *writes, size_t count);
// Reads `reads` of `buffer` of `size` bytes once the work submitted before completes.
Result<void, std::string> readBuffer(DeviceState &state, Buffer &buffer, uint64_t size,
                                     const VernonHalBufferRead *reads, size_t count);
// Destroys `buffer` once every submission made so far completes.
void destroyBuffer(DeviceState &state, Buffer &buffer);

} // namespace vernon::hal::vulkan

#endif
