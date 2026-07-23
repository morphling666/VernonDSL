#ifndef VERNON_RUNTIME_BACKEND_VULKAN_H
#define VERNON_RUNTIME_BACKEND_VULKAN_H

#include "VernonRuntime.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan_driver.h"
#endif

#include <string>

struct VernonDeviceBuffer;
struct VernonDeviceSampler;
struct VernonDeviceTexture;
namespace vernon::runtime {

#if defined(VERNON_HAS_VULKAN_RUNTIME)
struct VulkanContextState {
    VkInstance instance{};
    VkPhysicalDevice physicalDevice{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t queueFamily{};
    uint32_t maxPushConstantsSize{};
    VkCommandPool commandPool{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkSampler defaultImplicitSampler{};
    size_t defaultImplicitSamplerCreations{};
};

struct VulkanBufferState {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
};

struct VulkanTextureState {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    VkFormat format{VK_FORMAT_UNDEFINED};
    bool colorAttachment{};
};

struct VulkanSamplerState {
    VkSampler sampler{};
};

inline VulkanContextState &vulkanState(VernonRuntimeContext &context) {
    return runtimeBackendState<VulkanContextState>(context);
}

inline const VulkanContextState &vulkanState(const VernonRuntimeContext &context) {
    return runtimeBackendState<VulkanContextState>(context);
}

inline VulkanBufferState &vulkanBufferState(VernonDeviceBuffer &buffer) {
    return runtimeBackendState<VulkanBufferState>(buffer);
}

inline const VulkanBufferState &vulkanBufferState(const VernonDeviceBuffer &buffer) {
    return runtimeBackendState<VulkanBufferState>(buffer);
}

inline VulkanTextureState &vulkanTextureState(VernonDeviceTexture &texture) {
    return runtimeBackendState<VulkanTextureState>(texture);
}

inline const VulkanTextureState &vulkanTextureState(const VernonDeviceTexture &texture) {
    return runtimeBackendState<VulkanTextureState>(texture);
}

inline VulkanSamplerState &vulkanSamplerState(VernonDeviceSampler &sampler) {
    return runtimeBackendState<VulkanSamplerState>(sampler);
}

inline const VulkanSamplerState &vulkanSamplerState(const VernonDeviceSampler &sampler) {
    return runtimeBackendState<VulkanSamplerState>(sampler);
}

struct VulkanKernelState {
    VkShaderModule shader{};
    VkDescriptorSetLayout descriptorSetLayout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline pipeline{};
};
#else
struct VulkanKernelState {};
#endif

bool probeVulkan(std::string &diagnostic);
bool initializeVulkanContext(VernonRuntimeContext &context, uint32_t deviceIndex);
void destroyVulkanContext(VernonRuntimeContext &context);
VernonStatus synchronizeVulkan(VernonRuntimeContext &context);

#if defined(VERNON_HAS_VULKAN_RUNTIME)
bool createVulkanBuffer(VernonRuntimeContext &context, VkDeviceSize size, VkBuffer &buffer, VkDeviceMemory &memory);
bool createVulkanShaderModule(VernonRuntimeContext &context, const void *data, size_t size, VkShaderModule &module,
                              uint32_t pushConstantBaseOffset = 0);
void destroyVulkanShaderModule(VernonRuntimeContext &context, VkShaderModule &module);
void transitionVulkanImageLayout(VkCommandBuffer command, VernonDeviceTexture &texture, VkImageLayout newLayout);
bool beginVulkanCommands(VernonRuntimeContext &context, VkCommandBuffer &command, std::string &error);
bool submitVulkanCommands(VernonRuntimeContext &context, VkCommandBuffer command, std::string &error);
#endif
bool createVulkanBuffer(VernonDeviceBuffer &buffer);
void destroyVulkanBuffer(VernonDeviceBuffer &buffer);
VernonStatus copyToVulkanBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromVulkanBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

bool createVulkanTexture(VernonDeviceTexture &texture);
void destroyVulkanTexture(VernonDeviceTexture &texture);
VernonStatus copyToVulkanTexture(VernonDeviceTexture &texture, const void *source, size_t size);
VernonStatus copyFromVulkanTexture(const VernonDeviceTexture &texture, void *destination, size_t size);
bool createVulkanSampler(VernonDeviceSampler &sampler);
void destroyVulkanSampler(VernonDeviceSampler &sampler);

bool loadVulkanKernel(VernonRuntimeContext &context, const void *artifact, size_t artifactSize, const char *reflection,
                      size_t reflectionSize, const char *entry, size_t entrySize, VulkanKernelState &state,
                      ReflectedEntry &metadata);
void destroyVulkanKernel(VernonRuntimeContext &context, VulkanKernelState &state);
VernonStatus launchVulkanKernel(VernonRuntimeContext &context, const VulkanKernelState &state,
                                const ReflectedEntry &metadata, VernonLaunchSize globalSize,
                                const VernonLaunchArgument *arguments, size_t argumentCount);

} // namespace vernon::runtime

#endif
