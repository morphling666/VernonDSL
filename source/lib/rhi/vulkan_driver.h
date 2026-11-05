#ifndef VERNON_RHI_VULKAN_DRIVER_H
#define VERNON_RHI_VULKAN_DRIVER_H

#include "VernonRHI.h"
#include "platform/platform_library.h"

#include <vulkan/vulkan.h>

#include <mutex>
#include <string>

namespace vernon::rhi::vulkan {

struct VERNON_RHI_CAPI Driver {
    bool load();
    bool loadInstance(VkInstance instance);
    bool loadDevice(VkDevice device);

    platform::PlatformLibrary library;
    std::mutex mutex;
    std::string error;
    bool attempted{};
    bool available{};
    PFN_vkGetInstanceProcAddr getInstanceProcAddr{};
    PFN_vkEnumerateInstanceExtensionProperties enumerateInstanceExtensionProperties{};
    PFN_vkCreateInstance createInstance{};
    PFN_vkDestroyInstance destroyInstance{};
    PFN_vkEnumeratePhysicalDevices enumeratePhysicalDevices{};
    PFN_vkGetPhysicalDeviceQueueFamilyProperties getPhysicalDeviceQueueFamilyProperties{};
    PFN_vkGetPhysicalDeviceProperties getPhysicalDeviceProperties{};
    PFN_vkGetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties{};
    PFN_vkGetPhysicalDeviceFormatProperties getPhysicalDeviceFormatProperties{};
    PFN_vkGetPhysicalDeviceImageFormatProperties getPhysicalDeviceImageFormatProperties{};
    PFN_vkGetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2{};
    PFN_vkEnumerateDeviceExtensionProperties enumerateDeviceExtensionProperties{};
    PFN_vkCreateDevice createDevice{};
    PFN_vkGetDeviceProcAddr getDeviceProcAddr{};
    PFN_vkDestroyDevice destroyDevice{};
    PFN_vkDeviceWaitIdle deviceWaitIdle{};
    PFN_vkGetDeviceQueue getDeviceQueue{};
    PFN_vkCreateCommandPool createCommandPool{};
    PFN_vkDestroyCommandPool destroyCommandPool{};
    PFN_vkAllocateCommandBuffers allocateCommandBuffers{};
    PFN_vkFreeCommandBuffers freeCommandBuffers{};
    PFN_vkResetCommandBuffer resetCommandBuffer{};
    PFN_vkBeginCommandBuffer beginCommandBuffer{};
    PFN_vkEndCommandBuffer endCommandBuffer{};
    PFN_vkQueueSubmit queueSubmit{};
    PFN_vkQueueWaitIdle queueWaitIdle{};
    PFN_vkCreateFence createFence{};
    PFN_vkDestroyFence destroyFence{};
    PFN_vkWaitForFences waitForFences{};
    PFN_vkResetFences resetFences{};
    PFN_vkGetFenceStatus getFenceStatus{};
    PFN_vkCreateBuffer createBuffer{};
    PFN_vkDestroyBuffer destroyBuffer{};
    PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements{};
    PFN_vkAllocateMemory allocateMemory{};
    PFN_vkFreeMemory freeMemory{};
    PFN_vkBindBufferMemory bindBufferMemory{};
    PFN_vkMapMemory mapMemory{};
    PFN_vkUnmapMemory unmapMemory{};
    PFN_vkCmdCopyBuffer cmdCopyBuffer{};
    PFN_vkCreateShaderModule createShaderModule{};
    PFN_vkDestroyShaderModule destroyShaderModule{};
    PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout{};
    PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout{};
    PFN_vkCreatePipelineLayout createPipelineLayout{};
    PFN_vkDestroyPipelineLayout destroyPipelineLayout{};
    PFN_vkCreateComputePipelines createComputePipelines{};
    PFN_vkDestroyPipeline destroyPipeline{};
    PFN_vkCreateDescriptorPool createDescriptorPool{};
    PFN_vkDestroyDescriptorPool destroyDescriptorPool{};
    PFN_vkResetDescriptorPool resetDescriptorPool{};
    PFN_vkAllocateDescriptorSets allocateDescriptorSets{};
    PFN_vkFreeDescriptorSets freeDescriptorSets{};
    PFN_vkUpdateDescriptorSets updateDescriptorSets{};
    PFN_vkCmdBindPipeline cmdBindPipeline{};
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets{};
    PFN_vkCmdDispatch cmdDispatch{};
    PFN_vkCmdDispatchIndirect cmdDispatchIndirect{};
    PFN_vkCreateImage createImage{};
    PFN_vkDestroyImage destroyImage{};
    PFN_vkGetImageMemoryRequirements getImageMemoryRequirements{};
    PFN_vkBindImageMemory bindImageMemory{};
    PFN_vkCreateImageView createImageView{};
    PFN_vkDestroyImageView destroyImageView{};
    PFN_vkCreateSampler createSampler{};
    PFN_vkDestroySampler destroySampler{};
    PFN_vkCreateRenderPass createRenderPass{};
    PFN_vkDestroyRenderPass destroyRenderPass{};
    PFN_vkCreateFramebuffer createFramebuffer{};
    PFN_vkDestroyFramebuffer destroyFramebuffer{};
    PFN_vkCreateGraphicsPipelines createGraphicsPipelines{};
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier{};
    PFN_vkCmdCopyBufferToImage cmdCopyBufferToImage{};
    PFN_vkCmdCopyImageToBuffer cmdCopyImageToBuffer{};
    PFN_vkCmdCopyImage cmdCopyImage{};
    PFN_vkCmdBlitImage cmdBlitImage{};
    PFN_vkCmdBeginRenderPass cmdBeginRenderPass{};
    PFN_vkCmdEndRenderPass cmdEndRenderPass{};
    PFN_vkCmdClearAttachments cmdClearAttachments{};
    PFN_vkCmdBeginRendering cmdBeginRendering{};
    PFN_vkCmdEndRendering cmdEndRendering{};
    PFN_vkCmdSetViewport cmdSetViewport{};
    PFN_vkCmdSetScissor cmdSetScissor{};
    PFN_vkCmdSetStencilReference cmdSetStencilReference{};
    PFN_vkCmdBindVertexBuffers cmdBindVertexBuffers{};
    PFN_vkCmdBindIndexBuffer cmdBindIndexBuffer{};
    PFN_vkCmdDraw cmdDraw{};
    PFN_vkCmdDrawIndexed cmdDrawIndexed{};
    PFN_vkCmdDrawIndexedIndirect cmdDrawIndexedIndirect{};
    // Null unless the device enabled VK_KHR_draw_indirect_count.
    PFN_vkCmdDrawIndexedIndirectCount cmdDrawIndexedIndirectCount{};
    PFN_vkCmdPushConstants cmdPushConstants{};
    PFN_vkCreateQueryPool createQueryPool{};
    PFN_vkDestroyQueryPool destroyQueryPool{};
    PFN_vkCmdResetQueryPool cmdResetQueryPool{};
    PFN_vkCmdWriteTimestamp cmdWriteTimestamp{};
    PFN_vkGetQueryPoolResults getQueryPoolResults{};
    PFN_vkCreateSemaphore createSemaphore{};
    PFN_vkDestroySemaphore destroySemaphore{};
    // Null unless the instance enabled VK_KHR_surface.
    PFN_vkDestroySurfaceKHR destroySurface{};
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR getPhysicalDeviceSurfaceSupport{};
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR getPhysicalDeviceSurfaceCapabilities{};
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR getPhysicalDeviceSurfaceFormats{};
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR getPhysicalDeviceSurfacePresentModes{};
    // The platform's surface constructor, such as vkCreateWin32SurfaceKHR; null unless the instance enabled
    // `platformSurfaceExtension`. Its type needs the platform's headers, which only createPlatformSurface includes.
    PFN_vkVoidFunction createPlatformSurface{};
    // Null unless the device enabled VK_KHR_swapchain.
    PFN_vkCreateSwapchainKHR createSwapchain{};
    PFN_vkDestroySwapchainKHR destroySwapchain{};
    PFN_vkGetSwapchainImagesKHR getSwapchainImages{};
    PFN_vkAcquireNextImageKHR acquireNextImage{};
    PFN_vkQueuePresentKHR queuePresent{};
};

VERNON_RHI_CAPI Driver &driver();

// The instance extension that creates surfaces for the platform's windows, or null where the RHI presents none.
#if defined(_WIN32)
inline constexpr const char *platformSurfaceExtension = "VK_KHR_win32_surface";
#else
inline constexpr const char *platformSurfaceExtension = nullptr;
#endif

// Creates a surface for the platform window `descriptor` names, with the instance's platform surface extension.
VERNON_RHI_CAPI bool createPlatformSurface(VkInstance instance, const VernonRhiSurfaceDescriptor &descriptor,
                                           VkSurfaceKHR &surface, std::string &error);

} // namespace vernon::rhi::vulkan

#endif
