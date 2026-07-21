#ifndef VERNON_RUNTIME_BACKEND_VULKAN_DRIVER_H
#define VERNON_RUNTIME_BACKEND_VULKAN_DRIVER_H

#include "platform_library.h"

#include <vulkan/vulkan.h>

#include <mutex>
#include <string>

namespace vernon::runtime {

struct VulkanDriver {
  bool load();
  bool loadInstance(VkInstance instance);
  bool loadDevice(VkDevice device);

  PlatformLibrary library;
  std::mutex mutex;
  std::string error;
  bool attempted{};
  bool available{};
  PFN_vkGetInstanceProcAddr getInstanceProcAddr{};
  PFN_vkCreateInstance createInstance{};
  PFN_vkDestroyInstance destroyInstance{};
  PFN_vkEnumeratePhysicalDevices enumeratePhysicalDevices{};
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      getPhysicalDeviceQueueFamilyProperties{};
  PFN_vkGetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties{};
  PFN_vkGetPhysicalDeviceFormatProperties getPhysicalDeviceFormatProperties{};
  PFN_vkCreateDevice createDevice{};
  PFN_vkGetDeviceProcAddr getDeviceProcAddr{};
  PFN_vkDestroyDevice destroyDevice{};
  PFN_vkDeviceWaitIdle deviceWaitIdle{};
  PFN_vkGetDeviceQueue getDeviceQueue{};
  PFN_vkCreateCommandPool createCommandPool{};
  PFN_vkDestroyCommandPool destroyCommandPool{};
  PFN_vkAllocateCommandBuffers allocateCommandBuffers{};
  PFN_vkFreeCommandBuffers freeCommandBuffers{};
  PFN_vkBeginCommandBuffer beginCommandBuffer{};
  PFN_vkEndCommandBuffer endCommandBuffer{};
  PFN_vkQueueSubmit queueSubmit{};
  PFN_vkQueueWaitIdle queueWaitIdle{};
  PFN_vkCreateBuffer createBuffer{};
  PFN_vkDestroyBuffer destroyBuffer{};
  PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements{};
  PFN_vkAllocateMemory allocateMemory{};
  PFN_vkFreeMemory freeMemory{};
  PFN_vkBindBufferMemory bindBufferMemory{};
  PFN_vkMapMemory mapMemory{};
  PFN_vkUnmapMemory unmapMemory{};
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
  PFN_vkAllocateDescriptorSets allocateDescriptorSets{};
  PFN_vkUpdateDescriptorSets updateDescriptorSets{};
  PFN_vkCmdBindPipeline cmdBindPipeline{};
  PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets{};
  PFN_vkCmdDispatch cmdDispatch{};
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
  PFN_vkCmdBeginRenderPass cmdBeginRenderPass{};
  PFN_vkCmdEndRenderPass cmdEndRenderPass{};
  PFN_vkCmdSetViewport cmdSetViewport{};
  PFN_vkCmdSetScissor cmdSetScissor{};
  PFN_vkCmdBindVertexBuffers cmdBindVertexBuffers{};
  PFN_vkCmdBindIndexBuffer cmdBindIndexBuffer{};
  PFN_vkCmdDraw cmdDraw{};
  PFN_vkCmdDrawIndexed cmdDrawIndexed{};
  PFN_vkCmdPushConstants cmdPushConstants{};
};

VulkanDriver &vulkanDriver();

} // namespace vernon::runtime

#endif
