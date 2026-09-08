#include "vulkan_driver.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace vernon::rhi::vulkan {
namespace {

void appendCandidate(std::vector<std::string> &candidates, std::filesystem::path candidate) {
    if (candidate.empty())
        return;
    const std::string path = candidate.u8string();
    if (std::find(candidates.begin(), candidates.end(), path) == candidates.end())
        candidates.push_back(path);
}

std::vector<std::string> loaderCandidates() {
    std::vector<std::string> candidates;
    if (const char *overridePath = std::getenv("VERNON_VULKAN_LOADER"))
        appendCandidate(candidates, overridePath);

    if (const char *sdk = std::getenv("VULKAN_SDK")) {
#if defined(_WIN32)
        appendCandidate(candidates, std::filesystem::path(sdk) / "Bin" / "vulkan-1.dll");
#elif defined(__APPLE__)
        appendCandidate(candidates, std::filesystem::path(sdk) / "lib" / "libvulkan.1.dylib");
        appendCandidate(candidates, std::filesystem::path(sdk) / "macOS" / "lib" / "libvulkan.1.dylib");
#else
        appendCandidate(candidates, std::filesystem::path(sdk) / "lib" / "libvulkan.so.1");
#endif
    }

#if defined(_WIN32)
    appendCandidate(candidates, "vulkan-1.dll");
#elif defined(__APPLE__)
    appendCandidate(candidates, "libvulkan.1.dylib");
    if (const char *homebrew = std::getenv("HOMEBREW_PREFIX"))
        appendCandidate(candidates,
                        std::filesystem::path(homebrew) / "opt" / "vulkan-loader" / "lib" / "libvulkan.1.dylib");
    appendCandidate(candidates, "/opt/homebrew/opt/vulkan-loader/lib/libvulkan.1.dylib");
    appendCandidate(candidates, "/usr/local/opt/vulkan-loader/lib/libvulkan.1.dylib");
    appendCandidate(candidates, "/usr/local/lib/libvulkan.1.dylib");
#else
    appendCandidate(candidates, "libvulkan.so.1");
#endif
    return candidates;
}

bool openLoader(platform::PlatformLibrary &library, std::string &error) {
    std::string failures;
    for (const std::string &candidate : loaderCandidates()) {
        std::string candidateError;
        if (library.open(candidate.c_str(), candidateError))
            return true;
        if (!failures.empty())
            failures += "; ";
        failures += candidate + ": " + candidateError;
    }
    error = "Vulkan loader was not found; attempted " + failures;
    return false;
}

} // namespace

bool Driver::load() {
    std::lock_guard<std::mutex> guard(mutex);
    if (attempted)
        return available;
    attempted = true;
    if (!openLoader(library, error))
        return false;
    getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(library.symbol("vkGetInstanceProcAddr"));
    if (!getInstanceProcAddr) {
        error = "Vulkan loader is missing vkGetInstanceProcAddr";
        return false;
    }
    enumerateInstanceExtensionProperties = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        getInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    if (!enumerateInstanceExtensionProperties) {
        error = "Vulkan loader is missing vkEnumerateInstanceExtensionProperties";
        return false;
    }
    createInstance = reinterpret_cast<PFN_vkCreateInstance>(getInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!createInstance) {
        error = "Vulkan loader is missing vkCreateInstance";
        return false;
    }
    available = true;
    return true;
}

bool Driver::loadInstance(VkInstance instance) {
#define VERNON_LOAD_VULKAN_INSTANCE(member, symbol)                                                                    \
    member = reinterpret_cast<decltype(member)>(getInstanceProcAddr(instance, symbol));                                \
    if (!member) {                                                                                                     \
        error = std::string("Vulkan loader is missing ") + symbol;                                                     \
        return false;                                                                                                  \
    }
    VERNON_LOAD_VULKAN_INSTANCE(destroyInstance, "vkDestroyInstance");
    VERNON_LOAD_VULKAN_INSTANCE(enumeratePhysicalDevices, "vkEnumeratePhysicalDevices");
    VERNON_LOAD_VULKAN_INSTANCE(getPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties");
    VERNON_LOAD_VULKAN_INSTANCE(getPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
    VERNON_LOAD_VULKAN_INSTANCE(getPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
    VERNON_LOAD_VULKAN_INSTANCE(getPhysicalDeviceFormatProperties, "vkGetPhysicalDeviceFormatProperties");
    VERNON_LOAD_VULKAN_INSTANCE(getPhysicalDeviceFeatures2, "vkGetPhysicalDeviceFeatures2");
    VERNON_LOAD_VULKAN_INSTANCE(enumerateDeviceExtensionProperties, "vkEnumerateDeviceExtensionProperties");
    VERNON_LOAD_VULKAN_INSTANCE(createDevice, "vkCreateDevice");
    VERNON_LOAD_VULKAN_INSTANCE(getDeviceProcAddr, "vkGetDeviceProcAddr");
#undef VERNON_LOAD_VULKAN_INSTANCE
    return true;
}

bool Driver::loadDevice(VkDevice device) {
#define VERNON_LOAD_VULKAN_DEVICE(member, symbol)                                                                      \
    member = reinterpret_cast<decltype(member)>(getDeviceProcAddr(device, symbol));                                    \
    if (!member) {                                                                                                     \
        error = std::string("Vulkan device is missing ") + symbol;                                                     \
        return false;                                                                                                  \
    }
    VERNON_LOAD_VULKAN_DEVICE(destroyDevice, "vkDestroyDevice");
    VERNON_LOAD_VULKAN_DEVICE(deviceWaitIdle, "vkDeviceWaitIdle");
    VERNON_LOAD_VULKAN_DEVICE(getDeviceQueue, "vkGetDeviceQueue");
    VERNON_LOAD_VULKAN_DEVICE(createCommandPool, "vkCreateCommandPool");
    VERNON_LOAD_VULKAN_DEVICE(destroyCommandPool, "vkDestroyCommandPool");
    VERNON_LOAD_VULKAN_DEVICE(allocateCommandBuffers, "vkAllocateCommandBuffers");
    VERNON_LOAD_VULKAN_DEVICE(freeCommandBuffers, "vkFreeCommandBuffers");
    VERNON_LOAD_VULKAN_DEVICE(resetCommandBuffer, "vkResetCommandBuffer");
    VERNON_LOAD_VULKAN_DEVICE(beginCommandBuffer, "vkBeginCommandBuffer");
    VERNON_LOAD_VULKAN_DEVICE(endCommandBuffer, "vkEndCommandBuffer");
    VERNON_LOAD_VULKAN_DEVICE(queueSubmit, "vkQueueSubmit");
    VERNON_LOAD_VULKAN_DEVICE(queueWaitIdle, "vkQueueWaitIdle");
    VERNON_LOAD_VULKAN_DEVICE(createFence, "vkCreateFence");
    VERNON_LOAD_VULKAN_DEVICE(destroyFence, "vkDestroyFence");
    VERNON_LOAD_VULKAN_DEVICE(waitForFences, "vkWaitForFences");
    VERNON_LOAD_VULKAN_DEVICE(resetFences, "vkResetFences");
    VERNON_LOAD_VULKAN_DEVICE(createBuffer, "vkCreateBuffer");
    VERNON_LOAD_VULKAN_DEVICE(destroyBuffer, "vkDestroyBuffer");
    VERNON_LOAD_VULKAN_DEVICE(getBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
    VERNON_LOAD_VULKAN_DEVICE(allocateMemory, "vkAllocateMemory");
    VERNON_LOAD_VULKAN_DEVICE(freeMemory, "vkFreeMemory");
    VERNON_LOAD_VULKAN_DEVICE(bindBufferMemory, "vkBindBufferMemory");
    VERNON_LOAD_VULKAN_DEVICE(mapMemory, "vkMapMemory");
    VERNON_LOAD_VULKAN_DEVICE(unmapMemory, "vkUnmapMemory");
    VERNON_LOAD_VULKAN_DEVICE(cmdCopyBuffer, "vkCmdCopyBuffer");
    VERNON_LOAD_VULKAN_DEVICE(createShaderModule, "vkCreateShaderModule");
    VERNON_LOAD_VULKAN_DEVICE(destroyShaderModule, "vkDestroyShaderModule");
    VERNON_LOAD_VULKAN_DEVICE(createDescriptorSetLayout, "vkCreateDescriptorSetLayout");
    VERNON_LOAD_VULKAN_DEVICE(destroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
    VERNON_LOAD_VULKAN_DEVICE(createPipelineLayout, "vkCreatePipelineLayout");
    VERNON_LOAD_VULKAN_DEVICE(destroyPipelineLayout, "vkDestroyPipelineLayout");
    VERNON_LOAD_VULKAN_DEVICE(createComputePipelines, "vkCreateComputePipelines");
    VERNON_LOAD_VULKAN_DEVICE(destroyPipeline, "vkDestroyPipeline");
    VERNON_LOAD_VULKAN_DEVICE(createDescriptorPool, "vkCreateDescriptorPool");
    VERNON_LOAD_VULKAN_DEVICE(destroyDescriptorPool, "vkDestroyDescriptorPool");
    VERNON_LOAD_VULKAN_DEVICE(allocateDescriptorSets, "vkAllocateDescriptorSets");
    VERNON_LOAD_VULKAN_DEVICE(freeDescriptorSets, "vkFreeDescriptorSets");
    VERNON_LOAD_VULKAN_DEVICE(updateDescriptorSets, "vkUpdateDescriptorSets");
    VERNON_LOAD_VULKAN_DEVICE(cmdBindPipeline, "vkCmdBindPipeline");
    VERNON_LOAD_VULKAN_DEVICE(cmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    VERNON_LOAD_VULKAN_DEVICE(cmdDispatch, "vkCmdDispatch");
    VERNON_LOAD_VULKAN_DEVICE(createImage, "vkCreateImage");
    VERNON_LOAD_VULKAN_DEVICE(destroyImage, "vkDestroyImage");
    VERNON_LOAD_VULKAN_DEVICE(getImageMemoryRequirements, "vkGetImageMemoryRequirements");
    VERNON_LOAD_VULKAN_DEVICE(bindImageMemory, "vkBindImageMemory");
    VERNON_LOAD_VULKAN_DEVICE(createImageView, "vkCreateImageView");
    VERNON_LOAD_VULKAN_DEVICE(destroyImageView, "vkDestroyImageView");
    VERNON_LOAD_VULKAN_DEVICE(createSampler, "vkCreateSampler");
    VERNON_LOAD_VULKAN_DEVICE(destroySampler, "vkDestroySampler");
    VERNON_LOAD_VULKAN_DEVICE(createRenderPass, "vkCreateRenderPass");
    VERNON_LOAD_VULKAN_DEVICE(destroyRenderPass, "vkDestroyRenderPass");
    VERNON_LOAD_VULKAN_DEVICE(createFramebuffer, "vkCreateFramebuffer");
    VERNON_LOAD_VULKAN_DEVICE(destroyFramebuffer, "vkDestroyFramebuffer");
    VERNON_LOAD_VULKAN_DEVICE(createGraphicsPipelines, "vkCreateGraphicsPipelines");
    VERNON_LOAD_VULKAN_DEVICE(cmdPipelineBarrier, "vkCmdPipelineBarrier");
    VERNON_LOAD_VULKAN_DEVICE(cmdCopyBufferToImage, "vkCmdCopyBufferToImage");
    VERNON_LOAD_VULKAN_DEVICE(cmdCopyImageToBuffer, "vkCmdCopyImageToBuffer");
    VERNON_LOAD_VULKAN_DEVICE(cmdCopyImage, "vkCmdCopyImage");
    VERNON_LOAD_VULKAN_DEVICE(cmdBlitImage, "vkCmdBlitImage");
    VERNON_LOAD_VULKAN_DEVICE(cmdBeginRenderPass, "vkCmdBeginRenderPass");
    VERNON_LOAD_VULKAN_DEVICE(cmdEndRenderPass, "vkCmdEndRenderPass");
    VERNON_LOAD_VULKAN_DEVICE(cmdClearAttachments, "vkCmdClearAttachments");
    VERNON_LOAD_VULKAN_DEVICE(cmdSetViewport, "vkCmdSetViewport");
    VERNON_LOAD_VULKAN_DEVICE(cmdSetScissor, "vkCmdSetScissor");
    VERNON_LOAD_VULKAN_DEVICE(cmdSetStencilReference, "vkCmdSetStencilReference");
    VERNON_LOAD_VULKAN_DEVICE(cmdBindVertexBuffers, "vkCmdBindVertexBuffers");
    VERNON_LOAD_VULKAN_DEVICE(cmdBindIndexBuffer, "vkCmdBindIndexBuffer");
    VERNON_LOAD_VULKAN_DEVICE(cmdDraw, "vkCmdDraw");
    VERNON_LOAD_VULKAN_DEVICE(cmdDrawIndexed, "vkCmdDrawIndexed");
    VERNON_LOAD_VULKAN_DEVICE(cmdPushConstants, "vkCmdPushConstants");
#undef VERNON_LOAD_VULKAN_DEVICE
    cmdBeginRendering = reinterpret_cast<PFN_vkCmdBeginRendering>(getDeviceProcAddr(device, "vkCmdBeginRendering"));
    cmdEndRendering = reinterpret_cast<PFN_vkCmdEndRendering>(getDeviceProcAddr(device, "vkCmdEndRendering"));
    if (!cmdBeginRendering || !cmdEndRendering) {
        cmdBeginRendering =
            reinterpret_cast<PFN_vkCmdBeginRendering>(getDeviceProcAddr(device, "vkCmdBeginRenderingKHR"));
        cmdEndRendering = reinterpret_cast<PFN_vkCmdEndRendering>(getDeviceProcAddr(device, "vkCmdEndRenderingKHR"));
    }
    return true;
}

Driver &driver() {
    static Driver result;
    return result;
}

} // namespace vernon::rhi::vulkan
