#include "hal/vulkan/vulkan_state.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::hal::vulkan {
namespace {

bool check(VkResult result, const char *operation, std::string &error) {
    if (result == VK_SUCCESS)
        return true;
    error = std::string(operation) + " failed with Vulkan error " + std::to_string(static_cast<int>(result));
    return false;
}

bool hasExtension(const std::vector<VkExtensionProperties> &extensions, std::string_view name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties &extension) {
        return std::string_view(extension.extensionName) == name;
    });
}

} // namespace

uint32_t physicalDeviceTypeRank(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return 0;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return 1;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return 3;
    default:
        return 4;
    }
}

DeviceState::~DeviceState() { shutdown(); }

bool DeviceState::initialize(uint32_t deviceIndex, std::string &error) {
    shutdown();
    static std::mutex initializationMutex;
    std::lock_guard<std::mutex> guard(initializationMutex);
    Driver &api = driver();
    if (!api.load()) {
        error = api.error;
        return false;
    }
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "VernonHAL";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.pEngineName = "Vernon";
    application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_1;
    uint32_t instanceExtensionCount = 0;
    if (!check(api.enumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr),
               "vkEnumerateInstanceExtensionProperties", error))
        return false;
    std::vector<VkExtensionProperties> instanceExtensionProperties(instanceExtensionCount);
    if (instanceExtensionCount && !check(api.enumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount,
                                                                                  instanceExtensionProperties.data()),
                                         "vkEnumerateInstanceExtensionProperties", error))
        return false;
    std::vector<const char *> instanceExtensions;
    portabilityEnumeration = hasExtension(instanceExtensionProperties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (portabilityEnumeration)
        instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    const bool debugUtils = hasExtension(instanceExtensionProperties, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugUtils)
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    const bool surfaces = platformSurfaceExtension &&
                          hasExtension(instanceExtensionProperties, VK_KHR_SURFACE_EXTENSION_NAME) &&
                          hasExtension(instanceExtensionProperties, platformSurfaceExtension);
    if (surfaces) {
        instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        instanceExtensions.push_back(platformSurfaceExtension);
    }
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &application;
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.empty() ? nullptr : instanceExtensions.data();
    if (portabilityEnumeration)
        instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    if (!check(api.createInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance", error))
        return false;
    if (!api.loadInstance(instance)) {
        error = api.error;
        shutdown();
        return false;
    }
    if (debugUtils) {
        cmdBeginDebugUtilsLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            api.getInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
        cmdEndDebugUtilsLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            api.getInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
        cmdInsertDebugUtilsLabel = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
            api.getInstanceProcAddr(instance, "vkCmdInsertDebugUtilsLabelEXT"));
        if (!cmdBeginDebugUtilsLabel || !cmdEndDebugUtilsLabel || !cmdInsertDebugUtilsLabel) {
            cmdBeginDebugUtilsLabel = nullptr;
            cmdEndDebugUtilsLabel = nullptr;
            cmdInsertDebugUtilsLabel = nullptr;
        }
    }
    uint32_t deviceCount = 0;
    if (!check(api.enumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices", error)) {
        shutdown();
        return false;
    }
    if (!deviceCount) {
        error = "Vulkan loader found no physical devices";
        shutdown();
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (!check(api.enumeratePhysicalDevices(instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices",
               error)) {
        shutdown();
        return false;
    }
    struct DeviceCandidate {
        VkPhysicalDevice device{};
        VkPhysicalDeviceProperties properties{};
        VkDeviceSize deviceLocalMemory{};
        uint32_t queueFamily{};
        uint32_t timestampValidBits{};
    };
    std::vector<DeviceCandidate> candidates;
    candidates.reserve(devices.size());
    for (VkPhysicalDevice candidateDevice : devices) {
        uint32_t candidateQueueCount = 0;
        api.getPhysicalDeviceQueueFamilyProperties(candidateDevice, &candidateQueueCount, nullptr);
        std::vector<VkQueueFamilyProperties> candidateQueues(candidateQueueCount);
        api.getPhysicalDeviceQueueFamilyProperties(candidateDevice, &candidateQueueCount, candidateQueues.data());
        const auto candidateQueue =
            std::find_if(candidateQueues.begin(), candidateQueues.end(), [](const VkQueueFamilyProperties &family) {
                return (family.queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) ==
                       (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
            });
        if (candidateQueue == candidateQueues.end())
            continue;
        DeviceCandidate candidate;
        candidate.device = candidateDevice;
        candidate.queueFamily = static_cast<uint32_t>(std::distance(candidateQueues.begin(), candidateQueue));
        candidate.timestampValidBits = candidateQueue->timestampValidBits;
        api.getPhysicalDeviceProperties(candidateDevice, &candidate.properties);
        VkPhysicalDeviceMemoryProperties candidateMemory{};
        api.getPhysicalDeviceMemoryProperties(candidateDevice, &candidateMemory);
        for (uint32_t heapIndex = 0; heapIndex < candidateMemory.memoryHeapCount; ++heapIndex)
            if (candidateMemory.memoryHeaps[heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                candidate.deviceLocalMemory += candidateMemory.memoryHeaps[heapIndex].size;
        candidates.push_back(candidate);
    }
    std::stable_sort(
        candidates.begin(), candidates.end(), [](const DeviceCandidate &left, const DeviceCandidate &right) {
            const uint32_t leftRank = physicalDeviceTypeRank(left.properties.deviceType);
            const uint32_t rightRank = physicalDeviceTypeRank(right.properties.deviceType);
            return leftRank != rightRank ? leftRank < rightRank : left.deviceLocalMemory > right.deviceLocalMemory;
        });
    if (deviceIndex >= candidates.size()) {
        error = "Vulkan high-performance device index is unavailable";
        shutdown();
        return false;
    }
    physicalDevice = candidates[deviceIndex].device;
    queueFamily = candidates[deviceIndex].queueFamily;
    timestampValidBits = candidates[deviceIndex].timestampValidBits;
    const VkPhysicalDeviceProperties &properties = candidates[deviceIndex].properties;
    timestampPeriod = properties.limits.timestampPeriod;
    maxPushConstantsSize = properties.limits.maxPushConstantsSize;
    maxVertexInputAttributes = properties.limits.maxVertexInputAttributes;
    apiVersion = properties.apiVersion;
    descriptorBufferOffsetAlignment = (std::max)(properties.limits.minUniformBufferOffsetAlignment,
                                                 properties.limits.minStorageBufferOffsetAlignment);
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    uint32_t extensionCount = 0;
    if (!check(api.enumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr),
               "vkEnumerateDeviceExtensionProperties", error)) {
        shutdown();
        return false;
    }
    std::vector<VkExtensionProperties> extensions(extensionCount);
    if (extensionCount &&
        !check(api.enumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data()),
               "vkEnumerateDeviceExtensionProperties", error)) {
        shutdown();
        return false;
    }
    const bool dynamicRenderingIsCore =
        VK_API_VERSION_MAJOR(apiVersion) > 1 ||
        (VK_API_VERSION_MAJOR(apiVersion) == 1 && VK_API_VERSION_MINOR(apiVersion) >= 3);
    const bool dynamicRenderingDependenciesAreCore =
        VK_API_VERSION_MAJOR(apiVersion) > 1 ||
        (VK_API_VERSION_MAJOR(apiVersion) == 1 && VK_API_VERSION_MINOR(apiVersion) >= 2);
    const bool hasDynamicRenderingExtension = hasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    portabilitySubset = hasExtension(extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    const bool hasShaderAtomicFloat = hasExtension(extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portabilitySubsetFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shaderAtomicFloatFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    const bool canUseDynamicRendering =
        dynamicRenderingIsCore || (dynamicRenderingDependenciesAreCore && hasDynamicRenderingExtension);
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void *queriedFeatureChain = nullptr;
    if (hasShaderAtomicFloat) {
        shaderAtomicFloatFeatures.pNext = queriedFeatureChain;
        queriedFeatureChain = &shaderAtomicFloatFeatures;
    }
    if (portabilitySubset) {
        portabilitySubsetFeatures.pNext = queriedFeatureChain;
        queriedFeatureChain = &portabilitySubsetFeatures;
    }
    if (canUseDynamicRendering) {
        dynamicRenderingFeatures.pNext = queriedFeatureChain;
        queriedFeatureChain = &dynamicRenderingFeatures;
    }
    features.pNext = queriedFeatureChain;
    api.getPhysicalDeviceFeatures2(physicalDevice, &features);
    shaderBufferFloat32AtomicAdd =
        hasShaderAtomicFloat && shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd == VK_TRUE;
    if (canUseDynamicRendering)
        dynamicRendering = dynamicRenderingFeatures.dynamicRendering == VK_TRUE;
    imageCubeArray = features.features.imageCubeArray == VK_TRUE;
    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.imageCubeArray = features.features.imageCubeArray;
    multiDrawIndirect =
        features.features.multiDrawIndirect == VK_TRUE && features.features.drawIndirectFirstInstance == VK_TRUE;
    enabledFeatures.multiDrawIndirect = multiDrawIndirect ? VK_TRUE : VK_FALSE;
    enabledFeatures.drawIndirectFirstInstance = multiDrawIndirect ? VK_TRUE : VK_FALSE;
    deviceInfo.pEnabledFeatures = &enabledFeatures;
    std::vector<const char *> deviceExtensions;
    void *enabledFeatureChain = nullptr;
    if (shaderBufferFloat32AtomicAdd) {
        shaderAtomicFloatFeatures.pNext = enabledFeatureChain;
        enabledFeatureChain = &shaderAtomicFloatFeatures;
        deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    if (portabilitySubset) {
        portabilitySubsetFeatures.pNext = enabledFeatureChain;
        enabledFeatureChain = &portabilitySubsetFeatures;
        deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
    if (dynamicRendering) {
        dynamicRenderingFeatures.pNext = enabledFeatureChain;
        enabledFeatureChain = &dynamicRenderingFeatures;
        if (!dynamicRenderingIsCore)
            deviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    }
    // The extension rather than Vulkan 1.2, which the 1.1 instance does not admit.
    drawIndirectCount = hasExtension(extensions, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    if (drawIndirectCount)
        deviceExtensions.push_back(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    presentation = surfaces && hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (presentation)
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    deviceInfo.pNext = enabledFeatureChain;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
    if (!check(api.createDevice(physicalDevice, &deviceInfo, nullptr, &device), "vkCreateDevice", error)) {
        shutdown();
        return false;
    }
    if (!api.loadDevice(device)) {
        error = api.error;
        shutdown();
        return false;
    }
    dynamicRendering = dynamicRendering && api.cmdBeginRendering && api.cmdEndRendering;
    drawIndirectCount = drawIndirectCount && api.cmdDrawIndexedIndirectCount;
    presentation = presentation && api.destroySurface && api.getPhysicalDeviceSurfaceSupport &&
                   api.getPhysicalDeviceSurfaceCapabilities && api.getPhysicalDeviceSurfaceFormats &&
                   api.getPhysicalDeviceSurfacePresentModes && api.createPlatformSurface && api.createSwapchain &&
                   api.destroySwapchain && api.getSwapchainImages && api.acquireNextImage && api.queuePresent;
    api.getDeviceQueue(device, queueFamily, 0, &queue);
    api.getPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    CommandFrame initialFrame;
    if (!createCommandFrame(initialFrame, error)) {
        shutdown();
        return false;
    }
    availableCommandFrames.push_back(std::move(initialFrame));
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (!createSampler(defaultImplicitSampler, samplerInfo, error)) {
        shutdown();
        return false;
    }
    return true;
}

void DeviceState::shutdown() {
    Driver &api = driver();
    if (device) {
        if (api.deviceWaitIdle)
            api.deviceWaitIdle(device);
        destroyDeferred(true);
        destroySampler(defaultImplicitSampler);
        for (CommandFrame &frame : availableCommandFrames)
            destroyCommandFrame(frame);
        for (auto &entry : recordingCommandFrames)
            destroyCommandFrame(entry.second);
        for (CommandFrame &frame : inFlightCommandFrames)
            destroyCommandFrame(frame);
        for (StagingPool<StagingChunk> *pool : {&uploadStaging, &readbackStaging})
            pool->clear([this](StagingChunk &chunk) { destroyBuffer(chunk.buffer); });
        if (api.destroyDevice)
            api.destroyDevice(device, nullptr);
    }
    availableCommandFrames.clear();
    recordingCommandFrames.clear();
    inFlightCommandFrames.clear();
    pendingUploads = VK_NULL_HANDLE;
    deferredDestructions.clear();
    submittedSerial = 0;
    completedSerial = 0;
    deviceLost = false;
    if (instance && api.destroyInstance)
        api.destroyInstance(instance, nullptr);
    queue = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
    queueFamily = 0;
    timestampValidBits = 0;
    timestampPeriod = 0;
    maxPushConstantsSize = 0;
    maxVertexInputAttributes = 0;
    apiVersion = 0;
    descriptorBufferOffsetAlignment = 1;
    dynamicRendering = false;
    drawIndirectCount = false;
    multiDrawIndirect = false;
    shaderBufferFloat32AtomicAdd = false;
    presentation = false;
    portabilityEnumeration = false;
    portabilitySubset = false;
    imageCubeArray = false;
    cmdBeginDebugUtilsLabel = nullptr;
    cmdEndDebugUtilsLabel = nullptr;
    cmdInsertDebugUtilsLabel = nullptr;
    memoryProperties = {};
}

bool DeviceState::synchronize(std::string &error) {
    if (!flushUploads(error) || !device || !check(driver().deviceWaitIdle(device), "vkDeviceWaitIdle", error))
        return false;
    return retireCommandFrames(submittedSerial, error);
}

bool DeviceState::createCommandFrame(CommandFrame &frame, std::string &error) {
    Driver &api = driver();
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (!check(api.createCommandPool(device, &poolInfo, nullptr, &frame.pool), "vkCreateCommandPool", error))
        return false;
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = frame.pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    if (!check(api.allocateCommandBuffers(device, &allocation, &frame.command), "vkAllocateCommandBuffers", error) ||
        !check(api.createFence(device, &fenceInfo, nullptr, &frame.fence), "vkCreateFence", error)) {
        destroyCommandFrame(frame);
        return false;
    }
    return true;
}

void DeviceState::destroyCommandFrame(CommandFrame &frame) {
    Driver &api = driver();
    retireFrameMemory(frame.upload, frame.readback);
    if (frame.fence && api.destroyFence)
        api.destroyFence(device, frame.fence, nullptr);
    if (frame.pool && api.destroyCommandPool)
        api.destroyCommandPool(device, frame.pool, nullptr);
    frame = {};
}

void DeviceState::recycleCommandFrame(CommandFrame frame) {
    frame.serial = 0;
    frame.imageLayouts.clear();
    retireFrameMemory(frame.upload, frame.readback);
    availableCommandFrames.push_back(std::move(frame));
}

DeviceState::CommandFrame *DeviceState::recordingFrame(VkCommandBuffer command) {
    const auto found = recordingCommandFrames.find(reinterpret_cast<uintptr_t>(command));
    return found == recordingCommandFrames.end() ? nullptr : &found->second;
}

bool DeviceState::moveImageLayout(VkCommandBuffer command, Image &owner, size_t subresource, VkImageLayout layout,
                                  std::optional<VkImageLayout> &from, std::string &error) {
    CommandFrame *frame = recordingFrame(command);
    if (!frame || subresource >= owner.subresourceLayouts.size()) {
        error = "Vulkan image layout change requires an active recording and an image subresource";
        return false;
    }
    try {
        ImageLayoutUse &use = frame->imageLayouts[&owner];
        if (use.current.empty()) {
            use.initial.assign(owner.subresourceLayouts.size(), kUnusedLayout);
            use.current.assign(owner.subresourceLayouts.size(), kUnusedLayout);
        }
        VkImageLayout &current = use.current[subresource];
        if (current == kUnusedLayout)
            use.initial[subresource] = layout;
        from = current != kUnusedLayout ? std::optional<VkImageLayout>{current} : std::nullopt;
        current = layout;
    } catch (const std::bad_alloc &) {
        error = "Vulkan image layout tracking ran out of memory";
        return false;
    }
    return true;
}

bool DeviceState::movedToLayout(VkCommandBuffer command, Image &owner, const ImageSubresources &range,
                                VkImageLayout layout) {
    CommandFrame *frame = recordingFrame(command);
    if (!frame)
        return false;
    const auto found = frame->imageLayouts.find(&owner);
    if (found == frame->imageLayouts.end() || found->second.current.empty())
        return false;
    for (uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.arrayLayerCount; ++layer)
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.mipLevelCount; ++mip) {
            const size_t subresource = mip + size_t{layer} * owner.mipLevels;
            if (subresource >= found->second.current.size() || found->second.current[subresource] != layout)
                return false;
        }
    return true;
}

namespace {

// The barriers that move each subresource `layouts` uses to the layout it begins in, from its layout after the
// commands of `prior`, when given, or else as the image tracks it.
bool layoutBarriers(const std::unordered_map<Image *, DeviceState::ImageLayoutUse> &layouts,
                    const std::unordered_map<Image *, DeviceState::ImageLayoutUse> *prior,
                    std::vector<VkImageMemoryBarrier> &barriers, std::string &error) {
    try {
        for (const auto &[image, use] : layouts) {
            const DeviceState::ImageLayoutUse *before = nullptr;
            if (prior)
                if (const auto found = prior->find(image); found != prior->end())
                    before = &found->second;
            for (size_t subresource = 0; subresource < use.initial.size(); ++subresource) {
                const VkImageLayout layout = use.initial[subresource];
                VkImageLayout current = image->subresourceLayouts[subresource];
                if (before && subresource < before->current.size() &&
                    before->current[subresource] != DeviceState::kUnusedLayout)
                    current = before->current[subresource];
                if (layout == DeviceState::kUnusedLayout || layout == current)
                    continue;
                VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.oldLayout = current;
                barrier.newLayout = layout;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image->image;
                barrier.subresourceRange = {imageAspectMask(image->format),
                                            static_cast<uint32_t>(subresource % image->mipLevels), 1,
                                            static_cast<uint32_t>(subresource / image->mipLevels), 1};
                barriers.push_back(barrier);
            }
        }
    } catch (const std::bad_alloc &) {
        error = "Vulkan layout prologue ran out of memory";
        return false;
    }
    return true;
}

} // namespace

bool DeviceState::recordLayoutPrologue(CommandFrame &frame, bool &recorded, std::string &error) {
    recorded = false;
    std::vector<VkImageMemoryBarrier> barriers;
    if (!layoutBarriers(frame.imageLayouts, nullptr, barriers, error) || !recordPrologue(frame, barriers, error))
        return false;
    recorded = !barriers.empty();
    return true;
}

bool DeviceState::recordPrologue(CommandFrame &frame, const std::vector<VkImageMemoryBarrier> &barriers,
                                 std::string &error) {
    if (barriers.empty())
        return true;
    Driver &api = driver();
    if (!frame.prologue) {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = frame.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        if (!check(api.allocateCommandBuffers(device, &allocation, &frame.prologue), "vkAllocateCommandBuffers", error))
            return false;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(api.resetCommandBuffer(frame.prologue, 0), "vkResetCommandBuffer", error) ||
        !check(api.beginCommandBuffer(frame.prologue, &begin), "vkBeginCommandBuffer", error))
        return false;
    api.cmdPipelineBarrier(frame.prologue, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                           nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
    return check(api.endCommandBuffer(frame.prologue), "vkEndCommandBuffer", error);
}

void DeviceState::commitImageLayouts(CommandFrame &frame) {
    commitLayouts(frame.imageLayouts);
    frame.imageLayouts.clear();
}

void DeviceState::commitLayouts(const std::unordered_map<Image *, ImageLayoutUse> &layouts) {
    for (const auto &[image, use] : layouts) {
        for (size_t subresource = 0; subresource < use.current.size(); ++subresource)
            if (use.current[subresource] != kUnusedLayout)
                image->subresourceLayouts[subresource] = use.current[subresource];
        const VkImageLayout shared = image->subresourceLayouts.front();
        if (std::all_of(image->subresourceLayouts.begin(), image->subresourceLayouts.end(),
                        [shared](VkImageLayout layout) { return layout == shared; }))
            image->layout = shared;
    }
}

void DeviceState::retireFrameMemory(StagingArena<StagingChunk> &upload, StagingArena<StagingChunk> &readback) {
    uploadStaging.retire(upload);
    readbackStaging.retire(readback);
}

namespace {

// Submissions do not wait for each other on the host, so every recording first orders itself after all work recorded
// or submitted before it on the queue. Barriers recorded later only order work within the recording.
void recordQueueOrder(VkCommandBuffer command) {
    VkMemoryBarrier queueOrder{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    queueOrder.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    queueOrder.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    driver().cmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                                &queueOrder, 0, nullptr, 0, nullptr);
}

} // namespace

bool DeviceState::beginCommands(VkCommandBuffer &command, std::string &error) {
    if (deviceLost) {
        error = "Vulkan device was lost";
        return false;
    }
    if (!retireCommandFrames(completedSerial, error))
        return false;
    if (inFlightCommandFrames.size() >= maxFramesInFlight &&
        !waitForSerial(inFlightCommandFrames.front().serial, error))
        return false;
    Driver &api = driver();
    CommandFrame frame;
    if (!availableCommandFrames.empty()) {
        frame = std::move(availableCommandFrames.back());
        availableCommandFrames.pop_back();
    } else if (!createCommandFrame(frame, error)) {
        return false;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(api.resetFences(device, 1, &frame.fence), "vkResetFences", error) ||
        !check(api.resetCommandBuffer(frame.command, 0), "vkResetCommandBuffer", error) ||
        !check(api.beginCommandBuffer(frame.command, &begin), "vkBeginCommandBuffer", error)) {
        recycleCommandFrame(std::move(frame));
        return false;
    }
    recordQueueOrder(frame.command);
    command = frame.command;
    if (!recordingCommandFrames.emplace(reinterpret_cast<uintptr_t>(command), std::move(frame)).second) {
        error = "Vulkan command buffer is already recording";
        return false;
    }
    return true;
}

bool DeviceState::submitCommands(VkCommandBuffer command, uint64_t &serial, std::string &error,
                                 const SubmitSemaphores &semaphores) {
    serial = 0;
    Driver &api = driver();
    if (command != pendingUploads && !flushUploads(error)) {
        abandonCommands(command);
        return false;
    }
    const auto found = recordingCommandFrames.find(reinterpret_cast<uintptr_t>(command));
    if (found == recordingCommandFrames.end()) {
        error = "Vulkan command buffer does not belong to an active recording";
        return false;
    }
    CommandFrame frame = std::move(found->second);
    recordingCommandFrames.erase(found);
    bool prologue = false;
    if (!check(api.endCommandBuffer(command), "vkEndCommandBuffer", error) ||
        !recordLayoutPrologue(frame, prologue, error)) {
        recycleCommandFrame(std::move(frame));
        return false;
    }
    const std::array<VkCommandBuffer, 2> batch{frame.prologue, command};
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = prologue ? 2 : 1;
    submit.pCommandBuffers = prologue ? batch.data() : batch.data() + 1;
    if (semaphores.wait) {
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &semaphores.wait;
        submit.pWaitDstStageMask = &semaphores.waitStage;
    }
    if (semaphores.signal) {
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &semaphores.signal;
    }
    if (!check(api.queueSubmit(queue, 1, &submit, frame.fence), "vkQueueSubmit", error)) {
        recycleCommandFrame(std::move(frame));
        return false;
    }
    commitImageLayouts(frame);
    frame.serial = ++submittedSerial;
    serial = frame.serial;
    inFlightCommandFrames.push_back(std::move(frame));
    return true;
}

bool DeviceState::submitRecorded(VkCommandBuffer command, const std::unordered_map<Image *, ImageLayoutUse> &layouts,
                                 uint64_t &serial, std::string &error) {
    serial = 0;
    if (deviceLost) {
        error = "Vulkan device was lost";
        return false;
    }
    if (!retireCommandFrames(completedSerial, error))
        return false;
    if (inFlightCommandFrames.size() >= maxFramesInFlight &&
        !waitForSerial(inFlightCommandFrames.front().serial, error))
        return false;
    Driver &api = driver();
    // The pending uploads go in the same submission, ahead of `command`: their frame carries the fence, and the
    // transitions to `layouts` follow the uploads in their command buffer. Without uploads a frame carries the prologue
    // and the fence, and its own command buffer stays unused.
    CommandFrame frame;
    const bool uploads = pendingUploads != VK_NULL_HANDLE;
    if (uploads) {
        const auto found = recordingCommandFrames.find(reinterpret_cast<uintptr_t>(pendingUploads));
        if (found == recordingCommandFrames.end()) {
            error = "Vulkan upload commands do not belong to an active recording";
            return false;
        }
        frame = std::move(found->second);
        recordingCommandFrames.erase(found);
        pendingUploads = VK_NULL_HANDLE;
    } else if (!availableCommandFrames.empty()) {
        frame = std::move(availableCommandFrames.back());
        availableCommandFrames.pop_back();
    } else if (!createCommandFrame(frame, error)) {
        return false;
    }
    std::vector<VkImageMemoryBarrier> barriers;
    bool prologue = false;
    bool recorded = layoutBarriers(layouts, uploads ? &frame.imageLayouts : nullptr, barriers, error);
    if (recorded && uploads) {
        if (!barriers.empty())
            api.cmdPipelineBarrier(frame.command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                   static_cast<uint32_t>(barriers.size()), barriers.data());
        recorded = check(api.endCommandBuffer(frame.command), "vkEndCommandBuffer", error) &&
                   recordLayoutPrologue(frame, prologue, error);
    } else if (recorded) {
        recorded = check(api.resetFences(device, 1, &frame.fence), "vkResetFences", error) &&
                   recordPrologue(frame, barriers, error);
        prologue = !barriers.empty();
    }
    if (!recorded) {
        recycleCommandFrame(std::move(frame));
        return false;
    }
    std::array<VkCommandBuffer, 3> batch{};
    uint32_t count = 0;
    if (prologue)
        batch[count++] = frame.prologue;
    if (uploads)
        batch[count++] = frame.command;
    batch[count++] = command;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = count;
    submit.pCommandBuffers = batch.data();
    if (!check(api.queueSubmit(queue, 1, &submit, frame.fence), "vkQueueSubmit", error)) {
        recycleCommandFrame(std::move(frame));
        return false;
    }
    commitImageLayouts(frame);
    commitLayouts(layouts);
    frame.serial = ++submittedSerial;
    serial = frame.serial;
    inFlightCommandFrames.push_back(std::move(frame));
    return true;
}

bool DeviceState::beginUploadCommands(VkCommandBuffer &command, std::string &error) {
    if (!pendingUploads) {
        if (!beginCommands(command, error))
            return false;
        pendingUploads = command;
        return true;
    }
    command = pendingUploads;
    VkMemoryBarrier order{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    order.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    order.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    driver().cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &order,
                                0, nullptr, 0, nullptr);
    return true;
}

bool DeviceState::flushUploads(std::string &error) {
    if (!pendingUploads)
        return true;
    uint64_t serial{};
    return submitCommands(std::exchange(pendingUploads, VK_NULL_HANDLE), serial, error);
}

bool DeviceState::retireCommandFrames(uint64_t through, std::string &error) {
    Driver &api = driver();
    while (!inFlightCommandFrames.empty()) {
        CommandFrame &frame = inFlightCommandFrames.front();
        const VkResult state = frame.serial <= through ? api.waitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX)
                                                       : api.getFenceStatus(device, frame.fence);
        if (state == VK_NOT_READY)
            break;
        if (!check(state, frame.serial <= through ? "vkWaitForFences" : "vkGetFenceStatus", error)) {
            deviceLost = true;
            return false;
        }
        completedSerial = frame.serial;
        recycleCommandFrame(std::move(frame));
        inFlightCommandFrames.pop_front();
    }
    destroyDeferred(false);
    return true;
}

bool DeviceState::waitForSerial(uint64_t serial, std::string &error) {
    if (serial > submittedSerial) {
        error = "Vulkan serial was never submitted";
        return false;
    }
    if (!retireCommandFrames(serial, error))
        return false;
    if (deviceLost) {
        error = "Vulkan device was lost";
        return false;
    }
    return true;
}

bool DeviceState::pollSerial(uint64_t serial, bool &completed, std::string &error) {
    completed = false;
    if (serial > submittedSerial) {
        error = "Vulkan serial was never submitted";
        return false;
    }
    if (!retireCommandFrames(0, error))
        return false;
    completed = serial <= completedSerial;
    return true;
}

void DeviceState::abandonCommands(VkCommandBuffer command) {
    if (!command)
        return;
    const auto found = recordingCommandFrames.find(reinterpret_cast<uintptr_t>(command));
    if (found == recordingCommandFrames.end())
        return;
    if (command == pendingUploads)
        pendingUploads = VK_NULL_HANDLE;
    (void)driver().resetCommandBuffer(command, 0);
    recycleCommandFrame(std::move(found->second));
    recordingCommandFrames.erase(found);
}

void DeviceState::destroyAfterSubmittedWork(DeferredDestruction destruction) {
    // Pending uploads may use the object; the next submission submits them.
    destruction.serial = submittedSerial + (pendingUploads ? 1 : 0);
    if (destruction.serial <= completedSerial) {
        destroyNow(destruction);
        return;
    }
    deferredDestructions.push_back(std::move(destruction));
}

void DeviceState::destroyDeferred(bool all) {
    const auto retired = std::stable_partition(
        deferredDestructions.begin(), deferredDestructions.end(),
        [&](const DeferredDestruction &destruction) { return !all && destruction.serial > completedSerial; });
    for (auto destruction = retired; destruction != deferredDestructions.end(); ++destruction)
        destroyNow(*destruction);
    deferredDestructions.erase(retired, deferredDestructions.end());
}

void DeviceState::destroyNow(DeferredDestruction &destruction) {
    destroyBuffer(destruction.buffer);
    destroyImage(destruction.image);
    if (destruction.view)
        driver().destroyImageView(device, destruction.view, nullptr);
    destruction.view = VK_NULL_HANDLE;
    destroySampler(destruction.sampler);
}

std::optional<uint32_t> DeviceState::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                                                    VkMemoryPropertyFlags preferred) const {
    if (preferred) {
        const VkMemoryPropertyFlags preferredFlags = required | preferred;
        for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index)
            if ((typeBits & (uint32_t{1} << index)) &&
                (memoryProperties.memoryTypes[index].propertyFlags & preferredFlags) == preferredFlags)
                return index;
    }
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index)
        if ((typeBits & (uint32_t{1} << index)) &&
            (memoryProperties.memoryTypes[index].propertyFlags & required) == required)
            return index;
    return std::nullopt;
}

bool DeviceState::createBuffer(Buffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags properties, std::string &error,
                               VkMemoryPropertyFlags preferredMemoryProperties) {
    Driver &api = driver();
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(api.createBuffer(device, &bufferInfo, nullptr, &buffer.buffer), "vkCreateBuffer", error))
        return false;
    VkMemoryRequirements requirements{};
    api.getBufferMemoryRequirements(device, buffer.buffer, &requirements);
    const auto memoryType = findMemoryType(requirements.memoryTypeBits, properties, preferredMemoryProperties);
    if (!memoryType) {
        error = "Vulkan device has no compatible buffer memory type";
        destroyBuffer(buffer);
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = *memoryType;
    if (!check(api.allocateMemory(device, &allocation, nullptr, &buffer.memory), "vkAllocateMemory", error) ||
        !check(api.bindBufferMemory(device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory", error)) {
        destroyBuffer(buffer);
        return false;
    }
    return true;
}

void DeviceState::destroyBuffer(Buffer &buffer) {
    if (buffer.mapped && buffer.memory)
        driver().unmapMemory(device, buffer.memory);
    if (buffer.buffer)
        driver().destroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory)
        driver().freeMemory(device, buffer.memory, nullptr);
    buffer = {};
}

bool DeviceState::createImage(Image &image, const VkImageCreateInfo &imageInfo, const VkImageViewCreateInfo &viewInfo,
                              VkMemoryPropertyFlags properties, std::string &error) {
    Driver &api = driver();
    if (!check(api.createImage(device, &imageInfo, nullptr, &image.image), "vkCreateImage", error))
        return false;
    VkMemoryRequirements requirements{};
    api.getImageMemoryRequirements(device, image.image, &requirements);
    const auto memoryType = findMemoryType(requirements.memoryTypeBits, properties);
    if (!memoryType) {
        error = "Vulkan device has no compatible image memory type";
        destroyImage(image);
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = *memoryType;
    if (!check(api.allocateMemory(device, &allocation, nullptr, &image.memory), "vkAllocateMemory", error) ||
        !check(api.bindImageMemory(device, image.image, image.memory, 0), "vkBindImageMemory", error)) {
        destroyImage(image);
        return false;
    }
    VkImageViewCreateInfo concreteView = viewInfo;
    concreteView.image = image.image;
    if (!check(api.createImageView(device, &concreteView, nullptr, &image.view), "vkCreateImageView", error)) {
        destroyImage(image);
        return false;
    }
    image.format = imageInfo.format;
    image.layout = imageInfo.initialLayout;
    image.owned = true;
    return true;
}

void DeviceState::destroyImage(Image &image) {
    if (image.owned) {
        if (image.view)
            driver().destroyImageView(device, image.view, nullptr);
        if (image.image)
            driver().destroyImage(device, image.image, nullptr);
        if (image.memory)
            driver().freeMemory(device, image.memory, nullptr);
    }
    image = {};
}

bool DeviceState::createSampler(Sampler &sampler, const VkSamplerCreateInfo &createInfo, std::string &error) {
    return check(driver().createSampler(device, &createInfo, nullptr, &sampler.sampler), "vkCreateSampler", error);
}

void DeviceState::destroySampler(Sampler &sampler) {
    if (sampler.sampler)
        driver().destroySampler(device, sampler.sampler, nullptr);
    sampler = {};
}

bool DeviceState::acquireStaging(VkCommandBuffer command, bool upload, VkDeviceSize size, VkDeviceSize alignment,
                                 VkBuffer &buffer, VkDeviceSize &offset, uint8_t *&mapped, std::string &error) {
    if (!size || !alignment || (alignment & (alignment - 1)) != 0 ||
        size > (std::numeric_limits<VkDeviceSize>::max)() - (alignment - 1)) {
        error = "Vulkan staging allocation request is invalid";
        return false;
    }
    CommandFrame *frame = recordingFrame(command);
    if (!frame) {
        error = "Vulkan staging requires an active recording";
        return false;
    }
    StagingChunk *chunk = (upload ? uploadStaging : readbackStaging)
                              .acquire(upload ? frame->upload : frame->readback, size, alignment, offset,
                                       [&](VkDeviceSize capacity, StagingChunk &created) {
                                           return createStagingChunk(upload, capacity, created, error);
                                       });
    if (!chunk)
        return false;
    buffer = chunk->buffer.buffer;
    mapped = chunk->buffer.mapped + offset;
    return true;
}

bool DeviceState::createStagingChunk(bool upload, VkDeviceSize capacity, StagingChunk &chunk, std::string &error) {
    chunk.capacity = capacity;
    if (!createBuffer(chunk.buffer, capacity,
                      upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                             : VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, error,
                      upload ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
        return false;
    if (!check(driver().mapMemory(device, chunk.buffer.memory, 0, capacity, 0,
                                  reinterpret_cast<void **>(&chunk.buffer.mapped)),
               "vkMapMemory", error)) {
        destroyBuffer(chunk.buffer);
        return false;
    }
    return true;
}

namespace {

Result<void, std::string> failed(std::string message) { return Result<void, std::string>{err(std::move(message))}; }

// Copies `size` bytes that `fill` writes into frame staging to `destination`, in queue order. `copies` are relative
// to the staging allocation.
template <typename Fill>
Result<void, std::string> submitBufferUpload(DeviceState &state, VkBuffer destination, VkDeviceSize size,
                                             std::vector<VkBufferCopy> &copies, Fill &&fill) {
    std::string error;
    VkCommandBuffer command{};
    if (!state.beginUploadCommands(command, error))
        return failed("cannot begin the Vulkan upload: " + error);
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    // A failed upload leaves the pending recording with the earlier uploads, which still submit.
    if (!state.acquireStaging(command, true, size, 16, staging, stagingOffset, mapped, error))
        return failed("cannot stage the Vulkan upload: " + error);
    fill(mapped);
    for (VkBufferCopy &copy : copies)
        copy.srcOffset += stagingOffset;
    driver().cmdCopyBuffer(command, staging, destination, static_cast<uint32_t>(copies.size()), copies.data());
    return Result<void, std::string>{ok()};
}

// Host reads of an operation's staging wait for that operation only.
bool submitAndWait(DeviceState &state, VkCommandBuffer command, std::string &error) {
    uint64_t serial{};
    return state.submitCommands(command, serial, error) && state.waitForSerial(serial, error);
}

bool validRange(uint64_t offset, uint64_t size, uint64_t bufferSize) {
    return size != 0 && size <= (std::numeric_limits<size_t>::max)() && offset <= bufferSize &&
           size <= bufferSize - offset;
}

} // namespace

Result<uint64_t, std::string>
submitRecordedCommands(DeviceState &state, VkCommandBuffer command,
                       const std::unordered_map<Image *, DeviceState::ImageLayoutUse> &layouts) {
    std::lock_guard<std::mutex> guard(state.mutex);
    uint64_t serial{};
    std::string error;
    if (!state.submitRecorded(command, layouts, serial, error))
        return Result<uint64_t, std::string>{err(std::move(error))};
    return Result<uint64_t, std::string>{ok(serial)};
}

Result<void, std::string> waitForSubmission(DeviceState &state, uint64_t serial) {
    std::lock_guard<std::mutex> guard(state.mutex);
    std::string error;
    if (!state.waitForSerial(serial, error))
        return failed(std::move(error));
    return Result<void, std::string>{ok()};
}

Result<bool, std::string> submissionCompleted(DeviceState &state, uint64_t serial) {
    std::lock_guard<std::mutex> guard(state.mutex);
    if (serial <= state.completedSerial)
        return Result<bool, std::string>{ok(true)};
    bool completed = false;
    std::string error;
    if (!state.pollSerial(serial, completed, error))
        return Result<bool, std::string>{err(std::move(error))};
    return Result<bool, std::string>{ok(completed)};
}

Result<void, std::string> createBuffer(DeviceState &state, uint64_t size, uint32_t usage, VernonHalMemory memory,
                                       Buffer &buffer) {
    if (size == 0 || size > (std::numeric_limits<VkDeviceSize>::max)() - 3 || memory > VERNON_HAL_MEMORY_READBACK)
        return failed("the Vulkan buffer's size or memory is invalid");
    VkBufferUsageFlags native = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr std::pair<uint32_t, VkBufferUsageFlags> kUsages[] = {
        {VERNON_HAL_BUFFER_UNIFORM, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
        {VERNON_HAL_BUFFER_STORAGE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        {VERNON_HAL_BUFFER_VERTEX, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT},
        {VERNON_HAL_BUFFER_INDEX, VK_BUFFER_USAGE_INDEX_BUFFER_BIT},
        {VERNON_HAL_BUFFER_INDIRECT, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT},
    };
    for (const auto &[hal, vulkan] : kUsages)
        if (usage & hal)
            native |= vulkan;
    const VkDeviceSize allocationSize = (size + 3) & ~VkDeviceSize{3};
    const bool device = memory == VERNON_HAL_MEMORY_DEVICE;
    const VkMemoryPropertyFlags required =
        device ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
               : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags preferred = device ? 0 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    std::lock_guard<std::mutex> guard(state.mutex);
    std::string error;
    if (!state.createBuffer(buffer, allocationSize, native, required, error, preferred))
        return failed(std::move(error));
    if (!device) {
        void *mapped = nullptr;
        const VkResult result = driver().mapMemory(state.device, buffer.memory, 0, allocationSize, 0, &mapped);
        if (result != VK_SUCCESS) {
            state.destroyBuffer(buffer);
            return failed("vkMapMemory failed with Vulkan error " + std::to_string(static_cast<int>(result)));
        }
        buffer.mapped = static_cast<uint8_t *>(mapped);
    }
    return Result<void, std::string>{ok()};
}

Result<void, std::string> writeBuffer(DeviceState &state, Buffer &buffer, uint64_t size, VernonHalMemory memory,
                                      const VernonHalBufferWrite *writes, size_t count) {
    if (!buffer.buffer || !writes || count == 0 || count > (std::numeric_limits<uint32_t>::max)())
        return failed("the Vulkan buffer write is invalid");
    for (size_t index = 0; index < count; ++index)
        if (!writes[index].source || !validRange(writes[index].offset, writes[index].size, size))
            return failed("a Vulkan buffer write overruns the buffer");
    if (memory == VERNON_HAL_MEMORY_UPLOAD) {
        if (!buffer.mapped)
            return failed("the Vulkan upload buffer is not mapped");
        for (size_t index = 0; index < count; ++index)
            std::memcpy(buffer.mapped + writes[index].offset, writes[index].source,
                        static_cast<size_t>(writes[index].size));
        return Result<void, std::string>{ok()};
    }
    std::vector<VkBufferCopy> copies;
    copies.reserve(count);
    VkDeviceSize packedSize = 0;
    bool aligned = true;
    for (size_t index = 0; index < count; ++index) {
        packedSize = (packedSize + 15) & ~VkDeviceSize{15};
        copies.push_back({packedSize, writes[index].offset, writes[index].size});
        packedSize += writes[index].size;
        aligned = aligned && (writes[index].offset & 3) == 0 && (writes[index].size & 3) == 0;
    }
    std::lock_guard<std::mutex> guard(state.mutex);
    if (aligned)
        return submitBufferUpload(state, buffer.buffer, packedSize, copies, [&](uint8_t *staging) {
            for (size_t index = 0; index < count; ++index)
                std::memcpy(staging + copies[index].srcOffset, writes[index].source,
                            static_cast<size_t>(writes[index].size));
        });
    // Transfers move whole words, so the words that unaligned writes cover are read back, patched and written whole.
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> words;
    words.reserve(count);
    for (size_t index = 0; index < count; ++index)
        words.emplace_back(writes[index].offset & ~VkDeviceSize{3},
                           (writes[index].offset + writes[index].size + 3) & ~VkDeviceSize{3});
    std::sort(words.begin(), words.end());
    size_t merged = 0;
    for (const auto &[begin, end] : words) {
        if (merged && begin <= words[merged - 1].second)
            words[merged - 1].second = (std::max)(words[merged - 1].second, end);
        else
            words[merged++] = {begin, end};
    }
    words.resize(merged);
    VkDeviceSize wordBytes = 0;
    std::vector<VkBufferCopy> readbackCopies;
    readbackCopies.reserve(merged);
    for (const auto &[begin, end] : words) {
        readbackCopies.push_back({begin, wordBytes, end - begin});
        wordBytes += end - begin;
    }
    std::vector<uint8_t> contents(static_cast<size_t>(wordBytes));
    std::string error;
    VkCommandBuffer readback{};
    if (!state.beginCommands(readback, error))
        return failed("cannot begin the Vulkan upload's readback: " + error);
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(readback, false, wordBytes, 16, staging, stagingOffset, mapped, error)) {
        state.abandonCommands(readback);
        return failed("cannot stage the Vulkan upload's readback: " + error);
    }
    for (VkBufferCopy &copy : readbackCopies)
        copy.dstOffset += stagingOffset;
    driver().cmdCopyBuffer(readback, buffer.buffer, staging, static_cast<uint32_t>(readbackCopies.size()),
                           readbackCopies.data());
    if (!submitAndWait(state, readback, error))
        return failed("cannot read back the Vulkan upload's words: " + error);
    std::memcpy(contents.data(), mapped, contents.size());
    for (size_t index = 0; index < count; ++index)
        for (size_t word = 0; word < words.size(); ++word) {
            const auto [begin, end] = words[word];
            if (writes[index].offset < begin || writes[index].offset + writes[index].size > end)
                continue;
            std::memcpy(contents.data() + (readbackCopies[word].dstOffset - stagingOffset) + writes[index].offset -
                            begin,
                        writes[index].source, static_cast<size_t>(writes[index].size));
            break;
        }
    copies.clear();
    VkDeviceSize sourceOffset = 0;
    for (const auto &[begin, end] : words) {
        copies.push_back({sourceOffset, begin, end - begin});
        sourceOffset += end - begin;
    }
    return submitBufferUpload(state, buffer.buffer, wordBytes, copies,
                              [&](uint8_t *staged) { std::memcpy(staged, contents.data(), contents.size()); });
}

Result<void, std::string> readBuffer(DeviceState &state, Buffer &buffer, uint64_t size,
                                     const VernonHalBufferRead *reads, size_t count) {
    if (!buffer.buffer || !reads || count == 0 || count > (std::numeric_limits<uint32_t>::max)())
        return failed("the Vulkan buffer read is invalid");
    std::vector<VkBufferCopy> copies;
    copies.reserve(count);
    VkDeviceSize stagingSize = 0;
    for (size_t index = 0; index < count; ++index) {
        const VernonHalBufferRead &read = reads[index];
        if (!read.destination || !validRange(read.offset, read.size, size))
            return failed("a Vulkan buffer read overruns the buffer");
        const VkDeviceSize begin = read.offset & ~VkDeviceSize{3};
        const VkDeviceSize end = (read.offset + read.size + 3) & ~VkDeviceSize{3};
        stagingSize = (stagingSize + 3) & ~VkDeviceSize{3};
        copies.push_back({begin, stagingSize, end - begin});
        stagingSize += end - begin;
    }
    std::lock_guard<std::mutex> guard(state.mutex);
    std::string error;
    VkCommandBuffer command{};
    if (!state.beginCommands(command, error))
        return failed("cannot begin the Vulkan buffer read: " + error);
    VkBuffer staging{};
    VkDeviceSize stagingOffset{};
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(command, false, stagingSize, 16, staging, stagingOffset, mapped, error)) {
        state.abandonCommands(command);
        return failed("cannot stage the Vulkan buffer read: " + error);
    }
    for (VkBufferCopy &copy : copies)
        copy.dstOffset += stagingOffset;
    driver().cmdCopyBuffer(command, buffer.buffer, staging, static_cast<uint32_t>(copies.size()), copies.data());
    if (!submitAndWait(state, command, error))
        return failed("cannot read the Vulkan buffer: " + error);
    for (size_t index = 0; index < count; ++index) {
        const VkDeviceSize begin = reads[index].offset & ~VkDeviceSize{3};
        std::memcpy(reads[index].destination,
                    mapped + (copies[index].dstOffset - stagingOffset) + reads[index].offset - begin,
                    static_cast<size_t>(reads[index].size));
    }
    return Result<void, std::string>{ok()};
}

void destroyBuffer(DeviceState &state, Buffer &buffer) {
    std::lock_guard<std::mutex> guard(state.mutex);
    DeviceState::DeferredDestruction destruction;
    destruction.buffer = std::exchange(buffer, {});
    state.destroyAfterSubmittedWork(std::move(destruction));
}

} // namespace vernon::hal::vulkan
