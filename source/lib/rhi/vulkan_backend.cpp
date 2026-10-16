#include "vulkan_backend.h"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <new>
#include <string_view>
#include <vector>

namespace vernon::rhi::vulkan {
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
    commandBufferAllocations = 0;
    descriptorPoolCreations = 0;
    stagingBufferAllocations = 0;
    static std::mutex initializationMutex;
    std::lock_guard<std::mutex> guard(initializationMutex);
    Driver &api = driver();
    if (!api.load()) {
        error = api.error;
        return false;
    }
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "VernonRHI";
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
    maxComputeWorkGroupInvocations = properties.limits.maxComputeWorkGroupInvocations;
    descriptorBufferOffsetAlignment = (std::max)(properties.limits.minUniformBufferOffsetAlignment,
                                                 properties.limits.minStorageBufferOffsetAlignment);
    for (size_t index = 0; index < 3; ++index)
        maxComputeWorkGroupSize[index] = properties.limits.maxComputeWorkGroupSize[index];
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
    ++defaultImplicitSamplerCreations;
    return true;
}

bool DeviceState::initializeBorrowed(VkInstance borrowedInstance, VkPhysicalDevice borrowedPhysicalDevice,
                                     VkDevice borrowedDevice, VkQueue borrowedQueue, uint32_t borrowedQueueFamily,
                                     VkCommandBuffer commands, std::string &error) {
    shutdown();
    nativeObjectsBorrowed = true;
    instance = borrowedInstance;
    physicalDevice = borrowedPhysicalDevice;
    device = borrowedDevice;
    queue = borrowedQueue;
    queueFamily = borrowedQueueFamily;
    borrowedCommandBuffer = commands;
    Driver &api = driver();
    if (!api.load() || !api.loadInstance(instance) || !api.loadDevice(device)) {
        error = api.error;
        shutdown();
        return false;
    }
    VkPhysicalDeviceProperties properties{};
    api.getPhysicalDeviceProperties(physicalDevice, &properties);
    uint32_t queueFamilyCount = 0;
    api.getPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    api.getPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    timestampValidBits = queueFamily < queueFamilyCount ? queueFamilies[queueFamily].timestampValidBits : 0;
    timestampPeriod = properties.limits.timestampPeriod;
    maxPushConstantsSize = properties.limits.maxPushConstantsSize;
    maxVertexInputAttributes = properties.limits.maxVertexInputAttributes;
    apiVersion = properties.apiVersion;
    maxComputeWorkGroupInvocations = properties.limits.maxComputeWorkGroupInvocations;
    descriptorBufferOffsetAlignment = (std::max)(properties.limits.minUniformBufferOffsetAlignment,
                                                 properties.limits.minStorageBufferOffsetAlignment);
    for (size_t index = 0; index < 3; ++index)
        maxComputeWorkGroupSize[index] = properties.limits.maxComputeWorkGroupSize[index];
    api.getPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    return true;
}

void DeviceState::shutdown() {
    Driver &api = driver();
    if (device) {
        // A borrowed device's host has executed the borrowed command buffer before it destroys this state.
        retireBorrowedCommands(borrowedSubmissions);
        destroyCommandFrame(borrowedCommandFrame);
    }
    if (device && !nativeObjectsBorrowed) {
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
    }
    if (device) {
        for (StagingPool<StagingChunk> *pool : {&uploadStaging, &readbackStaging})
            pool->clear([this](StagingChunk &chunk) { destroyBuffer(chunk.buffer); });
        for (VkDescriptorPool pool : descriptorPools)
            api.destroyDescriptorPool(device, pool, nullptr);
    }
    descriptorPools.clear();
    currentDescriptorPool = 0;
    if (device && !nativeObjectsBorrowed && api.destroyDevice)
        api.destroyDevice(device, nullptr);
    availableCommandFrames.clear();
    recordingCommandFrames.clear();
    inFlightCommandFrames.clear();
    borrowedSubmissions = 0;
    deferredDestructions.clear();
    submittedSerial = 0;
    completedSerial = 0;
    deviceLost = false;
    if (instance && !nativeObjectsBorrowed && api.destroyInstance)
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
    maxComputeWorkGroupInvocations = 0;
    descriptorBufferOffsetAlignment = 1;
    dynamicRendering = false;
    shaderBufferFloat32AtomicAdd = false;
    portabilityEnumeration = false;
    portabilitySubset = false;
    nativeObjectsBorrowed = false;
    cmdBeginDebugUtilsLabel = nullptr;
    cmdEndDebugUtilsLabel = nullptr;
    cmdInsertDebugUtilsLabel = nullptr;
    borrowedCommandBuffer = VK_NULL_HANDLE;
    std::fill(std::begin(maxComputeWorkGroupSize), std::end(maxComputeWorkGroupSize), 0);
    memoryProperties = {};
    defaultImplicitSamplerCreations = 0;
}

bool DeviceState::synchronize(std::string &error) {
    if (nativeObjectsBorrowed)
        return true;
    if (!device || !check(driver().deviceWaitIdle(device), "vkDeviceWaitIdle", error))
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
    ++commandBufferAllocations;
    return true;
}

void DeviceState::destroyCommandFrame(CommandFrame &frame) {
    Driver &api = driver();
    retireStaging(frame);
    if (frame.fence && api.destroyFence)
        api.destroyFence(device, frame.fence, nullptr);
    if (frame.pool && api.destroyCommandPool)
        api.destroyCommandPool(device, frame.pool, nullptr);
    frame = {};
}

void DeviceState::recycleCommandFrame(CommandFrame frame) {
    frame.serial = 0;
    frame.imageLayouts.clear();
    retireStaging(frame);
    availableCommandFrames.push_back(std::move(frame));
}

DeviceState::CommandFrame *DeviceState::recordingFrame(VkCommandBuffer command) {
    if (nativeObjectsBorrowed)
        return command && command == borrowedCommandBuffer ? &borrowedCommandFrame : nullptr;
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
            use.initial[subresource] = nativeObjectsBorrowed ? owner.subresourceLayouts[subresource] : layout;
        from = current != kUnusedLayout ? std::optional<VkImageLayout>{current}
               : nativeObjectsBorrowed  ? std::optional<VkImageLayout>{use.initial[subresource]}
                                        : std::nullopt;
        current = layout;
    } catch (const std::bad_alloc &) {
        error = "Vulkan image layout tracking ran out of memory";
        return false;
    }
    return true;
}

VkImageLayout DeviceState::recordedImageLayout(VkCommandBuffer command, Image &owner, size_t subresource) {
    if (CommandFrame *frame = recordingFrame(command)) {
        const auto found = frame->imageLayouts.find(&owner);
        if (found != frame->imageLayouts.end() && found->second.current[subresource] != kUnusedLayout)
            return found->second.current[subresource];
    }
    return owner.subresourceLayouts[subresource];
}

bool DeviceState::recordLayoutPrologue(CommandFrame &frame, bool &recorded, std::string &error) {
    recorded = false;
    std::vector<VkImageMemoryBarrier> barriers;
    try {
        for (const auto &[image, use] : frame.imageLayouts)
            for (size_t subresource = 0; subresource < use.initial.size(); ++subresource) {
                const VkImageLayout layout = use.initial[subresource];
                if (layout == kUnusedLayout || layout == image->subresourceLayouts[subresource])
                    continue;
                VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.oldLayout = image->subresourceLayouts[subresource];
                barrier.newLayout = layout;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image->image;
                barrier.subresourceRange = {imageAspectMask(image->format),
                                            static_cast<uint32_t>(subresource % image->mipLevels), 1,
                                            static_cast<uint32_t>(subresource / image->mipLevels), 1};
                barriers.push_back(barrier);
            }
    } catch (const std::bad_alloc &) {
        error = "Vulkan layout prologue ran out of memory";
        return false;
    }
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
        ++commandBufferAllocations;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(api.resetCommandBuffer(frame.prologue, 0), "vkResetCommandBuffer", error) ||
        !check(api.beginCommandBuffer(frame.prologue, &begin), "vkBeginCommandBuffer", error))
        return false;
    api.cmdPipelineBarrier(frame.prologue, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                           nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
    if (!check(api.endCommandBuffer(frame.prologue), "vkEndCommandBuffer", error))
        return false;
    recorded = true;
    return true;
}

void DeviceState::commitImageLayouts(CommandFrame &frame) {
    for (auto &[image, use] : frame.imageLayouts) {
        for (size_t subresource = 0; subresource < use.current.size(); ++subresource)
            if (use.current[subresource] != kUnusedLayout)
                image->subresourceLayouts[subresource] = use.current[subresource];
        const VkImageLayout shared = image->subresourceLayouts.front();
        if (std::all_of(image->subresourceLayouts.begin(), image->subresourceLayouts.end(),
                        [shared](VkImageLayout layout) { return layout == shared; }))
            image->layout = shared;
    }
    frame.imageLayouts.clear();
}

void DeviceState::retireStaging(CommandFrame &frame) {
    uploadStaging.retire(frame.upload);
    readbackStaging.retire(frame.readback);
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
    if (nativeObjectsBorrowed) {
        command = borrowedCommandBuffer;
        if (!command)
            return false;
        recordQueueOrder(command);
        return true;
    }
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

bool DeviceState::submitCommands(VkCommandBuffer command, uint64_t &serial, std::string &error) {
    serial = 0;
    if (nativeObjectsBorrowed) {
        if (command != borrowedCommandBuffer)
            return false;
        commitImageLayouts(borrowedCommandFrame);
        serial = ++borrowedSubmissions;
        borrowedStaging.push_back(
            {serial, std::exchange(borrowedCommandFrame.upload, {}), std::exchange(borrowedCommandFrame.readback, {})});
        return true;
    }
    Driver &api = driver();
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

void DeviceState::retireBorrowedCommands(uint64_t serial) {
    while (!borrowedStaging.empty() && borrowedStaging.front().serial <= serial) {
        uploadStaging.retire(borrowedStaging.front().upload);
        readbackStaging.retire(borrowedStaging.front().readback);
        borrowedStaging.pop_front();
    }
}

void DeviceState::abandonCommands(VkCommandBuffer command) {
    if (!command)
        return;
    // The host executes whatever was recorded into its command buffer, including the abandoned recording's transitions.
    if (nativeObjectsBorrowed) {
        if (command == borrowedCommandBuffer)
            commitImageLayouts(borrowedCommandFrame);
        return;
    }
    const auto found = recordingCommandFrames.find(reinterpret_cast<uintptr_t>(command));
    if (found == recordingCommandFrames.end())
        return;
    (void)driver().resetCommandBuffer(command, 0);
    recycleCommandFrame(std::move(found->second));
    recordingCommandFrames.erase(found);
}

void DeviceState::destroyAfterSubmittedWork(DeferredDestruction destruction) {
    destruction.serial = submittedSerial;
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
    buffer.owned = true;
    return true;
}

void DeviceState::destroyBuffer(Buffer &buffer) {
    if (buffer.owned) {
        if (buffer.mapped && buffer.memory)
            driver().unmapMemory(device, buffer.memory);
        if (buffer.buffer)
            driver().destroyBuffer(device, buffer.buffer, nullptr);
        if (buffer.memory)
            driver().freeMemory(device, buffer.memory, nullptr);
    }
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
    if (!check(driver().createSampler(device, &createInfo, nullptr, &sampler.sampler), "vkCreateSampler", error))
        return false;
    sampler.owned = true;
    return true;
}

void DeviceState::destroySampler(Sampler &sampler) {
    if (sampler.sampler && sampler.owned)
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
                      upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, error,
                      upload ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
        return false;
    ++stagingBufferAllocations;
    if (!check(driver().mapMemory(device, chunk.buffer.memory, 0, capacity, 0,
                                  reinterpret_cast<void **>(&chunk.buffer.mapped)),
               "vkMapMemory", error)) {
        destroyBuffer(chunk.buffer);
        return false;
    }
    return true;
}

bool DeviceState::createDescriptorPool(std::string &error) {
    // Each pool doubles the last, up to 16 times the first, so many live binding sets need few pools.
    const uint32_t scale = uint32_t{1} << (std::min)(descriptorPools.size(), size_t{4});
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 * scale},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256 * scale},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 * scale},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256 * scale},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 256 * scale},
    };
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    info.maxSets = 256 * scale;
    info.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
    info.pPoolSizes = sizes;
    VkDescriptorPool pool{};
    if (!check(driver().createDescriptorPool(device, &info, nullptr, &pool), "vkCreateDescriptorPool", error))
        return false;
    descriptorPools.push_back(pool);
    currentDescriptorPool = descriptorPools.size() - 1;
    ++descriptorPoolCreations;
    return true;
}

bool DeviceState::allocateDescriptorSet(VkDescriptorSetLayout layout, DescriptorSet &set, std::string &error) {
    if (!layout || !device) {
        error = "Vulkan descriptor allocation state is incomplete";
        return false;
    }
    std::lock_guard<std::mutex> guard(descriptorMutex);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    // Freed sets return space to the pools they came from, so every pool is tried before the chain grows.
    const size_t poolCount = descriptorPools.size();
    for (size_t attempt = 0; attempt < poolCount; ++attempt) {
        const size_t index = (currentDescriptorPool + attempt) % poolCount;
        allocation.descriptorPool = descriptorPools[index];
        const VkResult result = driver().allocateDescriptorSets(device, &allocation, &set.set);
        if (result == VK_SUCCESS) {
            currentDescriptorPool = index;
            set.pool = descriptorPools[index];
            return true;
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL)
            return check(result, "vkAllocateDescriptorSets", error);
    }
    if (!createDescriptorPool(error))
        return false;
    allocation.descriptorPool = descriptorPools.back();
    if (!check(driver().allocateDescriptorSets(device, &allocation, &set.set), "vkAllocateDescriptorSets", error))
        return false;
    set.pool = descriptorPools.back();
    return true;
}

void DeviceState::freeDescriptorSet(DescriptorSet &set) {
    if (set.set && set.pool && device) {
        std::lock_guard<std::mutex> guard(descriptorMutex);
        (void)driver().freeDescriptorSets(device, set.pool, 1, &set.set);
    }
    set = {};
}

} // namespace vernon::rhi::vulkan
