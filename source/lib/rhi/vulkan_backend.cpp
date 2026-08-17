#include "vulkan_backend.h"

#include <algorithm>
#include <limits>
#include <mutex>
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
    const VkPhysicalDeviceProperties &properties = candidates[deviceIndex].properties;
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
    VkPhysicalDeviceFeatures enabledFeatures{};
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
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    if (!check(api.createCommandPool(device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool", error)) {
        shutdown();
        return false;
    }
    VkCommandBufferAllocateInfo commandAllocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandAllocation.commandPool = commandPool;
    commandAllocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandAllocation.commandBufferCount = 1;
    if (!check(api.allocateCommandBuffers(device, &commandAllocation, &frame.command), "vkAllocateCommandBuffers",
               error)) {
        shutdown();
        return false;
    }
    ++commandBufferAllocations;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (!check(api.createFence(device, &fenceInfo, nullptr, &frame.fence), "vkCreateFence", error)) {
        shutdown();
        return false;
    }
    const VkDescriptorPoolSize descriptorSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 256},
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorPoolInfo.maxSets = 256;
    descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(std::size(descriptorSizes));
    descriptorPoolInfo.pPoolSizes = descriptorSizes;
    if (!check(api.createDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool),
               "vkCreateDescriptorPool", error)) {
        shutdown();
        return false;
    }
    ++descriptorPoolCreations;
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
    if (device && !nativeObjectsBorrowed) {
        if (api.deviceWaitIdle)
            api.deviceWaitIdle(device);
        for (StagingRing *ring : {&uploadRing, &readbackRing}) {
            if (ring->mapped && ring->buffer.memory)
                api.unmapMemory(device, ring->buffer.memory);
            ring->mapped = nullptr;
            destroyBuffer(ring->buffer);
            ring->capacity = 0;
            ring->cursor = 0;
        }
        destroySampler(defaultImplicitSampler);
        if (descriptorPool && api.destroyDescriptorPool)
            api.destroyDescriptorPool(device, descriptorPool, nullptr);
        if (frame.fence && api.destroyFence)
            api.destroyFence(device, frame.fence, nullptr);
        frame = {};
        if (commandPool && api.destroyCommandPool)
            api.destroyCommandPool(device, commandPool, nullptr);
        if (api.destroyDevice)
            api.destroyDevice(device, nullptr);
    }
    if (instance && !nativeObjectsBorrowed && api.destroyInstance)
        api.destroyInstance(instance, nullptr);
    commandPool = VK_NULL_HANDLE;
    descriptorPool = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
    queueFamily = 0;
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
    borrowedCommandBuffer = VK_NULL_HANDLE;
    std::fill(std::begin(maxComputeWorkGroupSize), std::end(maxComputeWorkGroupSize), 0);
    memoryProperties = {};
    defaultImplicitSamplerCreations = 0;
}

bool DeviceState::synchronize(std::string &error) {
    if (nativeObjectsBorrowed)
        return true;
    return device && check(driver().deviceWaitIdle(device), "vkDeviceWaitIdle", error);
}

bool DeviceState::beginCommands(VkCommandBuffer &command, std::string &error) {
    if (nativeObjectsBorrowed) {
        command = borrowedCommandBuffer;
        return command != VK_NULL_HANDLE;
    }
    Driver &api = driver();
    if (frame.submitted &&
        !check(api.waitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", error))
        return false;
    if (!check(api.resetFences(device, 1, &frame.fence), "vkResetFences", error) ||
        !check(api.resetCommandBuffer(frame.command, 0), "vkResetCommandBuffer", error))
        return false;
    frame.submitted = false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    command = frame.command;
    return check(api.beginCommandBuffer(command, &begin), "vkBeginCommandBuffer", error);
}

bool DeviceState::submitCommands(VkCommandBuffer command, std::string &error) {
    if (nativeObjectsBorrowed)
        return command == borrowedCommandBuffer;
    Driver &api = driver();
    if (command != frame.command) {
        error = "Vulkan command buffer does not belong to the active frame";
        return false;
    }
    const bool recorded = check(api.endCommandBuffer(command), "vkEndCommandBuffer", error);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (!recorded || !check(api.queueSubmit(queue, 1, &submit, frame.fence), "vkQueueSubmit", error))
        return false;
    frame.submitted = true;
    if (!check(api.waitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", error))
        return false;
    // RuntimeCore caches binding sets across invocations. Resetting the shared
    // pool here would invalidate those live descriptor-set handles.
    uploadRing.cursor = 0;
    readbackRing.cursor = 0;
    return true;
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

bool DeviceState::acquireStaging(bool upload, VkDeviceSize size, VkDeviceSize alignment, VkBuffer &buffer,
                                 VkDeviceSize &offset, uint8_t *&mapped, std::string &error) {
    if (!size || !alignment || (alignment & (alignment - 1)) != 0 ||
        size > (std::numeric_limits<VkDeviceSize>::max)() - (alignment - 1)) {
        error = "Vulkan staging allocation request is invalid";
        return false;
    }
    StagingRing &ring = upload ? uploadRing : readbackRing;
    const VkDeviceSize alignedCursor = (ring.cursor + alignment - 1) & ~(alignment - 1);
    if (!ring.buffer.buffer || alignedCursor > ring.capacity || size > ring.capacity - alignedCursor) {
        if (ring.cursor != 0) {
            error = "Vulkan staging ring was exhausted before command submission";
            return false;
        }
        if (ring.buffer.buffer) {
            if (ring.mapped)
                driver().unmapMemory(device, ring.buffer.memory);
            ring.mapped = nullptr;
            destroyBuffer(ring.buffer);
        }
        // Keep the first allocation modest. MoltenVK maps host-visible Vulkan
        // memory to Metal buffers, and CI's virtualized devices can reject the
        // old 1 MiB minimum even for uploads that are only a few bytes.
        VkDeviceSize capacity = 64 * 1024;
        while (capacity < size) {
            if (capacity > (std::numeric_limits<VkDeviceSize>::max)() / 2) {
                error = "Vulkan staging ring capacity overflow";
                return false;
            }
            capacity *= 2;
        }
        if (!createBuffer(ring.buffer, capacity,
                          upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, error,
                          upload ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
            return false;
        ++stagingBufferAllocations;
        if (!check(
                driver().mapMemory(device, ring.buffer.memory, 0, capacity, 0, reinterpret_cast<void **>(&ring.mapped)),
                "vkMapMemory", error)) {
            destroyBuffer(ring.buffer);
            return false;
        }
        ring.capacity = capacity;
    }
    offset = (ring.cursor + alignment - 1) & ~(alignment - 1);
    buffer = ring.buffer.buffer;
    mapped = ring.mapped + offset;
    ring.cursor = offset + size;
    return true;
}

bool DeviceState::allocateDescriptorSet(VkDescriptorSetLayout layout, VkDescriptorSet &set, std::string &error) {
    if (!layout || !descriptorPool) {
        error = "Vulkan descriptor allocation state is incomplete";
        return false;
    }
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = descriptorPool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    return check(driver().allocateDescriptorSets(device, &allocation, &set), "vkAllocateDescriptorSets", error);
}

bool DeviceState::freeDescriptorSet(VkDescriptorSet set, std::string &error) {
    if (!set || !descriptorPool) {
        error = "Vulkan descriptor release state is incomplete";
        return false;
    }
    return check(driver().freeDescriptorSets(device, descriptorPool, 1, &set), "vkFreeDescriptorSets", error);
}

} // namespace vernon::rhi::vulkan
