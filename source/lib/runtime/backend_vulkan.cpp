#include "backend_vulkan.h"

#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

#if defined(VERNON_HAS_VULKAN_RUNTIME)
VernonStatus vkFail(VernonRuntimeContext &context, VkResult result, const char *operation) {
    if (result == VK_SUCCESS)
        return VERNON_STATUS_OK;
    context.error = std::string(operation) + " failed with VkResult " + std::to_string(result);
    return VERNON_STATUS_INTERNAL_ERROR;
}

bool vkSucceeded(VkResult result, const char *operation, std::string &error) {
    if (result == VK_SUCCESS)
        return true;
    error = std::string(operation) + " failed with Vulkan error " + std::to_string(static_cast<int>(result));
    return false;
}

std::optional<uint32_t> findMemoryType(const VernonRuntimeContext &context, uint32_t typeBits,
                                       VkMemoryPropertyFlags required) {
    const VulkanContextState &state = vulkanState(context);
    for (uint32_t index = 0; index < state.memoryProperties.memoryTypeCount; ++index)
        if ((typeBits & (uint32_t{1} << index)) &&
            (state.memoryProperties.memoryTypes[index].propertyFlags & required) == required)
            return index;
    return std::nullopt;
}

std::optional<VkFormat> textureFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_R8_UNORM:
        return VK_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_RG8_UNORM:
        return VK_FORMAT_R8G8_UNORM;
    case VERNON_TEXTURE_RGB8_UNORM:
        return VK_FORMAT_R8G8B8_UNORM;
    case VERNON_TEXTURE_RGBA8_UNORM:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case VERNON_TEXTURE_R16_FLOAT:
        return VK_FORMAT_R16_SFLOAT;
    case VERNON_TEXTURE_R32_FLOAT:
        return VK_FORMAT_R32_SFLOAT;
    }
    return std::nullopt;
}

std::optional<VkSamplerAddressMode> samplerAddressMode(VernonSamplerWrapMode mode) {
    switch (mode) {
    case VERNON_SAMPLER_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case VERNON_SAMPLER_MIRRORED_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case VERNON_SAMPLER_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case VERNON_SAMPLER_CLAMP_TO_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    }
    return std::nullopt;
}

std::optional<VkFilter> samplerFilter(VernonSamplerFilter filter) {
    if (filter == VERNON_SAMPLER_NEAREST)
        return VK_FILTER_NEAREST;
    if (filter == VERNON_SAMPLER_LINEAR)
        return VK_FILTER_LINEAR;
    return std::nullopt;
}

std::optional<size_t> rgba8ByteSize(uint32_t width, uint32_t height, uint32_t layers) {
    if (height && static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / height)
        return std::nullopt;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (layers && pixels > std::numeric_limits<size_t>::max() / layers)
        return std::nullopt;
    const size_t layeredPixels = pixels * layers;
    if (layeredPixels > std::numeric_limits<size_t>::max() / 4)
        return std::nullopt;
    return layeredPixels * 4;
}

template <typename Record> bool submitCommands(VernonRuntimeContext &context, Record &&record) {
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!beginVulkanCommands(context, command, context.error))
        return false;
    record(command);
    return submitVulkanCommands(context, command, context.error);
}

} // namespace

bool beginVulkanCommands(VernonRuntimeContext &context, VkCommandBuffer &command, std::string &error) {
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(context);
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = state.commandPool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    if (!vkSucceeded(driver.allocateCommandBuffers(state.device, &allocation, &command), "vkAllocateCommandBuffers",
                     error))
        return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkSucceeded(driver.beginCommandBuffer(command, &begin), "vkBeginCommandBuffer", error))
        return true;
    driver.freeCommandBuffers(state.device, state.commandPool, 1, &command);
    command = VK_NULL_HANDLE;
    return false;
}

bool submitVulkanCommands(VernonRuntimeContext &context, VkCommandBuffer command, std::string &error) {
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(context);
    const bool recorded = vkSucceeded(driver.endCommandBuffer(command), "vkEndCommandBuffer", error);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    const bool submitted =
        recorded && vkSucceeded(driver.queueSubmit(state.queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit", error) &&
        vkSucceeded(driver.queueWaitIdle(state.queue), "vkQueueWaitIdle", error);
    driver.freeCommandBuffers(state.device, state.commandPool, 1, &command);
    return submitted;
}

void transitionVulkanImageLayout(VkCommandBuffer command, VernonDeviceTexture &texture, VkImageLayout newLayout) {
    VulkanTextureState &state = vulkanTextureState(texture);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = state.layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = state.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = texture.mipLevels;
    barrier.subresourceRange.layerCount = texture.dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (state.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    } else if (state.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (state.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (state.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (state.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        destinationStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    vulkanDriver().cmdPipelineBarrier(command, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    state.layout = newLayout;
}

namespace {
#endif

} // namespace

bool initializeVulkanContext(VernonRuntimeContext &context, uint32_t deviceIndex) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    // VulkanDriver owns one mutable dispatch table. Initialization is rare and
    // must not race while instance/device function pointers are replaced.
    static std::mutex initializationMutex;
    std::lock_guard<std::mutex> guard(initializationMutex);
    VulkanDriver &driver = vulkanDriver();
    if (!driver.load()) {
        context.error = driver.error;
        return false;
    }
    installRuntimeBackendState(context, new VulkanContextState());
    VulkanContextState &state = vulkanState(context);
    const auto failInitialization = [&]() {
        const std::string diagnostic = context.error;
        destroyVulkanContext(context);
        destroyRuntimeBackendState(context);
        context.error = diagnostic;
        return false;
    };
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "VernonRuntime";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.pEngineName = "VernonRuntime";
    application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &application;
    if (vkFail(context, driver.createInstance(&instanceInfo, nullptr, &state.instance), "vkCreateInstance") !=
        VERNON_STATUS_OK)
        return failInitialization();
    if (!driver.loadInstance(state.instance)) {
        context.error = driver.error;
        return failInitialization();
    }
    uint32_t deviceCount = 0;
    if (driver.enumeratePhysicalDevices(state.instance, &deviceCount, nullptr) != VK_SUCCESS ||
        deviceIndex >= deviceCount) {
        context.error = "Vulkan device index is unavailable";
        return failInitialization();
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (driver.enumeratePhysicalDevices(state.instance, &deviceCount, devices.data()) != VK_SUCCESS)
        return failInitialization();
    state.physicalDevice = devices[deviceIndex];
    VkPhysicalDeviceProperties properties{};
    driver.getPhysicalDeviceProperties(state.physicalDevice, &properties);
    state.maxPushConstantsSize = properties.limits.maxPushConstantsSize;
    uint32_t queueCount = 0;
    driver.getPhysicalDeviceQueueFamilyProperties(state.physicalDevice, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queueCount);
    driver.getPhysicalDeviceQueueFamilyProperties(state.physicalDevice, &queueCount, queues.data());
    const auto queue = std::find_if(queues.begin(), queues.end(), [](const VkQueueFamilyProperties &family) {
        return family.queueFlags & VK_QUEUE_COMPUTE_BIT;
    });
    if (queue == queues.end()) {
        context.error = "Vulkan device has no compute queue";
        return failInitialization();
    }
    state.queueFamily = static_cast<uint32_t>(std::distance(queues.begin(), queue));
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = state.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (vkFail(context, driver.createDevice(state.physicalDevice, &deviceInfo, nullptr, &state.device),
               "vkCreateDevice") != VERNON_STATUS_OK)
        return failInitialization();
    if (!driver.loadDevice(state.device)) {
        context.error = driver.error;
        return failInitialization();
    }
    driver.getDeviceQueue(state.device, state.queueFamily, 0, &state.queue);
    driver.getPhysicalDeviceMemoryProperties(state.physicalDevice, &state.memoryProperties);
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = state.queueFamily;
    if (vkFail(context, driver.createCommandPool(state.device, &poolInfo, nullptr, &state.commandPool),
               "vkCreateCommandPool") != VERNON_STATUS_OK)
        return failInitialization();
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (vkFail(context, driver.createSampler(state.device, &samplerInfo, nullptr, &state.defaultImplicitSampler),
               "vkCreateSampler") != VERNON_STATUS_OK)
        return failInitialization();
    ++state.defaultImplicitSamplerCreations;
    return true;
#else
    (void)context;
    (void)deviceIndex;
    return false;
#endif
}

void destroyVulkanContext(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(context);
    if (state.device) {
        if (driver.deviceWaitIdle)
            driver.deviceWaitIdle(state.device);
        if (state.defaultImplicitSampler && driver.destroySampler)
            driver.destroySampler(state.device, state.defaultImplicitSampler, nullptr);
        if (state.commandPool && driver.destroyCommandPool)
            driver.destroyCommandPool(state.device, state.commandPool, nullptr);
        PFN_vkDestroyDevice destroyDevice = driver.destroyDevice;
        if (!destroyDevice && driver.getDeviceProcAddr)
            destroyDevice =
                reinterpret_cast<PFN_vkDestroyDevice>(driver.getDeviceProcAddr(state.device, "vkDestroyDevice"));
        if (destroyDevice)
            destroyDevice(state.device, nullptr);
    }
    if (state.instance) {
        PFN_vkDestroyInstance destroyInstance = driver.destroyInstance;
        if (!destroyInstance && driver.getInstanceProcAddr)
            destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
                driver.getInstanceProcAddr(state.instance, "vkDestroyInstance"));
        if (destroyInstance)
            destroyInstance(state.instance, nullptr);
    }
    state.commandPool = VK_NULL_HANDLE;
    state.defaultImplicitSampler = VK_NULL_HANDLE;
    state.queue = VK_NULL_HANDLE;
    state.device = VK_NULL_HANDLE;
    state.physicalDevice = VK_NULL_HANDLE;
    state.instance = VK_NULL_HANDLE;
#else
    (void)context;
#endif
}

bool probeVulkan(std::string &diagnostic) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    struct Probe {
        bool available{};
        std::string diagnostic;
    };
    static const Probe cached = [] {
        Probe result;
        VernonRuntimeContext context;
        context.backend = VERNON_RUNTIME_VULKAN;
        result.available = initializeVulkanContext(context, 0);
        result.diagnostic = context.error;
        if (result.available) {
            destroyVulkanContext(context);
            destroyRuntimeBackendState(context);
        }
        return result;
    }();
    diagnostic = cached.diagnostic;
    return cached.available;
#else
    diagnostic = "VernonRuntime was built without Vulkan support";
    return false;
#endif
}

#if defined(VERNON_HAS_VULKAN_RUNTIME)
bool createVulkanBuffer(VernonRuntimeContext &context, VkDeviceSize size, VkBuffer &buffer, VkDeviceMemory &memory) {
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(context);
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkFail(context, driver.createBuffer(state.device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer") !=
        VERNON_STATUS_OK)
        return false;
    VkMemoryRequirements requirements{};
    driver.getBufferMemoryRequirements(state.device, buffer, &requirements);
    const auto memoryType = findMemoryType(context, requirements.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!memoryType) {
        driver.destroyBuffer(state.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        context.error = "Vulkan device has no host-visible coherent memory type";
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = *memoryType;
    if (vkFail(context, driver.allocateMemory(state.device, &allocation, nullptr, &memory), "vkAllocateMemory") !=
        VERNON_STATUS_OK) {
        driver.destroyBuffer(state.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    if (vkFail(context, driver.bindBufferMemory(state.device, buffer, memory, 0), "vkBindBufferMemory") ==
        VERNON_STATUS_OK)
        return true;
    driver.freeMemory(state.device, memory, nullptr);
    driver.destroyBuffer(state.device, buffer, nullptr);
    memory = VK_NULL_HANDLE;
    buffer = VK_NULL_HANDLE;
    return false;
}

bool createVulkanShaderModule(VernonRuntimeContext &context, const void *data, size_t size, VkShaderModule &module,
                              uint32_t pushConstantBaseOffset) {
    if (!data || !size || size % sizeof(uint32_t) != 0) {
        context.error = "Vulkan shader artifact is not aligned SPIR-V";
        return false;
    }
    std::vector<uint32_t> alignedData;
    const uint32_t *spirv = static_cast<const uint32_t *>(data);
    if (pushConstantBaseOffset || reinterpret_cast<uintptr_t>(data) % alignof(uint32_t) != 0) {
        alignedData.resize(size / sizeof(uint32_t));
        std::memcpy(alignedData.data(), data, size);
        spirv = alignedData.data();
    }
    if (pushConstantBaseOffset) {
        constexpr uint32_t spirvMagic = 0x07230203;
        constexpr uint16_t opTypePointer = 32;
        constexpr uint16_t opVariable = 59;
        constexpr uint16_t opMemberDecorate = 72;
        constexpr uint32_t storageClassPushConstant = 9;
        constexpr uint32_t decorationOffset = 35;
        if (alignedData.size() < 5 || alignedData[0] != spirvMagic) {
            context.error = "Vulkan shader artifact has an invalid SPIR-V header";
            return false;
        }
        uint32_t pushConstantStruct = 0;
        std::vector<std::pair<uint32_t, uint32_t>> pushConstantPointers;
        for (size_t index = 5; index < alignedData.size();) {
            const uint32_t instruction = alignedData[index];
            const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16);
            const uint16_t opcode = static_cast<uint16_t>(instruction);
            if (!wordCount || wordCount > alignedData.size() - index) {
                context.error = "Vulkan shader artifact has malformed SPIR-V";
                return false;
            }
            if (opcode == opTypePointer && wordCount == 4 && alignedData[index + 2] == storageClassPushConstant)
                pushConstantPointers.emplace_back(alignedData[index + 1], alignedData[index + 3]);
            else if (opcode == opVariable && wordCount >= 4 && alignedData[index + 3] == storageClassPushConstant) {
                const uint32_t pointerType = alignedData[index + 1];
                const auto pointer = std::find_if(pushConstantPointers.begin(), pushConstantPointers.end(),
                                                  [=](const auto &entry) { return entry.first == pointerType; });
                if (pointer != pushConstantPointers.end())
                    pushConstantStruct = pointer->second;
            }
            index += wordCount;
        }
        bool relocated = false;
        for (size_t index = 5; index < alignedData.size();) {
            const uint32_t instruction = alignedData[index];
            const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16);
            const uint16_t opcode = static_cast<uint16_t>(instruction);
            if (opcode == opMemberDecorate && wordCount >= 5 && alignedData[index + 1] == pushConstantStruct &&
                alignedData[index + 3] == decorationOffset) {
                if (alignedData[index + 4] > std::numeric_limits<uint32_t>::max() - pushConstantBaseOffset) {
                    context.error = "Vulkan push-constant offset overflows uint32";
                    return false;
                }
                alignedData[index + 4] += pushConstantBaseOffset;
                relocated = true;
            }
            index += wordCount;
        }
        if (!pushConstantStruct || !relocated) {
            context.error = "Vulkan shader push-constant block could not be relocated";
            return false;
        }
    }
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = size;
    info.pCode = spirv;
    return vkFail(context, vulkanDriver().createShaderModule(vulkanState(context).device, &info, nullptr, &module),
                  "vkCreateShaderModule") == VERNON_STATUS_OK;
}

void destroyVulkanShaderModule(VernonRuntimeContext &context, VkShaderModule &module) {
    if (!module)
        return;
    vulkanDriver().destroyShaderModule(vulkanState(context).device, module, nullptr);
    module = VK_NULL_HANDLE;
}
#endif

bool createVulkanBuffer(VernonDeviceBuffer &buffer) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    auto *state = new VulkanBufferState();
    if (!createVulkanBuffer(*buffer.context, buffer.size, state->buffer, state->memory)) {
        delete state;
        return false;
    }
    installRuntimeBackendState(buffer, state);
    return true;
#else
    (void)buffer;
    return false;
#endif
}

void destroyVulkanBuffer(VernonDeviceBuffer &buffer) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(*buffer.context);
    VulkanBufferState &bufferState = vulkanBufferState(buffer);
    if (bufferState.buffer)
        driver.destroyBuffer(state.device, bufferState.buffer, nullptr);
    if (bufferState.memory)
        driver.freeMemory(state.device, bufferState.memory, nullptr);
    bufferState.buffer = VK_NULL_HANDLE;
    bufferState.memory = VK_NULL_HANDLE;
#else
    (void)buffer;
#endif
}

VernonStatus copyToVulkanBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!source || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer upload";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    void *mapped = nullptr;
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(*buffer.context);
    VulkanBufferState &bufferState = vulkanBufferState(buffer);
    if (vkFail(*buffer.context, driver.mapMemory(state.device, bufferState.memory, offset, size, 0, &mapped),
               "vkMapMemory") != VERNON_STATUS_OK)
        return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(mapped, source, size);
    driver.unmapMemory(state.device, bufferState.memory);
    return VERNON_STATUS_OK;
#else
    (void)buffer;
    (void)offset;
    (void)source;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus copyFromVulkanBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!destination || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer readback";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    void *mapped = nullptr;
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(*buffer.context);
    const VulkanBufferState &bufferState = vulkanBufferState(buffer);
    if (vkFail(*buffer.context, driver.mapMemory(state.device, bufferState.memory, offset, size, 0, &mapped),
               "vkMapMemory") != VERNON_STATUS_OK)
        return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(destination, mapped, size);
    driver.unmapMemory(state.device, bufferState.memory);
    return VERNON_STATUS_OK;
#else
    (void)buffer;
    (void)offset;
    (void)destination;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

bool createVulkanTexture(VernonDeviceTexture &texture) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VernonRuntimeContext &context = *texture.context;
    const auto format = textureFormat(texture.format);
    if (!format) {
        context.error = "sampled texture format is unsupported";
        return false;
    }
    uint32_t maximumMipLevels = 1;
    uint32_t maximumExtent = std::max({texture.width, texture.height, texture.depth});
    while (maximumExtent > 1) {
        maximumExtent >>= 1;
        ++maximumMipLevels;
    }
    if (texture.mipLevels > maximumMipLevels) {
        context.error = "sampled texture mip count exceeds its extent";
        return false;
    }
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &contextState = vulkanState(context);
    VkFormatProperties properties{};
    driver.getPhysicalDeviceFormatProperties(contextState.physicalDevice, *format, &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
        context.error = "Vulkan device cannot sample the requested texture format";
        return false;
    }
    auto *textureState = new VulkanTextureState();
    textureState->format = *format;
    textureState->colorAttachment = texture.dimension == VERNON_TEXTURE_2D &&
                                    (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
    installRuntimeBackendState(texture, textureState);
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.flags = texture.dimension == VERNON_TEXTURE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    imageInfo.imageType = texture.dimension == VERNON_TEXTURE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    imageInfo.format = *format;
    imageInfo.extent = {texture.width, texture.height, texture.dimension == VERNON_TEXTURE_3D ? texture.depth : 1};
    imageInfo.mipLevels = texture.mipLevels;
    imageInfo.arrayLayers = texture.dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (textureState->colorAttachment)
        imageInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkFail(context, driver.createImage(contextState.device, &imageInfo, nullptr, &textureState->image),
               "vkCreateImage") != VERNON_STATUS_OK) {
        destroyRuntimeBackendState(texture);
        return false;
    }
    VkMemoryRequirements requirements{};
    driver.getImageMemoryRequirements(contextState.device, textureState->image, &requirements);
    const auto memoryType = findMemoryType(context, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!memoryType) {
        context.error = "Vulkan device has no device-local image memory";
        destroyVulkanTexture(texture);
        destroyRuntimeBackendState(texture);
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = *memoryType;
    if (vkFail(context, driver.allocateMemory(contextState.device, &allocation, nullptr, &textureState->memory),
               "vkAllocateMemory") != VERNON_STATUS_OK ||
        vkFail(context, driver.bindImageMemory(contextState.device, textureState->image, textureState->memory, 0),
               "vkBindImageMemory") != VERNON_STATUS_OK) {
        destroyVulkanTexture(texture);
        destroyRuntimeBackendState(texture);
        return false;
    }
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = textureState->image;
    viewInfo.viewType = texture.dimension == VERNON_TEXTURE_3D     ? VK_IMAGE_VIEW_TYPE_3D
                        : texture.dimension == VERNON_TEXTURE_CUBE ? VK_IMAGE_VIEW_TYPE_CUBE
                                                                   : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = *format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = texture.mipLevels;
    viewInfo.subresourceRange.layerCount = texture.dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
    if (vkFail(context, driver.createImageView(contextState.device, &viewInfo, nullptr, &textureState->view),
               "vkCreateImageView") != VERNON_STATUS_OK) {
        destroyVulkanTexture(texture);
        destroyRuntimeBackendState(texture);
        return false;
    }
    return true;
#else
    (void)texture;
    return false;
#endif
}

void destroyVulkanTexture(VernonDeviceTexture &texture) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &contextState = vulkanState(*texture.context);
    VulkanTextureState &state = vulkanTextureState(texture);
    driver.deviceWaitIdle(contextState.device);
    if (state.view)
        driver.destroyImageView(contextState.device, state.view, nullptr);
    if (state.image)
        driver.destroyImage(contextState.device, state.image, nullptr);
    if (state.memory)
        driver.freeMemory(contextState.device, state.memory, nullptr);
    state.view = VK_NULL_HANDLE;
    state.image = VK_NULL_HANDLE;
    state.memory = VK_NULL_HANDLE;
#else
    (void)texture;
#endif
}

VernonStatus copyToVulkanTexture(VernonDeviceTexture &texture, const void *source, size_t size) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    const uint32_t layers = texture.dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
    const auto expected = rgba8ByteSize(texture.width, texture.height, layers);
    if ((texture.dimension != VERNON_TEXTURE_2D && texture.dimension != VERNON_TEXTURE_CUBE) ||
        texture.format != VERNON_TEXTURE_RGBA8_UNORM || texture.mipLevels != 1 || !source || !expected ||
        size != *expected) {
        texture.context->error = "Vulkan host upload supports one-mip RGBA8 2D and Cube textures only";
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }
    VernonRuntimeContext &context = *texture.context;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!createVulkanBuffer(context, size, staging, memory))
        return VERNON_STATUS_INTERNAL_ERROR;
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(context);
    VulkanTextureState &textureState = vulkanTextureState(texture);
    void *mapped = nullptr;
    if (vkFail(context, driver.mapMemory(state.device, memory, 0, size, 0, &mapped), "vkMapMemory") !=
        VERNON_STATUS_OK) {
        driver.destroyBuffer(state.device, staging, nullptr);
        driver.freeMemory(state.device, memory, nullptr);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(mapped, source, size);
    driver.unmapMemory(state.device, memory);
    const bool copied = submitCommands(context, [&](VkCommandBuffer command) {
        transitionVulkanImageLayout(command, texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = layers;
        region.imageExtent = {texture.width, texture.height, 1};
        driver.cmdCopyBufferToImage(command, staging, textureState.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                    &region);
        transitionVulkanImageLayout(command, texture, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    driver.destroyBuffer(state.device, staging, nullptr);
    driver.freeMemory(state.device, memory, nullptr);
    return copied ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#else
    (void)texture;
    (void)source;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus copyFromVulkanTexture(const VernonDeviceTexture &texture, void *destination, size_t size) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    const auto expected = rgba8ByteSize(texture.width, texture.height, 1);
    if (texture.dimension != VERNON_TEXTURE_2D || texture.format != VERNON_TEXTURE_RGBA8_UNORM ||
        texture.mipLevels != 1 || !destination || !expected || size != *expected) {
        texture.context->error = "Vulkan host readback supports one-mip RGBA8 2D textures only";
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }
    auto &mutableTexture = const_cast<VernonDeviceTexture &>(texture);
    VernonRuntimeContext &context = *texture.context;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!createVulkanBuffer(context, size, staging, memory))
        return VERNON_STATUS_INTERNAL_ERROR;
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(context);
    VulkanTextureState &textureState = vulkanTextureState(mutableTexture);
    const VkImageLayout restoreLayout = textureState.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                            : textureState.layout;
    const bool copied = submitCommands(context, [&](VkCommandBuffer command) {
        transitionVulkanImageLayout(command, mutableTexture, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {texture.width, texture.height, 1};
        driver.cmdCopyImageToBuffer(command, textureState.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                                    &region);
        transitionVulkanImageLayout(command, mutableTexture, restoreLayout);
    });
    if (copied) {
        void *mapped = nullptr;
        if (vkFail(context, driver.mapMemory(state.device, memory, 0, size, 0, &mapped), "vkMapMemory") ==
            VERNON_STATUS_OK) {
            std::memcpy(destination, mapped, size);
            driver.unmapMemory(state.device, memory);
        } else {
            driver.destroyBuffer(state.device, staging, nullptr);
            driver.freeMemory(state.device, memory, nullptr);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
    }
    driver.destroyBuffer(state.device, staging, nullptr);
    driver.freeMemory(state.device, memory, nullptr);
    return copied ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#else
    (void)texture;
    (void)destination;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

bool createVulkanSampler(VernonDeviceSampler &sampler) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    const auto wrapU = samplerAddressMode(sampler.descriptor.wrap_u);
    const auto wrapV = samplerAddressMode(sampler.descriptor.wrap_v);
    const auto wrapW = samplerAddressMode(sampler.descriptor.wrap_w);
    const auto minFilter = samplerFilter(sampler.descriptor.min_filter);
    const auto magFilter = samplerFilter(sampler.descriptor.mag_filter);
    if (!wrapU || !wrapV || !wrapW || !minFilter || !magFilter ||
        sampler.descriptor.mip_filter > VERNON_SAMPLER_LINEAR) {
        sampler.context->error = "sampler descriptor contains an invalid enum value";
        return false;
    }
    VkSamplerCreateInfo createInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    createInfo.magFilter = *magFilter;
    createInfo.minFilter = *minFilter;
    createInfo.mipmapMode = sampler.descriptor.mip_filter == VERNON_SAMPLER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                                                   : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    createInfo.addressModeU = *wrapU;
    createInfo.addressModeV = *wrapV;
    createInfo.addressModeW = *wrapW;
    createInfo.minLod = 0.0f;
    createInfo.maxLod = VK_LOD_CLAMP_NONE;
    createInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    auto *state = new VulkanSamplerState();
    if (vkFail(
            *sampler.context,
            vulkanDriver().createSampler(vulkanState(*sampler.context).device, &createInfo, nullptr, &state->sampler),
            "vkCreateSampler") != VERNON_STATUS_OK) {
        delete state;
        return false;
    }
    installRuntimeBackendState(sampler, state);
    return true;
#else
    (void)sampler;
    return false;
#endif
}

void destroyVulkanSampler(VernonDeviceSampler &sampler) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanSamplerState &samplerState = vulkanSamplerState(sampler);
    if (!samplerState.sampler)
        return;
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &state = vulkanState(*sampler.context);
    driver.deviceWaitIdle(state.device);
    driver.destroySampler(state.device, samplerState.sampler, nullptr);
    samplerState.sampler = VK_NULL_HANDLE;
#else
    (void)sampler;
#endif
}

bool loadVulkanKernel(VernonRuntimeContext &context, const void *artifact, size_t artifactSize, const char *reflection,
                      size_t reflectionSize, const char *entry, size_t entrySize, VulkanKernelState &state,
                      ReflectedEntry &metadata) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (artifactSize % sizeof(uint32_t) != 0) {
        context.error = "Vulkan artifact is not aligned SPIR-V";
        return false;
    }
    try {
        const nlohmann::json parsed = nlohmann::json::parse(reflection, reflection + reflectionSize, nullptr, false);
        const std::string entryName(entry, entrySize);
        if (parsed.is_discarded() || !parseReflection(parsed, entryName, metadata, context.error))
            return false;
        VulkanDriver &driver = vulkanDriver();
        if (!createVulkanShaderModule(context, artifact, artifactSize, state.shader))
            return false;
        VulkanContextState &contextState = vulkanState(context);
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(metadata.arguments.size());
        uint32_t index = 0;
        for (ReflectedArgument &argument : metadata.arguments) {
            if (argument.kind == "builtin")
                continue;
            if (argument.descriptorSet != 0) {
                context.error = "Vulkan compute supports descriptor set zero only";
                destroyVulkanKernel(context, state);
                return false;
            }
            if (argument.binding == UINT32_MAX)
                argument.binding = index;
            bindings.push_back(
                {argument.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
            ++index;
        }
        VkDescriptorSetLayoutCreateInfo descriptorInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptorInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        descriptorInfo.pBindings = bindings.data();
        if (vkFail(context,
                   driver.createDescriptorSetLayout(contextState.device, &descriptorInfo, nullptr,
                                                    &state.descriptorSetLayout),
                   "vkCreateDescriptorSetLayout") != VERNON_STATUS_OK) {
            destroyVulkanKernel(context, state);
            return false;
        }
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &state.descriptorSetLayout;
        if (vkFail(context,
                   driver.createPipelineLayout(contextState.device, &layoutInfo, nullptr, &state.pipelineLayout),
                   "vkCreatePipelineLayout") != VERNON_STATUS_OK) {
            destroyVulkanKernel(context, state);
            return false;
        }
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = state.shader;
        stage.pName = entryName.c_str();
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stage;
        pipelineInfo.layout = state.pipelineLayout;
        if (vkFail(context,
                   driver.createComputePipelines(contextState.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                                 &state.pipeline),
                   "vkCreateComputePipelines") != VERNON_STATUS_OK) {
            destroyVulkanKernel(context, state);
            return false;
        }
        return true;
    } catch (const std::exception &exception) {
        context.error = std::string("failed to load Vulkan artifact: ") + exception.what();
        destroyVulkanKernel(context, state);
        return false;
    }
#else
    (void)context;
    (void)artifact;
    (void)artifactSize;
    (void)reflection;
    (void)reflectionSize;
    (void)entry;
    (void)entrySize;
    (void)state;
    (void)metadata;
    return false;
#endif
}

void destroyVulkanKernel(VernonRuntimeContext &context, VulkanKernelState &state) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &contextState = vulkanState(context);
    driver.deviceWaitIdle(contextState.device);
    if (state.pipeline)
        driver.destroyPipeline(contextState.device, state.pipeline, nullptr);
    if (state.pipelineLayout)
        driver.destroyPipelineLayout(contextState.device, state.pipelineLayout, nullptr);
    if (state.descriptorSetLayout)
        driver.destroyDescriptorSetLayout(contextState.device, state.descriptorSetLayout, nullptr);
    if (state.shader)
        driver.destroyShaderModule(contextState.device, state.shader, nullptr);
    state = {};
#else
    (void)context;
    (void)state;
#endif
}

VernonStatus launchVulkanKernel(VernonRuntimeContext &context, const VulkanKernelState &state,
                                const ReflectedEntry &metadata, VernonLaunchSize globalSize,
                                const VernonLaunchArgument *arguments, size_t argumentCount) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (argumentCount > UINT32_MAX) {
        context.error = "Vulkan compute argument count exceeds API limits";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    VulkanDriver &driver = vulkanDriver();
    VulkanContextState &contextState = vulkanState(context);
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> scalarBuffers;
    scalarBuffers.reserve(argumentCount);
    const auto cleanup = [&] {
        if (commandBuffer)
            driver.freeCommandBuffers(contextState.device, contextState.commandPool, 1, &commandBuffer);
        if (descriptorPool)
            driver.destroyDescriptorPool(contextState.device, descriptorPool, nullptr);
        for (const auto &[buffer, memory] : scalarBuffers) {
            driver.destroyBuffer(contextState.device, buffer, nullptr);
            driver.freeMemory(contextState.device, memory, nullptr);
        }
    };
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(argumentCount)};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = argumentCount ? 1u : 0u;
    poolInfo.pPoolSizes = argumentCount ? &poolSize : nullptr;
    if (vkFail(context, driver.createDescriptorPool(contextState.device, &poolInfo, nullptr, &descriptorPool),
               "vkCreateDescriptorPool") != VERNON_STATUS_OK)
        return VERNON_STATUS_INTERNAL_ERROR;
    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &state.descriptorSetLayout;
    VkDescriptorSet descriptorSet{};
    if (vkFail(context, driver.allocateDescriptorSets(contextState.device, &setInfo, &descriptorSet),
               "vkAllocateDescriptorSets") != VERNON_STATUS_OK) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSet> writes;
    bufferInfos.reserve(argumentCount);
    writes.reserve(argumentCount);
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : metadata.arguments) {
        if (reflected.kind == "builtin")
            continue;
        const VernonLaunchArgument &argument = arguments[supplied++];
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        if (argument.kind == VERNON_LAUNCH_TENSOR) {
            buffer = vulkanBufferState(*argument.buffer).buffer;
            size = argument.buffer->size;
        } else {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (!createVulkanBuffer(context, argument.scalar_size, buffer, memory)) {
                cleanup();
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            scalarBuffers.emplace_back(buffer, memory);
            void *mapped = nullptr;
            if (vkFail(context, driver.mapMemory(contextState.device, memory, 0, argument.scalar_size, 0, &mapped),
                       "vkMapMemory") != VERNON_STATUS_OK) {
                cleanup();
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            std::memcpy(mapped, argument.scalar_data, argument.scalar_size);
            driver.unmapMemory(contextState.device, memory);
            size = argument.scalar_size;
        }
        bufferInfos.push_back({buffer, 0, size});
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptorSet;
        write.dstBinding = reflected.binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfos.back();
        writes.push_back(write);
    }
    driver.updateDescriptorSets(contextState.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandInfo.commandPool = contextState.commandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    if (vkFail(context, driver.allocateCommandBuffers(contextState.device, &commandInfo, &commandBuffer),
               "vkAllocateCommandBuffers") != VERNON_STATUS_OK) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkFail(context, driver.beginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer") !=
        VERNON_STATUS_OK) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    driver.cmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    driver.cmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipelineLayout, 0, 1,
                                 &descriptorSet, 0, nullptr);
    const uint32_t *workgroup = metadata.workgroup;
    if (!workgroup[0] || !workgroup[1] || !workgroup[2]) {
        context.error = "Vulkan workgroup dimensions must be positive";
        cleanup();
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    driver.cmdDispatch(commandBuffer, (globalSize.x - 1) / workgroup[0] + 1, (globalSize.y - 1) / workgroup[1] + 1,
                       (globalSize.z - 1) / workgroup[2] + 1);
    if (vkFail(context, driver.endCommandBuffer(commandBuffer), "vkEndCommandBuffer") != VERNON_STATUS_OK) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer;
    const bool submitted =
        vkFail(context, driver.queueSubmit(contextState.queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit") ==
            VERNON_STATUS_OK &&
        vkFail(context, driver.queueWaitIdle(contextState.queue), "vkQueueWaitIdle") == VERNON_STATUS_OK;
    cleanup();
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#else
    (void)context;
    (void)state;
    (void)metadata;
    (void)globalSize;
    (void)arguments;
    (void)argumentCount;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus synchronizeVulkan(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    return vkFail(context, vulkanDriver().deviceWaitIdle(vulkanState(context).device), "vkDeviceWaitIdle");
#else
    (void)context;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
