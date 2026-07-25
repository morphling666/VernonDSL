#include "backend_vulkan.h"

#include "rhi_adapter/adapter_internal.h"
#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
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
    return vulkanState(context).beginCommands(command, error);
}

bool submitVulkanCommands(VernonRuntimeContext &context, VkCommandBuffer command, std::string &error) {
    return vulkanState(context).submitCommands(command, error);
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
    rhi::vulkan::driver().cmdPipelineBarrier(command, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1,
                                             &barrier);
    state.layout = newLayout;
}

namespace {
#endif

} // namespace

bool initializeVulkanContext(VernonRuntimeContext &context, uint32_t deviceIndex) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    installRuntimeBackendState(context, new VulkanContextState());
    VulkanContextState &state = vulkanState(context);
    if (!state.initialize(deviceIndex, context.error)) {
        destroyRuntimeBackendState(context);
        return false;
    }
    state.adapter = createBorrowedVulkanRhiAdapter(state);
    if (!state.adapter) {
        context.error = "failed to create borrowed Vulkan RHI adapter";
        state.shutdown();
        destroyRuntimeBackendState(context);
        return false;
    }
    return true;
#else
    (void)context;
    (void)deviceIndex;
    return false;
#endif
}

void destroyVulkanContext(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanContextState &state = vulkanState(context);
    vernonRuntimeRhiAdapterDestroy(state.adapter);
    state.adapter = nullptr;
    state.shutdown();
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
    VulkanBufferState state;
    if (!vulkanState(context).createBuffer(
            state, size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, context.error))
        return false;
    buffer = state.buffer;
    memory = state.memory;
    return true;
}

void destroyVulkanBuffer(VernonRuntimeContext &context, VkBuffer &buffer, VkDeviceMemory &memory) {
    VulkanBufferState state{buffer, memory};
    vulkanState(context).destroyBuffer(state);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

#endif

bool createVulkanBuffer(VernonDeviceBuffer &buffer) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    auto *state = new VulkanBufferState();
    if (!vulkanState(*buffer.context)
             .createBuffer(*state, buffer.size,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer.context->error)) {
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
    vulkanState(*buffer.context).destroyBuffer(vulkanBufferState(buffer));
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
    rhi::vulkan::Driver &driver = rhi::vulkan::driver();
    VulkanContextState &state = vulkanState(*buffer.context);
    VulkanBufferState &bufferState = vulkanBufferState(buffer);
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(true, size, 16, staging, stagingOffset, mapped, buffer.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(mapped, source, size);
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!state.beginCommands(command, buffer.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    const VkBufferCopy copy{stagingOffset, offset, size};
    driver.cmdCopyBuffer(command, staging, bufferState.buffer, 1, &copy);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = bufferState.buffer;
    barrier.offset = offset;
    barrier.size = size;
    driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                              nullptr, 1, &barrier, 0, nullptr);
    return state.submitCommands(command, buffer.context->error) ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
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
    rhi::vulkan::Driver &driver = rhi::vulkan::driver();
    VulkanContextState &state = vulkanState(*buffer.context);
    const VulkanBufferState &bufferState = vulkanBufferState(buffer);
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(false, size, 16, staging, stagingOffset, mapped, buffer.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!state.beginCommands(command, buffer.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    VkBufferMemoryBarrier before{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.buffer = bufferState.buffer;
    before.offset = offset;
    before.size = size;
    driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                              nullptr, 1, &before, 0, nullptr);
    const VkBufferCopy copy{offset, stagingOffset, size};
    driver.cmdCopyBuffer(command, bufferState.buffer, staging, 1, &copy);
    VkBufferMemoryBarrier after = before;
    after.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                          VK_ACCESS_INDEX_READ_BIT;
    driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                              nullptr, 1, &after, 0, nullptr);
    if (!state.submitCommands(command, buffer.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(destination, mapped, size);
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
    rhi::vulkan::Driver &driver = rhi::vulkan::driver();
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
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.viewType = texture.dimension == VERNON_TEXTURE_3D     ? VK_IMAGE_VIEW_TYPE_3D
                        : texture.dimension == VERNON_TEXTURE_CUBE ? VK_IMAGE_VIEW_TYPE_CUBE
                                                                   : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = *format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = texture.mipLevels;
    viewInfo.subresourceRange.layerCount = texture.dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
    if (!contextState.createImage(*textureState, imageInfo, viewInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                  context.error)) {
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
    VulkanContextState &contextState = vulkanState(*texture.context);
    (void)contextState.synchronize(texture.context->error);
    contextState.destroyImage(vulkanTextureState(texture));
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
    rhi::vulkan::Driver &driver = rhi::vulkan::driver();
    VulkanContextState &state = vulkanState(context);
    VulkanTextureState &textureState = vulkanTextureState(texture);
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(true, size, 16, staging, stagingOffset, mapped, context.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(mapped, source, size);
    const bool copied = submitCommands(context, [&](VkCommandBuffer command) {
        transitionVulkanImageLayout(command, texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = stagingOffset;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = layers;
        region.imageExtent = {texture.width, texture.height, 1};
        driver.cmdCopyBufferToImage(command, staging, textureState.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                    &region);
        transitionVulkanImageLayout(command, texture, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
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
    rhi::vulkan::Driver &driver = rhi::vulkan::driver();
    VulkanContextState &state = vulkanState(context);
    VulkanTextureState &textureState = vulkanTextureState(mutableTexture);
    VkDeviceSize stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(false, size, 16, staging, stagingOffset, mapped, context.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    const VkImageLayout restoreLayout = textureState.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                            : textureState.layout;
    const bool copied = submitCommands(context, [&](VkCommandBuffer command) {
        transitionVulkanImageLayout(command, mutableTexture, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = stagingOffset;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {texture.width, texture.height, 1};
        driver.cmdCopyImageToBuffer(command, textureState.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                                    &region);
        transitionVulkanImageLayout(command, mutableTexture, restoreLayout);
    });
    if (copied)
        std::memcpy(destination, mapped, size);
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
    if (!vulkanState(*sampler.context).createSampler(*state, createInfo, sampler.context->error)) {
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
    VulkanContextState &state = vulkanState(*sampler.context);
    (void)state.synchronize(sampler.context->error);
    state.destroySampler(samplerState);
#else
    (void)sampler;
#endif
}

VernonStatus synchronizeVulkan(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    return vulkanState(context).synchronize(context.error) ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#else
    (void)context;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
