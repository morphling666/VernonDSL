#ifndef VERNON_RUNTIME_BACKEND_VULKAN_H
#define VERNON_RUNTIME_BACKEND_VULKAN_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "../rhi/vulkan_backend.h"
#endif

#include <string>
#include <vector>

struct VernonDeviceBuffer;
struct VernonDeviceSampler;
struct VernonDeviceTexture;
struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

#if defined(VERNON_HAS_VULKAN_RUNTIME)
struct VulkanContextState : rhi::vulkan::DeviceState {
    VernonRuntimeRhiAdapter *adapter{};
};
using VulkanBufferState = rhi::vulkan::Buffer;
using VulkanTextureState = rhi::vulkan::Image;
using VulkanSamplerState = rhi::vulkan::Sampler;

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

struct VulkanPipelineState {
    struct Binding {
        enum Source {
            EXTERNAL_VERTEX,
            EXTERNAL_TEXTURE,
            EXTERNAL_SAMPLER,
            EXTERNAL_UNIFORM,
            IMPLICIT_SAMPLER,
            RESOLUTION
        };
        Source source{};
        uint32_t externalSlot{};
        TensorPackingLayout packing;
        std::vector<uint8_t> storage;
    };

    VernonRuntimeCorePipeline *rhiComputePipeline{};
    VernonRuntimeCoreBindings *rhiComputeBindings{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiComputeLayout;
    std::vector<VernonRuntimeProviderBindingValue> rhiComputeValues;
    std::vector<uint64_t> rhiComputeResourceOffsets;
    uint32_t rhiComputeWorkgroup[3]{1, 1, 1};
    VernonRuntimeCorePipeline *rhiGraphicsPipeline{};
    VernonRuntimeCoreBindings *rhiGraphicsBindings{};
    VernonRuntimeCoreGraphicsVariant *rhiGraphicsVariant{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiGraphicsLayout;
    std::vector<VernonRuntimeProviderVertexAttribute> rhiGraphicsVertexAttributes;
    std::vector<VernonRuntimeProviderBindingValue> rhiGraphicsValues;
    std::vector<Binding> rhiGraphicsBindingPlan;
    std::vector<uint32_t> rhiGraphicsFormats;
    uint32_t rhiGraphicsDepthFormat{};
    uint64_t rhiGraphicsVertexLayoutIdentity{};
    uint32_t rhiGraphicsTopology{};
};
#endif

bool probeVulkan(std::string &diagnostic);
bool initializeVulkanContext(VernonRuntimeContext &context, uint32_t deviceIndex);
void destroyVulkanContext(VernonRuntimeContext &context);
VernonStatus synchronizeVulkan(VernonRuntimeContext &context);

#if defined(VERNON_HAS_VULKAN_RUNTIME)
bool createVulkanBuffer(VernonRuntimeContext &context, VkDeviceSize size, VkBuffer &buffer, VkDeviceMemory &memory);
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

} // namespace vernon::runtime

#endif
