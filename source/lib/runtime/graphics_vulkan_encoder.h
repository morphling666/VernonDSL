#ifndef VERNON_RUNTIME_GRAPHICS_VULKAN_ENCODER_H
#define VERNON_RUNTIME_GRAPHICS_VULKAN_ENCODER_H

#include "graphics_invocation_planner.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan_driver.h"
#endif

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime {

#if defined(VERNON_HAS_VULKAN_RUNTIME)
struct VulkanDescriptorBindingKey {
    uint32_t binding{};
    uint32_t descriptorType{};
    uint32_t descriptorCount{};
    uint32_t stageFlags{};

    bool operator<(const VulkanDescriptorBindingKey &other) const {
        return std::tie(binding, descriptorType, descriptorCount, stageFlags) <
               std::tie(other.binding, other.descriptorType, other.descriptorCount, other.stageFlags);
    }
};

struct VulkanDescriptorSetLayoutKey {
    std::vector<VulkanDescriptorBindingKey> bindings;

    bool operator<(const VulkanDescriptorSetLayoutKey &other) const { return bindings < other.bindings; }
};

struct VulkanPushConstantRangeKey {
    uint32_t stageFlags{};
    uint32_t offset{};
    uint32_t size{};

    bool operator<(const VulkanPushConstantRangeKey &other) const {
        return std::tie(stageFlags, offset, size) < std::tie(other.stageFlags, other.offset, other.size);
    }
};

struct VulkanPipelineLayoutKey {
    std::vector<VulkanDescriptorSetLayoutKey> descriptorSets;
    std::vector<VulkanPushConstantRangeKey> pushConstantRanges;

    bool operator<(const VulkanPipelineLayoutKey &other) const {
        return std::tie(descriptorSets, pushConstantRanges) < std::tie(other.descriptorSets, other.pushConstantRanges);
    }
};

struct VulkanRenderPassCompatibilityKey {
    // VK_FORMAT_UNDEFINED marks an unused color-attachment location.
    std::vector<uint32_t> colorFormatsByLocation;

    bool operator<(const VulkanRenderPassCompatibilityKey &other) const {
        return colorFormatsByLocation < other.colorFormatsByLocation;
    }
};

struct VulkanVertexBindingKey {
    uint32_t binding{};
    uint32_t stride{};
    uint32_t inputRate{};

    bool operator<(const VulkanVertexBindingKey &other) const {
        return std::tie(binding, stride, inputRate) < std::tie(other.binding, other.stride, other.inputRate);
    }
};

struct VulkanVertexAttributeKey {
    uint32_t location{};
    uint32_t binding{};
    uint32_t format{};
    uint32_t offset{};

    bool operator<(const VulkanVertexAttributeKey &other) const {
        return std::tie(location, binding, format, offset) <
               std::tie(other.location, other.binding, other.format, other.offset);
    }
};

struct VulkanGraphicsPipelineKey {
    VulkanRenderPassCompatibilityKey renderPass;
    uint32_t topology{};
    std::vector<VulkanVertexBindingKey> vertexBindings;
    std::vector<VulkanVertexAttributeKey> vertexAttributes;
    VulkanPipelineLayoutKey layout;

    bool operator<(const VulkanGraphicsPipelineKey &other) const {
        return std::tie(renderPass, topology, vertexBindings, vertexAttributes, layout) <
               std::tie(other.renderPass, other.topology, other.vertexBindings, other.vertexAttributes, other.layout);
    }
};

struct VulkanGraphicsCache {
    std::map<VulkanDescriptorSetLayoutKey, VkDescriptorSetLayout> descriptorSetLayouts;
    std::map<VulkanPipelineLayoutKey, VkPipelineLayout> pipelineLayouts;
    std::map<VulkanGraphicsPipelineKey, VkPipeline> graphicsPipelines;
    size_t descriptorSetLayoutCreations{};
    size_t pipelineLayoutCreations{};
    size_t graphicsPipelineCreations{};
};

struct VulkanPipelineState {
    VernonLoadedKernel *computeKernel{};
    VkShaderModule vertex{};
    VkShaderModule fragment{};
    std::string vertexEntry;
    std::string fragmentEntry;
    VulkanGraphicsCache graphicsCache;
};

struct VulkanGraphicsState {
    VernonRuntimeContext *context{};
    VkShaderModule vertex{};
    VkShaderModule fragment{};
    const std::string *vertexEntry{};
    const std::string *fragmentEntry{};
    VulkanGraphicsCache *cache{};
    bool barrier{};
};

struct VulkanPushConstantStageRange {
    uint32_t offset{};
    uint32_t size{};
};

struct VulkanPushConstantRanges {
    VulkanPushConstantStageRange vertex;
    VulkanPushConstantStageRange fragment;
};

bool planVulkanPushConstantRanges(const Variant &variant, uint32_t maximumSize, VulkanPushConstantRanges &ranges,
                                  std::string &error);

VernonStatus encodeAndSubmitVulkanGraphics(const VulkanGraphicsState &state, const Variant &variant,
                                           const VernonPipelineInvocation &invocation,
                                           const PlannedGraphicsInvocation &plan, std::string &error);

void destroyVulkanGraphicsCache(VernonRuntimeContext *context, VulkanGraphicsCache &cache);
#endif

} // namespace vernon::runtime

#endif
