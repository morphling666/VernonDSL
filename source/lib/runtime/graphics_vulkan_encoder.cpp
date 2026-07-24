#include "graphics_vulkan_encoder.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)

#include "backend_vulkan.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

VernonStatus fail(std::string &error, const std::string &message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    error = message;
    return status;
}

bool vkCheck(VkResult result, const char *operation, std::string &error) {
    if (result == VK_SUCCESS)
        return true;
    error = std::string(operation) + " failed with VkResult " + std::to_string(result);
    return false;
}

struct PushConstantLayout {
    uint32_t alignment{};
    uint32_t storageSize{};
    uint32_t matrixStride{};
};

std::optional<PushConstantLayout> pushConstantLayout(const ParameterUse &use, std::string &error) {
    uint32_t scalarSize = 0;
    if (use.dtype == "i32" || use.dtype == "u32" || use.dtype == "f32")
        scalarSize = 4;
    else if (use.dtype == "f16")
        scalarSize = 2;
    else if (use.dtype == "f64")
        scalarSize = 8;
    else {
        error = "Vulkan push constant has unsupported reflected dtype '" + use.dtype + "'";
        return std::nullopt;
    }
    if (use.shape.empty())
        return PushConstantLayout{scalarSize, scalarSize, 0};
    if (use.shape.size() > 2) {
        error = "Vulkan push constants support only scalars, vectors, and matrices";
        return std::nullopt;
    }
    const uint64_t rows = use.shape[0];
    if (rows < 2 || rows > 4) {
        error = "Vulkan push constant vector dimension must be between 2 and 4";
        return std::nullopt;
    }
    const uint32_t vectorAlignment = (rows == 2 ? 2u : 4u) * scalarSize;
    const uint32_t vectorSize = static_cast<uint32_t>(rows) * scalarSize;
    if (use.shape.size() == 1)
        return PushConstantLayout{vectorAlignment, vectorSize, 0};
    const uint64_t columns = use.shape[1];
    if (columns < 2 || columns > 4) {
        error = "Vulkan push constant matrix dimension must be between 2 and 4";
        return std::nullopt;
    }
    const uint32_t stride = (vectorSize + vectorAlignment - 1) / vectorAlignment * vectorAlignment;
    return PushConstantLayout{vectorAlignment, stride * static_cast<uint32_t>(columns), stride};
}

VkPrimitiveTopology topology(VernonPrimitiveTopology value) {
    if (value == VERNON_TOPOLOGY_LINE_LIST)
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    if (value == VERNON_TOPOLOGY_POINT_LIST)
        return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

VkFormat vertexFormat(uint32_t components) {
    if (components == 1)
        return VK_FORMAT_R32_SFLOAT;
    if (components == 2)
        return VK_FORMAT_R32G32_SFLOAT;
    if (components == 3)
        return VK_FORMAT_R32G32B32_SFLOAT;
    if (components == 4)
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    return VK_FORMAT_UNDEFINED;
}

template <typename Record> bool submit(VernonRuntimeContext *context, Record record, std::string &error) {
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!beginVulkanCommands(*context, command, error))
        return false;
    record(command);
    return submitVulkanCommands(*context, command, error);
}

} // namespace

bool planVulkanPushConstantRanges(const Variant &variant, uint32_t maximumSize, VulkanPushConstantRanges &ranges,
                                  std::string &error) {
    ranges = {};
    uint32_t nextOffset = 0;
    const std::array<std::pair<const char *, VulkanPushConstantStageRange *>, 2> stages{
        {{"vertex", &ranges.vertex}, {"fragment", &ranges.fragment}}};
    for (const auto &[stageName, range] : stages) {
        std::vector<const ParameterUse *> uses;
        auto collect = [&](const std::vector<Parameter> &parameters) {
            for (const Parameter &parameter : parameters)
                for (const ParameterUse &use : parameter.uses)
                    if (use.stage == stageName && use.interfaceKind == "uniform" && use.binding == UINT32_MAX)
                        uses.push_back(&use);
        };
        collect(variant.parameters);
        collect(variant.internalParameters);
        std::sort(uses.begin(), uses.end(),
                  [](const ParameterUse *left, const ParameterUse *right) { return left->index < right->index; });
        if (std::adjacent_find(uses.begin(), uses.end(), [](const ParameterUse *left, const ParameterUse *right) {
                return left->index == right->index;
            }) != uses.end()) {
            error = std::string("Vulkan ") + stageName + " push-constant reflection has duplicate indices";
            return false;
        }
        uint32_t localSize = 0;
        for (const ParameterUse *use : uses) {
            const std::optional<PushConstantLayout> layout = pushConstantLayout(*use, error);
            if (!layout) {
                error = std::string("Vulkan ") + stageName + " push constant: " + error;
                return false;
            }
            const uint32_t offset = (localSize + layout->alignment - 1) / layout->alignment * layout->alignment;
            if (offset > maximumSize || layout->storageSize > maximumSize - offset) {
                error = std::string("Vulkan ") + stageName + " push-constant block exceeds device limit of " +
                        std::to_string(maximumSize) + " bytes";
                return false;
            }
            localSize = offset + layout->storageSize;
        }
        range->offset = (nextOffset + 3) / 4 * 4;
        range->size = (localSize + 3) / 4 * 4;
        if (range->offset > maximumSize || range->size > maximumSize - range->offset) {
            error =
                "combined Vulkan push-constant blocks exceed device limit of " + std::to_string(maximumSize) + " bytes";
            return false;
        }
        nextOffset = range->offset + range->size;
    }
    return true;
}

VernonStatus encodeAndSubmitVulkanGraphics(const VulkanGraphicsState &state, const Variant &variant,
                                           const VernonPipelineInvocation &invocation,
                                           const PlannedGraphicsInvocation &plan, std::string &error) {
    VernonRuntimeContext *context = state.context;
    VulkanContextState &contextState = vulkanState(*context);
    VulkanDriver &driver = vulkanDriver();
    struct PushConstantUse {
        const Parameter *parameter;
        const ParameterUse *use;
        const VernonPipelineArgument *argument;
    };
    struct PushConstantBlock {
        VkShaderStageFlags stage;
        uint32_t offset;
        std::vector<uint8_t> data;
    };
    struct SampledResource {
        VernonDeviceTexture *texture{};
        VernonDeviceSampler *sampler{};
        bool implicitSampler{};
        VkShaderStageFlags stages{};
    };
    std::vector<VkVertexInputBindingDescription> bindingDescriptions;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceSize> vertexOffsets;
    std::vector<PushConstantUse> pushConstantUses;
    std::vector<PushConstantBlock> pushConstantBlocks;
    std::map<std::pair<uint32_t, uint32_t>, SampledResource> sampledResources;
    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];

    const std::array<uint64_t, 1> resolutionShape{2};
    const std::array<int64_t, 1> resolutionStrides{sizeof(float)};
    VernonPipelineArgument resolutionArgument{};
    resolutionArgument.kind = VERNON_PIPELINE_TENSOR;
    resolutionArgument.tensor.struct_size = sizeof(VernonTensorView);
    resolutionArgument.tensor.storage = VERNON_TENSOR_HOST;
    resolutionArgument.tensor.host_data = plan.resolution.data();
    resolutionArgument.tensor.dtype = VERNON_DATA_F32;
    resolutionArgument.tensor.access = VERNON_ACCESS_READ;
    resolutionArgument.tensor.rank = 1;
    resolutionArgument.tensor.shape = resolutionShape.data();
    resolutionArgument.tensor.byte_strides = resolutionStrides.data();
    resolutionArgument.tensor.byte_size = sizeof(plan.resolution);

    for (const auto &[binding, resource] : plan.sampledResources) {
        VkShaderStageFlags stages = 0;
        if (resource.stages & PLANNED_STAGE_VERTEX)
            stages |= VK_SHADER_STAGE_VERTEX_BIT;
        if (resource.stages & PLANNED_STAGE_FRAGMENT)
            stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
        sampledResources.emplace(binding,
                                 SampledResource{resource.texture, resource.sampler, resource.implicitSampler, stages});
    }
    for (const PlannedVertexInput &input : plan.vertexInputs) {
        const VernonTensorView &tensor = *input.tensor;
        const ParameterUse &use = *input.use;
        const uint32_t binding = static_cast<uint32_t>(bindingDescriptions.size());
        bindingDescriptions.push_back({binding, static_cast<uint32_t>(tensor.byte_strides[0]),
                                       input.instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
        vertexBuffers.push_back(vulkanBufferState(*tensor.buffer).buffer);
        vertexOffsets.push_back(tensor.byte_offset);
        if (input.components <= 4) {
            attributeDescriptions.push_back({use.location, binding, vertexFormat(input.components), 0});
        } else {
            const uint32_t columns = static_cast<uint32_t>(tensor.shape[1]);
            const uint32_t rows = static_cast<uint32_t>(tensor.shape[2]);
            for (uint32_t column = 0; column < columns; ++column)
                attributeDescriptions.push_back({use.location + column, binding, vertexFormat(rows),
                                                 static_cast<uint32_t>(column * tensor.byte_strides[1])});
        }
    }
    for (const Parameter &parameter : variant.parameters) {
        const VernonPipelineArgument &argument = *plan.arguments.at(parameter.slot);
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "vertex" && use.stage != "fragment")
                continue;
            if (use.interfaceKind == "uniform") {
                if (use.binding != UINT32_MAX)
                    return fail(error, "Vulkan descriptor uniforms are not implemented",
                                VERNON_STATUS_UNSUPPORTED_TARGET);
                pushConstantUses.push_back({&parameter, &use, &argument});
            }
        }
    }
    for (const Parameter &parameter : variant.internalParameters)
        for (const ParameterUse &use : parameter.uses)
            if ((use.stage == "vertex" || use.stage == "fragment") && parameter.systemValue == "resolution") {
                if (use.interfaceKind != "uniform" || use.binding != UINT32_MAX)
                    return fail(error, "Vulkan resolution system value must be a push constant");
                pushConstantUses.push_back({&parameter, &use, &resolutionArgument});
            }

    VulkanPushConstantRanges plannedPushConstantRanges;
    if (!planVulkanPushConstantRanges(variant, contextState.maxPushConstantsSize, plannedPushConstantRanges, error))
        return VERNON_STATUS_UNSUPPORTED_TARGET;

    for (const auto &[stageName, stageFlag] : std::array<std::pair<const char *, VkShaderStageFlags>, 2>{
             {{"vertex", VK_SHADER_STAGE_VERTEX_BIT}, {"fragment", VK_SHADER_STAGE_FRAGMENT_BIT}}}) {
        std::vector<PushConstantUse> stageUses;
        for (const PushConstantUse &value : pushConstantUses)
            if (value.use->stage == stageName)
                stageUses.push_back(value);
        if (stageUses.empty())
            continue;
        std::sort(stageUses.begin(), stageUses.end(), [](const PushConstantUse &left, const PushConstantUse &right) {
            return left.use->index < right.use->index;
        });
        if (std::adjacent_find(stageUses.begin(), stageUses.end(),
                               [](const PushConstantUse &left, const PushConstantUse &right) {
                                   return left.use->index == right.use->index;
                               }) != stageUses.end())
            return fail(error, std::string("Vulkan ") + stageName + " push-constant reflection has duplicate indices");
        const VulkanPushConstantStageRange &stageRange = stageFlag == VK_SHADER_STAGE_VERTEX_BIT
                                                             ? plannedPushConstantRanges.vertex
                                                             : plannedPushConstantRanges.fragment;
        PushConstantBlock block{stageFlag, stageRange.offset, {}};
        uint32_t size = 0;
        for (const PushConstantUse &value : stageUses) {
            const Parameter &parameter = *value.parameter;
            const ParameterUse &use = *value.use;
            const VernonPipelineArgument &argument = *value.argument;
            if (parameter.dtype != use.dtype || parameter.shape != use.shape)
                return fail(error, std::string("Vulkan ") + stageName +
                                       " push-constant reflection is inconsistent for '" + parameter.name + "'");
            std::string layoutError;
            const std::optional<PushConstantLayout> layout = pushConstantLayout(use, layoutError);
            if (!layout)
                return fail(error, std::string("Vulkan ") + stageName + " push-constant '" + parameter.name +
                                       "': " + layoutError);
            const std::optional<VernonDataType> dtype = pipelineDataType(use.dtype);
            if (!dtype || argument.kind != VERNON_PIPELINE_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
                argument.tensor.dtype != *dtype || argument.tensor.rank != use.shape.size())
                return fail(error, std::string("Vulkan ") + stageName +
                                       " push-constant value has the wrong type or size "
                                       "for '" +
                                       parameter.name + "'");
            for (size_t dimension = 0; dimension < use.shape.size(); ++dimension)
                if (argument.tensor.shape[dimension] != use.shape[dimension])
                    return fail(error, std::string("Vulkan ") + stageName +
                                           " push-constant value has the wrong shape for '" + parameter.name + "'");
            const uint32_t offset = (size + layout->alignment - 1) / layout->alignment * layout->alignment;
            size = offset + layout->storageSize;
            block.data.resize(size);
            const uint8_t *source = hostTensorData(argument.tensor);
            const size_t elementSize = dataTypeSize(argument.tensor.dtype);
            if (use.shape.empty()) {
                std::memcpy(block.data.data() + offset, source, elementSize);
            } else if (use.shape.size() == 1) {
                for (uint32_t row = 0; row < use.shape[0]; ++row)
                    std::memcpy(block.data.data() + offset + row * elementSize,
                                source + row * argument.tensor.byte_strides[0], elementSize);
            } else {
                const uint32_t rows = static_cast<uint32_t>(use.shape[0]);
                const uint32_t columns = static_cast<uint32_t>(use.shape[1]);
                for (uint32_t column = 0; column < columns; ++column)
                    for (uint32_t row = 0; row < rows; ++row)
                        std::memcpy(block.data.data() + offset + column * layout->matrixStride + row * elementSize,
                                    source + row * argument.tensor.byte_strides[0] +
                                        column * argument.tensor.byte_strides[1],
                                    elementSize);
            }
        }
        block.data.resize(stageRange.size);
        pushConstantBlocks.push_back(std::move(block));
    }

    for (const auto &[binding, resource] : sampledResources)
        if ((!resource.sampler || !vulkanSamplerState(*resource.sampler).sampler) && !resource.implicitSampler)
            return fail(error, "Vulkan sampler is not available");
    for (const VernonColorAttachment *attachment : plan.attachments)
        if (!vulkanTextureState(*attachment->texture).colorAttachment)
            return fail(error, "Vulkan render target is invalid");
    for (const auto &[binding, resource] : sampledResources)
        if (std::any_of(plan.attachments.begin(), plan.attachments.end(), [&](const VernonColorAttachment *attachment) {
                return attachment->texture == resource.texture;
            }))
            return fail(error, "a Vulkan texture cannot be sampled and rendered to in the "
                               "same invocation");

    std::vector<VkAttachmentDescription> attachmentDescriptions;
    std::vector<VkAttachmentReference> attachmentReferences(
        plan.maximumAttachmentLocation + 1, {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    std::vector<VkImageView> imageViews;
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        VkAttachmentDescription description{};
        description.format = vulkanTextureState(*plan.attachments[index]->texture).format;
        description.samples = VK_SAMPLE_COUNT_1_BIT;
        description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        description.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        description.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachmentDescriptions.push_back(description);
        attachmentReferences[plan.attachments[index]->location] = {static_cast<uint32_t>(index),
                                                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        imageViews.push_back(vulkanTextureState(*plan.attachments[index]->texture).view);
    }

    if (!state.cache)
        return fail(error, "Vulkan graphics cache is unavailable", VERNON_STATUS_INTERNAL_ERROR);
    VulkanGraphicsCache &cache = *state.cache;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkDescriptorSet> descriptorSets;
    auto cleanup = [&] {
        if (framebuffer)
            driver.destroyFramebuffer(contextState.device, framebuffer, nullptr);
        if (descriptorPool)
            driver.destroyDescriptorPool(contextState.device, descriptorPool, nullptr);
        if (renderPass)
            driver.destroyRenderPass(contextState.device, renderPass, nullptr);
    };
    if (std::any_of(sampledResources.begin(), sampledResources.end(),
                    [](const auto &entry) { return entry.second.implicitSampler && !entry.second.sampler; }) &&
        !contextState.defaultImplicitSampler)
        return fail(error, "Vulkan default sampler is unavailable", VERNON_STATUS_INTERNAL_ERROR);

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(attachmentReferences.size());
    subpass.pColorAttachments = attachmentReferences.data();
    VkRenderPassCreateInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescriptions.size());
    renderPassInfo.pAttachments = attachmentDescriptions.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    if (!vkCheck(driver.createRenderPass(contextState.device, &renderPassInfo, nullptr, &renderPass),
                 "vkCreateRenderPass", error))
        return VERNON_STATUS_INTERNAL_ERROR;

    VulkanPipelineLayoutKey pipelineLayoutKey;
    if (!sampledResources.empty()) {
        const uint32_t maximumSet = sampledResources.rbegin()->first.first;
        descriptorSetLayouts.resize(maximumSet + 1, VK_NULL_HANDLE);
        pipelineLayoutKey.descriptorSets.resize(maximumSet + 1);
        for (uint32_t set = 0; set <= maximumSet; ++set) {
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            for (const auto &[key, resource] : sampledResources)
                if (key.first == set) {
                    bindings.push_back(
                        {key.second, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, resource.stages, nullptr});
                    pipelineLayoutKey.descriptorSets[set].bindings.push_back(
                        {key.second, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, resource.stages});
                }
            const VulkanDescriptorSetLayoutKey &key = pipelineLayoutKey.descriptorSets[set];
            const auto cached = cache.descriptorSetLayouts.find(key);
            if (cached != cache.descriptorSetLayouts.end()) {
                descriptorSetLayouts[set] = cached->second;
            } else {
                VkDescriptorSetLayout layout = VK_NULL_HANDLE;
                VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                info.bindingCount = static_cast<uint32_t>(bindings.size());
                info.pBindings = bindings.data();
                if (!vkCheck(driver.createDescriptorSetLayout(contextState.device, &info, nullptr, &layout),
                             "vkCreateDescriptorSetLayout", error)) {
                    cleanup();
                    return VERNON_STATUS_INTERNAL_ERROR;
                }
                cache.descriptorSetLayouts.emplace(key, layout);
                ++cache.descriptorSetLayoutCreations;
                descriptorSetLayouts[set] = layout;
            }
        }
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      static_cast<uint32_t>(sampledResources.size())};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = static_cast<uint32_t>(descriptorSetLayouts.size());
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        if (!vkCheck(driver.createDescriptorPool(contextState.device, &poolInfo, nullptr, &descriptorPool),
                     "vkCreateDescriptorPool", error)) {
            cleanup();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        descriptorSets.resize(descriptorSetLayouts.size());
        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = descriptorPool;
        allocateInfo.descriptorSetCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        allocateInfo.pSetLayouts = descriptorSetLayouts.data();
        if (!vkCheck(driver.allocateDescriptorSets(contextState.device, &allocateInfo, descriptorSets.data()),
                     "vkAllocateDescriptorSets", error)) {
            cleanup();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        std::vector<VkDescriptorImageInfo> imageInfos;
        std::vector<VkWriteDescriptorSet> writes;
        imageInfos.reserve(sampledResources.size());
        writes.reserve(sampledResources.size());
        for (const auto &[key, resource] : sampledResources) {
            imageInfos.push_back(
                {resource.sampler ? vulkanSamplerState(*resource.sampler).sampler : contextState.defaultImplicitSampler,
                 vulkanTextureState(*resource.texture).view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = descriptorSets[key.first];
            write.dstBinding = key.second;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfos.back();
            writes.push_back(write);
        }
        driver.updateDescriptorSets(contextState.device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                                    nullptr);
    }

    std::vector<VkPushConstantRange> pushConstantRanges;
    for (const PushConstantBlock &block : pushConstantBlocks) {
        pushConstantRanges.push_back({block.stage, block.offset, static_cast<uint32_t>(block.data.size())});
        pipelineLayoutKey.pushConstantRanges.push_back(
            {block.stage, block.offset, static_cast<uint32_t>(block.data.size())});
    }
    const auto cachedPipelineLayout = cache.pipelineLayouts.find(pipelineLayoutKey);
    if (cachedPipelineLayout != cache.pipelineLayouts.end()) {
        pipelineLayout = cachedPipelineLayout->second;
    } else {
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        layoutInfo.pSetLayouts = descriptorSetLayouts.data();
        layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
        layoutInfo.pPushConstantRanges = pushConstantRanges.data();
        if (!vkCheck(driver.createPipelineLayout(contextState.device, &layoutInfo, nullptr, &pipelineLayout),
                     "vkCreatePipelineLayout", error)) {
            cleanup();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        cache.pipelineLayouts.emplace(pipelineLayoutKey, pipelineLayout);
        ++cache.pipelineLayoutCreations;
    }

    const VkPipelineShaderStageCreateInfo shaderStages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, state.vertex,
         state.vertexEntry->c_str(), nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, state.fragment,
         state.fragmentEntry->c_str(), nullptr}};
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInput.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInput.pVertexAttributeDescriptions = attributeDescriptions.data();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = topology(invocation.topology);
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkPipelineColorBlendAttachmentState> blendStates(attachmentDescriptions.size());
    for (auto &blend : blendStates)
        blend.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = static_cast<uint32_t>(blendStates.size());
    blend.pAttachments = blendStates.data();
    const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    VulkanGraphicsPipelineKey graphicsPipelineKey;
    graphicsPipelineKey.renderPass.colorFormatsByLocation.resize(attachmentReferences.size(), VK_FORMAT_UNDEFINED);
    for (const VernonColorAttachment *attachment : plan.attachments)
        graphicsPipelineKey.renderPass.colorFormatsByLocation[attachment->location] =
            vulkanTextureState(*attachment->texture).format;
    graphicsPipelineKey.topology = inputAssembly.topology;
    for (const VkVertexInputBindingDescription &binding : bindingDescriptions)
        graphicsPipelineKey.vertexBindings.push_back(
            {binding.binding, binding.stride, static_cast<uint32_t>(binding.inputRate)});
    for (const VkVertexInputAttributeDescription &attribute : attributeDescriptions)
        graphicsPipelineKey.vertexAttributes.push_back(
            {attribute.location, attribute.binding, static_cast<uint32_t>(attribute.format), attribute.offset});
    graphicsPipelineKey.layout = pipelineLayoutKey;
    const auto cachedGraphicsPipeline = cache.graphicsPipelines.find(graphicsPipelineKey);
    if (cachedGraphicsPipeline != cache.graphicsPipelines.end()) {
        graphicsPipeline = cachedGraphicsPipeline->second;
    } else {
        if (!vkCheck(driver.createGraphicsPipelines(contextState.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                                    &graphicsPipeline),
                     "vkCreateGraphicsPipelines", error)) {
            cleanup();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        cache.graphicsPipelines.emplace(std::move(graphicsPipelineKey), graphicsPipeline);
        ++cache.graphicsPipelineCreations;
    }

    VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(imageViews.size());
    framebufferInfo.pAttachments = imageViews.data();
    framebufferInfo.width = plan.attachmentWidth;
    framebufferInfo.height = plan.attachmentHeight;
    framebufferInfo.layers = 1;
    if (!vkCheck(driver.createFramebuffer(contextState.device, &framebufferInfo, nullptr, &framebuffer),
                 "vkCreateFramebuffer", error)) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }

    const bool submitted = submit(
        context,
        [&](VkCommandBuffer command) {
            if (state.barrier) {
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
                driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            }
            for (const VernonColorAttachment *attachment : plan.attachments)
                if (vulkanTextureState(*attachment->texture).layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                    transitionVulkanImageLayout(command, *attachment->texture,
                                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            for (const auto &[binding, resource] : sampledResources)
                if (vulkanTextureState(*resource.texture).layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    transitionVulkanImageLayout(command, *resource.texture, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            std::vector<VkClearValue> clearValues(plan.attachments.size());
            VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            begin.renderPass = renderPass;
            begin.framebuffer = framebuffer;
            begin.renderArea.extent = {plan.attachmentWidth, plan.attachmentHeight};
            begin.clearValueCount = static_cast<uint32_t>(clearValues.size());
            begin.pClearValues = clearValues.data();
            driver.cmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
            driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
            if (!descriptorSets.empty())
                driver.cmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                                             static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0,
                                             nullptr);
            for (const PushConstantBlock &block : pushConstantBlocks)
                driver.cmdPushConstants(command, pipelineLayout, block.stage, block.offset,
                                        static_cast<uint32_t>(block.data.size()), block.data.data());
            const bool hasScissor = invocation.scissor[2] && invocation.scissor[3];
            const float viewportY = static_cast<float>(hasViewport ? invocation.viewport[1] : 0);
            const float viewportHeight =
                static_cast<float>(hasViewport ? invocation.viewport[3] : plan.attachmentHeight);
            // Vulkan's framebuffer Y axis is opposite OpenGL's clip-space
            // convention. A negative viewport height keeps the common pipeline
            // coordinates and host readback orientation backend-independent.
            VkViewport viewport{static_cast<float>(hasViewport ? invocation.viewport[0] : 0),
                                viewportY + viewportHeight,
                                static_cast<float>(hasViewport ? invocation.viewport[2] : plan.attachmentWidth),
                                -viewportHeight,
                                0.0f,
                                1.0f};
            VkRect2D scissor{{static_cast<int32_t>(hasScissor ? invocation.scissor[0] : 0),
                              static_cast<int32_t>(hasScissor ? invocation.scissor[1] : 0)},
                             {hasScissor ? invocation.scissor[2] : plan.attachmentWidth,
                              hasScissor ? invocation.scissor[3] : plan.attachmentHeight}};
            driver.cmdSetViewport(command, 0, 1, &viewport);
            driver.cmdSetScissor(command, 0, 1, &scissor);
            if (!vertexBuffers.empty())
                driver.cmdBindVertexBuffers(command, 0, static_cast<uint32_t>(vertexBuffers.size()),
                                            vertexBuffers.data(), vertexOffsets.data());
            if (plan.indexBinding) {
                driver.cmdBindIndexBuffer(command, vulkanBufferState(*plan.indexBinding->buffer).buffer,
                                          plan.indexBinding->offset, VK_INDEX_TYPE_UINT32);
                driver.cmdDrawIndexed(command, plan.indexBinding->index_count, plan.instanceCount, 0, 0, 0);
            } else {
                driver.cmdDraw(command, plan.vertexCount, plan.instanceCount, 0, 0);
            }
            driver.cmdEndRenderPass(command);
        },
        error);
    cleanup();
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
}

void destroyVulkanGraphicsCache(VernonRuntimeContext *context, VulkanGraphicsCache &cache) {
    if (!context || !context->backendState)
        return;
    VulkanContextState &state = vulkanState(*context);
    if (!state.device)
        return;
    VulkanDriver &driver = vulkanDriver();
    for (const auto &[key, pipeline] : cache.graphicsPipelines)
        driver.destroyPipeline(state.device, pipeline, nullptr);
    for (const auto &[key, layout] : cache.pipelineLayouts)
        driver.destroyPipelineLayout(state.device, layout, nullptr);
    for (const auto &[key, layout] : cache.descriptorSetLayouts)
        driver.destroyDescriptorSetLayout(state.device, layout, nullptr);
    cache = {};
}

} // namespace vernon::runtime

#endif
