#include "adapter_common.h"

#if defined(VERNON_HAS_VULKAN_RHI)

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

struct PreparedShader {
    rhi::vulkan::DeviceState *device{};
    uint32_t stage{};
    VkShaderModule module{};
    std::string entry;
    std::vector<uint32_t> words;
};
struct PreparedLayout {
    struct Entry {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        uint32_t inlineOffset{UINT32_MAX};
    };
    rhi::vulkan::DeviceState *device{};
    std::vector<Entry> entries;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    VkDescriptorSetLayout descriptorSetLayout{};
    uint32_t pushConstantSize{};
};
struct PreparedPipeline {
    rhi::vulkan::DeviceState *device{};
    PreparedLayout *layout{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline pipeline{};
    VkRenderPass renderPass{};
    std::vector<VkShaderModule> specializedShaderModules;
    bool graphics{};
};
struct PreparedBindingSet {
    struct Slot {
        PreparedLayout::Entry entry{};
        VernonRuntimeProviderBindingValue value{};
        std::vector<uint8_t> inlineStorage;
        rhi::vulkan::Buffer inlineBuffer;
    };
    rhi::vulkan::DeviceState *device{};
    PreparedLayout *layout{};
    VkDescriptorSet descriptorSet{};
    std::vector<Slot> slots;
    std::mutex mutex;

    ~PreparedBindingSet() {
        if (!device)
            return;
        if (descriptorSet) {
            std::string ignored;
            (void)device->freeDescriptorSet(descriptorSet, ignored);
        }
        for (auto &slot : slots)
            if (slot.inlineBuffer.buffer)
                device->destroyBuffer(slot.inlineBuffer);
    }
};

VkShaderStageFlags stages(uint32_t value) {
    VkShaderStageFlags result = 0;
    if (value & VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
        result |= VK_SHADER_STAGE_COMPUTE_BIT;
    if (value & VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
        result |= VK_SHADER_STAGE_VERTEX_BIT;
    if (value & VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
        result |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return result;
}

VkDescriptorType descriptorType(VernonRuntimeProviderBindingKind kind) {
    switch (kind) {
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
        // Compute inline values use an internally owned storage buffer;
        // graphics inline values take the push-constant path instead.
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    case VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER:
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

VkFormat vertexFormat(uint32_t dtype, uint32_t components) {
    static constexpr VkFormat formats[][4] = {
        {VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT},
        {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT},
        {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
        {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
        {VK_FORMAT_R64_SFLOAT, VK_FORMAT_R64G64_SFLOAT, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED},
    };
    const int row = dtype == VERNON_RUNTIME_PROVIDER_I32   ? 0
                    : dtype == VERNON_RUNTIME_PROVIDER_U32 ? 1
                    : dtype == VERNON_RUNTIME_PROVIDER_F16 ? 2
                    : dtype == VERNON_RUNTIME_PROVIDER_F32 ? 3
                    : dtype == VERNON_RUNTIME_PROVIDER_F64 ? 4
                                                           : -1;
    return row < 0 || components == 0 || components > 4 ? VK_FORMAT_UNDEFINED : formats[row][components - 1];
}

bool relocatePushConstantOffsets(std::vector<uint32_t> &words, uint32_t delta) {
    if (delta == 0)
        return true;
    constexpr uint16_t kOpTypePointer = 32;
    constexpr uint16_t kOpVariable = 59;
    constexpr uint16_t kOpMemberDecorate = 72;
    constexpr uint32_t kPushConstantStorageClass = 9;
    constexpr uint32_t kOffsetDecoration = 35;
    uint32_t pushConstantPointerType = 0;
    uint32_t pushConstantStruct = 0;
    for (size_t index = 5; index < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[index] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[index]);
        if (wordCount == 0 || index + wordCount > words.size())
            return false;
        if (opcode == kOpVariable && wordCount >= 4 && words[index + 3] == kPushConstantStorageClass)
            pushConstantPointerType = words[index + 1];
        index += wordCount;
    }
    if (pushConstantPointerType == 0)
        return true;
    for (size_t index = 5; index < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[index] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[index]);
        if (opcode == kOpTypePointer && wordCount >= 4 && words[index + 1] == pushConstantPointerType &&
            words[index + 2] == kPushConstantStorageClass) {
            pushConstantStruct = words[index + 3];
            break;
        }
        index += wordCount;
    }
    if (pushConstantStruct == 0)
        return false;
    for (size_t index = 5; index < words.size();) {
        const uint16_t wordCount = static_cast<uint16_t>(words[index] >> 16);
        const uint16_t opcode = static_cast<uint16_t>(words[index]);
        if (opcode == kOpMemberDecorate && wordCount >= 5 && words[index + 1] == pushConstantStruct &&
            words[index + 3] == kOffsetDecoration)
            words[index + 4] += delta;
        index += wordCount;
    }
    return true;
}

VernonStatus failure(VernonRuntimeRhiAdapter &adapter, VkResult result, const char *operation) {
    return result == VK_SUCCESS
               ? VERNON_STATUS_OK
               : fail(adapter, std::string(operation) + " failed with VkResult " + std::to_string(result),
                      VERNON_STATUS_INTERNAL_ERROR);
}

uint32_t getCapabilities(void *) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return {0x56554c4b414eu, reinterpret_cast<uintptr_t>(adapter.vulkanDevice->physicalDevice),
            adapter.vulkanDevice->apiVersion};
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !descriptor->data ||
        descriptor->size < sizeof(uint32_t) || descriptor->size % sizeof(uint32_t) != 0 || !descriptor->entry.data ||
        descriptor->entry.size == 0 || descriptor->format.size != 5 ||
        std::memcmp(descriptor->format.data, "spirv", 5) != 0)
        return fail(adapter, "Vulkan adapter received an invalid SPIR-V shader");
    auto shader = std::unique_ptr<PreparedShader>(new (std::nothrow) PreparedShader());
    if (!shader)
        return fail(adapter, "Vulkan shader preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    shader->device = adapter.vulkanDevice;
    shader->stage = descriptor->stage;
    shader->entry.assign(descriptor->entry.data, descriptor->entry.size);
    shader->words.assign(static_cast<const uint32_t *>(descriptor->data),
                         static_cast<const uint32_t *>(descriptor->data) + descriptor->size / sizeof(uint32_t));
    const VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, descriptor->size,
                                              static_cast<const uint32_t *>(descriptor->data)};
    const VkResult result =
        rhi::vulkan::driver().createShaderModule(shader->device->device, &createInfo, nullptr, &shader->module);
    if (result != VK_SUCCESS)
        return failure(adapter, result, "vkCreateShaderModule");
    *output = toHandle(shader.release());
    adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return fail(adapter, "Vulkan adapter received an invalid pipeline layout");
    auto layout = std::unique_ptr<PreparedLayout>(new (std::nothrow) PreparedLayout());
    if (!layout)
        return fail(adapter, "Vulkan layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    layout->device = adapter.vulkanDevice;
    std::vector<VkDescriptorSetLayoutBinding> nativeBindings;
    try {
        layout->entries.reserve(descriptor->binding_count);
        for (size_t index = 0; index < descriptor->binding_count; ++index) {
            const auto &source = descriptor->bindings[index];
            const bool supported = source.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ||
                                   source.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
            if (!supported || source.array_count != 1)
                return fail(adapter, "Vulkan layout contains an unsupported binding", VERNON_STATUS_UNSUPPORTED_TARGET);
            PreparedLayout::Entry entry;
            entry.layout = source;
            if (source.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
                if (source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
                    nativeBindings.push_back(
                        {source.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
                else {
                    entry.inlineOffset = (layout->pushConstantSize + 3u) & ~3u;
                    layout->pushConstantSize = entry.inlineOffset + source.element_size;
                }
            } else if (source.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                const auto existing =
                    std::find_if(nativeBindings.begin(), nativeBindings.end(),
                                 [&](const auto &binding) { return binding.binding == source.binding; });
                if (existing == nativeBindings.end())
                    nativeBindings.push_back(
                        {source.binding, descriptorType(source.kind), 1, stages(source.stage_mask), nullptr});
                else if ((existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
                          source.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) ||
                         (existing->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER &&
                          source.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)) {
                    existing->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    existing->stageFlags |= stages(source.stage_mask);
                } else
                    return fail(adapter, "Vulkan layout contains duplicate descriptor bindings");
            }
            layout->entries.push_back(entry);
        }
        for (size_t index = 0; index < descriptor->vertex_attribute_count; ++index) {
            const auto &attribute = descriptor->vertex_attributes[index];
            const VkFormat format = vertexFormat(attribute.dtype, attribute.component_count);
            const bool bindingExists =
                std::any_of(layout->entries.begin(), layout->entries.end(), [&](const PreparedLayout::Entry &entry) {
                    return entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                           entry.layout.binding == attribute.binding;
                });
            VkFormatProperties properties{};
            if (format != VK_FORMAT_UNDEFINED)
                rhi::vulkan::driver().getPhysicalDeviceFormatProperties(layout->device->physicalDevice, format,
                                                                        &properties);
            if (!bindingExists || format == VK_FORMAT_UNDEFINED ||
                !(properties.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ||
                attribute.location >= layout->device->maxVertexInputAttributes)
                return fail(adapter, "Vulkan vertex attribute exceeds device location or format capabilities");
            layout->vertexAttributes.push_back(attribute);
        }
    } catch (const std::bad_alloc &) {
        return fail(adapter, "Vulkan layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
    if (layout->pushConstantSize > layout->device->maxPushConstantsSize)
        return fail(adapter, "Vulkan push constants exceed the device limit", VERNON_STATUS_UNSUPPORTED_TARGET);
    const VkDescriptorSetLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
                                                     static_cast<uint32_t>(nativeBindings.size()),
                                                     nativeBindings.data()};
    const VkResult result = rhi::vulkan::driver().createDescriptorSetLayout(layout->device->device, &createInfo,
                                                                            nullptr, &layout->descriptorSetLayout);
    if (result != VK_SUCCESS)
        return failure(adapter, result, "vkCreateDescriptorSetLayout");
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus createPipelineLayout(VernonRuntimeRhiAdapter &adapter, PreparedPipeline &pipeline) {
    VkPushConstantRange range{VK_SHADER_STAGE_ALL, 0, pipeline.layout->pushConstantSize};
    const VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                nullptr,
                                                0,
                                                1,
                                                &pipeline.layout->descriptorSetLayout,
                                                pipeline.layout->pushConstantSize ? 1u : 0u,
                                                pipeline.layout->pushConstantSize ? &range : nullptr};
    return failure(adapter,
                   rhi::vulkan::driver().createPipelineLayout(pipeline.device->device, &createInfo, nullptr,
                                                              &pipeline.pipelineLayout),
                   "vkCreatePipelineLayout");
}

VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout || !descriptor->shaders ||
        descriptor->shader_count == 0)
        return fail(adapter, "Vulkan adapter received an invalid pipeline");
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return fail(adapter, "Vulkan pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    pipeline->device = adapter.vulkanDevice;
    pipeline->layout = layout;
    pipeline->graphics = descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    VernonStatus status = createPipelineLayout(adapter, *pipeline);
    if (status != VERNON_STATUS_OK)
        return status;
    if (pipeline->graphics && descriptor->color_format_count == 0) {
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    }
    if (!pipeline->graphics) {
        auto *shader = descriptor->shader_count == 1 ? fromHandle<PreparedShader>(descriptor->shaders[0]) : nullptr;
        if (!shader || shader->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
            return fail(adapter, "Vulkan compute pipeline requires one compute shader");
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    nullptr,
                                                    0,
                                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                                    shader->module,
                                                    shader->entry.c_str(),
                                                    nullptr};
        const VkComputePipelineCreateInfo createInfo{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage, pipeline->pipelineLayout, {}, 0};
        status = failure(adapter,
                         rhi::vulkan::driver().createComputePipelines(pipeline->device->device, {}, 1, &createInfo,
                                                                      nullptr, &pipeline->pipeline),
                         "vkCreateComputePipelines");
    } else {
        PreparedShader *vertex = nullptr;
        PreparedShader *fragment = nullptr;
        for (size_t index = 0; index < descriptor->shader_count; ++index) {
            auto *shader = fromHandle<PreparedShader>(descriptor->shaders[index]);
            if (shader && shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
                vertex = shader;
            else if (shader && shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
                fragment = shader;
        }
        if (!vertex || !fragment)
            return fail(adapter, "Vulkan graphics provider requires vertex and fragment shaders");
        const auto stageOffset = [&](uint32_t stage) {
            uint32_t offset = UINT32_MAX;
            for (const auto &entry : layout->entries)
                if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE && entry.layout.stage_mask == stage &&
                    entry.inlineOffset != UINT32_MAX)
                    offset = std::min(offset, entry.inlineOffset);
            return offset == UINT32_MAX ? 0u : offset;
        };
        const auto specializeShader = [&](PreparedShader &shader, uint32_t offset, VkShaderModule &module) {
            if (offset == 0) {
                module = shader.module;
                return VERNON_STATUS_OK;
            }
            std::vector<uint32_t> words = shader.words;
            if (!relocatePushConstantOffsets(words, offset))
                return fail(adapter, "Vulkan provider could not relocate push-constant offsets");
            const VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                                      words.size() * sizeof(uint32_t), words.data()};
            const VkResult result =
                rhi::vulkan::driver().createShaderModule(pipeline->device->device, &moduleInfo, nullptr, &module);
            if (result != VK_SUCCESS)
                return failure(adapter, result, "vkCreateShaderModule for relocated push constants");
            pipeline->specializedShaderModules.push_back(module);
            return VERNON_STATUS_OK;
        };
        VkShaderModule vertexModule{};
        VkShaderModule fragmentModule{};
        status = specializeShader(*vertex, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_VERTEX), vertexModule);
        if (status == VERNON_STATUS_OK)
            status = specializeShader(*fragment, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT), fragmentModule);
        if (status != VERNON_STATUS_OK)
            return status;
        const VkPipelineShaderStageCreateInfo shaderStages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertexModule,
             vertex->entry.c_str(), nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
             fragmentModule, fragment->entry.c_str(), nullptr}};
        std::vector<VkVertexInputBindingDescription> vertexBindings;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
        for (const auto &entry : layout->entries)
            if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                vertexBindings.push_back(
                    {entry.layout.binding, 0,
                     entry.layout.divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
            }
        for (const VernonRuntimeProviderVertexAttribute &attribute : layout->vertexAttributes)
            vertexAttributes.push_back({attribute.location, attribute.binding,
                                        vertexFormat(attribute.dtype, attribute.component_count),
                                        attribute.relative_offset});
        for (auto &binding : vertexBindings) {
            if (binding.binding >= descriptor->vertex_stride_count)
                return fail(adapter, "Vulkan graphics pipeline is missing a vertex stride");
            binding.stride = descriptor->vertex_strides[binding.binding];
        }
        const VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(vertexBindings.size()),
            vertexBindings.data(),
            static_cast<uint32_t>(vertexAttributes.size()),
            vertexAttributes.data()};
        const VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
            descriptor->topology == 1   ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
            : descriptor->topology == 2 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                                        : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_FALSE};
        const VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        const VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                            nullptr,
                                                            0,
                                                            VK_FALSE,
                                                            VK_FALSE,
                                                            VK_POLYGON_MODE_FILL,
                                                            VK_CULL_MODE_NONE,
                                                            VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                            VK_FALSE,
                                                            0,
                                                            0,
                                                            0,
                                                            1};
        const VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<VkSampleCountFlagBits>(std::max(1u, descriptor->sample_count)),
            VK_FALSE,
            0,
            nullptr,
            VK_FALSE,
            VK_FALSE};
        const bool hasDepth = descriptor->depth_stencil_format != 0;
        const VkPipelineDepthStencilStateCreateInfo depthStencil{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            nullptr,
            0,
            hasDepth ? VK_TRUE : VK_FALSE,
            hasDepth ? VK_TRUE : VK_FALSE,
            VK_COMPARE_OP_LESS,
            VK_FALSE,
            VK_FALSE,
            {},
            {},
            0,
            1};
        std::array<VkPipelineColorBlendAttachmentState, 8> blendAttachments{};
        for (size_t index = 0; index < descriptor->color_format_count; ++index)
            blendAttachments[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        const VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                        nullptr,
                                                        0,
                                                        VK_FALSE,
                                                        VK_LOGIC_OP_COPY,
                                                        static_cast<uint32_t>(descriptor->color_format_count),
                                                        blendAttachments.data(),
                                                        {}};
        const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        const VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0,
                                                       2, dynamicStates};
        const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      static_cast<uint32_t>(descriptor->color_format_count),
                                                      reinterpret_cast<const VkFormat *>(descriptor->color_formats),
                                                      static_cast<VkFormat>(descriptor->depth_stencil_format),
                                                      VK_FORMAT_UNDEFINED};
        std::array<VkAttachmentDescription, 9> renderPassAttachments{};
        std::array<VkAttachmentReference, 8> renderPassReferences{};
        VkAttachmentReference depthReference{};
        if (!pipeline->device->dynamicRendering) {
            for (size_t index = 0; index < descriptor->color_format_count; ++index) {
                auto &attachment = renderPassAttachments[index];
                attachment.format = static_cast<VkFormat>(descriptor->color_formats[index]);
                attachment.samples = static_cast<VkSampleCountFlagBits>(std::max(1u, descriptor->sample_count));
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                renderPassReferences[index] = {static_cast<uint32_t>(index), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            }
            if (hasDepth) {
                auto &attachment = renderPassAttachments[descriptor->color_format_count];
                attachment.format = static_cast<VkFormat>(descriptor->depth_stencil_format);
                attachment.samples = static_cast<VkSampleCountFlagBits>(std::max(1u, descriptor->sample_count));
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depthReference = {static_cast<uint32_t>(descriptor->color_format_count),
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            }
            const VkSubpassDescription subpass{0,
                                               VK_PIPELINE_BIND_POINT_GRAPHICS,
                                               0,
                                               nullptr,
                                               static_cast<uint32_t>(descriptor->color_format_count),
                                               renderPassReferences.data(),
                                               nullptr,
                                               hasDepth ? &depthReference : nullptr,
                                               0,
                                               nullptr};
            const VkRenderPassCreateInfo renderPassInfo{
                VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                nullptr,
                0,
                static_cast<uint32_t>(descriptor->color_format_count + (hasDepth ? 1 : 0)),
                renderPassAttachments.data(),
                1,
                &subpass,
                0,
                nullptr};
            status = failure(adapter,
                             rhi::vulkan::driver().createRenderPass(pipeline->device->device, &renderPassInfo, nullptr,
                                                                    &pipeline->renderPass),
                             "vkCreateRenderPass");
            if (status != VERNON_STATUS_OK)
                return status;
        }
        const VkGraphicsPipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                                      pipeline->device->dynamicRendering ? &rendering : nullptr,
                                                      0,
                                                      2,
                                                      shaderStages,
                                                      &vertexInput,
                                                      &assembly,
                                                      nullptr,
                                                      &viewport,
                                                      &raster,
                                                      &multisample,
                                                      hasDepth ? &depthStencil : nullptr,
                                                      &blend,
                                                      &dynamic,
                                                      pipeline->pipelineLayout,
                                                      pipeline->renderPass,
                                                      0,
                                                      {},
                                                      0};
        status = failure(adapter,
                         rhi::vulkan::driver().createGraphicsPipelines(pipeline->device->device, {}, 1, &createInfo,
                                                                       nullptr, &pipeline->pipeline),
                         "vkCreateGraphicsPipelines");
    }
    if (status != VERNON_STATUS_OK)
        return status;
    *output = toHandle(pipeline.release());
    adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus retainResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return resource.identity == reinterpret_cast<uintptr_t>(adapter.vulkanDevice) && resource.resource.value
               ? VERNON_STATUS_OK
               : fail(adapter, "Vulkan adapter received a foreign or invalid resource");
}

void releaseResource(void *, VernonRuntimeProviderResourceReference) {}

VernonStatus updateBindingsImpl(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size())
        return fail(adapter, "Vulkan binding values do not match the prepared layout");
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    writes.reserve(valueCount);
    buffers.reserve(valueCount);
    images.reserve(valueCount);
    for (auto &slot : bindings.slots) {
        const auto value = std::find_if(values, values + valueCount, [&slot](const auto &candidate) {
            return candidate.slot == slot.entry.layout.slot;
        });
        if (value == values + valueCount || value->kind != slot.entry.layout.kind)
            return fail(adapter, "Vulkan binding slot or kind is invalid");
        slot.value = *value;
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            if (!value->inline_data || value->inline_size != slot.inlineStorage.size())
                return fail(adapter, "Vulkan inline or uniform-buffer binding size is invalid");
            std::memcpy(slot.inlineStorage.data(), value->inline_data, value->inline_size);
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                slot.entry.inlineOffset == UINT32_MAX) {
                void *mapped = nullptr;
                const VkResult result = rhi::vulkan::driver().mapMemory(
                    bindings.device->device, slot.inlineBuffer.memory, 0, value->inline_size, 0, &mapped);
                if (result != VK_SUCCESS)
                    return failure(adapter, result, "vkMapMemory");
                std::memcpy(mapped, value->inline_data, value->inline_size);
                rhi::vulkan::driver().unmapMemory(bindings.device->device, slot.inlineBuffer.memory);
            } else
                continue;
        }
    }
    for (auto &slot : bindings.slots) {
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER ||
            (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE && slot.entry.inlineOffset != UINT32_MAX) ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)
            continue;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = bindings.descriptorSet;
        write.dstBinding = slot.entry.layout.binding;
        write.descriptorCount = 1;
        write.descriptorType = descriptorType(slot.entry.layout.kind);
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
            slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
            const bool internallyOwned = slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                                         slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
            auto *buffer =
                internallyOwned ? &slot.inlineBuffer : fromHandle<rhi::vulkan::Buffer>(slot.value.resource.resource);
            buffers.push_back({buffer->buffer, internallyOwned ? 0 : slot.value.resource.offset,
                               internallyOwned ? slot.inlineStorage.size() : slot.value.resource.size});
            write.pBufferInfo = &buffers.back();
        } else {
            VkDescriptorImageInfo info{};
            auto *image = fromHandle<rhi::vulkan::Image>(slot.value.resource.resource);
            info.imageView = image->view;
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            const auto sampler = std::find_if(bindings.slots.begin(), bindings.slots.end(), [&](const auto &candidate) {
                return candidate.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                       candidate.entry.layout.binding == slot.entry.layout.binding;
            });
            if (sampler != bindings.slots.end()) {
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                info.sampler = (sampler->value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0
                                   ? bindings.device->defaultImplicitSampler.sampler
                                   : fromHandle<rhi::vulkan::Sampler>(sampler->value.resource.resource)->sampler;
            }
            images.push_back(info);
            write.pImageInfo = &images.back();
        }
        writes.push_back(write);
    }
    size_t bufferIndex = 0;
    size_t imageIndex = 0;
    for (auto &write : writes) {
        if (write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
            write.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            write.pBufferInfo = &buffers[bufferIndex++];
        else
            write.pImageInfo = &images[imageIndex++];
    }
    if (!writes.empty())
        rhi::vulkan::driver().updateDescriptorSets(bindings.device->device, static_cast<uint32_t>(writes.size()),
                                                   writes.data(), 0, nullptr);
    return VERNON_STATUS_OK;
}

VernonStatus createBindings(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || !layout)
        return fail(adapter, "Vulkan adapter received an invalid binding-set descriptor");
    auto bindings = std::unique_ptr<PreparedBindingSet>(new (std::nothrow) PreparedBindingSet());
    if (!bindings)
        return fail(adapter, "Vulkan binding preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    bindings->device = adapter.vulkanDevice;
    bindings->layout = layout;
    if (!bindings->device->allocateDescriptorSet(layout->descriptorSetLayout, bindings->descriptorSet, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    for (const auto &entry : layout->entries) {
        PreparedBindingSet::Slot slot;
        slot.entry = entry;
        if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
            entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            slot.inlineStorage.resize(entry.layout.element_size);
            if ((entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER || entry.inlineOffset == UINT32_MAX) &&
                !bindings->device->createBuffer(
                    slot.inlineBuffer, entry.layout.element_size,
                    entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                                                : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, adapter.error))
                return VERNON_STATUS_INTERNAL_ERROR;
        }
        bindings->slots.push_back(std::move(slot));
    }
    const VernonStatus status = updateBindingsImpl(adapter, *bindings, descriptor->values, descriptor->value_count);
    if (status != VERNON_STATUS_OK)
        return status;
    *output = toHandle(bindings.release());
    adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus updateBindings(void *data, VernonRuntimeProviderObject handle,
                            const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return fail(adapter, "Vulkan adapter received an invalid binding set");
    std::lock_guard<std::mutex> guard(bindings->mutex);
    return updateBindingsImpl(adapter, *bindings, values, valueCount);
}

void transitionImage(VkCommandBuffer command, rhi::vulkan::Image &image, VkImageLayout newLayout) {
    if (image.layout == newLayout)
        return;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = image.layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {static_cast<VkImageAspectFlags>(image.format == VK_FORMAT_D32_SFLOAT
                                                                    ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                                    : VK_IMAGE_ASPECT_COLOR_BIT),
                                0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask =
        newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        : newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : VK_ACCESS_SHADER_READ_BIT;
    rhi::vulkan::driver().cmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                             &barrier);
    image.layout = newLayout;
}

void bindResources(VkCommandBuffer command, PreparedPipeline &pipeline, PreparedBindingSet *bindings,
                   VkPipelineBindPoint bindPoint) {
    if (!bindings)
        return;
    auto &driver = rhi::vulkan::driver();
    driver.cmdBindDescriptorSets(command, bindPoint, pipeline.pipelineLayout, 0, 1, &bindings->descriptorSet, 0,
                                 nullptr);
    for (auto &slot : bindings->slots)
        if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE && slot.entry.inlineOffset != UINT32_MAX)
            driver.cmdPushConstants(command, pipeline.pipelineLayout, stages(slot.entry.layout.stage_mask),
                                    slot.entry.inlineOffset, static_cast<uint32_t>(slot.inlineStorage.size()),
                                    slot.inlineStorage.data());
}

VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || !pipeline || pipeline->graphics)
        return fail(adapter, "Vulkan adapter received an invalid dispatch");
    VkCommandBuffer command{};
    if (!pipeline->device->beginCommands(command, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    auto &driver = rhi::vulkan::driver();
    driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    bindResources(command, *pipeline, bindings, VK_PIPELINE_BIND_POINT_COMPUTE);
    driver.cmdDispatch(command, descriptor->group_count[0], descriptor->group_count[1], descriptor->group_count[2]);
    if (!pipeline->device->submitCommands(command, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || !pipeline || !pipeline->graphics || !pipeline->pipeline ||
        descriptor->color_attachment_count == 0 || descriptor->color_attachment_count > 8)
        return fail(adapter, "Vulkan adapter received an invalid draw");
    VkCommandBuffer command{};
    if (!pipeline->device->beginCommands(command, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::array<VkRenderingAttachmentInfo, 8> attachments{};
    std::array<VkImageView, 9> imageViews{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        auto *image = fromHandle<rhi::vulkan::Image>(descriptor->color_attachments[index].image.resource);
        transitionImage(command, *image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        imageViews[index] = image->view;
        attachments[index] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                              nullptr,
                              image->view,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_RESOLVE_MODE_NONE,
                              {},
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_ATTACHMENT_LOAD_OP_CLEAR,
                              VK_ATTACHMENT_STORE_OP_STORE,
                              {}};
    }
    VkRenderingAttachmentInfo depthAttachment{};
    if (descriptor->depth_stencil_attachment.resource.value) {
        auto *image = fromHandle<rhi::vulkan::Image>(descriptor->depth_stencil_attachment.resource);
        if (!image || image->format != VK_FORMAT_D32_SFLOAT)
            return fail(adapter, "Vulkan draw contains an invalid depth attachment");
        transitionImage(command, *image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        imageViews[descriptor->color_attachment_count] = image->view;
        depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                           nullptr,
                           image->view,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                           VK_RESOLVE_MODE_NONE,
                           {},
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_ATTACHMENT_LOAD_OP_CLEAR,
                           VK_ATTACHMENT_STORE_OP_DONT_CARE,
                           {}};
        depthAttachment.clearValue.depthStencil = {1.0f, 0};
    }
    const VkRenderingInfo rendering{
        VK_STRUCTURE_TYPE_RENDERING_INFO,
        nullptr,
        0,
        {{static_cast<int32_t>(descriptor->viewport[0]), static_cast<int32_t>(descriptor->viewport[1])},
         {descriptor->viewport[2], descriptor->viewport[3]}},
        1,
        0,
        static_cast<uint32_t>(descriptor->color_attachment_count),
        attachments.data(),
        descriptor->depth_stencil_attachment.resource.value ? &depthAttachment : nullptr,
        nullptr};
    auto &driver = rhi::vulkan::driver();
    VkFramebuffer framebuffer{};
    if (pipeline->device->dynamicRendering)
        driver.cmdBeginRendering(command, &rendering);
    else {
        const VkFramebufferCreateInfo framebufferInfo{
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            nullptr,
            0,
            pipeline->renderPass,
            static_cast<uint32_t>(descriptor->color_attachment_count +
                                  (descriptor->depth_stencil_attachment.resource.value ? 1 : 0)),
            imageViews.data(),
            descriptor->viewport[2],
            descriptor->viewport[3],
            1};
        const VkResult result =
            driver.createFramebuffer(pipeline->device->device, &framebufferInfo, nullptr, &framebuffer);
        if (result != VK_SUCCESS)
            return failure(adapter, result, "vkCreateFramebuffer");
        std::array<VkClearValue, 9> clearValues{};
        if (descriptor->depth_stencil_attachment.resource.value)
            clearValues[descriptor->color_attachment_count].depthStencil = {1.0f, 0};
        const VkRenderPassBeginInfo begin{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr,
            pipeline->renderPass,
            framebuffer,
            {{static_cast<int32_t>(descriptor->viewport[0]), static_cast<int32_t>(descriptor->viewport[1])},
             {descriptor->viewport[2], descriptor->viewport[3]}},
            static_cast<uint32_t>(descriptor->color_attachment_count +
                                  (descriptor->depth_stencil_attachment.resource.value ? 1 : 0)),
            clearValues.data()};
        driver.cmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    }
    driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
    bindResources(command, *pipeline, bindings, VK_PIPELINE_BIND_POINT_GRAPHICS);
    if (bindings)
        for (auto &slot : bindings->slots) {
            if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)
                transitionImage(command, *fromHandle<rhi::vulkan::Image>(slot.value.resource.resource),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            else if (slot.entry.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                auto *buffer = fromHandle<rhi::vulkan::Buffer>(slot.value.resource.resource);
                const VkDeviceSize offset = slot.value.resource.offset;
                driver.cmdBindVertexBuffers(command, slot.entry.layout.binding, 1, &buffer->buffer, &offset);
            }
        }
    const VkViewport viewport{static_cast<float>(descriptor->viewport[0]),
                              static_cast<float>(descriptor->viewport[1] + descriptor->viewport[3]),
                              static_cast<float>(descriptor->viewport[2]),
                              -static_cast<float>(descriptor->viewport[3]),
                              0,
                              1};
    const VkRect2D scissor{
        {static_cast<int32_t>(descriptor->viewport[0]), static_cast<int32_t>(descriptor->viewport[1])},
        {descriptor->viewport[2], descriptor->viewport[3]}};
    driver.cmdSetViewport(command, 0, 1, &viewport);
    driver.cmdSetScissor(command, 0, 1, &scissor);
    if (descriptor->index_count) {
        auto *buffer = fromHandle<rhi::vulkan::Buffer>(descriptor->index_buffer.resource);
        driver.cmdBindIndexBuffer(command, buffer->buffer, descriptor->index_buffer.offset, VK_INDEX_TYPE_UINT32);
        driver.cmdDrawIndexed(command, descriptor->index_count, descriptor->instance_count, 0, descriptor->first_vertex,
                              descriptor->first_instance);
    } else
        driver.cmdDraw(command, descriptor->vertex_count, descriptor->instance_count, descriptor->first_vertex,
                       descriptor->first_instance);
    if (pipeline->device->dynamicRendering)
        driver.cmdEndRendering(command);
    else
        driver.cmdEndRenderPass(command);
    if (!pipeline->device->submitCommands(command, adapter.error)) {
        if (framebuffer)
            driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    if (framebuffer)
        driver.destroyFramebuffer(pipeline->device->device, framebuffer, nullptr);
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

void destroyShader(void *, VernonRuntimeProviderObject handle) {
    auto *shader = fromHandle<PreparedShader>(handle);
    if (!shader)
        return;
    rhi::vulkan::driver().destroyShaderModule(shader->device->device, shader->module, nullptr);
    delete shader;
}
void destroyLayout(void *, VernonRuntimeProviderObject handle) {
    auto *layout = fromHandle<PreparedLayout>(handle);
    if (!layout)
        return;
    rhi::vulkan::driver().destroyDescriptorSetLayout(layout->device->device, layout->descriptorSetLayout, nullptr);
    delete layout;
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    auto *pipeline = fromHandle<PreparedPipeline>(handle);
    if (!pipeline)
        return;
    std::string ignored;
    (void)pipeline->device->synchronize(ignored);
    if (pipeline->pipeline)
        rhi::vulkan::driver().destroyPipeline(pipeline->device->device, pipeline->pipeline, nullptr);
    if (pipeline->renderPass)
        rhi::vulkan::driver().destroyRenderPass(pipeline->device->device, pipeline->renderPass, nullptr);
    for (VkShaderModule module : pipeline->specializedShaderModules)
        rhi::vulkan::driver().destroyShaderModule(pipeline->device->device, module, nullptr);
    rhi::vulkan::driver().destroyPipelineLayout(pipeline->device->device, pipeline->pipelineLayout, nullptr);
    delete pipeline;
}
void destroyBindings(void *, VernonRuntimeProviderObject handle) {
    auto *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return;
    delete bindings;
}

} // namespace

void initializeVulkanProvider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_RUNTIME_DEVICE_PROVIDER_ABI_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainResource;
    adapter.provider.release_resource = releaseResource;
    adapter.provider.create_binding_set = createBindings;
    adapter.provider.update_binding_set = updateBindings;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_binding_set = destroyBindings;
}

} // namespace vernon::runtime::rhi_adapter

#endif
