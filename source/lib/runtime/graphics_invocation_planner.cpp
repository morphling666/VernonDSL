#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>

namespace vernon::runtime {
namespace {

bool kindMatches(const Parameter &parameter, const VernonPipelineArgument &argument) {
    return (parameter.kind == "tensor" && argument.kind == VERNON_PIPELINE_TENSOR) ||
           (parameter.kind == "texture" && argument.kind == VERNON_PIPELINE_TEXTURE) ||
           (parameter.kind == "sampler" && argument.kind == VERNON_PIPELINE_SAMPLER);
}

bool validTensor(const VernonTensorView &tensor) {
    if (tensor.struct_size < sizeof(VernonTensorView) || tensor.access > VERNON_ACCESS_READ_WRITE ||
        (tensor.storage != VERNON_TENSOR_HOST && tensor.storage != VERNON_TENSOR_RHI_RESOURCE))
        return false;
    if (tensor.storage == VERNON_TENSOR_HOST && !tensorFitsAllocation(tensor))
        return false;
    if (tensor.storage == VERNON_TENSOR_HOST)
        return tensor.host_data != nullptr;
    return tensor.resource.identity && tensor.resource.resource.value && tensor.byte_offset <= tensor.resource.size &&
           tensor.byte_size <= tensor.resource.size;
}

uint32_t shaderStage(const ParameterUse &use) {
    if (use.stage == "vertex")
        return PLANNED_STAGE_VERTEX;
    if (use.stage == "fragment")
        return PLANNED_STAGE_FRAGMENT;
    return 0;
}

bool fail(std::string &error, const char *message) {
    error = message;
    return false;
}

bool rasterizationEqual(const VernonRasterizationState &left, const VernonRasterizationState &right) {
    return left.cull_mode == right.cull_mode && left.front_face == right.front_face &&
           left.depth_clamp == right.depth_clamp && left.depth_bias_enabled == right.depth_bias_enabled &&
           left.depth_bias_constant == right.depth_bias_constant && left.depth_bias_slope == right.depth_bias_slope;
}

bool stencilFaceEqual(const VernonStencilFaceState &left, const VernonStencilFaceState &right) {
    return left.stencil_fail == right.stencil_fail && left.depth_fail == right.depth_fail && left.pass == right.pass &&
           left.compare == right.compare;
}

bool depthStencilEqual(const VernonDepthStencilState &left, const VernonDepthStencilState &right) {
    return left.depth_test == right.depth_test && left.depth_write == right.depth_write &&
           left.depth_compare == right.depth_compare && left.stencil_test == right.stencil_test &&
           stencilFaceEqual(left.front, right.front) && stencilFaceEqual(left.back, right.back) &&
           left.stencil_read_mask == right.stencil_read_mask && left.stencil_write_mask == right.stencil_write_mask;
}

bool blendEqual(const VernonColorBlendState &left, const VernonColorBlendState &right) {
    return left.blend_enabled == right.blend_enabled && left.source_color_factor == right.source_color_factor &&
           left.destination_color_factor == right.destination_color_factor &&
           left.color_operation == right.color_operation && left.source_alpha_factor == right.source_alpha_factor &&
           left.destination_alpha_factor == right.destination_alpha_factor &&
           left.alpha_operation == right.alpha_operation && left.write_mask == right.write_mask;
}

} // namespace

bool planGraphicsState(const VernonPipelineInvocation &invocation, size_t colorCount, bool hasDepth, bool hasStencil,
                       PlannedGraphicsState &state, std::string &error) {
    state = {};
    state.depthStencil.depth_test = hasDepth;
    state.depthStencil.depth_write = hasDepth;
    state.depthStencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    state.depthStencil.stencil_read_mask = 0xff;
    state.depthStencil.stencil_write_mask = 0xff;
    state.stencilReference = invocation.stencil_reference;
    state.colorBlends.resize(colorCount);
    for (auto &blend : state.colorBlends)
        blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    if (invocation.graphics_state) {
        const VernonGraphicsState &source = *invocation.graphics_state;
        if (source.struct_size < sizeof(source) || source.color_blend_count != colorCount ||
            (colorCount && !source.color_blends))
            return fail(error, "graphics state does not match the render-target layout");
        state.rasterization = source.rasterization;
        state.depthStencil = source.depth_stencil;
        for (size_t index = 0; index < colorCount; ++index) {
            state.colorBlends[index] = source.color_blends[index];
        }
    }
    const auto validFace = [](const VernonStencilFaceState &face) {
        return face.stencil_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.depth_fail <= VERNON_RHI_STENCIL_DECREMENT_WRAP && face.pass <= VERNON_RHI_STENCIL_DECREMENT_WRAP &&
               face.compare <= VERNON_RHI_COMPARE_ALWAYS;
    };
    if (state.rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
        state.rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE || state.rasterization.depth_clamp > 1 ||
        state.rasterization.depth_bias_enabled > 1 || !std::isfinite(state.rasterization.depth_bias_constant) ||
        !std::isfinite(state.rasterization.depth_bias_slope) || state.depthStencil.depth_test > 1 ||
        state.depthStencil.depth_write > 1 || state.depthStencil.depth_compare > VERNON_RHI_COMPARE_ALWAYS ||
        state.depthStencil.stencil_test > 1 || !validFace(state.depthStencil.front) ||
        !validFace(state.depthStencil.back) || state.depthStencil.stencil_read_mask > 0xff ||
        state.depthStencil.stencil_write_mask > 0xff || state.stencilReference > 0xff ||
        (!hasDepth && (state.depthStencil.depth_test || state.depthStencil.depth_write)) ||
        (!hasStencil && state.depthStencil.stencil_test))
        return fail(error, "graphics state contains an invalid or unsupported value");
    for (const auto &blend : state.colorBlends)
        if (blend.blend_enabled > 1 || blend.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.color_operation > VERNON_RHI_BLEND_MAXIMUM || blend.alpha_operation > VERNON_RHI_BLEND_MAXIMUM ||
            (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
            return fail(error, "graphics blend state contains an invalid value");
    if (!state.rasterization.depth_bias_enabled) {
        state.rasterization.depth_bias_constant = 0;
        state.rasterization.depth_bias_slope = 0;
    }
    if (!state.depthStencil.depth_test)
        state.depthStencil.depth_compare = VERNON_RHI_COMPARE_ALWAYS;
    if (!hasDepth)
        state.depthStencil.depth_write = 0;
    if (!hasStencil || !state.depthStencil.stencil_test) {
        state.depthStencil.stencil_test = 0;
        state.depthStencil.front = {VERNON_RHI_STENCIL_ZERO, VERNON_RHI_STENCIL_ZERO, VERNON_RHI_STENCIL_ZERO,
                                    VERNON_RHI_COMPARE_NEVER};
        state.depthStencil.back = state.depthStencil.front;
        state.depthStencil.stencil_read_mask = 0;
        state.depthStencil.stencil_write_mask = 0;
    }
    for (auto &blend : state.colorBlends)
        if (!blend.blend_enabled) {
            blend.source_color_factor = VERNON_RHI_BLEND_ONE;
            blend.destination_color_factor = VERNON_RHI_BLEND_ZERO;
            blend.color_operation = VERNON_RHI_BLEND_ADD;
            blend.source_alpha_factor = VERNON_RHI_BLEND_ONE;
            blend.destination_alpha_factor = VERNON_RHI_BLEND_ZERO;
            blend.alpha_operation = VERNON_RHI_BLEND_ADD;
        }
    return true;
}

bool graphicsVariantKeysEqual(const GraphicsVariantKey &left, const GraphicsVariantKey &right) {
    return left.topology == right.topology && left.colorFormats == right.colorFormats &&
           left.depthStencilFormat == right.depthStencilFormat && left.sampleCount == right.sampleCount &&
           left.vertexStrides == right.vertexStrides && rasterizationEqual(left.rasterization, right.rasterization) &&
           depthStencilEqual(left.depthStencil, right.depthStencil) &&
           left.colorBlends.size() == right.colorBlends.size() &&
           std::equal(left.colorBlends.begin(), left.colorBlends.end(), right.colorBlends.begin(), blendEqual);
}

size_t graphicsVariantKeyHash(const GraphicsVariantKey &key) {
    size_t result = 0xcbf29ce484222325ull;
    const auto combine = [&](auto value) {
        result ^= std::hash<decltype(value)>{}(value) + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2);
    };
    const auto combineFace = [&](const VernonStencilFaceState &face) {
        combine(face.stencil_fail);
        combine(face.depth_fail);
        combine(face.pass);
        combine(face.compare);
    };
    combine(key.topology);
    for (uint32_t format : key.colorFormats)
        combine(format);
    combine(key.colorFormats.size());
    combine(key.depthStencilFormat);
    combine(key.sampleCount);
    for (uint32_t stride : key.vertexStrides)
        combine(stride);
    combine(key.vertexStrides.size());
    combine(key.rasterization.cull_mode);
    combine(key.rasterization.front_face);
    combine(key.rasterization.depth_clamp);
    combine(key.rasterization.depth_bias_enabled);
    combine(key.rasterization.depth_bias_constant);
    combine(key.rasterization.depth_bias_slope);
    combine(key.depthStencil.depth_test);
    combine(key.depthStencil.depth_write);
    combine(key.depthStencil.depth_compare);
    combine(key.depthStencil.stencil_test);
    combineFace(key.depthStencil.front);
    combineFace(key.depthStencil.back);
    combine(key.depthStencil.stencil_read_mask);
    combine(key.depthStencil.stencil_write_mask);
    for (const auto &blend : key.colorBlends) {
        combine(blend.blend_enabled);
        combine(blend.source_color_factor);
        combine(blend.destination_color_factor);
        combine(blend.color_operation);
        combine(blend.source_alpha_factor);
        combine(blend.destination_alpha_factor);
        combine(blend.alpha_operation);
        combine(blend.write_mask);
    }
    combine(key.colorBlends.size());
    return result;
}

VernonStatus ensureGraphicsVariant(VernonRuntimeCorePipeline *pipeline, const GraphicsVariantKey &key,
                                   PreparedGraphicsVariant &prepared) {
    if (prepared.handle && graphicsVariantKeysEqual(prepared.key, key))
        return VERNON_STATUS_OK;
    VernonRuntimeCoreGraphicsCompatibility compatibility{};
    compatibility.struct_size = sizeof(compatibility);
    compatibility.topology = key.topology;
    compatibility.color_formats = key.colorFormats.data();
    compatibility.color_format_count = key.colorFormats.size();
    compatibility.depth_stencil_format = key.depthStencilFormat;
    compatibility.sample_count = key.sampleCount;
    compatibility.vertex_strides = key.vertexStrides.data();
    compatibility.vertex_stride_count = key.vertexStrides.size();
    compatibility.rasterization = key.rasterization;
    compatibility.depth_stencil = key.depthStencil;
    compatibility.color_blends = key.colorBlends.data();
    compatibility.color_blend_count = key.colorBlends.size();
    VernonRuntimeCoreGraphicsVariant *replacement{};
    const VernonStatus status = vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &replacement);
    if (status != VERNON_STATUS_OK)
        return status;
    vernonRuntimeCoreGraphicsVariantDestroy(prepared.handle);
    prepared.key = key;
    prepared.handle = replacement;
    return VERNON_STATUS_OK;
}

void destroyGraphicsVariant(PreparedGraphicsVariant &prepared) {
    vernonRuntimeCoreGraphicsVariantDestroy(prepared.handle);
    prepared = {};
}

bool planGraphicsInvocation(const Variant &variant, const VernonPipelineInvocation &invocation,
                            PlannedGraphicsInvocation &plan, std::string &error) {
    plan = {};
    for (size_t index = 0; index < invocation.argument_count; ++index)
        if (!plan.arguments.emplace(invocation.arguments[index].slot, &invocation.arguments[index]).second)
            return fail(error, "duplicate pipeline argument slot");
    if (plan.arguments.size() != variant.parameters.size())
        return fail(error, "pipeline argument count does not match layout");

    for (const Parameter &parameter : variant.parameters) {
        const auto found = plan.arguments.find(parameter.slot);
        if (found == plan.arguments.end() || !kindMatches(parameter, *found->second))
            return fail(error, "pipeline argument kind does not match layout");
        const VernonPipelineArgument &argument = *found->second;
        if (argument.kind == VERNON_PIPELINE_TENSOR) {
            const ValueLayout &expectedLayout =
                parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
            if (!valueLayoutsEqual(argument.tensor.element_layout, pipelineValueLayout(expectedLayout)) ||
                !validTensor(argument.tensor))
                return fail(error, "pipeline Tensor argument does not match layout");
            if (parameter.source != "direct") {
                const bool allowLeading =
                    std::any_of(parameter.uses.begin(), parameter.uses.end(),
                                [](const ParameterUse &use) { return use.interfaceKind == "input"; });
                const size_t offset = allowLeading && argument.tensor.rank == parameter.shape.size() + 1 ? 1 : 0;
                if (argument.tensor.rank != parameter.shape.size() + offset)
                    return fail(error, "pipeline Tensor rank does not match layout");
                for (size_t dimension = 0; dimension < parameter.shape.size(); ++dimension)
                    if (parameter.shape[dimension] &&
                        parameter.shape[dimension] != argument.tensor.shape[dimension + offset])
                        return fail(error, "pipeline Tensor shape does not match layout");
            }
        } else if (argument.kind == VERNON_PIPELINE_TEXTURE) {
            if (!argument.texture.resource.identity || !argument.texture.resource.resource.value)
                return fail(error, "pipeline texture argument does not match layout");
            if (!parameter.dimension.empty() &&
                ((parameter.dimension == "2d" && argument.texture.dimension != VERNON_TEXTURE_2D) ||
                 (parameter.dimension == "3d" && argument.texture.dimension != VERNON_TEXTURE_3D) ||
                 (parameter.dimension == "cube" && argument.texture.dimension != VERNON_TEXTURE_CUBE)))
                return fail(error, "pipeline texture argument does not match layout");
            if (!parameter.format.empty()) {
                const auto format = pipelineTextureFormat(parameter.format);
                if (!format || argument.texture.format != *format)
                    return fail(error, "pipeline storage texture format does not match layout");
            }
            if ((parameter.access == "read" && argument.texture.access != VERNON_ACCESS_READ) ||
                (parameter.access == "write" && argument.texture.access != VERNON_ACCESS_WRITE) ||
                (parameter.access == "read_write" && argument.texture.access != VERNON_ACCESS_READ_WRITE))
                return fail(error, "pipeline texture access does not match layout");
        } else if (!argument.resource.identity || !argument.resource.resource.value) {
            return fail(error, "pipeline sampler belongs to another runtime");
        }
    }

    if (variant.vertex.empty())
        return true;

    if (!invocation.color_attachment_count || !invocation.color_attachments)
        return fail(error, "graphics pipeline requires color attachments");
    for (size_t index = 0; index < invocation.color_attachment_count; ++index) {
        const VernonColorAttachment &attachment = invocation.color_attachments[index];
        uint32_t width = attachment.width;
        uint32_t height = attachment.height;
        if (!attachment.resource.identity || !attachment.resource.resource.value)
            return fail(error, "render target is invalid");
        if (!width || !height)
            return fail(error, "render target extent is invalid");
        if (attachment.load_operation > VERNON_RHI_LOAD_DISCARD ||
            attachment.store_operation > VERNON_RHI_STORE_DISCARD)
            return fail(error, "render target attachment operation is invalid");
        if (!plan.attachmentWidth) {
            plan.attachmentWidth = width;
            plan.attachmentHeight = height;
        } else if (plan.attachmentWidth != width || plan.attachmentHeight != height) {
            return fail(error, "render target extents differ");
        }
        plan.attachments.push_back(&attachment);
    }
    if (invocation.depth_attachment) {
        const VernonDepthAttachment &attachment = *invocation.depth_attachment;
        uint32_t width = attachment.width;
        uint32_t height = attachment.height;
        VernonTextureFormat format = attachment.format;
        if (!attachment.resource.identity || !attachment.resource.resource.value)
            return fail(error, "depth attachment is invalid");
        if (!width || !height || width != plan.attachmentWidth || height != plan.attachmentHeight ||
            (format != VERNON_TEXTURE_D32_FLOAT && format != VERNON_TEXTURE_D32_FLOAT_S8_UINT))
            return fail(error, "depth attachment must be D32 or D32S8 with the render-target extent");
        if (attachment.load_operation > VERNON_RHI_LOAD_DISCARD ||
            attachment.store_operation > VERNON_RHI_STORE_DISCARD || !std::isfinite(attachment.clear_depth) ||
            attachment.clear_depth < 0.0f || attachment.clear_depth > 1.0f ||
            attachment.stencil_load_operation > VERNON_RHI_LOAD_DISCARD ||
            attachment.stencil_store_operation > VERNON_RHI_STORE_DISCARD || attachment.clear_stencil > 0xff ||
            (format == VERNON_TEXTURE_D32_FLOAT && attachment.clear_stencil != 0))
            return fail(error, "depth attachment operation is invalid");
        plan.depthAttachment = &attachment;
    }
    std::sort(plan.attachments.begin(), plan.attachments.end(),
              [](const VernonColorAttachment *left, const VernonColorAttachment *right) {
                  return left->location < right->location;
              });
    if (std::adjacent_find(plan.attachments.begin(), plan.attachments.end(),
                           [](const VernonColorAttachment *left, const VernonColorAttachment *right) {
                               return left->location == right->location;
                           }) != plan.attachments.end())
        return fail(error, "render target locations are duplicated");
    for (size_t index = 0; index < plan.attachments.size(); ++index)
        if (plan.attachments[index]->location != index)
            return fail(error, "render target locations must be contiguous from zero");
    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];
    plan.resolution = {static_cast<float>(hasViewport ? invocation.viewport[2] : plan.attachmentWidth),
                       static_cast<float>(hasViewport ? invocation.viewport[3] : plan.attachmentHeight)};
    plan.vertexCount = invocation.vertex_count;
    plan.instanceCount = invocation.instance_count;

    std::map<std::pair<uint32_t, uint32_t>, PlannedSampledResource> sampled;
    for (const Parameter &parameter : variant.parameters) {
        const VernonPipelineArgument &argument = *plan.arguments.at(parameter.slot);
        for (const ParameterUse &use : parameter.uses) {
            const uint32_t stage = shaderStage(use);
            if (!stage)
                continue;
            if (argument.kind == VERNON_PIPELINE_TEXTURE) {
                if (use.binding == UINT32_MAX)
                    return fail(error, "sampled texture is missing set/binding");
                PlannedSampledResource &resource = sampled[{use.descriptorSet, use.binding}];
                if (resource.imageResource.resource.value &&
                    resource.imageResource.resource.value != argument.texture.resource.resource.value)
                    return fail(error, "sampled texture binding is ambiguous");
                if (resource.imageResource.resource.value && !resource.explicitSampler &&
                    resource.samplerResource.resource.value != argument.texture.sampler_resource.resource.value)
                    return fail(error, "texture binding has conflicting sampler policies");
                resource.imageResource = argument.texture.resource;
                if (!resource.explicitSampler)
                    resource.samplerResource = argument.texture.sampler_resource;
                resource.stages |= stage;
                continue;
            }
            if (argument.kind == VERNON_PIPELINE_SAMPLER) {
                if (use.sampledTextureBindings.empty())
                    return fail(error, "sampler reflection has no paired sampled texture "
                                       "binding");
                for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                    PlannedSampledResource &resource = sampled[{binding.descriptorSet, binding.binding}];
                    if (resource.explicitSampler &&
                        resource.samplerResource.resource.value != argument.resource.resource.value)
                        return fail(error, "sampler pairing is ambiguous");
                    resource.samplerResource = argument.resource;
                    resource.explicitSampler = true;
                    resource.stages |= stage;
                }
                continue;
            }
            if (use.interfaceKind != "input")
                continue;
            const VernonTensorView &tensor = argument.tensor;
            if (argument.kind != VERNON_PIPELINE_TENSOR || tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !tensor.resource.resource.value || !tensor.rank || !tensor.shape || !tensor.byte_strides ||
                use.location == UINT32_MAX || use.attributeLeaves.empty())
                return fail(error, "graphics Tensor view is invalid");
            for (const AttributeLeaf &leaf : use.attributeLeaves)
                if (leaf.dtype != "i32" && leaf.dtype != "u32" && leaf.dtype != "f16" && leaf.dtype != "f32" &&
                    leaf.dtype != "f64")
                    return fail(error, "graphics Tensor dtype is unsupported");
            for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension)
                if (tensor.byte_strides[dimension] <= 0)
                    return fail(error, "graphics Tensor strides must be positive");
            if (tensor.shape[0] > std::numeric_limits<uint32_t>::max())
                return fail(error, "graphics Tensor leading dimension is too large");
            const uint32_t leading = static_cast<uint32_t>(tensor.shape[0]);
            const bool instanced = use.divisor != 0;
            uint32_t &inferred = instanced ? plan.instanceCount : plan.vertexCount;
            const uint32_t divisor = instanced ? std::max(use.divisor, 1u) : 1u;
            if (inferred) {
                const uint32_t requiredRecords = (inferred - 1) / divisor + 1;
                if (requiredRecords != leading)
                    return fail(error, "graphics Tensor leading dimensions conflict");
            } else {
                if (leading > std::numeric_limits<uint32_t>::max() / divisor)
                    return fail(error, "graphics inferred draw count is too large");
                inferred = leading * divisor;
            }
            uint64_t expectedStride = tensor.element_layout.byte_size;
            for (uint32_t dimension = tensor.rank; dimension-- > 1;) {
                if (tensor.byte_strides[dimension] != static_cast<int64_t>(expectedStride))
                    return fail(error, "graphics Tensor inner dimensions must be contiguous row-major");
                if (tensor.shape[dimension] > std::numeric_limits<uint64_t>::max() / expectedStride)
                    return fail(error, "graphics Tensor shape is too large");
                expectedStride *= tensor.shape[dimension];
            }
            if (static_cast<uint64_t>(tensor.byte_strides[0]) < expectedStride)
                return fail(error, "graphics Tensor record stride is too small");
            plan.vertexInputs.push_back({&use, &tensor, instanced});
        }
    }

    for (const Parameter &parameter : variant.internalParameters) {
        if (parameter.source != "implicit_sampler")
            continue;
        for (const ParameterUse &use : parameter.uses) {
            const uint32_t stage = shaderStage(use);
            if (!stage)
                continue;
            for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                PlannedSampledResource &resource = sampled[{binding.descriptorSet, binding.binding}];
                if (resource.explicitSampler)
                    return fail(error, "implicit and explicit samplers conflict");
                resource.implicitSampler = true;
                resource.stages |= stage;
            }
        }
    }
    for (const auto &[binding, resource] : sampled) {
        if (!resource.imageResource.resource.value ||
            (!resource.samplerResource.resource.value && !resource.implicitSampler))
            return fail(error, "sampled image requires paired texture and sampler");
        plan.sampledResources.emplace(binding, resource);
    }

    if (!plan.instanceCount)
        plan.instanceCount = 1;
    if (!plan.vertexCount)
        return fail(error, "graphics draw counts cannot be inferred");
    if (invocation.index_binding) {
        const VernonIndexBinding &index = *invocation.index_binding;
        if (!index.resource.identity || !index.resource.resource.value || index.type != VERNON_INDEX_U32 ||
            !index.index_count)
            return fail(error, "index binding is invalid");
        plan.indexBinding = &index;
    }
    return true;
}

} // namespace vernon::runtime
