#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"
#include "provider_image_description.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace vernon::runtime {
namespace {

bool kindMatches(const Parameter &parameter, const VernonProgramArgument &argument) {
    return (parameter.kind == "tensor" && argument.kind == VERNON_PROGRAM_TENSOR) ||
           (parameter.kind == "image" && argument.kind == VERNON_PROGRAM_IMAGE) ||
           (parameter.kind == "sampler" && argument.kind == VERNON_PROGRAM_SAMPLER);
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

} // namespace

bool planGraphicsState(const VernonStageInvocationDescriptor &invocation, size_t colorCount, bool hasDepth,
                       bool hasStencil, PlannedGraphicsState &state, std::string &error) {
    state = {};
    state.depthStencil.depth_test = hasDepth;
    state.depthStencil.depth_write = hasDepth;
    state.depthStencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    state.depthStencil.stencil_read_mask = 0xff;
    state.depthStencil.stencil_write_mask = 0xff;
    if (invocation.dynamic_state && invocation.dynamic_state->struct_size < sizeof(VernonDynamicState))
        return fail(error, "DynamicState control metadata is incomplete");
    state.stencilReference = invocation.dynamic_state ? invocation.dynamic_state->stencil_reference : 0;
    state.colorBlends.resize(colorCount);
    for (auto &blend : state.colorBlends)
        blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    if (invocation.graphics_state) {
        const VernonGraphicsState &source = *invocation.graphics_state;
        if (source.struct_size < sizeof(source))
            return fail(error, "graphics state metadata is incomplete");
        if (source.color_blend_count != colorCount || (colorCount && !source.color_blends))
            return fail(error, "graphics state color blend count does not match the color attachment count");
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
    if (!hasDepth && (state.depthStencil.depth_test || state.depthStencil.depth_write))
        return fail(error, "graphics depth test/write requires a depth attachment");
    if (!hasStencil && state.depthStencil.stencil_test)
        return fail(error, "graphics stencil test requires a stencil attachment");
    if (state.rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
        state.rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE || state.rasterization.depth_clamp > 1 ||
        state.rasterization.depth_bias_enabled > 1 || !std::isfinite(state.rasterization.depth_bias_constant) ||
        !std::isfinite(state.rasterization.depth_bias_slope) || state.depthStencil.depth_test > 1 ||
        state.depthStencil.depth_write > 1 || state.depthStencil.depth_compare > VERNON_RHI_COMPARE_ALWAYS ||
        state.depthStencil.stencil_test > 1 || !validFace(state.depthStencil.front) ||
        !validFace(state.depthStencil.back) || state.depthStencil.stencil_read_mask > 0xff ||
        state.depthStencil.stencil_write_mask > 0xff || state.stencilReference > 0xff)
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

bool planGraphicsInvocation(const StageBindingPlan &stagePlan, const VernonStageInvocationDescriptor &source,
                            DescribeImageResource describeImage, void *describeImageUserData,
                            PlannedGraphicsInvocation &plan, std::string &error) {
    plan = {};
    for (size_t index = 0; index < source.argument_count; ++index)
        if (!plan.arguments.emplace(source.arguments[index].slot, &source.arguments[index]).second)
            return fail(error, "duplicate pipeline argument slot");
    if (plan.arguments.size() != stagePlan.parameters.size())
        return fail(error, "pipeline argument count does not match layout");

    for (const Parameter &parameter : stagePlan.parameters) {
        const auto found = plan.arguments.find(parameter.slot);
        if (found == plan.arguments.end()) {
            error = "graphics pipeline argument '" + parameter.name + "' is missing at slot " +
                    std::to_string(parameter.slot);
            return false;
        }
        if (!kindMatches(parameter, *found->second)) {
            error = "graphics pipeline argument '" + parameter.name + "' expects " + parameter.kind +
                    " but received kind " + std::to_string(found->second->kind);
            return false;
        }
        const VernonProgramArgument &argument = *found->second;
        if (argument.kind == VERNON_PROGRAM_TENSOR) {
            const ValueLayout &expectedLayout = !parameter.elementLayout.leaves.empty() ? parameter.elementLayout
                                                : parameter.valueLayout                 ? *parameter.valueLayout
                                                                                        : parameter.elementLayout;
            if (!valueLayoutsEqual(argument.tensor.element_layout, pipelineValueLayout(expectedLayout)) ||
                !validTensor(argument.tensor))
                return fail(error, "pipeline Tensor argument does not match layout");
            if (parameter.source != StageParameterSource::Direct) {
                const bool allowLeading =
                    std::any_of(parameter.uses.begin(), parameter.uses.end(),
                                [](const ParameterUse &use) { return use.interfaceKind == "input"; });
                const size_t offset = allowLeading && argument.tensor.rank == parameter.shape.size() + 1 ? 1 : 0;
                if (argument.tensor.rank != parameter.shape.size() + offset) {
                    error = "pipeline Tensor '" + parameter.name + "' rank " + std::to_string(argument.tensor.rank) +
                            " does not match layout rank " + std::to_string(parameter.shape.size());
                    return false;
                }
                for (size_t dimension = 0; dimension < parameter.shape.size(); ++dimension)
                    if (parameter.shape[dimension] &&
                        parameter.shape[dimension] != argument.tensor.shape[dimension + offset])
                        return fail(error, "pipeline Tensor shape does not match layout");
            }
        } else if (argument.kind == VERNON_PROGRAM_IMAGE) {
            if (!argument.image.view.identity || !argument.image.view.resource.value)
                return fail(error, "pipeline image argument does not match layout");
        } else if (!argument.resource.identity || !argument.resource.resource.value) {
            return fail(error, "pipeline sampler belongs to another runtime");
        }
    }

    if (stagePlan.vertex.empty())
        return true;

    if (!source.graphics_state || source.graphics_state->struct_size < sizeof(VernonGraphicsState))
        return fail(error, "graphics pipeline state is missing or incomplete");
    if (!source.render_pass || source.render_pass->struct_size < sizeof(VernonRenderPass))
        return fail(error, "RenderPass control is missing or incomplete");
    if (!source.draw_command || source.draw_command->struct_size < sizeof(VernonDrawCommand))
        return fail(error, "DrawCommand control is missing or incomplete");
    if (source.dynamic_state && source.dynamic_state->struct_size < sizeof(VernonDynamicState))
        return fail(error, "DynamicState control metadata is incomplete");
    const VernonGraphicsState &graphicsState = *source.graphics_state;
    const VernonRenderPass &renderPass = *source.render_pass;
    const VernonDrawCommand &draw = *source.draw_command;
    if (graphicsState.topology > VERNON_TOPOLOGY_POINT_LIST)
        return fail(error, "graphics pipeline topology is invalid");
    plan.topology = graphicsState.topology;
    if ((renderPass.color_attachment_count && !renderPass.color_attachments) ||
        (!renderPass.color_attachment_count && !renderPass.depth_attachment))
        return fail(error, "graphics pipeline requires a color or depth attachment");
    if (!describeImage)
        return fail(error, "graphics image descriptor resolver is missing");
    for (size_t index = 0; index < renderPass.color_attachment_count; ++index) {
        const VernonColorAttachment &attachment = renderPass.color_attachments[index];
        if (!attachment.view.identity || !attachment.view.resource.value)
            return fail(error, "render target is invalid");
        if (attachment.load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
            attachment.store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD)
            return fail(error, "render target attachment operation is invalid");
        plan.attachments.push_back(&attachment);
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
    plan.attachmentFormats.reserve(plan.attachments.size());
    for (const VernonColorAttachment *attachment : plan.attachments) {
        VernonRuntimeProviderImageDescription description{};
        description.struct_size = sizeof(description);
        if (describeImage(describeImageUserData, attachment->view, &description) != VERNON_STATUS_OK ||
            !providerImageDescriptionIsCanonical(description) ||
            description.resource_kind != VERNON_RUNTIME_PROVIDER_IMAGE_VIEW ||
            description.view.dimension != VERNON_TEXTURE_2D ||
            !(description.image.usage & VERNON_IMAGE_COLOR_ATTACHMENT) ||
            description.view.subresources.aspects != VERNON_IMAGE_ASPECT_COLOR)
            return fail(error, "render target view is incompatible");
        const uint32_t width =
            std::max(description.image.extent.width >> description.view.subresources.base_mip_level, 1u);
        const uint32_t height =
            std::max(description.image.extent.height >> description.view.subresources.base_mip_level, 1u);
        if (!plan.attachmentWidth) {
            plan.attachmentWidth = width;
            plan.attachmentHeight = height;
        } else if (plan.attachmentWidth != width || plan.attachmentHeight != height) {
            return fail(error, "render target extents differ");
        }
        plan.attachmentFormats.push_back(description.view.format);
    }
    if (renderPass.depth_attachment) {
        const VernonDepthAttachment &attachment = *renderPass.depth_attachment;
        if (!attachment.view.identity || !attachment.view.resource.value)
            return fail(error, "depth attachment is invalid");
        VernonRuntimeProviderImageDescription description{};
        description.struct_size = sizeof(description);
        if (describeImage(describeImageUserData, attachment.view, &description) != VERNON_STATUS_OK ||
            !providerImageDescriptionIsCanonical(description) ||
            description.resource_kind != VERNON_RUNTIME_PROVIDER_IMAGE_VIEW)
            return fail(error, "depth attachment is stale");
        const VernonTextureFormat format = description.view.format;
        const uint32_t width =
            std::max(description.image.extent.width >> description.view.subresources.base_mip_level, 1u);
        const uint32_t height =
            std::max(description.image.extent.height >> description.view.subresources.base_mip_level, 1u);
        const bool firstAttachment = !plan.attachmentWidth;
        if (description.view.dimension != VERNON_TEXTURE_2D ||
            !(description.image.usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT) || !width || !height ||
            (!firstAttachment && (width != plan.attachmentWidth || height != plan.attachmentHeight)) ||
            (format != VERNON_TEXTURE_D32_FLOAT && format != VERNON_TEXTURE_D32_FLOAT_S8_UINT))
            return fail(error, "depth attachment must be D32 or D32S8 with the render-target extent");
        if (firstAttachment) {
            plan.attachmentWidth = width;
            plan.attachmentHeight = height;
        }
        if (attachment.load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
            attachment.store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD ||
            !std::isfinite(attachment.clear_depth) || attachment.clear_depth < 0.0f || attachment.clear_depth > 1.0f ||
            attachment.stencil_load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
            attachment.stencil_store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD ||
            attachment.clear_stencil > 0xff || (format == VERNON_TEXTURE_D32_FLOAT && attachment.clear_stencil != 0))
            return fail(error, "depth attachment operation is invalid");
        plan.depthAttachment = &attachment;
        plan.depthFormat = format;
    }
    const uint32_t *dynamicViewport = source.dynamic_state ? source.dynamic_state->viewport : nullptr;
    const uint32_t *dynamicScissor = source.dynamic_state ? source.dynamic_state->scissor : nullptr;
    const bool hasViewport = dynamicViewport && dynamicViewport[2] && dynamicViewport[3];
    const uint32_t *viewport = hasViewport ? dynamicViewport : renderPass.render_area;
    plan.viewport[0] = viewport[0];
    plan.viewport[1] = viewport[1];
    plan.viewport[2] = viewport[2] ? viewport[2] : plan.attachmentWidth;
    plan.viewport[3] = viewport[3] ? viewport[3] : plan.attachmentHeight;
    const bool hasScissor = dynamicScissor && dynamicScissor[2] && dynamicScissor[3];
    for (size_t index = 0; index < 4; ++index)
        plan.scissor[index] = hasScissor ? dynamicScissor[index] : plan.viewport[index];
    plan.resolution = {static_cast<float>(plan.viewport[2]), static_cast<float>(plan.viewport[3])};
    plan.vertexCount = draw.vertex_count;
    plan.instanceCount = draw.instance_count;

    std::map<std::pair<uint32_t, uint32_t>, PlannedSampledResource> sampled;
    for (const Parameter &parameter : stagePlan.parameters) {
        const VernonProgramArgument &argument = *plan.arguments.at(parameter.slot);
        for (const ParameterUse &use : parameter.uses) {
            const uint32_t stage = shaderStage(use);
            if (!stage)
                continue;
            if (argument.kind == VERNON_PROGRAM_IMAGE) {
                if (use.binding == UINT32_MAX)
                    return fail(error, "sampled image is missing set/binding");
                PlannedSampledResource &resource = sampled[{use.descriptorSet, use.binding}];
                if (resource.imageView.resource.value &&
                    resource.imageView.resource.value != argument.image.view.resource.value)
                    return fail(error, "sampled image binding is ambiguous");
                resource.imageView = argument.image.view;
                resource.stages |= stage;
                continue;
            }
            if (argument.kind == VERNON_PROGRAM_SAMPLER) {
                if (use.sampledImageBindings.empty())
                    return fail(error, "sampler reflection has no paired sampled image "
                                       "binding");
                for (const SampledImageBinding &binding : use.sampledImageBindings) {
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
            if (argument.kind != VERNON_PROGRAM_TENSOR || tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
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

    for (const Parameter &parameter : stagePlan.runtimeParameters) {
        if (parameter.source != StageParameterSource::ImplicitSampler)
            continue;
        for (const ParameterUse &use : parameter.uses) {
            const uint32_t stage = shaderStage(use);
            if (!stage)
                continue;
            for (const SampledImageBinding &binding : use.sampledImageBindings) {
                PlannedSampledResource &resource = sampled[{binding.descriptorSet, binding.binding}];
                if (resource.explicitSampler)
                    return fail(error, "implicit and explicit samplers conflict");
                resource.implicitSampler = true;
                resource.stages |= stage;
            }
        }
    }
    for (const auto &[binding, resource] : sampled) {
        if (!resource.imageView.resource.value ||
            (!resource.samplerResource.resource.value && !resource.implicitSampler))
            return fail(error, "sampled image requires paired image view and sampler");
        plan.sampledResources.emplace(binding, resource);
    }

    if (!plan.instanceCount)
        plan.instanceCount = 1;
    if (!plan.vertexCount)
        return fail(error, "graphics draw counts cannot be inferred");
    if (draw.index_binding) {
        const VernonIndexBinding &index = *draw.index_binding;
        if (!index.resource.identity || !index.resource.resource.value || index.type != VERNON_INDEX_U32 ||
            !index.index_count)
            return fail(error, "index binding is invalid");
        plan.indexBinding = &index;
    }
    return true;
}

} // namespace vernon::runtime
