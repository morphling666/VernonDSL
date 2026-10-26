#include "graphics_invocation_planner.h"
#include "VernonFormats.hpp"
#include "pipeline_metadata.h"
#include "provider_image_description.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

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

bool shaderStage(const ParameterUse &use) { return use.stage == "vertex" || use.stage == "fragment"; }

bool fail(std::string &error, const char *message) {
    error = message;
    return false;
}

} // namespace

bool planGraphicsState(const VernonGraphicsState *graphicsState, size_t colorCount, bool hasDepth, bool hasStencil,
                       PlannedGraphicsState &state, std::string &error) {
    state = {};
    state.depthStencil.depth_test = hasDepth;
    state.depthStencil.depth_write = hasDepth;
    state.depthStencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    state.depthStencil.stencil_read_mask = 0xff;
    state.depthStencil.stencil_write_mask = 0xff;
    state.colorBlends.resize(colorCount);
    for (auto &blend : state.colorBlends)
        blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    if (graphicsState) {
        const VernonGraphicsState &source = *graphicsState;
        if (source.struct_size < sizeof(source))
            return fail(error, "graphics state metadata is incomplete");
        if (source.topology > VERNON_TOPOLOGY_POINT_LIST)
            return fail(error, "graphics pipeline topology is invalid");
        state.topology = source.topology;
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
        state.depthStencil.stencil_write_mask > 0xff)
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

bool planRenderTargets(const VernonRenderPass *source, DescribeImageResource describeImage, void *describeImageUserData,
                       PlannedRenderTargets &targets, std::string &error) {
    targets = {};
    if (!source || source->struct_size < sizeof(VernonRenderPass))
        return fail(error, "RenderPass control is missing or incomplete");
    const VernonRenderPass &renderPass = *source;
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
        targets.attachments.push_back(&attachment);
    }
    std::sort(targets.attachments.begin(), targets.attachments.end(),
              [](const VernonColorAttachment *left, const VernonColorAttachment *right) {
                  return left->location < right->location;
              });
    if (std::adjacent_find(targets.attachments.begin(), targets.attachments.end(),
                           [](const VernonColorAttachment *left, const VernonColorAttachment *right) {
                               return left->location == right->location;
                           }) != targets.attachments.end())
        return fail(error, "render target locations are duplicated");
    for (size_t index = 0; index < targets.attachments.size(); ++index)
        if (targets.attachments[index]->location != index)
            return fail(error, "render target locations must be contiguous from zero");
    targets.attachmentFormats.reserve(targets.attachments.size());
    for (const VernonColorAttachment *attachment : targets.attachments) {
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
        if (!targets.attachmentWidth) {
            targets.attachmentWidth = width;
            targets.attachmentHeight = height;
        } else if (targets.attachmentWidth != width || targets.attachmentHeight != height) {
            return fail(error, "render target extents differ");
        }
        targets.attachmentFormats.push_back(description.view.format);
        targets.attachmentSubresources.push_back(providerImageSubresources(description));
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
        const bool firstAttachment = !targets.attachmentWidth;
        if (description.view.dimension != VERNON_TEXTURE_2D ||
            !(description.image.usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT) || !width || !height ||
            (!firstAttachment && (width != targets.attachmentWidth || height != targets.attachmentHeight)) ||
            !vernon::formats::hasDepth(format))
            return fail(error, "depth attachment must be D32 or D32S8 with the render-target extent");
        if (firstAttachment) {
            targets.attachmentWidth = width;
            targets.attachmentHeight = height;
        }
        if (attachment.load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
            attachment.store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD ||
            !std::isfinite(attachment.clear_depth) || attachment.clear_depth < 0.0f || attachment.clear_depth > 1.0f ||
            attachment.stencil_load_operation > VERNON_RUNTIME_PROVIDER_LOAD_DISCARD ||
            attachment.stencil_store_operation > VERNON_RUNTIME_PROVIDER_STORE_DISCARD ||
            attachment.clear_stencil > 0xff || (!vernon::formats::hasStencil(format) && attachment.clear_stencil != 0))
            return fail(error, "depth attachment operation is invalid");
        targets.depthAttachment = &attachment;
        targets.depthFormat = format;
        targets.attachmentSubresources.push_back(providerImageSubresources(description));
    }
    return true;
}

bool buildGraphicsDrawLayout(const StageBindingPlan &stagePlan, GraphicsDrawLayout &layout, std::string &error) {
    layout = {};
    struct Descriptor {
        uint32_t image{UINT32_MAX};
        uint32_t sampler{UINT32_MAX};
        bool implicitSampler{};
    };
    std::map<std::pair<uint32_t, uint32_t>, Descriptor> descriptors;
    const auto share = [&](uint32_t &bound, uint32_t parameter) {
        if (bound != UINT32_MAX && bound != parameter)
            layout.sharedDescriptors.emplace_back(bound, parameter);
        bound = parameter;
    };
    for (uint32_t index = 0; index < stagePlan.parameters.size(); ++index) {
        const Parameter &parameter = stagePlan.parameters[index];
        for (const ParameterUse &use : parameter.uses) {
            if (!shaderStage(use))
                continue;
            if (parameter.kind == "image") {
                if (use.binding == UINT32_MAX)
                    return fail(error, "sampled image is missing set/binding");
                share(descriptors[{use.descriptorSet, use.binding}].image, index);
                if (std::find(layout.sampledImages.begin(), layout.sampledImages.end(), index) ==
                    layout.sampledImages.end())
                    layout.sampledImages.push_back(index);
                continue;
            }
            if (parameter.kind == "sampler") {
                for (const SampledImageBinding &binding : use.sampledImageBindings)
                    share(descriptors[{binding.descriptorSet, binding.binding}].sampler, index);
                continue;
            }
            if (use.interfaceKind != "input")
                continue;
            if (parameter.kind != "tensor" || use.location == UINT32_MAX || use.attributeLeaves.empty())
                return fail(error, "graphics Tensor view is invalid");
            for (const AttributeLeaf &leaf : use.attributeLeaves)
                if (leaf.dtype != "i32" && leaf.dtype != "u32" && leaf.dtype != "f16" && leaf.dtype != "f32" &&
                    leaf.dtype != "f64")
                    return fail(error, "graphics Tensor dtype is unsupported");
            layout.vertexInputs.push_back({index, use.divisor});
        }
    }
    for (const Parameter &parameter : stagePlan.runtimeParameters) {
        if (parameter.source != StageParameterSource::ImplicitSampler)
            continue;
        for (const ParameterUse &use : parameter.uses) {
            if (!shaderStage(use))
                continue;
            for (const SampledImageBinding &binding : use.sampledImageBindings) {
                Descriptor &descriptor = descriptors[{binding.descriptorSet, binding.binding}];
                if (descriptor.sampler != UINT32_MAX)
                    return fail(error, "implicit and explicit samplers conflict");
                descriptor.implicitSampler = true;
            }
        }
    }
    for (const auto &[binding, descriptor] : descriptors)
        if (descriptor.image == UINT32_MAX || (descriptor.sampler == UINT32_MAX && !descriptor.implicitSampler))
            return fail(error, "sampled image requires paired image view and sampler");
    return true;
}

bool planGraphicsArgument(const StageBindingPlan &stagePlan, const GraphicsDrawLayout &layout, uint32_t index,
                          const VernonProgramArgument &argument, DescribeImageResource describeImage,
                          void *describeImageUserData, PlannedGraphicsArgument &planned, std::string &error) {
    planned = {};
    const Parameter &parameter = stagePlan.parameters[index];
    if (!kindMatches(parameter, argument)) {
        error = "graphics pipeline argument '" + parameter.name + "' expects " + parameter.kind +
                " but received kind " + std::to_string(argument.kind);
        return false;
    }
    if (argument.kind == VERNON_PROGRAM_IMAGE) {
        if (!argument.image.view.identity || !argument.image.view.resource.value)
            return fail(error, "pipeline image argument does not match layout");
        if (std::find(layout.sampledImages.begin(), layout.sampledImages.end(), index) == layout.sampledImages.end())
            return true;
        if (!describeImage)
            return fail(error, "graphics image descriptor resolver is missing");
        VernonRuntimeProviderImageDescription description{};
        description.struct_size = sizeof(description);
        if (describeImage(describeImageUserData, argument.image.view, &description) != VERNON_STATUS_OK ||
            !providerImageDescriptionIsCanonical(description) || !(description.image.usage & VERNON_IMAGE_SAMPLED))
            return fail(error, "sampled image is stale or lacks sampled usage");
        planned.subresources = providerImageSubresources(description);
        return true;
    }
    if (argument.kind == VERNON_PROGRAM_SAMPLER) {
        if (!argument.resource.identity || !argument.resource.resource.value)
            return fail(error, "pipeline sampler belongs to another runtime");
        return true;
    }
    const VernonTensorView &tensor = argument.tensor;
    const ValueLayout &expectedLayout =
        parameter.tensorArgument == TensorRepresentation::WholeValue ? *parameter.valueLayout : parameter.elementLayout;
    if (!valueLayoutsEqual(tensor.element_layout, pipelineValueLayout(expectedLayout)) || !validTensor(tensor))
        return fail(error, "pipeline Tensor argument does not match layout");
    if (parameter.source != StageParameterSource::Direct &&
        parameter.tensorArgument != TensorRepresentation::WholeValue) {
        const bool allowLeading = std::any_of(parameter.uses.begin(), parameter.uses.end(),
                                              [](const ParameterUse &use) { return use.interfaceKind == "input"; });
        const size_t offset = allowLeading && tensor.rank == parameter.shape.size() + 1 ? 1 : 0;
        if (tensor.rank != parameter.shape.size() + offset) {
            error = "pipeline Tensor '" + parameter.name + "' rank " + std::to_string(tensor.rank) +
                    " does not match layout rank " + std::to_string(parameter.shape.size());
            return false;
        }
        for (size_t dimension = 0; dimension < parameter.shape.size(); ++dimension)
            if (parameter.shape[dimension] && parameter.shape[dimension] != tensor.shape[dimension + offset])
                return fail(error, "pipeline Tensor shape does not match layout");
    }
    if (std::none_of(layout.vertexInputs.begin(), layout.vertexInputs.end(),
                     [&](const GraphicsDrawLayout::VertexInput &input) { return input.parameter == index; }))
        return true;
    if (tensor.storage != VERNON_TENSOR_RHI_RESOURCE || !tensor.resource.resource.value || !tensor.rank ||
        !tensor.shape || !tensor.byte_strides)
        return fail(error, "graphics Tensor view is invalid");
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension)
        if (tensor.byte_strides[dimension] <= 0)
            return fail(error, "graphics Tensor strides must be positive");
    if (tensor.shape[0] > std::numeric_limits<uint32_t>::max())
        return fail(error, "graphics Tensor leading dimension is too large");
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
    if (static_cast<uint64_t>(tensor.byte_strides[0]) > std::numeric_limits<uint32_t>::max() ||
        tensor.byte_offset >= tensor.resource.size)
        return fail(error, "graphics vertex argument is missing or invalid");
    return true;
}

GraphicsVertexRecords graphicsVertexRecords(const GraphicsDrawLayout &layout,
                                            const VernonProgramArgument *const *arguments) {
    GraphicsVertexRecords records;
    records.reserve(layout.vertexInputs.size());
    for (const GraphicsDrawLayout::VertexInput &input : layout.vertexInputs)
        records.emplace_back(static_cast<uint32_t>(arguments[input.parameter]->tensor.shape[0]), input.divisor);
    return records;
}

bool checkGraphicsSharedDescriptors(const GraphicsDrawLayout &layout, const VernonProgramArgument *const *arguments,
                                    std::string &error) {
    for (const auto &[first, second] : layout.sharedDescriptors) {
        const VernonProgramArgument &left = *arguments[first];
        const VernonProgramArgument &right = *arguments[second];
        const bool image = left.kind == VERNON_PROGRAM_IMAGE;
        if ((image ? left.image.view.resource.value : left.resource.resource.value) !=
            (image ? right.image.view.resource.value : right.resource.resource.value))
            return fail(error, image ? "sampled image binding is ambiguous" : "sampler pairing is ambiguous");
    }
    return true;
}

bool planGraphicsDraw(const GraphicsDrawLayout &layout, const VernonProgramArgument *const *arguments,
                      const PlannedRenderTargets &targets, const VernonRenderPass &renderPass,
                      const VernonDynamicState *dynamicState, const VernonDrawCommand &draw, PlannedGraphicsDraw &plan,
                      std::string &error) {
    return checkGraphicsSharedDescriptors(layout, arguments, error) &&
           planGraphicsDrawValues(graphicsVertexRecords(layout, arguments), targets, renderPass, dynamicState, draw,
                                  plan, error);
}

bool planGraphicsDrawValues(const GraphicsVertexRecords &records, const PlannedRenderTargets &targets,
                            const VernonRenderPass &renderPass, const VernonDynamicState *dynamicState,
                            const VernonDrawCommand &draw, PlannedGraphicsDraw &plan, std::string &error) {
    plan = {};
    if (draw.struct_size < sizeof(VernonDrawCommand))
        return fail(error, "DrawCommand control is missing or incomplete");
    if (dynamicState && dynamicState->struct_size < sizeof(VernonDynamicState))
        return fail(error, "DynamicState control metadata is incomplete");
    const uint32_t *dynamicViewport = dynamicState ? dynamicState->viewport : nullptr;
    const uint32_t *dynamicScissor = dynamicState ? dynamicState->scissor : nullptr;
    const bool hasViewport = dynamicViewport && dynamicViewport[2] && dynamicViewport[3];
    const uint32_t *viewport = hasViewport ? dynamicViewport : renderPass.render_area;
    plan.viewport[0] = viewport[0];
    plan.viewport[1] = viewport[1];
    plan.viewport[2] = viewport[2] ? viewport[2] : targets.attachmentWidth;
    plan.viewport[3] = viewport[3] ? viewport[3] : targets.attachmentHeight;
    const bool hasScissor = dynamicScissor && dynamicScissor[2] && dynamicScissor[3];
    for (size_t index = 0; index < 4; ++index)
        plan.scissor[index] = hasScissor ? dynamicScissor[index] : plan.viewport[index];
    plan.resolution = {static_cast<float>(plan.viewport[2]), static_cast<float>(plan.viewport[3])};
    plan.stencilReference = dynamicState ? dynamicState->stencil_reference : 0;
    if (plan.stencilReference > 0xff)
        return fail(error, "graphics state contains an invalid or unsupported value");
    plan.vertexCount = draw.vertex_count;
    plan.instanceCount = draw.instance_count;
    plan.firstInstance = draw.first_instance;
    for (const auto &[leading, divisor] : records) {
        if (!divisor) {
            if (!plan.vertexCount)
                plan.vertexCount = leading;
            else if (plan.vertexCount != leading)
                return fail(error, "graphics Tensor leading dimensions conflict");
            continue;
        }
        // A draw of no instances reads no instance records; an instance input may hold more records than a draw reads.
        if (!plan.instanceCount)
            continue;
        const uint64_t requiredRecords = (uint64_t{plan.firstInstance} + plan.instanceCount - 1) / divisor + 1;
        if (requiredRecords > leading)
            return fail(error, "graphics Tensor leading dimensions conflict");
    }
    if (!plan.vertexCount && !draw.index_binding)
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
