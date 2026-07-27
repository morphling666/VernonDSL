#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"
#include "tensor_bridge.h"

#include <algorithm>
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

} // namespace

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
            if (!valueLayoutsEqual(argument.tensor.element_layout, pipelineValueLayout(parameter.elementLayout)) ||
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
            format != VERNON_TEXTURE_D32_FLOAT)
            return fail(error, "depth attachment must be D32 with the render-target extent");
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
    plan.maximumAttachmentLocation = plan.attachments.back()->location;

    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];
    plan.resolution = {static_cast<float>(hasViewport ? invocation.viewport[2] : plan.attachmentWidth),
                       static_cast<float>(hasViewport ? invocation.viewport[3] : plan.attachmentHeight)};
    plan.vertexCount = invocation.vertex_count;
    plan.instanceCount = invocation.instance_count;

    struct SampledResourceDraft {
        VernonRuntimeProviderResourceReference imageResource{};
        VernonRuntimeProviderResourceReference textureSamplerResource{};
        VernonRuntimeProviderResourceReference explicitSamplerResource{};
        bool implicitSampler{};
        uint32_t stages{};
    };
    std::map<std::pair<uint32_t, uint32_t>, SampledResourceDraft> sampled;
    for (const Parameter &parameter : variant.parameters) {
        const VernonPipelineArgument &argument = *plan.arguments.at(parameter.slot);
        for (const ParameterUse &use : parameter.uses) {
            const uint32_t stage = shaderStage(use);
            if (!stage)
                continue;
            if (argument.kind == VERNON_PIPELINE_TEXTURE) {
                if (use.binding == UINT32_MAX)
                    return fail(error, "sampled texture is missing set/binding");
                SampledResourceDraft &resource = sampled[{use.descriptorSet, use.binding}];
                if (resource.imageResource.resource.value &&
                    resource.imageResource.resource.value != argument.texture.resource.resource.value)
                    return fail(error, "sampled texture binding is ambiguous");
                if (resource.imageResource.resource.value &&
                    resource.textureSamplerResource.resource.value != argument.texture.sampler_resource.resource.value)
                    return fail(error, "texture binding has conflicting sampler policies");
                resource.imageResource = argument.texture.resource;
                resource.textureSamplerResource = argument.texture.sampler_resource;
                resource.stages |= stage;
                continue;
            }
            if (argument.kind == VERNON_PIPELINE_SAMPLER) {
                if (use.sampledTextureBindings.empty())
                    return fail(error, "sampler reflection has no paired sampled texture "
                                       "binding");
                for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                    SampledResourceDraft &resource = sampled[{binding.descriptorSet, binding.binding}];
                    if (resource.explicitSamplerResource.resource.value &&
                        resource.explicitSamplerResource.resource.value != argument.resource.resource.value)
                        return fail(error, "sampler pairing is ambiguous");
                    resource.explicitSamplerResource = argument.resource;
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
                SampledResourceDraft &resource = sampled[{binding.descriptorSet, binding.binding}];
                if (resource.explicitSamplerResource.resource.value)
                    return fail(error, "implicit and explicit samplers conflict");
                resource.implicitSampler = true;
                resource.stages |= stage;
            }
        }
    }
    for (const auto &[binding, resource] : sampled) {
        if (!resource.imageResource.resource.value ||
            (!resource.explicitSamplerResource.resource.value && !resource.textureSamplerResource.resource.value &&
             !resource.implicitSampler))
            return fail(error, "sampled image requires paired texture and sampler");
        plan.sampledResources.emplace(binding, PlannedSampledResource{resource.imageResource,
                                                                      resource.explicitSamplerResource.resource.value
                                                                          ? resource.explicitSamplerResource
                                                                          : resource.textureSamplerResource,
                                                                      resource.implicitSampler, resource.stages});
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
