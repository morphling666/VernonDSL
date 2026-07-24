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

bool validTensor(const VernonTensorView &tensor, const void *expectedContext,
                 const GraphicsPlannerCallbacks &callbacks) {
    if (tensor.struct_size < sizeof(VernonTensorView) || tensor.access > VERNON_ACCESS_READ_WRITE ||
        (tensor.storage != VERNON_TENSOR_HOST && tensor.storage != VERNON_TENSOR_DEVICE))
        return false;
    if (!tensorFitsAllocation(tensor))
        return false;
    if (tensor.storage == VERNON_TENSOR_HOST)
        return tensor.host_data != nullptr;
    if (!tensor.buffer || !callbacks.bufferSnapshot)
        return false;
    const GraphicsResourceSnapshot snapshot = callbacks.bufferSnapshot(callbacks.userData, tensor.buffer);
    return snapshot.context == expectedContext && tensor.byte_size <= snapshot.bufferSize;
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
                            const void *expectedContext, const GraphicsPlannerCallbacks &callbacks,
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
            const std::optional<VernonDataType> dtype = pipelineDataType(parameter.dtype);
            if (!dtype || argument.tensor.dtype != *dtype || !validTensor(argument.tensor, expectedContext, callbacks))
                return fail(error, "pipeline Tensor argument does not match layout");
            const bool allowLeading =
                std::any_of(parameter.uses.begin(), parameter.uses.end(), [](const ParameterUse &use) {
                    return use.interfaceKind == "input" || use.interfaceKind == "instance";
                });
            const size_t offset = allowLeading && argument.tensor.rank == parameter.shape.size() + 1 ? 1 : 0;
            if (argument.tensor.rank != parameter.shape.size() + offset)
                return fail(error, "pipeline Tensor rank does not match layout");
            for (size_t dimension = 0; dimension < parameter.shape.size(); ++dimension)
                if (parameter.shape[dimension] &&
                    parameter.shape[dimension] != argument.tensor.shape[dimension + offset])
                    return fail(error, "pipeline Tensor shape does not match layout");
        } else if (argument.kind == VERNON_PIPELINE_TEXTURE) {
            if (!argument.texture.texture || !callbacks.textureSnapshot)
                return fail(error, "pipeline texture argument does not match layout");
            const GraphicsResourceSnapshot snapshot =
                callbacks.textureSnapshot(callbacks.userData, argument.texture.texture);
            if (snapshot.context != expectedContext || argument.texture.format != snapshot.textureFormat ||
                argument.texture.dimension != snapshot.textureDimension ||
                argument.texture.width != snapshot.textureWidth || argument.texture.height != snapshot.textureHeight ||
                argument.texture.depth != snapshot.textureDepth ||
                (argument.texture.sampler &&
                 (!callbacks.samplerContext ||
                  callbacks.samplerContext(callbacks.userData, argument.texture.sampler) != expectedContext)) ||
                (!parameter.dimension.empty() &&
                 ((parameter.dimension == "2d" && argument.texture.dimension != VERNON_TEXTURE_2D) ||
                  (parameter.dimension == "3d" && argument.texture.dimension != VERNON_TEXTURE_3D) ||
                  (parameter.dimension == "cube" && argument.texture.dimension != VERNON_TEXTURE_CUBE))))
                return fail(error, "pipeline texture argument does not match layout");
        } else if (!argument.sampler || !callbacks.samplerContext ||
                   callbacks.samplerContext(callbacks.userData, argument.sampler) != expectedContext) {
            return fail(error, "pipeline sampler belongs to another runtime");
        }
    }

    if (variant.vertex.empty())
        return true;

    if (!invocation.color_attachment_count || !invocation.color_attachments)
        return fail(error, "graphics pipeline requires color attachments");
    for (size_t index = 0; index < invocation.color_attachment_count; ++index) {
        const VernonColorAttachment &attachment = invocation.color_attachments[index];
        if (!attachment.texture || !callbacks.textureSnapshot)
            return fail(error, "render target is invalid");
        const GraphicsResourceSnapshot snapshot = callbacks.textureSnapshot(callbacks.userData, attachment.texture);
        if (snapshot.context != expectedContext || snapshot.textureDimension != VERNON_TEXTURE_2D)
            return fail(error, "render target is invalid");
        if (!plan.attachmentWidth) {
            plan.attachmentWidth = snapshot.textureWidth;
            plan.attachmentHeight = snapshot.textureHeight;
        } else if (plan.attachmentWidth != snapshot.textureWidth || plan.attachmentHeight != snapshot.textureHeight) {
            return fail(error, "render target extents differ");
        }
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
    plan.maximumAttachmentLocation = plan.attachments.back()->location;

    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];
    plan.resolution = {static_cast<float>(hasViewport ? invocation.viewport[2] : plan.attachmentWidth),
                       static_cast<float>(hasViewport ? invocation.viewport[3] : plan.attachmentHeight)};
    plan.vertexCount = invocation.vertex_count;
    plan.instanceCount = invocation.instance_count;

    struct SampledResourceDraft {
        VernonDeviceTexture *texture{};
        VernonDeviceSampler *textureSampler{};
        VernonDeviceSampler *explicitSampler{};
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
                if (resource.texture && resource.texture != argument.texture.texture)
                    return fail(error, "sampled texture binding is ambiguous");
                if (resource.texture && resource.textureSampler != argument.texture.sampler)
                    return fail(error, "texture binding has conflicting sampler policies");
                resource.texture = argument.texture.texture;
                resource.textureSampler = argument.texture.sampler;
                resource.stages |= stage;
                continue;
            }
            if (argument.kind == VERNON_PIPELINE_SAMPLER) {
                if (use.sampledTextureBindings.empty())
                    return fail(error, "sampler reflection has no paired sampled texture "
                                       "binding");
                for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                    SampledResourceDraft &resource = sampled[{binding.descriptorSet, binding.binding}];
                    if (resource.explicitSampler && resource.explicitSampler != argument.sampler)
                        return fail(error, "sampler pairing is ambiguous");
                    resource.explicitSampler = argument.sampler;
                    resource.stages |= stage;
                }
                continue;
            }
            if (use.interfaceKind != "input" && use.interfaceKind != "instance")
                continue;
            const VernonTensorView &tensor = argument.tensor;
            if (argument.kind != VERNON_PIPELINE_TENSOR || tensor.storage != VERNON_TENSOR_DEVICE ||
                tensor.dtype != VERNON_DATA_F32 || !tensor.buffer || !tensor.rank || !tensor.shape ||
                !tensor.byte_strides || use.location == UINT32_MAX)
                return fail(error, "graphics Tensor view is invalid");
            for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension)
                if (tensor.byte_strides[dimension] <= 0)
                    return fail(error, "graphics Tensor strides must be positive");
            uint64_t components = 1;
            for (uint32_t dimension = 1; dimension < tensor.rank; ++dimension) {
                if (tensor.shape[dimension] > std::numeric_limits<uint32_t>::max() / components)
                    return fail(error, "graphics Tensor component shape is unsupported");
                components *= tensor.shape[dimension];
            }
            if (tensor.shape[0] > std::numeric_limits<uint32_t>::max())
                return fail(error, "graphics Tensor leading dimension is too large");
            const uint32_t leading = static_cast<uint32_t>(tensor.shape[0]);
            const bool instanced = use.interfaceKind == "instance" || use.divisor != 0;
            uint32_t &inferred = instanced ? plan.instanceCount : plan.vertexCount;
            if (inferred && inferred != leading)
                return fail(error, "graphics Tensor leading dimensions conflict");
            inferred = leading;
            if (components <= 4) {
                if (tensor.rank > 1 && tensor.byte_strides[tensor.rank - 1] != sizeof(float))
                    return fail(error, "graphics Tensor components must be contiguous");
            } else if (tensor.rank == 3 && tensor.shape[2] <= 4) {
                if (tensor.byte_strides[2] != sizeof(float))
                    return fail(error, "graphics matrix rows must be contiguous");
            } else {
                return fail(error, "graphics Tensor component shape is unsupported");
            }
            plan.vertexInputs.push_back({&use, &tensor, static_cast<uint32_t>(components), instanced});
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
                if (resource.explicitSampler)
                    return fail(error, "implicit and explicit samplers conflict");
                resource.implicitSampler = true;
                resource.stages |= stage;
            }
        }
    }
    for (const auto &[binding, resource] : sampled) {
        if (!resource.texture || (!resource.explicitSampler && !resource.implicitSampler))
            return fail(error, "sampled image requires paired texture and sampler");
        plan.sampledResources.emplace(
            binding,
            PlannedSampledResource{resource.texture,
                                   resource.explicitSampler ? resource.explicitSampler : resource.textureSampler,
                                   resource.implicitSampler, resource.stages});
    }

    if (!plan.instanceCount)
        plan.instanceCount = 1;
    if (!plan.vertexCount)
        return fail(error, "graphics draw counts cannot be inferred");
    if (invocation.index_binding) {
        const VernonIndexBinding &index = *invocation.index_binding;
        if (!index.buffer || !callbacks.bufferSnapshot ||
            callbacks.bufferSnapshot(callbacks.userData, index.buffer).context != expectedContext ||
            index.type != VERNON_INDEX_U32 || !index.index_count)
            return fail(error, "index binding is invalid");
        plan.indexBinding = &index;
    }
    return true;
}

} // namespace vernon::runtime
