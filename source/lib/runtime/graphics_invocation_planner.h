#ifndef VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H
#define VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H

#include "VernonRuntime.h"
#include "pipeline_manifest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime {

struct GraphicsResourceSnapshot {
    const void *context{};
    size_t bufferSize{};
    VernonTextureDimension textureDimension{VERNON_TEXTURE_2D};
    VernonTextureFormat textureFormat{VERNON_TEXTURE_RGBA8_UNORM};
    uint32_t textureWidth{};
    uint32_t textureHeight{};
    uint32_t textureDepth{};
};

struct GraphicsPlannerCallbacks {
    const void *userData{};
    GraphicsResourceSnapshot (*bufferSnapshot)(const void *userData, const VernonDeviceBuffer *buffer){};
    GraphicsResourceSnapshot (*textureSnapshot)(const void *userData, const VernonDeviceTexture *texture){};
    const void *(*samplerContext)(const void *userData, const VernonDeviceSampler *sampler){};
};

using PipelineArgumentMap = std::unordered_map<uint32_t, const VernonPipelineArgument *>;

enum PlannedShaderStage : uint32_t { PLANNED_STAGE_VERTEX = 1u << 0, PLANNED_STAGE_FRAGMENT = 1u << 1 };

struct PlannedSampledResource {
    VernonDeviceTexture *texture{};
    VernonDeviceSampler *sampler{};
    VernonRuntimeProviderResourceReference imageResource{};
    VernonRuntimeProviderResourceReference samplerResource{};
    bool implicitSampler{};
    uint32_t stages{};
};

struct PlannedVertexInput {
    const ParameterUse *use{};
    const VernonTensorView *tensor{};
    bool instanced{};
};

struct PlannedGraphicsInvocation {
    PipelineArgumentMap arguments;
    std::vector<const VernonColorAttachment *> attachments;
    const VernonDepthAttachment *depthAttachment{};
    std::map<std::pair<uint32_t, uint32_t>, PlannedSampledResource> sampledResources;
    std::vector<PlannedVertexInput> vertexInputs;
    std::array<float, 2> resolution{};
    uint32_t attachmentWidth{};
    uint32_t attachmentHeight{};
    uint32_t maximumAttachmentLocation{};
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    const VernonIndexBinding *indexBinding{};
};

bool planGraphicsInvocation(const Variant &variant, const VernonPipelineInvocation &invocation,
                            const void *expectedContext, const GraphicsPlannerCallbacks &callbacks,
                            PlannedGraphicsInvocation &plan, std::string &error);

} // namespace vernon::runtime

#endif
