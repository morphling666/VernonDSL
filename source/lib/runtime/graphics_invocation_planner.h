#ifndef VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H
#define VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "graphics_variant_key.h"
#include "stage_binding_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::runtime {

using PipelineArgumentMap = std::unordered_map<uint32_t, const VernonProgramArgument *>;

enum PlannedShaderStage : uint32_t { PLANNED_STAGE_VERTEX = 1u << 0, PLANNED_STAGE_FRAGMENT = 1u << 1 };

struct PlannedSampledResource {
    VernonRuntimeProviderResourceReference imageView{};
    VernonRuntimeProviderResourceReference samplerResource{};
    bool implicitSampler{};
    bool explicitSampler{};
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
    std::vector<VernonTextureFormat> attachmentFormats;
    const VernonDepthAttachment *depthAttachment{};
    VernonTextureFormat depthFormat{};
    std::map<std::pair<uint32_t, uint32_t>, PlannedSampledResource> sampledResources;
    std::vector<PlannedVertexInput> vertexInputs;
    std::array<float, 2> resolution{};
    uint32_t attachmentWidth{};
    uint32_t attachmentHeight{};
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    const VernonIndexBinding *indexBinding{};
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    uint32_t viewport[4]{};
    uint32_t scissor[4]{};
};

using DescribeImageResource = VernonStatus (*)(void *userData, VernonRuntimeProviderResourceReference resource,
                                               VernonRuntimeProviderImageDescription *description);

struct PlannedGraphicsState {
    VernonRasterizationState rasterization{};
    VernonDepthStencilState depthStencil{};
    std::vector<VernonColorBlendState> colorBlends;
    uint32_t stencilReference{};
};

struct PreparedGraphicsVariant {
    GraphicsVariantKey key;
    VernonRuntimeCoreGraphicsVariant *handle{};
};

bool planGraphicsInvocation(const StageBindingPlan &stagePlan, const VernonStageInvocationDescriptor &invocation,
                            DescribeImageResource describeImage, void *describeImageUserData,
                            PlannedGraphicsInvocation &plan, std::string &error);
bool planGraphicsState(const VernonStageInvocationDescriptor &invocation, size_t colorCount, bool hasDepth,
                       bool hasStencil, PlannedGraphicsState &state, std::string &error);
VernonStatus ensureGraphicsVariant(VernonRuntimeCorePipeline *pipeline, const GraphicsVariantKey &key,
                                   PreparedGraphicsVariant &prepared);
void destroyGraphicsVariant(PreparedGraphicsVariant &prepared);

} // namespace vernon::runtime

#endif
