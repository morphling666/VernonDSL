#ifndef VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H
#define VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "graphics_variant_key.h"
#include "pipeline_groups.h"
#include "provider_image_description.h"
#include "resolved_stage_types.h"
#include "stage_binding_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime {

// What every draw of one graphics Stage reads, fixed when the Stage is prepared.
struct GraphicsDrawLayout {
    struct VertexInput {
        uint32_t parameter{};
        uint32_t divisor{};
    };
    std::vector<VertexInput> vertexInputs;
    // Image parameters the draws sample, by index.
    std::vector<uint32_t> sampledImages;
    // Parameters bound at one shader descriptor, whose arguments must name the same resource.
    std::vector<std::pair<uint32_t, uint32_t>> sharedDescriptors;
};

// What validating one argument of a graphics Stage found.
struct PlannedGraphicsArgument {
    // A sampled image's subresources; empty for every other argument.
    ProviderImageSubresources subresources{};
};

// `attachments` are sorted by location, and locations are contiguous from zero. `attachmentSubresources` holds the
// subresources of the color attachments in that order, followed by those of the depth attachment.
struct PlannedRenderTargets {
    std::vector<const VernonColorAttachment *> attachments;
    std::vector<VernonTextureFormat> attachmentFormats;
    std::vector<ProviderImageSubresources> attachmentSubresources;
    const VernonDepthAttachment *depthAttachment{};
    VernonTextureFormat depthFormat{};
    uint32_t attachmentWidth{};
    uint32_t attachmentHeight{};
};

struct PlannedGraphicsDraw {
    std::array<float, 2> resolution{};
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    uint32_t firstInstance{};
    uint32_t stencilReference{};
    const VernonIndexBinding *indexBinding{};
    uint32_t viewport[4]{};
    uint32_t scissor[4]{};
};

// One draw of a graphics command stream, encoded inside the stream's rendering scope.
struct GraphicsCommand {
    // Per Stage parameter.
    const VernonProgramArgument *const *arguments{};
    const PlannedRenderTargets *targets{};
    const VernonGraphicsState *graphicsState{};
    PlannedGraphicsDraw draw;
    GroupSources groups;
    VernonRuntimeProviderObject encoder{};
};

// What the commands of one stream bound last. A stream renders to targets of one format set, so the variant changes
// only with the pipeline, its state, or the vertex strides.
struct GraphicsStream {
    const void *pipeline{};
    const VernonGraphicsState *graphicsState{};
    std::vector<uint32_t> vertexStrides;
    std::vector<uint32_t> nextVertexStrides;
    // The vertex buffers of the command being encoded, by vertex binding.
    std::vector<VernonRuntimeProviderResourceReference> vertexBuffers;
};

using DescribeImageResource = VernonStatus (*)(void *userData, VernonRuntimeProviderResourceReference resource,
                                               VernonRuntimeProviderImageDescription *description);

struct PlannedGraphicsState {
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    VernonRasterizationState rasterization{};
    VernonDepthStencilState depthStencil{};
    std::vector<VernonColorBlendState> colorBlends;
};

struct PreparedGraphicsVariant {
    GraphicsVariantKey key;
    VernonRuntimeCoreGraphicsVariant *handle{};
};

// A standalone invocation: a stream of one command.
struct PlannedGraphicsInvocation {
    PlannedRenderTargets targets;
    std::vector<const VernonProgramArgument *> arguments;
    std::vector<PlannedGraphicsArgument> planned;
    PlannedGraphicsDraw draw;
};

bool buildGraphicsDrawLayout(const StageBindingPlan &stagePlan, GraphicsDrawLayout &layout, std::string &error);
bool planGraphicsArgument(const StageBindingPlan &stagePlan, const GraphicsDrawLayout &layout, uint32_t parameter,
                          const VernonProgramArgument &argument, DescribeImageResource describeImage,
                          void *describeImageUserData, PlannedGraphicsArgument &planned, std::string &error);
bool planRenderTargets(const VernonRenderPass *renderPass, DescribeImageResource describeImage,
                       void *describeImageUserData, PlannedRenderTargets &targets, std::string &error);
// `arguments` holds one validated argument per Stage parameter.
bool planGraphicsDraw(const GraphicsDrawLayout &layout, const VernonProgramArgument *const *arguments,
                      const PlannedRenderTargets &targets, const VernonRenderPass &renderPass,
                      const VernonDynamicState *dynamicState, const VernonDrawCommand &draw, PlannedGraphicsDraw &plan,
                      std::string &error);
bool planGraphicsInvocation(const StageBindingPlan &stagePlan, const GraphicsDrawLayout &layout,
                            const VernonStageInvocationDescriptor &invocation, DescribeImageResource describeImage,
                            void *describeImageUserData, PlannedGraphicsInvocation &plan, std::string &error);
bool planGraphicsState(const VernonGraphicsState *graphicsState, size_t colorCount, bool hasDepth, bool hasStencil,
                       PlannedGraphicsState &state, std::string &error);
VernonStatus ensureGraphicsVariant(VernonRuntimeCorePipeline *pipeline, const GraphicsVariantKey &key,
                                   PreparedGraphicsVariant &prepared);
void destroyGraphicsVariant(PreparedGraphicsVariant &prepared);

} // namespace vernon::runtime

#endif
