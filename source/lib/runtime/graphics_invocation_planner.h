#ifndef VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H
#define VERNON_RUNTIME_GRAPHICS_INVOCATION_PLANNER_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "graphics_variant_key.h"
#include "pipeline_groups.h"
#include "provider_image_description.h"
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

// Indexed draws from records, which replace the command's one draw; see VernonRuntimeCoreStreamDrawInvocation. A
// host stream leaves both resources zero.
struct GraphicsDrawStream {
    const VernonRuntimeProviderDrawRecord *hostRecords{};
    const uint32_t *hostCounts{};
    VernonRuntimeProviderResourceReference records{};
    VernonRuntimeProviderResourceReference counts{};
    uint32_t firstRecord{};
    uint32_t maxRecords{};
    uint32_t countIndex{};
};

using DescribeImageResource = VernonStatus (*)(void *userData, VernonRuntimeProviderResourceReference resource,
                                               VernonRuntimeProviderImageDescription *description);

struct PlannedGraphicsState {
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    VernonRasterizationState rasterization{};
    VernonDepthStencilState depthStencil{};
    std::vector<VernonColorBlendState> colorBlends;
};

bool buildGraphicsDrawLayout(const StageBindingPlan &stagePlan, GraphicsDrawLayout &layout, std::string &error);
bool planGraphicsArgument(const StageBindingPlan &stagePlan, const GraphicsDrawLayout &layout, uint32_t parameter,
                          const VernonProgramArgument &argument, DescribeImageResource describeImage,
                          void *describeImageUserData, PlannedGraphicsArgument &planned, std::string &error);
bool planRenderTargets(const VernonRenderPass *renderPass, DescribeImageResource describeImage,
                       void *describeImageUserData, PlannedRenderTargets &targets, std::string &error);
// The leading dimension and divisor of each vertex input of a draw, which its counts are checked against; divisor 0
// marks a per-vertex input.
using GraphicsVertexRecords = std::vector<std::pair<uint32_t, uint32_t>>;
// `arguments` holds one validated argument per Stage parameter.
GraphicsVertexRecords graphicsVertexRecords(const GraphicsDrawLayout &layout,
                                            const VernonProgramArgument *const *arguments);
bool checkGraphicsSharedDescriptors(const GraphicsDrawLayout &layout, const VernonProgramArgument *const *arguments,
                                    std::string &error);
bool planGraphicsDraw(const GraphicsDrawLayout &layout, const VernonProgramArgument *const *arguments,
                      const PlannedRenderTargets &targets, const VernonRenderPass &renderPass,
                      const VernonDynamicState *dynamicState, const VernonDrawCommand &draw, PlannedGraphicsDraw &plan,
                      std::string &error);
// The counts, viewport, scissor and stencil reference of a draw whose vertex inputs have `records`. A draw of no
// instances draws nothing.
bool planGraphicsDrawValues(const GraphicsVertexRecords &records, const PlannedRenderTargets &targets,
                            const VernonRenderPass &renderPass, const VernonDynamicState *dynamicState,
                            const VernonDrawCommand &draw, PlannedGraphicsDraw &plan, std::string &error);
bool planGraphicsState(const VernonGraphicsState *graphicsState, size_t colorCount, bool hasDepth, bool hasStencil,
                       PlannedGraphicsState &state, std::string &error);

} // namespace vernon::runtime

#endif
