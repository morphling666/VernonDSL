#ifndef VERNON_TESTS_SUPPORT_RESOLVED_STAGE_TEST_UTILS_H
#define VERNON_TESTS_SUPPORT_RESOLVED_STAGE_TEST_UTILS_H

#include "runtime/resolved_stage_invocation.h"

#include <vector>

namespace vernon::tests {

struct GraphicsInvocationControls {
    VernonGraphicsState state{};
    VernonRenderPass renderPass{};
    VernonDrawCommand draw{};
    VernonDynamicState dynamic{};
    std::vector<VernonColorBlendState> blends;

    GraphicsInvocationControls(const VernonColorAttachment *colors, size_t colorCount, uint32_t vertexCount = 0,
                               uint32_t instanceCount = 1,
                               VernonPrimitiveTopology topology = VERNON_TOPOLOGY_TRIANGLE_LIST) {
        state.struct_size = sizeof(state);
        state.topology = topology;
        blends.resize(colorCount);
        for (VernonColorBlendState &blend : blends)
            blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
        state.color_blends = blends.data();
        state.color_blend_count = blends.size();
        renderPass.struct_size = sizeof(renderPass);
        renderPass.color_attachments = colors;
        renderPass.color_attachment_count = colorCount;
        draw.struct_size = sizeof(draw);
        draw.vertex_count = vertexCount;
        draw.instance_count = instanceCount;
        dynamic.struct_size = sizeof(dynamic);
    }

    void bind(VernonStageInvocationDescriptor &invocation) {
        invocation.graphics_state = &state;
        invocation.render_pass = &renderPass;
        invocation.draw_command = &draw;
        invocation.dynamic_state = &dynamic;
    }
};

inline VernonStatus completeResolvedStageSubmission(VernonStageExecutable *stage,
                                                    const VernonStageInvocationDescriptor *invocation) {
    VernonSubmission *submission{};
    const VernonStatus submitStatus = runtime::submitResolvedStage(stage, invocation, &submission);
    if (submitStatus != VERNON_STATUS_OK)
        return submitStatus;
    const VernonStatus completionStatus = vernonSubmissionWait(submission);
    vernonSubmissionDestroy(submission);
    return completionStatus;
}

} // namespace vernon::tests

#endif
