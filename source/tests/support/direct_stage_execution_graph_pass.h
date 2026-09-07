#ifndef VERNON_TESTS_DIRECT_STAGE_EXECUTION_GRAPH_PASS_H
#define VERNON_TESTS_DIRECT_STAGE_EXECUTION_GRAPH_PASS_H

#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"

#include <string>

namespace vernon::tests {

class DirectStageGraphRenderPass final : public execution::RenderPass {
public:
    DirectStageGraphRenderPass(std::string name, execution::GraphImage target, VernonRuntimeContext *runtime,
                               VernonStageExecutable *stage, const VernonStageInvocationDescriptor *invocation,
                               VernonRhiLoadOperation load, VernonRhiStoreOperation store = VERNON_RHI_STORE_PRESERVE)
        : RenderPass(std::move(name)), target_(target), runtime_(runtime), stage_(stage), invocation_(invocation),
          load_(load), store_(store) {}

    void declare() override {
        execution::ColorAttachmentUse attachment{};
        attachment.image = target_;
        attachment.load = load_;
        attachment.store = store_;
        color(0, attachment);
        renderArea(0, 0, target_.width, target_.height);
    }

    VernonRhiStatus execute(execution::GraphicsEncoder &encoder, const execution::ExecutionResources &) override {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(runtime_, encoder.native(), &providerEncoder) != VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        return vernonRuntimeStageEncode(providerEncoder, stage_, invocation_) == VERNON_STATUS_OK
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    execution::GraphImage target_;
    VernonRuntimeContext *runtime_{};
    VernonStageExecutable *stage_{};
    const VernonStageInvocationDescriptor *invocation_{};
    VernonRhiLoadOperation load_{};
    VernonRhiStoreOperation store_{};
};

} // namespace vernon::tests

#endif
