#ifndef VERNON_EXECUTION_GRAPH_INTERNAL_H
#define VERNON_EXECUTION_GRAPH_INTERNAL_H

#include "VernonExecutionGraph.h"

#include <mutex>

namespace vernon::execution {

struct CompiledExecutionGraph::State {
    detail::ExecutionProvider provider{detail::ExecutionProvider::Cpu};
    VernonRhiDevice device{};
    uint64_t graphIdentity{};
    std::vector<std::unique_ptr<ExecutionPass>> passes;
    std::vector<GraphResource> resources;
    std::vector<detail::ExecutionResourceRecord> resourceRecords;
    std::vector<std::string> parameterNames;
    std::vector<uint32_t> schedule;
    std::vector<CompiledScope> scopes;
    AutodiffDagCheckpointPlan autodiffCheckpointPlan;
    std::vector<uint32_t> autodiffInitialResources;
    std::vector<std::vector<GraphByteRange>> autodiffInitialRanges;
    std::vector<uint32_t> autodiffTransactionResources;
    std::vector<std::vector<GraphByteRange>> autodiffTransactionRanges;
    std::vector<uint32_t> autodiffRestorationResources;
    std::vector<std::vector<GraphByteRange>> autodiffRestorationRanges;
    bool hasAutodiffCheckpointPlan{};
    std::vector<NamedDerivativeEndpoint> differentiableInputs;
    std::vector<NamedDerivativeEndpoint> objectives;

    ~State();
};

class ExecutionSubmission::Impl {
public:
    Impl(std::shared_ptr<CompiledExecutionGraph::State> retainedPlan,
         std::shared_ptr<const ExecutionBindings> retainedBindings)
        : plan(std::move(retainedPlan)), bindings(std::move(retainedBindings)) {}
    ~Impl();

    std::shared_ptr<CompiledExecutionGraph::State> plan;
    std::shared_ptr<const ExecutionBindings> bindings;
    mutable std::mutex stateMutex;
    State state{State::Pending};
    VernonRhiStatus status{VERNON_RHI_STATUS_OK};
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiCommandEncoderStats stats{};
};

namespace detail {

using CpuPassExecutor = VernonRhiStatus (*)(void *context, uint32_t scheduleOffset, ComputePass &pass,
                                            ComputeEncoder &encoder, const ExecutionResources &resources);

VernonRhiStatus executeCpuScheduleRange(VernonRhiDevice device, const std::vector<GraphResource> &resources,
                                        const std::vector<std::unique_ptr<ExecutionPass>> &passes,
                                        const std::vector<uint32_t> &schedule,
                                        std::shared_ptr<const ExecutionBindings> bindings, uint32_t beginOffset,
                                        uint32_t endOffset, CpuPassExecutor executor = nullptr,
                                        void *context = nullptr);

} // namespace detail

} // namespace vernon::execution

#endif
