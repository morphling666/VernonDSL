#ifndef VERNON_EXECUTION_GRAPH_INTERNAL_H
#define VERNON_EXECUTION_GRAPH_INTERNAL_H

#include "VernonExecutionGraph.h"
#include "execution_command_model.h"
#include "rhi/logical_resource_record.h"

#include <memory>
#include <mutex>
#include <string>

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
    detail::CommandDag commandDag;
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
using RhiCommandListEncoder = VernonRhiStatus (*)(void *context, VernonRhiCommandEncoder encoder);
using RhiCommandCompletionCallback = VernonRhiStatus (*)(void *context);

struct RhiCommandNodeEncoder {
    RhiCommandListEncoder encode{};
    void *context{};
    RhiCommandCompletionCallback complete{};
};

struct RhiCommandResourceBinding {
    uint64_t aliasDomain{};
    ResourceKind kind{ResourceKind::Buffer};
    VernonRhiBuffer buffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiImage image{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
};

inline CommandResourceAccess rhiBufferAccess(VernonRhiBuffer buffer, uint64_t offset, uint64_t size, AccessMode access,
                                             VernonRhiResourceState state, uint32_t stageMask = 0) {
    CommandResourceAccess result;
    result.resource = buffer.index;
    result.aliasDomain = rhi::encodeResourceKey(buffer);
    result.access = access;
    result.kind = ResourceKind::Buffer;
    result.bufferRange = {offset, size};
    result.state = state;
    result.stageMask = stageMask;
    return result;
}

inline void appendRhiBufferBinding(std::vector<RhiCommandResourceBinding> &bindings, VernonRhiBuffer buffer) {
    const uint64_t identity = rhi::encodeResourceKey(buffer);
    for (const RhiCommandResourceBinding &binding : bindings)
        if (binding.aliasDomain == identity && binding.kind == ResourceKind::Buffer)
            return;
    bindings.push_back({identity, ResourceKind::Buffer, buffer, {}});
}

struct RhiCommandDagExecutionStats {
    uint64_t submissions{};
    uint64_t waits{};
    uint64_t deviceWaitNanoseconds{};
};

struct RhiCommandExecutionPlan {
    CommandDag commands;
    std::vector<RhiCommandNodeEncoder> encoders;
    std::vector<RhiCommandResourceBinding> bindings;
    std::vector<std::shared_ptr<void>> retainedContexts;
    std::vector<CommandResourceAccess> initialAccesses;
};

class RhiCommandCompletion {
public:
    virtual ~RhiCommandCompletion() = default;
    virtual bool validationPhase() const { return false; }
    virtual void complete(bool succeeded, const RhiCommandDagExecutionStats &stats) = 0;
};

class RhiCommandPlanSink {
public:
    virtual ~RhiCommandPlanSink() = default;
    virtual VernonRhiStatus append(RhiCommandExecutionPlan plan) = 0;
    virtual void retain(std::shared_ptr<void> context) = 0;
    virtual void onCompletion(std::shared_ptr<RhiCommandCompletion> completion) = 0;
    virtual VernonRhiStatus flush() = 0;
};

using RhiCommandPlanExtension = VernonRhiStatus (*)(void *context, uint32_t offset, uint32_t predecessor,
                                                    RhiCommandExecutionPlan &plan);
using RhiPassCommandPlanner = VernonRhiStatus (*)(void *context, uint32_t offset, ComputePass &pass,
                                                  const ExecutionResources &resources, RhiCommandExecutionPlan &plan);

bool appendRhiCommandExecutionPlan(RhiCommandExecutionPlan &destination, RhiCommandExecutionPlan source, bool serialize,
                                   std::string &error);
bool validateRhiCommandExecutionPlan(const RhiCommandExecutionPlan &plan, std::string &error);

VernonRhiStatus submitRhiCommandDag(VernonRhiDevice device, uint32_t requiredCapabilities, const CommandDag &dag,
                                    const std::vector<RhiCommandNodeEncoder> &encoders,
                                    VernonRhiCompletion *outputCompletion,
                                    const std::vector<RhiCommandResourceBinding> *resourceBindings = nullptr);
VernonRhiStatus executeRhiCommandDagAndWait(VernonRhiDevice device, uint32_t requiredCapabilities,
                                            const CommandDag &dag, const std::vector<RhiCommandNodeEncoder> &encoders,
                                            RhiCommandDagExecutionStats *stats = nullptr,
                                            const std::vector<RhiCommandResourceBinding> *resourceBindings = nullptr,
                                            const std::vector<CommandResourceAccess> *initialAccesses = nullptr);
VernonRhiStatus executeRhiCommandPlanAndWait(VernonRhiDevice device, uint32_t requiredCapabilities,
                                             const RhiCommandExecutionPlan &plan,
                                             RhiCommandDagExecutionStats *stats = nullptr);
VernonRhiStatus submitRhiCommandList(VernonRhiDevice device, uint32_t requiredCapabilities,
                                     RhiCommandListEncoder encode, void *context,
                                     VernonRhiCompletion *outputCompletion);

VernonRhiStatus executeCpuScheduleRange(VernonRhiDevice device, const std::vector<GraphResource> &resources,
                                        const std::vector<std::unique_ptr<ExecutionPass>> &passes,
                                        const std::vector<uint32_t> &schedule,
                                        std::shared_ptr<const ExecutionBindings> bindings, uint32_t beginOffset,
                                        uint32_t endOffset, CpuPassExecutor executor = nullptr, void *context = nullptr,
                                        const std::vector<VernonRhiBuffer> *resourceBuffers = nullptr);

VernonRhiStatus submitRhiComputeCommandRange(VernonRhiDevice device, const std::vector<GraphResource> &resources,
                                             const std::vector<ExecutionResourceRecord> &resourceRecords,
                                             const std::vector<std::unique_ptr<ExecutionPass>> &passes,
                                             const std::vector<uint32_t> &schedule,
                                             const std::vector<CompiledScope> &scopes, const CommandDag &commandDag,
                                             std::shared_ptr<const ExecutionBindings> bindings, uint32_t beginOffset,
                                             uint32_t endOffset, CpuPassExecutor executor = nullptr,
                                             void *context = nullptr, VernonRhiCompletion *outputCompletion = nullptr,
                                             RhiCommandPlanExtension extension = nullptr,
                                             void *extensionContext = nullptr, RhiPassCommandPlanner planner = nullptr,
                                             void *plannerContext = nullptr,
                                             RhiCommandExecutionPlan *outputPlan = nullptr);

} // namespace detail

} // namespace vernon::execution

#endif
