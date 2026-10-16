#ifndef VERNON_EXECUTION_GRAPH_INTERNAL_H
#define VERNON_EXECUTION_GRAPH_INTERNAL_H

#include "execution_command_model.h"
#include "execution_graph/command_graph.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vernon::execution {

struct CompiledCommandGraph::State {
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

    ~State();
};

class ExecutionSubmission::Impl {
public:
    Impl(std::shared_ptr<CompiledCommandGraph::State> retainedPlan,
         std::shared_ptr<const ExecutionBindings> retainedBindings)
        : plan(std::move(retainedPlan)), bindings(std::move(retainedBindings)) {}
    ~Impl();

    std::shared_ptr<CompiledCommandGraph::State> plan;
    std::shared_ptr<const ExecutionBindings> bindings;
    mutable std::mutex stateMutex;
    State state{State::Pending};
    VernonRhiStatus status{VERNON_RHI_STATUS_OK};
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiCommandEncoderStats stats{};
};

namespace detail {

inline void releaseResourceLease(rhi::RetainedRhiResourceLease &lease) noexcept {
    if (!lease.active())
        return;
    if (lease.release().isErr())
        resultContractViolation();
}

inline void releaseExecutionResourceRecords(ExecutionProvider provider, VernonRhiDevice device,
                                            std::vector<ExecutionResourceRecord> &records) noexcept {
    if (provider != ExecutionProvider::Rhi)
        return;
    for (auto record = records.rbegin(); record != records.rend(); ++record) {
        if (record->graphOwned && vernonRhiDeviceDestroyBuffer(device, record->buffer) != VERNON_RHI_STATUS_OK)
            resultContractViolation();
        for (auto viewLease = record->imageViewLeases.rbegin(); viewLease != record->imageViewLeases.rend();
             ++viewLease)
            releaseResourceLease(*viewLease);
        if (record->resourceLease)
            releaseResourceLease(*record->resourceLease);
    }
}

using CpuPassExecutor = VernonRhiStatus (*)(void *context, uint32_t scheduleOffset, ComputePass &pass,
                                            ComputeEncoder &encoder, const ExecutionResources &resources);
using RhiCommandListEncoder = VernonRhiStatus (*)(void *context, VernonRhiCommandEncoder encoder);

// One Program Node, or the contiguous Nodes fused into one render scope. The label and Node count grow until
// submission as Nodes fuse.
struct CommandScope {
    std::string label;
    uint32_t firstNode{};
    uint32_t nodeCount{1};
};

using TimedCommandScopes = std::vector<std::shared_ptr<const CommandScope>>;

struct RhiCommandNodeEncoder {
    RhiCommandListEncoder encode{};
    void *context{};
    // Consecutive encoders sharing one scope object are recorded inside one debug group with its label, barriers
    // included, and timed as one scope.
    std::shared_ptr<const CommandScope> scope;
};

struct RhiCommandResourceBinding {
    uint64_t aliasDomain{};
    ResourceKind kind{ResourceKind::Buffer};
    VernonRhiBuffer buffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiImage image{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
};

inline CommandResourceAccess rhiBufferAccess(VernonRhiBuffer buffer, uint64_t offset, uint64_t size, AccessMode access,
                                             VernonRhiResourceState state, uint32_t stageMask = 0,
                                             uint32_t rhiAccessMask = 0) {
    CommandResourceAccess result;
    result.resource = buffer.index;
    result.aliasDomain = rhi::encodeResourceKey(buffer);
    result.access = access;
    result.kind = ResourceKind::Buffer;
    result.bufferRange = {offset, size};
    result.state = state;
    result.stageMask = stageMask;
    result.rhiAccessMask = rhiAccessMask;
    return result;
}

inline void appendRhiBufferBinding(std::vector<RhiCommandResourceBinding> &bindings, VernonRhiBuffer buffer) {
    const uint64_t identity = rhi::encodeResourceKey(buffer);
    for (const RhiCommandResourceBinding &binding : bindings)
        if (binding.aliasDomain == identity && binding.kind == ResourceKind::Buffer)
            return;
    bindings.push_back({identity, ResourceKind::Buffer, buffer, {}});
}

inline CommandResourceAccess rhiImageAccess(VernonRhiImage image, VernonRhiImageSubresourceRange subresources,
                                            AccessMode access, VernonRhiResourceState state, uint32_t stageMask = 0) {
    CommandResourceAccess result;
    result.resource = image.index;
    result.aliasDomain = rhi::encodeResourceKey(image);
    result.access = access;
    result.kind = ResourceKind::Image;
    result.imageSubresources = subresources;
    result.state = state;
    result.stageMask = stageMask;
    return result;
}

inline void appendRhiImageBinding(std::vector<RhiCommandResourceBinding> &bindings, VernonRhiImage image) {
    const uint64_t identity = rhi::encodeResourceKey(image);
    for (const RhiCommandResourceBinding &binding : bindings)
        if (binding.aliasDomain == identity && binding.kind == ResourceKind::Image)
            return;
    bindings.push_back({identity, ResourceKind::Image, {}, image});
}

struct RhiCommandExecutionPlan {
    CommandDag commands;
    std::vector<RhiCommandNodeEncoder> encoders;
    std::vector<RhiCommandResourceBinding> bindings;
    std::vector<std::shared_ptr<void>> retainedContexts;
    std::vector<CommandResourceAccess> initialAccesses;
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
// With `timedScopes`, each run of encoders sharing a scope records one timer, and the scopes are listed in timer
// order.
VernonRhiStatus submitRhiCommandPlan(VernonRhiDevice device, uint32_t requiredCapabilities,
                                     const RhiCommandExecutionPlan &plan, VernonRhiCompletion &outputCompletion,
                                     TimedCommandScopes *timedScopes = nullptr);
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
