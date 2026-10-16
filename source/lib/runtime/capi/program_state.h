#ifndef VERNON_RUNTIME_CAPI_PROGRAM_STATE_H
#define VERNON_RUNTIME_CAPI_PROGRAM_STATE_H

#include "VernonRuntime.h"
#include "runtime/autodiff/execution_workspace_pool.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/program_execution/submission_record.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/program_graph_linker.h"
#include "runtime/program_instance.h"
#include "runtime/program_invocation_context.h"
#include "runtime/recorded_graphics.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_lifecycle.h"
#include "runtime/runtime_state.h"
#include "runtime/stage_binding_plan.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Internal state of the Program C API handles that VernonRuntime.h leaves opaque.

struct RuntimeProgramBindingMetadata {
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    std::string layoutHash;
    std::vector<VernonValueLeafView> leaves;
};

struct RuntimeProgramBinding {
    VernonProgramArgument argument{};
    std::shared_ptr<const RuntimeProgramBindingMetadata> metadata;
    std::shared_ptr<void> lease;

    void refresh() {
        if (argument.kind != VERNON_PROGRAM_TENSOR)
            return;
        argument.tensor.shape = metadata->shape.empty() ? nullptr : metadata->shape.data();
        argument.tensor.byte_strides = metadata->strides.empty() ? nullptr : metadata->strides.data();
        argument.tensor.element_layout.layout_hash = {metadata->layoutHash.data(), metadata->layoutHash.size()};
        argument.tensor.element_layout.leaves = metadata->leaves.empty() ? nullptr : metadata->leaves.data();
    }
};

struct RuntimeProgramControl {
    enum Kind : uint32_t { RenderPass = 0, DrawCommand = 1, DynamicState = 2 };
    Kind kind{RenderPass};
    VernonRenderPass renderPass{};
    VernonDrawCommand draw{};
    VernonDynamicState dynamic{};
    std::vector<VernonColorAttachment> colors;
    std::optional<VernonDepthAttachment> depth;
    std::optional<VernonIndexBinding> index;
    std::vector<std::shared_ptr<void>> leases;

    void refresh() {
        renderPass.color_attachments = colors.empty() ? nullptr : colors.data();
        renderPass.color_attachment_count = colors.size();
        renderPass.depth_attachment = depth ? &*depth : nullptr;
        draw.index_binding = index ? &*index : nullptr;
    }
};

// Keeps an executable alive the way a Program instance does: its destroy fails while the lease exists.
class RuntimeExecutableLease {
public:
    using AcquireResult = vernon::Result<std::unique_ptr<RuntimeExecutableLease>, vernon::RuntimeError>;

    static AcquireResult acquire(VernonProgramExecutable &executable) {
        auto pin = executable.lifecycle.pin();
        if (pin.isErr())
            return AcquireResult{vernon::err(std::move(pin).error())};
        auto child = vernon::runtime::RuntimeChildLifecycle::reserve(executable.instanceOwner);
        if (child.isErr())
            return AcquireResult{vernon::err(std::move(child).error())};
        std::unique_ptr<RuntimeExecutableLease> lease(new RuntimeExecutableLease(executable, std::move(child).value()));
        auto published = lease->lifecycle_.publish();
        if (published.isErr())
            return AcquireResult{vernon::err(std::move(published).error())};
        return AcquireResult{vernon::ok(std::move(lease))};
    }
    RuntimeExecutableLease(const RuntimeExecutableLease &) = delete;
    RuntimeExecutableLease &operator=(const RuntimeExecutableLease &) = delete;
    ~RuntimeExecutableLease() noexcept {
        if (!lifecycle_.published())
            return;
        auto destruction = lifecycle_.beginDestroy();
        if (destruction.isErr() || destruction.value().commit().isErr())
            vernon::resultContractViolation();
    }

    VernonProgramExecutable &executable;

private:
    RuntimeExecutableLease(VernonProgramExecutable &value, vernon::runtime::RuntimeChildLifecycle child) noexcept
        : executable(value), lifecycle_(std::move(child)) {}

    vernon::runtime::RuntimeChildLifecycle lifecycle_;
};

struct RuntimeGraphicsPipelineState {
    uint64_t serial{};
    // The pass Node the pipeline draws in.
    VernonProgramExecutable *pass{};
    VernonProgramNodeId node{};
    // The Program whose Stage draws; for pipeline 0 it is `pass` and its Node is the pass Node.
    VernonProgramExecutable *source{};
    const vernon::runtime::program::Node *sourceNode{};
    const vernon::runtime::program::ResolvedNodePlan *sourcePlan{};
    std::vector<vernon::runtime::recorded::PipelineEndpoint> endpoints;
    // Only pipeline 0 keeps the pass Node's endpoint classification; other pipelines conform to it at creation.
    std::vector<vernon::runtime::recorded::NodeEndpoint> passEndpoints;
    // Empty for pipeline 0, which `pass` caches; see VernonProgramExecutable::passPipelines.
    std::vector<std::unique_ptr<RuntimeExecutableLease>> leases;
};

struct VernonGraphicsPipeline {
    VernonRuntimeContext *context{};
    std::shared_ptr<const RuntimeGraphicsPipelineState> state;
    std::unique_ptr<RuntimeExecutableLease> lease;
};

struct RuntimeBindingSetEntry {
    // Boundary slot of the origin pipeline's Program; its contract decides which pipelines accept the entry.
    const vernon::runtime::program::BoundarySlot *boundary{};
    std::shared_ptr<RuntimeProgramBinding> binding;
};

// A sealed set prepared for one pipeline: the set's values inside that pipeline's Program.
struct RuntimeBindingSetMatch {
    std::weak_ptr<const RuntimeGraphicsPipelineState> pipeline;
    // Pipeline endpoints the set supplies, index-aligned with RuntimeGraphicsPipelineState::endpoints.
    std::vector<char> endpoints;
    // Per entry: why it does not correspond to a per-draw parameter of the pipeline; empty when it does. A
    // corresponding entry stays bound across a switch to this pipeline even if other entries of its set do not.
    std::vector<vernon::runtime::program::Diagnostic> failures;
    std::shared_ptr<const vernon::runtime::program_execution::ProgramInvocationState> values;
};

struct RuntimeBindingSetState {
    VernonRuntimeContext *context{};
    std::shared_ptr<const RuntimeGraphicsPipelineState> origin;
    std::unique_ptr<RuntimeExecutableLease> lease;
    std::vector<RuntimeBindingSetEntry> entries;
    bool sealed{};
    std::mutex matchMutex;
    std::unordered_map<uint64_t, std::shared_ptr<const RuntimeBindingSetMatch>> matches;
};

struct VernonBindingSet {
    std::shared_ptr<RuntimeBindingSetState> state;
};

struct RuntimeRecordedCommand {
    VernonGraphicsCommandKind kind{};
    std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline;
    std::shared_ptr<RuntimeBindingSetState> set;
    std::array<uint32_t, 4> rect{};
    uint32_t stencilReference{};
    std::shared_ptr<const RuntimeProgramControl> draw;
    std::string label;
};

struct RuntimeGraphicsRecording {
    std::vector<RuntimeRecordedCommand> commands;
    std::vector<std::shared_ptr<void>> leases;
};

// Where one Stage parameter of a recorded draw takes its materialized argument from: the pass Node's own frame,
// or the frame of a binding set prepared for the draw's pipeline.
struct RecordedArgumentSource {
    bool fromSet{};
    size_t frame{};
    size_t parameter{};
};

struct RecordedDraw {
    std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline;
    std::vector<RecordedArgumentSource> arguments;
    std::shared_ptr<const RuntimeProgramControl> draw;
    std::shared_ptr<const RuntimeProgramControl> dynamic;
};

struct RecordedSetFrame {
    std::shared_ptr<const RuntimeBindingSetMatch> match;
    std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline;
};

// A debug group push, pop or marker, encoded before draw `draw` of its pass, or after the last draw when `draw` is
// the draw count.
struct RecordedDebugCommand {
    VernonGraphicsCommandKind kind{};
    std::string label;
    size_t draw{};
};

struct RecordedGraphicsPass {
    std::vector<RecordedSetFrame> frames;
    std::vector<RecordedDraw> draws;
    // Balanced and in recording order.
    std::vector<RecordedDebugCommand> debugCommands;
    VernonProgramGraphicsNodeTelemetry telemetry{};
};

struct vernon::runtime::RecordedGraphicsPasses {
    std::map<VernonProgramNodeId, RecordedGraphicsPass> nodes;
};

struct RuntimeProgramInstanceState {
    RuntimeProgramInstanceState(VernonProgramExecutable &value, vernon::runtime::RuntimeChildLifecycle childLifecycle,
                                vernon::OwnerRef invocationOwner)
        : lifecycle(std::move(childLifecycle)), owner(std::move(invocationOwner)), pipeline(&value), bindings(&value),
          workspaces(
              std::make_shared<vernon::runtime::ad::ExecutionWorkspacePool>(value.programAutodiff.canonicalExecution)) {
    }
    ~RuntimeProgramInstanceState() noexcept {
        if (!lifecycle.published())
            return;
        auto destruction = lifecycle.beginDestroy();
        auto close = owner.beginClose();
        if (destruction.isErr() || close.isErr())
            vernon::resultContractViolation();
        if (close.value().commit().isErr() || destruction.value().commit().isErr())
            vernon::resultContractViolation();
    }
    vernon::runtime::RuntimeChildLifecycle lifecycle;
    vernon::OwnerRef owner;
    VernonProgramExecutable *pipeline;
    vernon::runtime::program::ProgramInstance bindings;
    const std::shared_ptr<vernon::runtime::ad::ExecutionWorkspacePool> workspaces;
    // Measured on the instance's last Execute; admission charges the next one with them before it allocates. Before
    // the first Execute one RHI submission is assumed.
    std::atomic<uint64_t> submissionTransientBytes{};
    std::atomic<uint32_t> submissionRhiSubmissions{1};
};

struct VernonProgramInstance {
    explicit VernonProgramInstance(std::shared_ptr<RuntimeProgramInstanceState> value) noexcept
        : state(std::move(value)) {}
    std::shared_ptr<RuntimeProgramInstanceState> state;
};

struct VernonProgramInvocation {
    VernonProgramInvocation(VernonProgramInstance &value, vernon::runtime::RuntimeChildLifecycle childLifecycle,
                            std::unique_ptr<vernon::runtime::program::BindingTransaction> bindingTransaction) noexcept
        : instance(value.state), lifecycle(std::move(childLifecycle)), transaction(std::move(bindingTransaction)) {}
    ~VernonProgramInvocation() {
        if (pendingPullback)
            vernonProgramPullbackDestroy(pendingPullback);
    }
    std::shared_ptr<RuntimeProgramInstanceState> instance;
    vernon::runtime::RuntimeChildLifecycle lifecycle;
    std::unique_ptr<vernon::runtime::program::BindingTransaction> transaction;
    std::shared_ptr<const vernon::runtime::program::InvocationSnapshot> snapshot;
    std::map<VernonProgramNodeId, std::unique_ptr<vernon::runtime::ad::PullbackExecution>> nodePullbacks;
    VernonPullback *pendingPullback{};
    std::optional<uint64_t> checkpointMemoryBudget;
    std::string checkpointPolicy;
    std::shared_ptr<vernon::runtime::ad::CanonicalProgramExecutionWorkspace> executionWorkspace;
    std::map<VernonProgramNodeId, RuntimeGraphicsRecording> recordings;
    std::map<VernonProgramNodeId, VernonProgramGraphicsNodeTelemetry> graphicsTelemetry;
    vernon::runtime::program_execution::CompletionMode completionMode{
        vernon::runtime::program_execution::CompletionMode::Wait};
    bool timings{};
    // Produced by an Execute that submitted work, until it is taken.
    std::unique_ptr<VernonSubmission> submission;
    bool executed{};
    bool finished{};
    bool succeeded{};
};

struct RuntimeProgramGraphNode {
    uint32_t id{};
    std::string bundleId;
    std::string label;
    std::string contentHash;
    vernon::runtime::ProgramVariantDeployment deployment;
    std::filesystem::path bundleRoot;
};

struct RuntimeProgramGraphValue {
    vernon::runtime::ProgramGraphBoundaryKey source;
    std::vector<vernon::runtime::ProgramGraphBoundaryKey> destinations;
};

struct RuntimeProgramGraphStorage {
    VernonProgramArgumentKind kind{VERNON_PROGRAM_TENSOR};
    std::vector<vernon::runtime::ProgramGraphBoundaryKey> versions;
};

struct VernonProgramGraph {
    explicit VernonProgramGraph(vernon::runtime::RuntimeChildLifecycle childLifecycle) noexcept
        : lifecycle(std::move(childLifecycle)) {}
    vernon::runtime::RuntimeChildLifecycle lifecycle;
    VernonRuntimeContext *context{};
    uint64_t id{};
    std::vector<RuntimeProgramGraphNode> nodes;
    std::vector<RuntimeProgramGraphValue> values;
    std::vector<RuntimeProgramGraphStorage> storages;
    std::vector<vernon::runtime::ProgramGraphExport> exports;
};

namespace vernon::runtime::capi {

inline VernonRuntimeContext *instanceContext(const VernonProgramInstance *instance) {
    return instance && instance->state && instance->state->pipeline ? instance->state->pipeline->context : nullptr;
}

inline VernonProgramExecutable *invocationExecutable(const VernonProgramInvocation *invocation) {
    return invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
}

inline VernonRuntimeContext *invocationContext(const VernonProgramInvocation *invocation) {
    VernonProgramExecutable *executable = invocationExecutable(invocation);
    return executable ? executable->context : nullptr;
}

vernon::Result<std::shared_ptr<void>, vernon::RuntimeError>
retainProgramResourceLease(const VernonProgramResourceLease &lease) noexcept;

const program::Program *executableProgram(const VernonProgramExecutable &pipeline);
size_t publicBoundaryCount(const program::Program &program);
size_t mutationCapacity(const VernonProgramExecutable &pipeline);
const program::BoundarySlot *publicBoundaryAt(const program::Program &program, size_t publicIndex);
const program::BoundarySlot *parameterBoundaryAt(const VernonProgramExecutable &pipeline, size_t index);
const program::BoundarySlot *findParameterBoundary(const VernonProgramExecutable &pipeline, VernonStringView name);
uint64_t programControlKey(uint32_t slot, RuntimeProgramControl::Kind kind);
void buildExecutableBindingIndex(VernonProgramExecutable &pipeline);
bool bindableParameterSlot(const VernonProgramExecutable &pipeline, uint32_t slot);
const std::vector<uint32_t> *bindingAliasSlots(const VernonProgramExecutable &pipeline, uint32_t slot);
// Every boundary slot a binding of `slot` reaches, including `slot` itself when it has no aliases.
std::vector<uint32_t> boundSlotsOf(const VernonProgramExecutable &pipeline, uint32_t slot);
std::optional<program::BoundaryRole> reflectedBoundaryRole(VernonProgramBoundaryRole boundary);
const program::BoundarySlot *boundaryAt(const program::Program &program, program::BoundaryRole role, size_t index);
const ValueLayout *boundaryLayoutView(const VernonProgramExecutable &pipeline, const program::BoundarySlot &slot);
bool stringViewEquals(VernonStringView view, const std::string &value);
/* Collects the forward graph's graphics nodes in graph order, which is the order the controls query indexes by. */
void collectForwardGraphicsNodes(const VernonProgramExecutable *pipeline, std::vector<const program::Node *> &nodes);

const char *programArgumentDiagnostic(const program::BoundarySlot &boundary, const VernonProgramArgument &argument);
using ProgramBindingValueResult = vernon::Result<std::shared_ptr<RuntimeProgramBinding>, vernon::RuntimeError>;
// `argument` must have passed programArgumentDiagnostic.
ProgramBindingValueResult makeProgramBinding(const VernonProgramArgument &argument,
                                             const VernonProgramResourceLease *lease);
std::shared_ptr<RuntimeProgramControl> makeDrawCommandControl(const VernonDrawCommand *draw);

} // namespace vernon::runtime::capi

#endif
