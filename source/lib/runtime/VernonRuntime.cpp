#include "VernonRuntime.h"
#include "execution_graph/execution_graph_internal.h"
#include "rhi/rhi_internal.h"
#include "runtime/autodiff/execution_workspace_pool.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/autodiff/tape_allocator_abi.h"
#include "runtime/backend_cpu.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/graphics_scope_materializer.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/program_execution/device_commands.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/program_execution/materialized_node_frame.h"
#include "runtime/program_execution/program_boundary_contract.h"
#include "runtime/program_execution/program_forward.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/program_execution/resolved_transfer_executor.h"
#include "runtime/program_execution_backend.h"
#include "runtime/program_graph_linker.h"
#include "runtime/program_graphics_executor.h"
#include "runtime/program_instance.h"
#include "runtime/program_invocation_context.h"
#include "runtime/recorded_graphics.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime/stage_artifact.h"
#include "runtime/stage_binding_plan.h"
#include "runtime/tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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

struct RecordedGraphicsPass {
    std::vector<RecordedSetFrame> frames;
    std::vector<RecordedDraw> draws;
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
    // Produced by an Execute that submitted work, until it is taken.
    std::unique_ptr<VernonSubmission> submission;
    bool executed{};
    bool finished{};
    bool succeeded{};
};

struct RuntimeProgramGraphNode {
    uint32_t id{};
    std::string bundleId;
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

namespace {
VernonRuntimeContext *invocationContext(const VernonProgramInvocation *invocation) {
    return invocation && invocation->instance && invocation->instance->pipeline
               ? invocation->instance->pipeline->context
               : nullptr;
}

struct RuntimeDiagnosticState {
    const VernonRuntimeContext *context{};
    uint64_t generation{};
    std::string pending;
    std::string published;
    std::array<char, 192> emergency{};
    size_t emergencySize{};
    size_t depth{};
    uint64_t lastUse{};
    bool retired{};
};

struct RuntimeDiagnosticOverflowEntry {
    RuntimeDiagnosticState state;
    std::unique_ptr<RuntimeDiagnosticOverflowEntry> next;
};

struct RuntimeDiagnosticTable {
    std::array<RuntimeDiagnosticState, 16> entries{};
    std::unique_ptr<RuntimeDiagnosticOverflowEntry> overflow;
    RuntimeDiagnosticState emergency;
    RuntimeDiagnosticState discard;
    uint64_t clock{};
};
thread_local RuntimeDiagnosticTable invocationDiagnostics;
thread_local bool failNextDiagnosticOverflowAllocation;
std::atomic<uint64_t> nextDiagnosticGeneration{1};

void touchDiagnosticState(RuntimeDiagnosticState &state) noexcept {
    if (invocationDiagnostics.clock != std::numeric_limits<uint64_t>::max())
        ++invocationDiagnostics.clock;
    state.lastUse = invocationDiagnostics.clock;
}

RuntimeDiagnosticState *findDiagnosticState(const VernonRuntimeContext *context, uint64_t generation) noexcept {
    for (RuntimeDiagnosticState &state : invocationDiagnostics.entries) {
        if (state.context == context && state.generation == generation) {
            touchDiagnosticState(state);
            return &state;
        }
    }
    for (RuntimeDiagnosticOverflowEntry *entry = invocationDiagnostics.overflow.get(); entry;
         entry = entry->next.get()) {
        if (entry->state.context == context && entry->state.generation == generation) {
            touchDiagnosticState(entry->state);
            return &entry->state;
        }
    }
    RuntimeDiagnosticState &emergency = invocationDiagnostics.emergency;
    if (emergency.context == context && emergency.generation == generation) {
        touchDiagnosticState(emergency);
        return &emergency;
    }
    return nullptr;
}

RuntimeDiagnosticState *findDiagnosticState(const VernonRuntimeContext &context) noexcept {
    return findDiagnosticState(&context, context.diagnosticGeneration);
}

void initializeDiagnosticState(RuntimeDiagnosticState &state, VernonRuntimeContext &context) noexcept {
    state.context = &context;
    state.generation = context.diagnosticGeneration;
    state.pending.clear();
    state.published.clear();
    state.emergencySize = 0;
    state.depth = 0;
    state.retired = false;
    touchDiagnosticState(state);
}

RuntimeDiagnosticState *tryDiagnosticState(VernonRuntimeContext &context) noexcept {
    if (RuntimeDiagnosticState *state = findDiagnosticState(context))
        return state;
    RuntimeDiagnosticState *candidate = &invocationDiagnostics.entries.front();
    for (RuntimeDiagnosticState &state : invocationDiagnostics.entries) {
        if (!state.context) {
            candidate = &state;
            break;
        }
        if (!state.depth && (candidate->depth || state.lastUse < candidate->lastUse))
            candidate = &state;
    }
    if (!candidate->depth) {
        initializeDiagnosticState(*candidate, context);
        return candidate;
    }

    RuntimeDiagnosticOverflowEntry *last = nullptr;
    for (RuntimeDiagnosticOverflowEntry *entry = invocationDiagnostics.overflow.get(); entry;
         entry = entry->next.get()) {
        if (!entry->state.depth) {
            initializeDiagnosticState(entry->state, context);
            return &entry->state;
        }
        last = entry;
    }

    std::unique_ptr<RuntimeDiagnosticOverflowEntry> added;
    if (failNextDiagnosticOverflowAllocation)
        failNextDiagnosticOverflowAllocation = false;
    else
        added.reset(new (std::nothrow) RuntimeDiagnosticOverflowEntry);
    if (added) {
        RuntimeDiagnosticOverflowEntry *result = added.get();
        if (last)
            last->next = std::move(added);
        else
            invocationDiagnostics.overflow = std::move(added);
        initializeDiagnosticState(result->state, context);
        return &result->state;
    }

    RuntimeDiagnosticState &emergency = invocationDiagnostics.emergency;
    if (!emergency.depth) {
        initializeDiagnosticState(emergency, context);
        return &emergency;
    }
    return nullptr;
}

RuntimeDiagnosticState &diagnosticState(VernonRuntimeContext &context) noexcept {
    if (RuntimeDiagnosticState *state = tryDiagnosticState(context))
        return *state;
    return invocationDiagnostics.discard;
}
} // namespace

std::string &vernon::runtime::invocationDiagnostic(VernonRuntimeContext &context) {
    RuntimeDiagnosticState &state = diagnosticState(context);
    return state.depth ? state.pending : state.published;
}

const std::string *vernon::runtime::currentInvocationDiagnostic(const VernonRuntimeContext &context) {
    RuntimeDiagnosticState *state = findDiagnosticState(context);
    if (!state || state->emergencySize)
        return nullptr;
    return state->depth ? &state->pending : &state->published;
}

VernonStringView vernon::runtime::currentInvocationDiagnosticView(const VernonRuntimeContext &context) noexcept {
    RuntimeDiagnosticState *state = findDiagnosticState(context);
    if (!state)
        return {nullptr, 0};
    if (state->emergencySize)
        return {state->emergency.data(), state->emergencySize};
    const std::string &diagnostic = state->depth ? state->pending : state->published;
    return {diagnostic.data(), diagnostic.size()};
}

void vernon::runtime::clearInvocationDiagnostic(const VernonRuntimeContext &context) noexcept {
    RuntimeDiagnosticState *state = findDiagnosticState(context);
    if (state && state->depth)
        state->retired = true;
    else if (state)
        *state = {};
}

uint64_t vernon::runtime::diagnosticClockForTesting() noexcept { return invocationDiagnostics.clock; }

void vernon::runtime::setDiagnosticClockForTesting(uint64_t clock) noexcept {
    invocationDiagnostics.clock = clock;
    for (RuntimeDiagnosticState &state : invocationDiagnostics.entries)
        state.lastUse = clock;
    for (RuntimeDiagnosticOverflowEntry *entry = invocationDiagnostics.overflow.get(); entry; entry = entry->next.get())
        entry->state.lastUse = clock;
    invocationDiagnostics.emergency.lastUse = clock;
}

void vernon::runtime::failNextDiagnosticOverflowAllocationForTesting() noexcept {
    failNextDiagnosticOverflowAllocation = true;
}

uint64_t vernon::runtime::diagnosticGenerationCounterForTesting() noexcept {
    return nextDiagnosticGeneration.load(std::memory_order_relaxed);
}

void vernon::runtime::setDiagnosticGenerationCounterForTesting(uint64_t generation) noexcept {
    nextDiagnosticGeneration.store(generation, std::memory_order_relaxed);
}

vernon::runtime::RuntimeDiagnosticScope::RuntimeDiagnosticScope(const VernonRuntimeContext *context) noexcept
    : context_(context), generation_(context ? context->diagnosticGeneration : 0) {
    if (!context_)
        return;
    RuntimeDiagnosticState *state = tryDiagnosticState(*const_cast<VernonRuntimeContext *>(context_));
    state_ = state;
    if (state && state->depth++ == 0) {
        state->pending.clear();
        state->emergencySize = 0;
    }
}

vernon::runtime::RuntimeDiagnosticScope::~RuntimeDiagnosticScope() noexcept {
    auto *state = static_cast<RuntimeDiagnosticState *>(state_);
    if (!state || state->context != context_ || state->generation != generation_)
        return;
    if (!state->depth)
        return;
    if (--state->depth != 0)
        return;
    if (state->retired)
        *state = {};
    else if (!state->emergencySize)
        state->published = std::move(state->pending);
}

namespace {

using namespace vernon::runtime;

vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError> createRuntimeContext() noexcept {
    auto created = VernonRuntimeContext::create();
    if (created.isErr())
        return created;

    uint64_t generation = nextDiagnosticGeneration.load(std::memory_order_relaxed);
    do {
        if (generation == std::numeric_limits<uint64_t>::max())
            return vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>{
                vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ResourceExhausted,
                                                 {"runtime diagnostic generation exhausted", 0, 0}})};
    } while (!nextDiagnosticGeneration.compare_exchange_weak(generation, generation + 1, std::memory_order_relaxed,
                                                             std::memory_order_relaxed));
    created.value()->diagnosticGeneration = generation;
    return created;
}

vernon::runtime::program_execution::SubmissionBounds submissionBounds(const VernonRuntimeCreateOptions *options) {
    constexpr vernon::runtime::program_execution::SubmissionBounds defaults =
        vernon::runtime::program_execution::defaultSubmissionBounds;
    if (!options)
        return defaults;
    return {options->transient_budget_bytes ? options->transient_budget_bytes : defaults.transientBudgetBytes,
            options->max_pending_submissions ? options->max_pending_submissions : defaults.maxPendingSubmissions};
}

vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>
createRuntimeContextForBackend(VernonRuntimeBackend backend, const VernonRuntimeCreateOptions *options) {
    using Result = vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>;
    if (options && options->struct_size < sizeof(VernonRuntimeCreateOptions))
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                       {"create_runtime_context", options->struct_size, 0}})};
    auto created = createRuntimeContext();
    if (created.isErr())
        return created;
    created.value()->backend = backend;
    auto initialized = initializeBackend(*created.value(), options ? options->device_index : 0);
    if (initialized.isErr())
        return Result{vernon::err(std::move(initialized).error())};
    created.value()->submissions.configure(submissionBounds(options), created.value()->rhiDevice);
    return created;
}

vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>
createRuntimeContextForRhiDevice(VernonRuntimeBackend backend, VernonRhiDevice device,
                                 const VernonRuntimeCreateOptions *options) {
    using Result = vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>;
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX ||
        (options && (options->struct_size < sizeof(VernonRuntimeCreateOptions) || options->device_index)))
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                       {"create_runtime_context_for_rhi_device", 0, 0}})};
    auto created = createRuntimeContext();
    if (created.isErr())
        return created;
    created.value()->backend = backend;
    auto initialized = initializeBackendForRhiDevice(*created.value(), device);
    if (initialized.isErr())
        return Result{vernon::err(std::move(initialized).error())};
    created.value()->submissions.configure(submissionBounds(options), created.value()->rhiDevice);
    return created;
}

VernonStatus fail(VernonRuntimeContext *context, std::string_view error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) noexcept {
    if (!context)
        return status;
    RuntimeDiagnosticState *state = tryDiagnosticState(*context);
    if (!state)
        return status;
    try {
        (state->depth ? state->pending : state->published).assign(error);
        state->emergencySize = 0;
    } catch (...) {
        constexpr std::string_view fallback = "Runtime diagnostic allocation failed";
        state->emergencySize = std::min(fallback.size(), state->emergency.size() - 1);
        std::memcpy(state->emergency.data(), fallback.data(), state->emergencySize);
        state->emergency[state->emergencySize] = '\0';
    }
    return status;
}

VernonStatus publishRuntimeError(VernonRuntimeContext *context, vernon::RuntimeError error) noexcept {
    if (context) {
        RuntimeDiagnosticState *state = tryDiagnosticState(*context);
        if (!state)
            return vernon::toVernonStatus(error);
        try {
            (state->depth ? state->pending : state->published)
                .assign(error.context.operation ? error.context.operation : "Runtime operation failed");
            state->emergencySize = 0;
        } catch (...) {
            const size_t required =
                vernon::renderEmergencyDiagnostic(error, state->emergency.data(), state->emergency.size());
            state->emergencySize = std::min(required, state->emergency.size() - 1);
        }
    }
    return vernon::toVernonStatus(error);
}

template <typename Callback>
VernonStatus runtimeResultBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        RuntimeDiagnosticScope diagnostic(context);
        auto result = std::forward<Callback>(callback)();
        return result.isOk() ? VERNON_STATUS_OK : publishRuntimeError(context, std::move(result).error());
    } catch (const std::bad_alloc &) {
        return publishRuntimeError(
            context, {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        return publishRuntimeError(context,
                                   {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
}

template <typename Callback>
VernonStatus runtimeStatusBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        RuntimeDiagnosticScope diagnostic(context);
        return std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        return publishRuntimeError(
            context, {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        return publishRuntimeError(context,
                                   {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
}

template <typename Pointer, typename Callback>
Pointer runtimePointerBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        RuntimeDiagnosticScope diagnostic(context);
        return std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
    return nullptr;
}

template <typename Pointer, typename Callback>
Pointer runtimePointerResultBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    return runtimePointerBoundary<Pointer>(context, [&]() -> Pointer {
        auto result = std::forward<Callback>(callback)();
        if (result.isErr()) {
            (void)publishRuntimeError(context, std::move(result).error());
            return nullptr;
        }
        return std::move(result).value().release();
    });
}

VernonRuntimeOperationStatus runtimeOperationStatus(vernon::RuntimeErrorCode code) noexcept {
    return code == vernon::RuntimeErrorCode::NotReady ? VERNON_RUNTIME_OPERATION_INVALID_ARGUMENT
                                                      : static_cast<VernonRuntimeOperationStatus>(code);
}

template <typename Pointer, typename Callback>
VernonRuntimeOperationStatus runtimeHandleResultBoundary(VernonRuntimeContext *context, Pointer **output,
                                                         Callback &&callback) noexcept {
    if (output)
        *output = nullptr;
    try {
        RuntimeDiagnosticScope diagnostic(context);
        if (!output) {
            (void)fail(context, "Runtime result output is null");
            return VERNON_RUNTIME_OPERATION_INVALID_ARGUMENT;
        }
        auto result = std::forward<Callback>(callback)();
        if (result.isErr()) {
            vernon::RuntimeError error = std::move(result).error();
            const VernonStringView current =
                context ? currentInvocationDiagnosticView(*context) : VernonStringView{nullptr, 0};
            if (!current.size)
                (void)publishRuntimeError(context, error);
            return runtimeOperationStatus(error.code);
        }
        *output = std::move(result).value();
        return VERNON_RUNTIME_OPERATION_OK;
    } catch (const std::bad_alloc &) {
        const vernon::RuntimeError error{vernon::RuntimeErrorCode::ResourceExhausted,
                                         {"Runtime boundary allocation", 0, 0}};
        (void)publishRuntimeError(context, error);
        return VERNON_RUNTIME_OPERATION_RESOURCE_EXHAUSTED;
    } catch (...) {
        const vernon::RuntimeError error{vernon::RuntimeErrorCode::InternalFailure,
                                         {"Runtime boundary exception", 0, 0}};
        (void)publishRuntimeError(context, error);
        return VERNON_RUNTIME_OPERATION_INTERNAL_FAILURE;
    }
}

template <typename Value, typename Callback>
Value runtimeValueBoundary(const VernonRuntimeContext *context, Value fallback, Callback &&callback) noexcept {
    auto *mutableContext = const_cast<VernonRuntimeContext *>(context);
    try {
        RuntimeDiagnosticScope diagnostic(context);
        return std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        (void)publishRuntimeError(mutableContext,
                                  {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        (void)publishRuntimeError(mutableContext,
                                  {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
    return fallback;
}

template <typename Callback> void runtimeVoidBoundary(VernonRuntimeContext *context, Callback &&callback) noexcept {
    try {
        std::forward<Callback>(callback)();
    } catch (const std::bad_alloc &) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::ResourceExhausted, {"Runtime boundary allocation", 0, 0}});
    } catch (...) {
        (void)publishRuntimeError(context,
                                  {vernon::RuntimeErrorCode::InternalFailure, {"Runtime boundary exception", 0, 0}});
    }
}

using ProgramLeaseCallbackResult = vernon::Result<void, vernon::RuntimeError>;

ProgramLeaseCallbackResult retainForeignProgramResource(void *object, void (*retain)(void *)) noexcept {
    try {
        retain(object);
        return ProgramLeaseCallbackResult{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return ProgramLeaseCallbackResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ResourceExhausted,
                                             {"Program resource retain callback allocation failed", 0, 0}})};
    } catch (...) {
        return ProgramLeaseCallbackResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InternalFailure, {"Program resource retain callback failed", 0, 0}})};
    }
}

void releaseForeignProgramResource(void *object, void (*release)(void *)) noexcept {
    try {
        release(object);
    } catch (...) {
        vernon::resultContractViolation();
    }
}

vernon::Result<std::shared_ptr<void>, vernon::RuntimeError>
retainProgramResourceLease(const VernonProgramResourceLease &lease) noexcept {
    using Result = vernon::Result<std::shared_ptr<void>, vernon::RuntimeError>;
    if (lease.struct_size < sizeof(lease) || (lease.retain == nullptr) != (lease.release == nullptr))
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                       {"Program resource lease is invalid", 0, 0}})};
    if (!lease.retain)
        return Result{vernon::ok(std::shared_ptr<void>{})};
    auto retained = retainForeignProgramResource(lease.object, lease.retain);
    if (retained.isErr())
        return Result{vernon::err(std::move(retained).error())};
    try {
        return Result{vernon::ok(std::shared_ptr<void>(lease.object, [release = lease.release](void *object) noexcept {
            releaseForeignProgramResource(object, release);
        }))};
    } catch (const std::bad_alloc &) {
        releaseForeignProgramResource(lease.object, lease.release);
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ResourceExhausted,
                                                       {"Program resource lease allocation failed", 0, 0}})};
    } catch (...) {
        releaseForeignProgramResource(lease.object, lease.release);
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InternalFailure,
                                                       {"Program resource lease construction failed", 0, 0}})};
    }
}

std::string pipelineTargetKind(const nlohmann::json &root) {
    if (!root.contains("target") || !root["target"].is_object())
        return {};
    const nlohmann::json &target = root["target"];
    if (!target.contains("kind") || !target["kind"].is_string() || !target.contains("options") ||
        !target["options"].is_object() || target.size() != 2)
        return {};
    const std::string kind = target["kind"].get<std::string>();
    const nlohmann::json &options = target["options"];
    auto hasOnly = [&](std::initializer_list<std::string_view> allowed) {
        for (const auto &[key, value] : options.items())
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                return false;
        return true;
    };
    if (kind == "cpu") {
        if (!hasOnly({"triple", "processor", "features"}) || !options.contains("triple") ||
            !options["triple"].is_string() || options["triple"].get_ref<const std::string &>().empty())
            return {};
        if (options.contains("processor") && !options["processor"].is_string())
            return {};
        if (options.contains("features") &&
            (!options["features"].is_array() ||
             !std::all_of(options["features"].begin(), options["features"].end(), [](const nlohmann::json &feature) {
                 return feature.is_string() && !feature.get_ref<const std::string &>().empty();
             })))
            return {};
    } else if (kind == "opengl" || kind == "opengles") {
        if (!hasOnly({"version"}))
            return {};
        if (options.contains("version") &&
            (!options["version"].is_number_unsigned() || options["version"].get<uint64_t>() < 100 ||
             options["version"].get<uint64_t>() > 999))
            return {};
    } else if (kind == "metal") {
        if (!hasOnly({"platform"}) || !options.contains("platform") || !options["platform"].is_string() ||
            (options["platform"] != "macos" && options["platform"] != "ios"))
            return {};
    } else if (kind == "directx") {
        if (!hasOnly({"shader_model"}) || !options.contains("shader_model") ||
            !options["shader_model"].is_number_unsigned() || options["shader_model"].get<uint64_t>() < 60)
            return {};
    } else if (kind == "vulkan" || kind == "cuda") {
        if (!options.empty())
            return {};
    } else {
        return {};
    }
    return kind;
}

const program::Program *executableProgram(const VernonProgramExecutable &pipeline) {
    return &pipeline.executionPlan->resolvedProgram->program;
}

bool publicBoundarySlot(const program::Program &program, const program::BoundarySlot &slot) {
    (void)program;
    if (slot.role == program::BoundaryRole::Input)
        return true;
    return slot.role == program::BoundaryRole::Output && slot.publication != program::BoundaryPublication::None;
}

size_t publicBoundaryCount(const program::Program &program) {
    return static_cast<size_t>(
        std::count_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                      [&](const program::BoundarySlot &slot) { return publicBoundarySlot(program, slot); }));
}

size_t mutationCapacity(const VernonProgramExecutable &pipeline) {
    if (!pipeline.executionPlan || !pipeline.executionPlan->resolvedProgram)
        return 0;
    const program::Program &program = pipeline.executionPlan->resolvedProgram->program;
    std::set<uint32_t> renderPassControls;
    for (const program::Graph &graph : program.graphs)
        for (const program::Node &node : graph.nodes)
            if (program::executionKind(node) == program::ExecutionKind::Graphics)
                renderPassControls.insert(program::graphicsOperation(node).renderPassControl);
    return pipeline.executionPlan->publications.transactions.size() + renderPassControls.size();
}

const program::BoundarySlot *publicBoundaryAt(const program::Program &program, size_t publicIndex) {
    for (const program::BoundarySlot &slot : program.abi.boundarySlots)
        if (publicBoundarySlot(program, slot) && publicIndex-- == 0)
            return &slot;
    return nullptr;
}

const program::BoundarySlot *findPublicBoundary(const program::Program &program, VernonStringView name) {
    const auto found = std::find_if(
        program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(), [&](const program::BoundarySlot &slot) {
            return publicBoundarySlot(program, slot) && name.size == slot.path.size() &&
                   (name.size == 0 || std::equal(name.data, name.data + name.size, slot.path.data()));
        });
    return found == program.abi.boundarySlots.end() ? nullptr : &*found;
}

const program::BoundarySlot *parameterBoundaryAt(const VernonProgramExecutable &pipeline, size_t index) {
    const program::Program &program = *executableProgram(pipeline);
    if (!pipeline.programGraphId)
        return publicBoundaryAt(program, index);
    if (index >= pipeline.publicParameterSlots.size())
        return nullptr;
    const uint32_t slot = pipeline.publicParameterSlots[index];
    return slot < program.abi.boundarySlots.size() ? &program.abi.boundarySlots[slot] : nullptr;
}

const program::BoundarySlot *findParameterBoundary(const VernonProgramExecutable &pipeline, VernonStringView name) {
    if (!pipeline.programGraphId)
        return findPublicBoundary(*executableProgram(pipeline), name);
    for (size_t index = 0; index < pipeline.publicParameterSlots.size(); ++index) {
        const program::BoundarySlot *slot = parameterBoundaryAt(pipeline, index);
        if (slot && name.size == slot->path.size() &&
            (!name.size || std::equal(name.data, name.data + name.size, slot->path.data())))
            return slot;
    }
    return nullptr;
}

uint64_t programOwnerKey(const program::ProgramOwnerId &owner) {
    return (static_cast<uint64_t>(owner.kind) << 32) | owner.id;
}

uint64_t programControlKey(uint32_t slot, RuntimeProgramControl::Kind kind) {
    return (static_cast<uint64_t>(kind) + 1) << 32 | slot;
}

void buildExecutableBindingIndex(VernonProgramExecutable &pipeline) {
    const program::Program &program = *executableProgram(pipeline);
    pipeline.bindableParameterSlots.clear();
    pipeline.bindingAliases.clear();
    pipeline.graphicsControlKeys.clear();
    if (pipeline.programGraphId)
        pipeline.bindableParameterSlots.insert(pipeline.publicParameterSlots.begin(),
                                               pipeline.publicParameterSlots.end());
    for (const program::BoundarySlot &slot : program.abi.boundarySlots) {
        if (!pipeline.programGraphId && publicBoundarySlot(program, slot))
            pipeline.bindableParameterSlots.insert(slot.id);
        if (publicBoundarySlot(program, slot))
            pipeline.bindingAliases[programOwnerKey(slot.aliasOwner)].push_back(slot.id);
    }
    for (const program::Graph &graph : program.graphs)
        for (const program::Node &node : graph.nodes)
            if (program::executionKind(node) == program::ExecutionKind::Graphics) {
                const program::GraphicsOperation &graphics = program::graphicsOperation(node);
                pipeline.graphicsControlKeys.insert(
                    programControlKey(graphics.renderPassControl, RuntimeProgramControl::RenderPass));
                pipeline.graphicsControlKeys.insert(
                    programControlKey(graphics.drawCommandControl, RuntimeProgramControl::DrawCommand));
                pipeline.graphicsControlKeys.insert(
                    programControlKey(graphics.dynamicStateControl, RuntimeProgramControl::DynamicState));
            }
}

bool bindableParameterSlot(const VernonProgramExecutable &pipeline, uint32_t slot) {
    return pipeline.bindableParameterSlots.find(slot) != pipeline.bindableParameterSlots.end();
}

const std::vector<uint32_t> *bindingAliasSlots(const VernonProgramExecutable &pipeline, uint32_t slot) {
    const program::Program &program = *executableProgram(pipeline);
    if (slot >= program.abi.boundarySlots.size())
        return nullptr;
    const auto found = pipeline.bindingAliases.find(programOwnerKey(program.abi.boundarySlots[slot].aliasOwner));
    return found == pipeline.bindingAliases.end() ? nullptr : &found->second;
}

// Every boundary slot a binding of `slot` reaches, including `slot` itself when it has no aliases.
std::vector<uint32_t> boundSlotsOf(const VernonProgramExecutable &pipeline, uint32_t slot) {
    const std::vector<uint32_t> *aliases = bindingAliasSlots(pipeline, slot);
    return aliases ? *aliases : std::vector<uint32_t>{slot};
}

std::optional<program::BoundaryRole> reflectedBoundaryRole(VernonProgramBoundaryRole boundary) {
    switch (boundary) {
    case VERNON_PROGRAM_BOUNDARY_INPUT:
        return program::BoundaryRole::Input;
    case VERNON_PROGRAM_BOUNDARY_OUTPUT:
        return program::BoundaryRole::Output;
    case VERNON_PROGRAM_BOUNDARY_COTANGENT:
        return program::BoundaryRole::Cotangent;
    case VERNON_PROGRAM_BOUNDARY_GRADIENT:
        return program::BoundaryRole::Gradient;
    default:
        return std::nullopt;
    }
}

const program::BoundarySlot *boundaryAt(const program::Program &program, program::BoundaryRole role, size_t index) {
    for (const program::BoundarySlot &slot : program.abi.boundarySlots)
        if (slot.role == role && index-- == 0)
            return &slot;
    return nullptr;
}

const ValueLayout *boundaryLayoutView(const VernonProgramExecutable &pipeline, const program::BoundarySlot &slot) {
    const program::Program *program = executableProgram(pipeline);
    if (!program)
        return nullptr;
    const auto slotIndex = static_cast<size_t>(&slot - program->abi.boundarySlots.data());
    const program::ResolvedExecutionPlan &execution = *pipeline.executionPlan;
    return slotIndex < execution.boundaryLayoutViews.size() ? &execution.boundaryLayoutViews[slotIndex] : nullptr;
}

} // namespace

extern "C" {

VernonValueLayoutView vernonRuntimeGetScalarValueLayout(VernonDataType dtype) {
    return runtimeValueBoundary(nullptr, VernonValueLayoutView{}, [&] {
        static constexpr VernonValueLeafView leaves[] = {
            {VERNON_DATA_BOOL, 1, 0}, {VERNON_DATA_I32, 1, 0}, {VERNON_DATA_U32, 1, 0}, {VERNON_DATA_F16, 1, 0},
            {VERNON_DATA_F32, 1, 0},  {VERNON_DATA_F64, 1, 0}, {VERNON_DATA_U8, 1, 0},
        };
        static constexpr const char *hashes[] = {
            "3ca886485debde52d9dae8389b51daf52bf265e898ece94a57241849eea52fc7",
            "5221c466df6b1fe9046f6d2e7597efdc98f5a3aa66bb176fe65d02de0c39607f",
            "5d0250c80dab299ac915d5d0d21170d208e2f4d97216c89d0263a3c2d3bf5dc8",
            "937b700417d47a346038256ddb7c3ed7062303c531efba4d6dfd5e21583deec4",
            "cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
            "8f6e354f03614c96a53ba7e4ff053d14f7ce2c8f5066a5193d43f28db91268da",
            "",
        };
        auto sizeResult = dataTypeSize(dtype);
        const size_t index = static_cast<size_t>(dtype);
        if (sizeResult.isErr() || index >= std::size(leaves) || !hashes[index][0])
            return VernonValueLayoutView{};
        const size_t size = std::move(sizeResult).value();
        return VernonValueLayoutView{sizeof(VernonValueLayoutView),
                                     static_cast<uint32_t>(size),
                                     static_cast<uint32_t>(size),
                                     {hashes[index], std::strlen(hashes[index])},
                                     &leaves[index],
                                     1};
    });
}

VernonRuntimeCapabilities vernonRuntimeGetCapabilities(VernonRuntimeBackend backend) {
    return runtimeValueBoundary(nullptr, VernonRuntimeCapabilities{}, [&] {
        static thread_local std::string diagnostic;
        diagnostic.clear();
        VernonRuntimeCapabilities result{};
        result.available = probeBackend(backend, diagnostic).isOk();
        if (backend == VERNON_RUNTIME_CPU && result.available) {
            result.supports_compute = 1;
            result.supports_storage_buffers = 1;
        } else if ((backend == VERNON_RUNTIME_CUDA || backend == VERNON_RUNTIME_VULKAN ||
                    backend == VERNON_RUNTIME_DIRECTX12 || backend == VERNON_RUNTIME_METAL) &&
                   result.available) {
            result.supports_compute = result.available;
            result.supports_storage_buffers = result.available;
        } else if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
            result.supports_graphics = 1;
        }
        result.diagnostic = {diagnostic.data(), diagnostic.size()};
        return result;
    });
}

VernonRuntimeContext *vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                                                     const VernonRuntimeCreateOptions *options) {
    return runtimePointerResultBoundary<VernonRuntimeContext *>(
        nullptr, [&] { return createRuntimeContextForBackend(backend, options); });
}

VernonRuntimeContext *vernonRuntimeCreateForRhiDevice(VernonRuntimeBackend backend, VernonRhiDevice device,
                                                      const VernonRuntimeCreateOptions *options) {
    return runtimePointerResultBoundary<VernonRuntimeContext *>(
        nullptr, [&] { return createRuntimeContextForRhiDevice(backend, device, options); });
}

VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context) {
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return VERNON_STATUS_OK;
        auto operations = context->operations.beginDestroy();
        if (operations.isErr())
            return fail(context, "runtime context has an active operation", vernon::toVernonStatus(operations.error()));
        auto close = context->owner.beginClose();
        if (close.isErr()) {
            (void)operations.value().rollback();
            return fail(context, "runtime context still owns live handles", vernon::toVernonStatus(close.error()));
        }
        destroyBackend(*context);
        if (close.value().commit().isErr() || operations.value().commit().isErr())
            vernon::resultContractViolation();
        vernon::runtime::clearInvocationDiagnostic(*context);
        delete context;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeContextGetSubmissionTelemetry(VernonRuntimeContext *context,
                                                        VernonSubmissionTelemetry *output) {
    return runtimeStatusBoundary(context, [&] {
        if (!context || !output || output->struct_size < sizeof(*output))
            return fail(context, "invalid submission telemetry query");
        *output = context->submissions.telemetry();
        return VERNON_STATUS_OK;
    });
}

VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context) {
    return runtimeValueBoundary(nullptr, VernonStringView{nullptr, 0}, [&] {
        if (!context)
            return VernonStringView{nullptr, 0};
        return vernon::runtime::currentInvocationDiagnosticView(*context);
    });
}

VernonRuntimeCapabilities vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context) {
    return runtimeValueBoundary(context, VernonRuntimeCapabilities{}, [&] {
        VernonRuntimeCapabilities result{};
        if (!context)
            return result;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return result;
        fillBackendCapabilities(*context, result);
        result.diagnostic = currentInvocationDiagnosticView(*context);
        return result;
    });
}

VernonStatus vernonRuntimeRegisterStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    return runtimeResultBoundary(nullptr, [&] { return registerBackendStaticCpuEntries(entries, count); });
}

VernonStatus vernonRuntimeUnregisterStaticCpuEntries(const VernonStaticCpuEntry *entries, size_t count) {
    return runtimeResultBoundary(nullptr, [&] { return unregisterBackendStaticCpuEntries(entries, count); });
}

VernonStatus vernonRuntimeRegisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                           VernonCpuEntryPoint entryPoint) {
    return runtimeStatusBoundary(context, [&] {
        if (!context || context->backend != VERNON_RUNTIME_CPU || !symbol.data || !symbol.size || !entryPoint)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        std::lock_guard<std::mutex> lock(context->cpuEntriesMutex);
        auto [found, inserted] =
            context->cpuEntries.emplace(std::string(symbol.data, symbol.size), std::make_pair(entryPoint, size_t{1}));
        if (!inserted) {
            if (found->second.first != entryPoint)
                return VERNON_STATUS_INVALID_ARGUMENT;
            ++found->second.second;
        }
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeUnregisterCpuEntry(VernonRuntimeContext *context, VernonStringView symbol,
                                             VernonCpuEntryPoint entryPoint) {
    return runtimeStatusBoundary(context, [&] {
        if (!context || !symbol.data || !symbol.size || !entryPoint)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        std::lock_guard<std::mutex> lock(context->cpuEntriesMutex);
        const auto found = context->cpuEntries.find(std::string(symbol.data, symbol.size));
        if (found == context->cpuEntries.end() || found->second.first != entryPoint)
            return VERNON_STATUS_INVALID_ARGUMENT;
        if (!--found->second.second)
            context->cpuEntries.erase(found);
        return VERNON_STATUS_OK;
    });
}

namespace {

bool parseProgramDeployments(VernonRuntimeContext &context, const nlohmann::json &document,
                             std::vector<vernon::runtime::ProgramVariantDeployment> &result) {
    std::string previousKeyBytes;
    for (const nlohmann::json &variant : document["variants"]) {
        if (!variant.is_object() || variant.size() != 3 || !variant.contains("key") || !variant["key"].is_array() ||
            !variant.contains("program") || !variant["program"].is_object() || !variant.contains("artifact_system") ||
            !variant["artifact_system"].is_object()) {
            fail(&context, "Program bundle variant has unsupported or invalid members", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        vernon::runtime::ProgramVariantDeployment parsed;
        for (const nlohmann::json &assignment : variant["key"]) {
            if (!assignment.is_object() || assignment.size() != 2 || !assignment.contains("name") ||
                !assignment["name"].is_string() || assignment["name"].get_ref<const std::string &>().empty() ||
                !assignment.contains("value") || !assignment["value"].is_object() || assignment["value"].size() != 2 ||
                !assignment["value"].contains("tag") || !assignment["value"]["tag"].is_string() ||
                !assignment["value"].contains("value")) {
                fail(&context, "Program bundle specialization assignment is invalid", VERNON_STATUS_PARSE_ERROR);
                return false;
            }
            vernon::runtime::ProgramSpecialization specialization;
            specialization.name = assignment["name"].get<std::string>();
            const std::string tag = assignment["value"]["tag"].get<std::string>();
            const nlohmann::json &value = assignment["value"]["value"];
            if (tag == "bool" && value.is_boolean()) {
                specialization.kind = vernon::runtime::ProgramSpecializationKind::Bool;
                specialization.value = value.get<bool>();
            } else if (tag == "i32" && value.is_number_integer()) {
                const int64_t integer = value.get<int64_t>();
                if (integer < std::numeric_limits<int32_t>::min() || integer > std::numeric_limits<int32_t>::max()) {
                    fail(&context, "Program bundle i32 specialization is out of range", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::I32;
                specialization.value = static_cast<int32_t>(integer);
            } else if (tag == "u32" && value.is_number_unsigned()) {
                const uint64_t integer = value.get<uint64_t>();
                if (integer > std::numeric_limits<uint32_t>::max()) {
                    fail(&context, "Program bundle u32 specialization is out of range", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::U32;
                specialization.value = static_cast<uint32_t>(integer);
            } else if (tag == "f32" && value.is_number_float()) {
                const float scalar = value.get<float>();
                if (!std::isfinite(scalar)) {
                    fail(&context, "Program bundle f32 specialization must be finite", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::F32;
                specialization.value = scalar == 0.0f ? 0.0f : scalar;
            } else if (tag == "f64" && value.is_number_float()) {
                const double scalar = value.get<double>();
                if (!std::isfinite(scalar)) {
                    fail(&context, "Program bundle f64 specialization must be finite", VERNON_STATUS_PARSE_ERROR);
                    return false;
                }
                specialization.kind = vernon::runtime::ProgramSpecializationKind::F64;
                specialization.value = scalar == 0.0 ? 0.0 : scalar;
            } else if (tag == "type" && value.is_string() && !value.get_ref<const std::string &>().empty()) {
                specialization.kind = vernon::runtime::ProgramSpecializationKind::Type;
                specialization.value = value.get<std::string>();
            } else {
                fail(&context, "Program bundle specialization value does not match its tag", VERNON_STATUS_PARSE_ERROR);
                return false;
            }
            parsed.key.push_back(std::move(specialization));
        }
        if (!std::is_sorted(parsed.key.begin(), parsed.key.end()) ||
            std::adjacent_find(parsed.key.begin(), parsed.key.end(), [](const auto &left, const auto &right) {
                return left.name == right.name;
            }) != parsed.key.end()) {
            fail(&context, "Program bundle specialization key is not canonical", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        const std::string keyBytes = variant["key"].dump();
        if (!previousKeyBytes.empty() && keyBytes <= previousKeyBytes) {
            fail(&context, "Program bundle variants are not canonically ordered", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        previousKeyBytes = keyBytes;
        if (std::any_of(result.begin(), result.end(), [&](const vernon::runtime::ProgramVariantDeployment &existing) {
                return existing.key == parsed.key;
            })) {
            fail(&context, "Program bundle contains duplicate specialization variants", VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        vernon::runtime::program::Diagnostic diagnostic;
        auto program = vernon::runtime::program::parse(variant["program"], diagnostic);
        if (program.isErr()) {
            fail(&context,
                 diagnostic.code + (diagnostic.path.empty() ? ": " : " at " + diagnostic.path + ": ") +
                     diagnostic.message,
                 VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        parsed.program = std::move(program).value();
        auto artifacts = vernon::runtime::program::parseArtifactSystem(document["target"], document["blobs"],
                                                                       variant["artifact_system"], diagnostic);
        if (artifacts.isErr()) {
            fail(&context,
                 diagnostic.code + (diagnostic.path.empty() ? ": " : " at " + diagnostic.path + ": ") +
                     diagnostic.message,
                 VERNON_STATUS_PARSE_ERROR);
            return false;
        }
        parsed.artifactSystem = std::move(artifacts).value();
        result.push_back(std::move(parsed));
    }
    if (result.empty()) {
        fail(&context, "Program bundle declares no variants", VERNON_STATUS_PARSE_ERROR);
        return false;
    }
    return true;
}

extern "C++" {
vernon::Result<const vernon::runtime::ProgramVariantDeployment *, vernon::RuntimeError>
selectProgramDeployment(VernonRuntimeContext &context,
                        const std::vector<vernon::runtime::ProgramVariantDeployment> &variants,
                        const VernonProgramVariantSelector *selector) {
    using Result = vernon::Result<const vernon::runtime::ProgramVariantDeployment *, vernon::RuntimeError>;
    std::vector<vernon::runtime::ProgramSpecialization> requested;
    if (selector) {
        if (selector->struct_size < sizeof(*selector) ||
            (selector->specialization_count && !selector->specializations)) {
            fail(&context, "Program variant selector is invalid");
            return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                           {"Program variant selector is invalid", 0, 0}})};
        }
        requested.reserve(selector->specialization_count);
        for (size_t index = 0; index < selector->specialization_count; ++index) {
            const VernonProgramSpecialization &source = selector->specializations[index];
            if (source.struct_size < sizeof(source) || !source.name.data || !source.name.size) {
                fail(&context, "Program specialization is invalid");
                return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                               {"Program specialization is invalid", 0, 0}})};
            }
            vernon::runtime::ProgramSpecialization destination;
            destination.name.assign(source.name.data, source.name.size);
            switch (source.kind) {
            case VERNON_PROGRAM_SPECIALIZATION_BOOL:
                if (source.value.boolean_value > 1) {
                    fail(&context, "Program bool specialization must be zero or one");
                    return Result{
                        vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                         {"Program bool specialization must be zero or one", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::Bool;
                destination.value = source.value.boolean_value != 0;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_I32:
                destination.kind = vernon::runtime::ProgramSpecializationKind::I32;
                destination.value = source.value.i32_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_U32:
                destination.kind = vernon::runtime::ProgramSpecializationKind::U32;
                destination.value = source.value.u32_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_F32:
                if (!std::isfinite(source.value.f32_value)) {
                    fail(&context, "Program f32 specialization must be finite");
                    return Result{
                        vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                         {"Program f32 specialization must be finite", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::F32;
                destination.value = source.value.f32_value == 0.0f ? 0.0f : source.value.f32_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_F64:
                if (!std::isfinite(source.value.f64_value)) {
                    fail(&context, "Program f64 specialization must be finite");
                    return Result{
                        vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                         {"Program f64 specialization must be finite", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::F64;
                destination.value = source.value.f64_value == 0.0 ? 0.0 : source.value.f64_value;
                break;
            case VERNON_PROGRAM_SPECIALIZATION_TYPE:
                if (!source.value.type_value.data || !source.value.type_value.size) {
                    fail(&context, "Program type specialization requires a type identity");
                    return Result{vernon::err(
                        vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"Program type specialization requires a type identity", 0, 0}})};
                }
                destination.kind = vernon::runtime::ProgramSpecializationKind::Type;
                destination.value = std::string(source.value.type_value.data, source.value.type_value.size);
                break;
            default:
                fail(&context, "Program specialization kind is invalid");
                return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                               {"Program specialization kind is invalid", 0, 0}})};
            }
            requested.push_back(std::move(destination));
        }
    }
    std::sort(requested.begin(), requested.end());
    if (std::adjacent_find(requested.begin(), requested.end(), [](const auto &left, const auto &right) {
            return left.name == right.name;
        }) != requested.end()) {
        fail(&context, "Program variant selector contains duplicate specialization names");
        return Result{vernon::err(
            vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                 {"Program variant selector contains duplicate specialization names", 0, 0}})};
    }
    const auto found =
        std::find_if(variants.begin(), variants.end(), [&](const vernon::runtime::ProgramVariantDeployment &variant) {
            return variant.key == requested;
        });
    if (found == variants.end()) {
        fail(&context, "Program bundle has no matching variant", VERNON_STATUS_PARSE_ERROR);
        return Result{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure,
                                                       {"Program bundle has no matching variant", 0, 0}})};
    }
    return Result{vernon::ok(&*found)};
}

vernon::RuntimeErrorCode programLoadErrorCode(const vernon::runtime::program::ProgramLoadError &error) noexcept {
    if (error.code == "PROGRAM_RUNTIME_LIFECYCLE")
        return error.message.find("cannot publish") != std::string::npos ? vernon::RuntimeErrorCode::LifecycleFailure
                                                                         : vernon::RuntimeErrorCode::ResourceExhausted;
    if (error.code.find("UNSUPPORTED") != std::string::npos || error.code == "PROGRAM_ARTIFACT_TARGET")
        return vernon::RuntimeErrorCode::Unsupported;
    return vernon::RuntimeErrorCode::VerificationFailure;
}

vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>
resolveProgramDeployment(VernonRuntimeContext &context, const vernon::runtime::ProgramVariantDeployment &variant,
                         const std::filesystem::path &bundleRoot) {
    using Result = vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>;
    auto loaded = vernon::runtime::program::loadBackendProgramPipeline(context, variant.program, variant.artifactSystem,
                                                                       bundleRoot);
    if (loaded.isErr()) {
        const vernon::RuntimeErrorCode code = programLoadErrorCode(loaded.error());
        fail(&context, vernon::runtime::program::renderProgramLoadError(loaded.error()),
             vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
        return Result{vernon::err(vernon::RuntimeError{code, {"Program executable resolution", 0, 0}})};
    }
    return Result{vernon::ok(std::move(loaded).value().release())};
}
} // extern "C++"

} // namespace

VernonStatus vernonRuntimeProgramBundleInspectTarget(const void *bundleData, size_t bundleSize,
                                                     VernonRuntimeBackend *target) {
    return runtimeStatusBoundary(nullptr, [&] {
        if (!bundleData || !bundleSize || !target)
            return VERNON_STATUS_INVALID_ARGUMENT;
        const nlohmann::json root = nlohmann::json::parse(
            static_cast<const char *>(bundleData), static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
        if (root.is_discarded() || !root.is_object())
            return VERNON_STATUS_PARSE_ERROR;
        static const std::set<std::string> members{"compiler_contract_version",
                                                   "program_version",
                                                   "type",
                                                   "id",
                                                   "target",
                                                   "blobs",
                                                   "variants",
                                                   "content_hash"};
        std::set<std::string> actual;
        for (const auto &[name, unused] : root.items())
            actual.insert(name);
        if (actual != members || root.value("compiler_contract_version", 0) != VERNON_COMPILER_CONTRACT_VERSION ||
            root.value("program_version", 0) != VERNON_PROGRAM_VERSION || root.value("type", "") != "program" ||
            !root.contains("id") || !root["id"].is_string() || root["id"].get_ref<const std::string &>().empty() ||
            !root.contains("target") || !root["target"].is_object() || root["target"].size() != 2 ||
            !root["target"].contains("kind") || !root["target"]["kind"].is_string() ||
            !root["target"].contains("options") || !root["target"]["options"].is_object() || !root.contains("blobs") ||
            !root["blobs"].is_object() || !root.contains("variants") || !root["variants"].is_array() ||
            validateProgramBundleHash(root, true).isErr())
            return VERNON_STATUS_PARSE_ERROR;
        const std::string name = pipelineTargetKind(root);
        if (name.empty())
            return VERNON_STATUS_PARSE_ERROR;
        if (name == "cpu")
            *target = VERNON_RUNTIME_CPU;
        else if (name == "cuda")
            *target = VERNON_RUNTIME_CUDA;
        else if (name == "vulkan")
            *target = VERNON_RUNTIME_VULKAN;
        else if (name == "opengl")
            *target = VERNON_RUNTIME_OPENGL;
        else if (name == "opengles")
            *target = VERNON_RUNTIME_OPENGL_ES;
        else if (name == "directx")
            *target = VERNON_RUNTIME_DIRECTX12;
        else if (name == "metal")
            *target = VERNON_RUNTIME_METAL;
        else
            return VERNON_STATUS_UNSUPPORTED_TARGET;
        return VERNON_STATUS_OK;
    });
}

extern "C++" {
namespace {
using ProgramBundleHandleResult = vernon::Result<VernonProgramBundle *, vernon::RuntimeError>;

ProgramBundleHandleResult loadProgramBundleOperation(VernonRuntimeContext *context, const void *bundleData,
                                                     size_t bundleSize, const VernonProgramBundleLoadOptions *options) {
    if (!context)
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program bundle load invocation", 0, 0}})};
    auto contextPin = context->operations.tryPin();
    if (contextPin.isErr()) {
        fail(context, "runtime context is closing");
        return ProgramBundleHandleResult{vernon::err(vernon::toRuntimeError(contextPin.error()))};
    }
    auto child = RuntimeChildLifecycle::reserve(context->owner);
    if (child.isErr())
        return ProgramBundleHandleResult{vernon::err(std::move(child).error())};
    if (context->backend != VERNON_RUNTIME_CPU && context->backend != VERNON_RUNTIME_OPENGL &&
        context->backend != VERNON_RUNTIME_OPENGL_ES && context->backend != VERNON_RUNTIME_VULKAN &&
        context->backend != VERNON_RUNTIME_CUDA && context->backend != VERNON_RUNTIME_DIRECTX12 &&
        context->backend != VERNON_RUNTIME_METAL) {
        fail(context, "Program bundle backend is unsupported", VERNON_STATUS_UNSUPPORTED_TARGET);
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::Unsupported, {"Program bundle backend is unsupported", 0, 0}})};
    }
    if (!bundleData || !bundleSize || (options && options->struct_size < sizeof(VernonProgramBundleLoadOptions))) {
        invocationDiagnostic(*context) = "invalid Program bundle load invocation";
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program bundle load invocation", 0, 0}})};
    }
    std::optional<std::filesystem::path> bundleDirectory;
    if (options && options->bundle_directory && options->bundle_directory[0] != '\0')
        bundleDirectory = std::filesystem::u8path(options->bundle_directory);
    const nlohmann::json root = nlohmann::json::parse(
        static_cast<const char *>(bundleData), static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
    if (root.is_discarded()) {
        fail(context, "invalid Program bundle: malformed JSON", VERNON_STATUS_PARSE_ERROR);
        return ProgramBundleHandleResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure, {"Program bundle parse", 0, 0}})};
    }
    const char *expectedTarget = context->backend == VERNON_RUNTIME_CPU         ? "cpu"
                                 : context->backend == VERNON_RUNTIME_CUDA      ? "cuda"
                                 : context->backend == VERNON_RUNTIME_VULKAN    ? "vulkan"
                                 : context->backend == VERNON_RUNTIME_DIRECTX12 ? "directx"
                                 : context->backend == VERNON_RUNTIME_METAL
                                     ? "metal"
                                     : (context->backend == VERNON_RUNTIME_OPENGL_ES ? "opengles" : "opengl");
    static const std::set<std::string> canonicalMembers{
        "compiler_contract_version", "program_version", "type", "id", "target", "blobs", "variants", "content_hash"};
    std::set<std::string> members;
    if (root.is_object())
        for (const auto &[name, unused] : root.items())
            members.insert(name);
    if (members != canonicalMembers || root.value("compiler_contract_version", 0) != VERNON_COMPILER_CONTRACT_VERSION ||
        root.value("program_version", 0) != VERNON_PROGRAM_VERSION || root.value("type", "") != "program" ||
        !root["id"].is_string() || root["id"].get_ref<const std::string &>().empty() ||
        !root["content_hash"].is_string() || root["content_hash"].get_ref<const std::string &>().empty() ||
        !root["target"].is_object() || root["target"].size() != 2 || !root["target"].contains("kind") ||
        !root["target"]["kind"].is_string() || !root["target"].contains("options") ||
        !root["target"]["options"].is_object() || !root["blobs"].is_object() || !root["variants"].is_array()) {
        fail(context, "unsupported or invalid Program bundle", VERNON_STATUS_PARSE_ERROR);
        return ProgramBundleHandleResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure, {"Program bundle parse", 0, 0}})};
    }
    if (root["target"]["kind"].get_ref<const std::string &>() != expectedTarget) {
        fail(context, "Program bundle target does not match the Runtime backend", VERNON_STATUS_UNSUPPORTED_TARGET);
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::Unsupported, {"Program bundle target is unsupported", 0, 0}})};
    }
    auto bundle = std::make_unique<VernonProgramBundle>(std::move(child).value());
    bundle->context = context;
    bundle->id = root["id"].get<std::string>();
    bundle->contentHash = root["content_hash"].get<std::string>();
    if (bundleDirectory)
        bundle->bundleRoot = *bundleDirectory;
    if (!parseProgramDeployments(*context, root, bundle->deployments))
        return ProgramBundleHandleResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::ParseFailure, {"Program bundle parse", 0, 0}})};
    for (vernon::runtime::ProgramVariantDeployment &deployment : bundle->deployments)
        for (auto &[unused, blob] : deployment.artifactSystem.blobs)
            blob.bundleRoot = bundle->bundleRoot;
    auto bundleHash = validateProgramBundleHash(root, true);
    if (bundleHash.isErr()) {
        fail(context, renderStageArtifactError(std::move(bundleHash).error()), VERNON_STATUS_PARSE_ERROR);
        return ProgramBundleHandleResult{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::VerificationFailure,
                                                                          {"Program bundle verification", 0, 0}})};
    }
    auto published = bundle->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish Program bundle", vernon::toVernonStatus(published.error()));
        return ProgramBundleHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramBundleHandleResult{vernon::ok(bundle.release())};
}
} // namespace
} // extern "C++"

VernonRuntimeOperationStatus
vernonRuntimeLoadProgramBundleWithOptionsResult(VernonRuntimeContext *context, const void *bundleData,
                                                size_t bundleSize, const VernonProgramBundleLoadOptions *options,
                                                VernonProgramBundle **output) {
    return runtimeHandleResultBoundary(
        context, output, [&] { return loadProgramBundleOperation(context, bundleData, bundleSize, options); });
}

VernonProgramBundle *vernonRuntimeLoadProgramBundleWithOptions(VernonRuntimeContext *context, const void *bundleData,
                                                               size_t bundleSize,
                                                               const VernonProgramBundleLoadOptions *options) {
    VernonProgramBundle *output = nullptr;
    (void)vernonRuntimeLoadProgramBundleWithOptionsResult(context, bundleData, bundleSize, options, &output);
    return output;
}

VernonStringView vernonRuntimeProgramBundleGetId(const VernonProgramBundle *bundle) {
    return runtimeValueBoundary(bundle ? bundle->context : nullptr, VernonStringView{nullptr, 0}, [&] {
        if (!bundle)
            return VernonStringView{nullptr, 0};
        auto pin = bundle->lifecycle.pin();
        return pin.isOk() ? VernonStringView{bundle->id.data(), bundle->id.size()} : VernonStringView{nullptr, 0};
    });
}

namespace {

VernonValueLayoutView valueLayoutView(const ValueLayout &layout) {
    return {sizeof(VernonValueLayoutView),
            layout.byteSize,
            layout.alignment,
            {layout.layoutHash.data(), layout.layoutHash.size()},
            layout.abiLeaves.empty() ? nullptr : layout.abiLeaves.data(),
            layout.abiLeaves.size()};
}

bool fillBoundaryParameterView(const VernonProgramExecutable &pipeline, const program::BoundarySlot &source,
                               VernonProgramParameterView &destination) {
    const ValueLayout *layout = boundaryLayoutView(pipeline, source);
    if ((source.category == program::BoundaryCategory::Value ||
         source.category == program::BoundaryCategory::StorageView) &&
        (!source.layout || !layout))
        return false;
    destination = {source.id,
                   {source.path.data(), source.path.size()},
                   program_execution::boundaryArgumentKind(source.category),
                   layout ? valueLayoutView(*layout) : VernonValueLayoutView{},
                   program_execution::boundaryValueAccess(source.access),
                   static_cast<uint32_t>(source.outerShape.size()),
                   source.outerShape.empty() ? nullptr : source.outerShape.data()};
    return true;
}

VernonStatus fillBoundaryImageConstraintView(const program::BoundarySlot &source,
                                             VernonProgramImageConstraintView &destination) {
    if (source.category != program::BoundaryCategory::Texture || !source.storage ||
        source.storage->descriptorKind != program::StorageDescriptorKind::Image)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto dimension = artifactTextureDimension(source.storage->image.dimension);
    if (!dimension)
        return VERNON_STATUS_PARSE_ERROR;
    destination.dimension = *dimension;
    const auto hasUsage = [&](std::string_view usage) {
        return std::find(source.storage->image.usage.begin(), source.storage->image.usage.end(), usage) !=
               source.storage->image.usage.end();
    };
    destination.binding_role = hasUsage("color_attachment")           ? VERNON_IMAGE_BINDING_COLOR_ATTACHMENT
                               : hasUsage("depth_stencil_attachment") ? VERNON_IMAGE_BINDING_DEPTH_STENCIL_ATTACHMENT
                               : source.access == program::BoundaryAccess::Read ? VERNON_IMAGE_BINDING_SAMPLED
                                                                                : VERNON_IMAGE_BINDING_STORAGE;
    destination.sample_result_class = VERNON_IMAGE_SAMPLE_FLOAT;
    if (destination.binding_role == VERNON_IMAGE_BINDING_STORAGE) {
        const auto format = artifactStorageFormat(source.storage->image.format);
        if (!format)
            return VERNON_STATUS_PARSE_ERROR;
        destination.storage_format = *format;
    } else {
        destination.storage_format = static_cast<VernonTextureFormat>(0);
    }
    std::fill(std::begin(destination.reserved), std::end(destination.reserved), 0);
    return VERNON_STATUS_OK;
}

bool stringViewEquals(VernonStringView view, const std::string &value) {
    return view.size == value.size() && (!view.size || std::memcmp(view.data, value.data(), view.size) == 0);
}

const program::BoundarySlot *programGraphBoundary(const VernonProgramGraph &graph,
                                                  const VernonProgramNodeBindingToken *token) {
    if (!token || token->struct_size < sizeof(*token) || token->graph_id != graph.id ||
        token->node >= graph.nodes.size() || token->kind > VERNON_PROGRAM_SAMPLER)
        return nullptr;
    const RuntimeProgramGraphNode &node = graph.nodes[token->node];
    const auto &boundaries = node.deployment.program.abi.boundarySlots;
    return token->local_slot < boundaries.size() && boundaries[token->local_slot].id == token->local_slot
               ? &boundaries[token->local_slot]
               : nullptr;
}

} // namespace

void vernonRuntimeProgramBundleDestroy(VernonProgramBundle *bundle) {
    runtimeVoidBoundary(bundle ? bundle->context : nullptr, [&] {
        if (!bundle)
            return;
        auto destruction = bundle->lifecycle.beginDestroy();
        if (destruction.isErr()) {
            fail(bundle->context, "Program bundle has an active operation",
                 vernon::toVernonStatus(destruction.error()));
            return;
        }
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete bundle;
    });
}

extern "C++" {
namespace {
using ProgramGraphHandleResult = vernon::Result<VernonProgramGraph *, vernon::RuntimeError>;

ProgramGraphHandleResult createProgramGraphOperation(VernonRuntimeContext *context) {
    if (!context)
        return ProgramGraphHandleResult{vernon::err(
            vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument, {"invalid ProgramGraph creation", 0, 0}})};
    auto contextPin = context->operations.tryPin();
    if (contextPin.isErr()) {
        fail(context, "runtime context is closing", vernon::toVernonStatus(contextPin.error()));
        return ProgramGraphHandleResult{vernon::err(vernon::toRuntimeError(contextPin.error()))};
    }
    auto child = RuntimeChildLifecycle::reserve(context->owner);
    if (child.isErr()) {
        fail(context, "runtime context cannot admit a ProgramGraph", vernon::toVernonStatus(child.error()));
        return ProgramGraphHandleResult{vernon::err(std::move(child).error())};
    }
    auto graph = std::make_unique<VernonProgramGraph>(std::move(child).value());
    graph->context = context;
    graph->id = context->nextProgramGraphId++;
    auto published = graph->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish ProgramGraph", vernon::toVernonStatus(published.error()));
        return ProgramGraphHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramGraphHandleResult{vernon::ok(graph.release())};
}
} // namespace
} // extern "C++"

VernonRuntimeOperationStatus vernonRuntimeProgramGraphCreateResult(VernonRuntimeContext *context,
                                                                   VernonProgramGraph **output) {
    return runtimeHandleResultBoundary(context, output, [&] { return createProgramGraphOperation(context); });
}

VernonProgramGraph *vernonRuntimeProgramGraphCreate(VernonRuntimeContext *context) {
    VernonProgramGraph *output = nullptr;
    (void)vernonRuntimeProgramGraphCreateResult(context, &output);
    return output;
}

void vernonRuntimeProgramGraphDestroy(VernonProgramGraph *graph) {
    runtimeVoidBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return;
        auto destruction = graph->lifecycle.beginDestroy();
        if (destruction.isErr()) {
            fail(graph->context, "ProgramGraph has an active operation", vernon::toVernonStatus(destruction.error()));
            return;
        }
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete graph;
    });
}

VernonStatus vernonRuntimeProgramGraphAddProgram(VernonProgramGraph *graph, const VernonProgramBundle *bundle,
                                                 const VernonProgramVariantSelector *selector,
                                                 VernonProgramNodeId *node) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !bundle || !node)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph node");
        auto graphPin = graph->lifecycle.pin();
        auto bundlePin = bundle->lifecycle.pin();
        if (graphPin.isErr() || bundlePin.isErr())
            return fail(graph->context, "ProgramGraph or Program bundle is closing");
        if (bundle->context != graph->context || graph->nodes.size() >= UINT32_MAX)
            return fail(graph->context, "invalid ProgramGraph node");
        auto deploymentResult = selectProgramDeployment(*graph->context, bundle->deployments, selector);
        if (deploymentResult.isErr())
            return vernon::toVernonStatus(std::move(deploymentResult).error());
        const ProgramVariantDeployment *deployment = std::move(deploymentResult).value();
        RuntimeProgramGraphNode source;
        source.id = static_cast<uint32_t>(graph->nodes.size());
        source.bundleId = bundle->id;
        source.contentHash = bundle->contentHash;
        source.deployment = *deployment;
        source.bundleRoot = bundle->bundleRoot;
        graph->nodes.push_back(std::move(source));
        *node = graph->nodes.back().id;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphFindBoundary(const VernonProgramGraph *graph, VernonProgramNodeId node,
                                                   VernonProgramBoundaryRole role, VernonStringView name,
                                                   VernonProgramNodeBindingToken *token) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        const auto expectedRole = reflectedBoundaryRole(role);
        if (!graph)
            return fail(nullptr, "invalid ProgramGraph boundary lookup");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        if (!expectedRole || !token || token->struct_size < sizeof(*token) || (name.size && !name.data) ||
            node >= graph->nodes.size())
            return fail(graph->context, "invalid ProgramGraph boundary lookup");
        const program::Program &program = graph->nodes[node].deployment.program;
        const auto found = std::find_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                                        [&](const program::BoundarySlot &slot) {
                                            return slot.role == *expectedRole && stringViewEquals(name, slot.path);
                                        });
        if (found == program.abi.boundarySlots.end())
            return fail(graph->context, "ProgramGraph node boundary was not found");
        token->graph_id = graph->id;
        token->node = node;
        token->local_slot = found->id;
        token->kind = vernon::runtime::program_execution::boundaryArgumentKind(found->category);
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphCreateValue(VernonProgramGraph *graph,
                                                  const VernonProgramNodeBindingToken *source,
                                                  VernonProgramGraphValue *value) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Value source");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, source);
        if (!value || value->struct_size < sizeof(*value) || !boundary ||
            boundary->direction != program::BoundaryDirection::Output)
            return fail(graph->context, "invalid ProgramGraph Value source");
        if (graph->values.size() >= UINT32_MAX)
            return fail(graph->context, "ProgramGraph has too many Values");
        value->graph_id = graph->id;
        value->id = static_cast<uint32_t>(graph->values.size());
        value->kind = source->kind;
        graph->values.push_back({{source->node, source->local_slot}, {}});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphConnectValue(VernonProgramGraph *graph, const VernonProgramGraphValue *value,
                                                   const VernonProgramNodeBindingToken *destination) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Value destination");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, destination);
        if (!value || value->struct_size < sizeof(*value) || value->graph_id != graph->id ||
            value->id >= graph->values.size() || !boundary ||
            boundary->direction != program::BoundaryDirection::Input || value->kind != destination->kind)
            return fail(graph->context, "invalid ProgramGraph Value destination");
        graph->values[value->id].destinations.push_back({destination->node, destination->local_slot});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphCreateStorage(VernonProgramGraph *graph,
                                                    const VernonProgramNodeBindingToken *firstVersion,
                                                    VernonProgramGraphStorage *storage) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Storage source");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, firstVersion);
        if (!storage || storage->struct_size < sizeof(*storage) || !boundary ||
            boundary->direction != program::BoundaryDirection::Output ||
            boundary->aliasOwner.kind != program::ProgramOwnerKind::Storage)
            return fail(graph->context, "invalid ProgramGraph Storage source");
        if (graph->storages.size() >= UINT32_MAX)
            return fail(graph->context, "ProgramGraph has too many Storages");
        storage->graph_id = graph->id;
        storage->id = static_cast<uint32_t>(graph->storages.size());
        storage->kind = firstVersion->kind;
        graph->storages.push_back({firstVersion->kind, {{firstVersion->node, firstVersion->local_slot}}});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphAppendStorage(VernonProgramGraph *graph, const VernonProgramGraphStorage *storage,
                                                    const VernonProgramNodeBindingToken *nextVersion) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Storage version");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const program::BoundarySlot *boundary = programGraphBoundary(*graph, nextVersion);
        if (!storage || storage->struct_size < sizeof(*storage) || storage->graph_id != graph->id ||
            storage->id >= graph->storages.size() || !boundary ||
            boundary->direction != program::BoundaryDirection::Output ||
            boundary->aliasOwner.kind != program::ProgramOwnerKind::Storage || storage->kind != nextVersion->kind)
            return fail(graph->context, "invalid ProgramGraph Storage version");
        RuntimeProgramGraphStorage &target = graph->storages[storage->id];
        if (target.versions.back().node >= nextVersion->node)
            return fail(graph->context, "ProgramGraph Storage versions must follow node order");
        target.versions.push_back({nextVersion->node, nextVersion->local_slot});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphExportBoundary(VernonProgramGraph *graph,
                                                     const VernonProgramNodeBindingToken *boundaryToken,
                                                     VernonStringView graphName) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !boundaryToken || boundaryToken->struct_size < sizeof(*boundaryToken) ||
            boundaryToken->graph_id != graph->id || boundaryToken->node >= graph->nodes.size() || !graphName.data ||
            !graphName.size)
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph boundary export");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        graph->exports.push_back(
            {{boundaryToken->node, boundaryToken->local_slot}, std::string(graphName.data, graphName.size)});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramGraphExportValue(VernonProgramGraph *graph, const VernonProgramGraphValue *value,
                                                  VernonStringView graphName) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !value || value->struct_size < sizeof(*value) || value->graph_id != graph->id ||
            value->id >= graph->values.size())
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Value export");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const ProgramGraphBoundaryKey source = graph->values[value->id].source;
        const VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken), graph->id, source.node,
                                                  source.slot, value->kind};
        return vernonRuntimeProgramGraphExportBoundary(graph, &token, graphName);
    });
}

VernonStatus vernonRuntimeProgramGraphExportStorage(VernonProgramGraph *graph, const VernonProgramGraphStorage *storage,
                                                    VernonStringView graphName) {
    return runtimeStatusBoundary(graph ? graph->context : nullptr, [&] {
        if (!graph || !storage || storage->struct_size < sizeof(*storage) || storage->graph_id != graph->id ||
            storage->id >= graph->storages.size() || graph->storages[storage->id].versions.empty())
            return fail(graph ? graph->context : nullptr, "invalid ProgramGraph Storage export");
        auto graphPin = graph->lifecycle.pin();
        if (graphPin.isErr())
            return fail(graph->context, "ProgramGraph is closing");
        const ProgramGraphBoundaryKey final = graph->storages[storage->id].versions.back();
        const VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken), graph->id, final.node,
                                                  final.slot, storage->kind};
        return vernonRuntimeProgramGraphExportBoundary(graph, &token, graphName);
    });
}

extern "C++" {
namespace {
using ProgramGraphExecutableResult = vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>;

ProgramGraphExecutableResult resolveProgramGraphOperation(VernonProgramGraph *graph) {
    if (!graph)
        return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid ProgramGraph resolve invocation", 0, 0}})};
    auto graphPin = graph->lifecycle.pin();
    if (graphPin.isErr()) {
        fail(graph->context, "ProgramGraph is closing", vernon::toVernonStatus(graphPin.error()));
        return ProgramGraphExecutableResult{vernon::err(std::move(graphPin).error())};
    }
    if (graph->nodes.empty()) {
        fail(graph->context, "invalid ProgramGraph resolve invocation");
        return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid ProgramGraph resolve invocation", 0, 0}})};
    }
    std::vector<ProgramGraphNodeSource> sources;
    sources.reserve(graph->nodes.size());
    for (RuntimeProgramGraphNode &node : graph->nodes) {
        sources.push_back({node.id, node.bundleId, node.contentHash, &node.deployment, node.bundleRoot});
    }
    std::vector<ProgramGraphConnection> connections;
    for (const RuntimeProgramGraphValue &value : graph->values)
        for (ProgramGraphBoundaryKey destination : value.destinations)
            connections.push_back({value.source, destination, false});
    for (const RuntimeProgramGraphStorage &storage : graph->storages)
        for (size_t index = 1; index < storage.versions.size(); ++index)
            connections.push_back({storage.versions[index - 1], storage.versions[index], true});
    std::vector<ProgramGraphBoundaryKey> retainedBoundaries;
    retainedBoundaries.reserve(graph->storages.size());
    for (const RuntimeProgramGraphStorage &storage : graph->storages)
        retainedBoundaries.push_back(storage.versions.back());
    LinkedProgramDeployment linked;
    program::Diagnostic linkDiagnostic;
    if (!linkProgramGraph(sources, connections, retainedBoundaries, graph->exports, linked, linkDiagnostic)) {
        fail(graph->context,
             linkDiagnostic.code + (linkDiagnostic.path.empty() ? ": " : " at " + linkDiagnostic.path + ": ") +
                 linkDiagnostic.message,
             VERNON_STATUS_PARSE_ERROR);
        return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::VerificationFailure, {"ProgramGraph link verification", 0, 0}})};
    }
    auto loaded = program::loadBackendProgramPipeline(*graph->context, linked.deployment.program,
                                                      linked.deployment.artifactSystem, {});
    if (loaded.isErr()) {
        const vernon::RuntimeErrorCode code = programLoadErrorCode(loaded.error());
        fail(graph->context, program::renderProgramLoadError(loaded.error()),
             vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
        return ProgramGraphExecutableResult{
            vernon::err(vernon::RuntimeError{code, {"ProgramGraph executable resolution", 0, 0}})};
    }
    std::unique_ptr<VernonProgramExecutable, decltype(&vernonRuntimeProgramExecutableDestroy)> executable(
        std::move(loaded).value().release(), vernonRuntimeProgramExecutableDestroy);
    for (const RuntimeProgramGraphNode &node : graph->nodes) {
        if (!program::findGraph(node.deployment.program, "backward"))
            continue;
        auto childResult = program::loadBackendProgramPipeline(*graph->context, node.deployment.program,
                                                               node.deployment.artifactSystem, node.bundleRoot);
        if (childResult.isErr()) {
            const vernon::RuntimeErrorCode code = programLoadErrorCode(childResult.error());
            fail(graph->context, program::renderProgramLoadError(childResult.error()),
                 vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
            return ProgramGraphExecutableResult{
                vernon::err(vernon::RuntimeError{code, {"ProgramGraph child resolution", 0, 0}})};
        }
        VernonProgramExecutable *child = std::move(childResult).value().release();
        const auto mapping = linked.nodeMappings.find(node.id);
        if (mapping == linked.nodeMappings.end()) {
            vernonRuntimeProgramExecutableDestroy(child);
            fail(graph->context, "ProgramGraph child remapping is unavailable", VERNON_STATUS_INTERNAL_ERROR);
            return ProgramGraphExecutableResult{vernon::err(vernon::RuntimeError{
                vernon::RuntimeErrorCode::InternalFailure, {"ProgramGraph child remapping is unavailable", 0, 0}})};
        }
        ProgramGraphNodeAutodiffState state;
        state.executable = std::shared_ptr<VernonProgramExecutable>(child, vernonRuntimeProgramExecutableDestroy);
        state.globalValues = mapping->second.values;
        state.globalStorages = mapping->second.storages;
        executable->programGraphNodeAutodiff.emplace(node.id, std::move(state));
    }
    executable->id = linked.id;
    executable->programGraphId = graph->id;
    executable->publicParameterSlots.reserve(graph->exports.size());
    for (const ProgramGraphExport &exported : graph->exports) {
        const uint32_t slot = linked.boundarySlots.at(exported.boundary);
        if (std::find(executable->publicParameterSlots.begin(), executable->publicParameterSlots.end(), slot) ==
            executable->publicParameterSlots.end())
            executable->publicParameterSlots.push_back(slot);
    }
    buildExecutableBindingIndex(*executable);
    return ProgramGraphExecutableResult{vernon::ok(executable.release())};
}
} // namespace
} // extern "C++"

VernonRuntimeOperationStatus vernonRuntimeResolveProgramGraphResult(VernonProgramGraph *graph,
                                                                    VernonProgramExecutable **output) {
    VernonRuntimeContext *context = graph ? graph->context : nullptr;
    return runtimeHandleResultBoundary(context, output, [&] { return resolveProgramGraphOperation(graph); });
}

VernonProgramExecutable *vernonRuntimeResolveProgramGraph(VernonProgramGraph *graph) {
    VernonProgramExecutable *output = nullptr;
    (void)vernonRuntimeResolveProgramGraphResult(graph, &output);
    return output;
}

extern "C++" {
namespace {
using ProgramExecutableHandleResult = vernon::Result<VernonProgramExecutable *, vernon::RuntimeError>;

ProgramExecutableHandleResult resolveProgramOperation(VernonProgramBundle *bundle,
                                                      const VernonProgramVariantSelector *selector) {
    if (!bundle)
        return ProgramExecutableHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program resolve invocation", 0, 0}})};
    auto bundlePin = bundle->lifecycle.pin();
    if (bundlePin.isErr()) {
        fail(bundle->context, "Program bundle is closing", vernon::toVernonStatus(bundlePin.error()));
        return ProgramExecutableHandleResult{vernon::err(std::move(bundlePin).error())};
    }
    auto selected = selectProgramDeployment(*bundle->context, bundle->deployments, selector);
    if (selected.isErr())
        return ProgramExecutableHandleResult{vernon::err(std::move(selected).error())};
    auto executable = resolveProgramDeployment(*bundle->context, *std::move(selected).value(), bundle->bundleRoot);
    if (executable.isErr())
        return executable;
    executable.value()->id = bundle->id;
    buildExecutableBindingIndex(*executable.value());
    return executable;
}
} // namespace
} // extern "C++"

VernonRuntimeOperationStatus vernonRuntimeResolveProgramResult(VernonProgramBundle *bundle,
                                                               const VernonProgramVariantSelector *selector,
                                                               VernonProgramExecutable **output) {
    VernonRuntimeContext *context = bundle ? bundle->context : nullptr;
    return runtimeHandleResultBoundary(context, output, [&] { return resolveProgramOperation(bundle, selector); });
}

VernonProgramExecutable *vernonRuntimeResolveProgram(VernonProgramBundle *bundle,
                                                     const VernonProgramVariantSelector *selector) {
    VernonProgramExecutable *output = nullptr;
    (void)vernonRuntimeResolveProgramResult(bundle, selector, &output);
    return output;
}

size_t vernonRuntimeProgramExecutableGetParameterCount(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        if (!pipeline)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return size_t{0};
        return pipeline->programGraphId ? pipeline->publicParameterSlots.size()
                                        : publicBoundaryCount(*executableProgram(*pipeline));
    });
}

size_t vernonRuntimeProgramExecutableGetMutationCapacity(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        if (!pipeline)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        return pipelinePin.isOk() ? mutationCapacity(*pipeline) : size_t{0};
    });
}

VernonStatus vernonRuntimeProgramExecutableGetParameterByIndex(const VernonProgramExecutable *pipeline, size_t index,
                                                               VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !parameter)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = parameterBoundaryAt(*pipeline, index);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableFindParameter(const VernonProgramExecutable *pipeline, VernonStringView name,
                                                         VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !parameter || (name.size && !name.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = findParameterBoundary(*pipeline, name);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableGetParameterValueLeaf(const VernonProgramExecutable *pipeline,
                                                                 VernonStringView parameterName, size_t leafIndex,
                                                                 VernonProgramValueLeafView *leaf) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !leaf || leaf->struct_size < sizeof(VernonProgramValueLeafView) ||
            (parameterName.size && !parameterName.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = findParameterBoundary(*pipeline, parameterName);
        const ValueLayout *layout = slot ? boundaryLayoutView(*pipeline, *slot) : nullptr;
        if (!slot || !layout ||
            (slot->category != program::BoundaryCategory::Value &&
             slot->category != program::BoundaryCategory::StorageView) ||
            leafIndex >= layout->leaves.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const ValueLeaf &source = layout->leaves[leafIndex];
        leaf->value = layout->abiLeaves[leafIndex];
        leaf->path = source.abiPath.empty() ? nullptr : source.abiPath.data();
        leaf->path_count = source.abiPath.size();
        leaf->static_shape = source.shape.empty() ? nullptr : source.shape.data();
        leaf->static_rank = static_cast<uint32_t>(source.shape.size());
        return VERNON_STATUS_OK;
    });
}

size_t vernonRuntimeProgramExecutableGetBoundaryCount(const VernonProgramExecutable *pipeline,
                                                      VernonProgramBoundaryRole boundary) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return size_t{0};
        const program::Program &program = *executableProgram(*pipeline);
        return static_cast<size_t>(
            std::count_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                          [&](const program::BoundarySlot &slot) { return slot.role == *role; }));
    });
}

VernonStatus vernonRuntimeProgramExecutableGetBoundaryByIndex(const VernonProgramExecutable *pipeline,
                                                              VernonProgramBoundaryRole boundary, size_t index,
                                                              VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role || !parameter)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = boundaryAt(*executableProgram(*pipeline), *role, index);
        if (!slot)
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *slot, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableFindBoundary(const VernonProgramExecutable *pipeline,
                                                        VernonProgramBoundaryRole boundary, VernonStringView name,
                                                        VernonProgramParameterView *parameter) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role || !parameter || (name.size && !name.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::Program &program = *executableProgram(*pipeline);
        const auto found = std::find_if(
            program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
            [&](const program::BoundarySlot &slot) { return slot.role == *role && stringViewEquals(name, slot.path); });
        if (found == program.abi.boundarySlots.end())
            return VERNON_STATUS_INVALID_ARGUMENT;
        return fillBoundaryParameterView(*pipeline, *found, *parameter) ? VERNON_STATUS_OK : VERNON_STATUS_PARSE_ERROR;
    });
}

VernonStatus vernonRuntimeProgramExecutableGetBoundaryValueLeaf(const VernonProgramExecutable *pipeline,
                                                                VernonProgramBoundaryRole boundary, uint32_t slotId,
                                                                size_t leafIndex, VernonProgramValueLeafView *leaf) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        const auto role = reflectedBoundaryRole(boundary);
        if (!pipeline || !role || !leaf || leaf->struct_size < sizeof(*leaf))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::Program &program = *executableProgram(*pipeline);
        if (slotId >= program.abi.boundarySlots.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const program::BoundarySlot &slot = program.abi.boundarySlots[slotId];
        const ValueLayout *layout = boundaryLayoutView(*pipeline, slot);
        if (slot.id != slotId || slot.role != *role || !layout || leafIndex >= layout->leaves.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const ValueLeaf &source = layout->leaves[leafIndex];
        leaf->value = layout->abiLeaves[leafIndex];
        leaf->path = source.abiPath.empty() ? nullptr : source.abiPath.data();
        leaf->path_count = source.abiPath.size();
        leaf->static_shape = source.shape.empty() ? nullptr : source.shape.data();
        leaf->static_rank = static_cast<uint32_t>(source.shape.size());
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramExecutableGetImageConstraintByParameterIndex(
    const VernonProgramExecutable *pipeline, size_t parameterIndex, VernonProgramImageConstraintView *constraint) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonProgramImageConstraintView))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = parameterBoundaryAt(*pipeline, parameterIndex);
        return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
    });
}

VernonStatus vernonRuntimeProgramExecutableFindImageConstraint(const VernonProgramExecutable *pipeline,
                                                               VernonStringView parameterName,
                                                               VernonProgramImageConstraintView *constraint) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !constraint || constraint->struct_size < sizeof(VernonProgramImageConstraintView) ||
            (parameterName.size && !parameterName.data))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        const program::BoundarySlot *slot = findParameterBoundary(*pipeline, parameterName);
        return slot ? fillBoundaryImageConstraintView(*slot, *constraint) : VERNON_STATUS_INVALID_ARGUMENT;
    });
}

namespace {

/* Collects the forward graph's graphics nodes in graph order, which is the order the controls query indexes by. */
void collectForwardGraphicsNodes(const VernonProgramExecutable *pipeline, std::vector<const program::Node *> &nodes) {
    const program::Program *program = pipeline ? executableProgram(*pipeline) : nullptr;
    if (!program)
        return;
    const program::Graph *graph = program::findGraph(*program, "forward");
    if (!graph)
        return;
    for (const program::Node &node : graph->nodes) {
        if (program::executionKind(node) == program::ExecutionKind::Graphics)
            nodes.push_back(&node);
    }
}

} // namespace

size_t vernonRuntimeProgramExecutableGetGraphicsNodeCount(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary<size_t>(pipeline ? pipeline->context : nullptr, 0, [&] {
        if (!pipeline)
            return size_t{0};
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return size_t{0};
        std::vector<const program::Node *> nodes;
        collectForwardGraphicsNodes(pipeline, nodes);
        return nodes.size();
    });
}

VernonStatus vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(const VernonProgramExecutable *pipeline,
                                                                      size_t index,
                                                                      VernonProgramGraphicsControlsView *output) {
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline || !output)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pipelinePin = pipeline->lifecycle.pin();
        if (pipelinePin.isErr())
            return vernon::toVernonStatus(pipelinePin.error());
        std::vector<const program::Node *> nodes;
        collectForwardGraphicsNodes(pipeline, nodes);
        if (index >= nodes.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const program::Node &node = *nodes[index];
        const program::GraphicsOperation &graphics = program::graphicsOperation(node);
        *output = {sizeof(VernonProgramGraphicsControlsView), node.id, graphics.renderPassControl,
                   graphics.drawCommandControl, graphics.dynamicStateControl};
        return VERNON_STATUS_OK;
    });
}

void vernonRuntimeProgramExecutableDestroy(VernonProgramExecutable *pipeline) {
    runtimeVoidBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!pipeline)
            return;
        auto destruction = pipeline->lifecycle.beginDestroy();
        if (destruction.isErr()) {
            fail(pipeline->context, "Program executable has an active operation",
                 vernon::toVernonStatus(destruction.error()));
            return;
        }
        auto close = pipeline->instanceOwner.beginClose();
        if (close.isErr()) {
            (void)destruction.value().rollback();
            fail(pipeline->context, "Program executable still owns live instances",
                 vernon::toVernonStatus(close.error()));
            return;
        }
        if (close.value().commit().isErr() || destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete pipeline;
    });
}

VernonStringView vernonRuntimeProgramExecutableGetId(const VernonProgramExecutable *pipeline) {
    return runtimeValueBoundary(pipeline ? pipeline->context : nullptr, VernonStringView{nullptr, 0}, [&] {
        if (!pipeline)
            return VernonStringView{nullptr, 0};
        auto pin = pipeline->lifecycle.pin();
        return pin.isOk() ? VernonStringView{pipeline->id.data(), pipeline->id.size()} : VernonStringView{nullptr, 0};
    });
}

void vernon::runtime::destroyResolvedStage(VernonStageExecutable *stage) {
    if (!stage)
        return;
    RuntimeDiagnosticScope diagnostic(stage->context);
    if (!stage->lifecycle)
        vernon::resultContractViolation();
    auto destruction = stage->lifecycle.value().beginDestroy();
    if (destruction.isErr()) {
        fail(stage->context, "resolved Stage has an active operation", vernon::toVernonStatus(destruction.error()));
        return;
    }
    if (destruction.value().commit().isErr())
        vernon::resultContractViolation();
    delete stage;
}

extern "C++" {

namespace {

std::string programNodeName(const program::Node &node) {
    return "program." + std::to_string(node.id) + "." + (node.name.empty() ? node.stage : node.name);
}

bool resolveProgramGrid(vernon::runtime::program::DispatchMapping mapping, const uint64_t staticGrid[3],
                        const std::vector<VernonProgramArgument> &arguments, VernonLaunchSize &grid,
                        std::string &error) {
    grid = {static_cast<uint32_t>(staticGrid[0]), static_cast<uint32_t>(staticGrid[1]),
            static_cast<uint32_t>(staticGrid[2])};
    if (mapping == vernon::runtime::program::DispatchMapping::StaticGrid)
        return true;
    for (const VernonProgramArgument &argument : arguments) {
        if (argument.kind != VERNON_PROGRAM_TENSOR || (argument.tensor.rank && !argument.tensor.shape))
            continue;
        uint64_t count = 1;
        for (uint32_t dimension = 0; dimension < argument.tensor.rank; ++dimension) {
            const uint64_t extent = argument.tensor.shape[dimension];
            if (!extent) {
                count = 0;
                break;
            }
            if (count > std::numeric_limits<uint32_t>::max() / extent) {
                error = "linearized Program dispatch exceeds the portable uint32 range";
                return false;
            }
            count *= extent;
        }
        grid = {static_cast<uint32_t>(count), 1, 1};
        return true;
    }
    error = "linearized Program dispatch has no Tensor extent source";
    return false;
}

VernonStatus executeCpuProgramNode(VernonRuntimeContext &context, const program::Program &program,
                                   const program::Node &node, VernonStageExecutable &pipeline,
                                   vernon::runtime::program_execution::ProgramInvocationState &arena,
                                   vernon::runtime::program_execution::MaterializedNodeFrame &materialized,
                                   VernonLaunchSize grid,
                                   vernon::runtime::program_execution::SubmissionState &submission) {
    std::string error;
    if (!materialized.prepareHost(error))
        return fail(&context, programNodeName(node) + ": " + error, VERNON_STATUS_INTERNAL_ERROR);
    if (!grid.x || !grid.y || !grid.z)
        return VERNON_STATUS_OK;

    VernonAdTapeAllocator *allocator = nullptr;
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    const auto bindTape = [&](uint32_t valueId) {
        const std::vector<VernonProgramArgument> &arguments = arena.arguments();
        if (valueId >= arguments.size() || valueId >= program.values.size() ||
            !program::isTapeValue(program.values[valueId]))
            return;
        const VernonProgramArgument &argument = arguments[valueId];
        constexpr size_t allocatorDescriptorBytes = sizeof(VernonAdTapeAllocator *);
        constexpr size_t rootDescriptorBytes = sizeof(VernonAdRegionHandle);
        if (argument.kind != VERNON_PROGRAM_TENSOR || !argument.tensor.host_data ||
            argument.tensor.byte_size < allocatorDescriptorBytes + rootDescriptorBytes)
            return;
        const auto *bytes = static_cast<const uint8_t *>(argument.tensor.host_data);
        std::memcpy(&allocator, bytes, allocatorDescriptorBytes);
        std::memcpy(&root, bytes + allocatorDescriptorBytes, rootDescriptorBytes);
    };
    for (uint32_t operand : node.operands)
        bindTape(operand);
    for (uint32_t result : node.results)
        bindTape(result);
    vernon::runtime::setCpuProgramTape(pipeline, allocator, root);

    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = materialized.arguments.data();
    invocation.argument_count = materialized.arguments.size();
    invocation.compute_grid = grid;
    PlannedComputeLaunch plan;
    if (!planComputeInvocation(pipeline.bindingProjection, invocation, plan, error))
        return fail(&context, std::move(error));
    const uint32_t launchGrid[3]{plan.grid.x, plan.grid.y, plan.grid.z};
    const uint32_t workgroup[3]{pipeline.workgroupSize.x, pipeline.workgroupSize.y, pipeline.workgroupSize.z};
    if (!validateDispatchContract(pipeline.dispatchContract, launchGrid, workgroup, error))
        return fail(&context, std::move(error));

    submission = vernon::runtime::program_execution::SubmissionState::Indeterminate;
    auto invoked = invokeBackendComputePipeline(pipeline, plan);
    if (invoked.isErr()) {
        const VernonStatus status = vernon::toVernonStatus(std::move(invoked).error());
        std::string &detail = invocationDiagnostic(context);
        if (detail.empty())
            detail = "pipeline compute failed";
        detail = programNodeName(node) + ": " + detail;
        return status;
    }
    if (!commitComputeResults(plan, error))
        return fail(&context, programNodeName(node) + ": " + error, VERNON_STATUS_INTERNAL_ERROR);
    if (!materialized.commitHost(error))
        return fail(&context, programNodeName(node) + ": " + error, VERNON_STATUS_INTERNAL_ERROR);
    return VERNON_STATUS_OK;
}

struct ManagedGraphicsCommandContext {
    VernonStageExecutable &pipeline;
    vernon::runtime::program_execution::MaterializedNodeFrame materialized;
    std::shared_ptr<RuntimeProgramControl> renderPass;
    std::shared_ptr<const RuntimeProgramControl> draw;
    std::shared_ptr<const RuntimeProgramControl> dynamic;
    VernonStageInvocationDescriptor invocation{};
    PlannedGraphicsInvocation plan;
    VernonGraphicsState graphicsState{};
    std::vector<VernonColorBlendState> colorBlends;
    VernonDrawCommand defaultDraw{};
    // Recorded draws only: `materialized.arguments` point into these frames.
    std::vector<std::shared_ptr<const vernon::runtime::program_execution::MaterializedNodeFrame>> argumentFrames;
};

VernonStatus describeRuntimeImage(void *userData, VernonRuntimeProviderResourceReference resource,
                                  VernonRuntimeProviderImageDescription *description) {
    auto described = describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource);
    if (described.isErr())
        return vernon::toVernonStatus(std::move(described).error());
    *description = described.value();
    return VERNON_STATUS_OK;
}

struct ManagedGraphicsCommandBatch {
    VernonRuntimeContext *context{};
    GraphicsRenderScope scope;
    std::vector<std::shared_ptr<ManagedGraphicsCommandContext>> draws;
};

template <typename Handle> bool decodeRhiResourceKey(uint64_t value, Handle &handle) {
    const uint32_t encodedIndex = static_cast<uint32_t>(value);
    const uint32_t generation = static_cast<uint32_t>(value >> 32);
    if (!encodedIndex || !generation)
        return false;
    handle = {encodedIndex - 1, generation};
    return true;
}

bool appendGraphicsVertexResources(const PlannedGraphicsInvocation &graphics,
                                   vernon::execution::detail::CommandNode &command,
                                   std::vector<vernon::execution::detail::RhiCommandResourceBinding> &bindings) {
    for (const PlannedVertexInput &input : graphics.vertexInputs) {
        if (!input.tensor)
            return false;
        const VernonTensorView &tensor = *input.tensor;
        VernonRhiBuffer buffer{};
        if (!decodeRhiResourceKey(tensor.resource.resource.value, buffer) ||
            tensor.resource.offset > UINT64_MAX - tensor.byte_offset)
            return false;
        command.accesses.push_back(vernon::execution::detail::rhiBufferAccess(
            buffer, tensor.resource.offset + tensor.byte_offset, tensor.byte_size, vernon::execution::AccessMode::Read,
            VERNON_RHI_STATE_SHADER_READ, VERNON_RHI_STAGE_VERTEX, VERNON_RHI_ACCESS_VERTEX_READ));
        vernon::execution::detail::appendRhiBufferBinding(bindings, buffer);
    }
    if (graphics.indexBinding) {
        VernonRhiBuffer buffer{};
        const VernonRuntimeProviderResourceReference &resource = graphics.indexBinding->resource;
        const uint64_t indexSize = sizeof(uint32_t);
        if (!decodeRhiResourceKey(resource.resource.value, buffer) ||
            resource.offset > UINT64_MAX - graphics.indexBinding->offset)
            return false;
        command.accesses.push_back(vernon::execution::detail::rhiBufferAccess(
            buffer, resource.offset + graphics.indexBinding->offset, graphics.indexBinding->index_count * indexSize,
            vernon::execution::AccessMode::Read, VERNON_RHI_STATE_SHADER_READ, VERNON_RHI_STAGE_VERTEX,
            VERNON_RHI_ACCESS_INDEX_READ));
        vernon::execution::detail::appendRhiBufferBinding(bindings, buffer);
    }
    return true;
}

void prepareGraphicsPipelineState(const program::GraphicsOperation &operation, ManagedGraphicsCommandContext &context) {
    const program::GraphicsPipelineState &source = operation.pipelineState;
    context.colorBlends.clear();
    context.colorBlends.reserve(source.colorBlends.size());
    for (size_t location = 0; location < source.colorBlends.size(); ++location)
        context.colorBlends.push_back(source.colorBlends.at(static_cast<uint32_t>(location)));
    context.graphicsState = {};
    context.graphicsState.struct_size = sizeof(context.graphicsState);
    context.graphicsState.topology = source.topology;
    context.graphicsState.rasterization = source.rasterization;
    context.graphicsState.depth_stencil = source.depthStencil;
    context.graphicsState.color_blends = context.colorBlends.data();
    context.graphicsState.color_blend_count = context.colorBlends.size();
}

VernonRhiStatus encodeManagedGraphicsBatch(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &batch = *static_cast<ManagedGraphicsCommandBatch *>(opaque);
    const VernonRhiDevice device = batch.context->rhiDevice;
    auto providerEncoder = referenceBackendCommandEncoder(*batch.context, encoder);
    if (providerEncoder.isErr()) {
        invocationDiagnostic(*batch.context) = "failed to reference the managed graphics command encoder";
        return vernon::toVernonRhiStatus(std::move(providerEncoder).error());
    }
    if (auto begun =
            beginBackendRendering(*batch.context, providerEncoder.value(), graphicsRenderingDescriptor(batch.scope));
        begun.isErr())
        return vernon::toVernonRhiStatus(std::move(begun).error());
    VernonRhiStatus status = VERNON_RHI_STATUS_OK;
    for (const std::shared_ptr<ManagedGraphicsCommandContext> &context : batch.draws) {
        if (!context || context->pipeline.context->rhiDevice.index != device.index ||
            vernonRuntimeReferenceRhiCommandEncoder(context->pipeline.context, encoder,
                                                    &context->invocation.command_encoder) != VERNON_STATUS_OK) {
            if (context)
                invocationDiagnostic(*context->pipeline.context) =
                    "failed to reference the managed graphics command encoder";
            status = VERNON_RHI_STATUS_INVALID_ARGUMENT;
            break;
        }
        auto invoked = invokeBackendPipeline(context->pipeline, context->invocation, context->plan);
        if (invoked.isErr()) {
            const vernon::RuntimeError error = std::move(invoked).error();
            if (invocationDiagnostic(*context->pipeline.context).empty())
                invocationDiagnostic(*context->pipeline.context) =
                    error.context.operation ? error.context.operation : "managed graphics draw failed";
            status = vernon::toVernonRhiStatus(error);
            break;
        }
    }
    auto ended = endBackendRendering(*batch.context, providerEncoder.value());
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    return ended.isOk() ? VERNON_RHI_STATUS_OK : vernon::toVernonRhiStatus(std::move(ended).error());
}

VernonStatus executePipelineProgramGraphImpl(
    VernonRuntimeContext &context, const program::ResolvedExecutionPlan &execution, const program::Graph &graph,
    vernon::runtime::program_execution::ProgramInvocationState &arena,
    const vernon::runtime::program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint,
    vernon::runtime::program_execution::SubmissionRecord &submission,
    vernon::runtime::program_execution::SubmissionState &step) {
    step = vernon::runtime::program_execution::SubmissionState::NotSubmitted;
    const program::ResolvedProgram &resolved = *execution.resolvedProgram;
    const program::Program &canonicalProgram = resolved.program;
    const std::vector<VernonProgramArgument> &valueArguments = arena.arguments();
    if (valueArguments.size() != canonicalProgram.values.size())
        return fail(&context, "Program graph value set does not match its execution topology");
    const auto canonicalGraph =
        std::find_if(canonicalProgram.graphs.begin(), canonicalProgram.graphs.end(),
                     [&](const program::Graph &candidate) { return candidate.direction == graph.direction; });
    if (canonicalGraph == canonicalProgram.graphs.end())
        return fail(&context, "Program graph is not part of its resolved owner");
    const ProgramInvocationContext *invocationContext = arena.invocationContext();
    const auto control = [&](uint32_t slot,
                             RuntimeProgramControl::Kind kind) -> std::shared_ptr<RuntimeProgramControl> {
        if (!invocationContext)
            return nullptr;
        const std::shared_ptr<void> *payload = invocationContext->bindings.find(programControlKey(slot, kind));
        return payload ? std::static_pointer_cast<RuntimeProgramControl>(*payload) : nullptr;
    };
    std::string controlBindingError;
    if (!bindProgramGraphicsControlResources(
            context, canonicalProgram, graph, execution, arena,
            [&](uint32_t slot) -> const VernonRenderPass * {
                auto renderPass = control(slot, RuntimeProgramControl::RenderPass);
                if (!renderPass)
                    return nullptr;
                return &renderPass->renderPass;
            },
            controlBindingError))
        return fail(&context, std::move(controlBindingError));
    if (context.backend == VERNON_RUNTIME_CPU) {
        const std::optional<program::GraphDirection> direction = program::graphDirection(graph.direction);
        if (!direction)
            return fail(&context, "Program graph has an invalid direction");
        for (const program::Node &node : graph.nodes) {
            const program::ResolvedNodePlan *resolvedNode = execution.node(*direction, node.id);
            if (!resolvedNode || !resolvedNode->stage)
                return fail(&context, "pipeline node has no resolved kernel stage");
            if (program::executionKind(node) != program::ExecutionKind::Compute)
                return fail(&context, "CPU Program execution requires compute nodes");

            vernon::runtime::program_execution::MaterializedNodeFrame materialized;
            std::string error;
            if (!vernon::runtime::program_execution::materializeNodeFrame(arena, canonicalProgram, node, *resolvedNode,
                                                                          resolvePhysicalEndpoint, materialized, error))
                return fail(&context, std::move(error));
            const program::ComputeOperation &compute = program::computeOperation(node);
            uint64_t controlGrid[3]{};
            for (size_t axis = 0; axis < 3; ++axis) {
                auto controlValue = arena.resolveControl(canonicalProgram, compute.workgroups[axis]);
                if (controlValue.isErr())
                    return fail(&context, program_execution::programInvocationErrorMessage(controlValue.error()));
                controlGrid[axis] = controlValue.value();
                if (!controlGrid[axis] || controlGrid[axis] > UINT32_MAX)
                    return fail(&context, "Program compute grid is outside the portable uint32 range");
            }
            const auto *controls = std::get_if<program::ResolvedComputeControls>(&resolvedNode->controls);
            VernonLaunchSize grid{};
            if (!controls ||
                !resolveProgramGrid(controls->dispatchMapping, controlGrid, materialized.arguments, grid, error))
                return fail(&context, std::move(error));
            const VernonStatus status = executeCpuProgramNode(context, canonicalProgram, node, *resolvedNode->stage,
                                                              arena, materialized, grid, step);
            if (status != VERNON_STATUS_OK)
                return status;
        }
        if (vernon::runtime::program_execution::injectFailure(
                vernon::runtime::program_execution::FailureBoundary::Completion))
            return fail(&context, "injected CPU Program wait failure", VERNON_STATUS_INTERNAL_ERROR);
        step = vernon::runtime::program_execution::SubmissionState::Completed;
        return VERNON_STATUS_OK;
    }
    {
        vernon::execution::detail::RhiCommandExecutionPlan commandPlan;
        vernon::runtime::program_execution::ResolvedTransferExecutor transfers(context, arena);
        const std::optional<program::GraphDirection> graphDirection = program::graphDirection(graph.direction);
        std::string transferError;
        if (!graphDirection || !transfers.prepareGraph(*graphDirection, transferError))
            return fail(&context, std::move(transferError));
        std::shared_ptr<ManagedGraphicsCommandBatch> graphicsBatch;
        GraphicsScopeMaterializer graphicsScopeMaterializer;
        vernon::execution::detail::RhiCommandExecutionPlan graphicsNodePlan;
        const auto appendGraphicsBatch = [&]() {
            if (!graphicsBatch)
                return VERNON_STATUS_OK;
            std::string compositionError;
            if (!vernon::execution::detail::appendRhiCommandExecutionPlan(commandPlan, std::move(graphicsNodePlan),
                                                                          true, compositionError))
                return fail(&context, std::move(compositionError));
            graphicsNodePlan = {};
            graphicsBatch.reset();
            return VERNON_STATUS_OK;
        };
        const auto flushCommands = [&]() {
            if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                return pending;
            if (commandPlan.commands.nodes.empty())
                return VERNON_STATUS_OK;
            graphicsScopeMaterializer.reset();
            return submission.execute(std::exchange(commandPlan, {}), step);
        };
        std::vector<vernon::runtime::program_execution::MaterializedNodeFrame> materializedNodes;
        materializedNodes.reserve(graph.nodes.size());
        for (const program::Node &node : graph.nodes) {
            const std::optional<program::GraphDirection> direction = program::graphDirection(graph.direction);
            const program::ResolvedNodePlan *resolvedNode = direction ? execution.node(*direction, node.id) : nullptr;
            if (!resolvedNode)
                return fail(&context, "pipeline node has no resolved kernel stage");
            if (program::executionKind(node) == program::ExecutionKind::Graphics) {
                const program::GraphicsOperation &graphics = program::graphicsOperation(node);
                auto renderPass = control(graphics.renderPassControl, RuntimeProgramControl::RenderPass);
                if (!renderPass)
                    return fail(&context, "managed graphics node has no bound RenderPass control");
                auto draw = control(graphics.drawCommandControl, RuntimeProgramControl::DrawCommand);
                auto dynamic = control(graphics.dynamicStateControl, RuntimeProgramControl::DynamicState);
                const auto *graphicsControls = std::get_if<program::ResolvedGraphicsControls>(&resolvedNode->controls);
                if (!graphicsControls)
                    return fail(&context, "managed graphics node has no resolved attachment controls");
                auto stagedRenderPass = std::make_shared<RuntimeProgramControl>(*renderPass);
                bool usesStaging = false;
                for (const program::ResolvedGraphicsAttachment &attachment : graphicsControls->colorAttachments) {
                    auto view = arena.controlImage(attachment.storage);
                    if (view) {
                        if (attachment.location >= stagedRenderPass->colors.size())
                            return fail(&context, "managed graphics staging attachment has an invalid color location");
                        stagedRenderPass->colors[attachment.location].view = view.value().get();
                        usesStaging = true;
                    }
                }
                if (graphicsControls->depthStencilAttachment) {
                    auto view = arena.controlImage(graphicsControls->depthStencilAttachment->storage);
                    if (view) {
                        if (!stagedRenderPass->depth)
                            return fail(&context, "managed graphics staging has no depth attachment");
                        stagedRenderPass->depth->view = view.value().get();
                        usesStaging = true;
                    }
                }
                if (usesStaging) {
                    stagedRenderPass->refresh();
                    renderPass = std::move(stagedRenderPass);
                }
                if (renderPass->renderPass.color_attachment_count != graphics.colorAttachments.size())
                    return fail(&context, "managed graphics fragment outputs must exactly match the color attachments");
                if (graphics.depthStencilAttachment.has_value() !=
                    static_cast<bool>(renderPass->renderPass.depth_attachment))
                    return fail(&context, "managed graphics depth attachment does not match the canonical render pass");
                RecordedGraphicsPass *recordedPass = nullptr;
                if (invocationContext && invocationContext->recordedGraphics) {
                    auto &recordedNodes = invocationContext->recordedGraphics->nodes;
                    const auto found = recordedNodes.find(node.id);
                    recordedPass = found == recordedNodes.end() ? nullptr : &found->second;
                }
                materializedNodes.emplace_back();
                std::string materializationError;
                if (!vernon::runtime::program_execution::materializeNodeFrame(
                        arena, canonicalProgram, node, *resolvedNode, resolvePhysicalEndpoint, materializedNodes.back(),
                        materializationError))
                    return fail(&context, std::move(materializationError));
                std::vector<std::shared_ptr<const vernon::runtime::program_execution::MaterializedNodeFrame>> setFrames;
                std::vector<vernon::runtime::program_execution::DeviceBufferCopy> copiesBefore =
                    materializedNodes.back().deviceCopiesBefore;
                if (recordedPass) {
                    for (const RecordedSetFrame &frame : recordedPass->frames) {
                        const RuntimeGraphicsPipelineState &pipeline = *frame.pipeline;
                        auto materialized =
                            std::make_shared<vernon::runtime::program_execution::MaterializedNodeFrame>();
                        if (!vernon::runtime::program_execution::materializeNodeFrame(
                                *frame.match->values, *executableProgram(*pipeline.source), *pipeline.sourceNode,
                                *pipeline.sourcePlan, {}, *materialized, materializationError, &frame.match->endpoints))
                            return fail(&context, programNodeName(node) + ": binding set: " + materializationError);
                        copiesBefore.insert(copiesBefore.end(), materialized->deviceCopiesBefore.begin(),
                                            materialized->deviceCopiesBefore.end());
                        setFrames.push_back(std::move(materialized));
                    }
                }
                bool transferCommandsAppended = false;
                if (!copiesBefore.empty())
                    if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                        return pending;
                if (const VernonStatus status = transfers.appendBeforeConsumer(
                        {*direction, node.id}, copiesBefore, commandPlan, transferCommandsAppended, transferError);
                    status != VERNON_STATUS_OK)
                    return status;
                if (transferCommandsAppended)
                    graphicsBatch.reset();
                const auto planContext = [&](ManagedGraphicsCommandContext &graphicsContext,
                                             const program::GraphicsOperation &pipelineGraphics) {
                    prepareGraphicsPipelineState(pipelineGraphics, graphicsContext);
                    VernonStageInvocationDescriptor &invocation = graphicsContext.invocation;
                    invocation.struct_size = sizeof(invocation);
                    invocation.abi_version = VERNON_PROGRAM_VERSION;
                    invocation.arguments = graphicsContext.materialized.arguments.data();
                    invocation.argument_count = graphicsContext.materialized.arguments.size();
                    invocation.render_pass = &graphicsContext.renderPass->renderPass;
                    invocation.graphics_state = &graphicsContext.graphicsState;
                    std::string graphicsError;
                    if (!planGraphicsInvocation(graphicsContext.pipeline.bindingProjection, invocation,
                                                describeRuntimeImage, &context, graphicsContext.plan, graphicsError))
                        return fail(&context, std::move(graphicsError));
                    if (!checkProgramGraphicsAttachmentSignature(graphics, graphicsContext.plan, graphicsError))
                        return fail(&context, std::move(graphicsError));
                    return VERNON_STATUS_OK;
                };
                std::vector<std::shared_ptr<ManagedGraphicsCommandContext>> graphicsContexts;
                if (!recordedPass) {
                    auto graphicsContext = std::make_shared<ManagedGraphicsCommandContext>(
                        ManagedGraphicsCommandContext{*resolvedNode->stage, std::move(materializedNodes.back()),
                                                      std::move(renderPass), std::move(draw), std::move(dynamic)});
                    graphicsContext->defaultDraw = {sizeof(VernonDrawCommand), nullptr,
                                                    static_cast<uint32_t>(graphics.vertexCount),
                                                    static_cast<uint32_t>(graphics.instanceCount)};
                    graphicsContext->invocation.draw_command =
                        graphicsContext->draw ? &graphicsContext->draw->draw : &graphicsContext->defaultDraw;
                    graphicsContext->invocation.dynamic_state =
                        graphicsContext->dynamic ? &graphicsContext->dynamic->dynamic : nullptr;
                    if (const VernonStatus status = planContext(*graphicsContext, graphics); status != VERNON_STATUS_OK)
                        return status;
                    graphicsContexts.push_back(std::move(graphicsContext));
                } else {
                    auto passFrame = std::make_shared<const vernon::runtime::program_execution::MaterializedNodeFrame>(
                        std::move(materializedNodes.back()));
                    for (const RecordedDraw &recordedDraw : recordedPass->draws) {
                        const RuntimeGraphicsPipelineState &pipeline = *recordedDraw.pipeline;
                        VernonStageExecutable &stage = *pipeline.sourcePlan->stage;
                        vernon::runtime::program_execution::MaterializedNodeFrame composed;
                        composed.arguments.reserve(recordedDraw.arguments.size());
                        for (size_t parameter = 0; parameter < recordedDraw.arguments.size(); ++parameter) {
                            const RecordedArgumentSource &source = recordedDraw.arguments[parameter];
                            const auto &frame = source.fromSet ? *setFrames[source.frame] : *passFrame;
                            composed.arguments.push_back(frame.arguments[source.parameter]);
                            composed.arguments.back().slot = stage.bindingProjection.parameters[parameter].slot;
                        }
                        auto graphicsContext =
                            std::make_shared<ManagedGraphicsCommandContext>(ManagedGraphicsCommandContext{
                                stage, std::move(composed), renderPass, recordedDraw.draw, recordedDraw.dynamic});
                        graphicsContext->argumentFrames.push_back(passFrame);
                        graphicsContext->argumentFrames.insert(graphicsContext->argumentFrames.end(), setFrames.begin(),
                                                               setFrames.end());
                        graphicsContext->invocation.draw_command = &graphicsContext->draw->draw;
                        graphicsContext->invocation.dynamic_state = &graphicsContext->dynamic->dynamic;
                        if (const VernonStatus status =
                                planContext(*graphicsContext, program::graphicsOperation(*pipeline.sourceNode));
                            status != VERNON_STATUS_OK)
                            return status;
                        graphicsContexts.push_back(std::move(graphicsContext));
                    }
                    recordedPass->telemetry.render_scopes = 1;
                }
                const auto candidate =
                    std::find_if(execution.graphicsScopeCandidates.begin(), execution.graphicsScopeCandidates.end(),
                                 [&](const program::ResolvedGraphicsScopeCandidate &entry) {
                                     return entry.node == program::NodeKey{*direction, node.id};
                                 });
                if (candidate == execution.graphicsScopeCandidates.end())
                    return fail(&context, "managed graphics node has no resolved scope candidate");
                // A recorded Node without draws still opens its render scope, so its clears apply.
                PlannedRenderTargets undrawnTargets;
                if (graphicsContexts.empty()) {
                    std::string targetError;
                    if (!planRenderTargets(&renderPass->renderPass, describeRuntimeImage, &context, undrawnTargets,
                                           targetError) ||
                        !checkProgramGraphicsAttachmentSignature(graphics, undrawnTargets, targetError))
                        return fail(&context, programNodeName(node) + ": " + targetError);
                }
                const PlannedRenderTargets &scopeTargets =
                    graphicsContexts.empty() ? undrawnTargets : graphicsContexts.front()->plan;
                GraphicsScopeReads scopeReads;
                for (const auto &graphicsContext : graphicsContexts)
                    addGraphicsScopeReads(scopeReads, graphicsContext->plan);
                if (graphicsScopeReadsOverlap(scopeReads, scopeTargets))
                    return fail(&context, programNodeName(node) + ": samples an image it renders to");
                // Only a node following an open batch's scope can fuse into it.
                const GraphicsScopeMaterialization materialization =
                    graphicsScopeMaterializer.materialize(candidate->region, scopeTargets, scopeReads, !graphicsBatch);
                if (materialization != GraphicsScopeMaterialization::Fuse)
                    if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                        return pending;
                if (!graphicsBatch) {
                    graphicsBatch = std::make_shared<ManagedGraphicsCommandBatch>();
                    graphicsBatch->context = &context;
                    vernon::execution::detail::CommandNode drawCommand;
                    drawCommand.kind = vernon::execution::detail::CommandNodeKind::Derivative;
                    drawCommand.queue = vernon::execution::detail::CommandQueueClass::Graphics;
                    graphicsNodePlan.commands.nodes.push_back(std::move(drawCommand));
                    graphicsNodePlan.encoders.push_back({encodeManagedGraphicsBatch, graphicsBatch.get()});
                    graphicsNodePlan.retainedContexts.push_back(graphicsBatch);
                }
                graphicsBatch->scope = graphicsScopeMaterializer.scope();
                for (std::shared_ptr<ManagedGraphicsCommandContext> &graphicsContext : graphicsContexts) {
                    if (!appendGraphicsVertexResources(graphicsContext->plan, graphicsNodePlan.commands.nodes.back(),
                                                       graphicsNodePlan.bindings))
                        return fail(&context, "managed graphics node contains an invalid vertex or index resource");
                    graphicsBatch->draws.push_back(std::move(graphicsContext));
                }
                continue;
            }
            if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                return pending;
            graphicsScopeMaterializer.reset();
            materializedNodes.emplace_back();
            std::string materializationError;
            if (!vernon::runtime::program_execution::materializeNodeFrame(
                    arena, canonicalProgram, node, *resolvedNode, resolvePhysicalEndpoint, materializedNodes.back(),
                    materializationError))
                return fail(&context, std::move(materializationError));
            bool transferCommandsAppended = false;
            if (const VernonStatus status =
                    transfers.appendBeforeConsumer({*direction, node.id}, materializedNodes.back().deviceCopiesBefore,
                                                   commandPlan, transferCommandsAppended, transferError);
                status != VERNON_STATUS_OK)
                return status;
            vernon::execution::detail::RhiCommandExecutionPlan nodePlan;
            VernonLaunchSize grid{};
            std::string gridError;
            const program::ComputeOperation &compute = program::computeOperation(node);
            uint64_t staticGrid[3]{};
            for (size_t axis = 0; axis < 3; ++axis) {
                auto control = arena.resolveControl(canonicalProgram, compute.workgroups[axis]);
                if (control.isErr())
                    return fail(&context, program_execution::programInvocationErrorMessage(control.error()));
                staticGrid[axis] = std::move(control).value();
            }
            const auto *computeControls = std::get_if<program::ResolvedComputeControls>(&resolvedNode->controls);
            if (!computeControls || !resolveProgramGrid(computeControls->dispatchMapping, staticGrid,
                                                        materializedNodes.back().arguments, grid, gridError))
                return fail(&context, std::move(gridError));
            if (!grid.x || !grid.y || !grid.z)
                continue;
            const VernonStatus planned = vernon::runtime::program_execution::buildPipelineCommandPlan(
                context, {}, {}, *resolvedNode->stage, materializedNodes.back().arguments, grid,
                materializedNodes.back().deviceCopiesAfter, vernon::execution::detail::CommandNodeKind::Derivative,
                nodePlan);
            if (planned != VERNON_STATUS_OK)
                return planned;
            std::string compositionError;
            if (!vernon::execution::detail::appendRhiCommandExecutionPlan(commandPlan, std::move(nodePlan), true,
                                                                          compositionError))
                return fail(&context, std::move(compositionError));
        }
        return flushCommands();
    }
    return VERNON_STATUS_INTERNAL_ERROR;
}

VernonStatus encodeStageInvocation(VernonStageExecutable &pipeline, const VernonStageInvocationDescriptor &invocation) {
    if (!pipeline.bindingProjection.compute.empty()) {
        const uint32_t grid[3]{invocation.compute_grid.x, invocation.compute_grid.y, invocation.compute_grid.z};
        const uint32_t workgroup[3]{pipeline.workgroupSize.x, pipeline.workgroupSize.y, pipeline.workgroupSize.z};
        if (!validateDispatchContract(pipeline.dispatchContract, grid, workgroup,
                                      invocationDiagnostic(*pipeline.context)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        PlannedComputeLaunch plan;
        std::string planningError;
        if (!planComputeInvocation(pipeline.bindingProjection, invocation, plan, planningError))
            return fail(pipeline.context, planningError);
        auto invoked = invokeBackendComputePipeline(pipeline, plan);
        return invoked.isOk() ? VERNON_STATUS_OK : vernon::toVernonStatus(std::move(invoked).error());
    }

    PlannedGraphicsInvocation plan;
    std::string planningError;
    if (!planGraphicsInvocation(pipeline.bindingProjection, invocation, describeRuntimeImage, pipeline.context, plan,
                                planningError))
        return fail(pipeline.context, planningError);
    auto invoked = invokeBackendPipeline(pipeline, invocation, plan);
    return invoked.isOk() ? VERNON_STATUS_OK : vernon::toVernonStatus(std::move(invoked).error());
}

} // namespace

VernonStatus vernon::runtime::executePipelineProgramGraph(
    VernonRuntimeContext &context, const program::ResolvedExecutionPlan &execution, const program::Graph &graph,
    program_execution::ProgramInvocationState &frame,
    const program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint,
    program_execution::SubmissionRecord &submission, program_execution::SubmissionState &step) {
    return executePipelineProgramGraphImpl(context, execution, graph, frame, resolvePhysicalEndpoint, submission, step);
}

} // extern "C++"

VernonStatus vernon::runtime::submitResolvedStage(VernonStageExecutable *pipeline,
                                                  const VernonStageInvocationDescriptor *invocation,
                                                  VernonSubmission **output) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (output)
        *output = nullptr;
    if (!pipeline || !invocation || !output || invocation->struct_size < sizeof(VernonStageInvocationDescriptor) ||
        invocation->abi_version != VERNON_PROGRAM_VERSION || (invocation->argument_count && !invocation->arguments) ||
        invocation->command_encoder.value != 0)
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline submission");
    if (!pipeline->lifecycle)
        vernon::resultContractViolation();
    auto pipelinePin = pipeline->lifecycle.value().pin();
    if (pipelinePin.isErr())
        return fail(pipeline->context, "resolved Stage is closing");
    auto child = RuntimeChildLifecycle::reserve(pipeline->context->owner);
    if (child.isErr())
        return fail(pipeline->context, "runtime context cannot admit a submission",
                    vernon::toVernonStatus(child.error()));
    auto submission = std::make_unique<VernonSubmission>(
        *pipeline->context, std::make_shared<program_execution::SubmissionRecord>(*pipeline->context));
    submission->lifecycle.emplace(std::move(child).value());
    if (pipeline->context->rhiDevice.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        const VernonStatus status = encodeStageInvocation(*pipeline, *invocation);
        if (status != VERNON_STATUS_OK)
            return status;
        submission->record->setOutcome({program_execution::SubmissionState::Completed, {}});
        if (submission->lifecycle.value().publish().isErr())
            return fail(pipeline->context, "cannot publish pipeline submission", VERNON_STATUS_INTERNAL_ERROR);
        *output = submission.release();
        return VERNON_STATUS_OK;
    }

    pipeline->context->submissions.admit({0, 1});
    const bool graphics = pipeline->bindingProjection.compute.empty();
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = graphics ? VERNON_RHI_QUEUE_GRAPHICS : VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder native{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (vernonRhiDeviceCreateCommandEncoder(pipeline->context->rhiDevice, &descriptor, &native) != VERNON_RHI_STATUS_OK)
        return fail(pipeline->context, "failed to create the immediate command encoder");
    VernonStageInvocationDescriptor encoded = *invocation;
    auto referencedEncoder = referenceBackendCommandEncoder(*pipeline->context, native);
    VernonStatus status =
        referencedEncoder.isOk() ? VERNON_STATUS_OK : vernon::toVernonStatus(std::move(referencedEncoder).error());
    if (status == VERNON_STATUS_OK)
        encoded.command_encoder = referencedEncoder.value();
    bool rendering = false;
    PlannedGraphicsInvocation plan;
    if (status == VERNON_STATUS_OK && graphics) {
        std::string planningError;
        if (!planGraphicsInvocation(pipeline->bindingProjection, encoded, describeRuntimeImage, pipeline->context, plan,
                                    planningError))
            status = fail(pipeline->context, std::move(planningError));
        GraphicsScopeReads reads;
        if (status == VERNON_STATUS_OK) {
            addGraphicsScopeReads(reads, plan);
            if (graphicsScopeReadsOverlap(reads, plan))
                status = fail(pipeline->context, "graphics invocation samples an image it renders to");
        }
        if (status == VERNON_STATUS_OK) {
            const GraphicsRenderScope scope = graphicsRenderScope(plan, std::move(reads));
            auto begun =
                beginBackendRendering(*pipeline->context, encoded.command_encoder, graphicsRenderingDescriptor(scope));
            if (begun.isErr())
                status = vernon::toVernonStatus(std::move(begun).error());
            else
                rendering = true;
        }
    }
    if (status == VERNON_STATUS_OK && graphics) {
        auto invoked = invokeBackendPipeline(*pipeline, encoded, plan);
        if (invoked.isErr())
            status = vernon::toVernonStatus(std::move(invoked).error());
    } else if (status == VERNON_STATUS_OK) {
        status = encodeStageInvocation(*pipeline, encoded);
    }
    if (rendering) {
        auto ended = endBackendRendering(*pipeline->context, encoded.command_encoder);
        if (status == VERNON_STATUS_OK && ended.isErr())
            status = vernon::toVernonStatus(std::move(ended).error());
    }
    if (status == VERNON_STATUS_OK &&
        vernonRhiCommandEncoderFinish(pipeline->context->rhiDevice, native) != VERNON_RHI_STATUS_OK)
        status = fail(pipeline->context, "failed to finish the immediate command encoder");
    if (status != VERNON_STATUS_OK) {
        (void)vernonRhiDeviceDestroyCommandEncoder(pipeline->context->rhiDevice, native);
        return status;
    }
    if (submission->record->submit(native) != VERNON_RHI_STATUS_OK) {
        const VernonStringView detail = vernonRhiDeviceGetLastError(pipeline->context->rhiDevice);
        std::string error = "failed to submit the immediate command encoder";
        if (detail.data && detail.size)
            error.append(": ").append(detail.data, detail.size);
        return fail(pipeline->context, std::move(error));
    }
    submission->record->setOutcome({program_execution::SubmissionState::Submitted, {}});
    pipeline->context->submissions.push(submission->record, 0);
    if (submission->lifecycle.value().publish().isErr())
        return fail(pipeline->context, "cannot publish pipeline submission", VERNON_STATUS_INTERNAL_ERROR);
    *output = submission.release();
    return VERNON_STATUS_OK;
}

VernonStatus vernon::runtime::encodeResolvedStage(VernonRuntimeProviderObject encoder, VernonStageExecutable *pipeline,
                                                  const VernonStageInvocationDescriptor *invocation) {
    RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !invocation || invocation->struct_size < sizeof(VernonStageInvocationDescriptor) ||
        invocation->abi_version != VERNON_PROGRAM_VERSION || (invocation->argument_count && !invocation->arguments))
        return fail(pipeline ? pipeline->context : nullptr, "invalid pipeline invocation");
    if (!pipeline->lifecycle)
        vernon::resultContractViolation();
    auto pipelinePin = pipeline->lifecycle.value().pin();
    if (pipelinePin.isErr())
        return fail(pipeline->context, "resolved Stage is closing");
    VernonStageInvocationDescriptor encoded = *invocation;
    encoded.command_encoder = encoder;
    return encodeStageInvocation(*pipeline, encoded);
}

extern "C++" {
namespace {
using ProgramInstanceHandleResult = vernon::Result<VernonProgramInstance *, vernon::RuntimeError>;

ProgramInstanceHandleResult createProgramInstanceOperation(VernonProgramExecutable *pipeline) {
    VernonRuntimeContext *context = pipeline ? pipeline->context : nullptr;
    if (!pipeline || !pipeline->context) {
        fail(context, "invalid Program instance creation");
        return ProgramInstanceHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program instance creation", 0, 0}})};
    }
    auto pipelinePin = pipeline->lifecycle.pin();
    if (pipelinePin.isErr()) {
        fail(context, "Program executable is closing", vernon::toVernonStatus(pipelinePin.error()));
        return ProgramInstanceHandleResult{vernon::err(std::move(pipelinePin).error())};
    }
    auto child = RuntimeChildLifecycle::reserve(pipeline->instanceOwner);
    if (child.isErr()) {
        fail(context, "Program executable cannot admit an instance", vernon::toVernonStatus(child.error()));
        return ProgramInstanceHandleResult{vernon::err(std::move(child).error())};
    }
    auto invocationOwner = vernon::OwnerControlBlock::create();
    if (invocationOwner.isErr()) {
        fail(context, "cannot create Program instance invocation owner",
             vernon::toVernonStatus(invocationOwner.error()));
        return ProgramInstanceHandleResult{vernon::err(vernon::toRuntimeError(invocationOwner.error()))};
    }
    auto state = std::make_shared<RuntimeProgramInstanceState>(*pipeline, std::move(child).value(),
                                                               std::move(invocationOwner).value());
    auto instance = std::make_unique<VernonProgramInstance>(state);
    auto published = state->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish Program instance", vernon::toVernonStatus(published.error()));
        return ProgramInstanceHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramInstanceHandleResult{vernon::ok(instance.release())};
}
} // namespace
} // extern "C++"

VernonRuntimeOperationStatus vernonRuntimeProgramInstanceCreateResult(VernonProgramExecutable *pipeline,
                                                                      VernonProgramInstance **output) {
    VernonRuntimeContext *context = pipeline ? pipeline->context : nullptr;
    return runtimeHandleResultBoundary(context, output, [&] { return createProgramInstanceOperation(pipeline); });
}

VernonProgramInstance *vernonRuntimeProgramInstanceCreate(VernonProgramExecutable *pipeline) {
    VernonProgramInstance *output = nullptr;
    (void)vernonRuntimeProgramInstanceCreateResult(pipeline, &output);
    return output;
}

void vernonRuntimeProgramInstanceDestroy(VernonProgramInstance *instance) {
    VernonRuntimeContext *context =
        instance && instance->state && instance->state->pipeline ? instance->state->pipeline->context : nullptr;
    runtimeVoidBoundary(context, [&] { delete instance; });
}

extern "C++" {
namespace {
using ProgramInvocationHandleResult = vernon::Result<VernonProgramInvocation *, vernon::RuntimeError>;

ProgramInvocationHandleResult beginProgramInvocationOperation(VernonProgramInstance *instance) {
    VernonRuntimeContext *context =
        instance && instance->state && instance->state->pipeline ? instance->state->pipeline->context : nullptr;
    if (!instance || !instance->state || !instance->state->pipeline) {
        fail(context, "invalid Program invocation begin");
        return ProgramInvocationHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program invocation begin", 0, 0}})};
    }
    auto instancePin = instance->state->lifecycle.pin();
    if (instancePin.isErr()) {
        fail(context, "Program instance is closing", vernon::toVernonStatus(instancePin.error()));
        return ProgramInvocationHandleResult{vernon::err(std::move(instancePin).error())};
    }
    auto child = RuntimeChildLifecycle::reserve(instance->state->owner);
    if (child.isErr()) {
        fail(context, "Program instance cannot admit an invocation", vernon::toVernonStatus(child.error()));
        return ProgramInvocationHandleResult{vernon::err(std::move(child).error())};
    }
    auto transaction = instance->state->bindings.beginInvocation();
    if (transaction.isErr()) {
        fail(context, "cannot begin Program binding transaction", VERNON_STATUS_INTERNAL_ERROR);
        return ProgramInvocationHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InternalFailure, {"cannot begin Program binding transaction", 0, 0}})};
    }
    auto invocation =
        std::make_unique<VernonProgramInvocation>(*instance, std::move(child).value(), std::move(transaction).value());
    invocation->executionWorkspace = instance->state->workspaces->acquire();
    if (!invocation->executionWorkspace) {
        fail(context, "cannot acquire Program execution workspace", VERNON_STATUS_INTERNAL_ERROR);
        return ProgramInvocationHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::ResourceExhausted, {"cannot acquire Program execution workspace", 0, 0}})};
    }
    auto published = invocation->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish Program invocation", vernon::toVernonStatus(published.error()));
        return ProgramInvocationHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramInvocationHandleResult{vernon::ok(invocation.release())};
}
} // namespace
} // extern "C++"

VernonRuntimeOperationStatus vernonRuntimeProgramInstanceBeginInvocationResult(VernonProgramInstance *instance,
                                                                               VernonProgramInvocation **output) {
    VernonRuntimeContext *context =
        instance && instance->state && instance->state->pipeline ? instance->state->pipeline->context : nullptr;
    return runtimeHandleResultBoundary(context, output, [&] { return beginProgramInvocationOperation(instance); });
}

VernonProgramInvocation *vernonRuntimeProgramInstanceBeginInvocation(VernonProgramInstance *instance) {
    VernonProgramInvocation *output = nullptr;
    (void)vernonRuntimeProgramInstanceBeginInvocationResult(instance, &output);
    return output;
}

VernonStatus vernonRuntimeProgramInvocationSetAutodiffOptions(VernonProgramInvocation *invocation,
                                                              const VernonProgramAutodiffInvocationOptions *options) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeStatusBoundary(context, [&] {
        const bool reservedBytesSet =
            options && std::any_of(std::begin(options->reserved_bytes), std::end(options->reserved_bytes),
                                   [](uint8_t value) { return value != 0; });
        const bool reservedWordsSet = options && std::any_of(std::begin(options->reserved), std::end(options->reserved),
                                                             [](uint32_t value) { return value != 0; });
        if (!invocation || invocation->executed || invocation->finished || !options ||
            options->struct_size < sizeof(*options) ||
            options->abi_version != VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION ||
            options->has_checkpoint_memory_budget > 1 || reservedBytesSet || reservedWordsSet ||
            (options->checkpoint_policy.size && !options->checkpoint_policy.data))
            return fail(context, "invalid Program autodiff invocation options");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(context, "Program invocation is closing", vernon::toVernonStatus(invocationPin.error()));
        std::string policy;
        if (options->checkpoint_policy.size)
            policy.assign(options->checkpoint_policy.data, options->checkpoint_policy.size);
        if (!policy.empty() && policy != "min_memory" && policy != "balanced" && policy != "min_runtime")
            return fail(context, "Program autodiff invocation checkpoint policy is invalid");
        invocation->checkpointMemoryBudget = options->has_checkpoint_memory_budget
                                                 ? std::optional<uint64_t>(options->checkpoint_memory_budget)
                                                 : std::nullopt;
        invocation->checkpointPolicy.swap(policy);
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationSetOptions(VernonProgramInvocation *invocation,
                                                      const VernonProgramInvocationOptions *options) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeStatusBoundary(context, [&] {
        const bool reservedSet = options && std::any_of(std::begin(options->reserved), std::end(options->reserved),
                                                        [](uint32_t value) { return value != 0; });
        if (!invocation || invocation->executed || invocation->finished || !options ||
            options->struct_size < sizeof(*options) ||
            options->abi_version != VERNON_PROGRAM_INVOCATION_OPTIONS_VERSION || reservedSet ||
            (options->completion_mode != VERNON_COMPLETION_WAIT &&
             options->completion_mode != VERNON_COMPLETION_ASYNCHRONOUS))
            return fail(context, "invalid Program invocation options");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(context, "Program invocation is closing", vernon::toVernonStatus(invocationPin.error()));
        invocation->completionMode = options->completion_mode == VERNON_COMPLETION_ASYNCHRONOUS
                                         ? vernon::runtime::program_execution::CompletionMode::Asynchronous
                                         : vernon::runtime::program_execution::CompletionMode::Wait;
        return VERNON_STATUS_OK;
    });
}

extern "C++" {
namespace {
using ProgramBindingResult = vernon::Result<void, vernon::RuntimeError>;

ProgramBindingResult programBindingFailure(VernonRuntimeContext *context, vernon::RuntimeErrorCode code,
                                           const char *diagnostic) {
    (void)fail(context, diagnostic, vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
    return ProgramBindingResult{vernon::err(vernon::RuntimeError{code, {diagnostic, 0, 0}})};
}

const char *programArgumentDiagnostic(const program::BoundarySlot &boundary, const VernonProgramArgument &argument) {
    if (argument.kind < VERNON_PROGRAM_TENSOR || argument.kind > VERNON_PROGRAM_SAMPLER)
        return "Program binding has an invalid argument kind";
    if (!vernon::runtime::program_execution::argumentMatchesBoundary(boundary, argument))
        return "Program binding does not match its canonical boundary contract";
    if (argument.kind != VERNON_PROGRAM_TENSOR)
        return nullptr;
    const VernonTensorView &tensor = argument.tensor;
    if (tensor.struct_size < sizeof(tensor) || tensor.element_layout.struct_size < sizeof(VernonValueLayoutView) ||
        (tensor.rank && (!tensor.shape || !tensor.byte_strides)) ||
        (tensor.element_layout.layout_hash.size && !tensor.element_layout.layout_hash.data) ||
        (tensor.element_layout.leaf_count && !tensor.element_layout.leaves))
        return "persistent Program Tensor binding has incomplete metadata";
    return nullptr;
}

using ProgramBindingValueResult = vernon::Result<std::shared_ptr<RuntimeProgramBinding>, vernon::RuntimeError>;

// `argument` must have passed programArgumentDiagnostic.
ProgramBindingValueResult makeProgramBinding(const VernonProgramArgument &argument,
                                             const VernonProgramResourceLease *lease) {
    auto binding = std::make_shared<RuntimeProgramBinding>();
    binding->argument = argument;
    if (argument.kind == VERNON_PROGRAM_TENSOR) {
        const VernonTensorView &tensor = argument.tensor;
        auto metadata = std::make_shared<RuntimeProgramBindingMetadata>();
        if (tensor.rank) {
            metadata->shape.assign(tensor.shape, tensor.shape + tensor.rank);
            metadata->strides.assign(tensor.byte_strides, tensor.byte_strides + tensor.rank);
        }
        if (tensor.element_layout.layout_hash.size)
            metadata->layoutHash.assign(tensor.element_layout.layout_hash.data, tensor.element_layout.layout_hash.size);
        if (tensor.element_layout.leaf_count)
            metadata->leaves.assign(tensor.element_layout.leaves,
                                    tensor.element_layout.leaves + tensor.element_layout.leaf_count);
        binding->metadata = std::move(metadata);
    }
    if (lease) {
        auto retained = retainProgramResourceLease(*lease);
        if (retained.isErr())
            return ProgramBindingValueResult{vernon::err(std::move(retained).error())};
        binding->lease = std::move(retained).value();
    }
    binding->refresh();
    return ProgramBindingValueResult{vernon::ok(std::move(binding))};
}

ProgramBindingResult tryReuseProgramBindingOperation(VernonProgramInvocation *invocation, uint32_t slot,
                                                     const VernonProgramBindingToken *token, uint64_t uploadBytes,
                                                     uint64_t uploadRanges, uint8_t *reused) {
    VernonRuntimeContext *context = invocationContext(invocation);
    if (reused)
        *reused = 0;
    if (!invocation || invocation->executed || invocation->finished || !token || token->struct_size < sizeof(*token) ||
        !token->size || !token->data || !reused)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "invalid persistent Program binding reuse query");
    auto invocationPin = invocation->lifecycle.pin();
    if (invocationPin.isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                     "Program invocation is closing");
    if (!bindableParameterSlot(*invocation->instance->pipeline, slot))
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "Program binding reuse query does not reference an exported parameter");
    const std::string_view key(static_cast<const char *>(token->data), token->size);
    const std::vector<uint32_t> *aliases = bindingAliasSlots(*invocation->instance->pipeline, slot);
    if (!aliases)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InternalFailure,
                                     "Program binding reuse query has no resolved aliases");
    bool allMatch = true;
    for (uint32_t alias : *aliases) {
        auto matches = invocation->transaction->matches(alias, key);
        if (matches.isErr())
            return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                         "Program binding reuse query is not in the staging phase");
        allMatch &= matches.value();
    }
    if (allMatch) {
        if (invocation->transaction->observeReuses(aliases->size()).isErr() ||
            invocation->transaction->observeUploads(uploadBytes, uploadRanges).isErr())
            return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                         "Program binding reuse telemetry is not in the staging phase");
        *reused = 1;
    }
    return ProgramBindingResult{vernon::ok()};
}
} // namespace
} // extern "C++"

VernonStatus vernonRuntimeProgramInvocationTryReuse(VernonProgramInvocation *invocation, uint32_t slot,
                                                    const VernonProgramBindingToken *token, uint64_t uploadBytes,
                                                    uint64_t uploadRanges, uint8_t *reused) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(context, [&] {
        return tryReuseProgramBindingOperation(invocation, slot, token, uploadBytes, uploadRanges, reused);
    });
}

extern "C++" {
namespace {
ProgramBindingResult bindProgramInvocationOperation(VernonProgramInvocation *invocation,
                                                    const VernonProgramBindingToken *token,
                                                    const VernonProgramArgument *argument,
                                                    const VernonProgramResourceLease *lease, uint64_t uploadBytes,
                                                    uint64_t uploadRanges) {
    VernonRuntimeContext *context = invocationContext(invocation);
    if (!invocation || invocation->executed || invocation->finished || !token || token->struct_size < sizeof(*token) ||
        !token->size || !token->data || !argument)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "invalid persistent Program binding update");
    auto invocationPin = invocation->lifecycle.pin();
    if (invocationPin.isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                     "Program invocation is closing");
    const program::Program &program = *executableProgram(*invocation->instance->pipeline);
    if (!bindableParameterSlot(*invocation->instance->pipeline, argument->slot))
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "Program binding does not reference an exported parameter");
    if (const char *diagnostic = programArgumentDiagnostic(program.abi.boundarySlots[argument->slot], *argument))
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument, diagnostic);
    const std::string_view key(static_cast<const char *>(token->data), token->size);
    const std::vector<uint32_t> *aliases = bindingAliasSlots(*invocation->instance->pipeline, argument->slot);
    if (!aliases)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InternalFailure,
                                     "Program binding has no resolved aliases");
    for (uint32_t alias : *aliases) {
        if (alias >= program.abi.boundarySlots.size() || program.abi.boundarySlots[alias].id != alias)
            return programBindingFailure(context, vernon::RuntimeErrorCode::VerificationFailure,
                                         "Program binding alias does not reference a canonical boundary slot");
    }
    bool allMatch = true;
    for (uint32_t alias : *aliases) {
        auto matches = invocation->transaction->matches(alias, key);
        if (matches.isErr())
            return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                         "Program binding update is not in the staging phase");
        allMatch &= matches.value();
    }
    if (allMatch) {
        if (invocation->transaction->observeReuses(aliases->size()).isErr() ||
            invocation->transaction->observeUploads(uploadBytes, uploadRanges).isErr())
            return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                         "Program binding telemetry update is not in the staging phase");
        return ProgramBindingResult{vernon::ok()};
    }
    auto made = makeProgramBinding(*argument, lease);
    if (made.isErr())
        return ProgramBindingResult{vernon::err(std::move(made).error())};
    const std::shared_ptr<RuntimeProgramBinding> binding = std::move(made).value();
    std::vector<std::pair<uint64_t, vernon::runtime::program::BindingEntry>> staged;
    staged.reserve(aliases->size());
    for (uint32_t alias : *aliases) {
        auto remapped = std::make_shared<RuntimeProgramBinding>(*binding);
        remapped->argument.slot = alias;
        if (remapped->argument.kind == VERNON_PROGRAM_TENSOR) {
            const program::BoundarySlot &boundary = program.abi.boundarySlots[alias];
            remapped->argument.tensor.access = vernon::runtime::program_execution::boundaryValueAccess(boundary.access);
        }
        remapped->refresh();
        staged.emplace_back(alias, vernon::runtime::program::BindingEntry{std::string(key), std::move(remapped)});
    }
    if (invocation->transaction->stageMany(std::move(staged), uploadBytes, uploadRanges).isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::ResourceExhausted,
                                     "cannot stage persistent Program binding");
    return ProgramBindingResult{vernon::ok()};
}
} // namespace
} // extern "C++"

VernonStatus vernonRuntimeProgramInvocationBind(VernonProgramInvocation *invocation,
                                                const VernonProgramBindingToken *token,
                                                const VernonProgramArgument *argument,
                                                const VernonProgramResourceLease *lease, uint64_t uploadBytes,
                                                uint64_t uploadRanges) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(context, [&] {
        return bindProgramInvocationOperation(invocation, token, argument, lease, uploadBytes, uploadRanges);
    });
}

extern "C++" {
namespace {
using ProgramControlResult = vernon::Result<void, vernon::RuntimeError>;

vernon::Result<std::string_view, vernon::RuntimeError>
programControlToken(const VernonProgramBindingToken *token) noexcept {
    if (!token || token->struct_size < sizeof(*token) || !token->data || !token->size)
        return vernon::Result<std::string_view, vernon::RuntimeError>{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"Program control binding token is invalid", 0, 0}})};
    return vernon::Result<std::string_view, vernon::RuntimeError>{
        vernon::ok(std::string_view{static_cast<const char *>(token->data), token->size})};
}

ProgramControlResult retainProgramControlLease(RuntimeProgramControl &control,
                                               const VernonProgramResourceLease &lease) {
    auto retained = retainProgramResourceLease(lease);
    if (retained.isErr())
        return ProgramControlResult{vernon::err(std::move(retained).error())};
    if (retained.value())
        control.leases.push_back(std::move(retained).value());
    return ProgramControlResult{vernon::ok()};
}

ProgramControlResult stageProgramControl(VernonProgramInvocation *invocation, uint32_t slot,
                                         const VernonProgramBindingToken *token,
                                         std::shared_ptr<RuntimeProgramControl> control) {
    if (!invocation || invocation->executed || invocation->finished || !control)
        return ProgramControlResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid persistent Program control update", slot, 0}})};
    auto invocationPin = invocation->lifecycle.pin();
    if (invocationPin.isErr())
        return ProgramControlResult{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::LifecycleFailure,
                                                                     {"Program invocation is closing", slot, 0}})};
    auto keyResult = programControlToken(token);
    if (keyResult.isErr())
        return ProgramControlResult{vernon::err(std::move(keyResult).error())};
    const std::string_view key = std::move(keyResult).value();
    const uint64_t storageSlot = programControlKey(slot, control->kind);
    if (invocation->instance->pipeline->graphicsControlKeys.find(storageSlot) ==
        invocation->instance->pipeline->graphicsControlKeys.end())
        return ProgramControlResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"Program control slot or kind is not resolved", slot, 0}})};
    auto matches = invocation->transaction->matches(storageSlot, key);
    if (matches.isErr())
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::LifecycleFailure,
                                             {"Program control update is not in the staging phase", slot, 0}})};
    if (matches.value()) {
        if (invocation->transaction->observeReuses(1).isErr())
            return ProgramControlResult{
                vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::LifecycleFailure,
                                                 {"Program control reuse is not in the staging phase", slot, 0}})};
        return ProgramControlResult{vernon::ok()};
    }
    control->refresh();
    if (invocation->transaction->stage(storageSlot, std::string(key), std::move(control), 0, 0).isErr())
        return ProgramControlResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::ResourceExhausted, {"cannot stage persistent Program control", slot, 0}})};
    return ProgramControlResult{vernon::ok()};
}

ProgramControlResult bindProgramRenderPassOperation(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                    const VernonProgramBindingToken *token,
                                                    const VernonRenderPass *renderPass,
                                                    const VernonProgramResourceLease *leases, size_t leaseCount) {
    if (!renderPass || renderPass->struct_size < sizeof(*renderPass) ||
        (renderPass->color_attachment_count && !renderPass->color_attachments) || (leaseCount && !leases))
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"invalid persistent Program RenderPass control", controlSlot, 0}})};
    auto control = std::make_shared<RuntimeProgramControl>();
    control->kind = RuntimeProgramControl::RenderPass;
    control->renderPass = *renderPass;
    control->colors.assign(renderPass->color_attachments,
                           renderPass->color_attachments + renderPass->color_attachment_count);
    if (renderPass->depth_attachment)
        control->depth = *renderPass->depth_attachment;
    for (size_t index = 0; index < leaseCount; ++index) {
        auto retained = retainProgramControlLease(*control, leases[index]);
        if (retained.isErr())
            return retained;
    }
    return stageProgramControl(invocation, controlSlot, token, std::move(control));
}

std::shared_ptr<RuntimeProgramControl> makeDrawCommandControl(const VernonDrawCommand *draw) {
    if (!draw || draw->struct_size < sizeof(*draw) || !draw->instance_count ||
        (draw->index_binding && (draw->vertex_count || draw->index_binding->type != VERNON_INDEX_U32)))
        return nullptr;
    auto control = std::make_shared<RuntimeProgramControl>();
    control->kind = RuntimeProgramControl::DrawCommand;
    control->draw = *draw;
    if (draw->index_binding)
        control->index = *draw->index_binding;
    control->refresh();
    return control;
}

ProgramControlResult bindProgramDrawCommandOperation(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                     const VernonProgramBindingToken *token,
                                                     const VernonDrawCommand *draw,
                                                     const VernonProgramResourceLease *lease) {
    std::shared_ptr<RuntimeProgramControl> control = makeDrawCommandControl(draw);
    if (!control)
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"invalid persistent Program DrawCommand control", controlSlot, 0}})};
    if (lease) {
        auto retained = retainProgramControlLease(*control, *lease);
        if (retained.isErr())
            return retained;
    }
    return stageProgramControl(invocation, controlSlot, token, std::move(control));
}

ProgramControlResult bindProgramDynamicStateOperation(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                      const VernonProgramBindingToken *token,
                                                      const VernonDynamicState *dynamicState) {
    if (!dynamicState || dynamicState->struct_size < sizeof(*dynamicState) || dynamicState->stencil_reference > 0xff)
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"invalid persistent Program DynamicState control", controlSlot, 0}})};
    auto control = std::make_shared<RuntimeProgramControl>();
    control->kind = RuntimeProgramControl::DynamicState;
    control->dynamic = *dynamicState;
    return stageProgramControl(invocation, controlSlot, token, std::move(control));
}
} // namespace
} // extern "C++"

VernonStatus vernonRuntimeProgramInvocationBindRenderPass(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                          const VernonProgramBindingToken *token,
                                                          const VernonRenderPass *renderPass,
                                                          const VernonProgramResourceLease *leases, size_t leaseCount) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(context, [&] {
        return bindProgramRenderPassOperation(invocation, controlSlot, token, renderPass, leases, leaseCount);
    });
}

VernonStatus vernonRuntimeProgramInvocationBindDrawCommand(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                           const VernonProgramBindingToken *token,
                                                           const VernonDrawCommand *draw,
                                                           const VernonProgramResourceLease *lease) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(
        context, [&] { return bindProgramDrawCommandOperation(invocation, controlSlot, token, draw, lease); });
}

VernonStatus vernonRuntimeProgramInvocationBindDynamicState(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                            const VernonProgramBindingToken *token,
                                                            const VernonDynamicState *dynamicState) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(
        context, [&] { return bindProgramDynamicStateOperation(invocation, controlSlot, token, dynamicState); });
}

extern "C++" {
namespace {
std::atomic<uint64_t> graphicsPipelineSerial{0};

VernonStatus failDiagnostic(VernonRuntimeContext *context, const program::Diagnostic &diagnostic,
                            VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    return fail(context,
                diagnostic.code + " at " + (diagnostic.path.empty() ? std::string("<node>") : diagnostic.path) + ": " +
                    diagnostic.message,
                status);
}

struct ForwardGraphicsNode {
    const program::Node *node{};
    const program::ResolvedNodePlan *plan{};
};

std::optional<ForwardGraphicsNode> forwardGraphicsNode(const VernonProgramExecutable &executable,
                                                       VernonProgramNodeId id) {
    std::vector<const program::Node *> nodes;
    collectForwardGraphicsNodes(&executable, nodes);
    for (const program::Node *node : nodes) {
        if (node->id != id)
            continue;
        const program::ResolvedNodePlan *plan = executable.executionPlan->node(program::GraphDirection::Forward, id);
        if (!plan || !plan->stage)
            return std::nullopt;
        return ForwardGraphicsNode{node, plan};
    }
    return std::nullopt;
}

recorded::NodeEndpointsResult classifyExecutableNode(const VernonProgramExecutable &executable,
                                                     const ForwardGraphicsNode &node) {
    return recorded::classifyNodeEndpoints(*executableProgram(executable), *node.plan,
                                           [&](uint32_t slot) { return bindableParameterSlot(executable, slot); });
}

using PipelineStateResult = vernon::Result<std::shared_ptr<const RuntimeGraphicsPipelineState>, std::string>;

PipelineStateResult passPipelineState(VernonProgramExecutable &executable, VernonProgramNodeId nodeId) {
    std::lock_guard lock(executable.passPipelineMutex);
    if (const auto found = executable.passPipelines.find(nodeId); found != executable.passPipelines.end())
        return PipelineStateResult{vernon::ok(found->second)};
    const std::optional<ForwardGraphicsNode> node = forwardGraphicsNode(executable, nodeId);
    if (!node)
        return PipelineStateResult{vernon::err(std::string("node is not a resolved graphics Node of the executable"))};
    auto endpoints = classifyExecutableNode(executable, *node);
    if (endpoints.isErr())
        return PipelineStateResult{vernon::err(std::move(endpoints).error())};
    auto state = std::make_shared<RuntimeGraphicsPipelineState>();
    state->serial = ++graphicsPipelineSerial;
    state->pass = &executable;
    state->node = nodeId;
    state->source = &executable;
    state->sourceNode = node->node;
    state->sourcePlan = node->plan;
    state->passEndpoints = std::move(endpoints).value();
    state->endpoints = recorded::passPipelineEndpoints(state->passEndpoints);
    executable.passPipelines.emplace(nodeId, state);
    return PipelineStateResult{vernon::ok(std::move(state))};
}

bool sameBoundaryContract(const program::BoundarySlot &left, const program::BoundarySlot &right) {
    const bool sameLayout = left.layout.has_value() == right.layout.has_value() &&
                            (!left.layout || (left.layout->layoutHash == right.layout->layoutHash &&
                                              left.layout->byteSize == right.layout->byteSize &&
                                              left.layout->alignment == right.layout->alignment));
    const bool sameStorage = left.storage.has_value() == right.storage.has_value() &&
                             (!left.storage || left.storage->descriptorKind == right.storage->descriptorKind);
    return left.role == right.role && left.category == right.category && left.access == right.access &&
           left.logicalType == right.logicalType && left.outerShape == right.outerShape && sameLayout && sameStorage;
}

bool hostArgumentNeedsUpload(const VernonProgramArgument &argument, const program::TargetBinding &target) {
    return argument.kind == VERNON_PROGRAM_TENSOR && argument.tensor.storage == VERNON_TENSOR_HOST &&
           target.carrier != program::TargetCarrier::InlineValue &&
           target.carrier != program::TargetCarrier::UniformBuffer;
}

// Resolves the parameter `boundary` of another Program to the per-draw endpoints it supplies in `pipeline`.
program::Diagnostic correspondParameter(const program::BoundarySlot &boundary, const VernonProgramArgument &argument,
                                        const RuntimeGraphicsPipelineState &pipeline, const std::string &path,
                                        std::vector<char> &supplied, const program::BoundarySlot *&target) {
    const VernonProgramExecutable &executable = *pipeline.source;
    const program::Program &program = *executableProgram(executable);
    target = findParameterBoundary(executable, {boundary.path.data(), boundary.path.size()});
    if (!target || !sameBoundaryContract(boundary, *target))
        return {recorded::kBindingSetMismatch, recorded::kInvocationEntryPhase, path,
                "parameter '" + boundary.path + "' has no counterpart with equal reflection in the pipeline"};
    const std::vector<uint32_t> aliases = boundSlotsOf(executable, target->id);
    // A Node that writes the parameter in place makes the endpoint read a later Value of the same Storage.
    const auto aliasReaches = [&](uint32_t value) {
        const std::optional<uint32_t> storage =
            value < program.values.size() ? program.values[value].storage : std::nullopt;
        return std::any_of(aliases.begin(), aliases.end(), [&](uint32_t alias) {
            if (alias >= program.abi.boundarySlots.size())
                return false;
            const uint32_t bound = program.abi.boundarySlots[alias].value;
            return bound == value ||
                   (storage && bound < program.values.size() && program.values[bound].storage == storage);
        });
    };
    supplied.assign(pipeline.endpoints.size(), 0);
    bool any = false;
    for (size_t index = 0; index < pipeline.endpoints.size(); ++index) {
        const recorded::PipelineEndpoint &endpoint = pipeline.endpoints[index];
        const program::NodeEndpointProjection &projection = pipeline.sourcePlan->projections[index];
        if (endpoint.parameterSlot &&
            std::find(aliases.begin(), aliases.end(), *endpoint.parameterSlot) != aliases.end()) {
            if (hostArgumentNeedsUpload(argument, projection.target))
                return {recorded::kRecordingInvalid, recorded::kInvocationEntryPhase, path,
                        "parameter '" + boundary.path + "' supplies host data to a device endpoint, which needs a " +
                            "Runtime upload; bind a device resource"};
            supplied[index] = 1;
            any = true;
        } else if (endpoint.fixedPassParameter && aliasReaches(projection.value)) {
            return {recorded::kRecordingInvalid, recorded::kInvocationEntryPhase, path,
                    "parameter '" + boundary.path + "' binds an endpoint the pass Node fixes"};
        }
    }
    if (!any)
        return {recorded::kBindingSetMismatch, recorded::kInvocationEntryPhase, path,
                "parameter '" + boundary.path + "' is not a per-draw parameter of the pipeline"};
    return {};
}

std::shared_ptr<const RuntimeBindingSetMatch>
matchBindingSet(RuntimeBindingSetState &set, const std::shared_ptr<const RuntimeGraphicsPipelineState> &pipeline) {
    {
        std::lock_guard lock(set.matchMutex);
        const auto found = set.matches.find(pipeline->serial);
        if (found != set.matches.end() && !found->second->pipeline.expired())
            return found->second;
    }
    const VernonProgramExecutable &executable = *pipeline->source;
    const program::Program &program = *executableProgram(executable);
    auto match = std::make_shared<RuntimeBindingSetMatch>();
    match->pipeline = pipeline;
    match->endpoints.assign(pipeline->endpoints.size(), 0);
    match->failures.resize(set.entries.size());
    std::vector<vernon::runtime::program_execution::ProgramValueState> values(program.values.size());
    std::vector<char> supplied;
    for (size_t entryIndex = 0; entryIndex < set.entries.size(); ++entryIndex) {
        const RuntimeBindingSetEntry &entry = set.entries[entryIndex];
        const program::BoundarySlot *target = nullptr;
        match->failures[entryIndex] = correspondParameter(*entry.boundary, entry.binding->argument, *pipeline,
                                                          entry.boundary->path, supplied, target);
        if (match->failures[entryIndex])
            continue;
        if (entry.binding->argument.kind == VERNON_PROGRAM_TENSOR && target->value < program.values.size() &&
            entry.binding->argument.tensor.rank < program.values[target->value].shape.size()) {
            match->failures[entryIndex] = {recorded::kBindingSetMismatch, recorded::kInvocationEntryPhase,
                                           entry.boundary->path,
                                           "Tensor rank is lower than the pipeline parameter's shape"};
            continue;
        }
        for (size_t index = 0; index < supplied.size(); ++index)
            match->endpoints[index] |= supplied[index];
        for (uint32_t alias : boundSlotsOf(executable, target->id)) {
            const program::BoundarySlot &boundary = program.abi.boundarySlots[alias];
            if (boundary.role != program::BoundaryRole::Input || boundary.value >= values.size())
                continue;
            vernon::runtime::program_execution::ProgramValueState &value = values[boundary.value];
            value.argument = entry.binding->argument;
            value.argument.slot = alias;
            if (value.argument.kind != VERNON_PROGRAM_TENSOR) {
                value.ownership = vernon::runtime::program_execution::ProgramValueOwnership::BorrowedDevice;
                continue;
            }
            const VernonTensorView &tensor = value.argument.tensor;
            value.argument.tensor.access = vernon::runtime::program_execution::boundaryValueAccess(boundary.access);
            value.ownership = tensor.storage == VERNON_TENSOR_HOST
                                  ? vernon::runtime::program_execution::ProgramValueOwnership::BorrowedHost
                                  : vernon::runtime::program_execution::ProgramValueOwnership::BorrowedDevice;
            const size_t rank = std::min<size_t>(program.values[boundary.value].shape.size(), tensor.rank);
            value.concreteShape.emplace(tensor.shape, tensor.shape + rank);
            value.strides.assign(tensor.byte_strides, tensor.byte_strides + rank);
        }
    }
    match->values = std::make_shared<const vernon::runtime::program_execution::ProgramInvocationState>(
        *executable.executionPlan, std::move(values), std::map<uint32_t, program_execution::ProgramStorageBacking>{});
    std::lock_guard lock(set.matchMutex);
    for (auto entry = set.matches.begin(); entry != set.matches.end();)
        entry = entry->second->pipeline.expired() ? set.matches.erase(entry) : std::next(entry);
    set.matches[pipeline->serial] = match;
    return match;
}

program::Diagnostic prefixedDiagnostic(program::Diagnostic diagnostic, const std::string &path) {
    diagnostic.path = path + "." + diagnostic.path;
    return diagnostic;
}

using GraphicsPipelineHandleResult = vernon::Result<VernonGraphicsPipeline *, vernon::RuntimeError>;

GraphicsPipelineHandleResult createGraphicsPipelineOperation(VernonProgramExecutable *executable,
                                                             VernonProgramNodeId nodeId,
                                                             VernonProgramExecutable *source) {
    VernonRuntimeContext *context = executable ? executable->context : nullptr;
    const auto failure = [&](vernon::RuntimeErrorCode code) {
        return GraphicsPipelineHandleResult{
            vernon::err(vernon::RuntimeError{code, {"graphics pipeline creation", nodeId, 0}})};
    };
    if (!executable || !context || (source && source->context != context)) {
        fail(context, "invalid graphics pipeline creation");
        return failure(vernon::RuntimeErrorCode::InvalidArgument);
    }
    auto passLease = RuntimeExecutableLease::acquire(*executable);
    if (passLease.isErr()) {
        fail(context, "graphics pipeline executable cannot admit a pipeline");
        return GraphicsPipelineHandleResult{vernon::err(std::move(passLease).error())};
    }
    auto pass = passPipelineState(*executable, nodeId);
    if (pass.isErr()) {
        fail(context, "graphics pipeline Node " + std::to_string(nodeId) + ": " + pass.error());
        return failure(vernon::RuntimeErrorCode::InvalidArgument);
    }
    auto handle = std::make_unique<VernonGraphicsPipeline>();
    handle->context = context;
    if (!source) {
        handle->state = std::move(pass).value();
        handle->lease = std::move(passLease).value();
        return GraphicsPipelineHandleResult{vernon::ok(handle.release())};
    }
    const std::shared_ptr<const RuntimeGraphicsPipelineState> &passState = pass.value();
    auto sourceLease = RuntimeExecutableLease::acquire(*source);
    if (sourceLease.isErr()) {
        fail(context, "graphics pipeline source executable cannot admit a pipeline");
        return GraphicsPipelineHandleResult{vernon::err(std::move(sourceLease).error())};
    }
    const program::Graph *graph = program::findGraph(*executableProgram(*source), "forward");
    const std::optional<ForwardGraphicsNode> sourceNode = !source->programGraphId && graph && graph->nodes.size() == 1
                                                              ? forwardGraphicsNode(*source, graph->nodes.front().id)
                                                              : std::nullopt;
    if (!sourceNode) {
        failDiagnostic(context, {recorded::kPipelineConformance, recorded::kPipelineCreationPhase, "source",
                                 "pipeline source is not a resolved one-node graphics Program"});
        return failure(vernon::RuntimeErrorCode::InvalidArgument);
    }
    auto sourceEndpoints = classifyExecutableNode(*source, *sourceNode);
    if (sourceEndpoints.isErr()) {
        failDiagnostic(context, {recorded::kPipelineConformance, recorded::kPipelineCreationPhase, "source",
                                 sourceEndpoints.error()});
        return failure(vernon::RuntimeErrorCode::InvalidArgument);
    }
    auto endpoints = recorded::conformPipeline(
        program::graphicsOperation(*passState->sourceNode), *passState->sourcePlan, passState->passEndpoints,
        program::graphicsOperation(*sourceNode->node), *sourceNode->plan, sourceEndpoints.value());
    if (endpoints.isErr()) {
        failDiagnostic(context, endpoints.error());
        return failure(vernon::RuntimeErrorCode::InvalidArgument);
    }
    auto state = std::make_shared<RuntimeGraphicsPipelineState>();
    state->serial = ++graphicsPipelineSerial;
    state->pass = executable;
    state->node = nodeId;
    state->source = source;
    state->sourceNode = sourceNode->node;
    state->sourcePlan = sourceNode->plan;
    state->endpoints = std::move(endpoints).value();
    state->leases.push_back(std::move(passLease).value());
    state->leases.push_back(std::move(sourceLease).value());
    handle->state = std::move(state);
    return GraphicsPipelineHandleResult{vernon::ok(handle.release())};
}

using BindingSetHandleResult = vernon::Result<VernonBindingSet *, vernon::RuntimeError>;

BindingSetHandleResult createBindingSetOperation(VernonGraphicsPipeline *pipeline) {
    VernonRuntimeContext *context = pipeline ? pipeline->context : nullptr;
    if (!pipeline || !pipeline->state) {
        fail(context, "invalid binding set creation");
        return BindingSetHandleResult{vernon::err(
            vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument, {"binding set creation", 0, 0}})};
    }
    auto state = std::make_shared<RuntimeBindingSetState>();
    state->context = context;
    state->origin = pipeline->state;
    if (state->origin->leases.empty()) {
        auto lease = RuntimeExecutableLease::acquire(*state->origin->pass);
        if (lease.isErr()) {
            fail(context, "binding set executable cannot admit a binding set");
            return BindingSetHandleResult{vernon::err(std::move(lease).error())};
        }
        state->lease = std::move(lease).value();
    }
    auto handle = std::make_unique<VernonBindingSet>();
    handle->state = std::move(state);
    return BindingSetHandleResult{vernon::ok(handle.release())};
}

struct WrittenResources {
    struct Buffer {
        uint64_t identity{};
        uint64_t resource{};
        uint64_t begin{};
        uint64_t end{};
    };
    std::vector<Buffer> buffers;
    std::vector<uint64_t> images;

    // Saturates instead of wrapping, so an out-of-range view still overlaps what it would reach.
    static Buffer range(const VernonRuntimeProviderResourceReference &reference, uint64_t offset, uint64_t size) {
        const uint64_t begin = offset > UINT64_MAX - reference.offset ? UINT64_MAX : reference.offset + offset;
        return {reference.identity, reference.resource.value, begin,
                size > UINT64_MAX - begin ? UINT64_MAX : begin + size};
    }

    bool overlapsBuffer(const VernonRuntimeProviderResourceReference &reference, uint64_t offset, uint64_t size) const {
        const Buffer view = range(reference, offset, size);
        return std::any_of(buffers.begin(), buffers.end(), [&](const Buffer &buffer) {
            return buffer.identity == view.identity && buffer.resource == view.resource && view.begin < buffer.end &&
                   buffer.begin < view.end;
        });
    }
};

std::optional<uint64_t> imageParent(VernonRuntimeContext &context, VernonRuntimeProviderResourceReference view) {
    auto described = describeBackendImage(context, view);
    if (described.isErr())
        return std::nullopt;
    return described.value().parent_identity ? described.value().parent_identity : view.resource.value;
}

bool addWrittenArgument(VernonRuntimeContext &context, const VernonProgramArgument &argument,
                        WrittenResources &written) {
    if (argument.kind == VERNON_PROGRAM_TENSOR) {
        if (argument.tensor.storage != VERNON_TENSOR_RHI_RESOURCE)
            return true;
        written.buffers.push_back(
            WrittenResources::range(argument.tensor.resource, argument.tensor.byte_offset, argument.tensor.byte_size));
        return true;
    }
    if (argument.kind != VERNON_PROGRAM_IMAGE)
        return true;
    const std::optional<uint64_t> parent = imageParent(context, argument.image.view);
    if (!parent)
        return false;
    written.images.push_back(*parent);
    return true;
}

// Empty when the argument's image cannot be described.
std::optional<bool> argumentIsWritten(VernonRuntimeContext &context, const VernonProgramArgument &argument,
                                      const WrittenResources &written) {
    if (argument.kind == VERNON_PROGRAM_TENSOR)
        return argument.tensor.storage == VERNON_TENSOR_RHI_RESOURCE &&
               written.overlapsBuffer(argument.tensor.resource, argument.tensor.byte_offset, argument.tensor.byte_size);
    if (argument.kind != VERNON_PROGRAM_IMAGE)
        return false;
    const std::optional<uint64_t> parent = imageParent(context, argument.image.view);
    if (!parent)
        return std::nullopt;
    return std::find(written.images.begin(), written.images.end(), *parent) != written.images.end();
}

// Resources the invocation's Nodes write that callers can also reach: written boundary arguments and attachments.
bool collectWrittenResources(VernonRuntimeContext &context, const VernonProgramExecutable &executable,
                             const vernon::runtime::program::InvocationSnapshot &snapshot, WrittenResources &written) {
    const program::Program &program = *executableProgram(executable);
    for (const program::BoundarySlot &slot : program.abi.boundarySlots) {
        if (slot.access == program::BoundaryAccess::Read)
            continue;
        const std::shared_ptr<void> *payload = snapshot.find(slot.id);
        if (payload &&
            !addWrittenArgument(context, std::static_pointer_cast<RuntimeProgramBinding>(*payload)->argument, written))
            return false;
    }
    std::vector<const program::Node *> nodes;
    collectForwardGraphicsNodes(&executable, nodes);
    for (const program::Node *node : nodes) {
        const std::shared_ptr<void> *payload = snapshot.find(
            programControlKey(program::graphicsOperation(*node).renderPassControl, RuntimeProgramControl::RenderPass));
        if (!payload)
            continue;
        const auto &control = *std::static_pointer_cast<RuntimeProgramControl>(*payload);
        std::vector<VernonRuntimeProviderResourceReference> views;
        for (const VernonColorAttachment &color : control.colors)
            views.push_back(color.view);
        if (control.depth)
            views.push_back(control.depth->view);
        for (const VernonRuntimeProviderResourceReference &view : views) {
            const std::optional<uint64_t> parent = imageParent(context, view);
            if (!parent)
                return false;
            written.images.push_back(*parent);
        }
    }
    return true;
}

struct BoundParameter {
    std::shared_ptr<RuntimeBindingSetState> set;
    size_t entry{};
};

VernonStatus validateGraphicsRecording(VernonProgramInvocation &invocation, VernonProgramNodeId nodeId,
                                       const RuntimeGraphicsRecording &recording, const WrittenResources &written,
                                       RecordedGraphicsPass &pass) {
    VernonProgramExecutable &executable = *invocation.instance->pipeline;
    VernonRuntimeContext &context = *executable.context;
    const std::string nodePath = "node " + std::to_string(nodeId);
    const auto invalid = [&](const std::string &path, const std::string &message) {
        return failDiagnostic(&context, {recorded::kRecordingInvalid, recorded::kInvocationEntryPhase, path, message});
    };
    auto passPipeline = passPipelineState(executable, nodeId);
    if (passPipeline.isErr())
        return invalid(nodePath, passPipeline.error());
    const program::GraphicsOperation &graphics = program::graphicsOperation(*passPipeline.value()->sourceNode);
    VernonDynamicState dynamic{};
    dynamic.struct_size = sizeof(dynamic);
    // Shared by consecutive draws until a dynamic-state command changes `dynamic`.
    std::shared_ptr<const RuntimeProgramControl> dynamicControl;
    if (const std::shared_ptr<void> *payload = invocation.snapshot->find(
            programControlKey(graphics.dynamicStateControl, RuntimeProgramControl::DynamicState))) {
        dynamicControl = std::static_pointer_cast<const RuntimeProgramControl>(*payload);
        dynamic = dynamicControl->dynamic;
    }

    std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline;
    std::map<std::string, BoundParameter> bound;
    pass.telemetry = {sizeof(VernonProgramGraphicsNodeTelemetry), 0, 0, 0, 0};
    const auto frameFor = [&](const std::shared_ptr<RuntimeBindingSetState> &set) {
        std::shared_ptr<const RuntimeBindingSetMatch> match = matchBindingSet(*set, pipeline);
        for (size_t index = 0; index < pass.frames.size(); ++index)
            if (pass.frames[index].match == match)
                return index;
        pass.frames.push_back({std::move(match), pipeline});
        return pass.frames.size() - 1;
    };
    for (size_t index = 0; index < recording.commands.size(); ++index) {
        const RuntimeRecordedCommand &command = recording.commands[index];
        const std::string path = nodePath + ".commands[" + std::to_string(index) + "]";
        switch (command.kind) {
        case VERNON_GRAPHICS_COMMAND_BIND_PIPELINE: {
            std::shared_ptr<const RuntimeGraphicsPipelineState> next =
                command.pipeline ? command.pipeline : passPipeline.value();
            if (next->pass != &executable || next->node != nodeId)
                return invalid(path, "pipeline was created for another graphics Node");
            pipeline = std::move(next);
            ++pass.telemetry.pipeline_binds;
            for (auto entry = bound.begin(); entry != bound.end();) {
                const bool keeps = !matchBindingSet(*entry->second.set, pipeline)->failures[entry->second.entry];
                entry = keeps ? std::next(entry) : bound.erase(entry);
            }
            break;
        }
        case VERNON_GRAPHICS_COMMAND_BIND_SET: {
            if (!pipeline)
                return invalid(path, "binding set is bound before any pipeline");
            const RecordedSetFrame &frame = pass.frames[frameFor(command.set)];
            for (const program::Diagnostic &failure : frame.match->failures)
                if (failure)
                    return failDiagnostic(&context, prefixedDiagnostic(failure, path));
            for (size_t entry = 0; entry < command.set->entries.size(); ++entry) {
                const RuntimeBindingSetEntry &binding = command.set->entries[entry];
                const std::optional<bool> isWritten = argumentIsWritten(context, binding.binding->argument, written);
                if (!isWritten)
                    return invalid(path + "." + binding.boundary->path, "cannot describe the bound image");
                if (*isWritten)
                    return failDiagnostic(&context, {recorded::kBindingSetHazard, recorded::kInvocationEntryPhase,
                                                     path + "." + binding.boundary->path,
                                                     "resource is written by a Node of the same invocation"});
                bound[binding.boundary->path] = {command.set, entry};
            }
            ++pass.telemetry.set_binds;
            break;
        }
        case VERNON_GRAPHICS_COMMAND_SET_VIEWPORT:
            std::copy(command.rect.begin(), command.rect.end(), dynamic.viewport);
            dynamicControl.reset();
            break;
        case VERNON_GRAPHICS_COMMAND_SET_SCISSOR:
            std::copy(command.rect.begin(), command.rect.end(), dynamic.scissor);
            dynamicControl.reset();
            break;
        case VERNON_GRAPHICS_COMMAND_SET_STENCIL_REFERENCE:
            dynamic.stencil_reference = command.stencilReference;
            dynamicControl.reset();
            break;
        case VERNON_GRAPHICS_COMMAND_DRAW: {
            if (!pipeline)
                return invalid(path, "draw has no bound pipeline");
            const std::optional<VernonIndexBinding> &index = command.draw->index;
            if (index && written.overlapsBuffer(index->resource, index->offset,
                                                static_cast<uint64_t>(index->index_count) * sizeof(uint32_t)))
                return failDiagnostic(&context,
                                      {recorded::kBindingSetHazard, recorded::kInvocationEntryPhase, path + ".index",
                                       "index buffer is written by a Node of the same invocation"});
            if (!dynamicControl) {
                auto control = std::make_shared<RuntimeProgramControl>();
                control->kind = RuntimeProgramControl::DynamicState;
                control->dynamic = dynamic;
                dynamicControl = std::move(control);
            }
            RecordedDraw draw;
            draw.pipeline = pipeline;
            draw.draw = command.draw;
            draw.dynamic = dynamicControl;
            const program::Program &program = *executableProgram(*pipeline->source);
            for (size_t parameter = 0; parameter < pipeline->endpoints.size(); ++parameter) {
                const recorded::PipelineEndpoint &endpoint = pipeline->endpoints[parameter];
                if (endpoint.fixedPassParameter) {
                    draw.arguments.push_back({false, 0, *endpoint.fixedPassParameter});
                    continue;
                }
                const std::string &name = program.abi.boundarySlots[*endpoint.parameterSlot].path;
                const auto binding = bound.find(name);
                if (binding != bound.end()) {
                    const size_t frame = frameFor(binding->second.set);
                    if (pass.frames[frame].match->endpoints[parameter]) {
                        draw.arguments.push_back({true, frame, parameter});
                        continue;
                    }
                }
                if (!endpoint.initialPassParameter)
                    return invalid(path, "per-draw parameter '" + name + "' is unbound");
                draw.arguments.push_back({false, 0, *endpoint.initialPassParameter});
            }
            pass.draws.push_back(std::move(draw));
            ++pass.telemetry.draws;
            break;
        }
        }
    }
    return VERNON_STATUS_OK;
}

VernonStatus validateGraphicsRecordings(VernonProgramInvocation &invocation,
                                        vernon::runtime::RecordedGraphicsPasses &passes) {
    VernonProgramExecutable &executable = *invocation.instance->pipeline;
    WrittenResources written;
    if (!collectWrittenResources(*executable.context, executable, *invocation.snapshot, written))
        return fail(executable.context, "cannot describe the images a recorded invocation writes");
    for (const auto &[node, recording] : invocation.recordings) {
        const VernonStatus status = validateGraphicsRecording(invocation, node, recording, written, passes.nodes[node]);
        if (status != VERNON_STATUS_OK)
            return status;
    }
    return VERNON_STATUS_OK;
}
} // namespace
} // extern "C++"

VernonRuntimeOperationStatus vernonRuntimeGraphicsPipelineCreate(VernonProgramExecutable *executable,
                                                                 VernonProgramNodeId node,
                                                                 VernonProgramExecutable *source,
                                                                 VernonGraphicsPipeline **output) {
    VernonRuntimeContext *context = executable ? executable->context : nullptr;
    return runtimeHandleResultBoundary(context, output,
                                       [&] { return createGraphicsPipelineOperation(executable, node, source); });
}

void vernonRuntimeGraphicsPipelineDestroy(VernonGraphicsPipeline *pipeline) {
    runtimeVoidBoundary(pipeline ? pipeline->context : nullptr, [&] { delete pipeline; });
}

VernonRuntimeOperationStatus vernonRuntimeBindingSetCreate(VernonGraphicsPipeline *pipeline,
                                                           VernonBindingSet **output) {
    return runtimeHandleResultBoundary(pipeline ? pipeline->context : nullptr, output,
                                       [&] { return createBindingSetOperation(pipeline); });
}

VernonStatus vernonRuntimeBindingSetBind(VernonBindingSet *set, const VernonProgramArgument *argument,
                                         const VernonProgramResourceLease *lease) {
    VernonRuntimeContext *context = set && set->state ? set->state->context : nullptr;
    return runtimeStatusBoundary(context, [&] {
        if (!set || !set->state || !argument)
            return fail(context, "invalid binding set binding");
        RuntimeBindingSetState &state = *set->state;
        if (state.sealed)
            return fail(context, "binding set is sealed");
        const VernonProgramExecutable &executable = *state.origin->source;
        if (!bindableParameterSlot(executable, argument->slot))
            return fail(context, "binding set argument does not reference an exported parameter");
        const program::BoundarySlot &boundary = executableProgram(executable)->abi.boundarySlots[argument->slot];
        if (const char *diagnostic = programArgumentDiagnostic(boundary, *argument))
            return fail(context, diagnostic);
        std::vector<char> supplied;
        const program::BoundarySlot *target = nullptr;
        if (program::Diagnostic diagnostic =
                correspondParameter(boundary, *argument, *state.origin, boundary.path, supplied, target)) {
            diagnostic.phase = "binding";
            return failDiagnostic(context, diagnostic);
        }
        auto made = makeProgramBinding(*argument, lease);
        if (made.isErr())
            return publishRuntimeError(context, std::move(made).error());
        const auto existing =
            std::find_if(state.entries.begin(), state.entries.end(),
                         [&](const RuntimeBindingSetEntry &entry) { return entry.boundary == &boundary; });
        if (existing != state.entries.end())
            existing->binding = std::move(made).value();
        else
            state.entries.push_back({&boundary, std::move(made).value()});
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeBindingSetSeal(VernonBindingSet *set) {
    VernonRuntimeContext *context = set && set->state ? set->state->context : nullptr;
    return runtimeStatusBoundary(context, [&] {
        if (!set || !set->state || set->state->sealed)
            return fail(context, "invalid binding set seal");
        set->state->sealed = true;
        return VERNON_STATUS_OK;
    });
}

void vernonRuntimeBindingSetDestroy(VernonBindingSet *set) {
    runtimeVoidBoundary(set && set->state ? set->state->context : nullptr, [&] { delete set; });
}

VernonStatus vernonRuntimeProgramInvocationRecordGraphics(VernonProgramInvocation *invocation, VernonProgramNodeId node,
                                                          const VernonGraphicsCommand *commands, size_t commandCount,
                                                          const VernonProgramResourceLease *leases, size_t leaseCount) {
    VernonProgramExecutable *executable = invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
    VernonRuntimeContext *context = executable ? executable->context : nullptr;
    return runtimeStatusBoundary(context, [&] {
        if (!invocation || !executable || invocation->executed || invocation->finished || (commandCount && !commands) ||
            (leaseCount && !leases))
            return fail(context, "invalid graphics recording");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(context, "Program invocation is closing", vernon::toVernonStatus(invocationPin.error()));
        if (!forwardGraphicsNode(*executable, node))
            return fail(context, "graphics recording Node " + std::to_string(node) +
                                     " is not a resolved graphics Node of the executable");
        RuntimeGraphicsRecording appended;
        appended.commands.reserve(commandCount);
        for (size_t index = 0; index < commandCount; ++index) {
            const VernonGraphicsCommand &source = commands[index];
            const std::string path = "graphics recording command " + std::to_string(index);
            RuntimeRecordedCommand command;
            command.kind = source.kind;
            switch (source.kind) {
            case VERNON_GRAPHICS_COMMAND_BIND_PIPELINE:
                if (source.pipeline && (!source.pipeline->state || source.pipeline->context != context))
                    return fail(context, path + " has an invalid pipeline or one of another Runtime context");
                if (source.pipeline)
                    command.pipeline = source.pipeline->state;
                break;
            case VERNON_GRAPHICS_COMMAND_BIND_SET:
                if (!source.binding_set || !source.binding_set->state || !source.binding_set->state->sealed)
                    return fail(context, path + " binds a missing or unsealed binding set");
                if (source.binding_set->state->context != context)
                    return fail(context, path + " binds a binding set of another Runtime context");
                command.set = source.binding_set->state;
                break;
            case VERNON_GRAPHICS_COMMAND_SET_VIEWPORT:
            case VERNON_GRAPHICS_COMMAND_SET_SCISSOR:
                std::copy(std::begin(source.rect), std::end(source.rect), command.rect.begin());
                break;
            case VERNON_GRAPHICS_COMMAND_SET_STENCIL_REFERENCE:
                if (source.stencil_reference > 0xff)
                    return fail(context, path + " has a stencil reference above 255");
                command.stencilReference = source.stencil_reference;
                break;
            case VERNON_GRAPHICS_COMMAND_DRAW:
                command.draw = makeDrawCommandControl(&source.draw);
                if (!command.draw)
                    return fail(context, path + " has an invalid DrawCommand");
                break;
            default:
                return fail(context, path + " has an unknown kind");
            }
            appended.commands.push_back(std::move(command));
        }
        for (size_t index = 0; index < leaseCount; ++index) {
            auto retained = retainProgramResourceLease(leases[index]);
            if (retained.isErr())
                return publishRuntimeError(context, std::move(retained).error());
            if (retained.value())
                appended.leases.push_back(std::move(retained).value());
        }
        RuntimeGraphicsRecording &recording = invocation->recordings[node];
        recording.commands.insert(recording.commands.end(), std::make_move_iterator(appended.commands.begin()),
                                  std::make_move_iterator(appended.commands.end()));
        recording.leases.insert(recording.leases.end(), std::make_move_iterator(appended.leases.begin()),
                                std::make_move_iterator(appended.leases.end()));
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationGetGraphicsTelemetry(const VernonProgramInvocation *invocation,
                                                                VernonProgramNodeId node,
                                                                VernonProgramGraphicsNodeTelemetry *output) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeStatusBoundary(context, [&] {
        if (!invocation || !output || output->struct_size < sizeof(*output))
            return fail(context, "invalid graphics telemetry query");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(context, "Program invocation is closing", vernon::toVernonStatus(invocationPin.error()));
        const auto found = invocation->graphicsTelemetry.find(node);
        if (found == invocation->graphicsTelemetry.end())
            return fail(context, "graphics Node " + std::to_string(node) + " was not recorded by Execute");
        *output = found->second;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationExecute(VernonProgramInvocation *invocation, uint8_t retainPullback,
                                                   VernonInvocationMutationOutcome *publicOutcome) {
    VernonProgramExecutable *pipeline = invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!invocation || invocation->executed || invocation->finished || !pipeline)
            return fail(pipeline ? pipeline->context : nullptr, "invalid persistent Program invocation");
        if (!vernon::runtime::program_execution::preparePublicOutcome(publicOutcome, mutationCapacity(*pipeline)))
            return fail(pipeline->context, "invalid Program mutation outcome storage");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        auto submissionChild = vernon::runtime::RuntimeChildLifecycle::reserve(pipeline->context->owner);
        if (submissionChild.isErr())
            return fail(pipeline->context, "runtime context cannot admit a submission",
                        vernon::toVernonStatus(submissionChild.error()));
        auto submission = std::make_unique<VernonSubmission>(
            *pipeline->context, std::make_shared<vernon::runtime::program_execution::SubmissionRecord>(
                                    *pipeline->context, invocation->completionMode));
        submission->lifecycle.emplace(std::move(submissionChild).value());
        invocation->executed = true;
        auto snapshot = invocation->transaction->freeze();
        if (snapshot.isErr())
            return fail(pipeline->context, "cannot freeze Program invocation bindings", VERNON_STATUS_INTERNAL_ERROR);
        invocation->snapshot = std::move(snapshot).value();
        std::vector<VernonProgramArgument> &arguments = invocation->executionWorkspace->boundaryArguments;
        arguments.clear();
        const program::Program &program = *executableProgram(*pipeline);
        const size_t count = publicBoundaryCount(program);
        if (arguments.capacity() < count)
            arguments.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            const program::BoundarySlot *parameter = publicBoundaryAt(program, index);
            if (!parameter)
                return fail(pipeline->context, "cannot reflect persistent Program boundary slot");
            const std::shared_ptr<void> *payload = invocation->snapshot->find(parameter->id);
            if (!payload)
                return fail(pipeline->context,
                            "persistent Program invocation has an unbound boundary slot '" + parameter->path + "'");
            auto binding = std::static_pointer_cast<RuntimeProgramBinding>(*payload);
            arguments.push_back(binding->argument);
        }
        std::shared_ptr<vernon::runtime::RecordedGraphicsPasses> recordedGraphics;
        if (!invocation->recordings.empty()) {
            recordedGraphics = std::make_shared<vernon::runtime::RecordedGraphicsPasses>();
            if (const VernonStatus status = validateGraphicsRecordings(*invocation, *recordedGraphics);
                status != VERNON_STATUS_OK)
                return status;
        }
        const ProgramInvocationContext programContext{
            *invocation->snapshot,        invocation->checkpointMemoryBudget,
            invocation->checkpointPolicy, invocation->executionWorkspace.get(),
            recordedGraphics.get(),
        };
        VernonPullback *pullback = nullptr;
        vernon::runtime::program_execution::InvocationMutationOutcome outcome;
        vernon::runtime::program_execution::SubmissionRecord &record = *submission->record;
        pipeline->context->submissions.admit({invocation->instance->submissionTransientBytes.load(),
                                              invocation->instance->submissionRhiSubmissions.load()});
        vernon::runtime::program_execution::TransientAllocationMeter meter(*pipeline->context);
        const VernonStatus status = vernon::runtime::program_execution::forwardProgramInvocation(
            *pipeline, arguments.data(), arguments.size(), pullback, outcome, record, &programContext,
            pipeline->programGraphId && retainPullback ? &invocation->nodePullbacks : nullptr, retainPullback != 0);
        invocation->instance->submissionTransientBytes.store(meter.bytes());
        invocation->instance->submissionRhiSubmissions.store(record.submittedCount());
        vernon::runtime::program_execution::publishPublicOutcome(outcome, publicOutcome);
        if (recordedGraphics)
            for (const auto &[node, pass] : recordedGraphics->nodes)
                invocation->graphicsTelemetry[node] = pass.telemetry;
        if (outcome.submission != vernon::runtime::program_execution::SubmissionState::NotSubmitted) {
            const bool pending = outcome.submission == vernon::runtime::program_execution::SubmissionState::Submitted;
            record.retain(invocation->snapshot);
            record.retain(invocation->executionWorkspace);
            if (recordedGraphics) {
                record.retain(std::move(recordedGraphics));
                record.retain(std::make_shared<std::map<VernonProgramNodeId, RuntimeGraphicsRecording>>(
                    std::exchange(invocation->recordings, {})));
            }
            record.setOutcome(std::move(outcome));
            if (pending)
                pipeline->context->submissions.push(submission->record, meter.bytes());
            invocation->submission = std::move(submission);
        }
        if (status != VERNON_STATUS_OK) {
            if (pullback)
                vernonProgramPullbackDestroy(pullback);
            return status;
        }
        invocation->pendingPullback = pullback;
        invocation->succeeded = true;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationCommit(VernonProgramInvocation *invocation,
                                                  VernonPullback **outputPullback) {
    VernonProgramExecutable *pipeline = invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (outputPullback)
            *outputPullback = nullptr;
        if (!invocation || !invocation->executed || !invocation->succeeded || invocation->finished || !pipeline)
            return fail(pipeline ? pipeline->context : nullptr, "invalid Program invocation commit");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        auto snapshot = invocation->transaction->commit();
        if (snapshot.isErr())
            return fail(pipeline->context, "cannot commit Program invocation bindings", VERNON_STATUS_INTERNAL_ERROR);
        invocation->snapshot = std::move(snapshot).value();
        invocation->finished = true;
        if (invocation->pendingPullback)
            vernon::runtime::program_execution::attachProgramSnapshot(*invocation->pendingPullback,
                                                                      invocation->snapshot);
        if (outputPullback)
            *outputPullback = std::exchange(invocation->pendingPullback, nullptr);
        else if (invocation->pendingPullback) {
            vernonProgramPullbackDestroy(invocation->pendingPullback);
            invocation->pendingPullback = nullptr;
        }
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationTakeSubmission(VernonProgramInvocation *invocation,
                                                          VernonSubmission **output) {
    VernonProgramExecutable *pipeline = invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (output)
            *output = nullptr;
        if (!invocation || !pipeline || !output)
            return fail(pipeline ? pipeline->context : nullptr, "invalid Program invocation submission retrieval");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        if (!invocation->submission)
            return fail(pipeline->context, "Program invocation has no submission to take");
        if (invocation->submission->lifecycle.value().publish().isErr())
            return fail(pipeline->context, "cannot publish Program invocation submission",
                        VERNON_STATUS_INTERNAL_ERROR);
        *output = invocation->submission.release();
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationGetNodePullback(VernonProgramInvocation *invocation,
                                                           VernonProgramNodeId node, VernonPullback **outputPullback) {
    VernonProgramExecutable *pipeline = invocation && invocation->instance ? invocation->instance->pipeline : nullptr;
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (outputPullback)
            *outputPullback = nullptr;
        if (!invocation || !invocation->finished || !invocation->succeeded || !pipeline || !pipeline->programGraphId ||
            !outputPullback)
            return fail(pipeline ? pipeline->context : nullptr, "invalid ProgramGraph node pullback retrieval");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        const auto declared = pipeline->programGraphNodeAutodiff.find(node);
        if (declared == pipeline->programGraphNodeAutodiff.end())
            return fail(pipeline->context, "ProgramGraph node is not differentiable");
        const auto retained = invocation->nodePullbacks.find(node);
        if (retained == invocation->nodePullbacks.end() || !retained->second)
            return fail(pipeline->context, "ProgramGraph node pullback is unavailable");
        auto transfer = vernon::runtime::program_execution::makeRetainedProgramPullback(*pipeline, retained->second,
                                                                                        invocation->snapshot);
        if (transfer.isErr())
            return fail(pipeline->context, "cannot retain ProgramGraph node pullback", VERNON_STATUS_INTERNAL_ERROR);
        *outputPullback = std::move(transfer).value().release();
        invocation->nodePullbacks.erase(retained);
        return VERNON_STATUS_OK;
    });
}

void vernonRuntimeProgramInvocationRollback(VernonProgramInvocation *invocation) {
    VernonRuntimeContext *context = invocationContext(invocation);
    runtimeVoidBoundary(context, [&] {
        if (!invocation || invocation->finished)
            return;
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return;
        invocation->transaction->rollback();
        if (invocation->pendingPullback) {
            vernonProgramPullbackDestroy(invocation->pendingPullback);
            invocation->pendingPullback = nullptr;
        }
        invocation->nodePullbacks.clear();
        invocation->succeeded = false;
        invocation->finished = true;
    });
}

void vernonRuntimeProgramInvocationDestroy(VernonProgramInvocation *invocation) {
    VernonRuntimeContext *context = invocationContext(invocation);
    runtimeVoidBoundary(context, [&] {
        if (!invocation)
            return;
        auto destruction = invocation->lifecycle.beginDestroy();
        if (destruction.isErr())
            return;
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete invocation;
    });
}

VernonStatus vernonRuntimeProgramInstanceGetTelemetry(const VernonProgramInstance *instance,
                                                      VernonProgramBindingTelemetry *output) {
    VernonRuntimeContext *context =
        instance && instance->state && instance->state->pipeline ? instance->state->pipeline->context : nullptr;
    return runtimeStatusBoundary(context, [&] {
        if (!instance || !instance->state || !output || output->struct_size < sizeof(*output))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto instancePin = instance->state->lifecycle.pin();
        if (instancePin.isErr())
            return vernon::toVernonStatus(instancePin.error());
        const vernon::runtime::program::BindingTelemetry telemetry = instance->state->bindings.telemetry();
        *output = {sizeof(*output),         telemetry.prepareCount, telemetry.reuseCount,
                   telemetry.rollbackCount, telemetry.uploadBytes,  telemetry.uploadRanges};
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonSubmissionGetState(const VernonSubmission *submission, VernonSubmissionState *output) {
    return runtimeStatusBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission || !output || !submission->lifecycle)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pin = submission->lifecycle.value().pin();
        if (pin.isErr())
            return vernon::toVernonStatus(pin.error());
        return submission->record->poll(*output);
    });
}

VernonStatus vernonSubmissionWait(VernonSubmission *submission) {
    return runtimeStatusBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission || !submission->lifecycle)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pin = submission->lifecycle.value().pin();
        if (pin.isErr())
            return vernon::toVernonStatus(pin.error());
        return submission->record->wait();
    });
}

VernonStatus vernonSubmissionGetMutationOutcome(const VernonSubmission *submission,
                                                VernonInvocationMutationOutcome *outcome) {
    return runtimeStatusBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission || !outcome || !submission->lifecycle)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pin = submission->lifecycle.value().pin();
        if (pin.isErr())
            return vernon::toVernonStatus(pin.error());
        const vernon::runtime::program_execution::InvocationMutationOutcome *completed = nullptr;
        if (const VernonStatus status = submission->record->completionOutcome(completed); status != VERNON_STATUS_OK)
            return status;
        if (!completed)
            return VERNON_STATUS_NOT_READY;
        if (!vernon::runtime::program_execution::preparePublicOutcome(outcome, completed->boundaries.size()))
            return fail(submission->context, "invalid submission mutation outcome storage");
        vernon::runtime::program_execution::publishPublicOutcome(*completed, outcome);
        return VERNON_STATUS_OK;
    });
}

void vernonSubmissionDestroy(VernonSubmission *submission) {
    runtimeVoidBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission)
            return;
        if (!submission->lifecycle)
            vernon::resultContractViolation();
        auto destruction = submission->lifecycle.value().beginDestroy();
        if (destruction.isErr())
            return;
        // The context may destroy its device once the destruction commits.
        (void)submission->record->wait();
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete submission;
    });
}

VernonStatus vernonRuntimeReferenceRhiBuffer(VernonRuntimeContext *context, VernonRhiBuffer buffer, uint64_t offset,
                                             uint64_t size, VernonRuntimeProviderResourceReference *output) {
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        if (!output)
            return fail(context, "invalid RHI buffer reference");
        auto referenced = referenceBackendRhiBuffer(*context, buffer, offset, size);
        if (referenced.isErr())
            return publishRuntimeError(context, std::move(referenced).error());
        *output = referenced.value();
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeReferenceRhiImageView(VernonRuntimeContext *context, VernonRhiImageView view,
                                                VernonRuntimeProviderResourceReference *output) {
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        if (!output)
            return fail(context, "invalid RHI image view reference");
        auto referenced = referenceBackendRhiImageView(*context, view);
        if (referenced.isErr())
            return publishRuntimeError(context, std::move(referenced).error());
        *output = referenced.value();
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeReferenceRhiSampler(VernonRuntimeContext *context, VernonRhiSampler sampler,
                                              VernonRuntimeProviderResourceReference *output) {
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        if (!output)
            return fail(context, "invalid RHI sampler reference");
        auto referenced = referenceBackendRhiSampler(*context, sampler);
        if (referenced.isErr())
            return publishRuntimeError(context, std::move(referenced).error());
        *output = referenced.value();
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeReferenceRhiCommandEncoder(VernonRuntimeContext *context, VernonRhiCommandEncoder encoder,
                                                     VernonRuntimeProviderObject *output) {
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto contextPin = context->operations.tryPin();
        if (contextPin.isErr())
            return vernon::toVernonStatus(contextPin.error());
        if (!output)
            return fail(context, "invalid RHI command encoder reference");
        auto referenced = referenceBackendCommandEncoder(*context, encoder);
        if (referenced.isErr())
            return publishRuntimeError(context, std::move(referenced).error());
        *output = referenced.value();
        return VERNON_STATUS_OK;
    });
}

} // extern "C"

VernonStageExecutable::~VernonStageExecutable() {
    if (backendState)
        vernon::runtime::destroyBackendPipeline(*this);
}
