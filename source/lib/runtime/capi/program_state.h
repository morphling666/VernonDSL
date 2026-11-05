#ifndef VERNON_RUNTIME_CAPI_PROGRAM_STATE_H
#define VERNON_RUNTIME_CAPI_PROGRAM_STATE_H

#include "VernonRuntime.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/pipeline_groups.h"
#include "runtime/program_execution/materialized_node_frame.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/program_graph_linker.h"
#include "runtime/program_graphics_executor.h"
#include "runtime/recorded_graphics.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_lifecycle.h"
#include "runtime/runtime_state.h"
#include "runtime/stage_binding_plan.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
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
    // A binding set's host Tensor points here: a sealed set's values never change after seal.
    std::vector<uint8_t> ownedHostBytes;

    void refresh() {
        if (argument.kind != VERNON_PROGRAM_TENSOR)
            return;
        argument.tensor.shape = metadata->shape.empty() ? nullptr : metadata->shape.data();
        argument.tensor.byte_strides = metadata->strides.empty() ? nullptr : metadata->strides.data();
        argument.tensor.element_layout.layout_hash = {metadata->layoutHash.data(), metadata->layoutHash.size()};
        argument.tensor.element_layout.leaves = metadata->leaves.empty() ? nullptr : metadata->leaves.data();
    }
};

// Keeps an executable alive: its destroy fails while the lease exists.
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
    vernon::runtime::ProgramGraphicsState graphicsState;
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

// A match's argument frame lowered for its pipeline's Stage once, then read by every pass that draws with it.
struct RuntimeLoweredSetFrame {
    vernon::runtime::program_execution::MaterializedNodeFrame frame;
    // Per Stage parameter; meaningful for the parameters the match supplies.
    std::vector<vernon::runtime::PlannedGraphicsArgument> planned;
};

// A sealed set prepared for one pipeline: the set's values inside that pipeline's Program.
struct RuntimeBindingSetMatch {
    std::weak_ptr<const RuntimeGraphicsPipelineState> pipeline;
    // Pipeline endpoints the set supplies, index-aligned with RuntimeGraphicsPipelineState::endpoints.
    std::vector<char> endpoints;
    // Per entry: why it does not correspond to a per-draw parameter of the pipeline; empty when it does. A
    // corresponding entry stays bound across a switch to this pipeline even if other entries of its set do not.
    std::vector<vernon::runtime::program::Diagnostic> failures;
    // Per entry: the pipeline endpoints it supplies; empty for an entry that fails.
    std::vector<std::vector<uint32_t>> entryEndpoints;
    std::shared_ptr<const vernon::runtime::program_execution::ProgramInvocationState> values;
    // The groups whose values only this match decides; draws in flight share them.
    mutable vernon::runtime::GroupStore groups;
    mutable std::mutex loweringMutex;
    // Made by the first pass that draws with the match; stays null for a frame that needs per-pass copies or Runtime
    // carriers, which every pass lowers again.
    mutable std::shared_ptr<const RuntimeLoweredSetFrame> lowered;
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

struct RecordedSetFrame {
    std::shared_ptr<const RuntimeBindingSetMatch> match;
    std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline;
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

vernon::Result<std::shared_ptr<void>, vernon::RuntimeError>
retainProgramResourceLease(const VernonProgramResourceLease &lease) noexcept;

const program::Program *executableProgram(const VernonProgramExecutable &pipeline);
size_t publicBoundaryCount(const program::Program &program);
const program::BoundarySlot *publicBoundaryAt(const program::Program &program, size_t publicIndex);
const program::BoundarySlot *parameterBoundaryAt(const VernonProgramExecutable &pipeline, size_t index);
const program::BoundarySlot *findParameterBoundary(const VernonProgramExecutable &pipeline, VernonStringView name);
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

} // namespace vernon::runtime::capi

#endif
