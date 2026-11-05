#ifndef VERNON_RUNTIME_SCHEDULE_PASS_PLAN_H
#define VERNON_RUNTIME_SCHEDULE_PASS_PLAN_H

// The pass planning that every graphics device of a linked schedule shares: a submission plans each pass of its
// command buffer from the record it is given, reusing the plan an earlier call made of the same table and inputs, and
// groups adjacent graphics passes into render scopes where their attachments allow.

#include "VernonRHI.h"
#include "VernonSchedule.h"
#include "hal/hal.h"
#include "hal/native_buffers.h"
#include "runtime/capi/graphics_recording.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/graphics_scope_materializer.h"
#include "runtime/pipeline_groups.h"
#include "runtime/program_graphics_executor.h"
#include "runtime/resolved_execution_plan.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

struct VernonRuntimeContext;
struct VernonProgramExecutable;

namespace vernon::hal {

template <typename T> using Made = Result<T, std::string>;

// A graphics node of a resolved Program. The plan keeps the node's Stage, and with it the pipeline it prepared, alive;
// `node` and `resolved` point into the plan.
struct GraphicsSource {
    std::shared_ptr<const runtime::program::ResolvedExecutionPlan> plan;
    const runtime::program::Node *node{};
    const runtime::program::ResolvedNodePlan *resolved{};
    // The Program the node belongs to, which outlives the schedule; table entries' pipelines must be created for it.
    const VernonProgramExecutable *executable{};
};

// The executable of a pass node: its Program node and, for a graphics node, the node's own pipeline state.
class PassExecutable : public Executable {
public:
    explicit PassExecutable(std::shared_ptr<const GraphicsSource> source);

    const GraphicsSource &source() const { return *source_; }
    const VernonGraphicsState &state() const { return state_.state; }

private:
    std::shared_ptr<const GraphicsSource> source_;
    runtime::ProgramGraphicsState state_;
};

// A debug command of a planned pass table, copied from the call that planned it.
struct PlannedDebug {
    uint32_t kind{};
    std::string label;
};

const std::string &nodeLabel(const runtime::program::Node &node);

template <typename T> T readRecord(const uint8_t *record, uint64_t offset) {
    T value;
    std::memcpy(&value, record + offset, sizeof(value));
    return value;
}

// The binding set entries a pass table's entries have bound, per endpoint of the pipeline the current entry draws
// with, as a graphics recording binds them: a set binds the endpoints its entries correspond to in the current
// pipeline, and a pipeline change keeps each bound entry for the endpoint of the same parameter path.
class TableBindings {
public:
    Made<void> select(const std::shared_ptr<const RuntimeGraphicsPipelineState> &pipeline);
    Made<void> bind(const VernonBindingSet *set);
    bool supplies(size_t endpoint) const { return endpoint < bound_.size() && bound_[endpoint].set; }

    // Ends planning, keeping only the lowered frames. A cached plan must not own the last pipeline: a pipeline other
    // than pipeline 0 leases the Program it draws with, which cannot be destroyed while leased, and a cached plan
    // outlives the table it planned. The frames' draws are encoded only while the caller keeps the table, and with it
    // the pipelines.
    void release() {
        pipeline_.reset();
        bound_.clear();
    }

    // The frame of the set bound at `endpoint`, lowered for the current pipeline once, which holds the Stage's
    // argument at the endpoint and owns its groups.
    Made<const runtime::capi::InvocationSetFrame *> frame(VernonRuntimeContext &context, size_t endpoint);

private:
    struct Bound {
        RuntimeBindingSetState *set{};
        size_t entry{};
    };

    std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline_;
    std::vector<Bound> bound_;
    // The frames of the current pipeline, by set; a pipeline change lowers them again.
    std::unordered_map<RuntimeBindingSetState *, runtime::capi::InvocationSetFrame> frames_;
    // The matches whose groups the frames' owners name, kept while the plan is.
    std::vector<std::shared_ptr<const RuntimeBindingSetMatch>> retained_;
};

// The extents and byte strides a Stage argument points into when the Stage reads a view of the caller's Value.
struct StagedView {
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

// One draw command of a pass: the node's draw or stream, or a table entry's stream. The draw points into the
// caller's index binding, which the call reads until it returns.
struct PlannedDraw {
    const runtime::program::Program *program{};
    const runtime::program::ResolvedNodePlan *resolved{};
    VernonStageExecutable *stage{};
    const VernonGraphicsState *state{};
    std::vector<VernonProgramArgument> arguments;
    std::vector<StagedView> views;
    std::vector<const VernonProgramArgument *> argumentPointers;
    // Per Stage parameter: who supplies its argument, and the set frame that lowered it, or null for an argument the
    // call adapts. The frames are valid only while the draw is planned.
    std::vector<runtime::ArgumentOwner> owners;
    std::vector<const runtime::capi::InvocationSetFrame *> lowered;
    // (Stage parameter, node parameter) of each argument that is a node argument, which a reused plan takes the host
    // bytes of again.
    std::vector<std::pair<uint32_t, uint32_t>> nodeParameters;
    VernonDrawCommand drawCommand{};
    VernonDynamicState dynamic{};
    runtime::PlannedGraphicsDraw draw;
    bool streamed{};
    runtime::GraphicsDrawStream stream;
    // Encoded before the draw.
    std::vector<PlannedDebug> debug;
};

// The extents and byte strides of an internal buffer whose size follows the call, or of a tape's launch counts, as one
// call's argument names them.
struct InternalShape {
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

// A leaf copy as one call runs it: each region's bytes from `source` to `destination`, at offsets in the buffers
// the references name, with adjacent elements' ranges merged.
struct CopyRegion {
    uint64_t source{};
    uint64_t destination{};
    uint64_t size{};
};

struct PlannedCopy {
    VernonRuntimeProviderResourceReference source{};
    VernonRuntimeProviderResourceReference destination{};
    std::vector<CopyRegion> regions;
};

// A pass as one call draws or dispatches it. The targets point into the caller's RenderPass. Moving a plan keeps the
// addresses of its arguments and draws, which the owners of the draws' arguments and their groups name. A leaf copy
// is a compute pass with no executable and its `copy`.
struct PlannedPass {
    const PassExecutable *executable{};
    runtime::PlannedRenderTargets targets;
    runtime::GraphicsScopeReads reads;
    // Reserved before it is filled: the transient groups of the stream are keyed by each draw's address.
    std::vector<PlannedDraw> draws;
    size_t scope{};
    bool compute{};
    // The node's arguments, which the owners of the draws' arguments name.
    std::vector<VernonProgramArgument> nodeArguments;
    TableBindings bindings;
    std::vector<VernonProgramArgument> arguments;
    std::vector<StagedView> views;
    // Per parameter, what an argument of an internal buffer whose size follows the call points into.
    std::vector<InternalShape> internalShapes;
    VernonLaunchSize grid{};
    // The id of the cached plan this is, or zero for a plan of one call.
    uint64_t plan{};
    // The table's debug commands after its last entry.
    std::vector<PlannedDebug> trailingDebug;
    std::optional<PlannedCopy> copy;
};

// A table pass as an earlier call planned it: the table and revision it planned, and the words of what else of that
// call its draws depend on.
struct CachedPass {
    const VernonSchedulePassTable *table{};
    uint64_t revision{};
    std::vector<uint64_t> key;
    PlannedPass pass;
    // (draw, Stage parameter, node parameter) of each draw argument that is a host Tensor of the node, whose bytes a
    // reuse takes again; the key holds the rest of every argument.
    std::vector<std::tuple<PlannedDraw *, uint32_t, uint32_t>> hostArguments;
};

struct PassCache {
    // Held while a call plans and encodes, so a plan it reuses stays as it was planned until encoded.
    std::mutex mutex;
    // By command, the most recent plan last.
    std::vector<std::deque<CachedPass>> commands;
    uint64_t nextPlan{1};
};

// The passes of one call in command order, each planned in `fresh` or reused from the cache, and the render scopes
// the graphics passes fall in; a compute pass ends the scope before it.
struct PlannedPasses {
    std::deque<PlannedPass> fresh;
    std::vector<PlannedPass *> passes;
    std::vector<runtime::GraphicsRenderScope> scopes;
    // The buffer ranges the arguments of PassOffsets::zeroed parameters name, each once, which the call zeroes before
    // its first pass; a device's recording key holds them.
    std::vector<VernonRuntimeProviderResourceReference> zeroed;
    // The cached plans this call evicted, whose dependents the device drops.
    std::vector<uint64_t> evicted;
};

// The command buffer of a device that plans passes: the pass commands, the plans of their table passes and what the
// last submission drew.
class PassCommandBuffer : public CommandBuffer {
public:
    explicit PassCommandBuffer(std::vector<DispatchCommand> commands);
    const std::vector<DispatchCommand> &commands() const { return commands_; }
    // The plans of the table passes, which the schedule's calls, one at a time, reuse.
    PassCache &passCache() const { return *passCache_; }

    GraphicsTelemetry telemetry() const override {
        return {renderScopes_.load(std::memory_order_relaxed), draws_.load(std::memory_order_relaxed)};
    }
    void setTelemetry(uint64_t renderScopes, uint64_t draws) const {
        renderScopes_.store(renderScopes, std::memory_order_relaxed);
        draws_.store(draws, std::memory_order_relaxed);
    }
    void setTimed(bool timed) override { timed_.store(timed, std::memory_order_relaxed); }
    bool timed() const { return timed_.load(std::memory_order_relaxed); }

private:
    std::vector<DispatchCommand> commands_;
    std::unique_ptr<PassCache> passCache_;
    mutable std::atomic<uint64_t> renderScopes_{};
    mutable std::atomic<uint64_t> draws_{};
    std::atomic<bool> timed_{};
};

// Checks that every command is a pass of a pass executable.
Made<void> checkPassCommands(const std::vector<DispatchCommand> &commands, const char *device);

std::string runtimeMessage(VernonRuntimeContext &context, const RuntimeError &error, const char *fallback);

// The device buffers of the passes' internal buffers, allocated from the device's buffers on first use and kept by the
// device. A buffer whose size follows the call grows to the largest call so far; the buffer it outgrows stays until
// the device is idle, since a submission still running may read it.
class InternalBuffers {
public:
    struct Internal {
        // Holds the key alive, so that no other buffer takes its address while the entry exists.
        std::shared_ptr<const InternalBuffer> buffer;
        std::unique_ptr<Buffer> allocation;
        VernonRuntimeProviderResourceReference reference{};
        // The bytes a call zeroes, which the allocation may exceed.
        uint64_t bytes{};
    };

    // `buffers` outlives this; it is null on a device that cannot allocate, whose passes then read no internal buffer.
    explicit InternalBuffers(DeviceBuffers *buffers) : buffers_(buffers) {}
    InternalBuffers(const InternalBuffers &) = delete;
    InternalBuffers &operator=(const InternalBuffers &) = delete;
    // The device must be idle.
    ~InternalBuffers() = default;

    // The entries of the internal buffers the commands read, each once, as the call's planning sized them.
    Made<std::vector<const Internal *>> used(const std::vector<DispatchCommand> &commands);
    // The argument of Stage parameter `parameter` of a pass: the caller's, an attachment of a RenderPass the record
    // binds, or the internal buffer it reads, whose shape and strides for this call `shape` holds.
    Made<VernonProgramArgument> argument(const PassOffsets &offsets, size_t parameter, const uint8_t *record,
                                         InternalShape &shape);

private:
    Made<Internal> allocate(std::shared_ptr<const InternalBuffer> buffer, uint64_t bytes);
    // Called with mutex_ held; grows the entry to at least `bytes`.
    Made<const Internal *> internal(const std::shared_ptr<const InternalBuffer> &buffer, uint64_t bytes);
    // Called with mutex_ held; the buffer holding a tape's launch counts.
    Made<const Internal *> launch(const std::array<uint32_t, 3> &counts);

    DeviceBuffers *buffers_;
    std::mutex mutex_;
    std::map<const InternalBuffer *, Internal> internals_;
    std::vector<Internal> outgrown_;
    // Written once when created, since a recording in flight may read it; a schedule launches few distinct grids.
    std::map<std::array<uint32_t, 3>, Internal> launches_;
};

void addWords(std::vector<uint64_t> &key, const void *data, size_t size);
void addReference(std::vector<uint64_t> &key, const VernonRuntimeProviderResourceReference &reference);
// The words of a render scope's attachments and their load and store operations.
void addScopeKey(std::vector<uint64_t> &key, const runtime::GraphicsRenderScope &scope);
void addCopyKey(std::vector<uint64_t> &key, const PlannedCopy &copy);

VernonStatus describeImage(void *userData, VernonRuntimeProviderResourceReference resource,
                           VernonRuntimeProviderImageDescription *description);

class PassPlanner {
public:
    PassPlanner(VernonRuntimeContext &context, InternalBuffers &internals) : context_(context), internals_(internals) {}

    // Plans every pass of `commands` from `record`; the caller holds the cache's mutex until it encoded them.
    Made<void> plan(const PassCommandBuffer &commands, const uint8_t *record, PlannedPasses &planned);

    // What a graphics pass reads of the call besides its arguments.
    struct PassCall;

private:
    Made<void> planCompute(const PassExecutable &executable, const PassOffsets &offsets, const uint8_t *record,
                           PlannedPass &pass);
    Made<void> planCopy(const DispatchCommand &command, const uint8_t *record, PlannedPass &pass);
    Made<void> planInputs(const PassExecutable &executable, const PassOffsets &offsets, const uint8_t *record,
                          PlannedPass &pass, PassCall &call);
    Made<PlannedPass *> planGraphics(const PassExecutable &executable, const PassOffsets &offsets,
                                     const uint8_t *record, PassCache &cache, size_t command, PlannedPasses &planned);
    Made<void> planPass(const PassCall &call, PlannedPass &pass);
    Made<void> planDraw(const std::string &pass, size_t entry, const VernonRenderPass &renderPass,
                        const VernonDynamicState *dynamic, PlannedPass &planned, PlannedDraw &draw);

    VernonRuntimeContext &context_;
    InternalBuffers &internals_;
};

} // namespace vernon::hal

#endif
