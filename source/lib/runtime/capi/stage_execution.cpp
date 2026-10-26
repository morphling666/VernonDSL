#include "VernonRuntime.h"
#include "execution_graph/execution_graph_internal.h"
#include "runtime/autodiff/tape_allocator_abi.h"
#include "runtime/backend_cpu.h"
#include "runtime/backend_stage_pipeline.h"
#include "runtime/capi/boundary.h"
#include "runtime/capi/program_state.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/graphics_scope_materializer.h"
#include "runtime/prepared_binding_plan.h"
#include "runtime/prepared_graphics_draw.h"
#include "runtime/program_execution/device_commands.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/program_execution/materialized_node_frame.h"
#include "runtime/program_execution/resolved_transfer_executor.h"
#include "runtime/program_execution/submission_queue.h"
#include "runtime/program_execution/submission_record.h"
#include "runtime/program_graphics_executor.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_dispatch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace vernon::runtime;
using namespace vernon::runtime::capi;

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

namespace {

const std::string &programNodeLabel(const program::Node &node) { return node.name.empty() ? node.stage : node.name; }

std::string programNodeName(const program::Node &node) {
    return "program." + std::to_string(node.id) + "." + programNodeLabel(node);
}

using vernon::execution::detail::CommandScope;

std::shared_ptr<CommandScope> nodeScope(const program::Node &node) {
    return std::make_shared<CommandScope>(CommandScope{programNodeLabel(node), node.id});
}

void setScope(std::vector<vernon::execution::detail::RhiCommandNodeEncoder> &encoders,
              const std::shared_ptr<const CommandScope> &scope) {
    for (vernon::execution::detail::RhiCommandNodeEncoder &encoder : encoders)
        encoder.scope = scope;
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
    auto stream = beginCommandStream(context, plan.commandEncoder);
    if (stream.isErr())
        return fail(&context, programNodeName(node) + ": cannot begin CPU commands",
                    vernon::toVernonStatus(std::move(stream).error()));
    auto invoked = invokeBackendComputePipeline(pipeline, stream.value(), plan, nullptr);
    auto ended = stream.value().end();
    if (invoked.isErr() || ended.isErr()) {
        const VernonStatus status =
            vernon::toVernonStatus(invoked.isErr() ? std::move(invoked).error() : std::move(ended).error());
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

VernonStatus describeRuntimeImage(void *userData, VernonRuntimeProviderResourceReference resource,
                                  VernonRuntimeProviderImageDescription *description) {
    auto described = describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource);
    if (described.isErr())
        return vernon::toVernonStatus(std::move(described).error());
    *description = described.value();
    return VERNON_STATUS_OK;
}

template <typename Handle> bool decodeRhiResourceKey(uint64_t value, Handle &handle) {
    const uint32_t encodedIndex = static_cast<uint32_t>(value);
    const uint32_t generation = static_cast<uint32_t>(value >> 32);
    if (!encodedIndex || !generation)
        return false;
    handle = {encodedIndex - 1, generation};
    return true;
}

// The vertex and index buffer reads of draws, merged per buffer and kind of read.
class GraphicsBufferReads {
public:
    bool add(const VernonRuntimeProviderResourceReference &resource, uint64_t offset, uint64_t size, bool index) {
        VernonRhiBuffer buffer{};
        if (!decodeRhiResourceKey(resource.resource.value, buffer) || resource.offset > UINT64_MAX - offset ||
            size > UINT64_MAX - resource.offset - offset)
            return false;
        merge({resource.resource.value, index}, {resource.offset + offset, resource.offset + offset + size});
        return true;
    }
    void merge(const GraphicsBufferReads &other) {
        for (const auto &[key, range] : other.ranges_)
            merge(key, range);
    }
    void append(vernon::execution::detail::CommandNode &command,
                std::vector<vernon::execution::detail::RhiCommandResourceBinding> &bindings) const {
        // Keys are ordered by buffer, so both kinds of read of one buffer are adjacent and bind it once.
        const uint64_t *previous = nullptr;
        for (const auto &[key, range] : ranges_) {
            VernonRhiBuffer buffer{};
            decodeRhiResourceKey(key.first, buffer);
            command.accesses.push_back(vernon::execution::detail::rhiBufferAccess(
                buffer, range.begin, range.end - range.begin, vernon::execution::AccessMode::Read,
                VERNON_RHI_STATE_SHADER_READ, VERNON_RHI_STAGE_VERTEX,
                key.second ? VERNON_RHI_ACCESS_INDEX_READ : VERNON_RHI_ACCESS_VERTEX_READ));
            if (!previous || *previous != key.first)
                bindings.push_back(
                    {vernon::rhi::encodeResourceKey(buffer), vernon::execution::ResourceKind::Buffer, buffer, {}});
            previous = &key.first;
        }
    }

private:
    struct Range {
        uint64_t begin{};
        uint64_t end{};
    };
    using Key = std::pair<uint64_t, bool>;

    void merge(const Key &key, const Range &range) {
        const auto [found, inserted] = ranges_.try_emplace(key, range);
        if (inserted)
            return;
        found->second.begin = std::min(found->second.begin, range.begin);
        found->second.end = std::max(found->second.end, range.end);
    }

    std::map<Key, Range> ranges_;
};

bool addVertexReads(const GraphicsDrawLayout &layout, const VernonProgramArgument *const *arguments,
                    GraphicsBufferReads &reads) {
    for (const GraphicsDrawLayout::VertexInput &input : layout.vertexInputs) {
        const VernonTensorView &tensor = arguments[input.parameter]->tensor;
        if (!reads.add(tensor.resource, tensor.byte_offset, tensor.byte_size, false))
            return false;
    }
    return true;
}

void addSampledRead(const GraphicsDrawLayout &layout, uint32_t parameter, const VernonProgramArgument &argument,
                    const PlannedGraphicsArgument &planned, GraphicsScopeReads &reads) {
    if (std::find(layout.sampledImages.begin(), layout.sampledImages.end(), parameter) != layout.sampledImages.end())
        addGraphicsScopeRead(reads, argument.image.view, planned.subresources);
}

// A binding set's frame as one invocation draws with it, and the owner of the arguments it supplies.
struct InvocationSetFrame {
    std::shared_ptr<const RuntimeLoweredSetFrame> lowered;
    ArgumentOwner owner;
};

// A match lowers its frame once and keeps it, unless the frame needs per-invocation copies or carriers; such a frame
// and its groups belong to the invocation.
bool lowerSetFrame(VernonRuntimeContext &context, const RecordedSetFrame &frame, InvocationSetFrame &output,
                   std::string &error) {
    const RuntimeBindingSetMatch &match = *frame.match;
    std::lock_guard lock(match.loweringMutex);
    if (!match.lowered) {
        const RuntimeGraphicsPipelineState &pipeline = *frame.pipeline;
        const VernonStageExecutable &stage = *pipeline.sourcePlan->stage;
        auto lowered = std::make_shared<RuntimeLoweredSetFrame>();
        if (!vernon::runtime::program_execution::materializeNodeFrame(
                *match.values, *executableProgram(*pipeline.source), *pipeline.sourceNode, *pipeline.sourcePlan, {},
                lowered->frame, error, &match.endpoints))
            return false;
        lowered->planned.resize(stage.bindingProjection.parameters.size());
        for (uint32_t parameter = 0; parameter < lowered->planned.size(); ++parameter)
            if (match.endpoints[parameter] &&
                !planGraphicsArgument(stage.bindingProjection, stage.drawLayout, parameter,
                                      lowered->frame.arguments[parameter], describeRuntimeImage, &context,
                                      lowered->planned[parameter], error))
                return false;
        const vernon::runtime::program_execution::MaterializedNodeFrame &made = lowered->frame;
        if (!made.copiesBefore.empty() || !made.copiesAfter.empty() || !made.deviceCopiesBefore.empty() ||
            !made.deviceCopiesAfter.empty() || !made.deviceEndpointCarriers.empty()) {
            output = {lowered, {lowered.get(), nullptr, nullptr}};
            return true;
        }
        match.lowered = std::move(lowered);
    }
    output = {match.lowered, {&match, &match.groups, nullptr}};
    return true;
}

// One draw of a managed graphics batch. Its arguments and their owners start at `firstArgument` in the batch, one per
// Stage parameter.
struct ManagedGraphicsDraw {
    VernonStageExecutable *stage{};
    const VernonGraphicsState *graphicsState{};
    const PlannedRenderTargets *targets{};
    size_t firstArgument{};
    // Identifies the draw inputs no parameter decides.
    const void *command{};
    PlannedGraphicsDraw draw;
    // What planning a recorded draw's values reads of its arguments; null for a draw without a recording.
    std::shared_ptr<const GraphicsVertexRecords> records;
};

// What one graphics Node adds to a batch; its draws count arguments from its own first argument.
struct ManagedGraphicsNode {
    std::shared_ptr<PlannedRenderTargets> targets{std::make_shared<PlannedRenderTargets>()};
    std::vector<ManagedGraphicsDraw> draws;
    std::vector<const VernonProgramArgument *> arguments;
    std::vector<ArgumentOwner> owners;
    GraphicsScopeReads reads;
    GraphicsBufferReads bufferReads;
    std::vector<std::shared_ptr<const void>> retained;
    // The control that `targets` point into.
    std::shared_ptr<const RuntimeProgramControl> renderPass;
    // What the Node's key names; see GraphicsNodeKey.
    std::vector<std::weak_ptr<const void>> identities;
    // False when an input of the Node lives only as long as the invocation, so that its commands cannot be stored.
    bool storable{true};
    // The draw values of a recorded Node.
    std::shared_ptr<const RecordedDrawPayload> payload;
};

struct ManagedGraphicsBatchMember {
    std::string label;
    // End of the member's draws in `ManagedGraphicsCommandBatch::draws`.
    size_t drawEnd{};
    // Positions count from the member's first draw.
    std::vector<RecordedDebugCommand> debugCommands;
    // The frame of the Node, and whether the Node is keyed, so that its groups belong to the stored scope.
    const vernon::runtime::program_execution::MaterializedNodeFrame *frame{};
    bool keyed{};
};

// What decides the commands of one graphics Node: Nodes with equal keys plan and encode equal commands. The pointers
// name immutable inputs; a stored scope replays only while each is alive, since an input made after one died may take
// its address.
struct GraphicsNodeKey {
    // The validated recording, or the draw and dynamic-state controls of a Node without one.
    const void *pass{};
    const void *draw{};
    const void *dynamic{};
    const void *renderPass{};
    // The Node, its staged attachments and the arguments of its frame.
    std::vector<uint64_t> words;

    bool operator==(const GraphicsNodeKey &other) const {
        return pass == other.pass && draw == other.draw && dynamic == other.dynamic && renderPass == other.renderPass &&
               words == other.words;
    }
};

constexpr uint32_t kFixedArgument = UINT32_MAX;

// An updatable group of a stored scope, and what fills it again when the scope replays.
struct StoredGroupSite {
    VernonRuntimeCoreGroup *group{};
    const PreparedGraphicsBindingPlan *plan{};
    std::vector<uint32_t> slots;
    std::array<float, 2> resolution{};
    // The scope member whose frame supplies the arguments that `frameArguments` index.
    size_t member{};
    // Per Stage parameter: the index of its argument in the member's frame, or kFixedArgument for the argument in
    // `arguments`, which a binding set lowered once and keeps while the scope can replay.
    std::vector<uint32_t> frameArguments;
    std::vector<const VernonProgramArgument *> arguments;
    // The inline bytes of the group's values, as last written.
    std::vector<uint8_t> bytes;
};

constexpr size_t kDrawPayloadWords = sizeof(VernonRuntimeProviderDrawPayload) / sizeof(uint32_t);

// A recorded draw of a stored Node: what planning other values for it reads, and where its payload is in the batch's.
struct StoredGraphicsDraw {
    std::shared_ptr<const GraphicsVertexRecords> records;
    VernonPrimitiveTopology topology{};
    // Other values replay the draw only with an equal resolution, which the stored groups were filled for.
    std::array<float, 2> resolution{};
    size_t payloadWord{SIZE_MAX};
};

// The draw values a match of a stored Node last planned, and by draw the payload of each of their draws whose values
// differ from the stored ones. Planning reads only the stored Node and the values, so a later match reuses it for the
// draws whose values it still has, whether or not the replay that planned it was encoded.
struct MatchedDrawValues {
    std::shared_ptr<const RecordedDrawPayload> values;
    std::vector<std::pair<size_t, VernonRuntimeProviderDrawPayload>> draws;
};

// What a stored Node shows the Nodes after it: what fusing decides on, and the buffers its draws read.
struct StoredGraphicsMember {
    // A copy of the Node's render pass without its leases, which `targets` point into.
    std::shared_ptr<const RuntimeProgramControl> renderPass;
    PlannedRenderTargets targets;
    GraphicsScopeReads reads;
    GraphicsBufferReads bufferReads;
    // The draw values of a recorded Node as stored, and its draws.
    std::shared_ptr<const RecordedDrawPayload> payload;
    std::vector<StoredGraphicsDraw> draws;
    // Read and replaced only with std::atomic_load and std::atomic_store: concurrent invocations match the same scope.
    mutable std::shared_ptr<const MatchedDrawValues> matched;
};

} // namespace

struct vernon::runtime::StoredGraphicsScope {
    explicit StoredGraphicsScope(VernonRuntimeProviderGroupLifetime lifetime) : groups(lifetime) {}

    std::vector<GraphicsNodeKey> keys;
    std::vector<StoredGraphicsMember> members;
    StoredPacketBatch batch;
    std::vector<std::weak_ptr<const void>> identities;
    // The groups of the arguments that no binding set owns: updatable when the provider writes updatable groups, else
    // persistent, holding only arguments that a Program instance binding does not own.
    GroupStore groups;
    // The updatable groups of `groups`. A replay writes those whose bytes changed before it encodes `batch`, and
    // `replays` keeps concurrent replays from interleaving their writes.
    mutable std::vector<StoredGroupSite> sites;
    mutable std::mutex replays;

    bool alive() const {
        return std::none_of(identities.begin(), identities.end(),
                            [](const std::weak_ptr<const void> &identity) { return identity.expired(); });
    }
};

namespace {

// What planning a graphics Node reads, kept for a Node of a replayed scope until the scope either completes or has to
// plan its Nodes after all.
struct GraphicsMemberInputs {
    const program::Node *node{};
    const program::ResolvedNodePlan *resolved{};
    RecordedGraphicsNode *recorded{};
    std::shared_ptr<vernon::runtime::program_execution::MaterializedNodeFrame> frame;
    std::vector<ArgumentOwner> owners;
    // The bound control, and the control with the staged attachments the Node renders to.
    std::shared_ptr<const RuntimeProgramControl> boundRenderPass;
    std::shared_ptr<const RuntimeProgramControl> renderPass;
    std::shared_ptr<const RuntimeProgramControl> draw;
    std::shared_ptr<const RuntimeProgramControl> dynamic;
    // Empty when the Node's commands cannot be stored.
    std::optional<GraphicsNodeKey> key;
    // The payload words, by position in the replayed batch's, of the draws whose values differ from the stored ones.
    std::vector<std::pair<size_t, VernonRuntimeProviderDrawPayload>> payload;
};

// Whether the draw values of `member` replay `stored`: each draw whose values differ from the stored ones, planned
// with them, keeps what the stored commands depend on, its payload then going to `payload`. Only the draws whose values
// also differ from those of the last match are planned.
bool replaysDrawValues(const StoredGraphicsMember &stored, const GraphicsMemberInputs &member,
                       std::vector<std::pair<size_t, VernonRuntimeProviderDrawPayload>> &payload) {
    payload.clear();
    const std::shared_ptr<const RecordedDrawPayload> values =
        member.recorded ? member.recorded->payload : std::shared_ptr<const RecordedDrawPayload>();
    if (!values || values == stored.payload)
        return true;
    if (values->draws.size() != stored.draws.size())
        return false;
    const auto sameValues = [&](const RecordedDrawPayload &other, size_t index) {
        return values->draws[index] == other.draws[index] &&
               sameDynamicState(values->dynamics[index]->dynamic, other.dynamics[index]->dynamic);
    };
    const auto toPayload = [&](const MatchedDrawValues &matched) {
        for (const auto &[index, words] : matched.draws)
            payload.emplace_back(stored.draws[index].payloadWord, words);
    };
    const std::shared_ptr<const MatchedDrawValues> last = std::atomic_load(&stored.matched);
    if (last && last->values == values) {
        toPayload(*last);
        return true;
    }
    auto matched = std::make_shared<MatchedDrawValues>();
    matched->values = values;
    size_t reused = 0;
    std::string error;
    for (size_t index = 0; index < stored.draws.size(); ++index) {
        if (sameValues(*stored.payload, index))
            continue;
        if (last) {
            while (reused < last->draws.size() && last->draws[reused].first < index)
                ++reused;
            if (reused < last->draws.size() && last->draws[reused].first == index && sameValues(*last->values, index)) {
                matched->draws.push_back(last->draws[reused]);
                continue;
            }
        }
        const StoredGraphicsDraw &draw = stored.draws[index];
        PlannedGraphicsDraw planned;
        VernonRuntimeCoreDrawInvocation invocation;
        VernonRuntimeProviderDrawPayload words{};
        if (draw.payloadWord == SIZE_MAX ||
            !planGraphicsDrawValues(*draw.records, stored.targets, stored.renderPass->renderPass,
                                    &values->dynamics[index]->dynamic, values->draws[index]->draw, planned, error) ||
            planned.resolution != draw.resolution ||
            !prepareGraphicsDrawInvocation(planned, draw.topology, invocation, error) ||
            vernonRuntimeCoreDrawPayload(&invocation, &words) != VERNON_STATUS_OK)
            return false;
        matched->draws.emplace_back(index, words);
    }
    toPayload(*matched);
    std::atomic_store(&stored.matched, std::shared_ptr<const MatchedDrawValues>(std::move(matched)));
    return true;
}

// The draws of one rendering scope, which either plans and encodes its members, storing them in `built`, or replays a
// stored scope. Recorded draws point into validated recordings, which the submission keeps until the batch is encoded.
struct ManagedGraphicsCommandBatch {
    VernonRuntimeContext *context{};
    // The plan's stream, which the plan retains with the batch.
    EncoderCommandStream *stream{};
    GraphicsRenderScope scope;
    std::vector<ManagedGraphicsDraw> draws;
    std::vector<const VernonProgramArgument *> arguments;
    std::vector<ArgumentOwner> owners;
    std::vector<ManagedGraphicsBatchMember> members;
    // Its label gains `+<member label>` as each member fuses.
    std::shared_ptr<CommandScope> commandScope;
    // The distinct resolutions of the draws, which identify their command inputs.
    std::deque<std::array<float, 2>> resolutions;
    std::vector<std::shared_ptr<const void>> retained;
    // Published to `scopes` under `opener` once encoded, when every member can be stored; keyed members' groups.
    std::shared_ptr<StoredGraphicsScope> built;
    // The groups of the members that are not keyed.
    GroupStore groups{VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT};
    bool storable{true};
    StoredGraphicsScopes *scopes{};
    uint64_t opener{};
    // The stored scope replayed instead, whose first `inputs.size()` members matched.
    std::shared_ptr<const StoredGraphicsScope> replay;
    std::vector<GraphicsMemberInputs> inputs;
};

const void *resolutionIdentity(ManagedGraphicsCommandBatch &batch, const std::array<float, 2> &resolution) {
    for (const std::array<float, 2> &known : batch.resolutions)
        if (known == resolution)
            return &known;
    return &batch.resolutions.emplace_back(resolution);
}

void appendManagedGraphicsNode(ManagedGraphicsCommandBatch &batch, ManagedGraphicsNode &&node,
                               const std::optional<GraphicsNodeKey> &key) {
    const size_t offset = batch.arguments.size();
    batch.arguments.insert(batch.arguments.end(), node.arguments.begin(), node.arguments.end());
    batch.owners.insert(batch.owners.end(), node.owners.begin(), node.owners.end());
    batch.draws.reserve(batch.draws.size() + node.draws.size());
    for (ManagedGraphicsDraw &draw : node.draws) {
        draw.firstArgument += offset;
        draw.command = resolutionIdentity(batch, draw.draw.resolution);
        batch.draws.push_back(draw);
    }
    batch.storable = batch.storable && node.storable && key;
    if (batch.storable) {
        auto renderPass = std::make_shared<RuntimeProgramControl>(*node.renderPass);
        renderPass->leases.clear();
        renderPass->refresh();
        StoredGraphicsMember &stored = batch.built->members.emplace_back(
            StoredGraphicsMember{renderPass, *node.targets, node.reads, node.bufferReads, node.payload, {}});
        if (node.payload)
            for (const ManagedGraphicsDraw &draw : node.draws)
                stored.draws.push_back({draw.records, draw.graphicsState->topology, draw.draw.resolution});
        for (const VernonColorAttachment *&attachment : stored.targets.attachments)
            attachment = &renderPass->colors[attachment - node.renderPass->colors.data()];
        if (stored.targets.depthAttachment)
            stored.targets.depthAttachment = &*renderPass->depth;
        batch.built->keys.push_back(*key);
        batch.built->identities.insert(batch.built->identities.end(), node.identities.begin(), node.identities.end());
    }
    batch.retained.push_back(std::move(node.targets));
    batch.retained.insert(batch.retained.end(), std::make_move_iterator(node.retained.begin()),
                          std::make_move_iterator(node.retained.end()));
}
// The draw of a keyed member whose groups a stored scope makes.
struct GroupSiteDraw {
    CommandStream &commands;
    ManagedGraphicsCommandBatch &batch;
    const ManagedGraphicsDraw &draw;
    size_t member{};
};

// Writes a group's first values before the draw binds it, and keeps what fills it on replay.
VernonStatus recordGroupSite(void *userData, VernonRuntimeCoreGroup *group, const PreparedGraphicsBindingPlan &plan,
                             const std::vector<uint32_t> &slots, const VernonRuntimeProviderBindingValue *values,
                             size_t valueCount) {
    const auto &made = *static_cast<const GroupSiteDraw *>(userData);
    const std::vector<VernonProgramArgument> &frame = made.batch.members[made.member].frame->arguments;
    const size_t parameters = made.draw.stage->bindingProjection.parameters.size();
    StoredGroupSite site{group, &plan, slots, made.draw.draw.resolution, made.member};
    site.arguments.assign(made.batch.arguments.begin() + static_cast<std::ptrdiff_t>(made.draw.firstArgument),
                          made.batch.arguments.begin() +
                              static_cast<std::ptrdiff_t>(made.draw.firstArgument + parameters));
    site.frameArguments.assign(parameters, kFixedArgument);
    for (size_t parameter = 0; parameter < parameters; ++parameter)
        if (const VernonProgramArgument *argument = site.arguments[parameter];
            argument >= frame.data() && argument < frame.data() + frame.size())
            site.frameArguments[parameter] = static_cast<uint32_t>(argument - frame.data());
    for (size_t value = 0; value < valueCount; ++value)
        if (inlineGroupValue(values[value])) {
            const auto *bytes = static_cast<const uint8_t *>(values[value].payload.inline_value.data);
            site.bytes.insert(site.bytes.end(), bytes, bytes + values[value].payload.inline_value.size);
        }
    if (auto updated = made.commands.updateGroup(group, values, valueCount); updated.isErr())
        return vernon::toVernonStatus(std::move(updated).error());
    made.batch.built->sites.push_back(std::move(site));
    return VERNON_STATUS_OK;
}

// Fills the updatable groups of the replayed scope from the frames of the Nodes it replays, and writes those whose
// bytes changed. Called outside a rendering scope, with the scope's replays serialized.
RuntimeResult<void> updateStoredGroups(CommandStream &commands, const ManagedGraphicsCommandBatch &batch) {
    std::vector<const VernonProgramArgument *> arguments;
    std::vector<VernonRuntimeProviderBindingValue> values;
    std::vector<std::vector<uint8_t>> storage;
    std::vector<uint8_t> bytes;
    for (StoredGroupSite &site : batch.replay->sites) {
        const std::vector<VernonProgramArgument> &frame = batch.inputs[site.member].frame->arguments;
        arguments = site.arguments;
        for (size_t parameter = 0; parameter < arguments.size(); ++parameter)
            if (site.frameArguments[parameter] != kFixedArgument)
                arguments[parameter] = &frame[site.frameArguments[parameter]];
        values.assign(site.slots.size(), {});
        storage.resize(site.slots.size());
        bytes.clear();
        std::string error;
        for (size_t value = 0; value < site.slots.size(); ++value) {
            if (!fillPreparedGraphicsBindingValue(*site.plan, {arguments.data(), site.resolution}, site.slots[value],
                                                  values[value], storage[value], error)) {
                if (invocationDiagnostic(*batch.context).empty())
                    invocationDiagnostic(*batch.context) = error;
                return RuntimeResult<void>{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                                                            {"refill_stored_graphics_group", 0, 0}})};
            }
            values[value].slot = static_cast<uint32_t>(value);
            if (inlineGroupValue(values[value])) {
                const auto *data = static_cast<const uint8_t *>(values[value].payload.inline_value.data);
                bytes.insert(bytes.end(), data, data + values[value].payload.inline_value.size);
            }
        }
        if (bytes == site.bytes)
            continue;
        if (auto updated = commands.updateGroup(site.group, values.data(), values.size()); updated.isErr())
            return updated;
        site.bytes.swap(bytes);
    }
    return RuntimeResult<void>{vernon::ok()};
}

// `stored`, the draw as the scope being built stores it, learns where the draw's payload is.
VernonRhiStatus encodeManagedGraphicsDraw(CommandStream &commands, GraphicsStream &stream,
                                          ManagedGraphicsCommandBatch &batch, const ManagedGraphicsDraw &draw,
                                          size_t member, StoredGraphicsDraw *stored) {
    const bool keyed = batch.members[member].keyed;
    GroupSiteDraw made{commands, batch, draw, member};
    const GraphicsCommand command{
        batch.arguments.data() + draw.firstArgument,
        draw.targets,
        draw.graphicsState,
        draw.draw,
        {batch.owners.data() + draw.firstArgument, draw.command, keyed ? &batch.built->groups : &batch.groups, {}},
        commands.encoder(),
        keyed ? GraphicsGroupSink{recordGroupSite, &made} : GraphicsGroupSink{}};
    auto invoked = invokeBackendPipeline(*draw.stage, commands.packets(), command, stream);
    if (invoked.isOk()) {
        if (stored) {
            size_t words = 0;
            vernonRuntimeCorePacketsGetPayload(commands.packets(), &words);
            stored->payloadWord = words - kDrawPayloadWords;
        }
        return VERNON_RHI_STATUS_OK;
    }
    const vernon::RuntimeError error = std::move(invoked).error();
    if (invocationDiagnostic(*draw.stage->context).empty())
        invocationDiagnostic(*draw.stage->context) =
            error.context.operation ? error.context.operation : "managed graphics draw failed";
    return vernon::toVernonRhiStatus(error);
}

VernonRhiStatus recordedCommandStatus(VernonRuntimeContext &context, VernonStatus status, const char *operation) {
    if (status == VERNON_STATUS_OK)
        return VERNON_RHI_STATUS_OK;
    if (invocationDiagnostic(context).empty())
        invocationDiagnostic(context) = operation;
    return vernon::toVernonRhiStatus(vernon::runtimeErrorFromStatus(status, {operation, 0, 0}));
}

VernonStatus encodeRecordedDebugCommand(VernonRuntimeCorePackets *packets, const RecordedDebugCommand &command) {
    const VernonStringView label{command.label.data(), command.label.size()};
    switch (command.kind) {
    case VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP:
        return vernonRuntimeCorePacketsPushDebugGroup(packets, label);
    case VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP:
        return vernonRuntimeCorePacketsPopDebugGroup(packets);
    default:
        return vernonRuntimeCorePacketsInsertDebugMarker(packets, label);
    }
}

VernonRhiStatus encodeManagedGraphicsMember(CommandStream &commands, GraphicsStream &stream,
                                            ManagedGraphicsCommandBatch &batch, size_t memberIndex, size_t firstDraw) {
    const ManagedGraphicsBatchMember &member = batch.members[memberIndex];
    std::vector<StoredGraphicsDraw> *stored =
        batch.storable && batch.scopes ? &batch.built->members[memberIndex].draws : nullptr;
    size_t command = 0;
    for (size_t draw = firstDraw;; ++draw) {
        for (; command < member.debugCommands.size() && member.debugCommands[command].draw <= draw - firstDraw;
             ++command)
            if (const VernonRhiStatus status = recordedCommandStatus(
                    *batch.context, encodeRecordedDebugCommand(commands.packets(), member.debugCommands[command]),
                    "managed graphics debug command failed");
                status != VERNON_RHI_STATUS_OK)
                return status;
        if (draw == member.drawEnd)
            return VERNON_RHI_STATUS_OK;
        if (const VernonRhiStatus status =
                encodeManagedGraphicsDraw(commands, stream, batch, batch.draws[draw], memberIndex,
                                          stored && !stored->empty() ? &(*stored)[draw - firstDraw] : nullptr);
            status != VERNON_RHI_STATUS_OK)
            return status;
    }
}

VernonRhiStatus encodeManagedGraphicsBatch(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &batch = *static_cast<ManagedGraphicsCommandBatch *>(opaque);
    auto acquired = batch.stream->acquire(encoder);
    if (acquired.isErr()) {
        if (invocationDiagnostic(*batch.context).empty())
            invocationDiagnostic(*batch.context) = "failed to acquire the managed graphics command stream";
        return vernon::toVernonRhiStatus(std::move(acquired).error());
    }
    CommandStream &commands = *acquired.value();
    if (batch.replay) {
        const std::lock_guard lock(batch.replay->replays);
        if (auto flushed = commands.flush(); flushed.isErr())
            return vernon::toVernonRhiStatus(std::move(flushed).error());
        if (auto updated = updateStoredGroups(commands, batch); updated.isErr())
            return vernon::toVernonRhiStatus(std::move(updated).error());
        std::vector<uint32_t> patched;
        for (const GraphicsMemberInputs &member : batch.inputs)
            for (const auto &[word, payload] : member.payload) {
                if (patched.empty())
                    patched = batch.replay->batch.payload();
                std::memcpy(patched.data() + word, &payload, sizeof(payload));
            }
        auto encoded = commands.encode(batch.replay->batch, patched.empty() ? batch.replay->batch.payload() : patched);
        return encoded.isOk() ? VERNON_RHI_STATUS_OK : vernon::toVernonRhiStatus(std::move(encoded).error());
    }
    // A stored batch's payload starts with its own draws, where they are patched on replay.
    if (auto flushed = commands.flush(); flushed.isErr())
        return vernon::toVernonRhiStatus(std::move(flushed).error());
    GraphicsStream stream;
    if (auto begun = commands.beginRendering(graphicsRenderingDescriptor(batch.scope)); begun.isErr())
        return vernon::toVernonRhiStatus(std::move(begun).error());
    VernonRhiStatus status = VERNON_RHI_STATUS_OK;
    // Members get their own groups only inside a fused scope; an unfused scope's group already carries the label.
    const bool memberGroups = batch.members.size() > 1;
    size_t firstDraw = 0;
    for (size_t memberIndex = 0; memberIndex < batch.members.size(); ++memberIndex) {
        const ManagedGraphicsBatchMember &member = batch.members[memberIndex];
        if (memberGroups)
            status = recordedCommandStatus(
                *batch.context,
                vernonRuntimeCorePacketsPushDebugGroup(commands.packets(), {member.label.data(), member.label.size()}),
                "managed graphics debug group failed");
        if (status == VERNON_RHI_STATUS_OK)
            status = encodeManagedGraphicsMember(commands, stream, batch, memberIndex, firstDraw);
        if (memberGroups && status == VERNON_RHI_STATUS_OK)
            status = recordedCommandStatus(*batch.context, vernonRuntimeCorePacketsPopDebugGroup(commands.packets()),
                                           "managed graphics debug group failed");
        if (status != VERNON_RHI_STATUS_OK)
            return status;
        firstDraw = member.drawEnd;
    }
    if (auto ended = commands.endRendering(); ended.isErr())
        return vernon::toVernonRhiStatus(std::move(ended).error());
    if (!batch.storable || !batch.scopes) {
        auto flushed = commands.flush();
        return flushed.isOk() ? VERNON_RHI_STATUS_OK : vernon::toVernonRhiStatus(std::move(flushed).error());
    }
    if (auto flushed = commands.flush(batch.built->batch); flushed.isErr())
        return vernon::toVernonRhiStatus(std::move(flushed).error());
    batch.scopes->publish(batch.opener, batch.built,
                          [](const StoredGraphicsScope &left, const StoredGraphicsScope &right) {
                              return left.keys == right.keys || !left.alive();
                          });
    return VERNON_RHI_STATUS_OK;
}

bool frameIsStorable(const vernon::runtime::program_execution::MaterializedNodeFrame &frame) {
    return frame.copiesBefore.empty() && frame.copiesAfter.empty() && frame.deviceCopiesBefore.empty() &&
           frame.deviceCopiesAfter.empty() && frame.deviceEndpointCarriers.empty();
}

void addReferenceKey(std::vector<uint64_t> &words, const VernonRuntimeProviderResourceReference &reference) {
    words.insert(words.end(), {reference.identity, reference.resource.value, reference.offset, reference.size});
}

void addBytesKey(std::vector<uint64_t> &words, const void *data, size_t size) {
    words.push_back(size);
    const size_t first = words.size();
    words.resize(first + (size + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    if (size)
        std::memcpy(words.data() + first, data, size);
}

// False for an argument whose commands cannot be stored. Host bytes are keyed only with `hostBytes`.
bool addArgumentKey(std::vector<uint64_t> &words, const VernonProgramArgument &argument, bool hostBytes) {
    words.push_back((static_cast<uint64_t>(argument.kind) << 32) | argument.slot);
    switch (argument.kind) {
    case VERNON_PROGRAM_IMAGE:
        addReferenceKey(words, argument.image.view);
        return true;
    case VERNON_PROGRAM_SAMPLER:
        addReferenceKey(words, argument.resource);
        return true;
    case VERNON_PROGRAM_TENSOR:
        break;
    default:
        return false;
    }
    const VernonTensorView &tensor = argument.tensor;
    const VernonValueLayoutView &layout = tensor.element_layout;
    words.insert(words.end(), {static_cast<uint64_t>(tensor.storage), static_cast<uint64_t>(tensor.access), tensor.rank,
                               tensor.byte_offset, tensor.byte_size, layout.byte_size, layout.alignment});
    addBytesKey(words, layout.layout_hash.data, layout.layout_hash.size);
    if (tensor.rank && (!tensor.shape || !tensor.byte_strides))
        return false;
    for (uint32_t axis = 0; axis < tensor.rank; ++axis)
        words.insert(words.end(), {tensor.shape[axis], static_cast<uint64_t>(tensor.byte_strides[axis])});
    if (tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
        addReferenceKey(words, tensor.resource);
        return true;
    }
    if (tensor.storage != VERNON_TENSOR_HOST || (tensor.byte_size && !tensor.host_data))
        return false;
    if (hostBytes)
        addBytesKey(words, static_cast<const uint8_t *>(tensor.host_data) + tensor.byte_offset, tensor.byte_size);
    return true;
}

// Empty when the commands of the Node cannot be stored. Arguments are keyed by their structure and resources. Their
// host bytes only reach group values, so with `updatable` groups a replay writes them and they are not keyed; without,
// an argument that a Program instance binding owns is keyed by the binding, which never changes, and any other by its
// host bytes as well.
std::optional<GraphicsNodeKey> graphicsNodeKey(uint64_t node, const GraphicsMemberInputs &member, bool updatable) {
    const vernon::runtime::program_execution::MaterializedNodeFrame &frame = *member.frame;
    if (!frameIsStorable(frame))
        return std::nullopt;
    GraphicsNodeKey key;
    key.pass = member.recorded ? member.recorded->pass.get() : nullptr;
    if (!key.pass) {
        key.draw = member.draw.get();
        key.dynamic = member.dynamic.get();
    }
    key.renderPass = member.boundRenderPass.get();
    key.words.push_back(node);
    for (const VernonColorAttachment &color : member.renderPass->colors)
        addReferenceKey(key.words, color.view);
    if (member.renderPass->depth)
        addReferenceKey(key.words, member.renderPass->depth->view);
    for (size_t parameter = 0; parameter < frame.arguments.size(); ++parameter) {
        const ArgumentOwner &owner = member.owners[parameter];
        if (!updatable && owner.groups) {
            key.words.insert(key.words.end(), {1, reinterpret_cast<uintptr_t>(owner.identity)});
            continue;
        }
        key.words.push_back(2);
        if (!addArgumentKey(key.words, frame.arguments[parameter], !updatable))
            return std::nullopt;
    }
    return key;
}
// The owner of each argument of `frame`, the frame of the Node that `node` keys: the Program instance binding that the
// argument borrows unchanged, whose groups the instance keeps for the Node, otherwise the frame itself.
std::vector<ArgumentOwner> frameOwners(const ProgramInvocationContext *invocation,
                                       const std::vector<uint32_t> &valueSlots, uint64_t node,
                                       const program::ResolvedNodePlan &plan,
                                       const vernon::runtime::program_execution::MaterializedNodeFrame &frame) {
    std::vector<ArgumentOwner> owners(frame.arguments.size(), ArgumentOwner{&frame, nullptr, nullptr});
    if (!invocation || !invocation->groups)
        return owners;
    GroupStore *groups = nullptr;
    for (size_t parameter = 0; parameter < owners.size(); ++parameter) {
        const uint32_t value = plan.projections[parameter].value;
        if (!frame.borrowed[parameter] || value >= valueSlots.size() || valueSlots[value] == UINT32_MAX)
            continue;
        const std::shared_ptr<void> *payload = invocation->bindings.find(valueSlots[value]);
        if (!payload)
            continue;
        if (!groups)
            groups = &invocation->groups->node(node);
        owners[parameter] = {payload->get(), groups, payload};
    }
    return owners;
}

// Plans the one draw of a Node that has no recording: every argument comes from the Node's own frame.
bool planUnrecordedGraphicsNode(VernonRuntimeContext &context, VernonStageExecutable &stage,
                                const program::GraphicsOperation &graphics,
                                const vernon::runtime::program_execution::MaterializedNodeFrame &frame,
                                const std::vector<ArgumentOwner> &owners, const RuntimeProgramControl &renderPass,
                                const std::shared_ptr<const RuntimeProgramControl> &drawControl,
                                const std::shared_ptr<const RuntimeProgramControl> &dynamic, ManagedGraphicsNode &node,
                                std::string &error) {
    for (uint32_t parameter = 0; parameter < stage.bindingProjection.parameters.size(); ++parameter) {
        const VernonProgramArgument &argument = frame.arguments[parameter];
        PlannedGraphicsArgument planned;
        if (!planGraphicsArgument(stage.bindingProjection, stage.drawLayout, parameter, argument, describeRuntimeImage,
                                  &context, planned, error))
            return false;
        addSampledRead(stage.drawLayout, parameter, argument, planned, node.reads);
        node.arguments.push_back(&argument);
        node.owners.push_back(owners[parameter]);
    }
    if (!addVertexReads(stage.drawLayout, node.arguments.data(), node.bufferReads))
        return error = "contains an invalid vertex resource", false;
    auto state = std::make_shared<ProgramGraphicsState>();
    state->describe(graphics);
    const VernonDrawCommand defaultDraw{sizeof(VernonDrawCommand), nullptr, static_cast<uint32_t>(graphics.vertexCount),
                                        static_cast<uint32_t>(graphics.instanceCount)};
    ManagedGraphicsDraw &draw = node.draws.emplace_back();
    draw.stage = &stage;
    draw.graphicsState = &state->state;
    draw.targets = node.targets.get();
    if (!planGraphicsDraw(stage.drawLayout, node.arguments.data(), *node.targets, renderPass.renderPass,
                          dynamic ? &dynamic->dynamic : nullptr, drawControl ? drawControl->draw : defaultDraw,
                          draw.draw, error))
        return false;
    if (draw.draw.indexBinding &&
        !node.bufferReads.add(draw.draw.indexBinding->resource, draw.draw.indexBinding->offset,
                              static_cast<uint64_t>(draw.draw.indexBinding->index_count) * sizeof(uint32_t), true))
        return error = "contains an invalid index resource", false;
    node.retained.push_back(std::move(state));
    node.retained.push_back(drawControl);
    return true;
}

// Plans the recorded draws of a Node. A draw whose argument sources equal the previous draw's shares its arguments.
// A draw's index read is its whole index view, so that the reads of a Node depend only on its recording's structure.
bool planRecordedGraphicsNode(VernonRuntimeContext &context, const RecordedGraphicsPass &pass,
                              const std::shared_ptr<const RecordedDrawPayload> &payload,
                              const vernon::runtime::program_execution::MaterializedNodeFrame &frame,
                              const std::vector<ArgumentOwner> &owners, const std::vector<InvocationSetFrame> &sets,
                              const RuntimeProgramControl &renderPass, ManagedGraphicsNode &node, std::string &error) {
    if (payload->draws.size() != pass.draws.size() || payload->dynamics.size() != pass.draws.size())
        return error = "recording payload does not match its draws", false;
    std::shared_ptr<const GraphicsVertexRecords> records;
    // Per pipeline and Stage parameter: whether the pass frame's argument was validated.
    std::unordered_map<const RuntimeGraphicsPipelineState *, std::vector<char>> passChecked;
    std::vector<char> setRead(sets.size());
    const std::vector<RecordedArgumentSource> *sources = nullptr;
    size_t firstArgument = 0;
    for (size_t index = 0; index < pass.draws.size(); ++index) {
        const RecordedDraw &recorded = pass.draws[index];
        const RuntimeGraphicsPipelineState &pipeline = *recorded.pipeline;
        VernonStageExecutable &stage = *pipeline.sourcePlan->stage;
        if (recorded.arguments.get() != sources) {
            sources = recorded.arguments.get();
            firstArgument = node.arguments.size();
            std::vector<char> &checked = passChecked[&pipeline];
            checked.resize(sources->size());
            for (uint32_t parameter = 0; parameter < sources->size(); ++parameter) {
                const RecordedArgumentSource &source = (*sources)[parameter];
                if (source.fromSet) {
                    const InvocationSetFrame &set = sets[source.frame];
                    if (!std::exchange(setRead[source.frame], 1))
                        for (uint32_t image : stage.drawLayout.sampledImages)
                            if (pass.frames[source.frame].match->endpoints[image])
                                addGraphicsScopeRead(node.reads, set.lowered->frame.arguments[image].image.view,
                                                     set.lowered->planned[image].subresources);
                    node.arguments.push_back(&set.lowered->frame.arguments[source.parameter]);
                    node.owners.push_back(set.owner);
                    continue;
                }
                const VernonProgramArgument &argument = frame.arguments[source.parameter];
                if (!std::exchange(checked[parameter], 1)) {
                    PlannedGraphicsArgument planned;
                    if (!planGraphicsArgument(stage.bindingProjection, stage.drawLayout, parameter, argument,
                                              describeRuntimeImage, &context, planned, error))
                        return false;
                    addSampledRead(stage.drawLayout, parameter, argument, planned, node.reads);
                }
                node.arguments.push_back(&argument);
                node.owners.push_back(owners[source.parameter]);
            }
            if (!addVertexReads(stage.drawLayout, node.arguments.data() + firstArgument, node.bufferReads))
                return error = "contains an invalid vertex resource", false;
            if (!checkGraphicsSharedDescriptors(stage.drawLayout, node.arguments.data() + firstArgument, error))
                return false;
            records = std::make_shared<const GraphicsVertexRecords>(
                graphicsVertexRecords(stage.drawLayout, node.arguments.data() + firstArgument));
        }
        ManagedGraphicsDraw &draw = node.draws.emplace_back();
        draw.stage = &stage;
        draw.graphicsState = &pipeline.graphicsState.state;
        draw.targets = node.targets.get();
        draw.firstArgument = firstArgument;
        draw.records = records;
        if (!planGraphicsDrawValues(*records, *node.targets, renderPass.renderPass, &payload->dynamics[index]->dynamic,
                                    payload->draws[index]->draw, draw.draw, error))
            return false;
        if (draw.draw.indexBinding &&
            !node.bufferReads.add(draw.draw.indexBinding->resource, 0, draw.draw.indexBinding->resource.size, true))
            return error = "contains an invalid index resource", false;
    }
    node.payload = payload;
    node.retained.push_back(payload);
    return true;
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
            const auto begin = std::chrono::steady_clock::now();
            const VernonStatus status = executeCpuProgramNode(context, canonicalProgram, node, *resolvedNode->stage,
                                                              arena, materialized, grid, step);
            if (status != VERNON_STATUS_OK)
                return status;
            submission.recordHostScope(node.id, begin, std::chrono::steady_clock::now());
        }
        if (vernon::runtime::program_execution::injectFailure(
                vernon::runtime::program_execution::FailureBoundary::Completion))
            return fail(&context, "injected CPU Program wait failure", VERNON_STATUS_INTERNAL_ERROR);
        step = vernon::runtime::program_execution::SubmissionState::Completed;
        return VERNON_STATUS_OK;
    }
    {
        vernon::execution::detail::ValidatedRhiCommandPlan commandPlan;
        vernon::runtime::program_execution::ResolvedTransferExecutor transfers(context, arena);
        const std::optional<program::GraphDirection> graphDirection = program::graphDirection(graph.direction);
        std::string transferError;
        if (!graphDirection || !transfers.prepareGraph(*graphDirection, transferError))
            return fail(&context, std::move(transferError));
        std::shared_ptr<ManagedGraphicsCommandBatch> graphicsBatch;
        GraphicsScopeMaterializer graphicsScopeMaterializer;
        vernon::execution::detail::RhiCommandExecutionPlan graphicsNodePlan;
        GraphicsBufferReads graphicsBufferReads;
        // The submitted plan's producers share one command stream on its encoder.
        std::shared_ptr<EncoderCommandStream> commandStream;
        const auto sharedCommandStream = [&]() -> EncoderCommandStream & {
            if (!commandStream)
                commandStream = std::make_shared<EncoderCommandStream>(context);
            return *commandStream;
        };
        const auto retainCommandStream = [&](vernon::execution::detail::RhiCommandExecutionPlan &fragment) {
            fragment.finishers.push_back({EncoderCommandStream::finish, commandStream.get()});
            fragment.retainedContexts.push_back(commandStream);
        };
        // Lowers the binding sets a recorded Node draws with. Without `copies`, the sets must need no copies: they were
        // lowered once and kept by a stored scope that planned no transfers for them.
        const auto lowerSets = [&](const GraphicsMemberInputs &member, std::vector<InvocationSetFrame> &sets,
                                   std::vector<vernon::runtime::program_execution::DeviceBufferCopy> *copies) {
            const RecordedGraphicsPass *pass = member.recorded ? member.recorded->pass.get() : nullptr;
            if (!pass)
                return VERNON_STATUS_OK;
            sets.resize(pass->frames.size());
            std::string error;
            for (size_t index = 0; index < sets.size(); ++index) {
                if (!lowerSetFrame(context, pass->frames[index], sets[index], error))
                    return fail(&context, programNodeName(*member.node) + ": binding set: " + error);
                const auto &setCopies = sets[index].lowered->frame.deviceCopiesBefore;
                if (copies)
                    copies->insert(copies->end(), setCopies.begin(), setCopies.end());
                else if (!setCopies.empty())
                    return fail(&context, programNodeName(*member.node) + ": binding set needs unplanned copies",
                                VERNON_STATUS_INTERNAL_ERROR);
            }
            return VERNON_STATUS_OK;
        };
        const auto planMember = [&](const GraphicsMemberInputs &member, std::vector<InvocationSetFrame> &sets,
                                    ManagedGraphicsNode &graphicsNode) {
            const program::Node &node = *member.node;
            const program::GraphicsOperation &graphics = program::graphicsOperation(node);
            const RecordedGraphicsPass *pass = member.recorded ? member.recorded->pass.get() : nullptr;
            std::string error;
            if (!planRenderTargets(&member.renderPass->renderPass, describeRuntimeImage, &context,
                                   *graphicsNode.targets, error) ||
                !checkProgramGraphicsAttachmentSignature(graphics, *graphicsNode.targets, error))
                return fail(&context, programNodeName(node) + ": " + error);
            const bool planned =
                pass ? planRecordedGraphicsNode(context, *pass, member.recorded->payload, *member.frame, member.owners,
                                                sets, *member.renderPass, graphicsNode, error)
                     : planUnrecordedGraphicsNode(context, *member.resolved->stage, graphics, *member.frame,
                                                  member.owners, *member.renderPass, member.draw, member.dynamic,
                                                  graphicsNode, error);
            if (!planned)
                return fail(&context, programNodeName(node) + ": " + error);
            if (graphicsScopeReadsOverlap(graphicsNode.reads, *graphicsNode.targets))
                return fail(&context, programNodeName(node) + ": samples an image it renders to");
            graphicsNode.storable = frameIsStorable(*member.frame);
            for (InvocationSetFrame &set : sets) {
                graphicsNode.storable = graphicsNode.storable && set.owner.groups;
                graphicsNode.retained.push_back(std::move(set.lowered));
            }
            graphicsNode.renderPass = member.renderPass;
            graphicsNode.retained.insert(graphicsNode.retained.end(), {member.frame, member.renderPass});
            for (const ArgumentOwner &owner : member.owners)
                if (owner.groups && owner.lifetime)
                    graphicsNode.identities.emplace_back(*owner.lifetime);
            for (const std::shared_ptr<const void> &identity :
                 {std::shared_ptr<const void>(member.boundRenderPass), std::shared_ptr<const void>(member.draw),
                  std::shared_ptr<const void>(member.dynamic),
                  member.recorded ? std::shared_ptr<const void>(member.recorded->pass) : nullptr})
                if (identity)
                    graphicsNode.identities.emplace_back(identity);
            return VERNON_STATUS_OK;
        };
        const auto appendPlanned = [&](const GraphicsMemberInputs &member, ManagedGraphicsNode &&graphicsNode) {
            graphicsBufferReads.merge(graphicsNode.bufferReads);
            const RecordedGraphicsPass *pass = member.recorded ? member.recorded->pass.get() : nullptr;
            appendManagedGraphicsNode(*graphicsBatch, std::move(graphicsNode), member.key);
            graphicsBatch->members.push_back({programNodeLabel(*member.node), graphicsBatch->draws.size(),
                                              pass ? pass->debugCommands : std::vector<RecordedDebugCommand>{},
                                              member.frame.get(), member.key.has_value()});
        };
        const auto planMembers = [&](std::vector<GraphicsMemberInputs> &&members) {
            for (GraphicsMemberInputs &member : members) {
                std::vector<InvocationSetFrame> sets;
                if (const VernonStatus status = lowerSets(member, sets, nullptr); status != VERNON_STATUS_OK)
                    return status;
                ManagedGraphicsNode graphicsNode;
                if (const VernonStatus status = planMember(member, sets, graphicsNode); status != VERNON_STATUS_OK)
                    return status;
                appendPlanned(member, std::move(graphicsNode));
            }
            return VERNON_STATUS_OK;
        };
        // A replayed scope whose Nodes stop matching its stored members plans the Nodes it already matched.
        const auto stopReplay = [&]() {
            graphicsBatch->replay.reset();
            return planMembers(std::exchange(graphicsBatch->inputs, {}));
        };
        const auto appendGraphicsBatch = [&]() {
            if (!graphicsBatch)
                return VERNON_STATUS_OK;
            if (graphicsBatch->replay && graphicsBatch->inputs.size() != graphicsBatch->replay->keys.size())
                if (const VernonStatus status = stopReplay(); status != VERNON_STATUS_OK)
                    return status;
            graphicsBufferReads.append(graphicsNodePlan.commands.nodes.back(), graphicsNodePlan.bindings);
            graphicsBufferReads = {};
            std::string compositionError;
            if (!commandPlan.append(std::move(graphicsNodePlan), compositionError))
                return fail(&context, std::move(compositionError));
            graphicsNodePlan = {};
            graphicsBatch.reset();
            return VERNON_STATUS_OK;
        };
        // A Node's transfers are planned apart so that a graphics Node can end the open render batch before them.
        const auto appendNodeTransfers =
            [&](std::vector<vernon::execution::detail::RhiCommandExecutionPlan> &&nodeTransfers,
                const std::shared_ptr<const CommandScope> &scope) {
                for (vernon::execution::detail::RhiCommandExecutionPlan &fragment : nodeTransfers) {
                    setScope(fragment.encoders, scope);
                    std::string compositionError;
                    if (!commandPlan.append(std::move(fragment), compositionError))
                        return fail(&context, std::move(compositionError));
                }
                return VERNON_STATUS_OK;
            };
        const auto flushCommands = [&]() {
            if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                return pending;
            if (commandPlan.empty())
                return VERNON_STATUS_OK;
            graphicsScopeMaterializer.reset();
            commandStream.reset();
            return submission.execute(std::exchange(commandPlan, {}), step);
        };
        // Per Value, the boundary slot whose instance binding supplies it.
        StoredGraphicsScopes *storedScopes = invocationContext ? invocationContext->storedGraphics : nullptr;
        const bool updatesGroups = storedScopes && backendUpdatesGroups(context);
        std::vector<uint32_t> valueSlots;
        if (invocationContext && invocationContext->groups) {
            valueSlots.assign(canonicalProgram.values.size(), UINT32_MAX);
            for (const program::BoundarySlot &slot : canonicalProgram.abi.boundarySlots)
                if (slot.value < valueSlots.size())
                    valueSlots[slot.value] = slot.id;
        }
        std::vector<vernon::runtime::program_execution::MaterializedNodeFrame> materializedNodes;
        materializedNodes.reserve(graph.nodes.size());
        for (const program::Node &node : graph.nodes) {
            const std::optional<program::GraphDirection> direction = program::graphDirection(graph.direction);
            const program::ResolvedNodePlan *resolvedNode = direction ? execution.node(*direction, node.id) : nullptr;
            if (!resolvedNode)
                return fail(&context, "pipeline node has no resolved kernel stage");
            const uint64_t nodeKey = (static_cast<uint64_t>(*direction) << 32) | node.id;
            if (program::executionKind(node) == program::ExecutionKind::Graphics) {
                const program::GraphicsOperation &graphics = program::graphicsOperation(node);
                auto renderPass = control(graphics.renderPassControl, RuntimeProgramControl::RenderPass);
                if (!renderPass)
                    return fail(&context, "managed graphics node has no bound RenderPass control");
                GraphicsMemberInputs member;
                member.node = &node;
                member.resolved = resolvedNode;
                member.boundRenderPass = renderPass;
                member.draw = control(graphics.drawCommandControl, RuntimeProgramControl::DrawCommand);
                member.dynamic = control(graphics.dynamicStateControl, RuntimeProgramControl::DynamicState);
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
                member.renderPass = renderPass;
                if (invocationContext && invocationContext->recordedGraphics) {
                    auto &recordedNodes = invocationContext->recordedGraphics->nodes;
                    const auto found = recordedNodes.find(node.id);
                    member.recorded = found == recordedNodes.end() ? nullptr : &found->second;
                }
                member.frame = std::make_shared<vernon::runtime::program_execution::MaterializedNodeFrame>();
                std::string graphicsError;
                if (!vernon::runtime::program_execution::materializeNodeFrame(arena, canonicalProgram, node,
                                                                              *resolvedNode, resolvePhysicalEndpoint,
                                                                              *member.frame, graphicsError))
                    return fail(&context, std::move(graphicsError));
                member.owners = frameOwners(invocationContext, valueSlots, nodeKey, *resolvedNode, *member.frame);
                if (storedScopes)
                    member.key = graphicsNodeKey(nodeKey, member, updatesGroups);
                // The groups of a keyed Node belong to its scope, which finds each by the arguments it holds.
                if (member.key && updatesGroups)
                    for (size_t parameter = 0; parameter < member.owners.size(); ++parameter)
                        member.owners[parameter] = {&member.frame->arguments[parameter], nullptr, nullptr};
                if (member.recorded)
                    member.recorded->telemetry.render_scopes = 1;

                // A stored Node with an equal key renders to the same targets and reads the same resources, so the
                // Node is planned only once it is known that no stored scope replays it.
                const StoredGraphicsMember *stored = nullptr;
                bool continues = false;
                std::shared_ptr<const StoredGraphicsScope> opening;
                std::vector<std::pair<size_t, VernonRuntimeProviderDrawPayload>> openingPayload;
                if (member.key) {
                    if (graphicsBatch && graphicsBatch->replay) {
                        const size_t next = graphicsBatch->inputs.size();
                        continues = next < graphicsBatch->replay->keys.size() &&
                                    graphicsBatch->replay->keys[next] == *member.key &&
                                    replaysDrawValues(graphicsBatch->replay->members[next], member, member.payload);
                        if (continues)
                            stored = &graphicsBatch->replay->members[next];
                    }
                    for (const std::shared_ptr<const StoredGraphicsScope> &scope : storedScopes->find(nodeKey))
                        if (scope->keys.front() == *member.key && scope->alive() &&
                            replaysDrawValues(scope->members.front(), member, openingPayload)) {
                            opening = scope;
                            if (!stored)
                                stored = &scope->members.front();
                            break;
                        }
                }
                std::vector<vernon::runtime::program_execution::DeviceBufferCopy> copiesBefore =
                    member.frame->deviceCopiesBefore;
                std::vector<InvocationSetFrame> sets;
                if (!stored)
                    if (const VernonStatus status = lowerSets(member, sets, &copiesBefore); status != VERNON_STATUS_OK)
                        return status;
                std::vector<vernon::execution::detail::RhiCommandExecutionPlan> nodeTransfers;
                if (const VernonStatus status =
                        transfers.planBeforeConsumer({*direction, node.id}, copiesBefore, nodeTransfers);
                    status != VERNON_STATUS_OK)
                    return status;
                const std::shared_ptr<CommandScope> scope = nodeScope(node);
                // Transfers run before this Node's draws, so they end the open scope and this Node opens its own.
                if (!nodeTransfers.empty()) {
                    if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                        return pending;
                    if (const VernonStatus appended = appendNodeTransfers(std::move(nodeTransfers), scope);
                        appended != VERNON_STATUS_OK)
                        return appended;
                }
                ManagedGraphicsNode graphicsNode;
                if (!stored)
                    if (const VernonStatus status = planMember(member, sets, graphicsNode); status != VERNON_STATUS_OK)
                        return status;
                const auto candidate =
                    std::find_if(execution.graphicsScopeCandidates.begin(), execution.graphicsScopeCandidates.end(),
                                 [&](const program::ResolvedGraphicsScopeCandidate &entry) {
                                     return entry.node == program::NodeKey{*direction, node.id};
                                 });
                if (candidate == execution.graphicsScopeCandidates.end())
                    return fail(&context, "managed graphics node has no resolved scope candidate");
                // Only a node following an open batch's scope can fuse into it.
                const GraphicsScopeMaterialization materialization = graphicsScopeMaterializer.materialize(
                    candidate->region, stored ? stored->targets : *graphicsNode.targets,
                    stored ? stored->reads : graphicsNode.reads, !graphicsBatch);
                const bool fuses = graphicsBatch && materialization == GraphicsScopeMaterialization::Fuse;
                if (!fuses) {
                    if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                        return pending;
                } else if (graphicsBatch->replay && !continues) {
                    if (const VernonStatus status = stopReplay(); status != VERNON_STATUS_OK)
                        return status;
                }
                if (!graphicsBatch) {
                    graphicsBatch = std::make_shared<ManagedGraphicsCommandBatch>();
                    graphicsBatch->context = &context;
                    graphicsBatch->stream = &sharedCommandStream();
                    graphicsBatch->scopes = storedScopes;
                    graphicsBatch->opener = nodeKey;
                    if (opening)
                        member.payload = std::move(openingPayload);
                    graphicsBatch->replay = std::move(opening);
                    graphicsBatch->built =
                        std::make_shared<StoredGraphicsScope>(updatesGroups ? VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE
                                                                            : VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT);
                    retainCommandStream(graphicsNodePlan);
                    graphicsBatch->commandScope = scope;
                    vernon::execution::detail::CommandNode drawCommand;
                    drawCommand.kind = vernon::execution::detail::CommandNodeKind::Derivative;
                    drawCommand.queue = vernon::execution::detail::CommandQueueClass::Graphics;
                    graphicsNodePlan.commands.nodes.push_back(std::move(drawCommand));
                    graphicsNodePlan.encoders.push_back(
                        {encodeManagedGraphicsBatch, graphicsBatch.get(), graphicsBatch->commandScope});
                    graphicsNodePlan.retainedContexts.push_back(graphicsBatch);
                } else {
                    graphicsBatch->commandScope->label += "+" + programNodeLabel(node);
                    ++graphicsBatch->commandScope->nodeCount;
                }
                graphicsBatch->scope = graphicsScopeMaterializer.scope();
                if (graphicsBatch->replay) {
                    graphicsBufferReads.merge(stored->bufferReads);
                    graphicsBatch->inputs.push_back(std::move(member));
                    continue;
                }
                if (stored) {
                    std::vector<GraphicsMemberInputs> members;
                    members.push_back(std::move(member));
                    if (const VernonStatus status = planMembers(std::move(members)); status != VERNON_STATUS_OK)
                        return status;
                    continue;
                }
                appendPlanned(member, std::move(graphicsNode));
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
            const std::shared_ptr<const CommandScope> scope = nodeScope(node);
            std::vector<vernon::execution::detail::RhiCommandExecutionPlan> nodeTransfers;
            if (const VernonStatus status = transfers.planBeforeConsumer(
                    {*direction, node.id}, materializedNodes.back().deviceCopiesBefore, nodeTransfers);
                status != VERNON_STATUS_OK)
                return status;
            if (const VernonStatus appended = appendNodeTransfers(std::move(nodeTransfers), scope);
                appended != VERNON_STATUS_OK)
                return appended;
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
                context, sharedCommandStream(), {}, {}, *resolvedNode->stage, materializedNodes.back().arguments,
                frameOwners(invocationContext, valueSlots, nodeKey, *resolvedNode, materializedNodes.back()), grid,
                materializedNodes.back().deviceCopiesAfter, vernon::execution::detail::CommandNodeKind::Derivative,
                nodePlan);
            if (planned != VERNON_STATUS_OK)
                return planned;
            retainCommandStream(nodePlan);
            setScope(nodePlan.encoders, scope);
            std::string compositionError;
            if (!commandPlan.append(std::move(nodePlan), compositionError))
                return fail(&context, std::move(compositionError));
        }
        return flushCommands();
    }
    return VERNON_STATUS_INTERNAL_ERROR;
}

VernonStatus encodeStageInvocation(VernonStageExecutable &pipeline, CommandStream &commands,
                                   const VernonStageInvocationDescriptor &invocation, const ArgumentOwner *owners) {
    if (pipeline.bindingProjection.compute.empty())
        return fail(pipeline.context, "only a compute Stage is invoked alone");
    const uint32_t grid[3]{invocation.compute_grid.x, invocation.compute_grid.y, invocation.compute_grid.z};
    const uint32_t workgroup[3]{pipeline.workgroupSize.x, pipeline.workgroupSize.y, pipeline.workgroupSize.z};
    if (!validateDispatchContract(pipeline.dispatchContract, grid, workgroup, invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    PlannedComputeLaunch plan;
    std::string planningError;
    if (!planComputeInvocation(pipeline.bindingProjection, invocation, plan, planningError))
        return fail(pipeline.context, planningError);
    auto invoked = invokeBackendComputePipeline(pipeline, commands, plan, owners);
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

VernonStatus vernon::runtime::encodeResolvedStage(CommandStream &stream, VernonStageExecutable *pipeline,
                                                  const VernonStageInvocationDescriptor *invocation,
                                                  const ArgumentOwner *owners) {
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
    encoded.command_encoder = stream.encoder();
    return encodeStageInvocation(*pipeline, stream, encoded, owners);
}

VernonStageExecutable::~VernonStageExecutable() {
    if (backendState)
        vernon::runtime::destroyBackendPipeline(*this);
}
