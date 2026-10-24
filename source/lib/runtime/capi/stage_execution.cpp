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
};

struct ManagedGraphicsBatchMember {
    std::string label;
    // End of the member's draws in `ManagedGraphicsCommandBatch::draws`.
    size_t drawEnd{};
    // Positions count from the member's first draw.
    std::vector<RecordedDebugCommand> debugCommands;
};

// Recorded draws point into the invocation's recordings, which the submission keeps until the batch is encoded.
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
};

const void *resolutionIdentity(ManagedGraphicsCommandBatch &batch, const std::array<float, 2> &resolution) {
    for (const std::array<float, 2> &known : batch.resolutions)
        if (known == resolution)
            return &known;
    return &batch.resolutions.emplace_back(resolution);
}

void appendManagedGraphicsNode(ManagedGraphicsCommandBatch &batch, ManagedGraphicsNode &&node) {
    const size_t offset = batch.arguments.size();
    batch.arguments.insert(batch.arguments.end(), node.arguments.begin(), node.arguments.end());
    batch.owners.insert(batch.owners.end(), node.owners.begin(), node.owners.end());
    batch.draws.reserve(batch.draws.size() + node.draws.size());
    for (ManagedGraphicsDraw &draw : node.draws) {
        draw.firstArgument += offset;
        draw.command = resolutionIdentity(batch, draw.draw.resolution);
        batch.draws.push_back(draw);
    }
    batch.retained.push_back(std::move(node.targets));
    batch.retained.insert(batch.retained.end(), std::make_move_iterator(node.retained.begin()),
                          std::make_move_iterator(node.retained.end()));
}

VernonRhiStatus encodeManagedGraphicsDraw(CommandStream &commands, GraphicsStream &stream,
                                          const ManagedGraphicsCommandBatch &batch, const ManagedGraphicsDraw &draw) {
    const GraphicsCommand command{batch.arguments.data() + draw.firstArgument,
                                  draw.targets,
                                  draw.graphicsState,
                                  draw.draw,
                                  {batch.owners.data() + draw.firstArgument, draw.command, &commands.transient(), {}},
                                  commands.encoder()};
    auto invoked = invokeBackendPipeline(*draw.stage, commands.commands(), command, stream);
    if (invoked.isOk())
        return VERNON_RHI_STATUS_OK;
    const vernon::RuntimeError error = std::move(invoked).error();
    if (invocationDiagnostic(*draw.stage->context).empty())
        invocationDiagnostic(*draw.stage->context) =
            error.context.operation ? error.context.operation : "managed graphics draw failed";
    return vernon::toVernonRhiStatus(error);
}

VernonRhiStatus encodeRecordedDebugCommand(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                           const RecordedDebugCommand &command) {
    const VernonStringView label{command.label.data(), command.label.size()};
    switch (command.kind) {
    case VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP:
        return vernonRhiCommandEncoderPushDebugGroup(device, encoder, label);
    case VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP:
        return vernonRhiCommandEncoderPopDebugGroup(device, encoder);
    default:
        return vernonRhiCommandEncoderInsertDebugMarker(device, encoder, label);
    }
}

VernonRhiStatus encodeManagedGraphicsMember(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                            CommandStream &commands, GraphicsStream &stream,
                                            const ManagedGraphicsCommandBatch &batch,
                                            const ManagedGraphicsBatchMember &member, size_t firstDraw) {
    size_t command = 0;
    for (size_t draw = firstDraw;; ++draw) {
        for (; command < member.debugCommands.size() && member.debugCommands[command].draw <= draw - firstDraw;
             ++command)
            if (const VernonRhiStatus status =
                    encodeRecordedDebugCommand(device, encoder, member.debugCommands[command]);
                status != VERNON_RHI_STATUS_OK)
                return status;
        if (draw == member.drawEnd)
            return VERNON_RHI_STATUS_OK;
        if (const VernonRhiStatus status = encodeManagedGraphicsDraw(commands, stream, batch, batch.draws[draw]);
            status != VERNON_RHI_STATUS_OK)
            return status;
    }
}

VernonRhiStatus encodeManagedGraphicsBatch(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &batch = *static_cast<ManagedGraphicsCommandBatch *>(opaque);
    const VernonRhiDevice device = batch.context->rhiDevice;
    auto acquired = batch.stream->acquire(encoder);
    if (acquired.isErr()) {
        if (invocationDiagnostic(*batch.context).empty())
            invocationDiagnostic(*batch.context) = "failed to acquire the managed graphics command stream";
        return vernon::toVernonRhiStatus(std::move(acquired).error());
    }
    CommandStream &commands = *acquired.value();
    GraphicsStream stream;
    if (auto begun =
            beginBackendRendering(*batch.context, commands.commands(), graphicsRenderingDescriptor(batch.scope));
        begun.isErr())
        return vernon::toVernonRhiStatus(std::move(begun).error());
    VernonRhiStatus status = VERNON_RHI_STATUS_OK;
    // Members get their own groups only inside a fused scope; an unfused scope's group already carries the label.
    const bool memberGroups = batch.members.size() > 1;
    size_t firstDraw = 0;
    for (const ManagedGraphicsBatchMember &member : batch.members) {
        if (memberGroups)
            status = vernonRhiCommandEncoderPushDebugGroup(device, encoder, {member.label.data(), member.label.size()});
        if (status == VERNON_RHI_STATUS_OK)
            status = encodeManagedGraphicsMember(device, encoder, commands, stream, batch, member, firstDraw);
        if (memberGroups && status == VERNON_RHI_STATUS_OK)
            status = vernonRhiCommandEncoderPopDebugGroup(device, encoder);
        if (status != VERNON_RHI_STATUS_OK)
            break;
        firstDraw = member.drawEnd;
    }
    auto ended = endBackendRendering(*batch.context, commands.commands());
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    return ended.isOk() ? VERNON_RHI_STATUS_OK : vernon::toVernonRhiStatus(std::move(ended).error());
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
bool planRecordedGraphicsNode(VernonRuntimeContext &context, const RecordedGraphicsPass &pass,
                              const vernon::runtime::program_execution::MaterializedNodeFrame &frame,
                              const std::vector<ArgumentOwner> &owners, const std::vector<InvocationSetFrame> &sets,
                              const RuntimeProgramControl &renderPass, ManagedGraphicsNode &node, std::string &error) {
    // Per pipeline and Stage parameter: whether the pass frame's argument was validated.
    std::unordered_map<const RuntimeGraphicsPipelineState *, std::vector<char>> passChecked;
    std::vector<char> setRead(sets.size());
    const std::vector<RecordedArgumentSource> *sources = nullptr;
    size_t firstArgument = 0;
    for (const RecordedDraw &recorded : pass.draws) {
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
        }
        ManagedGraphicsDraw &draw = node.draws.emplace_back();
        draw.stage = &stage;
        draw.graphicsState = &pipeline.graphicsState.state;
        draw.targets = node.targets.get();
        draw.firstArgument = firstArgument;
        if (!planGraphicsDraw(stage.drawLayout, node.arguments.data() + firstArgument, *node.targets,
                              renderPass.renderPass, &recorded.dynamic->dynamic, recorded.draw->draw, draw.draw, error))
            return false;
        if (draw.draw.indexBinding &&
            !node.bufferReads.add(draw.draw.indexBinding->resource, draw.draw.indexBinding->offset,
                                  static_cast<uint64_t>(draw.draw.indexBinding->index_count) * sizeof(uint32_t), true))
            return error = "contains an invalid index resource", false;
    }
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
        const auto appendGraphicsBatch = [&]() {
            if (!graphicsBatch)
                return VERNON_STATUS_OK;
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
                auto frame = std::make_shared<vernon::runtime::program_execution::MaterializedNodeFrame>();
                std::string graphicsError;
                if (!vernon::runtime::program_execution::materializeNodeFrame(
                        arena, canonicalProgram, node, *resolvedNode, resolvePhysicalEndpoint, *frame, graphicsError))
                    return fail(&context, std::move(graphicsError));
                std::vector<vernon::runtime::program_execution::DeviceBufferCopy> copiesBefore =
                    frame->deviceCopiesBefore;
                std::vector<InvocationSetFrame> sets;
                if (recordedPass) {
                    sets.resize(recordedPass->frames.size());
                    for (size_t index = 0; index < sets.size(); ++index) {
                        if (!lowerSetFrame(context, recordedPass->frames[index], sets[index], graphicsError))
                            return fail(&context, programNodeName(node) + ": binding set: " + graphicsError);
                        const auto &setCopies = sets[index].lowered->frame.deviceCopiesBefore;
                        copiesBefore.insert(copiesBefore.end(), setCopies.begin(), setCopies.end());
                    }
                }
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
                if (!planRenderTargets(&renderPass->renderPass, describeRuntimeImage, &context, *graphicsNode.targets,
                                       graphicsError) ||
                    !checkProgramGraphicsAttachmentSignature(graphics, *graphicsNode.targets, graphicsError))
                    return fail(&context, programNodeName(node) + ": " + graphicsError);
                const std::vector<ArgumentOwner> owners =
                    frameOwners(invocationContext, valueSlots, nodeKey, *resolvedNode, *frame);
                const bool planned =
                    recordedPass ? planRecordedGraphicsNode(context, *recordedPass, *frame, owners, sets, *renderPass,
                                                            graphicsNode, graphicsError)
                                 : planUnrecordedGraphicsNode(context, *resolvedNode->stage, graphics, *frame, owners,
                                                              *renderPass, draw, dynamic, graphicsNode, graphicsError);
                if (!planned)
                    return fail(&context, programNodeName(node) + ": " + graphicsError);
                if (recordedPass)
                    recordedPass->telemetry.render_scopes = 1;
                if (graphicsScopeReadsOverlap(graphicsNode.reads, *graphicsNode.targets))
                    return fail(&context, programNodeName(node) + ": samples an image it renders to");
                const auto candidate =
                    std::find_if(execution.graphicsScopeCandidates.begin(), execution.graphicsScopeCandidates.end(),
                                 [&](const program::ResolvedGraphicsScopeCandidate &entry) {
                                     return entry.node == program::NodeKey{*direction, node.id};
                                 });
                if (candidate == execution.graphicsScopeCandidates.end())
                    return fail(&context, "managed graphics node has no resolved scope candidate");
                // Only a node following an open batch's scope can fuse into it.
                const GraphicsScopeMaterialization materialization = graphicsScopeMaterializer.materialize(
                    candidate->region, *graphicsNode.targets, graphicsNode.reads, !graphicsBatch);
                if (materialization != GraphicsScopeMaterialization::Fuse)
                    if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                        return pending;
                if (!graphicsBatch) {
                    graphicsBatch = std::make_shared<ManagedGraphicsCommandBatch>();
                    graphicsBatch->context = &context;
                    graphicsBatch->stream = &sharedCommandStream();
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
                graphicsBufferReads.merge(graphicsNode.bufferReads);
                graphicsNode.retained.push_back(std::move(frame));
                graphicsNode.retained.push_back(std::move(renderPass));
                for (InvocationSetFrame &set : sets)
                    graphicsNode.retained.push_back(std::move(set.lowered));
                appendManagedGraphicsNode(*graphicsBatch, std::move(graphicsNode));
                graphicsBatch->members.push_back(
                    {programNodeLabel(node), graphicsBatch->draws.size(),
                     recordedPass ? recordedPass->debugCommands : std::vector<RecordedDebugCommand>{}});
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

// Encodes the planned `invocation`, alone, on `commands` inside their rendering scope.
VernonStatus encodePlannedGraphics(VernonStageExecutable &pipeline, CommandStream &commands,
                                   const VernonStageInvocationDescriptor &invocation,
                                   const PlannedGraphicsInvocation &plan) {
    GraphicsStream stream;
    const GraphicsCommand command{plan.arguments.data(),
                                  &plan.targets,
                                  invocation.graphics_state,
                                  plan.draw,
                                  {nullptr, &plan, &commands.transient(), {}},
                                  commands.encoder()};
    auto invoked = invokeBackendPipeline(pipeline, commands.commands(), command, stream);
    return invoked.isOk() ? VERNON_STATUS_OK : vernon::toVernonStatus(std::move(invoked).error());
}

VernonStatus encodeStageInvocation(VernonStageExecutable &pipeline, CommandStream &commands,
                                   const VernonStageInvocationDescriptor &invocation, const ArgumentOwner *owners) {
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
        auto invoked = invokeBackendComputePipeline(pipeline, commands, plan, owners);
        return invoked.isOk() ? VERNON_STATUS_OK : vernon::toVernonStatus(std::move(invoked).error());
    }
    PlannedGraphicsInvocation plan;
    std::string planningError;
    if (!planGraphicsInvocation(pipeline.bindingProjection, pipeline.drawLayout, invocation, describeRuntimeImage,
                                pipeline.context, plan, planningError))
        return fail(pipeline.context, std::move(planningError));
    return encodePlannedGraphics(pipeline, commands, invocation, plan);
}

} // namespace

VernonStatus vernon::runtime::executePipelineProgramGraph(
    VernonRuntimeContext &context, const program::ResolvedExecutionPlan &execution, const program::Graph &graph,
    program_execution::ProgramInvocationState &frame,
    const program_execution::ResolvePhysicalEndpoint &resolvePhysicalEndpoint,
    program_execution::SubmissionRecord &submission, program_execution::SubmissionState &step) {
    return executePipelineProgramGraphImpl(context, execution, graph, frame, resolvePhysicalEndpoint, submission, step);
}

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
        auto commands = beginCommandStream(*pipeline->context, invocation->command_encoder);
        if (commands.isErr())
            return vernon::toVernonStatus(std::move(commands).error());
        VernonStatus status = encodeStageInvocation(*pipeline, commands.value(), *invocation, nullptr);
        if (auto ended = commands.value().end(); status == VERNON_STATUS_OK && ended.isErr())
            status = vernon::toVernonStatus(std::move(ended).error());
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
    EncoderCommandStream stream(*pipeline->context);
    auto acquired = stream.acquire(native);
    VernonStatus status = acquired.isOk() ? VERNON_STATUS_OK : vernon::toVernonStatus(std::move(acquired).error());
    CommandStream *commands = status == VERNON_STATUS_OK ? acquired.value() : nullptr;
    if (commands)
        encoded.command_encoder = commands->encoder();
    bool rendering = false;
    PlannedGraphicsInvocation plan;
    if (status == VERNON_STATUS_OK && graphics) {
        std::string planningError;
        if (!planGraphicsInvocation(pipeline->bindingProjection, pipeline->drawLayout, encoded, describeRuntimeImage,
                                    pipeline->context, plan, planningError))
            status = fail(pipeline->context, std::move(planningError));
        GraphicsScopeReads reads;
        if (status == VERNON_STATUS_OK) {
            for (uint32_t image : pipeline->drawLayout.sampledImages)
                addGraphicsScopeRead(reads, plan.arguments[image]->image.view, plan.planned[image].subresources);
            if (graphicsScopeReadsOverlap(reads, plan.targets))
                status = fail(pipeline->context, "graphics invocation samples an image it renders to");
        }
        if (status == VERNON_STATUS_OK) {
            const GraphicsRenderScope scope = graphicsRenderScope(plan.targets, std::move(reads));
            auto begun =
                beginBackendRendering(*pipeline->context, commands->commands(), graphicsRenderingDescriptor(scope));
            if (begun.isErr())
                status = vernon::toVernonStatus(std::move(begun).error());
            else
                rendering = true;
        }
    }
    if (status == VERNON_STATUS_OK && graphics) {
        status = encodePlannedGraphics(*pipeline, *commands, encoded, plan);
    } else if (status == VERNON_STATUS_OK) {
        status = encodeStageInvocation(*pipeline, *commands, encoded, nullptr);
    }
    if (rendering) {
        auto ended = endBackendRendering(*pipeline->context, commands->commands());
        if (status == VERNON_STATUS_OK && ended.isErr())
            status = vernon::toVernonStatus(std::move(ended).error());
    }
    if (commands) {
        auto ended = commands->end();
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
