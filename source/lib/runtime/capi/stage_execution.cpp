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
#include <map>
#include <memory>
#include <optional>
#include <string>
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

struct ManagedGraphicsBatchMember {
    std::string label;
    // End of the member's draws in `ManagedGraphicsCommandBatch::draws`.
    size_t drawEnd{};
    // Positions count from the member's first draw.
    std::vector<RecordedDebugCommand> debugCommands;
};

struct ManagedGraphicsCommandBatch {
    VernonRuntimeContext *context{};
    GraphicsRenderScope scope;
    std::vector<std::shared_ptr<ManagedGraphicsCommandContext>> draws;
    std::vector<ManagedGraphicsBatchMember> members;
    // Its label gains `+<member label>` as each member fuses.
    std::shared_ptr<CommandScope> commandScope;
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

VernonRhiStatus encodeManagedGraphicsDraw(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                          const std::shared_ptr<ManagedGraphicsCommandContext> &context) {
    if (!context || context->pipeline.context->rhiDevice.index != device.index ||
        vernonRuntimeReferenceRhiCommandEncoder(context->pipeline.context, encoder,
                                                &context->invocation.command_encoder) != VERNON_STATUS_OK) {
        if (context)
            invocationDiagnostic(*context->pipeline.context) =
                "failed to reference the managed graphics command encoder";
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    auto invoked = invokeBackendPipeline(context->pipeline, context->invocation, context->plan);
    if (invoked.isOk())
        return VERNON_RHI_STATUS_OK;
    const vernon::RuntimeError error = std::move(invoked).error();
    if (invocationDiagnostic(*context->pipeline.context).empty())
        invocationDiagnostic(*context->pipeline.context) =
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
        if (const VernonRhiStatus status = encodeManagedGraphicsDraw(device, encoder, batch.draws[draw]);
            status != VERNON_RHI_STATUS_OK)
            return status;
    }
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
    // Members get their own groups only inside a fused scope; an unfused scope's group already carries the label.
    const bool memberGroups = batch.members.size() > 1;
    size_t firstDraw = 0;
    for (const ManagedGraphicsBatchMember &member : batch.members) {
        if (memberGroups)
            status = vernonRhiCommandEncoderPushDebugGroup(device, encoder, {member.label.data(), member.label.size()});
        if (status == VERNON_RHI_STATUS_OK)
            status = encodeManagedGraphicsMember(device, encoder, batch, member, firstDraw);
        if (memberGroups && status == VERNON_RHI_STATUS_OK)
            status = vernonRhiCommandEncoderPopDebugGroup(device, encoder);
        if (status != VERNON_RHI_STATUS_OK)
            break;
        firstDraw = member.drawEnd;
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
        // A Node's transfers are planned apart so that a graphics Node can end the open render batch before them.
        const auto appendNodeTransfers = [&](vernon::execution::detail::RhiCommandExecutionPlan &&nodeTransfers,
                                             const std::shared_ptr<const CommandScope> &scope) {
            setScope(nodeTransfers.encoders, scope);
            std::string compositionError;
            if (!vernon::execution::detail::appendRhiCommandExecutionPlan(commandPlan, std::move(nodeTransfers), true,
                                                                          compositionError))
                return fail(&context, std::move(compositionError));
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
                vernon::execution::detail::RhiCommandExecutionPlan nodeTransfers;
                bool transferCommandsAppended = false;
                if (const VernonStatus status = transfers.appendBeforeConsumer(
                        {*direction, node.id}, copiesBefore, nodeTransfers, transferCommandsAppended, transferError);
                    status != VERNON_STATUS_OK)
                    return status;
                const std::shared_ptr<CommandScope> scope = nodeScope(node);
                // Transfers run before this Node's draws, so they end the open scope and this Node opens its own.
                if (transferCommandsAppended) {
                    if (const VernonStatus pending = appendGraphicsBatch(); pending != VERNON_STATUS_OK)
                        return pending;
                    if (const VernonStatus appended = appendNodeTransfers(std::move(nodeTransfers), scope);
                        appended != VERNON_STATUS_OK)
                        return appended;
                }
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
                for (std::shared_ptr<ManagedGraphicsCommandContext> &graphicsContext : graphicsContexts) {
                    if (!appendGraphicsVertexResources(graphicsContext->plan, graphicsNodePlan.commands.nodes.back(),
                                                       graphicsNodePlan.bindings))
                        return fail(&context, "managed graphics node contains an invalid vertex or index resource");
                    graphicsBatch->draws.push_back(std::move(graphicsContext));
                }
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
            vernon::execution::detail::RhiCommandExecutionPlan nodeTransfers;
            bool transferCommandsAppended = false;
            if (const VernonStatus status =
                    transfers.appendBeforeConsumer({*direction, node.id}, materializedNodes.back().deviceCopiesBefore,
                                                   nodeTransfers, transferCommandsAppended, transferError);
                status != VERNON_STATUS_OK)
                return status;
            if (transferCommandsAppended)
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
                context, {}, {}, *resolvedNode->stage, materializedNodes.back().arguments, grid,
                materializedNodes.back().deviceCopiesAfter, vernon::execution::detail::CommandNodeKind::Derivative,
                nodePlan);
            if (planned != VERNON_STATUS_OK)
                return planned;
            setScope(nodePlan.encoders, scope);
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

VernonStageExecutable::~VernonStageExecutable() {
    if (backendState)
        vernon::runtime::destroyBackendPipeline(*this);
}
