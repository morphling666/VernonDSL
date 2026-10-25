#include "runtime/capi/graphics_recording.h"
#include "runtime/capi/boundary.h"
#include "runtime/capi/program_state.h"
#include "runtime/program_execution/program_boundary_contract.h"
#include "runtime/recorded_graphics.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_dispatch.h"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace vernon::runtime;
using namespace vernon::runtime::capi;

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
    state->graphicsState.describe(program::graphicsOperation(*state->sourceNode));
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
    match->entryEndpoints.resize(set.entries.size());
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
        for (uint32_t index = 0; index < supplied.size(); ++index)
            if (supplied[index]) {
                match->endpoints[index] = 1;
                match->entryEndpoints[entryIndex].push_back(index);
            }
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
    state->graphicsState.describe(program::graphicsOperation(*state->sourceNode));
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

std::optional<uint64_t> imageParent(VernonRuntimeContext &context, VernonRuntimeProviderResourceReference view) {
    auto described = describeBackendImage(context, view);
    if (described.isErr())
        return std::nullopt;
    return described.value().parent_identity ? described.value().parent_identity : view.resource.value;
}

bool addWrittenArgument(VernonRuntimeContext &context, const VernonProgramArgument &argument,
                        RecordedWrittenResources &written) {
    if (argument.kind == VERNON_PROGRAM_TENSOR) {
        if (argument.tensor.storage != VERNON_TENSOR_RHI_RESOURCE)
            return true;
        written.buffers.push_back(RecordedWrittenResources::range(argument.tensor.resource, argument.tensor.byte_offset,
                                                                  argument.tensor.byte_size));
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
                                      const RecordedWrittenResources &written) {
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
                             const vernon::runtime::program::InvocationSnapshot &snapshot,
                             RecordedWrittenResources &written) {
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

// The set entry that supplies one endpoint of the bound pipeline; null `set` leaves the endpoint unbound.
struct BoundParameter {
    RuntimeBindingSetState *set{};
    size_t entry{};
    size_t frame{};
};

const std::string &endpointPath(const RuntimeGraphicsPipelineState &pipeline, size_t endpoint) {
    return executableProgram(*pipeline.source)->abi.boundarySlots[*pipeline.endpoints[endpoint].parameterSlot].path;
}

struct SetPipelineKey {
    const RuntimeBindingSetState *set{};
    const RuntimeGraphicsPipelineState *pipeline{};

    bool operator==(const SetPipelineKey &other) const { return set == other.set && pipeline == other.pipeline; }
};

struct SetPipelineHash {
    size_t operator()(const SetPipelineKey &key) const {
        return std::hash<const void *>{}(key.set) * 31 + std::hash<const void *>{}(key.pipeline);
    }
};

std::string commandPath(VernonProgramNodeId node, size_t command) {
    return "node " + std::to_string(node) + ".commands[" + std::to_string(command) + "]";
}

// Everything except the draw values, which buildDrawPayload collects, and hazards, which depend on what each
// invocation writes.
VernonStatus validateGraphicsRecording(VernonProgramExecutable &executable,
                                       const RuntimeGraphicsRecordingState &recording,
                                       const std::shared_ptr<const RuntimeGraphicsPipelineState> &passPipeline,
                                       RecordedGraphicsPass &pass) {
    VernonRuntimeContext &context = *executable.context;
    const VernonProgramNodeId nodeId = recording.node;
    const auto invalid = [&](const std::string &path, const std::string &message) {
        return failDiagnostic(&context, {recorded::kRecordingInvalid, recorded::kInvocationEntryPhase, path, message});
    };
    std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline;
    // Per endpoint of `pipeline`.
    std::vector<BoundParameter> bound;
    std::unordered_map<SetPipelineKey, size_t, SetPipelineHash> frames;
    std::vector<char> frameChecked;
    std::unordered_set<const RuntimeBindingSetState *> distinctSets;
    // Shared by consecutive draws until a bind changes what their parameters take.
    std::shared_ptr<const std::vector<RecordedArgumentSource>> arguments;
    size_t openDebugGroups = 0;
    pass.telemetry = {sizeof(VernonProgramGraphicsNodeTelemetry), 0, 0, 0, 0};
    const auto frameFor = [&](RuntimeBindingSetState &set) {
        const auto [found, inserted] = frames.try_emplace({&set, pipeline.get()}, pass.frames.size());
        if (inserted) {
            pass.frames.push_back({matchBindingSet(set, pipeline), pipeline});
            frameChecked.push_back(0);
        }
        return found->second;
    };
    for (size_t index = 0; index < recording.commands.size(); ++index) {
        const RuntimeRecordedCommand &command = recording.commands[index];
        const auto path = [&] { return commandPath(nodeId, index); };
        switch (command.kind) {
        case VERNON_GRAPHICS_COMMAND_BIND_PIPELINE: {
            std::shared_ptr<const RuntimeGraphicsPipelineState> next =
                command.pipeline ? command.pipeline : passPipeline;
            if (next->pass != &executable || next->node != nodeId)
                return invalid(path(), "pipeline was created for another graphics Node");
            // A parameter stays bound across the switch, by name, when its entry corresponds in the new pipeline.
            std::unordered_map<std::string_view, BoundParameter> named;
            for (size_t endpoint = 0; endpoint < bound.size(); ++endpoint)
                if (bound[endpoint].set)
                    named[endpointPath(*pipeline, endpoint)] = bound[endpoint];
            pipeline = std::move(next);
            bound.assign(pipeline->endpoints.size(), {});
            for (size_t endpoint = 0; endpoint < bound.size() && !named.empty(); ++endpoint) {
                if (!pipeline->endpoints[endpoint].parameterSlot)
                    continue;
                const auto found = named.find(endpointPath(*pipeline, endpoint));
                if (found == named.end())
                    continue;
                const size_t frame = frameFor(*found->second.set);
                const RuntimeBindingSetMatch &match = *pass.frames[frame].match;
                if (!match.failures[found->second.entry] && match.endpoints[endpoint])
                    bound[endpoint] = {found->second.set, found->second.entry, frame};
            }
            arguments.reset();
            ++pass.telemetry.pipeline_binds;
            break;
        }
        case VERNON_GRAPHICS_COMMAND_BIND_SET: {
            if (!pipeline)
                return invalid(path(), "binding set is bound before any pipeline");
            const size_t frame = frameFor(*command.set);
            const RuntimeBindingSetMatch &match = *pass.frames[frame].match;
            if (!std::exchange(frameChecked[frame], 1))
                for (const program::Diagnostic &failure : match.failures)
                    if (failure)
                        return failDiagnostic(&context, prefixedDiagnostic(failure, path()));
            if (distinctSets.insert(command.set.get()).second)
                pass.sets.emplace_back(index, command.set);
            for (size_t entry = 0; entry < command.set->entries.size(); ++entry)
                for (uint32_t endpoint : match.entryEndpoints[entry])
                    bound[endpoint] = {command.set.get(), entry, frame};
            arguments.reset();
            ++pass.telemetry.set_binds;
            break;
        }
        case VERNON_GRAPHICS_COMMAND_SET_VIEWPORT:
        case VERNON_GRAPHICS_COMMAND_SET_SCISSOR:
        case VERNON_GRAPHICS_COMMAND_SET_STENCIL_REFERENCE:
            break;
        case VERNON_GRAPHICS_COMMAND_DRAW: {
            if (!pipeline)
                return invalid(path(), "draw has no bound pipeline");
            if (!arguments) {
                auto sources = std::make_shared<std::vector<RecordedArgumentSource>>();
                sources->reserve(pipeline->endpoints.size());
                for (size_t parameter = 0; parameter < pipeline->endpoints.size(); ++parameter) {
                    const recorded::PipelineEndpoint &endpoint = pipeline->endpoints[parameter];
                    if (endpoint.fixedPassParameter) {
                        sources->push_back({false, 0, *endpoint.fixedPassParameter});
                        continue;
                    }
                    if (bound[parameter].set) {
                        sources->push_back({true, bound[parameter].frame, parameter});
                        continue;
                    }
                    if (!endpoint.initialPassParameter)
                        return invalid(path(),
                                       "per-draw parameter '" + endpointPath(*pipeline, parameter) + "' is unbound");
                    sources->push_back({false, 0, *endpoint.initialPassParameter});
                }
                arguments = std::move(sources);
            }
            pass.draws.push_back({pipeline, arguments});
            break;
        }
        case VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP:
            if (!openDebugGroups)
                return invalid(path(), "pops a debug group that is not open");
            --openDebugGroups;
            pass.debugCommands.push_back({command.kind, {}, pass.draws.size()});
            break;
        case VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP:
            ++openDebugGroups;
            [[fallthrough]];
        case VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER:
            pass.debugCommands.push_back({command.kind, command.label, pass.draws.size()});
            break;
        }
    }
    if (openDebugGroups)
        return invalid("node " + std::to_string(nodeId), "pushes debug groups it does not pop");
    pass.leases = recording.leases;
    return VERNON_STATUS_OK;
}

// The draw values of `recording`, whose dynamic state starts from `initialDynamic`, null when the invocation sets none.
std::shared_ptr<const RecordedDrawPayload>
buildDrawPayload(const RuntimeGraphicsRecordingState &recording,
                 const std::shared_ptr<const RuntimeProgramControl> &initialDynamic) {
    auto payload = std::make_shared<RecordedDrawPayload>();
    VernonDynamicState dynamic{};
    dynamic.struct_size = sizeof(dynamic);
    // Shared by consecutive draws until a dynamic-state command changes `dynamic`.
    std::shared_ptr<const RuntimeProgramControl> dynamicControl = initialDynamic;
    if (dynamicControl)
        dynamic = dynamicControl->dynamic;
    for (size_t index = 0; index < recording.commands.size(); ++index) {
        const RuntimeRecordedCommand &command = recording.commands[index];
        switch (command.kind) {
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
        case VERNON_GRAPHICS_COMMAND_DRAW:
            if (!dynamicControl) {
                auto control = std::make_shared<RuntimeProgramControl>();
                control->kind = RuntimeProgramControl::DynamicState;
                control->dynamic = dynamic;
                dynamicControl = std::move(control);
            }
            if (command.draw->index)
                payload->indices.emplace_back(index, *command.draw->index);
            payload->draws.push_back(command.draw);
            payload->dynamics.push_back(dynamicControl);
            payload->drawn += command.draw->draw.instance_count != 0;
            break;
        default:
            break;
        }
    }
    return payload;
}

VernonStatus checkGraphicsHazards(VernonRuntimeContext &context, VernonProgramNodeId node,
                                  const RecordedGraphicsPass &pass, const RecordedDrawPayload &payload,
                                  const RecordedWrittenResources &written) {
    for (const auto &[command, set] : pass.sets)
        for (const RuntimeBindingSetEntry &binding : set->entries) {
            const std::optional<bool> isWritten = argumentIsWritten(context, binding.binding->argument, written);
            const std::string path = commandPath(node, command) + "." + binding.boundary->path;
            if (!isWritten)
                return failDiagnostic(&context, {recorded::kRecordingInvalid, recorded::kInvocationEntryPhase, path,
                                                 "cannot describe the bound image"});
            if (*isWritten)
                return failDiagnostic(&context, {recorded::kBindingSetHazard, recorded::kInvocationEntryPhase, path,
                                                 "resource is written by a Node of the same invocation"});
        }
    for (const auto &[command, index] : payload.indices)
        if (written.overlapsBuffer(index.resource, 0, index.resource.size))
            return failDiagnostic(&context, {recorded::kBindingSetHazard, recorded::kInvocationEntryPhase,
                                             commandPath(node, command) + ".index",
                                             "index buffer is written by a Node of the same invocation"});
    return VERNON_STATUS_OK;
}
} // namespace

VernonStatus vernon::runtime::capi::validateGraphicsRecordings(VernonProgramInvocation &invocation,
                                                               RecordedGraphicsPasses &passes) {
    VernonProgramExecutable &executable = *invocation.instance->pipeline;
    VernonRuntimeContext &context = *executable.context;
    RecordedWrittenResources written;
    if (!collectWrittenResources(context, executable, *invocation.snapshot, written))
        return fail(&context, "cannot describe the images a recorded invocation writes");
    for (const auto &[node, recording] : invocation.recordings) {
        auto passPipeline = passPipelineState(executable, node);
        if (passPipeline.isErr())
            return failDiagnostic(&context, {recorded::kRecordingInvalid, recorded::kInvocationEntryPhase,
                                             "node " + std::to_string(node), passPipeline.error()});
        const program::GraphicsOperation &graphics = program::graphicsOperation(*passPipeline.value()->sourceNode);
        std::shared_ptr<const RuntimeProgramControl> initialDynamic;
        VernonDynamicState dynamic{};
        if (const std::shared_ptr<void> *payload = invocation.snapshot->find(
                programControlKey(graphics.dynamicStateControl, RuntimeProgramControl::DynamicState))) {
            initialDynamic = std::static_pointer_cast<const RuntimeProgramControl>(*payload);
            dynamic = initialDynamic->dynamic;
        }
        if (!recording->validated) {
            auto validated = std::make_shared<RecordedGraphicsPass>();
            if (const VernonStatus status =
                    validateGraphicsRecording(executable, *recording, passPipeline.value(), *validated);
                status != VERNON_STATUS_OK)
                return status;
            recording->validated = std::move(validated);
        }
        if (!recording->payload || !sameDynamicState(recording->payloadDynamic, dynamic)) {
            recording->payload = buildDrawPayload(*recording, initialDynamic);
            recording->payloadDynamic = dynamic;
        }
        if ((!written.buffers.empty() || !written.images.empty()) &&
            (recording->hazardFreePass != recording->validated || recording->hazardFreePayload != recording->payload ||
             !(recording->hazardFreeWritten == written))) {
            if (const VernonStatus status =
                    checkGraphicsHazards(context, node, *recording->validated, *recording->payload, written);
                status != VERNON_STATUS_OK)
                return status;
            recording->hazardFreePass = recording->validated;
            recording->hazardFreePayload = recording->payload;
            recording->hazardFreeWritten = written;
        }
        VernonProgramGraphicsNodeTelemetry telemetry = recording->validated->telemetry;
        telemetry.draws = recording->payload->drawn;
        passes.nodes[node] = {recording->validated, recording->payload, telemetry};
    }
    return VERNON_STATUS_OK;
}

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
        std::shared_ptr<RuntimeProgramBinding> binding = std::move(made).value();
        VernonTensorView &tensor = binding->argument.tensor;
        if (binding->argument.kind == VERNON_PROGRAM_TENSOR && tensor.storage == VERNON_TENSOR_HOST) {
            const auto *bytes = static_cast<const uint8_t *>(tensor.host_data);
            binding->ownedHostBytes.assign(bytes, bytes + tensor.byte_size);
            tensor.host_data = binding->ownedHostBytes.data();
        }
        const auto existing =
            std::find_if(state.entries.begin(), state.entries.end(),
                         [&](const RuntimeBindingSetEntry &entry) { return entry.boundary == &boundary; });
        if (existing != state.entries.end())
            existing->binding = std::move(binding);
        else
            state.entries.push_back({&boundary, std::move(binding)});
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

namespace {
using GraphicsRecordingHandleResult = vernon::Result<VernonGraphicsRecording *, vernon::RuntimeError>;

GraphicsRecordingHandleResult createGraphicsRecordingOperation(VernonProgramInstance *instance,
                                                               VernonProgramNodeId node) {
    VernonRuntimeContext *context = instance && instance->state ? instance->state->pipeline->context : nullptr;
    const auto invalid = [&](const std::string &message) {
        fail(context, message);
        return GraphicsRecordingHandleResult{vernon::err(
            vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument, {"graphics recording creation", 0, 0}})};
    };
    if (!context)
        return invalid("invalid graphics recording creation");
    if (!forwardGraphicsNode(*instance->state->pipeline, node))
        return invalid("graphics recording Node " + std::to_string(node) +
                       " is not a resolved graphics Node of the executable");
    auto state = std::make_shared<RuntimeGraphicsRecordingState>();
    state->context = context;
    state->instance = instance->state;
    state->node = node;
    auto handle = std::make_unique<VernonGraphicsRecording>();
    handle->state = std::move(state);
    return GraphicsRecordingHandleResult{vernon::ok(handle.release())};
}

VernonStatus convertGraphicsCommand(VernonRuntimeContext *context, const VernonGraphicsCommand &source, size_t index,
                                    RuntimeRecordedCommand &command) {
    const auto path = [&] { return "graphics recording command " + std::to_string(index); };
    command.kind = source.kind;
    switch (source.kind) {
    case VERNON_GRAPHICS_COMMAND_BIND_PIPELINE:
        if (source.pipeline && (!source.pipeline->state || source.pipeline->context != context))
            return fail(context, path() + " has an invalid pipeline or one of another Runtime context");
        if (source.pipeline)
            command.pipeline = source.pipeline->state;
        break;
    case VERNON_GRAPHICS_COMMAND_BIND_SET:
        if (!source.binding_set || !source.binding_set->state || !source.binding_set->state->sealed)
            return fail(context, path() + " binds a missing or unsealed binding set");
        if (source.binding_set->state->context != context)
            return fail(context, path() + " binds a binding set of another Runtime context");
        command.set = source.binding_set->state;
        break;
    case VERNON_GRAPHICS_COMMAND_SET_VIEWPORT:
    case VERNON_GRAPHICS_COMMAND_SET_SCISSOR:
        std::copy(std::begin(source.rect), std::end(source.rect), command.rect.begin());
        break;
    case VERNON_GRAPHICS_COMMAND_SET_STENCIL_REFERENCE:
        if (source.stencil_reference > 0xff)
            return fail(context, path() + " has a stencil reference above 255");
        command.stencilReference = source.stencil_reference;
        break;
    case VERNON_GRAPHICS_COMMAND_DRAW:
        command.draw = makeDrawCommandControl(&source.draw);
        if (!command.draw)
            return fail(context, path() + " has an invalid DrawCommand");
        break;
    case VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP:
    case VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER:
        if (source.label.size && !source.label.data)
            return fail(context, path() + " has an invalid label");
        command.label.assign(source.label.size ? source.label.data : "", source.label.size);
        break;
    case VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP:
        break;
    default:
        return fail(context, path() + " has an unknown kind");
    }
    return VERNON_STATUS_OK;
}

VernonRuntimeContext *recordingContext(const VernonGraphicsRecording *recording) {
    return recording && recording->state ? recording->state->context : nullptr;
}
} // namespace

VernonRuntimeOperationStatus vernonRuntimeGraphicsRecordingCreate(VernonProgramInstance *instance,
                                                                  VernonProgramNodeId node,
                                                                  VernonGraphicsRecording **output) {
    VernonRuntimeContext *context = instance && instance->state ? instance->state->pipeline->context : nullptr;
    return runtimeHandleResultBoundary(context, output,
                                       [&] { return createGraphicsRecordingOperation(instance, node); });
}

void vernonRuntimeGraphicsRecordingDestroy(VernonGraphicsRecording *recording) {
    runtimeVoidBoundary(recordingContext(recording), [&] { delete recording; });
}

VernonStatus vernonRuntimeGraphicsRecordingReset(VernonGraphicsRecording *recording) {
    VernonRuntimeContext *context = recordingContext(recording);
    return runtimeStatusBoundary(context, [&] {
        if (!context)
            return fail(context, "invalid graphics recording reset");
        RuntimeGraphicsRecordingState &state = *recording->state;
        state.commands.clear();
        state.leases.clear();
        state.validated.reset();
        state.payload.reset();
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeGraphicsRecordingSetCommand(VernonGraphicsRecording *recording, size_t index,
                                                      const VernonGraphicsCommand *command) {
    VernonRuntimeContext *context = recordingContext(recording);
    return runtimeStatusBoundary(context, [&] {
        if (!context || !command || index >= recording->state->commands.size())
            return fail(context, "invalid graphics recording command update");
        RuntimeRecordedCommand &target = recording->state->commands[index];
        const std::string path = "graphics recording command " + std::to_string(index);
        switch (command->kind) {
        case VERNON_GRAPHICS_COMMAND_SET_VIEWPORT:
        case VERNON_GRAPHICS_COMMAND_SET_SCISSOR:
        case VERNON_GRAPHICS_COMMAND_SET_STENCIL_REFERENCE:
        case VERNON_GRAPHICS_COMMAND_DRAW:
            break;
        default:
            return fail(context, path + " can only be replaced by a viewport, scissor, stencil reference or draw");
        }
        if (command->kind != target.kind)
            return fail(context, path + " cannot change its kind");
        RuntimeRecordedCommand replacement;
        if (const VernonStatus status = convertGraphicsCommand(context, *command, index, replacement);
            status != VERNON_STATUS_OK)
            return status;
        if (replacement.kind == VERNON_GRAPHICS_COMMAND_DRAW) {
            const std::optional<VernonIndexBinding> &previous = target.draw->index;
            const std::optional<VernonIndexBinding> &next = replacement.draw->index;
            if (previous.has_value() != next.has_value() ||
                (previous && (previous->resource.identity != next->resource.identity ||
                              previous->resource.resource.value != next->resource.resource.value)))
                return fail(context, path + " cannot change whether it is indexed or its index buffer");
        }
        target = std::move(replacement);
        recording->state->payload.reset();
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationBindRecording(VernonProgramInvocation *invocation,
                                                         VernonGraphicsRecording *recording) {
    VernonProgramExecutable *executable = invocationExecutable(invocation);
    VernonRuntimeContext *context = executable ? executable->context : nullptr;
    return runtimeStatusBoundary(context, [&] {
        if (!invocation || !executable || invocation->executed || invocation->finished || !recording ||
            !recording->state)
            return fail(context, "invalid graphics recording binding");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(context, "Program invocation is closing", vernon::toVernonStatus(invocationPin.error()));
        const std::weak_ptr<RuntimeProgramInstanceState> &owner = recording->state->instance;
        if (owner.owner_before(invocation->instance) || invocation->instance.owner_before(owner))
            return fail(context, "graphics recording belongs to another Program instance");
        invocation->recordings[recording->state->node] = recording->state;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeGraphicsRecordingAppend(VernonGraphicsRecording *recording,
                                                  const VernonGraphicsCommand *commands, size_t commandCount,
                                                  const VernonProgramResourceLease *leases, size_t leaseCount) {
    VernonRuntimeContext *context = recordingContext(recording);
    return runtimeStatusBoundary(context, [&] {
        if (!context || (commandCount && !commands) || (leaseCount && !leases))
            return fail(context, "invalid graphics recording");
        std::vector<RuntimeRecordedCommand> appended;
        std::vector<std::shared_ptr<void>> appendedLeases;
        appended.reserve(commandCount);
        for (size_t index = 0; index < commandCount; ++index) {
            RuntimeRecordedCommand command;
            if (const VernonStatus status = convertGraphicsCommand(context, commands[index], index, command);
                status != VERNON_STATUS_OK)
                return status;
            appended.push_back(std::move(command));
        }
        for (size_t index = 0; index < leaseCount; ++index) {
            auto retained = retainProgramResourceLease(leases[index]);
            if (retained.isErr())
                return publishRuntimeError(context, std::move(retained).error());
            if (retained.value())
                appendedLeases.push_back(std::move(retained).value());
        }
        RuntimeGraphicsRecordingState &state = *recording->state;
        state.commands.insert(state.commands.end(), std::make_move_iterator(appended.begin()),
                              std::make_move_iterator(appended.end()));
        state.leases.insert(state.leases.end(), std::make_move_iterator(appendedLeases.begin()),
                            std::make_move_iterator(appendedLeases.end()));
        state.validated.reset();
        state.payload.reset();
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
