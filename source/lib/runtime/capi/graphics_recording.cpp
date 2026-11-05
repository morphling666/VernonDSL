#include "runtime/capi/graphics_recording.h"
#include "runtime/capi/boundary.h"
#include "runtime/capi/program_state.h"
#include "runtime/graphics_invocation_planner.h"
#include "runtime/program_execution/materialized_node_frame.h"
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

} // namespace

PipelineStateResult vernon::runtime::capi::passPipelineState(VernonProgramExecutable &executable,
                                                             VernonProgramNodeId nodeId) {
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

namespace {

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

VernonStatus describeRuntimeImage(void *userData, VernonRuntimeProviderResourceReference resource,
                                  VernonRuntimeProviderImageDescription *description) {
    auto described = describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource);
    if (described.isErr())
        return vernon::toVernonStatus(std::move(described).error());
    *description = described.value();
    return VERNON_STATUS_OK;
}

} // namespace

std::shared_ptr<const RuntimeBindingSetMatch>
vernon::runtime::capi::matchBindingSet(RuntimeBindingSetState &set,
                                       const std::shared_ptr<const RuntimeGraphicsPipelineState> &pipeline) {
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

const std::string &vernon::runtime::capi::endpointPath(const RuntimeGraphicsPipelineState &pipeline, size_t endpoint) {
    return executableProgram(*pipeline.source)->abi.boundarySlots[*pipeline.endpoints[endpoint].parameterSlot].path;
}

bool vernon::runtime::capi::lowerSetFrame(VernonRuntimeContext &context, const RecordedSetFrame &frame,
                                          InvocationSetFrame &output, std::string &error) {
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
            output = {lowered, {lowered.get(), nullptr}};
            return true;
        }
        match.lowered = std::move(lowered);
    }
    output = {match.lowered, {&match, &match.groups}};
    return true;
}

namespace {

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

} // namespace

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
            const auto *bytes = static_cast<const uint8_t *>(tensor.host_data) + tensor.byte_offset;
            binding->ownedHostBytes.assign(bytes, bytes + tensor.byte_size);
            tensor.host_data = binding->ownedHostBytes.data();
            tensor.byte_offset = 0;
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
