#include "runtime/capi/program_state.h"
#include "runtime/program_execution/program_boundary_contract.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <set>

namespace vernon::runtime::capi {

namespace {

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

bool publicBoundarySlot(const program::BoundarySlot &slot) {
    if (slot.role == program::BoundaryRole::Input)
        return true;
    return slot.role == program::BoundaryRole::Output && slot.publication != program::BoundaryPublication::None;
}

const program::BoundarySlot *findPublicBoundary(const program::Program &program, VernonStringView name) {
    const auto found = std::find_if(
        program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(), [&](const program::BoundarySlot &slot) {
            return publicBoundarySlot(slot) && name.size == slot.path.size() &&
                   (name.size == 0 || std::equal(name.data, name.data + name.size, slot.path.data()));
        });
    return found == program.abi.boundarySlots.end() ? nullptr : &*found;
}

uint64_t programOwnerKey(const program::ProgramOwnerId &owner) {
    return (static_cast<uint64_t>(owner.kind) << 32) | owner.id;
}

} // namespace

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

const program::Program *executableProgram(const VernonProgramExecutable &pipeline) {
    return &pipeline.executionPlan->resolvedProgram->program;
}

size_t publicBoundaryCount(const program::Program &program) {
    return static_cast<size_t>(
        std::count_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(),
                      [&](const program::BoundarySlot &slot) { return publicBoundarySlot(slot); }));
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
        if (publicBoundarySlot(slot) && publicIndex-- == 0)
            return &slot;
    return nullptr;
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
        if (!pipeline.programGraphId && publicBoundarySlot(slot))
            pipeline.bindableParameterSlots.insert(slot.id);
        if (publicBoundarySlot(slot))
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

bool stringViewEquals(VernonStringView view, const std::string &value) {
    return view.size == value.size() && (!view.size || std::memcmp(view.data, value.data(), view.size) == 0);
}

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

} // namespace vernon::runtime::capi
