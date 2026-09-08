#include "program_invocation_builder.h"

#include "program_boundary_binder.h"
#include "program_invocation_values.h"
#include "program_tape_scratch.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/program_execution/publication_transaction.h"
#include "runtime_autodiff_internal.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace vernon::runtime::ad {
using program_execution::ProgramInvocationState;
using program_execution::ProgramStorageBacking;
using program_execution::ProgramValueOwnership;
using program_execution::ProgramValueState;
namespace {

struct InvocationBuildRequest {
    const std::vector<char> &required;
    const CanonicalBoundaryBindings *boundaries{};
    const std::vector<std::vector<uint8_t>> *captures{};
    const std::vector<std::vector<uint64_t>> *captureShapes{};
    const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures{};
    const std::vector<program_execution::CanonicalValueSnapshot> *retainedSnapshots{};
};

bool attachResiduals(const InvocationBuildRequest &request, const program::Program &execution,
                     std::vector<ProgramValueState> &values, std::map<uint32_t, ProgramStorageBacking> &backings,
                     std::string &error) {
    if (!request.captures && !request.captureShapes)
        return true;
    if (!request.captures || !request.captureShapes || request.captures->size() != execution.values.size() ||
        request.captureShapes->size() != execution.values.size())
        return error = "Program autodiff retained state is incomplete", false;
    for (size_t value = 0; value < request.captures->size(); ++value) {
        if (!(*request.captureShapes)[value].empty())
            values[value].concreteShape = (*request.captureShapes)[value];
        if ((*request.captures)[value].empty())
            continue;
        values[value].ownership = ProgramValueOwnership::RetainedResidual;
        const program::Value &slot = execution.values[value];
        if (!program::isTapeValueType(slot.type) && slot.storage) {
            ProgramStorageBacking &backing = backings[*slot.storage];
            backing.bytes = std::max(backing.bytes, (*request.captures)[value].size());
            backing.sized = true;
        }
    }
    return true;
}

bool build(VernonRuntimeContext &context, const program::Program &execution, const program::ResolvedExecutionPlan *plan,
           std::vector<ProgramValueState> &values, std::map<uint32_t, ProgramStorageBacking> &storageBackings,
           ProgramTapeScratch &tapeScratch, const InvocationBuildRequest &request,
           std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    values.assign(execution.values.size(), {});
    std::vector<char> typedControls(execution.values.size());
    std::set<uint32_t> typedControlStorages;
    for (const program::Graph &graph : execution.graphs)
        for (const program::GraphInput &input : graph.inputs)
            if (input.kind == program::GraphInputKind::Control && input.value < execution.values.size()) {
                typedControls[input.value] = 1;
                if (execution.values[input.value].storage)
                    typedControlStorages.insert(*execution.values[input.value].storage);
            }

    std::vector<char> live = request.required;
    live.resize(execution.values.size());
    if (request.boundaries)
        for (const auto &[slot, value] : request.boundaries->valueBySlot) {
            (void)slot;
            if (value < live.size())
                live[value] = 1;
        }
    if (request.captures)
        for (size_t value = 0; value < request.captures->size() && value < live.size(); ++value)
            live[value] |= !(*request.captures)[value].empty();
    if (request.tapeCaptures)
        for (size_t value = 0; value < request.tapeCaptures->size() && value < live.size(); ++value)
            live[value] |= static_cast<bool>((*request.tapeCaptures)[value]);
    for (const program::Value &slot : execution.values)
        if (slot.storage && typedControlStorages.count(*slot.storage))
            live[slot.id] = 0;

    std::vector<char> liveStorage(execution.storages.size());
    for (const program::Value &slot : execution.values)
        if (live[slot.id] && slot.storage && *slot.storage < liveStorage.size() &&
            !typedControlStorages.count(*slot.storage))
            liveStorage[*slot.storage] = 1;

    std::map<uint32_t, ProgramStorageBacking> backings;
    for (const program::Storage &slot : execution.storages)
        backings.emplace(slot.id, ProgramStorageBacking{
                                      slot.initialValue,
                                      static_cast<size_t>(slot.buffer.byteLength),
                                      slot.buffer.byteLength > 0,
                                      std::nullopt,
                                  });
    std::map<uint32_t, VernonProgramArgument> externalValues;
    if (request.boundaries) {
        ProgramBoundaryBindingRequest binding{
            request.boundaries->arguments,
            request.boundaries->argumentCount,
            request.boundaries->valueBySlot,
            &request.boundaries->publication,
        };
        if (!plan ||
            !bindProgramBoundaries(context, execution, *plan, binding, externalValues, backings, live, error) ||
            !request.boundaries->publication.applyConcreteShapes(execution, values, error))
            return false;
        for (auto &[value, argument] : externalValues) {
            if (value >= execution.values.size() || execution.values[value].storage ||
                argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
                !argument.tensor.host_data || !argument.tensor.byte_size)
                continue;
            const auto *data = static_cast<const uint8_t *>(argument.tensor.host_data) + argument.tensor.byte_offset;
            values[value].ownedHostBytes.assign(data, data + argument.tensor.byte_size);
            argument.tensor.host_data = values[value].ownedHostBytes.data();
            argument.tensor.byte_offset = 0;
        }
    }
    if (request.retainedSnapshots) {
        for (const auto &snapshot : *request.retainedSnapshots) {
            if (snapshot.value >= values.size() || snapshot.logical.argument.kind != VERNON_PROGRAM_TENSOR)
                continue;
            externalValues.emplace(snapshot.value, snapshot.logical.argument);
            values[snapshot.value].concreteShape = snapshot.logical.concreteShape;
            values[snapshot.value].strides = snapshot.logical.strides;
        }
    }

    std::vector<std::optional<ValueLayout>> layouts(execution.values.size());
    for (const program::Value &slot : execution.values) {
        if (typedControls[slot.id] || program::isTapeValueType(slot.type) ||
            (slot.storage && typedControlStorages.count(*slot.storage)))
            continue;
        layouts[slot.id] = resolvedProgramValueLayout(slot);
        const auto external = externalValues.find(slot.id);
        const VernonProgramArgument *argument = external != externalValues.end() ? &external->second
                                                : slot.storage && backings[*slot.storage].external
                                                    ? &*backings[*slot.storage].external
                                                    : nullptr;
        if (argument && argument->kind != VERNON_PROGRAM_TENSOR) {
            values[slot.id].argument = *argument;
            continue;
        }
        const bool dynamic = programValueHasDynamicShape(slot);
        if (!layouts[slot.id] || !layouts[slot.id]->byteSize || (!dynamic && !programValueByteSize(slot)))
            return error = "Program Value '" + slot.name + "' has no materializable tensor layout", false;
        if (argument) {
            const VernonTensorView &tensor = argument->tensor;
            if (tensor.rank < slot.shape.size() || (tensor.rank && (!tensor.shape || !tensor.byte_strides)))
                return error = "Program invocation Tensor has incomplete shape metadata", false;
            values[slot.id].concreteShape = shape::ConcreteShape();
            values[slot.id].strides.clear();
            if (!slot.shape.empty()) {
                values[slot.id].concreteShape->assign(tensor.shape, tensor.shape + slot.shape.size());
                values[slot.id].strides.assign(tensor.byte_strides, tensor.byte_strides + slot.shape.size());
            }
        }
        if (!slot.storage)
            continue;
        ProgramStorageBacking &backing = backings[*slot.storage];
        if (backing.owner == UINT32_MAX || backing.owner >= execution.values.size())
            backing.owner = slot.id;
        if (!dynamic) {
            if (const std::optional<size_t> bytes = programValueByteSize(slot))
                backing.bytes = std::max(backing.bytes, *bytes);
            backing.sized = true;
        }
    }
    if (!attachResiduals(request, execution, values, backings, error) ||
        !materializeProgramOwnedStorages(execution, plan, values, liveStorage, layouts, backings, error) ||
        !materializeProgramValues(execution, plan, values, live, layouts, backings, externalValues,
                                  request.tapeCaptures, tapePolicy, tapeScratch, error))
        return false;
    storageBackings = std::move(backings);
    return true;
}

} // namespace

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan, std::vector<ProgramValueState> &values,
                                  std::map<uint32_t, ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const ForwardInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    if (program_execution::injectFailure(program_execution::FailureBoundary::Allocation))
        return error = "injected Program invocation allocation failure", false;
    return build(context, execution, plan, values, storageBackings, tapeScratch, {spec.requiredValues, &spec.bindings},
                 std::move(tapePolicy), error);
}

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan, std::vector<ProgramValueState> &values,
                                  std::map<uint32_t, ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const PullbackInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    return build(context, execution, plan, values, storageBackings, tapeScratch,
                 {spec.requiredValues, &spec.bindings, &spec.captures, &spec.captureShapes, spec.tapeCaptures,
                  spec.retainedSnapshots},
                 std::move(tapePolicy), error);
}

} // namespace vernon::runtime::ad
