#include "program_invocation_builder.h"

#include "program_boundary_binder.h"
#include "program_invocation_values.h"
#include "program_tape_scratch.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/program_execution/publication_transaction.h"

#include <algorithm>
#include <map>
#include <optional>

namespace vernon::runtime::ad {
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
        if (!program::isTapeValue(slot) && slot.storage) {
            ProgramStorageBacking &backing = backings[*slot.storage];
            backing.bytes = std::max(backing.bytes, (*request.captures)[value].size());
            backing.sized = true;
        }
    }
    return true;
}

bool build(VernonRuntimeContext &context, const program::Program &execution, const program::ResolvedExecutionPlan *plan,
           const ProgramInvocationPreparation &preparation, ProgramBoundaryBindingScratch &bindingScratch,
           std::vector<ProgramValueState> &values, std::map<uint32_t, ProgramStorageBacking> &storageBackings,
           ProgramTapeScratch &tapeScratch, const InvocationBuildRequest &request,
           std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    values.resize(execution.values.size());
    for (size_t valueIndex = 0; valueIndex < values.size(); ++valueIndex) {
        ProgramValueState &value = values[valueIndex];
        value.ownedHostBytes.clear();
        value.strides.clear();
        if (plan && valueIndex < plan->preparedValueShapes.size() && plan->preparedValueShapes[valueIndex]) {
            if (!value.concreteShape)
                value.concreteShape.emplace();
            *value.concreteShape = *plan->preparedValueShapes[valueIndex];
        } else {
            value.concreteShape.reset();
        }
        if (value.boundTensorLayout) {
            value.boundTensorLayout->shape.clear();
            value.boundTensorLayout->byteStrides.clear();
        }
        value.argument = {};
        value.stagedDeviceInitial.reset();
        value.ownership = ProgramValueOwnership::OwnedInvocation;
    }
    for (const auto &[storage, prepared] : preparation.storageBackings)
        storageBackings.insert_or_assign(storage, prepared);

    bindingScratch.reset(execution);
    std::vector<char> &live = bindingScratch.liveValues;
    std::copy(request.required.begin(), request.required.end(), live.begin());
    if (request.boundaries)
        for (const auto &[slot, value] : request.boundaries->valueBySlot) {
            (void)slot;
            if (value < live.size())
                live[value] = 1;
        }
    if (request.captures)
        for (size_t value = 0; value < request.captures->size() && value < live.size(); ++value)
            live[value] = live[value] || !(*request.captures)[value].empty();
    if (request.tapeCaptures)
        for (size_t value = 0; value < request.tapeCaptures->size() && value < live.size(); ++value)
            live[value] = live[value] || static_cast<bool>((*request.tapeCaptures)[value]);
    for (const program::Value &slot : execution.values)
        if (slot.storage && *slot.storage < preparation.typedControlStorages.size() &&
            preparation.typedControlStorages[*slot.storage])
            live[slot.id] = 0;

    std::vector<char> &liveStorage = bindingScratch.liveStorages;
    for (const program::Value &slot : execution.values)
        if (live[slot.id] && slot.storage && *slot.storage < liveStorage.size() &&
            (*slot.storage >= preparation.typedControlStorages.size() ||
             !preparation.typedControlStorages[*slot.storage]))
            liveStorage[*slot.storage] = 1;

    auto &backings = storageBackings;
    if (request.boundaries) {
        ProgramBoundaryBindingRequest binding{
            request.boundaries->arguments,
            request.boundaries->argumentCount,
            request.boundaries->valueBySlot,
            &request.boundaries->publication,
        };
        if (!plan || !bindProgramBoundaries(context, execution, *plan, binding, bindingScratch, backings, live, error))
            return false;
        auto applied = request.boundaries->publication.applyConcreteShapes(execution, values);
        if (applied.isErr())
            return error = program_execution::publicationErrorMessage(applied.error()), false;
        for (size_t value = 0; value < bindingScratch.externalValues.size(); ++value) {
            std::optional<VernonProgramArgument> &external = bindingScratch.externalValues[value];
            if (!external)
                continue;
            VernonProgramArgument &argument = *external;
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
            bindingScratch.externalValues[snapshot.value] = snapshot.logical.argument;
            values[snapshot.value].concreteShape = snapshot.logical.concreteShape;
            values[snapshot.value].strides = snapshot.logical.strides;
        }
    }

    for (const program::Value &slot : execution.values) {
        const bool typedControlStorage = slot.storage && *slot.storage < preparation.typedControlStorages.size() &&
                                         preparation.typedControlStorages[*slot.storage];
        if (preparation.typedControls[slot.id] || preparation.tapeValues[slot.id] || typedControlStorage)
            continue;
        const std::optional<VernonProgramArgument> &external = bindingScratch.externalValues[slot.id];
        const VernonProgramArgument *argument = external ? &*external
                                                : slot.storage && backings[*slot.storage].external
                                                    ? &*backings[*slot.storage].external
                                                    : nullptr;
        if (argument && argument->kind != VERNON_PROGRAM_TENSOR) {
            values[slot.id].argument = *argument;
            continue;
        }
        const bool dynamic = preparation.dynamicShapes[slot.id];
        if (!preparation.layouts[slot.id] || !preparation.layouts[slot.id]->byteSize ||
            (!dynamic && !preparation.materializableStaticValues[slot.id]))
            return error = "Program Value '" + slot.name + "' has no materializable tensor layout", false;
        if (argument) {
            const VernonTensorView &tensor = argument->tensor;
            if (tensor.rank < slot.shape.size() || (tensor.rank && (!tensor.shape || !tensor.byte_strides)))
                return error = "Program invocation Tensor has incomplete shape metadata", false;
            if (!values[slot.id].concreteShape)
                values[slot.id].concreteShape.emplace();
            values[slot.id].concreteShape->clear();
            values[slot.id].strides.clear();
            if (!slot.shape.empty()) {
                values[slot.id].concreteShape->assign(tensor.shape, tensor.shape + slot.shape.size());
                values[slot.id].strides.assign(tensor.byte_strides, tensor.byte_strides + slot.shape.size());
            }
        }
    }
    if (!attachResiduals(request, execution, values, backings, error) ||
        !materializeProgramOwnedStorages(execution, plan, preparation, values, liveStorage, backings, error) ||
        !materializeProgramValues(execution, plan, preparation, values, live, backings, bindingScratch.externalValues,
                                  request.tapeCaptures, tapePolicy, tapeScratch, error))
        return false;
    return true;
}

} // namespace

ProgramInvocationPreparation prepareProgramInvocation(const program::Program &execution) {
    ProgramInvocationPreparation result;
    result.typedControls.resize(execution.values.size());
    result.typedControlStorages.resize(execution.storages.size());
    result.tapeValues.resize(execution.values.size());
    result.dynamicShapes.resize(execution.values.size());
    result.materializableStaticValues.resize(execution.values.size());
    result.staticByteSizes.resize(execution.values.size());
    result.layouts.resize(execution.values.size());
    result.forwardRequiredValues.resize(execution.values.size());
    result.backwardRequiredValues.resize(execution.values.size());
    if (const program::Graph *forward = program::findGraph(execution, "forward")) {
        program::markGraphValues(*forward, result.forwardRequiredValues);
        for (uint32_t value : program::residualCaptures(execution))
            if (value < result.forwardRequiredValues.size())
                result.forwardRequiredValues[value] = 1;
    }
    if (const program::Graph *backward = program::findGraph(execution, "backward"))
        program::markGraphValues(*backward, result.backwardRequiredValues);
    for (const program::Graph &graph : execution.graphs)
        for (const program::GraphInput &input : graph.inputs)
            if (input.kind == program::GraphInputKind::Control && input.value < execution.values.size()) {
                result.typedControls[input.value] = 1;
                const std::optional<uint32_t> storage = execution.values[input.value].storage;
                if (storage && *storage < result.typedControlStorages.size())
                    result.typedControlStorages[*storage] = 1;
            }
    for (const program::Storage &slot : execution.storages)
        result.storageBackings.emplace(slot.id, ProgramStorageBacking{
                                                    slot.initialValue,
                                                    static_cast<size_t>(slot.buffer.byteLength),
                                                    slot.buffer.byteLength > 0,
                                                    std::nullopt,
                                                });
    for (const program::Value &slot : execution.values) {
        result.tapeValues[slot.id] = program::isTapeValue(slot);
        result.dynamicShapes[slot.id] = programValueHasDynamicShape(slot);
        const bool typedControlStorage = slot.storage && *slot.storage < result.typedControlStorages.size() &&
                                         result.typedControlStorages[*slot.storage];
        if (result.typedControls[slot.id] || result.tapeValues[slot.id] || typedControlStorage)
            continue;
        result.layouts[slot.id] = resolvedProgramValueLayout(slot);
        const std::optional<size_t> staticBytes =
            result.dynamicShapes[slot.id] ? std::nullopt : programValueByteSize(slot);
        result.staticByteSizes[slot.id] = staticBytes;
        result.materializableStaticValues[slot.id] = staticBytes.has_value();
        if (!slot.storage)
            continue;
        auto backing = result.storageBackings.find(*slot.storage);
        if (backing == result.storageBackings.end())
            continue;
        if (backing->second.owner == UINT32_MAX || backing->second.owner >= execution.values.size())
            backing->second.owner = slot.id;
        if (staticBytes) {
            backing->second.bytes = std::max(backing->second.bytes, *staticBytes);
            backing->second.sized = true;
        }
    }
    return result;
}

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan,
                                  const ProgramInvocationPreparation &preparation,
                                  ProgramBoundaryBindingScratch &bindingScratch, std::vector<ProgramValueState> &values,
                                  std::map<uint32_t, ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const ForwardInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    if (program_execution::injectFailure(program_execution::FailureBoundary::Allocation))
        return error = "injected Program invocation allocation failure", false;
    return build(context, execution, plan, preparation, bindingScratch, values, storageBackings, tapeScratch,
                 {spec.requiredValues, &spec.bindings}, std::move(tapePolicy), error);
}

bool buildProgramInvocationValues(VernonRuntimeContext &context, const program::Program &execution,
                                  const program::ResolvedExecutionPlan *plan,
                                  const ProgramInvocationPreparation &preparation,
                                  ProgramBoundaryBindingScratch &bindingScratch, std::vector<ProgramValueState> &values,
                                  std::map<uint32_t, ProgramStorageBacking> &storageBackings,
                                  ProgramTapeScratch &tapeScratch, const PullbackInvocationSpec &spec,
                                  std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    return build(context, execution, plan, preparation, bindingScratch, values, storageBackings, tapeScratch,
                 {spec.requiredValues, &spec.bindings, &spec.captures, &spec.captureShapes, spec.tapeCaptures,
                  spec.retainedSnapshots},
                 std::move(tapePolicy), error);
}

} // namespace vernon::runtime::ad
