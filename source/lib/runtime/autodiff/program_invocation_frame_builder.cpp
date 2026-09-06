#include "program_invocation_frame_builder.h"

#include "program_boundary_binder.h"
#include "program_publication.h"
#include "program_value_materializer.h"
#include "runtime_autodiff_internal.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>
#include <set>

namespace vernon::runtime::ad {
namespace {

struct FrameRequest {
    const std::vector<char> &required;
    const CanonicalForwardBindings *canonical{};
    ProgramLeafFrameSource bound;
    ProgramLeafFrameSource results;
    const std::vector<std::vector<uint8_t>> *captures{};
    const std::vector<std::vector<uint64_t>> *captureShapes{};
    const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures{};
    const LogicalValueFrame *retainedValues{};
};

bool applyBoundShape(LogicalProgramValue &host, ProgramStorageState *backing, const VernonAdValue &value,
                     const program::Value &slot, std::string &error) {
    if (value.rank < slot.shape.size() || (value.rank && !value.shape))
        return error = "Program autodiff bound value rank does not match its Program type", false;
    shape::ConcreteShape concrete(value.shape, value.shape + slot.shape.size());
    if (!shape::matches(shape::decodeRuntimeContractShape(slot.shape), concrete))
        return error = "Program autodiff bound value shape does not match its Program type", false;
    host.concreteShape = std::move(concrete);
    if (backing) {
        backing->bytes = std::max(backing->bytes, value.size);
        backing->sized = true;
    }
    return true;
}

bool applyLeaves(const ProgramLeafFrameSource &source, const program::Program &execution,
                 const std::vector<std::optional<ValueLayout>> &layouts, std::vector<LogicalProgramValue> &storage,
                 std::map<uint32_t, ProgramStorageState> &backings, std::string &error) {
    if (!source.values && !source.signature && !source.bindings)
        return true;
    if (!source.values || !source.signature || !source.bindings ||
        source.bindings->size() != source.signature->size() || source.values->value_count != source.signature->size())
        return error = "Program autodiff leaf set does not match its canonical boundary", false;
    for (size_t index = 0; index < source.bindings->size(); ++index) {
        const ProgramLeafBinding &binding = (*source.bindings)[index];
        const ValueAbi &abi = (*source.signature)[index];
        const VernonAdValue *value = findValue(*source.values, abi.path);
        if (!value)
            return error = "Program autodiff leaf '" + abi.path + "' is absent from the supplied frame", false;
        if (binding.value >= storage.size())
            return error = "Program autodiff leaf '" + abi.path + "' exceeds the graph value frame", false;
        if (!matchesProgramValueAbi(*value, abi))
            return error = "Program autodiff leaf '" + abi.path + "' does not match graph reflection (dtype " +
                           std::to_string(value->dtype) + ", rank " + std::to_string(value->rank) + ", bytes " +
                           std::to_string(value->size) + ")",
                   false;
        const program::Value &slot = execution.values[binding.value];
        ProgramStorageState *backing = slot.storage ? &backings[*slot.storage] : nullptr;
        if (!layouts[binding.value] || !applyBoundShape(storage[binding.value], backing, *value, slot, error))
            return false;
        if (backing && backing->owner < storage.size() && !storage[backing->owner].concreteShape)
            storage[backing->owner].concreteShape = storage[binding.value].concreteShape;
    }
    return true;
}

bool attachResiduals(const FrameRequest &request, const program::Program &execution,
                     std::vector<LogicalProgramValue> &storage, std::map<uint32_t, ProgramStorageState> &backings,
                     std::string &error) {
    if (!request.captures && !request.captureShapes)
        return true;
    if (!request.captures || !request.captureShapes || request.captures->size() != execution.values.size() ||
        request.captureShapes->size() != execution.values.size())
        return error = "Program autodiff capture state is incomplete", false;
    for (size_t value = 0; value < request.captures->size(); ++value) {
        if (!(*request.captureShapes)[value].empty())
            storage[value].concreteShape = (*request.captureShapes)[value];
        if ((*request.captures)[value].empty())
            continue;
        storage[value].ownership = ProgramValueOwnership::RetainedResidual;
        const program::Value &slot = execution.values[value];
        if (!program::isTapeValueType(slot.type) && slot.storage) {
            ProgramStorageState &backing = backings[*slot.storage];
            backing.bytes = std::max(backing.bytes, (*request.captures)[value].size());
            backing.sized = true;
        }
    }
    return true;
}

bool build(VernonRuntimeContext &context, const program::Program &execution, const VernonProgramTopology *topology,
           std::vector<LogicalProgramValue> &storage, const FrameRequest &request,
           std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    storage.assign(execution.values.size(), {});
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
    const auto markBindings = [&](const ProgramLeafFrameSource &source) {
        if (source.bindings)
            for (const ProgramLeafBinding &binding : *source.bindings)
                if (binding.value < live.size())
                    live[binding.value] = 1;
    };
    markBindings(request.bound);
    markBindings(request.results);
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

    std::map<uint32_t, ProgramStorageState> backings;
    for (const program::Storage &slot : execution.storages)
        backings.emplace(slot.id, ProgramStorageState{slot.initialValue, static_cast<size_t>(slot.buffer.byteLength),
                                                      slot.buffer.byteLength > 0, std::nullopt});
    std::map<uint32_t, VernonProgramArgument> externalValues;
    if (request.canonical) {
        ProgramBoundaryBindingRequest binding{request.canonical->invocation, request.canonical->valueBySlot,
                                              &request.canonical->publications};
        if (!bindProgramBoundaries(context, execution, binding, externalValues, backings, live, error) ||
            !applyProgramPublicationShapes(execution, request.canonical->publications, storage, error))
            return false;
        for (auto &[value, argument] : externalValues) {
            if (value >= execution.values.size() || execution.values[value].storage ||
                argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
                !argument.tensor.host_data || !argument.tensor.byte_size)
                continue;
            const auto *data = static_cast<const uint8_t *>(argument.tensor.host_data) + argument.tensor.byte_offset;
            storage[value].owned.assign(data, data + argument.tensor.byte_size);
            argument.tensor.host_data = storage[value].owned.data();
            argument.tensor.byte_offset = 0;
        }
    } else if (request.retainedValues) {
        const std::vector<LogicalProgramValue> &retained = request.retainedValues->hostValues();
        for (size_t value = 0; value < retained.size() && value < live.size(); ++value) {
            const VernonProgramArgument &argument = retained[value].argument;
            if (!live[value] || argument.kind != VERNON_PROGRAM_TENSOR ||
                argument.tensor.storage != VERNON_TENSOR_HOST || !argument.tensor.host_data)
                continue;
            externalValues.emplace(static_cast<uint32_t>(value), argument);
            storage[value].concreteShape = retained[value].concreteShape;
            storage[value].strides = retained[value].strides;
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
            storage[slot.id].argument = *argument;
            continue;
        }
        const bool dynamic = programValueHasDynamicShape(slot);
        if (!layouts[slot.id] || !layouts[slot.id]->byteSize || (!dynamic && !programValueByteSize(slot)))
            return error = "Program autodiff value '" + slot.name + "' (id " + std::to_string(slot.id) + ", type " +
                           slot.type + ") has no materializable tensor layout",
                   false;
        if (argument) {
            const VernonTensorView &tensor = argument->tensor;
            if (tensor.rank < slot.shape.size() || (tensor.rank && (!tensor.shape || !tensor.byte_strides)))
                return error = "Program invocation Tensor binding has incomplete shape metadata", false;
            storage[slot.id].concreteShape = shape::ConcreteShape();
            storage[slot.id].strides.clear();
            if (!slot.shape.empty()) {
                storage[slot.id].concreteShape->assign(tensor.shape, tensor.shape + slot.shape.size());
                storage[slot.id].strides.assign(tensor.byte_strides, tensor.byte_strides + slot.shape.size());
            }
        }
        if (!slot.storage)
            continue;
        ProgramStorageState &backing = backings[*slot.storage];
        if (backing.owner == UINT32_MAX || backing.owner >= execution.values.size())
            backing.owner = slot.id;
        if (!dynamic) {
            if (const std::optional<size_t> bytes = programValueByteSize(slot))
                backing.bytes = std::max(backing.bytes, *bytes);
            backing.sized = true;
        }
    }
    if (!applyLeaves(request.bound, execution, layouts, storage, backings, error) ||
        !applyLeaves(request.results, execution, layouts, storage, backings, error) ||
        !attachResiduals(request, execution, storage, backings, error) ||
        !materializeProgramOwnedStorages(execution, topology, storage, liveStorage, layouts, backings, error) ||
        !materializeProgramValues(execution, topology, storage, live, layouts, backings, externalValues,
                                  request.tapeCaptures, tapePolicy, error))
        return false;
    return true;
}

} // namespace

bool matchesProgramValueAbi(const VernonAdValue &value, const ValueAbi &abi) {
    if (value.dtype != abi.dtype || value.rank != abi.logicalShape.size() || (value.rank && !value.shape))
        return false;
    bool dynamic = false;
    bool empty = false;
    for (size_t index = 0; index < abi.logicalShape.size(); ++index) {
        empty |= value.shape[index] == 0;
        if (!abi.logicalShape[index])
            dynamic = true;
        else if (value.shape[index] != abi.logicalShape[index])
            return false;
    }
    if (!value.data && !empty)
        return false;
    return dynamic ? (empty ? value.size == 0 : value.size > 0) : value.size == abi.byteSize;
}

bool buildLogicalValueFrame(VernonRuntimeContext &context, const program::Program &execution,
                            const VernonProgramTopology *topology, std::vector<LogicalProgramValue> &storage,
                            const ForwardInvocationSpec &spec, std::shared_ptr<AutodiffMemoryPolicy> tapePolicy,
                            std::string &error) {
    if (const auto *canonical = std::get_if<CanonicalForwardBindings>(&spec.bindings))
        return build(context, execution, topology, storage, {spec.requiredValues, canonical}, std::move(tapePolicy),
                     error);
    const HostForwardBindings &host = std::get<HostForwardBindings>(spec.bindings);
    return build(context, execution, topology, storage, {spec.requiredValues, nullptr, host.inputs, host.outputs},
                 std::move(tapePolicy), error);
}

bool buildLogicalValueFrame(VernonRuntimeContext &context, const program::Program &execution,
                            const VernonProgramTopology *topology, std::vector<LogicalProgramValue> &storage,
                            const PullbackInvocationSpec &spec, std::shared_ptr<AutodiffMemoryPolicy> tapePolicy,
                            std::string &error) {
    return build(context, execution, topology, storage,
                 {spec.requiredValues, nullptr, spec.cotangents, spec.gradients, &spec.captures, &spec.captureShapes,
                  spec.tapeCaptures, spec.retainedValues},
                 std::move(tapePolicy), error);
}

} // namespace vernon::runtime::ad
