#include "runtime_autodiff_internal.h"

#include "execution_graph/execution_graph_checkpoint_planner_internal.h"
#include "host_tape_allocator.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {
namespace {

const ProgramGraph *findGraph(const ExecutableProgram &execution, const char *direction) {
    const auto found = std::find_if(execution.graphs.begin(), execution.graphs.end(),
                                    [&](const ProgramGraph &graph) { return graph.direction == direction; });
    return found == execution.graphs.end() ? nullptr : &*found;
}

const Parameter *findParameter(const Variant &variant, const std::string &name) {
    const auto external = std::find_if(variant.parameters.begin(), variant.parameters.end(),
                                       [&](const Parameter &parameter) { return parameter.name == name; });
    if (external != variant.parameters.end())
        return &*external;
    const auto internal = std::find_if(variant.internalParameters.begin(), variant.internalParameters.end(),
                                       [&](const Parameter &parameter) { return parameter.name == name; });
    return internal == variant.internalParameters.end() ? nullptr : &*internal;
}

std::optional<size_t> parameterByteSize(const Parameter &parameter, const ProgramValueSlot *slot = nullptr) {
    const ValueLayout &layout = slot && slot->valueLayout ? *slot->valueLayout
                                : parameter.valueLayout   ? *parameter.valueLayout
                                                          : parameter.elementLayout;
    if (!layout.byteSize)
        return std::nullopt;
    size_t result = layout.byteSize;
    const std::vector<uint64_t> &shape = slot ? slot->shape : parameter.shape;
    for (uint64_t extent : shape) {
        if (!extent || extent > std::numeric_limits<size_t>::max() / result)
            return std::nullopt;
        result *= static_cast<size_t>(extent);
    }
    return result;
}

std::optional<size_t> valueByteSize(const ProgramValueSlot &slot, const Parameter *parameter = nullptr) {
    if (slot.valueLayout && slot.valueLayout->byteSize) {
        Parameter synthesized;
        synthesized.kind = "tensor";
        synthesized.shape = slot.shape;
        synthesized.valueLayout = *slot.valueLayout;
        return parameterByteSize(synthesized, &slot);
    }
    return parameter ? parameterByteSize(*parameter, &slot) : std::nullopt;
}

Parameter parameterFromSlot(const ProgramValueSlot &slot) {
    Parameter parameter;
    parameter.name = slot.name;
    parameter.kind = "tensor";
    parameter.access = "read_write";
    parameter.shape = slot.shape;
    if (slot.valueLayout) {
        parameter.valueLayout = *slot.valueLayout;
        parameter.elementLayout = *slot.valueLayout;
        rebuildValueLayoutPathViews(*parameter.valueLayout);
        rebuildValueLayoutPathViews(parameter.elementLayout);
    }
    return parameter;
}

VernonValueLayoutView layoutView(const ValueLayout &layout) {
    return {sizeof(VernonValueLayoutView),
            layout.byteSize,
            layout.alignment,
            {layout.layoutHash.data(), layout.layoutHash.size()},
            layout.abiLeaves.empty() ? nullptr : layout.abiLeaves.data(),
            layout.abiLeaves.size()};
}

void rebuildVariantLayouts(Variant &variant) {
    const auto rebuild = [](Parameter &parameter) {
        if (parameter.valueLayout)
            rebuildValueLayoutPathViews(*parameter.valueLayout);
        rebuildValueLayoutPathViews(parameter.elementLayout);
    };
    for (Parameter &parameter : variant.parameters)
        rebuild(parameter);
    for (Parameter &parameter : variant.internalParameters)
        rebuild(parameter);
}

struct HostProgramValue {
    std::vector<uint8_t> owned;
    std::vector<int64_t> strides;
    std::vector<uint64_t> runtimeShape;
    VernonPipelineArgument argument{};
};

struct ProgramResidualPlan {
    execution::AutodiffDagCheckpointPlan checkpoint;
    std::vector<uint32_t> retainedValues;
    uint32_t replayEnd{};
};

bool planProgramResiduals(const ExecutableProgram &execution, const Variant &variant,
                          const std::vector<HostProgramValue> &materialized, uint64_t memoryBudget,
                          ProgramResidualPlan &result, std::string &error) {
    const ProgramGraph *forward = findGraph(execution, "forward");
    if (!forward)
        return error = "Program autodiff topology has no forward graph", false;
    std::vector<std::optional<uint32_t>> producers(execution.values.size());
    std::vector<execution::detail::AutodiffDagNode> nodes(forward->nodes.size());
    for (const ProgramNode &node : forward->nodes) {
        execution::detail::AutodiffDagNode &planned = nodes[node.id];
        planned.predecessors = node.dependencies;
        planned.replayable = true;
        planned.replayCost = 1;
        planned.recomputationCost = 1;
        for (uint32_t value : node.results)
            producers[value] = node.id;
    }
    uint64_t initialStateBytes = 0;
    const auto materializedBytes = [&](uint32_t value, const ProgramValueSlot &slot,
                                       const Parameter *parameter) -> std::optional<size_t> {
        if (value < materialized.size() && materialized[value].argument.tensor.byte_size)
            return materialized[value].argument.tensor.byte_size;
        return valueByteSize(slot, parameter);
    };
    for (uint32_t value : forward->arguments) {
        const ProgramValueSlot &slot = execution.values[value];
        const Parameter *parameter = findParameter(variant, slot.name);
        const std::optional<size_t> bytes = materializedBytes(value, slot, parameter);
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - initialStateBytes)
            return error = "Program autodiff initial state size overflows", false;
        initialStateBytes += *bytes;
    }
    for (uint32_t value : execution.backwardCaptures()) {
        if (!producers[value])
            continue;
        const ProgramValueSlot &slot = execution.values[value];
        const Parameter *parameter = findParameter(variant, slot.name);
        const std::optional<size_t> bytes = materializedBytes(value, slot, parameter);
        execution::detail::AutodiffDagNode &node = nodes[*producers[value]];
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - node.residualBytes)
            return error = "Program autodiff residual size overflows", false;
        node.residualBytes += *bytes;
        node.retainedAllocationBytes += *bytes;
        node.forwardPeakBytes += *bytes;
    }
    if (!execution::detail::planDagAutodiffCheckpoints(nodes, memoryBudget, result.checkpoint, error, initialStateBytes,
                                                       true, 0, true, 0, true, 0,
                                                       execution::detail::AutodiffCheckpointPolicy::Balanced))
        return false;
    const uint32_t retainBegin =
        result.checkpoint.replaySegments.empty() ? 0 : result.checkpoint.replaySegments.back().beginStep;
    for (uint32_t value : execution.backwardCaptures()) {
        if (!producers[value] || *producers[value] >= retainBegin)
            result.retainedValues.push_back(value);
        else
            result.replayEnd = std::max(result.replayEnd, *producers[value] + 1);
    }
    if (result.replayEnd)
        for (uint32_t value : forward->arguments)
            if (std::find(result.retainedValues.begin(), result.retainedValues.end(), value) ==
                result.retainedValues.end())
                result.retainedValues.push_back(value);
    std::sort(result.retainedValues.begin(), result.retainedValues.end());
    return true;
}

const ValueLayout *stableValueLayout(const ProgramValueSlot &slot, const Parameter *parameter) {
    if (slot.valueLayout)
        return slot.valueLayout.get();
    if (!parameter)
        return nullptr;
    return parameter->valueLayout ? &*parameter->valueLayout : &parameter->elementLayout;
}

bool fillHostTensor(HostProgramValue &value, const ProgramValueSlot &slot, const ValueLayout &layout, size_t byteSize,
                    void *hostData, std::string &error) {
    const std::vector<uint64_t> &shape = value.runtimeShape.empty() ? slot.shape : value.runtimeShape;
    value.strides.resize(shape.size());
    size_t stride = layout.byteSize;
    for (size_t index = shape.size(); index-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            error = "Program autodiff tensor stride overflows";
            return false;
        }
        value.strides[index] = static_cast<int64_t>(stride);
        const uint64_t extent = shape[index] ? shape[index] : 1;
        if (extent > std::numeric_limits<size_t>::max() / stride) {
            error = "Program autodiff tensor footprint overflows";
            return false;
        }
        stride *= static_cast<size_t>(extent);
    }
    value.argument.kind = VERNON_PIPELINE_TENSOR;
    value.argument.tensor.struct_size = sizeof(VernonTensorView);
    value.argument.tensor.storage = VERNON_TENSOR_HOST;
    value.argument.tensor.host_data = hostData;
    value.argument.tensor.element_layout = layoutView(layout);
    value.argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    value.argument.tensor.rank = static_cast<uint32_t>(shape.size());
    value.argument.tensor.shape = shape.empty() ? nullptr : shape.data();
    value.argument.tensor.byte_strides = value.strides.empty() ? nullptr : value.strides.data();
    value.argument.tensor.byte_size = byteSize;
    return true;
}

struct StorageBacking {
    uint32_t owner{UINT32_MAX};
    size_t bytes{};
    bool sized{};
};

bool dynamicShape(const std::vector<uint64_t> &shape) {
    return std::any_of(shape.begin(), shape.end(), [](uint64_t extent) { return extent == 0; });
}

bool programValueMatches(const VernonAdValue &value, const ValueAbi &abi) {
    if (value.dtype != abi.dtype || !value.data || value.rank != abi.logicalShape.size() ||
        (value.rank && !value.shape))
        return false;
    bool dynamic = false;
    for (size_t index = 0; index < abi.logicalShape.size(); ++index) {
        if (!abi.logicalShape[index]) {
            dynamic = true;
            continue;
        }
        if (value.shape[index] != abi.logicalShape[index])
            return false;
    }
    return dynamic ? value.size > 0 : value.size == abi.byteSize;
}

bool applyBoundShape(HostProgramValue &host, StorageBacking *backing, const VernonAdValue &value,
                     const ProgramValueSlot &slot, const ValueLayout &layout, std::string &error) {
    if (value.rank != slot.shape.size()) {
        error = "Program autodiff bound value rank does not match its Program type";
        return false;
    }
    host.runtimeShape.assign(value.shape, value.shape + value.rank);
    if (backing) {
        if (value.size > backing->bytes)
            backing->bytes = value.size;
        backing->sized = true;
    }
    (void)layout;
    return true;
}

bool evaluateOwnedExtents(const ProgramStorageSlot &storage, const std::vector<HostProgramValue> &hosts,
                          const std::vector<const ValueLayout *> &layouts, size_t &bytes, std::vector<uint64_t> &shape,
                          std::string &error) {
    bytes = 0;
    shape.clear();
    const uint32_t owner = storage.initialValue;
    if (owner >= layouts.size() || !layouts[owner] || !layouts[owner]->byteSize) {
        error = "owned dyn Storage has no element layout";
        return false;
    }
    uint64_t product = 1;
    shape.reserve(storage.byteLengthExtents.size());
    for (const ProgramBufferExtent &extent : storage.byteLengthExtents) {
        uint64_t length = extent.staticValue;
        if (!extent.isStatic) {
            if (extent.value >= hosts.size() || hosts[extent.value].runtimeShape.size() <= extent.axis ||
                !hosts[extent.value].runtimeShape[extent.axis]) {
                error = "owned dyn like-source dimension is unavailable at invocation bind";
                return false;
            }
            length = hosts[extent.value].runtimeShape[extent.axis];
        }
        if (!length || product > std::numeric_limits<uint64_t>::max() / length) {
            error = "owned dyn Storage extent overflows";
            return false;
        }
        product *= length;
        shape.push_back(length);
    }
    if (layouts[owner]->byteSize > std::numeric_limits<uint64_t>::max() / product) {
        error = "owned dyn Storage byte length overflows";
        return false;
    }
    bytes = static_cast<size_t>(layouts[owner]->byteSize * product);
    return bytes != 0;
}

struct ProgramLeafBinding {
    uint32_t value{UINT32_MAX};
    size_t byteOffset{};
    size_t elementStride{};
    size_t leafElementBytes{};
    size_t elementCount{};
};

bool materializeValues(const ExecutableProgram &execution, const Variant &variant,
                       std::vector<HostProgramValue> &storage, std::vector<VernonPipelineArgument> &arguments,
                       const VernonAdValueSet *boundLeaves, const std::vector<ValueAbi> *boundSignature,
                       const std::vector<ProgramLeafBinding> *boundBindings,
                       const std::vector<std::vector<uint8_t>> *captures,
                       const std::vector<std::vector<uint64_t>> *captureShapes, std::string &error) {
    storage.assign(execution.values.size(), {});
    arguments.resize(execution.values.size());
    std::map<uint32_t, StorageBacking> backings;
    for (const ProgramStorageSlot &slot : execution.storages) {
        StorageBacking backing;
        backing.owner = slot.initialValue;
        backing.bytes = static_cast<size_t>(slot.byteLength);
        backing.sized = slot.byteLength > 0;
        backings.emplace(slot.id, backing);
    }
    std::vector<const ValueLayout *> layouts(execution.values.size());
    for (const ProgramValueSlot &slot : execution.values) {
        const Parameter *parameter = findParameter(variant, slot.name);
        const ValueLayout *layout = stableValueLayout(slot, parameter);
        const bool dyn = dynamicShape(slot.shape);
        if ((!parameter && !slot.valueLayout) || (parameter && parameter->kind != "tensor" && !slot.valueLayout) ||
            !layout || !layout->byteSize || (!dyn && !valueByteSize(slot, parameter))) {
            error = "Program autodiff value has no materializable tensor parameter";
            return false;
        }
        layouts[slot.id] = layout;
        if (!slot.storage)
            continue;
        StorageBacking &backing = backings[*slot.storage];
        if (backing.owner == UINT32_MAX || backing.owner >= execution.values.size())
            backing.owner = slot.id;
        if (!dyn) {
            const std::optional<size_t> bytes = valueByteSize(slot, parameter);
            if (bytes && *bytes > backing.bytes)
                backing.bytes = *bytes;
            backing.sized = true;
        }
    }
    if (boundLeaves && boundSignature && boundBindings) {
        if (boundBindings->size() != boundSignature->size() || boundLeaves->value_count != boundSignature->size()) {
            error = "Program autodiff leaf set does not match its canonical boundary";
            return false;
        }
        for (size_t index = 0; index < boundBindings->size(); ++index) {
            const ProgramLeafBinding &binding = (*boundBindings)[index];
            const VernonAdValue *value = findValue(*boundLeaves, (*boundSignature)[index].path);
            if (!value || binding.value >= storage.size() || !programValueMatches(*value, (*boundSignature)[index])) {
                error = "Program autodiff leaf does not match graph reflection";
                return false;
            }
            const ProgramValueSlot &slot = execution.values[binding.value];
            StorageBacking *backing = slot.storage ? &backings[*slot.storage] : nullptr;
            if (!applyBoundShape(storage[binding.value], backing, *value, slot, *layouts[binding.value], error))
                return false;
        }
    }
    if (captures && captureShapes) {
        if (captures->size() != execution.values.size() || captureShapes->size() != execution.values.size()) {
            error = "Program autodiff capture state is incomplete";
            return false;
        }
        for (uint32_t value : execution.backwardCaptures()) {
            if (value >= storage.size() || (*captures)[value].empty())
                continue;
            storage[value].runtimeShape = (*captureShapes)[value];
            const ProgramValueSlot &slot = execution.values[value];
            if (!slot.storage)
                continue;
            StorageBacking &backing = backings[*slot.storage];
            if ((*captures)[value].size() > backing.bytes)
                backing.bytes = (*captures)[value].size();
            backing.sized = true;
        }
    }
    for (const ProgramStorageSlot &slot : execution.storages) {
        StorageBacking &backing = backings[slot.id];
        if (backing.sized)
            continue;
        if (slot.byteLengthExtents.empty()) {
            error = "Program autodiff storage has no materializable backing";
            return false;
        }
        size_t bytes = 0;
        std::vector<uint64_t> shape;
        if (!evaluateOwnedExtents(slot, storage, layouts, bytes, shape, error)) {
            if (captures)
                return false;
            error.clear();
            backing.bytes = 1;
            backing.sized = true;
            continue;
        }
        backing.bytes = bytes;
        backing.sized = true;
        if (backing.owner < storage.size())
            storage[backing.owner].runtimeShape = shape;
    }
    for (auto &[storageId, backing] : backings) {
        if (backing.owner >= storage.size() || !backing.bytes) {
            error = "Program autodiff storage has no materializable backing";
            return false;
        }
        storage[backing.owner].owned.resize(backing.bytes);
    }
    for (const ProgramValueSlot &slot : execution.values) {
        HostProgramValue &value = storage[slot.id];
        void *hostData = nullptr;
        size_t byteSize = 0;
        if (slot.storage) {
            const StorageBacking &backing = backings[*slot.storage];
            hostData = storage[backing.owner].owned.data();
            byteSize = backing.bytes;
            if (value.runtimeShape.empty())
                value.runtimeShape = storage[backing.owner].runtimeShape;
        } else if (!dynamicShape(slot.shape)) {
            const std::optional<size_t> bytes = valueByteSize(slot, findParameter(variant, slot.name));
            if (!bytes) {
                error = "Program autodiff value has no materializable tensor parameter";
                return false;
            }
            value.owned.resize(*bytes);
            hostData = value.owned.data();
            byteSize = *bytes;
        } else {
            value.owned.resize(1);
            hostData = value.owned.data();
            byteSize = 1;
        }
        if (!fillHostTensor(value, slot, *layouts[slot.id], byteSize, hostData, error))
            return false;
        arguments[slot.id] = value.argument;
    }
    return true;
}

bool transferLeaves(const VernonAdValueSet &supplied, const std::vector<ValueAbi> &signature,
                    const std::vector<ProgramLeafBinding> &bindings, std::vector<HostProgramValue> &storage,
                    bool publish, std::string &error) {
    if (bindings.size() != signature.size() || supplied.value_count != signature.size())
        return error = "Program autodiff leaf set does not match its canonical boundary", false;
    for (size_t index = 0; index < bindings.size(); ++index) {
        const ValueAbi &abi = signature[index];
        const VernonAdValue *value = findValue(supplied, abi.path);
        const ProgramLeafBinding &binding = bindings[index];
        if (!value || !programValueMatches(*value, abi) || binding.value >= storage.size())
            return error = "Program autodiff leaf does not match graph reflection", false;
        auto *packed = static_cast<uint8_t *>(const_cast<void *>(storage[binding.value].argument.tensor.host_data));
        auto *leaf = static_cast<uint8_t *>(const_cast<void *>(value->data));
        size_t elementCount = binding.elementCount;
        if (!elementCount) {
            const size_t bytes = publish ? storage[binding.value].argument.tensor.byte_size : value->size;
            if (!binding.elementStride || bytes % binding.elementStride)
                return error = "Program autodiff leaf does not match its dynamic footprint", false;
            elementCount = bytes / binding.elementStride;
        }
        for (size_t element = 0; element < elementCount; ++element) {
            uint8_t *packedElement = packed + element * binding.elementStride + binding.byteOffset;
            uint8_t *leafElement = leaf + element * binding.leafElementBytes;
            if (publish)
                std::memcpy(leafElement, packedElement, binding.leafElementBytes);
            else
                std::memcpy(packedElement, leafElement, binding.leafElementBytes);
        }
    }
    return true;
}

VernonStatus fail(VernonRuntimeContext &context, const std::string &error) {
    invocationDiagnostic(context) = error;
    return VERNON_STATUS_INVALID_ARGUMENT;
}

class ProgramPullback final : public PullbackExecution {
public:
    ProgramPullback(VernonRuntimeContext &context, std::shared_ptr<VernonPipelineTopology> topology, Variant variant,
                    Signature signature, std::vector<ProgramLeafBinding> cotangentBindings,
                    std::vector<ProgramLeafBinding> gradientBindings, ProgramResidualPlan plan,
                    std::vector<std::vector<uint8_t>> residuals, std::vector<std::vector<uint64_t>> residualShapes)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)),
          signature_(std::move(signature)), cotangentBindings_(std::move(cotangentBindings)),
          gradientBindings_(std::move(gradientBindings)), plan_(std::move(plan)), residuals_(std::move(residuals)),
          residualShapes_(std::move(residualShapes)) {
        rebuildVariantLayouts(variant_);
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        const ProgramGraph *backward = findGraph(topology_->execution, "backward");
        if (!backward || (!backward->arguments.empty() && !cotangents))
            return fail(*context_, "Program pullback has no backward graph or required cotangents");
        std::vector<HostProgramValue> storage;
        std::vector<VernonPipelineArgument> values;
        std::string error;
        if (!materializeValues(topology_->execution, variant_, storage, values, cotangents,
                               cotangents ? &signature_.cotangents : nullptr,
                               cotangents ? &cotangentBindings_ : nullptr, &residuals_, &residualShapes_, error))
            return fail(*context_, error);
        if (cotangents &&
            !transferLeaves(*cotangents, signature_.cotangents, cotangentBindings_, storage, false, error))
            return fail(*context_, error);
        size_t temporaryBytes = 0;
        for (const HostProgramValue &value : storage) {
            if (value.owned.size() > std::numeric_limits<size_t>::max() - temporaryBytes)
                return fail(*context_, "Program pullback temporary memory accounting overflows");
            temporaryBytes += value.owned.size();
        }
        if (temporaryBytes > options.maximumTemporaryBytes)
            return fail(*context_, "Program pullback exceeds its temporary memory limit");
        for (uint32_t value : plan_.retainedValues) {
            if (value >= residuals_.size() || residuals_[value].size() != values[value].tensor.byte_size)
                return fail(*context_, "Program pullback residual state is incomplete");
            std::memcpy(const_cast<void *>(storage[value].argument.tensor.host_data), residuals_[value].data(),
                        residuals_[value].size());
        }
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology_;
        if (plan_.replayEnd) {
            const ProgramGraph *forward = findGraph(topology_->execution, "forward");
            ProgramGraph replay = *forward;
            replay.nodes.resize(plan_.replayEnd);
            const VernonStatus replayStatus = executePipelineProgramGraph(proxy, replay, values);
            if (replayStatus != VERNON_STATUS_OK)
                return replayStatus;
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *backward, values);
        if (status == VERNON_STATUS_OK &&
            transferLeaves(gradients, signature_.gradients, gradientBindings_, storage, true, error)) {
            ++usage_.submissions;
            ++usage_.waits;
            ++usage_.atomicPublications;
            usage_.temporaryAllocationBytes += temporaryBytes;
            return status;
        }
        return status == VERNON_STATUS_OK ? fail(*context_, error) : status;
    }

    PullbackMemoryUsage memoryUsage() const override {
        size_t bytes = 0;
        for (const std::vector<uint8_t> &residual : residuals_)
            bytes += residual.size();
        return {plan_.checkpoint.logicalResidualBytes, bytes, bytes, bytes, 0};
    }

    PullbackControlPlaneUsage controlPlaneUsage() const override { return usage_; }

private:
    VernonRuntimeContext *context_;
    std::shared_ptr<VernonPipelineTopology> topology_;
    Variant variant_;
    Signature signature_;
    std::vector<ProgramLeafBinding> cotangentBindings_;
    std::vector<ProgramLeafBinding> gradientBindings_;
    ProgramResidualPlan plan_;
    std::vector<std::vector<uint8_t>> residuals_;
    std::vector<std::vector<uint64_t>> residualShapes_;
    PullbackControlPlaneUsage usage_;
};

class ProgramExecutable final : public Executable {
public:
    ProgramExecutable(VernonRuntimeContext &context, std::weak_ptr<VernonPipelineTopology> topology, Variant variant)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)) {
        rebuildVariantLayouts(variant_);
        const std::shared_ptr<VernonPipelineTopology> locked = topology_.lock();
        const auto append = [&](const std::vector<ProgramAdSignatureBinding> &bindings, std::vector<ValueAbi> &values,
                                std::vector<ProgramLeafBinding> &leafBindings) {
            for (const ProgramAdSignatureBinding &binding : bindings) {
                const ProgramValueSlot &slot = locked->execution.values[binding.value];
                const Parameter *parameter = findParameter(variant_, slot.name);
                std::string error;
                Parameter linked = parameter ? *parameter : parameterFromSlot(slot);
                linked.shape = slot.shape;
                if (slot.valueLayout)
                    linked.valueLayout = *slot.valueLayout;
                if (parameter || slot.valueLayout) {
                    const ValueLayout &layout = linked.valueLayout ? *linked.valueLayout : linked.elementLayout;
                    const size_t begin = values.size();
                    if (!appendParameterValueAbi(linked, binding.path, values, error)) {
                        signatureError_ = std::move(error);
                        continue;
                    }
                    size_t elementCount = 1;
                    for (uint64_t extent : linked.shape)
                        elementCount *= static_cast<size_t>(extent);
                    if (values.size() - begin != layout.leaves.size()) {
                        signatureError_ = "Program aggregate ABI contains duplicate leaf paths";
                        continue;
                    }
                    for (const ValueLeaf &leaf : layout.leaves) {
                        const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
                        leafBindings.push_back({binding.value, leaf.byteOffset, layout.byteSize,
                                                dtype ? dtypeSize(*dtype) * static_cast<size_t>(leaf.scalarCount) : 0,
                                                elementCount});
                    }
                } else {
                    signatureError_ = "Program signature value has no linked canonical parameter";
                }
            }
        };
        append(locked->execution.adSignature.inputs, signature_.inputs, inputBindings_);
        append(locked->execution.adSignature.outputs, signature_.outputs, outputBindings_);
        append(locked->execution.adSignature.cotangents, signature_.cotangents, cotangentBindings_);
        append(locked->execution.adSignature.gradients, signature_.gradients, gradientBindings_);
        const auto sortBoundary = [](std::vector<ValueAbi> &values, std::vector<ProgramLeafBinding> &bindings) {
            std::vector<size_t> order(values.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](size_t left, size_t right) { return values[left].path < values[right].path; });
            std::vector<ValueAbi> sortedValues;
            std::vector<ProgramLeafBinding> sortedBindings;
            sortedValues.reserve(order.size());
            sortedBindings.reserve(order.size());
            for (size_t index : order) {
                sortedValues.push_back(std::move(values[index]));
                sortedBindings.push_back(bindings[index]);
            }
            values = std::move(sortedValues);
            bindings = std::move(sortedBindings);
        };
        sortBoundary(signature_.inputs, inputBindings_);
        sortBoundary(signature_.outputs, outputBindings_);
        sortBoundary(signature_.cotangents, cotangentBindings_);
        sortBoundary(signature_.gradients, gradientBindings_);
    }

    const Signature &signature() const override { return signature_; }

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize, const VernonAdValueSet &inputs,
                         VernonAdValueSet *outputs, std::unique_ptr<PullbackExecution> &pullback) override {
        const std::shared_ptr<VernonPipelineTopology> topology = topology_.lock();
        const ProgramGraph *forward = topology ? findGraph(topology->execution, "forward") : nullptr;
        if (!signatureError_.empty())
            return fail(*context_, signatureError_);
        if (!topology || !forward || !outputs || target.encodedInvocation())
            return fail(*context_, "Program autodiff forward requires a live pipeline and host API values");
        std::vector<HostProgramValue> storage;
        std::vector<VernonPipelineArgument> values;
        std::string error;
        if (!materializeValues(topology->execution, variant_, storage, values, &inputs, &signature_.inputs,
                               &inputBindings_, nullptr, nullptr, error) ||
            !transferLeaves(inputs, signature_.inputs, inputBindings_, storage, false, error))
            return fail(*context_, error);
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology;
        const VernonStatus status = executePipelineProgramGraph(proxy, *forward, values);
        if (status != VERNON_STATUS_OK)
            return status;
        if (!transferLeaves(*outputs, signature_.outputs, outputBindings_, storage, true, error))
            return fail(*context_, error);
        ProgramResidualPlan plan;
        if (!planProgramResiduals(topology->execution, variant_, storage,
                                  context_->autodiffMemoryPolicy->invocationLimit(), plan, error))
            return fail(*context_, error);
        std::vector<std::vector<uint8_t>> residuals(topology->execution.values.size());
        std::vector<std::vector<uint64_t>> residualShapes(topology->execution.values.size());
        for (uint32_t value : topology->execution.backwardCaptures())
            residualShapes[value] = storage[value].runtimeShape;
        for (uint32_t value : plan.retainedValues) {
            const HostProgramValue &slot = storage[value];
            const auto *data = static_cast<const uint8_t *>(slot.argument.tensor.host_data);
            residuals[value].assign(data, data + slot.argument.tensor.byte_size);
            residualShapes[value] = slot.runtimeShape;
        }
        pullback = std::make_unique<ProgramPullback>(*context_, topology, variant_, signature_, cotangentBindings_,
                                                     gradientBindings_, std::move(plan), std::move(residuals),
                                                     std::move(residualShapes));
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext *context_;
    std::weak_ptr<VernonPipelineTopology> topology_;
    Variant variant_;
    Signature signature_;
    std::vector<ProgramLeafBinding> inputBindings_;
    std::vector<ProgramLeafBinding> outputBindings_;
    std::vector<ProgramLeafBinding> cotangentBindings_;
    std::vector<ProgramLeafBinding> gradientBindings_;
    std::string signatureError_;
};

} // namespace

bool resolveProgramAutodiff(VernonLoadedPipeline &pipeline,
                            const std::vector<AutodiffDerivativeGroup> &derivativeGroups) {
    if (!pipeline.context || !pipeline.topology)
        return false;
    if (pipeline.context->backend != VERNON_RUNTIME_CPU) {
        invocationDiagnostic(*pipeline.context) = "Program autodiff currently requires the CPU backend";
        return false;
    }
    if (!findGraph(pipeline.topology->execution, "forward") || !findGraph(pipeline.topology->execution, "backward")) {
        invocationDiagnostic(*pipeline.context) = "Program autodiff topology requires forward and backward graphs";
        return false;
    }
    const auto physicallyEquivalent = [](const ValueLayout &left, const ValueLayout &right) {
        if (left.byteSize != right.byteSize || left.alignment != right.alignment ||
            left.leaves.size() != right.leaves.size())
            return false;
        for (size_t index = 0; index < left.leaves.size(); ++index) {
            const ValueLeaf &a = left.leaves[index];
            const ValueLeaf &b = right.leaves[index];
            if (a.dtype != b.dtype || a.scalarCount != b.scalarCount || a.byteOffset != b.byteOffset ||
                a.shape != b.shape)
                return false;
        }
        return true;
    };
    for (const ProgramValueSlot &value : pipeline.topology->execution.values) {
        const Parameter *parameter = findParameter(pipeline.variant, value.name);
        const ValueLayout *parameterLayout =
            parameter ? (parameter->valueLayout ? &*parameter->valueLayout : &parameter->elementLayout) : nullptr;
        const ValueLayout *valueLayout = value.valueLayout ? value.valueLayout.get() : parameterLayout;
        if (!valueLayout || (!dynamicShape(value.shape) && !valueByteSize(value, parameter)) ||
            (value.valueLayout && !value.valueLayout->layoutHash.empty() && parameterLayout &&
             value.valueLayout->layoutHash != parameterLayout->layoutHash &&
             !physicallyEquivalent(*value.valueLayout, *parameterLayout))) {
            invocationDiagnostic(*pipeline.context) = "Program autodiff topology has an unsupported value ABI";
            return false;
        }
    }
    if (!pipeline.context->autodiffMemoryPolicy)
        pipeline.context->autodiffMemoryPolicy = std::make_shared<AutodiffMemoryPolicy>();
    auto executable = std::make_shared<ProgramExecutable>(*pipeline.context, pipeline.topology, pipeline.variant);
    std::vector<AutodiffDerivativeGroup> groups = derivativeGroups;
    const auto appendGroups = [](AutodiffDerivativeRole role, const std::vector<ProgramAdSignatureBinding> &declared,
                                 const std::vector<ValueAbi> &leaves, std::vector<AutodiffDerivativeGroup> &result) {
        for (const ProgramAdSignatureBinding &binding : declared) {
            AutodiffDerivativeGroup group{role, binding.path, {}};
            for (const ValueAbi &leaf : leaves)
                if (leaf.path == binding.path || (leaf.path.size() > binding.path.size() &&
                                                  leaf.path.compare(0, binding.path.size(), binding.path) == 0 &&
                                                  leaf.path[binding.path.size()] == '.'))
                    group.leafPaths.push_back(leaf.path);
            std::sort(group.leafPaths.begin(), group.leafPaths.end());
            result.push_back(std::move(group));
        }
    };
    if (groups.empty()) {
        appendGroups(AutodiffDerivativeRole::Gradient, pipeline.topology->execution.adSignature.gradients,
                     executable->signature().gradients, groups);
        std::sort(groups.begin(), groups.end(),
                  [](const AutodiffDerivativeGroup &left, const AutodiffDerivativeGroup &right) {
                      return left.declaredPath < right.declaredPath;
                  });
        const size_t gradientCount = groups.size();
        appendGroups(AutodiffDerivativeRole::Cotangent, pipeline.topology->execution.adSignature.cotangents,
                     executable->signature().cotangents, groups);
        std::sort(groups.begin() + static_cast<std::ptrdiff_t>(gradientCount), groups.end(),
                  [](const AutodiffDerivativeGroup &left, const AutodiffDerivativeGroup &right) {
                      return left.declaredPath < right.declaredPath;
                  });
    }
    std::string groupError;
    const bool completeDerivativeBoundary =
        !executable->signature().cotangents.empty() && !executable->signature().gradients.empty();
    const bool validGroups = !completeDerivativeBoundary || validateAutodiffDerivativeGroups(groups, groupError);
    const bool validSignature = !completeDerivativeBoundary || validateDerivativeGroupsAgainstSignature(
                                                                   *pipeline.context, groups, executable->signature());
    if (!validGroups || !validSignature) {
        if (!groupError.empty())
            invocationDiagnostic(*pipeline.context) = std::move(groupError);
        else if (invocationDiagnostic(*pipeline.context).empty())
            invocationDiagnostic(*pipeline.context) = "Program autodiff derivative groups do not match its signature";
        return false;
    }
    pipeline.topology->differentiated = VernonDifferentiatedPipeline{std::move(executable), std::move(groups)};
    return true;
}

} // namespace vernon::runtime::ad
