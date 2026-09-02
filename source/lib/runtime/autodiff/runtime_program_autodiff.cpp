#include "runtime_autodiff_internal.h"

#include "execution_graph/execution_graph_checkpoint_planner_internal.h"
#include "host_tape_allocator.h"
#include "program_shape_resolver.h"
#include "program_value_arena.h"
#include "runtime/program_manifest.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime_autodiff_memory_usage.h"
#include "runtime_gpu_argument_binding.h"
#include "runtime_gpu_replay.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
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

using HostProgramValue = ProgramHostValue;

bool fillTapeHostValue(HostProgramValue &value, std::shared_ptr<HostStaticTapeBatch> batch, std::string &error) {
    if (!batch) {
        error = "Program autodiff tape has no allocator batch";
        return false;
    }
    VernonAdTapeAllocator *descriptor = batch->descriptor(0);
    if (!descriptor) {
        error = "Program autodiff tape has no allocator descriptor";
        return false;
    }
    value.tapeBatch = std::move(batch);
    const VernonAdRegionHandle root = value.tapeBatch->rootRegion(0);
    value.owned.resize(sizeof(descriptor) + sizeof(root));
    std::memcpy(value.owned.data(), &descriptor, sizeof(descriptor));
    std::memcpy(value.owned.data() + sizeof(descriptor), &root, sizeof(root));
    value.argument = {};
    value.argument.kind = VERNON_PIPELINE_TENSOR;
    value.argument.tensor.struct_size = sizeof(VernonTensorView);
    value.argument.tensor.storage = VERNON_TENSOR_HOST;
    value.argument.tensor.host_data = value.owned.data();
    value.argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    value.argument.tensor.byte_size = value.owned.size();
    return true;
}

struct ProgramResidualPlan {
    execution::AutodiffDagCheckpointPlan checkpoint;
    std::vector<uint32_t> retainedValues;
    uint32_t replayEnd{};
};

execution::detail::AutodiffCheckpointPolicy programCheckpointPolicy(const std::string &name) {
    if (name == "min_memory")
        return execution::detail::AutodiffCheckpointPolicy::MinMemory;
    if (name == "min_runtime")
        return execution::detail::AutodiffCheckpointPolicy::MinRuntime;
    return execution::detail::AutodiffCheckpointPolicy::Balanced;
}

std::string programPassName(const ProgramNode &node) {
    std::string name = node.name;
    const std::string_view suffixes[] = {".forward_with_tape", ".vjp"};
    for (std::string_view suffix : suffixes)
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            name.resize(name.size() - suffix.size());
            break;
        }
    if (name.find('.') == std::string::npos)
        name.insert(0, "program.");
    return name;
}

std::vector<AutodiffPullbackPassTelemetry> collectProgramPassTelemetry(const ProgramGraph &forward,
                                                                       const ExecutableProgram &execution,
                                                                       const std::vector<HostProgramValue> &storage,
                                                                       const ProgramResidualPlan &plan) {
    std::vector<char> retained(execution.values.size());
    for (uint32_t value : plan.retainedValues)
        if (value < retained.size())
            retained[value] = 1;
    std::vector<char> captured(execution.values.size());
    for (uint32_t value : execution.residualCaptures())
        if (value < captured.size())
            captured[value] = 1;
    std::vector<AutodiffPullbackPassTelemetry> telemetry;
    telemetry.reserve(forward.nodes.size());
    for (const ProgramNode &node : forward.nodes) {
        AutodiffPullbackPassTelemetry item;
        item.scheduleOffset = node.id;
        item.passName = programPassName(node);
        uint64_t estimated = 0;
        uint64_t logical = 0;
        uint64_t resident = 0;
        uint64_t allocated = 0;
        uint64_t retainedBytes = 0;
        bool hasTape = false;
        bool keptTape = false;
        bool dynamicTape = false;
        bool hasControlHistory = false;
        bool hasTensorResidual = false;
        for (uint32_t value : node.results) {
            if (value >= storage.size())
                continue;
            const HostProgramValue &slot = storage[value];
            if (slot.tapeBatch) {
                hasTape = true;
                estimated += slot.tapeBatch->logicalBytes();
                dynamicTape |= slot.tapeBatch->hasDynamicLanes();
                hasControlHistory |= slot.tapeBatch->hasControlHistory();
                if (value < retained.size() && retained[value]) {
                    keptTape = true;
                    logical += slot.tapeBatch->logicalBytes();
                    const uint64_t residentBytes = slot.tapeBatch->isCompacted() ? slot.tapeBatch->logicalBytes()
                                                                                 : slot.tapeBatch->residentBytes();
                    resident += residentBytes;
                    allocated += slot.tapeBatch->allocatedBytes();
                    retainedBytes += slot.tapeBatch->allocatedBytes();
                }
            } else if (value < captured.size() && captured[value] && value < retained.size() && retained[value]) {
                hasTensorResidual = true;
                retainedBytes += slot.argument.tensor.byte_size;
            }
        }
        item.estimatedTapeBytes = estimated;
        item.logicalResidualBytes = hasControlHistory ? logical : 0;
        item.residentTapeBytes = hasControlHistory ? resident : 0;
        item.allocatedTapeBytes = hasControlHistory ? allocated : 0;
        item.retainedAllocationBytes = retainedBytes;
        if (hasTape) {
            const char *kind = dynamicTape ? "dynamic_capture" : "static_capture";
            item.residualSourceKind = keptTape ? kind : std::string(kind) + "+pure_rematerialization";
            item.controlHistoryKind = hasControlHistory && keptTape ? "dynamic_capture" : "none";
            item.peakTemporaryTapeBytes = std::max(allocated, estimated);
            if (!item.peakTemporaryTapeBytes)
                item.peakTemporaryTapeBytes = 1;
        } else if (hasTensorResidual) {
            item.residualSourceKind = "static_capture";
            item.controlHistoryKind = "none";
        }
        telemetry.push_back(std::move(item));
    }
    return telemetry;
}

bool planProgramResiduals(const ExecutableProgram &execution, const Variant &variant,
                          const std::vector<HostProgramValue> &materialized, uint64_t memoryBudget,
                          execution::detail::AutodiffCheckpointPolicy policy, bool rematerializeTapes,
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
        if (value < materialized.size() && materialized[value].tapeBatch)
            return std::max<size_t>(materialized[value].tapeBatch->logicalBytes(), 1);
        if (value < materialized.size() && materialized[value].argument.tensor.byte_size)
            return materialized[value].argument.tensor.byte_size;
        if (isProgramAdTapeType(slot.type))
            return 1;
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
    for (uint32_t value : execution.residualCaptures()) {
        if (!producers[value])
            continue;
        if (rematerializeTapes && isProgramAdTapeType(execution.values[value].type))
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
                                                       true, 0, true, 0, true, 0, policy))
        return false;
    const uint32_t retainBegin =
        result.checkpoint.replaySegments.empty() ? 0 : result.checkpoint.replaySegments.back().beginStep;
    for (uint32_t value : execution.residualCaptures()) {
        if (!producers[value])
            continue;
        if ((rematerializeTapes && isProgramAdTapeType(execution.values[value].type)) ||
            *producers[value] < retainBegin)
            result.replayEnd = std::max(result.replayEnd, *producers[value] + 1);
    }
    for (uint32_t value : execution.residualCaptures()) {
        if (!producers[value]) {
            result.retainedValues.push_back(value);
        } else if (*producers[value] < result.replayEnd) {
            continue;
        } else {
            result.retainedValues.push_back(value);
        }
    }
    std::sort(result.retainedValues.begin(), result.retainedValues.end());
    result.retainedValues.erase(std::unique(result.retainedValues.begin(), result.retainedValues.end()),
                                result.retainedValues.end());
    std::set<uint32_t> retainedStorages;
    for (uint32_t value : result.retainedValues)
        if (value < execution.values.size() && execution.values[value].storage)
            retainedStorages.insert(*execution.values[value].storage);
    uint64_t logicalResidualBytes = 0;
    for (uint32_t value : execution.residualCaptures()) {
        if (!producers[value])
            continue;
        const ProgramValueSlot &slot = execution.values[value];
        const bool retained = std::binary_search(result.retainedValues.begin(), result.retainedValues.end(), value) ||
                              (slot.storage && retainedStorages.find(*slot.storage) != retainedStorages.end());
        if (!retained)
            continue;
        const Parameter *parameter = findParameter(variant, slot.name);
        const std::optional<size_t> bytes = materializedBytes(value, slot, parameter);
        if (!bytes || *bytes > std::numeric_limits<uint64_t>::max() - logicalResidualBytes)
            return error = "Program autodiff logical residual size overflows", false;
        logicalResidualBytes += *bytes;
    }
    if (rematerializeTapes)
        result.checkpoint.logicalResidualBytes = logicalResidualBytes;
    if (result.replayEnd) {
        result.retainedValues.insert(result.retainedValues.end(), forward->arguments.begin(), forward->arguments.end());
        std::sort(result.retainedValues.begin(), result.retainedValues.end());
        result.retainedValues.erase(std::unique(result.retainedValues.begin(), result.retainedValues.end()),
                                    result.retainedValues.end());
    }
    return true;
}

bool sealProgramTapeValues(std::vector<HostProgramValue> &storage, std::vector<VernonPipelineArgument> &values,
                           std::string &error) {
    if (values.size() < storage.size())
        return error = "Program autodiff tape value arena is incomplete", false;
    for (size_t value = 0; value < storage.size(); ++value) {
        HostProgramValue &slot = storage[value];
        if (!slot.tapeBatch)
            continue;
        if (!slot.tapeBatch->compact(true))
            return error = "Program autodiff tape could not be compacted", false;
        if (!fillTapeHostValue(slot, slot.tapeBatch, error))
            return false;
        values[value] = slot.argument;
    }
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
    if (!value.concreteShape) {
        value.concreteShape = shape::concrete(shape::decodeRuntimeContractShape(slot.shape));
        if (!value.concreteShape) {
            error = "Program autodiff tensor shape was not resolved before allocation";
            return false;
        }
    }
    if (!shape::rowMajorByteStrides(*value.concreteShape, layout.byteSize, value.strides)) {
        error = "Program autodiff tensor layout overflows";
        return false;
    }
    value.argument.kind = VERNON_PIPELINE_TENSOR;
    value.argument.tensor.struct_size = sizeof(VernonTensorView);
    value.argument.tensor.storage = VERNON_TENSOR_HOST;
    value.argument.tensor.host_data = hostData;
    value.argument.tensor.element_layout = layoutView(layout);
    value.argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    value.argument.tensor.rank = static_cast<uint32_t>(value.concreteShape->size());
    value.argument.tensor.shape = value.concreteShape->empty() ? nullptr : value.concreteShape->data();
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
    return !shape::isConcrete(shape::decodeRuntimeContractShape(shape));
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
    if (value.rank < slot.shape.size() || (value.rank && !value.shape)) {
        error = "Program autodiff bound value rank does not match its Program type";
        return false;
    }
    // Host numpy may include trailing leaf axes (Vector/Matrix cells). Runtime extents
    // are the TensorView prefix; extra axes are payload layout, not like-source shape.
    shape::ConcreteShape concrete(value.shape, value.shape + slot.shape.size());
    if (!shape::matches(shape::decodeRuntimeContractShape(slot.shape), concrete)) {
        error = "Program autodiff bound value shape does not match its Program type";
        return false;
    }
    host.concreteShape = std::move(concrete);
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
            if (extent.value >= hosts.size() || !hosts[extent.value].concreteShape ||
                hosts[extent.value].concreteShape->size() <= extent.axis) {
                error = "owned dyn like-source dimension is unavailable at invocation bind";
                return false;
            }
            length = (*hosts[extent.value].concreteShape)[extent.axis];
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

bool applyProgramBoundLeaves(const VernonAdValueSet &leaves, const std::vector<ValueAbi> &signature,
                             const std::vector<ProgramLeafBinding> &bindings, const ExecutableProgram &execution,
                             const std::vector<const ValueLayout *> &layouts, std::vector<HostProgramValue> &storage,
                             std::map<uint32_t, StorageBacking> &backings, std::string &error) {
    if (bindings.size() != signature.size() || leaves.value_count != signature.size())
        return error = "Program autodiff leaf set does not match its canonical boundary", false;
    for (size_t index = 0; index < bindings.size(); ++index) {
        const ProgramLeafBinding &binding = bindings[index];
        const VernonAdValue *value = findValue(leaves, signature[index].path);
        if (!value || binding.value >= storage.size() || !programValueMatches(*value, signature[index])) {
            error = "Program autodiff leaf does not match graph reflection";
            return false;
        }
        const ProgramValueSlot &slot = execution.values[binding.value];
        StorageBacking *backing = slot.storage ? &backings[*slot.storage] : nullptr;
        if (!applyBoundShape(storage[binding.value], backing, *value, slot, *layouts[binding.value], error))
            return false;
        if (backing && backing->owner < storage.size() && !storage[backing->owner].concreteShape)
            storage[backing->owner].concreteShape = storage[binding.value].concreteShape;
    }
    return true;
}

bool programTapePayloadStride(const std::string &type, size_t &stride, std::string &error) {
    stride = 1;
    constexpr std::string_view prefix = "!vernon.ad_tape<";
    if (type.size() <= prefix.size() || type.compare(0, prefix.size(), prefix.data(), prefix.size()) != 0 ||
        type.back() != '>')
        return true;
    std::string_view digits = std::string_view(type).substr(prefix.size(), type.size() - prefix.size() - 1);
    uint64_t bytes = 0;
    const auto [end, parseError] = std::from_chars(digits.data(), digits.data() + digits.size(), bytes);
    if (parseError != std::errc{} || end != digits.data() + digits.size() || !bytes ||
        bytes > std::numeric_limits<size_t>::max()) {
        error = "Program autodiff tape type has an invalid payload stride";
        return false;
    }
    stride = static_cast<size_t>(bytes);
    return true;
}

bool programTapeLaneCount(const ExecutableProgram &execution, const VernonPipelineTopology *topology,
                          uint32_t tapeValue, size_t &lanes, std::string &error) {
    lanes = 1;
    for (const ProgramGraph &graph : execution.graphs) {
        for (const ProgramNode &node : graph.nodes) {
            const bool used = std::find(node.results.begin(), node.results.end(), tapeValue) != node.results.end() ||
                              std::find(node.operands.begin(), node.operands.end(), tapeValue) != node.operands.end();
            if (!used)
                continue;
            uint32_t workgroup[3]{256, 1, 1};
            if (topology) {
                const auto found = topology->stageIndices.find(node.stage);
                if (found != topology->stageIndices.end() && found->second < topology->stages.size()) {
                    const VernonLoadedPipeline *pipeline = topology->stages[found->second].pipeline.get();
                    if (pipeline) {
                        workgroup[0] = pipeline->workgroupSize.x ? pipeline->workgroupSize.x : 1;
                        workgroup[1] = pipeline->workgroupSize.y ? pipeline->workgroupSize.y : 1;
                        workgroup[2] = pipeline->workgroupSize.z ? pipeline->workgroupSize.z : 1;
                    }
                }
            }
            size_t volume = 1;
            for (int axis = 0; axis < 3; ++axis) {
                const size_t groups = node.grid[axis] ? static_cast<size_t>(node.grid[axis]) : 1;
                const size_t wg = workgroup[axis] ? workgroup[axis] : 1;
                if (groups > std::numeric_limits<size_t>::max() / wg ||
                    volume > std::numeric_limits<size_t>::max() / (groups * wg)) {
                    error = "Program autodiff tape dispatch volume overflows";
                    return false;
                }
                volume *= groups * wg;
            }
            lanes = std::max(lanes, volume);
        }
    }
    return true;
}

bool materializeValues(const ExecutableProgram &execution, const VernonPipelineTopology *topology,
                       const Variant &variant, std::vector<HostProgramValue> &storage,
                       std::vector<VernonPipelineArgument> &arguments, const std::vector<char> &requiredValues,
                       const VernonAdValueSet *boundLeaves, const std::vector<ValueAbi> *boundSignature,
                       const std::vector<ProgramLeafBinding> *boundBindings, const VernonAdValueSet *resultLeaves,
                       const std::vector<ValueAbi> *resultSignature,
                       const std::vector<ProgramLeafBinding> *resultBindings,
                       const std::vector<std::vector<uint8_t>> *captures,
                       const std::vector<std::vector<uint64_t>> *captureShapes,
                       const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures,
                       std::shared_ptr<AutodiffMemoryPolicy> tapePolicy, std::string &error) {
    storage.assign(execution.values.size(), {});
    arguments.resize(execution.values.size());
    std::vector<char> live = requiredValues;
    live.resize(execution.values.size());
    const auto mark = [&](uint32_t value) {
        if (value < live.size())
            live[value] = 1;
    };
    const auto markBindings = [&](const std::vector<ProgramLeafBinding> *bindings) {
        if (!bindings)
            return;
        for (const ProgramLeafBinding &binding : *bindings)
            mark(binding.value);
    };
    markBindings(boundBindings);
    markBindings(resultBindings);
    if (captures)
        for (size_t value = 0; value < captures->size() && value < live.size(); ++value)
            if (!(*captures)[value].empty())
                mark(static_cast<uint32_t>(value));
    if (tapeCaptures)
        for (size_t value = 0; value < tapeCaptures->size() && value < live.size(); ++value)
            if ((*tapeCaptures)[value])
                mark(static_cast<uint32_t>(value));
    std::vector<char> liveStorage(execution.storages.size());
    for (const ProgramValueSlot &slot : execution.values) {
        if (!live[slot.id] || !slot.storage || *slot.storage >= liveStorage.size())
            continue;
        liveStorage[*slot.storage] = 1;
    }
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
        if (isProgramAdTapeType(slot.type))
            continue;
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
    if (boundLeaves && boundSignature && boundBindings &&
        !applyProgramBoundLeaves(*boundLeaves, *boundSignature, *boundBindings, execution, layouts, storage, backings,
                                 error))
        return false;
    if (resultLeaves && resultSignature && resultBindings &&
        !applyProgramBoundLeaves(*resultLeaves, *resultSignature, *resultBindings, execution, layouts, storage,
                                 backings, error))
        return false;
    if (captures && captureShapes) {
        if (captures->size() != execution.values.size() || captureShapes->size() != execution.values.size()) {
            error = "Program autodiff capture state is incomplete";
            return false;
        }
        for (size_t value = 0; value < captures->size(); ++value) {
            if (value >= storage.size())
                continue;
            if (!(*captureShapes)[value].empty())
                storage[value].concreteShape = (*captureShapes)[value];
            if ((*captures)[value].empty())
                continue;
            const ProgramValueSlot &slot = execution.values[value];
            if (isProgramAdTapeType(slot.type) || !slot.storage)
                continue;
            StorageBacking &backing = backings[*slot.storage];
            if ((*captures)[value].size() > backing.bytes)
                backing.bytes = (*captures)[value].size();
            backing.sized = true;
        }
    }
    if (!resolveProgramShapes(execution, topology, storage, error))
        return false;
    for (const ProgramStorageSlot &slot : execution.storages) {
        if (slot.id >= liveStorage.size() || !liveStorage[slot.id])
            continue;
        if (slot.initialValue < execution.values.size() &&
            isProgramAdTapeType(execution.values[slot.initialValue].type))
            continue;
        StorageBacking &backing = backings[slot.id];
        if (backing.sized)
            continue;
        if (slot.byteLengthExtents.empty()) {
            for (const ProgramValueSlot &value : execution.values) {
                if (!value.storage || *value.storage != slot.id)
                    continue;
                const std::optional<size_t> bytes = valueByteSize(value);
                if (bytes) {
                    backing.bytes = std::max(backing.bytes, *bytes);
                    backing.sized = true;
                    continue;
                }
                if (value.id < storage.size() && value.id < layouts.size() && layouts[value.id] &&
                    storage[value.id].concreteShape) {
                    size_t elements = 0;
                    if (shape::checkedElementCount(*storage[value.id].concreteShape, elements) &&
                        layouts[value.id]->byteSize &&
                        elements <= std::numeric_limits<size_t>::max() / layouts[value.id]->byteSize) {
                        const size_t runtimeBytes = elements * layouts[value.id]->byteSize;
                        backing.bytes = std::max(backing.bytes, runtimeBytes);
                        backing.sized = true;
                        continue;
                    }
                }
                if (!topology)
                    continue;
                for (const VernonResolvedProgramStage &stage : topology->stages)
                    for (size_t bindingIndex = 0; stage.pipeline && bindingIndex < stage.bindings.size() &&
                                                  bindingIndex < stage.pipeline->variant.parameters.size();
                         ++bindingIndex) {
                        if (stage.bindings[bindingIndex].value != value.id)
                            continue;
                        const std::optional<size_t> physicalBytes =
                            parameterByteSize(stage.pipeline->variant.parameters[bindingIndex], &value);
                        if (physicalBytes) {
                            backing.bytes = std::max(backing.bytes, *physicalBytes);
                            backing.sized = true;
                        }
                    }
            }
            if (backing.sized)
                continue;
            const std::string ownerName =
                slot.initialValue < execution.values.size() ? execution.values[slot.initialValue].name : std::string();
            error = "Program autodiff storage '" + ownerName + "' (id " + std::to_string(slot.id) +
                    ") has neither a boundary backing nor a resolvable live Value extent";
            return false;
        }
        size_t bytes = 0;
        std::vector<uint64_t> shape;
        if (!evaluateOwnedExtents(slot, storage, layouts, bytes, shape, error)) {
            error.clear();
            continue;
        }
        backing.bytes = bytes;
        backing.sized = true;
        if (backing.owner < storage.size())
            storage[backing.owner].concreteShape = shape;
    }
    for (;;) {
        if (!resolveProgramShapes(execution, topology, storage, error))
            return false;
        bool progress = false;
        bool pending = false;
        for (const ProgramStorageSlot &slot : execution.storages) {
            if (slot.id >= liveStorage.size() || !liveStorage[slot.id] || slot.byteLengthExtents.empty())
                continue;
            StorageBacking &backing = backings[slot.id];
            if (backing.sized)
                continue;
            pending = true;
            size_t bytes = 0;
            std::vector<uint64_t> shape;
            std::string extentError;
            if (!evaluateOwnedExtents(slot, storage, layouts, bytes, shape, extentError))
                continue;
            backing.bytes = bytes;
            backing.sized = true;
            if (backing.owner < storage.size())
                storage[backing.owner].concreteShape = std::move(shape);
            progress = true;
        }
        if (!pending)
            break;
        if (!progress) {
            error = "owned dyn Storage extents contain unresolved like-source dependencies:";
            for (const ProgramStorageSlot &slot : execution.storages) {
                if (slot.id >= liveStorage.size() || !liveStorage[slot.id] || backings[slot.id].sized ||
                    slot.byteLengthExtents.empty())
                    continue;
                error += " storage " + std::to_string(slot.id);
                for (const ProgramBufferExtent &extent : slot.byteLengthExtents)
                    if (!extent.isStatic)
                        error += " <- value " + std::to_string(extent.value) + " axis " + std::to_string(extent.axis);
            }
            return false;
        }
    }
    for (const ProgramStorageSlot &slot : execution.storages) {
        if (slot.id >= liveStorage.size() || !liveStorage[slot.id])
            continue;
        if (slot.initialValue < execution.values.size() &&
            isProgramAdTapeType(execution.values[slot.initialValue].type))
            continue;
        StorageBacking &backing = backings[slot.id];
        if (backing.owner >= storage.size() || !backing.bytes) {
            error = "Program autodiff storage has no materializable backing";
            return false;
        }
        storage[backing.owner].owned.resize(backing.bytes);
    }
    for (const ProgramValueSlot &slot : execution.values) {
        if (!live[slot.id])
            continue;
        HostProgramValue &value = storage[slot.id];
        if (isProgramAdTapeType(slot.type)) {
            std::shared_ptr<HostStaticTapeBatch> batch;
            if (tapeCaptures && slot.id < tapeCaptures->size())
                batch = (*tapeCaptures)[slot.id];
            if (!batch) {
                if (!tapePolicy) {
                    error = "Program autodiff tape has no memory policy";
                    return false;
                }
                size_t laneCount = 1;
                size_t payloadStride = 1;
                if (!programTapeLaneCount(execution, topology, slot.id, laneCount, error) ||
                    !programTapePayloadStride(slot.type, payloadStride, error))
                    return false;
                batch = HostStaticTapeBatch::create(laneCount, payloadStride, tapePolicy->invocationLimit(), tapePolicy,
                                                    nullptr);
            }
            if (!fillTapeHostValue(value, std::move(batch), error))
                return false;
            arguments[slot.id] = value.argument;
            continue;
        }
        void *hostData = nullptr;
        size_t byteSize = 0;
        if (slot.storage) {
            const StorageBacking &backing = backings[*slot.storage];
            hostData = storage[backing.owner].owned.data();
            byteSize = backing.bytes;
            if (!value.concreteShape)
                value.concreteShape = storage[backing.owner].concreteShape;
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
            error = "Program autodiff value has no storage";
            return false;
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

struct ProgramTapeState {
    uint32_t value{};
    VernonLaunchSize grid{};
    VernonLaunchSize workgroup{};
    const program::TargetBinding *tape{};
    const program::TargetBinding *segment{};
    const program::TargetBinding *status{};
    const program::TargetBinding *launch{};
    size_t stride{16};
};

bool multiplySize(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool carrierShape(const program::TargetBinding &binding, size_t bytes, std::vector<uint64_t> &shape,
                  std::vector<int64_t> &strides, std::string &error) {
    gpu::InternalBufferView view;
    if (!gpu::materializeInternalBufferView(binding.shape, binding.elementLayout, bytes, view)) {
        error = "Program tape carrier allocation does not match its element ABI";
        return false;
    }
    shape = std::move(view.shape);
    strides = std::move(view.strides);
    return true;
}

bool allocateProgramTapeState(ProgramValueArena &arena, VernonRuntimeContext &context, ProgramTapeState &state,
                              std::string &error) {
    size_t groupCount = 1;
    size_t workgroupVolume = 1;
    for (uint32_t extent : {state.grid.x, state.grid.y, state.grid.z})
        if (!extent || !multiplySize(groupCount, extent, groupCount)) {
            error = "Program tape dispatch group count overflows";
            return false;
        }
    for (uint32_t extent : {state.workgroup.x, state.workgroup.y, state.workgroup.z})
        if (!extent || !multiplySize(workgroupVolume, extent, workgroupVolume)) {
            error = "Program tape workgroup volume overflows";
            return false;
        }
    size_t groupTapeBytes = 0;
    size_t tapeBytes = 0;
    size_t segmentBytes = 0;
    if (!gpu::normalizeTapeStride(state.stride, state.stride) ||
        !multiplySize(state.stride, workgroupVolume, groupTapeBytes) ||
        !multiplySize(groupTapeBytes, groupCount, tapeBytes) ||
        !multiplySize(sizeof(gpu::Segment), groupCount, segmentBytes)) {
        error = "Program tape carrier size overflows";
        return false;
    }
    const auto allocate = [&](const program::TargetBinding *binding, size_t bytes) {
        if (!binding)
            return true;
        std::vector<uint64_t> shape;
        std::vector<int64_t> strides;
        return carrierShape(*binding, bytes, shape, strides, error) &&
               arena.allocateCarrier(context, state.value, *binding, bytes, std::move(shape), std::move(strides),
                                     error);
    };
    if (!allocate(state.tape, tapeBytes) || !allocate(state.segment, segmentBytes) ||
        !allocate(state.status, sizeof(gpu::BatchSummary)) || !allocate(state.launch, sizeof(uint32_t) * 3))
        return false;
    std::vector<gpu::Segment> segments(groupCount);
    for (size_t group = 0; group < groupCount; ++group)
        if (!gpu::initializeSegment(state.grid, state.workgroup, group, state.stride, workgroupVolume,
                                    segments[group])) {
            error = "Program replay segment initialization overflows";
            return false;
        }
    const gpu::BatchSummary clear{};
    const uint32_t launch[3]{state.grid.x, state.grid.y, state.grid.z};
    return (!state.segment || arena.uploadCarrier(state.value, program::CarrierSemantic::ReplaySegment, segments.data(),
                                                  segmentBytes, error)) &&
           (!state.status ||
            arena.uploadCarrier(state.value, program::CarrierSemantic::ReplayStatus, &clear, sizeof(clear), error)) &&
           (!state.launch ||
            arena.uploadCarrier(state.value, program::CarrierSemantic::LaunchMetadata, launch, sizeof(launch), error));
}

bool prepareProgramTapeStates(ProgramValueArena &arena, VernonRuntimeContext &context,
                              const VernonPipelineTopology &topology, const ProgramGraph &forward,
                              std::vector<ProgramTapeState> &states, std::string &error) {
    std::map<uint32_t, ProgramTapeState> byValue;
    for (const ProgramNode &node : forward.nodes) {
        const auto stageIndex = topology.stageIndices.find(node.stage);
        if (stageIndex == topology.stageIndices.end()) {
            error = "Program tape stage is not resolved";
            return false;
        }
        const VernonResolvedProgramStage &stage = topology.stages[stageIndex->second];
        for (const VernonProgramStageBinding &binding : stage.bindings) {
            if (!binding.target || binding.target->semantic == program::CarrierSemantic::Value ||
                binding.target->semantic == program::CarrierSemantic::Resource)
                continue;
            ProgramTapeState &state = byValue[binding.value];
            state.value = binding.value;
            state.grid = {static_cast<uint32_t>(node.grid[0]), static_cast<uint32_t>(node.grid[1]),
                          static_cast<uint32_t>(node.grid[2])};
            state.workgroup = stage.pipeline->workgroupSize;
            if (binding.target->semantic == program::CarrierSemantic::TapeData)
                state.tape = &*binding.target;
            else if (binding.target->semantic == program::CarrierSemantic::ReplaySegment)
                state.segment = &*binding.target;
            else if (binding.target->semantic == program::CarrierSemantic::ReplayStatus)
                state.status = &*binding.target;
            else
                state.launch = &*binding.target;
        }
    }
    states.clear();
    states.reserve(byValue.size());
    for (auto &[value, state] : byValue) {
        (void)value;
        if (!state.tape || !state.segment) {
            error = "Program tape logical value has an incomplete carrier bundle";
            return false;
        }
        if (!allocateProgramTapeState(arena, context, state, error))
            return false;
        states.push_back(state);
    }
    return true;
}

bool validateProgramTapeStates(ProgramValueArena &arena, std::vector<ProgramTapeState> &states, bool &retry,
                               std::string &error) {
    constexpr uint32_t tapeReady = 0;
    constexpr uint32_t tapeOverflow = 1;
    retry = false;
    for (ProgramTapeState &state : states) {
        if (!state.status)
            continue;
        gpu::BatchSummary summary{};
        if (!arena.downloadCarrier(state.value, program::CarrierSemantic::ReplayStatus, &summary, sizeof(summary),
                                   error))
            return false;
        if (summary.status == tapeReady)
            continue;
        if (summary.status != tapeOverflow || summary.requiredBytes <= state.stride) {
            error = "Program tape carrier reported an invalid replay status";
            return false;
        }
        state.stride = summary.requiredBytes;
        retry = true;
    }
    return true;
}

class ProgramPullback final : public PullbackExecution {
public:
    ProgramPullback(VernonRuntimeContext &context, std::shared_ptr<VernonPipelineTopology> topology, Variant variant,
                    Signature signature, std::vector<ProgramLeafBinding> cotangentBindings,
                    std::vector<ProgramLeafBinding> gradientBindings, ProgramResidualPlan plan,
                    std::vector<std::vector<uint8_t>> residuals, std::vector<std::vector<uint64_t>> residualShapes,
                    std::vector<std::shared_ptr<HostStaticTapeBatch>> tapeResiduals,
                    std::vector<AutodiffPullbackPassTelemetry> passTelemetry,
                    std::unique_ptr<ProgramValueArena> deviceState = nullptr)
        : context_(&context), topology_(std::move(topology)), variant_(std::move(variant)),
          signature_(std::move(signature)), cotangentBindings_(std::move(cotangentBindings)),
          gradientBindings_(std::move(gradientBindings)), plan_(std::move(plan)), residuals_(std::move(residuals)),
          residualShapes_(std::move(residualShapes)), tapeResiduals_(std::move(tapeResiduals)),
          passTelemetry_(std::move(passTelemetry)), deviceState_(std::move(deviceState)) {
        rebuildVariantLayouts(variant_);
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        const ProgramGraph *backward = findGraph(topology_->execution, "backward");
        if (!backward || (!backward->arguments.empty() && !cotangents))
            return fail(*context_, "Program pullback has no backward graph or required cotangents");
        if (deviceState_) {
            std::vector<HostProgramValue> storage;
            std::vector<VernonPipelineArgument> values;
            std::string error;
            std::vector<char> required(topology_->execution.values.size());
            markProgramGraphValues(*backward, required);
            std::vector<std::vector<uint8_t>> captureBytes(topology_->execution.values.size());
            std::vector<std::vector<uint64_t>> captureShapes(topology_->execution.values.size());
            for (uint32_t value : topology_->execution.residualCaptures())
                if (value < captureShapes.size() && value < deviceState_->hostValues().size() &&
                    deviceState_->hostValues()[value].concreteShape)
                    captureShapes[value] = *deviceState_->hostValues()[value].concreteShape;
            if (!materializeValues(topology_->execution, topology_.get(), variant_, storage, values, required,
                                   cotangents, cotangents ? &signature_.cotangents : nullptr,
                                   cotangents ? &cotangentBindings_ : nullptr, &gradients, &signature_.gradients,
                                   &gradientBindings_, &captureBytes, &captureShapes, nullptr,
                                   context_->autodiffMemoryPolicy, error) ||
                (cotangents &&
                 !transferLeaves(*cotangents, signature_.cotangents, cotangentBindings_, storage, false, error)))
                return fail(*context_, error);
            ProgramValueArena derivatives(std::move(storage));
            if (!derivatives.materializeDevice(*context_, required, error))
                return fail(*context_, error);
            for (uint32_t value : topology_->execution.residualCaptures())
                if (!derivatives.adoptRetainedValue(value, *deviceState_, error))
                    return fail(*context_, error);
            VernonLoadedPipeline proxy;
            proxy.context = context_;
            proxy.variant = variant_;
            proxy.topology = topology_;
            const VernonStatus status = executePipelineProgramGraph(proxy, *backward, derivatives);
            if (status != VERNON_STATUS_OK)
                return status;
            std::vector<char> downloads(topology_->execution.values.size());
            for (const ProgramLeafBinding &binding : gradientBindings_)
                if (binding.value < downloads.size())
                    downloads[binding.value] = 1;
            if (!derivatives.downloadLogicalToHost(downloads, error) ||
                !transferLeaves(gradients, signature_.gradients, gradientBindings_, derivatives.hostValues(), true,
                                error))
                return fail(*context_, error);
            ++usage_.submissions;
            ++usage_.waits;
            ++usage_.readbacks;
            ++usage_.atomicPublications;
            return VERNON_STATUS_OK;
        }
        std::vector<HostProgramValue> storage;
        std::vector<VernonPipelineArgument> values;
        std::string error;
        std::vector<char> required(topology_->execution.values.size());
        markProgramGraphValues(*backward, required);
        if (plan_.replayEnd) {
            const ProgramGraph *forward = findGraph(topology_->execution, "forward");
            if (!forward)
                return fail(*context_, "Program pullback replay has no forward graph");
            ProgramGraph replay = *forward;
            replay.nodes.resize(plan_.replayEnd);
            markProgramGraphValues(replay, required);
        }
        for (uint32_t value : topology_->execution.residualCaptures())
            if (value < required.size())
                required[value] = 1;
        for (uint32_t value : plan_.retainedValues)
            if (value < required.size())
                required[value] = 1;
        if (!materializeValues(topology_->execution, topology_.get(), variant_, storage, values, required, cotangents,
                               cotangents ? &signature_.cotangents : nullptr,
                               cotangents ? &cotangentBindings_ : nullptr, &gradients, &signature_.gradients,
                               &gradientBindings_, &residuals_, &residualShapes_, &tapeResiduals_,
                               context_->autodiffMemoryPolicy, error))
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
            if (value < tapeResiduals_.size() && tapeResiduals_[value])
                continue;
            std::memcpy(const_cast<void *>(storage[value].argument.tensor.host_data), residuals_[value].data(),
                        residuals_[value].size());
        }
        ProgramValueArena arena(std::move(storage));
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology_;
        if (plan_.replayEnd) {
            const ProgramGraph *forward = findGraph(topology_->execution, "forward");
            ProgramGraph replay = *forward;
            replay.nodes.resize(plan_.replayEnd);
            const VernonStatus replayStatus = executePipelineProgramGraph(proxy, replay, arena);
            if (replayStatus != VERNON_STATUS_OK)
                return replayStatus;
            if (!sealProgramTapeValues(arena.hostValues(), arena.logicalArguments(), error))
                return fail(*context_, error);
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *backward, arena);
        if (status == VERNON_STATUS_OK &&
            transferLeaves(gradients, signature_.gradients, gradientBindings_, arena.hostValues(), true, error)) {
            ++usage_.submissions;
            ++usage_.waits;
            ++usage_.atomicPublications;
            usage_.temporaryAllocationBytes += temporaryBytes;
            return status;
        }
        return status == VERNON_STATUS_OK ? fail(*context_, error) : status;
    }

    PullbackMemoryUsage memoryUsage() const override {
        MemoryAccounting memory;
        memory.logicalPayloadBytes = plan_.checkpoint.logicalResidualBytes;
        const auto add = [](size_t &total, size_t bytes) {
            if (bytes > std::numeric_limits<size_t>::max() - total)
                total = std::numeric_limits<size_t>::max();
            else
                total += bytes;
        };
        for (const auto &batch : tapeResiduals_) {
            if (!batch)
                continue;
            // compact(true) keeps write descriptors for the packed Program tape ABI.
            // Graph AD compact(false) drops that construction storage from resident.
            add(memory.residentBytes, batch->isCompacted() ? batch->logicalBytes() : batch->residentBytes());
            add(memory.allocatedBytes, batch->allocatedBytes());
        }
        for (size_t value = 0; value < residuals_.size(); ++value) {
            const std::vector<uint8_t> &residual = residuals_[value];
            if (residual.empty())
                continue;
            if (value >= tapeResiduals_.size() || !tapeResiduals_[value])
                add(memory.residentBytes, residual.size());
            add(memory.retainedAllocationBytes, residual.size());
        }
        add(memory.retainedAllocationBytes, memory.allocatedBytes);
        return pullbackMemoryUsage(memory);
    }

    PullbackControlPlaneUsage controlPlaneUsage() const override { return usage_; }

    AutodiffPullbackCheckpointPlan checkpointPlan() const override {
        AutodiffPullbackCheckpointPlan snapshot;
        snapshot.present = true;
        snapshot.peakBytes = plan_.checkpoint.peakBytes;
        snapshot.memoryBudget = plan_.checkpoint.memoryBudget;
        snapshot.logicalResidualBytes = plan_.checkpoint.logicalResidualBytes;
        snapshot.retainedAllocationBytes = plan_.checkpoint.retainedAllocationBytes;
        snapshot.initialStateBytes = plan_.checkpoint.initialStateBytes;
        snapshot.restorationBytes = plan_.checkpoint.restorationBytes;
        snapshot.transactionBytes = plan_.checkpoint.transactionBytes;
        snapshot.persistentCheckpointBytes = plan_.checkpoint.persistentCheckpointBytes;
        snapshot.backwardValueBytes = plan_.checkpoint.backwardValueBytes;
        snapshot.replayCost = plan_.checkpoint.replayCost;
        snapshot.recomputationCost = plan_.checkpoint.recomputationCost;
        snapshot.selectedPolicy = plan_.checkpoint.selectedPolicy;
        return snapshot;
    }

    std::vector<AutodiffPullbackPassTelemetry> passTelemetry() const override { return passTelemetry_; }

    uint64_t peakRuntimeManagedBytes() const override {
        const PullbackMemoryUsage usage = memoryUsage();
        const uint64_t held =
            std::max(std::max(usage.retainedAllocationBytes, usage.allocatedBytes), usage.peakTemporaryBytes);
        return std::max(held, plan_.checkpoint.peakBytes);
    }

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
    std::vector<std::shared_ptr<HostStaticTapeBatch>> tapeResiduals_;
    std::vector<AutodiffPullbackPassTelemetry> passTelemetry_;
    std::unique_ptr<ProgramValueArena> deviceState_;
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
        std::vector<HostProgramValue> hostStorage;
        std::vector<VernonPipelineArgument> values;
        std::string error;
        std::vector<char> required(topology->execution.values.size());
        markProgramGraphValues(*forward, required);
        for (uint32_t value : topology->execution.residualCaptures())
            if (value < required.size())
                required[value] = 1;
        if (!materializeValues(topology->execution, topology.get(), variant_, hostStorage, values, required, &inputs,
                               &signature_.inputs, &inputBindings_, outputs, &signature_.outputs, &outputBindings_,
                               nullptr, nullptr, nullptr, context_->autodiffMemoryPolicy, error) ||
            !transferLeaves(inputs, signature_.inputs, inputBindings_, hostStorage, false, error))
            return fail(*context_, error);
        ProgramValueArena arena(std::move(hostStorage));
        VernonLoadedPipeline proxy;
        proxy.context = context_;
        proxy.variant = variant_;
        proxy.topology = topology;
        if (context_->backend != VERNON_RUNTIME_CPU) {
            if (!arena.materializeDevice(*context_, required, error))
                return fail(*context_, error);
            std::vector<ProgramTapeState> tapeStates;
            if (!prepareProgramTapeStates(arena, *context_, *topology, *forward, tapeStates, error))
                return fail(*context_, error);
            for (;;) {
                const VernonStatus status = executePipelineProgramGraph(proxy, *forward, arena);
                if (status != VERNON_STATUS_OK)
                    return status;
                bool retry = false;
                if (!validateProgramTapeStates(arena, tapeStates, retry, error))
                    return fail(*context_, error);
                if (!retry)
                    break;
                if (!arena.restoreDeviceValuesFromHost(required, error))
                    return fail(*context_, error);
                for (ProgramTapeState &state : tapeStates)
                    if (!allocateProgramTapeState(arena, *context_, state, error))
                        return fail(*context_, error);
            }
            std::vector<char> downloads(topology->execution.values.size());
            for (const ProgramLeafBinding &binding : outputBindings_)
                if (binding.value < downloads.size())
                    downloads[binding.value] = 1;
            if (!arena.downloadLogicalToHost(downloads, error) ||
                !transferLeaves(*outputs, signature_.outputs, outputBindings_, arena.hostValues(), true, error))
                return fail(*context_, error);
            ProgramResidualPlan plan;
            plan.retainedValues = topology->execution.residualCaptures();
            std::vector<AutodiffPullbackPassTelemetry> telemetry =
                collectProgramPassTelemetry(*forward, topology->execution, arena.hostValues(), plan);
            pullback = std::make_unique<ProgramPullback>(
                *context_, topology, variant_, signature_, cotangentBindings_, gradientBindings_, std::move(plan),
                std::vector<std::vector<uint8_t>>(topology->execution.values.size()),
                std::vector<std::vector<uint64_t>>(topology->execution.values.size()),
                std::vector<std::shared_ptr<HostStaticTapeBatch>>(topology->execution.values.size()),
                std::move(telemetry), std::make_unique<ProgramValueArena>(std::move(arena)));
            return VERNON_STATUS_OK;
        }
        const VernonStatus status = executePipelineProgramGraph(proxy, *forward, arena);
        if (status != VERNON_STATUS_OK)
            return status;
        std::vector<HostProgramValue> &storage = arena.hostValues();
        if (!transferLeaves(*outputs, signature_.outputs, outputBindings_, storage, true, error))
            return fail(*context_, error);
        ProgramResidualPlan plan;
        uint64_t memoryBudget = context_->autodiffMemoryPolicy->invocationLimit();
        const bool rematerializeTapes = topology->programCheckpointMemoryBudget.has_value();
        if (rematerializeTapes)
            memoryBudget = *topology->programCheckpointMemoryBudget;
        if (!planProgramResiduals(topology->execution, variant_, storage, memoryBudget,
                                  programCheckpointPolicy(topology->programCheckpointPolicy), rematerializeTapes, plan,
                                  error))
            return fail(*context_, error);
        std::vector<std::vector<uint8_t>> residuals(topology->execution.values.size());
        std::vector<std::vector<uint64_t>> residualShapes(topology->execution.values.size());
        std::vector<std::shared_ptr<HostStaticTapeBatch>> tapeResiduals(topology->execution.values.size());
        for (uint32_t value : topology->execution.residualCaptures())
            if (storage[value].concreteShape)
                residualShapes[value] = *storage[value].concreteShape;
        for (uint32_t value : plan.retainedValues) {
            HostProgramValue &slot = storage[value];
            if (slot.tapeBatch) {
                if (!slot.tapeBatch->compact(true))
                    return fail(*context_, "Program autodiff tape could not be compacted");
                if (!fillTapeHostValue(slot, slot.tapeBatch, error))
                    return fail(*context_, error);
                tapeResiduals[value] = slot.tapeBatch;
            }
            const auto *data = static_cast<const uint8_t *>(slot.argument.tensor.host_data);
            if (!data || !slot.argument.tensor.byte_size)
                return fail(*context_, "Program autodiff residual state is incomplete");
            residuals[value].assign(data, data + slot.argument.tensor.byte_size);
            if (slot.concreteShape)
                residualShapes[value] = *slot.concreteShape;
        }
        std::vector<AutodiffPullbackPassTelemetry> telemetry =
            collectProgramPassTelemetry(*forward, topology->execution, storage, plan);
        pullback = std::make_unique<ProgramPullback>(
            *context_, topology, variant_, signature_, cotangentBindings_, gradientBindings_, std::move(plan),
            std::move(residuals), std::move(residualShapes), std::move(tapeResiduals), std::move(telemetry));
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
        if (isProgramAdTapeType(value.type))
            continue;
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
