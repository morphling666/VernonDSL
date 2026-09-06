#include "program_value_materializer.h"

#include "runtime/autodiff/program_shape_resolver.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <string_view>

namespace vernon::runtime::ad {
namespace {

std::optional<size_t> parameterByteSize(const Parameter &parameter, const program::Value *slot) {
    const std::optional<ValueLayout> slotLayout = slot ? resolvedProgramValueLayout(*slot) : std::nullopt;
    const ValueLayout &layout = slotLayout              ? *slotLayout
                                : parameter.valueLayout ? *parameter.valueLayout
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

bool fillHostTensor(LogicalProgramValue &value, const program::Value &slot, const ValueLayout &layout, size_t byteSize,
                    void *hostData, std::string &error) {
    if (!value.concreteShape) {
        value.concreteShape = shape::concrete(shape::decodeRuntimeContractShape(slot.shape));
        if (!value.concreteShape)
            return error = "Program autodiff tensor shape was not resolved before allocation", false;
    }
    if (value.strides.empty()) {
        if (!shape::rowMajorByteStrides(*value.concreteShape, layout.byteSize, value.strides))
            return error = "Program Value " + std::to_string(slot.id) + " ('" + slot.name +
                           "') has a zero or overflowing materialized layout",
                   false;
    } else if (value.strides.size() != value.concreteShape->size()) {
        return error = "Program autodiff tensor shape and strides have different ranks", false;
    }
    value.argument.kind = VERNON_PROGRAM_TENSOR;
    value.argument.tensor.struct_size = sizeof(VernonTensorView);
    value.argument.tensor.storage = VERNON_TENSOR_HOST;
    value.argument.tensor.host_data = hostData;
    value.argument.tensor.element_layout = programValueLayoutView(layout);
    value.argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    value.argument.tensor.rank = static_cast<uint32_t>(value.concreteShape->size());
    value.argument.tensor.shape = value.concreteShape->empty() ? nullptr : value.concreteShape->data();
    value.argument.tensor.byte_strides = value.strides.empty() ? nullptr : value.strides.data();
    value.argument.tensor.byte_size = byteSize;
    return true;
}

bool evaluateOwnedExtents(const program::Program &program, const program::Storage &storage,
                          const std::vector<LogicalProgramValue> &hosts,
                          const std::vector<std::optional<ValueLayout>> &layouts, size_t &bytes,
                          std::vector<uint64_t> &concreteShape, std::string &error) {
    bytes = 0;
    concreteShape.clear();
    const uint32_t owner = storage.initialValue;
    if (owner >= layouts.size() || !layouts[owner] || !layouts[owner]->byteSize)
        return error = "owned dyn Storage has no element layout", false;
    uint64_t product = 1;
    concreteShape.reserve(storage.buffer.byteLengthExtents.size());
    const std::string graph = owner < program.values.size() ? program.values[owner].origin.graph : std::string();
    for (const program::ControlComponent &extent : storage.buffer.byteLengthExtents) {
        uint64_t length = extent.value;
        if (extent.kind != program::ControlKind::Static) {
            uint32_t value = UINT32_MAX;
            if (!program::resolveControlValue(program, extent, graph, value) || value >= hosts.size() ||
                !hosts[value].concreteShape || hosts[value].concreteShape->size() <= extent.axis)
                return error = "owned dyn like-source dimension is unavailable at invocation bind", false;
            length = (*hosts[value].concreteShape)[extent.axis];
        }
        if (length && product > std::numeric_limits<uint64_t>::max() / length)
            return error = "owned dyn Storage extent overflows", false;
        product *= length;
        concreteShape.push_back(length);
    }
    if (product && layouts[owner]->byteSize > std::numeric_limits<uint64_t>::max() / product)
        return error = "owned dyn Storage byte length overflows", false;
    bytes = static_cast<size_t>(layouts[owner]->byteSize * product);
    return true;
}

bool tapePayloadStride(const std::string &type, size_t &stride, std::string &error) {
    stride = 1;
    constexpr std::string_view prefix = "!vernon.ad_tape<";
    if (type.size() <= prefix.size() || type.compare(0, prefix.size(), prefix.data(), prefix.size()) != 0 ||
        type.back() != '>')
        return true;
    std::string_view digits = std::string_view(type).substr(prefix.size(), type.size() - prefix.size() - 1);
    uint64_t bytes = 0;
    const auto [end, parseError] = std::from_chars(digits.data(), digits.data() + digits.size(), bytes);
    if (parseError != std::errc{} || end != digits.data() + digits.size() || !bytes ||
        bytes > std::numeric_limits<size_t>::max())
        return error = "Program autodiff tape type has an invalid payload stride", false;
    stride = static_cast<size_t>(bytes);
    return true;
}

bool tapeLaneCount(const program::Program &execution, const VernonProgramTopology *topology,
                   const std::vector<LogicalProgramValue> &hosts, uint32_t tapeValue, size_t &lanes,
                   std::string &error) {
    lanes = 1;
    for (const program::Graph &graph : execution.graphs) {
        for (const program::Node &node : graph.nodes) {
            const bool used = std::find(node.results.begin(), node.results.end(), tapeValue) != node.results.end() ||
                              std::find(node.operands.begin(), node.operands.end(), tapeValue) != node.operands.end();
            if (!used)
                continue;
            uint32_t workgroup[3]{256, 1, 1};
            if (topology) {
                const auto found = topology->nodes.find({graph.direction, node.id});
                if (found != topology->nodes.end()) {
                    const VernonProgramExecutable *pipeline = found->second.pipeline;
                    if (pipeline) {
                        workgroup[0] = pipeline->workgroupSize.x ? pipeline->workgroupSize.x : 1;
                        workgroup[1] = pipeline->workgroupSize.y ? pipeline->workgroupSize.y : 1;
                        workgroup[2] = pipeline->workgroupSize.z ? pipeline->workgroupSize.z : 1;
                    }
                }
            }
            size_t volume = 1;
            if (program::executionKind(node) != program::ExecutionKind::Compute)
                continue;
            const program::ComputeOperation &compute = program::computeOperation(node);
            for (int axis = 0; axis < 3; ++axis) {
                uint64_t resolvedGroups = 0;
                if (!resolveProgramControl(execution, hosts, compute.workgroups[axis], resolvedGroups, error) ||
                    !resolvedGroups || resolvedGroups > std::numeric_limits<size_t>::max())
                    return error = error.empty() ? "Program autodiff tape dispatch control is invalid" : error, false;
                const size_t groups = static_cast<size_t>(resolvedGroups);
                const size_t wg = workgroup[axis] ? workgroup[axis] : 1;
                if (groups > std::numeric_limits<size_t>::max() / wg ||
                    volume > std::numeric_limits<size_t>::max() / (groups * wg))
                    return error = "Program autodiff tape dispatch volume overflows", false;
                volume *= groups * wg;
            }
            lanes = std::max(lanes, volume);
        }
    }
    return true;
}

bool attachTape(const program::Program &execution, const VernonProgramTopology *topology,
                const std::vector<LogicalProgramValue> &hosts, const program::Value &slot,
                const std::vector<std::shared_ptr<HostStaticTapeBatch>> *captures,
                const std::shared_ptr<AutodiffMemoryPolicy> &policy, LogicalProgramValue &value, std::string &error) {
    std::shared_ptr<HostStaticTapeBatch> batch;
    if (captures && slot.id < captures->size())
        batch = (*captures)[slot.id];
    if (!batch) {
        if (!policy)
            return error = "Program autodiff tape has no memory policy", false;
        size_t lanes = 1;
        size_t stride = 1;
        if (!tapeLaneCount(execution, topology, hosts, slot.id, lanes, error) ||
            !tapePayloadStride(slot.type, stride, error))
            return false;
        batch = HostStaticTapeBatch::create(lanes, stride, policy->invocationLimit(), policy, nullptr);
    }
    return fillProgramTapeHostValue(value, std::move(batch), error);
}

} // namespace

std::optional<ValueLayout> resolvedProgramValueLayout(const program::Value &value) {
    if (!value.layout)
        return std::nullopt;
    return program::materializeValueLayout(*value.layout, value.type);
}

std::optional<size_t> programValueByteSize(const program::Value &value, const Parameter *parameter) {
    const std::optional<ValueLayout> layout = resolvedProgramValueLayout(value);
    if (layout && layout->byteSize) {
        Parameter synthesized;
        synthesized.kind = "tensor";
        synthesized.shape = value.shape;
        synthesized.valueLayout = *layout;
        return parameterByteSize(synthesized, &value);
    }
    return parameter ? parameterByteSize(*parameter, &value) : std::nullopt;
}

VernonValueLayoutView programValueLayoutView(const ValueLayout &layout) {
    return {sizeof(VernonValueLayoutView),
            layout.byteSize,
            layout.alignment,
            {layout.layoutHash.data(), layout.layoutHash.size()},
            layout.abiLeaves.empty() ? nullptr : layout.abiLeaves.data(),
            layout.abiLeaves.size()};
}

bool fillProgramTapeHostValue(LogicalProgramValue &value, std::shared_ptr<HostStaticTapeBatch> batch,
                              std::string &error) {
    if (!batch)
        return error = "Program autodiff tape has no allocator batch", false;
    VernonAdTapeAllocator *descriptor = batch->descriptor(0);
    if (!descriptor)
        return error = "Program autodiff tape has no allocator descriptor", false;
    value.tapeBatch = std::move(batch);
    value.ownership = ProgramValueOwnership::TapeCarrier;
    const VernonAdRegionHandle root = value.tapeBatch->rootRegion(0);
    constexpr size_t descriptorBytes = sizeof(VernonAdTapeAllocator *);
    value.owned.resize(descriptorBytes + sizeof(root));
    std::memcpy(value.owned.data(), &descriptor, descriptorBytes);
    std::memcpy(value.owned.data() + descriptorBytes, &root, sizeof(root));
    value.argument = {};
    value.argument.kind = VERNON_PROGRAM_TENSOR;
    value.argument.tensor.struct_size = sizeof(VernonTensorView);
    value.argument.tensor.storage = VERNON_TENSOR_HOST;
    value.argument.tensor.host_data = value.owned.data();
    value.argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    value.argument.tensor.byte_size = value.owned.size();
    return true;
}

bool materializeProgramOwnedStorages(const program::Program &execution, const VernonProgramTopology *topology,
                                     std::vector<LogicalProgramValue> &storage, const std::vector<char> &liveStorage,
                                     const std::vector<std::optional<ValueLayout>> &layouts,
                                     std::map<uint32_t, ProgramStorageState> &backings, std::string &error) {
    if (!resolveProgramShapes(execution, topology, storage, error))
        return false;
    for (const program::Storage &slot : execution.storages) {
        if (slot.id >= liveStorage.size() || !liveStorage[slot.id] ||
            (slot.initialValue < execution.values.size() &&
             program::isTapeValueType(execution.values[slot.initialValue].type)))
            continue;
        ProgramStorageState &backing = backings[slot.id];
        if (backing.external || backing.sized)
            continue;
        if (slot.buffer.byteLengthExtents.empty()) {
            for (const program::Value &value : execution.values) {
                if (!value.storage || *value.storage != slot.id)
                    continue;
                const std::optional<size_t> bytes = programValueByteSize(value);
                if (bytes) {
                    backing.bytes = std::max(backing.bytes, *bytes);
                    backing.sized = true;
                } else if (value.id < storage.size() && value.id < layouts.size() && layouts[value.id] &&
                           storage[value.id].concreteShape) {
                    size_t elements = 0;
                    if (shape::checkedElementCount(*storage[value.id].concreteShape, elements) &&
                        layouts[value.id]->byteSize &&
                        elements <= std::numeric_limits<size_t>::max() / layouts[value.id]->byteSize) {
                        backing.bytes = std::max(backing.bytes, elements * layouts[value.id]->byteSize);
                        backing.sized = true;
                    }
                }
            }
            if (!backing.sized) {
                const std::string ownerName = slot.initialValue < execution.values.size()
                                                  ? execution.values[slot.initialValue].name
                                                  : std::string();
                error = "Program autodiff storage '" + ownerName + "' (id " + std::to_string(slot.id) +
                        ") has neither a boundary backing nor a resolvable live Value extent";
                return false;
            }
            continue;
        }
        size_t bytes = 0;
        std::vector<uint64_t> concreteShape;
        if (evaluateOwnedExtents(execution, slot, storage, layouts, bytes, concreteShape, error)) {
            backing.bytes = bytes;
            backing.sized = true;
            if (backing.owner < storage.size())
                storage[backing.owner].concreteShape = std::move(concreteShape);
        } else {
            error.clear();
        }
    }
    for (;;) {
        if (!resolveProgramShapes(execution, topology, storage, error))
            return false;
        bool progress = false;
        bool pending = false;
        for (const program::Storage &slot : execution.storages) {
            if (slot.id >= liveStorage.size() || !liveStorage[slot.id] || slot.buffer.byteLengthExtents.empty())
                continue;
            ProgramStorageState &backing = backings[slot.id];
            if (backing.external || backing.sized)
                continue;
            pending = true;
            size_t bytes = 0;
            std::vector<uint64_t> concreteShape;
            std::string extentError;
            if (!evaluateOwnedExtents(execution, slot, storage, layouts, bytes, concreteShape, extentError))
                continue;
            backing.bytes = bytes;
            backing.sized = true;
            if (backing.owner < storage.size())
                storage[backing.owner].concreteShape = std::move(concreteShape);
            progress = true;
        }
        if (!pending)
            break;
        if (!progress) {
            error = "owned dyn Storage extents contain unresolved like-source dependencies:";
            for (const program::Storage &slot : execution.storages) {
                if (slot.id >= liveStorage.size() || !liveStorage[slot.id] || backings[slot.id].external ||
                    backings[slot.id].sized || slot.buffer.byteLengthExtents.empty())
                    continue;
                error += " storage " + std::to_string(slot.id);
                for (const program::ControlComponent &extent : slot.buffer.byteLengthExtents)
                    if (extent.kind != program::ControlKind::Static)
                        error +=
                            " <- control " + std::to_string(extent.reference) + " axis " + std::to_string(extent.axis);
            }
            return false;
        }
    }
    for (const program::Storage &slot : execution.storages) {
        if (slot.id >= liveStorage.size() || !liveStorage[slot.id] ||
            (slot.initialValue < execution.values.size() &&
             program::isTapeValueType(execution.values[slot.initialValue].type)))
            continue;
        ProgramStorageState &backing = backings[slot.id];
        if (backing.external)
            continue;
        if (backing.owner >= storage.size() || !backing.sized)
            return error = "Program autodiff storage has no materializable backing", false;
        storage[backing.owner].owned.resize(std::max<size_t>(backing.bytes, 1));
        if (backing.initial) {
            const VernonTensorView &initial = backing.initial->tensor;
            if (initial.byte_size > backing.bytes) {
                error = "staged Program Storage input exceeds its canonical backing";
                return false;
            }
            std::memcpy(storage[backing.owner].owned.data(), initial.host_data, initial.byte_size);
        }
    }
    return true;
}

bool materializeProgramValues(const program::Program &execution, const VernonProgramTopology *topology,
                              std::vector<LogicalProgramValue> &storage, const std::vector<char> &live,
                              const std::vector<std::optional<ValueLayout>> &layouts,
                              const std::map<uint32_t, ProgramStorageState> &backings,
                              const std::map<uint32_t, VernonProgramArgument> &externalValues,
                              const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures,
                              const std::shared_ptr<AutodiffMemoryPolicy> &tapePolicy, std::string &error) {
    for (const program::Value &slot : execution.values) {
        if (!live[slot.id])
            continue;
        LogicalProgramValue &value = storage[slot.id];
        if (program::isTapeValueType(slot.type)) {
            if (!attachTape(execution, topology, storage, slot, tapeCaptures, tapePolicy, value, error))
                return false;
            continue;
        }
        void *hostData = nullptr;
        size_t byteSize = 0;
        if (slot.storage) {
            const ProgramStorageState &backing = backings.at(*slot.storage);
            if (backing.external) {
                value.argument = *backing.external;
                if (value.argument.kind == VERNON_PROGRAM_TENSOR) {
                    value.argument.tensor.element_layout = programValueLayoutView(*layouts[slot.id]);
                    if (value.concreteShape) {
                        value.argument.tensor.rank = static_cast<uint32_t>(value.concreteShape->size());
                        value.argument.tensor.shape =
                            value.concreteShape->empty() ? nullptr : value.concreteShape->data();
                        value.argument.tensor.byte_strides = value.strides.empty() ? nullptr : value.strides.data();
                    }
                }
                continue;
            }
            hostData = storage[backing.owner].owned.data();
            byteSize = backing.bytes;
            if (!value.concreteShape)
                value.concreteShape = storage[backing.owner].concreteShape;
        } else if (const auto external = externalValues.find(slot.id); external != externalValues.end()) {
            value.argument = external->second;
            value.ownership =
                value.argument.kind == VERNON_PROGRAM_TENSOR && value.argument.tensor.storage == VERNON_TENSOR_HOST
                    ? ProgramValueOwnership::BorrowedHost
                    : ProgramValueOwnership::BorrowedDevice;
            if (value.argument.kind == VERNON_PROGRAM_TENSOR)
                value.argument.tensor.element_layout = programValueLayoutView(*layouts[slot.id]);
            continue;
        } else if (!programValueHasDynamicShape(slot)) {
            const std::optional<size_t> bytes = programValueByteSize(slot);
            if (!bytes)
                return error = "Program autodiff value has no materializable tensor parameter", false;
            value.owned.resize(*bytes);
            hostData = value.owned.data();
            byteSize = *bytes;
        } else {
            return error = "Program autodiff value has no storage", false;
        }
        if (!fillHostTensor(value, slot, *layouts[slot.id], byteSize, hostData, error))
            return false;
    }
    return true;
}

} // namespace vernon::runtime::ad
