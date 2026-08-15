#include "runtime_autodiff_internal.h"

#include "host_effect_transaction.h"
#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/cpu_workgroup_dispatch.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
#include "host_tape_test_hooks.h"
#endif

namespace vernon::runtime::ad {

namespace {

constexpr size_t kCpuAdGradientDispatchLimit = 512u * 1024u * 1024u;
constexpr size_t kCpuAdInlineInvocationLimit = 16u * 1024u;

bool checkedAddBytes(size_t &total, size_t count, size_t bytes) {
    if (count && bytes > std::numeric_limits<size_t>::max() / count)
        return false;
    const size_t additional = count * bytes;
    if (additional > std::numeric_limits<size_t>::max() - total)
        return false;
    total += additional;
    return true;
}

struct HostFrameLeaf {
    ValueAbi value;
    size_t frameOffset{};
};

struct HostTensorView {
    std::string access;
    std::vector<int64_t> shape;
    ValueLayout elementLayout;
    std::vector<ValueAbi> leaves;
};

struct HostArgument {
    std::string name;
    std::string builtin;
    std::string accumulationOwnership;
    size_t offset{};
    size_t size{};
    std::optional<HostTensorView> tensorView;
    std::vector<HostFrameLeaf> leaves;
};

struct HostProfileLayout {
    size_t argumentsSize{};
    size_t resultsSize{};
    std::optional<size_t> tapeAllocatorOffset;
    std::optional<size_t> tapeRootRegionOffset;
    uint32_t workgroup[3]{1, 1, 1};
    DispatchContract dispatchContract;
    std::vector<HostArgument> arguments;
    std::vector<HostFrameLeaf> results;
};

struct StorageRange {
    uintptr_t begin{};
    uintptr_t end{};
    bool writable{};
};

struct StagedTensorView {
    const HostArgument *argument{};
    size_t elementCount{};
    std::vector<uint64_t> shape;
    std::vector<uint8_t> packed;
    std::vector<uint8_t *> leafShadows;
};

using RuntimeTensorShapes = std::unordered_map<std::string, std::vector<uint64_t>>;
using RetainedPrimalLeaves = std::unordered_map<std::string, std::vector<uint8_t>>;
struct RetainedPrimalTensorView {
    std::vector<uint64_t> shape;
    std::vector<uint8_t> packed;
};
using RetainedPrimalTensorViews = std::unordered_map<std::string, RetainedPrimalTensorView>;
struct RetainedPrimalTensorViewRef {
    const std::vector<uint64_t> *shape{};
    uint8_t *packed{};
};
using RetainedPrimalTensorViewRefs = std::unordered_map<std::string, RetainedPrimalTensorViewRef>;

struct OwnedAdValue {
    std::string path;
    VernonDataType dtype{};
    std::shared_ptr<std::vector<uint8_t>> bytes;
    std::vector<uint64_t> shape;
    VernonAdValue value{};

    OwnedAdValue() = default;
    explicit OwnedAdValue(const VernonAdValue &source, bool copyBytes = true)
        : path(source.path.data, source.path.size), dtype(source.dtype),
          bytes(std::make_shared<std::vector<uint8_t>>(source.size)) {
        if (source.rank)
            shape.assign(source.shape, source.shape + source.rank);
        if (copyBytes && source.size)
            std::memcpy(bytes->data(), source.data, source.size);
        refresh();
    }
    OwnedAdValue(OwnedAdValue &&other) noexcept
        : path(std::move(other.path)), dtype(other.dtype), bytes(std::move(other.bytes)),
          shape(std::move(other.shape)) {
        refresh();
    }
    OwnedAdValue &operator=(OwnedAdValue &&other) noexcept {
        path = std::move(other.path);
        dtype = other.dtype;
        bytes = std::move(other.bytes);
        shape = std::move(other.shape);
        refresh();
        return *this;
    }
    OwnedAdValue(const OwnedAdValue &) = delete;
    OwnedAdValue &operator=(const OwnedAdValue &) = delete;

    void refresh() {
        value = {sizeof(VernonAdValue),
                 {path.data(), path.size()},
                 dtype,
                 !bytes || bytes->empty() ? nullptr : bytes->data(),
                 bytes ? bytes->size() : 0,
                 static_cast<uint32_t>(shape.size()),
                 shape.empty() ? nullptr : shape.data()};
    }
};

struct OwnedAdValueSet {
    std::vector<OwnedAdValue> storage;
    std::vector<VernonAdValue> views;
    VernonAdValueSet set{};

    OwnedAdValueSet() = default;
    explicit OwnedAdValueSet(const VernonAdValueSet &source, bool copyBytes = true) {
        storage.reserve(source.value_count);
        std::unordered_map<const void *, std::shared_ptr<std::vector<uint8_t>>> sharedBytes;
        for (size_t index = 0; index < source.value_count; ++index) {
            storage.emplace_back(source.values[index], copyBytes);
            if (copyBytes) {
                const VernonAdValue &value = source.values[index];
                const auto [found, inserted] = sharedBytes.emplace(value.data, storage.back().bytes);
                if (!inserted && found->second->size() == value.size)
                    storage.back().bytes = found->second;
            }
        }
        refresh();
    }
    OwnedAdValueSet(OwnedAdValueSet &&other) noexcept : storage(std::move(other.storage)) { refresh(); }
    OwnedAdValueSet &operator=(OwnedAdValueSet &&other) noexcept {
        storage = std::move(other.storage);
        refresh();
        return *this;
    }
    OwnedAdValueSet(const OwnedAdValueSet &) = delete;
    OwnedAdValueSet &operator=(const OwnedAdValueSet &) = delete;

    void refresh() {
        views.clear();
        views.reserve(storage.size());
        for (OwnedAdValue &value : storage) {
            value.refresh();
            views.push_back(value.value);
        }
        set = {sizeof(VernonAdValueSet), views.empty() ? nullptr : views.data(), views.size(), {}};
    }

    size_t retainedBytes() const {
        size_t result = 0;
        std::unordered_set<const void *> counted;
        for (const OwnedAdValue &value : storage)
            if ((value.bytes && counted.insert(value.bytes.get()).second &&
                 !checkedAddBytes(result, value.bytes->capacity(), sizeof(uint8_t))) ||
                !checkedAddBytes(result, value.shape.capacity(), sizeof(uint64_t)))
                return std::numeric_limits<size_t>::max();
        return result;
    }
};

bool materializeTensorViewShape(const HostArgument &argument, const VernonAdValueSet &inputs,
                                std::vector<uint64_t> &shape) {
    const HostTensorView &view = *argument.tensorView;
    for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
        const ValueAbi &leaf = view.leaves[leafIndex];
        const ValueLeaf &layoutLeaf = view.elementLayout.leaves[leafIndex];
        const VernonAdValue *value = findValue(inputs, leaf.path);
        if (!value || value->dtype != leaf.dtype || value->rank != view.shape.size() + layoutLeaf.shape.size() ||
            (value->rank && !value->shape))
            return false;
        std::vector<uint64_t> candidate(value->shape, value->shape + view.shape.size());
        for (size_t dimension = 0; dimension < view.shape.size(); ++dimension)
            if (view.shape[dimension] >= 0 && candidate[dimension] != static_cast<uint64_t>(view.shape[dimension]))
                return false;
        for (size_t dimension = 0; dimension < layoutLeaf.shape.size(); ++dimension)
            if (value->shape[view.shape.size() + dimension] != layoutLeaf.shape[dimension])
                return false;
        size_t scalarCount = 1;
        for (uint64_t extent : candidate) {
            if (scalarCount && extent > SIZE_MAX / scalarCount)
                return false;
            scalarCount *= static_cast<size_t>(extent);
        }
        const size_t scalarSize = dtypeSize(leaf.dtype);
        if (!scalarSize || (scalarCount && layoutLeaf.scalarCount > SIZE_MAX / scalarCount))
            return false;
        const size_t leafScalarCount = scalarCount * layoutLeaf.scalarCount;
        if ((leafScalarCount && scalarSize > SIZE_MAX / leafScalarCount) || value->size != leafScalarCount * scalarSize)
            return false;
        if (shape.empty())
            shape = std::move(candidate);
        else if (shape != candidate)
            return false;
    }
    return !shape.empty();
}

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT);
bool writeTensorViewDescriptor(const HostArgument &argument, const std::vector<uint64_t> &shape, void *data,
                               uint8_t *arguments);

bool appendStorageRange(const void *data, size_t size, bool writable, std::vector<StorageRange> &ranges) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
    if (size > std::numeric_limits<uintptr_t>::max() - begin)
        return false;
    const StorageRange range{begin, begin + size, writable};
    for (const StorageRange &existing : ranges)
        if (hostByteRangesHaveWritableOverlap(reinterpret_cast<const void *>(range.begin), range.end - range.begin,
                                              range.writable, reinterpret_cast<const void *>(existing.begin),
                                              existing.end - existing.begin, existing.writable))
            return false;
    ranges.push_back(range);
    return true;
}

VernonStatus stageForwardInputs(VernonRuntimeContext &context, const HostProfileLayout &layout,
                                const VernonAdValueSet &inputs, std::vector<uint8_t> &arguments,
                                HostEffectTransaction &transaction, std::vector<StorageRange> &storageRanges,
                                std::vector<StagedTensorView> &tensorViews, bool preserveWriteOnly = false) {
    tensorViews.reserve(layout.arguments.size());
    for (const HostArgument &argument : layout.arguments) {
        if (!argument.builtin.empty())
            continue;
        if (argument.tensorView) {
            const HostTensorView &view = *argument.tensorView;
            if (view.leaves.size() != view.elementLayout.leaves.size())
                return fail(context, "autodiff forward TensorView has no canonical element layout");
            const bool writable = view.access != "read";
            const bool preserve = preserveWriteOnly || view.access != "write";
            StagedTensorView staged;
            staged.argument = &argument;
            staged.elementCount = 1;
            if (!materializeTensorViewShape(argument, inputs, staged.shape))
                return fail(context, "autodiff forward TensorView inputs do not match profile reflection");
            for (uint64_t extent : staged.shape) {
                if (staged.elementCount && extent > std::numeric_limits<size_t>::max() / staged.elementCount)
                    return fail(context, "autodiff forward TensorView element count overflows");
                staged.elementCount *= static_cast<size_t>(extent);
            }
            if (!view.elementLayout.byteSize ||
                staged.elementCount > std::numeric_limits<size_t>::max() / view.elementLayout.byteSize)
                return fail(context, "autodiff forward TensorView byte size overflows");
            try {
                staged.packed.resize(staged.elementCount * view.elementLayout.byteSize);
                staged.leafShadows.reserve(view.leaves.size());
            } catch (const std::bad_alloc &) {
                return fail(context, "cannot allocate native CPU autodiff TensorView shadow",
                            VERNON_STATUS_INTERNAL_ERROR);
            } catch (const std::length_error &) {
                return fail(context, "cannot allocate native CPU autodiff TensorView shadow",
                            VERNON_STATUS_INTERNAL_ERROR);
            }
            for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
                const ValueAbi &leaf = view.leaves[leafIndex];
                const VernonAdValue *value = findValue(inputs, leaf.path);
                if (!value)
                    return fail(context, "autodiff forward TensorView input '" + leaf.path +
                                             "' does not match profile reflection");
                if (!appendStorageRange(value->data, value->size, writable, storageRanges))
                    return fail(context, "observable native CPU autodiff Storage/output ranges overlap or overflow");
                auto *source = static_cast<const uint8_t *>(value->data);
                auto *shadow =
                    writable
                        ? static_cast<uint8_t *>(transaction.stageStorage(value->data, value->size, preserve, true))
                        : const_cast<uint8_t *>(source);
                if (value->size && !shadow)
                    return fail(context, "cannot allocate native CPU autodiff Storage shadow",
                                VERNON_STATUS_INTERNAL_ERROR);
                staged.leafShadows.push_back(shadow);
                if (!preserve)
                    continue;
                const ValueLeaf &layoutLeaf = view.elementLayout.leaves[leafIndex];
                const size_t scalarBytes = dtypeSize(leaf.dtype);
                if (!scalarBytes || layoutLeaf.scalarCount > SIZE_MAX / scalarBytes)
                    return fail(context, "autodiff TensorView leaf byte size overflows");
                const size_t leafBytes = layoutLeaf.scalarCount * scalarBytes;
                if (layoutLeaf.byteOffset > view.elementLayout.byteSize ||
                    leafBytes > view.elementLayout.byteSize - layoutLeaf.byteOffset)
                    return fail(context, "autodiff TensorView leaf exceeds its canonical element layout");
                const uint8_t *packedSource = writable ? shadow : source;
                for (size_t element = 0; element < staged.elementCount; ++element)
                    std::memcpy(staged.packed.data() + element * view.elementLayout.byteSize + layoutLeaf.byteOffset,
                                packedSource + element * leafBytes, leafBytes);
            }
            tensorViews.push_back(std::move(staged));
            if (!writeTensorViewDescriptor(argument, tensorViews.back().shape, tensorViews.back().packed.data(),
                                           arguments.data()))
                return fail(context, "autodiff forward TensorView descriptor overflows");
            continue;
        }
        for (const HostFrameLeaf &leaf : argument.leaves) {
            const VernonAdValue *value = findValue(inputs, leaf.value.path);
            if (!value || !valueMatches(*value, leaf.value))
                return fail(context,
                            "autodiff forward input '" + leaf.value.path +
                                "' does not match profile reflection (expected " + std::to_string(leaf.value.byteSize) +
                                " bytes at rank " + std::to_string(leaf.value.logicalShape.size()) + ", received " +
                                (value ? std::to_string(value->size) + " bytes at rank " + std::to_string(value->rank)
                                       : std::string("no value")) +
                                ")");
            std::memcpy(arguments.data() + leaf.frameOffset, value->data, value->size);
        }
    }
    return VERNON_STATUS_OK;
}

void restoreReplayReadWriteShadows(std::vector<StagedTensorView> &tensorViews) {
    for (StagedTensorView &staged : tensorViews) {
        if (!staged.argument || !staged.argument->tensorView || staged.argument->tensorView->access != "read_write")
            continue;
        const HostTensorView &view = *staged.argument->tensorView;
        for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
            const ValueLeaf &layoutLeaf = view.elementLayout.leaves[leafIndex];
            const size_t leafBytes = layoutLeaf.scalarCount * dtypeSize(view.leaves[leafIndex].dtype);
            const uint8_t *source = staged.leafShadows[leafIndex];
            for (size_t element = 0; element < staged.elementCount; ++element)
                std::memcpy(staged.packed.data() + element * view.elementLayout.byteSize + layoutLeaf.byteOffset,
                            source + element * leafBytes, leafBytes);
        }
    }
}

VernonStatus flushStagedTensorViews(VernonRuntimeContext &context, const std::vector<StagedTensorView> &tensorViews) {
    for (const StagedTensorView &staged : tensorViews) {
        if (!staged.argument || !staged.argument->tensorView || staged.argument->tensorView->access == "read" ||
            staged.argument->tensorView->leaves.size() != staged.leafShadows.size())
            continue;
        const HostTensorView &view = *staged.argument->tensorView;
        const ValueLayout &layout = view.elementLayout;
        for (size_t leafIndex = 0; leafIndex < view.leaves.size(); ++leafIndex) {
            const ValueAbi &leaf = view.leaves[leafIndex];
            const ValueLeaf &layoutLeaf = layout.leaves[leafIndex];
            const size_t scalarBytes = dtypeSize(leaf.dtype);
            if (!scalarBytes || layoutLeaf.scalarCount > SIZE_MAX / scalarBytes)
                return fail(context, "autodiff TensorView leaf byte size overflows");
            const size_t leafBytes = layoutLeaf.scalarCount * scalarBytes;
            for (size_t element = 0; element < staged.elementCount; ++element)
                std::memcpy(staged.leafShadows[leafIndex] + element * leafBytes,
                            staged.packed.data() + element * layout.byteSize + layoutLeaf.byteOffset, leafBytes);
        }
    }
    return VERNON_STATUS_OK;
}

bool writeInvocationBuiltin(const HostArgument &argument, const CpuLaneCoordinates &coordinates, uint8_t *arguments) {
    if ((argument.builtin != "global_invocation_id" && argument.builtin != "local_invocation_id" &&
         argument.builtin != "workgroup_id") ||
        argument.leaves.size() != 1 || argument.leaves.front().value.dtype != VERNON_DATA_U32 ||
        argument.leaves.front().value.logicalShape != std::vector<uint64_t>{3})
        return false;
    const uint32_t *value = argument.builtin == "global_invocation_id"
                                ? coordinates.global
                                : (argument.builtin == "local_invocation_id" ? coordinates.local : coordinates.group);
    std::memcpy(arguments + argument.leaves.front().frameOffset, value, sizeof(uint32_t) * 3);
    return true;
}

VernonStatus fail(VernonRuntimeContext &context, std::string message, VernonStatus status) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

const char *allocatorFailure(VernonAdTapeAllocatorStatus status) {
    switch (status) {
    case VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED:
        return "capacity exhausted";
    case VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW:
        return "arithmetic overflow";
    case VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE:
        return "host allocation failed";
    case VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE:
        return "invalid state";
    case VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI:
        return "invalid ABI";
    case VERNON_AD_TAPE_ALLOCATOR_OK:
        return "no failure";
    }
    return "unknown failure";
}

bool appendLeafPath(const nlohmann::json &value, std::string &path, std::string &error) {
    if (!value.contains("path") || !value["path"].is_array()) {
        error = "autodiff Value leaf path is invalid";
        return false;
    }
    for (const nlohmann::json &component : value["path"]) {
        std::string name;
        if (component.is_string())
            name = component.get<std::string>();
        else if (component.is_number_unsigned())
            name = std::to_string(component.get<uint64_t>());
        else {
            error = "autodiff Value leaf path is invalid";
            return false;
        }
        if (name.empty() || name.find('.') != std::string::npos) {
            error = "autodiff Value leaf path is invalid";
            return false;
        }
        path += "." + name;
    }
    return true;
}

bool parseLeaf(const nlohmann::json &value, const std::string &path, size_t baseOffset, HostFrameLeaf &leaf,
               std::string &error) {
    if (!value.is_object() || !value.contains("dtype") || !value["dtype"].is_string() ||
        !value.contains("byte_offset") || !value["byte_offset"].is_number_unsigned() ||
        !value.contains("scalar_count") || !value["scalar_count"].is_number_unsigned()) {
        error = "autodiff profile contains an invalid Value leaf";
        return false;
    }
    const auto dtype = pipelineDataType(value["dtype"].get<std::string>());
    const size_t scalarSize = dtype ? dtypeSize(*dtype) : 0;
    const size_t scalarCount = value["scalar_count"].get<size_t>();
    if (!scalarSize || !scalarCount || scalarCount > SIZE_MAX / scalarSize) {
        error = "autodiff Runtime profile contains an unsupported or empty Value leaf";
        return false;
    }
    const size_t relativeOffset = value["byte_offset"].get<size_t>();
    if (relativeOffset > SIZE_MAX - baseOffset) {
        error = "autodiff scalar leaf offset overflows";
        return false;
    }
    std::vector<uint64_t> shape;
    if (value.contains("shape")) {
        if (!value["shape"].is_array()) {
            error = "autodiff Value leaf shape is invalid";
            return false;
        }
        size_t shapedCount = 1;
        for (const nlohmann::json &extentValue : value["shape"]) {
            if (!extentValue.is_number_unsigned()) {
                error = "autodiff Value leaf shape is invalid";
                return false;
            }
            const uint64_t extent = extentValue.get<uint64_t>();
            if (!extent || extent > SIZE_MAX / shapedCount) {
                error = "autodiff Value leaf shape is empty or overflows";
                return false;
            }
            shapedCount *= static_cast<size_t>(extent);
            shape.push_back(extent);
        }
        if (shapedCount != scalarCount) {
            error = "autodiff Value leaf shape does not match scalar_count";
            return false;
        }
    } else if (scalarCount != 1) {
        error = "autodiff non-scalar Value leaf has no shape";
        return false;
    }
    std::string leafPath = path;
    if (!appendLeafPath(value, leafPath, error))
        return false;
    leaf = {{std::move(leafPath), *dtype, scalarCount * scalarSize, scalarSize, std::move(shape)},
            baseOffset + relativeOffset};
    return true;
}

bool parseProfile(const Stage &stage, HostProfileLayout &layout, std::string &error) {
    const nlohmann::json root = nlohmann::json::parse(stage.reflection, nullptr, false);
    if (root.is_discarded()) {
        error = "autodiff artifact reflection is invalid JSON";
        return false;
    }
    const nlohmann::json *entry = nullptr;
    if (root.contains("entries") && root["entries"].is_array())
        for (const nlohmann::json &candidate : root["entries"])
            if (candidate.is_object() && candidate.value("name", "") == stage.entry) {
                entry = &candidate;
                break;
            }
    if (!entry || !entry->contains("arguments") || !(*entry)["arguments"].is_array() || !entry->contains("results") ||
        !(*entry)["results"].is_array() || !entry->contains("physical_layouts") ||
        !(*entry)["physical_layouts"].is_object()) {
        error = "autodiff artifact reflection has no selected profile ABI";
        return false;
    }
    const auto host = (*entry)["physical_layouts"].find("host_value");
    if (host == (*entry)["physical_layouts"].end() || !host->is_object() || !host->contains("packed_arguments_size") ||
        !(*host)["packed_arguments_size"].is_number_unsigned() || !host->contains("packed_results_size") ||
        !(*host)["packed_results_size"].is_number_unsigned()) {
        error = "autodiff profile has no host Value ABI";
        return false;
    }
    layout.argumentsSize = (*host)["packed_arguments_size"].get<size_t>();
    layout.resultsSize = (*host)["packed_results_size"].get<size_t>();
    if (!entry->contains("workgroup_size") || !(*entry)["workgroup_size"].is_array() ||
        (*entry)["workgroup_size"].size() != 3) {
        error = "autodiff profile has no workgroup size";
        return false;
    }
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        const nlohmann::json &extent = (*entry)["workgroup_size"][dimension];
        if (!extent.is_number_unsigned() || !extent.get<uint32_t>()) {
            error = "autodiff profile has an invalid workgroup size";
            return false;
        }
        layout.workgroup[dimension] = extent.get<uint32_t>();
    }
    if (!parseDispatchContract(*entry, layout.dispatchContract, error))
        return false;
    if (!layout.argumentsSize) {
        error = "autodiff profile has an empty host argument ABI";
        return false;
    }
    for (const nlohmann::json &value : (*entry)["arguments"]) {
        if (!value.is_object() || !value.contains("physical_layouts") || !value["physical_layouts"].is_object()) {
            error = "autodiff profile contains an invalid argument ABI";
            return false;
        }
        const auto physical = value["physical_layouts"].find("host_value");
        const std::string physicalKind =
            physical != value["physical_layouts"].end() && physical->is_object() ? physical->value("kind", "") : "";
        const nlohmann::json *physicalExtent =
            physicalKind == "cpu_call" && physical->contains("root") && (*physical)["root"].is_object()
                ? &(*physical)["root"]
                : (physicalKind == "resource_binding" ? &*physical : nullptr);
        if (physical == value["physical_layouts"].end() || !physical->is_object() ||
            !physical->contains("frame_offset") || !(*physical)["frame_offset"].is_number_unsigned() ||
            !physicalExtent || !physicalExtent->contains("size") || !(*physicalExtent)["size"].is_number_unsigned()) {
            error = "autodiff argument has no host Value layout";
            return false;
        }
        HostArgument argument;
        argument.name = value.value("vernon.source_name", "");
        argument.builtin = value.value("builtin", "");
        argument.accumulationOwnership = value.value("vernon.accumulation_ownership", "");
        if (!argument.accumulationOwnership.empty() && argument.accumulationOwnership != "invocation_private" &&
            argument.accumulationOwnership != "atomic_shared" && argument.accumulationOwnership != "workgroup_shared" &&
            argument.accumulationOwnership != "none") {
            error = "autodiff argument has unsupported accumulation ownership";
            return false;
        }
        argument.offset = (*physical)["frame_offset"].get<size_t>();
        argument.size = (*physicalExtent)["size"].get<size_t>();
        const bool isTensorView = physical->value("kind", "") == "resource_binding" &&
                                  physical->value("resource_kind", "") == "tensor_view_descriptor";
        if (argument.offset > layout.argumentsSize || argument.size > layout.argumentsSize - argument.offset) {
            error = "autodiff argument host Value layout is out of bounds";
            return false;
        }
        if (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
            if (value.value("kind", "") != "builtin" || argument.size != sizeof(VernonAdTapeAllocator *) ||
                layout.tapeAllocatorOffset) {
                error = "autodiff tape allocator builtin has an invalid host ABI";
                return false;
            }
            layout.tapeAllocatorOffset = argument.offset;
            layout.arguments.push_back(std::move(argument));
            continue;
        }
        if (argument.builtin == "ad_tape_root_region") {
            if (value.value("kind", "") != "builtin" || argument.size != sizeof(VernonAdRegionHandle) ||
                layout.tapeRootRegionOffset) {
                error = "autodiff root tape region builtin has an invalid host ABI";
                return false;
            }
            layout.tapeRootRegionOffset = argument.offset;
            layout.arguments.push_back(std::move(argument));
            continue;
        }
        if (argument.name.empty()) {
            error = "autodiff argument has no source name";
            return false;
        }
        if (isTensorView) {
            if (!value.contains("source_shape") || !value["source_shape"].is_array() ||
                !value.contains("element_layout") || !value["element_layout"].is_object()) {
                error = "autodiff TensorView argument has an invalid element ABI";
                return false;
            }
            ValueLayout elementLayout;
            if (!parsePipelineValueLayout(value["element_layout"], elementLayout, error) ||
                elementLayout.leaves.empty() || !elementLayout.byteSize) {
                if (error.empty())
                    error = "autodiff TensorView argument has an invalid canonical element layout";
                return false;
            }
            std::vector<int64_t> shape;
            const std::string access = value.value("access", "");
            for (const nlohmann::json &extentValue : value["source_shape"]) {
                if (!extentValue.is_number_integer()) {
                    error = "autodiff TensorView argument has an invalid shape";
                    return false;
                }
                const int64_t extent = extentValue.get<int64_t>();
                if (!extent || extent < -1) {
                    error = "autodiff TensorView argument shape is empty or invalid";
                    return false;
                }
                shape.push_back(extent);
            }
            if (shape.empty() || argument.size != sizeof(uint64_t) * (2 + 2 * shape.size()) ||
                (access != "read" && access != "write" && access != "read_write")) {
                error = "autodiff TensorView argument has an invalid host descriptor";
                return false;
            }
            HostTensorView tensorView;
            tensorView.access = access;
            tensorView.shape = shape;
            for (const ValueLeaf &layoutLeaf : elementLayout.leaves) {
                const auto dtype = pipelineDataType(layoutLeaf.dtype);
                const size_t scalarSize = dtype ? dtypeSize(*dtype) : 0;
                if (!scalarSize || !layoutLeaf.scalarCount) {
                    error = "autodiff TensorView element leaf size overflows";
                    return false;
                }
                std::string path = argument.name;
                for (const ValuePathComponent &component : layoutLeaf.path)
                    path += component.field ? "." + *component.field : "." + std::to_string(component.index);
                std::vector<uint64_t> leafShape;
                leafShape.reserve(shape.size() + layoutLeaf.shape.size());
                for (int64_t extent : shape)
                    leafShape.push_back(extent < 0 ? 0 : static_cast<uint64_t>(extent));
                leafShape.insert(leafShape.end(), layoutLeaf.shape.begin(), layoutLeaf.shape.end());
                size_t byteSize = layoutLeaf.scalarCount * scalarSize;
                for (int64_t extent : shape) {
                    if (extent < 0) {
                        byteSize = 0;
                        break;
                    }
                    if (static_cast<uint64_t>(extent) > SIZE_MAX / byteSize) {
                        error = "autodiff TensorView element leaf size overflows";
                        return false;
                    }
                    byteSize *= static_cast<size_t>(extent);
                }
                tensorView.leaves.push_back({std::move(path), *dtype, byteSize, scalarSize, std::move(leafShape)});
            }
            tensorView.elementLayout = std::move(elementLayout);
            argument.tensorView = std::move(tensorView);
            layout.arguments.push_back(std::move(argument));
            continue;
        }
        if (!value.contains("value_layout") || !value["value_layout"].is_object() ||
            !value["value_layout"].contains("leaves") || !value["value_layout"]["leaves"].is_array()) {
            error = "autodiff profile contains an invalid argument Value ABI";
            return false;
        }
        for (const nlohmann::json &leafValue : value["value_layout"]["leaves"]) {
            HostFrameLeaf leaf;
            if (!parseLeaf(leafValue, argument.name, argument.offset, leaf, error))
                return false;
            if (leaf.frameOffset < argument.offset || leaf.frameOffset > argument.offset + argument.size ||
                leaf.value.byteSize > argument.offset + argument.size - leaf.frameOffset) {
                error = "autodiff argument Value leaf is out of bounds";
                return false;
            }
            argument.leaves.push_back(std::move(leaf));
        }
        if (argument.leaves.empty()) {
            error = "autodiff argument contains no Value leaves";
            return false;
        }
        layout.arguments.push_back(std::move(argument));
    }
    if ((*entry)["results"].empty())
        return true;
    if ((*entry)["results"].size() != 1) {
        error = "CPU autodiff profile supports at most one result";
        return false;
    }
    const nlohmann::json &result = (*entry)["results"][0];
    if (!result.is_object() || !result.contains("physical_layouts") || !result["physical_layouts"].is_object() ||
        !result.contains("value_layout") || !result["value_layout"].is_object() ||
        !result["value_layout"].contains("leaves") || !result["value_layout"]["leaves"].is_array()) {
        error = "autodiff profile contains an invalid result ABI";
        return false;
    }
    const auto physical = result["physical_layouts"].find("host_value");
    if (physical == result["physical_layouts"].end() || !physical->is_object() || !physical->contains("frame_offset") ||
        !(*physical)["frame_offset"].is_number_unsigned() || physical->value("kind", "") != "cpu_call" ||
        !physical->contains("root") || !(*physical)["root"].is_object() || !(*physical)["root"].contains("size") ||
        !(*physical)["root"]["size"].is_number_unsigned()) {
        error = "autodiff result has no host Value layout";
        return false;
    }
    const size_t resultOffset = (*physical)["frame_offset"].get<size_t>();
    const size_t resultSize = (*physical)["root"]["size"].get<size_t>();
    if (resultOffset > layout.resultsSize || resultSize > layout.resultsSize - resultOffset) {
        error = "autodiff result host Value layout is out of bounds";
        return false;
    }
    for (const nlohmann::json &leafValue : result["value_layout"]["leaves"]) {
        HostFrameLeaf leaf;
        if (!parseLeaf(leafValue, "output", resultOffset, leaf, error))
            return false;
        if (leaf.frameOffset < resultOffset || leaf.frameOffset > resultOffset + resultSize ||
            leaf.value.byteSize > resultOffset + resultSize - leaf.frameOffset) {
            error = "autodiff result Value leaf is out of bounds";
            return false;
        }
        layout.results.push_back(std::move(leaf));
    }
    if (layout.results.empty()) {
        error = "autodiff profile result contains no scalar leaves";
        return false;
    }
    return true;
}

VernonStatus prepareGradientDestinations(VernonRuntimeContext &context, const Signature &signature,
                                         VernonAdValueSet &gradients, std::vector<VernonAdValue *> &destinations,
                                         std::vector<std::vector<uint8_t>> &stagedGradients) {
    if (gradients.value_count != signature.gradients.size())
        return fail(context, "invalid pullback invocation");
    destinations.reserve(signature.gradients.size());
    stagedGradients.reserve(signature.gradients.size());
    for (const ValueAbi &expected : signature.gradients) {
        VernonAdValue *gradient = findValue(gradients, expected.path);
        if (!gradient || !valueMatches(*gradient, expected))
            return fail(context, "gradient output does not match backward reflection");
        destinations.push_back(gradient);
    }
    try {
        for (const ValueAbi &expected : signature.gradients)
            stagedGradients.emplace_back(expected.byteSize, uint8_t{0});
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate staged pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(context, "cannot allocate staged pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
    }
    return VERNON_STATUS_OK;
}

bool writeTensorViewDescriptor(const HostArgument &argument, const std::vector<uint64_t> &shape, void *data,
                               uint8_t *arguments) {
    if (!argument.tensorView || shape.size() != argument.tensorView->shape.size() ||
        argument.size != sizeof(uint64_t) * (2 + 2 * shape.size()))
        return false;
    uint8_t *descriptor = arguments + argument.offset;
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);
    const uint64_t zero = 0;
    std::memcpy(descriptor, &pointer, sizeof(pointer));
    std::memcpy(descriptor + sizeof(uint64_t), &zero, sizeof(zero));
    uint64_t stride = 1;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (stride && shape[dimension] > UINT64_MAX / stride)
            return false;
        std::memcpy(descriptor + sizeof(uint64_t) * (2 + dimension), &shape[dimension], sizeof(uint64_t));
        std::memcpy(descriptor + sizeof(uint64_t) * (2 + shape.size() + dimension), &stride, sizeof(uint64_t));
        stride *= shape[dimension];
    }
    return true;
}

bool tensorViewDescriptorShape(const HostArgument &argument, const std::vector<uint64_t> &logicalShape,
                               std::vector<uint64_t> &descriptorShape) {
    if (!argument.tensorView)
        return false;
    const size_t rank = argument.tensorView->shape.size();
    if (logicalShape.size() < rank)
        return false;
    descriptorShape.assign(logicalShape.begin(), logicalShape.begin() + rank);
    for (size_t dimension = 0; dimension < rank; ++dimension)
        if (argument.tensorView->shape[dimension] >= 0 &&
            descriptorShape[dimension] != static_cast<uint64_t>(argument.tensorView->shape[dimension]))
            return false;
    return std::any_of(argument.tensorView->elementLayout.leaves.begin(),
                       argument.tensorView->elementLayout.leaves.end(), [&](const ValueLeaf &leaf) {
                           return leaf.shape.size() == logicalShape.size() - rank &&
                                  std::equal(leaf.shape.begin(), leaf.shape.end(), logicalShape.begin() + rank);
                       });
}

bool isShapeSource(const HostArgument &argument) { return argument.name.rfind("shape.", 0) == 0; }
bool isPrimalSource(const HostArgument &argument) { return argument.name.rfind("primal.", 0) == 0; }

std::string tensorOwnerName(const HostArgument &argument) {
    std::string name = isShapeSource(argument) ? argument.name.substr(6) : argument.name;
    const size_t separator = name.find('.');
    if (separator != std::string::npos)
        name.resize(separator);
    return name;
}

bool materializeRuntimeSignature(Signature &signature, const VernonAdValueSet &inputs) {
    for (ValueAbi &input : signature.inputs) {
        const VernonAdValue *value = findValue(inputs, input.path);
        if (!value || value->dtype != input.dtype || (value->rank && !value->shape))
            return false;
        input.byteSize = value->size;
        input.logicalShape.assign(value->shape, value->shape + value->rank);
    }
    auto materializeDerivative = [&](ValueAbi &derivative) {
        const auto primal = std::find_if(signature.inputs.begin(), signature.inputs.end(),
                                         [&](const ValueAbi &value) { return value.path == derivative.path; });
        if (primal == signature.inputs.end())
            return false;
        const size_t primalScalarSize = dtypeSize(primal->dtype);
        const size_t derivativeScalarSize = dtypeSize(derivative.dtype);
        if (!primalScalarSize || !derivativeScalarSize || primal->byteSize % primalScalarSize ||
            primal->byteSize / primalScalarSize > SIZE_MAX / derivativeScalarSize)
            return false;
        derivative.byteSize = primal->byteSize / primalScalarSize * derivativeScalarSize;
        derivative.logicalShape = primal->logicalShape;
        return true;
    };
    for (ValueAbi &output : signature.outputs)
        if (!materializeDerivative(output))
            return false;
    for (ValueAbi &cotangent : signature.cotangents)
        if (!materializeDerivative(cotangent))
            return false;
    for (ValueAbi &gradient : signature.gradients)
        if (!materializeDerivative(gradient))
            return false;
    return true;
}

VernonStatus accumulateFloatingBytes(VernonDataType dtype, uint8_t *destination, const uint8_t *source,
                                     size_t byteSize) {
    auto accumulate = [&](auto scalar) {
        using Scalar = decltype(scalar);
        if (byteSize % sizeof(Scalar))
            return false;
        for (size_t offset = 0; offset < byteSize; offset += sizeof(Scalar)) {
            Scalar current;
            Scalar contribution;
            std::memcpy(&current, destination + offset, sizeof(Scalar));
            std::memcpy(&contribution, source + offset, sizeof(Scalar));
            current += contribution;
            std::memcpy(destination + offset, &current, sizeof(Scalar));
        }
        return true;
    };
    if ((dtype == VERNON_DATA_F32 && accumulate(float{})) || (dtype == VERNON_DATA_F64 && accumulate(double{})))
        return VERNON_STATUS_OK;
    return VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus accumulateBackwardResults(const HostProfileLayout &layout, const Signature &signature,
                                       const uint8_t *results, const std::vector<size_t> &resultGradientIndices,
                                       uint8_t *privateGradients, const std::vector<size_t> &privateGradientOffsets) {
    if (layout.results.size() != resultGradientIndices.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < resultGradientIndices.size(); ++index) {
        const size_t gradientIndex = resultGradientIndices[index];
        const HostFrameLeaf &source = layout.results[index];
        if (gradientIndex >= privateGradientOffsets.size() ||
            privateGradientOffsets[gradientIndex] == std::numeric_limits<size_t>::max())
            return VERNON_STATUS_INVALID_ARGUMENT;
        uint8_t *destination = privateGradients + privateGradientOffsets[gradientIndex];
        const uint8_t *contribution = results + source.frameOffset;
        if (VernonStatus accumulateStatus =
                accumulateFloatingBytes(source.value.dtype, destination, contribution, source.value.byteSize);
            accumulateStatus != VERNON_STATUS_OK)
            return accumulateStatus;
    }
    return VERNON_STATUS_OK;
}

VernonStatus accumulateGradientBytes(VernonRuntimeContext &context, const ValueAbi &abi, const uint8_t *source,
                                     std::vector<uint8_t> &destination) {
    const size_t scalarSize = dtypeSize(abi.dtype);
    if (!scalarSize || destination.size() != abi.byteSize)
        return fail(context, "CPU pullback gradient accumulation ABI is inconsistent");
    const VernonStatus status = accumulateFloatingBytes(abi.dtype, destination.data(), source, abi.byteSize);
    return status == VERNON_STATUS_OK
               ? status
               : fail(context, "CPU pullback can only accumulate well-formed floating gradients", status);
}

void commitGradientDestinations(const std::vector<VernonAdValue *> &destinations,
                                const std::vector<std::vector<uint8_t>> &stagedGradients) {
    for (size_t index = 0; index < destinations.size(); ++index)
        if (!stagedGradients[index].empty())
            std::memcpy(destinations[index]->data, stagedGradients[index].data(), stagedGradients[index].size());
}

enum class CpuResidualStorage {
    None,
    Static,
    Dynamic,
};

enum class CpuPlanningPolicy {
    MinMemory,
    Balanced,
    MinRuntime,
};

struct CpuResidualPlan {
    CpuResidualStorage storage{};
    CpuPlanningPolicy policy{};
    size_t staticTapeStride{};
    bool permitsWholeDispatchRetention{};
};

struct CpuAutodiffProgram {
    HostProfileLayout primalLayout;
    HostProfileLayout forwardLayout;
    HostProfileLayout backwardLayout;
    std::shared_ptr<CpuKernelState> primal;
    std::shared_ptr<CpuKernelState> forward;
    std::shared_ptr<CpuKernelState> backward;
    Signature signature;
    std::vector<size_t> resultGradientIndices;
    std::unordered_set<std::string> requiredPrimalTensorOwners;
};

struct CpuBackwardOwnedState {
    Signature signature;
    RuntimeTensorShapes tensorShapes;
    RetainedPrimalLeaves retainedPrimals;
    RetainedPrimalTensorViews retainedTensorViews;
    RetainedPrimalTensorViewRefs retainedTensorViewRefs;

    CpuBackwardOwnedState(Signature runtimeSignature, RuntimeTensorShapes shapes, RetainedPrimalLeaves primals,
                          RetainedPrimalTensorViews tensorViews)
        : signature(std::move(runtimeSignature)), tensorShapes(std::move(shapes)), retainedPrimals(std::move(primals)),
          retainedTensorViews(std::move(tensorViews)) {
        retainedTensorViewRefs.reserve(retainedTensorViews.size());
        for (auto &[owner, view] : retainedTensorViews)
            retainedTensorViewRefs.emplace(owner, RetainedPrimalTensorViewRef{&view.shape, view.packed.data()});
    }
};

struct CpuBackwardBorrowedState {
    std::reference_wrapper<const Signature> signature;
    std::reference_wrapper<const RuntimeTensorShapes> tensorShapes;
    std::reference_wrapper<const RetainedPrimalLeaves> retainedPrimals;
    std::reference_wrapper<const RetainedPrimalTensorViewRefs> retainedTensorViews;
};

class CpuBackwardExecutor {
public:
    CpuBackwardExecutor(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                        Signature signature, VernonLaunchSize computeGrid,
                        std::shared_ptr<HostStaticTapeBatch> tapeBatch, RuntimeTensorShapes tensorShapes,
                        RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews,
                        std::optional<size_t> groupLinear = std::nullopt)
        : context_(context), program_(std::move(program)),
          state_(std::in_place_type<CpuBackwardOwnedState>, std::move(signature), std::move(tensorShapes),
                 std::move(retainedPrimals), std::move(retainedTensorViews)),
          computeGrid_(computeGrid), tapeBatch_(std::move(tapeBatch)), groupLinear_(groupLinear) {}

    CpuBackwardExecutor(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                        CpuBackwardBorrowedState state, VernonLaunchSize computeGrid,
                        std::shared_ptr<HostStaticTapeBatch> tapeBatch, size_t groupLinear)
        : context_(context), program_(std::move(program)), state_(std::move(state)), computeGrid_(computeGrid),
          tapeBatch_(std::move(tapeBatch)), groupLinear_(groupLinear) {}

    PullbackMemoryUsage memoryUsage() const {
        PullbackMemoryUsage usage;
        if (tapeBatch_) {
            usage.logicalResidualBytes = tapeBatch_->logicalBytes();
            usage.residentBytes = tapeBatch_->residentBytes();
            usage.allocatedBytes = tapeBatch_->allocatedBytes();
        }
        size_t retainedBytes = usage.allocatedBytes;
        const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_);
        if (!owned) {
            usage.retainedAllocationBytes = retainedBytes;
            return usage;
        }
        for (const auto &retained : owned->retainedPrimals)
            if (!checkedAddBytes(retainedBytes, retained.second.capacity(), sizeof(uint8_t))) {
                usage.retainedAllocationBytes = std::numeric_limits<size_t>::max();
                return usage;
            }
        for (const auto &retained : owned->retainedTensorViews)
            if (!checkedAddBytes(retainedBytes, retained.second.shape.capacity(), sizeof(uint64_t)) ||
                !checkedAddBytes(retainedBytes, retained.second.packed.capacity(), sizeof(uint8_t))) {
                usage.retainedAllocationBytes = std::numeric_limits<size_t>::max();
                return usage;
            }
        usage.retainedAllocationBytes = retainedBytes;
        return usage;
    }

    VernonStatus applyInto(const VernonAdValueSet *cotangents, std::vector<std::vector<uint8_t>> &stagedGradients) {
        if (!groupLinear_ || externalStagedGradients_)
            return fail(context_, "CPU segmented pullback accumulation is unavailable", VERNON_STATUS_INTERNAL_ERROR);
        externalStagedGradients_ = &stagedGradients;
        VernonAdValueSet unused{sizeof(VernonAdValueSet), nullptr, 0, {}};
        const VernonStatus status = apply(cotangents, unused);
        externalStagedGradients_ = nullptr;
        return status;
    }

    std::shared_ptr<HostStaticTapeBatch> releaseTapeBatch() { return std::move(tapeBatch_); }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) {
        const Signature *const signature_ = &runtimeSignature();
        const RuntimeTensorShapes *const tensorShapes_ = &runtimeTensorShapes();
        const RetainedPrimalLeaves *const retainedPrimals_ = &retainedPrimalLeaves();
        const RetainedPrimalTensorViewRefs *const retainedTensorViews_ = &retainedPrimalTensorViews();
        const uint32_t grid[3]{computeGrid_.x, computeGrid_.y, computeGrid_.z};
        const HostProfileLayout &layout = program_->backwardLayout;
        if (!validateDispatchContract(layout.dispatchContract, grid, layout.workgroup, invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        VernonLaunchSize extent{};
        if (!invocationExtent(computeGrid_, {layout.workgroup[0], layout.workgroup[1], layout.workgroup[2]}, extent))
            return fail(context_, "CPU pullback invocation extent overflows");
        if (signature_->cotangents.empty())
            return fail(context_, "CPU pullback has no cotangent leaves");
        if ((cotangents && cotangents->value_count != signature_->cotangents.size()) ||
            (!cotangents && signature_->cotangents.size() != 1))
            return fail(context_, "CPU pullback requires exactly the reflected output cotangent leaves");
        std::unordered_map<std::string_view, size_t> gradientIndices;
        gradientIndices.reserve(signature_->gradients.size());
        for (size_t index = 0; index < signature_->gradients.size(); ++index)
            gradientIndices.emplace(signature_->gradients[index].path, index);
        std::vector<std::string> gradientOwnership(signature_->gradients.size());
        std::vector<bool> storageGradients(signature_->gradients.size());
        for (size_t gradientIndex : program_->resultGradientIndices) {
            if (gradientIndex >= gradientOwnership.size())
                return fail(context_, "CPU pullback result gradient index is invalid");
            gradientOwnership[gradientIndex] = "invocation_private";
        }
        for (const HostArgument &argument : layout.arguments)
            if (argument.tensorView && !isShapeSource(argument) && !isPrimalSource(argument))
                if (const auto gradient = gradientIndices.find(argument.name); gradient != gradientIndices.end()) {
                    gradientOwnership[gradient->second] = argument.accumulationOwnership;
                    storageGradients[gradient->second] = true;
                }
        if (std::any_of(gradientOwnership.begin(), gradientOwnership.end(),
                        [](const std::string &ownership) { return ownership.empty(); }))
            return fail(context_, "CPU pullback gradient has no reflected ownership");
        size_t invocationCount = 0;
        if (!carrierCount(extent, invocationCount))
            return fail(context_, "CPU pullback dispatch size overflows");
        size_t workgroupVolume = 1;
        for (uint32_t dimension : layout.workgroup) {
            if (workgroupVolume > SIZE_MAX / dimension)
                return fail(context_, "CPU pullback workgroup size overflows");
            workgroupVolume *= dimension;
        }
        const bool segmented = groupLinear_.has_value();
        const size_t frameCount = segmented ? workgroupVolume : invocationCount;
        const bool inlineExecution = segmented || invocationCount <= kCpuAdInlineInvocationLimit;
        std::vector<bool> sharedCotangents(signature_->cotangents.size());
        if (cotangents) {
            for (size_t index = 0; index < signature_->cotangents.size(); ++index) {
                const ValueAbi &logical = signature_->cotangents[index];
                ValueAbi carried = logical;
                if (!materializeCarrierValue(carried, extent))
                    return fail(context_, "CPU pullback cotangent size overflows");
                const VernonAdValue *value = findValue(*cotangents, logical.path);
                if (!value)
                    return fail(context_, "output cotangent does not match backward reflection");
                if (valueMatches(*value, logical))
                    sharedCotangents[index] = true;
                else if (!valueMatches(*value, carried))
                    return fail(context_, "output cotangent does not match backward reflection");
            }
        }
        size_t requiredGradientBytes = 0;
        for (size_t index = 0; index < signature_->cotangents.size(); ++index)
            if (!checkedAddBytes(requiredGradientBytes, sharedCotangents[index] ? 1 : invocationCount,
                                 signature_->cotangents[index].byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
        for (size_t gradientIndex = 0; gradientIndex < signature_->gradients.size(); ++gradientIndex) {
            const size_t byteSize = signature_->gradients[gradientIndex].byteSize;
            if (!checkedAddBytes(requiredGradientBytes, 1, byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
            const size_t carrierCount =
                storageGradients[gradientIndex] || gradientOwnership[gradientIndex] == "none" ? 0
                : gradientOwnership[gradientIndex] == "invocation_private" ? (inlineExecution ? 1 : frameCount)
                                                                           : 0;
            if (!storageGradients[gradientIndex] && gradientOwnership[gradientIndex] != "none" &&
                gradientOwnership[gradientIndex] != "invocation_private")
                return fail(context_, "CPU pullback non-Storage gradient requires invocation-private ownership");
            if (!checkedAddBytes(requiredGradientBytes, carrierCount, byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
        }
        if (requiredGradientBytes > kCpuAdGradientDispatchLimit)
            return fail(context_, "CPU pullback requires " + std::to_string(requiredGradientBytes) +
                                      " gradient bytes, exceeding the " + std::to_string(kCpuAdGradientDispatchLimit) +
                                      "-byte dispatch limit");
        std::vector<const HostArgument *> cotangentArguments;
        for (const HostArgument &argument : layout.arguments)
            if (argument.builtin.empty() && !isShapeSource(argument) && !isPrimalSource(argument) &&
                gradientIndices.find(argument.name) == gradientIndices.end())
                cotangentArguments.push_back(&argument);
        if (cotangentArguments.size() != signature_->cotangents.size())
            return fail(context_, "CPU pullback has an invalid cotangent ABI");
        std::vector<std::vector<uint8_t>> cotangentBytes;
        std::vector<const uint8_t *> directCotangents(signature_->cotangents.size());
        for (size_t index = 0; index < signature_->cotangents.size(); ++index) {
            const HostArgument &argument = *cotangentArguments[index];
            const size_t leafCount = argument.tensorView ? argument.tensorView->leaves.size() : argument.leaves.size();
            if (leafCount != 1 || argument.name != signature_->cotangents[index].path)
                return fail(context_, "CPU pullback cotangent reflection is inconsistent");
            ValueAbi abi = signature_->cotangents[index];
            if (!materializeCarrierValue(abi, extent))
                return fail(context_, "CPU pullback cotangent size overflows");
            const size_t byteSize = sharedCotangents[index] ? signature_->cotangents[index].byteSize : abi.byteSize;
            if (segmented && cotangents && !sharedCotangents[index]) {
                const VernonAdValue *value = findValue(*cotangents, abi.path);
                if (!value || value->size != byteSize)
                    return fail(context_, "output cotangent does not match backward reflection");
                directCotangents[index] = static_cast<const uint8_t *>(value->data);
                cotangentBytes.emplace_back();
                continue;
            }
            std::vector<uint8_t> bytes(byteSize);
            if (cotangents) {
                const VernonAdValue *value = findValue(*cotangents, abi.path);
                if (!value || value->size != byteSize)
                    return fail(context_, "output cotangent does not match backward reflection");
                std::memcpy(bytes.data(), value->data, byteSize);
            } else {
                if (signature_->cotangents.size() != 1 ||
                    !makeCotangentBytes(nullptr, abi, bytes, invocationDiagnostic(context_)))
                    return VERNON_STATUS_INVALID_ARGUMENT;
            }
            cotangentBytes.push_back(std::move(bytes));
        }
        std::vector<VernonAdValue *> destinations;
        std::vector<std::vector<uint8_t>> localStagedGradients;
        std::vector<std::vector<uint8_t>> &stagedGradients =
            externalStagedGradients_ ? *externalStagedGradients_ : localStagedGradients;
        if (!externalStagedGradients_)
            if (VernonStatus status =
                    prepareGradientDestinations(context_, *signature_, gradients, destinations, stagedGradients);
                status != VERNON_STATUS_OK)
                return status;
        if (stagedGradients.size() != signature_->gradients.size())
            return fail(context_, "CPU segmented pullback gradient staging is inconsistent",
                        VERNON_STATUS_INTERNAL_ERROR);
        size_t argumentFrameBytes = 0;
        size_t resultFrameBytes = 0;
        if (!checkedAddBytes(argumentFrameBytes, frameCount, layout.argumentsSize) ||
            !checkedAddBytes(resultFrameBytes, frameCount, layout.resultsSize))
            return fail(context_, "CPU pullback invocation frame size overflows");
        std::vector<uint8_t> packedArguments(argumentFrameBytes);
        std::vector<uint8_t> packedResults(resultFrameBytes);
        std::vector<const void *> laneArguments(frameCount);
        std::vector<void *> laneResults(frameCount);
        std::vector<size_t> privateGradientOffsets(signature_->gradients.size(), std::numeric_limits<size_t>::max());
        size_t privateGradientStride = 0;
        for (size_t gradientIndex = 0; gradientIndex < signature_->gradients.size(); ++gradientIndex) {
            if (storageGradients[gradientIndex] || gradientOwnership[gradientIndex] != "invocation_private")
                continue;
            privateGradientOffsets[gradientIndex] = privateGradientStride;
            if (!checkedAddBytes(privateGradientStride, 1, signature_->gradients[gradientIndex].byteSize))
                return fail(context_, "CPU pullback private gradient layout overflows");
        }
        size_t privateGradientBytes = 0;
        if (!checkedAddBytes(privateGradientBytes, inlineExecution ? 1 : frameCount, privateGradientStride))
            return fail(context_, "CPU pullback private gradient storage overflows");
        std::vector<uint8_t> privateGradients(privateGradientBytes);
        uint8_t shapeSourceSentinel{};
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
        HostTapeTraversalMetrics *traversalMetrics = currentHostTapeTraversalMetrics();
#endif
        std::mutex dispatchFailureMutex;
        std::string dispatchFailure;
        const auto failDispatch = [&](std::string message, VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
            std::lock_guard lock(dispatchFailureMutex);
            if (dispatchFailure.empty())
                dispatchFailure = std::move(message);
            return status;
        };
        CpuWorkgroupScheduler &scheduler = cpuWorkgroupScheduler(context_);
        const CpuRangeCallback executeRange = [&](VernonCpuRangeV1 &range) {
            std::vector<HostStaticTapeBatch::Reader> staticReaders;
            if (tapeBatch_) {
                staticReaders.resize(range.lane_end - range.lane_begin);
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const size_t tapeIndex =
                        segmented ? localLinear : cpuRangeCoordinates(range, localLinear).linearIndex;
                    if (!tapeBatch_->initializeReader(tapeIndex, staticReaders[localLinear - range.lane_begin]))
                        return failDispatch("CPU pullback tape reader could not be initialized");
                }
            }
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                auto prepareLane = [&]() -> VernonStatus {
                    const size_t invocationIndex = coordinates.linearIndex;
                    const size_t frameIndex = segmented ? localLinear : invocationIndex;
                    uint8_t *arguments = packedArguments.data() + frameIndex * layout.argumentsSize;
                    uint8_t *results = packedResults.data() + frameIndex * layout.resultsSize;
                    const size_t tableIndex = segmented ? localLinear : invocationIndex;
                    laneArguments[tableIndex] = arguments;
                    laneResults[tableIndex] = results;
                    for (const HostArgument &argument : layout.arguments) {
                        if (isPrimalSource(argument)) {
                            if (argument.tensorView) {
                                std::string owner = argument.name.substr(7);
                                const auto retained = retainedTensorViews_->find(owner);
                                if (retained == retainedTensorViews_->end() || !retained->second.shape ||
                                    !writeTensorViewDescriptor(argument, *retained->second.shape,
                                                               retained->second.packed, arguments))
                                    return failDispatch("CPU pullback required TensorView primal was not retained");
                                continue;
                            }
                            for (const HostFrameLeaf &leaf : argument.leaves) {
                                std::string path = leaf.value.path;
                                if (path.rfind("primal.", 0) != 0)
                                    return failDispatch("CPU pullback primal reflection path is invalid");
                                path.erase(0, 7);
                                const auto retained = retainedPrimals_->find(path);
                                if (retained == retainedPrimals_->end() ||
                                    retained->second.size() != leaf.value.byteSize)
                                    return failDispatch("CPU pullback required primal was not retained");
                                std::memcpy(arguments + leaf.frameOffset, retained->second.data(),
                                            retained->second.size());
                            }
                            continue;
                        }
                        if (!argument.tensorView)
                            continue;
                        uint8_t *data = &shapeSourceSentinel;
                        const std::vector<uint64_t> *shape = nullptr;
                        std::vector<uint64_t> descriptorShape;
                        if (isShapeSource(argument)) {
                            const auto retained = tensorShapes_->find(tensorOwnerName(argument));
                            if (retained == tensorShapes_->end())
                                return failDispatch("CPU pullback TensorView argument has no retained forward shape");
                            shape = &retained->second;
                        } else {
                            const auto gradient = gradientIndices.find(argument.name);
                            if (gradient == gradientIndices.end())
                                continue;
                            const size_t gradientIndex = gradient->second;
                            if (!tensorViewDescriptorShape(argument, signature_->gradients[gradientIndex].logicalShape,
                                                           descriptorShape))
                                return failDispatch("CPU pullback gradient descriptor shape is inconsistent");
                            shape = &descriptorShape;
                            data = gradientOwnership[gradientIndex] == "none" ? &shapeSourceSentinel
                                                                              : stagedGradients[gradientIndex].data();
                        }
                        if (!shape || !writeTensorViewDescriptor(argument, *shape, data, arguments))
                            return failDispatch("CPU pullback TensorView descriptor overflows");
                    }
                    if (tapeBatch_) {
                        if (!layout.tapeAllocatorOffset || !layout.tapeRootRegionOffset)
                            return failDispatch("CPU pullback dynamic tape ABI is incomplete");
                        HostStaticTapeBatch::Reader &reader = staticReaders[localLinear - range.lane_begin];
                        VernonAdTapeAllocator *descriptor = reader.descriptor();
                        const VernonAdRegionHandle root = reader.rootRegion();
                        if (!descriptor)
                            return failDispatch("CPU pullback tape view is unavailable");
                        if (!root)
                            return failDispatch("CPU pullback dynamic tape has no root region");
                        std::memcpy(arguments + *layout.tapeAllocatorOffset, &descriptor,
                                    sizeof(VernonAdTapeAllocator *));
                        std::memcpy(arguments + *layout.tapeRootRegionOffset, &root, sizeof(root));
                    } else if (layout.tapeAllocatorOffset || layout.tapeRootRegionOffset) {
                        return failDispatch("CPU pullback no-Tape ABI contains Tape arguments");
                    }
                    for (size_t cotangentIndex = 0; cotangentIndex < cotangentArguments.size(); ++cotangentIndex) {
                        const HostArgument &argument = *cotangentArguments[cotangentIndex];
                        const uint8_t *source =
                            directCotangents[cotangentIndex]
                                ? directCotangents[cotangentIndex] +
                                      invocationIndex * signature_->cotangents[cotangentIndex].byteSize
                                : cotangentBytes[cotangentIndex].data() +
                                      (sharedCotangents[cotangentIndex]
                                           ? 0
                                           : invocationIndex * signature_->cotangents[cotangentIndex].byteSize);
                        if (argument.tensorView) {
                            std::vector<uint64_t> descriptorShape;
                            if (!tensorViewDescriptorShape(
                                    argument, signature_->cotangents[cotangentIndex].logicalShape, descriptorShape) ||
                                !writeTensorViewDescriptor(argument, descriptorShape, const_cast<uint8_t *>(source),
                                                           arguments))
                                return failDispatch("CPU pullback cotangent descriptor is inconsistent");
                        } else {
                            const HostFrameLeaf &leaf = argument.leaves.front();
                            std::memcpy(arguments + leaf.frameOffset, source,
                                        signature_->cotangents[cotangentIndex].byteSize);
                        }
                    }
                    for (const HostArgument &argument : layout.arguments) {
                        if (argument.builtin.empty() || argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN ||
                            argument.builtin == "ad_tape_root_region")
                            continue;
                        if (!writeInvocationBuiltin(argument, coordinates, arguments))
                            return failDispatch("native CPU pullback has an unsupported builtin ABI");
                    }
                    return VERNON_STATUS_OK;
                };
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
                const VernonStatus prepareStatus = withHostTapeTraversalMetrics(traversalMetrics, prepareLane);
#else
                const VernonStatus prepareStatus = prepareLane();
#endif
                if (prepareStatus != VERNON_STATUS_OK)
                    return prepareStatus;
            }
            const size_t firstInvocation = cpuRangeCoordinates(range, range.lane_begin).linearIndex;
            const size_t firstTableIndex = segmented ? range.lane_begin : firstInvocation;
            range.arguments = laneArguments[firstTableIndex];
            range.arguments_size = layout.argumentsSize;
            range.results = laneResults[firstInvocation];
            range.results_size = layout.resultsSize;
            range.lane_arguments = laneArguments.data();
            range.lane_results = laneResults.data();
            range.lane_table_count = laneArguments.size();
            const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
            auto invokeBackward = [&] { return program_->backward->entry(&invocation); };
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
            const VernonStatus status = withHostTapeTraversalMetrics(traversalMetrics, invokeBackward);
#else
            const VernonStatus status = invokeBackward();
#endif
            if (status != VERNON_STATUS_OK)
                return failDispatch("autodiff backward profile invocation failed", status);
            if (range.outcome == VERNON_CPU_RANGE_YIELDED_V1)
                return VERNON_STATUS_OK;
            if (range.outcome != VERNON_CPU_RANGE_COMPLETE_V1 ||
                range.completed_lanes != range.lane_end - range.lane_begin)
                return failDispatch("autodiff backward lanes completed at different barrier phases");
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                const size_t frameIndex = segmented ? localLinear : invocationIndex;
                uint8_t *results = packedResults.data() + frameIndex * layout.resultsSize;
                uint8_t *lanePrivateGradients =
                    privateGradients.data() + (inlineExecution ? 0 : frameIndex * privateGradientStride);
                if (VernonStatus accumulateStatus =
                        accumulateBackwardResults(layout, *signature_, results, program_->resultGradientIndices,
                                                  lanePrivateGradients, privateGradientOffsets);
                    accumulateStatus != VERNON_STATUS_OK)
                    return failDispatch("CPU pullback result gradient ABI is inconsistent", accumulateStatus);
                const size_t tableIndex = segmented ? localLinear : invocationIndex;
                laneArguments[tableIndex] = nullptr;
                laneResults[tableIndex] = nullptr;
            }
            return VERNON_STATUS_OK;
        };
        const VernonStatus dispatchStatus =
            segmented         ? scheduler.dispatchGroupInline(grid, layout.workgroup, *groupLinear_, executeRange)
            : inlineExecution ? scheduler.dispatchInline(grid, layout.workgroup, executeRange)
                              : scheduler.dispatch(grid, layout.workgroup, executeRange);
        if (dispatchStatus != VERNON_STATUS_OK) {
            const std::string diagnostic = !dispatchFailure.empty() ? dispatchFailure
                                           : scheduler.lastDiagnostic().empty()
                                               ? "CPU pullback workgroup execution failed"
                                               : scheduler.lastDiagnostic();
            return fail(context_, diagnostic, dispatchStatus);
        }
        for (size_t gradientIndex = 0; gradientIndex < signature_->gradients.size(); ++gradientIndex) {
            if (gradientOwnership[gradientIndex] == "none" || storageGradients[gradientIndex]) {
                continue;
            } else {
                const size_t contributionCount = inlineExecution ? 1 : frameCount;
                for (size_t contribution = 0; contribution < contributionCount; ++contribution)
                    if (VernonStatus status =
                            accumulateGradientBytes(context_, signature_->gradients[gradientIndex],
                                                    privateGradients.data() + contribution * privateGradientStride +
                                                        privateGradientOffsets[gradientIndex],
                                                    stagedGradients[gradientIndex]);
                        status != VERNON_STATUS_OK)
                        return status;
            }
        }
        if (!externalStagedGradients_)
            commitGradientDestinations(destinations, stagedGradients);
        return VERNON_STATUS_OK;
    }

private:
    const Signature &runtimeSignature() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->signature;
        return std::get<CpuBackwardBorrowedState>(state_).signature.get();
    }

    const RuntimeTensorShapes &runtimeTensorShapes() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->tensorShapes;
        return std::get<CpuBackwardBorrowedState>(state_).tensorShapes.get();
    }

    const RetainedPrimalLeaves &retainedPrimalLeaves() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->retainedPrimals;
        return std::get<CpuBackwardBorrowedState>(state_).retainedPrimals.get();
    }

    const RetainedPrimalTensorViewRefs &retainedPrimalTensorViews() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->retainedTensorViewRefs;
        return std::get<CpuBackwardBorrowedState>(state_).retainedTensorViews.get();
    }

    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::variant<CpuBackwardOwnedState, CpuBackwardBorrowedState> state_;
    VernonLaunchSize computeGrid_{};
    std::shared_ptr<HostStaticTapeBatch> tapeBatch_;
    std::optional<size_t> groupLinear_;
    std::vector<std::vector<uint8_t>> *externalStagedGradients_{};
};

class NoTapeCpuPullback final : public PullbackExecution {
public:
    NoTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                      Signature signature, VernonLaunchSize computeGrid, RuntimeTensorShapes tensorShapes,
                      RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews)
        : executor_(context, std::move(program), std::move(signature), computeGrid, nullptr, std::move(tensorShapes),
                    std::move(retainedPrimals), std::move(retainedTensorViews)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &) override {
        return executor_.apply(cotangents, gradients);
    }

    PullbackMemoryUsage memoryUsage() const override { return executor_.memoryUsage(); }

private:
    CpuBackwardExecutor executor_;
};

class RetainedTapeCpuPullback final : public PullbackExecution {
public:
    RetainedTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                            Signature signature, VernonLaunchSize computeGrid,
                            std::shared_ptr<HostStaticTapeBatch> tapeBatch, RuntimeTensorShapes tensorShapes,
                            RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews)
        : executor_(context, std::move(program), std::move(signature), computeGrid, std::move(tapeBatch),
                    std::move(tensorShapes), std::move(retainedPrimals), std::move(retainedTensorViews)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &) override {
        return executor_.apply(cotangents, gradients);
    }

    PullbackMemoryUsage memoryUsage() const override { return executor_.memoryUsage(); }

private:
    CpuBackwardExecutor executor_;
};

template <typename Invoke, typename Finish>
VernonStatus runCpuForward(VernonRuntimeContext &context, const HostProfileLayout &layout, const Signature &signature,
                           VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                           const std::unordered_set<std::string> &requiredPrimalTensorOwners,
                           std::unique_ptr<PullbackExecution> &pullback, Invoke &&invoke, Finish &&finish,
                           std::optional<size_t> groupLinear = std::nullopt, bool compactAllGroups = false) {
    const uint32_t grid[3]{computeGrid.x, computeGrid.y, computeGrid.z};
    if (!validateDispatchContract(layout.dispatchContract, grid, layout.workgroup, invocationDiagnostic(context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonLaunchSize extent{};
    size_t invocationCount = 0;
    if (!invocationExtent(computeGrid, {layout.workgroup[0], layout.workgroup[1], layout.workgroup[2]}, extent) ||
        !carrierCount(extent, invocationCount))
        return fail(context, "native CPU autodiff launch size overflows");
    if (inputs.value_count != signature.inputs.size() || outputs.value_count != 0 || !layout.results.empty())
        return fail(context, "autodiff forward values do not match the host profile");
    std::vector<uint8_t> arguments(layout.argumentsSize);
    std::vector<StorageRange> storageRanges;
    std::vector<StagedTensorView> tensorViews;
    HostEffectTransaction transaction(0);
    if (VernonStatus status = stageForwardInputs(context, layout, inputs, arguments, transaction, storageRanges,
                                                 tensorViews, groupLinear.has_value());
        status != VERNON_STATUS_OK)
        return status;
    Signature runtimeSignature = signature;
    if (!materializeRuntimeSignature(runtimeSignature, inputs))
        return fail(context, "cannot materialize native CPU autodiff runtime signature");
    RuntimeTensorShapes tensorShapes;
    RetainedPrimalTensorViews retainedTensorViews;
    tensorShapes.reserve(tensorViews.size());
    for (const StagedTensorView &view : tensorViews) {
        const std::string owner = tensorOwnerName(*view.argument);
        const auto [retained, inserted] = tensorShapes.emplace(owner, view.shape);
        if (!inserted && retained->second != view.shape)
            return fail(context, "native CPU autodiff retained incompatible TensorView shapes for one owner");
        if (requiredPrimalTensorOwners.find(owner) != requiredPrimalTensorOwners.end())
            retainedTensorViews.emplace(owner, RetainedPrimalTensorView{view.shape, view.packed});
    }
    CpuWorkgroupScheduler &scheduler = cpuWorkgroupScheduler(context);
    std::mutex invocationFailureMutex;
    std::string invocationFailure;
    const auto recordInvocationFailure = [&](std::string diagnostic) {
        std::lock_guard lock(invocationFailureMutex);
        if (invocationFailure.empty())
            invocationFailure = std::move(diagnostic);
    };
    struct ForwardInvocationFrame {
        uint8_t *arguments{};
        uint8_t *results{};
    };
    std::vector<ForwardInvocationFrame> invocationFrames;
    std::vector<uint8_t> argumentFrames;
    std::vector<uint8_t> resultFrames;
    std::vector<const void *> laneArguments;
    std::vector<void *> laneResults;
    size_t frameCount = invocationCount;
    try {
        size_t workgroupVolume = 1;
        for (uint32_t dimension : layout.workgroup) {
            if (workgroupVolume > SIZE_MAX / dimension)
                return fail(context, "CPU autodiff workgroup size overflows");
            workgroupVolume *= dimension;
        }
        frameCount = groupLinear || compactAllGroups ? workgroupVolume : invocationCount;
        size_t argumentFrameBytes = 0;
        size_t resultFrameBytes = 0;
        if (!checkedAddBytes(argumentFrameBytes, frameCount, layout.argumentsSize) ||
            !checkedAddBytes(resultFrameBytes, frameCount, layout.resultsSize))
            return fail(context, "CPU autodiff lane frame size overflows");
        invocationFrames.resize(frameCount);
        argumentFrames.resize(argumentFrameBytes);
        resultFrames.resize(resultFrameBytes);
        laneArguments.resize(frameCount);
        laneResults.resize(frameCount);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate persistent CPU autodiff lane frames");
    }
    const CpuRangeCallback executeRange = [&](VernonCpuRangeV1 &range) {
        try {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                const size_t frameIndex = groupLinear || compactAllGroups ? localLinear : invocationIndex;
                ForwardInvocationFrame &frame = invocationFrames[frameIndex];
                if (!frame.arguments && layout.argumentsSize) {
                    frame.arguments = argumentFrames.data() + frameIndex * layout.argumentsSize;
                    std::memcpy(frame.arguments, arguments.data(), layout.argumentsSize);
                }
                if (!frame.results && layout.resultsSize)
                    frame.results = resultFrames.data() + frameIndex * layout.resultsSize;
                const size_t tableIndex = groupLinear || compactAllGroups ? localLinear : invocationIndex;
                laneArguments[tableIndex] = frame.arguments;
                laneResults[tableIndex] = frame.results;
            }
        } catch (const std::bad_alloc &) {
            recordInvocationFailure("cannot allocate active CPU autodiff lane frames");
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        const size_t firstInvocation = cpuRangeCoordinates(range, range.lane_begin).linearIndex;
        const size_t firstTableIndex = groupLinear || compactAllGroups ? range.lane_begin : firstInvocation;
        range.arguments = laneArguments[firstTableIndex];
        range.arguments_size = layout.argumentsSize;
        range.results = laneResults[firstInvocation];
        range.results_size = layout.resultsSize;
        range.lane_arguments = laneArguments.data();
        range.lane_results = laneResults.data();
        range.lane_table_count = laneArguments.size();
        std::string diagnostic;
        VernonStatus status = invoke(range, invocationFrames, diagnostic);
        if (status != VERNON_STATUS_OK && !diagnostic.empty())
            recordInvocationFailure(std::move(diagnostic));
        if (status == VERNON_STATUS_OK && range.outcome == VERNON_CPU_RANGE_COMPLETE_V1) {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                const size_t frameIndex = groupLinear || compactAllGroups ? localLinear : invocationIndex;
                const size_t tableIndex = frameIndex;
                laneArguments[tableIndex] = nullptr;
                laneResults[tableIndex] = nullptr;
                invocationFrames[frameIndex] = {};
            }
        }
        return status;
    };
    const VernonStatus dispatchStatus =
        groupLinear        ? scheduler.dispatchGroupInline(grid, layout.workgroup, *groupLinear, executeRange)
        : compactAllGroups ? scheduler.dispatchInline(grid, layout.workgroup, executeRange)
        : invocationCount <= kCpuAdInlineInvocationLimit
            ? scheduler.dispatchInline(grid, layout.workgroup, executeRange)
            : scheduler.dispatch(grid, layout.workgroup, executeRange);
    if (dispatchStatus != VERNON_STATUS_OK)
        return fail(context,
                    invocationFailure.empty()
                        ? (scheduler.lastDiagnostic().empty() ? "CPU autodiff forward workgroup execution failed"
                                                              : scheduler.lastDiagnostic())
                        : invocationFailure,
                    dispatchStatus);
    std::unique_ptr<PullbackExecution> pendingPullback;
    if (VernonStatus status = finish(invocationCount, runtimeSignature, std::move(tensorShapes),
                                     std::move(retainedTensorViews), pendingPullback);
        status != VERNON_STATUS_OK)
        return status;
    if (!pendingPullback)
        return fail(context, "native CPU autodiff did not create a pullback", VERNON_STATUS_INTERNAL_ERROR);
    if (VernonStatus status = flushStagedTensorViews(context, tensorViews); status != VERNON_STATUS_OK)
        return status;
    if (!transaction.commit(nullptr))
        return fail(context, "autodiff effect transaction was already committed");
    pullback = std::move(pendingPullback);
    return VERNON_STATUS_OK;
}

std::unordered_set<std::string> requiredPrimalTensorOwners(const HostProfileLayout &backwardLayout) {
    std::unordered_set<std::string> owners;
    for (const HostArgument &argument : backwardLayout.arguments)
        if (isPrimalSource(argument) && argument.tensorView)
            owners.insert(argument.name.substr(7));
    return owners;
}

VernonStatus retainRequiredPrimalLeaves(VernonRuntimeContext &context, const HostProfileLayout &backwardLayout,
                                        const VernonAdValueSet &inputs, RetainedPrimalLeaves &retainedPrimals) {
    for (const HostArgument &argument : backwardLayout.arguments) {
        if (!isPrimalSource(argument) || argument.tensorView)
            continue;
        for (const HostFrameLeaf &leaf : argument.leaves) {
            std::string path = leaf.value.path;
            if (path.rfind("primal.", 0) != 0)
                return fail(context, "CPU primal reflection path is invalid");
            path.erase(0, 7);
            const VernonAdValue *value = findValue(inputs, path);
            if (!value || value->size != leaf.value.byteSize || value->dtype != leaf.value.dtype)
                return fail(context, "CPU required primal does not match the forward inputs");
            retainedPrimals[path] = std::vector<uint8_t>(static_cast<const uint8_t *>(value->data),
                                                         static_cast<const uint8_t *>(value->data) + value->size);
        }
    }
    return VERNON_STATUS_OK;
}

class NoTapeStructuredCpuExecutable final : public HostExecutable {
public:
    NoTapeStructuredCpuExecutable(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program)
        : context_(context), program_(std::move(program)) {}

    const Signature &signature() const override { return program_->signature; }

    VernonStatus forward(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        const HostProfileLayout &forwardLayout = program_->forwardLayout;
        const HostProfileLayout &backwardLayout = program_->backwardLayout;
        const uint32_t grid[3]{computeGrid.x, computeGrid.y, computeGrid.z};
        if (!validateDispatchContract(backwardLayout.dispatchContract, grid, backwardLayout.workgroup,
                                      invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        RetainedPrimalLeaves retainedPrimals;
        if (VernonStatus status = retainRequiredPrimalLeaves(context_, backwardLayout, inputs, retainedPrimals);
            status != VERNON_STATUS_OK)
            return status;
        return runCpuForward(
            context_, forwardLayout, program_->signature, computeGrid, inputs, outputs,
            program_->requiredPrimalTensorOwners, pullback,
            [&](VernonCpuRangeV1 &range, auto &frames, std::string &diagnostic) {
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                    uint8_t *arguments = frames[coordinates.linearIndex].arguments;
                    for (const HostArgument &argument : forwardLayout.arguments)
                        if (!argument.builtin.empty() && !writeInvocationBuiltin(argument, coordinates, arguments)) {
                            diagnostic = "native CPU no-Tape autodiff has an unsupported builtin ABI";
                            return VERNON_STATUS_INVALID_ARGUMENT;
                        }
                }
                const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
                return program_->forward->entry(&invocation);
            },
            [&](size_t, Signature runtimeSignature, RuntimeTensorShapes tensorShapes,
                RetainedPrimalTensorViews retainedTensorViews, std::unique_ptr<PullbackExecution> &pending) {
                pending = std::make_unique<NoTapeCpuPullback>(
                    context_, program_, std::move(runtimeSignature), computeGrid, std::move(tensorShapes),
                    std::move(retainedPrimals), std::move(retainedTensorViews));
                return VERNON_STATUS_OK;
            });
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
};

class CpuTapeRangeExecutor {
public:
    struct SegmentScratch {
        std::vector<uint8_t> argumentFrames;
        std::vector<const void *> laneArguments;
        std::shared_ptr<HostStaticTapeBatch> recyclableTape;
        std::shared_ptr<HostTapeDispatchBudget> tapeBudget;
    };

    CpuTapeRangeExecutor(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                         std::shared_ptr<const CpuResidualPlan> plan, std::shared_ptr<HostTapeMemoryPolicy> policy)
        : context_(context), program_(std::move(program)), plan_(std::move(plan)), policy_(std::move(policy)) {}

    std::shared_ptr<HostStaticTapeBatch> allocate(size_t laneCount,
                                                  const std::shared_ptr<HostTapeDispatchBudget> &budget) const {
        return HostStaticTapeBatch::create(laneCount, plan_->staticTapeStride, policy_->invocationLimit(), policy_,
                                           budget);
    }

    VernonStatus tryAllocateWholeDispatch(size_t invocationCount, std::shared_ptr<HostStaticTapeBatch> &tape,
                                          std::shared_ptr<HostTapeDispatchBudget> &budget, bool &admitted) const {
        admitted = false;
        size_t constructionBytes = 0;
        if (!hostStaticTapeBatchPureStaticBytes(invocationCount, plan_->staticTapeStride, constructionBytes))
            return fail(context_, "CPU whole-dispatch Tape size overflows");
        if (constructionBytes > policy_->contextLimit())
            return VERNON_STATUS_OK;
        budget = HostTapeDispatchBudget::reserve(policy_, constructionBytes);
        if (!budget)
            return VERNON_STATUS_OK;
        tape = allocate(invocationCount, budget);
        if (!tape)
            return fail(context_, "CPU whole-dispatch Tape allocation disagrees with its admitted static plan",
                        VERNON_STATUS_INTERNAL_ERROR);
        admitted = true;
        return VERNON_STATUS_OK;
    }

    template <typename ArgumentsAt, typename TapeIndexAt>
    VernonStatus executeRange(VernonCpuRangeV1 &range, HostStaticTapeBatch &tape, ArgumentsAt &&argumentsAt,
                              TapeIndexAt &&tapeIndexAt, std::string &diagnostic) const {
        const HostProfileLayout &layout = program_->forwardLayout;
        for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
            const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
            uint8_t *arguments = argumentsAt(localLinear, coordinates);
            const size_t tapeIndex = tapeIndexAt(localLinear, coordinates);
            for (const HostArgument &argument : layout.arguments) {
                if (argument.builtin.empty())
                    continue;
                if (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
                    VernonAdTapeAllocator *descriptor = tape.descriptor(tapeIndex);
                    if (!descriptor) {
                        diagnostic = "CPU Tape dispatch lane is unavailable";
                        return VERNON_STATUS_INTERNAL_ERROR;
                    }
                    std::memcpy(arguments + argument.offset, &descriptor, sizeof(VernonAdTapeAllocator *));
                } else if (!writeInvocationBuiltin(argument, coordinates, arguments)) {
                    diagnostic = "CPU Tape dispatch has an unsupported builtin ABI";
                    return VERNON_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
        const VernonStatus status = program_->forward->entry(&invocation);
        if (status != VERNON_STATUS_OK || range.outcome == VERNON_CPU_RANGE_YIELDED_V1)
            return status;
        if (range.outcome != VERNON_CPU_RANGE_COMPLETE_V1 ||
            range.completed_lanes != range.lane_end - range.lane_begin) {
            diagnostic = "CPU Tape forward lanes completed at different barrier phases";
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
            const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
            const size_t tapeIndex = tapeIndexAt(localLinear, coordinates);
            VernonAdTapeAllocator *descriptor = tape.descriptor(tapeIndex);
            if (!descriptor) {
                diagnostic = "CPU Tape dispatch has no allocator descriptor";
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            if (descriptor->status != VERNON_AD_TAPE_ALLOCATOR_OK) {
                diagnostic = std::string("autodiff tape allocator ") + allocatorFailure(descriptor->status) +
                             " after requiring " + std::to_string(descriptor->required_bytes) + " bytes under the " +
                             std::to_string(policy_->contextLimit()) + "-byte context limit";
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            if (!tape.rootRegion(tapeIndex)) {
                diagnostic = "CPU Tape dispatch did not seal its Tape";
                return VERNON_STATUS_INTERNAL_ERROR;
            }
        }
        return VERNON_STATUS_OK;
    }

    VernonStatus finish(HostStaticTapeBatch &tape, HostTapeDispatchBudget &budget, std::string_view scope,
                        bool retainConstructionStorage = false) const {
        if (!tape.compact(retainConstructionStorage))
            return fail(context_, "CPU " + std::string(scope) + " Tape could not be compacted");
        budget.commit();
        return VERNON_STATUS_OK;
    }

    VernonStatus replayPreparedGroup(VernonLaunchSize computeGrid, size_t groupLinear,
                                     const std::vector<uint8_t> &baseArguments,
                                     std::shared_ptr<HostStaticTapeBatch> &tapeBatch, SegmentScratch &scratch,
                                     const PullbackApplyOptions &options) const {
        const HostProfileLayout &layout = program_->forwardLayout;
        size_t workgroupVolume = 1;
        for (uint32_t dimension : layout.workgroup) {
            if (workgroupVolume > SIZE_MAX / dimension)
                return fail(context_, "CPU bounded replay workgroup size overflows");
            workgroupVolume *= dimension;
        }
        if (scratch.recyclableTape &&
            scratch.recyclableTape->constructionBytes() <= options.maximumReusableConstructionBytes &&
            scratch.recyclableTape->resetRecyclableConstruction()) {
            tapeBatch = std::move(scratch.recyclableTape);
        } else {
            scratch.recyclableTape.reset();
            scratch.tapeBudget.reset();
            size_t maximumTapeBytes = 0;
            if (!checkedAddBytes(maximumTapeBytes, workgroupVolume, policy_->invocationLimit()))
                return fail(context_, "CPU bounded replay segment Tape upper bound overflows",
                            VERNON_STATUS_INTERNAL_ERROR);
            const size_t applyLimit =
                std::min({maximumTapeBytes, policy_->contextLimit(), options.maximumTemporaryBytes});
            scratch.tapeBudget = HostTapeDispatchBudget::reserve(policy_, applyLimit);
            if (!scratch.tapeBudget)
                return fail(context_, "CPU bounded replay cannot reserve one complete-workgroup Tape segment",
                            VERNON_STATUS_INTERNAL_ERROR);
            tapeBatch = allocate(workgroupVolume, scratch.tapeBudget);
            if (!tapeBatch)
                return fail(context_, "CPU bounded replay workgroup Tape exceeds the context budget",
                            VERNON_STATUS_INTERNAL_ERROR);
        }
        size_t argumentFrameBytes = 0;
        if (!checkedAddBytes(argumentFrameBytes, workgroupVolume, layout.argumentsSize))
            return fail(context_, "CPU bounded replay argument frames overflow");
        try {
            scratch.argumentFrames.resize(argumentFrameBytes);
            scratch.laneArguments.assign(workgroupVolume, nullptr);
        } catch (const std::bad_alloc &) {
            return fail(context_, "cannot allocate CPU bounded replay segment scratch", VERNON_STATUS_INTERNAL_ERROR);
        } catch (const std::length_error &) {
            return fail(context_, "CPU bounded replay segment scratch exceeds container limits",
                        VERNON_STATUS_INTERNAL_ERROR);
        }
        std::string invocationFailure;
        const CpuRangeCallback execute = [&](VernonCpuRangeV1 &range) {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                if (!scratch.laneArguments[localLinear]) {
                    uint8_t *arguments = scratch.argumentFrames.data() + localLinear * layout.argumentsSize;
                    std::memcpy(arguments, baseArguments.data(), layout.argumentsSize);
                    scratch.laneArguments[localLinear] = arguments;
                }
            }
            range.arguments = scratch.laneArguments[range.lane_begin];
            range.arguments_size = layout.argumentsSize;
            range.lane_arguments = scratch.laneArguments.data();
            range.lane_table_count = scratch.laneArguments.size();
            return executeRange(
                range, *tapeBatch,
                [&](size_t localLinear, const CpuLaneCoordinates &) {
                    return scratch.argumentFrames.data() + localLinear * layout.argumentsSize;
                },
                [](size_t localLinear, const CpuLaneCoordinates &) { return localLinear; }, invocationFailure);
        };
        CpuWorkgroupScheduler &scheduler = cpuWorkgroupScheduler(context_);
        const uint32_t grid[3]{computeGrid.x, computeGrid.y, computeGrid.z};
        const VernonStatus status = scheduler.dispatchGroupInline(grid, layout.workgroup, groupLinear, execute);
        if (status != VERNON_STATUS_OK)
            return fail(context_, invocationFailure.empty() ? scheduler.lastDiagnostic() : invocationFailure, status);
        const bool retainConstruction = tapeBatch->constructionBytes() <= options.maximumReusableConstructionBytes;
        return finish(*tapeBatch, *scratch.tapeBudget, "bounded replay segment", retainConstruction);
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::shared_ptr<const CpuResidualPlan> plan_;
    std::shared_ptr<HostTapeMemoryPolicy> policy_;
};

class SegmentReplayCpuPullback final : public PullbackExecution {
public:
    SegmentReplayCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                             std::shared_ptr<const CpuResidualPlan> plan,
                             std::shared_ptr<HostTapeMemoryPolicy> tapePolicy, Signature signature,
                             VernonLaunchSize computeGrid, OwnedAdValueSet retainedInputs)
        : context_(context), program_(std::move(program)), plan_(std::move(plan)), tapePolicy_(std::move(tapePolicy)),
          signature_(std::move(signature)), computeGrid_(computeGrid), retainedInputs_(std::move(retainedInputs)) {}

    PullbackMemoryUsage memoryUsage() const override {
        return {0, 0, 0, retainedInputs_.retainedBytes(), peakTemporaryBytes_.load(std::memory_order_relaxed)};
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        std::vector<VernonAdValue *> destinations;
        std::vector<std::vector<uint8_t>> stagedGradients;
        if (VernonStatus status =
                prepareGradientDestinations(context_, signature_, gradients, destinations, stagedGradients);
            status != VERNON_STATUS_OK)
            return status;
        OwnedAdValueSet replayInputs(retainedInputs_.set);
        std::vector<uint8_t> arguments(program_->forwardLayout.argumentsSize);
        std::vector<StorageRange> storageRanges;
        std::vector<StagedTensorView> tensorViews;
        HostEffectTransaction transaction(0);
        if (VernonStatus status = stageForwardInputs(context_, program_->forwardLayout, replayInputs.set, arguments,
                                                     transaction, storageRanges, tensorViews);
            status != VERNON_STATUS_OK)
            return status;
        RuntimeTensorShapes tensorShapes;
        for (const StagedTensorView &view : tensorViews) {
            const std::string owner = tensorOwnerName(*view.argument);
            tensorShapes.emplace(owner, view.shape);
        }
        RetainedPrimalLeaves retainedPrimals;
        if (VernonStatus status =
                retainRequiredPrimalLeaves(context_, program_->backwardLayout, replayInputs.set, retainedPrimals);
            status != VERNON_STATUS_OK)
            return status;
        size_t groupCount = 1;
        for (uint32_t dimension : {computeGrid_.x, computeGrid_.y, computeGrid_.z}) {
            if (groupCount > SIZE_MAX / dimension)
                return fail(context_, "CPU bounded replay group count overflows");
            groupCount *= dimension;
        }
        CpuTapeRangeExecutor executor(context_, program_, plan_, tapePolicy_);
        CpuTapeRangeExecutor::SegmentScratch scratch;
        bool replayShadowIsPristine = true;
        for (size_t group = groupCount; group-- > 0;) {
            if (!replayShadowIsPristine)
                restoreReplayReadWriteShadows(tensorViews);
            replayShadowIsPristine = false;
            std::shared_ptr<HostStaticTapeBatch> tapeBatch;
            if (VernonStatus status =
                    executor.replayPreparedGroup(computeGrid_, group, arguments, tapeBatch, scratch, options);
                status != VERNON_STATUS_OK)
                return status;
            size_t observed = peakTemporaryBytes_.load(std::memory_order_relaxed);
            const size_t allocated = scratch.tapeBudget->usage().peakBytes;
            while (observed < allocated &&
                   !peakTemporaryBytes_.compare_exchange_weak(observed, allocated, std::memory_order_relaxed)) {
            }
            RetainedPrimalTensorViewRefs retainedTensorViews;
            for (StagedTensorView &view : tensorViews) {
                const std::string owner = tensorOwnerName(*view.argument);
                if (program_->requiredPrimalTensorOwners.find(owner) != program_->requiredPrimalTensorOwners.end())
                    retainedTensorViews.emplace(owner, RetainedPrimalTensorViewRef{&view.shape, view.packed.data()});
            }
            CpuBackwardExecutor segment(
                context_, program_,
                CpuBackwardBorrowedState{signature_, tensorShapes, retainedPrimals, retainedTensorViews}, computeGrid_,
                std::move(tapeBatch), group);
            if (VernonStatus status = segment.applyInto(cotangents, stagedGradients); status != VERNON_STATUS_OK)
                return status;
            std::shared_ptr<HostStaticTapeBatch> completedTape = segment.releaseTapeBatch();
            if (completedTape && completedTape->constructionBytes() <= options.maximumReusableConstructionBytes &&
                completedTape->markConstructionRecyclable())
                scratch.recyclableTape = std::move(completedTape);
        }
        commitGradientDestinations(destinations, stagedGradients);
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::shared_ptr<const CpuResidualPlan> plan_;
    std::shared_ptr<HostTapeMemoryPolicy> tapePolicy_;
    Signature signature_;
    VernonLaunchSize computeGrid_{};
    OwnedAdValueSet retainedInputs_;
    std::atomic<size_t> peakTemporaryBytes_{};
};

class TapedStructuredCpuExecutable final : public HostExecutable {
public:
    TapedStructuredCpuExecutable(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                                 std::shared_ptr<const CpuResidualPlan> plan)
        : context_(context), program_(std::move(program)), plan_(std::move(plan)), tapePolicy_(context.cpuTapePolicy) {}

    const Signature &signature() const override { return program_->signature; }

    VernonStatus forward(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        const HostProfileLayout &forwardLayout = program_->forwardLayout;
        VernonLaunchSize extent{};
        size_t invocationCount = 0;
        if (!invocationExtent(computeGrid,
                              {forwardLayout.workgroup[0], forwardLayout.workgroup[1], forwardLayout.workgroup[2]},
                              extent) ||
            !carrierCount(extent, invocationCount))
            return fail(context_, "CPU autodiff launch size overflows");
        if (plan_->permitsWholeDispatchRetention) {
            CpuTapeRangeExecutor executor(context_, program_, plan_, tapePolicy_);
            std::shared_ptr<HostStaticTapeBatch> tapeBatch;
            std::shared_ptr<HostTapeDispatchBudget> budget;
            bool admitted = false;
            if (VernonStatus status = executor.tryAllocateWholeDispatch(invocationCount, tapeBatch, budget, admitted);
                status != VERNON_STATUS_OK)
                return status;
            if (admitted)
                return forwardWithRetainedTape(computeGrid, inputs, outputs, pullback, std::move(tapeBatch),
                                               std::move(budget));
        }
        OwnedAdValueSet retainedInputs(inputs);
        return runCpuForward(
            context_, program_->primalLayout, program_->signature, computeGrid, inputs, outputs,
            program_->requiredPrimalTensorOwners, pullback,
            [&](VernonCpuRangeV1 &range, auto &frames, std::string &diagnostic) {
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                    uint8_t *arguments = frames[localLinear].arguments;
                    for (const HostArgument &argument : program_->primalLayout.arguments)
                        if (!argument.builtin.empty() && !writeInvocationBuiltin(argument, coordinates, arguments)) {
                            diagnostic = "CPU bounded replay primal has an unsupported builtin ABI";
                            return VERNON_STATUS_INVALID_ARGUMENT;
                        }
                }
                const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
                return program_->primal->entry(&invocation);
            },
            [&](size_t, Signature runtimeSignature, RuntimeTensorShapes, RetainedPrimalTensorViews,
                std::unique_ptr<PullbackExecution> &pending) {
                pending = std::make_unique<SegmentReplayCpuPullback>(context_, program_, plan_, tapePolicy_,
                                                                     std::move(runtimeSignature), computeGrid,
                                                                     std::move(retainedInputs));
                return VERNON_STATUS_OK;
            },
            std::nullopt, true);
    }

private:
    VernonStatus forwardWithRetainedTape(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs,
                                         VernonAdValueSet &outputs, std::unique_ptr<PullbackExecution> &pullback,
                                         std::shared_ptr<HostStaticTapeBatch> tapeBatch,
                                         std::shared_ptr<HostTapeDispatchBudget> dispatchBudget) {
        CpuTapeRangeExecutor executor(context_, program_, plan_, tapePolicy_);
        RetainedPrimalLeaves retainedPrimals;
        if (VernonStatus status =
                retainRequiredPrimalLeaves(context_, program_->backwardLayout, inputs, retainedPrimals);
            status != VERNON_STATUS_OK)
            return status;
        return runCpuForward(
            context_, program_->forwardLayout, program_->signature, computeGrid, inputs, outputs,
            program_->requiredPrimalTensorOwners, pullback,
            [&](VernonCpuRangeV1 &range, auto &frames, std::string &diagnostic) {
                return executor.executeRange(
                    range, *tapeBatch,
                    [&](size_t, const CpuLaneCoordinates &coordinates) {
                        return frames[coordinates.linearIndex].arguments;
                    },
                    [](size_t, const CpuLaneCoordinates &coordinates) { return coordinates.linearIndex; }, diagnostic);
            },
            [&](size_t, Signature runtimeSignature, RuntimeTensorShapes tensorShapes,
                RetainedPrimalTensorViews retainedTensorViews, std::unique_ptr<PullbackExecution> &pending) {
                if (VernonStatus status = executor.finish(*tapeBatch, *dispatchBudget, "whole-dispatch");
                    status != VERNON_STATUS_OK)
                    return status;
                pending = std::make_unique<RetainedTapeCpuPullback>(
                    context_, program_, std::move(runtimeSignature), computeGrid, std::move(tapeBatch),
                    std::move(tensorShapes), std::move(retainedPrimals), std::move(retainedTensorViews));
                return VERNON_STATUS_OK;
            });
    }

    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::shared_ptr<const CpuResidualPlan> plan_;
    std::shared_ptr<HostTapeMemoryPolicy> tapePolicy_;
};

bool validateDifferentiationSignature(VernonRuntimeContext &context, const Signature &signature) {
    if (signature.outputs.empty() || signature.cotangents.empty()) {
        invocationDiagnostic(context) = "autodiff output or cotangent signature is empty";
        return false;
    }
    std::set<std::string> outputPaths;
    for (const ValueAbi &output : signature.outputs)
        if (output.path.empty() || !outputPaths.insert(output.path).second) {
            invocationDiagnostic(context) = "autodiff output leaf paths are empty or duplicated";
            return false;
        }
    std::set<std::string> cotangentPaths;
    for (const ValueAbi &cotangent : signature.cotangents) {
        const auto output = std::find_if(signature.outputs.begin(), signature.outputs.end(),
                                         [&](const ValueAbi &value) { return value.path == cotangent.path; });
        if (cotangent.path.empty() || !cotangentPaths.insert(cotangent.path).second ||
            output == signature.outputs.end() || !derivativeAbiMatches(*output, cotangent)) {
            invocationDiagnostic(context) = "autodiff output and cotangent profile ABIs do not match";
            return false;
        }
    }
    std::set<std::string> inputPaths;
    for (const ValueAbi &input : signature.inputs)
        if (input.path.empty() || !inputPaths.insert(input.path).second) {
            invocationDiagnostic(context) = "autodiff input leaf paths are empty or duplicated";
            return false;
        }
    std::set<std::string> gradientPaths;
    for (const ValueAbi &gradient : signature.gradients) {
        if (gradient.path.empty() || !gradientPaths.insert(gradient.path).second) {
            invocationDiagnostic(context) = "autodiff gradient leaf paths are empty or duplicated";
            return false;
        }
        const auto primal = std::find_if(signature.inputs.begin(), signature.inputs.end(),
                                         [&](const ValueAbi &input) { return input.path == gradient.path; });
        if (primal == signature.inputs.end() || !derivativeAbiMatches(*primal, gradient)) {
            invocationDiagnostic(context) = "autodiff gradient leaf ABI does not match its primal input";
            return false;
        }
    }
    return true;
}

bool finishStructuredCpuExecutable(VernonRuntimeContext &context, const Stage &primalStage, const Stage &forwardStage,
                                   const Stage &backwardStage, std::shared_ptr<CpuKernelState> primalKernel,
                                   std::shared_ptr<CpuKernelState> forwardKernel,
                                   std::shared_ptr<CpuKernelState> backwardKernel,
                                   const std::vector<std::string> &gradientPaths, uint64_t staticTapeBytesHint,
                                   const std::string &residualStorage, const std::string &selectedPolicy,
                                   std::shared_ptr<Executable> &executable) {
    HostProfileLayout primal;
    HostProfileLayout forward;
    HostProfileLayout backward;
    if (!parseProfile(primalStage, primal, invocationDiagnostic(context)) ||
        !parseProfile(forwardStage, forward, invocationDiagnostic(context)) ||
        !parseProfile(backwardStage, backward, invocationDiagnostic(context)))
        return false;
    if (!std::equal(std::begin(primal.workgroup), std::end(primal.workgroup), std::begin(forward.workgroup)) ||
        !std::equal(std::begin(forward.workgroup), std::end(forward.workgroup), std::begin(backward.workgroup))) {
        invocationDiagnostic(context) = "structured CPU autodiff profiles use different workgroup sizes";
        return false;
    }
    if (!primal.results.empty() || primal.tapeAllocatorOffset || primal.tapeRootRegionOffset) {
        invocationDiagnostic(context) = "structured CPU autodiff primal profile has an invalid Tape ABI";
        return false;
    }
    CpuResidualStorage storage{};
    if (residualStorage == "none")
        storage = CpuResidualStorage::None;
    else if (residualStorage == "static")
        storage = CpuResidualStorage::Static;
    else if (residualStorage == "dynamic")
        storage = CpuResidualStorage::Dynamic;
    else {
        invocationDiagnostic(context) = "structured CPU autodiff planning metadata is invalid";
        return false;
    }
    CpuPlanningPolicy policy{};
    if (selectedPolicy == "min_memory")
        policy = CpuPlanningPolicy::MinMemory;
    else if (selectedPolicy == "balanced")
        policy = CpuPlanningPolicy::Balanced;
    else if (selectedPolicy == "min_runtime")
        policy = CpuPlanningPolicy::MinRuntime;
    else {
        invocationDiagnostic(context) = "structured CPU autodiff planning metadata is invalid";
        return false;
    }
    const bool noTape = storage == CpuResidualStorage::None;
    if ((noTape != (staticTapeBytesHint == 0)) || staticTapeBytesHint > std::numeric_limits<size_t>::max()) {
        invocationDiagnostic(context) = "structured CPU autodiff planning metadata is invalid";
        return false;
    }
    const bool validNoTape = noTape && forward.results.empty() && !forward.tapeAllocatorOffset &&
                             !forward.tapeRootRegionOffset && !backward.tapeAllocatorOffset &&
                             !backward.tapeRootRegionOffset;
    const bool validTape = !noTape && forward.results.empty() && forward.tapeAllocatorOffset &&
                           !forward.tapeRootRegionOffset && backward.tapeAllocatorOffset &&
                           backward.tapeRootRegionOffset;
    if (!validNoTape && !validTape) {
        invocationDiagnostic(context) = "structured CPU autodiff profile Tape ABI disagrees with its static hint";
        return false;
    }
    Signature signature;
    signature.storageObjectives = true;
    for (const HostArgument &argument : forward.arguments) {
        if (!argument.builtin.empty())
            continue;
        if (argument.tensorView)
            signature.inputs.insert(signature.inputs.end(), argument.tensorView->leaves.begin(),
                                    argument.tensorView->leaves.end());
        else
            for (const HostFrameLeaf &leaf : argument.leaves)
                signature.inputs.push_back(leaf.value);
    }
    std::vector<const HostArgument *> cotangentArguments;
    for (const HostArgument &argument : backward.arguments)
        if (argument.builtin.empty() && !isShapeSource(argument) && !isPrimalSource(argument) &&
            std::find(gradientPaths.begin(), gradientPaths.end(), argument.name) == gradientPaths.end())
            cotangentArguments.push_back(&argument);
    for (const HostArgument *argument : cotangentArguments) {
        const size_t leafCount = argument->tensorView ? argument->tensorView->leaves.size() : argument->leaves.size();
        if (argument->name.empty() || leafCount != 1) {
            invocationDiagnostic(context) = "structured CPU autodiff cotangent is not one canonical leaf";
            return false;
        }
        ValueAbi cotangent =
            argument->tensorView ? argument->tensorView->leaves.front() : argument->leaves.front().value;
        cotangent.path = argument->name;
        signature.cotangents.push_back(std::move(cotangent));
    }
    for (const ValueAbi &cotangent : signature.cotangents) {
        const auto input = std::find_if(signature.inputs.begin(), signature.inputs.end(),
                                        [&](const ValueAbi &value) { return value.path == cotangent.path; });
        if (input == signature.inputs.end()) {
            invocationDiagnostic(context) = "structured CPU autodiff Storage objective is not a primal input";
            return false;
        }
        signature.outputs.push_back(*input);
    }
    std::vector<size_t> resultGradientIndices;
    size_t resultIndex = 0;
    for (const std::string &gradientPath : gradientPaths) {
        const auto storageArgument =
            std::find_if(backward.arguments.begin(), backward.arguments.end(), [&](const HostArgument &argument) {
                return argument.builtin.empty() && argument.tensorView && argument.name == gradientPath;
            });
        ValueAbi gradient;
        if (storageArgument != backward.arguments.end()) {
            if (storageArgument->tensorView->leaves.size() != 1 || storageArgument->accumulationOwnership.empty()) {
                invocationDiagnostic(context) =
                    "structured CPU autodiff storage gradient has no canonical leaf or ownership";
                return false;
            }
            gradient = storageArgument->tensorView->leaves.front();
        } else {
            if (resultIndex >= backward.results.size()) {
                invocationDiagnostic(context) = "structured CPU autodiff has too few value gradient results";
                return false;
            }
            gradient = backward.results[resultIndex++].value;
            resultGradientIndices.push_back(signature.gradients.size());
        }
        gradient.path = gradientPath;
        signature.gradients.push_back(std::move(gradient));
    }
    if (resultIndex != backward.results.size()) {
        invocationDiagnostic(context) = "structured CPU autodiff has unclaimed value gradient results";
        return false;
    }
    if (!validateDifferentiationSignature(context, signature))
        return false;
    auto program = std::make_shared<CpuAutodiffProgram>();
    program->primalLayout = std::move(primal);
    program->forwardLayout = std::move(forward);
    program->backwardLayout = std::move(backward);
    program->primal = std::move(primalKernel);
    program->forward = std::move(forwardKernel);
    program->backward = std::move(backwardKernel);
    program->signature = std::move(signature);
    program->resultGradientIndices = std::move(resultGradientIndices);
    program->requiredPrimalTensorOwners = requiredPrimalTensorOwners(program->backwardLayout);
    if (noTape) {
        executable = std::make_shared<NoTapeStructuredCpuExecutable>(context, std::move(program));
    } else {
        if (!context.cpuTapePolicy)
            context.cpuTapePolicy = std::make_shared<HostTapeMemoryPolicy>();
        auto plan = std::make_shared<CpuResidualPlan>(
            CpuResidualPlan{storage, policy, static_cast<size_t>(staticTapeBytesHint),
                            storage == CpuResidualStorage::Static && policy != CpuPlanningPolicy::MinMemory});
        executable = std::make_shared<TapedStructuredCpuExecutable>(context, std::move(program), std::move(plan));
    }
    return true;
}

bool loadStructuredCpuExecutable(VernonRuntimeContext &context, const Stage &primalStage, const Stage &forwardStage,
                                 const Stage &backwardStage, const std::vector<std::string> &gradientPaths,
                                 uint64_t staticTapeBytesHint, const std::string &residualStorage,
                                 const std::string &selectedPolicy, std::shared_ptr<Executable> &executable) {
    if (!primalStage.cpuArtifact || !forwardStage.cpuArtifact || !backwardStage.cpuArtifact) {
        invocationDiagnostic(context) = "CPU autodiff profiles have no loadable artifacts";
        return false;
    }
    auto primalKernel = std::make_shared<CpuKernelState>();
    auto forwardKernel = std::make_shared<CpuKernelState>();
    auto backwardKernel = std::make_shared<CpuKernelState>();
    ReflectedEntry reflection;
    if (!loadCpuNativeArtifact(*primalStage.cpuArtifact, *primalKernel, reflection, invocationDiagnostic(context)) ||
        !loadCpuNativeArtifact(*forwardStage.cpuArtifact, *forwardKernel, reflection, invocationDiagnostic(context)) ||
        !loadCpuNativeArtifact(*backwardStage.cpuArtifact, *backwardKernel, reflection, invocationDiagnostic(context)))
        return false;
    return finishStructuredCpuExecutable(context, primalStage, forwardStage, backwardStage, std::move(primalKernel),
                                         std::move(forwardKernel), std::move(backwardKernel), gradientPaths,
                                         staticTapeBytesHint, residualStorage, selectedPolicy, executable);
}

bool loadStructuredCpuEntryExecutable(VernonRuntimeContext &context, const Stage &primalStage,
                                      VernonCpuEntryPoint primalEntry, const Stage &forwardStage,
                                      VernonCpuEntryPoint forwardEntry, const Stage &backwardStage,
                                      VernonCpuEntryPoint backwardEntry, const std::vector<std::string> &gradientPaths,
                                      uint64_t staticTapeBytesHint, const std::string &residualStorage,
                                      const std::string &selectedPolicy, std::shared_ptr<Executable> &executable) {
    auto primalKernel = std::make_shared<CpuKernelState>();
    auto forwardKernel = std::make_shared<CpuKernelState>();
    auto backwardKernel = std::make_shared<CpuKernelState>();
    ReflectedEntry reflection;
    if (!loadCpuEntry(primalEntry, primalStage.reflection.data(), primalStage.reflection.size(),
                      primalStage.entry.data(), primalStage.entry.size(), *primalKernel, reflection,
                      invocationDiagnostic(context)) ||
        !loadCpuEntry(forwardEntry, forwardStage.reflection.data(), forwardStage.reflection.size(),
                      forwardStage.entry.data(), forwardStage.entry.size(), *forwardKernel, reflection,
                      invocationDiagnostic(context)) ||
        !loadCpuEntry(backwardEntry, backwardStage.reflection.data(), backwardStage.reflection.size(),
                      backwardStage.entry.data(), backwardStage.entry.size(), *backwardKernel, reflection,
                      invocationDiagnostic(context)))
        return false;
    return finishStructuredCpuExecutable(context, primalStage, forwardStage, backwardStage, std::move(primalKernel),
                                         std::move(forwardKernel), std::move(backwardKernel), gradientPaths,
                                         staticTapeBytesHint, residualStorage, selectedPolicy, executable);
}

} // namespace

bool createCpuExecutable(VernonRuntimeContext &context, const Stage &primalStage, const Stage &forwardStage,
                         const Stage &backwardStage, const std::vector<std::string> &gradientPaths,
                         uint64_t staticTapeBytesHint, const std::string &residualStorage,
                         const std::string &selectedPolicy, std::shared_ptr<Executable> &executable) {
    return loadStructuredCpuExecutable(context, primalStage, forwardStage, backwardStage, gradientPaths,
                                       staticTapeBytesHint, residualStorage, selectedPolicy, executable);
}

bool createCpuEntryExecutable(VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry,
                              VernonStringView primalReflection, VernonStringView primalName,
                              VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
                              VernonStringView forwardName, VernonCpuEntryPoint backwardEntry,
                              VernonStringView backwardReflection, VernonStringView backwardName,
                              VernonStringView forwardProtocol, VernonStringView backwardProtocol,
                              const std::vector<std::string> &gradientPaths, uint64_t staticTapeBytesHint,
                              const std::string &residualStorage, const std::string &selectedPolicy,
                              std::shared_ptr<Executable> &executable) {
    if (!primalEntry || !forwardEntry || !backwardEntry || !primalReflection.data || !primalReflection.size ||
        !forwardReflection.data || !forwardReflection.size || !backwardReflection.data || !backwardReflection.size ||
        !primalName.data || !primalName.size || !forwardName.data || !forwardName.size || !backwardName.data ||
        !backwardName.size || !forwardProtocol.data || !forwardProtocol.size || !backwardProtocol.data ||
        !backwardProtocol.size) {
        invocationDiagnostic(context) = "direct CPU autodiff profiles are invalid";
        return false;
    }
    Stage primalStage;
    primalStage.entry.assign(primalName.data, primalName.size);
    primalStage.reflection.assign(primalReflection.data, primalReflection.size);
    Stage forwardStage;
    forwardStage.entry.assign(forwardName.data, forwardName.size);
    forwardStage.reflection.assign(forwardReflection.data, forwardReflection.size);
    Stage backwardStage;
    backwardStage.entry.assign(backwardName.data, backwardName.size);
    backwardStage.reflection.assign(backwardReflection.data, backwardReflection.size);
    if (std::string_view(forwardProtocol.data, forwardProtocol.size) != "dynamic_v2" ||
        std::string_view(backwardProtocol.data, backwardProtocol.size) != "dynamic_v2") {
        invocationDiagnostic(context) = "direct structured CPU autodiff requires explicit dynamic_v2 profiles";
        return false;
    }
    return loadStructuredCpuEntryExecutable(context, primalStage, primalEntry, forwardStage, forwardEntry,
                                            backwardStage, backwardEntry, gradientPaths, staticTapeBytesHint,
                                            residualStorage, selectedPolicy, executable);
}

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
void setHostTapeMemoryPolicyForTesting(VernonRuntimeContext &context, std::shared_ptr<HostTapeMemoryPolicy> policy) {
    context.cpuTapePolicy = std::move(policy);
}
#endif

} // namespace vernon::runtime::ad
