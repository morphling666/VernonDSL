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
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
#include "host_tape_test_hooks.h"
#endif

namespace vernon::runtime::ad {

namespace {

constexpr size_t kCpuAdGradientDispatchLimit = 512u * 1024u * 1024u;

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
                               std::vector<uint8_t> &arguments);

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
                                std::vector<StagedTensorView> &tensorViews) {
    tensorViews.reserve(layout.arguments.size());
    for (const HostArgument &argument : layout.arguments) {
        if (!argument.builtin.empty())
            continue;
        if (argument.tensorView) {
            const HostTensorView &view = *argument.tensorView;
            if (view.leaves.size() != view.elementLayout.leaves.size())
                return fail(context, "autodiff forward TensorView has no canonical element layout");
            const bool writable = view.access != "read";
            const bool preserve = view.access != "write";
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
                                           arguments))
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

bool writeInvocationBuiltin(const HostArgument &argument, const CpuLaneCoordinates &coordinates,
                            std::vector<uint8_t> &arguments) {
    if ((argument.builtin != "global_invocation_id" && argument.builtin != "local_invocation_id" &&
         argument.builtin != "workgroup_id") ||
        argument.leaves.size() != 1 || argument.leaves.front().value.dtype != VERNON_DATA_U32 ||
        argument.leaves.front().value.logicalShape != std::vector<uint64_t>{3})
        return false;
    const uint32_t *value = argument.builtin == "global_invocation_id"
                                ? coordinates.global
                                : (argument.builtin == "local_invocation_id" ? coordinates.local : coordinates.group);
    std::memcpy(arguments.data() + argument.leaves.front().frameOffset, value, sizeof(uint32_t) * 3);
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
                               std::vector<uint8_t> &arguments) {
    if (!argument.tensorView || shape.size() != argument.tensorView->shape.size() ||
        argument.size != sizeof(uint64_t) * (2 + 2 * shape.size()))
        return false;
    uint8_t *descriptor = arguments.data() + argument.offset;
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

bool isShapeSource(const HostArgument &argument) { return argument.name.rfind("shape.", 0) == 0; }

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
                                       const std::vector<uint8_t> &results,
                                       const std::vector<size_t> &resultGradientIndices,
                                       std::vector<std::vector<uint8_t>> &stagedGradients) {
    if (layout.results.size() != resultGradientIndices.size())
        return VERNON_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < resultGradientIndices.size(); ++index) {
        const size_t gradientIndex = resultGradientIndices[index];
        const HostFrameLeaf &source = layout.results[index];
        uint8_t *destination = stagedGradients[gradientIndex].data();
        const uint8_t *contribution = results.data() + source.frameOffset;
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

struct CpuPullbackInvocationState {
    std::vector<uint8_t> packedArguments;
    std::vector<uint8_t> packedResults;
    std::shared_ptr<const HostTapeSnapshot> tape;
    std::vector<std::vector<uint8_t>> privateGradients;
};

struct CpuPullbackWorkgroupState {
    std::vector<std::vector<uint8_t>> sharedGradientStaging;
};

class StructuredCpuPullbackExecution final : public PullbackExecution {
public:
    StructuredCpuPullbackExecution(VernonRuntimeContext &context, std::shared_ptr<CpuKernelState> backward,
                                   HostProfileLayout layout, Signature signature, VernonLaunchSize computeGrid,
                                   size_t invocationCount,
                                   std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots,
                                   RuntimeTensorShapes tensorShapes, std::vector<size_t> resultGradientIndices)
        : context_(context), backward_(std::move(backward)), layout_(std::move(layout)),
          signature_(std::move(signature)), computeGrid_(computeGrid), tensorShapes_(std::move(tensorShapes)),
          resultGradientIndices_(std::move(resultGradientIndices)) {
        invocations_.reserve(invocationCount);
        for (size_t index = 0; index < invocationCount; ++index)
            invocations_.push_back(
                {{}, {}, index < tapeSnapshots.size() ? std::move(tapeSnapshots[index]) : nullptr, {}});
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) override {
        const uint32_t grid[3]{computeGrid_.x, computeGrid_.y, computeGrid_.z};
        if (!validateDispatchContract(layout_.dispatchContract, grid, layout_.workgroup,
                                      invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        VernonLaunchSize extent{};
        if (!invocationExtent(computeGrid_, {layout_.workgroup[0], layout_.workgroup[1], layout_.workgroup[2]}, extent))
            return fail(context_, "CPU pullback invocation extent overflows");
        if (signature_.cotangents.empty())
            return fail(context_, "CPU pullback has no cotangent leaves");
        if ((cotangents && cotangents->value_count != signature_.cotangents.size()) ||
            (!cotangents && signature_.cotangents.size() != 1))
            return fail(context_, "CPU pullback requires exactly the reflected output cotangent leaves");
        std::unordered_map<std::string_view, size_t> gradientIndices;
        gradientIndices.reserve(signature_.gradients.size());
        for (size_t index = 0; index < signature_.gradients.size(); ++index)
            gradientIndices.emplace(signature_.gradients[index].path, index);
        std::vector<std::string> gradientOwnership(signature_.gradients.size());
        for (size_t gradientIndex : resultGradientIndices_) {
            if (gradientIndex >= gradientOwnership.size())
                return fail(context_, "CPU pullback result gradient index is invalid");
            gradientOwnership[gradientIndex] = "invocation_private";
        }
        for (const HostArgument &argument : layout_.arguments)
            if (argument.tensorView && !isShapeSource(argument))
                if (const auto gradient = gradientIndices.find(argument.name); gradient != gradientIndices.end())
                    gradientOwnership[gradient->second] = argument.accumulationOwnership;
        if (std::any_of(gradientOwnership.begin(), gradientOwnership.end(),
                        [](const std::string &ownership) { return ownership.empty(); }))
            return fail(context_, "CPU pullback gradient has no reflected ownership");
        size_t invocationCount = 0;
        size_t workgroupCount = 0;
        if (!carrierCount(extent, invocationCount) || !carrierCount(computeGrid_, workgroupCount))
            return fail(context_, "CPU pullback dispatch size overflows");
        size_t requiredGradientBytes = 0;
        for (const ValueAbi &cotangent : signature_.cotangents)
            if (!checkedAddBytes(requiredGradientBytes, invocationCount, cotangent.byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
        for (size_t gradientIndex = 0; gradientIndex < signature_.gradients.size(); ++gradientIndex) {
            const size_t byteSize = signature_.gradients[gradientIndex].byteSize;
            if (!checkedAddBytes(requiredGradientBytes, 1, byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
            const size_t carrierCount = gradientOwnership[gradientIndex] == "none"                 ? 0
                                        : gradientOwnership[gradientIndex] == "invocation_private" ? invocationCount
                                                                                                   : workgroupCount;
            if (!checkedAddBytes(requiredGradientBytes, carrierCount, byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
        }
        if (requiredGradientBytes > kCpuAdGradientDispatchLimit)
            return fail(context_, "CPU pullback requires " + std::to_string(requiredGradientBytes) +
                                      " gradient bytes, exceeding the " + std::to_string(kCpuAdGradientDispatchLimit) +
                                      "-byte dispatch limit");
        std::vector<const HostArgument *> cotangentArguments;
        for (const HostArgument &argument : layout_.arguments)
            if (argument.builtin.empty() && !isShapeSource(argument) &&
                gradientIndices.find(argument.name) == gradientIndices.end())
                cotangentArguments.push_back(&argument);
        if (cotangentArguments.size() != signature_.cotangents.size())
            return fail(context_, "CPU pullback has an invalid cotangent ABI");
        std::vector<std::vector<uint8_t>> cotangentBytes;
        for (size_t index = 0; index < signature_.cotangents.size(); ++index) {
            const HostArgument &argument = *cotangentArguments[index];
            const size_t leafCount = argument.tensorView ? argument.tensorView->leaves.size() : argument.leaves.size();
            if (leafCount != 1 || argument.name != signature_.cotangents[index].path)
                return fail(context_, "CPU pullback cotangent reflection is inconsistent");
            ValueAbi abi = signature_.cotangents[index];
            if (!materializeCarrierValue(abi, extent))
                return fail(context_, "CPU pullback cotangent size overflows");
            std::vector<uint8_t> bytes(abi.byteSize);
            if (cotangents) {
                const VernonAdValue *value = findValue(*cotangents, abi.path);
                if (!value || !valueMatches(*value, abi))
                    return fail(context_, "output cotangent does not match backward reflection");
                std::memcpy(bytes.data(), value->data, abi.byteSize);
            } else {
                if (signature_.cotangents.size() != 1 ||
                    !makeCotangentBytes(nullptr, abi, bytes, invocationDiagnostic(context_)))
                    return VERNON_STATUS_INVALID_ARGUMENT;
            }
            cotangentBytes.push_back(std::move(bytes));
        }
        std::vector<VernonAdValue *> destinations;
        std::vector<std::vector<uint8_t>> stagedGradients;
        if (VernonStatus status =
                prepareGradientDestinations(context_, signature_, gradients, destinations, stagedGradients);
            status != VERNON_STATUS_OK)
            return status;
        std::vector<CpuPullbackInvocationState> invocations = invocations_;
        std::vector<const void *> laneArguments(invocations.size());
        std::vector<void *> laneResults(invocations.size());
        for (CpuPullbackInvocationState &invocation : invocations)
            invocation.privateGradients.assign(signature_.gradients.size(), {});
        std::vector<CpuPullbackWorkgroupState> workgroups(workgroupCount);
        for (CpuPullbackWorkgroupState &workgroup : workgroups)
            workgroup.sharedGradientStaging.resize(signature_.gradients.size());
        for (size_t gradientIndex = 0; gradientIndex < signature_.gradients.size(); ++gradientIndex) {
            if (gradientOwnership[gradientIndex] == "atomic_shared" ||
                gradientOwnership[gradientIndex] == "workgroup_shared") {
                for (CpuPullbackWorkgroupState &workgroup : workgroups)
                    workgroup.sharedGradientStaging[gradientIndex].resize(signature_.gradients[gradientIndex].byteSize);
            } else if (gradientOwnership[gradientIndex] == "invocation_private") {
                for (CpuPullbackInvocationState &invocation : invocations)
                    invocation.privateGradients[gradientIndex].resize(signature_.gradients[gradientIndex].byteSize);
            }
        }
        const auto workgroupIndex = [&](size_t invocationIndex) {
            const size_t x = invocationIndex % extent.x;
            const size_t y = (invocationIndex / extent.x) % extent.y;
            const size_t z = invocationIndex / (static_cast<size_t>(extent.x) * extent.y);
            return x / layout_.workgroup[0] +
                   static_cast<size_t>(computeGrid_.x) *
                       (y / layout_.workgroup[1] + static_cast<size_t>(computeGrid_.y) * (z / layout_.workgroup[2]));
        };
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
        const VernonStatus dispatchStatus = scheduler.dispatch(grid, layout_.workgroup, [&](VernonCpuRangeV1 &range) {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                auto prepareLane = [&]() -> VernonStatus {
                    const size_t invocationIndex = coordinates.linearIndex;
                    CpuPullbackInvocationState &invocation = invocations[invocationIndex];
                    if (invocation.packedArguments.empty() && layout_.argumentsSize)
                        invocation.packedArguments.resize(layout_.argumentsSize);
                    if (invocation.packedResults.empty() && layout_.resultsSize)
                        invocation.packedResults.resize(layout_.resultsSize);
                    std::vector<uint8_t> &arguments = invocation.packedArguments;
                    laneArguments[invocationIndex] = arguments.data();
                    laneResults[invocationIndex] = invocation.packedResults.data();
                    for (const HostArgument &argument : layout_.arguments) {
                        if (!argument.tensorView)
                            continue;
                        const std::string shapeName = tensorOwnerName(argument);
                        const auto shape = tensorShapes_.find(shapeName);
                        if (shape == tensorShapes_.end())
                            return failDispatch("CPU pullback TensorView argument has no retained forward shape");
                        uint8_t *data = &shapeSourceSentinel;
                        if (!isShapeSource(argument)) {
                            const auto gradient = gradientIndices.find(argument.name);
                            if (gradient == gradientIndices.end())
                                continue;
                            const size_t gradientIndex = gradient->second;
                            data = (gradientOwnership[gradientIndex] == "atomic_shared" ||
                                    gradientOwnership[gradientIndex] == "workgroup_shared")
                                       ? workgroups[workgroupIndex(invocationIndex)]
                                             .sharedGradientStaging[gradientIndex]
                                             .data()
                                   : gradientOwnership[gradientIndex] == "invocation_private"
                                       ? invocation.privateGradients[gradientIndex].data()
                                       : &shapeSourceSentinel;
                        }
                        if (!writeTensorViewDescriptor(argument, shape->second, data, arguments))
                            return failDispatch("CPU pullback TensorView descriptor overflows");
                    }
                    if (!layout_.tapeAllocatorOffset || !layout_.tapeRootRegionOffset || !invocation.tape)
                        return failDispatch("CPU pullback dynamic tape ABI is incomplete");
                    const std::shared_ptr<const HostTapeSnapshot> &snapshot = invocation.tape;
                    VernonAdTapeAllocator *descriptor = snapshot->descriptor();
                    const VernonAdRegionHandle root = snapshot->rootRegion();
                    if (!root)
                        return failDispatch("CPU pullback dynamic tape has no root region");
                    std::memcpy(arguments.data() + *layout_.tapeAllocatorOffset, &descriptor,
                                sizeof(VernonAdTapeAllocator *));
                    std::memcpy(arguments.data() + *layout_.tapeRootRegionOffset, &root, sizeof(root));
                    for (size_t cotangentIndex = 0; cotangentIndex < cotangentArguments.size(); ++cotangentIndex) {
                        const HostArgument &argument = *cotangentArguments[cotangentIndex];
                        uint8_t *source = cotangentBytes[cotangentIndex].data() +
                                          invocationIndex * signature_.cotangents[cotangentIndex].byteSize;
                        if (argument.tensorView) {
                            const std::string shapeName = tensorOwnerName(argument);
                            const auto shape = tensorShapes_.find(shapeName);
                            if (shape == tensorShapes_.end() ||
                                !writeTensorViewDescriptor(argument, shape->second, source, arguments))
                                return failDispatch("CPU pullback cotangent descriptor is inconsistent");
                        } else {
                            const HostFrameLeaf &leaf = argument.leaves.front();
                            std::memcpy(arguments.data() + leaf.frameOffset, source,
                                        signature_.cotangents[cotangentIndex].byteSize);
                        }
                    }
                    for (const HostArgument &argument : layout_.arguments) {
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
            range.arguments = laneArguments[firstInvocation];
            range.arguments_size = layout_.argumentsSize;
            range.results = laneResults[firstInvocation];
            range.results_size = layout_.resultsSize;
            range.lane_arguments = laneArguments.data();
            range.lane_results = laneResults.data();
            range.lane_table_count = laneArguments.size();
            const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
            auto invokeBackward = [&] { return backward_->entry(&invocation); };
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
                CpuPullbackInvocationState &lane = invocations[invocationIndex];
                if (VernonStatus accumulateStatus = accumulateBackwardResults(
                        layout_, signature_, lane.packedResults, resultGradientIndices_, lane.privateGradients);
                    accumulateStatus != VERNON_STATUS_OK)
                    return failDispatch("CPU pullback result gradient ABI is inconsistent", accumulateStatus);
                laneArguments[invocationIndex] = nullptr;
                laneResults[invocationIndex] = nullptr;
                lane.packedArguments = {};
                lane.packedResults = {};
                lane.tape.reset();
            }
            return VERNON_STATUS_OK;
        });
        if (dispatchStatus != VERNON_STATUS_OK) {
            const std::string diagnostic = !dispatchFailure.empty() ? dispatchFailure
                                           : scheduler.lastDiagnostic().empty()
                                               ? "CPU pullback workgroup execution failed"
                                               : scheduler.lastDiagnostic();
            return fail(context_, diagnostic, dispatchStatus);
        }
        for (size_t gradientIndex = 0; gradientIndex < signature_.gradients.size(); ++gradientIndex) {
            if (gradientOwnership[gradientIndex] == "none") {
                continue;
            } else if (gradientOwnership[gradientIndex] == "atomic_shared" ||
                       gradientOwnership[gradientIndex] == "workgroup_shared") {
                for (const CpuPullbackWorkgroupState &workgroup : workgroups)
                    if (VernonStatus status = accumulateGradientBytes(
                            context_, signature_.gradients[gradientIndex],
                            workgroup.sharedGradientStaging[gradientIndex].data(), stagedGradients[gradientIndex]);
                        status != VERNON_STATUS_OK)
                        return status;
            } else {
                for (const CpuPullbackInvocationState &invocation : invocations)
                    if (VernonStatus status = accumulateGradientBytes(context_, signature_.gradients[gradientIndex],
                                                                      invocation.privateGradients[gradientIndex].data(),
                                                                      stagedGradients[gradientIndex]);
                        status != VERNON_STATUS_OK)
                        return status;
            }
        }
        commitGradientDestinations(destinations, stagedGradients);
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<CpuKernelState> backward_;
    HostProfileLayout layout_;
    Signature signature_;
    VernonLaunchSize computeGrid_{};
    std::vector<CpuPullbackInvocationState> invocations_;
    RuntimeTensorShapes tensorShapes_;
    std::vector<size_t> resultGradientIndices_;
};

template <typename Invoke, typename Finish>
VernonStatus runCpuForward(VernonRuntimeContext &context, const HostProfileLayout &layout, const Signature &signature,
                           VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                           std::unique_ptr<PullbackExecution> &pullback, Invoke &&invoke, Finish &&finish) {
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
    if (VernonStatus status =
            stageForwardInputs(context, layout, inputs, arguments, transaction, storageRanges, tensorViews);
        status != VERNON_STATUS_OK)
        return status;
    Signature runtimeSignature = signature;
    if (!materializeRuntimeSignature(runtimeSignature, inputs))
        return fail(context, "cannot materialize native CPU autodiff runtime signature");
    RuntimeTensorShapes tensorShapes;
    tensorShapes.reserve(tensorViews.size());
    for (const StagedTensorView &view : tensorViews) {
        const auto [retained, inserted] = tensorShapes.emplace(tensorOwnerName(*view.argument), view.shape);
        if (!inserted && retained->second != view.shape)
            return fail(context, "native CPU autodiff retained incompatible TensorView shapes for one owner");
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
        std::vector<uint8_t> arguments;
        std::vector<uint8_t> results;
    };
    std::vector<ForwardInvocationFrame> invocationFrames;
    std::vector<const void *> laneArguments;
    std::vector<void *> laneResults;
    try {
        invocationFrames.resize(invocationCount);
        laneArguments.resize(invocationCount);
        laneResults.resize(invocationCount);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate persistent CPU autodiff lane frames");
    }
    const VernonStatus dispatchStatus = scheduler.dispatch(grid, layout.workgroup, [&](VernonCpuRangeV1 &range) {
        try {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                ForwardInvocationFrame &frame = invocationFrames[invocationIndex];
                if (frame.arguments.empty() && layout.argumentsSize)
                    frame.arguments = arguments;
                if (frame.results.empty() && layout.resultsSize)
                    frame.results.resize(layout.resultsSize);
                laneArguments[invocationIndex] = frame.arguments.data();
                laneResults[invocationIndex] = frame.results.data();
            }
        } catch (const std::bad_alloc &) {
            recordInvocationFailure("cannot allocate active CPU autodiff lane frames");
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        const size_t firstInvocation = cpuRangeCoordinates(range, range.lane_begin).linearIndex;
        range.arguments = laneArguments[firstInvocation];
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
                laneArguments[invocationIndex] = nullptr;
                laneResults[invocationIndex] = nullptr;
                invocationFrames[invocationIndex].arguments = {};
                invocationFrames[invocationIndex].results = {};
            }
        }
        return status;
    });
    if (dispatchStatus != VERNON_STATUS_OK)
        return fail(context,
                    invocationFailure.empty()
                        ? (scheduler.lastDiagnostic().empty() ? "CPU autodiff forward workgroup execution failed"
                                                              : scheduler.lastDiagnostic())
                        : invocationFailure,
                    dispatchStatus);
    std::unique_ptr<PullbackExecution> pendingPullback;
    if (VernonStatus status = finish(invocationCount, runtimeSignature, std::move(tensorShapes), pendingPullback);
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

class StructuredCpuExecutable final : public HostExecutable {
public:
    StructuredCpuExecutable(VernonRuntimeContext &context, HostProfileLayout forwardLayout,
                            HostProfileLayout backwardLayout, std::shared_ptr<CpuKernelState> forward,
                            std::shared_ptr<CpuKernelState> backward, Signature signature,
                            std::vector<size_t> resultGradientIndices)
        : context_(context), forwardLayout_(std::move(forwardLayout)), backwardLayout_(std::move(backwardLayout)),
          forward_(std::move(forward)), backward_(std::move(backward)), signature_(std::move(signature)),
          tapePolicy_(context.cpuTapePolicy), resultGradientIndices_(std::move(resultGradientIndices)) {}

    const Signature &signature() const override { return signature_; }

    VernonStatus forward(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        const uint32_t grid[3]{computeGrid.x, computeGrid.y, computeGrid.z};
        if (!validateDispatchContract(backwardLayout_.dispatchContract, grid, backwardLayout_.workgroup,
                                      invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        VernonLaunchSize extent{};
        size_t invocationCount = 0;
        if (!invocationExtent(computeGrid,
                              {forwardLayout_.workgroup[0], forwardLayout_.workgroup[1], forwardLayout_.workgroup[2]},
                              extent) ||
            !carrierCount(extent, invocationCount))
            return fail(context_, "native CPU autodiff launch size overflows");
        size_t maximumTapeBytes = 0;
        if (!checkedAddBytes(maximumTapeBytes, invocationCount, tapePolicy_->invocationLimit()))
            return fail(context_, "CPU autodiff dispatch tape upper bound overflows");
        const size_t reservationBytes = std::min(maximumTapeBytes, tapePolicy_->contextLimit());
        std::shared_ptr<HostTapeDispatchBudget> tapeBudget =
            HostTapeDispatchBudget::reserve(tapePolicy_, reservationBytes);
        if (!tapeBudget)
            return fail(context_, "CPU autodiff forward cannot reserve its " + std::to_string(reservationBytes) +
                                      "-byte dispatch tape budget under the " +
                                      std::to_string(tapePolicy_->contextLimit()) + "-byte context limit");
        std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots(invocationCount);
        std::vector<std::unique_ptr<HostDynamicTape>> activeTapes(invocationCount);
        const VernonStatus status = runCpuForward(
            context_, forwardLayout_, signature_, computeGrid, inputs, outputs, pullback,
            [&](VernonCpuRangeV1 &range, auto &frames, std::string &diagnostic) {
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                    std::vector<uint8_t> &arguments = frames[coordinates.linearIndex].arguments;
                    std::unique_ptr<HostDynamicTape> &tape = activeTapes[coordinates.linearIndex];
                    for (const HostArgument &argument : forwardLayout_.arguments) {
                        if (argument.builtin.empty())
                            continue;
                        if (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
                            if (!tape)
                                tape = std::make_unique<HostDynamicTape>(tapePolicy_->invocationLimit(), tapePolicy_,
                                                                         tapeBudget);
                            VernonAdTapeAllocator *descriptor = &tape->descriptor();
                            std::memcpy(arguments.data() + argument.offset, &descriptor,
                                        sizeof(VernonAdTapeAllocator *));
                        } else if (!writeInvocationBuiltin(argument, coordinates, arguments)) {
                            diagnostic = "native CPU autodiff has an unsupported builtin ABI";
                            return VERNON_STATUS_INVALID_ARGUMENT;
                        }
                    }
                }
                const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
                const VernonStatus status = forward_->entry(&invocation);
                if (status != VERNON_STATUS_OK) {
                    diagnostic = "autodiff forward profile invocation failed with status " +
                                 std::to_string(static_cast<int>(status)) + " for group (" +
                                 std::to_string(range.group[0]) + ", " + std::to_string(range.group[1]) + ", " +
                                 std::to_string(range.group[2]) + ") lanes [" + std::to_string(range.lane_begin) +
                                 ", " + std::to_string(range.lane_end) + ")";
                    return status;
                }
                if (range.outcome == VERNON_CPU_RANGE_YIELDED_V1)
                    return VERNON_STATUS_OK;
                if (range.outcome != VERNON_CPU_RANGE_COMPLETE_V1 ||
                    range.completed_lanes != range.lane_end - range.lane_begin) {
                    diagnostic = "autodiff forward lanes completed at different barrier phases";
                    return VERNON_STATUS_INTERNAL_ERROR;
                }
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                    std::unique_ptr<HostDynamicTape> &tape = activeTapes[invocationIndex];
                    if (!tape) {
                        diagnostic = "autodiff forward profile has no dynamic tape descriptor";
                        return VERNON_STATUS_INVALID_ARGUMENT;
                    }
                    const VernonAdTapeAllocatorStatus allocatorStatus = tape->descriptor().status;
                    if (allocatorStatus != VERNON_AD_TAPE_ALLOCATOR_OK) {
                        diagnostic = std::string("autodiff tape allocator ") + allocatorFailure(allocatorStatus) +
                                     " after requiring " + std::to_string(tape->descriptor().required_bytes) +
                                     " bytes for one invocation under the " +
                                     std::to_string(tapePolicy_->invocationLimit()) +
                                     "-byte per-invocation limit and " + std::to_string(tapeBudget->capacity()) +
                                     "-byte dispatch budget within the " + std::to_string(tapePolicy_->contextLimit()) +
                                     "-byte context limit";
                        return allocatorStatus == VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE
                                   ? VERNON_STATUS_INTERNAL_ERROR
                                   : VERNON_STATUS_INVALID_ARGUMENT;
                    }
                    std::shared_ptr<const HostTapeSnapshot> snapshot = tape->takeSnapshot();
                    if (!snapshot) {
                        diagnostic = "autodiff forward profile did not seal its tape";
                        return VERNON_STATUS_INVALID_ARGUMENT;
                    }
                    tapeSnapshots[invocationIndex] = std::move(snapshot);
                    tape.reset();
                }
                return VERNON_STATUS_OK;
            },
            [&](size_t invocationCount, Signature runtimeSignature, RuntimeTensorShapes tensorShapes,
                std::unique_ptr<PullbackExecution> &pending) {
                pending = std::make_unique<StructuredCpuPullbackExecution>(
                    context_, backward_, backwardLayout_, std::move(runtimeSignature), computeGrid, invocationCount,
                    std::move(tapeSnapshots), std::move(tensorShapes), resultGradientIndices_);
                return VERNON_STATUS_OK;
            });
        if (status == VERNON_STATUS_OK)
            tapeBudget->commit();
        return status;
    }

private:
    VernonRuntimeContext &context_;
    HostProfileLayout forwardLayout_;
    HostProfileLayout backwardLayout_;
    std::shared_ptr<CpuKernelState> forward_;
    std::shared_ptr<CpuKernelState> backward_;
    Signature signature_;
    std::shared_ptr<HostTapeMemoryPolicy> tapePolicy_;
    std::vector<size_t> resultGradientIndices_;
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

bool finishStructuredCpuExecutable(VernonRuntimeContext &context, const Stage &forwardStage, const Stage &backwardStage,
                                   std::shared_ptr<CpuKernelState> forwardKernel,
                                   std::shared_ptr<CpuKernelState> backwardKernel,
                                   const std::vector<std::string> &gradientPaths,
                                   std::shared_ptr<Executable> &executable) {
    HostProfileLayout forward;
    HostProfileLayout backward;
    if (!parseProfile(forwardStage, forward, invocationDiagnostic(context)) ||
        !parseProfile(backwardStage, backward, invocationDiagnostic(context)))
        return false;
    if (!std::equal(std::begin(forward.workgroup), std::end(forward.workgroup), std::begin(backward.workgroup))) {
        invocationDiagnostic(context) = "structured CPU autodiff profiles use different workgroup sizes";
        return false;
    }
    if (!forward.results.empty() || !forward.tapeAllocatorOffset || forward.tapeRootRegionOffset ||
        !backward.tapeAllocatorOffset || !backward.tapeRootRegionOffset) {
        invocationDiagnostic(context) = "structured CPU autodiff profile does not use the dynamic tape ABI";
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
        if (argument.builtin.empty() && !isShapeSource(argument) &&
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
    if (!context.cpuTapePolicy)
        context.cpuTapePolicy = std::make_shared<HostTapeMemoryPolicy>();
    executable = std::make_shared<StructuredCpuExecutable>(context, std::move(forward), std::move(backward),
                                                           std::move(forwardKernel), std::move(backwardKernel),
                                                           std::move(signature), std::move(resultGradientIndices));
    return true;
}

bool loadStructuredCpuExecutable(VernonRuntimeContext &context, const Stage &forwardStage, const Stage &backwardStage,
                                 const std::vector<std::string> &gradientPaths,
                                 std::shared_ptr<Executable> &executable) {
    if (!forwardStage.cpuArtifact || !backwardStage.cpuArtifact) {
        invocationDiagnostic(context) = "CPU autodiff profiles have no loadable artifacts";
        return false;
    }
    auto forwardKernel = std::make_shared<CpuKernelState>();
    auto backwardKernel = std::make_shared<CpuKernelState>();
    ReflectedEntry reflection;
    if (!loadCpuNativeArtifact(*forwardStage.cpuArtifact, *forwardKernel, reflection, invocationDiagnostic(context)) ||
        !loadCpuNativeArtifact(*backwardStage.cpuArtifact, *backwardKernel, reflection, invocationDiagnostic(context)))
        return false;
    return finishStructuredCpuExecutable(context, forwardStage, backwardStage, std::move(forwardKernel),
                                         std::move(backwardKernel), gradientPaths, executable);
}

bool loadStructuredCpuEntryExecutable(VernonRuntimeContext &context, const Stage &forwardStage,
                                      VernonCpuEntryPoint forwardEntry, const Stage &backwardStage,
                                      VernonCpuEntryPoint backwardEntry, const std::vector<std::string> &gradientPaths,
                                      std::shared_ptr<Executable> &executable) {
    auto forwardKernel = std::make_shared<CpuKernelState>();
    auto backwardKernel = std::make_shared<CpuKernelState>();
    ReflectedEntry reflection;
    if (!loadCpuEntry(forwardEntry, forwardStage.reflection.data(), forwardStage.reflection.size(),
                      forwardStage.entry.data(), forwardStage.entry.size(), *forwardKernel, reflection,
                      invocationDiagnostic(context)) ||
        !loadCpuEntry(backwardEntry, backwardStage.reflection.data(), backwardStage.reflection.size(),
                      backwardStage.entry.data(), backwardStage.entry.size(), *backwardKernel, reflection,
                      invocationDiagnostic(context)))
        return false;
    return finishStructuredCpuExecutable(context, forwardStage, backwardStage, std::move(forwardKernel),
                                         std::move(backwardKernel), gradientPaths, executable);
}

} // namespace

bool createCpuExecutable(VernonRuntimeContext &context, const Stage &forwardStage, const Stage &backwardStage,
                         const std::vector<std::string> &gradientPaths, std::shared_ptr<Executable> &executable) {
    return loadStructuredCpuExecutable(context, forwardStage, backwardStage, gradientPaths, executable);
}

bool createCpuEntryExecutable(VernonRuntimeContext &context, VernonCpuEntryPoint forwardEntry,
                              VernonStringView forwardReflection, VernonStringView forwardName,
                              VernonCpuEntryPoint backwardEntry, VernonStringView backwardReflection,
                              VernonStringView backwardName, VernonStringView forwardProtocol,
                              VernonStringView backwardProtocol, const std::vector<std::string> &gradientPaths,
                              std::shared_ptr<Executable> &executable) {
    if (!forwardEntry || !backwardEntry || !forwardReflection.data || !forwardReflection.size ||
        !backwardReflection.data || !backwardReflection.size || !forwardName.data || !forwardName.size ||
        !backwardName.data || !backwardName.size || !forwardProtocol.data || !forwardProtocol.size ||
        !backwardProtocol.data || !backwardProtocol.size) {
        invocationDiagnostic(context) = "direct CPU autodiff profiles are invalid";
        return false;
    }
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
    return loadStructuredCpuEntryExecutable(context, forwardStage, forwardEntry, backwardStage, backwardEntry,
                                            gradientPaths, executable);
}

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
void setHostTapeMemoryPolicyForTesting(VernonRuntimeContext &context, std::shared_ptr<HostTapeMemoryPolicy> policy) {
    context.cpuTapePolicy = std::move(policy);
}
#endif

} // namespace vernon::runtime::ad
