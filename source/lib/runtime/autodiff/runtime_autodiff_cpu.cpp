#include "runtime_autodiff_internal.h"

#include "host_effect_transaction.h"
#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
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
        if ((range.writable || existing.writable) && range.begin < existing.end && existing.begin < range.end)
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

bool writeGlobalInvocationId(const HostArgument &argument, VernonLaunchSize computeGrid, size_t invocationIndex,
                             std::vector<uint8_t> &arguments) {
    if (argument.builtin != "global_invocation_id" || argument.leaves.size() != 1 ||
        argument.leaves.front().value.dtype != VERNON_DATA_U32 ||
        argument.leaves.front().value.logicalShape != std::vector<uint64_t>{3})
        return false;
    const uint32_t coordinates[] = {
        static_cast<uint32_t>(invocationIndex % computeGrid.x),
        static_cast<uint32_t>((invocationIndex / computeGrid.x) % computeGrid.y),
        static_cast<uint32_t>(invocationIndex / (static_cast<size_t>(computeGrid.x) * computeGrid.y)),
    };
    std::memcpy(arguments.data() + argument.leaves.front().frameOffset, coordinates, sizeof(coordinates));
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

VernonStatus accumulateFloatingBytes(VernonRuntimeContext &context, VernonDataType dtype, uint8_t *destination,
                                     const uint8_t *source, size_t byteSize) {
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
    return fail(context, "CPU pullback can only accumulate well-formed floating gradients");
}

VernonStatus invokeBackwardAndAccumulate(VernonRuntimeContext &context, const CpuKernelState &backward,
                                         const HostProfileLayout &layout, const Signature &signature,
                                         std::vector<uint8_t> &arguments,
                                         const std::vector<size_t> &resultGradientIndices,
                                         std::vector<std::vector<uint8_t>> &stagedGradients) {
    std::vector<uint8_t> results(layout.resultsSize);
    const VernonCpuInvocation invocation{arguments.data(), arguments.size(), results.data(), results.size(), nullptr};
    const VernonStatus status = backward.entry(&invocation);
    if (status != VERNON_STATUS_OK)
        return fail(context, "autodiff backward profile invocation failed", status);
    if (layout.results.size() != resultGradientIndices.size())
        return fail(context, "CPU pullback result gradient ABI is inconsistent");
    for (size_t index = 0; index < resultGradientIndices.size(); ++index) {
        const size_t gradientIndex = resultGradientIndices[index];
        const HostFrameLeaf &source = layout.results[index];
        uint8_t *destination = stagedGradients[gradientIndex].data();
        const uint8_t *contribution = results.data() + source.frameOffset;
        if (VernonStatus accumulateStatus =
                accumulateFloatingBytes(context, source.value.dtype, destination, contribution, source.value.byteSize);
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
    return accumulateFloatingBytes(context, abi.dtype, destination.data(), source, abi.byteSize);
}

void commitGradientDestinations(const std::vector<VernonAdValue *> &destinations,
                                const std::vector<std::vector<uint8_t>> &stagedGradients) {
    for (size_t index = 0; index < destinations.size(); ++index)
        if (!stagedGradients[index].empty())
            std::memcpy(destinations[index]->data, stagedGradients[index].data(), stagedGradients[index].size());
}

class StructuredCpuPullbackExecution final : public PullbackExecution {
public:
    StructuredCpuPullbackExecution(VernonRuntimeContext &context, std::shared_ptr<CpuKernelState> backward,
                                   HostProfileLayout layout, Signature signature, VernonLaunchSize computeGrid,
                                   std::vector<std::vector<uint8_t>> arguments,
                                   std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots,
                                   RuntimeTensorShapes tensorShapes, std::vector<size_t> resultGradientIndices)
        : context_(context), backward_(std::move(backward)), layout_(std::move(layout)),
          signature_(std::move(signature)), computeGrid_(computeGrid), arguments_(std::move(arguments)),
          tapeSnapshots_(std::move(tapeSnapshots)), tensorShapes_(std::move(tensorShapes)),
          resultGradientIndices_(std::move(resultGradientIndices)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) override {
        if (signature_.cotangents.empty())
            return fail(context_, "CPU pullback has no cotangent leaves");
        if ((cotangents && cotangents->value_count != signature_.cotangents.size()) ||
            (!cotangents && signature_.cotangents.size() != 1))
            return fail(context_, "CPU pullback requires exactly the reflected output cotangent leaves");
        std::unordered_map<std::string_view, size_t> gradientIndices;
        gradientIndices.reserve(signature_.gradients.size());
        for (size_t index = 0; index < signature_.gradients.size(); ++index)
            gradientIndices.emplace(signature_.gradients[index].path, index);
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
            if (!materializeCarrierValue(abi, computeGrid_))
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
        std::vector<std::vector<uint8_t>> invocationStorageGradients(signature_.gradients.size());
        uint8_t shapeSourceSentinel{};
        for (const HostArgument &argument : layout_.arguments) {
            if (!argument.tensorView)
                continue;
            const bool shapeSource = isShapeSource(argument);
            const std::string shapeName = tensorOwnerName(argument);
            const auto shape = tensorShapes_.find(shapeName);
            if (shape == tensorShapes_.end())
                return fail(context_, "CPU pullback TensorView argument has no retained forward shape");
            void *data = &shapeSourceSentinel;
            if (!shapeSource) {
                const auto gradient = gradientIndices.find(argument.name);
                if (gradient == gradientIndices.end())
                    continue;
                const size_t index = gradient->second;
                invocationStorageGradients[index].resize(signature_.gradients[index].byteSize);
                data = invocationStorageGradients[index].data();
            }
            for (std::vector<uint8_t> &arguments : arguments_)
                if (!writeTensorViewDescriptor(argument, shape->second, data, arguments))
                    return fail(context_, "CPU pullback TensorView descriptor overflows");
        }
        for (size_t invocationIndex = 0; invocationIndex < arguments_.size(); ++invocationIndex) {
            std::vector<uint8_t> &arguments = arguments_[invocationIndex];
            for (std::vector<uint8_t> &gradient : invocationStorageGradients)
                std::fill(gradient.begin(), gradient.end(), uint8_t{0});
            if (!layout_.tapeAllocatorOffset || !layout_.tapeRootRegionOffset ||
                invocationIndex >= tapeSnapshots_.size())
                return fail(context_, "CPU pullback dynamic tape ABI is incomplete");
            const std::shared_ptr<const HostTapeSnapshot> &snapshot = tapeSnapshots_[invocationIndex];
            VernonAdTapeAllocator *descriptor = snapshot->descriptor();
            const VernonAdRegionHandle root = snapshot->rootRegion();
            if (!root)
                return fail(context_, "CPU pullback dynamic tape has no root region");
            std::memcpy(arguments.data() + *layout_.tapeAllocatorOffset, &descriptor, sizeof(VernonAdTapeAllocator *));
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
                        return fail(context_, "CPU pullback cotangent descriptor is inconsistent");
                } else {
                    const HostFrameLeaf &leaf = argument.leaves.front();
                    std::memcpy(arguments.data() + leaf.frameOffset, source,
                                signature_.cotangents[cotangentIndex].byteSize);
                }
            }
            if (VernonStatus status = invokeBackwardAndAccumulate(context_, *backward_, layout_, signature_, arguments,
                                                                  resultGradientIndices_, stagedGradients);
                status != VERNON_STATUS_OK)
                return status;
            for (size_t gradientIndex = 0; gradientIndex < invocationStorageGradients.size(); ++gradientIndex) {
                const std::vector<uint8_t> &contribution = invocationStorageGradients[gradientIndex];
                if (!contribution.empty())
                    if (VernonStatus status =
                            accumulateGradientBytes(context_, signature_.gradients[gradientIndex], contribution.data(),
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
    std::vector<std::vector<uint8_t>> arguments_;
    std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots_;
    RuntimeTensorShapes tensorShapes_;
    std::vector<size_t> resultGradientIndices_;
};

template <typename Invoke, typename Finish>
VernonStatus runCpuForward(VernonRuntimeContext &context, const HostProfileLayout &layout, const Signature &signature,
                           VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                           std::unique_ptr<PullbackExecution> &pullback, Invoke &&invoke, Finish &&finish) {
    size_t invocationCount = 0;
    if (!carrierCount(computeGrid, invocationCount))
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
    for (size_t invocationIndex = 0; invocationIndex < invocationCount; ++invocationIndex) {
        std::vector<uint8_t> results(layout.resultsSize);
        if (VernonStatus status = invoke(invocationIndex, arguments, results); status != VERNON_STATUS_OK)
            return status;
    }
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
        std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots;
        return runCpuForward(
            context_, forwardLayout_, signature_, computeGrid, inputs, outputs, pullback,
            [&](size_t invocationIndex, std::vector<uint8_t> &arguments, std::vector<uint8_t> &results) {
                std::unique_ptr<HostDynamicTape> tape;
                for (const HostArgument &argument : forwardLayout_.arguments) {
                    if (argument.builtin.empty())
                        continue;
                    if (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
                        tape = std::make_unique<HostDynamicTape>(tapePolicy_->invocationLimit(), tapePolicy_);
                        VernonAdTapeAllocator *descriptor = &tape->descriptor();
                        std::memcpy(arguments.data() + argument.offset, &descriptor, sizeof(VernonAdTapeAllocator *));
                    } else if (!writeGlobalInvocationId(argument, computeGrid, invocationIndex, arguments)) {
                        return fail(context_, "native CPU autodiff has an unsupported builtin ABI");
                    }
                }
                const VernonCpuInvocation invocation{arguments.data(), arguments.size(), results.data(), results.size(),
                                                     nullptr};
                const VernonStatus status = forward_->entry(&invocation);
                if (status != VERNON_STATUS_OK)
                    return fail(context_, "autodiff forward profile invocation failed", status);
                if (!tape)
                    return fail(context_, "autodiff forward profile has no dynamic tape descriptor");
                const VernonAdTapeAllocatorStatus allocatorStatus = tape->descriptor().status;
                if (allocatorStatus != VERNON_AD_TAPE_ALLOCATOR_OK)
                    return fail(context_, std::string("autodiff tape allocator ") + allocatorFailure(allocatorStatus),
                                allocatorStatus == VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE
                                    ? VERNON_STATUS_INTERNAL_ERROR
                                    : VERNON_STATUS_INVALID_ARGUMENT);
                std::shared_ptr<const HostTapeSnapshot> snapshot = tape->takeSnapshot();
                if (!snapshot)
                    return fail(context_, "autodiff forward profile did not seal its tape");
                tapeSnapshots.push_back(std::move(snapshot));
                return VERNON_STATUS_OK;
            },
            [&](size_t invocationCount, Signature runtimeSignature, RuntimeTensorShapes tensorShapes,
                std::unique_ptr<PullbackExecution> &pending) {
                std::vector<std::vector<uint8_t>> backwardArguments(
                    invocationCount, std::vector<uint8_t>(backwardLayout_.argumentsSize));
                pending = std::make_unique<StructuredCpuPullbackExecution>(
                    context_, backward_, backwardLayout_, std::move(runtimeSignature), computeGrid,
                    std::move(backwardArguments), std::move(tapeSnapshots), std::move(tensorShapes),
                    resultGradientIndices_);
                return VERNON_STATUS_OK;
            });
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
            if (storageArgument->tensorView->leaves.size() != 1) {
                invocationDiagnostic(context) =
                    "structured CPU autodiff storage gradient is not one canonical reflected leaf";
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
