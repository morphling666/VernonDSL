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
#include <utility>
#include <vector>

namespace vernon::runtime::ad {

namespace {

struct HostFrameLeaf {
    ValueAbi value;
    size_t frameOffset{};
};

struct HostTensorView {
    std::string access;
    std::vector<uint64_t> shape;
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
    std::vector<uint8_t> packed;
    std::vector<uint8_t *> leafShadows;
};

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT);

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
            for (uint64_t extent : view.shape) {
                if (!extent || extent > std::numeric_limits<size_t>::max() / staged.elementCount)
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
                if (!value || !valueMatches(*value, leaf))
                    return fail(context, "autodiff forward TensorView input '" + leaf.path +
                                             "' does not match profile reflection");
                if (!appendStorageRange(value->data, value->size, writable, storageRanges))
                    return fail(context, "observable native CPU autodiff Storage/output ranges overlap or overflow");
                auto *shadow =
                    static_cast<uint8_t *>(transaction.stageStorage(value->data, value->size, preserve, writable));
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
                for (size_t element = 0; element < staged.elementCount; ++element)
                    std::memcpy(staged.packed.data() + element * view.elementLayout.byteSize + layoutLeaf.byteOffset,
                                shadow + element * leafBytes, leafBytes);
            }
            tensorViews.push_back(std::move(staged));
            uint8_t *descriptor = arguments.data() + argument.offset;
            const uintptr_t pointer = reinterpret_cast<uintptr_t>(tensorViews.back().packed.data());
            const uint64_t zero = 0;
            std::memcpy(descriptor, &pointer, sizeof(pointer));
            std::memcpy(descriptor + sizeof(uint64_t), &zero, sizeof(zero));
            const size_t rank = view.shape.size();
            for (size_t dimension = 0; dimension < rank; ++dimension) {
                const uint64_t extent = view.shape[dimension];
                std::memcpy(descriptor + sizeof(uint64_t) * (2 + dimension), &extent, sizeof(extent));
            }
            uint64_t stride = 1;
            for (size_t dimension = rank; dimension-- > 0;) {
                std::memcpy(descriptor + sizeof(uint64_t) * (2 + rank + dimension), &stride, sizeof(stride));
                stride *= view.shape[dimension];
            }
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
    if (!layout.argumentsSize || !layout.resultsSize) {
        error = "autodiff profile has an empty host Value ABI";
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
            std::vector<uint64_t> shape;
            const std::string access = value.value("access", "");
            size_t elementCount = 1;
            for (const nlohmann::json &extentValue : value["source_shape"]) {
                if (!extentValue.is_number_unsigned()) {
                    error = "autodiff TensorView argument has an invalid shape";
                    return false;
                }
                const uint64_t extent = extentValue.get<uint64_t>();
                if (!extent || extent > SIZE_MAX / elementCount) {
                    error = "autodiff TensorView argument shape is empty or overflows";
                    return false;
                }
                elementCount *= static_cast<size_t>(extent);
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
                if (!scalarSize || !layoutLeaf.scalarCount || elementCount > SIZE_MAX / layoutLeaf.scalarCount ||
                    elementCount * layoutLeaf.scalarCount > SIZE_MAX / scalarSize) {
                    error = "autodiff TensorView element leaf size overflows";
                    return false;
                }
                std::string path = argument.name;
                for (const ValuePathComponent &component : layoutLeaf.path)
                    path += component.field ? "." + *component.field : "." + std::to_string(component.index);
                std::vector<uint64_t> leafShape = shape;
                leafShape.insert(leafShape.end(), layoutLeaf.shape.begin(), layoutLeaf.shape.end());
                tensorView.leaves.push_back({std::move(path), *dtype,
                                             elementCount * layoutLeaf.scalarCount * scalarSize, scalarSize,
                                             std::move(leafShape)});
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
    if ((*entry)["results"].size() != 1) {
        error = "CPU autodiff profile requires one result";
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
    size_t index = 0;
    for (const nlohmann::json &leafValue : result["value_layout"]["leaves"]) {
        HostFrameLeaf leaf;
        if (!parseLeaf(leafValue, "result." + std::to_string(index++), resultOffset, leaf, error))
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

VernonStatus invokeBackwardAndAccumulate(VernonRuntimeContext &context, const CpuKernelState &backward,
                                         const HostProfileLayout &layout, const Signature &signature,
                                         std::vector<uint8_t> &arguments,
                                         std::vector<std::vector<uint8_t>> &stagedGradients) {
    std::vector<uint8_t> results(layout.resultsSize);
    const VernonCpuInvocation invocation{arguments.data(), arguments.size(), results.data(), results.size(), nullptr};
    const VernonStatus status = backward.entry(&invocation);
    if (status != VERNON_STATUS_OK)
        return fail(context, "autodiff backward profile invocation failed", status);
    for (size_t index = 0; index < signature.gradients.size(); ++index) {
        const HostFrameLeaf &source = layout.results[index];
        uint8_t *destination = stagedGradients[index].data();
        const uint8_t *contribution = results.data() + source.frameOffset;
        const size_t scalarSize = dtypeSize(source.value.dtype);
        for (size_t offset = 0; offset < source.value.byteSize; offset += scalarSize) {
            if (source.value.dtype == VERNON_DATA_F32) {
                float current;
                float value;
                std::memcpy(&current, destination + offset, sizeof(current));
                std::memcpy(&value, contribution + offset, sizeof(value));
                current += value;
                std::memcpy(destination + offset, &current, sizeof(current));
            } else if (source.value.dtype == VERNON_DATA_F64) {
                double current;
                double value;
                std::memcpy(&current, destination + offset, sizeof(current));
                std::memcpy(&value, contribution + offset, sizeof(value));
                current += value;
                std::memcpy(destination + offset, &current, sizeof(current));
            } else {
                return fail(context, "CPU pullback can only reduce floating gradients");
            }
        }
    }
    return VERNON_STATUS_OK;
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
                                   std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots)
        : context_(context), backward_(std::move(backward)), layout_(std::move(layout)),
          signature_(std::move(signature)), computeGrid_(computeGrid), arguments_(std::move(arguments)),
          tapeSnapshots_(std::move(tapeSnapshots)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) override {
        if (signature_.cotangents.size() != 1)
            return fail(context_, "CPU pullback requires one cotangent leaf");
        ValueAbi cotangentAbi = signature_.cotangents.front();
        if (!materializeCarrierValue(cotangentAbi, computeGrid_))
            return fail(context_, "CPU pullback cotangent size overflows");
        std::vector<uint8_t> cotangent;
        if (!makeCotangentBytes(cotangents, cotangentAbi, cotangent, invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        const HostArgument *cotangentArgument = nullptr;
        for (const HostArgument &argument : layout_.arguments)
            if (argument.builtin.empty()) {
                if (cotangentArgument || argument.leaves.size() != 1)
                    return fail(context_, "CPU pullback has an invalid cotangent ABI");
                cotangentArgument = &argument;
            }
        if (!cotangentArgument)
            return fail(context_, "CPU pullback has no cotangent ABI");
        const HostFrameLeaf &cotangentLeaf = cotangentArgument->leaves.front();
        std::vector<VernonAdValue *> destinations;
        std::vector<std::vector<uint8_t>> stagedGradients;
        if (VernonStatus status =
                prepareGradientDestinations(context_, signature_, gradients, destinations, stagedGradients);
            status != VERNON_STATUS_OK)
            return status;
        for (size_t invocationIndex = 0; invocationIndex < arguments_.size(); ++invocationIndex) {
            std::vector<uint8_t> &arguments = arguments_[invocationIndex];
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
            std::memcpy(arguments.data() + cotangentLeaf.frameOffset,
                        cotangent.data() + invocationIndex * signature_.cotangents.front().byteSize,
                        signature_.cotangents.front().byteSize);
            if (VernonStatus status =
                    invokeBackwardAndAccumulate(context_, *backward_, layout_, signature_, arguments, stagedGradients);
                status != VERNON_STATUS_OK)
                return status;
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
};

template <typename Invoke, typename Finish>
VernonStatus runCpuForward(VernonRuntimeContext &context, const HostProfileLayout &layout, const Signature &signature,
                           VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                           std::unique_ptr<PullbackExecution> &pullback, Invoke &&invoke, Finish &&finish) {
    size_t invocationCount = 0;
    if (!carrierCount(computeGrid, invocationCount))
        return fail(context, "native CPU autodiff launch size overflows");
    if (inputs.value_count != signature.inputs.size() || outputs.value_count != 1 || signature.outputs.size() != 1)
        return fail(context, "autodiff forward values do not match the host profile");
    std::vector<uint8_t> arguments(layout.argumentsSize);
    std::vector<StorageRange> storageRanges;
    std::vector<StagedTensorView> tensorViews;
    VernonAdValue *output = findValue(outputs, signature.outputs.front().path);
    ValueAbi outputAbi = signature.outputs.front();
    if (!materializeCarrierValue(outputAbi, computeGrid))
        return fail(context, "native CPU autodiff output size overflows");
    if (!output || !valueMatches(*output, outputAbi))
        return fail(context, "autodiff output does not match profile reflection");
    if (!appendStorageRange(output->data, output->size, true, storageRanges))
        return fail(context, "native CPU autodiff output address range overflows");
    HostEffectTransaction transaction(outputAbi.byteSize);
    uint8_t *stagedOutput = transaction.stagedOutput();
    if (outputAbi.byteSize && !stagedOutput)
        return fail(context, "cannot allocate native CPU autodiff output shadow", VERNON_STATUS_INTERNAL_ERROR);
    if (VernonStatus status =
            stageForwardInputs(context, layout, inputs, arguments, transaction, storageRanges, tensorViews);
        status != VERNON_STATUS_OK)
        return status;
    for (size_t invocationIndex = 0; invocationIndex < invocationCount; ++invocationIndex) {
        std::vector<uint8_t> results(layout.resultsSize);
        if (VernonStatus status = invoke(invocationIndex, arguments, results); status != VERNON_STATUS_OK)
            return status;
        std::memcpy(stagedOutput + invocationIndex * signature.outputs.front().byteSize,
                    results.data() + layout.results.front().frameOffset, signature.outputs.front().byteSize);
    }
    std::unique_ptr<PullbackExecution> pendingPullback;
    if (VernonStatus status = finish(invocationCount, pendingPullback); status != VERNON_STATUS_OK)
        return status;
    if (!pendingPullback)
        return fail(context, "native CPU autodiff did not create a pullback", VERNON_STATUS_INTERNAL_ERROR);
    if (VernonStatus status = flushStagedTensorViews(context, tensorViews); status != VERNON_STATUS_OK)
        return status;
    if (!transaction.commit(output->data))
        return fail(context, "autodiff effect transaction was already committed");
    pullback = std::move(pendingPullback);
    return VERNON_STATUS_OK;
}

class StructuredCpuExecutable final : public HostExecutable {
public:
    StructuredCpuExecutable(VernonRuntimeContext &context, HostProfileLayout forwardLayout,
                            HostProfileLayout backwardLayout, std::shared_ptr<CpuKernelState> forward,
                            std::shared_ptr<CpuKernelState> backward, Signature signature)
        : context_(context), forwardLayout_(std::move(forwardLayout)), backwardLayout_(std::move(backwardLayout)),
          forward_(std::move(forward)), backward_(std::move(backward)), signature_(std::move(signature)),
          tapePolicy_(context.cpuTapePolicy) {}

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
            [&](size_t invocationCount, std::unique_ptr<PullbackExecution> &pending) {
                std::vector<std::vector<uint8_t>> backwardArguments(
                    invocationCount, std::vector<uint8_t>(backwardLayout_.argumentsSize));
                pending = std::make_unique<StructuredCpuPullbackExecution>(
                    context_, backward_, backwardLayout_, signature_, computeGrid, std::move(backwardArguments),
                    std::move(tapeSnapshots));
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
};

class LegacyCpuPullbackExecution final : public PullbackExecution {
public:
    LegacyCpuPullbackExecution(VernonRuntimeContext &context, std::shared_ptr<CpuKernelState> backward,
                               HostProfileLayout layout, Signature signature, VernonLaunchSize computeGrid,
                               std::vector<std::vector<uint8_t>> arguments)
        : context_(context), backward_(std::move(backward)), layout_(std::move(layout)),
          signature_(std::move(signature)), computeGrid_(computeGrid), arguments_(std::move(arguments)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) override {
        if (signature_.cotangents.size() != 1)
            return fail(context_, "legacy CPU pullback requires one cotangent leaf");
        ValueAbi cotangentAbi = signature_.cotangents.front();
        if (!materializeCarrierValue(cotangentAbi, computeGrid_))
            return fail(context_, "CPU pullback cotangent size overflows");
        std::vector<uint8_t> cotangent;
        if (!makeCotangentBytes(cotangents, cotangentAbi, cotangent, invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        if (layout_.arguments.size() != 2 || layout_.arguments[1].leaves.size() != 1)
            return fail(context_, "legacy CPU pullback has an invalid cotangent ABI");
        const HostFrameLeaf &cotangentLeaf = layout_.arguments[1].leaves.front();
        std::vector<VernonAdValue *> destinations;
        std::vector<std::vector<uint8_t>> stagedGradients;
        if (VernonStatus status =
                prepareGradientDestinations(context_, signature_, gradients, destinations, stagedGradients);
            status != VERNON_STATUS_OK)
            return status;
        for (size_t invocationIndex = 0; invocationIndex < arguments_.size(); ++invocationIndex) {
            std::vector<uint8_t> &arguments = arguments_[invocationIndex];
            std::memcpy(arguments.data() + cotangentLeaf.frameOffset,
                        cotangent.data() + invocationIndex * signature_.cotangents.front().byteSize,
                        signature_.cotangents.front().byteSize);
            if (VernonStatus status =
                    invokeBackwardAndAccumulate(context_, *backward_, layout_, signature_, arguments, stagedGradients);
                status != VERNON_STATUS_OK)
                return status;
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
};

class LegacyCpuExecutable final : public HostExecutable {
public:
    LegacyCpuExecutable(VernonRuntimeContext &context, HostProfileLayout forwardLayout,
                        HostProfileLayout backwardLayout, std::shared_ptr<CpuKernelState> forward,
                        std::shared_ptr<CpuKernelState> backward, Signature signature)
        : context_(context), forwardLayout_(std::move(forwardLayout)), backwardLayout_(std::move(backwardLayout)),
          forward_(std::move(forward)), backward_(std::move(backward)), signature_(std::move(signature)) {}

    const Signature &signature() const override { return signature_; }

    VernonStatus forward(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        std::vector<std::vector<uint8_t>> backwardArguments;
        return runCpuForward(
            context_, forwardLayout_, signature_, computeGrid, inputs, outputs, pullback,
            [&](size_t invocationIndex, std::vector<uint8_t> &arguments, std::vector<uint8_t> &results) {
                for (const HostArgument &argument : forwardLayout_.arguments)
                    if (!argument.builtin.empty() &&
                        !writeGlobalInvocationId(argument, computeGrid, invocationIndex, arguments))
                        return fail(context_, "native CPU autodiff has an unsupported builtin ABI");
                const VernonCpuInvocation invocation{arguments.data(), arguments.size(), results.data(), results.size(),
                                                     nullptr};
                const VernonStatus status = forward_->entry(&invocation);
                if (status != VERNON_STATUS_OK)
                    return fail(context_, "autodiff forward profile invocation failed", status);
                backwardArguments.emplace_back(backwardLayout_.argumentsSize);
                for (size_t index = 0; index < signature_.tape.size(); ++index) {
                    const HostFrameLeaf &source = forwardLayout_.results[index + 1];
                    const HostFrameLeaf &destination = backwardLayout_.arguments[0].leaves[index];
                    std::memcpy(backwardArguments.back().data() + destination.frameOffset,
                                results.data() + source.frameOffset, source.value.byteSize);
                }
                return VERNON_STATUS_OK;
            },
            [&](size_t invocationCount, std::unique_ptr<PullbackExecution> &pending) {
                if (backwardArguments.size() != invocationCount)
                    return fail(context_, "legacy CPU autodiff captured an incomplete tape",
                                VERNON_STATUS_INTERNAL_ERROR);
                pending = std::make_unique<LegacyCpuPullbackExecution>(context_, backward_, backwardLayout_, signature_,
                                                                       computeGrid, std::move(backwardArguments));
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
};

bool validateDifferentiationSignature(VernonRuntimeContext &context, const Signature &signature) {
    if (signature.outputs.empty() || signature.outputs.size() != signature.cotangents.size()) {
        invocationDiagnostic(context) = "autodiff output and cotangent leaf counts do not match";
        return false;
    }
    for (size_t index = 0; index < signature.outputs.size(); ++index)
        if (signature.outputs[index].path != signature.cotangents[index].path ||
            !derivativeAbiMatches(signature.outputs[index], signature.cotangents[index])) {
            invocationDiagnostic(context) = "autodiff output and cotangent profile ABIs do not match";
            return false;
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
    size_t backwardValueArguments = 0;
    for (const HostArgument &argument : backward.arguments)
        backwardValueArguments += argument.builtin.empty();
    if (!forward.tapeAllocatorOffset || forward.tapeRootRegionOffset || !backward.tapeAllocatorOffset ||
        !backward.tapeRootRegionOffset || forward.results.size() != 1 || backwardValueArguments != 1 ||
        backward.results.size() != gradientPaths.size()) {
        invocationDiagnostic(context) = "structured CPU autodiff profile does not use the dynamic tape ABI";
        return false;
    }
    Signature signature;
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
    ValueAbi output = forward.results.front().value;
    output.path = "output";
    signature.outputs.push_back(std::move(output));
    const HostArgument *cotangentArgument = nullptr;
    for (const HostArgument &argument : backward.arguments)
        if (argument.builtin.empty())
            cotangentArgument = &argument;
    ValueAbi cotangent = cotangentArgument->leaves.front().value;
    cotangent.path = "output";
    signature.cotangents.push_back(std::move(cotangent));
    if (!derivativeAbiMatches(signature.outputs.front(), signature.cotangents.front())) {
        invocationDiagnostic(context) = "autodiff output and cotangent profile ABIs do not match";
        return false;
    }
    for (size_t index = 0; index < gradientPaths.size(); ++index) {
        ValueAbi gradient = backward.results[index].value;
        gradient.path = gradientPaths[index];
        signature.gradients.push_back(std::move(gradient));
    }
    if (!validateDifferentiationSignature(context, signature))
        return false;
    if (!context.cpuTapePolicy)
        context.cpuTapePolicy = std::make_shared<HostTapeMemoryPolicy>();
    executable = std::make_shared<StructuredCpuExecutable>(context, std::move(forward), std::move(backward),
                                                           std::move(forwardKernel), std::move(backwardKernel),
                                                           std::move(signature));
    return true;
}

bool finishLegacyCpuExecutable(VernonRuntimeContext &context, const Stage &forwardStage, const Stage &backwardStage,
                               std::shared_ptr<CpuKernelState> forwardKernel,
                               std::shared_ptr<CpuKernelState> backwardKernel,
                               const std::vector<std::string> &gradientPaths, std::shared_ptr<Executable> &executable) {
    HostProfileLayout forward;
    HostProfileLayout backward;
    if (!parseProfile(forwardStage, forward, invocationDiagnostic(context)) ||
        !parseProfile(backwardStage, backward, invocationDiagnostic(context)))
        return false;
    if (forward.tapeAllocatorOffset || forward.tapeRootRegionOffset || backward.tapeAllocatorOffset ||
        backward.tapeRootRegionOffset || forward.results.size() < 2 || backward.arguments.size() != 2 ||
        backward.arguments[0].leaves.size() + 1 != forward.results.size() || backward.arguments[1].leaves.size() != 1 ||
        backward.results.size() != gradientPaths.size()) {
        invocationDiagnostic(context) = "legacy CPU autodiff profile does not match its explicit fixed protocol";
        return false;
    }
    Signature signature;
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
    ValueAbi output = forward.results.front().value;
    output.path = "output";
    signature.outputs.push_back(std::move(output));
    for (size_t index = 1; index < forward.results.size(); ++index) {
        ValueAbi tape = forward.results[index].value;
        tape.path = "tape." + std::to_string(index - 1);
        signature.tape.push_back(std::move(tape));
    }
    ValueAbi cotangent = backward.arguments[1].leaves.front().value;
    cotangent.path = "output";
    signature.cotangents.push_back(std::move(cotangent));
    if (!derivativeAbiMatches(signature.outputs.front(), signature.cotangents.front())) {
        invocationDiagnostic(context) = "autodiff output and cotangent profile ABIs do not match";
        return false;
    }
    for (size_t index = 0; index < signature.tape.size(); ++index) {
        const ValueAbi &backwardTape = backward.arguments[0].leaves[index].value;
        if (!sameValueAbi(signature.tape[index], backwardTape)) {
            invocationDiagnostic(context) = "legacy CPU autodiff tape leaves do not match backward reflection";
            return false;
        }
    }
    for (size_t index = 0; index < gradientPaths.size(); ++index) {
        ValueAbi gradient = backward.results[index].value;
        gradient.path = gradientPaths[index];
        signature.gradients.push_back(std::move(gradient));
    }
    if (!validateDifferentiationSignature(context, signature))
        return false;
    executable = std::make_shared<LegacyCpuExecutable>(context, std::move(forward), std::move(backward),
                                                       std::move(forwardKernel), std::move(backwardKernel),
                                                       std::move(signature));
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

bool loadLegacyCpuExecutable(VernonRuntimeContext &context, const Stage &forwardStage, const Stage &backwardStage,
                             const std::vector<std::string> &gradientPaths, std::shared_ptr<Executable> &executable) {
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
    return finishLegacyCpuExecutable(context, forwardStage, backwardStage, std::move(forwardKernel),
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
    if (!forwardStage.autodiff || !backwardStage.autodiff || forwardStage.autodiff->protocol != "dynamic_v2" ||
        backwardStage.autodiff->protocol != "dynamic_v2") {
        invocationDiagnostic(context) = "structured CPU autodiff requires explicit dynamic_v2 profiles";
        return false;
    }
    return loadStructuredCpuExecutable(context, forwardStage, backwardStage, gradientPaths, executable);
}

bool createLegacyCpuExecutable(VernonRuntimeContext &context, const Stage &forwardStage, const Stage &backwardStage,
                               const std::vector<std::string> &gradientPaths, std::shared_ptr<Executable> &executable) {
    if (!forwardStage.autodiff || !backwardStage.autodiff || forwardStage.autodiff->protocol != "legacy_fixed" ||
        backwardStage.autodiff->protocol != "legacy_fixed") {
        invocationDiagnostic(context) = "legacy CPU autodiff requires explicit legacy_fixed profiles";
        return false;
    }
    return loadLegacyCpuExecutable(context, forwardStage, backwardStage, gradientPaths, executable);
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
    forwardStage.autodiff = AutodiffStageMetadata{"", std::string(forwardProtocol.data, forwardProtocol.size), ""};
    Stage backwardStage;
    backwardStage.entry.assign(backwardName.data, backwardName.size);
    backwardStage.reflection.assign(backwardReflection.data, backwardReflection.size);
    backwardStage.autodiff = AutodiffStageMetadata{"", std::string(backwardProtocol.data, backwardProtocol.size), ""};
    if (forwardStage.autodiff->protocol != "dynamic_v2" || backwardStage.autodiff->protocol != "dynamic_v2") {
        invocationDiagnostic(context) = "direct structured CPU autodiff requires explicit dynamic_v2 profiles";
        return false;
    }
    return loadStructuredCpuEntryExecutable(context, forwardStage, forwardEntry, backwardStage, backwardEntry,
                                            gradientPaths, executable);
}

} // namespace vernon::runtime::ad
