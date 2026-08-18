#include "runtime_cpu_preparation.h"

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

namespace vernon::runtime::ad::cpu {

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
            if (argument.size != sizeof(uint64_t) * (2 + 2 * shape.size()) ||
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

} // namespace vernon::runtime::ad::cpu
