#include "pipeline_metadata.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace vernon::runtime {

bool parseReflection(const nlohmann::json &root, const std::string &selected, ReflectedEntry &output,
                     std::string &error) {
    if (!root.is_object() || root.value("gpu_launch_abi_version", 0) != 1 || !root.contains("entries") ||
        !root["entries"].is_array()) {
        error = "unsupported or invalid compute reflection";
        return false;
    }
    for (const nlohmann::json &entry : root["entries"]) {
        if (!entry.is_object() || entry.value("name", "") != selected)
            continue;
        output.cpuArgumentsSize = entry.value("cpu_arguments_size", size_t{0});
        if (entry.contains("workgroup_size") && entry["workgroup_size"].is_array() &&
            entry["workgroup_size"].size() == 3)
            for (size_t index = 0; index < 3; ++index)
                output.workgroup[index] = entry["workgroup_size"][index].get<uint32_t>();
        if (!output.cpuArgumentsSize || !entry.contains("arguments") || !entry["arguments"].is_array()) {
            error = "CPU entry has no argument layout";
            return false;
        }
        for (const nlohmann::json &value : entry["arguments"]) {
            if (!value.is_object()) {
                error = "CPU entry contains invalid argument reflection";
                return false;
            }
            ReflectedArgument argument;
            argument.kind = value.value("kind", "scalar");
            if (argument.kind == "tensor_value")
                argument.kind = "scalar";
            argument.builtin = value.value("builtin", "");
            argument.cpuOffset = value.value("cpu_offset", size_t{0});
            argument.cpuSize = value.value("cpu_size", size_t{0});
            argument.physicalSize = value.value("physical_size", argument.cpuSize);
            argument.alignment = value.value("alignment", size_t{1});
            argument.descriptorSet = value.value("vernon.set", uint32_t{0});
            argument.binding = value.value("vernon.binding", UINT32_MAX);
            if (value.contains("storage_leaves") && value["storage_leaves"].is_array()) {
                for (const nlohmann::json &leaf : value["storage_leaves"]) {
                    if (!leaf.is_object() || !leaf.contains("element_size") || !leaf.contains("binding")) {
                        error = "Tensor storage-leaf reflection is invalid";
                        return false;
                    }
                    argument.storageLeaves.push_back({leaf["element_size"].get<size_t>(),
                                                      leaf.value("byte_offset", size_t{0}),
                                                      leaf["binding"].get<uint32_t>()});
                }
            }
            const std::string dtype = value.value("dtype", "");
            const size_t elementSize = value.value("element_abi_size", dtype == "f64"    ? size_t{8}
                                                                       : dtype == "f16"  ? size_t{2}
                                                                       : dtype == "bool" ? size_t{1}
                                                                                         : size_t{4});
            argument.tensorElementSize = elementSize;
            if (value.contains("shape") && value["shape"].is_array()) {
                size_t elements = 1;
                for (const nlohmann::json &dimension : value["shape"]) {
                    if (!dimension.is_number_unsigned()) {
                        elements = 0;
                        break;
                    }
                    elements *= dimension.get<size_t>();
                }
                argument.tensorElements = elements;
                argument.tensorBytes = elements * elementSize;
            }
            if (!argument.cpuSize || argument.cpuOffset > output.cpuArgumentsSize ||
                argument.cpuSize > output.cpuArgumentsSize - argument.cpuOffset ||
                (argument.kind != "tensor" && argument.kind != "scalar" && argument.kind != "builtin")) {
                error = "CPU argument layout is invalid";
                return false;
            }
            output.arguments.push_back(std::move(argument));
        }
        return true;
    }
    error = "selected CPU entry is absent from reflection";
    return false;
}

std::optional<VernonPipelineArgumentKind> pipelineArgumentKind(const std::string &kind) {
    if (kind == "tensor")
        return VERNON_PIPELINE_TENSOR;
    if (kind == "texture")
        return VERNON_PIPELINE_TEXTURE;
    if (kind == "sampler")
        return VERNON_PIPELINE_SAMPLER;
    return std::nullopt;
}

std::optional<VernonDataType> pipelineDataType(const std::string &dtype) {
    if (dtype == "bool")
        return VERNON_DATA_BOOL;
    if (dtype == "i32")
        return VERNON_DATA_I32;
    if (dtype == "u32")
        return VERNON_DATA_U32;
    if (dtype == "f16")
        return VERNON_DATA_F16;
    if (dtype == "f32")
        return VERNON_DATA_F32;
    if (dtype == "f64")
        return VERNON_DATA_F64;
    if (dtype == "u8")
        return VERNON_DATA_U8;
    return std::nullopt;
}

std::optional<VernonValueAccess> pipelineValueAccess(const std::string &access) {
    if (access == "read")
        return VERNON_ACCESS_READ;
    if (access == "write")
        return VERNON_ACCESS_WRITE;
    if (access == "read_write")
        return VERNON_ACCESS_READ_WRITE;
    return std::nullopt;
}

} // namespace vernon::runtime
