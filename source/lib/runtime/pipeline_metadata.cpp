#include "pipeline_metadata.h"
#include "pipeline_manifest.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace vernon::runtime {

const char *physicalValueProfileName(VernonRuntimeBackend backend, const std::string &transport) {
    switch (backend) {
    case VERNON_RUNTIME_CPU:
        return "host_value";
    case VERNON_RUNTIME_CUDA:
        return "cuda_kernel_parameter";
    case VERNON_RUNTIME_VULKAN:
        if (transport == "push_constant")
            return "vulkan_push_constant";
        if (transport == "uniform_buffer")
            return "vulkan_std140_uniform_buffer";
        return "vulkan_std430_storage_buffer";
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        if (transport == "native_uniform" || transport == "push_constant")
            return "opengl_native_uniform";
        if (transport == "uniform_buffer")
            return "vulkan_std140_uniform_buffer";
        return "vulkan_std430_storage_buffer";
    case VERNON_RUNTIME_DIRECTX12:
        if (transport == "uniform_buffer" || transport == "push_constant")
            return "directx_constant_buffer";
        return "vulkan_std430_storage_buffer";
    }
    return "";
}

bool parseReflection(const nlohmann::json &root, const std::string &selected, ReflectedEntry &output,
                     VernonRuntimeBackend backend, std::string &error) {
    if (!root.is_object() || root.value("gpu_launch_abi_version", 0) != 1 || !root.contains("entries") ||
        !root["entries"].is_array()) {
        error = "unsupported or invalid compute reflection";
        return false;
    }
    for (const nlohmann::json &entry : root["entries"]) {
        if (!entry.is_object() || entry.value("name", "") != selected)
            continue;
        const char *profileName = physicalValueProfileName(backend, "storage_buffer");
        const auto entryLayouts = entry.find("physical_layouts");
        if (entryLayouts == entry.end() || !entryLayouts->is_object()) {
            error = "entry reflection has no physical layout table";
            return false;
        }
        const auto entryLayout = entryLayouts->find(profileName);
        if (entryLayout == entryLayouts->end() || !entryLayout->is_object() ||
            entryLayout->value("profile", std::string()) != profileName) {
            error = "entry reflection has no physical profile for the selected target";
            return false;
        }
        if (entryLayout->contains("packed_arguments_size")) {
            const size_t size = entryLayout->value("packed_arguments_size", size_t{0});
            if (!size) {
                error = "entry reflection has an invalid packed argument size";
                return false;
            }
            output.packedArguments = PackedArgumentsLayout{size};
        }
        if (backend == VERNON_RUNTIME_CPU && !output.packedArguments) {
            error = "CPU entry has no packed argument layout";
            return false;
        }
        if (entry.contains("workgroup_size") && entry["workgroup_size"].is_array() &&
            entry["workgroup_size"].size() == 3)
            for (size_t index = 0; index < 3; ++index)
                output.workgroup[index] = entry["workgroup_size"][index].get<uint32_t>();
        if (!entry.contains("arguments") || !entry["arguments"].is_array()) {
            error = "entry has no argument layout";
            return false;
        }
        for (const nlohmann::json &value : entry["arguments"]) {
            if (!value.is_object()) {
                error = "entry contains invalid argument reflection";
                return false;
            }
            ReflectedArgument argument;
            argument.kind = value.value("kind", "scalar");
            if (argument.kind == "tensor_value")
                argument.kind = "scalar";
            argument.builtin = value.value("builtin", "");
            const auto physicalLayouts = value.find("physical_layouts");
            if (physicalLayouts == value.end() || !physicalLayouts->is_object()) {
                error = "argument reflection has no physical layout table";
                return false;
            }
            const auto physical = physicalLayouts->find(profileName);
            if (physical == physicalLayouts->end() || !physical->is_object() ||
                physical->value("profile", std::string()) != profileName) {
                error = "argument reflection has no physical profile for the selected target";
                return false;
            }
            argument.physical.offset = physical->value("offset", size_t{0});
            argument.physical.size = physical->value("size", size_t{0});
            argument.physical.alignment = physical->value("alignment", size_t{1});
            if (!argument.physical.alignment || (argument.physical.alignment & (argument.physical.alignment - 1)) ||
                argument.physical.offset % argument.physical.alignment) {
                error = "argument reflection has an invalid physical alignment";
                return false;
            }
            const bool cudaStaticTensorValue =
                backend == VERNON_RUNTIME_CUDA && argument.kind == "tensor" && argument.physical.size != 0;
            if (cudaStaticTensorValue)
                argument.kind = "scalar";
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
            if (cudaStaticTensorValue)
                argument.storageLeaves.clear();
            size_t elementSize = 0;
            if (argument.kind == "tensor") {
                auto layout = value.find("element_layout");
                if (layout == value.end() || !layout->is_object() || !layout->contains("byte_size") ||
                    !(*layout)["byte_size"].is_number_unsigned()) {
                    error = "Tensor reflection has no canonical element layout";
                    return false;
                }
                elementSize = (*layout)["byte_size"].get<size_t>();
                if (!elementSize) {
                    error = "Tensor reflection has an empty canonical element layout";
                    return false;
                }
            }
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
            if (argument.kind != "tensor" && argument.kind != "scalar" && argument.kind != "builtin") {
                error = "argument layout has an unsupported kind";
                return false;
            }
            if (backend == VERNON_RUNTIME_CPU &&
                (!argument.physical.size || argument.physical.offset > output.packedArguments->size ||
                 argument.physical.size > output.packedArguments->size - argument.physical.offset)) {
                error = "CPU argument layout is invalid";
                return false;
            }
            if (backend != VERNON_RUNTIME_CPU && argument.kind == "scalar" && !argument.physical.size) {
                error = "value argument has no physical size for the selected target";
                return false;
            }
            output.arguments.push_back(std::move(argument));
        }
        return true;
    }
    error = "selected entry is absent from reflection";
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

VernonValueLayoutView pipelineValueLayout(const ValueLayout &layout) {
    return {sizeof(VernonValueLayoutView),
            layout.byteSize,
            layout.alignment,
            {layout.layoutHash.data(), layout.layoutHash.size()},
            layout.abiLeaves.empty() ? nullptr : layout.abiLeaves.data(),
            layout.abiLeaves.size()};
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
