#include "pipeline_metadata.h"
#include "pipeline_manifest.h"

#include "VernonVersions.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>
#include <string_view>
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
    case VERNON_RUNTIME_METAL:
        return "metal_constant_buffer";
    }
    return "";
}

bool parseReflection(const nlohmann::json &root, const std::string &selected, ReflectedEntry &output,
                     VernonRuntimeBackend backend, std::string &error) {
    if (!root.is_object() || root.value("compiler_contract_version", 0) != VERNON_COMPILER_CONTRACT_VERSION ||
        root.value("pipeline_version", 0) != VERNON_PIPELINE_VERSION || !root.contains("entries") ||
        !root["entries"].is_array()) {
        error = "unsupported or invalid compute reflection";
        return false;
    }
    for (const nlohmann::json &entry : root["entries"]) {
        if (!entry.is_object() || entry.value("name", "") != selected)
            continue;
        const char *profileName = physicalValueProfileName(backend, "storage_buffer");
        const std::string_view selectedProfile = profileName;
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
        if (!parseDispatchContract(entry, output.dispatchContract, error))
            return false;
        if (!entry.contains("arguments") || !entry["arguments"].is_array()) {
            error = "entry has no argument layout";
            return false;
        }
        for (const nlohmann::json &value : entry["arguments"]) {
            if (!value.is_object() || !value.contains("kind") || !value["kind"].is_string()) {
                error = "entry contains argument reflection without an explicit kind";
                return false;
            }
            ReflectedArgument argument;
            argument.sourceName = value.value("vernon.source_name", "");
            argument.kind = value["kind"].get<std::string>();
            if (argument.kind != "scalar" && argument.kind != "tensor_value" && argument.kind != "tensor" &&
                argument.kind != "image" && argument.kind != "sampler" && argument.kind != "builtin") {
                error = "entry contains argument reflection with an unsupported kind";
                return false;
            }
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
            const auto parseExtent = [](const nlohmann::json &object, const char *key, size_t &extent) {
                const auto value = object.find(key);
                if (value == object.end() || !value->is_number_integer())
                    return false;
                if (value->is_number_unsigned()) {
                    const uint64_t parsed = value->get<uint64_t>();
                    if (parsed > std::numeric_limits<size_t>::max())
                        return false;
                    extent = static_cast<size_t>(parsed);
                    return true;
                }
                const int64_t parsed = value->get<int64_t>();
                if (parsed < 0 || static_cast<uint64_t>(parsed) > std::numeric_limits<size_t>::max())
                    return false;
                extent = static_cast<size_t>(parsed);
                return true;
            };
            if (physical->contains("frame_offset") &&
                !parseExtent(*physical, "frame_offset", argument.physical.offset)) {
                error = "argument reflection has an invalid frame offset";
                return false;
            }
            const std::string planKind = physical->value("kind", "");
            if (planKind == "resource_binding") {
                const std::string resourceKind = physical->value("resource_kind", "");
                const bool handle = resourceKind == "host_pointer" || resourceKind == "tensor_view_descriptor";
                const bool descriptor = resourceKind == "strided_memref_storage_leaves" ||
                                        resourceKind == "descriptor_storage_leaves" ||
                                        resourceKind == "image_reference" || resourceKind == "sampler_descriptor";
                const bool profileMatches = selectedProfile == "host_value" ? handle
                                            : selectedProfile == "cuda_kernel_parameter"
                                                ? resourceKind == "strided_memref_storage_leaves"
                                                : resourceKind == "descriptor_storage_leaves" ||
                                                      resourceKind == "image_reference" ||
                                                      resourceKind == "sampler_descriptor";
                if ((!handle && !descriptor) || !profileMatches ||
                    (handle && (!parseExtent(*physical, "size", argument.physical.size) ||
                                !parseExtent(*physical, "alignment", argument.physical.alignment))) ||
                    (descriptor && (physical->contains("size") || physical->contains("alignment")))) {
                    error = "resource binding reflection does not match its resource kind";
                    return false;
                }
                if (descriptor) {
                    argument.physical.size = 0;
                    argument.physical.alignment = 1;
                }
            } else if (planKind == "cpu_call" || planKind == "kernel_parameter" || planKind == "byte_transport" ||
                       planKind == "native_uniform") {
                const bool planMatchesProfile =
                    selectedProfile == "host_value" ? planKind == "cpu_call" || planKind == "byte_transport"
                    : selectedProfile == "cuda_kernel_parameter" ? planKind == "kernel_parameter"
                    : selectedProfile == "opengl_native_uniform" ? planKind == "native_uniform"
                                                                 : planKind == "byte_transport";
                if (!planMatchesProfile) {
                    error = "typed value interface plan kind does not match its profile";
                    return false;
                }
                const auto root = physical->find("root");
                if (root == physical->end() || !root->is_object() ||
                    !parseExtent(*root, "size", argument.physical.size) ||
                    !parseExtent(*root, "alignment", argument.physical.alignment)) {
                    error = "typed value interface plan has no recursive root";
                    return false;
                }
            } else {
                error = "argument reflection has an unsupported interface plan";
                return false;
            }
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
            if (value.contains("tensor_view_descriptor")) {
                const auto &descriptor = value["tensor_view_descriptor"];
                if (!descriptor.is_object() || !descriptor.contains("rank") ||
                    !descriptor["rank"].is_number_unsigned() || !descriptor.contains("offset_binding") ||
                    !descriptor["offset_binding"].is_number_unsigned() || !descriptor.contains("extent_bindings") ||
                    !descriptor["extent_bindings"].is_array() || !descriptor.contains("stride_bindings") ||
                    !descriptor["stride_bindings"].is_array()) {
                    error = "TensorView reflection has an invalid descriptor layout";
                    return false;
                }
                TensorViewDescriptorLayout layout;
                layout.rank = descriptor["rank"].get<uint32_t>();
                layout.offsetBinding = descriptor["offset_binding"].get<uint32_t>();
                layout.extentBindings = descriptor["extent_bindings"].get<std::vector<uint32_t>>();
                layout.strideBindings = descriptor["stride_bindings"].get<std::vector<uint32_t>>();
                if (!layout.rank || layout.extentBindings.size() != layout.rank ||
                    layout.strideBindings.size() != layout.rank) {
                    error = "TensorView descriptor rank does not match its binding sequence";
                    return false;
                }
                argument.tensorViewDescriptor = std::move(layout);
            }
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
            if (value.contains("source_shape") && value["source_shape"].is_array())
                for (const nlohmann::json &dimension : value["source_shape"]) {
                    if (!dimension.is_number_integer()) {
                        error = "TensorView source shape reflection is invalid";
                        return false;
                    }
                    argument.sourceShape.push_back(dimension.get<int64_t>());
                }
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
            if (argument.kind != "tensor" && argument.kind != "scalar" && argument.kind != "builtin" &&
                argument.kind != "image" && argument.kind != "sampler") {
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
        const auto footprintArgument = [&](const std::string &owner) {
            return std::find_if(output.arguments.begin(), output.arguments.end(),
                                [&](const ReflectedArgument &candidate) { return candidate.sourceName == owner; });
        };
        if (entry.contains("effects")) {
            const nlohmann::json &effects = entry["effects"];
            if (!effects.is_array()) {
                error = "storage-effect reflection is invalid";
                return false;
            }
            for (const nlohmann::json &value : effects) {
                if (!value.is_object() || !value.contains("kind") || !value["kind"].is_string() ||
                    !value.contains("owner") || !value["owner"].is_string() || !value.contains("region") ||
                    !value["region"].is_string() || !value.contains("indices") || !value["indices"].is_array()) {
                    error = "storage-effect reflection is invalid";
                    return false;
                }
                const std::string kind = value["kind"].get<std::string>();
                if (kind != "read" && kind != "read_write")
                    continue;
                const std::string owner = value["owner"].get<std::string>();
                const auto argument = footprintArgument(owner);
                if (argument == output.arguments.end() || argument->kind != "tensor") {
                    error = "TensorView read footprint names an unknown or non-TensorView argument";
                    return false;
                }
                TensorViewWriteFootprint footprint;
                footprint.argument = static_cast<uint32_t>(std::distance(output.arguments.begin(), argument));
                footprint.owner = owner;
                const std::string region = value["region"].get<std::string>();
                footprint.wholeView = region == "unknown";
                if (!footprint.wholeView && region != "element") {
                    error = "TensorView read footprint has an unsupported region";
                    return false;
                }
                for (const nlohmann::json &index : value["indices"]) {
                    if (!index.is_number_unsigned()) {
                        error = "TensorView read footprint has an invalid element index";
                        return false;
                    }
                    footprint.indices.push_back(index.get<uint64_t>());
                }
                const size_t rank = argument->tensorViewDescriptor ? argument->tensorViewDescriptor->rank
                                                                   : argument->sourceShape.size();
                if ((footprint.wholeView && !footprint.indices.empty()) ||
                    (!footprint.wholeView && footprint.indices.size() != rank)) {
                    error = "TensorView read footprint rank does not match its reflected shape";
                    return false;
                }
                for (size_t axis = 0; axis < footprint.indices.size() && axis < argument->sourceShape.size(); ++axis)
                    if (argument->sourceShape[axis] >= 0 &&
                        footprint.indices[axis] >= static_cast<uint64_t>(argument->sourceShape[axis])) {
                        error = "TensorView read footprint exceeds its reflected shape";
                        return false;
                    }
                output.readFootprints.push_back(std::move(footprint));
            }
        }
        if (entry.contains("tensor_view_write_footprints")) {
            const nlohmann::json &footprints = entry["tensor_view_write_footprints"];
            if (!footprints.is_array()) {
                error = "TensorView write-footprint reflection is invalid";
                return false;
            }
            for (const nlohmann::json &value : footprints) {
                if (!value.is_object() || value.value("version", 0) != 1 || !value.contains("owner") ||
                    !value["owner"].is_string() || !value.contains("kind") || !value["kind"].is_string() ||
                    !value.contains("indices") || !value["indices"].is_array()) {
                    error = "TensorView write-footprint reflection is invalid";
                    return false;
                }
                const std::string owner = value["owner"].get<std::string>();
                const auto argument = footprintArgument(owner);
                if (argument == output.arguments.end()) {
                    error = "TensorView write footprint names an unknown argument";
                    return false;
                }
                if (argument->kind != "tensor") {
                    error = "TensorView write footprint names a non-TensorView argument";
                    return false;
                }
                TensorViewWriteFootprint footprint;
                footprint.argument = static_cast<uint32_t>(std::distance(output.arguments.begin(), argument));
                footprint.owner = owner;
                const std::string kind = value["kind"].get<std::string>();
                footprint.wholeView = kind == "whole_view";
                if (!footprint.wholeView && kind != "exact_element") {
                    error = "TensorView write footprint has an unsupported kind";
                    return false;
                }
                for (const nlohmann::json &index : value["indices"]) {
                    if (!index.is_number_unsigned()) {
                        error = "TensorView write footprint has an invalid element index";
                        return false;
                    }
                    footprint.indices.push_back(index.get<uint64_t>());
                }
                const size_t rank = argument->tensorViewDescriptor ? argument->tensorViewDescriptor->rank
                                                                   : argument->sourceShape.size();
                if ((footprint.wholeView && !footprint.indices.empty()) ||
                    (!footprint.wholeView && footprint.indices.size() != rank)) {
                    error = "TensorView write footprint rank does not match its reflected shape";
                    return false;
                }
                for (size_t axis = 0; axis < footprint.indices.size() && axis < argument->sourceShape.size(); ++axis)
                    if (argument->sourceShape[axis] >= 0 &&
                        footprint.indices[axis] >= static_cast<uint64_t>(argument->sourceShape[axis])) {
                        error = "TensorView write footprint exceeds its reflected shape";
                        return false;
                    }
                output.writeFootprints.push_back(std::move(footprint));
            }
        }
        return true;
    }
    error = "selected entry is absent from reflection";
    return false;
}

std::optional<VernonPipelineArgumentKind> pipelineArgumentKind(const std::string &kind) {
    if (kind == "tensor")
        return VERNON_PIPELINE_TENSOR;
    if (kind == "image")
        return VERNON_PIPELINE_IMAGE;
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

bool configureImageBindingLayout(const Parameter &parameter, VernonRuntimeProviderBindingLayoutEntry &layout) {
    if (parameter.kind != "image")
        return false;
    const auto dimension = pipelineTextureDimension(parameter.dimension);
    if (!dimension)
        return false;
    layout.image_dimension = *dimension;
    if (parameter.bindingRole == "sampled") {
        if (parameter.sampleResultClass != "float")
            return false;
        layout.sample_result_class = VERNON_IMAGE_SAMPLE_FLOAT;
        return true;
    }
    if (parameter.bindingRole != "storage")
        return false;
    const auto format = pipelineTextureFormat(parameter.exactStorageFormat);
    if (!format)
        return false;
    layout.storage_image_format = *format;
    return true;
}

} // namespace vernon::runtime
