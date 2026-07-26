#include "pipeline_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string_view>

namespace vernon::runtime {

std::optional<VernonTextureDimension> pipelineTextureDimension(const std::string &dimension) {
    if (dimension == "2d")
        return VERNON_TEXTURE_2D;
    if (dimension == "3d")
        return VERNON_TEXTURE_3D;
    if (dimension == "cube")
        return VERNON_TEXTURE_CUBE;
    return std::nullopt;
}

std::optional<VernonTextureFormat> pipelineTextureFormat(const std::string &format) {
    if (format == "rgba8_unorm")
        return VERNON_TEXTURE_RGBA8_UNORM;
    if (format == "rgba8_srgb")
        return VERNON_TEXTURE_RGBA8_SRGB;
    if (format == "rgba16_float")
        return VERNON_TEXTURE_RGBA16_FLOAT;
    if (format == "rgba32_float")
        return VERNON_TEXTURE_RGBA32_FLOAT;
    if (format == "r8_unorm")
        return VERNON_TEXTURE_R8_UNORM;
    if (format == "r16_float")
        return VERNON_TEXTURE_R16_FLOAT;
    if (format == "r32_float")
        return VERNON_TEXTURE_R32_FLOAT;
    if (format == "rg8_unorm")
        return VERNON_TEXTURE_RG8_UNORM;
    if (format == "rgb8_unorm")
        return VERNON_TEXTURE_RGB8_UNORM;
    if (format == "r11g11b10_float")
        return VERNON_TEXTURE_R11G11B10_FLOAT;
    return std::nullopt;
}

namespace {

bool parseUint64(const nlohmann::json &value, uint64_t &result) {
    if (!value.is_number_integer())
        return false;
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
        return true;
    }
    const int64_t parsed = value.get<int64_t>();
    if (parsed < 0)
        return false;
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool parseUse(const nlohmann::json &value, ParameterUse &use, std::string &error) {
    if (!value.is_object()) {
        error = "pipeline parameter use must be an object";
        return false;
    }
    use.stage = value.value("stage", "");
    use.interfaceKind = value.value("interface", "");
    use.uniformName = value.value("uniform_name", "");
    if (value.contains("dtype") && value["dtype"].is_string())
        use.dtype = value["dtype"].get<std::string>();
    use.index = value.value("index", 0u);
    use.location = value.value("vernon.location", UINT32_MAX);
    use.divisor = value.value("vernon.instance_divisor", 0u);
    use.descriptorSet = value.value("vernon.set", 0u);
    use.binding = value.value("vernon.binding", UINT32_MAX);
    if (value.contains("attribute_leaves")) {
        const nlohmann::json &leaves = value["attribute_leaves"];
        if (!leaves.is_array()) {
            error = "attribute_leaves must be an array";
            return false;
        }
        for (const nlohmann::json &leaf : leaves) {
            uint64_t locationOffset = 0;
            uint64_t componentCount = 0;
            uint64_t byteOffset = 0;
            if (!leaf.is_object() || !leaf.contains("location_offset") ||
                !parseUint64(leaf["location_offset"], locationOffset) || !leaf.contains("component_count") ||
                !parseUint64(leaf["component_count"], componentCount) || !leaf.contains("byte_offset") ||
                !parseUint64(leaf["byte_offset"], byteOffset) || locationOffset > UINT32_MAX ||
                componentCount > UINT32_MAX || byteOffset > UINT32_MAX) {
                error = "attribute leaf must contain unsigned location_offset, component_count, and byte_offset";
                return false;
            }
            use.attributeLeaves.push_back({static_cast<uint32_t>(locationOffset), static_cast<uint32_t>(componentCount),
                                           static_cast<uint32_t>(byteOffset)});
        }
    }
    if (value.contains("uniform_layout")) {
        const nlohmann::json &layout = value["uniform_layout"];
        uint64_t size = 0;
        uint64_t alignment = 0;
        if (!layout.is_object() || !layout.contains("storage") || !layout["storage"].is_string() ||
            !layout.contains("size") || !parseUint64(layout["size"], size) || !layout.contains("alignment") ||
            !parseUint64(layout["alignment"], alignment) || !layout.contains("byte_strides") ||
            !layout["byte_strides"].is_array()) {
            error = "uniform_layout must contain storage, size, alignment, and byte_strides";
            return false;
        }
        UniformLayout parsed;
        parsed.storage = layout["storage"].get<std::string>();
        parsed.matrixOrder = layout.value("matrix_order", "");
        parsed.size = size;
        parsed.alignment = alignment;
        for (const nlohmann::json &stride : layout["byte_strides"]) {
            uint64_t byteStride = 0;
            if (!parseUint64(stride, byteStride)) {
                error = "uniform_layout byte strides must be unsigned";
                return false;
            }
            parsed.byteStrides.push_back(byteStride);
        }
        if ((parsed.storage != "inline" && parsed.storage != "uniform_buffer") || parsed.size == 0 ||
            parsed.alignment == 0 ||
            (!parsed.matrixOrder.empty() && parsed.matrixOrder != "row_major" &&
             parsed.matrixOrder != "column_major")) {
            error = "uniform_layout contains unsupported physical layout metadata";
            return false;
        }
        use.uniformLayout = std::move(parsed);
    }
    if (value.contains("sampled_texture_bindings")) {
        const nlohmann::json &bindings = value["sampled_texture_bindings"];
        if (!bindings.is_array()) {
            error = "sampled_texture_bindings must be an array";
            return false;
        }
        for (const nlohmann::json &binding : bindings) {
            if (!binding.is_object() || !binding.contains("set") || !binding["set"].is_number_unsigned() ||
                !binding.contains("binding") || !binding["binding"].is_number_unsigned()) {
                error = "sampled texture binding must contain unsigned set/binding";
                return false;
            }
            use.sampledTextureBindings.push_back({binding["set"].get<uint32_t>(), binding["binding"].get<uint32_t>()});
        }
    }
    if (value.contains("shape") && value["shape"].is_array())
        for (const nlohmann::json &dimension : value["shape"])
            use.shape.push_back(dimension.is_number_unsigned() ? dimension.get<uint64_t>() : uint64_t{0});
    if (use.uniformLayout && use.uniformLayout->byteStrides.size() != use.shape.size()) {
        error = "uniform_layout byte-stride rank does not match the logical Tensor shape";
        return false;
    }
    if (use.stage.empty() || use.interfaceKind.empty()) {
        error = "pipeline parameter use is missing stage/interface metadata";
        return false;
    }
    return true;
}

void parseStaticType(const std::string &type, std::string &dtype, std::vector<uint64_t> &shape) {
    if (type.rfind("tensor<", 0) != 0 || type.size() < 9 || type.back() != '>') {
        dtype = type;
        return;
    }
    const std::string body = type.substr(7, type.size() - 8);
    size_t begin = 0;
    while (true) {
        const size_t separator = body.find('x', begin);
        if (separator == std::string::npos) {
            dtype = body.substr(begin);
            return;
        }
        const std::string dimension = body.substr(begin, separator - begin);
        if (dimension == "?")
            shape.push_back(0);
        else {
            try {
                shape.push_back(std::stoull(dimension));
            } catch (...) {
                dtype.clear();
                shape.clear();
                return;
            }
        }
        begin = separator + 1;
    }
}

bool parseUint32(const nlohmann::json &value, uint32_t &result) {
    if (!value.is_number_integer())
        return false;
    if (value.is_number_unsigned()) {
        const uint64_t parsed = value.get<uint64_t>();
        if (parsed > UINT32_MAX)
            return false;
        result = static_cast<uint32_t>(parsed);
    } else {
        const int64_t parsed = value.get<int64_t>();
        if (parsed < 0 || static_cast<uint64_t>(parsed) > UINT32_MAX)
            return false;
        result = static_cast<uint32_t>(parsed);
    }
    return true;
}

bool parseVersion(const nlohmann::json &value, RuntimeVersion &version) {
    if (!value.is_array() || value.size() != 2 || !parseUint32(value[0], version.major) ||
        !parseUint32(value[1], version.minor))
        return false;
    return version.major != 0;
}

bool hasOnlyKeys(const nlohmann::json &value, std::initializer_list<std::string_view> allowed) {
    for (auto row = value.begin(); row != value.end(); ++row)
        if (std::find(allowed.begin(), allowed.end(), row.key()) == allowed.end())
            return false;
    return true;
}

} // namespace

bool Variant::validate(std::string &error) const {
    const auto validTextureDimension = [](const std::string &dimension) {
        return dimension == "2d" || dimension == "3d" || dimension == "cube";
    };
    const auto validTextureFormat = [](const std::string &format) {
        return format.empty() || format == "rgba8_unorm" || format == "rgba8_srgb" || format == "rgba16_float" ||
               format == "rgba32_float" || format == "r8_unorm" || format == "r16_float" || format == "r32_float" ||
               format == "rg8_unorm" || format == "rgb8_unorm" || format == "r11g11b10_float";
    };
    if (!std::is_sorted(key.begin(), key.end()) || std::adjacent_find(key.begin(), key.end()) != key.end()) {
        error = "pipeline variant feature key is not canonical";
        return false;
    }
    if (!std::is_sorted(parameters.begin(), parameters.end(),
                        [](const Parameter &left, const Parameter &right) { return left.slot < right.slot; }) ||
        std::adjacent_find(parameters.begin(), parameters.end(), [](const Parameter &left, const Parameter &right) {
            return left.slot == right.slot;
        }) != parameters.end()) {
        error = "pipeline variant parameter slots are not unique and sorted";
        return false;
    }
    if (!std::is_sorted(internalParameters.begin(), internalParameters.end(),
                        [](const Parameter &left, const Parameter &right) { return left.name < right.name; }) ||
        std::adjacent_find(internalParameters.begin(), internalParameters.end(),
                           [](const Parameter &left, const Parameter &right) { return left.name == right.name; }) !=
            internalParameters.end()) {
        error = "pipeline variant internal parameters are not unique and sorted";
        return false;
    }
    for (const Parameter &parameter : parameters) {
        if (parameter.name.empty() || parameter.uses.empty() || !parameter.source.empty() ||
            !parameter.systemValue.empty()) {
            error = "external pipeline parameter invariant failed";
            return false;
        }
        const bool textureConstraintsValid =
            parameter.kind == "texture"
                ? validTextureDimension(parameter.dimension) && validTextureFormat(parameter.textureFormat)
                : parameter.dimension.empty() && parameter.textureFormat.empty();
        if (!textureConstraintsValid) {
            error = "pipeline parameter texture constraint invariant failed";
            return false;
        }
    }
    for (const Parameter &parameter : internalParameters) {
        const bool implicitSampler =
            parameter.source == "implicit_sampler" && parameter.kind == "sampler" && parameter.systemValue.empty();
        const bool resolution = parameter.source == "system_value" && parameter.systemValue == "resolution" &&
                                parameter.kind == "tensor" && parameter.dtype == "f32" &&
                                parameter.shape == std::vector<uint64_t>{2};
        if (parameter.name.empty() || parameter.uses.empty() || (!implicitSampler && !resolution)) {
            error = "internal pipeline parameter invariant failed";
            return false;
        }
        for (const ParameterUse &use : parameter.uses) {
            if (implicitSampler && use.sampledTextureBindings.size() != 1) {
                error = "implicit sampler must have exactly one sampled texture binding";
                return false;
            }
            if (resolution && !use.sampledTextureBindings.empty()) {
                error = "resolution cannot have sampled texture bindings";
                return false;
            }
        }
    }
    if (program.empty() || (!compute.empty() && program.size() != 1)) {
        error = "pipeline variant must contain either one compute program or one graphics program";
        return false;
    }
    return true;
}

bool parseRuntimeRequirements(const nlohmann::json &root, const std::string &target, RuntimeRequirements &requirements,
                              std::string &error) {
    if (!root.contains("runtime_requirements")) {
        error = "pipeline manifest requires runtime_requirements";
        return false;
    }
    const nlohmann::json &value = root["runtime_requirements"];
    if (!value.is_object() || !value.contains("backend") || !value["backend"].is_string() ||
        !value.contains("features") || !value["features"].is_array()) {
        error = "runtime_requirements must contain backend and features";
        return false;
    }
    requirements.backend = value["backend"].get<std::string>();
    if (requirements.backend != target) {
        error = "runtime_requirements backend does not match pipeline target";
        return false;
    }
    for (const nlohmann::json &feature : value["features"]) {
        if (!feature.is_string()) {
            error = "runtime requirement feature must be a string";
            return false;
        }
        requirements.features.push_back(feature.get<std::string>());
    }
    static constexpr std::string_view knownFeatures[] = {"compute", "instancing", "samplers", "tensor_views",
                                                         "textures"};
    if (!std::is_sorted(requirements.features.begin(), requirements.features.end()) ||
        std::adjacent_find(requirements.features.begin(), requirements.features.end()) != requirements.features.end() ||
        std::any_of(requirements.features.begin(), requirements.features.end(), [](const std::string &feature) {
            return std::find(std::begin(knownFeatures), std::end(knownFeatures), feature) == std::end(knownFeatures);
        })) {
        error = "runtime requirement features must be known, unique, and sorted";
        return false;
    }
    if (target == "cpu") {
        if (!hasOnlyKeys(value, {"backend", "features", "target_triple", "object_format", "invocation_abi_version"}) ||
            !value.contains("target_triple") || !value["target_triple"].is_string() ||
            value["target_triple"].get_ref<const std::string &>().empty() || !value.contains("object_format") ||
            !value["object_format"].is_string() || !value.contains("invocation_abi_version") ||
            !parseUint32(value["invocation_abi_version"], requirements.invocationAbiVersion)) {
            error = "CPU runtime requirements are invalid";
            return false;
        }
        requirements.targetTriple = value["target_triple"].get<std::string>();
        requirements.objectFormat = value["object_format"].get<std::string>();
        if ((requirements.objectFormat != "coff" && requirements.objectFormat != "elf" &&
             requirements.objectFormat != "macho" && requirements.objectFormat != "wasm") ||
            requirements.invocationAbiVersion == 0) {
            error = "CPU runtime requirements are invalid";
            return false;
        }
        return true;
    }
    if (target == "opengl" || target == "opengles") {
        if (!hasOnlyKeys(value, {"backend", "features", "glsl_version", "profile", "api_version"}) ||
            !value.contains("glsl_version") || !parseUint32(value["glsl_version"], requirements.glslVersion) ||
            !value.contains("profile") || !value["profile"].is_string() || !value.contains("api_version") ||
            !parseVersion(value["api_version"], requirements.apiVersion)) {
            error = "OpenGL runtime requirements are invalid";
            return false;
        }
        requirements.profile = value["profile"].get<std::string>();
        if ((target == "opengles" && requirements.profile != "es") ||
            (target == "opengl" && requirements.profile != "core" && requirements.profile != "compatibility")) {
            error = "OpenGL runtime requirement profile does not match target";
            return false;
        }
        if (requirements.glslVersion != glslVersionForApi(requirements.apiVersion)) {
            error = "OpenGL GLSL and API versions are inconsistent";
            return false;
        }
        return true;
    }
    if (target == "vulkan") {
        if (!hasOnlyKeys(value, {"backend", "features", "api_version", "spirv_version", "compute_workgroup_size"}) ||
            !value.contains("api_version") || !parseVersion(value["api_version"], requirements.apiVersion) ||
            !value.contains("spirv_version") || !parseVersion(value["spirv_version"], requirements.shaderVersion)) {
            error = "Vulkan runtime requirements are invalid";
            return false;
        }
        if (value.contains("compute_workgroup_size")) {
            const nlohmann::json &workgroup = value["compute_workgroup_size"];
            if (!workgroup.is_array() || workgroup.size() != 3) {
                error = "Vulkan compute workgroup requirement must have three dimensions";
                return false;
            }
            for (size_t index = 0; index < 3; ++index) {
                if (!parseUint32(workgroup[index], requirements.computeWorkgroupSize[index]) ||
                    requirements.computeWorkgroupSize[index] == 0) {
                    error = "Vulkan compute workgroup dimensions must be positive";
                    return false;
                }
            }
        }
        return true;
    }
    if (target == "directx") {
        if (!hasOnlyKeys(value, {"backend", "features", "api_version", "minimum_feature_level", "shader_model",
                                 "root_signature_version", "compute_workgroup_size"}) ||
            !value.contains("api_version") || !parseVersion(value["api_version"], requirements.apiVersion) ||
            !value.contains("minimum_feature_level") ||
            !parseVersion(value["minimum_feature_level"], requirements.minimumFeatureLevel) ||
            !value.contains("shader_model") || !parseVersion(value["shader_model"], requirements.shaderVersion) ||
            !value.contains("root_signature_version") ||
            !parseVersion(value["root_signature_version"], requirements.rootSignatureVersion)) {
            error = "DirectX runtime requirements are invalid";
            return false;
        }
        if (requirements.apiVersion.major != 12 || requirements.shaderVersion.major < 6 ||
            requirements.rootSignatureVersion.major != 1) {
            error = "DirectX runtime requires D3D12, Shader Model 6+, and root signature 1.x";
            return false;
        }
        if (value.contains("compute_workgroup_size")) {
            const nlohmann::json &workgroup = value["compute_workgroup_size"];
            if (!workgroup.is_array() || workgroup.size() != 3) {
                error = "DirectX compute workgroup requirement must have three dimensions";
                return false;
            }
            for (size_t index = 0; index < 3; ++index)
                if (!parseUint32(workgroup[index], requirements.computeWorkgroupSize[index]) ||
                    requirements.computeWorkgroupSize[index] == 0) {
                    error = "DirectX compute workgroup dimensions must be positive";
                    return false;
                }
        }
        return true;
    }
    if (target == "cuda") {
        if (!hasOnlyKeys(value, {"backend", "features", "ptx_version", "minimum_compute_capability", "address_size"}) ||
            !value.contains("ptx_version") || !parseVersion(value["ptx_version"], requirements.shaderVersion) ||
            !value.contains("minimum_compute_capability") ||
            !parseVersion(value["minimum_compute_capability"], requirements.minimumComputeCapability) ||
            !value.contains("address_size") || !parseUint32(value["address_size"], requirements.addressSize)) {
            error = "CUDA runtime requirements are invalid";
            return false;
        }
        if (requirements.addressSize != 32 && requirements.addressSize != 64) {
            error = "CUDA PTX address size must be 32 or 64";
            return false;
        }
        return true;
    }
    error = "runtime requirements are unsupported for pipeline target";
    return false;
}

bool runtimeVersionAtLeast(RuntimeVersion actual, RuntimeVersion required) {
    return actual.major > required.major || (actual.major == required.major && actual.minor >= required.minor);
}

uint32_t glslVersionForApi(RuntimeVersion apiVersion) { return apiVersion.major * 100 + apiVersion.minor * 10; }

bool parseVariant(const nlohmann::json &value, Variant &variant, std::string &error) {
    if (!value.is_object() || !value.contains("key") || !value.contains("program") || !value.contains("parameters") ||
        !value["key"].is_array() || !value["program"].is_object() || !value["parameters"].is_array()) {
        error = "pipeline variant tables are invalid";
        return false;
    }
    for (const nlohmann::json &feature : value["key"]) {
        if (!feature.is_string()) {
            error = "pipeline variant feature is not a string";
            return false;
        }
        variant.key.push_back(feature.get<std::string>());
    }
    if (!std::is_sorted(variant.key.begin(), variant.key.end()) ||
        std::adjacent_find(variant.key.begin(), variant.key.end()) != variant.key.end()) {
        error = "pipeline variant feature key is not canonical";
        return false;
    }
    auto parseParameter = [&](const nlohmann::json &row, bool internal, Parameter &parameter) {
        if (!row.is_object() || (!internal && !row.contains("slot")) || !row.contains("uses") ||
            !row["uses"].is_array()) {
            error = internal ? "internal pipeline parameter record is invalid" : "pipeline parameter record is invalid";
            return false;
        }
        if (!internal)
            parameter.slot = row["slot"].get<uint32_t>();
        parameter.name = row.value("name", "");
        parameter.kind = row.value("kind", "");
        parameter.source = row.value("source", "");
        parameter.systemValue = row.value("system_value", "");
        parameter.dtype = row.value("dtype", "");
        parameter.access = row.value("access", "read");
        parameter.dimension = row.value("dimension", "");
        parameter.textureFormat = row.value("texture_format", "");
        if (row.contains("shape") && row["shape"].is_array())
            for (const nlohmann::json &dimension : row["shape"])
                parameter.shape.push_back(dimension.is_number_unsigned() ? dimension.get<uint64_t>() : uint64_t{0});
        for (const nlohmann::json &useValue : row["uses"]) {
            ParameterUse use;
            if (!parseUse(useValue, use, error))
                return false;
            parameter.uses.push_back(std::move(use));
        }
        if (parameter.name.empty() || parameter.uses.empty()) {
            error =
                internal ? "internal pipeline parameter has no name or uses" : "pipeline parameter has no name or uses";
            return false;
        }
        if (parameter.kind == "texture") {
            if (!pipelineTextureDimension(parameter.dimension) ||
                (!parameter.textureFormat.empty() && !pipelineTextureFormat(parameter.textureFormat))) {
                error = "texture parameter has an invalid dimension or texture_format";
                return false;
            }
        } else if (!parameter.dimension.empty() || !parameter.textureFormat.empty()) {
            error = "non-texture parameter contains texture constraints";
            return false;
        }
        if (internal) {
            const bool implicitSampler =
                parameter.source == "implicit_sampler" && parameter.kind == "sampler" && parameter.systemValue.empty();
            const bool resolution = parameter.source == "system_value" && parameter.systemValue == "resolution" &&
                                    parameter.kind == "tensor" && parameter.dtype == "f32" &&
                                    parameter.shape == std::vector<uint64_t>{2};
            if (!implicitSampler && !resolution) {
                error = "internal pipeline parameter metadata is unsupported";
                return false;
            }
        } else if (!parameter.source.empty() || !parameter.systemValue.empty()) {
            error = "external pipeline parameter contains internal metadata";
            return false;
        }
        return true;
    };
    for (const nlohmann::json &row : value["parameters"]) {
        Parameter parameter;
        if (!parseParameter(row, false, parameter))
            return false;
        variant.parameters.push_back(std::move(parameter));
    }
    const nlohmann::json internalRows = value.value("internal_parameters", nlohmann::json::array());
    if (!internalRows.is_array()) {
        error = "internal_parameters must be an array";
        return false;
    }
    for (const nlohmann::json &row : internalRows) {
        Parameter parameter;
        if (!parseParameter(row, true, parameter))
            return false;
        variant.internalParameters.push_back(std::move(parameter));
    }
    std::sort(variant.parameters.begin(), variant.parameters.end(),
              [](const Parameter &left, const Parameter &right) { return left.slot < right.slot; });
    if (std::adjacent_find(variant.parameters.begin(), variant.parameters.end(),
                           [](const Parameter &left, const Parameter &right) { return left.slot == right.slot; }) !=
        variant.parameters.end()) {
        error = "pipeline variant contains duplicate parameter slots";
        return false;
    }
    std::sort(variant.internalParameters.begin(), variant.internalParameters.end(),
              [](const Parameter &left, const Parameter &right) { return left.name < right.name; });
    if (std::adjacent_find(variant.internalParameters.begin(), variant.internalParameters.end(),
                           [](const Parameter &left, const Parameter &right) { return left.name == right.name; }) !=
        variant.internalParameters.end()) {
        error = "pipeline variant contains duplicate internal parameters";
        return false;
    }
    for (const Parameter &parameter : variant.internalParameters) {
        if (parameter.source != "implicit_sampler")
            continue;
        for (const ParameterUse &use : parameter.uses)
            if (use.sampledTextureBindings.empty()) {
                error = "implicit sampler has no paired sampled texture binding";
                return false;
            }
    }
    for (const nlohmann::json &row : value.value("outputs", nlohmann::json::array())) {
        if (!row.is_object()) {
            error = "pipeline output record is invalid";
            return false;
        }
        Output output;
        output.name = row.value("name", "");
        output.kind = row.value("kind", "texture");
        output.dtype = row.value("dtype", "");
        output.access = row.value("access", "write");
        output.location = row.value("location", UINT32_MAX);
        if (output.name.empty() && output.location != UINT32_MAX)
            output.name = "output_" + std::to_string(output.location);
        if (row.contains("shape") && row["shape"].is_array())
            for (const nlohmann::json &dimension : row["shape"])
                output.shape.push_back(dimension.is_number_unsigned() ? dimension.get<uint64_t>() : uint64_t{0});
        if (output.dtype.empty())
            parseStaticType(row.value("type", ""), output.dtype, output.shape);
        if (output.name.empty() || output.dtype.empty() || output.location == UINT32_MAX) {
            error = "pipeline output metadata is incomplete";
            return false;
        }
        variant.outputs.push_back(std::move(output));
    }
    for (const auto &[stage, artifact] : value["program"].items()) {
        if (!artifact.is_string() || artifact.get_ref<const std::string &>().empty()) {
            error = "pipeline program stage artifact id is invalid";
            return false;
        }
        variant.program.emplace(stage, artifact.get<std::string>());
        if (stage == "compute")
            variant.compute = artifact.get<std::string>();
        else if (stage == "vertex")
            variant.vertex = artifact.get<std::string>();
        else if (stage == "fragment")
            variant.fragment = artifact.get<std::string>();
    }
    if (variant.vertex.empty() != variant.fragment.empty()) {
        error = "graphics pipeline requires both vertex and fragment stages";
        return false;
    }
    return variant.validate(error);
}

} // namespace vernon::runtime
