#include "pipeline_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>

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
    if (vertex.empty() != fragment.empty() || (compute.empty() && vertex.empty())) {
        error = "pipeline variant has no valid execution stages";
        return false;
    }
    return true;
}

bool parseVariant(const nlohmann::json &value, Variant &variant, std::string &error) {
    if (!value.is_object() || !value.contains("key") || !value.contains("parameters") || !value.contains("steps") ||
        !value["key"].is_array() || !value["parameters"].is_array() || !value["steps"].is_array()) {
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
    for (const nlohmann::json &step : value["steps"]) {
        const std::string kind = step.value("kind", "");
        if (kind == "dispatch")
            variant.compute = step.value("stage", "");
        else if (kind == "barrier")
            variant.barrier = true;
        else if (kind == "draw") {
            variant.vertex = step.value("vertex", "");
            variant.fragment = step.value("fragment", "");
        } else {
            error = "pipeline step kind is unsupported";
            return false;
        }
    }
    if (variant.vertex.empty() != variant.fragment.empty()) {
        error = "graphics pipeline requires both vertex and fragment stages";
        return false;
    }
    return variant.validate(error);
}

} // namespace vernon::runtime
