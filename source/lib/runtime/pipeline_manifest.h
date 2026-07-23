#ifndef VERNON_RUNTIME_PIPELINE_MANIFEST_H
#define VERNON_RUNTIME_PIPELINE_MANIFEST_H

#include "VernonRuntime.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime {

struct SampledTextureBinding {
    uint32_t descriptorSet{};
    uint32_t binding{UINT32_MAX};
};

struct ParameterUse {
    std::string stage;
    std::string interfaceKind;
    std::string uniformName;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint32_t index{};
    uint32_t location{UINT32_MAX};
    uint32_t divisor{};
    uint32_t descriptorSet{};
    uint32_t binding{UINT32_MAX};
    std::vector<SampledTextureBinding> sampledTextureBindings;
};

struct Parameter {
    uint32_t slot{};
    std::string name;
    std::string kind;
    std::string source;
    std::string systemValue;
    std::string dtype;
    std::string access;
    std::string dimension;
    std::string textureFormat;
    std::vector<uint64_t> shape;
    std::vector<ParameterUse> uses;
};

struct Output {
    std::string name;
    std::string kind;
    std::string dtype;
    std::string access;
    std::vector<uint64_t> shape;
    uint32_t location{UINT32_MAX};
};

struct Variant {
    std::vector<std::string> key;
    std::vector<Parameter> parameters;
    std::vector<Parameter> internalParameters;
    std::vector<Output> outputs;
    std::string compute;
    std::string vertex;
    std::string fragment;
    bool barrier{};

    bool validate(std::string &error) const;
};

std::optional<VernonTextureDimension> pipelineTextureDimension(const std::string &dimension);

std::optional<VernonTextureFormat> pipelineTextureFormat(const std::string &format);

bool parseVariant(const nlohmann::json &value, Variant &variant, std::string &error);

} // namespace vernon::runtime

#endif
