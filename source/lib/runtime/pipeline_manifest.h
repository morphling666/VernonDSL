#ifndef VERNON_RUNTIME_PIPELINE_MANIFEST_H
#define VERNON_RUNTIME_PIPELINE_MANIFEST_H

#include "VernonRuntime.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <map>
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

struct RuntimeVersion {
    uint32_t major{};
    uint32_t minor{};
};

struct RuntimeRequirements {
    bool present{};
    std::string backend;
    std::vector<std::string> features;
    RuntimeVersion apiVersion;
    RuntimeVersion shaderVersion;
    RuntimeVersion minimumFeatureLevel;
    RuntimeVersion rootSignatureVersion;
    RuntimeVersion minimumComputeCapability;
    uint32_t computeWorkgroupSize[3]{1, 1, 1};
    uint32_t glslVersion{};
    uint32_t addressSize{};
    uint32_t invocationAbiVersion{};
    std::string targetTriple;
    std::string objectFormat;
    std::string profile;
};

struct Variant {
    std::vector<std::string> key;
    std::vector<Parameter> parameters;
    std::vector<Parameter> internalParameters;
    std::vector<Output> outputs;
    std::map<std::string, std::string> program;
    std::string compute;
    std::string vertex;
    std::string fragment;

    bool validate(std::string &error) const;
};

std::optional<VernonTextureDimension> pipelineTextureDimension(const std::string &dimension);

std::optional<VernonTextureFormat> pipelineTextureFormat(const std::string &format);

bool parseVariant(const nlohmann::json &value, Variant &variant, std::string &error);
bool parseRuntimeRequirements(const nlohmann::json &root, const std::string &target, RuntimeRequirements &requirements,
                              std::string &error);
bool runtimeVersionAtLeast(RuntimeVersion actual, RuntimeVersion required);
uint32_t glslVersionForApi(RuntimeVersion apiVersion);

} // namespace vernon::runtime

#endif
