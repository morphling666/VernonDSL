#ifndef VERNON_RUNTIME_PIPELINE_MANIFEST_H
#define VERNON_RUNTIME_PIPELINE_MANIFEST_H

#include "VernonRuntime.h"
#include "autodiff/autodiff_metadata.h"
#include "program_manifest.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime {

struct SampledImageBinding {
    uint32_t descriptorSet{};
    uint32_t binding{UINT32_MAX};
};

enum class InterfacePlanKind {
    CpuCall,
    KernelParameter,
    ByteTransport,
    NativeUniform,
};

enum class TransportNodeKind {
    Scalar,
    Product,
    Array,
};

struct TransportNode {
    TransportNodeKind kind{TransportNodeKind::Scalar};
    std::string representation;
    uint64_t offset{};
    uint64_t size{};
    uint64_t alignment{1};
    std::vector<uint64_t> shape;
    std::vector<uint64_t> byteStrides;
    std::vector<TransportNode> children;
};

struct InterfacePlan {
    InterfacePlanKind kind{InterfacePlanKind::ByteTransport};
    std::string profile;
    std::string canonicalLayoutHash;
    uint64_t frameOffset{};
    std::optional<TransportNode> root;
};

struct AttributeLeaf {
    uint32_t locationOffset{};
    std::string dtype;
    uint32_t componentCount{};
    uint32_t byteOffset{};
};

struct ValuePathComponent {
    std::optional<std::string> field;
    uint64_t index{};
};

struct ValueLeaf {
    ValueLeaf() = default;
    ValueLeaf(std::string dtype, uint32_t scalarCount, uint32_t byteOffset)
        : dtype(std::move(dtype)), scalarCount(scalarCount), byteOffset(byteOffset) {}

    std::vector<ValuePathComponent> path;
    std::vector<uint64_t> shape;
    std::string dtype;
    uint32_t scalarCount{};
    uint32_t byteOffset{};
    std::vector<VernonValuePathComponentView> abiPath;
};

struct ValueLayout {
    std::string logicalType;
    std::string structName;
    std::string layoutHash;
    uint32_t byteSize{};
    uint32_t alignment{};
    std::vector<ValueLeaf> leaves;
    std::vector<VernonValueLeafView> abiLeaves;
};

struct TensorViewDescriptorUse {
    uint32_t rank{};
    uint32_t offsetBinding{UINT32_MAX};
    std::vector<uint32_t> extentBindings;
    std::vector<uint32_t> strideBindings;
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
    std::vector<SampledImageBinding> sampledImageBindings;
    std::vector<AttributeLeaf> attributeLeaves;
    std::string transport;
    std::optional<ValueLayout> valueLayout;
    std::optional<InterfacePlan> interfacePlan;
    std::optional<TensorViewDescriptorUse> tensorViewDescriptor;
};

enum class AutodiffResourceRole {
    None,
    Input,
    Storage,
    Output,
    Tape,
    Cotangent,
    Gradient,
    Primal,
    RetainedPrimal,
    ReplaySegment,
    ReplayStatus,
    LaunchMetadata,
};

struct Parameter {
    uint32_t slot{};
    std::string name;
    std::string kind;
    std::string source;
    std::string systemValue;
    std::optional<ValueLayout> valueLayout;
    ValueLayout elementLayout;
    std::string access;
    std::string addressSpace;
    std::string dimension;
    std::string bindingRole;
    std::string sampleResultClass;
    std::string exactStorageFormat;
    AutodiffResourceRole autodiffRole{AutodiffResourceRole::None};
    std::string autodiffSource;
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
    std::string targetTriple;
    std::string objectFormat;
    std::string profile;
    std::string applePlatform;
    RuntimeVersion minimumOsVersion;
};

struct Variant {
    std::vector<std::string> key;
    std::vector<Parameter> parameters;
    std::vector<Parameter> internalParameters;
    std::vector<Output> outputs;
    std::map<std::string, std::string> program;
    std::optional<ExecutableProgram> executable;
    std::string compute;
    std::string vertex;
    std::string fragment;

    bool validate(std::string &error) const;
};

struct AutodiffLaunchPlan {
    VernonLaunchSize workgroupSize{1, 1, 1};
};

struct AutodiffProfile {
    std::vector<std::string> key;
    std::string primal;
    std::string forwardWithTape;
    std::string backward;
    uint64_t staticTapeBytesHint{};
    std::string residualStorage;
    std::vector<std::string> requiredPrimalPaths;
    std::map<std::string, uint64_t> sourceKindCounts;
    std::map<std::string, uint64_t> costComponents;
    std::string selectedPolicy;
    bool wholeDispatchRetentionPermitted{};
    AutodiffLaunchPlan launch;
};

struct AutodiffManifest {
    std::vector<AutodiffDerivativeGroup> derivativeGroups;
    std::vector<AutodiffProfile> profiles;
};

std::optional<VernonTextureDimension> pipelineTextureDimension(const std::string &dimension);
std::optional<VernonTextureFormat> pipelineTextureFormat(const std::string &format);

bool parseVariant(const nlohmann::json &value, Variant &variant, std::string &error);
bool parseAutodiffManifest(const nlohmann::json &root, AutodiffManifest &manifest, std::string &error);
bool normalizeLegacySingleComputeExecution(const Variant &variant, ExecutableProgram &execution, std::string &error);
bool validatePipelineRootSchema(const nlohmann::json &root, std::string &error);
bool parsePipelineValueLayout(const nlohmann::json &value, ValueLayout &layout, std::string &error);
bool parsePipelineInterfacePlan(const nlohmann::json &value, InterfacePlan &plan, std::string &error);
void rebuildValueLayoutPathViews(ValueLayout &layout);
bool parseRuntimeRequirements(const nlohmann::json &root, const std::string &target, RuntimeRequirements &requirements,
                              std::string &error);
bool runtimeVersionAtLeast(RuntimeVersion actual, RuntimeVersion required);
uint32_t glslVersionForApi(RuntimeVersion apiVersion);

} // namespace vernon::runtime

#endif
