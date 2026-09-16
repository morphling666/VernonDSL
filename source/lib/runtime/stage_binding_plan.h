#ifndef VERNON_RUNTIME_STAGE_BINDING_PLAN_H
#define VERNON_RUNTIME_STAGE_BINDING_PLAN_H

#include "VernonRuntime.h"
#include "autodiff/autodiff_metadata.h"
#include "transport_node.h"

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

enum class MetadataFieldKind : uint8_t {
    Offset,
    Extent,
    Stride,
};

struct MetadataFieldIdentity {
    uint32_t argument{UINT32_MAX};
    MetadataFieldKind kind{MetadataFieldKind::Offset};
    std::optional<uint32_t> dimension;
};

struct PhysicalMetadataMember {
    uint32_t semanticOrdinal{UINT32_MAX};
    uint64_t byteOffset{};
    uint64_t byteSize{};
    uint64_t alignment{};
};

// One immutable, entry-owned TensorView metadata ABI. `fields` is the
// canonical semantic order; `members` is a bijective physical projection.
struct MetadataCarrier {
    std::string profile;
    std::string representation;
    std::string carrier;
    uint64_t encodedSize{};
    uint64_t size{};
    uint64_t alignment{};
    uint32_t descriptorSet{UINT32_MAX};
    uint32_t binding{UINT32_MAX};
    uint32_t parameterOrdinal{UINT32_MAX};
    std::vector<MetadataFieldIdentity> fields;
    std::vector<PhysicalMetadataMember> members;
    InterfacePlan interfacePlan;
};

enum class TensorRepresentation {
    ElementStream,
    WholeValue,
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
    TensorRepresentation tensorPacking{TensorRepresentation::ElementStream};
    std::optional<ValueLayout> valueLayout;
    std::optional<InterfacePlan> interfacePlan;
};

enum class StageParameterSource {
    Direct,
    ImplicitSampler,
    Resolution,
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
    StageParameterSource source;
    // Representation of the invocation-owned Tensor descriptor.
    TensorRepresentation tensorArgument{TensorRepresentation::ElementStream};
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
    bool invocationCarrier{};
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

struct StageBindingPlan {
    std::vector<Parameter> parameters;
    std::vector<Parameter> runtimeParameters;
    std::vector<Output> outputs;
    std::optional<MetadataCarrier> metadataCarrier;
    std::map<std::string, std::string> artifactKeys;
    std::string compute;
    std::string vertex;
    std::string fragment;
};

// Immutable physical ABI projection for one reusable Stage implementation.
// Program-to-endpoint projection is performed once while resolving a Node.

std::optional<VernonTextureDimension> artifactTextureDimension(const std::string &dimension);
std::optional<VernonTextureFormat> artifactTextureFormat(const std::string &format);

bool parseArtifactValueLayout(const nlohmann::json &value, ValueLayout &layout, std::string &error);
bool parseArtifactInterfacePlan(const nlohmann::json &value, InterfacePlan &plan, std::string &error);
bool validateStageBindingPlan(const StageBindingPlan &plan, std::string &error);
void rebuildValueLayoutPathViews(ValueLayout &layout);
void rebuildStageBindingLayoutViews(StageBindingPlan &plan);
bool runtimeVersionAtLeast(RuntimeVersion actual, RuntimeVersion required);
uint32_t glslVersionForApi(RuntimeVersion apiVersion);

} // namespace vernon::runtime

#endif
