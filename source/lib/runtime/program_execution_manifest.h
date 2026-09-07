#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_MANIFEST_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_MANIFEST_H

#include "VernonProgramPlanTypes.h"
#include "stage_artifact.h"
#include "stage_binding_plan.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace vernon::runtime::program {

struct Diagnostic {
    std::string code;
    std::string phase;
    std::string path;
    std::string message;

    explicit operator bool() const { return !code.empty(); }
};

struct StageContract {
    std::string operation;
    std::string contractHash;
};

struct CodeModule {
    std::string role;
    std::string format;
    std::string entryPoint;
    std::string blob;
    uint64_t offset{};
    uint64_t byteLength{};
    std::string sha256;
};

struct EndpointAbiBinding {
    std::string semantic;
    std::optional<uint32_t> axis;
    std::string carrier;
    uint32_t slot{};
    uint64_t byteOffset{};
    uint64_t byteSize{};
    uint64_t alignment{};
};

struct ReflectedEndpoint {
    std::string tag;
    std::string module;
    std::string interfaceKind;
    uint32_t index{};
    std::string role;
    std::string type;
    std::string layoutHash;
    std::string elementLayoutHash;
    std::string transport;
    std::string access;
    std::string builtin;
    std::string autodiffCarrier;
    uint32_t viewRank{};
    std::vector<int64_t> viewShape;
    bool viewDescriptor{};
    std::string imageDimension;
    std::string imageFormat;
    std::vector<EndpointAbiBinding> abiBindings;
    std::string writeFootprintKind;
    std::vector<uint32_t> writeFootprintIndices;
};

struct GraphicsVertexInput {
    uint32_t location{};
    uint32_t endpointIndex{};
    std::string format;
    uint64_t byteOffset{};
    uint64_t byteStride{};
    std::string step;
    uint32_t divisor{};
};

struct GraphicsFragmentOutput {
    uint32_t location{};
    std::string type;
};

struct CompiledEndpointAbi {
    std::string module;
    std::string interfaceKind;
    uint32_t index{};
    std::string builtin;
    std::string valueTransport;
    uint32_t descriptorSet{UINT32_MAX};
    uint32_t binding{UINT32_MAX};
    std::optional<vernon::runtime::InterfacePlan> interfacePlan;
    std::optional<uint64_t> packedFrameOffset;
    std::optional<vernon::runtime::ValueLayout> elementLayout;
    std::vector<vernon::runtime::SampledImageBinding> sampledImageBindings;
};

struct StageArtifact {
    std::string operation;
    std::string contractHash;
    std::string backend;
    std::vector<std::string> requiredFeatures;
    std::vector<CodeModule> modules;
    std::vector<ReflectedEndpoint> endpoints;
    std::vector<CompiledEndpointAbi> compiledAbi;
    std::vector<vernon::runtime::NativeResourceSlot> nativeSlots;
    uint32_t workgroupSize[3]{1, 1, 1};
    std::vector<std::string> capabilities;
    std::string graphicsTopology;
    std::vector<GraphicsVertexInput> vertexInputs;
    std::vector<GraphicsFragmentOutput> fragmentOutputs;
    bool requiresUnitWorkgroup{};
    std::vector<uint32_t> unitGridAxes;
};

struct Blob {
    uint64_t byteLength{};
    std::string sha256;
    std::string uri;
};

struct ArtifactSystem {
    std::string target;
    std::map<std::string, Blob> blobs;
    std::map<std::string, StageArtifact> stages;
};

struct Parameter {
    uint32_t id{};
    std::string path;
    uint32_t value{};
};

enum class StorageOwnership {
    Owned,
    Borrowed,
};

enum class StorageLifetime {
    Invocation,
    Instance,
    Pullback,
};

enum class StorageMutability {
    ReadOnly,
    Mutable,
};

enum class ControlKind {
    Static,
    Value,
    Parameter,
    Capture,
};

struct ControlComponent {
    ControlKind kind{ControlKind::Static};
    uint64_t value{};
    uint32_t reference{};
    bool hasAxis{};
    uint32_t axis{};
};

struct BufferDescriptor {
    uint64_t byteLength{};
    uint64_t alignment{};
    std::string memory;
    std::vector<std::string> usage;
    std::vector<ControlComponent> byteLengthExtents;
};

enum class StorageDescriptorKind {
    Buffer,
    Image,
    Opaque,
};

struct ImageDescriptor {
    std::string dimension;
    std::vector<uint64_t> extent;
    std::vector<ControlComponent> extentControls;
    std::string format;
    uint32_t sampleCount{};
    uint32_t mipLevels{};
    uint32_t arrayLayers{};
    std::vector<std::string> aspects;
    std::vector<std::string> usage;
};

struct Storage {
    uint32_t id{};
    std::string name;
    uint32_t initialValue{};
    StorageOwnership ownership{StorageOwnership::Borrowed};
    StorageLifetime lifetime{StorageLifetime::Invocation};
    StorageMutability mutability{StorageMutability::ReadOnly};
    StorageDescriptorKind descriptorKind{StorageDescriptorKind::Buffer};
    BufferDescriptor buffer;
    ImageDescriptor image;
    std::string opaqueContractHash;
};

enum class OriginKind {
    Argument,
    Parameter,
    Allocation,
    NodeResult,
};

struct Origin {
    OriginKind kind{OriginKind::Argument};
    std::string graph;
    uint32_t slot{};
    uint32_t parameter{};
    uint32_t node{};
};

struct LayoutPathComponent {
    std::string field;
    std::optional<uint32_t> index;
};

struct LayoutLeaf {
    std::vector<LayoutPathComponent> path;
    std::string dtype;
    uint64_t byteOffset{};
    uint64_t scalarCount{};
    std::vector<uint64_t> shape;
};

struct ValueLayout {
    std::string scope;
    std::string layoutHash;
    uint64_t byteSize{};
    uint64_t alignment{};
    std::vector<LayoutLeaf> leaves;
};

struct CanonicalValueType {
    std::string dtype;
    std::vector<uint64_t> innerShape;
    bool rankedValue{};
};

struct Value {
    uint32_t id{};
    std::string name;
    std::string type;
    CanonicalValueType canonicalType;
    std::vector<uint64_t> shape;
    Origin origin;
    std::optional<uint32_t> storage;
    std::optional<ValueLayout> layout;
};

enum class GraphInputKind {
    UserInput,
    InvocationControl,
    Control,
    Parameter,
    Allocation,
};

struct GraphInput {
    GraphInputKind kind{GraphInputKind::UserInput};
    uint32_t value{};
    uint32_t slot{};
    uint32_t axis{};
    uint32_t parameter{};
    uint32_t storage{};
};

struct GraphOutput {
    uint32_t value{};
    std::string disposition;
};

enum class BindingTag {
    Value,
    Resource,
};

enum class ValueBindingDirection {
    Input,
    Result,
};

struct ValueEndpointProjection {
    uint32_t value{};
    std::optional<uint32_t> leaf;
    uint32_t physicalLeaf{};
    ValueBindingDirection direction{ValueBindingDirection::Input};
};

struct EndpointBinding {
    std::string module;
    std::string interfaceKind;
    uint32_t index{};
    BindingTag tag{BindingTag::Value};
    std::vector<ValueEndpointProjection> projections;
    uint32_t access{};
    std::optional<uint32_t> resourceLeaf;
};

enum class AccessKind {
    Read,
    Initialize,
    Write,
    Attachment,
};

struct ResourceAccess {
    AccessKind kind{AccessKind::Read};
    uint32_t storage{};
    uint32_t value{};
    uint32_t before{};
    uint32_t after{};
    std::optional<uint32_t> view;
    std::string access;
};

struct ComputeOperation {
    ControlComponent workgroups[3];
};

enum class ExecutionKind {
    Compute,
    Graphics,
};

struct GraphicsAttachmentSignature {
    uint32_t location{};
    uint32_t access{};
    std::vector<VernonTextureFormat> formats;
    std::vector<uint32_t> sampleCounts;
    uint32_t aspects{};
};

struct GraphicsPipelineState {
    VernonPrimitiveTopology topology{VERNON_TOPOLOGY_TRIANGLE_LIST};
    VernonRasterizationState rasterization{};
    VernonDepthStencilState depthStencil{};
    std::map<uint32_t, VernonColorBlendState> colorBlends;
};

struct GraphicsOperation {
    GraphicsPipelineState pipelineState;
    std::vector<GraphicsAttachmentSignature> colorAttachments;
    std::optional<GraphicsAttachmentSignature> depthStencilAttachment;
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    uint32_t renderPassControl{};
    uint32_t drawCommandControl{};
    uint32_t dynamicStateControl{};
};

using NodeOperation = std::variant<ComputeOperation, GraphicsOperation>;

struct Node {
    uint32_t id{};
    std::string name;
    std::string stage;
    std::vector<uint32_t> operands;
    std::vector<uint32_t> results;
    std::vector<EndpointBinding> bindings;
    std::vector<ResourceAccess> accesses;
    NodeOperation operation;
};

inline ExecutionKind executionKind(const Node &node) {
    return std::holds_alternative<ComputeOperation>(node.operation) ? ExecutionKind::Compute : ExecutionKind::Graphics;
}

inline const ComputeOperation &computeOperation(const Node &node) { return std::get<ComputeOperation>(node.operation); }

inline const GraphicsOperation &graphicsOperation(const Node &node) {
    return std::get<GraphicsOperation>(node.operation);
}

struct Graph {
    std::string name;
    std::string direction;
    std::vector<GraphInput> inputs;
    std::vector<uint32_t> captures;
    std::vector<GraphOutput> outputs;
    std::vector<Node> nodes;
};

enum class BoundaryRole {
    Input,
    Output,
    Cotangent,
    Gradient,
};

enum class BoundaryDirection {
    Input,
    Output,
};

enum class BoundaryCategory {
    Value,
    StorageView,
    Texture,
    Sampler,
};

enum class BoundaryAccess {
    Read,
    Write,
    ReadWrite,
};

struct BoundaryStorage {
    uint32_t id{};
    StorageDescriptorKind descriptorKind{StorageDescriptorKind::Buffer};
    BufferDescriptor buffer;
    ImageDescriptor image;
    std::string opaqueContractHash;
};

enum class ProgramOwnerKind {
    Value,
    Storage,
};

struct ProgramOwnerId {
    ProgramOwnerKind kind{ProgramOwnerKind::Value};
    uint32_t id{};
};

enum class BoundaryPublication {
    None,
    CommitAfterSuccess,
    InPlace,
};

struct BoundarySlot {
    uint32_t id{};
    std::string path;
    uint32_t value{};
    BoundaryRole role{BoundaryRole::Input};
    BoundaryDirection direction{BoundaryDirection::Input};
    BoundaryCategory category{BoundaryCategory::Value};
    BoundaryAccess access{BoundaryAccess::Read};
    std::string logicalType;
    std::vector<uint64_t> outerShape;
    ProgramOwnerId aliasOwner;
    BoundaryPublication publication{BoundaryPublication::None};
    std::optional<ValueLayout> layout;
    std::optional<BoundaryStorage> storage;
};

struct PublicationTarget {
    uint32_t slot{};
    uint32_t value{};
    BoundaryRole role{BoundaryRole::Output};
    ProgramOwnerId aliasOwner;
};

struct PublicationPlan {
    std::vector<PublicationTarget> targets;
};

struct BoundaryReference {
    uint32_t slot{};
    std::string path;
};

struct DerivativeProjection {
    BoundaryReference derivative;
    BoundaryReference primal;
    std::vector<LayoutPathComponent> valuePath;
};

struct TapePlan {
    uint32_t value{};
    bool forwardProducer{};
    bool backwardConsumer{};
    std::vector<vernon::program_plan::TapeCarrier> requiredCarriers;
    std::vector<vernon::program_plan::TapeCarrier> optionalCarriers;
};

struct ProgramAbi {
    std::vector<BoundarySlot> boundarySlots;
    std::vector<DerivativeProjection> derivativeProjections;
    std::vector<TapePlan> tapePlans;
    PublicationPlan publication;
};

PublicationPlan derivePublicationPlan(const std::vector<BoundarySlot> &slots);
const PublicationTarget *findPublicationTarget(const ProgramAbi &abi, uint32_t slot);

struct ResidualCapture {
    uint32_t value{};
};

struct ResidualContract {
    std::vector<ResidualCapture> captures;
};

struct Program {
    std::map<std::string, StageContract> stages;
    std::vector<Parameter> parameters;
    std::vector<Storage> storages;
    std::vector<Value> values;
    std::vector<Graph> graphs;
    ProgramAbi abi;
    std::optional<ResidualContract> residualContract;
};

struct ResolvedGraph {
    std::vector<std::vector<uint32_t>> predecessors;
};

struct ResolvedStage {
    std::string artifact;
    StageArtifact stage;
};

struct ResolvedProgram {
    Program program;
    std::map<std::string, ResolvedStage> stages;
    std::vector<ResolvedGraph> graphs;
};

bool parse(const nlohmann::json &value, Program &program, Diagnostic &diagnostic);
bool parseArtifactSystem(const nlohmann::json &target, const nlohmann::json &blobs, const nlohmann::json &value,
                         ArtifactSystem &artifacts, Diagnostic &diagnostic);
bool resolve(Program program, const ArtifactSystem &artifacts, ResolvedProgram &resolved, Diagnostic &diagnostic);
const Graph *findGraph(const Program &program, std::string_view direction);
std::vector<uint32_t> residualCaptures(const Program &program);
void markGraphValues(const Graph &graph, std::vector<char> &live);
bool isTapeValueType(std::string_view type);
bool resolveControlValue(const Program &program, const ControlComponent &control, std::string_view graph,
                         uint32_t &valueId);

} // namespace vernon::runtime::program

#endif
