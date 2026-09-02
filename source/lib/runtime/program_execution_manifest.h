#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_MANIFEST_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_MANIFEST_H

#include "pipeline_bundle.h"
#include "pipeline_manifest.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
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
    Argument,
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

struct Value {
    uint32_t id{};
    std::string name;
    std::string type;
    std::vector<uint64_t> shape;
    Origin origin;
    std::optional<uint32_t> storage;
    std::optional<ValueLayout> layout;
};

enum class GraphInputKind {
    UserInput,
    Parameter,
    Allocation,
};

struct GraphInput {
    GraphInputKind kind{GraphInputKind::UserInput};
    uint32_t value{};
    uint32_t slot{};
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

struct EndpointBinding {
    std::string module;
    std::string interfaceKind;
    uint32_t index{};
    BindingTag tag{BindingTag::Value};
    uint32_t value{};
    uint32_t access{};
    std::optional<uint32_t> leaf;
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

struct GraphicsOperation {
    std::vector<uint32_t> attachmentAccesses;
    uint64_t vertexCount{};
    uint64_t instanceCount{};
};

struct Node {
    uint32_t id{};
    std::string name;
    std::string stage;
    std::vector<uint32_t> operands;
    std::vector<uint32_t> results;
    std::vector<EndpointBinding> bindings;
    std::vector<ResourceAccess> accesses;
    std::string operation;
    ComputeOperation compute;
    GraphicsOperation graphics;
};

struct Graph {
    std::string name;
    std::string direction;
    std::vector<GraphInput> inputs;
    std::vector<uint32_t> captures;
    std::vector<GraphOutput> outputs;
    std::vector<Node> nodes;
};

struct SignatureBinding {
    std::string path;
    uint32_t value{};
    std::optional<uint32_t> primal;
    std::optional<std::string> disposition;
};

struct Signature {
    std::vector<SignatureBinding> inputs;
    std::vector<SignatureBinding> outputs;
    std::vector<SignatureBinding> cotangents;
    std::vector<SignatureBinding> gradients;
};

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
    Signature signature;
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

struct InvocationBuffer {
    void *data{};
    size_t byteLength{};
};

struct Invocation {
    std::vector<InvocationBuffer> arguments;
    std::vector<uint64_t> parameters;
};

struct StageResource {
    void *data{};
    size_t byteLength{};
    std::string access;
    std::vector<uint64_t> shape;
};

struct StageInvocation {
    const ResolvedStage *stage{};
    const Node *node{};
    uint32_t workgroups[3]{1, 1, 1};
    std::vector<StageResource> resources;
};

using StageExecutor = std::function<bool(const StageInvocation &, Diagnostic &)>;

struct ExecutionResult {
    std::vector<std::vector<uint8_t>> outputs;
};

bool parse(const nlohmann::json &value, Program &program, Diagnostic &diagnostic);
bool parseArtifactSystem(const nlohmann::json &value, ArtifactSystem &artifacts, Diagnostic &diagnostic);
bool resolve(Program program, const ArtifactSystem &artifacts, const std::map<std::string, std::string> &stageBindings,
             ResolvedProgram &resolved, Diagnostic &diagnostic);
bool resolveControlValue(const Program &program, const ControlComponent &control, std::string_view graph,
                         uint32_t &valueId);
bool execute(const ResolvedProgram &program, const Invocation &invocation, const StageExecutor &executor,
             ExecutionResult &result, Diagnostic &diagnostic);

} // namespace vernon::runtime::program

#endif
