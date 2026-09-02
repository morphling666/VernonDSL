#ifndef VERNON_RUNTIME_TARGET_BINDING_PLAN_H
#define VERNON_RUNTIME_TARGET_BINDING_PLAN_H

#include "VernonRuntime.h"
#include "pipeline_bundle.h"
#include "pipeline_manifest.h"
#include "pipeline_metadata.h"
#include "program_execution_manifest.h"
#include "shape_layout.h"

#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime::program {

enum class SourceRepresentation {
    WholeValueBytes,
    ElementStream,
    TensorViewDescriptor,
    ResourceHandle,
    SystemValue,
};

enum class TargetCarrier {
    InlineValue,
    UniformBuffer,
    StorageBuffer,
    VertexBuffer,
    IndexBuffer,
    Image,
    Sampler,
    Attachment,
};

enum class CarrierSemantic {
    Value,
    Resource,
    TapeData,
    ReplaySegment,
    ReplayStatus,
    LaunchMetadata,
};

enum class ViewAxisSource {
    LogicalAxis,
    Constant,
    InvocationLinearCarrier,
};

enum class DispatchMapping {
    StaticGrid,
    FirstTensorElementCount,
};

struct ViewAxisTransform {
    ViewAxisSource source{ViewAxisSource::LogicalAxis};
    uint32_t logicalAxis{};
    uint64_t constantExtent{1};
    bool zeroStride{};
};

struct ViewTransform {
    std::vector<ViewAxisTransform> axes;
};

struct ProgramProjection {
    uint32_t value{UINT32_MAX};
    std::optional<size_t> leaf;
};

struct TargetEndpointIdentity {
    std::string module;
    std::string interfaceKind;
    uint32_t index{};
    uint32_t portableSlot{UINT32_MAX};
    std::string access;
};

struct TargetNativeLocation {
    uint32_t descriptorSet{UINT32_MAX};
    uint32_t binding{UINT32_MAX};
    uint32_t location{UINT32_MAX};
};

struct TargetPhysicalTransport {
    vernon::runtime::ValueLayout canonicalProjection;
    vernon::runtime::InterfacePlan targetAbi;
    TargetNativeLocation native;
};

struct TargetBinding {
    TargetEndpointIdentity endpoint;
    ProgramProjection projection;
    SourceRepresentation source{SourceRepresentation::ResourceHandle};
    TargetCarrier carrier{TargetCarrier::StorageBuffer};
    CarrierSemantic semantic{CarrierSemantic::Resource};
    std::string name;
    std::string sourceName;
    std::string kind;
    std::string reflectedKind;
    std::string role;
    std::string access;
    std::string dimension;
    std::string imageFormat;
    std::string builtin;
    std::optional<ViewTransform> viewTransform;
    shape::DeclaredShape shape;
    std::vector<int64_t> viewShape;
    std::optional<vernon::runtime::ValueLayout> wholeValueLayout;
    vernon::runtime::ValueLayout elementLayout;
    std::optional<TargetPhysicalTransport> transport;
    TargetNativeLocation native;
    std::vector<vernon::runtime::AttributeLeaf> attributeLeaves;
    std::vector<vernon::runtime::SampledImageBinding> sampledImageBindings;
    std::vector<vernon::runtime::ReflectedStorageLeaf> storageLeaves;
    std::optional<vernon::runtime::TensorViewDescriptorUse> tensorViewDescriptor;
    vernon::runtime::PhysicalArgumentLayout physical;
    std::string writeFootprintKind;
    std::vector<uint32_t> writeFootprintIndices;
    uint32_t divisor{};
};

struct TargetOutput {
    uint32_t location{};
    std::string type;
};

struct TargetModule {
    std::string role;
    std::string entryPoint;
    std::string format;
};

struct TargetBindingPlan {
    VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
    std::string operation;
    std::string topology;
    uint32_t workgroupSize[3]{1, 1, 1};
    vernon::runtime::DispatchContract dispatch;
    uint64_t packedArgumentsSize{};
    std::vector<TargetBinding> bindings;
    std::vector<TargetOutput> outputs;
    std::vector<TargetModule> modules;
    std::vector<vernon::runtime::TensorViewWriteFootprint> readFootprints;
    std::vector<vernon::runtime::TensorViewWriteFootprint> writeFootprints;
    std::vector<vernon::runtime::NativeResourceSlot> nativeSlots;
    DispatchMapping dispatchMapping{DispatchMapping::StaticGrid};
};

struct ResolvedExecutableNode {
    const Node *node{};
    const ResolvedStage *stage{};
    TargetBindingPlan plan;
};

struct ResolvedExecutablePlan {
    std::vector<ResolvedExecutableNode> nodes;
};

vernon::runtime::ValueLayout materializeValueLayout(const ValueLayout &layout, std::string logicalType = {});
bool buildTargetBindingPlan(const ResolvedProgram &program, const Node &node, const ResolvedStage &stage,
                            VernonRuntimeBackend backend, TargetBindingPlan &plan, Diagnostic &diagnostic);
bool buildResolvedExecutablePlan(const ResolvedProgram &program, VernonRuntimeBackend backend,
                                 ResolvedExecutablePlan &plan, Diagnostic &diagnostic);
bool materializeTargetBindingPlan(const TargetBindingPlan &plan, vernon::runtime::Variant &variant,
                                  vernon::runtime::ReflectedEntry &reflection, Diagnostic &diagnostic);
bool materializeGraphicsTargetBindingPlan(const TargetBindingPlan &plan, vernon::runtime::Variant &variant,
                                          Diagnostic &diagnostic);

} // namespace vernon::runtime::program

#endif
