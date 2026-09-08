#ifndef VERNON_RUNTIME_RESOLVED_EXECUTION_PLAN_H
#define VERNON_RUNTIME_RESOLVED_EXECUTION_PLAN_H

#include "backend_stage_pipeline.h"
#include "program_execution_manifest.h"
#include "target_binding_plan.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace vernon::runtime::program {

enum class GraphDirection {
    Forward,
    Backward,
    Replay,
};

struct NodeKey {
    GraphDirection graph{GraphDirection::Forward};
    uint32_t node{};

    bool operator<(const NodeKey &other) const {
        return graph != other.graph ? graph < other.graph : node < other.node;
    }
    bool operator==(const NodeKey &other) const { return graph == other.graph && node == other.node; }
    bool operator!=(const NodeKey &other) const { return !(*this == other); }
};

struct GraphValueKey {
    GraphDirection graph{GraphDirection::Forward};
    uint32_t value{};

    bool operator<(const GraphValueKey &other) const {
        return graph != other.graph ? graph < other.graph : value < other.value;
    }
    bool operator==(const GraphValueKey &other) const { return graph == other.graph && value == other.value; }
};

struct NodeEndpointProjection {
    uint32_t value{};
    std::optional<size_t> logicalLeaf;
    TargetBinding target;
};

struct ResolvedComputeControls {
    std::array<ControlComponent, 3> workgroups;
    DispatchMapping dispatchMapping{DispatchMapping::StaticGrid};
};

struct ResolvedGraphicsAttachment {
    uint32_t access{};
    uint32_t storage{};
    uint32_t location{};
    uint32_t aspects{};
};

struct ResolvedGraphicsControls {
    uint32_t renderPassControl{};
    uint32_t drawCommandControl{};
    uint32_t dynamicStateControl{};
    std::vector<ResolvedGraphicsAttachment> colorAttachments;
    std::optional<ResolvedGraphicsAttachment> depthStencilAttachment;
};

using ResolvedOperationControls = std::variant<ResolvedComputeControls, ResolvedGraphicsControls>;

struct ResolvedNodePlan {
    NodeKey key;
    std::shared_ptr<VernonStageExecutable> stage;
    std::vector<NodeEndpointProjection> projections;
    ResolvedOperationControls controls;
};

enum class ResidencyRequirement {
    Host,
    Device,
};

struct AliasDomainPlan {
    uint32_t storage{};
    std::vector<uint32_t> values;
    std::map<GraphDirection, ResidencyRequirement> residency;
};

enum class TransferKind {
    HostUpload,
    DeviceCopy,
    Readback,
};

enum class TransferEndpointKind {
    Boundary,
    Node,
    Storage,
};

struct TransferEndpoint {
    TransferEndpointKind kind{TransferEndpointKind::Storage};
    GraphDirection graph{GraphDirection::Forward};
    uint32_t id{};
};

struct ResolvedTransferEdge {
    TransferKind kind{TransferKind::HostUpload};
    TransferEndpoint producer;
    TransferEndpoint consumer;
    uint32_t value{};
    std::optional<uint32_t> storage;
    uint32_t order{};
};

struct ResolvedTransferPlan {
    std::vector<ResolvedTransferEdge> edges;
};

enum class HazardKind {
    Canonical,
    ReadAfterWrite,
    WriteAfterRead,
    WriteAfterWrite,
};

enum class BarrierRequirement {
    Execution,
    Memory,
    AttachmentTransition,
};

struct ResolvedDependencyEdge {
    NodeKey predecessor;
    NodeKey successor;
    HazardKind hazard{HazardKind::Canonical};
    BarrierRequirement barrier{BarrierRequirement::Execution};
    std::optional<uint32_t> storage;
};

struct ResolvedHazardPlan {
    std::vector<ResolvedDependencyEdge> edges;
    std::map<NodeKey, std::vector<uint32_t>> predecessors;
};

enum class AttachmentTransition {
    Preserve,
    Initialize,
    ReadWrite,
};

struct ResolvedAttachmentTransition {
    uint32_t storage{};
    uint32_t location{};
    uint32_t aspects{};
    AttachmentTransition transition{AttachmentTransition::Preserve};
};

struct ResolvedGraphicsScopePlan {
    NodeKey node;
    std::vector<ResolvedAttachmentTransition> attachments;
};

struct ResolvedTapeRequirement {
    uint32_t value{};
    bool forwardProducer{};
    bool backwardConsumer{};
    std::vector<vernon::program_plan::TapeCarrier> requiredCarriers;
    std::vector<vernon::program_plan::TapeCarrier> optionalCarriers;
};

enum class CheckpointPolicy {
    Retain,
    Rematerialize,
};

struct ResolvedAutodiffPlan {
    std::vector<ResolvedTapeRequirement> tapes;
    std::vector<uint32_t> residualValues;
    std::vector<NodeKey> replayNodes;
    CheckpointPolicy checkpointPolicy{CheckpointPolicy::Retain};
};

enum class PublicationCommitMode {
    CommitAfterSuccess,
    InPlace,
};

struct ResolvedPublicationTransaction {
    uint32_t slot{};
    uint32_t value{};
    ProgramOwnerId stagingOwner;
    PublicationCommitMode mode{PublicationCommitMode::CommitAfterSuccess};
};

struct ResolvedPublicationPlan {
    std::vector<ResolvedPublicationTransaction> transactions;
};

// Immutable canonical execution authority. Backend handles occur only through
// reusable Stage references; every other record is logical Program planning.
struct ResolvedExecutionPlan {
    std::shared_ptr<const ResolvedProgram> resolvedProgram;
    std::vector<vernon::runtime::ValueLayout> boundaryLayoutViews;
    std::vector<std::shared_ptr<VernonStageExecutable>> stageCache;
    std::map<NodeKey, ResolvedNodePlan> nodes;
    std::map<uint32_t, AliasDomainPlan> aliasDomains;
    std::map<GraphValueKey, ResidencyRequirement> residency;
    ResolvedTransferPlan transfers;
    ResolvedHazardPlan hazards;
    std::vector<ResolvedGraphicsScopePlan> graphicsScopes;
    ResolvedAutodiffPlan autodiff;
    ResolvedPublicationPlan publications;

    const ResolvedNodePlan *node(GraphDirection graph, uint32_t nodeId) const;
    bool requiresDevice(GraphDirection graph, uint32_t value) const;
    const std::vector<uint32_t> &predecessors(GraphDirection graph, uint32_t nodeId) const;
};

std::optional<GraphDirection> graphDirection(std::string_view direction);
bool buildResolvedExecutionPolicies(ResolvedExecutionPlan &plan, Diagnostic &diagnostic);
bool validateResolvedExecutionPlan(const ResolvedExecutionPlan &plan, Diagnostic &diagnostic);

} // namespace vernon::runtime::program

#endif
