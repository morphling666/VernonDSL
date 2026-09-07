#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace {

using namespace vernon::runtime;
using namespace vernon::runtime::program;

TargetBinding projection(uint32_t value, TargetCarrier carrier, ValueBindingDirection direction) {
    TargetBinding target;
    target.endpoint.interfaceKind = "argument";
    target.projection.value = value;
    target.projection.direction = direction;
    target.carrier = carrier;
    return target;
}

Value value(uint32_t id, OriginKind origin, uint32_t storage, uint32_t node = 0) {
    Value result;
    result.id = id;
    result.storage = storage;
    result.origin.kind = origin;
    result.origin.graph = "forward";
    result.origin.node = node;
    return result;
}

ResourceAccess access(AccessKind kind, uint32_t storage, uint32_t value, const char *mode = "") {
    ResourceAccess result;
    result.kind = kind;
    result.storage = storage;
    result.value = value;
    result.access = mode;
    return result;
}

struct PlanFixture {
    VernonRuntimeContext context;
    std::shared_ptr<VernonStageExecutable> compute = std::make_shared<VernonStageExecutable>();
    std::shared_ptr<VernonStageExecutable> graphics = std::make_shared<VernonStageExecutable>();
    std::shared_ptr<ResolvedProgram> resolved = std::make_shared<ResolvedProgram>();
    ResolvedExecutionPlan plan;

    PlanFixture() {
        context.backend = VERNON_RUNTIME_OPENGL;
        compute->context = &context;
        compute->bindingProjection.parameters.resize(2);
        graphics->context = &context;

        Program &program = resolved->program;
        program.storages = {{0}, {1}};
        program.values = {
            value(0, OriginKind::Argument, 0),
            value(1, OriginKind::NodeResult, 0, 0),
            value(2, OriginKind::NodeResult, 1, 1),
        };

        Node produce;
        produce.id = 0;
        produce.name = "produce";
        produce.results = {1};
        produce.accesses = {access(AccessKind::Read, 0, 0), access(AccessKind::Write, 0, 1)};
        produce.operation = ComputeOperation{};

        Node consume;
        consume.id = 1;
        consume.name = "consume";
        consume.operands = {1};
        consume.results = {2};
        consume.accesses = {access(AccessKind::Read, 0, 1), access(AccessKind::Initialize, 1, 2)};
        consume.operation = ComputeOperation{};

        Node readAlias;
        readAlias.id = 2;
        readAlias.name = "read_alias";
        readAlias.operands = {2};
        readAlias.accesses = {access(AccessKind::Read, 1, 2)};
        readAlias.operation = ComputeOperation{};

        Node overwriteAlias;
        overwriteAlias.id = 3;
        overwriteAlias.name = "overwrite_alias";
        overwriteAlias.accesses = {access(AccessKind::Write, 1, 2)};
        overwriteAlias.operation = ComputeOperation{};

        Node draw;
        draw.id = 4;
        draw.name = "draw";
        draw.accesses = {access(AccessKind::Attachment, 1, 2, "read_write")};
        GraphicsOperation graphicsOperation;
        graphicsOperation.colorAttachments.push_back({0, 0, {}, {}, 1});
        draw.operation = graphicsOperation;

        Graph forward;
        forward.name = "forward";
        forward.direction = "forward";
        forward.nodes = {produce, consume, readAlias, overwriteAlias, draw};
        program.graphs.push_back(std::move(forward));

        Graph replay;
        replay.name = "replay";
        replay.direction = "replay";
        Node replayNode;
        replayNode.id = 0;
        replayNode.name = "replay";
        replayNode.operation = ComputeOperation{};
        replay.nodes.push_back(replayNode);
        program.graphs.push_back(std::move(replay));

        resolved->graphs.resize(2);
        resolved->graphs[0].predecessors.resize(5);
        resolved->graphs[1].predecessors.resize(1);

        program.residualContract = ResidualContract{{{1}}};
        program.abi.tapePlans.push_back({2,
                                         true,
                                         true,
                                         {vernon::program_plan::TapeCarrier::TapeData},
                                         {vernon::program_plan::TapeCarrier::ReplayStatus}});
        BoundarySlot commit;
        commit.id = 0;
        commit.value = 2;
        commit.role = BoundaryRole::Output;
        commit.aliasOwner = {ProgramOwnerKind::Storage, 1};
        commit.publication = BoundaryPublication::CommitAfterSuccess;
        BoundarySlot inPlace;
        inPlace.id = 1;
        inPlace.value = 1;
        inPlace.role = BoundaryRole::Output;
        inPlace.aliasOwner = {ProgramOwnerKind::Storage, 0};
        inPlace.publication = BoundaryPublication::InPlace;
        program.abi.boundarySlots = {commit, inPlace};
        program.abi.publication.targets.push_back({0, 2, BoundaryRole::Output, commit.aliasOwner});

        plan.resolvedProgram = resolved;
        plan.stageCache = {compute, graphics};
        const auto computeControls = ResolvedComputeControls{};
        plan.nodes.emplace(
            NodeKey{GraphDirection::Forward, 0},
            ResolvedNodePlan{
                {GraphDirection::Forward, 0},
                compute,
                {{0, std::nullopt, projection(0, TargetCarrier::StorageBuffer, ValueBindingDirection::Input)},
                 {1, std::nullopt, projection(1, TargetCarrier::StorageBuffer, ValueBindingDirection::Result)}},
                computeControls});
        plan.nodes.emplace(
            NodeKey{GraphDirection::Forward, 1},
            ResolvedNodePlan{
                {GraphDirection::Forward, 1},
                compute,
                {{1, std::nullopt, projection(1, TargetCarrier::InlineValue, ValueBindingDirection::Input)},
                 {2, std::nullopt, projection(2, TargetCarrier::StorageBuffer, ValueBindingDirection::Result)}},
                computeControls});
        plan.nodes.emplace(NodeKey{GraphDirection::Forward, 2},
                           ResolvedNodePlan{{GraphDirection::Forward, 2}, graphics, {}, computeControls});
        plan.nodes.emplace(NodeKey{GraphDirection::Forward, 3},
                           ResolvedNodePlan{{GraphDirection::Forward, 3}, graphics, {}, computeControls});
        ResolvedGraphicsControls graphicsControls;
        graphicsControls.colorAttachments.push_back({0, 1, 0, 1});
        plan.nodes.emplace(NodeKey{GraphDirection::Forward, 4},
                           ResolvedNodePlan{{GraphDirection::Forward, 4}, graphics, {}, graphicsControls});
        plan.nodes.emplace(NodeKey{GraphDirection::Replay, 0},
                           ResolvedNodePlan{{GraphDirection::Replay, 0}, graphics, {}, computeControls});
    }
};

TEST(ResolvedExecutionPlan, SharedStageRetainsNoNodeProjectionState) {
    PlanFixture fixture;
    Diagnostic diagnostic;
    ASSERT_TRUE(buildResolvedExecutionPolicies(fixture.plan, diagnostic)) << diagnostic.message;
    ASSERT_TRUE(validateResolvedExecutionPlan(fixture.plan, diagnostic)) << diagnostic.message;
    const ResolvedNodePlan *first = fixture.plan.node(GraphDirection::Forward, 0);
    const ResolvedNodePlan *second = fixture.plan.node(GraphDirection::Forward, 1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->stage.get(), second->stage.get());
    EXPECT_NE(first->projections[0].value, second->projections[0].value);
    EXPECT_EQ(first->stage->bindingProjection.parameters.size(), 2u);
}

TEST(ResolvedExecutionPlan, DerivesTypedResidencyTransferHazardGraphicsAndAutodiffPolicies) {
    PlanFixture fixture;
    Diagnostic diagnostic;
    ASSERT_TRUE(buildResolvedExecutionPolicies(fixture.plan, diagnostic)) << diagnostic.message;
    ASSERT_TRUE(validateResolvedExecutionPlan(fixture.plan, diagnostic)) << diagnostic.message;

    ASSERT_EQ(fixture.plan.aliasDomains.at(0).values, std::vector<uint32_t>({0, 1}));
    EXPECT_TRUE(fixture.plan.requiresDevice(GraphDirection::Forward, 0));
    EXPECT_TRUE(fixture.plan.requiresDevice(GraphDirection::Forward, 1));
    const auto transfer = std::find_if(
        fixture.plan.transfers.edges.begin(), fixture.plan.transfers.edges.end(), [](const ResolvedTransferEdge &edge) {
            return edge.kind == TransferKind::Readback && edge.value == 1 && edge.consumer.id == 1;
        });
    EXPECT_NE(transfer, fixture.plan.transfers.edges.end());

    const auto hasHazard = [&](HazardKind kind, uint32_t predecessor, uint32_t successor) {
        return std::any_of(fixture.plan.hazards.edges.begin(), fixture.plan.hazards.edges.end(),
                           [&](const ResolvedDependencyEdge &edge) {
                               return edge.hazard == kind && edge.predecessor.node == predecessor &&
                                      edge.successor.node == successor;
                           });
    };
    EXPECT_TRUE(hasHazard(HazardKind::ReadAfterWrite, 0, 1));
    EXPECT_TRUE(hasHazard(HazardKind::WriteAfterRead, 2, 3));
    EXPECT_TRUE(hasHazard(HazardKind::WriteAfterWrite, 1, 3));

    ASSERT_EQ(fixture.plan.graphicsScopes.size(), 1u);
    EXPECT_EQ(fixture.plan.graphicsScopes[0].attachments[0].transition, AttachmentTransition::ReadWrite);
    ASSERT_EQ(fixture.plan.autodiff.tapes.size(), 1u);
    EXPECT_EQ(fixture.plan.autodiff.residualValues, std::vector<uint32_t>({1}));
    ASSERT_EQ(fixture.plan.autodiff.replayNodes.size(), 1u);
    EXPECT_EQ(fixture.plan.autodiff.checkpointPolicy, CheckpointPolicy::Rematerialize);
    ASSERT_EQ(fixture.plan.publications.transactions.size(), 2u);
    EXPECT_EQ(fixture.plan.publications.transactions[0].mode, PublicationCommitMode::CommitAfterSuccess);
    EXPECT_EQ(fixture.plan.publications.transactions[1].mode, PublicationCommitMode::InPlace);
}

TEST(ResolvedExecutionPlan, OrdersDeviceResultTransferBeforeUniformConsumer) {
    PlanFixture fixture;
    fixture.plan.nodes.at({GraphDirection::Forward, 1}).projections[0].target.carrier = TargetCarrier::UniformBuffer;
    Diagnostic diagnostic;
    ASSERT_TRUE(buildResolvedExecutionPolicies(fixture.plan, diagnostic)) << diagnostic.message;
    ASSERT_TRUE(validateResolvedExecutionPlan(fixture.plan, diagnostic)) << diagnostic.message;

    const auto transfer = std::find_if(
        fixture.plan.transfers.edges.begin(), fixture.plan.transfers.edges.end(), [](const ResolvedTransferEdge &edge) {
            return edge.kind == TransferKind::DeviceCopy && edge.value == 1 &&
                   edge.producer.kind == TransferEndpointKind::Node && edge.producer.id == 0 &&
                   edge.consumer.kind == TransferEndpointKind::Node && edge.consumer.id == 1;
        });
    ASSERT_NE(transfer, fixture.plan.transfers.edges.end());
    EXPECT_TRUE(fixture.plan.requiresDevice(GraphDirection::Forward, 1));
    const std::vector<uint32_t> &predecessors = fixture.plan.predecessors(GraphDirection::Forward, 1);
    EXPECT_NE(std::find(predecessors.begin(), predecessors.end(), 0), predecessors.end());
}

TEST(ResolvedExecutionPlan, CpuUsesSameLogicalUniformProjectionWithoutDeviceTransfer) {
    PlanFixture fixture;
    fixture.context.backend = VERNON_RUNTIME_CPU;
    fixture.plan.nodes.at({GraphDirection::Forward, 1}).projections[0].target.carrier = TargetCarrier::UniformBuffer;
    Diagnostic diagnostic;
    ASSERT_TRUE(buildResolvedExecutionPolicies(fixture.plan, diagnostic)) << diagnostic.message;
    ASSERT_TRUE(validateResolvedExecutionPlan(fixture.plan, diagnostic)) << diagnostic.message;

    EXPECT_FALSE(fixture.plan.requiresDevice(GraphDirection::Forward, 1));
    EXPECT_EQ(fixture.plan.node(GraphDirection::Forward, 1)->projections[0].target.projection.value, 1u);
    EXPECT_TRUE(std::none_of(
        fixture.plan.transfers.edges.begin(), fixture.plan.transfers.edges.end(),
        [](const ResolvedTransferEdge &edge) { return edge.kind == TransferKind::DeviceCopy && edge.value == 1; }));
}

TEST(ResolvedExecutionPlan, RejectsPhysicalReadWithoutDominatingProducer) {
    PlanFixture fixture;
    fixture.resolved->program.values[1].origin.node = 99;
    Diagnostic diagnostic;
    ASSERT_TRUE(buildResolvedExecutionPolicies(fixture.plan, diagnostic)) << diagnostic.message;
    EXPECT_FALSE(validateResolvedExecutionPlan(fixture.plan, diagnostic));
    EXPECT_EQ(diagnostic.code, "PROGRAM_EXECUTION_PLAN");
    EXPECT_NE(diagnostic.message.find("dominating producer"), std::string::npos);
}

TEST(ResolvedExecutionPlan, RejectsMissingAliasBarrierAndPublicationStagingTarget) {
    {
        PlanFixture fixture;
        Diagnostic diagnostic;
        ASSERT_TRUE(buildResolvedExecutionPolicies(fixture.plan, diagnostic)) << diagnostic.message;
        fixture.plan.hazards.edges.erase(
            std::remove_if(fixture.plan.hazards.edges.begin(), fixture.plan.hazards.edges.end(),
                           [](const ResolvedDependencyEdge &edge) {
                               return edge.hazard == HazardKind::WriteAfterRead && edge.predecessor.node == 2 &&
                                      edge.successor.node == 3;
                           }),
            fixture.plan.hazards.edges.end());
        EXPECT_FALSE(validateResolvedExecutionPlan(fixture.plan, diagnostic));
        EXPECT_NE(diagnostic.message.find("conflicting alias"), std::string::npos);
    }
    {
        PlanFixture fixture;
        fixture.resolved->program.abi.publication.targets.clear();
        Diagnostic diagnostic;
        ASSERT_TRUE(buildResolvedExecutionPolicies(fixture.plan, diagnostic)) << diagnostic.message;
        EXPECT_FALSE(validateResolvedExecutionPlan(fixture.plan, diagnostic));
        EXPECT_NE(diagnostic.message.find("staging target"), std::string::npos);
    }
}

} // namespace
