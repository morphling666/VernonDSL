#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

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

// Node 0 writes Value 1, which node 1 reads to write Value 2; node 2 draws.
struct PlanFixture {
    std::unique_ptr<VernonRuntimeContext> context = [] {
        auto created = VernonRuntimeContext::create();
        return created.isOk() ? std::move(created).value() : nullptr;
    }();
    std::shared_ptr<VernonStageExecutable> compute = std::make_shared<VernonStageExecutable>();
    std::shared_ptr<VernonStageExecutable> graphics = std::make_shared<VernonStageExecutable>();
    std::shared_ptr<ResolvedProgram> resolved = std::make_shared<ResolvedProgram>();
    ResolvedExecutionPlan plan;

    PlanFixture() {
        EXPECT_NE(context, nullptr);
        context->backend = VERNON_RUNTIME_OPENGL;
        compute->context = context.get();
        compute->bindingProjection.parameters.resize(2);
        graphics->context = context.get();

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
        produce.operation = ComputeOperation{};

        Node consume;
        consume.id = 1;
        consume.name = "consume";
        consume.operands = {1};
        consume.results = {2};
        consume.operation = ComputeOperation{};

        Node draw;
        draw.id = 2;
        draw.name = "draw";
        draw.operation = GraphicsOperation{};

        Graph forward;
        forward.name = "forward";
        forward.direction = "forward";
        forward.nodes = {produce, consume, draw};
        program.graphs.push_back(std::move(forward));

        plan.resolvedProgram = resolved;
        plan.stageCache = {compute, graphics};
        plan.nodes.emplace(
            NodeKey{GraphDirection::Forward, 0},
            ResolvedNodePlan{
                {GraphDirection::Forward, 0},
                compute,
                {{0, std::nullopt, projection(0, TargetCarrier::StorageBuffer, ValueBindingDirection::Input)},
                 {1, std::nullopt, projection(1, TargetCarrier::StorageBuffer, ValueBindingDirection::Result)}},
                ResolvedComputeControls{}});
        plan.nodes.emplace(
            NodeKey{GraphDirection::Forward, 1},
            ResolvedNodePlan{
                {GraphDirection::Forward, 1},
                compute,
                {{1, std::nullopt, projection(1, TargetCarrier::InlineValue, ValueBindingDirection::Input)},
                 {2, std::nullopt, projection(2, TargetCarrier::StorageBuffer, ValueBindingDirection::Result)}},
                ResolvedComputeControls{}});
        plan.nodes.emplace(NodeKey{GraphDirection::Forward, 2},
                           ResolvedNodePlan{{GraphDirection::Forward, 2}, graphics, {}, ResolvedGraphicsControls{}});
    }
};

TEST(ResolvedExecutionPlan, SharedStageRetainsNoNodeProjectionState) {
    PlanFixture fixture;
    const ExecutionPlanResult completed = completeResolvedExecutionPlan(fixture.plan);
    ASSERT_TRUE(completed.isOk()) << completed.error().message;
    const ResolvedNodePlan *first = fixture.plan.node(GraphDirection::Forward, 0);
    const ResolvedNodePlan *second = fixture.plan.node(GraphDirection::Forward, 1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->stage.get(), second->stage.get());
    EXPECT_NE(first->projections[0].value, second->projections[0].value);
    EXPECT_EQ(first->stage->bindingProjection.parameters.size(), 2u);
    EXPECT_EQ(fixture.plan.declaredValueShapes.size(), fixture.resolved->program.values.size());
}

TEST(ResolvedExecutionPlan, RejectsAReadOfAResultNoEarlierNodeProduces) {
    for (const uint32_t producer : {2u, 99u}) {
        SCOPED_TRACE(producer);
        PlanFixture fixture;
        fixture.resolved->program.values[1].origin.node = producer;
        const ExecutionPlanResult completed = completeResolvedExecutionPlan(fixture.plan);
        ASSERT_TRUE(completed.isErr());
        EXPECT_EQ(completed.error().code, "PROGRAM_EXECUTION_PLAN");
        EXPECT_NE(completed.error().message.find("no earlier Node"), std::string::npos);
    }
}

TEST(ResolvedExecutionPlan, RejectsANodeWhoseProjectionsDoNotMatchItsStage) {
    PlanFixture fixture;
    fixture.plan.nodes.at({GraphDirection::Forward, 1}).projections.pop_back();
    const ExecutionPlanResult completed = completeResolvedExecutionPlan(fixture.plan);
    ASSERT_TRUE(completed.isErr());
    EXPECT_NE(completed.error().message.find("Stage endpoint ABI"), std::string::npos);
}

TEST(ResolvedExecutionPlan, RejectsANodeWithoutAResolvedPlan) {
    PlanFixture fixture;
    fixture.plan.nodes.erase({GraphDirection::Forward, 2});
    const ExecutionPlanResult completed = completeResolvedExecutionPlan(fixture.plan);
    ASSERT_TRUE(completed.isErr());
    EXPECT_NE(completed.error().message.find("no resolved plan"), std::string::npos);
}

} // namespace
