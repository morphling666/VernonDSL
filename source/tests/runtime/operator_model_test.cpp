#include "operator/elementwise_operator.h"
#include "operator/operator_execution.h"
#include "operator/operator_lowering.h"
#include "operator/operator_model.h"
#include "operator/operator_validation.h"

#include <gtest/gtest.h>

#include <limits>

namespace vernon::ops {
namespace {

TensorViewDescriptor view(uintptr_t owner, std::vector<uint64_t> shape, std::vector<int64_t> strides,
                          uint64_t offset = 0, VernonDataType dtype = VERNON_DATA_F32, uint64_t allocationBytes = 64) {
    return {dtype, std::move(shape), std::move(strides), offset, allocationBytes, owner};
}

TEST(OperatorModel, ValidatesStridedAndNegativeStrideFootprints) {
    std::string error;
    EXPECT_TRUE(validateTensorViewDescriptor(view(1, {}, {}, 4, VERNON_DATA_F32, 8), error)) << error;
    error.clear();
    EXPECT_TRUE(validateTensorViewDescriptor(view(1, {2, 3}, {16, 4}, 0, VERNON_DATA_F32, 28), error)) << error;
    error.clear();
    EXPECT_TRUE(validateTensorViewDescriptor(view(1, {3}, {-4}, 8, VERNON_DATA_F32, 12), error)) << error;
    error.clear();
    EXPECT_FALSE(validateTensorViewDescriptor(view(1, {3}, {-4}, 4, VERNON_DATA_F32, 12), error));
    EXPECT_EQ(error, "operator tensor view exceeds its physical owner");
    error.clear();
    EXPECT_FALSE(validateTensorViewDescriptor(
        view(1, {std::numeric_limits<uint64_t>::max()}, {std::numeric_limits<int64_t>::max()}), error));
    error.clear();
    EXPECT_FALSE(validateTensorViewDescriptor(
        view(1, {2}, {std::numeric_limits<int64_t>::min()}, std::numeric_limits<uint64_t>::max()), error));
}

TEST(OperatorModel, BuildsTopologicallyOrderedElementwiseDag) {
    OperatorDag dag;
    const uint32_t left = dag.addLeaf(view(1, {4}, {4}, 0, VERNON_DATA_F32, 16));
    const uint32_t right = dag.addLeaf(view(2, {4}, {4}, 0, VERNON_DATA_F32, 16));
    dag.addElementwise(ElementwiseOperatorKind::Add, {left, right}, view(3, {4}, {4}, 0, VERNON_DATA_F32, 16));
    std::string error;
    EXPECT_TRUE(validateOperatorDag(dag, error)) << error;
}

TEST(OperatorModel, LowersDeviceAddToDerivativeCommandNode) {
    OperatorDag dag;
    const uint32_t left = dag.addLeaf(view(1, {4}, {4}, 0, VERNON_DATA_F32, 16));
    const uint32_t right = dag.addLeaf(view(2, {4}, {4}, 0, VERNON_DATA_F32, 16));
    dag.addElementwise(ElementwiseOperatorKind::Add, {left, right}, view(3, {4}, {4}, 0, VERNON_DATA_F32, 16));
    execution::detail::CommandDag commands;
    std::string error;
    ASSERT_TRUE(lowerOperatorDagToCommandDag(dag, commands, error)) << error;
    ASSERT_EQ(commands.nodes.size(), 1u);
    EXPECT_EQ(commands.nodes.front().kind, execution::detail::CommandNodeKind::Derivative);
    EXPECT_EQ(commands.nodes.front().queue, execution::detail::CommandQueueClass::Compute);
    ASSERT_EQ(commands.nodes.front().accesses.size(), 3u);
    EXPECT_EQ(commands.nodes.front().accesses[0].resource, left);
    EXPECT_EQ(commands.nodes.front().accesses[2].resource, 2u);
    EXPECT_EQ(commands.nodes.front().accesses[2].access, execution::AccessMode::Write);
}

VernonRhiStatus encodeOperator(void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }

TEST(OperatorModel, BuildsProductionExecutionPlanForMultiNodeDag) {
    OperatorDag dag;
    const uint32_t left = dag.addLeaf(view(1, {4}, {4}, 0, VERNON_DATA_F32, 16));
    const uint32_t right = dag.addLeaf(view(2, {4}, {4}, 0, VERNON_DATA_F32, 16));
    const uint32_t temporary =
        dag.addElementwise(ElementwiseOperatorKind::Add, {left, right}, view(3, {4}, {4}, 0, VERNON_DATA_F32, 16));
    const uint32_t extra = dag.addLeaf(view(4, {4}, {4}, 0, VERNON_DATA_F32, 16));
    dag.addElementwise(ElementwiseOperatorKind::Add, {temporary, extra}, view(5, {4}, {4}, 0, VERNON_DATA_F32, 16));
    std::vector<execution::detail::RhiCommandNodeEncoder> encoders = {
        {encodeOperator, nullptr, nullptr},
        {encodeOperator, nullptr, nullptr},
    };
    std::vector<execution::detail::RhiCommandResourceBinding> bindings;
    for (uint64_t owner = 1; owner <= 5; ++owner)
        bindings.push_back({owner, execution::ResourceKind::Buffer, {static_cast<uint32_t>(owner), 1}, {}});
    OperatorExecutionPlan plan;
    std::string error;
    ASSERT_TRUE(buildOperatorExecutionPlan(dag, encoders, bindings, plan, error)) << error;
    ASSERT_EQ(plan.commands.nodes.size(), 2u);
    ASSERT_EQ(plan.encoders.size(), 2u);
    EXPECT_EQ(plan.commands.nodes[1].predecessors, std::vector<uint32_t>({0}));
    EXPECT_EQ(plan.commands.nodes[0].accesses.size(), 3u);
    EXPECT_EQ(plan.commands.nodes[1].accesses.size(), 3u);
}

TEST(OperatorModel, RejectsIncompleteExecutionPlan) {
    OperatorDag dag;
    const uint32_t left = dag.addLeaf(view(1, {4}, {4}));
    const uint32_t right = dag.addLeaf(view(2, {4}, {4}));
    dag.addElementwise(ElementwiseOperatorKind::Add, {left, right}, view(3, {4}, {4}));
    OperatorExecutionPlan plan;
    std::string error;
    EXPECT_FALSE(
        buildOperatorExecutionPlan(dag, std::vector<execution::detail::RhiCommandNodeEncoder>(1), {}, plan, error));
    EXPECT_EQ(error, "RHI command plan has an invalid node encoder");
    error.clear();
    EXPECT_FALSE(buildOperatorExecutionPlan(dag, {{encodeOperator, nullptr, nullptr}}, {}, plan, error));
    EXPECT_EQ(error, "RHI command plan has an unbound resource access");
}

TEST(OperatorModel, RejectsIncompatibleElementwiseShapes) {
    ElementwiseAddPlan plan;
    std::string error;
    EXPECT_FALSE(planElementwiseAdd(view(1, {4}, {4}), view(2, {2}, {4}), view(3, {4}, {4}), plan, error));
    EXPECT_EQ(error, "elementwise operator input shapes do not match");
}

TEST(OperatorModel, SelectsExplicitFallbackForUnsupportedDtypeAndAliasing) {
    ElementwiseAddPlan plan;
    std::string error;
    ASSERT_TRUE(planElementwiseAdd(view(1, {4}, {8}, 0, VERNON_DATA_F64), view(2, {4}, {8}, 0, VERNON_DATA_F64),
                                   view(3, {4}, {8}, 0, VERNON_DATA_F64), plan, error))
        << error;
    EXPECT_FALSE(plan.device);
    EXPECT_FALSE(plan.fallbackReason.empty());

    ASSERT_TRUE(planElementwiseAdd(view(1, {4}, {4}), view(2, {4}, {4}), view(1, {4}, {4}), plan, error)) << error;
    EXPECT_FALSE(plan.device);
    EXPECT_EQ(plan.fallbackReason, "device elementwise Add requires an invocation-private output owner");
}

} // namespace
} // namespace vernon::ops
