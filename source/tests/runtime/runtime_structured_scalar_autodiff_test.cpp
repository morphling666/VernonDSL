#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using vernon::tests::HostTensor;
using vernon::tests::LinkedVjp;
using vernon::tests::ResolvedProgram;

constexpr VernonProgramBoundaryRole kCotangent = VERNON_PROGRAM_BOUNDARY_COTANGENT;
constexpr VernonProgramBoundaryRole kGradient = VERNON_PROGRAM_BOUNDARY_GRADIENT;

double objective(double x, double y, double z) {
    const double linear = x + y;
    const double difference = x - y;
    return linear * difference / y - z + std::sin(x) + std::cos(y) + std::exp(z) + std::log(x) + std::sqrt(y) +
           std::acos(z) + std::atan2(y, x) + std::abs(y - z) + std::pow(x, y);
}

double finiteDifference(double x, double y, double z, unsigned argument) {
    constexpr double epsilon = 1e-4;
    double positive[] = {x, y, z};
    double negative[] = {x, y, z};
    positive[argument] += epsilon;
    negative[argument] -= epsilon;
    return (objective(positive[0], positive[1], positive[2]) - objective(negative[0], negative[1], negative[2])) /
           (2.0 * epsilon);
}

class StructuredScalarAutodiff : public testing::Test {
protected:
    void SetUp() override { ASSERT_NE(context_, nullptr); }
    void TearDown() override { EXPECT_EQ(vernonRuntimeDestroy(context_), VERNON_STATUS_OK); }

    VernonRuntimeContext *context_{vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr)};
};

// Each grid workgroup writes the objective into a 3 x 2 block of the output.
struct ObjectiveCall {
    ObjectiveCall(LinkedVjp &vjp, uint32_t workgroups, float x, float y, float z)
        : output(std::vector<float>(size_t{6} * workgroups), {1, 3, uint64_t{2} * workgroups}),
          seeds(std::vector<float>(output.values.size()), output.shape) {
        vjp.forwardRecord().bindScalar("x", x);
        vjp.forwardRecord().bindScalar("y", y);
        vjp.forwardRecord().bindScalar("z", z);
        vjp.forwardRecord().bind("output", output.view());
        vjp.forwardRecord().bindGrid(workgroups);
        vjp.backwardRecord().bind("output", seeds.view(), kCotangent);
        for (size_t index = 0; index < 3; ++index)
            vjp.backwardRecord().bind(std::string(1, "xyz"[index]), gradients[index].view(), kGradient);
    }

    HostTensor<float> output;
    HostTensor<float> seeds;
    HostTensor<float> gradients[3]{HostTensor<float>({0.0f}, {}), HostTensor<float>({0.0f}, {}),
                                   HostTensor<float>({0.0f}, {})};
};

TEST_F(StructuredScalarAutodiff, ProfilesMatchAnalyticVjp) {
    const ResolvedProgram program(context_, VERNON_STRUCTURED_SCALAR_AUTODIFF_MANIFEST);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(context_);
    EXPECT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(program.get()), 1u);

    for (const auto &[x, y, z] : {std::array<float, 3>{1.2f, 0.7f, 0.2f}, std::array<float, 3>{0.8f, 0.3f, 0.6f}}) {
        LinkedVjp vjp(context_, program.get());
        ASSERT_TRUE(vjp) << vjp.error();
        ObjectiveCall call(vjp, 1, x, y, z);
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        for (float value : call.output.values)
            EXPECT_NEAR(value, objective(x, y, z), 2e-5);
        // One forward call's residuals serve every pullback of it.
        for (float seed : {1.75f, 0.5f}) {
            std::fill(call.seeds.values.begin(), call.seeds.values.end(), seed);
            ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
            for (unsigned index = 0; index < 3; ++index)
                EXPECT_NEAR(call.gradients[index].values[0], 6.0 * seed * finiteDifference(x, y, z, index), 3e-3);
        }
    }

    // Two workgroups: every lane's seed weighs its own output.
    LinkedVjp vjp(context_, program.get());
    ASSERT_TRUE(vjp) << vjp.error();
    ObjectiveCall call(vjp, 2, 1.2f, 0.7f, 0.2f);
    ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
    for (float value : call.output.values)
        EXPECT_NEAR(value, objective(1.2, 0.7, 0.2), 2e-5);
    for (size_t lane = 0; lane < call.seeds.values.size(); ++lane)
        call.seeds.values[lane] = static_cast<float>(lane + 1);
    ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
    for (unsigned index = 0; index < 3; ++index)
        EXPECT_NEAR(call.gradients[index].values[0], 78.0 * finiteDifference(1.2f, 0.7f, 0.2f, index), 3e-3);
}

// The tapes a forward call records grow with its grid, and the residual block holds them until its pullback.
TEST_F(StructuredScalarAutodiff, TapesScaleWithTheGridAndServeTheirPullback) {
    const ResolvedProgram program(context_, VERNON_STRUCTURED_SCALAR_BALANCED_AUTODIFF_MANIFEST);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(context_);
    uint64_t previous = 0;
    for (uint32_t workgroups : {512u, 1024u}) {
        SCOPED_TRACE(workgroups);
        LinkedVjp vjp(context_, program.get());
        ASSERT_TRUE(vjp) << vjp.error();
        ObjectiveCall call(vjp, workgroups, 1.2f, 0.7f, 0.2f);
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        for (float value : call.output.values)
            ASSERT_NEAR(value, objective(1.2, 0.7, 0.2), 2e-5);
        const VernonScheduleVjpPass tapes = vjp.passTotals();
        EXPECT_GT(tapes.tape_count, 0u);
        EXPECT_GT(tapes.tape_logical_bytes, previous);
        EXPECT_GE(tapes.tape_resident_bytes, tapes.tape_logical_bytes);
        EXPECT_GE(tapes.tape_allocated_bytes, tapes.tape_resident_bytes);
        previous = tapes.tape_logical_bytes;

        std::fill(call.seeds.values.begin(), call.seeds.values.end(), 1.0f);
        ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
        const double lanes = static_cast<double>(call.output.values.size());
        for (unsigned index = 0; index < 3; ++index)
            EXPECT_NEAR(call.gradients[index].values[0], lanes * finiteDifference(1.2, 0.7, 0.2, index), 0.004 * lanes);
    }
}

// output = (1 + 2 count) x through a loop of `count` iterations, which the forward call records on a dynamic tape.
struct LoopCall {
    LoopCall(LinkedVjp &vjp, int32_t count) {
        vjp.forwardRecord().bindScalar("x", 1.25f);
        vjp.forwardRecord().bindScalar("count", count);
        vjp.forwardRecord().bind("output", output.view());
        vjp.forwardRecord().bindGrid(1);
        vjp.backwardRecord().bind("output", seed.view(), kCotangent);
        vjp.backwardRecord().bind("x", gradient.view(), kGradient);
    }

    HostTensor<float> output{std::vector<float>{-17.0f}};
    HostTensor<float> seed{std::vector<float>{1.0f}};
    HostTensor<float> gradient{std::vector<float>{0.0f}, {}};
};

TEST_F(StructuredScalarAutodiff, DynamicTapesGrowLinearlyWithExecutedIterations) {
    const ResolvedProgram program(context_, VERNON_STRUCTURED_DYNAMIC_AUTODIFF_MANIFEST);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(context_);
    LinkedVjp vjp(context_, program.get());
    ASSERT_TRUE(vjp) << vjp.error();
    const auto measure = [&](int32_t count) -> uint64_t {
        SCOPED_TRACE(count);
        LoopCall call(vjp, count);
        EXPECT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        EXPECT_FLOAT_EQ(call.output.values[0], (1.0f + 2.0f * count) * 1.25f);
        const VernonScheduleVjpPass tapes = vjp.passTotals();
        EXPECT_GT(tapes.dynamic_tape, 0u);
        EXPECT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
        EXPECT_FLOAT_EQ(call.gradient.values[0], 1.0f + 2.0f * count);
        return tapes.tape_logical_bytes;
    };
    const uint64_t zero = measure(0);
    measure(1);
    const uint64_t medium = measure(32) - zero;
    const uint64_t longLoop = measure(1500) - zero;
    ASSERT_GT(medium, 0u);
    EXPECT_GE(longLoop, medium * 40u);
    EXPECT_LE(longLoop, medium * 50u);
}

// A checkpointed forward call whose inputs alone exceed the budget fails before it writes anything.
TEST_F(StructuredScalarAutodiff, CheckpointBudgetBelowTheInputsRejectsTheForwardCall) {
    const ResolvedProgram program(context_, VERNON_STRUCTURED_SCALAR_AUTODIFF_MANIFEST);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(context_);
    uint64_t inputs = 0;
    {
        const VernonScheduleCheckpointOptions unbounded{sizeof(VernonScheduleCheckpointOptions),
                                                        std::numeric_limits<uint64_t>::max(),
                                                        VERNON_SCHEDULE_CHECKPOINT_BALANCED};
        LinkedVjp vjp(context_, program.get(), &unbounded);
        ASSERT_TRUE(vjp) << vjp.error();
        ObjectiveCall call(vjp, 1, 1.2f, 0.7f, 0.2f);
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        VernonScheduleCheckpoint checkpoint{sizeof(VernonScheduleCheckpoint)};
        ASSERT_EQ(vernonScheduleResidualsGetCheckpoint(vjp.residuals(), &checkpoint), VERNON_STATUS_OK);
        ASSERT_EQ(checkpoint.present, 1u);
        EXPECT_LE(checkpoint.held_bytes, checkpoint.memory_budget);
        inputs = checkpoint.initial_state_bytes;
        ASSERT_GT(inputs, 0u);
        std::fill(call.seeds.values.begin(), call.seeds.values.end(), 1.0f);
        ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
        for (unsigned index = 0; index < 3; ++index)
            EXPECT_NEAR(call.gradients[index].values[0], 6.0 * finiteDifference(1.2, 0.7, 0.2, index), 3e-3);
    }

    const VernonScheduleCheckpointOptions bounded{sizeof(VernonScheduleCheckpointOptions), inputs - 1,
                                                  VERNON_SCHEDULE_CHECKPOINT_BALANCED};
    LinkedVjp vjp(context_, program.get(), &bounded);
    ASSERT_TRUE(vjp) << vjp.error();
    ObjectiveCall call(vjp, 1, 1.2f, 0.7f, 0.2f);
    std::fill(call.output.values.begin(), call.output.values.end(), -17.0f);
    EXPECT_EQ(vjp.forward(), VERNON_STATUS_INVALID_ARGUMENT);
    const std::string rejection = vjp.error();
    EXPECT_NE(rejection.find("checkpoint memory budget"), std::string::npos) << rejection;
    for (float value : call.output.values)
        EXPECT_FLOAT_EQ(value, -17.0f);
}

TEST_F(StructuredScalarAutodiff, PreservesF64PrimalCotangentAndGradientDtypes) {
    const ResolvedProgram program(context_, VERNON_STRUCTURED_F64_AUTODIFF_MANIFEST);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(context_);
    for (VernonProgramBoundaryRole role : {VERNON_PROGRAM_BOUNDARY_OUTPUT, kCotangent, kGradient}) {
        VernonProgramParameterView boundary{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(program.get(), role, 0, &boundary),
                  VERNON_STATUS_OK);
        VernonProgramValueLeafView leaf{sizeof(VernonProgramValueLeafView)};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryValueLeaf(program.get(), role, boundary.slot, 0, &leaf),
                  VERNON_STATUS_OK);
        EXPECT_EQ(leaf.value.dtype, VERNON_DATA_F64);
    }

    LinkedVjp vjp(context_, program.get());
    ASSERT_TRUE(vjp) << vjp.error();
    HostTensor<double> output({0.0});
    HostTensor<double> seed({1.0});
    HostTensor<double> gradient({0.0}, {});
    vjp.forwardRecord().bindScalar("x", 1.5);
    vjp.forwardRecord().bind("output", output.view());
    vjp.forwardRecord().bindGrid(1);
    ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
    ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_DOUBLE_EQ(output.values[0], 3.75);
    vjp.backwardRecord().bind("output", seed.view(), kCotangent);
    vjp.backwardRecord().bind("x", gradient.view(), kGradient);
    ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_DOUBLE_EQ(gradient.values[0], 4.0);
    seed.values[0] = 2.0;
    ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_DOUBLE_EQ(gradient.values[0], 8.0);
}

} // namespace
