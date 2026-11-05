#include "acceptance_descriptor.h"
#include "runtime/autodiff/runtime_autodiff_memory.h"
#include "runtime/autodiff/runtime_gpu_replay.h"
#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using vernon::runtime::ad::gpu::BatchSummary;
using vernon::runtime::ad::gpu::Segment;
using vernon::tests::AcceptanceDescriptor;
using vernon::tests::DeviceTensor;
using vernon::tests::LinkedVjp;
using vernon::tests::ResolvedProgram;

#include "language_contract_acceptance_descriptors.inc"

TEST(RuntimeGpuAutodiff, MemoryAccountingSeparatesLogicalObservationalAndBudgetedBytes) {
    vernon::runtime::ad::MemoryAccounting usage;
    usage.logicalPayloadBytes = 10;
    usage.retainedAllocationBytes = 20;
    usage.residentBytes = 30;
    usage.allocatedBytes = 40;
    usage.checkpointBytes = 50;
    usage.observeTemporary(60);
    usage.observeTemporary(5);

    size_t budgeted = 0;
    ASSERT_TRUE(usage.budgetedBytes(budgeted));
    EXPECT_EQ(budgeted, 130u);
    EXPECT_EQ(usage.logicalPayloadBytes, 10u);
    EXPECT_EQ(usage.residentBytes, 30u);
    EXPECT_EQ(usage.allocatedBytes, 40u);
    EXPECT_EQ(usage.peakTemporaryBytes, 60u);
}

TEST(RuntimeGpuAutodiff, ReplayBudgetSelectsLargestBoundedBatch) {
    vernon::runtime::ad::gpu::BatchBudget budget;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          10, 100, 20, budget));
    EXPECT_EQ(budget.capacity, 3u);
    EXPECT_EQ(budget.tapeBytes, 300u);
    EXPECT_EQ(budget.segmentBytes, 3 * sizeof(Segment));
    EXPECT_EQ(budget.statusBytes, 20u);
    EXPECT_EQ(budget.memory.peakTemporaryBytes, 420 + 6 * sizeof(Segment));
    EXPECT_FALSE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 299, 1,
                                                           100, 20, budget));
}

TEST(RuntimeGpuAutodiff, ReplayBudgetShrinksBatchAfterDynamicStrideGrowth) {
    vernon::runtime::ad::gpu::BatchBudget initial;
    vernon::runtime::ad::gpu::BatchBudget grown;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          10, 100, 20, initial));
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          10, 300, 20, grown));
    EXPECT_EQ(initial.capacity, 3u);
    EXPECT_EQ(grown.capacity, 1u);
    EXPECT_EQ(grown.memory.peakTemporaryBytes, 420 + 2 * sizeof(Segment));
    EXPECT_LE(grown.memory.peakTemporaryBytes, 1000u);
}

TEST(RuntimeGpuAutodiff, ReplayBudgetUsesSharedPlanningPolicy) {
    using vernon::runtime::ad::PlanningPolicy;
    vernon::runtime::ad::gpu::BatchBudget minMemory;
    vernon::runtime::ad::gpu::BatchBudget balanced;
    vernon::runtime::ad::gpu::BatchBudget minRuntime;
    constexpr size_t groupCount = 1000000;
    constexpr size_t maximumBytes = 600 * 1024 * 1024;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(PlanningPolicy::MinMemory, 0, maximumBytes, groupCount, 1024,
                                                          sizeof(BatchSummary), minMemory));
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(PlanningPolicy::Balanced, 0, maximumBytes, groupCount, 1024,
                                                          sizeof(BatchSummary), balanced));
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(PlanningPolicy::MinRuntime, 0, maximumBytes, groupCount, 1024,
                                                          sizeof(BatchSummary), minRuntime));
    EXPECT_LT(minMemory.capacity, balanced.capacity);
    EXPECT_LE(balanced.capacity, minRuntime.capacity);
    EXPECT_LT(minMemory.memory.peakTemporaryBytes, balanced.memory.peakTemporaryBytes);
    EXPECT_LE(balanced.memory.peakTemporaryBytes, minRuntime.memory.peakTemporaryBytes);
    EXPECT_GT(minMemory.batchCount, balanced.batchCount);
    EXPECT_GE(balanced.batchCount, minRuntime.batchCount);
}

TEST(RuntimeGpuAutodiff, MinMemoryReplayUsesOneTransactionWhenWholePassFits) {
    vernon::runtime::ad::gpu::BatchBudget budget;
    constexpr size_t groupCount = 4096;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinMemory, 0,
                                                          1024 * 1024, groupCount, 64, sizeof(BatchSummary), budget));
    EXPECT_EQ(budget.capacity, groupCount);
    EXPECT_EQ(budget.batchCount, 1u);
    EXPECT_TRUE(budget.wholeDispatch);
}

TEST(RuntimeGpuAutodiff, ProvenStaticReplayUsesPolicyCapacityWhenWholePassFits) {
    vernon::runtime::ad::gpu::BatchBudget budget;
    constexpr size_t groupCount = 1000000;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinMemory, 0,
                                                          2ull * 1024 * 1024 * 1024, groupCount, 1024,
                                                          sizeof(BatchSummary), budget, false));
    EXPECT_LT(budget.capacity, groupCount);
    EXPECT_GT(budget.batchCount, 1u);
    EXPECT_FALSE(budget.wholeDispatch);
}

TEST(RuntimeGpuAutodiff, ReplaySegmentPreservesVirtualCoordinates) {
    Segment segment{};
    ASSERT_TRUE(vernon::runtime::ad::gpu::initializeSegment({3, 2, 2}, {4, 8, 1}, 10, 256, 32, segment));
    EXPECT_EQ(segment.virtualWorkgroup[0], 1u);
    EXPECT_EQ(segment.virtualWorkgroup[1], 1u);
    EXPECT_EQ(segment.virtualWorkgroup[2], 1u);
    EXPECT_EQ(segment.virtualGlobalBase[0], 4u);
    EXPECT_EQ(segment.virtualGlobalBase[1], 8u);
    EXPECT_EQ(segment.virtualGlobalBase[2], 1u);
    EXPECT_EQ(segment.tapeStride, 256u);
    EXPECT_EQ(segment.capacityBytes, 8192u);
}

TEST(RuntimeGpuAutodiff, ReplayBatchSegmentsUseDisjointTapeRanges) {
    Segment first{};
    Segment second{};
    ASSERT_TRUE(vernon::runtime::ad::gpu::initializeBatchSegment({8, 1, 1}, {4, 1, 1}, 6, 0, 256, 4, 2048, first));
    ASSERT_TRUE(vernon::runtime::ad::gpu::initializeBatchSegment({8, 1, 1}, {4, 1, 1}, 7, 1, 256, 4, 2048, second));
    EXPECT_EQ(first.virtualWorkgroup[0], 6u);
    EXPECT_EQ(second.virtualWorkgroup[0], 7u);
    EXPECT_EQ(first.tapeBase, 0u);
    EXPECT_EQ(second.tapeBase, 1024u);
    EXPECT_EQ(first.capacityBytes, 2048u);
    EXPECT_EQ(second.capacityBytes, 2048u);
}

TEST(RuntimeGpuAutodiff, ReplaySchedulerOwnsReverseBatchProgressAndResize) {
    vernon::runtime::ad::gpu::ReverseBatchScheduler scheduler(10, 4);
    ASSERT_TRUE(scheduler.valid());
    EXPECT_EQ(scheduler.current().begin, 6u);
    EXPECT_EQ(scheduler.current().count, 4u);
    scheduler.commit();
    EXPECT_EQ(scheduler.current().begin, 2u);
    EXPECT_EQ(scheduler.current().count, 4u);
    ASSERT_TRUE(scheduler.setCapacity(3));
    EXPECT_EQ(scheduler.current().begin, 3u);
    EXPECT_EQ(scheduler.current().count, 3u);
    scheduler.commit();
    EXPECT_EQ(scheduler.current().begin, 0u);
    EXPECT_EQ(scheduler.current().count, 3u);
    scheduler.commit();
    EXPECT_TRUE(scheduler.empty());
}

TEST(RuntimeGpuAutodiff, DynamicTapeRetryUsesLargestLaneRequirement) {
    const BatchSummary summary{257, 1};
    size_t stride = 0;
    size_t bytes = 0;
    uint32_t status = 0;
    ASSERT_TRUE(vernon::runtime::ad::gpu::requiredTapeBytes(summary, 4, 128, stride, bytes, status));
    EXPECT_EQ(stride, 260u);
    EXPECT_EQ(bytes, 1040u);
    EXPECT_EQ(status, 1u);
}

TEST(RuntimeGpuAutodiff, TapeStrideKeepsAdjacentLaneWordsDisjoint) {
    size_t stride = 0;
    ASSERT_TRUE(vernon::runtime::ad::gpu::normalizeTapeStride(17, stride));
    EXPECT_EQ(stride, 20u);
    ASSERT_TRUE(vernon::runtime::ad::gpu::normalizeTapeStride(0, stride));
    EXPECT_EQ(stride, 16u);
    EXPECT_FALSE(vernon::runtime::ad::gpu::normalizeTapeStride(std::numeric_limits<size_t>::max(), stride));
}

TEST(RuntimeGpuAutodiff, TapeReadbackIsOneFixedBatchSummary) {
    vernon::runtime::ad::gpu::BatchBudget oneGroup;
    vernon::runtime::ad::gpu::BatchBudget manyGroups;
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000, 1,
                                                          100, sizeof(BatchSummary), oneGroup));
    ASSERT_TRUE(vernon::runtime::ad::gpu::planBatchBudget(vernon::runtime::ad::PlanningPolicy::MinRuntime, 100, 1000,
                                                          100, 100, sizeof(BatchSummary), manyGroups));
    EXPECT_EQ(oneGroup.statusBytes, sizeof(BatchSummary));
    EXPECT_EQ(manyGroups.statusBytes, sizeof(BatchSummary));
}

constexpr VernonProgramBoundaryRole kCotangent = VERNON_PROGRAM_BOUNDARY_COTANGENT;
constexpr VernonProgramBoundaryRole kGradient = VERNON_PROGRAM_BOUNDARY_GRADIENT;

class RuntimeGpuAutodiffMatrix : public vernon::tests::HalProgramTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::programVjpFixtureRequirements(GetParam().runtime);
    }

    // A device scalar a backward call writes a gradient into.
    DeviceTensor scalarGradient() const {
        const float zero = 0.0f;
        return DeviceTensor(device_, &zero, sizeof(zero), {}, sizeof(zero));
    }

    // loss = values * values elementwise, with no tape: every pullback reads only the forward inputs.
    void runNoTapeVjp(VernonProgramExecutable *program, const AcceptanceDescriptor &acceptance) {
        ASSERT_EQ(acceptance.expectedCount, 8u);
        LinkedVjp vjp(runtime(), program);
        ASSERT_TRUE(vjp) << vjp.error();
        const DeviceTensor values(device_, std::vector<float>{2.0f, 3.0f, 5.0f, 7.0f}, {2, 2});
        const DeviceTensor loss(device_, std::vector<float>(4), {2, 2});
        vjp.forwardRecord().bind("values", values);
        vjp.forwardRecord().bind("loss", loss);
        vjp.forwardRecord().bindGrid(1);
        ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        const std::vector<float> forward = loss.values<float>();
        for (size_t index = 0; index < forward.size(); ++index)
            EXPECT_FLOAT_EQ(forward[index], acceptance.expected[index]);

        const DeviceTensor cotangent(device_, std::vector<float>(4, 1.0f), {2, 2});
        DeviceTensor gradient(device_, std::vector<float>(4), {2, 2});
        vjp.backwardRecord().bind("loss", cotangent, kCotangent);
        vjp.backwardRecord().bind("values", gradient, kGradient);
        for (int pullback = 0; pullback < 2; ++pullback) {
            SCOPED_TRACE(pullback);
            const std::vector<float> cleared(4, -23.0f);
            gradient.buffer().write(cleared.data(), cleared.size() * sizeof(float));
            ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
            const std::vector<float> result = gradient.values<float>();
            for (size_t index = 0; index < result.size(); ++index)
                EXPECT_FLOAT_EQ(result[index], acceptance.expected[index + 4]);
        }
    }

    // A 192-lane workgroup reduction, which is no power of two, feeds two losses that both reach `scale`.
    void runNonPowerOfTwoReductionVjp(VernonProgramExecutable *program, const AcceptanceDescriptor &acceptance) {
        ASSERT_EQ(acceptance.expectedCount, 1u);
        constexpr size_t laneCount = 192;
        LinkedVjp vjp(runtime(), program);
        ASSERT_TRUE(vjp) << vjp.error();
        std::vector<float> initial(laneCount);
        for (size_t lane = 0; lane < laneCount; ++lane)
            initial[lane] = static_cast<float>(lane + 1);
        const DeviceTensor values(device_, initial);
        const DeviceTensor carriedLoss(device_, std::vector<float>(laneCount));
        const DeviceTensor sharedLoss(device_, std::vector<float>(laneCount));
        vjp.forwardRecord().bindScalar("scale", 2.0f);
        vjp.forwardRecord().bind("values", values);
        vjp.forwardRecord().bind("carried_loss", carriedLoss);
        vjp.forwardRecord().bind("shared_loss", sharedLoss);
        vjp.forwardRecord().bindGrid(2);
        ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        const std::vector<float> carried = carriedLoss.values<float>();
        const std::vector<float> shared = sharedLoss.values<float>();
        for (size_t lane = 0; lane < laneCount; ++lane) {
            EXPECT_FLOAT_EQ(carried[lane], 2.0f * initial[lane]);
            EXPECT_FLOAT_EQ(shared[lane], 2.0f * initial[lane] * initial[lane]);
        }

        const DeviceTensor seeds(device_, std::vector<float>(laneCount, 1.0f));
        const DeviceTensor gradient = scalarGradient();
        vjp.backwardRecord().bind("carried_loss", seeds, kCotangent);
        vjp.backwardRecord().bind("shared_loss", seeds, kCotangent);
        vjp.backwardRecord().bind("scale", gradient, kGradient);
        ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
        EXPECT_NEAR(gradient.values<float>()[0], acceptance.expected[0], 0.5f);
    }

    // Eight lanes over two workgroups record a static tape of x and y, or a dynamic one of x through `count`
    // iterations, that every pullback of the forward call replays.
    void runCapturedTapeVjp(VernonProgramExecutable *program, bool dynamic, const AcceptanceDescriptor &acceptance) {
        ASSERT_EQ(acceptance.expectedCount, dynamic ? 2u : 3u);
        constexpr size_t laneCount = 8;
        LinkedVjp vjp(runtime(), program);
        ASSERT_TRUE(vjp) << vjp.error();
        const DeviceTensor output(device_, std::vector<float>(laneCount));
        vjp.forwardRecord().bindScalar("x", dynamic ? 2.0f : 3.0f);
        if (dynamic)
            vjp.forwardRecord().bindScalar("count", int32_t{2});
        else
            vjp.forwardRecord().bindScalar("y", 2.0f);
        vjp.forwardRecord().bind("output", output);
        vjp.forwardRecord().bindGrid(2);
        ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        for (float value : output.values<float>())
            EXPECT_FLOAT_EQ(value, acceptance.expected[0]);
        const VernonScheduleVjpPass tapes = vjp.passTotals();
        EXPECT_GT(tapes.tape_count, 0u);
        EXPECT_GT(tapes.tape_logical_bytes, 0u);
        EXPECT_GE(tapes.tape_allocated_bytes, tapes.tape_logical_bytes);

        const DeviceTensor seeds(device_, std::vector<float>(laneCount, 1.0f));
        const DeviceTensor xGradient = scalarGradient();
        const DeviceTensor yGradient = scalarGradient();
        vjp.backwardRecord().bind("output", seeds, kCotangent);
        vjp.backwardRecord().bind("x", xGradient, kGradient);
        if (!dynamic)
            vjp.backwardRecord().bind("y", yGradient, kGradient);
        ASSERT_TRUE(vjp.backwardRecord().unbound().empty());
        for (int pullback = 0; pullback < 2; ++pullback) {
            SCOPED_TRACE(pullback);
            ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
            EXPECT_NEAR(xGradient.values<float>()[0], acceptance.expected[1], 1e-4f);
            if (!dynamic)
                EXPECT_NEAR(yGradient.values<float>()[0], acceptance.expected[2], 1e-4f);
        }
    }
};

TEST_P(RuntimeGpuAutodiffMatrix, ExecutesRegisteredAcceptanceOracles) {
    size_t executed = 0;
    for (const AcceptanceDescriptor &acceptance : acceptanceDescriptors) {
        if (acceptance.suite != "gpu_autodiff")
            continue;
        ++executed;
        SCOPED_TRACE(acceptance.fixtureId);
        const ResolvedProgram program(runtime(), fixture(acceptance.fixtureId).manifestPath);
        ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
        if (acceptance.oracle == "no_tape_vjp")
            runNoTapeVjp(program.get(), acceptance);
        else if (acceptance.oracle == "reduction_vjp")
            runNonPowerOfTwoReductionVjp(program.get(), acceptance);
        else if (acceptance.oracle == "static_vjp")
            runCapturedTapeVjp(program.get(), false, acceptance);
        else if (acceptance.oracle == "dynamic_vjp")
            runCapturedTapeVjp(program.get(), true, acceptance);
        else
            FAIL() << "GPU autodiff acceptance has no executor: " << acceptance.oracle;
    }
    EXPECT_GT(executed, 0u);
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, RuntimeGpuAutodiffMatrix,
                         testing::ValuesIn(vernon::tests::programFixtureCases("gpu_autodiff")),
                         [](const testing::TestParamInfo<vernon::tests::ProgramFixtureManifest> &info) {
                             return std::string(info.param.target);
                         });

} // namespace
