#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

using vernon::tests::HostTensor;
using vernon::tests::LinkedVjp;
using vernon::tests::ResolvedProgram;

// Each lane of a 4-lane workgroup writes 2 * scale * the carrier of its next neighbour across a barrier.
TEST(RuntimeWorkgroupAutodiff, ReplaysBarriersInReverseForEveryLaneGradient) {
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const ResolvedProgram program(context, VERNON_STRUCTURED_STORAGE_WORKGROUP_AUTODIFF_MANIFEST);
        ASSERT_TRUE(program) << vernon::tests::runtimeError(context);
        LinkedVjp vjp(context, program.get());
        ASSERT_TRUE(vjp) << vjp.error();

        constexpr size_t laneCount = 8;
        constexpr float scale = 1.5f;
        HostTensor<float> carriers({1.0f, 2.0f, 4.0f, 8.0f, 3.0f, 5.0f, 7.0f, 11.0f});
        HostTensor<float> output(std::vector<float>(laneCount), {1, 1, laneCount});
        vjp.forwardRecord().bind("carriers", carriers.view());
        vjp.forwardRecord().bindScalar("scale", scale);
        vjp.forwardRecord().bind("output", output.view());
        vjp.forwardRecord().bindGrid(2);
        ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        const VernonScheduleVjpPass tapes = vjp.passTotals();
        EXPECT_GT(tapes.tape_logical_bytes, 0u);
        EXPECT_GE(tapes.tape_resident_bytes, tapes.tape_logical_bytes);
        EXPECT_GE(tapes.tape_allocated_bytes, tapes.tape_resident_bytes);
        for (size_t lane = 0; lane < laneCount; ++lane) {
            const size_t neighbor = lane / 4 * 4 + (lane + 1) % 4;
            EXPECT_FLOAT_EQ(output.values[lane], carriers.values[neighbor] * scale * 2.0f);
        }

        HostTensor<float> seeds({1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 11.0f, 13.0f, 17.0f}, {1, 1, laneCount});
        HostTensor<float> carrierGradients{std::vector<float>(laneCount)};
        HostTensor<float> scaleGradient({0.0f}, {});
        vjp.backwardRecord().bind("output", seeds.view(), VERNON_PROGRAM_BOUNDARY_COTANGENT);
        vjp.backwardRecord().bind("carriers", carrierGradients.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
        vjp.backwardRecord().bind("scale", scaleGradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
        ASSERT_TRUE(vjp.backwardRecord().unbound().empty());
        float expectedScaleGradient = 0.0f;
        for (size_t lane = 0; lane < laneCount; ++lane)
            expectedScaleGradient += 2.0f * carriers.values[lane / 4 * 4 + (lane + 1) % 4] * seeds.values[lane];
        // A second pullback of the same forward call replays the same tapes.
        for (int pullback = 0; pullback < 2; ++pullback) {
            SCOPED_TRACE(pullback);
            std::fill(carrierGradients.values.begin(), carrierGradients.values.end(), 0.0f);
            scaleGradient.values[0] = 0.0f;
            ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
            for (size_t lane = 0; lane < laneCount; ++lane) {
                const size_t consumer = lane / 4 * 4 + (lane + 3) % 4;
                EXPECT_FLOAT_EQ(carrierGradients.values[lane], 2.0f * scale * seeds.values[consumer]);
            }
            EXPECT_FLOAT_EQ(scaleGradient.values[0], expectedScaleGradient);
        }
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
