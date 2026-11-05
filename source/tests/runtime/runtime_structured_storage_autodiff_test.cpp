#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using vernon::tests::HostTensor;
using vernon::tests::LinkedVjp;
using vernon::tests::ResolvedProgram;

// values[index] = scale * source[index] in place, then loss = values[0] + values[index].
TEST(RuntimeStructuredStorageAutodiff, ExecutesDynamicIndexMutationAndFreshStorageGradient) {
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const ResolvedProgram program(context, VERNON_STRUCTURED_STORAGE_AUTODIFF_MANIFEST);
        ASSERT_TRUE(program) << vernon::tests::runtimeError(context);
        LinkedVjp vjp(context, program.get());
        ASSERT_TRUE(vjp) << vjp.error();

        HostTensor<float> values({2.0f, 3.0f, 5.0f}, {1, 1, 1, 3});
        HostTensor<float> source({7.0f, 6.0f, 9.0f}, {1, 1, 1, 3});
        HostTensor<float> loss({0.0f}, {1, 1, 1});
        vjp.forwardRecord().bind("values", values.view());
        vjp.forwardRecord().bind("source", source.view());
        vjp.forwardRecord().bindScalar("index", int32_t{1});
        vjp.forwardRecord().bindScalar("scale", 4.0f);
        vjp.forwardRecord().bind("loss", loss.view());
        vjp.forwardRecord().bindGrid(1);
        ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        EXPECT_FLOAT_EQ(loss.values[0], 26.0f);
        EXPECT_EQ(values.values, (std::vector<float>{2.0f, 24.0f, 5.0f}));

        HostTensor<float> seed({1.0f}, {1, 1, 1});
        HostTensor<float> scaleGradient({0.0f}, {});
        HostTensor<float> sourceGradient({0.0f, 0.0f, 0.0f}, {1, 1, 1, 3});
        HostTensor<float> valuesGradient({0.0f, 0.0f, 0.0f}, {1, 1, 1, 3});
        vjp.backwardRecord().bind("loss", seed.view(), VERNON_PROGRAM_BOUNDARY_COTANGENT);
        vjp.backwardRecord().bind("scale", scaleGradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
        vjp.backwardRecord().bind("source", sourceGradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
        vjp.backwardRecord().bind("values", valuesGradient.view(), VERNON_PROGRAM_BOUNDARY_GRADIENT);
        ASSERT_TRUE(vjp.backwardRecord().unbound().empty());
        ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
        EXPECT_FLOAT_EQ(scaleGradient.values[0], 6.0f);
        EXPECT_EQ(sourceGradient.values, (std::vector<float>{0.0f, 4.0f, 0.0f}));
        // The write replaced values[1], so only values[0] reaches the loss.
        EXPECT_EQ(valuesGradient.values, (std::vector<float>{1.0f, 0.0f, 0.0f}));
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
