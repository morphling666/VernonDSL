#include "VernonRuntime.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

extern "C" VernonStatus vernonRegisterAutodiffFixture(void);

namespace {

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data, error.size);
}

TEST(RuntimeCookedAutodiff, LoadsStaticallyLinkedObjectsAndExecutesReusablePullback) {
    ASSERT_EQ(vernonRegisterAutodiffFixture(), VERNON_STATUS_OK);

    const std::filesystem::path manifestPath = VERNON_COOKED_AUTODIFF_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    auto runCase = [&](float factor, float selector, const float (&expectedOutput)[2],
                       const float (&expectedValueGradient)[2]) {
        float value[]{2.0f, 3.0f};
        int32_t count = 2;
        float outputValue[2]{};
        VernonAdValue inputValues[]{
            {sizeof(VernonAdValue), {"value", 5}, VERNON_DATA_F32, value, sizeof(value), {}},
            {sizeof(VernonAdValue), {"factor", 6}, VERNON_DATA_F32, &factor, sizeof(factor), {}},
            {sizeof(VernonAdValue), {"selector", 8}, VERNON_DATA_F32, &selector, sizeof(selector), {}},
            {sizeof(VernonAdValue), {"count", 5}, VERNON_DATA_I32, &count, sizeof(count), {}},
        };
        VernonAdValue output{sizeof(VernonAdValue), {"output", 6},       VERNON_DATA_F32,
                             outputValue,           sizeof(outputValue), {}};
        VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 4, {}};
        VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
        VernonPullback *pullback = nullptr;
        ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
            << lastError(context);
        ASSERT_NE(pullback, nullptr);
        EXPECT_FLOAT_EQ(outputValue[0], expectedOutput[0]);
        EXPECT_FLOAT_EQ(outputValue[1], expectedOutput[1]);

        float valueGradient[2]{};
        float factorGradient = 1.0f;
        float selectorGradient = 1.0f;
        VernonAdValue gradientValues[]{
            {sizeof(VernonAdValue), {"value", 5}, VERNON_DATA_F32, valueGradient, sizeof(valueGradient), {}},
            {sizeof(VernonAdValue), {"factor", 6}, VERNON_DATA_F32, &factorGradient, sizeof(factorGradient), {}},
            {sizeof(VernonAdValue), {"selector", 8}, VERNON_DATA_F32, &selectorGradient, sizeof(selectorGradient), {}},
        };
        VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientValues, 3, {}};
        float seedValue[]{1.0f, 2.0f};
        VernonAdValue seed{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedValue, sizeof(seedValue), {}};
        VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
        ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
        EXPECT_FLOAT_EQ(valueGradient[0], expectedValueGradient[0]);
        EXPECT_FLOAT_EQ(valueGradient[1], expectedValueGradient[1]);
        EXPECT_FLOAT_EQ(factorGradient, 0.0f);
        EXPECT_FLOAT_EQ(selectorGradient, 0.0f);

        seedValue[0] = 2.0f;
        seedValue[1] = 3.0f;
        ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
        EXPECT_FLOAT_EQ(valueGradient[0], expectedOutput[0]);
        EXPECT_FLOAT_EQ(valueGradient[1], expectedOutput[1]);
        EXPECT_FLOAT_EQ(factorGradient, 0.0f);
        EXPECT_FLOAT_EQ(selectorGradient, 0.0f);
        vernonPullbackDestroy(pullback);
    };
    const float positiveOutput[]{4.0f, 6.0f};
    const float positiveGradient[]{2.0f, 4.0f};
    runCase(0.0f, 1.0f, positiveOutput, positiveGradient);
    const float negativeOutput[]{6.0f, 9.0f};
    const float negativeGradient[]{3.0f, 6.0f};
    runCase(2.0f, -1.0f, negativeOutput, negativeGradient);

    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
