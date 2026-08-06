#include "VernonRuntime.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

extern "C" VernonStatus vernonRegisterStructuredScalarAutodiffFixture(void);

namespace {

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data, error.size);
}

TEST(RuntimeStructuredScalarAutodiff, ProfilesMatchAnalyticVjp) {
    ASSERT_EQ(vernonRegisterStructuredScalarAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_SCALAR_AUTODIFF_MANIFEST;
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

    auto objective = [](double x, double y, double z) {
        const double linear = x + y;
        const double difference = x - y;
        return linear * difference / y - z + std::sin(x) + std::cos(y) + std::exp(z) + std::log(x) + std::sqrt(y) +
               std::acos(z) + std::atan2(y, x) + std::abs(y - z) + std::pow(x, y);
    };
    auto finiteDifference = [&](double x, double y, double z, unsigned argument) {
        constexpr double epsilon = 1e-4;
        double positive[] = {x, y, z};
        double negative[] = {x, y, z};
        positive[argument] += epsilon;
        negative[argument] -= epsilon;
        return (objective(positive[0], positive[1], positive[2]) - objective(negative[0], negative[1], negative[2])) /
               (2.0 * epsilon);
    };
    auto runCase = [&](float x, float y, float z) {
        VernonAdValue inputValues[] = {
            {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), {}},
            {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &y, sizeof(y), {}},
            {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &z, sizeof(z), {}},
        };
        float outputValue{};
        VernonAdValue output{sizeof(VernonAdValue), {"output", 6},       VERNON_DATA_F32,
                             &outputValue,          sizeof(outputValue), {}};
        VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 3, {}};
        VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
        VernonPullback *pullback = nullptr;
        ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
            << lastError(context);
        ASSERT_NE(pullback, nullptr);
        EXPECT_NEAR(outputValue, objective(x, y, z), 2e-5);

        float seedValue = 1.75f;
        VernonAdValue seed{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &seedValue, sizeof(seedValue), {}};
        VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
        float gradientValuesStorage[3]{};
        VernonAdValue gradientValues[] = {
            {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValuesStorage[0], sizeof(float), {}},
            {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &gradientValuesStorage[1], sizeof(float), {}},
            {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &gradientValuesStorage[2], sizeof(float), {}},
        };
        VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientValues, 3, {}};
        ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
        for (unsigned index = 0; index < 3; ++index)
            EXPECT_NEAR(gradientValuesStorage[index], seedValue * finiteDifference(x, y, z, index), 3e-3);

        seedValue = 0.5f;
        ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
        for (unsigned index = 0; index < 3; ++index)
            EXPECT_NEAR(gradientValuesStorage[index], seedValue * finiteDifference(x, y, z, index), 3e-3);
        vernonPullbackDestroy(pullback);
    };
    runCase(1.2f, 0.7f, 0.2f);
    runCase(0.8f, 0.3f, 0.6f);

    float x = 1.2f;
    float y = 0.7f;
    float z = 0.2f;
    VernonAdValue inputValues[] = {
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 0, nullptr},
        {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &y, sizeof(y), 0, nullptr},
        {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &z, sizeof(z), 0, nullptr},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 3, {0, 0, 0, 0}};
    float outputValues[2]{};
    const uint64_t invocationShape[]{1, 1, 2};
    VernonAdValue output{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, outputValues, sizeof(outputValues), 3,
                         invocationShape};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {0, 0, 0, 0}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {2, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_NEAR(outputValues[0], objective(x, y, z), 2e-5);
    EXPECT_NEAR(outputValues[1], objective(x, y, z), 2e-5);

    float seedValues[]{1.0f, 2.0f};
    VernonAdValue seed{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedValues, sizeof(seedValues), 3,
                       invocationShape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {0, 0, 0, 0}};
    float gradientStorage[3]{};
    VernonAdValue gradientValues[] = {
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientStorage[0], sizeof(float), 0, nullptr},
        {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &gradientStorage[1], sizeof(float), 0, nullptr},
        {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &gradientStorage[2], sizeof(float), 0, nullptr},
    };
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientValues, 3, {0, 0, 0, 0}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    for (unsigned index = 0; index < 3; ++index)
        EXPECT_NEAR(gradientStorage[index], 3.0 * finiteDifference(x, y, z, index), 3e-3);
    vernonPullbackDestroy(pullback);

    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
