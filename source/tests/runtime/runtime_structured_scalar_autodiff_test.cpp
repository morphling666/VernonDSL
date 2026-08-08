#include "VernonRuntime.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/host_tape_test_hooks.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

extern "C" VernonStatus vernonRegisterStructuredScalarAutodiffFixture(void);
extern "C" VernonStatus vernonRegisterStructuredDynamicAutodiffFixture(void);
extern "C" VernonStatus vernonRegisterStructuredF64AutodiffFixture(void);

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
        const uint64_t outputShape[]{1};
        float outputValue{};
        VernonAdValue inputValues[] = {
            {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), {}},
            {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &y, sizeof(y), {}},
            {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &z, sizeof(z), {}},
            {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 1, outputShape},
        };
        VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 4, {}};
        VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
        VernonPullback *pullback = nullptr;
        ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
            << lastError(context);
        ASSERT_NE(pullback, nullptr);
        EXPECT_NEAR(outputValue, objective(x, y, z), 2e-5);

        float seedValue = 1.75f;
        const uint64_t seedShape[]{1};
        VernonAdValue seed{
            sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &seedValue, sizeof(seedValue), 1, seedShape};
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
    float outputValue{};
    const uint64_t outputShape[]{1};
    VernonAdValue inputValues[] = {
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 0, nullptr},
        {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &y, sizeof(y), 0, nullptr},
        {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &z, sizeof(z), 0, nullptr},
        {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 1, outputShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 4, {0, 0, 0, 0}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {0, 0, 0, 0}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {2, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_NEAR(outputValue, objective(x, y, z), 2e-5);

    float seedValues[]{1.0f, 2.0f};
    const uint64_t invocationShape[]{1, 1, 2, 1};
    VernonAdValue seed{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedValues, sizeof(seedValues), 4,
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

TEST(RuntimeStructuredScalarAutodiff, CpuRejectsLegacyFixedProtocolAtResolve) {
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
    ASSERT_TRUE(bundle->autodiff.has_value());
    bundle->autodiff->protocol = "legacy_fixed";

    EXPECT_EQ(vernonRuntimeResolvePipeline(bundle, {nullptr, 0}), nullptr);
    EXPECT_EQ(lastError(context), "CPU autodiff profile uses an unsupported protocol");

    vernonRuntimePipelineBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
TEST(RuntimeStructuredScalarAutodiff, DynamicTapeTraversalScalesLinearlyWithExecutedRecords) {
    ASSERT_EQ(vernonRegisterStructuredDynamicAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_DYNAMIC_AUTODIFF_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    vernon::runtime::ad::HostTapeTraversalMetrics metrics;
    vernon::runtime::ad::HostTapeTraversalScope traversalScope(metrics);
    vernon::runtime::ad::setHostTapeMemoryPolicyForTesting(
        *context,
        std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(
            vernon::runtime::ad::kDefaultHostTapeInvocationLimit, vernon::runtime::ad::kDefaultHostTapeContextLimit));
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    auto measure = [&](int32_t count) {
        float x = 1.25f;
        float output = 0.0f;
        const uint64_t shape[]{1};
        VernonAdValue inputValues[]{
            {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), {}},
            {sizeof(VernonAdValue), {"count", 5}, VERNON_DATA_I32, &count, sizeof(count), {}},
            {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &output, sizeof(output), 1, shape},
        };
        VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
        VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
        VernonPullback *pullback = nullptr;
        EXPECT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
            << lastError(context);
        EXPECT_NE(pullback, nullptr);
        if (!pullback)
            return metrics;
        EXPECT_FLOAT_EQ(output, (1.0f + 2.0f * count) * x);

        metrics.reset();
        float gradient = 0.0f;
        VernonAdValue gradientValue{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradient, sizeof(gradient), {}};
        VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradientValue, 1, {}};
        EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
        EXPECT_FLOAT_EQ(gradient, 1.0f + 2.0f * count);
        vernonPullbackDestroy(pullback);
        return metrics;
    };

    const auto zero = measure(0);
    const auto one = measure(1);
    const auto medium = measure(32);
    const auto longLoop = measure(1500);
    EXPECT_EQ(zero.recordResolutions, zero.leafReads + zero.childReads);
    EXPECT_EQ(one.recordResolutions, one.leafReads + one.childReads);
    EXPECT_EQ(medium.recordResolutions, medium.leafReads + medium.childReads);
    EXPECT_EQ(longLoop.recordResolutions, longLoop.leafReads + longLoop.childReads);
    const size_t mediumVariable = medium.recordResolutions - zero.recordResolutions;
    const size_t longVariable = longLoop.recordResolutions - zero.recordResolutions;
    ASSERT_GT(mediumVariable, 0u);
    EXPECT_GE(longVariable, mediumVariable * 40u);
    EXPECT_LE(longVariable, mediumVariable * 50u);
    EXPECT_LE(longLoop.regionLookups,
              longLoop.leafReads + 2u * longLoop.childReads + longLoop.executedCountReads + longLoop.exitKindReads);

    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}
#endif

TEST(RuntimeStructuredScalarAutodiff, PreservesF64PrimalCotangentAndGradientDtypes) {
    ASSERT_EQ(vernonRegisterStructuredF64AutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_F64_AUTODIFF_MANIFEST;
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

    VernonAdValueMetadataView outputMetadata{sizeof(VernonAdValueMetadataView)};
    VernonAdValueMetadataView cotangentMetadata{sizeof(VernonAdValueMetadataView)};
    VernonAdValueMetadataView gradientMetadata{sizeof(VernonAdValueMetadataView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputByIndex(pipeline, 0, &outputMetadata), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdCotangentByIndex(pipeline, 0, &cotangentMetadata), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradientByIndex(pipeline, 0, &gradientMetadata), VERNON_STATUS_OK);
    EXPECT_EQ(outputMetadata.dtype, VERNON_DATA_F64);
    EXPECT_EQ(cotangentMetadata.dtype, VERNON_DATA_F64);
    EXPECT_EQ(gradientMetadata.dtype, VERNON_DATA_F64);

    double x = 1.5;
    double output = 0.0;
    const uint64_t shape[]{1};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F64, &x, sizeof(x), {}},
        {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F64, &output, sizeof(output), 1, shape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_DOUBLE_EQ(output, 3.75);

    double gradient = 0.0;
    VernonAdValue gradientValue{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F64, &gradient, sizeof(gradient), {}};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradientValue, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_DOUBLE_EQ(gradient, 4.0);
    double seed = 2.0;
    VernonAdValue cotangent{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F64, &seed, sizeof(seed), 1, shape};
    VernonAdValueSet cotangents{sizeof(VernonAdValueSet), &cotangent, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_DOUBLE_EQ(gradient, 8.0);

    vernonPullbackDestroy(pullback);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
