#include "VernonRuntime.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/host_tape_test_hooks.h"
#include "runtime/autodiff/runtime_autodiff_telemetry.h"
#include "runtime/runtime_state.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern "C" VernonStatus vernonRegisterStructuredScalarAutodiffFixture(void);
extern "C" VernonStatus vernonRegisterStructuredScalarBalancedAutodiffFixture(void);
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
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    EXPECT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(pipeline), 1u);

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
        constexpr size_t laneCount = 6;
        const uint64_t outputShape[]{1, 3, 2};
        float outputValues[laneCount]{};
        VernonAdValue inputValues[] = {
            {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), {}},
            {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &y, sizeof(y), {}},
            {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &z, sizeof(z), {}},
            {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, outputValues, sizeof(outputValues), 3, outputShape},
        };
        VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 4, {}};
        VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
        VernonPullback *pullback = nullptr;
        ASSERT_EQ(vernon::tests::completeCanonicalAutodiffInvocation(pipeline, {1, 1, 1}, inputs, outputs, &pullback),
                  VERNON_STATUS_OK)
            << lastError(context);
        ASSERT_NE(pullback, nullptr);
        for (float outputValue : outputValues)
            EXPECT_NEAR(outputValue, objective(x, y, z), 2e-5);

        float seedValue = 1.75f;
        float seedValues[laneCount];
        std::fill(std::begin(seedValues), std::end(seedValues), seedValue);
        VernonAdValue seed{
            sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedValues, sizeof(seedValues), 3, outputShape};
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
            EXPECT_NEAR(gradientValuesStorage[index], 6.0 * seedValue * finiteDifference(x, y, z, index), 3e-3);

        seedValue = 0.5f;
        std::fill(std::begin(seedValues), std::end(seedValues), seedValue);
        ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
        for (unsigned index = 0; index < 3; ++index)
            EXPECT_NEAR(gradientValuesStorage[index], 6.0 * seedValue * finiteDifference(x, y, z, index), 3e-3);

        std::array<VernonStatus, 2> concurrentStatuses{};
        std::array<std::array<float, 3>, 2> concurrentGradientStorage{};
        std::array<std::thread, 2> workers;
        for (size_t worker = 0; worker < workers.size(); ++worker) {
            workers[worker] = std::thread([&, worker] {
                const float concurrentSeedValue = static_cast<float>(worker + 1);
                float concurrentSeedValues[laneCount];
                std::fill(std::begin(concurrentSeedValues), std::end(concurrentSeedValues), concurrentSeedValue);
                VernonAdValue concurrentSeed{sizeof(VernonAdValue),
                                             {"output", 6},
                                             VERNON_DATA_F32,
                                             concurrentSeedValues,
                                             sizeof(concurrentSeedValues),
                                             3,
                                             outputShape};
                VernonAdValueSet concurrentSeeds{sizeof(VernonAdValueSet), &concurrentSeed, 1, {}};
                VernonAdValue concurrentGradientValues[] = {
                    {sizeof(VernonAdValue),
                     {"x", 1},
                     VERNON_DATA_F32,
                     &concurrentGradientStorage[worker][0],
                     sizeof(float),
                     {}},
                    {sizeof(VernonAdValue),
                     {"y", 1},
                     VERNON_DATA_F32,
                     &concurrentGradientStorage[worker][1],
                     sizeof(float),
                     {}},
                    {sizeof(VernonAdValue),
                     {"z", 1},
                     VERNON_DATA_F32,
                     &concurrentGradientStorage[worker][2],
                     sizeof(float),
                     {}},
                };
                VernonAdValueSet concurrentGradients{sizeof(VernonAdValueSet), concurrentGradientValues, 3, {}};
                concurrentStatuses[worker] = vernonPullbackApply(pullback, &concurrentSeeds, &concurrentGradients);
            });
        }
        for (std::thread &worker : workers)
            worker.join();
        for (size_t worker = 0; worker < workers.size(); ++worker) {
            ASSERT_EQ(concurrentStatuses[worker], VERNON_STATUS_OK) << lastError(context);
            for (unsigned index = 0; index < 3; ++index)
                EXPECT_NEAR(concurrentGradientStorage[worker][index],
                            6.0 * static_cast<double>(worker + 1) * finiteDifference(x, y, z, index), 3e-3);
        }
        vernonPullbackDestroy(pullback);
    };
    runCase(1.2f, 0.7f, 0.2f);
    runCase(0.8f, 0.3f, 0.6f);

    float x = 1.2f;
    float y = 0.7f;
    float z = 0.2f;
    constexpr size_t laneCount = 12;
    float outputValues[laneCount]{};
    const uint64_t outputShape[]{1, 3, 4};
    VernonAdValue inputValues[] = {
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 0, nullptr},
        {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &y, sizeof(y), 0, nullptr},
        {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &z, sizeof(z), 0, nullptr},
        {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, outputValues, sizeof(outputValues), 3, outputShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 4, {0, 0, 0, 0}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {0, 0, 0, 0}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernon::tests::completeCanonicalAutodiffInvocation(pipeline, {2, 1, 1}, inputs, outputs, &pullback),
              VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    for (float outputValue : outputValues)
        EXPECT_NEAR(outputValue, objective(x, y, z), 2e-5);

    float seedValues[laneCount];
    for (size_t lane = 0; lane < laneCount; ++lane)
        seedValues[lane] = static_cast<float>(lane + 1);
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedValues, sizeof(seedValues), 3, outputShape};
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
        EXPECT_NEAR(gradientStorage[index], 78.0 * finiteDifference(x, y, z, index), 3e-3);
    vernonPullbackDestroy(pullback);

    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
TEST(RuntimeStructuredScalarAutodiff, CanonicalTapeRetentionUsesRuntimeMemoryAccounting) {
    ASSERT_EQ(vernonRegisterStructuredScalarBalancedAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_SCALAR_BALANCED_AUTODIFF_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const nlohmann::json manifestJson = nlohmann::json::parse(manifest);
    ASSERT_EQ(manifestJson.at("variants").size(), 1u);
    ASSERT_FALSE(manifestJson.contains("autodiff"));
    const std::string bundleDirectory = manifestPath.parent_path().string();

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
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
    auto run = [&](VernonProgramExecutable *pipeline, VernonLaunchSize grid, auto &&verifyAfterForward,
                   auto &&verifyAfterBackward) {
        float x = 1.2f;
        float y = 0.7f;
        float z = 0.2f;
        const size_t workgroupCount = static_cast<size_t>(grid.x) * grid.y * grid.z;
        std::vector<float> output(workgroupCount * 6);
        const uint64_t outputShape[]{static_cast<uint64_t>(grid.z), static_cast<uint64_t>(grid.y) * 3,
                                     static_cast<uint64_t>(grid.x) * 2};
        VernonAdValue inputValues[]{
            {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), {}},
            {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &y, sizeof(y), {}},
            {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &z, sizeof(z), {}},
            {sizeof(VernonAdValue),
             {"output", 6},
             VERNON_DATA_F32,
             output.data(),
             output.size() * sizeof(float),
             3,
             outputShape},
        };
        VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
        VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
        VernonPullback *pullback = nullptr;
        ASSERT_EQ(vernon::tests::completeCanonicalAutodiffInvocation(pipeline, grid, inputs, outputs, &pullback),
                  VERNON_STATUS_OK)
            << lastError(context);
        ASSERT_NE(pullback, nullptr);
        for (float value : output)
            EXPECT_NEAR(value, objective(x, y, z), 2e-5);
        verifyAfterForward(pullback);

        std::vector<float> seeds(output.size(), 1.0f);
        VernonAdValue seed{sizeof(VernonAdValue),
                           {"output", 6},
                           VERNON_DATA_F32,
                           seeds.data(),
                           seeds.size() * sizeof(float),
                           3,
                           outputShape};
        VernonAdValueSet cotangents{sizeof(VernonAdValueSet), &seed, 1, {}};
        float gradientStorage[3]{};
        VernonAdValue gradientValues[]{
            {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientStorage[0], sizeof(float), {}},
            {sizeof(VernonAdValue), {"y", 1}, VERNON_DATA_F32, &gradientStorage[1], sizeof(float), {}},
            {sizeof(VernonAdValue), {"z", 1}, VERNON_DATA_F32, &gradientStorage[2], sizeof(float), {}},
        };
        VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientValues, std::size(gradientValues), {}};
        const VernonPullbackApplyOptions applyOptions{sizeof(VernonPullbackApplyOptions),
                                                      VERNON_PULLBACK_APPLY_OPTIONS_VERSION,
                                                      std::numeric_limits<uint64_t>::max(),
                                                      std::numeric_limits<uint64_t>::max(),
                                                      {}};
        ASSERT_EQ(vernonPullbackApplyWithOptions(pullback, &cotangents, &gradients, &applyOptions), VERNON_STATUS_OK)
            << lastError(context);
        for (unsigned index = 0; index < 3; ++index)
            EXPECT_NEAR(gradientStorage[index], static_cast<double>(output.size()) * finiteDifference(x, y, z, index),
                        0.004 * output.size());
        verifyAfterBackward(pullback);
        vernonPullbackDestroy(pullback);
    };

    VernonProgramExecutable *pipeline = nullptr;
    for (VernonLaunchSize grid : {VernonLaunchSize{512, 1, 1}, VernonLaunchSize{1024, 1, 1}}) {
        auto retainedPolicy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(
            std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());
        vernon::runtime::ad::setHostTapeMemoryPolicyForTesting(*context, retainedPolicy);
        pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
        ASSERT_NE(pipeline, nullptr) << lastError(context);
        run(
            pipeline, grid,
            [&](VernonPullback *pullback) {
                const vernon::runtime::AutodiffPullbackMemoryUsage usage =
                    vernon::runtime::autodiffPullbackMemoryUsage(pullback);
                EXPECT_GT(usage.logicalResidualBytes, 0u);
                EXPECT_GE(usage.residentBytes, usage.logicalResidualBytes);
                EXPECT_GT(usage.allocatedBytes, 0u);
                EXPECT_EQ(usage.peakTemporaryBytes, 0u);
                EXPECT_EQ(vernon::runtime::ad::hostTapeMemoryPolicyChargedBytesForTesting(*retainedPolicy),
                          usage.allocatedBytes);
            },
            [&](VernonPullback *pullback) {
                const vernon::runtime::AutodiffPullbackMemoryUsage usage =
                    vernon::runtime::autodiffPullbackMemoryUsage(pullback);
                EXPECT_GT(usage.allocatedBytes, 0u);
                EXPECT_EQ(usage.peakTemporaryBytes, 0u);
            });
        EXPECT_EQ(vernon::runtime::ad::hostTapeMemoryPolicyChargedBytesForTesting(*retainedPolicy), 0u);
        vernonRuntimeProgramExecutableDestroy(pipeline);
    }
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

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
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
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
        EXPECT_EQ(vernon::tests::completeCanonicalAutodiffInvocation(pipeline, {1, 1, 1}, inputs, outputs, &pullback),
                  VERNON_STATUS_OK)
            << lastError(context);
        EXPECT_NE(pullback, nullptr);
        if (!pullback)
            return metrics;
        EXPECT_FLOAT_EQ(output, (1.0f + 2.0f * count) * x);

        metrics.reset();
        float seedValue = 1.0f;
        VernonAdValue seed{
            sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &seedValue, sizeof(seedValue), 1, shape};
        VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
        float gradient = 0.0f;
        VernonAdValue gradientValue{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradient, sizeof(gradient), {}};
        VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradientValue, 1, {}};
        EXPECT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
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

    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeStructuredScalarAutodiff, DynamicTapeBudgetFailureDoesNotPublishGradients) {
    ASSERT_EQ(vernonRegisterStructuredDynamicAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_DYNAMIC_AUTODIFF_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    auto calibrationPolicy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(
        std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());
    vernon::runtime::ad::setHostTapeMemoryPolicyForTesting(*context, calibrationPolicy);
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    float x = 1.25f;
    int32_t count = 1;
    float output = -17.0f;
    const uint64_t shape[]{1};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), {}},
        {sizeof(VernonAdValue), {"count", 5}, VERNON_DATA_I32, &count, sizeof(count), {}},
        {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &output, sizeof(output), 1, shape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *calibrationPullback = nullptr;
    ASSERT_EQ(
        vernon::tests::completeCanonicalAutodiffInvocation(pipeline, {1, 1, 1}, inputs, outputs, &calibrationPullback),
        VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(calibrationPullback, nullptr);
    float calibrationGradient = 0.0f;
    float seedValue = 1.0f;
    VernonAdValue seed{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &seedValue, sizeof(seedValue), 1, shape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    VernonAdValue calibrationGradientValue{sizeof(VernonAdValue),       {"x", 1}, VERNON_DATA_F32, &calibrationGradient,
                                           sizeof(calibrationGradient), {}};
    VernonAdValueSet calibrationGradients{sizeof(VernonAdValueSet), &calibrationGradientValue, 1, {}};
    ASSERT_EQ(vernonPullbackApply(calibrationPullback, &seeds, &calibrationGradients), VERNON_STATUS_OK)
        << lastError(context);
    const size_t oneInvocationBytes = calibrationPolicy->usage().peakBytes;
    ASSERT_GT(oneInvocationBytes, 0u);
    vernonPullbackDestroy(calibrationPullback);
    EXPECT_EQ(vernon::runtime::ad::hostTapeMemoryPolicyChargedBytesForTesting(*calibrationPolicy), 0u);
    vernonRuntimeProgramExecutableDestroy(pipeline);

    auto boundedPolicy =
        std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(oneInvocationBytes, oneInvocationBytes);
    vernon::runtime::ad::setHostTapeMemoryPolicyForTesting(*context, boundedPolicy);
    pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    count = 1500;
    output = -31.0f;
    VernonPullback *rejectedPullback = nullptr;
    EXPECT_EQ(
        vernon::tests::completeCanonicalAutodiffInvocation(pipeline, {1, 1, 1}, inputs, outputs, &rejectedPullback),
        VERNON_STATUS_INVALID_ARGUMENT);
    const std::string rejectionError = lastError(context);
    EXPECT_NE(rejectionError.find("autodiff tape allocator host allocation failed"), std::string::npos)
        << rejectionError;
    EXPECT_NE(rejectionError.find("context limit"), std::string::npos) << rejectionError;
    EXPECT_EQ(rejectedPullback, nullptr);
    EXPECT_FLOAT_EQ(output, -31.0f);
    EXPECT_EQ(vernon::runtime::ad::hostTapeMemoryPolicyChargedBytesForTesting(*boundedPolicy), 0u);

    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
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
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    VernonAdValueMetadataView outputMetadata{sizeof(VernonAdValueMetadataView)};
    VernonAdValueMetadataView cotangentMetadata{sizeof(VernonAdValueMetadataView)};
    VernonAdValueMetadataView gradientMetadata{sizeof(VernonAdValueMetadataView)};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetAdOutputByIndex(pipeline, 0, &outputMetadata), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetAdCotangentByIndex(pipeline, 0, &cotangentMetadata), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetAdGradientByIndex(pipeline, 0, &gradientMetadata), VERNON_STATUS_OK);
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
    ASSERT_EQ(vernon::tests::completeCanonicalAutodiffInvocation(pipeline, {1, 1, 1}, inputs, outputs, &pullback),
              VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_DOUBLE_EQ(output, 3.75);

    double gradient = 0.0;
    double seed = 1.0;
    VernonAdValue cotangent{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F64, &seed, sizeof(seed), 1, shape};
    VernonAdValueSet cotangents{sizeof(VernonAdValueSet), &cotangent, 1, {}};
    VernonAdValue gradientValue{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F64, &gradient, sizeof(gradient), {}};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradientValue, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_DOUBLE_EQ(gradient, 4.0);
    seed = 2.0;
    ASSERT_EQ(vernonPullbackApply(pullback, &cotangents, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_DOUBLE_EQ(gradient, 8.0);

    vernonPullbackDestroy(pullback);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
