#include "VernonRuntime.h"
#include "runtime/autodiff/runtime_autodiff_telemetry.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

extern "C" VernonStatus vernonRegisterStructuredStorageWorkgroupAutodiffFixture(void);

namespace {

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data, error.size);
}

TEST(RuntimeWorkgroupAutodiff, ReplaysBarriersInReverseForEveryLaneGradient) {
    ASSERT_EQ(vernonRegisterStructuredStorageWorkgroupAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_STORAGE_WORKGROUP_AUTODIFF_MANIFEST;
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

    constexpr size_t laneCount = 8;
    float carriers[laneCount]{1.0f, 2.0f, 4.0f, 8.0f, 3.0f, 5.0f, 7.0f, 11.0f};
    float scale = 1.5f;
    float output[laneCount]{};
    const uint64_t vectorShape[]{laneCount};
    const uint64_t outputShape[]{1, 1, laneCount};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"carriers", 8}, VERNON_DATA_F32, carriers, sizeof(carriers), 1, vectorShape},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), 0, nullptr},
        {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, output, sizeof(output), 3, outputShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernon::tests::completeCanonicalAutodiffInvocation(pipeline, {2, 1, 1}, inputs, outputs, &pullback),
              VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    const vernon::runtime::AutodiffPullbackMemoryUsage memory = vernon::runtime::autodiffPullbackMemoryUsage(pullback);
    EXPECT_GT(memory.logicalResidualBytes, 0u);
    EXPECT_GT(memory.residentBytes, 0u);
    EXPECT_GT(memory.allocatedBytes, 0u);
    for (size_t lane = 0; lane < laneCount; ++lane) {
        const size_t groupBase = lane / 4 * 4;
        const size_t neighbor = groupBase + (lane + 1) % 4;
        EXPECT_FLOAT_EQ(output[lane], carriers[neighbor] * scale * 2.0f);
    }

    float seedValues[laneCount];
    const float weights[laneCount]{1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 11.0f, 13.0f, 17.0f};
    for (size_t lane = 0; lane < laneCount; ++lane)
        seedValues[lane] = weights[lane];
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedValues, sizeof(seedValues), 3, outputShape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    float carrierGradients[laneCount]{};
    float scaleGradient{};
    VernonAdValue gradientValues[]{
        {sizeof(VernonAdValue),
         {"carriers", 8},
         VERNON_DATA_F32,
         carrierGradients,
         sizeof(carrierGradients),
         1,
         vectorShape},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scaleGradient, sizeof(scaleGradient), 0, nullptr},
    };
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientValues, std::size(gradientValues), {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);

    float expectedScaleGradient = 0.0f;
    for (size_t lane = 0; lane < laneCount; ++lane) {
        const size_t groupBase = lane / 4 * 4;
        const size_t consumer = groupBase + (lane + 3) % 4;
        EXPECT_FLOAT_EQ(carrierGradients[lane], 2.0f * scale * weights[consumer]);
        const size_t neighbor = groupBase + (lane + 1) % 4;
        expectedScaleGradient += 2.0f * carriers[neighbor] * weights[lane];
    }
    EXPECT_FLOAT_EQ(scaleGradient, expectedScaleGradient);
    std::fill(std::begin(carrierGradients), std::end(carrierGradients), 0.0f);
    scaleGradient = 0.0f;
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    for (size_t lane = 0; lane < laneCount; ++lane) {
        const size_t groupBase = lane / 4 * 4;
        const size_t consumer = groupBase + (lane + 3) % 4;
        EXPECT_FLOAT_EQ(carrierGradients[lane], 2.0f * scale * weights[consumer]);
    }
    EXPECT_FLOAT_EQ(scaleGradient, expectedScaleGradient);

    vernonPullbackDestroy(pullback);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
