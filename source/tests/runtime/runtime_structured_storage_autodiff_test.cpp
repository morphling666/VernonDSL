#include "VernonRuntime.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/host_tape_test_hooks.h"
#include "runtime/autodiff/runtime_direct_autodiff.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

extern "C" VernonStatus vernonRegisterStructuredStorageAutodiffFixture(void);

namespace {

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data, error.size);
}

TEST(RuntimeStructuredStorageAutodiff, ExecutesDynamicIndexMutationAndFreshStorageGradient) {
    ASSERT_EQ(vernonRegisterStructuredStorageAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_STORAGE_AUTODIFF_MANIFEST;
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

    const uint64_t storageShape[]{1, 1, 1, 3};
    float values[]{2.0f, 3.0f, 5.0f};
    float source[]{7.0f, 6.0f, 9.0f};
    int32_t index = 1;
    float scale = 4.0f;
    float loss{};
    const uint64_t lossShape[]{1, 1, 1};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values, sizeof(values), 4, storageShape},
        {sizeof(VernonAdValue), {"source", 6}, VERNON_DATA_F32, source, sizeof(source), 4, storageShape},
        {sizeof(VernonAdValue), {"index", 5}, VERNON_DATA_I32, &index, sizeof(index), {}},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), {}},
        {sizeof(VernonAdValue), {"loss", 4}, VERNON_DATA_F32, &loss, sizeof(loss), 3, lossShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 5, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    inputValues[1].data = values;
    VernonPullback *rejectedPullback = nullptr;
    EXPECT_EQ(vernonAdProgramForward(pipeline, {1, 1, 1}, &inputs, &outputs, &rejectedPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(rejectedPullback, nullptr);
    inputValues[1].data = source;

    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdProgramForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(loss, 26.0f);
    EXPECT_FLOAT_EQ(values[0], 2.0f);
    EXPECT_FLOAT_EQ(values[1], 24.0f);
    EXPECT_FLOAT_EQ(values[2], 5.0f);
    source[1] = 100.0f;

    float scaleGradient{};
    float sourceGradient[3]{};
    float valuesGradient[3]{};
    VernonAdValue gradientValues[]{
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scaleGradient, sizeof(scaleGradient), {}},
        {sizeof(VernonAdValue),
         {"source", 6},
         VERNON_DATA_F32,
         sourceGradient,
         sizeof(sourceGradient),
         4,
         storageShape},
        {sizeof(VernonAdValue),
         {"values", 6},
         VERNON_DATA_F32,
         valuesGradient,
         sizeof(valuesGradient),
         4,
         storageShape},
    };
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientValues, 3, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(scaleGradient, 6.0f);
    EXPECT_FLOAT_EQ(sourceGradient[0], 0.0f);
    EXPECT_FLOAT_EQ(sourceGradient[1], 4.0f);
    EXPECT_FLOAT_EQ(sourceGradient[2], 0.0f);
    EXPECT_FLOAT_EQ(valuesGradient[0], 1.0f);
    EXPECT_FLOAT_EQ(valuesGradient[1], 0.0f);
    EXPECT_FLOAT_EQ(valuesGradient[2], 0.0f);

    vernonPullbackDestroy(pullback);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeStructuredStorageAutodiff, ReplayUsesRetainedCallerOwnedPrimalVersion) {
    ASSERT_EQ(vernonRegisterStructuredStorageAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_STORAGE_AUTODIFF_MANIFEST;
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

    const uint64_t storageShape[]{1, 1, 1, 3};
    const uint64_t lossShape[]{1, 1, 1};
    float values[]{2.0f, 3.0f, 5.0f};
    float source[]{7.0f, 6.0f, 9.0f};
    int32_t index = 1;
    float scale = 4.0f;
    float loss{};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values, sizeof(values), 4, storageShape},
        {sizeof(VernonAdValue), {"source", 6}, VERNON_DATA_F32, source, sizeof(source), 4, storageShape},
        {sizeof(VernonAdValue), {"index", 5}, VERNON_DATA_I32, &index, sizeof(index), {}},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), {}},
        {sizeof(VernonAdValue), {"loss", 4}, VERNON_DATA_F32, &loss, sizeof(loss), 3, lossShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdProgramForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    ASSERT_FLOAT_EQ(loss, 26.0f);

    source[1] = 100.0f;
    float scaleGradient{};
    float sourceGradient[3]{};
    float valuesGradient[3]{};
    VernonAdValue gradientValues[]{
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scaleGradient, sizeof(scaleGradient), {}},
        {sizeof(VernonAdValue),
         {"source", 6},
         VERNON_DATA_F32,
         sourceGradient,
         sizeof(sourceGradient),
         4,
         storageShape},
        {sizeof(VernonAdValue),
         {"values", 6},
         VERNON_DATA_F32,
         valuesGradient,
         sizeof(valuesGradient),
         4,
         storageShape},
    };
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientValues, std::size(gradientValues), {}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(scaleGradient, 6.0f);
    EXPECT_FLOAT_EQ(source[1], 100.0f);

    source[1] = -20.0f;
    scaleGradient = 0.0f;
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(scaleGradient, 6.0f);
    EXPECT_FLOAT_EQ(source[1], -20.0f);

    vernonPullbackDestroy(pullback);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
TEST(RuntimeStructuredStorageAutodiff, NoTapeProfileDoesNotReserveHostTape) {
    ASSERT_EQ(vernonRegisterStructuredStorageAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_STORAGE_AUTODIFF_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    auto tapePolicy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(0, 0);
    vernon::runtime::ad::setHostTapeMemoryPolicyForTesting(*context, tapePolicy);
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    const uint64_t storageShape[]{2, 1, 1, 3};
    const uint64_t lossShape[]{2, 1, 1};
    float values[]{2.0f, 3.0f, 5.0f, 11.0f, 13.0f, 17.0f};
    float source[]{7.0f, 6.0f, 9.0f, 19.0f, 23.0f, 29.0f};
    int32_t index = 1;
    float scale = 4.0f;
    float loss[]{-17.0f, -31.0f};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values, sizeof(values), 4, storageShape},
        {sizeof(VernonAdValue), {"source", 6}, VERNON_DATA_F32, source, sizeof(source), 4, storageShape},
        {sizeof(VernonAdValue), {"index", 5}, VERNON_DATA_I32, &index, sizeof(index), {}},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), {}},
        {sizeof(VernonAdValue), {"loss", 4}, VERNON_DATA_F32, loss, sizeof(loss), 3, lossShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, std::size(inputValues), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdProgramForward(pipeline, {2, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_EQ(vernon::runtime::ad::hostTapeMemoryPolicyChargedBytesForTesting(*tapePolicy), 0u);
    const vernon::runtime::AutodiffPullbackMemoryUsage memoryUsage =
        vernon::runtime::autodiffPullbackMemoryUsage(pullback);
    EXPECT_EQ(memoryUsage.logicalResidualBytes, 0u);
    EXPECT_GT(memoryUsage.retainedAllocationBytes, 0u);
    EXPECT_EQ(memoryUsage.residentBytes, 0u);
    EXPECT_EQ(memoryUsage.allocatedBytes, 0u);
    EXPECT_FLOAT_EQ(loss[0], 26.0f);
    EXPECT_FLOAT_EQ(loss[1], 103.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernon::runtime::ad::hostTapeMemoryPolicyChargedBytesForTesting(*tapePolicy), 0u);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}
#endif

} // namespace
