#include "VernonRuntime.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
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
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    const uint64_t storageShape[]{3};
    float values[]{2.0f, 3.0f, 5.0f};
    float source[]{7.0f, 6.0f, 9.0f};
    int32_t index = 1;
    float scale = 4.0f;
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values, sizeof(values), 1, storageShape},
        {sizeof(VernonAdValue), {"source", 6}, VERNON_DATA_F32, source, sizeof(source), 1, storageShape},
        {sizeof(VernonAdValue), {"index", 5}, VERNON_DATA_I32, &index, sizeof(index), {}},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), {}},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 4, {}};
    float outputValue{};
    VernonAdValue output{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    inputValues[1].data = values;
    VernonPullback *rejectedPullback = nullptr;
    EXPECT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &rejectedPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(rejectedPullback, nullptr);
    inputValues[1].data = source;

    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, 26.0f);
    EXPECT_FLOAT_EQ(values[0], 2.0f);
    EXPECT_FLOAT_EQ(values[1], 24.0f);
    EXPECT_FLOAT_EQ(values[2], 5.0f);

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
         1,
         storageShape},
        {sizeof(VernonAdValue),
         {"values", 6},
         VERNON_DATA_F32,
         valuesGradient,
         sizeof(valuesGradient),
         1,
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
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
