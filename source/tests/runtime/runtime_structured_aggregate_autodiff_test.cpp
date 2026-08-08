#include "VernonRuntime.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

extern "C" VernonStatus vernonRegisterStructuredAggregateAutodiffFixture(void);

namespace {

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data, error.size);
}

TEST(RuntimeStructuredAggregateAutodiff, ExecutesAggregateInputAndStorageObjectivePullback) {
    ASSERT_EQ(vernonRegisterStructuredAggregateAutodiffFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_STRUCTURED_AGGREGATE_AUTODIFF_MANIFEST;
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

    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputCount(pipeline), 1u);
    VernonAdValueMetadataView outputMetadata{sizeof(VernonAdValueMetadataView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputByIndex(pipeline, 0, &outputMetadata), VERNON_STATUS_OK);
    EXPECT_EQ(std::string_view(outputMetadata.path.data, outputMetadata.path.size), "output");
    EXPECT_EQ(outputMetadata.dtype, VERNON_DATA_F32);
    ASSERT_EQ(outputMetadata.rank, 1u);
    EXPECT_EQ(outputMetadata.shape[0], 2u);
    EXPECT_EQ(vernonRuntimeLoadedPipelineGetAdCotangentCount(pipeline), 1u);
    VernonAdValueMetadataView cotangentMetadata{sizeof(VernonAdValueMetadataView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdCotangentByIndex(pipeline, 0, &cotangentMetadata), VERNON_STATUS_OK);
    EXPECT_EQ(std::string_view(cotangentMetadata.path.data, cotangentMetadata.path.size), "output");
    EXPECT_EQ(cotangentMetadata.dtype, VERNON_DATA_F32);
    EXPECT_EQ(vernonRuntimeLoadedPipelineGetAdGradientCount(pipeline), 3u);
    VernonAdValueMetadataView gradientMetadata{sizeof(VernonAdValueMetadataView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradientByIndex(pipeline, 0, &gradientMetadata), VERNON_STATUS_OK);
    EXPECT_FALSE(std::string_view(gradientMetadata.path.data, gradientMetadata.path.size).empty());
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdDerivativeGroupCount(pipeline), 4u);
    VernonAdDerivativeGroupView group{sizeof(VernonAdDerivativeGroupView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdDerivativeGroupByIndex(pipeline, 0, &group), VERNON_STATUS_OK);
    EXPECT_EQ(group.role, VERNON_AD_DERIVATIVE_GRADIENT);
    EXPECT_GE(group.leaf_count, 1u);
    VernonStringView groupLeaf{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdDerivativeGroupLeaf(pipeline, 0, 0, &groupLeaf), VERNON_STATUS_OK);
    EXPECT_FALSE(std::string_view(groupLeaf.data, groupLeaf.size).empty());

    auto expectField = [](const VernonValuePathComponentView &component, std::string_view name) {
        EXPECT_EQ(component.kind, VERNON_VALUE_PATH_FIELD);
        EXPECT_EQ(std::string_view(component.field.data, component.field.size), name);
    };
    VernonPipelineParameterView valueParameter{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"value", 5}, &valueParameter), VERNON_STATUS_OK);
    ASSERT_EQ(valueParameter.rank, 1u);
    EXPECT_EQ(valueParameter.static_shape[0], 2u);
    VernonPipelineValueLeafView valueLeaf{sizeof(VernonPipelineValueLeafView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterValueLeaf(pipeline, {"value", 5}, 0, &valueLeaf),
              VERNON_STATUS_OK);
    EXPECT_EQ(valueLeaf.value.dtype, VERNON_DATA_F32);
    EXPECT_EQ(valueLeaf.value.scalar_count, 1u);
    EXPECT_EQ(valueLeaf.value.byte_offset, 0u);
    EXPECT_EQ(valueLeaf.static_rank, 0u);
    EXPECT_EQ(valueLeaf.path_count, 0u);

    VernonPipelineValueLeafView scaleLeaf{sizeof(VernonPipelineValueLeafView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterValueLeaf(pipeline, {"parameters", 10}, 0, &scaleLeaf),
              VERNON_STATUS_OK);
    EXPECT_EQ(scaleLeaf.value.dtype, VERNON_DATA_F32);
    EXPECT_EQ(scaleLeaf.value.scalar_count, 1u);
    EXPECT_EQ(scaleLeaf.value.byte_offset, 0u);
    EXPECT_EQ(scaleLeaf.static_rank, 0u);
    ASSERT_EQ(scaleLeaf.path_count, 2u);
    expectField(scaleLeaf.path[0], "inner");
    expectField(scaleLeaf.path[1], "scale");

    VernonPipelineValueLeafView biasLeaf{sizeof(VernonPipelineValueLeafView)};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterValueLeaf(pipeline, {"parameters", 10}, 1, &biasLeaf),
              VERNON_STATUS_OK);
    EXPECT_EQ(biasLeaf.value.dtype, VERNON_DATA_F32);
    EXPECT_EQ(biasLeaf.value.scalar_count, 1u);
    EXPECT_EQ(biasLeaf.value.byte_offset, 4u);
    EXPECT_EQ(biasLeaf.static_rank, 0u);
    ASSERT_EQ(biasLeaf.path_count, 2u);
    expectField(biasLeaf.path[0], "inner");
    expectField(biasLeaf.path[1], "bias");

    const uint64_t tensorShape[]{2};
    float value[]{2.0f, 3.0f};
    float scale = 4.0f;
    float bias = 1.0f;
    float outputValues[2]{};
    VernonAdValue inputValues[]{
        {sizeof(VernonAdValue), {"value", 5}, VERNON_DATA_F32, value, sizeof(value), 1, tensorShape},
        {sizeof(VernonAdValue), {"parameters.inner.scale", 22}, VERNON_DATA_F32, &scale, sizeof(scale), {}},
        {sizeof(VernonAdValue), {"parameters.inner.bias", 21}, VERNON_DATA_F32, &bias, sizeof(bias), {}},
        {sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, outputValues, sizeof(outputValues), 1, tensorShape},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues, 4, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValues[0], 9.0f);
    EXPECT_FLOAT_EQ(outputValues[1], 13.0f);

    float valueSeeds[]{5.0f, 7.0f};
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, valueSeeds, sizeof(valueSeeds), 1, tensorShape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    float biasGradient{};
    float scaleGradient{};
    float valueGradient[2]{};
    VernonAdValue gradientLeaves[]{
        {sizeof(VernonAdValue),
         {"parameters.inner.bias", 21},
         VERNON_DATA_F32,
         &biasGradient,
         sizeof(biasGradient),
         {}},
        {sizeof(VernonAdValue),
         {"parameters.inner.scale", 22},
         VERNON_DATA_F32,
         &scaleGradient,
         sizeof(scaleGradient),
         {}},
        {sizeof(VernonAdValue), {"value", 5}, VERNON_DATA_F32, valueGradient, sizeof(valueGradient), 1, tensorShape},
    };
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), gradientLeaves, 3, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(biasGradient, 12.0f);
    EXPECT_FLOAT_EQ(scaleGradient, 31.0f);
    EXPECT_FLOAT_EQ(valueGradient[0], 20.0f);
    EXPECT_FLOAT_EQ(valueGradient[1], 28.0f);

    vernonPullbackDestroy(pullback);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
