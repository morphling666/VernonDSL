#include "VernonRuntime.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
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
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);

    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryCount(pipeline, VERNON_PROGRAM_BOUNDARY_OUTPUT), 1u);
    VernonProgramParameterView outputBoundary{};
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(pipeline, VERNON_PROGRAM_BOUNDARY_OUTPUT, 0, &outputBoundary),
        VERNON_STATUS_OK);
    EXPECT_EQ(std::string_view(outputBoundary.name.data, outputBoundary.name.size), "output");
    VernonProgramValueLeafView outputLeaf{sizeof(VernonProgramValueLeafView)};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryValueLeaf(pipeline, VERNON_PROGRAM_BOUNDARY_OUTPUT,
                                                                 outputBoundary.slot, 0, &outputLeaf),
              VERNON_STATUS_OK);
    EXPECT_EQ(outputLeaf.value.dtype, VERNON_DATA_F32);
    ASSERT_EQ(outputBoundary.rank, 1u);
    EXPECT_EQ(outputBoundary.static_shape[0], 2u);
    EXPECT_EQ(outputLeaf.static_rank, 0u);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryCount(pipeline, VERNON_PROGRAM_BOUNDARY_COTANGENT), 1u);
    VernonProgramParameterView cotangentBoundary{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(pipeline, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0,
                                                               &cotangentBoundary),
              VERNON_STATUS_OK);
    EXPECT_EQ(std::string_view(cotangentBoundary.name.data, cotangentBoundary.name.size), "output");
    VernonProgramValueLeafView cotangentLeaf{sizeof(VernonProgramValueLeafView)};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryValueLeaf(pipeline, VERNON_PROGRAM_BOUNDARY_COTANGENT,
                                                                 cotangentBoundary.slot, 0, &cotangentLeaf),
              VERNON_STATUS_OK);
    EXPECT_EQ(cotangentLeaf.value.dtype, VERNON_DATA_F32);
    EXPECT_GE(vernonRuntimeProgramExecutableGetBoundaryCount(pipeline, VERNON_PROGRAM_BOUNDARY_GRADIENT), 1u);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(pipeline), 3u);
    VernonAdDerivativeGroupView group{sizeof(VernonAdDerivativeGroupView)};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(pipeline, 0, &group), VERNON_STATUS_OK);
    EXPECT_EQ(group.role, VERNON_AD_DERIVATIVE_GRADIENT);
    EXPECT_GE(group.leaf_count, 1u);
    VernonStringView groupLeaf{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(pipeline, 0, 0, &groupLeaf), VERNON_STATUS_OK);
    EXPECT_FALSE(std::string_view(groupLeaf.data, groupLeaf.size).empty());

    auto expectField = [](const VernonValuePathComponentView &component, std::string_view name) {
        EXPECT_EQ(component.kind, VERNON_VALUE_PATH_FIELD);
        EXPECT_EQ(std::string_view(component.field.data, component.field.size), name);
    };
    VernonProgramParameterView valueParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"value", 5}, &valueParameter), VERNON_STATUS_OK);
    EXPECT_EQ(valueParameter.rank, 0u);
    VernonProgramValueLeafView valueLeaf{sizeof(VernonProgramValueLeafView)};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterValueLeaf(pipeline, {"value", 5}, 0, &valueLeaf),
              VERNON_STATUS_OK);
    EXPECT_EQ(valueLeaf.value.dtype, VERNON_DATA_F32);
    EXPECT_EQ(valueLeaf.value.scalar_count, 2u);
    EXPECT_EQ(valueLeaf.value.byte_offset, 0u);
    ASSERT_EQ(valueLeaf.static_rank, 1u);
    EXPECT_EQ(valueLeaf.static_shape[0], 2u);
    EXPECT_EQ(valueLeaf.path_count, 0u);

    VernonProgramValueLeafView scaleLeaf{sizeof(VernonProgramValueLeafView)};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterValueLeaf(pipeline, {"parameters", 10}, 0, &scaleLeaf),
              VERNON_STATUS_OK);
    EXPECT_EQ(scaleLeaf.value.dtype, VERNON_DATA_F32);
    EXPECT_EQ(scaleLeaf.value.scalar_count, 1u);
    EXPECT_EQ(scaleLeaf.value.byte_offset, 0u);
    EXPECT_EQ(scaleLeaf.static_rank, 0u);
    ASSERT_EQ(scaleLeaf.path_count, 2u);
    expectField(scaleLeaf.path[0], "inner");
    expectField(scaleLeaf.path[1], "scale");

    VernonProgramValueLeafView biasLeaf{sizeof(VernonProgramValueLeafView)};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterValueLeaf(pipeline, {"parameters", 10}, 1, &biasLeaf),
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
    struct Parameters {
        float scale;
        float bias;
    } parameters{4.0f, 1.0f};
    float outputValues[2]{};
    VernonProgramParameterView parametersParameter{};
    VernonProgramParameterView outputParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"parameters", 10}, &parametersParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"output", 6}, &outputParameter), VERNON_STATUS_OK);
    const int64_t tensorStrides[]{sizeof(float)};
    auto argument = [&](const VernonProgramParameterView &parameter, void *data, size_t byteSize, uint32_t rank,
                        const uint64_t *shape, const int64_t *strides) {
        VernonProgramArgument result{};
        result.slot = parameter.slot;
        result.kind = VERNON_PROGRAM_TENSOR;
        result.tensor.struct_size = sizeof(VernonTensorView);
        result.tensor.storage = VERNON_TENSOR_HOST;
        result.tensor.host_data = data;
        result.tensor.element_layout = parameter.element_layout;
        result.tensor.access = parameter.access;
        result.tensor.rank = rank;
        result.tensor.shape = shape;
        result.tensor.byte_strides = strides;
        result.tensor.byte_size = byteSize;
        return result;
    };
    VernonProgramArgument arguments[]{
        argument(valueParameter, value, sizeof(value), 0, nullptr, nullptr),
        argument(parametersParameter, &parameters, sizeof(parameters), 0, nullptr, nullptr),
        argument(outputParameter, outputValues, sizeof(outputValues), 1, tensorShape, tensorStrides),
    };
    VernonPullback *rejectedPullback = reinterpret_cast<VernonPullback *>(uintptr_t{1});
    EXPECT_EQ(vernon::tests::completeCanonicalComputeInvocation(pipeline, arguments, std::size(arguments), {2, 1, 1},
                                                                &rejectedPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(rejectedPullback, nullptr);
    EXPECT_NE(lastError(context).find("dispatch grid axis 0 must equal 1"), std::string::npos);
    EXPECT_FLOAT_EQ(outputValues[0], 0.0f);
    EXPECT_FLOAT_EQ(outputValues[1], 0.0f);

    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(pipeline, arguments, std::size(arguments), {1, 1, 1},
                                                                &pullback),
              VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValues[0], 9.0f);
    EXPECT_FLOAT_EQ(outputValues[1], 13.0f);

    float valueSeeds[]{5.0f, 7.0f};
    VernonProgramParameterView cotangentParameter{};
    VernonProgramParameterView parametersGradientParameter{};
    VernonProgramParameterView valueGradientParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(pipeline, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0,
                                                               &cotangentParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindBoundary(pipeline, VERNON_PROGRAM_BOUNDARY_GRADIENT, {"parameters", 10},
                                                         &parametersGradientParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindBoundary(pipeline, VERNON_PROGRAM_BOUNDARY_GRADIENT, {"value", 5},
                                                         &valueGradientParameter),
              VERNON_STATUS_OK);
    Parameters parametersGradient{};
    float valueGradient[2]{};
    VernonProgramArgument derivativeArguments[]{
        argument(cotangentParameter, valueSeeds, sizeof(valueSeeds), 1, tensorShape, tensorStrides),
        argument(parametersGradientParameter, &parametersGradient, sizeof(parametersGradient), 0, nullptr, nullptr),
        argument(valueGradientParameter, valueGradient, sizeof(valueGradient), 0, nullptr, nullptr),
    };
    ASSERT_EQ(vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments)),
              VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(parametersGradient.bias, 12.0f);
    EXPECT_FLOAT_EQ(parametersGradient.scale, 31.0f);
    EXPECT_FLOAT_EQ(valueGradient[0], 20.0f);
    EXPECT_FLOAT_EQ(valueGradient[1], 28.0f);

    vernonProgramPullbackDestroy(pullback);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
