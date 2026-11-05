#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace {

using vernon::tests::HostTensor;
using vernon::tests::LinkedVjp;
using vernon::tests::ResolvedProgram;

struct ContextDeleter {
    void operator()(VernonRuntimeContext *context) const { EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK); }
};

TEST(RuntimeStructuredAggregateAutodiff, ExecutesAggregateInputAndStorageObjectivePullback) {
    const std::unique_ptr<VernonRuntimeContext, ContextDeleter> owner(
        vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr));
    VernonRuntimeContext *context = owner.get();
    ASSERT_NE(context, nullptr);
    const ResolvedProgram program(context, VERNON_STRUCTURED_AGGREGATE_AUTODIFF_MANIFEST);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(context);
    VernonProgramExecutable *pipeline = program.get();

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

    struct Parameters {
        float scale;
        float bias;
    };
    float value[]{2.0f, 3.0f};
    Parameters parameters{4.0f, 1.0f};
    HostTensor<float> output({0.0f, 0.0f});
    LinkedVjp vjp(context, pipeline);
    ASSERT_TRUE(vjp) << vjp.error();
    vjp.forwardRecord().bind("value", vernon::tests::hostView(value, sizeof(value)));
    vjp.forwardRecord().bind("parameters", vernon::tests::hostView(&parameters, sizeof(parameters)));
    vjp.forwardRecord().bind("output", output.view());
    vjp.forwardRecord().bindGrid(1);
    ASSERT_TRUE(vjp.forwardRecord().unbound().empty()) << vjp.forwardRecord().unbound().front();
    ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_FLOAT_EQ(output.values[0], 9.0f);
    EXPECT_FLOAT_EQ(output.values[1], 13.0f);

    HostTensor<float> seeds({5.0f, 7.0f});
    Parameters parametersGradient{};
    float valueGradient[2]{};
    vjp.backwardRecord().bind("output", seeds.view(), VERNON_PROGRAM_BOUNDARY_COTANGENT);
    vjp.backwardRecord().bind("parameters", vernon::tests::hostView(&parametersGradient, sizeof(parametersGradient)),
                              VERNON_PROGRAM_BOUNDARY_GRADIENT);
    vjp.backwardRecord().bind("value", vernon::tests::hostView(valueGradient, sizeof(valueGradient)),
                              VERNON_PROGRAM_BOUNDARY_GRADIENT);
    ASSERT_TRUE(vjp.backwardRecord().unbound().empty());
    ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_FLOAT_EQ(parametersGradient.bias, 12.0f);
    EXPECT_FLOAT_EQ(parametersGradient.scale, 31.0f);
    EXPECT_FLOAT_EQ(valueGradient[0], 20.0f);
    EXPECT_FLOAT_EQ(valueGradient[1], 28.0f);
}

} // namespace
