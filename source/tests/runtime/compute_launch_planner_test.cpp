#include "runtime/compute_launch_planner.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>

namespace {

using namespace vernon::runtime;

TEST(ComputeLaunchPlannerTest, PlacesArgumentsDirectlyByReflectionIndex) {
    Variant variant;
    Parameter contiguous;
    contiguous.slot = 0;
    contiguous.kind = "tensor";
    contiguous.uses.push_back({"compute", "buffer", "", "f32", {2}, 1, UINT32_MAX, 0, 0, 0, {}});
    Parameter strided;
    strided.slot = 1;
    strided.kind = "tensor";
    strided.uses.push_back({"compute", "buffer", "", "f32", {2}, 0, UINT32_MAX, 0, 0, 1, {}});
    variant.parameters = {contiguous, strided};

    const std::array<float, 2> contiguousValues{3, 4};
    const std::array<float, 3> stridedValues{1, -1, 2};
    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> contiguousStride{sizeof(float)};
    const std::array<int64_t, 1> stridedStride{2 * sizeof(float)};
    VernonPipelineArgument supplied[2]{};
    supplied[0].slot = 0;
    supplied[0].kind = VERNON_PIPELINE_TENSOR;
    supplied[0].tensor = {sizeof(VernonTensorView),
                          VERNON_TENSOR_HOST,
                          {contiguousValues.data()},
                          VERNON_DATA_F32,
                          VERNON_ACCESS_READ,
                          1,
                          shape.data(),
                          contiguousStride.data(),
                          0,
                          sizeof(contiguousValues)};
    supplied[1].slot = 1;
    supplied[1].kind = VERNON_PIPELINE_TENSOR;
    supplied[1].tensor = {sizeof(VernonTensorView),
                          VERNON_TENSOR_HOST,
                          {stridedValues.data()},
                          VERNON_DATA_F32,
                          VERNON_ACCESS_READ,
                          1,
                          shape.data(),
                          stridedStride.data(),
                          0,
                          sizeof(stridedValues)};
    const ComputeArgumentMap arguments{{0, &supplied[0]}, {1, &supplied[1]}};
    VernonPipelineInvocation invocation{};
    invocation.compute_grid = {4, 1, 1};

    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_TRUE(planComputeLaunch(variant, arguments, invocation, nullptr, {}, plan, error)) << error;
    ASSERT_EQ(plan.arguments.size(), 2u);
    ASSERT_EQ(plan.hostTensorStorage.size(), 1u);
    const std::array<float, 2> expectedPacked{1, 2};
    EXPECT_EQ(std::memcmp(plan.arguments[0].scalar_data, expectedPacked.data(), sizeof(expectedPacked)), 0);
    EXPECT_EQ(plan.arguments[1].scalar_data, contiguousValues.data());
    EXPECT_EQ(plan.grid.x, 4u);
    EXPECT_EQ(plan.grid.y, 1u);
    EXPECT_EQ(plan.grid.z, 1u);
}

TEST(ComputeLaunchPlannerTest, AcceptsValidatedStridedDeviceTensorView) {
    Variant variant;
    Parameter parameter;
    parameter.slot = 0;
    parameter.kind = "tensor";
    parameter.uses.push_back({"compute", "buffer", "", "f32", {2, 3}, 0, UINT32_MAX, 0, 0, 0, {}});
    variant.parameters = {parameter};

    const std::array<uint64_t, 2> shape{2, 3};
    const std::array<int64_t, 2> strides{6 * sizeof(float), -static_cast<int64_t>(sizeof(float))};
    auto *buffer = reinterpret_cast<VernonDeviceBuffer *>(uintptr_t{1});
    VernonPipelineArgument supplied{};
    supplied.slot = 0;
    supplied.kind = VERNON_PIPELINE_TENSOR;
    supplied.tensor = {sizeof(VernonTensorView),
                       VERNON_TENSOR_DEVICE,
                       {buffer},
                       VERNON_DATA_F32,
                       VERNON_ACCESS_READ,
                       2,
                       shape.data(),
                       strides.data(),
                       2 * sizeof(float),
                       12 * sizeof(float)};
    const ComputeArgumentMap arguments{{0, &supplied}};
    VernonPipelineInvocation invocation{};
    int context = 0;
    ComputePlannerCallbacks callbacks{
        &context,
        [](const void *userData, const VernonDeviceBuffer *) -> const void * { return userData; },
    };

    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_TRUE(planComputeLaunch(variant, arguments, invocation, &context, callbacks, plan, error)) << error;
    ASSERT_EQ(plan.arguments.size(), 1u);
    EXPECT_EQ(plan.arguments[0].buffer, buffer);
    EXPECT_EQ(plan.grid.x, 3u);
    EXPECT_EQ(plan.grid.y, 2u);
    EXPECT_EQ(plan.grid.z, 1u);
}

} // namespace
