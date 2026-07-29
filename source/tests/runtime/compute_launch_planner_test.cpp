#include "runtime/compute_launch_planner.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>

namespace {

using namespace vernon::runtime;

void setScalarLayout(Parameter &parameter, const char *dtype, VernonDataType dataType) {
    const VernonValueLayoutView view = vernonRuntimeGetScalarValueLayout(dataType);
    parameter.elementLayout.logicalType = dtype;
    parameter.elementLayout.layoutHash.assign(view.layout_hash.data, view.layout_hash.size);
    parameter.elementLayout.byteSize = view.byte_size;
    parameter.elementLayout.alignment = view.alignment;
    parameter.elementLayout.leaves = {{dtype, 1, 0}};
    parameter.elementLayout.abiLeaves.assign(view.leaves, view.leaves + view.leaf_count);
}

TEST(ComputeLaunchPlannerTest, PlacesArgumentsDirectlyByReflectionIndex) {
    Variant variant;
    Parameter contiguous;
    contiguous.slot = 0;
    contiguous.kind = "tensor";
    setScalarLayout(contiguous, "f32", VERNON_DATA_F32);
    contiguous.source = "direct";
    contiguous.uses.push_back({"compute", "buffer", "", "f32", {2}, 1, UINT32_MAX, 0, 0, 0, {}});
    contiguous.uses.back().physicalValueLayout =
        PhysicalValueLayout{"vulkan_std430_storage_buffer", "storage_buffer", 8, 4, {4}};
    Parameter strided;
    strided.slot = 1;
    strided.kind = "tensor";
    setScalarLayout(strided, "f32", VERNON_DATA_F32);
    strided.source = "direct";
    strided.uses.push_back({"compute", "buffer", "", "f32", {2}, 0, UINT32_MAX, 0, 0, 1, {}});
    strided.uses.back().physicalValueLayout =
        PhysicalValueLayout{"vulkan_std430_storage_buffer", "storage_buffer", 8, 4, {4}};
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
                          vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
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
                          vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                          VERNON_ACCESS_READ,
                          1,
                          shape.data(),
                          stridedStride.data(),
                          0,
                          sizeof(stridedValues)};
    VernonPipelineInvocation invocation{};
    invocation.arguments = supplied;
    invocation.argument_count = 2;
    invocation.compute_grid = {4, 1, 1};

    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_TRUE(planComputeInvocation(variant, invocation, plan, error)) << error;
    ASSERT_EQ(plan.arguments.size(), 2u);
    ASSERT_EQ(plan.hostTensorStorage.size(), 2u);
    const std::array<float, 2> expectedPacked{1, 2};
    EXPECT_EQ(std::memcmp(plan.arguments[0].scalarData, expectedPacked.data(), sizeof(expectedPacked)), 0);
    EXPECT_EQ(std::memcmp(plan.arguments[1].scalarData, contiguousValues.data(), sizeof(contiguousValues)), 0);
    EXPECT_EQ(plan.grid.x, 4u);
    EXPECT_EQ(plan.grid.y, 1u);
    EXPECT_EQ(plan.grid.z, 1u);
}

TEST(ComputeLaunchPlannerTest, AcceptsValidatedStridedRhiTensorView) {
    Variant variant;
    Parameter parameter;
    parameter.slot = 0;
    parameter.kind = "tensor";
    setScalarLayout(parameter, "f32", VERNON_DATA_F32);
    parameter.source = "direct";
    parameter.uses.push_back({"compute", "buffer", "", "f32", {2, 3}, 0, UINT32_MAX, 0, 0, 0, {}});
    parameter.uses.back().elementStrides = {6, -1};
    parameter.uses.back().elementOffset = 2;
    variant.parameters = {parameter};

    const std::array<uint64_t, 2> shape{2, 3};
    const std::array<int64_t, 2> strides{6 * sizeof(float), -static_cast<int64_t>(sizeof(float))};
    const VernonRuntimeProviderResourceReference resource{1, {2}, 0, 12 * sizeof(float)};
    VernonPipelineArgument supplied{};
    supplied.slot = 0;
    supplied.kind = VERNON_PIPELINE_TENSOR;
    supplied.tensor.struct_size = sizeof(VernonTensorView);
    supplied.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    supplied.tensor.resource = resource;
    supplied.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    supplied.tensor.access = VERNON_ACCESS_READ;
    supplied.tensor.rank = 2;
    supplied.tensor.shape = shape.data();
    supplied.tensor.byte_strides = strides.data();
    supplied.tensor.byte_offset = 2 * sizeof(float);
    supplied.tensor.byte_size = 12 * sizeof(float);
    VernonPipelineInvocation invocation{};
    invocation.arguments = &supplied;
    invocation.argument_count = 1;
    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_TRUE(planComputeInvocation(variant, invocation, plan, error)) << error;
    ASSERT_EQ(plan.arguments.size(), 1u);
    EXPECT_EQ(plan.arguments[0].resource.identity, resource.identity);
    EXPECT_EQ(plan.arguments[0].resource.resource.value, resource.resource.value);
    EXPECT_EQ(plan.grid.x, 3u);
    EXPECT_EQ(plan.grid.y, 2u);
    EXPECT_EQ(plan.grid.z, 1u);

    const std::array<int64_t, 2> incompatibleStrides{6 * sizeof(float), sizeof(float)};
    supplied.tensor.byte_strides = incompatibleStrides.data();
    ASSERT_FALSE(planComputeInvocation(variant, invocation, plan, error));
    EXPECT_EQ(error, "pipeline TensorView layout does not match specialization");

    supplied.tensor.byte_strides = strides.data();
    supplied.tensor.byte_size = 6 * sizeof(float);
    ASSERT_FALSE(planComputeInvocation(variant, invocation, plan, error));
    EXPECT_EQ(error, "pipeline Tensor argument does not match layout");
}

} // namespace
