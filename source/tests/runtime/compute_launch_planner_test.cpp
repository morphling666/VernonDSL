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

ParameterUse packedF32Use(uint32_t index, const std::string &layoutHash) {
    ParameterUse use;
    use.stage = "compute";
    use.interfaceKind = "value";
    use.dtype = "f32";
    use.shape = {2};
    use.index = index;
    use.transport = "storage_buffer";
    TransportNode scalar{TransportNodeKind::Scalar, "f32", 0, 4, 4};
    TransportNode array{TransportNodeKind::Array, "", 0, 8, 4, {2}, {4}, {std::move(scalar)}};
    InterfacePlan plan;
    plan.kind = InterfacePlanKind::ByteTransport;
    plan.profile = "vulkan_std430_storage_buffer";
    plan.canonicalLayoutHash = layoutHash;
    plan.root = std::move(array);
    use.interfacePlan = std::move(plan);
    return use;
}

Parameter storageF32Parameter(uint32_t slot, uint32_t index, const char *access) {
    Parameter parameter;
    parameter.slot = slot;
    parameter.kind = "tensor";
    parameter.source = "direct";
    parameter.access = access;
    setScalarLayout(parameter, "f32", VERNON_DATA_F32);
    ParameterUse use;
    use.stage = "compute";
    use.interfaceKind = "storage";
    use.index = index;
    parameter.uses.push_back(std::move(use));
    return parameter;
}

TEST(ComputeLaunchPlannerTest, PlacesArgumentsDirectlyByReflectionIndex) {
    Variant variant;
    Parameter contiguous;
    contiguous.slot = 0;
    contiguous.kind = "tensor";
    setScalarLayout(contiguous, "f32", VERNON_DATA_F32);
    contiguous.source = "direct";
    contiguous.uses.push_back(packedF32Use(2, contiguous.elementLayout.layoutHash));
    Parameter strided;
    strided.slot = 1;
    strided.kind = "tensor";
    setScalarLayout(strided, "f32", VERNON_DATA_F32);
    strided.source = "direct";
    strided.uses.push_back(packedF32Use(0, strided.elementLayout.layoutHash));
    variant.parameters = {contiguous, strided};

    const std::array<float, 2> contiguousValues{3, 4};
    const std::array<float, 3> stridedValues{1, -1, 2};
    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> contiguousStride{sizeof(float)};
    const std::array<int64_t, 1> stridedStride{2 * sizeof(float)};
    VernonProgramArgument supplied[2]{};
    supplied[0].slot = 0;
    supplied[0].kind = VERNON_PROGRAM_TENSOR;
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
    supplied[1].kind = VERNON_PROGRAM_TENSOR;
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
    VernonProgramSubmitDescriptor invocation{};
    invocation.arguments = supplied;
    invocation.argument_count = 2;
    invocation.compute_grid = {4, 1, 1};

    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_TRUE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error)) << error;
    ASSERT_EQ(plan.arguments.size(), 3u);
    ASSERT_EQ(plan.hostTensorStorage.size(), 2u);
    const std::array<float, 2> expectedPacked{1, 2};
    const auto &first = std::get<ComputeScalarArgument>(plan.arguments[0]);
    EXPECT_EQ(std::memcmp(first.data, expectedPacked.data(), sizeof(expectedPacked)), 0);
    EXPECT_TRUE(std::holds_alternative<ComputeTensorArgument>(plan.arguments[1]));
    const auto &third = std::get<ComputeScalarArgument>(plan.arguments[2]);
    EXPECT_EQ(std::memcmp(third.data, contiguousValues.data(), sizeof(contiguousValues)), 0);
    EXPECT_EQ(plan.grid.x, 4u);
    EXPECT_EQ(plan.grid.y, 1u);
    EXPECT_EQ(plan.grid.z, 1u);
}

TEST(ComputeLaunchPlannerTest, ReusesTensorViewArtifactAcrossDispatchLayouts) {
    Variant variant;
    Parameter parameter;
    parameter.slot = 0;
    parameter.kind = "tensor";
    setScalarLayout(parameter, "f32", VERNON_DATA_F32);
    parameter.source = "direct";
    parameter.access = "read";
    parameter.uses.push_back({"compute", "buffer", "", "f32", {2, 0}, 0, UINT32_MAX, 0, 0, 0, {}});
    parameter.uses.back().tensorViewDescriptor = TensorViewDescriptorUse{2, 1, {2, 3}, {4, 5}};
    variant.parameters = {parameter};

    const std::array<uint64_t, 2> shape{2, 3};
    const std::array<int64_t, 2> strides{6 * sizeof(float), -static_cast<int64_t>(sizeof(float))};
    const VernonRuntimeProviderResourceReference resource{1, {2}, 0, 12 * sizeof(float)};
    VernonProgramArgument supplied{};
    supplied.slot = 0;
    supplied.kind = VERNON_PROGRAM_TENSOR;
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
    VernonProgramSubmitDescriptor invocation{};
    invocation.arguments = &supplied;
    invocation.argument_count = 1;
    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_TRUE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error)) << error;
    ASSERT_EQ(plan.arguments.size(), 1u);
    const auto &firstPlan = std::get<ComputeTensorArgument>(plan.arguments[0]);
    EXPECT_EQ(firstPlan.resource.identity, resource.identity);
    EXPECT_EQ(firstPlan.resource.resource.value, resource.resource.value);
    EXPECT_EQ(plan.grid.x, 3u);
    EXPECT_EQ(plan.grid.y, 2u);
    EXPECT_EQ(plan.grid.z, 1u);

    const std::array<uint64_t, 2> secondShape{2, 4};
    const std::array<int64_t, 2> secondStrides{4 * sizeof(float), sizeof(float)};
    supplied.tensor.shape = secondShape.data();
    supplied.tensor.byte_strides = secondStrides.data();
    supplied.tensor.byte_offset = 0;
    ASSERT_TRUE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error)) << error;
    ASSERT_TRUE(std::get<ComputeTensorArgument>(plan.arguments[0]).tensorView);
    EXPECT_EQ(*computeBindingDescriptorValue(plan.arguments[0], {ComputeBindingSourceKind::TensorExtent, 0, 1}), 4);
    EXPECT_EQ(*computeBindingDescriptorValue(plan.arguments[0], {ComputeBindingSourceKind::TensorStride, 0, 0}), 4);

    const std::array<uint64_t, 2> invalidStaticShape{3, 4};
    supplied.tensor.shape = invalidStaticShape.data();
    ASSERT_FALSE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error));
    EXPECT_EQ(error, "pipeline TensorView descriptor violates static shape or element stride");

    supplied.tensor.shape = shape.data();
    supplied.tensor.byte_strides = strides.data();
    supplied.tensor.byte_offset = 2 * sizeof(float);
    supplied.tensor.byte_size = 6 * sizeof(float);
    ASSERT_FALSE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error));
    EXPECT_EQ(error, "pipeline Tensor argument '' at byte offset 8 requires 36 bytes but its allocation has 24");
}

TEST(ComputeLaunchPlannerTest, PacksRankZeroTensorViewDescriptor) {
    Variant variant;
    Parameter parameter = storageF32Parameter(0, 0, "read_write");
    parameter.shape = {};
    parameter.uses.front().shape = {};
    parameter.uses.front().tensorViewDescriptor = TensorViewDescriptorUse{0, 1, {}, {}};
    variant.parameters = {parameter};

    VernonProgramArgument supplied{};
    supplied.slot = 0;
    supplied.kind = VERNON_PROGRAM_TENSOR;
    supplied.tensor.struct_size = sizeof(VernonTensorView);
    supplied.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    supplied.tensor.resource = {1, {2}, 0, sizeof(float)};
    supplied.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    supplied.tensor.access = VERNON_ACCESS_READ_WRITE;
    supplied.tensor.rank = 0;
    supplied.tensor.byte_size = sizeof(float);
    VernonProgramSubmitDescriptor invocation{};
    invocation.arguments = &supplied;
    invocation.argument_count = 1;
    invocation.compute_grid = {1, 1, 1};

    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_TRUE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error)) << error;
    ASSERT_EQ(plan.arguments.size(), 1u);
    const auto &argument = std::get<ComputeTensorArgument>(plan.arguments.front());
    ASSERT_NE(argument.tensorView, nullptr);
    EXPECT_EQ(argument.tensorView->rank, 0u);
    EXPECT_EQ(argument.tensorViewSize, 16u);
}

TEST(ComputeLaunchPlannerTest, RejectsTensorViewAccessMismatchBeforeDispatch) {
    Variant variant;
    Parameter parameter;
    parameter.slot = 0;
    parameter.kind = "tensor";
    parameter.source = "direct";
    parameter.access = "read";
    setScalarLayout(parameter, "f32", VERNON_DATA_F32);
    parameter.uses.push_back({"compute", "buffer", "", "f32", {2}, 0, UINT32_MAX, 0, 0, 0, {}});
    parameter.uses.back().tensorViewDescriptor = TensorViewDescriptorUse{1, 1, {2}, {3}};
    variant.parameters = {parameter};

    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> strides{sizeof(float)};
    VernonProgramArgument supplied{};
    supplied.slot = 0;
    supplied.kind = VERNON_PROGRAM_TENSOR;
    supplied.tensor.struct_size = sizeof(VernonTensorView);
    supplied.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    supplied.tensor.resource = {1, {2}, 0, 2 * sizeof(float)};
    supplied.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    supplied.tensor.access = VERNON_ACCESS_WRITE;
    supplied.tensor.rank = 1;
    supplied.tensor.shape = shape.data();
    supplied.tensor.byte_strides = strides.data();
    supplied.tensor.byte_size = 2 * sizeof(float);
    VernonProgramSubmitDescriptor invocation{};
    invocation.arguments = &supplied;
    invocation.argument_count = 1;

    PlannedComputeLaunch plan;
    std::string error;
    ASSERT_FALSE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error));
    EXPECT_EQ(error, "pipeline Tensor argument access does not match reflection");
}

TEST(ComputeLaunchPlannerTest, EnforcesInjectiveAndPairwisePhysicalTensorAliases) {
    Variant variant;
    variant.parameters = {storageF32Parameter(0, 0, "write")};

    const std::array<float, 8> values{};
    const std::array<uint64_t, 1> shape{2};
    const std::array<int64_t, 1> zeroStride{0};
    VernonProgramArgument supplied[2]{};
    supplied[0].slot = 0;
    supplied[0].kind = VERNON_PROGRAM_TENSOR;
    supplied[0].tensor = {sizeof(VernonTensorView),
                          VERNON_TENSOR_HOST,
                          {values.data()},
                          vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32),
                          VERNON_ACCESS_WRITE,
                          1,
                          shape.data(),
                          zeroStride.data(),
                          0,
                          sizeof(values)};
    VernonProgramSubmitDescriptor invocation{};
    invocation.arguments = supplied;
    invocation.argument_count = 1;
    invocation.compute_grid = {1, 1, 1};
    PlannedComputeLaunch plan;
    std::string error;
    EXPECT_FALSE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error));
    EXPECT_EQ(error, "writable pipeline Tensor argument must have an injective byte layout");

    const std::array<int64_t, 1> sparseStride{2 * sizeof(float)};
    supplied[0].tensor.byte_strides = sparseStride.data();
    supplied[0].tensor.access = VERNON_ACCESS_READ;
    supplied[1] = supplied[0];
    supplied[1].slot = 1;
    supplied[1].tensor.access = VERNON_ACCESS_WRITE;
    supplied[1].tensor.byte_offset = sizeof(float);
    variant.parameters = {storageF32Parameter(0, 0, "read"), storageF32Parameter(1, 1, "write")};
    invocation.argument_count = 2;
    ASSERT_TRUE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error)) << error;

    supplied[1].tensor.byte_offset = 2;
    EXPECT_FALSE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error));
    EXPECT_EQ(error, "pipeline Tensor arguments have incompatible physical overlap");

    supplied[0].tensor.access = VERNON_ACCESS_READ;
    supplied[1].tensor.access = VERNON_ACCESS_READ;
    variant.parameters[1].access = "read";
    EXPECT_TRUE(planComputeInvocation(variant, {1, 1, 1}, invocation, plan, error)) << error;
}

} // namespace
