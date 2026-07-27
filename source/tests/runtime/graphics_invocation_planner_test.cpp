#include "runtime/graphics_invocation_planner.h"
#include "runtime/pipeline_metadata.h"

#include <gtest/gtest.h>

#include <array>
namespace {

using namespace vernon::runtime;

bool planVertexTensor(const char *dtype, VernonDataType dataType, const std::vector<uint64_t> &valueShape,
                      const std::vector<AttributeLeaf> &leaves, const std::vector<uint64_t> &tensorShape,
                      const std::vector<int64_t> &tensorStrides, uint32_t divisor, uint32_t instanceCount,
                      PlannedGraphicsInvocation &plan, std::string &error) {
    Variant variant;
    variant.vertex = "vertex";
    Parameter parameter;
    parameter.name = "value";
    parameter.kind = "tensor";
    const VernonValueLayoutView scalarLayout = vernonRuntimeGetScalarValueLayout(dataType);
    parameter.elementLayout.logicalType = dtype;
    parameter.elementLayout.layoutHash = scalarLayout.layout_hash.data
                                             ? std::string(scalarLayout.layout_hash.data, scalarLayout.layout_hash.size)
                                             : "test-layout";
    parameter.elementLayout.byteSize = scalarLayout.byte_size ? scalarLayout.byte_size : tensorStrides.back();
    parameter.elementLayout.alignment = parameter.elementLayout.byteSize;
    parameter.elementLayout.leaves = {{dtype, 1, 0}};
    parameter.elementLayout.abiLeaves = {
        {static_cast<uint32_t>(dataType), 1, 0},
    };
    parameter.shape = valueShape;
    ParameterUse use;
    use.stage = "vertex";
    use.interfaceKind = "input";
    use.dtype = dtype;
    use.shape = valueShape;
    use.location = 2;
    use.divisor = divisor;
    use.attributeLeaves = leaves;
    parameter.uses.push_back(std::move(use));
    variant.parameters.push_back(std::move(parameter));

    VernonPipelineArgument argument{};
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = {1, {2}, 0, 4096};
    argument.tensor.element_layout = pipelineValueLayout(variant.parameters.front().elementLayout);
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = static_cast<uint32_t>(tensorShape.size());
    argument.tensor.shape = tensorShape.data();
    argument.tensor.byte_strides = tensorStrides.data();
    argument.tensor.byte_size = 4096;

    VernonColorAttachment attachment{};
    attachment.resource = {1, {3}, 0, 1024};
    attachment.width = 16;
    attachment.height = 16;
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.vertex_count = divisor ? 3 : 0;
    invocation.instance_count = instanceCount;
    return planGraphicsInvocation(variant, invocation, plan, error);
}

TEST(GraphicsInvocationPlanner, PlansSortedTargetsPairingResolutionCountsAndIndex) {
    const VernonRuntimeProviderResourceReference vertexBuffer{1, {11}, 0, 48};
    const VernonRuntimeProviderResourceReference indexBuffer{2, {12}, 0, 24};
    const VernonRuntimeProviderResourceReference sampledTexture{3, {13}, 0, 0};
    const VernonRuntimeProviderResourceReference firstTarget{4, {14}, 0, 0};
    const VernonRuntimeProviderResourceReference secondTarget{5, {15}, 0, 0};
    const VernonRuntimeProviderResourceReference sampler{6, {16}, 0, 0};

    Variant variant;
    variant.vertex = "vertex";
    variant.fragment = "fragment";
    Parameter vertices;
    vertices.slot = 0;
    vertices.name = "vertices";
    vertices.kind = "tensor";
    const VernonValueLayoutView f32Layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    vertices.elementLayout.logicalType = "f32";
    vertices.elementLayout.layoutHash.assign(f32Layout.layout_hash.data, f32Layout.layout_hash.size);
    vertices.elementLayout.byteSize = f32Layout.byte_size;
    vertices.elementLayout.alignment = f32Layout.alignment;
    vertices.elementLayout.leaves = {{"f32", 1, 0}};
    vertices.elementLayout.abiLeaves.assign(f32Layout.leaves, f32Layout.leaves + f32Layout.leaf_count);
    vertices.shape = {3};
    vertices.uses.push_back({"vertex", "input", "", "f32", {3}, 0, 2, 0, 0, UINT32_MAX, {}, {{0, "f32", 3, 0}}});
    Parameter texture;
    texture.slot = 1;
    texture.name = "albedo";
    texture.kind = "texture";
    texture.dimension = "2d";
    texture.uses.push_back({"fragment", "uniform", "albedo", "", {}, 0, UINT32_MAX, 0, 0, 5, {}});
    variant.parameters = {vertices, texture};
    Parameter implicitSampler;
    implicitSampler.name = "__sampler";
    implicitSampler.kind = "sampler";
    implicitSampler.source = "implicit_sampler";
    implicitSampler.uses.push_back({"fragment", "uniform", "", "", {}, 1, UINT32_MAX, 0, 0, UINT32_MAX, {{0, 5}}});
    variant.internalParameters.push_back(implicitSampler);

    const std::array<uint64_t, 2> vertexShape{4, 3};
    std::array<int64_t, 2> vertexStrides{12, 4};
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[0].tensor.resource = vertexBuffer;
    arguments[0].tensor.element_layout = pipelineValueLayout(vertices.elementLayout);
    arguments[0].tensor.access = VERNON_ACCESS_READ;
    arguments[0].tensor.rank = 2;
    arguments[0].tensor.shape = vertexShape.data();
    arguments[0].tensor.byte_strides = vertexStrides.data();
    arguments[0].tensor.byte_size = 48;
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_TEXTURE;
    arguments[1].texture = {
        VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_2D, 16, 8, 1, sampledTexture, sampler};
    const VernonColorAttachment attachments[] = {
        {1, secondTarget, 64, 32, VERNON_TEXTURE_RGBA8_UNORM},
        {0, firstTarget, 64, 32, VERNON_TEXTURE_RGBA8_UNORM},
    };
    const VernonIndexBinding index{VERNON_INDEX_U32, 0, 6, indexBuffer};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = std::size(arguments);
    invocation.index_binding = &index;
    invocation.color_attachments = attachments;
    invocation.color_attachment_count = std::size(attachments);
    invocation.viewport[2] = 20;
    invocation.viewport[3] = 10;

    PlannedGraphicsInvocation plan;
    std::string error;
    ASSERT_TRUE(planGraphicsInvocation(variant, invocation, plan, error)) << error;
    ASSERT_EQ(plan.attachments.size(), 2u);
    EXPECT_EQ(plan.attachments[0]->location, 0u);
    EXPECT_EQ(plan.attachments[1]->location, 1u);
    EXPECT_EQ(plan.attachmentWidth, 64u);
    EXPECT_EQ(plan.attachmentHeight, 32u);
    EXPECT_EQ(plan.resolution, (std::array<float, 2>{20.0f, 10.0f}));
    EXPECT_EQ(plan.vertexCount, 4u);
    EXPECT_EQ(plan.instanceCount, 1u);
    EXPECT_EQ(plan.indexBinding, &index);
    ASSERT_EQ(plan.vertexInputs.size(), 1u);
    EXPECT_EQ(plan.vertexInputs[0].use->attributeLeaves.size(), 1u);
    const auto sampled = plan.sampledResources.find({0, 5});
    ASSERT_NE(sampled, plan.sampledResources.end());
    EXPECT_EQ(sampled->second.imageResource.identity, sampledTexture.identity);
    EXPECT_EQ(sampled->second.samplerResource.identity, sampler.identity);
    EXPECT_TRUE(sampled->second.implicitSampler);
    EXPECT_EQ(sampled->second.stages, PLANNED_STAGE_FRAGMENT);

    vertexStrides[0] = -12;
    arguments[0].tensor.byte_offset = 36;
    EXPECT_FALSE(planGraphicsInvocation(variant, invocation, plan, error));
    EXPECT_EQ(error, "graphics Tensor strides must be positive");
}

TEST(GraphicsInvocationPlanner, AcceptsFormalVertexNumericTypesAndRejectsNonVertexTypes) {
    struct Format {
        const char *name;
        VernonDataType type;
        uint32_t size;
    };
    constexpr Format supported[] = {{"i32", VERNON_DATA_I32, 4},
                                    {"u32", VERNON_DATA_U32, 4},
                                    {"f16", VERNON_DATA_F16, 2},
                                    {"f32", VERNON_DATA_F32, 4},
                                    {"f64", VERNON_DATA_F64, 8}};
    PlannedGraphicsInvocation plan;
    std::string error;
    for (const Format &format : supported) {
        const std::vector<uint64_t> shape{4, 3};
        const std::vector<int64_t> strides{3 * format.size, format.size};
        ASSERT_TRUE(planVertexTensor(format.name, format.type, {3}, {{0, format.name, 3, 0}}, shape, strides, 0, 0,
                                     plan, error))
            << format.name << ": " << error;
        ASSERT_EQ(plan.vertexInputs.size(), 1u);
        EXPECT_EQ(plan.vertexCount, 4u);
    }

    for (const Format format : {Format{"bool", VERNON_DATA_BOOL, 1}, Format{"u8", VERNON_DATA_U8, 1}}) {
        const std::vector<uint64_t> shape{4, 3};
        const std::vector<int64_t> strides{3, 1};
        EXPECT_FALSE(planVertexTensor(format.name, format.type, {3}, {{0, format.name, 3, 0}}, shape, strides, 0, 0,
                                      plan, error));
        EXPECT_EQ(error, "graphics Tensor dtype is unsupported");
    }
}

TEST(GraphicsInvocationPlanner, PlansNonSquareAttributesWithInterleavedRecordPadding) {
    PlannedGraphicsInvocation plan;
    std::string error;
    const std::vector<uint64_t> shape{5, 2, 3};
    const std::vector<int64_t> strides{32, 12, 4};
    ASSERT_TRUE(planVertexTensor("f32", VERNON_DATA_F32, {2, 3}, {{0, "f32", 4, 0}, {1, "f32", 2, 16}}, shape, strides,
                                 0, 0, plan, error))
        << error;
    ASSERT_EQ(plan.vertexInputs.size(), 1u);
    EXPECT_EQ(plan.vertexCount, 5u);
}

TEST(GraphicsInvocationPlanner, AppliesInstanceDivisorsToInferredAndExplicitCounts) {
    PlannedGraphicsInvocation plan;
    std::string error;
    const std::vector<uint64_t> shape{2, 4};
    const std::vector<int64_t> strides{24, 4};
    ASSERT_TRUE(planVertexTensor("f32", VERNON_DATA_F32, {4}, {{0, "f32", 4, 0}}, shape, strides, 3, 0, plan, error))
        << error;
    EXPECT_EQ(plan.instanceCount, 6u);
    ASSERT_TRUE(planVertexTensor("f32", VERNON_DATA_F32, {4}, {{0, "f32", 4, 0}}, shape, strides, 3, 5, plan, error))
        << error;
    EXPECT_EQ(plan.instanceCount, 5u);
    EXPECT_FALSE(planVertexTensor("f32", VERNON_DATA_F32, {4}, {{0, "f32", 4, 0}}, shape, strides, 3, 7, plan, error));
    EXPECT_EQ(error, "graphics Tensor leading dimensions conflict");
}

} // namespace
