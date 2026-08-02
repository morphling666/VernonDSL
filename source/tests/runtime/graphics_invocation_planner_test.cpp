#include "runtime/graphics_invocation_planner.h"
#include "runtime/pipeline_metadata.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
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
    invocation.abi_version = VERNON_PIPELINE_VERSION;
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
    VernonColorAttachment attachments[] = {
        {1, secondTarget, 64, 32, VERNON_TEXTURE_RGBA8_UNORM},
        {0, firstTarget, 64, 32, VERNON_TEXTURE_RGBA8_UNORM},
    };
    VernonIndexBinding index{VERNON_INDEX_U32, 0, 6, indexBuffer};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
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

    index.type = static_cast<VernonIndexType>(1);
    EXPECT_FALSE(planGraphicsInvocation(variant, invocation, plan, error));
    EXPECT_EQ(error, "index binding is invalid");
    index.type = VERNON_INDEX_U32;
    attachments[0].location = 2;
    EXPECT_FALSE(planGraphicsInvocation(variant, invocation, plan, error));
    EXPECT_EQ(error, "render target locations must be contiguous from zero");
    attachments[0].location = 1;
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

TEST(GraphicsInvocationPlanner, NormalizesGraphicsStateForEveryProvider) {
    VernonRhiColorBlendState blends[2]{};
    blends[0].blend_enabled = 1;
    blends[0].source_color_factor = VERNON_RHI_BLEND_SOURCE_ALPHA;
    blends[0].destination_color_factor = VERNON_RHI_BLEND_ONE_MINUS_SOURCE_ALPHA;
    blends[0].color_operation = VERNON_RHI_BLEND_ADD;
    blends[0].source_alpha_factor = VERNON_RHI_BLEND_ONE;
    blends[0].destination_alpha_factor = VERNON_RHI_BLEND_ZERO;
    blends[0].alpha_operation = VERNON_RHI_BLEND_ADD;
    blends[0].write_mask = VERNON_RHI_COLOR_WRITE_RED | VERNON_RHI_COLOR_WRITE_GREEN;
    blends[1].write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    VernonGraphicsState source{};
    source.struct_size = sizeof(source);
    source.rasterization.cull_mode = VERNON_RHI_CULL_BACK;
    source.rasterization.front_face = VERNON_RHI_FRONT_FACE_CLOCKWISE;
    source.depth_stencil.depth_test = 1;
    source.depth_stencil.depth_write = 1;
    source.depth_stencil.depth_compare = VERNON_RHI_COMPARE_GREATER_EQUAL;
    source.depth_stencil.stencil_test = 1;
    source.depth_stencil.front.compare = VERNON_RHI_COMPARE_ALWAYS;
    source.depth_stencil.front.pass = VERNON_RHI_STENCIL_REPLACE;
    source.depth_stencil.stencil_read_mask = 0xff;
    source.depth_stencil.stencil_write_mask = 0xff;
    source.depth_stencil.back = source.depth_stencil.front;
    source.color_blends = blends;
    source.color_blend_count = std::size(blends);
    VernonPipelineInvocation invocation{};
    invocation.graphics_state = &source;
    invocation.stencil_reference = 3;

    PlannedGraphicsState planned;
    std::string error;
    ASSERT_TRUE(planGraphicsState(invocation, std::size(blends), true, true, planned, error)) << error;
    EXPECT_EQ(planned.rasterization.cull_mode, VERNON_RHI_CULL_BACK);
    EXPECT_EQ(planned.depthStencil.depth_compare, VERNON_RHI_COMPARE_GREATER_EQUAL);
    EXPECT_EQ(planned.stencilReference, 3u);
    ASSERT_EQ(planned.colorBlends.size(), std::size(blends));
    EXPECT_EQ(planned.colorBlends[0].write_mask, VERNON_RHI_COLOR_WRITE_RED | VERNON_RHI_COLOR_WRITE_GREEN);

    source.color_blend_count = 1;
    EXPECT_FALSE(planGraphicsState(invocation, std::size(blends), true, true, planned, error));
    EXPECT_EQ(error, "graphics state does not match the render-target layout");
}

TEST(GraphicsInvocationPlanner, NormalizesInactiveGraphicsState) {
    VernonRhiColorBlendState blend{};
    blend.source_color_factor = VERNON_RHI_BLEND_DESTINATION_COLOR;
    blend.destination_color_factor = VERNON_RHI_BLEND_SOURCE_ALPHA;
    blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    VernonGraphicsState source{};
    source.struct_size = sizeof(source);
    source.rasterization.depth_bias_constant = 42;
    source.rasterization.depth_bias_slope = 7;
    source.depth_stencil.depth_compare = VERNON_RHI_COMPARE_GREATER;
    source.depth_stencil.front.pass = VERNON_RHI_STENCIL_REPLACE;
    source.depth_stencil.stencil_read_mask = 0xff;
    source.depth_stencil.stencil_write_mask = 0xff;
    source.color_blends = &blend;
    source.color_blend_count = 1;
    VernonPipelineInvocation invocation{};
    invocation.graphics_state = &source;
    PlannedGraphicsState planned;
    std::string error;
    ASSERT_TRUE(planGraphicsState(invocation, 1, true, false, planned, error)) << error;
    EXPECT_EQ(planned.rasterization.depth_bias_constant, 0);
    EXPECT_EQ(planned.depthStencil.depth_compare, VERNON_RHI_COMPARE_ALWAYS);
    EXPECT_EQ(planned.depthStencil.stencil_read_mask, 0u);
    EXPECT_EQ(planned.depthStencil.front.stencil_fail, VERNON_RHI_STENCIL_ZERO);
    EXPECT_EQ(planned.depthStencil.front.depth_fail, VERNON_RHI_STENCIL_ZERO);
    EXPECT_EQ(planned.depthStencil.front.pass, VERNON_RHI_STENCIL_ZERO);
    EXPECT_EQ(planned.depthStencil.front.compare, VERNON_RHI_COMPARE_NEVER);
    EXPECT_EQ(planned.depthStencil.back.pass, VERNON_RHI_STENCIL_ZERO);
    EXPECT_EQ(planned.colorBlends[0].source_color_factor, VERNON_RHI_BLEND_ONE);
    EXPECT_EQ(planned.colorBlends[0].destination_color_factor, VERNON_RHI_BLEND_ZERO);
}

TEST(GraphicsInvocationPlanner, RejectsNonFiniteAndOutOfRangeState) {
    VernonGraphicsState source{};
    source.struct_size = sizeof(source);
    source.rasterization.depth_bias_constant = std::numeric_limits<float>::infinity();
    VernonPipelineInvocation invocation{};
    invocation.graphics_state = &source;
    PlannedGraphicsState planned;
    std::string error;
    EXPECT_FALSE(planGraphicsState(invocation, 0, false, false, planned, error));
    source.rasterization.depth_bias_constant = 0;
    source.depth_stencil.stencil_read_mask = 0x100;
    EXPECT_FALSE(planGraphicsState(invocation, 0, false, false, planned, error));
    source.depth_stencil.stencil_read_mask = 0;
    invocation.stencil_reference = 0x100;
    EXPECT_FALSE(planGraphicsState(invocation, 0, false, false, planned, error));
}

TEST(GraphicsInvocationPlanner, VariantKeyUsesOnlyCanonicalStaticFields) {
    GraphicsVariantKey first{};
    first.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    first.colorFormats = {3};
    first.vertexStrides = {8};
    first.colorBlends.resize(1);
    first.colorBlends[0].write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    GraphicsVariantKey same = first;
    EXPECT_TRUE(graphicsVariantKeysEqual(first, same));
    EXPECT_EQ(graphicsVariantKeyHash(first), graphicsVariantKeyHash(same));
    same.colorBlends[0].write_mask = VERNON_RHI_COLOR_WRITE_RED;
    EXPECT_FALSE(graphicsVariantKeysEqual(first, same));
}

} // namespace
