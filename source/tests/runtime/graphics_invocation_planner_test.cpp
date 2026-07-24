#include "runtime/graphics_invocation_planner.h"

#include <gtest/gtest.h>

#include <array>
#include <unordered_map>

namespace {

using namespace vernon::runtime;

struct Resources {
    const void *context{};
    std::unordered_map<const void *, GraphicsResourceSnapshot> buffers;
    std::unordered_map<const void *, GraphicsResourceSnapshot> textures;
    std::unordered_map<const void *, const void *> samplers;
};

GraphicsResourceSnapshot bufferSnapshot(const void *userData, const VernonDeviceBuffer *buffer) {
    const auto &resources = *static_cast<const Resources *>(userData);
    const auto found = resources.buffers.find(buffer);
    return found == resources.buffers.end() ? GraphicsResourceSnapshot{} : found->second;
}

GraphicsResourceSnapshot textureSnapshot(const void *userData, const VernonDeviceTexture *texture) {
    const auto &resources = *static_cast<const Resources *>(userData);
    const auto found = resources.textures.find(texture);
    return found == resources.textures.end() ? GraphicsResourceSnapshot{} : found->second;
}

const void *samplerContext(const void *userData, const VernonDeviceSampler *sampler) {
    const auto &resources = *static_cast<const Resources *>(userData);
    const auto found = resources.samplers.find(sampler);
    return found == resources.samplers.end() ? nullptr : found->second;
}

TEST(GraphicsInvocationPlanner, PlansSortedTargetsPairingResolutionCountsAndIndex) {
    int contextStorage = 0;
    int vertexBufferStorage = 0;
    int indexBufferStorage = 0;
    int sampledTextureStorage = 0;
    int firstTargetStorage = 0;
    int secondTargetStorage = 0;
    int samplerStorage = 0;
    const void *context = &contextStorage;
    auto *vertexBuffer = reinterpret_cast<VernonDeviceBuffer *>(&vertexBufferStorage);
    auto *indexBuffer = reinterpret_cast<VernonDeviceBuffer *>(&indexBufferStorage);
    auto *sampledTexture = reinterpret_cast<VernonDeviceTexture *>(&sampledTextureStorage);
    auto *firstTarget = reinterpret_cast<VernonDeviceTexture *>(&firstTargetStorage);
    auto *secondTarget = reinterpret_cast<VernonDeviceTexture *>(&secondTargetStorage);
    auto *sampler = reinterpret_cast<VernonDeviceSampler *>(&samplerStorage);

    Resources resources;
    resources.context = context;
    resources.buffers[vertexBuffer] = {context, 48};
    resources.buffers[indexBuffer] = {context, 24};
    resources.textures[sampledTexture] = {context, 0, VERNON_TEXTURE_2D, VERNON_TEXTURE_RGBA8_UNORM, 16, 8, 1};
    resources.textures[firstTarget] = {context, 0, VERNON_TEXTURE_2D, VERNON_TEXTURE_RGBA8_UNORM, 64, 32, 1};
    resources.textures[secondTarget] = resources.textures[firstTarget];
    resources.samplers[sampler] = context;
    const GraphicsPlannerCallbacks callbacks{&resources, &bufferSnapshot, &textureSnapshot, &samplerContext};

    Variant variant;
    variant.vertex = "vertex";
    variant.fragment = "fragment";
    Parameter vertices;
    vertices.slot = 0;
    vertices.name = "vertices";
    vertices.kind = "tensor";
    vertices.dtype = "f32";
    vertices.shape = {3};
    vertices.uses.push_back({"vertex", "input", "", "f32", {3}, 0, 2, 0, 0, UINT32_MAX, {}});
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
    arguments[0].tensor.storage = VERNON_TENSOR_DEVICE;
    arguments[0].tensor.buffer = vertexBuffer;
    arguments[0].tensor.dtype = VERNON_DATA_F32;
    arguments[0].tensor.access = VERNON_ACCESS_READ;
    arguments[0].tensor.rank = 2;
    arguments[0].tensor.shape = vertexShape.data();
    arguments[0].tensor.byte_strides = vertexStrides.data();
    arguments[0].tensor.byte_size = 48;
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_TEXTURE;
    arguments[1].texture = {sampledTexture, VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_2D, 16, 8, 1,
                            sampler};
    const VernonColorAttachment attachments[] = {{1, secondTarget}, {0, firstTarget}};
    const VernonIndexBinding index{indexBuffer, VERNON_INDEX_U32, 0, 6};
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
    ASSERT_TRUE(planGraphicsInvocation(variant, invocation, context, callbacks, plan, error)) << error;
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
    EXPECT_EQ(plan.vertexInputs[0].components, 3u);
    const auto sampled = plan.sampledResources.find({0, 5});
    ASSERT_NE(sampled, plan.sampledResources.end());
    EXPECT_EQ(sampled->second.texture, sampledTexture);
    EXPECT_EQ(sampled->second.sampler, sampler);
    EXPECT_TRUE(sampled->second.implicitSampler);
    EXPECT_EQ(sampled->second.stages, PLANNED_STAGE_FRAGMENT);

    vertexStrides[0] = -12;
    arguments[0].tensor.byte_offset = 36;
    EXPECT_FALSE(planGraphicsInvocation(variant, invocation, context, callbacks, plan, error));
    EXPECT_EQ(error, "graphics Tensor strides must be positive");
}

} // namespace
