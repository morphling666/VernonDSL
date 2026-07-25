#include "VernonRuntime.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

#ifndef VERNON_VULKAN_CUBE_MAP_BUNDLE
#error VERNON_VULKAN_CUBE_MAP_BUNDLE must name the cooked CubeMap bundle
#endif

namespace {

VernonDeviceTexture *createTexture2D(VernonRuntimeContext *runtime, uint32_t width, uint32_t height,
                                     VernonTextureFormat format) {
    const VernonTextureDescriptor descriptor{
        sizeof(VernonTextureDescriptor), VERNON_TEXTURE_2D, format, width, height, 1, 1, {0, 0, 0, 0}};
    return vernonRuntimeTextureCreate(runtime, &descriptor);
}

VernonPipelineParameterView parameter(VernonLoadedPipeline *pipeline, const char *name) {
    VernonPipelineParameterView result{};
    const VernonStringView view{name, std::char_traits<char>::length(name)};
    EXPECT_EQ(vernonRuntimeLoadedPipelineFindParameter(pipeline, view, &result), VERNON_STATUS_OK) << name;
    return result;
}

void expectRuntimeOk(VernonRuntimeContext *runtime, VernonStatus status) {
    if (status == VERNON_STATUS_OK)
        return;
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    if (error.data)
        std::fwrite(error.data, 1, error.size, stderr);
    std::fputc('\n', stderr);
    EXPECT_EQ(status, VERNON_STATUS_OK);
}

} // namespace

TEST(RuntimeVulkanCubeMap, CooksSamplesAndRendersBothAttachments) {
    if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN).available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    const std::filesystem::path manifestPath = VERNON_VULKAN_CUBE_MAP_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_VULKAN, nullptr);
    ASSERT_TRUE(runtime);
    const std::string bundleDirectory = manifestPath.parent_path().u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_TRUE(loaded);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    ASSERT_TRUE(pipeline);

    constexpr float positions[] = {
        -1.0f, -1.0f, 1.0f, 3.0f, -1.0f, 1.0f, -1.0f, 3.0f, 1.0f,
    };
    VernonDeviceBuffer *vertices = vernonRuntimeBufferAllocate(runtime, sizeof(positions), alignof(float));
    ASSERT_TRUE(vertices);
    ASSERT_EQ(vernonRuntimeCopyFromHost(vertices, 0, positions, sizeof(positions)), VERNON_STATUS_OK);

    VernonTextureDescriptor cubeDescriptor{};
    cubeDescriptor.struct_size = sizeof(cubeDescriptor);
    cubeDescriptor.dimension = VERNON_TEXTURE_CUBE;
    cubeDescriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
    cubeDescriptor.width = 1;
    cubeDescriptor.height = 1;
    cubeDescriptor.depth = 1;
    cubeDescriptor.mip_levels = 1;
    VernonDeviceTexture *cube = vernonRuntimeTextureCreate(runtime, &cubeDescriptor);
    ASSERT_TRUE(cube);
    constexpr std::array<uint8_t, 24> cubePixels = {
        64, 200, 100, 255, 64, 200, 100, 255, 64, 200, 100, 255,
        64, 200, 100, 255, 64, 200, 100, 255, 64, 200, 100, 255,
    };
    ASSERT_EQ(vernonRuntimeTextureCopyFromHost(cube, cubePixels.data(), cubePixels.size()), VERNON_STATUS_OK);

    VernonSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    samplerDescriptor.wrap_u = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescriptor.wrap_v = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescriptor.wrap_w = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescriptor.min_filter = VERNON_SAMPLER_NEAREST;
    samplerDescriptor.mag_filter = VERNON_SAMPLER_NEAREST;
    samplerDescriptor.mip_filter = VERNON_SAMPLER_NEAREST;
    VernonDeviceSampler *sampler = vernonRuntimeSamplerCreate(runtime, &samplerDescriptor);
    ASSERT_TRUE(sampler);

    VernonDeviceTexture *color = createTexture2D(runtime, 32, 32, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceTexture *bloom = createTexture2D(runtime, 32, 32, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_TRUE(color);
    ASSERT_TRUE(bloom);

    constexpr uint64_t vertexShape[] = {3, 3};
    constexpr int64_t vertexStrides[] = {3 * sizeof(float), sizeof(float)};
    constexpr uint64_t matrixShape[] = {4, 4};
    constexpr int64_t matrixStrides[] = {4 * sizeof(float), sizeof(float)};
    constexpr std::array<float, 16> identity = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    };
    const VernonPipelineParameterView positionParameter = parameter(pipeline, "aPos");
    const VernonPipelineParameterView cubeParameter = parameter(pipeline, "cubeMap");
    const VernonPipelineParameterView modelParameter = parameter(pipeline, "model");
    const VernonPipelineParameterView projectionParameter = parameter(pipeline, "projection");
    const VernonPipelineParameterView viewParameter = parameter(pipeline, "view");

    std::array<VernonPipelineArgument, 5> arguments{};
    arguments[0].slot = positionParameter.slot;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_DEVICE;
    arguments[0].tensor.buffer = vertices;
    arguments[0].tensor.dtype = VERNON_DATA_F32;
    arguments[0].tensor.access = VERNON_ACCESS_READ;
    arguments[0].tensor.rank = 2;
    arguments[0].tensor.shape = vertexShape;
    arguments[0].tensor.byte_strides = vertexStrides;
    arguments[0].tensor.byte_size = sizeof(positions);
    arguments[1].slot = cubeParameter.slot;
    arguments[1].kind = VERNON_PIPELINE_TEXTURE;
    arguments[1].texture = {cube,   VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_CUBE, 1, 1, 1,
                            sampler};
    for (size_t index = 2; index < arguments.size(); ++index) {
        arguments[index].kind = VERNON_PIPELINE_TENSOR;
        arguments[index].tensor.struct_size = sizeof(VernonTensorView);
        arguments[index].tensor.storage = VERNON_TENSOR_HOST;
        arguments[index].tensor.host_data = identity.data();
        arguments[index].tensor.dtype = VERNON_DATA_F32;
        arguments[index].tensor.access = VERNON_ACCESS_READ;
        arguments[index].tensor.rank = 2;
        arguments[index].tensor.shape = matrixShape;
        arguments[index].tensor.byte_strides = matrixStrides;
        arguments[index].tensor.byte_size = identity.size() * sizeof(float);
    }
    arguments[2].slot = modelParameter.slot;
    arguments[3].slot = projectionParameter.slot;
    arguments[4].slot = viewParameter.slot;

    const VernonColorAttachment attachments[] = {{0, color}, {1, bloom}};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = arguments.data();
    invocation.argument_count = arguments.size();
    invocation.color_attachments = attachments;
    invocation.color_attachment_count = std::size(attachments);
    invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    invocation.vertex_count = 3;
    invocation.instance_count = 1;
    const VernonStatus invokeStatus = vernonRuntimePipelineInvoke(pipeline, &invocation);
    expectRuntimeOk(runtime, invokeStatus);
    ASSERT_EQ(invokeStatus, VERNON_STATUS_OK);

    std::vector<uint8_t> colorPixels(32 * 32 * 4);
    std::vector<uint8_t> bloomPixels(32 * 32 * 4);
    ASSERT_EQ(vernonRuntimeTextureCopyToHost(color, colorPixels.data(), colorPixels.size()), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeTextureCopyToHost(bloom, bloomPixels.data(), bloomPixels.size()), VERNON_STATUS_OK);
    const size_t center = (16 * 32 + 16) * 4;
    EXPECT_NEAR(colorPixels[center], 64, 10);
    EXPECT_NEAR(colorPixels[center + 1], 200, 10);
    EXPECT_NEAR(colorPixels[center + 2], 100, 10);
    EXPECT_GT(colorPixels[center + 3], 240);
    EXPECT_LT(bloomPixels[center], 10);
    EXPECT_LT(bloomPixels[center + 1], 10);
    EXPECT_LT(bloomPixels[center + 2], 10);
    EXPECT_GT(bloomPixels[center + 3], 240);

    ASSERT_EQ(vernonRuntimeTextureFree(bloom), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeTextureFree(color), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeSamplerFree(sampler), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeTextureFree(cube), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeBufferFree(vertices), VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}
