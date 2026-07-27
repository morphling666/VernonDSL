#include "VernonRuntime.h"
#include "runtime_rhi_test_utils.h"

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

vernon::tests::RhiImage createTexture2D(vernon::tests::RhiRuntime &context, uint32_t width, uint32_t height) {
    return vernon::tests::createImage(context, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, width, height, 1,
                                      VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
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

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_TRUE(runtime);
    const std::string bundleDirectory = manifestPath.parent_path().u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_TRUE(loaded);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    const VernonStringView resolveError = vernonRuntimeGetLastError(runtime);
    ASSERT_TRUE(pipeline) << (resolveError.data ? std::string(resolveError.data, resolveError.size) : std::string{});

    constexpr float positions[] = {
        -1.0f, -1.0f, 1.0f, 3.0f, -1.0f, 1.0f, -1.0f, 3.0f, 1.0f,
    };
    auto vertices =
        vernon::tests::createBuffer(context, sizeof(positions), alignof(float), VERNON_RHI_BUFFER_VERTEX, positions);
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto cube = vernon::tests::createImage(context, VERNON_RHI_IMAGE_CUBE, VERNON_RHI_FORMAT_RGBA8_UNORM, 1, 1, 1,
                                           VERNON_RHI_IMAGE_SAMPLED, 6);
    ASSERT_NE(cube.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr std::array<uint8_t, 24> cubePixels = {
        64, 200, 100, 255, 64, 200, 100, 255, 64, 200, 100, 255,
        64, 200, 100, 255, 64, 200, 100, 255, 64, 200, 100, 255,
    };
    std::array<VernonRhiImageUploadDescriptor, 6> uploads{};
    for (size_t index = 0; index < uploads.size(); ++index)
        uploads[index] = {sizeof(VernonRhiImageUploadDescriptor),
                          0,
                          static_cast<uint32_t>(index),
                          1,
                          1,
                          1,
                          VERNON_RHI_IMAGE_DATA_RGBA,
                          VERNON_RHI_IMAGE_DATA_UINT8,
                          cubePixels.data() + index * 4};
    ASSERT_EQ(vernonRhiDeviceUploadImage(context.device, cube.handle, uploads.data(), uploads.size()),
              VERNON_RHI_STATUS_OK);
    auto sampler = vernon::tests::createSampler(context);
    ASSERT_NE(sampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    auto color = createTexture2D(context, 32, 32);
    auto bloom = createTexture2D(context, 32, 32);
    ASSERT_NE(color.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(bloom.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

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
    arguments[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[0].tensor.resource = vertices.reference;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[0].tensor.access = VERNON_ACCESS_READ;
    arguments[0].tensor.rank = 2;
    arguments[0].tensor.shape = vertexShape;
    arguments[0].tensor.byte_strides = vertexStrides;
    arguments[0].tensor.byte_size = sizeof(positions);
    arguments[1].slot = cubeParameter.slot;
    arguments[1].kind = VERNON_PIPELINE_TEXTURE;
    arguments[1].texture = {
        VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_CUBE, 1, 1, 1, cube.reference,
        sampler.reference};
    for (size_t index = 2; index < arguments.size(); ++index) {
        arguments[index].kind = VERNON_PIPELINE_TENSOR;
        arguments[index].tensor.struct_size = sizeof(VernonTensorView);
        arguments[index].tensor.storage = VERNON_TENSOR_HOST;
        arguments[index].tensor.host_data = identity.data();
        arguments[index].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
        arguments[index].tensor.access = VERNON_ACCESS_READ;
        arguments[index].tensor.rank = 2;
        arguments[index].tensor.shape = matrixShape;
        arguments[index].tensor.byte_strides = matrixStrides;
        arguments[index].tensor.byte_size = identity.size() * sizeof(float);
    }
    arguments[2].slot = modelParameter.slot;
    arguments[3].slot = projectionParameter.slot;
    arguments[4].slot = viewParameter.slot;

    const VernonColorAttachment attachments[] = {
        {0, color.reference, 32, 32, VERNON_TEXTURE_RGBA8_UNORM},
        {1, bloom.reference, 32, 32, VERNON_TEXTURE_RGBA8_UNORM},
    };
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
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, color.handle, colorPixels.data(), colorPixels.size()),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, bloom.handle, bloomPixels.data(), bloomPixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (16 * 32 + 16) * 4;
    EXPECT_NEAR(colorPixels[center], 64, 10);
    EXPECT_NEAR(colorPixels[center + 1], 200, 10);
    EXPECT_NEAR(colorPixels[center + 2], 100, 10);
    EXPECT_GT(colorPixels[center + 3], 240);
    EXPECT_LT(bloomPixels[center], 10);
    EXPECT_LT(bloomPixels[center + 1], 10);
    EXPECT_LT(bloomPixels[center + 2], 10);
    EXPECT_GT(bloomPixels[center + 3], 240);

    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, bloom.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, color.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(context.device, sampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, cube.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}
