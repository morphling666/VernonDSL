#include "VernonRuntime.h"
#include "runtime/content_hash.h"
#include "runtime/runtime_test_hooks.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

#ifndef VERNON_VULKAN_PIPELINE_BUNDLE
#error VERNON_VULKAN_PIPELINE_BUNDLE must name the cooked pipeline bundle
#endif

TEST(RuntimeVulkanPipeline, ReusesGraphicsObjectsAcrossInvocations) {
    if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN).available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    const std::filesystem::path manifestPath = VERNON_VULKAN_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_TRUE(!bundle.empty());
    nlohmann::json document = nlohmann::json::parse(bundle);
    nlohmann::json &parameters = document["variants"][0]["parameters"];
    const auto samplerRow = std::find_if(parameters.begin(), parameters.end(),
                                         [](const nlohmann::json &row) { return row.value("kind", "") == "sampler"; });
    ASSERT_NE(samplerRow, parameters.end());
    nlohmann::json implicitSampler = *samplerRow;
    implicitSampler.erase("slot");
    implicitSampler["source"] = "implicit_sampler";
    parameters.erase(samplerRow);
    document["variants"][0]["internal_parameters"] = nlohmann::json::array({std::move(implicitSampler)});
    document.erase("content_hash");
    const std::string canonical = document.dump(-1, ' ', false);
    document["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    bundle = document.dump(-1, ' ', false);

    VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_VULKAN, 0);
    ASSERT_TRUE(runtime);
    size_t supportedFormatCount = 0;
    for (VernonTextureFormat format :
         {VERNON_TEXTURE_R8_UNORM, VERNON_TEXTURE_RG8_UNORM, VERNON_TEXTURE_RGB8_UNORM, VERNON_TEXTURE_RGBA8_UNORM,
          VERNON_TEXTURE_RGBA8_SRGB, VERNON_TEXTURE_RGBA16_FLOAT, VERNON_TEXTURE_RGBA32_FLOAT,
          VERNON_TEXTURE_R11G11B10_FLOAT}) {
        VernonTextureDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = VERNON_TEXTURE_2D;
        descriptor.format = format;
        descriptor.width = 4;
        descriptor.height = 4;
        descriptor.depth = 1;
        descriptor.mip_levels = 1;
        VernonDeviceTexture *texture = vernonRuntimeTextureCreate(runtime, &descriptor);
        if (!texture) {
            const VernonStringView error = vernonRuntimeGetLastError(runtime);
            ASSERT_TRUE(error.data && error.size);
            continue;
        }
        ++supportedFormatCount;
        ASSERT_TRUE(vernonRuntimeTextureFree(texture) == VERNON_STATUS_OK);
    }
    ASSERT_TRUE(supportedFormatCount >= 3);
    for (VernonTextureDimension dimension : {VERNON_TEXTURE_3D, VERNON_TEXTURE_CUBE}) {
        VernonTextureDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = dimension;
        descriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
        descriptor.width = 4;
        descriptor.height = 4;
        descriptor.depth = dimension == VERNON_TEXTURE_3D ? 4 : 1;
        descriptor.mip_levels = 1;
        VernonDeviceTexture *texture = vernonRuntimeTextureCreate(runtime, &descriptor);
        ASSERT_TRUE(texture);
        ASSERT_TRUE(vernonRuntimeTextureFree(texture) == VERNON_STATUS_OK);
    }
    const std::string bundleDirectory = manifestPath.parent_path().u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    if (!loaded) {
        const VernonStringView error = vernonRuntimeGetLastError(runtime);
        std::fwrite(error.data, 1, error.size, stderr);
        std::fputc('\n', stderr);
    }
    ASSERT_TRUE(loaded);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterCount(pipeline), 2u);

    constexpr float positions[] = {-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    VernonDeviceBuffer *vertices = vernonRuntimeBufferAllocate(runtime, sizeof(positions), alignof(float));
    ASSERT_TRUE(vertices);
    ASSERT_TRUE(vernonRuntimeCopyFromHost(vertices, 0, positions, sizeof(positions)) == VERNON_STATUS_OK);
    VernonDeviceTexture *firstTarget = vernonRuntimeTextureCreate2D(runtime, 32, 32, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_TRUE(firstTarget);
    VernonDeviceTexture *secondTarget = vernonRuntimeTextureCreate2D(runtime, 48, 24, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_TRUE(secondTarget);
    VernonTextureDescriptor sampledDescriptor{};
    sampledDescriptor.struct_size = sizeof(sampledDescriptor);
    sampledDescriptor.dimension = VERNON_TEXTURE_2D;
    sampledDescriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
    sampledDescriptor.width = 1;
    sampledDescriptor.height = 1;
    sampledDescriptor.depth = 1;
    sampledDescriptor.mip_levels = 1;
    VernonDeviceTexture *sampled = vernonRuntimeTextureCreate(runtime, &sampledDescriptor);
    ASSERT_TRUE(sampled);
    constexpr uint8_t sampledPixel[] = {64, 200, 100, 255};
    ASSERT_TRUE(vernonRuntimeTextureCopyFromHost(sampled, sampledPixel, sizeof(sampledPixel)) == VERNON_STATUS_OK);
    VernonSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    samplerDescriptor.wrap_u = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescriptor.wrap_v = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescriptor.wrap_w = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescriptor.min_filter = VERNON_SAMPLER_NEAREST;
    samplerDescriptor.mag_filter = VERNON_SAMPLER_NEAREST;
    samplerDescriptor.mip_filter = VERNON_SAMPLER_NEAREST;
    VernonDeviceSampler *textureSampler = vernonRuntimeSamplerCreate(runtime, &samplerDescriptor);
    ASSERT_TRUE(textureSampler);
    const uint64_t shape[] = {3, 2};
    const uint64_t strides[] = {2 * sizeof(float), sizeof(float)};
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0; // Slots are stable and sorted by source name.
    arguments[0].kind = VERNON_PIPELINE_TEXTURE;
    arguments[0].texture = {sampled,       VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_2D, 1, 1, 1,
                            textureSampler};
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_DEVICE;
    arguments[1].tensor.buffer = vertices;
    arguments[1].tensor.dtype = VERNON_DATA_F32;
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = 2;
    arguments[1].tensor.shape = shape;
    arguments[1].tensor.byte_strides = strides;
    arguments[1].tensor.byte_size = sizeof(positions);
    VernonColorAttachment attachment{0, firstTarget};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = std::size(arguments);
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    invocation.instance_count = 1;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    const vernon::runtime::VulkanGraphicsCacheStats firstStats =
        vernon::runtime::getVulkanGraphicsCacheStats(runtime, pipeline);
    EXPECT_EQ(firstStats.defaultImplicitSamplerCreations, 1u);
    EXPECT_EQ(firstStats.descriptorSetLayoutCreations, 1u);
    EXPECT_EQ(firstStats.pipelineLayoutCreations, 1u);
    EXPECT_EQ(firstStats.graphicsPipelineCreations, 1u);

    constexpr uint8_t secondSampledPixel[] = {180, 40, 220, 255};
    ASSERT_TRUE(vernonRuntimeTextureCopyFromHost(sampled, secondSampledPixel, sizeof(secondSampledPixel)) ==
                VERNON_STATUS_OK);
    arguments[0].texture.sampler = nullptr;
    attachment.texture = secondTarget;
    invocation.viewport[0] = 4;
    invocation.viewport[1] = 3;
    invocation.viewport[2] = 24;
    invocation.viewport[3] = 12;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    const vernon::runtime::VulkanGraphicsCacheStats secondStats =
        vernon::runtime::getVulkanGraphicsCacheStats(runtime, pipeline);
    EXPECT_EQ(secondStats.defaultImplicitSamplerCreations, firstStats.defaultImplicitSamplerCreations);
    EXPECT_EQ(secondStats.descriptorSetLayoutCreations, firstStats.descriptorSetLayoutCreations);
    EXPECT_EQ(secondStats.pipelineLayoutCreations, firstStats.pipelineLayoutCreations);
    EXPECT_EQ(secondStats.graphicsPipelineCreations, firstStats.graphicsPipelineCreations);

    std::vector<uint8_t> pixels(32 * 32 * 4);
    ASSERT_TRUE(vernonRuntimeTextureCopyToHost(firstTarget, pixels.data(), pixels.size()) == VERNON_STATUS_OK);
    bool rendered = false;
    for (size_t index = 0; index < pixels.size(); index += 4)
        rendered |= pixels[index] != 0 || pixels[index + 1] != 0 || pixels[index + 2] != 0;
    ASSERT_TRUE(rendered);
    const size_t center = (16 * 32 + 16) * 4;
    ASSERT_TRUE(pixels[center] > 55 && pixels[center] < 75);
    ASSERT_TRUE(pixels[center + 1] > 190 && pixels[center + 1] < 210);
    ASSERT_TRUE(pixels[center + 2] > 90 && pixels[center + 2] < 110);

    std::vector<uint8_t> secondPixels(48 * 24 * 4);
    ASSERT_TRUE(vernonRuntimeTextureCopyToHost(secondTarget, secondPixels.data(), secondPixels.size()) ==
                VERNON_STATUS_OK);
    const size_t secondCenter = (9 * 48 + 16) * 4;
    EXPECT_TRUE(secondPixels[secondCenter] > 170 && secondPixels[secondCenter] < 190);
    EXPECT_TRUE(secondPixels[secondCenter + 1] > 30 && secondPixels[secondCenter + 1] < 50);
    EXPECT_TRUE(secondPixels[secondCenter + 2] > 210 && secondPixels[secondCenter + 2] < 230);
    const size_t outsideViewport = (1 * 48 + 1) * 4;
    EXPECT_EQ(secondPixels[outsideViewport], 0u);
    EXPECT_EQ(secondPixels[outsideViewport + 1], 0u);
    EXPECT_EQ(secondPixels[outsideViewport + 2], 0u);

    ASSERT_TRUE(vernonRuntimeSamplerFree(textureSampler) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeTextureFree(sampled) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeTextureFree(secondTarget) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeTextureFree(firstTarget) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeBufferFree(vertices) == VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
