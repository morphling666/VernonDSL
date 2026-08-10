#include "VernonRuntime.h"
#include "runtime/content_hash.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime_rhi_test_utils.h"
#include "vernon_test_support.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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
#ifndef VERNON_VULKAN_COMPUTE_PIPELINE_BUNDLE
#error VERNON_VULKAN_COMPUTE_PIPELINE_BUNDLE must name the cooked compute pipeline bundle
#endif
namespace {

VernonRhiFormat rhiFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_R8_UNORM:
        return VERNON_RHI_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_RG8_UNORM:
        return VERNON_RHI_FORMAT_RG8_UNORM;
    case VERNON_TEXTURE_RGB8_UNORM:
        return VERNON_RHI_FORMAT_RGB8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return VERNON_RHI_FORMAT_RGBA8_SRGB;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return VERNON_RHI_FORMAT_RGBA16_FLOAT;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return VERNON_RHI_FORMAT_RGBA32_FLOAT;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return VERNON_RHI_FORMAT_R11G11B10_FLOAT;
    default:
        return VERNON_RHI_FORMAT_RGBA8_UNORM;
    }
}

vernon::tests::RhiImage createTexture2D(vernon::tests::RhiRuntime &context, uint32_t width, uint32_t height,
                                        VernonTextureFormat format, uint32_t usage) {
    return vernon::tests::createImage(context, VERNON_RHI_IMAGE_2D, rhiFormat(format), width, height, 1, usage);
}

} // namespace

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

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_TRUE(runtime);
    size_t supportedFormatCount = 0;
    for (VernonTextureFormat format :
         {VERNON_TEXTURE_R8_UNORM, VERNON_TEXTURE_RG8_UNORM, VERNON_TEXTURE_RGB8_UNORM, VERNON_TEXTURE_RGBA8_UNORM,
          VERNON_TEXTURE_RGBA8_SRGB, VERNON_TEXTURE_RGBA16_FLOAT, VERNON_TEXTURE_RGBA32_FLOAT,
          VERNON_TEXTURE_R11G11B10_FLOAT}) {
        auto texture = createTexture2D(context, 4, 4, format, VERNON_RHI_IMAGE_SAMPLED);
        if (texture.handle.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
            const VernonStringView error = vernonRhiDeviceGetLastError(context.device);
            ASSERT_TRUE(error.data && error.size);
            continue;
        }
        ++supportedFormatCount;
        ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, texture.handle), VERNON_RHI_STATUS_OK);
    }
    ASSERT_TRUE(supportedFormatCount >= 3);
    for (VernonTextureDimension dimension : {VERNON_TEXTURE_3D, VERNON_TEXTURE_CUBE}) {
        auto texture = vernon::tests::createImage(
            context, dimension == VERNON_TEXTURE_3D ? VERNON_RHI_IMAGE_3D : VERNON_RHI_IMAGE_CUBE,
            VERNON_RHI_FORMAT_RGBA8_UNORM, 4, 4, dimension == VERNON_TEXTURE_3D ? 4 : 1, VERNON_RHI_IMAGE_SAMPLED,
            dimension == VERNON_TEXTURE_CUBE ? 6 : 1);
        ASSERT_NE(texture.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, texture.handle), VERNON_RHI_STATUS_OK);
    }
    const std::string bundleDirectory = manifestPath.parent_path().u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    ASSERT_TRUE(document.contains("runtime_requirements"));
    nlohmann::json unsupportedDocument = document;
    unsupportedDocument["runtime_requirements"]["api_version"] = nlohmann::json::array({99, 0});
    unsupportedDocument.erase("content_hash");
    const std::string unsupportedCanonical = unsupportedDocument.dump(-1, ' ', false);
    unsupportedDocument["content_hash"] =
        vernon::runtime::sha256Hex(unsupportedCanonical.data(), unsupportedCanonical.size());
    const std::string unsupportedBundle = unsupportedDocument.dump(-1, ' ', false);
    ASSERT_FALSE(vernonRuntimeLoadPipelineBundleWithOptions(runtime, unsupportedBundle.data(), unsupportedBundle.size(),
                                                            &options));
    const VernonStringView unsupportedError = vernonRuntimeGetLastError(runtime);
    ASSERT_NE(std::string(unsupportedError.data, unsupportedError.size).find("pipeline requires Vulkan 99.0"),
              std::string::npos);

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
    auto vertices =
        vernon::tests::createBuffer(context, sizeof(positions), alignof(float), VERNON_RHI_BUFFER_VERTEX, positions);
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint32_t indices[]{0, 1, 2};
    auto indexBuffer =
        vernon::tests::createBuffer(context, sizeof(indices), alignof(uint32_t), VERNON_RHI_BUFFER_INDEX, indices);
    ASSERT_NE(indexBuffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto firstTarget = createTexture2D(context, 32, 32, VERNON_TEXTURE_RGBA8_UNORM, VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    ASSERT_NE(firstTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto secondTarget = createTexture2D(context, 48, 24, VERNON_TEXTURE_RGBA8_UNORM, VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    ASSERT_NE(secondTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto firstDepth =
        vernon::tests::createImage(context, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 32, 32, 1,
                                   VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE);
    ASSERT_NE(firstDepth.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto secondDepth =
        vernon::tests::createImage(context, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 48, 24, 1,
                                   VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE);
    ASSERT_NE(secondDepth.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto sampled = createTexture2D(context, 1, 1, VERNON_TEXTURE_RGBA8_UNORM, VERNON_RHI_IMAGE_SAMPLED);
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint8_t sampledPixel[] = {64, 200, 100, 255};
    VernonRhiImageUploadDescriptor upload{sizeof(VernonRhiImageUploadDescriptor),
                                          0,
                                          0,
                                          0,
                                          0,
                                          0,
                                          1,
                                          1,
                                          1,
                                          VERNON_RHI_IMAGE_DATA_RGBA,
                                          VERNON_RHI_IMAGE_DATA_UINT8,
                                          sampledPixel};
    ASSERT_EQ(vernonRhiDeviceUploadImage(context.device, sampled.handle, &upload, 1), VERNON_RHI_STATUS_OK)
        << vernon::test::text(vernonRhiDeviceGetLastError(context.device));
    auto textureSampler = vernon::tests::createSampler(context);
    ASSERT_NE(textureSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const uint64_t shape[] = {3, 2};
    const int64_t strides[] = {2 * sizeof(float), sizeof(float)};
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0; // Slots are stable and sorted by source name.
    arguments[0].kind = VERNON_PIPELINE_TEXTURE;
    arguments[0].texture = {VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ,      VERNON_TEXTURE_2D, 1, 1, 1,
                            sampled.reference,          textureSampler.reference};
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[1].tensor.resource = vertices.reference;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = 2;
    arguments[1].tensor.shape = shape;
    arguments[1].tensor.byte_strides = strides;
    arguments[1].tensor.byte_size = sizeof(positions);
    VernonColorAttachment attachment{0, firstTarget.reference, 32, 32, VERNON_TEXTURE_RGBA8_UNORM};
    VernonDepthAttachment depthAttachment{firstDepth.reference,
                                          32,
                                          32,
                                          VERNON_TEXTURE_D32_FLOAT_S8_UINT,
                                          VERNON_RHI_LOAD_CLEAR,
                                          VERNON_RHI_STORE_PRESERVE,
                                          1.0f,
                                          VERNON_RHI_LOAD_CLEAR,
                                          VERNON_RHI_STORE_PRESERVE,
                                          7};
    VernonRhiColorBlendState blend{};
    blend.source_color_factor = VERNON_RHI_BLEND_ONE;
    blend.destination_color_factor = VERNON_RHI_BLEND_ZERO;
    blend.source_alpha_factor = VERNON_RHI_BLEND_ONE;
    blend.destination_alpha_factor = VERNON_RHI_BLEND_ZERO;
    blend.color_operation = VERNON_RHI_BLEND_ADD;
    blend.alpha_operation = VERNON_RHI_BLEND_ADD;
    blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    VernonGraphicsState graphicsState{};
    graphicsState.struct_size = sizeof(graphicsState);
    graphicsState.rasterization.front_face = VERNON_RHI_FRONT_FACE_CLOCKWISE;
    graphicsState.depth_stencil.depth_test = 1;
    graphicsState.depth_stencil.depth_write = 1;
    graphicsState.depth_stencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    graphicsState.depth_stencil.stencil_test = 1;
    graphicsState.depth_stencil.front.compare = VERNON_RHI_COMPARE_ALWAYS;
    graphicsState.depth_stencil.front.pass = VERNON_RHI_STENCIL_REPLACE;
    graphicsState.depth_stencil.stencil_read_mask = 0xff;
    graphicsState.depth_stencil.stencil_write_mask = 0xff;
    graphicsState.depth_stencil.back = graphicsState.depth_stencil.front;
    graphicsState.color_blends = &blend;
    graphicsState.color_blend_count = 1;
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = std::size(arguments);
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.depth_attachment = &depthAttachment;
    invocation.graphics_state = &graphicsState;
    invocation.stencil_reference = 3;
    VernonIndexBinding indexBinding{VERNON_INDEX_U32, 0, std::size(indices), indexBuffer.reference};
    invocation.index_binding = &indexBinding;
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
    EXPECT_EQ(firstStats.bindingSnapshotCreations, 1u);
    EXPECT_EQ(firstStats.commandBufferAllocations, 1u);
    EXPECT_EQ(firstStats.descriptorPoolCreations, 1u);
    EXPECT_GT(firstStats.stagingBufferAllocations, 0u);
    EXPECT_EQ(firstStats.renderPassCreations, firstStats.dynamicRendering ? 0u : 1u);
    EXPECT_EQ(firstStats.lastStencilReference, 3u);
    EXPECT_TRUE(firstStats.lastDrawIndexed);

    constexpr uint8_t secondSampledPixel[] = {180, 40, 220, 255};
    upload.data = secondSampledPixel;
    ASSERT_EQ(vernonRhiDeviceUploadImage(context.device, sampled.handle, &upload, 1), VERNON_RHI_STATUS_OK);
    arguments[0].texture.sampler_resource = {};
    attachment.resource = secondTarget.reference;
    attachment.width = 48;
    attachment.height = 24;
    depthAttachment.resource = secondDepth.reference;
    depthAttachment.width = 48;
    depthAttachment.height = 24;
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
    EXPECT_EQ(secondStats.bindingSnapshotCreations, firstStats.bindingSnapshotCreations + 1);
    EXPECT_EQ(secondStats.commandBufferAllocations, firstStats.commandBufferAllocations);
    EXPECT_EQ(secondStats.descriptorPoolCreations, firstStats.descriptorPoolCreations);
    EXPECT_EQ(secondStats.stagingBufferAllocations, firstStats.stagingBufferAllocations);
    EXPECT_EQ(secondStats.renderPassCreations, firstStats.renderPassCreations);

    invocation.stencil_reference = 9;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    const vernon::runtime::VulkanGraphicsCacheStats warmStats =
        vernon::runtime::getVulkanGraphicsCacheStats(runtime, pipeline);
    EXPECT_EQ(warmStats.defaultImplicitSamplerCreations, secondStats.defaultImplicitSamplerCreations);
    EXPECT_EQ(warmStats.descriptorSetLayoutCreations, secondStats.descriptorSetLayoutCreations);
    EXPECT_EQ(warmStats.pipelineLayoutCreations, secondStats.pipelineLayoutCreations);
    EXPECT_EQ(warmStats.graphicsPipelineCreations, secondStats.graphicsPipelineCreations);
    EXPECT_EQ(warmStats.bindingSnapshotCreations, secondStats.bindingSnapshotCreations);
    EXPECT_EQ(warmStats.commandBufferAllocations, secondStats.commandBufferAllocations);
    EXPECT_EQ(warmStats.descriptorPoolCreations, secondStats.descriptorPoolCreations);
    EXPECT_EQ(warmStats.stagingBufferAllocations, secondStats.stagingBufferAllocations);
    EXPECT_EQ(warmStats.renderPassCreations, secondStats.renderPassCreations);
    EXPECT_EQ(warmStats.lastStencilReference, 9u);
    EXPECT_TRUE(warmStats.lastDrawIndexed);

    std::vector<uint8_t> pixels(32 * 32 * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = 32;
    download.height = 32;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, firstTarget.handle, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    bool rendered = false;
    for (size_t index = 0; index < pixels.size(); index += 4)
        rendered |= pixels[index] != 0 || pixels[index + 1] != 0 || pixels[index + 2] != 0;
    ASSERT_TRUE(rendered);
    const size_t center = (16 * 32 + 16) * 4;
    ASSERT_TRUE(pixels[center] > 55 && pixels[center] < 75);
    ASSERT_TRUE(pixels[center + 1] > 190 && pixels[center + 1] < 210);
    ASSERT_TRUE(pixels[center + 2] > 90 && pixels[center + 2] < 110);

    std::vector<uint8_t> secondPixels(48 * 24 * 4);
    download.width = 48;
    download.height = 24;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, secondTarget.handle, &download, secondPixels.data(),
                                           secondPixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t secondCenter = (9 * 48 + 16) * 4;
    EXPECT_TRUE(secondPixels[secondCenter] > 170 && secondPixels[secondCenter] < 190);
    EXPECT_TRUE(secondPixels[secondCenter + 1] > 30 && secondPixels[secondCenter + 1] < 50);
    EXPECT_TRUE(secondPixels[secondCenter + 2] > 210 && secondPixels[secondCenter + 2] < 230);
    const size_t outsideViewport = (1 * 48 + 1) * 4;
    EXPECT_EQ(secondPixels[outsideViewport], 0u);
    EXPECT_EQ(secondPixels[outsideViewport + 1], 0u);
    EXPECT_EQ(secondPixels[outsideViewport + 2], 0u);

    std::vector<uint8_t> depthStencilPixels(32 * 32 * 8);
    download.width = 32;
    download.height = 32;
    download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, firstDepth.handle, &download, depthStencilPixels.data(),
                                           depthStencilPixels.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(depthStencilPixels[(16 * 32 + 16) * 8 + sizeof(float)], 3u);
    EXPECT_EQ(depthStencilPixels[(1 * 32 + 1) * 8 + sizeof(float)], 7u);

    attachment.resource = firstTarget.reference;
    attachment.width = 32;
    attachment.height = 32;
    depthAttachment.resource = firstDepth.reference;
    depthAttachment.width = 32;
    depthAttachment.height = 32;
    std::fill(std::begin(invocation.viewport), std::end(invocation.viewport), 0);
    const size_t commandsBeforeGraph = vernon::runtime::getRhiAdapterRecordedCommandCount(runtime);
    {
        vernon::execution::ExecutionGraph graph(context.device);
        const auto graphTarget =
            graph.importImage(firstTarget.handle, {firstTarget.handle.index, firstTarget.handle.generation},
                              VERNON_RHI_FORMAT_RGBA8_UNORM, 32, 32, 1, 1, true);
        graph.emplacePass<vernon::tests::RuntimeGraphRenderPass>("first", graphTarget, runtime, pipeline, &invocation,
                                                                 VERNON_RHI_LOAD_CLEAR);
        graph.emplacePass<vernon::tests::RuntimeGraphRenderPass>("second", graphTarget, runtime, pipeline, &invocation,
                                                                 VERNON_RHI_LOAD_PRESERVE);
        ASSERT_EQ(graph.execute(), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(graph.lastStats().rendering_scope_count, 1u);
        EXPECT_EQ(graph.lastStats().draw_count, 2u);
        EXPECT_EQ(graph.lastStats().submission_count, 1u);
    }
    EXPECT_EQ(vernon::runtime::getRhiAdapterRecordedCommandCount(runtime) - commandsBeforeGraph, 2u);

    ASSERT_EQ(vernonRhiDeviceDestroySampler(context.device, textureSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, sampled.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, secondDepth.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, firstDepth.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, secondTarget.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, firstTarget.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, indexBuffer.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    EXPECT_EQ(vernon::runtime::getRhiAdapterLivePreparedPipelineCount(runtime), 0u);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeVulkanPipeline, DispatchesComputeBundleThroughRuntimeCoreProvider) {
    if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN).available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";
    const std::filesystem::path manifestPath = VERNON_VULKAN_COMPUTE_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());
    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_NE(runtime, nullptr);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_NE(loaded, nullptr);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);
    VernonPipelineParameterView valuesParameter{};
    VernonPipelineParameterView factorParameter{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"values", 6}, &valuesParameter), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"factor", 6}, &factorParameter), VERNON_STATUS_OK);

    constexpr std::array<float, 4> source{0, 1, 2, 3};
    auto buffer =
        vernon::tests::createBuffer(context, sizeof(source), alignof(float), VERNON_RHI_BUFFER_STORAGE, source.data());
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto secondBuffer =
        vernon::tests::createBuffer(context, sizeof(source), alignof(float), VERNON_RHI_BUFFER_STORAGE, source.data());
    ASSERT_NE(secondBuffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint64_t shape[]{4};
    constexpr uint64_t scalarShape[]{1};
    constexpr int64_t strides[]{sizeof(float)};
    constexpr float factor = 3.0f;
    constexpr float secondFactor = 5.0f;
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = valuesParameter.slot;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[0].tensor.resource = buffer.reference;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[0].tensor.access = VERNON_ACCESS_WRITE;
    arguments[0].tensor.rank = 1;
    arguments[0].tensor.shape = shape;
    arguments[0].tensor.byte_strides = strides;
    arguments[0].tensor.byte_size = sizeof(source);
    arguments[1].slot = factorParameter.slot;
    arguments[1].kind = VERNON_PIPELINE_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_HOST;
    arguments[1].tensor.host_data = &factor;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = 1;
    arguments[1].tensor.shape = scalarShape;
    arguments[1].tensor.byte_strides = strides;
    arguments[1].tensor.byte_size = sizeof(factor);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = std::size(arguments);
    invocation.compute_grid = {4, 1, 1};
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(context.device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerEncoder{};
    ASSERT_EQ(vernonRuntimeReferenceRhiCommandEncoder(runtime, encoder, &providerEncoder), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    arguments[0].tensor.access = VERNON_ACCESS_READ_WRITE;
    ASSERT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK)
        << "RHI: " << vernon::test::text(vernonRhiDeviceGetLastError(context.device))
        << "; runtime: " << vernon::test::text(vernonRuntimeGetLastError(runtime));
    constexpr uint64_t secondShape[]{2};
    constexpr int64_t secondStrides[]{2 * sizeof(float)};
    arguments[0].tensor.resource = secondBuffer.reference;
    arguments[0].tensor.shape = secondShape;
    arguments[0].tensor.byte_strides = secondStrides;
    arguments[0].tensor.byte_offset = sizeof(float);
    arguments[1].tensor.host_data = &secondFactor;
    invocation.compute_grid = {2, 1, 1};
    ASSERT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(context.device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(context.device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(context.device, encoder), VERNON_RHI_STATUS_OK);
    std::array<float, 4> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output.data(), sizeof(output)),
              VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < output.size(); ++index)
        EXPECT_EQ(output[index], source[index] * factor);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, secondBuffer.handle, 0, output.data(), sizeof(output)),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output[0], source[0]);
    EXPECT_EQ(output[1], source[1] * secondFactor);
    EXPECT_EQ(output[2], source[2]);
    EXPECT_EQ(output[3], source[3] * secondFactor);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, secondBuffer.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}
