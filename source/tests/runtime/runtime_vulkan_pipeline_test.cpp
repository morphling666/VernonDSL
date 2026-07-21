#include "VernonRuntime.h"

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

TEST(RuntimeVulkanPipeline, SupportsTexturesAndRendering) {
  if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN).available)
    GTEST_SKIP() << "Vulkan runtime backend is unavailable";

  const std::filesystem::path manifestPath = VERNON_VULKAN_PIPELINE_BUNDLE;
  std::ifstream input(manifestPath, std::ios::binary);
  const std::string bundle((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  ASSERT_TRUE(!bundle.empty());

  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_VULKAN, 0);
  ASSERT_TRUE(runtime);
  size_t supportedFormatCount = 0;
  for (VernonTextureFormat format :
       {VERNON_TEXTURE_R8_UNORM, VERNON_TEXTURE_RG8_UNORM,
        VERNON_TEXTURE_RGB8_UNORM, VERNON_TEXTURE_RGBA8_UNORM,
        VERNON_TEXTURE_RGBA8_SRGB, VERNON_TEXTURE_RGBA16_FLOAT,
        VERNON_TEXTURE_RGBA32_FLOAT, VERNON_TEXTURE_R11G11B10_FLOAT}) {
    VernonTextureDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_TEXTURE_2D;
    descriptor.format = format;
    descriptor.width = 4;
    descriptor.height = 4;
    descriptor.depth = 1;
    descriptor.mip_levels = 1;
    VernonDeviceTexture *texture =
        vernonRuntimeTextureCreate(runtime, &descriptor);
    if (!texture) {
      const VernonStringView error = vernonRuntimeGetLastError(runtime);
      ASSERT_TRUE(error.data && error.size);
      continue;
    }
    ++supportedFormatCount;
    ASSERT_TRUE(vernonRuntimeTextureFree(texture) == VERNON_STATUS_OK);
  }
  ASSERT_TRUE(supportedFormatCount >= 3);
  for (VernonTextureDimension dimension :
       {VERNON_TEXTURE_3D, VERNON_TEXTURE_CUBE}) {
    VernonTextureDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = dimension;
    descriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
    descriptor.width = 4;
    descriptor.height = 4;
    descriptor.depth = dimension == VERNON_TEXTURE_3D ? 4 : 1;
    descriptor.mip_levels = 1;
    VernonDeviceTexture *texture =
        vernonRuntimeTextureCreate(runtime, &descriptor);
    ASSERT_TRUE(texture);
    ASSERT_TRUE(vernonRuntimeTextureFree(texture) == VERNON_STATUS_OK);
  }
  const std::string bundleDirectory = manifestPath.parent_path().u8string();
  VernonPipelineBundleLoadOptions options{};
  options.struct_size = sizeof(options);
  options.bundle_directory = bundleDirectory.c_str();
  VernonPipelineBundle *loaded = vernonRuntimeLoadPipelineBundleWithOptions(
      runtime, bundle.data(), bundle.size(), &options);
  if (!loaded) {
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    std::fwrite(error.data, 1, error.size, stderr);
    std::fputc('\n', stderr);
  }
  ASSERT_TRUE(loaded);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
  ASSERT_TRUE(pipeline);

  constexpr float positions[] = {-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
  VernonDeviceBuffer *vertices =
      vernonRuntimeBufferAllocate(runtime, sizeof(positions), alignof(float));
  ASSERT_TRUE(vertices);
  ASSERT_TRUE(vernonRuntimeCopyFromHost(vertices, 0, positions,
                                        sizeof(positions)) == VERNON_STATUS_OK);
  VernonDeviceTexture *target =
      vernonRuntimeTextureCreate2D(runtime, 32, 32, VERNON_TEXTURE_RGBA8_UNORM);
  ASSERT_TRUE(target);
  VernonTextureDescriptor sampledDescriptor{};
  sampledDescriptor.struct_size = sizeof(sampledDescriptor);
  sampledDescriptor.dimension = VERNON_TEXTURE_2D;
  sampledDescriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
  sampledDescriptor.width = 1;
  sampledDescriptor.height = 1;
  sampledDescriptor.depth = 1;
  sampledDescriptor.mip_levels = 1;
  VernonDeviceTexture *sampled =
      vernonRuntimeTextureCreate(runtime, &sampledDescriptor);
  ASSERT_TRUE(sampled);
  constexpr uint8_t sampledPixel[] = {64, 200, 100, 255};
  ASSERT_TRUE(vernonRuntimeTextureCopyFromHost(sampled, sampledPixel,
                                               sizeof(sampledPixel)) ==
              VERNON_STATUS_OK);
  VernonSamplerDescriptor samplerDescriptor{};
  samplerDescriptor.struct_size = sizeof(samplerDescriptor);
  samplerDescriptor.wrap_u = VERNON_SAMPLER_CLAMP_TO_EDGE;
  samplerDescriptor.wrap_v = VERNON_SAMPLER_CLAMP_TO_EDGE;
  samplerDescriptor.wrap_w = VERNON_SAMPLER_CLAMP_TO_EDGE;
  samplerDescriptor.min_filter = VERNON_SAMPLER_NEAREST;
  samplerDescriptor.mag_filter = VERNON_SAMPLER_NEAREST;
  samplerDescriptor.mip_filter = VERNON_SAMPLER_NEAREST;
  VernonDeviceSampler *sampler =
      vernonRuntimeSamplerCreate(runtime, &samplerDescriptor);
  ASSERT_TRUE(sampler);

  const uint64_t shape[] = {3, 2};
  const uint64_t strides[] = {2 * sizeof(float), sizeof(float)};
  VernonPipelineArgument arguments[3]{};
  arguments[0].slot = 0; // Slots are stable and sorted by source name.
  arguments[0].kind = VERNON_PIPELINE_TEXTURE;
  arguments[0].texture = {sampled,
                          VERNON_TEXTURE_RGBA8_UNORM,
                          VERNON_ACCESS_READ,
                          VERNON_TEXTURE_2D,
                          1,
                          1,
                          1};
  arguments[1].slot = 1;
  arguments[1].kind = VERNON_PIPELINE_TENSOR;
  arguments[1].tensor = {
      vertices, VERNON_DATA_F32, VERNON_ACCESS_READ, 2, shape, strides, 0};
  arguments[2].slot = 2;
  arguments[2].kind = VERNON_PIPELINE_SAMPLER;
  arguments[2].sampler = sampler;
  VernonColorAttachment attachment{0, target};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.arguments = arguments;
  invocation.argument_count = 3;
  invocation.color_attachments = &attachment;
  invocation.color_attachment_count = 1;
  invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  invocation.instance_count = 1;
  ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) ==
              VERNON_STATUS_OK);

  std::vector<uint8_t> pixels(32 * 32 * 4);
  ASSERT_TRUE(vernonRuntimeTextureCopyToHost(
                  target, pixels.data(), pixels.size()) == VERNON_STATUS_OK);
  bool rendered = false;
  for (size_t index = 0; index < pixels.size(); index += 4)
    rendered |=
        pixels[index] != 0 || pixels[index + 1] != 0 || pixels[index + 2] != 0;
  ASSERT_TRUE(rendered);
  const size_t center = (16 * 32 + 16) * 4;
  ASSERT_TRUE(pixels[center] > 55 && pixels[center] < 75);
  ASSERT_TRUE(pixels[center + 1] > 190 && pixels[center + 1] < 210);
  ASSERT_TRUE(pixels[center + 2] > 90 && pixels[center + 2] < 110);

  ASSERT_TRUE(vernonRuntimeSamplerFree(sampler) == VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeTextureFree(sampled) == VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeTextureFree(target) == VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeBufferFree(vertices) == VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(loaded);
  ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
