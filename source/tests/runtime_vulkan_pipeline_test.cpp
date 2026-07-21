#include "VernonRuntime.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef VERNON_VULKAN_PIPELINE_BUNDLE
#error VERNON_VULKAN_PIPELINE_BUNDLE must name the cooked pipeline bundle
#endif

int main() {
  if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN).available)
    return 0;

  const std::filesystem::path manifestPath = VERNON_VULKAN_PIPELINE_BUNDLE;
  std::ifstream input(manifestPath, std::ios::binary);
  const std::string bundle((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  assert(!bundle.empty());

  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_VULKAN, 0);
  assert(runtime);
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
  assert(loaded);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
  assert(pipeline);

  constexpr float positions[] = {-0.8f, -0.8f, 0.0f, 0.8f, -0.8f,
                                 0.0f,  0.0f,  0.8f, 0.0f};
  VernonDeviceBuffer *vertices =
      vernonRuntimeBufferAllocate(runtime, sizeof(positions), alignof(float));
  assert(vertices);
  assert(vernonRuntimeCopyFromHost(vertices, 0, positions, sizeof(positions)) ==
         VERNON_STATUS_OK);
  VernonDeviceTexture *target =
      vernonRuntimeTextureCreate2D(runtime, 32, 32, VERNON_TEXTURE_RGBA8_UNORM);
  assert(target);

  const uint64_t shape[] = {3, 3};
  const uint64_t strides[] = {3 * sizeof(float), sizeof(float)};
  VernonPipelineArgument arguments[2]{};
  arguments[0].slot = 2; // Slots are stable across variants and sorted by name.
  arguments[0].kind = VERNON_PIPELINE_TENSOR;
  arguments[0].tensor = {
      vertices, VERNON_DATA_F32, VERNON_ACCESS_READ, 2, shape, strides, 0};
  constexpr float tint[] = {0.25f, 1.0f, 0.5f, 1.0f};
  constexpr uint64_t tintShape[] = {4};
  arguments[1].slot = 3;
  arguments[1].kind = VERNON_PIPELINE_INLINE_VALUE;
  arguments[1].inline_value = {VERNON_DATA_F32, 1, tintShape, tint,
                               sizeof(tint)};
  VernonColorAttachment attachment{0, target};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.arguments = arguments;
  invocation.argument_count = 2;
  invocation.color_attachments = &attachment;
  invocation.color_attachment_count = 1;
  invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  invocation.instance_count = 1;
  assert(vernonRuntimePipelineInvoke(pipeline, &invocation) ==
         VERNON_STATUS_OK);

  std::vector<uint8_t> pixels(32 * 32 * 4);
  assert(vernonRuntimeTextureCopyToHost(target, pixels.data(), pixels.size()) ==
         VERNON_STATUS_OK);
  bool rendered = false;
  for (size_t index = 0; index < pixels.size(); index += 4)
    rendered |=
        pixels[index] != 0 || pixels[index + 1] != 0 || pixels[index + 2] != 0;
  assert(rendered);
  const size_t center = (16 * 32 + 16) * 4;
  assert(pixels[center] > 40 && pixels[center] < 90);
  assert(pixels[center + 1] > 240);
  assert(pixels[center + 2] > 110 && pixels[center + 2] < 150);

  assert(vernonRuntimeTextureFree(target) == VERNON_STATUS_OK);
  assert(vernonRuntimeBufferFree(vertices) == VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(loaded);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
  return 0;
}
