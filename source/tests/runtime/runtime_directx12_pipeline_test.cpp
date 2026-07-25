#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "runtime/runtime_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef VERNON_DIRECTX_PIPELINE_BUNDLE
#error VERNON_DIRECTX_PIPELINE_BUNDLE must name the cooked pipeline bundle
#endif

TEST(RuntimeDirectX12Pipeline, RendersSampledTriangleWithWarp) {
    vernon::runtime::setDirectX12WarpForTests(true);
    const std::filesystem::path manifestPath = VERNON_DIRECTX_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_DIRECTX12, 0);
    ASSERT_NE(runtime, nullptr);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_NE(loaded, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                              vernonRuntimeGetLastError(runtime).size);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);

    constexpr std::array<float, 6> positions{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    VernonDeviceBuffer *vertices = vernonRuntimeBufferAllocate(runtime, sizeof(positions), alignof(float));
    ASSERT_NE(vertices, nullptr);
    ASSERT_EQ(vernonRuntimeCopyFromHost(vertices, 0, positions.data(), sizeof(positions)), VERNON_STATUS_OK);
    VernonDeviceTexture *sampled = vernonRuntimeTextureCreate2D(runtime, 1, 1, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceTexture *target = vernonRuntimeTextureCreate2D(runtime, 32, 32, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(sampled, nullptr);
    ASSERT_NE(target, nullptr);
    constexpr std::array<uint8_t, 4> color{64, 200, 100, 255};
    ASSERT_EQ(vernonRuntimeTextureCopyFromHost(sampled, color.data(), color.size()), VERNON_STATUS_OK);
    VernonSamplerDescriptor samplerDescription{};
    samplerDescription.struct_size = sizeof(samplerDescription);
    samplerDescription.wrap_u = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescription.wrap_v = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescription.wrap_w = VERNON_SAMPLER_CLAMP_TO_EDGE;
    samplerDescription.min_filter = VERNON_SAMPLER_NEAREST;
    samplerDescription.mag_filter = VERNON_SAMPLER_NEAREST;
    samplerDescription.mip_filter = VERNON_SAMPLER_NEAREST;
    VernonDeviceSampler *sampler = vernonRuntimeSamplerCreate(runtime, &samplerDescription);
    ASSERT_NE(sampler, nullptr);

    constexpr uint64_t shape[] = {3, 2};
    constexpr int64_t strides[] = {2 * sizeof(float), sizeof(float)};
    VernonPipelineArgument arguments[3]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TEXTURE;
    arguments[0].texture = {sampled, VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_2D, 1, 1, 1,
                            nullptr};
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
    arguments[2].slot = 2;
    arguments[2].kind = VERNON_PIPELINE_SAMPLER;
    arguments[2].sampler = sampler;
    VernonColorAttachment attachment{0, target};
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
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsPipelineCreationCount(pipeline), 1u);
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsPipelineCreationCount(pipeline), 1u);

    std::vector<uint8_t> pixels(32 * 32 * 4);
    ASSERT_EQ(vernonRuntimeTextureCopyToHost(target, pixels.data(), pixels.size()), VERNON_STATUS_OK);
    const size_t center = (16 * 32 + 16) * 4;
    EXPECT_NEAR(pixels[center], color[0], 2);
    EXPECT_NEAR(pixels[center + 1], color[1], 2);
    EXPECT_NEAR(pixels[center + 2], color[2], 2);

    EXPECT_EQ(vernonRuntimeSamplerFree(sampler), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeTextureFree(target), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeTextureFree(sampled), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeBufferFree(vertices), VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernon::runtime::setDirectX12WarpForTests(false);
}

TEST(RuntimeDirectX12Pipeline, CompilesAndDispatchesComputeDxil) {
    static constexpr char module[] = R"(
module {
  func.func @increment(
      %values: !vernon.tensor_view<f32, 1, "read_write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %id: index {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %value = "vernon.intrinsic"(%values, %id) {
      name = "tensor_view_load"
    } : (!vernon.tensor_view<f32, 1, "read_write">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.intrinsic"(%values, %id, %sum) {
      name = "tensor_view_store"
    } : (!vernon.tensor_view<f32, 1, "read_write">, index, f32) -> ()
    return
  }
}
)";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    VernonCompileResult *compiled =
        vernonCompilerCompileMlir(compiler, module, std::strlen(module), VERNON_TARGET_DIRECTX);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK) << std::string(
        vernonCompileResultGetDiagnostics(compiled).data, vernonCompileResultGetDiagnostics(compiled).size);
    ASSERT_EQ(vernonCompileResultGetArtifactCount(compiled), 1u);

    vernon::runtime::setDirectX12WarpForTests(true);
    VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_DIRECTX12, 0);
    ASSERT_NE(runtime, nullptr);
    const VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
    const VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    ASSERT_GE(artifact.size, 4u);
    ASSERT_EQ(std::memcmp(artifact.data, "DXBC", 4), 0)
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[0])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[1])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[2])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[3]));
    VernonLoadedKernel *kernel = vernonRuntimeLoadArtifact(runtime, artifact.data, artifact.size, reflection.data,
                                                           reflection.size, "increment", std::strlen("increment"));
    ASSERT_NE(kernel, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                              vernonRuntimeGetLastError(runtime).size);
    std::array<float, 8> values{0, 1, 2, 3, 4, 5, 6, 7};
    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(runtime, sizeof(values), alignof(float));
    ASSERT_NE(buffer, nullptr);
    ASSERT_EQ(vernonRuntimeCopyFromHost(buffer, 0, values.data(), sizeof(values)), VERNON_STATUS_OK);
    VernonLaunchArgument argument{VERNON_LAUNCH_TENSOR, buffer, nullptr, 0};
    ASSERT_EQ(vernonRuntimeLaunch(kernel, {8, 1, 1}, &argument, 1), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    std::array<float, 8> result{};
    ASSERT_EQ(vernonRuntimeCopyToHost(buffer, 0, result.data(), sizeof(result)), VERNON_STATUS_OK);
    for (size_t index = 0; index < values.size(); ++index)
        EXPECT_EQ(result[index], values[index] + 1.0f);

    EXPECT_EQ(vernonRuntimeBufferFree(buffer), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeKernelUnload(kernel), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernon::runtime::setDirectX12WarpForTests(false);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}
