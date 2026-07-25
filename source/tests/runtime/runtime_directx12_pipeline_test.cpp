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
#ifndef VERNON_DIRECTX_COMPUTE_PIPELINE_BUNDLE
#error VERNON_DIRECTX_COMPUTE_PIPELINE_BUNDLE must name the cooked compute pipeline bundle
#endif

namespace {

VernonDeviceTexture *createTexture2D(VernonRuntimeContext *runtime, uint32_t width, uint32_t height,
                                     VernonTextureFormat format) {
    const VernonTextureDescriptor descriptor{
        sizeof(VernonTextureDescriptor), VERNON_TEXTURE_2D, format, width, height, 1, 1, {0, 0, 0, 0}};
    return vernonRuntimeTextureCreate(runtime, &descriptor);
}

} // namespace

TEST(RuntimeDirectX12Pipeline, RendersSampledTriangleWithWarp) {
    vernon::runtime::setDirectX12WarpForTests(true);
    const std::filesystem::path manifestPath = VERNON_DIRECTX_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_DIRECTX12, nullptr);
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
    VernonDeviceTexture *sampled = createTexture2D(runtime, 1, 1, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceTexture *target = createTexture2D(runtime, 32, 32, VERNON_TEXTURE_RGBA8_UNORM);
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
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsRootSignatureCreationCount(pipeline), 1u);
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsPipelineCreationCount(pipeline), 1u);
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsRootSignatureCreationCount(pipeline), 1u);

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
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_DIRECTX12, nullptr);
    ASSERT_NE(runtime, nullptr);
    const VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
    const VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    ASSERT_GE(artifact.size, 4u);
    ASSERT_EQ(std::memcmp(artifact.data, "DXBC", 4), 0)
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[0])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[1])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[2])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[3]));
    VernonLoadedPipeline *pipeline = vernonRuntimeLoadArtifact(runtime, artifact.data, artifact.size, reflection.data,
                                                               reflection.size, "increment", std::strlen("increment"));
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);
    std::array<float, 8> values{0, 1, 2, 3, 4, 5, 6, 7};
    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(runtime, sizeof(values), alignof(float));
    ASSERT_NE(buffer, nullptr);
    ASSERT_EQ(vernonRuntimeCopyFromHost(buffer, 0, values.data(), sizeof(values)), VERNON_STATUS_OK);
    const uint64_t shape[]{8};
    const int64_t strides[]{sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_DEVICE;
    argument.tensor.buffer = buffer;
    argument.tensor.dtype = VERNON_DATA_F32;
    argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(values);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {8, 1, 1};
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    std::array<float, 8> result{};
    ASSERT_EQ(vernonRuntimeCopyToHost(buffer, 0, result.data(), sizeof(result)), VERNON_STATUS_OK);
    for (size_t index = 0; index < values.size(); ++index)
        EXPECT_EQ(result[index], values[index] + 1.0f);

    EXPECT_EQ(vernonRuntimeBufferFree(buffer), VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernon::runtime::setDirectX12WarpForTests(false);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST(RuntimeDirectX12Pipeline, DispatchesComputeBundleThroughRuntimeCoreProvider) {
    vernon::runtime::setDirectX12WarpForTests(true);
    const std::filesystem::path manifestPath = VERNON_DIRECTX_COMPUTE_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_DIRECTX12, nullptr);
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
    VernonPipelineParameterView valuesParameter{};
    VernonPipelineParameterView factorParameter{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"values", 6}, &valuesParameter), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"factor", 6}, &factorParameter), VERNON_STATUS_OK);

    constexpr std::array<float, 4> source{0, 1, 2, 3};
    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(runtime, sizeof(source), alignof(float));
    ASSERT_NE(buffer, nullptr);
    ASSERT_EQ(vernonRuntimeCopyFromHost(buffer, 0, source.data(), sizeof(source)), VERNON_STATUS_OK);
    constexpr uint64_t shape[] = {4};
    constexpr uint64_t scalarShape[] = {1};
    constexpr int64_t strides[] = {sizeof(float)};
    constexpr float factor = 3.0f;
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = valuesParameter.slot;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_DEVICE;
    arguments[0].tensor.buffer = buffer;
    arguments[0].tensor.dtype = VERNON_DATA_F32;
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
    arguments[1].tensor.dtype = VERNON_DATA_F32;
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = 1;
    arguments[1].tensor.shape = scalarShape;
    arguments[1].tensor.byte_strides = strides;
    arguments[1].tensor.byte_size = sizeof(factor);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = std::size(arguments);
    invocation.compute_grid = {4, 1, 1};
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);

    std::array<float, 4> output{};
    ASSERT_EQ(vernonRuntimeCopyToHost(buffer, 0, output.data(), sizeof(output)), VERNON_STATUS_OK);
    for (size_t index = 0; index < output.size(); ++index)
        EXPECT_EQ(output[index], source[index] * factor);

    EXPECT_EQ(vernonRuntimeBufferFree(buffer), VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernon::runtime::setDirectX12WarpForTests(false);
}
