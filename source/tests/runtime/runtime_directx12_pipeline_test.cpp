#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "VernonVersions.h"
#include "runtime/rhi_adapter/adapter_directx12_test_hooks.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <d3d12.h>

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
#ifndef VERNON_DIRECTX_RESOLUTION_PIPELINE_BUNDLE
#error VERNON_DIRECTX_RESOLUTION_PIPELINE_BUNDLE must name the cooked resolution pipeline bundle
#endif
namespace {

vernon::tests::RhiImage createTexture2D(vernon::tests::RhiRuntime &context, uint32_t width, uint32_t height) {
    return vernon::tests::createImage(context, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, width, height, 1,
                                      VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
}

} // namespace

TEST(RuntimeDirectX12Pipeline, MapsEveryGraphicsStateEnumerationExplicitly) {
    constexpr D3D12_BLEND blendFactors[]{D3D12_BLEND_ZERO,       D3D12_BLEND_ONE,
                                         D3D12_BLEND_SRC_COLOR,  D3D12_BLEND_INV_SRC_COLOR,
                                         D3D12_BLEND_DEST_COLOR, D3D12_BLEND_INV_DEST_COLOR,
                                         D3D12_BLEND_SRC_ALPHA,  D3D12_BLEND_INV_SRC_ALPHA,
                                         D3D12_BLEND_DEST_ALPHA, D3D12_BLEND_INV_DEST_ALPHA};
    for (uint32_t index = 0; index < std::size(blendFactors); ++index)
        EXPECT_EQ(vernon::runtime::getDirectX12BlendFactorMapping(index), blendFactors[index]);
    constexpr D3D12_BLEND_OP blendOperations[]{D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_SUBTRACT, D3D12_BLEND_OP_REV_SUBTRACT,
                                               D3D12_BLEND_OP_MIN, D3D12_BLEND_OP_MAX};
    for (uint32_t index = 0; index < std::size(blendOperations); ++index)
        EXPECT_EQ(vernon::runtime::getDirectX12BlendOperationMapping(index), blendOperations[index]);
    constexpr D3D12_COMPARISON_FUNC compareOperations[]{
        D3D12_COMPARISON_FUNC_NEVER,         D3D12_COMPARISON_FUNC_LESS,    D3D12_COMPARISON_FUNC_EQUAL,
        D3D12_COMPARISON_FUNC_LESS_EQUAL,    D3D12_COMPARISON_FUNC_GREATER, D3D12_COMPARISON_FUNC_NOT_EQUAL,
        D3D12_COMPARISON_FUNC_GREATER_EQUAL, D3D12_COMPARISON_FUNC_ALWAYS};
    for (uint32_t index = 0; index < std::size(compareOperations); ++index)
        EXPECT_EQ(vernon::runtime::getDirectX12CompareOperationMapping(index), compareOperations[index]);
    constexpr D3D12_STENCIL_OP stencilOperations[]{
        D3D12_STENCIL_OP_KEEP,     D3D12_STENCIL_OP_ZERO,   D3D12_STENCIL_OP_REPLACE, D3D12_STENCIL_OP_INCR_SAT,
        D3D12_STENCIL_OP_DECR_SAT, D3D12_STENCIL_OP_INVERT, D3D12_STENCIL_OP_INCR,    D3D12_STENCIL_OP_DECR};
    for (uint32_t index = 0; index < std::size(stencilOperations); ++index)
        EXPECT_EQ(vernon::runtime::getDirectX12StencilOperationMapping(index), stencilOperations[index]);
    constexpr D3D12_CULL_MODE cullModes[]{D3D12_CULL_MODE_NONE, D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_BACK};
    for (uint32_t index = 0; index < std::size(cullModes); ++index)
        EXPECT_EQ(vernon::runtime::getDirectX12CullModeMapping(index), cullModes[index]);
}

TEST(RuntimeDirectX12Pipeline, GeneratesMipmapsWithEmbeddedComputeShader) {
    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_DIRECTX12, nullptr, true);
    ASSERT_NE(context.runtime, nullptr);
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    descriptor.width = 4;
    descriptor.height = 4;
    descriptor.depth = 1;
    descriptor.mip_levels = 3;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.usage =
        VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(context.device, &descriptor, &image), VERNON_RHI_STATUS_OK);
    std::array<uint8_t, 4 * 4 * 4> pixels{};
    for (size_t index = 0; index < pixels.size(); index += 4) {
        pixels[index] = 64;
        pixels[index + 1] = 128;
        pixels[index + 2] = 192;
        pixels[index + 3] = 255;
    }
    VernonRhiImageUploadDescriptor upload{};
    upload.struct_size = sizeof(upload);
    upload.width = 4;
    upload.height = 4;
    upload.depth = 1;
    upload.source_format = VERNON_RHI_IMAGE_DATA_RGBA;
    upload.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
    upload.data = pixels.data();
    ASSERT_EQ(vernonRhiDeviceUploadImage(context.device, image, &upload, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_OK);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.mip_level = 1;
    download.width = 2;
    download.height = 2;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    std::array<uint8_t, 2 * 2 * 4> result{};
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, image, &download, result.data(), result.size()),
              VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < result.size(); index += 4) {
        EXPECT_EQ(result[index], 64);
        EXPECT_EQ(result[index + 1], 128);
        EXPECT_EQ(result[index + 2], 192);
        EXPECT_EQ(result[index + 3], 255);
    }
    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, image), VERNON_RHI_STATUS_OK);
    vernon::tests::destroyRhiRuntime(context);
}

TEST(RuntimeDirectX12Pipeline, RendersSampledTriangleWithWarp) {
    const std::filesystem::path manifestPath = VERNON_DIRECTX_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_DIRECTX12, nullptr, true);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_NE(runtime, nullptr);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonProgramBundle *loaded =
        vernonRuntimeLoadProgramBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_NE(loaded, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                              vernonRuntimeGetLastError(runtime).size);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(loaded, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);
    VernonProgramParameterView imageParameter{};
    VernonProgramParameterView positionParameter{};
    VernonProgramParameterView samplerParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"image", 5}, &imageParameter), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"position", 8}, &positionParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"sampler", 7}, &samplerParameter),
              VERNON_STATUS_OK);

    constexpr std::array<float, 6> positions{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    auto vertices = vernon::tests::createBuffer(context, sizeof(positions), alignof(float), VERNON_RHI_BUFFER_VERTEX,
                                                positions.data());
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto sampled = createTexture2D(context, 1, 1);
    auto target = createTexture2D(context, 32, 32);
    auto depth =
        vernon::tests::createImage(context, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_D32_FLOAT, 32, 32, 1,
                                   VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE);
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(depth.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr std::array<uint8_t, 4> color{64, 200, 100, 255};
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
                                          color.data()};
    ASSERT_EQ(vernonRhiDeviceUploadImage(context.device, sampled.handle, &upload, 1), VERNON_RHI_STATUS_OK);
    auto sampler = vernon::tests::createSampler(context);
    ASSERT_NE(sampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    constexpr uint64_t shape[] = {3, 2};
    constexpr int64_t strides[] = {2 * sizeof(float), sizeof(float)};
    VernonProgramArgument arguments[3]{};
    arguments[0].slot = imageParameter.slot;
    arguments[0].kind = VERNON_PROGRAM_IMAGE;
    arguments[0].image = {sampled.reference};
    arguments[1].slot = positionParameter.slot;
    arguments[1].kind = VERNON_PROGRAM_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[1].tensor.resource = vertices.reference;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = 2;
    arguments[1].tensor.shape = shape;
    arguments[1].tensor.byte_strides = strides;
    arguments[1].tensor.byte_size = sizeof(positions);
    arguments[2].slot = samplerParameter.slot;
    arguments[2].kind = VERNON_PROGRAM_SAMPLER;
    arguments[2].resource = sampler.reference;
    VernonColorAttachment attachment{0, target.reference};
    VernonDepthAttachment depthAttachment{};
    depthAttachment.view = depth.reference;
    depthAttachment.width = 32;
    depthAttachment.height = 32;
    depthAttachment.format = VERNON_TEXTURE_D32_FLOAT;
    depthAttachment.load_operation = VERNON_RHI_LOAD_CLEAR;
    depthAttachment.store_operation = VERNON_RHI_STORE_PRESERVE;
    depthAttachment.clear_depth = 1.0f;
    vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
    graphics.renderPass.depth_attachment = &depthAttachment;
    ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, arguments, std::size(arguments), graphics),
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsPipelineCreationCount(pipeline), 1u);
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsRootSignatureCreationCount(pipeline), 1u);
    const vernon::runtime::DirectX12DepthStencilStateStats depthStencilStats =
        vernon::runtime::getDirectX12DepthStencilStateStats(runtime);
    EXPECT_TRUE(depthStencilStats.depthEnable);
    EXPECT_EQ(depthStencilStats.depthWriteMask, D3D12_DEPTH_WRITE_MASK_ALL);
    EXPECT_EQ(depthStencilStats.depthFunction, D3D12_COMPARISON_FUNC_LESS);
    EXPECT_FALSE(depthStencilStats.stencilEnable);
    ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, arguments, std::size(arguments), graphics),
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsPipelineCreationCount(pipeline), 1u);
    EXPECT_EQ(vernon::runtime::getDirectX12GraphicsRootSignatureCreationCount(pipeline), 1u);

    std::vector<uint8_t> pixels(32 * 32 * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = 32;
    download.height = 32;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, target.handle, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (16 * 32 + 16) * 4;
    EXPECT_NEAR(pixels[center], color[0], 2);
    EXPECT_NEAR(pixels[center + 1], color[1], 2);
    EXPECT_NEAR(pixels[center + 2], color[2], 2);

    EXPECT_EQ(vernonRhiDeviceDestroySampler(context.device, sampler.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, depth.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, target.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, sampled.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    EXPECT_EQ(vernon::runtime::getRhiAdapterLivePreparedPipelineCount(runtime), 0u);
    vernonRuntimeProgramBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeDirectX12Pipeline, SuppliesEffectiveResolutionWithWarp) {
    const std::filesystem::path manifestPath = VERNON_DIRECTX_RESOLUTION_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_DIRECTX12, nullptr, true);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_NE(runtime, nullptr);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonProgramBundle *loaded =
        vernonRuntimeLoadProgramBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_NE(loaded, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                              vernonRuntimeGetLastError(runtime).size);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(loaded, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);

    constexpr std::array<float, 6> positions{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    auto vertices = vernon::tests::createBuffer(context, sizeof(positions), alignof(float), VERNON_RHI_BUFFER_VERTEX,
                                                positions.data());
    auto target = createTexture2D(context, 32, 32);
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonProgramParameterView positionParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"position", 8}, &positionParameter),
              VERNON_STATUS_OK);
    constexpr uint64_t shape[] = {3, 2};
    constexpr int64_t strides[] = {2 * sizeof(float), sizeof(float)};
    VernonProgramArgument argument{};
    argument.slot = positionParameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = vertices.reference;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = 2;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(positions);
    VernonColorAttachment attachment{0, target.reference};
    vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1);
    graphics.dynamic.viewport[2] = 8;
    graphics.dynamic.viewport[3] = 16;
    ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, &argument, 1, graphics), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);

    std::vector<uint8_t> pixels(32 * 32 * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = 32;
    download.height = 32;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, target.handle, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (8 * 32 + 4) * 4;
    EXPECT_NEAR(pixels[center], 64, 2);
    EXPECT_NEAR(pixels[center + 1], 128, 2);
    EXPECT_NEAR(pixels[center + 2], 0, 2);

    graphics.dynamic.viewport[2] = 0;
    graphics.dynamic.viewport[3] = 0;
    ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, &argument, 1, graphics), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, target.handle, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t attachmentCenter = (16 * 32 + 16) * 4;
    EXPECT_NEAR(pixels[attachmentCenter], 255, 2);
    EXPECT_NEAR(pixels[attachmentCenter + 1], 255, 2);
    EXPECT_NEAR(pixels[attachmentCenter + 2], 0, 2);

    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, target.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeDirectX12Pipeline, CompilesAndDispatchesComputeDxil) {
    static constexpr char module[] = R"(
module attributes {)" VERNON_MLIR_VERSION_ATTRIBUTES R"(} {
  func.func @increment(
      %values: !vernon.tensor_view<f32, [-1], "read_write", "device"> {
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
    %value = "vernon.load"(%values, %id) :
      (!vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.store"(%sum, %values, %id) :
      (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
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

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_DIRECTX12, nullptr, true);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_NE(runtime, nullptr);
    const VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
    const VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    ASSERT_GE(artifact.size, 4u);
    ASSERT_EQ(std::memcmp(artifact.data, "DXBC", 4), 0)
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[0])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[1])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[2])) << " "
        << static_cast<unsigned>(static_cast<unsigned char>(artifact.data[3]));
    VernonStageExecutable *pipeline = vernonRuntimeLoadArtifact(runtime, artifact.data, artifact.size, reflection.data,
                                                                reflection.size, "increment", std::strlen("increment"));
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);
    std::array<float, 8> values{0, 1, 2, 3, 4, 5, 6, 7};
    auto buffer =
        vernon::tests::createBuffer(context, sizeof(values), alignof(float), VERNON_RHI_BUFFER_STORAGE, values.data());
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const uint64_t shape[]{8};
    const int64_t strides[]{sizeof(float)};
    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = buffer.reference;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(values);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {8, 1, 1};
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    std::array<float, 8> result{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, result.data(), sizeof(result)),
              VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < values.size(); ++index)
        EXPECT_EQ(result[index], values[index] + 1.0f);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST(RuntimeDirectX12Pipeline, DispatchesComputeBundleThroughRuntimeCoreProvider) {
    const std::filesystem::path manifestPath = VERNON_DIRECTX_COMPUTE_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_DIRECTX12, nullptr, true);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_NE(runtime, nullptr);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonProgramBundle *loaded =
        vernonRuntimeLoadProgramBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_NE(loaded, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                              vernonRuntimeGetLastError(runtime).size);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(loaded, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);
    VernonProgramParameterView valuesParameter{};
    VernonProgramParameterView factorParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"values", 6}, &valuesParameter), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"factor", 6}, &factorParameter), VERNON_STATUS_OK);

    constexpr std::array<float, 4> source{0, 1, 2, 3};
    auto buffer =
        vernon::tests::createBuffer(context, sizeof(source), alignof(float), VERNON_RHI_BUFFER_STORAGE, source.data());
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint64_t shape[] = {4};
    constexpr uint64_t scalarShape[] = {1};
    constexpr int64_t strides[] = {sizeof(float)};
    constexpr float factor = 3.0f;
    constexpr float secondFactor = 5.0f;
    VernonProgramArgument arguments[2]{};
    arguments[0].slot = valuesParameter.slot;
    arguments[0].kind = VERNON_PROGRAM_TENSOR;
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
    arguments[1].kind = VERNON_PROGRAM_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_HOST;
    arguments[1].tensor.host_data = &factor;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = 1;
    arguments[1].tensor.shape = scalarShape;
    arguments[1].tensor.byte_strides = strides;
    arguments[1].tensor.byte_size = sizeof(factor);
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(pipeline, arguments, std::size(arguments), {4, 1, 1}),
              VERNON_STATUS_INVALID_ARGUMENT);
    arguments[0].tensor.access = VERNON_ACCESS_READ_WRITE;
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(pipeline, arguments, std::size(arguments), {4, 1, 1}),
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);

    std::array<float, 4> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output.data(), sizeof(output)),
              VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < output.size(); ++index)
        EXPECT_EQ(output[index], source[index] * factor);

    constexpr uint64_t secondShape[] = {2};
    constexpr int64_t secondStrides[] = {2 * sizeof(float)};
    arguments[0].tensor.shape = secondShape;
    arguments[0].tensor.byte_strides = secondStrides;
    arguments[0].tensor.byte_offset = sizeof(float);
    arguments[1].tensor.host_data = &secondFactor;
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(pipeline, arguments, std::size(arguments), {2, 1, 1}),
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output.data(), sizeof(output)),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output[0], source[0] * factor);
    EXPECT_EQ(output[1], source[1] * factor * secondFactor);
    EXPECT_EQ(output[2], source[2] * factor);
    EXPECT_EQ(output[3], source[3] * factor * secondFactor);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}
