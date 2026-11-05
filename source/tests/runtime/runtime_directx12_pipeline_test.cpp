#include "VernonRuntime.h"
#include "runtime/rhi_adapter/adapter_directx12_test_hooks.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <d3d12.h>

#include <array>
#include <iterator>

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
    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_DIRECTX12, nullptr,
                                                   vernon::tests::configuredDirectXTestDevicePreference());
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
    upload.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
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
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
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
