#pragma once

#include "backend_test_matrix.h"
#include "runtime_rhi_test_utils.h"
#include "vernon_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace vernon::tests {

inline void runSampledTextureRuntimeOracle(OwnedRhiRuntime &owned, VernonProgramExecutable *executable,
                                           const double *expected, size_t expectedCount) {
    constexpr uint8_t sampledPixel[]{255, 0, 0, 255};
    RhiImage sampled = createImage(owned.context(), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, 1, 1, 1,
                                   VERNON_RHI_IMAGE_SAMPLED);
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
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
    ASSERT_EQ(vernonRhiDeviceUploadImage(owned.device(), sampled.handle, &upload, 1), VERNON_RHI_STATUS_OK);
    RhiImageView sampledView =
        createImageView(owned.context(), sampled, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    RhiSampler sampler = createSampler(owned.context());
    ASSERT_NE(sampledView.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(sampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    std::vector<VernonProgramArgument> arguments;
    const size_t parameterCount = vernonRuntimeProgramExecutableGetParameterCount(executable);
    arguments.reserve(parameterCount);
    for (size_t index = 0; index < parameterCount; ++index) {
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &parameter), VERNON_STATUS_OK);
        VernonProgramArgument argument{};
        argument.slot = parameter.slot;
        argument.kind = parameter.kind;
        if (parameter.kind == VERNON_PROGRAM_IMAGE)
            argument.image = {sampledView.reference};
        else {
            ASSERT_EQ(parameter.kind, VERNON_PROGRAM_SAMPLER);
            argument.resource = sampler.reference;
        }
        arguments.push_back(argument);
    }

    constexpr uint32_t extent = 32;
    RhiImage target = createImage(owned.context(), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, extent, extent,
                                  1, VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    RhiImageView targetView =
        createImageView(owned.context(), target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(targetView.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonColorAttachment attachment{};
    attachment.location = 0;
    attachment.view = targetView.reference;
    attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    attachment.clear_color[3] = 1.0f;
    CanonicalGraphicsControls graphics(&attachment, 1, 3);
    graphics.renderPass.render_area[2] = extent;
    graphics.renderPass.render_area[3] = extent;
    graphics.dynamic.viewport[2] = extent;
    graphics.dynamic.viewport[3] = extent;
    graphics.dynamic.scissor[2] = extent;
    graphics.dynamic.scissor[3] = extent;
    ASSERT_EQ(completeCanonicalInvocation(executable, arguments.data(), arguments.size(), graphics), VERNON_STATUS_OK)
        << runtimeDiagnostic(vernonRuntimeGetLastError(owned.runtime()));

    std::vector<uint8_t> pixels(extent * extent * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = extent;
    download.height = extent;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(owned.device(), target.handle, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (extent / 2 * extent + extent / 2) * 4;
    const std::array<uint8_t, 4> actual{pixels[center], pixels[center + 1], pixels[center + 2], pixels[center + 3]};
    ASSERT_EQ(expectedCount, actual.size());
    for (size_t channel = 0; channel < actual.size(); ++channel)
        EXPECT_NEAR(actual[channel], expected[channel], 1);

    EXPECT_EQ(vernonRhiDeviceDestroyImageView(owned.device(), targetView.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(owned.device(), target.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySampler(owned.device(), sampler.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(owned.device(), sampledView.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(owned.device(), sampled.handle), VERNON_RHI_STATUS_OK);
}

} // namespace vernon::tests
