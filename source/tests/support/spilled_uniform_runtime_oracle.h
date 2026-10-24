#pragma once

#include "runtime_rhi_test_utils.h"
#include "vernon_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace vernon::tests {

/* Draws `spilled_uniform_color_asset`, whose fragment reads three matrices of which at least one spills to a uniform
 * buffer of a transient group, once to warm up and then repeatedly with changing matrices. `allocations` reports the
 * backend's native allocation counters; each must be positive after the warm draw and unchanged after the repeats. */
template <class Allocations>
void runSpilledUniformRuntimeOracle(RhiRuntime &context, VernonProgramExecutable *pipeline, Allocations allocations) {
    VernonRuntimeContext *runtime = context.runtime;
    constexpr float positions[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    auto vertices = createBuffer(context, sizeof(positions), alignof(float), VERNON_RHI_BUFFER_VERTEX, positions);
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto target = createImage(context, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, 8, 8, 1,
                              VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const auto targetView = createImageView(context, target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);

    // Symmetric, so that the product does not depend on the matrices' storage order.
    using Matrix = std::array<float, 16>;
    const auto diagonal = [](float x, float y, float z) {
        return Matrix{x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1};
    };
    const auto translation = [](float x, float y, float z) {
        return Matrix{1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, x, y, z, 1};
    };
    Matrix first = diagonal(0.5f, 1.0f, 1.0f);
    Matrix second = diagonal(1.0f, 0.25f, 1.0f);
    Matrix third{};
    constexpr uint64_t matrixShape[] = {4, 4};
    constexpr int64_t matrixStrides[] = {16, 4};
    constexpr uint64_t positionShape[] = {3, 2};
    constexpr int64_t positionStrides[] = {2 * sizeof(float), sizeof(float)};
    std::array<VernonProgramArgument, 4> arguments{};
    const std::array<std::string_view, 4> names = {"first", "second", "third", "position"};
    for (size_t index = 0; index < names.size(); ++index) {
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {names[index].data(), names[index].size()},
                                                              &parameter),
                  VERNON_STATUS_OK)
            << names[index];
        VernonProgramArgument &argument = arguments[index];
        argument.slot = parameter.slot;
        argument.kind = VERNON_PROGRAM_TENSOR;
        argument.tensor.struct_size = sizeof(VernonTensorView);
        argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
        argument.tensor.access = VERNON_ACCESS_READ;
        argument.tensor.rank = 2;
        argument.tensor.storage = VERNON_TENSOR_HOST;
        argument.tensor.shape = matrixShape;
        argument.tensor.byte_strides = matrixStrides;
        argument.tensor.byte_size = sizeof(Matrix);
    }
    arguments[0].tensor.host_data = first.data();
    arguments[1].tensor.host_data = second.data();
    arguments[2].tensor.host_data = third.data();
    arguments[3].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[3].tensor.resource = vertices.reference;
    arguments[3].tensor.shape = positionShape;
    arguments[3].tensor.byte_strides = positionStrides;
    arguments[3].tensor.byte_size = sizeof(positions);
    VernonColorAttachment attachment{0, targetView.reference};
    CanonicalGraphicsControls graphics(&attachment, 1, 3);
    std::vector<uint8_t> pixels(8 * 8 * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = 8;
    download.height = 8;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    const auto drawsColor = [&](float x, float y, float z, std::array<int, 3> expected) {
        third = translation(x, y, z);
        ASSERT_EQ(completeCanonicalInvocation(pipeline, arguments.data(), arguments.size(), graphics), VERNON_STATUS_OK)
            << vernon::test::text(vernonRuntimeGetLastError(runtime));
        ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, target.handle, &download, pixels.data(), pixels.size()),
                  VERNON_RHI_STATUS_OK);
        const size_t center = (4 * 8 + 4) * 4;
        for (size_t channel = 0; channel < expected.size(); ++channel)
            EXPECT_NEAR(pixels[center + channel], expected[channel], 2) << channel;
    };

    drawsColor(1.0f, 1.0f, 0.75f, {128, 64, 191});
    const std::vector<size_t> warm = allocations();
    for (size_t counter = 0; counter < warm.size(); ++counter)
        EXPECT_GT(warm[counter], 0u) << counter;
    for (int repetition = 0; repetition < 8; ++repetition) {
        drawsColor(0.0f, 1.0f, 1.0f, {0, 64, 255});
        drawsColor(1.0f, 0.0f, 0.5f, {128, 0, 128});
    }
    EXPECT_EQ(allocations(), warm);

    ASSERT_EQ(vernonRhiDeviceDestroyImageView(context.device, targetView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(context.device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, vertices.handle), VERNON_RHI_STATUS_OK);
}

} // namespace vernon::tests
