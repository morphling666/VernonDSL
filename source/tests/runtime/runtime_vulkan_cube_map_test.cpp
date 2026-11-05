#include "VernonRuntime.h"
#include "VernonSchedule.h"
#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#ifndef VERNON_VULKAN_CUBE_MAP_BUNDLE
#error VERNON_VULKAN_CUBE_MAP_BUNDLE must name the cooked CubeMap bundle
#endif

TEST(RuntimeVulkanCubeMap, CooksSamplesAndRendersBothAttachments) {
    vernon::tests::HalTestDevice device;
    vernon::tests::BackendTestRequirements requirements;
    requirements.graphics = true;
    const vernon::tests::BackendProbeResult probe = device.initialize(VERNON_RUNTIME_VULKAN, requirements);
    if (!probe.available()) {
        if (probe.skippable())
            GTEST_SKIP() << probe.reason;
        FAIL() << probe.reason;
    }
    vernon::tests::ResolvedProgram program(device.runtime(), VERNON_VULKAN_CUBE_MAP_BUNDLE);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(device.runtime());
    vernon::tests::LinkedProgram linked(device.runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();

    const std::vector<float> positions{-1.0f, -1.0f, 1.0f, 3.0f, -1.0f, 1.0f, -1.0f, 3.0f, 1.0f};
    const vernon::tests::DeviceTensor vertices(device, positions.data(), positions.size() * sizeof(float), {3, 3},
                                               sizeof(float), {}, 0, VERNON_HAL_BUFFER_VERTEX);
    vernon::tests::HalImage cube(device.get(), VERNON_TEXTURE_RGBA8_UNORM, 1, 1, VERNON_IMAGE_SAMPLED,
                                 VERNON_IMAGE_ASPECT_COLOR, VERNON_TEXTURE_CUBE);
    ASSERT_TRUE(cube);
    constexpr uint8_t face[]{64, 200, 100, 255};
    for (uint32_t layer = 0; layer < 6; ++layer)
        ASSERT_EQ(
            cube.write(face, VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_UINT8, VERNON_IMAGE_ASPECT_COLOR, layer),
            VERNON_HAL_OK)
            << vernon::tests::halError();
    const vernon::tests::HalImage color(device.get(), VERNON_TEXTURE_RGBA8_UNORM, 32, 32,
                                        VERNON_IMAGE_COLOR_ATTACHMENT);
    const vernon::tests::HalImage bloom(device.get(), VERNON_TEXTURE_RGBA8_UNORM, 32, 32,
                                        VERNON_IMAGE_COLOR_ATTACHMENT);
    ASSERT_TRUE(color && bloom);

    std::array<float, 16> identity{};
    for (size_t index = 0; index < 4; ++index)
        identity[index * 5] = 1.0f;
    constexpr uint64_t matrixShape[] = {4, 4};
    constexpr int64_t matrixStrides[] = {4 * sizeof(float), sizeof(float)};
    for (const char *name : {"model", "projection", "view"})
        linked.record().bind(name,
                             vernon::tests::hostView(identity.data(), sizeof(identity), 2, matrixShape, matrixStrides));
    linked.record().bind("aPos", vertices);
    linked.record().bindImage("cubeMap", cube.reference());

    vernon::tests::PassControls pass(color.reference(), 32);
    std::array<VernonColorAttachment, 2> attachments{pass.attachment, pass.attachment};
    attachments[1].location = 1;
    attachments[1].view = bloom.reference();
    pass.renderPass.color_attachments = attachments.data();
    pass.renderPass.color_attachment_count = static_cast<uint32_t>(attachments.size());
    const VernonDrawCommand draw{sizeof(VernonDrawCommand), nullptr, 3, 1, 0};
    pass.writeTo(linked.record(), 0, &draw);
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();

    const size_t center = (16 * 32 + 16) * 4;
    const std::vector<uint8_t> colorPixels = color.rgba8();
    const std::vector<uint8_t> bloomPixels = bloom.rgba8();
    EXPECT_NEAR(colorPixels[center], 64, 10);
    EXPECT_NEAR(colorPixels[center + 1], 200, 10);
    EXPECT_NEAR(colorPixels[center + 2], 100, 10);
    EXPECT_GT(colorPixels[center + 3], 240);
    EXPECT_LT(bloomPixels[center], 10);
    EXPECT_LT(bloomPixels[center + 1], 10);
    EXPECT_LT(bloomPixels[center + 2], 10);
    EXPECT_GT(bloomPixels[center + 3], 240);
}
