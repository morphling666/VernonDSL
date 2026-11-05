#include "VernonRuntime.h"
#include "VernonSchedule.h"
#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#ifndef VERNON_VULKAN_PROGRAM_BUNDLE
#error VERNON_VULKAN_PROGRAM_BUNDLE must name the cooked Program bundle
#endif
#ifndef VERNON_VULKAN_COMPUTE_PROGRAM_BUNDLE
#error VERNON_VULKAN_COMPUTE_PROGRAM_BUNDLE must name the cooked compute Program bundle
#endif
#ifndef VERNON_VULKAN_MULTISET_TEXTURE_BUNDLE
#error VERNON_VULKAN_MULTISET_TEXTURE_BUNDLE must name the cooked multi-set texture Program bundle
#endif
#ifndef VERNON_VULKAN_SPILLED_UNIFORM_BUNDLE
#error VERNON_VULKAN_SPILLED_UNIFORM_BUNDLE must name the cooked spilled uniform Program bundle
#endif

namespace {

using vernon::tests::DeviceTensor;
using vernon::tests::HalBuffer;
using vernon::tests::HalImage;
using vernon::tests::HalSampler;
using vernon::tests::HalTestDevice;
using vernon::tests::LinkedProgram;
using vernon::tests::PassControls;
using vernon::tests::ResolvedProgram;

class VulkanSchedule : public testing::Test {
protected:
    void SetUp() override {
        vernon::tests::BackendTestRequirements requirements;
        requirements.graphics = true;
        const vernon::tests::BackendProbeResult probe = device_.initialize(VERNON_RUNTIME_VULKAN, requirements);
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
        }
    }

    HalTestDevice device_;
};

std::array<uint8_t, 4> texel(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t x, uint32_t y) {
    const size_t index = (static_cast<size_t>(y) * width + x) * 4;
    return {pixels[index], pixels[index + 1], pixels[index + 2], pixels[index + 3]};
}

void expectNear(const std::array<uint8_t, 4> &actual, const std::array<int, 4> &expected, int tolerance) {
    for (size_t channel = 0; channel < actual.size(); ++channel)
        EXPECT_NEAR(actual[channel], expected[channel], tolerance) << "channel " << channel;
}

} // namespace

TEST_F(VulkanSchedule, ResolvesDistinctDescriptorSetsWithRepeatedBindingNumbers) {
    ResolvedProgram program(device_.runtime(), VERNON_VULKAN_MULTISET_TEXTURE_BUNDLE);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(device_.runtime());
    LinkedProgram linked(device_.runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();

    HalImage sampled(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, 1, 1, VERNON_IMAGE_SAMPLED);
    constexpr uint8_t red[]{255, 0, 0, 255};
    ASSERT_EQ(sampled.write(red, VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_UINT8), VERNON_HAL_OK);
    const HalSampler sampler(device_.get(), VERNON_HAL_FILTER_NEAREST);
    constexpr uint32_t extent = 32;
    const HalImage target(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, extent, extent, VERNON_IMAGE_COLOR_ATTACHMENT);
    const PassControls pass(target.reference(), extent);
    const VernonDrawCommand draw{sizeof(VernonDrawCommand), nullptr, 3, 1, 0};
    linked.record().bindImage("vertex_image", sampled.reference());
    linked.record().bindImage("fragment_image", sampled.reference());
    linked.record().bindSampler("vertex_sampler", sampler.reference());
    pass.writeTo(linked.record(), 0, &draw);
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    expectNear(texel(target.rgba8(), extent, extent / 2, extent / 2), {255, 0, 0, 255}, 1);
}

// Each call redraws into the attachments, viewport and sampled texels the caller gives it, through the schedule it
// recorded once.
TEST_F(VulkanSchedule, RedrawsIntoTheAttachmentsOfEachCall) {
    ResolvedProgram program(device_.runtime(), VERNON_VULKAN_PROGRAM_BUNDLE);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(device_.runtime());
    LinkedProgram linked(device_.runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();

    const std::vector<float> positions{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    const DeviceTensor vertices(device_, positions.data(), positions.size() * sizeof(float), {3, 2}, sizeof(float), {},
                                0, VERNON_HAL_BUFFER_VERTEX);
    constexpr uint32_t indices[]{0, 1, 2};
    const HalBuffer indexBuffer(device_.get(), sizeof(indices), indices, VERNON_HAL_BUFFER_INDEX);
    HalImage sampled(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, 1, 1, VERNON_IMAGE_SAMPLED);
    constexpr uint8_t firstTexel[]{64, 200, 100, 255};
    ASSERT_EQ(sampled.write(firstTexel, VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_UINT8), VERNON_HAL_OK);
    const HalSampler sampler(device_.get(), VERNON_HAL_FILTER_NEAREST);
    const HalImage firstTarget(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, 32, 32, VERNON_IMAGE_COLOR_ATTACHMENT);
    const HalImage secondTarget(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, 48, 24, VERNON_IMAGE_COLOR_ATTACHMENT);
    const HalImage firstDepth(device_.get(), VERNON_TEXTURE_D32_FLOAT, 32, 32, VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
                              VERNON_IMAGE_ASPECT_DEPTH);
    const HalImage secondDepth(device_.get(), VERNON_TEXTURE_D32_FLOAT, 48, 24, VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
                               VERNON_IMAGE_ASPECT_DEPTH);
    ASSERT_TRUE(firstTarget && secondTarget && firstDepth && secondDepth);

    PassControls pass(firstTarget.reference(), 32);
    pass.attachment.clear_color.f32[3] = 0.0f;
    VernonDepthAttachment depth{};
    depth.view = firstDepth.reference();
    depth.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    depth.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    depth.clear_depth = 1.0f;
    pass.renderPass.depth_attachment = &depth;
    const VernonIndexBinding indexBinding{VERNON_INDEX_U32, 0, 3, indexBuffer.reference(), 0};
    const VernonDrawCommand draw{sizeof(VernonDrawCommand), &indexBinding, 0, 1, 0};
    linked.record().bind("position", vertices);
    linked.record().bindImage("image", sampled.reference());
    linked.record().bindSampler("sampler", sampler.reference());
    pass.writeTo(linked.record(), 0, &draw);
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    EXPECT_EQ(linked.telemetry().draws, 1u);
    expectNear(texel(firstTarget.rgba8(), 32, 16, 16), {64, 200, 100, 255}, 10);

    constexpr uint8_t secondTexel[]{180, 40, 220, 255};
    ASSERT_EQ(sampled.write(secondTexel, VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_UINT8), VERNON_HAL_OK);
    pass.attachment.view = secondTarget.reference();
    depth.view = secondDepth.reference();
    pass.renderPass.render_area[2] = 48;
    pass.renderPass.render_area[3] = 24;
    pass.dynamic.viewport[0] = 4;
    pass.dynamic.viewport[1] = 3;
    pass.dynamic.viewport[2] = 24;
    pass.dynamic.viewport[3] = 12;
    pass.dynamic.scissor[2] = 48;
    pass.dynamic.scissor[3] = 24;
    pass.writeTo(linked.record(), 0, &draw);
    for (int call = 0; call < 2; ++call)
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    const std::vector<uint8_t> second = secondTarget.rgba8();
    expectNear(texel(second, 48, 16, 9), {180, 40, 220, 255}, 10);
    expectNear(texel(second, 48, 1, 1), {0, 0, 0, 0}, 0);
    expectNear(texel(firstTarget.rgba8(), 32, 16, 16), {64, 200, 100, 255}, 10);
}

// The fragment's three matrices exceed the push constant budget, so they spill to uniform memory that each call
// rewrites.
TEST_F(VulkanSchedule, SpilledUniformsTakeTheValuesOfEachCall) {
    ResolvedProgram program(device_.runtime(), VERNON_VULKAN_SPILLED_UNIFORM_BUNDLE);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(device_.runtime());
    LinkedProgram linked(device_.runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();

    const std::vector<float> positions{-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    const DeviceTensor vertices(device_, positions.data(), positions.size() * sizeof(float), {3, 2}, sizeof(float), {},
                                0, VERNON_HAL_BUFFER_VERTEX);
    const HalImage target(device_.get(), VERNON_TEXTURE_RGBA8_UNORM, 8, 8, VERNON_IMAGE_COLOR_ATTACHMENT);
    const PassControls pass(target.reference(), 8);
    const VernonDrawCommand draw{sizeof(VernonDrawCommand), nullptr, 3, 1, 0};

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
    constexpr uint64_t shape[] = {4, 4};
    constexpr int64_t strides[] = {16, 4};
    linked.record().bind("first", vernon::tests::hostView(first.data(), sizeof(Matrix), 2, shape, strides));
    linked.record().bind("second", vernon::tests::hostView(second.data(), sizeof(Matrix), 2, shape, strides));
    linked.record().bind("third", vernon::tests::hostView(third.data(), sizeof(Matrix), 2, shape, strides));
    linked.record().bind("position", vertices);
    pass.writeTo(linked.record(), 0, &draw);
    const auto drawsColor = [&](float x, float y, float z, std::array<int, 4> expected) {
        third = translation(x, y, z);
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
        expectNear(texel(target.rgba8(), 8, 4, 4), expected, 2);
    };
    drawsColor(1.0f, 1.0f, 0.75f, {128, 64, 191, 255});
    for (int repetition = 0; repetition < 8; ++repetition) {
        drawsColor(0.0f, 1.0f, 1.0f, {0, 64, 255, 255});
        drawsColor(1.0f, 0.0f, 0.5f, {128, 0, 128, 255});
    }
}

TEST_F(VulkanSchedule, DispatchesIntoTheViewOfEachCall) {
    ResolvedProgram program(device_.runtime(), VERNON_VULKAN_COMPUTE_PROGRAM_BUNDLE);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(device_.runtime());
    LinkedProgram linked(device_.runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();

    const std::vector<float> source{0, 1, 2, 3};
    const DeviceTensor dense(device_, source);
    const DeviceTensor strided(device_, source.data(), source.size() * sizeof(float), {2}, sizeof(float),
                               {2 * sizeof(float)}, sizeof(float));
    float factor = 3.0f;
    constexpr uint64_t scalarShape[] = {1};
    constexpr int64_t scalarStrides[] = {sizeof(float)};
    linked.record().bind("values", dense);
    linked.record().bind("factor", vernon::tests::hostView(&factor, sizeof(factor), 1, scalarShape, scalarStrides));
    linked.record().bindGrid(4);
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    linked.record().bind("values", strided);
    linked.record().bindGrid(2);
    factor = 5.0f;
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();

    const std::vector<float> first = dense.values<float>();
    for (size_t index = 0; index < first.size(); ++index)
        EXPECT_EQ(first[index], source[index] * 3.0f);
    EXPECT_EQ(strided.values<float>(), (std::vector<float>{0, 5, 2, 15}));
}
