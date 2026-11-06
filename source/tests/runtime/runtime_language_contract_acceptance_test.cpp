#include "VernonFormats.hpp"
#include "acceptance_descriptor.h"
#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using vernon::tests::AcceptanceDescriptor;
using vernon::tests::DeviceTensor;
using vernon::tests::HalImage;
using vernon::tests::HalSampler;
using vernon::tests::HalTestDevice;
using vernon::tests::LinkedProgram;
using vernon::tests::LinkedVjp;
using vernon::tests::PassControls;
using vernon::tests::ProgramFixtureManifest;
using vernon::tests::ResolvedProgram;

#include "language_contract_acceptance_descriptors.inc"

bool isAcceptanceFixture(std::string_view fixtureId) {
    return std::any_of(acceptanceDescriptors, std::end(acceptanceDescriptors),
                       [&](const AcceptanceDescriptor &acceptance) {
                           return acceptance.suite == "language_contract" && acceptance.fixtureId == fixtureId;
                       });
}

const AcceptanceDescriptor &acceptanceFor(std::string_view fixtureId) {
    const auto *result = std::find_if(
        acceptanceDescriptors, std::end(acceptanceDescriptors), [&](const AcceptanceDescriptor &acceptance) {
            return acceptance.suite == "language_contract" && acceptance.fixtureId == fixtureId;
        });
    if (result == std::end(acceptanceDescriptors))
        throw std::logic_error("unknown acceptance fixture");
    return *result;
}

std::vector<ProgramFixtureManifest> acceptanceFixtures() {
    std::vector<ProgramFixtureManifest> fixtures;
    for (const ProgramFixtureManifest &fixture : vernon::tests::programFixtureManifestTable)
        if (isAcceptanceFixture(fixture.fixtureId))
            fixtures.push_back(fixture);
    return fixtures;
}

vernon::tests::BackendTestRequirements requirementsFor(const ProgramFixtureManifest &fixture) {
    const AcceptanceDescriptor &acceptance = acceptanceFor(fixture.fixtureId);
    vernon::tests::BackendTestRequirements requirements;
    requirements.compute = acceptance.compute;
    requirements.graphics = acceptance.graphics;
    requirements.storageBuffers = acceptance.storageBuffers;
    requirements.storageTexture = acceptance.storageTexture;
    requirements.deviceAtomics = acceptance.deviceAtomics;
    requirements.f32AtomicAdd = acceptance.f32AtomicAdd;
    requirements.textureSamplerOperations = acceptance.textureSamplerOperations;
    requirements.programVjp = acceptance.programVjp;
    if (requirements.compute)
        vernon::tests::setOpenGLComputeApiRequirement(requirements, fixture.runtime);
    // The oracle reads depth and stencil back, which OpenGL ES does from 3.1.
    if (acceptance.oracle == "fragment_discard" && fixture.runtime == VERNON_RUNTIME_OPENGL_ES)
        vernon::tests::setOpenGLComputeApiRequirement(requirements, fixture.runtime);
    return requirements;
}

std::string testName(const testing::TestParamInfo<ProgramFixtureManifest> &info) {
    std::string result = std::string(info.param.fixtureId) + "_" + std::string(info.param.target);
    for (char &value : result)
        if (!std::isalnum(static_cast<unsigned char>(value)))
            value = '_';
    return result;
}

void runFloatBufferOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                          const AcceptanceDescriptor &acceptance) {
    const std::vector<float> expected(acceptance.expected, acceptance.expected + acceptance.expectedCount);
    const DeviceTensor buffer(device, std::vector<float>(expected.size()));
    LinkedProgram schedule(device.runtime(), executable);
    ASSERT_TRUE(schedule) << schedule.error();
    ASSERT_TRUE(schedule.record().bind(acceptance.parameter, buffer));
    schedule.record().bindGrid(acceptance.grid.x, acceptance.grid.y, acceptance.grid.z);
    ASSERT_TRUE(schedule.record().unbound().empty());
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();
    EXPECT_EQ(buffer.values<float>(), expected);
}

void runSignedStrideVjpOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                              const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 5u);
    LinkedVjp vjp(device.runtime(), executable);
    ASSERT_TRUE(vjp) << vjp.error();
    // The source view walks two of the four floats backwards from the last.
    const std::vector<int64_t> reversed{-2 * static_cast<int64_t>(sizeof(float))};
    const DeviceTensor source(device, std::vector<float>{10, 2, 20, 3}, {2}, reversed, 3 * sizeof(float));
    const DeviceTensor output(device, std::vector<float>(1));
    ASSERT_TRUE(vjp.forwardRecord().bind("source", source));
    ASSERT_TRUE(vjp.forwardRecord().bind("output", output));
    vjp.forwardRecord().bindGrid(1);
    ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_FLOAT_EQ(output.values<float>()[0], static_cast<float>(acceptance.expected[0]));

    const DeviceTensor cotangent(device, std::vector<float>{1});
    const DeviceTensor gradient(device, std::vector<float>(4), {2}, reversed, 3 * sizeof(float));
    ASSERT_TRUE(vjp.backwardRecord().bind("output", cotangent, VERNON_PROGRAM_BOUNDARY_COTANGENT));
    ASSERT_TRUE(vjp.backwardRecord().bind("source", gradient, VERNON_PROGRAM_BOUNDARY_GRADIENT));
    ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
    const std::vector<float> gradients = gradient.values<float>();
    for (size_t index = 0; index < gradients.size(); ++index)
        EXPECT_FLOAT_EQ(gradients[index], static_cast<float>(acceptance.expected[index + 1]));
}

void runFanOutVjpOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                        const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 3u);
    LinkedVjp vjp(device.runtime(), executable);
    ASSERT_TRUE(vjp) << vjp.error();
    const DeviceTensor source(device, std::vector<float>{2});
    const DeviceTensor square(device, std::vector<float>(1));
    const DeviceTensor cube(device, std::vector<float>(1));
    ASSERT_TRUE(vjp.forwardRecord().bind("source", source));
    ASSERT_TRUE(vjp.forwardRecord().bind("square", square));
    ASSERT_TRUE(vjp.forwardRecord().bind("cube", cube));
    vjp.forwardRecord().bindGrid(1);
    ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_FLOAT_EQ(square.values<float>()[0], static_cast<float>(acceptance.expected[0]));
    EXPECT_FLOAT_EQ(cube.values<float>()[0], static_cast<float>(acceptance.expected[1]));

    const DeviceTensor squareCotangent(device, std::vector<float>{1});
    const DeviceTensor cubeCotangent(device, std::vector<float>{2});
    const DeviceTensor gradient(device, std::vector<float>(1));
    ASSERT_TRUE(vjp.backwardRecord().bind("square", squareCotangent));
    ASSERT_TRUE(vjp.backwardRecord().bind("cube", cubeCotangent));
    ASSERT_TRUE(vjp.backwardRecord().bind("source", gradient));
    ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
    EXPECT_FLOAT_EQ(gradient.values<float>()[0], static_cast<float>(acceptance.expected[2]));
}

void runStructuredViewOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                             const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 4u);
    struct Pair {
        float left;
        float right;
    };
    struct Payload {
        float value;
        int32_t index;
    };
    const std::array<Pair, 2> pairValues{{{1, 2}, {5, 7}}};
    const std::array<Payload, 1> payloadValues{{{3, 4}}};
    const DeviceTensor matrix(device, std::vector<float>{1, 2, 3, 4}, {2, 2});
    const DeviceTensor pairs(device, pairValues.data(), sizeof(pairValues), {2}, sizeof(Pair));
    const DeviceTensor payloads(device, payloadValues.data(), sizeof(payloadValues), {1}, sizeof(Payload));
    const DeviceTensor output(device, std::vector<float>(4));
    LinkedProgram schedule(device.runtime(), executable);
    ASSERT_TRUE(schedule) << schedule.error();
    ASSERT_TRUE(schedule.record().bind("matrix", matrix));
    ASSERT_TRUE(schedule.record().bind("pairs", pairs));
    ASSERT_TRUE(schedule.record().bind("payloads", payloads));
    ASSERT_TRUE(schedule.record().bind("output", output));
    schedule.record().bindGrid(1);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();
    const std::vector<float> values = output.values<float>();
    for (size_t index = 0; index < values.size(); ++index)
        EXPECT_FLOAT_EQ(values[index], static_cast<float>(acceptance.expected[index]));
    std::array<Pair, 2> pairResult{};
    pairs.buffer().read(pairResult.data(), sizeof(pairResult));
    EXPECT_FLOAT_EQ(pairResult[0].left, 5);
    EXPECT_FLOAT_EQ(pairResult[0].right, 7);
    std::array<Payload, 1> payloadResult{};
    payloads.buffer().read(payloadResult.data(), sizeof(payloadResult));
    EXPECT_FLOAT_EQ(payloadResult[0].value, 4);
    EXPECT_EQ(payloadResult[0].index, 6);
}

constexpr uint32_t kExtent = 32;

std::array<uint8_t, 4> centerPixel(const std::vector<uint8_t> &pixels, uint32_t extent = kExtent) {
    const size_t center = (extent / 2 * extent + extent / 2) * 4;
    return {pixels[center], pixels[center + 1], pixels[center + 2], pixels[center + 3]};
}

// Draws the Program's three vertices into a clear opaque black target of `format`, read back as 8-bit RGBA.
std::array<uint8_t, 4> renderArgumentFreeTriangle(const HalTestDevice &device, VernonProgramExecutable *executable,
                                                  VernonTextureFormat format = VERNON_TEXTURE_RGBA8_UNORM,
                                                  uint32_t firstInstance = 0,
                                                  const VernonIndexBinding *indexBinding = nullptr) {
    EXPECT_EQ(vernonRuntimeProgramExecutableGetParameterCount(executable), 0u);
    const HalImage target(device.get(), format, kExtent, kExtent, VERNON_IMAGE_COLOR_ATTACHMENT);
    LinkedProgram schedule(device.runtime(), executable);
    EXPECT_TRUE(schedule) << schedule.error();
    if (!schedule || !target)
        return {};
    const PassControls controls(target.reference(), kExtent);
    VernonDrawCommand draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    draw.first_instance = firstInstance;
    draw.index_binding = indexBinding;
    controls.writeTo(schedule.record(), 0, &draw);
    EXPECT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();
    return centerPixel(target.rgba8());
}

void runGraphicsTriangleOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                               const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 4u);
    const std::array<uint8_t, 4> pixel = renderArgumentFreeTriangle(device, executable);
    for (size_t channel = 0; channel < pixel.size(); ++channel)
        EXPECT_NEAR(pixel[channel], acceptance.expected[channel], 1);
}

// The raw and generated contract vertices move every instance but instance 0 off the target, so a draw whose first
// instance is 1 leaves the clear color wherever `instance_index` includes the first instance.
void runFirstInstanceOracle(const HalTestDevice &device, VernonProgramExecutable *executable) {
    EXPECT_EQ(renderArgumentFreeTriangle(device, executable, VERNON_TEXTURE_RGBA8_UNORM, 1),
              (std::array<uint8_t, 4>{0, 0, 0, 255}));
}

// The contract vertices collapse every `vertex_index` from 2 up onto one point, so indices 3..5 drawn with base vertex
// -3 cover the center only where `vertex_index` includes the base vertex.
void runBaseVertexOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                         const AcceptanceDescriptor &acceptance) {
    const uint32_t indices[] = {3, 4, 5};
    const vernon::tests::HalBuffer indexBuffer(device.get(), sizeof(indices), indices, VERNON_HAL_BUFFER_INDEX);
    ASSERT_TRUE(indexBuffer);
    const VernonIndexBinding binding{VERNON_INDEX_U32, 0, 3, indexBuffer.reference(), -3};
    const std::array<uint8_t, 4> pixel =
        renderArgumentFreeTriangle(device, executable, VERNON_TEXTURE_RGBA8_UNORM, 0, &binding);
    for (size_t channel = 0; channel < pixel.size(); ++channel)
        EXPECT_NEAR(pixel[channel], acceptance.expected[channel], 1);
}

// The asset covers the target with one triangle and discards fragments whose (x + y) parity is odd.
void runFragmentDiscardOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                              const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 4u);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterCount(executable), 0u);
    constexpr uint32_t stencilReference = 7;
    const HalImage target(device.get(), VERNON_TEXTURE_RGBA8_UNORM, kExtent, kExtent, VERNON_IMAGE_COLOR_ATTACHMENT);
    const HalImage depthImage(device.get(), VERNON_TEXTURE_D32_FLOAT_S8_UINT, kExtent, kExtent,
                              VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
                              VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
    ASSERT_TRUE(target);
    ASSERT_TRUE(depthImage);

    PassControls controls(target.reference(), kExtent);
    controls.attachment.clear_color.f32[3] = 0.0f;
    VernonDepthAttachment depth{};
    depth.view = depthImage.reference();
    depth.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    depth.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    depth.clear_depth = 1.0f;
    depth.stencil_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    depth.stencil_store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    controls.renderPass.depth_attachment = &depth;
    controls.dynamic.stencil_reference = stencilReference;
    VernonDrawCommand draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    LinkedProgram schedule(device.runtime(), executable);
    ASSERT_TRUE(schedule) << schedule.error();
    controls.writeTo(schedule.record(), 0, &draw);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();

    const std::vector<uint8_t> pixels = target.rgba8();
    const std::vector<float> depths = depthImage.read<float>(VERNON_HAL_IMAGE_DATA_DEPTH, VERNON_HAL_IMAGE_DATA_FLOAT32,
                                                             sizeof(float), VERNON_IMAGE_ASPECT_DEPTH);
    const std::vector<uint8_t> stencils = depthImage.read<uint8_t>(
        VERNON_HAL_IMAGE_DATA_STENCIL, VERNON_HAL_IMAGE_DATA_UINT8, 1, VERNON_IMAGE_ASPECT_STENCIL);
    constexpr uint32_t keptParity = 0;
    size_t kept = 0;
    for (uint32_t y = 0; y < kExtent; ++y) {
        for (uint32_t x = 0; x < kExtent; ++x) {
            SCOPED_TRACE("pixel " + std::to_string(x) + "," + std::to_string(y));
            const size_t index = static_cast<size_t>(y) * kExtent + x;
            const uint8_t *pixel = pixels.data() + index * 4;
            if ((x + y) % 2 == keptParity) {
                ++kept;
                for (size_t channel = 0; channel < 4; ++channel)
                    ASSERT_NEAR(pixel[channel], acceptance.expected[channel], 1);
                ASSERT_LT(depths[index], 1.0f);
                ASSERT_EQ(stencils[index], stencilReference);
            } else {
                for (size_t channel = 0; channel < 4; ++channel)
                    ASSERT_EQ(pixel[channel], 0);
                ASSERT_EQ(depths[index], 1.0f);
                ASSERT_EQ(stencils[index], 0);
            }
        }
    }
    EXPECT_EQ(kept, kExtent * kExtent / 2);
}

void renderSpecializedVariant(const HalTestDevice &device, const ProgramFixtureManifest &fixture,
                              const VernonProgramSpecialization &specialization, double expectedRed,
                              VernonTextureFormat format = VERNON_TEXTURE_RGBA8_UNORM) {
    const VernonProgramVariantSelector selector{sizeof(VernonProgramVariantSelector), &specialization, 1, {}};
    const ResolvedProgram program(device.runtime(), std::string(fixture.manifestPath), &selector);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(device.runtime());
    const std::array<uint8_t, 4> pixel = renderArgumentFreeTriangle(device, program.get(), format);
    EXPECT_NEAR(pixel[0], expectedRed, 1);
    EXPECT_LT(pixel[1], 10);
}

// `expected` holds (value, red channel) pairs; each value selects one cooked variant through the u32
// specialization named by the descriptor.
void runSpecializationPixelsOracle(const HalTestDevice &device, const ProgramFixtureManifest &fixture,
                                   const AcceptanceDescriptor &acceptance) {
    ASSERT_FALSE(acceptance.specialization.empty());
    ASSERT_GT(acceptance.expectedCount, 0u);
    ASSERT_EQ(acceptance.expectedCount % 2, 0u);
    for (size_t index = 0; index < acceptance.expectedCount; index += 2) {
        const auto value = static_cast<uint32_t>(acceptance.expected[index]);
        SCOPED_TRACE(std::string(acceptance.specialization) + "=" + std::to_string(value));
        VernonProgramSpecialization specialization{};
        specialization.struct_size = sizeof(specialization);
        specialization.name = {acceptance.specialization.data(), acceptance.specialization.size()};
        specialization.kind = VERNON_PROGRAM_SPECIALIZATION_U32;
        specialization.value.u32_value = value;
        ASSERT_NO_FATAL_FAILURE(
            renderSpecializedVariant(device, fixture, specialization, acceptance.expected[index + 1]));
    }
}

// `variantValues[i]` is the type identity selecting one cooked variant through the type specialization
// named by the descriptor, and `expected[i]` is the red channel that variant renders.
void runTypeSpecializationPixelsOracle(const HalTestDevice &device, const ProgramFixtureManifest &fixture,
                                       const AcceptanceDescriptor &acceptance) {
    ASSERT_FALSE(acceptance.specialization.empty());
    ASSERT_GT(acceptance.variantValueCount, 0u);
    ASSERT_EQ(acceptance.expectedCount, acceptance.variantValueCount);
    for (size_t index = 0; index < acceptance.variantValueCount; ++index) {
        const std::string_view identity = acceptance.variantValues[index];
        SCOPED_TRACE(std::string(acceptance.specialization) + "=" + std::string(identity));
        VernonProgramSpecialization specialization{};
        specialization.struct_size = sizeof(specialization);
        specialization.name = {acceptance.specialization.data(), acceptance.specialization.size()};
        specialization.kind = VERNON_PROGRAM_SPECIALIZATION_TYPE;
        specialization.value.type_value = {identity.data(), identity.size()};
        ASSERT_NO_FATAL_FAILURE(renderSpecializedVariant(device, fixture, specialization, acceptance.expected[index]));
    }
}

// `variantValues[i]` names the color target format selecting one cooked variant through the format specialization
// named by the descriptor; the variant renders into an image of that format, and `expected[i]` is its red channel.
void runFormatSpecializationPixelsOracle(const HalTestDevice &device, const ProgramFixtureManifest &fixture,
                                         const AcceptanceDescriptor &acceptance) {
    ASSERT_FALSE(acceptance.specialization.empty());
    ASSERT_GT(acceptance.variantValueCount, 0u);
    ASSERT_EQ(acceptance.expectedCount, acceptance.variantValueCount);
    for (size_t index = 0; index < acceptance.variantValueCount; ++index) {
        const std::string_view name = acceptance.variantValues[index];
        SCOPED_TRACE(std::string(acceptance.specialization) + "=" + std::string(name));
        const vernon::formats::Format *format = vernon::formats::find(name);
        ASSERT_NE(format, nullptr);
        VernonProgramSpecialization specialization{};
        specialization.struct_size = sizeof(specialization);
        specialization.name = {acceptance.specialization.data(), acceptance.specialization.size()};
        specialization.kind = VERNON_PROGRAM_SPECIALIZATION_FORMAT;
        specialization.value.format_value = format->texture;
        ASSERT_NO_FATAL_FAILURE(
            renderSpecializedVariant(device, fixture, specialization, acceptance.expected[index], format->texture));
    }
}

// The asset samples a one-texel red image across the target through its Image and Sampler parameters.
void runSampledPixelOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                           const AcceptanceDescriptor &acceptance) {
    constexpr uint8_t sampledPixel[]{255, 0, 0, 255};
    HalImage sampled(device.get(), VERNON_TEXTURE_RGBA8_UNORM, 1, 1, VERNON_IMAGE_SAMPLED);
    ASSERT_TRUE(sampled);
    ASSERT_EQ(sampled.write(sampledPixel, VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_UINT8), VERNON_HAL_OK)
        << vernon::tests::halError();
    const HalSampler sampler(device.get(), VERNON_HAL_FILTER_NEAREST);
    ASSERT_TRUE(sampler);
    const HalImage target(device.get(), VERNON_TEXTURE_RGBA8_UNORM, kExtent, kExtent, VERNON_IMAGE_COLOR_ATTACHMENT);
    ASSERT_TRUE(target);

    LinkedProgram schedule(device.runtime(), executable);
    ASSERT_TRUE(schedule) << schedule.error();
    for (size_t index = 0; index < vernonRuntimeProgramExecutableGetParameterCount(executable); ++index) {
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &parameter), VERNON_STATUS_OK);
        const std::string_view name(parameter.name.data, parameter.name.size);
        if (parameter.kind == VERNON_PROGRAM_IMAGE) {
            VernonProgramImageConstraintView constraint{};
            constraint.struct_size = sizeof(constraint);
            ASSERT_EQ(vernonRuntimeProgramExecutableGetImageConstraintByParameterIndex(executable, index, &constraint),
                      VERNON_STATUS_OK);
            EXPECT_EQ(constraint.binding_role, VERNON_IMAGE_BINDING_SAMPLED);
            EXPECT_EQ(constraint.sample_result_class, VERNON_IMAGE_SAMPLE_FLOAT);
            ASSERT_TRUE(schedule.record().bindImage(name, sampled.reference()));
        } else {
            ASSERT_EQ(parameter.kind, VERNON_PROGRAM_SAMPLER);
            ASSERT_TRUE(schedule.record().bindSampler(name, sampler.reference()));
        }
    }
    const PassControls controls(target.reference(), kExtent);
    VernonDrawCommand draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    controls.writeTo(schedule.record(), 0, &draw);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();
    const std::array<uint8_t, 4> actual = centerPixel(target.rgba8());
    ASSERT_EQ(acceptance.expectedCount, actual.size());
    for (size_t channel = 0; channel < actual.size(); ++channel)
        EXPECT_NEAR(actual[channel], acceptance.expected[channel], 1);
}

void runStorageTexelOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                           const AcceptanceDescriptor &acceptance) {
    const HalImage image(device.get(), VERNON_TEXTURE_RGBA32_FLOAT, 1, 1, VERNON_IMAGE_STORAGE);
    ASSERT_TRUE(image);
    LinkedProgram schedule(device.runtime(), executable);
    ASSERT_TRUE(schedule) << schedule.error();
    ASSERT_TRUE(schedule.record().bindImage("image", image.reference()));
    schedule.record().bindGrid(1);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();
    const std::vector<float> texel =
        image.read<float>(VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_FLOAT32, 4 * sizeof(float));
    ASSERT_EQ(acceptance.expectedCount, texel.size());
    for (size_t channel = 0; channel < texel.size(); ++channel)
        EXPECT_FLOAT_EQ(texel[channel], static_cast<float>(acceptance.expected[channel]));
}

// The asset samples a 4x1 texture halfway between neighbouring texel centers and at the right edge, so only linear
// filtering with clamp-to-edge addressing from the bound Sampler yields `expected`.
void runComputeSampledTexelsOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                                   const AcceptanceDescriptor &acceptance) {
    constexpr uint32_t extent = 4;
    constexpr std::array<uint8_t, extent * 4> texels{0, 0, 0, 255, 51, 0, 0, 255, 102, 0, 0, 255, 255, 0, 0, 255};
    HalImage image(device.get(), VERNON_TEXTURE_RGBA8_UNORM, extent, 1, VERNON_IMAGE_SAMPLED);
    ASSERT_TRUE(image);
    ASSERT_EQ(image.write(texels.data(), VERNON_HAL_IMAGE_DATA_RGBA, VERNON_HAL_IMAGE_DATA_UINT8), VERNON_HAL_OK)
        << vernon::tests::halError();
    const HalSampler sampler(device.get(), VERNON_HAL_FILTER_LINEAR);
    ASSERT_TRUE(sampler);
    const DeviceTensor output(device, std::vector<float>(acceptance.expectedCount));
    LinkedProgram schedule(device.runtime(), executable);
    ASSERT_TRUE(schedule) << schedule.error();
    ASSERT_TRUE(schedule.record().bind(acceptance.parameter, output));
    ASSERT_TRUE(schedule.record().bindImage("image", image.reference()));
    ASSERT_TRUE(schedule.record().bindSampler("image_sampler", sampler.reference()));
    schedule.record().bindGrid(acceptance.grid.x, acceptance.grid.y, acceptance.grid.z);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();
    const std::vector<float> values = output.values<float>();
    for (size_t index = 0; index < values.size(); ++index)
        EXPECT_NEAR(values[index], acceptance.expected[index], 2e-3) << "invocation " << index;
}

// `expected` holds the 4x4 source texels in row order. The asset covers the target and writes, at each fragment, the
// texel addressed by its window coordinates.
void runTexelFetchOracle(const HalTestDevice &device, VernonProgramExecutable *executable,
                         const AcceptanceDescriptor &acceptance) {
    constexpr uint32_t extent = 4;
    ASSERT_EQ(acceptance.expectedCount, extent * extent);
    std::vector<uint32_t> texels(acceptance.expectedCount);
    std::transform(acceptance.expected, acceptance.expected + acceptance.expectedCount, texels.begin(),
                   [](double value) { return static_cast<uint32_t>(value); });
    HalImage source(device.get(), VERNON_TEXTURE_R32_UINT, extent, extent, VERNON_IMAGE_SAMPLED);
    ASSERT_TRUE(source);
    ASSERT_EQ(source.write(texels.data(), VERNON_HAL_IMAGE_DATA_RED, VERNON_HAL_IMAGE_DATA_UINT32), VERNON_HAL_OK)
        << vernon::tests::halError();
    const HalImage target(device.get(), VERNON_TEXTURE_R32_UINT, extent, extent, VERNON_IMAGE_COLOR_ATTACHMENT);
    ASSERT_TRUE(target);

    LinkedProgram schedule(device.runtime(), executable);
    ASSERT_TRUE(schedule) << schedule.error();
    ASSERT_TRUE(schedule.record().bindImage(acceptance.parameter, source.reference()));
    PassControls controls(target.reference(), extent);
    controls.attachment.clear_color = {};
    VernonDrawCommand draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    controls.writeTo(schedule.record(), 0, &draw);
    ASSERT_EQ(schedule.call(), VERNON_STATUS_OK) << schedule.error();
    EXPECT_EQ(target.read<uint32_t>(VERNON_HAL_IMAGE_DATA_RED, VERNON_HAL_IMAGE_DATA_UINT32, sizeof(uint32_t)), texels);
}

class LanguageContractAcceptance : public vernon::tests::HalProgramTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override { return requirementsFor(GetParam()); }
};

TEST_P(LanguageContractAcceptance, LoadsResolvesAndExecutesCanonicalProgramAsset) {
    const ProgramFixtureManifest fixture = GetParam();
    const AcceptanceDescriptor &acceptance = acceptanceFor(fixture.fixtureId);
    if (acceptance.oracle == "specialization_pixels")
        return runSpecializationPixelsOracle(device_, fixture, acceptance);
    if (acceptance.oracle == "type_specialization_pixels")
        return runTypeSpecializationPixelsOracle(device_, fixture, acceptance);
    if (acceptance.oracle == "format_specialization_pixels")
        return runFormatSpecializationPixelsOracle(device_, fixture, acceptance);
    const ResolvedProgram program(runtime(), std::string(fixture.manifestPath));
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    if (acceptance.oracle == "float_buffer") {
        runFloatBufferOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "graphics_triangle") {
        runGraphicsTriangleOracle(device_, program.get(), acceptance);
        const std::string_view id = fixture.fixtureId;
        if (id.find("graphics_raw") != std::string_view::npos ||
            id.find("graphics_generated") != std::string_view::npos) {
            runFirstInstanceOracle(device_, program.get());
            runBaseVertexOracle(device_, program.get(), acceptance);
        }
    } else if (acceptance.oracle == "fragment_discard") {
        runFragmentDiscardOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "sampled_pixel") {
        runSampledPixelOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "storage_texel") {
        runStorageTexelOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "compute_sampled_texels") {
        runComputeSampledTexelsOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "texel_fetch") {
        runTexelFetchOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "signed_stride_vjp") {
        runSignedStrideVjpOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "fan_out_vjp") {
        runFanOutVjpOracle(device_, program.get(), acceptance);
    } else if (acceptance.oracle == "structured_view") {
        runStructuredViewOracle(device_, program.get(), acceptance);
    } else {
        FAIL() << "oracle '" << acceptance.oracle << "' has no executor";
    }
}

INSTANTIATE_TEST_SUITE_P(CanonicalMatrix, LanguageContractAcceptance, testing::ValuesIn(acceptanceFixtures()),
                         testName);

} // namespace
