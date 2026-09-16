#include "../support/acceptance_descriptor.h"
#include "../support/program_fixture_runtime_test.h"
#include "../support/sampled_texture_runtime_oracle.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using vernon::tests::AcceptanceDescriptor;
using vernon::tests::ProgramFixtureManifest;

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

vernon::tests::BackendTestRow backendFor(const ProgramFixtureManifest &fixture) {
    for (const auto &backend : vernon::tests::backendTestMatrix)
        if (backend.runtime == fixture.runtime)
            return backend;
    return {};
}

std::vector<ProgramFixtureManifest> acceptanceFixtures() {
    std::vector<ProgramFixtureManifest> fixtures;
    for (const ProgramFixtureManifest &fixture : vernon::tests::programFixtureManifestTable)
        if (isAcceptanceFixture(fixture.fixtureId))
            fixtures.push_back(fixture);
    return fixtures;
}

std::vector<ProgramFixtureManifest> acceptanceFixtures(bool cpu) {
    std::vector<ProgramFixtureManifest> fixtures;
    for (const ProgramFixtureManifest &fixture : acceptanceFixtures())
        if ((fixture.runtime == VERNON_RUNTIME_CPU) == cpu)
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
    return requirements;
}

std::string testName(const testing::TestParamInfo<ProgramFixtureManifest> &info) {
    std::string result = std::string(info.param.fixtureId) + "_" + std::string(info.param.target);
    for (char &value : result)
        if (!std::isalnum(static_cast<unsigned char>(value)))
            value = '_';
    return result;
}

bool skipDirectXWorkgroupAtomicContention(const ProgramFixtureManifest &fixture) {
    const char *skip = std::getenv("VERNON_TEST_SKIP_DIRECTX_WORKGROUP_ATOMIC_CONTENTION");
    return skip && std::strcmp(skip, "1") == 0 && fixture.runtime == VERNON_RUNTIME_DIRECTX12 &&
           fixture.fixtureId == "floating_contention";
}

void runFloatBufferOracle(vernon::tests::OwnedRhiRuntime &owned, VernonProgramExecutable *executable,
                          const AcceptanceDescriptor &acceptance) {
    const std::vector<float> expected(acceptance.expected, acceptance.expected + acceptance.expectedCount);
    VernonProgramParameterView parameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(
                  executable, {acceptance.parameter.data(), acceptance.parameter.size()}, &parameter),
              VERNON_STATUS_OK);

    std::vector<float> initial(expected.size());
    vernon::tests::RhiBuffer buffer = vernon::tests::createBuffer(
        owned.context(), initial.size() * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE, initial.data());
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const uint64_t shape[]{expected.size()};
    const int64_t strides[]{sizeof(float)};
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = buffer.reference;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = initial.size() * sizeof(float);
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(executable, &argument, 1, acceptance.grid),
              VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRhiDeviceDownloadBuffer(owned.device(), buffer.handle, 0, initial.data(), initial.size() * sizeof(float)),
        VERNON_RHI_STATUS_OK);
    EXPECT_EQ(initial, expected);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), buffer.handle), VERNON_RHI_STATUS_OK);
}

void runHostFloatBufferOracle(VernonProgramExecutable *executable, const AcceptanceDescriptor &acceptance) {
    const std::vector<float> expected(acceptance.expected, acceptance.expected + acceptance.expectedCount);
    VernonProgramParameterView parameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(
                  executable, {acceptance.parameter.data(), acceptance.parameter.size()}, &parameter),
              VERNON_STATUS_OK);

    std::vector<float> result(expected.size());
    const uint64_t shape[]{expected.size()};
    const int64_t strides[]{sizeof(float)};
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = result.data();
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = result.size() * sizeof(float);
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(executable, &argument, 1, acceptance.grid),
              VERNON_STATUS_OK);
    EXPECT_EQ(result, expected);
}

VernonProgramArgument hostTensorArgument(const VernonProgramParameterView &parameter, void *data, size_t byteSize,
                                         uint32_t rank, const uint64_t *shape, const int64_t *strides,
                                         size_t byteOffset = 0) {
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = data;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = parameter.access;
    argument.tensor.rank = rank;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_offset = byteOffset;
    argument.tensor.byte_size = byteSize;
    return argument;
}

VernonStringView stringView(const char *value) { return {value, std::strlen(value)}; }

void runSignedStrideVjpHostOracle(VernonProgramExecutable *executable, const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 5u);
    VernonProgramParameterView sourceParameter{};
    VernonProgramParameterView outputParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, stringView("source"), &sourceParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, stringView("output"), &outputParameter),
              VERNON_STATUS_OK);
    std::array<float, 4> source{10, 2, 20, 3};
    std::array<float, 1> output{};
    const uint64_t sourceShape[]{2};
    const int64_t sourceStrides[]{-2 * static_cast<int64_t>(sizeof(float))};
    const uint64_t outputShape[]{1};
    const int64_t outputStrides[]{sizeof(float)};
    const std::array arguments{
        hostTensorArgument(sourceParameter, source.data(), sizeof(source), 1, sourceShape, sourceStrides,
                           3 * sizeof(float)),
        hostTensorArgument(outputParameter, output.data(), sizeof(output), 1, outputShape, outputStrides),
    };
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(executable, arguments.data(), arguments.size(),
                                                                {1, 1, 1}, &pullback),
              VERNON_STATUS_OK);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(output[0], static_cast<float>(acceptance.expected[0]));

    std::array<float, 1> cotangent{1};
    std::array<float, 4> gradient{};
    VernonProgramParameterView cotangentParameter{};
    VernonProgramParameterView gradientParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindBoundary(executable, VERNON_PROGRAM_BOUNDARY_COTANGENT,
                                                         stringView("output"), &cotangentParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindBoundary(executable, VERNON_PROGRAM_BOUNDARY_GRADIENT,
                                                         stringView("source"), &gradientParameter),
              VERNON_STATUS_OK);
    const std::array derivativeArguments{
        hostTensorArgument(cotangentParameter, cotangent.data(), sizeof(cotangent), 1, outputShape, outputStrides),
        hostTensorArgument(gradientParameter, gradient.data(), sizeof(gradient), 1, sourceShape, sourceStrides,
                           3 * sizeof(float)),
    };
    ASSERT_EQ(vernonProgramPullbackApply(pullback, derivativeArguments.data(), derivativeArguments.size(), nullptr),
              VERNON_STATUS_OK);
    for (size_t index = 0; index < gradient.size(); ++index)
        EXPECT_FLOAT_EQ(gradient[index], static_cast<float>(acceptance.expected[index + 1]));
    vernonProgramPullbackDestroy(pullback);
}

void runFanOutVjpHostOracle(VernonProgramExecutable *executable, const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 3u);
    std::array<float, 1> source{2};
    std::array<float, 1> square{};
    std::array<float, 1> cube{};
    const uint64_t shape[]{1};
    const int64_t strides[]{sizeof(float)};
    std::array<VernonProgramArgument, 3> arguments{};
    constexpr std::array names{"source", "square", "cube"};
    const std::array<void *, 3> values{source.data(), square.data(), cube.data()};
    for (size_t index = 0; index < arguments.size(); ++index) {
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, stringView(names[index]), &parameter),
                  VERNON_STATUS_OK);
        arguments[index] = hostTensorArgument(parameter, values[index], sizeof(float), 1, shape, strides);
    }
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(executable, arguments.data(), arguments.size(),
                                                                {1, 1, 1}, &pullback),
              VERNON_STATUS_OK);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(square[0], static_cast<float>(acceptance.expected[0]));
    EXPECT_FLOAT_EQ(cube[0], static_cast<float>(acceptance.expected[1]));

    std::array<float, 1> squareCotangent{1};
    std::array<float, 1> cubeCotangent{2};
    std::array<float, 1> gradient{};
    std::array<vernon::tests::DerivativeLeafFixture, 2> cotangentLeaves{{
        {sizeof(vernon::tests::DerivativeLeafFixture), stringView("square"), VERNON_DATA_F32, squareCotangent.data(),
         sizeof(float), 1, shape},
        {sizeof(vernon::tests::DerivativeLeafFixture), stringView("cube"), VERNON_DATA_F32, cubeCotangent.data(),
         sizeof(float), 1, shape},
    }};
    vernon::tests::DerivativeLeafFixture gradientLeaf{sizeof(vernon::tests::DerivativeLeafFixture),
                                                      stringView("source"),
                                                      VERNON_DATA_F32,
                                                      gradient.data(),
                                                      sizeof(float),
                                                      1,
                                                      shape};
    const vernon::tests::DerivativeLeafSetFixture cotangents{sizeof(vernon::tests::DerivativeLeafSetFixture),
                                                             cotangentLeaves.data(), cotangentLeaves.size()};
    vernon::tests::DerivativeLeafSetFixture gradients{sizeof(vernon::tests::DerivativeLeafSetFixture), &gradientLeaf,
                                                      1};
    ASSERT_EQ(vernon::tests::applyCanonicalPullback(executable, pullback, &cotangents, &gradients), VERNON_STATUS_OK);
    EXPECT_FLOAT_EQ(gradient[0], static_cast<float>(acceptance.expected[2]));
    vernonProgramPullbackDestroy(pullback);
}

void runStructuredViewHostOracle(VernonProgramExecutable *executable, const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(acceptance.expectedCount, 4u);
    struct Pair {
        float left;
        float right;
    };
    struct Payload {
        float value;
        int32_t index;
    };
    std::array<float, 4> matrix{1, 2, 3, 4};
    std::array<Pair, 2> pairs{{{1, 2}, {5, 7}}};
    std::array<Payload, 1> payloads{{{3, 4}}};
    std::array<float, 4> output{};
    const uint64_t matrixShape[]{2, 2};
    const int64_t matrixStrides[]{2 * sizeof(float), sizeof(float)};
    const uint64_t pairsShape[]{2};
    const int64_t pairsStrides[]{sizeof(Pair)};
    const uint64_t payloadShape[]{1};
    const int64_t payloadStrides[]{sizeof(Payload)};
    const uint64_t outputShape[]{4};
    const int64_t outputStrides[]{sizeof(float)};

    std::array<VernonProgramArgument, 4> arguments{};
    constexpr std::array names{"matrix", "pairs", "payloads", "output"};
    for (size_t index = 0; index < arguments.size(); ++index) {
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, stringView(names[index]), &parameter),
                  VERNON_STATUS_OK);
        if (index == 0)
            arguments[index] =
                hostTensorArgument(parameter, matrix.data(), sizeof(matrix), 2, matrixShape, matrixStrides);
        else if (index == 1)
            arguments[index] = hostTensorArgument(parameter, pairs.data(), sizeof(pairs), 1, pairsShape, pairsStrides);
        else if (index == 2)
            arguments[index] =
                hostTensorArgument(parameter, payloads.data(), sizeof(payloads), 1, payloadShape, payloadStrides);
        else
            arguments[index] =
                hostTensorArgument(parameter, output.data(), sizeof(output), 1, outputShape, outputStrides);
    }
    ASSERT_EQ(
        vernon::tests::completeCanonicalComputeInvocation(executable, arguments.data(), arguments.size(), {1, 1, 1}),
        VERNON_STATUS_OK);
    for (size_t index = 0; index < output.size(); ++index)
        EXPECT_FLOAT_EQ(output[index], static_cast<float>(acceptance.expected[index]));
    EXPECT_FLOAT_EQ(pairs[0].left, 5);
    EXPECT_FLOAT_EQ(pairs[0].right, 7);
    EXPECT_FLOAT_EQ(payloads[0].value, 4);
    EXPECT_EQ(payloads[0].index, 6);
}

void runArgumentFreeTriangleOracle(vernon::tests::OwnedRhiRuntime &owned, VernonProgramExecutable *executable,
                                   const AcceptanceDescriptor &acceptance) {
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterCount(executable), 0u);
    constexpr uint32_t extent = 32;
    vernon::tests::RhiImage target =
        vernon::tests::createImage(owned.context(), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, extent, extent,
                                   1, VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    vernon::tests::RhiImageView targetView =
        vernon::tests::createImageView(owned.context(), target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(targetView.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonColorAttachment attachment{};
    attachment.location = 0;
    attachment.view = targetView.reference;
    attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    attachment.clear_color[3] = 1.0f;
    vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
    graphics.renderPass.render_area[2] = extent;
    graphics.renderPass.render_area[3] = extent;
    graphics.dynamic.viewport[2] = extent;
    graphics.dynamic.viewport[3] = extent;
    graphics.dynamic.scissor[2] = extent;
    graphics.dynamic.scissor[3] = extent;

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(executable);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    VernonProgramGraphicsControlsView controls{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable, 0, &controls), VERNON_STATUS_OK);
    ASSERT_EQ(graphics.bind(invocation, controls), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), VERNON_STATUS_OK)
        << vernon::tests::runtimeDiagnostic(vernonRuntimeGetLastError(owned.runtime()));
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, nullptr), VERNON_STATUS_OK)
        << vernon::tests::runtimeDiagnostic(vernonRuntimeGetLastError(owned.runtime()));
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);

    std::vector<uint8_t> pixels(extent * extent * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = extent;
    download.height = extent;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(owned.device(), target.handle, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (extent / 2 * extent + extent / 2) * 4;
    if (acceptance.oracle == "specialization_pixels") {
        ASSERT_EQ(acceptance.expectedCount, 2u);
        EXPECT_NEAR(pixels[center], acceptance.expected[0], 1);
        EXPECT_LT(pixels[center + 1], 10);
    } else {
        ASSERT_EQ(acceptance.expectedCount, 4u);
        for (size_t channel = 0; channel < 4; ++channel)
            EXPECT_NEAR(pixels[center + channel], acceptance.expected[channel], 1);
    }
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(owned.device(), targetView.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(owned.device(), target.handle), VERNON_RHI_STATUS_OK);
}

void runSampledTextureOracle(vernon::tests::OwnedRhiRuntime &owned, VernonProgramExecutable *executable,
                             const AcceptanceDescriptor &acceptance) {
    vernon::tests::runSampledTextureRuntimeOracle(owned, executable, acceptance.expected, acceptance.expectedCount);
}

void runStorageTextureOracle(vernon::tests::OwnedRhiRuntime &owned, VernonProgramExecutable *executable,
                             const AcceptanceDescriptor &acceptance) {
    vernon::tests::RhiImage image = vernon::tests::createImage(
        owned.context(), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA32_FLOAT, 1, 1, 1, VERNON_RHI_IMAGE_STORAGE);
    vernon::tests::RhiImageView view =
        vernon::tests::createImageView(owned.context(), image, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA32_FLOAT);
    ASSERT_NE(image.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(view.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonProgramParameterView parameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(executable, stringView("image"), &parameter),
              VERNON_STATUS_OK);
    VernonProgramArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_IMAGE;
    argument.image = {view.reference};
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(executable, &argument, 1, {1, 1, 1}), VERNON_STATUS_OK);

    std::array<float, 4> texel{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = 1;
    download.height = 1;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(owned.device(), image.handle, &download, texel.data(), sizeof(texel)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(acceptance.expectedCount, texel.size());
    for (size_t channel = 0; channel < texel.size(); ++channel)
        EXPECT_FLOAT_EQ(texel[channel], static_cast<float>(acceptance.expected[channel]));
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(owned.device(), view.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(owned.device(), image.handle), VERNON_RHI_STATUS_OK);
}

class LanguageContractCpuAcceptance : public testing::TestWithParam<ProgramFixtureManifest> {};

TEST_P(LanguageContractCpuAcceptance, LoadsResolvesAndExecutesCanonicalProgramAsset) {
    const ProgramFixtureManifest fixture = GetParam();
    const AcceptanceDescriptor &acceptance = acceptanceFor(fixture.fixtureId);
    const vernon::tests::BackendTestRow backend = backendFor(fixture);
    ASSERT_FALSE(backend.name.empty());
    const vernon::tests::BackendTestRequirements requirements = requirementsFor(fixture);
    ASSERT_EQ(fixture.prepare(), VERNON_STATUS_OK);
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    const vernon::tests::BackendProbeResult runtimeProbe =
        vernon::tests::probeRuntimeBackend(backend, requirements, context);
    ASSERT_TRUE(runtimeProbe.available()) << runtimeProbe.reason;
    {
        vernon::tests::OwnedProgramExecutable program(context, std::string(fixture.manifestPath));
        ASSERT_TRUE(program) << vernon::tests::runtimeDiagnostic(vernonRuntimeGetLastError(context));
        if (acceptance.oracle == "float_buffer")
            runHostFloatBufferOracle(program.get(), acceptance);
        else if (acceptance.oracle == "signed_stride_vjp")
            runSignedStrideVjpHostOracle(program.get(), acceptance);
        else if (acceptance.oracle == "fan_out_vjp")
            runFanOutVjpHostOracle(program.get(), acceptance);
        else if (acceptance.oracle == "structured_view")
            runStructuredViewHostOracle(program.get(), acceptance);
        else
            FAIL() << "oracle '" << acceptance.oracle << "' has no CPU executor";
    }
    vernonRuntimeDestroy(context);
}

class LanguageContractGpuAcceptance : public vernon::tests::ProgramFixtureRuntimeTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override { return requirementsFor(GetParam()); }
};

TEST_P(LanguageContractGpuAcceptance, LoadsResolvesAndExecutesCanonicalProgramAsset) {
    const ProgramFixtureManifest fixture = GetParam();
    if (skipDirectXWorkgroupAtomicContention(fixture))
        GTEST_SKIP() << "GitHub-hosted Windows virtual graphics does not execute groupshared CAS correctly";
    const AcceptanceDescriptor &acceptance = acceptanceFor(fixture.fixtureId);
    vernon::tests::OwnedProgramExecutable program(runtime().runtime, std::string(fixture.manifestPath));
    ASSERT_TRUE(program) << vernon::tests::runtimeDiagnostic(vernonRuntimeGetLastError(runtime().runtime));
    if (acceptance.oracle == "float_buffer")
        runFloatBufferOracle(owned(), program.get(), acceptance);
    else if (acceptance.oracle == "graphics_triangle" || acceptance.oracle == "specialization_pixels")
        runArgumentFreeTriangleOracle(owned(), program.get(), acceptance);
    else if (acceptance.oracle == "sampled_pixel")
        runSampledTextureOracle(owned(), program.get(), acceptance);
    else if (acceptance.oracle == "storage_texel")
        runStorageTextureOracle(owned(), program.get(), acceptance);
    else if (acceptance.oracle == "signed_stride_vjp")
        runSignedStrideVjpHostOracle(program.get(), acceptance);
    else if (acceptance.oracle == "fan_out_vjp")
        runFanOutVjpHostOracle(program.get(), acceptance);
    else if (acceptance.oracle == "structured_view")
        runStructuredViewHostOracle(program.get(), acceptance);
    else
        FAIL() << "oracle '" << acceptance.oracle << "' has no GPU executor";
}

INSTANTIATE_TEST_SUITE_P(CanonicalMatrix, LanguageContractCpuAcceptance, testing::ValuesIn(acceptanceFixtures(true)),
                         testName);
INSTANTIATE_TEST_SUITE_P(CanonicalMatrix, LanguageContractGpuAcceptance, testing::ValuesIn(acceptanceFixtures(false)),
                         testName);

} // namespace
