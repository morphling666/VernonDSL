#include "VernonRHI.h"
#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "VernonRuntimeRHIAdapter.h"
#include "rhi/metal_backend.h"
#include "rhi/rhi_internal.h"
#include "rhi/sampler_filter.h"
#include "runtime/rhi_adapter/adapter_test_hooks.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime_rhi_test_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

VernonRhiDevice createMetalDevice() {
    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = VERNON_RHI_BACKEND_METAL;
    return vernonRhiCreateDevice(&descriptor);
}

const vernon::rhi::metal::DeviceState *metalDeviceState(VernonRhiDevice device) {
    auto state = vernon::rhi::deviceState(device, VERNON_RHI_BACKEND_METAL);
    return state.isOk() ? static_cast<const vernon::rhi::metal::DeviceState *>(state.value()) : nullptr;
}

bool supportsMetalArgumentBufferEncoding(VernonRhiDevice device) {
    const auto *state = metalDeviceState(device);
    return state && state->argumentBufferEncodingSupported;
}

bool supportsMetalArgumentBuffersTier2(VernonRhiDevice device) {
    const auto *state = metalDeviceState(device);
    return state && state->argumentBufferEncodingSupported && state->argumentBuffersTier >= MTLArgumentBuffersTier2;
}

} // namespace

TEST(RuntimeMetal, RequiresTier2BeforePreparingOverLimitArgumentBuffers) {
    using vernon::runtime::validateMetalArgumentBufferLimitsForTesting;
    EXPECT_TRUE(validateMetalArgumentBufferLimitsForTesting(31, 31, 16, false, 0));
    EXPECT_FALSE(validateMetalArgumentBufferLimitsForTesting(32, 0, 0, false, 0));
    EXPECT_FALSE(validateMetalArgumentBufferLimitsForTesting(0, 32, 0, false, 0));
    EXPECT_FALSE(validateMetalArgumentBufferLimitsForTesting(0, 0, 17, false, 0));
    EXPECT_FALSE(validateMetalArgumentBufferLimitsForTesting(0, 1, 0, true, 0));
    EXPECT_TRUE(validateMetalArgumentBufferLimitsForTesting(32, 32, 17, true, 1));
}

TEST(RuntimeMetal, CreatesDeviceAndRoundTripsAllBufferMemoryClasses) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    constexpr std::array<uint32_t, 4> source{0x12345678u, 2u, 3u, 0xabcdef01u};
    for (VernonRhiMemoryClass memoryClass :
         {VERNON_RHI_MEMORY_DEVICE, VERNON_RHI_MEMORY_UPLOAD, VERNON_RHI_MEMORY_READBACK}) {
        VernonRhiBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = sizeof(source);
        descriptor.usage =
            VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
        descriptor.memory_class = memoryClass;
        VernonRhiBuffer buffer{};
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &buffer), VERNON_RHI_STATUS_OK);
        EXPECT_TRUE(vernonRhiDeviceIsBufferValid(device, buffer));
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, buffer, 0, source.data(), sizeof(source)), VERNON_RHI_STATUS_OK);
        std::array<uint32_t, source.size()> destination{};
        ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, destination.data(), sizeof(destination)),
                  VERNON_RHI_STATUS_OK);
        EXPECT_EQ(destination, source);
        void *native = nullptr;
        EXPECT_EQ(vernonRhiDeviceGetBufferNativeHandle(device, buffer, &native), VERNON_RHI_STATUS_OK);
        EXPECT_NE(native, nullptr);
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
        EXPECT_FALSE(vernonRhiDeviceIsBufferValid(device, buffer));
    }

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    VernonRhiCompletionState completionState{};
    ASSERT_EQ(vernonRhiCompletionGetState(device, completion, &completionState), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(completionState == VERNON_RHI_COMPLETION_PENDING || completionState == VERNON_RHI_COMPLETION_SUCCEEDED);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionGetState(device, completion, &completionState), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(completionState, VERNON_RHI_COMPLETION_SUCCEEDED);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionGetState(device, completion, &completionState), VERNON_RHI_STATUS_INVALID_ARGUMENT);

    VernonRhiCommandEncoder shutdownEncoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &shutdownEncoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, shutdownEncoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion shutdownCompletion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, shutdownEncoder, &shutdownCompletion), VERNON_RHI_STATUS_OK);

    vernonRhiDestroyDevice(device);
    EXPECT_EQ(vernonRhiCompletionGetState(device, shutdownCompletion, &completionState), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionWait(device, shutdownCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, shutdownCompletion), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, BoundsInFlightSubmissionsUntilCompletionObservation) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    std::vector<VernonRhiCompletion> completions;
    VernonRhiCommandEncoder blocked{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiStatus status = VERNON_RHI_STATUS_OK;
    for (size_t attempt = 0; attempt < 64 && status == VERNON_RHI_STATUS_OK; ++attempt) {
        VernonRhiCommandEncoder encoder{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        VernonRhiCompletion completion{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        status = vernonRhiDeviceSubmit(device, encoder, &completion);
        if (status == VERNON_RHI_STATUS_OK)
            completions.push_back(completion);
        else
            blocked = encoder;
    }
    ASSERT_EQ(status, VERNON_RHI_STATUS_RESOURCE_EXHAUSTED);
    ASSERT_GT(completions.size(), 1u);
    ASSERT_LT(completions.size(), 64u);
    ASSERT_NE(blocked.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_EQ(vernonRhiCompletionWait(device, completions.front()), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion replacement{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    EXPECT_EQ(vernonRhiDeviceSubmit(device, blocked, &replacement), VERNON_RHI_STATUS_OK);
    if (replacement.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        completions.push_back(replacement);
    for (VernonRhiCompletion completion : completions)
        EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, KeepsDeviceAliveWhileIndependentEncoderIsRecording) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);

    vernonRhiDestroyDevice(device);
    EXPECT_TRUE(vernon::rhi::deviceExists(device));
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
    EXPECT_FALSE(vernon::rhi::deviceExists(device));
}

TEST(RuntimeMetal, CopiesBufferEntirelyOnDevice) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr std::array<uint32_t, 4> source{3u, 5u, 8u, 13u};
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = sizeof(source);
    descriptor.usage =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer sourceBuffer{}, destinationBuffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &sourceBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &destinationBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, sourceBuffer, 0, source.data(), sizeof(source)),
              VERNON_RHI_STATUS_OK);
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderCopyBuffer(device, encoder, sourceBuffer, 0, destinationBuffer, 0, sizeof(source)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    std::array<uint32_t, source.size()> destination{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, destinationBuffer, 0, destination.data(), sizeof(destination)),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destination, source);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, sourceBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, destinationBuffer), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ValidatesMslHostVersionAndComputeLimitsWithSpecificErrors) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(VERNON_RUNTIME_METAL, device, nullptr);
    ASSERT_NE(runtime, nullptr);

    vernon::runtime::RuntimeRequirements requirements;
    requirements.backend = "metal";
    requirements.features = {"compute"};
    requirements.applePlatform = "macos";
    requirements.shaderVersion = {9, 9};
    EXPECT_FALSE(vernon::runtime::validateRuntimeRequirements(*runtime, requirements));
    EXPECT_NE(std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size)
                  .find("unsupported MSL 9.9"),
              std::string::npos);

    requirements.shaderVersion = {2, 4};
    requirements.minimumOsVersion = {999, 0};
    EXPECT_FALSE(vernon::runtime::validateRuntimeRequirements(*runtime, requirements));
    EXPECT_NE(std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size)
                  .find("host provides"),
              std::string::npos);

    requirements.minimumOsVersion = {};
    requirements.computeWorkgroupSize[0] = UINT32_MAX;
    EXPECT_FALSE(vernon::runtime::validateRuntimeRequirements(*runtime, requirements));
    EXPECT_NE(std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size)
                  .find("Metal device limit"),
              std::string::npos);

    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, RoundTripsTextureGeneratesMipmapsAndCreatesSampler) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

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
        VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION | VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_OK);
    ASSERT_TRUE(vernonRhiDeviceIsImageValid(device, image));

    std::array<uint8_t, 4 * 4 * 4> source{};
    for (size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<uint8_t>(index * 3);
    VernonRhiImageUploadDescriptor upload{};
    upload.struct_size = sizeof(upload);
    upload.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    upload.width = descriptor.width;
    upload.height = descriptor.height;
    upload.depth = 1;
    upload.source_format = VERNON_RHI_IMAGE_DATA_RGBA;
    upload.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
    upload.data = source.data();
    ASSERT_EQ(vernonRhiDeviceUploadImage(device, image, &upload, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceGenerateImageMipmaps(device, image), VERNON_RHI_STATUS_OK);
    std::array<uint8_t, source.size()> destination{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = descriptor.width;
    download.height = descriptor.height;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, destination.data(), destination.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destination, source);
    uint64_t nativeImage = 0;
    EXPECT_EQ(vernonRhiDeviceGetImageNativeHandle(device, image, &nativeImage), VERNON_RHI_STATUS_OK);
    EXPECT_NE(nativeImage, 0u);

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    samplerDescriptor.min_filter = VERNON_RHI_FILTER_LINEAR;
    samplerDescriptor.mag_filter = VERNON_RHI_FILTER_LINEAR;
    samplerDescriptor.mip_filter = VERNON_RHI_FILTER_LINEAR;
    samplerDescriptor.address_u = VERNON_RHI_ADDRESS_REPEAT;
    samplerDescriptor.address_v = VERNON_RHI_ADDRESS_CLAMP_TO_EDGE;
    samplerDescriptor.address_w = VERNON_RHI_ADDRESS_MIRRORED_REPEAT;
    samplerDescriptor.max_anisotropy = 1.0f;
    VernonRhiSampler sampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &sampler), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(vernonRhiDeviceIsSamplerValid(device, sampler));
    EXPECT_EQ(vernonRhiDeviceDestroySampler(device, sampler), VERNON_RHI_STATUS_OK);
    EXPECT_FALSE(vernonRhiDeviceIsSamplerValid(device, sampler));

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    EXPECT_FALSE(vernonRhiDeviceIsImageValid(device, image));
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, CopiesImageMipLayerRegionsAndRejectsInvalidCopies) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    descriptor.width = 4;
    descriptor.height = 4;
    descriptor.depth = 1;
    descriptor.mip_levels = 2;
    descriptor.array_layers = 2;
    descriptor.sample_count = 1;
    descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage source{};
    VernonRhiImage destination{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &source), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &destination), VERNON_RHI_STATUS_OK);
    const std::array<uint8_t, 16> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    VernonRhiImageUploadDescriptor upload{};
    upload.struct_size = sizeof(upload);
    upload.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    upload.mip_level = 1;
    upload.array_layer = 1;
    upload.width = 2;
    upload.height = 2;
    upload.depth = 1;
    upload.source_format = VERNON_RHI_IMAGE_DATA_RGBA;
    upload.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
    upload.data = expected.data();
    ASSERT_EQ(vernonRhiDeviceUploadImage(device, source, &upload, 1), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiImageCopyRegion region{sizeof(VernonRhiImageCopyRegion), 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 1,
                                    VERNON_RHI_IMAGE_ASPECT_COLOR,    {}};
    ASSERT_EQ(vernonRhiCommandEncoderCopyImage(device, encoder, source, destination, &region, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);

    std::array<uint8_t, 16> actual{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.mip_level = 1;
    download.array_layer = 1;
    download.width = 2;
    download.height = 2;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, destination, &download, actual.data(), actual.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(actual, expected);

    VernonRhiCommandEncoder invalidEncoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &invalidEncoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderCopyImage(device, invalidEncoder, source, source, &region, 1),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    region.width = 3;
    EXPECT_EQ(vernonRhiCommandEncoderCopyImage(device, invalidEncoder, source, destination, &region, 1),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    region.width = 2;
    const std::array<VernonRhiImageCopyRegion, 2> overlapping{region, region};
    EXPECT_EQ(vernonRhiCommandEncoderCopyImage(device, invalidEncoder, source, destination, overlapping.data(),
                                               overlapping.size()),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    VernonRhiImageDescriptor incompatibleDescriptor = descriptor;
    incompatibleDescriptor.format = VERNON_RHI_FORMAT_R8_UNORM;
    VernonRhiImage incompatible{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &incompatibleDescriptor, &incompatible), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderCopyImage(device, invalidEncoder, source, incompatible, &region, 1),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, incompatible), VERNON_RHI_STATUS_OK);
    VernonRhiImageDescriptor wrongUsageDescriptor = descriptor;
    wrongUsageDescriptor.usage = VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage wrongUsage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &wrongUsageDescriptor, &wrongUsage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderCopyImage(device, invalidEncoder, wrongUsage, destination, &region, 1),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, wrongUsage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, invalidEncoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, destination), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, source), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, CopiesThreeDimensionalImageDepthExtent) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_3D;
    descriptor.format = VERNON_RHI_FORMAT_R8_UNORM;
    descriptor.width = 2;
    descriptor.height = 2;
    descriptor.depth = 2;
    descriptor.mip_levels = 1;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage source{};
    VernonRhiImage destination{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &source), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &destination), VERNON_RHI_STATUS_OK);
    const std::array<uint8_t, 8> expected{1, 2, 3, 4, 5, 6, 7, 8};
    VernonRhiImageUploadDescriptor upload{};
    upload.struct_size = sizeof(upload);
    upload.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    upload.width = 2;
    upload.height = 2;
    upload.depth = 2;
    upload.source_format = VERNON_RHI_IMAGE_DATA_RED;
    upload.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
    upload.data = expected.data();
    ASSERT_EQ(vernonRhiDeviceUploadImage(device, source, &upload, 1), VERNON_RHI_STATUS_OK);
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiImageCopyRegion region{sizeof(VernonRhiImageCopyRegion), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2,
                                    VERNON_RHI_IMAGE_ASPECT_COLOR,    {}};
    ASSERT_EQ(vernonRhiCommandEncoderCopyImage(device, encoder, source, destination, &region, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    std::array<uint8_t, 8> actual{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = 2;
    download.height = 2;
    download.depth = 2;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RED;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, destination, &download, actual.data(), actual.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, destination), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, source), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, CopiesDepthImageAspect) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.format = VERNON_RHI_FORMAT_D32_FLOAT;
    descriptor.width = 2;
    descriptor.height = 2;
    descriptor.depth = 1;
    descriptor.mip_levels = 1;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage source{};
    VernonRhiImage destination{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &source), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &destination), VERNON_RHI_STATUS_OK);
    const std::array<float, 4> expected{0.1f, 0.25f, 0.5f, 1.0f};
    VernonRhiImageUploadDescriptor upload{};
    upload.struct_size = sizeof(upload);
    upload.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH;
    upload.width = 2;
    upload.height = 2;
    upload.depth = 1;
    upload.source_format = VERNON_RHI_IMAGE_DATA_DEPTH;
    upload.source_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    upload.data = expected.data();
    ASSERT_EQ(vernonRhiDeviceUploadImage(device, source, &upload, 1), VERNON_RHI_STATUS_OK);
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiImageCopyRegion region{sizeof(VernonRhiImageCopyRegion), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1,
                                    VERNON_RHI_IMAGE_ASPECT_DEPTH,    {}};
    ASSERT_EQ(vernonRhiCommandEncoderCopyImage(device, encoder, source, destination, &region, 1), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);
    std::array<float, 4> actual{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH;
    download.width = 2;
    download.height = 2;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, destination, &download, actual.data(), sizeof(actual)),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, destination), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, source), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, CreatesImageViewsAndRetainsTheirParentImage) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 4;
    imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 3;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);

    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_SRGB;
    viewDescriptor.base_mip_level = 1;
    viewDescriptor.mip_level_count = 2;
    viewDescriptor.base_array_layer = 0;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);
    uint64_t native = 0;
    EXPECT_EQ(vernonRhiDeviceGetImageViewNativeHandle(device, view, &native), VERNON_RHI_STATUS_OK);
    EXPECT_NE(native, 0u);

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    EXPECT_FALSE(vernonRhiDeviceIsImageValid(device, image));
    EXPECT_EQ(vernonRhiDeviceGetImageViewNativeHandle(device, view, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceGetImageViewNativeHandle(device, view, &native), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, RejectsBufferDescriptorsThatWouldChangeSemantics) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 16;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE | (uint32_t{1} << 31);
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    EXPECT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ValidatesAndRecordsHazardTrackedBarriers) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 16;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);
    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 2;
    imageDescriptor.height = 2;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_STORAGE | VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    std::array<VernonRhiBarrier, 2> barriers{};
    barriers[0].struct_size = sizeof(VernonRhiBarrier);
    barriers[0].buffer = buffer;
    barriers[0].source_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barriers[0].destination_stage_mask = VERNON_RHI_STAGE_COMPUTE;
    barriers[0].source_access = VERNON_RHI_ACCESS_SHADER_WRITE;
    barriers[0].destination_access = VERNON_RHI_ACCESS_SHADER_READ;
    barriers[0].old_state = VERNON_RHI_STATE_SHADER_WRITE;
    barriers[0].new_state = VERNON_RHI_STATE_SHADER_READ;
    barriers[1] = barriers[0];
    barriers[1].image = image;
    barriers[1].is_image = 1;
    barriers[1].image_subresources = {0, 1, 0, 1, VERNON_RHI_IMAGE_ASPECT_COLOR};
    VernonRhiBarrier invalid = barriers[0];
    invalid.destination_stage_mask = uint32_t{1} << 31;
    EXPECT_EQ(vernonRhiCommandEncoderBarrier(device, encoder, &invalid, 1), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiCommandEncoderBarrier(device, encoder, barriers.data(), barriers.size()), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, DecodesEverySamplerFilterCombination) {
    const std::array<std::array<bool, 2>, 6> expected{{
        {false, false},
        {true, false},
        {false, false},
        {true, true},
        {true, false},
        {false, true},
    }};
    for (uint32_t value = VERNON_RHI_FILTER_NEAREST; value <= VERNON_RHI_FILTER_NEAREST_MIPMAP_LINEAR; ++value) {
        VernonRhiSamplerDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.min_filter = value;
        descriptor.mag_filter = VERNON_RHI_FILTER_LINEAR;
        descriptor.mip_filter = VERNON_RHI_FILTER_NEAREST;
        descriptor.max_anisotropy = 1.0f;
        vernon::rhi::SamplerFilter filter;
        ASSERT_TRUE(vernon::rhi::decodeSamplerFilter(descriptor, filter));
        EXPECT_EQ(filter.minLinear, expected[value][0]);
        EXPECT_EQ(filter.mipLinear, expected[value][1]);
        EXPECT_TRUE(filter.magLinear);
    }
}

TEST(RuntimeMetal, RetainedBufferDelaysSlotReuse) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = 16;
    descriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer first{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &first), VERNON_RHI_STATUS_OK);
    auto resourceResult = vernon::rhi::bufferResource(device, first);
    ASSERT_TRUE(resourceResult.isOk());
    const uint64_t resource = resourceResult.value();
    auto retained = vernon::rhi::retainResource(device, vernon::rhi::ResourceKind::Buffer, resource);
    ASSERT_TRUE(retained.isOk());
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, first), VERNON_RHI_STATUS_OK);
    VernonRhiBuffer replacement{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &replacement), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacement.index, first.index);
    ASSERT_TRUE(retained.value().release().isOk());
    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, first.index);
    EXPECT_NE(recycled.generation, first.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, RoundTripsThreeDimensionalCubeAndDepthTextures) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    auto roundTrip = [&](VernonRhiImageDimension dimension, VernonRhiFormat format, uint32_t width, uint32_t height,
                         uint32_t depth, uint32_t layers, VernonRhiImageDataFormat sourceFormat,
                         VernonRhiImageDataType sourceType, const void *source, size_t byteSize) {
        VernonRhiImageDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = dimension;
        descriptor.format = format;
        descriptor.width = width;
        descriptor.height = height;
        descriptor.depth = depth;
        descriptor.mip_levels = 1;
        descriptor.array_layers = layers;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
        VernonRhiImage image{};
        EXPECT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_OK);
        if (!vernonRhiDeviceIsImageValid(device, image))
            return;
        const size_t layerSize = byteSize / layers;
        std::vector<VernonRhiImageUploadDescriptor> uploads(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            uploads[layer].struct_size = sizeof(VernonRhiImageUploadDescriptor);
            uploads[layer].aspect =
                format == VERNON_RHI_FORMAT_D32_FLOAT ? VERNON_RHI_IMAGE_ASPECT_DEPTH : VERNON_RHI_IMAGE_ASPECT_COLOR;
            uploads[layer].array_layer = dimension == VERNON_RHI_IMAGE_3D ? 0 : layer;
            uploads[layer].width = width;
            uploads[layer].height = height;
            uploads[layer].depth = depth;
            uploads[layer].source_format = sourceFormat;
            uploads[layer].source_type = sourceType;
            uploads[layer].data = static_cast<const uint8_t *>(source) + layer * layerSize;
        }
        EXPECT_EQ(vernonRhiDeviceUploadImage(device, image, uploads.data(), uploads.size()), VERNON_RHI_STATUS_OK);
        std::vector<uint8_t> destination(byteSize);
        VernonRhiImageDownloadDescriptor download{};
        download.struct_size = sizeof(download);
        download.aspect =
            format == VERNON_RHI_FORMAT_D32_FLOAT ? VERNON_RHI_IMAGE_ASPECT_DEPTH : VERNON_RHI_IMAGE_ASPECT_COLOR;
        download.width = width;
        download.height = height;
        download.depth = depth;
        download.destination_format = sourceFormat;
        download.destination_type = sourceType;
        std::vector<VernonRhiImageDownload> downloads(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            downloads[layer].descriptor = download;
            downloads[layer].descriptor.array_layer = dimension == VERNON_RHI_IMAGE_3D ? 0 : layer;
            downloads[layer].destination = destination.data() + layer * layerSize;
            downloads[layer].size = layerSize;
        }
        EXPECT_EQ(vernonRhiDeviceDownloadImageBatch(device, image, downloads.data(), downloads.size()),
                  VERNON_RHI_STATUS_OK);
        EXPECT_EQ(std::memcmp(destination.data(), source, byteSize), 0);
        EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    };

    std::array<uint8_t, 2 * 2 * 2 * 4> volume{};
    for (size_t index = 0; index < volume.size(); ++index)
        volume[index] = static_cast<uint8_t>(index);
    roundTrip(VERNON_RHI_IMAGE_3D, VERNON_RHI_FORMAT_RGBA8_UNORM, 2, 2, 2, 1, VERNON_RHI_IMAGE_DATA_RGBA,
              VERNON_RHI_IMAGE_DATA_UINT8, volume.data(), volume.size());

    std::array<uint8_t, 6 * 2 * 2 * 4> cube{};
    for (size_t index = 0; index < cube.size(); ++index)
        cube[index] = static_cast<uint8_t>(index * 3);
    roundTrip(VERNON_RHI_IMAGE_CUBE, VERNON_RHI_FORMAT_RGBA8_UNORM, 2, 2, 1, 6, VERNON_RHI_IMAGE_DATA_RGBA,
              VERNON_RHI_IMAGE_DATA_UINT8, cube.data(), cube.size());

    const std::array<float, 4> depth{0.0f, 0.25f, 0.5f, 1.0f};
    roundTrip(VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_D32_FLOAT, 2, 2, 1, 1, VERNON_RHI_IMAGE_DATA_DEPTH,
              VERNON_RHI_IMAGE_DATA_FLOAT32, depth.data(), sizeof(depth));
    vernonRhiDestroyDevice(device);
}

namespace {

VernonRuntimeProviderShaderDescriptor mslShader(uint32_t stage, std::string_view source, std::string_view entry) {
    VernonRuntimeProviderShaderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.stage = stage;
    descriptor.format = {"msl", 3};
    descriptor.data = source.data();
    descriptor.size = source.size();
    descriptor.entry = {entry.data(), entry.size()};
    return descriptor;
}

VernonRuntimeProviderBindingLayoutEntry computeEntry(uint32_t slot, VernonRuntimeProviderBindingKind kind,
                                                     uint32_t access, uint32_t elementSize) {
    VernonRuntimeProviderBindingLayoutEntry entry{};
    entry.slot = slot;
    entry.binding = slot;
    entry.kind = kind;
    entry.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    entry.access = access;
    entry.array_count = 1;
    entry.element_size = elementSize;
    return entry;
}

VernonRuntimeCorePipeline *prepareComputePipeline(const VernonRuntimeDeviceProvider *provider,
                                                  const VernonRuntimeProviderShaderDescriptor &shader,
                                                  VernonRuntimeCoreGroupLayout *const *groupLayouts,
                                                  size_t groupLayoutCount) {
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.group_layouts = groupLayouts;
    descriptor.group_layout_count = groupLayoutCount;
    descriptor.workgroup_size[0] = descriptor.workgroup_size[1] = descriptor.workgroup_size[2] = 1;
    VernonRuntimeCorePipeline *pipeline = nullptr;
    EXPECT_EQ(vernonRuntimeCorePreparePipeline(provider, &descriptor, &pipeline), VERNON_STATUS_OK);
    return pipeline;
}

VernonRuntimeCoreGroup *persistentGroup(VernonRuntimeCoreGroupLayout *layout,
                                        const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    VernonRuntimeCoreGroup *group = nullptr;
    EXPECT_EQ(
        vernonRuntimeCoreCreateGroup(layout, values, valueCount, VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &group),
        VERNON_STATUS_OK);
    return group;
}

// An RHI command encoder and the provider's reference to it.
struct Recording {
    VernonRhiCommandEncoder encoder{};
    VernonRuntimeProviderObject providerEncoder{};
};

Recording beginRecording(VernonRhiDevice device, VernonRuntimeRhiAdapter *adapter, uint32_t queue) {
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = queue;
    Recording recording;
    EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &recording.encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, recording.encoder, &recording.providerEncoder),
              VERNON_STATUS_OK);
    return recording;
}

// Encodes one dispatch of `pipeline` with `groups` bound at their indices.
void encodeDispatch(const VernonRuntimeDeviceProvider *provider, const Recording &recording,
                    VernonRuntimeCorePipeline *pipeline, std::initializer_list<VernonRuntimeCoreGroup *> groups,
                    uint32_t groupCount) {
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(provider, recording.providerEncoder, &commands), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreBindPipeline(commands, pipeline), VERNON_STATUS_OK);
    uint32_t index = 0;
    for (VernonRuntimeCoreGroup *group : groups)
        EXPECT_EQ(vernonRuntimeCoreBindGroup(commands, index++, group), VERNON_STATUS_OK);
    const uint32_t counts[3]{groupCount, 1, 1};
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, counts, nullptr, 0), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
}

void submitAndWait(VernonRhiDevice device, const Recording &recording) {
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, recording.encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::tests::completeSubmission(device, recording.encoder), VERNON_RHI_STATUS_OK);
}

constexpr std::string_view kAddValueSource = R"(
#include <metal_stdlib>
using namespace metal;
struct AddArguments {
    device uint *values [[id(0)]];
    constant uint *amount [[id(1)]];
};
kernel void add_value(constant AddArguments &arguments [[buffer(0)]],
                      uint index [[thread_position_in_grid]]) {
    arguments.values[index] += *arguments.amount;
}
)";

// add_value's argument buffer: a storage buffer and an inline value, which the group places in its inline values.
std::array<VernonRuntimeProviderBindingLayoutEntry, 2> addValueEntries() {
    return {computeEntry(0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, 3, sizeof(uint32_t)),
            computeEntry(1, VERNON_RUNTIME_PROVIDER_INLINE_VALUE, 1, sizeof(uint32_t))};
}

std::array<VernonRuntimeProviderBindingValue, 2> addValueBindings(VernonRuntimeProviderResourceReference values,
                                                                  const uint32_t &amount) {
    std::array<VernonRuntimeProviderBindingValue, 2> bindings{};
    bindings[0].slot = 0;
    bindings[0].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    bindings[0].payload.buffer.resource = values;
    bindings[1].slot = 1;
    bindings[1].kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
    bindings[1].payload.inline_value.data = &amount;
    bindings[1].payload.inline_value.size = sizeof(amount);
    return bindings;
}

VernonRhiBuffer createStorageBuffer(VernonRhiDevice device, const void *data, uint64_t size) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.usage =
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    EXPECT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &buffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceUploadBuffer(device, buffer, 0, data, size), VERNON_RHI_STATUS_OK);
    return buffer;
}

} // namespace

TEST(RuntimeMetal, ProviderRejectsDuplicateArgumentBufferMembers) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_METAL);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    std::array<VernonRuntimeProviderBindingLayoutEntry, 2> entries{
        computeEntry(0, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, 1, sizeof(uint32_t)),
        computeEntry(1, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER, 1, sizeof(uint32_t))};
    entries[1].binding = entries[0].binding;
    VernonRuntimeCoreGroupLayout *layout = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGroupLayout(provider, entries.data(), entries.size(), &layout),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(layout, nullptr);

    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ProviderCompilesBindsAndDispatchesCompute) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    if (!supportsMetalArgumentBufferEncoding(device)) {
        vernonRhiDestroyDevice(device);
        GTEST_SKIP() << "Metal argument-buffer encoding is unavailable on this device";
    }
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_METAL);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);
    EXPECT_EQ(provider->get_capabilities(provider->user_data) & VERNON_RUNTIME_PROVIDER_COMPUTE,
              VERNON_RUNTIME_PROVIDER_COMPUTE);

    const auto entries = addValueEntries();
    VernonRuntimeCoreGroupLayout *groupLayout = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(provider, entries.data(), entries.size(), &groupLayout),
              VERNON_STATUS_OK);
    const auto shader = mslShader(VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, kAddValueSource, "add_value");
    VernonRuntimeCorePipeline *pipeline = prepareComputePipeline(provider, shader, &groupLayout, 1);
    ASSERT_NE(pipeline, nullptr);

    constexpr std::array<uint32_t, 4> input{1, 2, 3, 4};
    VernonRhiBuffer buffer = createStorageBuffer(device, input.data(), sizeof(input));
    VernonRuntimeProviderResourceReference bufferReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffer, 0, sizeof(input), &bufferReference),
              VERNON_STATUS_OK);
    const uint32_t amount = 7;
    const auto values = addValueBindings(bufferReference, amount);
    VernonRuntimeCoreGroup *group = persistentGroup(groupLayout, values.data(), values.size());
    ASSERT_NE(group, nullptr);
    auto invalidValues = values;
    invalidValues[1].payload.inline_value.size = 0;
    VernonRuntimeCoreGroup *invalidGroup = nullptr;
    EXPECT_EQ(vernonRuntimeCoreCreateGroup(groupLayout, invalidValues.data(), invalidValues.size(),
                                           VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT, {}, &invalidGroup),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(invalidGroup, nullptr);

    const Recording recording = beginRecording(device, adapter, VERNON_RHI_QUEUE_COMPUTE);
    encodeDispatch(provider, recording, pipeline, {group}, input.size());
    submitAndWait(device, recording);
    std::array<uint32_t, input.size()> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, output.data(), sizeof(output)), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output, (std::array<uint32_t, 4>{8, 9, 10, 11}));

    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(groupLayout);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ProviderPlacesTransientGroupsInTheTransientMemoryOfRetiredRecordings) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    if (!supportsMetalArgumentBufferEncoding(device)) {
        vernonRhiDestroyDevice(device);
        GTEST_SKIP() << "Metal argument-buffer encoding is unavailable on this device";
    }
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_METAL);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    const auto entries = addValueEntries();
    VernonRuntimeCoreGroupLayout *groupLayout = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(provider, entries.data(), entries.size(), &groupLayout),
              VERNON_STATUS_OK);
    const auto shader = mslShader(VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, kAddValueSource, "add_value");
    VernonRuntimeCorePipeline *pipeline = prepareComputePipeline(provider, shader, &groupLayout, 1);
    ASSERT_NE(pipeline, nullptr);
    constexpr std::array<uint32_t, 4> input{1, 2, 3, 4};
    VernonRhiBuffer buffer = createStorageBuffer(device, input.data(), sizeof(input));
    VernonRuntimeProviderResourceReference bufferReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffer, 0, sizeof(input), &bufferReference),
              VERNON_STATUS_OK);

    const auto addThroughTransientGroup = [&](uint32_t amount) {
        const Recording recording = beginRecording(device, adapter, VERNON_RHI_QUEUE_COMPUTE);
        const auto values = addValueBindings(bufferReference, amount);
        VernonRuntimeCoreGroup *group = nullptr;
        ASSERT_EQ(vernonRuntimeCoreCreateGroup(groupLayout, values.data(), values.size(),
                                               VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT, recording.providerEncoder,
                                               &group),
                  VERNON_STATUS_OK);
        encodeDispatch(provider, recording, pipeline, {group}, input.size());
        vernonRuntimeCoreGroupDestroy(group);
        submitAndWait(device, recording);
    };
    addThroughTransientGroup(1);
    const size_t warmChunks = metalDeviceState(device)->transientChunkCreations;
    EXPECT_GT(warmChunks, 0u);
    uint32_t added = 1;
    for (uint32_t amount = 2; amount < 10; ++amount) {
        addThroughTransientGroup(amount);
        added += amount;
    }
    EXPECT_EQ(metalDeviceState(device)->transientChunkCreations, warmChunks);
    std::array<uint32_t, input.size()> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, output.data(), sizeof(output)), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output, (std::array<uint32_t, 4>{1 + added, 2 + added, 3 + added, 4 + added}));

    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(groupLayout);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ProviderBindsMoreThanThirtyBuffersAcrossGroupsAndRetainsResources) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    if (!supportsMetalArgumentBuffersTier2(device)) {
        vernonRhiDestroyDevice(device);
        GTEST_SKIP() << "Metal Argument Buffers Tier 2 encoding is unavailable on this device";
    }
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_METAL);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    constexpr uint32_t kGroupCount = 2;
    constexpr uint32_t kBuffersPerGroup = 16;
    std::string source = "#include <metal_stdlib>\nusing namespace metal;\n";
    for (uint32_t group = 0; group < kGroupCount; ++group) {
        source += "struct Group" + std::to_string(group) + " {\n";
        for (uint32_t index = 0; index < kBuffersPerGroup; ++index)
            source += "device uint *b" + std::to_string(group * kBuffersPerGroup + index) + " [[id(" +
                      std::to_string(index) + ")]];\n";
        source += "};\n";
    }
    source += "kernel void sum_buffers(constant Group0 &g0 [[buffer(0)]], "
              "constant Group1 &g1 [[buffer(1)]]) {\nuint total = 0;\n";
    for (uint32_t index = 0; index < kGroupCount * kBuffersPerGroup; ++index)
        source += "total += g" + std::to_string(index / kBuffersPerGroup) + ".b" + std::to_string(index) + "[0];\n";
    source += "g0.b0[0] = total;\n}\n";

    std::array<VernonRuntimeCoreGroupLayout *, kGroupCount> groupLayouts{};
    for (uint32_t group = 0; group < kGroupCount; ++group) {
        std::array<VernonRuntimeProviderBindingLayoutEntry, kBuffersPerGroup> entries{};
        for (uint32_t index = 0; index < kBuffersPerGroup; ++index)
            entries[index] = computeEntry(index, VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER,
                                          group == 0 && index == 0 ? 3u : 1u, sizeof(uint32_t));
        ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(provider, entries.data(), entries.size(), &groupLayouts[group]),
                  VERNON_STATUS_OK);
    }
    const auto shader = mslShader(VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, source, "sum_buffers");
    VernonRuntimeCorePipeline *pipeline = prepareComputePipeline(provider, shader, groupLayouts.data(), kGroupCount);
    ASSERT_NE(pipeline, nullptr);

    std::array<VernonRhiBuffer, kGroupCount * kBuffersPerGroup> buffers{};
    std::array<std::array<VernonRuntimeProviderBindingValue, kBuffersPerGroup>, kGroupCount> values{};
    for (uint32_t index = 0; index < buffers.size(); ++index) {
        const uint32_t initial = index + 1;
        buffers[index] = createStorageBuffer(device, &initial, sizeof(initial));
        VernonRuntimeProviderBindingValue &value = values[index / kBuffersPerGroup][index % kBuffersPerGroup];
        value.slot = index % kBuffersPerGroup;
        value.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
        ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffers[index], 0, sizeof(initial),
                                                         &value.payload.buffer.resource),
                  VERNON_STATUS_OK);
    }
    std::array<VernonRuntimeCoreGroup *, kGroupCount> groups{};
    for (uint32_t group = 0; group < kGroupCount; ++group) {
        groups[group] = persistentGroup(groupLayouts[group], values[group].data(), values[group].size());
        ASSERT_NE(groups[group], nullptr);
    }
    const uint32_t replacementValue = 1000;
    VernonRhiBuffer replacement = createStorageBuffer(device, &replacementValue, sizeof(replacementValue));

    const Recording recording = beginRecording(device, adapter, VERNON_RHI_QUEUE_COMPUTE);
    encodeDispatch(provider, recording, pipeline, {groups[0], groups[1]}, 1);
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, replacement, 0, sizeof(replacementValue),
                                                     &values.back().back().payload.buffer.resource),
              VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *replacementGroup =
        persistentGroup(groupLayouts.back(), values.back().data(), values.back().size());
    ASSERT_NE(replacementGroup, nullptr);
    vernonRuntimeCoreGroupDestroy(groups.back());
    groups.back() = replacementGroup;
    for (size_t index = 1; index < buffers.size(); ++index)
        ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, buffers[index]), VERNON_RHI_STATUS_OK);
    submitAndWait(device, recording);
    uint32_t output = 0;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffers[0], 0, &output, sizeof(output)), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output, 528u);

    for (VernonRuntimeCoreGroup *group : groups)
        vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    for (VernonRuntimeCoreGroupLayout *groupLayout : groupLayouts)
        vernonRuntimeCoreGroupLayoutDestroy(groupLayout);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffers[0]), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ProviderEncodesSampledStorageTexturesAndSamplerInArgumentBuffer) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    if (!supportsMetalArgumentBuffersTier2(device)) {
        vernonRhiDestroyDevice(device);
        GTEST_SKIP() << "Metal Argument Buffers Tier 2 encoding is unavailable on this device";
    }
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_METAL);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);
    static constexpr std::string_view source = R"(
#include <metal_stdlib>
using namespace metal;
struct TextureArguments {
    texture2d<float, access::sample> input [[id(0)]];
    sampler input_sampler [[id(1)]];
    texture2d<float, access::write> output [[id(2)]];
};
kernel void copy_texture(constant TextureArguments &arguments [[buffer(0)]]) {
    arguments.output.write(arguments.input.sample(arguments.input_sampler, float2(0.5)), uint2(0));
}
)";
    const std::array<VernonRuntimeProviderBindingLayoutEntry, 3> entries{
        computeEntry(0, VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE, 1, 0),
        computeEntry(1, VERNON_RUNTIME_PROVIDER_SAMPLER, 1, 0),
        computeEntry(2, VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE, 2, 0)};
    VernonRuntimeCoreGroupLayout *groupLayout = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGroupLayout(provider, entries.data(), entries.size(), &groupLayout),
              VERNON_STATUS_OK);
    const auto shader = mslShader(VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, source, "copy_texture");
    VernonRuntimeCorePipeline *pipeline = prepareComputePipeline(provider, shader, &groupLayout, 1);
    ASSERT_NE(pipeline, nullptr);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = imageDescriptor.height = imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = imageDescriptor.array_layers = imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage input{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &input), VERNON_RHI_STATUS_OK);
    constexpr std::array<uint8_t, 4> color{51, 102, 153, 255};
    VernonRhiImageUploadDescriptor upload{};
    upload.struct_size = sizeof(upload);
    upload.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    upload.width = upload.height = upload.depth = 1;
    upload.source_format = VERNON_RHI_IMAGE_DATA_RGBA;
    upload.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
    upload.data = color.data();
    ASSERT_EQ(vernonRhiDeviceUploadImage(device, input, &upload, 1), VERNON_RHI_STATUS_OK);
    imageDescriptor.usage = VERNON_RHI_IMAGE_STORAGE | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    VernonRhiImage output{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &output), VERNON_RHI_STATUS_OK);
    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    samplerDescriptor.min_filter = samplerDescriptor.mag_filter = samplerDescriptor.mip_filter =
        VERNON_RHI_FILTER_NEAREST;
    samplerDescriptor.address_u = samplerDescriptor.address_v = samplerDescriptor.address_w =
        VERNON_RHI_ADDRESS_CLAMP_TO_EDGE;
    samplerDescriptor.max_anisotropy = 1.0f;
    VernonRhiSampler sampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &sampler), VERNON_RHI_STATUS_OK);

    std::array<VernonRuntimeProviderBindingValue, 3> values{};
    values[0].slot = 0;
    values[0].kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImage(adapter, input, &values[0].payload.image.view), VERNON_STATUS_OK);
    values[1].slot = 1;
    values[1].kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceSampler(adapter, sampler, &values[1].payload.sampler.resource),
              VERNON_STATUS_OK);
    values[2].slot = 2;
    values[2].kind = VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImage(adapter, output, &values[2].payload.image.view), VERNON_STATUS_OK);
    VernonRuntimeCoreGroup *group = persistentGroup(groupLayout, values.data(), values.size());
    ASSERT_NE(group, nullptr);

    const Recording recording = beginRecording(device, adapter, VERNON_RHI_QUEUE_COMPUTE);
    encodeDispatch(provider, recording, pipeline, {group}, 1);
    submitAndWait(device, recording);
    std::array<uint8_t, 4> result{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = 1;
    download.height = 1;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, output, &download, result.data(), result.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(result, color);

    vernonRuntimeCoreGroupDestroy(group);
    vernonRuntimeCorePipelineDestroy(pipeline);
    vernonRuntimeCoreGroupLayoutDestroy(groupLayout);
    EXPECT_EQ(vernonRhiDeviceDestroySampler(device, sampler), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, input), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, output), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ProviderEncodesColorAndDepthAttachments) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_METAL);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    static constexpr std::string_view source = R"(
#include <metal_stdlib>
using namespace metal;
struct VertexOutput { float4 position [[position]]; };
vertex VertexOutput vertex_main(uint index [[vertex_id]]) {
    const float2 positions[] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    return {float4(positions[index % 3], index < 3 ? 0.2 : 0.8, 1.0)};
}
fragment float4 fragment_main(VertexOutput input [[stage_in]], uint primitive [[primitive_id]]) {
    return primitive == 0 ? float4(1.0, 0.25, 0.0, 1.0) : float4(0.0, 1.0, 0.0, 1.0);
}
)";
    const std::array<VernonRuntimeProviderShaderDescriptor, 2> shaders{
        mslShader(VERNON_RUNTIME_PROVIDER_STAGE_VERTEX, source, "vertex_main"),
        mslShader(VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT, source, "fragment_main")};
    const uint32_t colorFormat = vernon::rhi::metal::pixelFormat(VERNON_RHI_FORMAT_RGBA8_UNORM);
    const uint32_t depthStencilFormat = vernon::rhi::metal::pixelFormat(VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT);
    VernonRasterizationState rasterization{};
    rasterization.cull_mode = VERNON_RHI_CULL_NONE;
    rasterization.front_face = VERNON_RHI_FRONT_FACE_COUNTER_CLOCKWISE;
    VernonDepthStencilState depthStencil{};
    depthStencil.depth_test = 1;
    depthStencil.depth_write = 1;
    depthStencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    depthStencil.stencil_test = 1;
    depthStencil.front.compare = VERNON_RHI_COMPARE_ALWAYS;
    depthStencil.front.stencil_fail = VERNON_RHI_STENCIL_KEEP;
    depthStencil.front.depth_fail = VERNON_RHI_STENCIL_KEEP;
    depthStencil.front.pass = VERNON_RHI_STENCIL_REPLACE;
    depthStencil.stencil_read_mask = 0xff;
    depthStencil.stencil_write_mask = 0xff;
    depthStencil.back = depthStencil.front;
    VernonColorBlendState blend{};
    blend.source_color_factor = VERNON_RHI_BLEND_ONE;
    blend.destination_color_factor = VERNON_RHI_BLEND_ZERO;
    blend.source_alpha_factor = VERNON_RHI_BLEND_ONE;
    blend.destination_alpha_factor = VERNON_RHI_BLEND_ZERO;
    blend.color_operation = VERNON_RHI_BLEND_ADD;
    blend.alpha_operation = VERNON_RHI_BLEND_ADD;
    blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    VernonRuntimeCorePipelineDescriptor pipelineDescriptor{};
    pipelineDescriptor.struct_size = sizeof(pipelineDescriptor);
    pipelineDescriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    pipelineDescriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
    pipelineDescriptor.shaders = shaders.data();
    pipelineDescriptor.shader_count = shaders.size();
    pipelineDescriptor.color_formats = &colorFormat;
    pipelineDescriptor.color_format_count = 1;
    pipelineDescriptor.depth_stencil_format = depthStencilFormat;
    pipelineDescriptor.sample_count = 1;
    pipelineDescriptor.rasterization = rasterization;
    pipelineDescriptor.depth_stencil = depthStencil;
    pipelineDescriptor.color_blends = &blend;
    pipelineDescriptor.color_blend_count = 1;
    VernonRuntimeCorePipeline *pipeline = nullptr;
    ASSERT_EQ(vernonRuntimeCorePreparePipeline(provider, &pipelineDescriptor, &pipeline), VERNON_STATUS_OK);
    VernonRuntimeCoreGraphicsCompatibility compatibility{};
    compatibility.struct_size = sizeof(compatibility);
    compatibility.color_formats = &colorFormat;
    compatibility.color_format_count = 1;
    compatibility.depth_stencil_format = depthStencilFormat;
    compatibility.sample_count = 1;
    compatibility.rasterization = rasterization;
    compatibility.depth_stencil = depthStencil;
    compatibility.color_blends = &blend;
    compatibility.color_blend_count = 1;
    EXPECT_EQ(vernon::runtime::getRhiAdapterPreparationStats(*adapter).livePreparedPipelines, 0u);
    compatibility.depth_stencil.depth_compare = UINT32_MAX;
    VernonRuntimeCoreGraphicsVariant *rejectedVariant = nullptr;
    EXPECT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &rejectedVariant),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(rejectedVariant, nullptr);
    EXPECT_EQ(vernon::runtime::getRhiAdapterPreparationStats(*adapter).livePreparedPipelines, 0u);
    compatibility.depth_stencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    VernonRuntimeCoreGraphicsVariant *variant = nullptr;
    ASSERT_EQ(vernonRuntimeCorePrepareGraphicsVariant(pipeline, &compatibility, &variant), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::getRhiAdapterPreparationStats(*adapter).livePreparedPipelines, 1u);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.width = 4;
    imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = imageDescriptor.format;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference imageReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImageView(adapter, view, &imageReference), VERNON_STATUS_OK);
    VernonRhiImageDescriptor depthDescriptor = imageDescriptor;
    depthDescriptor.format = VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    depthDescriptor.usage = VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    VernonRhiImage depthImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &depthDescriptor, &depthImage), VERNON_RHI_STATUS_OK);
    viewDescriptor.image = depthImage;
    viewDescriptor.format = depthDescriptor.format;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL;
    VernonRhiImageView depthView{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &depthView), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference depthReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImageView(adapter, depthView, &depthReference), VERNON_STATUS_OK);
    constexpr std::array<uint32_t, 3> indices{0, 1, 2};
    VernonRhiBufferDescriptor indexDescriptor{};
    indexDescriptor.struct_size = sizeof(indexDescriptor);
    indexDescriptor.size = sizeof(indices);
    indexDescriptor.usage = VERNON_RHI_BUFFER_INDEX;
    indexDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer indexBuffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &indexDescriptor, &indexBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, indexBuffer, 0, indices.data(), sizeof(indices)),
              VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference indexReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, indexBuffer, 0, sizeof(indices), &indexReference),
              VERNON_STATUS_OK);

    const Recording recording = beginRecording(device, adapter, VERNON_RHI_QUEUE_GRAPHICS);
    VernonRuntimeCoreCommands *commands = nullptr;
    ASSERT_EQ(vernonRuntimeCoreBeginCommands(provider, recording.providerEncoder, &commands), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreBindGraphicsVariant(commands, variant), VERNON_STATUS_OK);
    const uint32_t groupCount[3]{1, 1, 1};
    EXPECT_EQ(vernonRuntimeCoreDispatch(commands, groupCount, nullptr, 0), VERNON_STATUS_INVALID_ARGUMENT);
    VernonRuntimeProviderColorAttachment attachment{};
    attachment.view = imageReference;
    attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    VernonRuntimeProviderDepthStencilAttachment depthAttachment{};
    depthAttachment.view = depthReference;
    depthAttachment.depth_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    depthAttachment.depth_store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    depthAttachment.clear_depth = 1.0f;
    depthAttachment.stencil_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    depthAttachment.stencil_store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    depthAttachment.clear_stencil = 7;
    VernonRuntimeProviderRenderingDescriptor rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.color_attachments = &attachment;
    rendering.color_attachment_count = 1;
    rendering.depth_stencil_attachment = &depthAttachment;
    rendering.render_area[2] = imageDescriptor.width;
    rendering.render_area[3] = imageDescriptor.height;
    rendering.layers = 1;
    ASSERT_EQ(vernonRuntimeCoreBeginRendering(commands, &rendering), VERNON_STATUS_OK);
    constexpr VernonClearColor deferredClear{{1, 0, 0, 1}};
    ASSERT_EQ(vernonRhiCommandEncoderClearColorAttachment(device, recording.encoder, 0, &deferredClear),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderClearDepthStencilAttachment(
                  device, recording.encoder, 0.25f, 2, VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL),
              VERNON_RHI_STATUS_OK);
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 6;
    draw.instance_count = 1;
    draw.stencil_reference = 7;
    draw.viewport[2] = imageDescriptor.width;
    draw.viewport[3] = imageDescriptor.height;
    draw.scissor[2] = imageDescriptor.width;
    draw.scissor[3] = imageDescriptor.height;
    draw.index_buffer = indexReference;
    draw.index_count = indices.size();
    draw.index_type = 1;
    EXPECT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    draw.index_buffer = {};
    draw.index_count = 0;
    draw.index_type = 0;
    ASSERT_EQ(vernonRuntimeCoreDraw(commands, &draw), VERNON_STATUS_OK);
    const auto drawStats = vernon::runtime::getRhiAdapterPreparationStats(*adapter);
    EXPECT_EQ(drawStats.lastStencilReference, 7u);
    EXPECT_FALSE(drawStats.lastDrawIndexed);
    constexpr VernonClearColor explicitClear{{0, 0, 0, 1}};
    EXPECT_EQ(vernonRhiCommandEncoderClearColorAttachment(device, recording.encoder, 0, &explicitClear),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderClearDepthStencilAttachment(
                  device, recording.encoder, 0.5f, 3, VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreEndRendering(commands), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeCoreEndCommands(commands), VERNON_STATUS_OK);
    submitAndWait(device, recording);
    std::array<uint8_t, 4 * 4 * 4> pixels{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = 4;
    download.height = 4;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < pixels.size(); index += 4) {
        EXPECT_EQ(pixels[index], 0);
        EXPECT_EQ(pixels[index + 1], 0);
        EXPECT_EQ(pixels[index + 2], 0);
        EXPECT_EQ(pixels[index + 3], 255);
    }
    std::array<uint8_t, 4 * 4 * 8> depthStencilPixels{};
    download.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL;
    download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, depthImage, &download, depthStencilPixels.data(),
                                           depthStencilPixels.size()),
              VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < depthStencilPixels.size(); index += 8) {
        float depth = 0.0f;
        std::memcpy(&depth, depthStencilPixels.data() + index, sizeof(depth));
        EXPECT_FLOAT_EQ(depth, 0.5f);
        EXPECT_EQ(depthStencilPixels[index + 4], 3);
    }
    std::array<uint8_t, 4 * 4> stencil{};
    download.aspect = VERNON_RHI_IMAGE_ASPECT_STENCIL;
    download.destination_format = VERNON_RHI_IMAGE_DATA_STENCIL;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, depthImage, &download, stencil.data(), stencil.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(std::all_of(stencil.begin(), stencil.end(), [](uint8_t value) { return value == 3; }));
    std::array<float, 4 * 4> depth{};
    download.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH;
    download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, depthImage, &download, depth.data(), sizeof(depth)),
              VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(std::all_of(depth.begin(), depth.end(), [](float value) { return value == 0.5f; }));

    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, depthView), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, depthImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, indexBuffer), VERNON_RHI_STATUS_OK);
    vernonRuntimeCoreGraphicsVariantDestroy(variant);
    vernonRuntimeCorePipelineDestroy(pipeline);
    EXPECT_EQ(vernon::runtime::getRhiAdapterPreparationStats(*adapter).livePreparedPipelines, 0u);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}