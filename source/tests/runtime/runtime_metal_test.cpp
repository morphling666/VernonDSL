#include "../../lib/rhi/metal_backend.h"
#include "../../lib/rhi/rhi_internal.h"
#include "../../lib/rhi/sampler_filter.h"
#include "../../lib/runtime/rhi_adapter/adapter_test_hooks.h"
#include "../../lib/runtime/runtime_dispatch.h"
#include "../../lib/runtime/runtime_test_hooks.h"
#include "VernonRHI.h"
#include "VernonRuntime.h"
#include "VernonRuntimeRHIAdapter.h"
#include "runtime_rhi_test_utils.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

namespace {

VernonRhiDevice createMetalDevice() {
    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = VERNON_RHI_BACKEND_METAL;
    return vernonRhiCreateDevice(&descriptor);
}

const vernon::rhi::metal::DeviceState *metalDeviceState(VernonRhiDevice device) {
    return static_cast<const vernon::rhi::metal::DeviceState *>(
        vernon::rhi::deviceState(device, VERNON_RHI_BACKEND_METAL));
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
    EXPECT_EQ(vernonRhiCompletionGetState(device, shutdownCompletion, &completionState),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
}

TEST(RuntimeMetal, RecordsIndependentCommandEncoders) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    EXPECT_NE(vernon::rhi::deviceCommandCapabilities(device) & vernon::rhi::BackendCommandIndependentRecording, 0u);

    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder first{}, second{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &first), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &second), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, second), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, first), VERNON_RHI_STATUS_OK);

    VernonRhiCompletion firstCompletion{}, secondCompletion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, second, &secondCompletion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(device, first, &firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionWait(device, firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCompletionWait(device, secondCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, firstCompletion), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device, secondCompletion), VERNON_RHI_STATUS_OK);
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
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(VERNON_RUNTIME_METAL, device);
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

TEST(RuntimeMetal, RejectsImageDescriptorsThatWouldChangeSemantics) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    descriptor.width = 2;
    descriptor.height = 2;
    descriptor.depth = 2;
    descriptor.mip_levels = 1;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    EXPECT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    descriptor.depth = 1;
    descriptor.mip_levels = 3;
    EXPECT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    descriptor.mip_levels = 1;
    descriptor.usage |= uint32_t{1} << 31;
    EXPECT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_INVALID_ARGUMENT);

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
    const uint64_t resource = vernon::rhi::bufferResource(device, first);
    ASSERT_NE(resource, 0u);
    ASSERT_TRUE(vernon::rhi::retainResource(device, vernon::rhi::ResourceKind::Buffer, resource));
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, first), VERNON_RHI_STATUS_OK);
    VernonRhiBuffer replacement{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &descriptor, &replacement), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacement.index, first.index);
    vernon::rhi::releaseResource(device, vernon::rhi::ResourceKind::Buffer, resource);
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
        download.width = width;
        download.height = height;
        download.depth = depth;
        download.destination_format = sourceFormat;
        download.destination_type = sourceType;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            download.array_layer = dimension == VERNON_RHI_IMAGE_3D ? 0 : layer;
            EXPECT_EQ(vernonRhiDeviceDownloadImage(device, image, &download, destination.data() + layer * layerSize,
                                                   layerSize),
                      VERNON_RHI_STATUS_OK);
        }
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

TEST(RuntimeMetal, ProviderRejectsDuplicateArgumentBufferMembers) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, VERNON_RHI_BACKEND_METAL);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    std::array<VernonRuntimeProviderBindingLayoutEntry, 2> entries{};
    for (uint32_t index = 0; index < entries.size(); ++index) {
        entries[index].slot = index;
        entries[index].set = 0;
        entries[index].binding = 0;
        entries[index].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
        entries[index].stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        entries[index].access = 1;
        entries[index].array_count = 1;
        entries[index].element_size = sizeof(uint32_t);
    }
    VernonRuntimeProviderPipelineLayoutDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.bindings = entries.data();
    descriptor.binding_count = entries.size();
    VernonRuntimeProviderObject layout{};
    EXPECT_EQ(provider->prepare_pipeline_layout(provider->user_data, &descriptor, &layout),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(layout.value, 0u);

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

    static constexpr char source[] = R"(
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
    VernonRuntimeProviderShaderDescriptor shaderDescriptor{};
    shaderDescriptor.struct_size = sizeof(shaderDescriptor);
    shaderDescriptor.stage = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    shaderDescriptor.format = {"msl", 3};
    shaderDescriptor.data = source;
    shaderDescriptor.size = sizeof(source) - 1;
    shaderDescriptor.entry = {"add_value", 9};
    VernonRuntimeProviderObject shader{};
    ASSERT_EQ(provider->prepare_shader(provider->user_data, &shaderDescriptor, &shader), VERNON_STATUS_OK);

    std::array<VernonRuntimeProviderBindingLayoutEntry, 2> entries{};
    entries[0].slot = 0;
    entries[0].set = 0;
    entries[0].binding = 0;
    entries[0].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    entries[0].stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    entries[0].access = 3;
    entries[0].array_count = 1;
    entries[0].element_size = sizeof(uint32_t);
    entries[1].slot = 1;
    entries[1].set = 0;
    entries[1].binding = 1;
    entries[1].kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
    entries[1].stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    entries[1].access = 1;
    entries[1].array_count = 1;
    entries[1].element_size = sizeof(uint32_t);
    VernonRuntimeProviderPipelineLayoutDescriptor layoutDescriptor{};
    layoutDescriptor.struct_size = sizeof(layoutDescriptor);
    layoutDescriptor.bindings = entries.data();
    layoutDescriptor.binding_count = entries.size();
    VernonRuntimeProviderObject layout{};
    ASSERT_EQ(provider->prepare_pipeline_layout(provider->user_data, &layoutDescriptor, &layout), VERNON_STATUS_OK);

    constexpr std::array<uint32_t, 4> input{1, 2, 3, 4};
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(input);
    bufferDescriptor.usage =
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, buffer, 0, input.data(), sizeof(input)), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference bufferReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffer, 0, sizeof(input), &bufferReference),
              VERNON_STATUS_OK);
    const uint32_t amount = 7;
    std::array<VernonRuntimeProviderBindingValue, 2> values{};
    values[0].slot = 0;
    values[0].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    values[0].payload.buffer.resource = bufferReference;
    values[1].slot = 1;
    values[1].kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
    values[1].payload.inline_value.data = &amount;
    values[1].payload.inline_value.size = sizeof(amount);
    VernonRuntimeProviderBindingSetDescriptor bindingDescriptor{};
    bindingDescriptor.struct_size = sizeof(bindingDescriptor);
    bindingDescriptor.layout = layout;
    bindingDescriptor.values = values.data();
    bindingDescriptor.value_count = values.size();
    VernonRuntimeProviderObject bindings{};
    ASSERT_EQ(provider->create_binding_set(provider->user_data, &bindingDescriptor, &bindings), VERNON_STATUS_OK);
    auto invalidValues = values;
    invalidValues[1].payload.inline_value.size = 0;
    EXPECT_EQ(provider->update_binding_set(provider->user_data, bindings, invalidValues.data(), invalidValues.size()),
              VERNON_STATUS_INVALID_ARGUMENT);

    VernonRuntimeProviderPipelineDescriptor pipelineDescriptor{};
    pipelineDescriptor.struct_size = sizeof(pipelineDescriptor);
    pipelineDescriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    pipelineDescriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    pipelineDescriptor.shaders = &shader;
    pipelineDescriptor.shader_count = 1;
    pipelineDescriptor.layout = layout;
    pipelineDescriptor.workgroup_size[0] = 1;
    pipelineDescriptor.workgroup_size[1] = 1;
    pipelineDescriptor.workgroup_size[2] = 1;
    VernonRuntimeProviderObject pipeline{};
    ASSERT_EQ(provider->prepare_pipeline(provider->user_data, &pipelineDescriptor, &pipeline), VERNON_STATUS_OK);

    VernonRhiCommandEncoderDescriptor commandDescriptor{};
    commandDescriptor.struct_size = sizeof(commandDescriptor);
    commandDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder command{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &commandDescriptor, &command), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerCommand{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, command, &providerCommand), VERNON_STATUS_OK);
    VernonRuntimeProviderDispatchDescriptor dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.pipeline = pipeline;
    dispatch.bindings = bindings;
    dispatch.group_count[0] = input.size();
    dispatch.group_count[1] = 1;
    dispatch.group_count[2] = 1;
    ASSERT_EQ(provider->encode_dispatch(provider->user_data, providerCommand, &dispatch), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, command), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(device, command), VERNON_RHI_STATUS_OK);
    std::array<uint32_t, input.size()> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, output.data(), sizeof(output)), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output, (std::array<uint32_t, 4>{8, 9, 10, 11}));

    provider->destroy_binding_set(provider->user_data, bindings);
    provider->destroy_pipeline(provider->user_data, pipeline);
    provider->destroy_pipeline_layout(provider->user_data, layout);
    provider->destroy_shader(provider->user_data, shader);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST(RuntimeMetal, ProviderBindsMoreThanThirtyBuffersAcrossDescriptorSetsAndRetainsResources) {
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

    std::string source = "#include <metal_stdlib>\nusing namespace metal;\nstruct Set0 {\n";
    for (uint32_t index = 0; index < 16; ++index)
        source += "device uint *b" + std::to_string(index) + " [[id(" + std::to_string(index) + ")]];\n";
    source += "};\nstruct Set1 {\n";
    for (uint32_t index = 0; index < 16; ++index)
        source += "device uint *b" + std::to_string(index + 16) + " [[id(" + std::to_string(index) + ")]];\n";
    source += "};\nkernel void sum_buffers(constant Set0 &s0 [[buffer(0)]], "
              "constant Set1 &s1 [[buffer(1)]]) {\nuint total = 0;\n";
    for (uint32_t index = 0; index < 16; ++index)
        source += "total += s0.b" + std::to_string(index) + "[0];\n";
    for (uint32_t index = 16; index < 32; ++index)
        source += "total += s1.b" + std::to_string(index) + "[0];\n";
    source += "s0.b0[0] = total;\n}\n";

    VernonRuntimeProviderShaderDescriptor shaderDescriptor{};
    shaderDescriptor.struct_size = sizeof(shaderDescriptor);
    shaderDescriptor.stage = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    shaderDescriptor.format = {"msl", 3};
    shaderDescriptor.data = source.data();
    shaderDescriptor.size = source.size();
    shaderDescriptor.entry = {"sum_buffers", 11};
    VernonRuntimeProviderObject shader{};
    ASSERT_EQ(provider->prepare_shader(provider->user_data, &shaderDescriptor, &shader), VERNON_STATUS_OK);

    std::array<VernonRuntimeProviderBindingLayoutEntry, 32> entries{};
    for (uint32_t index = 0; index < entries.size(); ++index) {
        entries[index].slot = index;
        entries[index].set = index / 16;
        entries[index].binding = index % 16;
        entries[index].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
        entries[index].stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        entries[index].access = index == 0 ? 3u : 1u;
        entries[index].array_count = 1;
        entries[index].element_size = sizeof(uint32_t);
    }
    VernonRuntimeProviderPipelineLayoutDescriptor layoutDescriptor{};
    layoutDescriptor.struct_size = sizeof(layoutDescriptor);
    layoutDescriptor.bindings = entries.data();
    layoutDescriptor.binding_count = entries.size();
    VernonRuntimeProviderObject layout{};
    ASSERT_EQ(provider->prepare_pipeline_layout(provider->user_data, &layoutDescriptor, &layout), VERNON_STATUS_OK);

    VernonRuntimeProviderPipelineDescriptor pipelineDescriptor{};
    pipelineDescriptor.struct_size = sizeof(pipelineDescriptor);
    pipelineDescriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    pipelineDescriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    pipelineDescriptor.shaders = &shader;
    pipelineDescriptor.shader_count = 1;
    pipelineDescriptor.layout = layout;
    pipelineDescriptor.workgroup_size[0] = 1;
    pipelineDescriptor.workgroup_size[1] = 1;
    pipelineDescriptor.workgroup_size[2] = 1;
    VernonRuntimeProviderObject pipeline{};
    ASSERT_EQ(provider->prepare_pipeline(provider->user_data, &pipelineDescriptor, &pipeline), VERNON_STATUS_OK);

    std::array<VernonRhiBuffer, 32> buffers{};
    std::array<VernonRuntimeProviderResourceReference, 32> references{};
    std::array<VernonRuntimeProviderBindingValue, 32> values{};
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(uint32_t);
    bufferDescriptor.usage =
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    for (uint32_t index = 0; index < buffers.size(); ++index) {
        const uint32_t initial = index + 1;
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffers[index]), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, buffers[index], 0, &initial, sizeof(initial)),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(
            vernonRuntimeRhiAdapterReferenceBuffer(adapter, buffers[index], 0, sizeof(initial), &references[index]),
            VERNON_STATUS_OK);
        values[index].slot = index;
        values[index].kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
        values[index].payload.buffer.resource = references[index];
    }
    VernonRuntimeProviderBindingSetDescriptor bindingDescriptor{};
    bindingDescriptor.struct_size = sizeof(bindingDescriptor);
    bindingDescriptor.layout = layout;
    bindingDescriptor.values = values.data();
    bindingDescriptor.value_count = values.size();
    VernonRuntimeProviderObject bindings{};
    ASSERT_EQ(provider->create_binding_set(provider->user_data, &bindingDescriptor, &bindings), VERNON_STATUS_OK);
    VernonRhiBuffer replacement{};
    const uint32_t replacementValue = 1000;
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &replacement), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, replacement, 0, &replacementValue, sizeof(replacementValue)),
              VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor commandDescriptor{};
    commandDescriptor.struct_size = sizeof(commandDescriptor);
    commandDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder command{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &commandDescriptor, &command), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerCommand{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, command, &providerCommand), VERNON_STATUS_OK);
    VernonRuntimeProviderDispatchDescriptor dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.pipeline = pipeline;
    dispatch.bindings = bindings;
    dispatch.group_count[0] = dispatch.group_count[1] = dispatch.group_count[2] = 1;
    ASSERT_EQ(provider->encode_dispatch(provider->user_data, providerCommand, &dispatch), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, replacement, 0, sizeof(replacementValue),
                                                     &values.back().payload.buffer.resource),
              VERNON_STATUS_OK);
    ASSERT_EQ(provider->update_binding_set(provider->user_data, bindings, values.data(), values.size()),
              VERNON_STATUS_OK);
    for (size_t index = 1; index < buffers.size(); ++index)
        ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, buffers[index]), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, command), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(device, command), VERNON_RHI_STATUS_OK);
    uint32_t output = 0;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffers[0], 0, &output, sizeof(output)), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output, 528u);

    provider->destroy_binding_set(provider->user_data, bindings);
    provider->destroy_pipeline(provider->user_data, pipeline);
    provider->destroy_pipeline_layout(provider->user_data, layout);
    provider->destroy_shader(provider->user_data, shader);
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
    static constexpr char source[] = R"(
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
    VernonRuntimeProviderShaderDescriptor shaderDescriptor{};
    shaderDescriptor.struct_size = sizeof(shaderDescriptor);
    shaderDescriptor.stage = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    shaderDescriptor.format = {"msl", 3};
    shaderDescriptor.data = source;
    shaderDescriptor.size = sizeof(source) - 1;
    shaderDescriptor.entry = {"copy_texture", 12};
    VernonRuntimeProviderObject shader{};
    ASSERT_EQ(provider->prepare_shader(provider->user_data, &shaderDescriptor, &shader), VERNON_STATUS_OK);

    std::array<VernonRuntimeProviderBindingLayoutEntry, 3> entries{};
    for (uint32_t index = 0; index < entries.size(); ++index) {
        entries[index].slot = index;
        entries[index].set = 0;
        entries[index].binding = index;
        entries[index].stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        entries[index].array_count = 1;
        entries[index].access = index == 2 ? 2u : 1u;
    }
    entries[0].kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
    entries[1].kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
    entries[2].kind = VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    VernonRuntimeProviderPipelineLayoutDescriptor layoutDescriptor{};
    layoutDescriptor.struct_size = sizeof(layoutDescriptor);
    layoutDescriptor.bindings = entries.data();
    layoutDescriptor.binding_count = entries.size();
    VernonRuntimeProviderObject layout{};
    ASSERT_EQ(provider->prepare_pipeline_layout(provider->user_data, &layoutDescriptor, &layout), VERNON_STATUS_OK);
    VernonRuntimeProviderPipelineDescriptor pipelineDescriptor{};
    pipelineDescriptor.struct_size = sizeof(pipelineDescriptor);
    pipelineDescriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    pipelineDescriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    pipelineDescriptor.shaders = &shader;
    pipelineDescriptor.shader_count = 1;
    pipelineDescriptor.layout = layout;
    pipelineDescriptor.workgroup_size[0] = pipelineDescriptor.workgroup_size[1] = pipelineDescriptor.workgroup_size[2] =
        1;
    VernonRuntimeProviderObject pipeline{};
    ASSERT_EQ(provider->prepare_pipeline(provider->user_data, &pipelineDescriptor, &pipeline), VERNON_STATUS_OK);

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
    VernonRuntimeProviderBindingSetDescriptor bindingDescriptor{};
    bindingDescriptor.struct_size = sizeof(bindingDescriptor);
    bindingDescriptor.layout = layout;
    bindingDescriptor.values = values.data();
    bindingDescriptor.value_count = values.size();
    VernonRuntimeProviderObject bindings{};
    ASSERT_EQ(provider->create_binding_set(provider->user_data, &bindingDescriptor, &bindings), VERNON_STATUS_OK);

    VernonRhiCommandEncoderDescriptor commandDescriptor{};
    commandDescriptor.struct_size = sizeof(commandDescriptor);
    commandDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder command{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &commandDescriptor, &command), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerCommand{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, command, &providerCommand), VERNON_STATUS_OK);
    VernonRuntimeProviderDispatchDescriptor dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.pipeline = pipeline;
    dispatch.bindings = bindings;
    dispatch.group_count[0] = dispatch.group_count[1] = dispatch.group_count[2] = 1;
    ASSERT_EQ(provider->encode_dispatch(provider->user_data, providerCommand, &dispatch), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, command), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(device, command), VERNON_RHI_STATUS_OK);
    std::array<uint8_t, 4> result{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = 1;
    download.height = 1;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, output, &download, result.data(), result.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(result, color);

    provider->destroy_binding_set(provider->user_data, bindings);
    provider->destroy_pipeline(provider->user_data, pipeline);
    provider->destroy_pipeline_layout(provider->user_data, layout);
    provider->destroy_shader(provider->user_data, shader);
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

    static constexpr char source[] = R"(
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
    std::array<VernonRuntimeProviderObject, 2> shaders{};
    for (size_t index = 0; index < shaders.size(); ++index) {
        VernonRuntimeProviderShaderDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.stage = index == 0 ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
        descriptor.format = {"msl", 3};
        descriptor.data = source;
        descriptor.size = sizeof(source) - 1;
        descriptor.entry = index == 0 ? VernonStringView{"vertex_main", 11} : VernonStringView{"fragment_main", 13};
        ASSERT_EQ(provider->prepare_shader(provider->user_data, &descriptor, &shaders[index]), VERNON_STATUS_OK);
    }
    VernonRuntimeProviderPipelineLayoutDescriptor layoutDescriptor{};
    layoutDescriptor.struct_size = sizeof(layoutDescriptor);
    VernonRuntimeProviderObject layout{};
    ASSERT_EQ(provider->prepare_pipeline_layout(provider->user_data, &layoutDescriptor, &layout), VERNON_STATUS_OK);
    const uint32_t colorFormat = vernon::rhi::metal::pixelFormat(VERNON_RHI_FORMAT_RGBA8_UNORM);
    VernonRuntimeProviderPipelineDescriptor pipelineDescriptor{};
    pipelineDescriptor.struct_size = sizeof(pipelineDescriptor);
    pipelineDescriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    pipelineDescriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
    pipelineDescriptor.shaders = shaders.data();
    pipelineDescriptor.shader_count = shaders.size();
    pipelineDescriptor.layout = layout;
    pipelineDescriptor.color_formats = &colorFormat;
    pipelineDescriptor.color_format_count = 1;
    pipelineDescriptor.depth_stencil_format = vernon::rhi::metal::pixelFormat(VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT);
    pipelineDescriptor.sample_count = 1;
    pipelineDescriptor.rasterization.cull_mode = VERNON_RHI_CULL_NONE;
    pipelineDescriptor.rasterization.front_face = VERNON_RHI_FRONT_FACE_COUNTER_CLOCKWISE;
    pipelineDescriptor.depth_stencil.depth_test = 1;
    pipelineDescriptor.depth_stencil.depth_write = 1;
    pipelineDescriptor.depth_stencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    pipelineDescriptor.depth_stencil.stencil_test = 1;
    pipelineDescriptor.depth_stencil.front.compare = VERNON_RHI_COMPARE_ALWAYS;
    pipelineDescriptor.depth_stencil.front.stencil_fail = VERNON_RHI_STENCIL_KEEP;
    pipelineDescriptor.depth_stencil.front.depth_fail = VERNON_RHI_STENCIL_KEEP;
    pipelineDescriptor.depth_stencil.front.pass = VERNON_RHI_STENCIL_REPLACE;
    pipelineDescriptor.depth_stencil.stencil_read_mask = 0xff;
    pipelineDescriptor.depth_stencil.stencil_write_mask = 0xff;
    pipelineDescriptor.depth_stencil.back = pipelineDescriptor.depth_stencil.front;
    VernonColorBlendState blend{};
    blend.source_color_factor = VERNON_RHI_BLEND_ONE;
    blend.destination_color_factor = VERNON_RHI_BLEND_ZERO;
    blend.source_alpha_factor = VERNON_RHI_BLEND_ONE;
    blend.destination_alpha_factor = VERNON_RHI_BLEND_ZERO;
    blend.color_operation = VERNON_RHI_BLEND_ADD;
    blend.alpha_operation = VERNON_RHI_BLEND_ADD;
    blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
    pipelineDescriptor.color_blends = &blend;
    pipelineDescriptor.color_blend_count = 1;
    EXPECT_EQ(vernon::runtime::getRhiAdapterPreparationStats(*adapter).livePreparedPipelines, 0u);
    pipelineDescriptor.depth_stencil.depth_compare = UINT32_MAX;
    VernonRuntimeProviderObject rejectedPipeline{};
    EXPECT_EQ(provider->prepare_pipeline(provider->user_data, &pipelineDescriptor, &rejectedPipeline),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(rejectedPipeline.value, 0u);
    EXPECT_EQ(vernon::runtime::getRhiAdapterPreparationStats(*adapter).livePreparedPipelines, 0u);
    pipelineDescriptor.depth_stencil.depth_compare = VERNON_RHI_COMPARE_LESS;
    VernonRuntimeProviderObject pipeline{};
    ASSERT_EQ(provider->prepare_pipeline(provider->user_data, &pipelineDescriptor, &pipeline), VERNON_STATUS_OK);
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
    VernonRhiImage secondImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &secondImage), VERNON_RHI_STATUS_OK);
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
    VernonRuntimeProviderResourceReference secondImageReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImage(adapter, secondImage, &secondImageReference), VERNON_STATUS_OK);
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

    VernonRhiCommandEncoderDescriptor commandDescriptor{};
    commandDescriptor.struct_size = sizeof(commandDescriptor);
    commandDescriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder command{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &commandDescriptor, &command), VERNON_RHI_STATUS_OK);
    VernonRhiColorAttachment rhiAttachment{};
    rhiAttachment.view = view;
    rhiAttachment.initial_state = VERNON_RHI_STATE_UNDEFINED;
    rhiAttachment.final_state = VERNON_RHI_STATE_TRANSFER_SOURCE;
    rhiAttachment.load_operation = VERNON_RHI_LOAD_CLEAR;
    rhiAttachment.store_operation = VERNON_RHI_STORE_PRESERVE;
    VernonRhiDepthStencilAttachment rhiDepth{};
    rhiDepth.view = depthView;
    rhiDepth.initial_state = VERNON_RHI_STATE_UNDEFINED;
    rhiDepth.final_state = VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT;
    rhiDepth.depth_load_operation = VERNON_RHI_LOAD_CLEAR;
    rhiDepth.depth_store_operation = VERNON_RHI_STORE_PRESERVE;
    rhiDepth.clear_depth = 1.0f;
    rhiDepth.stencil_load_operation = VERNON_RHI_LOAD_CLEAR;
    rhiDepth.stencil_store_operation = VERNON_RHI_STORE_PRESERVE;
    rhiDepth.clear_stencil = 7;
    VernonRhiRenderingDescriptor rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.color_attachments = &rhiAttachment;
    rendering.color_attachment_count = 1;
    rendering.depth_stencil_attachment = &rhiDepth;
    rendering.width = imageDescriptor.width;
    rendering.height = imageDescriptor.height;
    rendering.layers = 1;
    VernonRuntimeProviderObject providerCommand{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceCommandEncoder(adapter, command, &providerCommand), VERNON_STATUS_OK);
    VernonRuntimeProviderDispatchDescriptor invalidDispatch{};
    invalidDispatch.struct_size = sizeof(invalidDispatch);
    invalidDispatch.pipeline = pipeline;
    invalidDispatch.group_count[0] = invalidDispatch.group_count[1] = invalidDispatch.group_count[2] = 1;
    EXPECT_EQ(provider->encode_dispatch(provider->user_data, providerCommand, &invalidDispatch),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(device, command, &rendering), VERNON_RHI_STATUS_OK);
    constexpr float deferredClear[]{1, 0, 0, 1};
    ASSERT_EQ(vernonRhiCommandEncoderClearColorAttachment(device, command, 0, deferredClear), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderClearDepthStencilAttachment(
                  device, command, 0.25f, 2, VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL),
              VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderColorAttachment attachment{};
    attachment.view = imageReference;
    attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    attachment.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    VernonRuntimeProviderDrawDescriptor draw{};
    draw.struct_size = sizeof(draw);
    draw.pipeline = pipeline;
    draw.vertex_count = 6;
    draw.instance_count = 1;
    draw.color_attachments = &attachment;
    draw.color_attachment_count = 1;
    draw.depth_stencil_view = depthReference;
    draw.depth_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    draw.depth_store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    draw.clear_depth = 1.0f;
    draw.stencil_load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    draw.stencil_store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    draw.clear_stencil = 7;
    draw.stencil_reference = 7;
    draw.viewport[2] = imageDescriptor.width;
    draw.viewport[3] = imageDescriptor.height;
    draw.scissor[2] = imageDescriptor.width;
    draw.scissor[3] = imageDescriptor.height;
    draw.index_buffer = indexReference;
    draw.index_count = indices.size();
    draw.index_type = 1;
    EXPECT_EQ(provider->encode_draw(provider->user_data, providerCommand, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    draw.index_buffer = {};
    draw.index_count = 0;
    draw.index_type = 0;
    ASSERT_EQ(provider->encode_draw(provider->user_data, providerCommand, &draw), VERNON_STATUS_OK);
    const auto drawStats = vernon::runtime::getRhiAdapterPreparationStats(*adapter);
    EXPECT_EQ(drawStats.lastStencilReference, 7u);
    EXPECT_FALSE(drawStats.lastDrawIndexed);
    attachment.view = secondImageReference;
    EXPECT_EQ(provider->encode_draw(provider->user_data, providerCommand, &draw), VERNON_STATUS_INVALID_ARGUMENT);
    attachment.view = imageReference;
    constexpr float explicitClear[]{0, 0, 0, 1};
    EXPECT_EQ(vernonRhiCommandEncoderClearColorAttachment(device, command, 0, explicitClear), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderClearDepthStencilAttachment(
                  device, command, 0.5f, 3, VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device, command), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, command), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(device, command), VERNON_RHI_STATUS_OK);
    std::array<uint8_t, 4 * 4 * 4> pixels{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
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
    std::array<uint8_t, 4 * 4 * 8> depthStencil{};
    download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, depthImage, &download, depthStencil.data(), depthStencil.size()),
              VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < depthStencil.size(); index += 8) {
        float depth = 0.0f;
        std::memcpy(&depth, depthStencil.data() + index, sizeof(depth));
        EXPECT_FLOAT_EQ(depth, 0.5f);
        EXPECT_EQ(depthStencil[index + 4], 3);
    }

    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, depthView), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, depthImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, secondImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, indexBuffer), VERNON_RHI_STATUS_OK);
    provider->destroy_pipeline(provider->user_data, pipeline);
    EXPECT_EQ(vernon::runtime::getRhiAdapterPreparationStats(*adapter).livePreparedPipelines, 0u);
    provider->destroy_pipeline_layout(provider->user_data, layout);
    for (const auto shader : shaders)
        provider->destroy_shader(provider->user_data, shader);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

#if defined(VERNON_METAL_COMPUTE_PIPELINE_BUNDLE)
TEST(RuntimeMetal, PublicRuntimeLoadsDispatchesAndReadsBackCookedBundle) {
    const std::filesystem::path manifestPath = VERNON_METAL_COMPUTE_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    if (!supportsMetalArgumentBufferEncoding(device)) {
        vernonRhiDestroyDevice(device);
        GTEST_SKIP() << "Metal argument-buffer encoding is unavailable on this device";
    }
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(VERNON_RUNTIME_METAL, device);
    ASSERT_NE(runtime, nullptr);
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetContextCapabilities(runtime);
    EXPECT_TRUE(capabilities.supports_compute);
    EXPECT_TRUE(capabilities.supports_storage_buffers);
    EXPECT_TRUE(capabilities.supports_graphics);

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

    constexpr std::array<float, 4> source{1, 2, 3, 4};
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(source);
    bufferDescriptor.alignment = alignof(float);
    bufferDescriptor.usage =
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, buffer, 0, source.data(), sizeof(source)), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference bufferReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiBuffer(runtime, buffer, 0, sizeof(source), &bufferReference), VERNON_STATUS_OK);

    constexpr uint64_t shape[]{4};
    constexpr uint64_t scalarShape[]{1};
    constexpr int64_t strides[]{sizeof(float)};
    constexpr float factor = 3.0f;
    VernonProgramArgument arguments[2]{};
    arguments[0].slot = valuesParameter.slot;
    arguments[0].kind = VERNON_PROGRAM_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[0].tensor.resource = bufferReference;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[0].tensor.access = VERNON_ACCESS_READ_WRITE;
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
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(pipeline, arguments, std::size(arguments), {4, 1, 1}),
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    std::array<float, source.size()> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(device, buffer, 0, output.data(), sizeof(output)), VERNON_RHI_STATUS_OK);
    for (size_t index = 0; index < output.size(); ++index)
        EXPECT_EQ(output[index], source[index] * factor * factor);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(device);
}
#endif

#if defined(VERNON_METAL_RESOLUTION_PIPELINE_BUNDLE)
TEST(RuntimeMetal, PublicRuntimeBindsCookedResolutionUniform) {
    const std::filesystem::path manifestPath = VERNON_METAL_RESOLUTION_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(VERNON_RUNTIME_METAL, device);
    ASSERT_NE(runtime, nullptr);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonProgramBundle *loaded =
        vernonRuntimeLoadProgramBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_NE(loaded, nullptr);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(loaded, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << std::string(vernonRuntimeGetLastError(runtime).data,
                                                vernonRuntimeGetLastError(runtime).size);
    VernonProgramParameterView positionParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"position", 8}, &positionParameter),
              VERNON_STATUS_OK);

    constexpr float positions[]{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(positions);
    bufferDescriptor.alignment = alignof(float);
    bufferDescriptor.usage = VERNON_RHI_BUFFER_VERTEX;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer vertices{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &vertices), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, vertices, 0, positions, sizeof(positions)), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference vertexReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiBuffer(runtime, vertices, 0, sizeof(positions), &vertexReference),
              VERNON_STATUS_OK);

    VernonRhiImageDescriptor targetDescriptor{};
    targetDescriptor.struct_size = sizeof(targetDescriptor);
    targetDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    targetDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    targetDescriptor.width = 32;
    targetDescriptor.height = 32;
    targetDescriptor.depth = 1;
    targetDescriptor.mip_levels = 1;
    targetDescriptor.array_layers = 1;
    targetDescriptor.sample_count = 1;
    targetDescriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    VernonRhiImage target{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &targetDescriptor, &target), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor targetViewDescriptor{};
    targetViewDescriptor.struct_size = sizeof(targetViewDescriptor);
    targetViewDescriptor.image = target;
    targetViewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    targetViewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    targetViewDescriptor.mip_level_count = 1;
    targetViewDescriptor.array_layer_count = 1;
    targetViewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView targetView{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &targetViewDescriptor, &targetView), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference targetReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiImageView(runtime, targetView, &targetReference), VERNON_STATUS_OK);

    constexpr uint64_t shape[]{3, 2};
    constexpr int64_t strides[]{2 * sizeof(float), sizeof(float)};
    VernonProgramArgument position{};
    position.slot = positionParameter.slot;
    position.kind = VERNON_PROGRAM_TENSOR;
    position.tensor.struct_size = sizeof(VernonTensorView);
    position.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    position.tensor.resource = vertexReference;
    position.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    position.tensor.access = VERNON_ACCESS_READ;
    position.tensor.rank = 2;
    position.tensor.shape = shape;
    position.tensor.byte_strides = strides;
    position.tensor.byte_size = sizeof(positions);
    VernonColorAttachment attachment{0, targetReference};
    const vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1);
    ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, &position, 1, graphics), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);
    std::vector<uint8_t> pixels(32 * 32 * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = 32;
    download.height = 32;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, target, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (16 * 32 + 16) * 4;
    EXPECT_GT(pixels[center], 250);
    EXPECT_GT(pixels[center + 1], 250);

    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, targetView), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, target), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, vertices), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(device);
}
#endif

#if defined(VERNON_METAL_GRAPHICS_PIPELINE_BUNDLE)
TEST(RuntimeMetal, PublicRuntimeLoadsAndDrawsCookedGraphicsBundle) {
    const std::filesystem::path manifestPath = VERNON_METAL_GRAPHICS_PIPELINE_BUNDLE;
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string bundle((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundle.empty());

    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    if (!supportsMetalArgumentBufferEncoding(device)) {
        vernonRhiDestroyDevice(device);
        GTEST_SKIP() << "Metal argument-buffer encoding is unavailable on this device";
    }
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(VERNON_RUNTIME_METAL, device);
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
    VernonProgramParameterView samplerParameter{};
    VernonProgramParameterView positionParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"image", 5}, &imageParameter), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"sampler", 7}, &samplerParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"position", 8}, &positionParameter),
              VERNON_STATUS_OK);

    constexpr float positions[]{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = sizeof(positions);
    bufferDescriptor.alignment = alignof(float);
    bufferDescriptor.usage = VERNON_RHI_BUFFER_VERTEX;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer vertices{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &vertices), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, vertices, 0, positions, sizeof(positions)), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference vertexReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiBuffer(runtime, vertices, 0, sizeof(positions), &vertexReference),
              VERNON_STATUS_OK);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 1;
    imageDescriptor.height = 1;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage sampled{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &sampled), VERNON_RHI_STATUS_OK);
    constexpr uint8_t sampledPixel[]{64, 200, 100, 128};
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
    ASSERT_EQ(vernonRhiDeviceUploadImage(device, sampled, &upload, 1), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor sampledViewDescriptor{};
    sampledViewDescriptor.struct_size = sizeof(sampledViewDescriptor);
    sampledViewDescriptor.image = sampled;
    sampledViewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    sampledViewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    sampledViewDescriptor.mip_level_count = 1;
    sampledViewDescriptor.array_layer_count = 1;
    sampledViewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView sampledView{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &sampledViewDescriptor, &sampledView), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference sampledReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiImageView(runtime, sampledView, &sampledReference), VERNON_STATUS_OK);

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    samplerDescriptor.min_filter = VERNON_RHI_FILTER_LINEAR;
    samplerDescriptor.mag_filter = VERNON_RHI_FILTER_LINEAR;
    samplerDescriptor.mip_filter = VERNON_RHI_FILTER_LINEAR;
    samplerDescriptor.max_anisotropy = 1.0f;
    VernonRhiSampler sampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &sampler), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference samplerReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiSampler(runtime, sampler, &samplerReference), VERNON_STATUS_OK);

    imageDescriptor.width = 32;
    imageDescriptor.height = 32;
    imageDescriptor.usage =
        VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage target{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &target), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor targetViewDescriptor = sampledViewDescriptor;
    targetViewDescriptor.image = target;
    VernonRhiImageView targetView{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &targetViewDescriptor, &targetView), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference targetReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiImageView(runtime, targetView, &targetReference), VERNON_STATUS_OK);

    constexpr uint64_t shape[]{3, 2};
    constexpr int64_t strides[]{2 * sizeof(float), sizeof(float)};
    VernonProgramArgument arguments[3]{};
    arguments[0].slot = imageParameter.slot;
    arguments[0].kind = VERNON_PROGRAM_IMAGE;
    arguments[0].image = {sampledReference};
    arguments[1].slot = samplerParameter.slot;
    arguments[1].kind = VERNON_PROGRAM_SAMPLER;
    arguments[1].resource = samplerReference;
    arguments[2].slot = positionParameter.slot;
    arguments[2].kind = VERNON_PROGRAM_TENSOR;
    arguments[2].tensor.struct_size = sizeof(VernonTensorView);
    arguments[2].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[2].tensor.resource = vertexReference;
    arguments[2].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[2].tensor.access = VERNON_ACCESS_READ;
    arguments[2].tensor.rank = 2;
    arguments[2].tensor.shape = shape;
    arguments[2].tensor.byte_strides = strides;
    arguments[2].tensor.byte_size = sizeof(positions);
    constexpr std::array<uint32_t, 3> indices{0, 1, 2};
    bufferDescriptor.size = sizeof(indices);
    bufferDescriptor.alignment = alignof(uint32_t);
    bufferDescriptor.usage = VERNON_RHI_BUFFER_INDEX;
    VernonRhiBuffer indexBuffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &indexBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(device, indexBuffer, 0, indices.data(), sizeof(indices)),
              VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference indexReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiBuffer(runtime, indexBuffer, 0, sizeof(indices), &indexReference),
              VERNON_STATUS_OK);
    VernonColorAttachment attachment{0, targetReference};
    vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
    VernonIndexBinding invalidIndex{static_cast<VernonIndexType>(1), 0, indices.size(), indexReference};
    graphics.draw.index_binding = &invalidIndex;
    EXPECT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, arguments, std::size(arguments), graphics),
              VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStringView indexError = vernonRuntimeGetLastError(runtime);
    EXPECT_NE(std::string(indexError.data, indexError.size).find("DrawCommand"), std::string::npos);
    graphics.draw.index_binding = nullptr;
    ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, arguments, std::size(arguments), graphics),
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(runtime).data, vernonRuntimeGetLastError(runtime).size);

    std::vector<uint8_t> pixels(32 * 32 * 4);
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = 32;
    download.height = 32;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, target, &download, pixels.data(), pixels.size()),
              VERNON_RHI_STATUS_OK);
    const size_t center = (16 * 32 + 16) * 4;
    EXPECT_NEAR(pixels[center], sampledPixel[0], 2);
    EXPECT_NEAR(pixels[center + 1], sampledPixel[1], 2);
    EXPECT_NEAR(pixels[center + 2], sampledPixel[2], 2);

    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, targetView), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, target), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySampler(device, sampler), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, sampledView), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, sampled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, indexBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, vertices), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(device);
}
#endif
