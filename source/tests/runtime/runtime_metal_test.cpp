#include "../../lib/rhi/rhi_internal.h"
#include "../../lib/rhi/sampler_filter.h"
#include "VernonRHI.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

VernonRhiDevice createMetalDevice() {
    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = VERNON_RHI_BACKEND_METAL;
    return vernonRhiCreateDevice(&descriptor);
}

} // namespace

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
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceSynchronize(device), VERNON_RHI_STATUS_OK);

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
    ASSERT_EQ(vernonRhiDeviceDownloadImage(device, image, destination.data(), destination.size()),
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

TEST(RuntimeMetal, RejectsBarriersUntilMetalSynchronizationIsImplemented) {
    VernonRhiDevice device = createMetalDevice();
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 16;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer buffer{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    VernonRhiBarrier barrier{};
    barrier.struct_size = sizeof(barrier);
    barrier.buffer = buffer;
    barrier.old_state = VERNON_RHI_STATE_SHADER_WRITE;
    barrier.new_state = VERNON_RHI_STATE_SHADER_READ;
    EXPECT_EQ(vernonRhiCommandEncoderBarrier(device, encoder, &barrier, 1), VERNON_RHI_STATUS_UNSUPPORTED);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
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
        EXPECT_EQ(vernonRhiDeviceDownloadImage(device, image, destination.data(), destination.size()),
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
