#import <Foundation/Foundation.h>

#include "VernonRHI.h"

#include <array>
#include <cstdint>

namespace {

bool testBuffers(VernonRhiDevice device) {
    constexpr std::array<uint32_t, 4> source{0x12345678u, 2u, 3u, 0xabcdef01u};
    for (VernonRhiMemoryClass memoryClass :
         {VERNON_RHI_MEMORY_DEVICE, VERNON_RHI_MEMORY_UPLOAD, VERNON_RHI_MEMORY_READBACK}) {
        VernonRhiBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = sizeof(source);
        descriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION |
                           VERNON_RHI_BUFFER_STORAGE;
        descriptor.memory_class = memoryClass;
        VernonRhiBuffer buffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        std::array<uint32_t, source.size()> destination{};
        const bool success =
            vernonRhiDeviceCreateBuffer(device, &descriptor, &buffer) == VERNON_RHI_STATUS_OK &&
            vernonRhiDeviceUploadBuffer(device, buffer, 0, source.data(), sizeof(source)) == VERNON_RHI_STATUS_OK &&
            vernonRhiDeviceDownloadBuffer(device, buffer, 0, destination.data(), sizeof(destination)) ==
                VERNON_RHI_STATUS_OK &&
            destination == source;
        if (buffer.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyBuffer(device, buffer);
        if (!success)
            return false;
    }
    return true;
}

bool testImageAndSampler(VernonRhiDevice device) {
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
    VernonRhiImage image{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    if (vernonRhiDeviceCreateImage(device, &descriptor, &image) != VERNON_RHI_STATUS_OK)
        return false;

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
    std::array<uint8_t, source.size()> destination{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = descriptor.width;
    download.height = descriptor.height;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    bool success = vernonRhiDeviceUploadImage(device, image, &upload, 1) == VERNON_RHI_STATUS_OK &&
                   vernonRhiDeviceGenerateImageMipmaps(device, image) == VERNON_RHI_STATUS_OK &&
                   vernonRhiDeviceDownloadImage(device, image, &download, destination.data(), destination.size()) ==
                       VERNON_RHI_STATUS_OK &&
                   destination == source;

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    samplerDescriptor.min_filter = VERNON_RHI_FILTER_NEAREST_MIPMAP_LINEAR;
    samplerDescriptor.mag_filter = VERNON_RHI_FILTER_LINEAR;
    samplerDescriptor.mip_filter = VERNON_RHI_FILTER_NEAREST;
    samplerDescriptor.address_u = VERNON_RHI_ADDRESS_REPEAT;
    samplerDescriptor.address_v = VERNON_RHI_ADDRESS_CLAMP_TO_EDGE;
    samplerDescriptor.address_w = VERNON_RHI_ADDRESS_MIRRORED_REPEAT;
    samplerDescriptor.max_anisotropy = 1.0f;
    VernonRhiSampler sampler{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    success = success &&
              vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &sampler) == VERNON_RHI_STATUS_OK;
    if (sampler.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        success = vernonRhiDeviceDestroySampler(device, sampler) == VERNON_RHI_STATUS_OK && success;
    success = vernonRhiDeviceDestroyImage(device, image) == VERNON_RHI_STATUS_OK && success;
    return success;
}

} // namespace

int main() {
    @autoreleasepool {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = VERNON_RHI_BACKEND_METAL;
        const VernonRhiDevice device = vernonRhiCreateDevice(&descriptor);
        if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
            return 1;
        const bool success = testBuffers(device) && testImageAndSampler(device) &&
                             vernonRhiDeviceSynchronize(device) == VERNON_RHI_STATUS_OK;
        vernonRhiDestroyDevice(device);
        NSLog(@"Vernon iOS Metal smoke test %@", success ? @"passed" : @"failed");
        return success ? 0 : 2;
    }
}
