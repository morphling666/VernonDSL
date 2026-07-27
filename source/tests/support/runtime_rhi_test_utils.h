#ifndef VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H
#define VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H

#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>

namespace vernon::tests {

struct RhiRuntime {
    VernonRhiDevice device{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeContext *runtime{};
};

struct RhiBuffer {
    VernonRhiBuffer handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

struct RhiImage {
    VernonRhiImage handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

struct RhiSampler {
    VernonRhiSampler handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

inline VernonRhiBackend rhiBackend(VernonRuntimeBackend backend) {
    switch (backend) {
    case VERNON_RUNTIME_CUDA:
        return VERNON_RHI_BACKEND_CUDA;
    case VERNON_RUNTIME_VULKAN:
        return VERNON_RHI_BACKEND_VULKAN;
    case VERNON_RUNTIME_DIRECTX12:
        return VERNON_RHI_BACKEND_DIRECTX12;
    case VERNON_RUNTIME_OPENGL:
        return VERNON_RHI_BACKEND_OPENGL;
    case VERNON_RUNTIME_OPENGL_ES:
        return VERNON_RHI_BACKEND_OPENGL_ES;
    default:
        return VERNON_RHI_BACKEND_CUDA;
    }
}

inline RhiRuntime createRhiRuntime(VernonRuntimeBackend backend,
                                   const VernonOpenGLContextCallbacks *openglCallbacks = nullptr,
                                   bool forceSoftware = false) {
    RhiRuntime result;
    if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES)
        result.device = vernonRhiCreateOpenGLDevice(openglCallbacks, backend == VERNON_RUNTIME_OPENGL_ES);
    else {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = rhiBackend(backend);
        descriptor.flags = forceSoftware ? VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE : 0;
        result.device = vernonRhiCreateDevice(&descriptor);
    }
    if (result.device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        result.runtime = vernonRuntimeCreateForRhiDevice(backend, result.device);
    return result;
}

inline void destroyRhiRuntime(RhiRuntime &context) {
    if (context.runtime)
        vernonRuntimeDestroy(context.runtime);
    if (context.device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        vernonRhiDestroyDevice(context.device);
    context = {};
}

inline RhiBuffer createBuffer(RhiRuntime &context, uint64_t size, uint64_t alignment, uint32_t usage,
                              const void *initialData = nullptr) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.alignment = alignment;
    descriptor.usage = usage | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    RhiBuffer result;
    if (vernonRhiDeviceCreateBuffer(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (initialData &&
        vernonRhiDeviceUploadBuffer(context.device, result.handle, 0, initialData, size) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiBuffer(context.runtime, result.handle, 0, size, &result.reference) != VERNON_STATUS_OK)
        return result;
    return result;
}

inline RhiImage createImage(RhiRuntime &context, VernonRhiImageDimension dimension, VernonRhiFormat format,
                            uint32_t width, uint32_t height, uint32_t depth, uint32_t usage, uint32_t arrayLayers = 1) {
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = dimension;
    descriptor.format = format;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.depth = depth;
    descriptor.mip_levels = 1;
    descriptor.array_layers = arrayLayers;
    descriptor.sample_count = 1;
    descriptor.usage = usage | VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    RhiImage result;
    if (vernonRhiDeviceCreateImage(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiImage(context.runtime, result.handle, &result.reference) != VERNON_STATUS_OK)
        return result;
    return result;
}

inline RhiSampler createSampler(RhiRuntime &context, uint32_t filter = VERNON_RHI_FILTER_NEAREST,
                                uint32_t address = VERNON_RHI_ADDRESS_CLAMP_TO_EDGE) {
    VernonRhiSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.mip_filter = filter;
    descriptor.address_u = address;
    descriptor.address_v = address;
    descriptor.address_w = address;
    RhiSampler result;
    if (vernonRhiDeviceCreateSampler(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiSampler(context.runtime, result.handle, &result.reference) != VERNON_STATUS_OK)
        return result;
    return result;
}

} // namespace vernon::tests

#endif
