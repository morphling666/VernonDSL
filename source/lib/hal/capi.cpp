#include "VernonHal.h"

#include "VernonSchedule.h"
#include "hal/cuda/cuda_device.h"
#include "hal/opengl/opengl_resources.h"
#include "hal/resources.h"
#include "hal/vulkan/vulkan_resources.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <string>

namespace {

using vernon::hal::Failure;

thread_local std::string lastError;

VernonHalStatus report(VernonHalStatus status, const std::string &message) noexcept {
    try {
        lastError = message;
    } catch (...) {
        lastError.clear();
    }
    return status;
}

VernonHalStatus report(const Failure &failure) noexcept { return report(failure.status, failure.message); }

VernonHalStatus invalid(const char *message) noexcept { return report(VERNON_HAL_INVALID_ARGUMENT, message); }

template <typename Body> VernonHalStatus boundary(Body &&body) noexcept {
    try {
        return body();
    } catch (const std::bad_alloc &) {
        return report(VERNON_HAL_RESOURCE_EXHAUSTED, "out of memory");
    } catch (...) {
        return report(VERNON_HAL_INTERNAL_ERROR, "an unexpected exception reached the HAL boundary");
    }
}

std::string runtimeError(const VernonRuntimeContext *context, const char *what) {
    std::string message = what;
    const VernonStringView detail = vernonRuntimeGetLastError(context);
    if (detail.data && detail.size)
        message += ": " + std::string(detail.data, detail.size);
    return message;
}

// The RHI device that a CUDA, Vulkan or OpenGL HAL device runs on.
vernon::hal::Done<VernonRhiDevice> createRhiDevice(const VernonHalDeviceDescriptor &descriptor) {
    using Created = vernon::hal::Done<VernonRhiDevice>;
    uint32_t flags = 0;
    if (descriptor.flags & VERNON_HAL_DEVICE_VALIDATION)
        flags |= VERNON_RHI_DEVICE_VALIDATION;
    if (descriptor.flags & VERNON_HAL_DEVICE_LABELS)
        flags |= VERNON_RHI_DEVICE_LABELS;
    if (descriptor.flags & VERNON_HAL_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT)
        flags |= VERNON_RHI_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT;
    const VernonRhiDevice invalidDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiDevice device = invalidDevice;
    switch (descriptor.backend) {
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        if (!descriptor.opengl)
            return vernon::hal::fail<VernonRhiDevice>(VERNON_HAL_INVALID_ARGUMENT, "an OpenGL device adopts a context");
        device = vernonRhiCreateOpenGLDevice(descriptor.opengl, descriptor.backend == VERNON_RUNTIME_OPENGL_ES, flags);
        break;
    case VERNON_RUNTIME_CUDA:
    case VERNON_RUNTIME_VULKAN: {
        VernonRhiOwnedDeviceDescriptor owned{};
        owned.struct_size = sizeof(owned);
        owned.backend = descriptor.backend == VERNON_RUNTIME_CUDA ? VERNON_RHI_BACKEND_CUDA : VERNON_RHI_BACKEND_VULKAN;
        owned.device_index = descriptor.device_index;
        owned.flags = flags;
        if (descriptor.flags & VERNON_HAL_DEVICE_FORCE_SOFTWARE)
            owned.flags |= VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE;
        device = vernonRhiCreateDevice(&owned);
        break;
    }
    case VERNON_RUNTIME_DIRECTX12:
    case VERNON_RUNTIME_METAL:
        return vernon::hal::fail<VernonRhiDevice>(VERNON_HAL_UNSUPPORTED,
                                                  "DirectX 12 and Metal have no HAL device yet");
    default:
        return vernon::hal::fail<VernonRhiDevice>(VERNON_HAL_INVALID_ARGUMENT, "the backend is unknown");
    }
    if (device.index == invalidDevice.index) {
        std::string message = "cannot create the device";
        const VernonStringView detail = vernonRhiDeviceGetLastError(invalidDevice);
        if (detail.data && detail.size)
            message += ": " + std::string(detail.data, detail.size);
        return vernon::hal::fail<VernonRhiDevice>(VERNON_HAL_UNSUPPORTED, std::move(message));
    }
    return Created{vernon::ok(device)};
}

bool fits(uint64_t offset, uint64_t size, uint64_t total) { return offset <= total && size <= total - offset; }

bool inside(const VernonImageDescriptor &image, const VernonHalImageRegion &region) {
    if (region.mip_level >= image.mip_level_count || region.array_layer >= image.array_layer_count || !region.width ||
        !region.height || !region.depth || !region.aspect)
        return false;
    const auto extent = [&](uint32_t size) { return std::max<uint32_t>(1, size >> region.mip_level); };
    return fits(region.offset_x, region.width, extent(image.extent.width)) &&
           fits(region.offset_y, region.height, extent(image.extent.height)) &&
           fits(region.offset_z, region.depth, extent(image.extent.depth));
}

} // namespace

extern "C" {

VernonStringView vernonHalGetLastError(void) { return VernonStringView{lastError.data(), lastError.size()}; }

VernonHalStatus vernonHalCreateDevice(const VernonHalDeviceDescriptor *descriptor, VernonHalDevice **output) {
    return boundary([&] {
        if (!output)
            return invalid("the device has no output");
        *output = nullptr;
        if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
            return invalid("the device descriptor is invalid");
        auto device = std::make_unique<VernonHalDevice>();
        device->backend = descriptor->backend;
        if (descriptor->backend == VERNON_RUNTIME_CPU) {
            if (descriptor->opengl)
                return invalid("only an OpenGL device adopts a context");
            VernonRuntimeCreateOptions options{};
            options.struct_size = sizeof(options);
            options.device_index = descriptor->device_index;
            device->runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, &options);
            if (!device->runtime)
                return report(VERNON_HAL_INTERNAL_ERROR, runtimeError(nullptr, "cannot create the CPU runtime"));
            device->resources = vernon::hal::createCpuResources();
        } else {
            auto rhi = createRhiDevice(*descriptor);
            if (rhi.isErr())
                return report(rhi.error());
            const VernonRhiDevice handle = rhi.value();
            device->release = [handle] { (void)vernonRhiDestroyDevice(handle); };
            device->runtime = vernonRuntimeCreateForRhiDevice(descriptor->backend, handle, nullptr);
            if (!device->runtime) {
                device->release();
                return report(VERNON_HAL_INTERNAL_ERROR, runtimeError(nullptr, "cannot create the device's runtime"));
            }
            if (descriptor->backend == VERNON_RUNTIME_CUDA)
                device->resources = vernon::hal::createCudaResources(handle);
#if VERNON_HAS_VULKAN_RUNTIME
            else if (descriptor->backend == VERNON_RUNTIME_VULKAN)
                device->resources = vernon::hal::createVulkanResources(handle);
#endif
            else if (descriptor->backend == VERNON_RUNTIME_OPENGL || descriptor->backend == VERNON_RUNTIME_OPENGL_ES)
                device->resources = vernon::hal::createOpenGLResources(handle);
        }
        if (!device->resources) {
            (void)vernonRuntimeDestroy(device->runtime);
            if (device->release)
                device->release();
            return report(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate the device");
        }
        *output = device.release();
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalDestroyDevice(VernonHalDevice *device) {
    return boundary([&] {
        if (!device)
            return invalid("the device is null");
        if (device->objects.load())
            return invalid("the device still has objects");
        // The runtime goes first: it fails, leaving the device whole, while schedules remain linked on it.
        if (vernonRuntimeDestroy(device->runtime) != VERNON_STATUS_OK)
            return report(VERNON_HAL_INVALID_ARGUMENT, runtimeError(device->runtime, "cannot destroy the runtime"));
        device->resources.reset();
        if (device->release)
            device->release();
        delete device;
        return VERNON_HAL_OK;
    });
}

VernonRuntimeBackend vernonHalDeviceGetBackend(const VernonHalDevice *device) {
    return device ? device->backend : VERNON_RUNTIME_CPU;
}

VernonRuntimeContext *vernonHalDeviceGetRuntime(VernonHalDevice *device) { return device ? device->runtime : nullptr; }

VernonHalStatus vernonHalDeviceGetFormatCapabilities(VernonHalDevice *device, VernonTextureFormat format,
                                                     uint32_t *capabilities) {
    return boundary([&] {
        if (!device || !capabilities)
            return invalid("the device or the output is null");
        auto queried = device->resources->formatCapabilities(format);
        if (queried.isErr())
            return report(queried.error());
        *capabilities = queried.value();
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalDeviceWait(VernonHalDevice *device, uint64_t point) {
    return boundary([&] {
        if (!device)
            return invalid("the device is null");
        if (vernonScheduleWait(device->runtime, point) != VERNON_STATUS_OK)
            return report(VERNON_HAL_INTERNAL_ERROR, runtimeError(device->runtime, "the device failed the submission"));
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalCreateBuffer(VernonHalDevice *device, const VernonHalBufferDescriptor *descriptor,
                                      VernonHalBuffer **output) {
    return boundary([&] {
        if (!output)
            return invalid("the buffer has no output");
        *output = nullptr;
        constexpr uint32_t kUsage = VERNON_HAL_BUFFER_UNIFORM | VERNON_HAL_BUFFER_STORAGE | VERNON_HAL_BUFFER_VERTEX |
                                    VERNON_HAL_BUFFER_INDEX | VERNON_HAL_BUFFER_INDIRECT;
        if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->size ||
            (descriptor->usage & ~kUsage) || descriptor->memory > VERNON_HAL_MEMORY_READBACK)
            return invalid("the buffer descriptor is invalid");
        auto created = device->resources->createBuffer(*descriptor);
        if (created.isErr())
            return report(created.error());
        VernonHalBuffer *buffer = std::move(created).value().release();
        buffer->device = device;
        buffer->descriptor = *descriptor;
        ++device->objects;
        *output = buffer;
        return VERNON_HAL_OK;
    });
}

void vernonHalDestroyBuffer(VernonHalBuffer *buffer) {
    if (!buffer)
        return;
    --buffer->device->objects;
    delete buffer;
}

uint64_t vernonHalBufferGetSize(const VernonHalBuffer *buffer) { return buffer ? buffer->descriptor.size : 0; }

VernonHalStatus vernonHalBufferWrite(VernonHalBuffer *buffer, const VernonHalBufferWrite *writes, size_t write_count) {
    return boundary([&] {
        if (!buffer || (write_count && !writes))
            return invalid("the buffer or its writes are null");
        for (size_t index = 0; index < write_count; ++index)
            if ((writes[index].size && !writes[index].source) ||
                !fits(writes[index].offset, writes[index].size, buffer->descriptor.size))
                return invalid("a write lies outside the buffer or has no source");
        auto written = buffer->device->resources->write(*buffer, writes, write_count);
        return written.isErr() ? report(written.error()) : VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalBufferRead(VernonHalBuffer *buffer, const VernonHalBufferRead *reads, size_t read_count) {
    return boundary([&] {
        if (!buffer || (read_count && !reads))
            return invalid("the buffer or its reads are null");
        for (size_t index = 0; index < read_count; ++index)
            if ((reads[index].size && !reads[index].destination) ||
                !fits(reads[index].offset, reads[index].size, buffer->descriptor.size))
                return invalid("a read lies outside the buffer or has no destination");
        auto read = buffer->device->resources->read(*buffer, reads, read_count);
        return read.isErr() ? report(read.error()) : VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalBufferGetAddress(const VernonHalBuffer *buffer, void **output) {
    return boundary([&] {
        if (!buffer || !output)
            return invalid("the buffer or the output is null");
        *output = nullptr;
        auto address = buffer->device->resources->address(*buffer);
        if (address.isErr())
            return report(address.error());
        *output = address.value();
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalCreateImage(VernonHalDevice *device, const VernonImageDescriptor *descriptor,
                                     VernonHalImage **output) {
    return boundary([&] {
        if (!output)
            return invalid("the image has no output");
        *output = nullptr;
        if (!device || !descriptor)
            return invalid("the device or the image descriptor is null");
        auto created = device->resources->createImage(*descriptor);
        if (created.isErr())
            return report(created.error());
        VernonHalImage *image = std::move(created).value().release();
        image->device = device;
        image->descriptor = *descriptor;
        ++device->objects;
        *output = image;
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalDestroyImage(VernonHalImage *image) {
    return boundary([&] {
        if (!image)
            return invalid("the image is null");
        if (image->surface)
            return invalid("a surface's image belongs to the surface");
        if (image->views.load())
            return invalid("the image still has views");
        --image->device->objects;
        delete image;
        return VERNON_HAL_OK;
    });
}

const VernonImageDescriptor *vernonHalImageGetDescriptor(const VernonHalImage *image) {
    return image ? &image->descriptor : nullptr;
}

VernonHalStatus vernonHalImageWrite(VernonHalImage *image, const VernonHalImageWrite *writes, size_t write_count) {
    return boundary([&] {
        if (!image || (write_count && !writes))
            return invalid("the image or its writes are null");
        for (size_t index = 0; index < write_count; ++index)
            if (!writes[index].source || !inside(image->descriptor, writes[index].region))
                return invalid("a write lies outside the image or has no source");
        auto written = image->device->resources->write(*image, writes, write_count);
        return written.isErr() ? report(written.error()) : VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalImageRead(VernonHalImage *image, const VernonHalImageRead *reads, size_t read_count) {
    return boundary([&] {
        if (!image || (read_count && !reads))
            return invalid("the image or its reads are null");
        for (size_t index = 0; index < read_count; ++index)
            if (!reads[index].destination || !inside(image->descriptor, reads[index].region))
                return invalid("a read lies outside the image or has no destination");
        auto read = image->device->resources->read(*image, reads, read_count);
        return read.isErr() ? report(read.error()) : VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalImageGenerateMipmaps(VernonHalImage *image) {
    return boundary([&] {
        if (!image)
            return invalid("the image is null");
        auto generated = image->device->resources->generateMipmaps(*image);
        return generated.isErr() ? report(generated.error()) : VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalCreateImageView(VernonHalImage *image, const VernonImageViewDescriptor *descriptor,
                                         VernonHalImageView **output) {
    return boundary([&] {
        if (!output)
            return invalid("the view has no output");
        *output = nullptr;
        if (!image || !descriptor)
            return invalid("the image or the view descriptor is null");
        auto created = image->device->resources->createView(*image, *descriptor);
        if (created.isErr())
            return report(created.error());
        VernonHalImageView *view = std::move(created).value().release();
        view->image = image;
        view->descriptor = *descriptor;
        ++image->views;
        ++image->device->objects;
        *output = view;
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalDestroyImageView(VernonHalImageView *view) {
    return boundary([&] {
        if (!view)
            return invalid("the view is null");
        VernonHalImage *image = view->image;
        if (image->surface)
            return invalid("a surface image's view belongs to the surface");
        --image->views;
        --image->device->objects;
        delete view;
        return VERNON_HAL_OK;
    });
}

VernonHalImage *vernonHalImageViewGetImage(const VernonHalImageView *view) { return view ? view->image : nullptr; }

VernonHalDevice *vernonHalBufferGetDevice(const VernonHalBuffer *buffer) { return buffer ? buffer->device : nullptr; }

VernonHalDevice *vernonHalImageGetDevice(const VernonHalImage *image) { return image ? image->device : nullptr; }

VernonHalDevice *vernonHalSamplerGetDevice(const VernonHalSampler *sampler) {
    return sampler ? sampler->device : nullptr;
}

VernonHalStatus vernonHalCreateSampler(VernonHalDevice *device, const VernonHalSamplerDescriptor *descriptor,
                                       VernonHalSampler **output) {
    return boundary([&] {
        if (!output)
            return invalid("the sampler has no output");
        *output = nullptr;
        if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor))
            return invalid("the sampler descriptor is invalid");
        const auto filter = [](VernonHalFilter value) {
            return value == VERNON_HAL_FILTER_NEAREST || value == VERNON_HAL_FILTER_LINEAR;
        };
        const auto address = [](VernonHalAddressMode value) {
            return value == VERNON_HAL_ADDRESS_REPEAT || value == VERNON_HAL_ADDRESS_CLAMP_TO_EDGE ||
                   value == VERNON_HAL_ADDRESS_MIRRORED_REPEAT;
        };
        if (!filter(descriptor->min_filter) || !filter(descriptor->mag_filter) || !filter(descriptor->mip_filter) ||
            !address(descriptor->address_u) || !address(descriptor->address_v) || !address(descriptor->address_w) ||
            !std::isfinite(descriptor->max_anisotropy) || descriptor->max_anisotropy < 0.0f ||
            descriptor->max_anisotropy > 16.0f)
            return invalid("the sampler filter, address mode or anisotropy is out of range");
        auto created = device->resources->createSampler(*descriptor);
        if (created.isErr())
            return report(created.error());
        VernonHalSampler *sampler = std::move(created).value().release();
        sampler->device = device;
        sampler->descriptor = *descriptor;
        ++device->objects;
        *output = sampler;
        return VERNON_HAL_OK;
    });
}

void vernonHalDestroySampler(VernonHalSampler *sampler) {
    if (!sampler)
        return;
    --sampler->device->objects;
    delete sampler;
}

VernonHalStatus vernonHalCreateSurface(VernonHalDevice *device, const VernonHalSurfaceDescriptor *descriptor,
                                       VernonHalSurface **output) {
    return boundary([&] {
        if (!output)
            return invalid("the surface has no output");
        *output = nullptr;
        if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor))
            return invalid("the surface descriptor is invalid");
        auto created = device->resources->createSurface(*descriptor);
        if (created.isErr())
            return report(created.error());
        VernonHalSurface *surface = std::move(created).value().release();
        surface->device = device;
        ++device->objects;
        *output = surface;
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalDestroySurface(VernonHalSurface *surface) {
    return boundary([&] {
        if (!surface)
            return invalid("the surface is null");
        if (surface->acquired)
            return invalid("the surface has an acquired image");
        --surface->device->objects;
        delete surface;
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalSurfaceGetCapabilities(VernonHalSurface *surface, VernonHalSurfaceCapabilities *output) {
    return boundary([&] {
        if (!surface || !output)
            return invalid("the surface or the output is null");
        auto queried = surface->device->resources->capabilities(*surface);
        if (queried.isErr())
            return report(queried.error());
        *output = queried.value();
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalSurfaceConfigure(VernonHalSurface *surface,
                                          const VernonHalSurfaceConfiguration *configuration) {
    return boundary([&] {
        if (!surface || !configuration || configuration->struct_size < sizeof(*configuration))
            return invalid("the surface configuration is invalid");
        if (surface->acquired)
            return invalid("the surface has an acquired image");
        if (!configuration->width || !configuration->height)
            return invalid("the surface's extent is zero");
        if (!(configuration->usage & VERNON_IMAGE_COLOR_ATTACHMENT))
            return invalid("a surface's images are color attachments");
        auto configured = surface->device->resources->configure(*surface, *configuration);
        return configured.isErr() ? report(configured.error()) : VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalSurfaceAcquire(VernonHalSurface *surface, VernonHalSurfaceImage *output) {
    return boundary([&] {
        if (!surface || !output)
            return invalid("the surface or the output is null");
        if (surface->acquired)
            return invalid("the surface already has an acquired image");
        auto acquired = surface->device->resources->acquire(*surface);
        if (acquired.isErr())
            return report(acquired.error());
        surface->acquired = true;
        *output = acquired.value();
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalSurfacePresent(VernonHalSurface *surface) {
    return boundary([&] {
        if (!surface)
            return invalid("the surface is null");
        if (!surface->acquired)
            return invalid("the surface has no acquired image");
        surface->acquired = false;
        auto presented = surface->device->resources->present(*surface);
        return presented.isErr() ? report(presented.error()) : VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalReferenceBuffer(const VernonHalBuffer *buffer, uint64_t offset, uint64_t size,
                                         VernonRuntimeProviderResourceReference *output) {
    return boundary([&] {
        if (!buffer || !output)
            return invalid("the buffer or the output is null");
        if (!fits(offset, size, buffer->descriptor.size))
            return invalid("the referenced range lies outside the buffer");
        VernonHalDevice &device = *buffer->device;
        auto made = device.resources->reference(*device.runtime, *buffer, offset, size);
        if (made.isErr())
            return report(made.error());
        *output = made.value();
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalReferenceImageView(const VernonHalImageView *view,
                                            VernonRuntimeProviderResourceReference *output) {
    return boundary([&] {
        if (!view || !output)
            return invalid("the view or the output is null");
        VernonHalDevice &device = *view->image->device;
        auto made = device.resources->reference(*device.runtime, *view);
        if (made.isErr())
            return report(made.error());
        *output = made.value();
        return VERNON_HAL_OK;
    });
}

VernonHalStatus vernonHalReferenceSampler(const VernonHalSampler *sampler,
                                          VernonRuntimeProviderResourceReference *output) {
    return boundary([&] {
        if (!sampler || !output)
            return invalid("the sampler or the output is null");
        VernonHalDevice &device = *sampler->device;
        auto made = device.resources->reference(*device.runtime, *sampler);
        if (made.isErr())
            return report(made.error());
        *output = made.value();
        return VERNON_HAL_OK;
    });
}

} // extern "C"
