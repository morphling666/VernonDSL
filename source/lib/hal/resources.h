#ifndef VERNON_HAL_RESOURCES_H
#define VERNON_HAL_RESOURCES_H

// The objects of VernonHal.h and the interface each backend implements them through. The C API checks what every
// backend relies on (null handles, struct sizes, transfer ranges inside their resources, counts of children, the
// surface's acquire state) and keeps the counts; a backend checks what its device decides.

#include "VernonHal.h"
#include "VernonResult.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace vernon::hal {

struct Failure {
    VernonHalStatus status{VERNON_HAL_INTERNAL_ERROR};
    std::string message;
};

template <typename T> using Done = Result<T, Failure>;

template <typename T = void> Done<T> fail(VernonHalStatus status, std::string message) {
    return Done<T>{err(Failure{status, std::move(message)})};
}

} // namespace vernon::hal

struct VernonHalBuffer {
    virtual ~VernonHalBuffer() = default;
    VernonHalDevice *device{};
    VernonHalBufferDescriptor descriptor{};
};

struct VernonHalImage {
    virtual ~VernonHalImage() = default;
    VernonHalDevice *device{};
    VernonImageDescriptor descriptor{};
    std::atomic<uint32_t> views{};
    // Set on a surface's image, which only the surface destroys.
    VernonHalSurface *surface{};
};

struct VernonHalImageView {
    virtual ~VernonHalImageView() = default;
    VernonHalImage *image{};
    VernonImageViewDescriptor descriptor{};
};

struct VernonHalSampler {
    virtual ~VernonHalSampler() = default;
    VernonHalDevice *device{};
    VernonHalSamplerDescriptor descriptor{};
};

struct VernonHalSurface {
    virtual ~VernonHalSurface() = default;
    VernonHalDevice *device{};
    bool acquired{};
};

namespace vernon::hal {

// Image transfers name one region each; the C API has checked it lies inside the image and that buffer ranges lie
// inside their buffer.
class Resources {
public:
    virtual ~Resources() = default;

    virtual Done<uint32_t> formatCapabilities(VernonTextureFormat format) = 0;
    virtual Done<std::unique_ptr<VernonHalBuffer>> createBuffer(const VernonHalBufferDescriptor &descriptor) = 0;
    virtual Done<void> write(VernonHalBuffer &buffer, const VernonHalBufferWrite *writes, size_t count) = 0;
    virtual Done<void> read(VernonHalBuffer &buffer, const VernonHalBufferRead *reads, size_t count) = 0;
    virtual Done<void *> address(const VernonHalBuffer &buffer) = 0;
    virtual Done<std::unique_ptr<VernonHalImage>> createImage(const VernonImageDescriptor &descriptor) = 0;
    virtual Done<void> write(VernonHalImage &image, const VernonHalImageWrite *writes, size_t count) = 0;
    virtual Done<void> read(VernonHalImage &image, const VernonHalImageRead *reads, size_t count) = 0;
    virtual Done<void> generateMipmaps(VernonHalImage &image) = 0;
    virtual Done<std::unique_ptr<VernonHalImageView>> createView(VernonHalImage &image,
                                                                 const VernonImageViewDescriptor &descriptor) = 0;
    virtual Done<std::unique_ptr<VernonHalSampler>> createSampler(const VernonHalSamplerDescriptor &descriptor) = 0;
    virtual Done<std::unique_ptr<VernonHalSurface>> createSurface(const VernonHalSurfaceDescriptor &descriptor) = 0;
    virtual Done<VernonHalSurfaceCapabilities> capabilities(VernonHalSurface &surface) = 0;
    virtual Done<void> configure(VernonHalSurface &surface, const VernonHalSurfaceConfiguration &configuration) = 0;
    // The image and view belong to the surface and stay valid until the next present; the image's `surface` is set.
    virtual Done<VernonHalSurfaceImage> acquire(VernonHalSurface &surface) = 0;
    virtual Done<void> present(VernonHalSurface &surface) = 0;
    // The references schedule arguments name resources by, made on the device's runtime.
    virtual Done<VernonRuntimeProviderResourceReference>
    reference(VernonRuntimeContext &runtime, const VernonHalBuffer &buffer, uint64_t offset, uint64_t size) = 0;
    virtual Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &runtime,
                                                                   const VernonHalImageView &view) = 0;
    virtual Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &runtime,
                                                                   const VernonHalSampler &sampler) = 0;
};

// The CPU device's resources, in host memory.
std::unique_ptr<Resources> createCpuResources();

} // namespace vernon::hal

struct VernonHalDevice {
    VernonRuntimeBackend backend{};
    VernonRuntimeContext *runtime{};
    std::unique_ptr<vernon::hal::Resources> resources;
    // Buffers, images, views, samplers and surfaces not yet destroyed.
    std::atomic<uint64_t> objects{};
    // Releases what the resources and the runtime were created on, after both are gone.
    std::function<void()> release;
};

#endif
