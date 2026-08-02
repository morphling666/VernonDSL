#include "rhi_internal.h"
#include "rhi_test_hooks.h"

#include <array>
#include <string>
#include <utility>

namespace {

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }
thread_local std::string creationError;

auto backends() {
    return std::array{
        &vernon::rhi::openGLBackendDispatch(),
#if defined(VERNON_HAS_CUDA_RHI)
        &vernon::rhi::cudaBackendDispatch(),
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
        &vernon::rhi::vulkanBackendDispatch(),
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
        &vernon::rhi::directX12BackendDispatch(),
#endif
#if defined(VERNON_HAS_METAL_RHI)
        &vernon::rhi::metalBackendDispatch(),
#endif
    };
}

const vernon::rhi::BackendDispatch *dispatch(VernonRhiDevice device) {
    for (const auto *candidate : backends())
        if (candidate->ownsDevice(device))
            return candidate;
    return nullptr;
}

const vernon::rhi::BackendDispatch *dispatch(VernonRhiBackend backend) {
    for (const auto *candidate : backends()) {
        if (candidate->backend == backend)
            return candidate;
        if (candidate->backend == VERNON_RHI_BACKEND_OPENGL && backend == VERNON_RHI_BACKEND_OPENGL_ES)
            return candidate;
    }
    return nullptr;
}

} // namespace

VernonRhiDevice vernon::rhi::createDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    creationError.clear();
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->flags & ~static_cast<uint32_t>(VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE)) ||
        (descriptor->backend != VERNON_RHI_BACKEND_DIRECTX12 && descriptor->flags))
        return invalidDevice();
    const BackendDispatch *backend = dispatch(descriptor->backend);
    if (!backend) {
        setDeviceCreationError("requested RHI backend is unavailable in this build");
        return invalidDevice();
    }
    return backend->createOwnedDevice(descriptor);
}

void vernon::rhi::setDeviceCreationError(std::string error) { creationError = std::move(error); }

VernonStringView vernon::rhi::deviceCreationError() { return {creationError.data(), creationError.size()}; }

void vernon::rhi::destroyDevice(VernonRhiDevice device) {
    if (deviceHasActiveCommandEncoder(device))
        return;
    if (const BackendDispatch *backend = dispatch(device))
        backend->destroyDevice(device);
}

VernonStringView vernon::rhi::deviceLastError(VernonRhiDevice device) {
    const BackendDispatch *backend = dispatch(device);
    return backend ? backend->lastError(device) : VernonStringView{};
}

VernonRhiStatus vernon::rhi::synchronizeDevice(VernonRhiDevice device) {
    const BackendDispatch *backend = dispatch(device);
    return backend ? backend->synchronize(device) : VERNON_RHI_STATUS_INVALID_ARGUMENT;
}

bool vernon::rhi::deviceExists(VernonRhiDevice device) { return dispatch(device) != nullptr; }

uint64_t vernon::rhi::getTrackedBufferState(VernonRhiDevice device, VernonRhiBuffer buffer) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->trackedBufferState ? backend->trackedBufferState(device, buffer) : UINT64_MAX;
}

void *vernon::rhi::deviceState(VernonRhiDevice device, VernonRhiBackend expected) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend || (backend->backend != expected &&
                     !(backend->backend == VERNON_RHI_BACKEND_OPENGL && expected == VERNON_RHI_BACKEND_OPENGL_ES)))
        return nullptr;
    return backend->deviceState(device);
}

#define VERNON_DISPATCH_STATUS(device, operation, ...)                                                                 \
    do {                                                                                                               \
        const auto *backend = dispatch(device);                                                                        \
        if (!backend)                                                                                                  \
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;                                                                 \
        return backend->operation ? backend->operation(device, __VA_ARGS__) : VERNON_RHI_STATUS_UNSUPPORTED;           \
    } while (false)

extern "C" VernonRhiStatus vernonRhiDeviceCreateBuffer(VernonRhiDevice device,
                                                       const VernonRhiBufferDescriptor *descriptor,
                                                       VernonRhiBuffer *output) {
    VERNON_DISPATCH_STATUS(device, createBuffer, descriptor, output);
}

extern "C" VernonRhiStatus vernonRhiDeviceUploadBuffer(VernonRhiDevice device, VernonRhiBuffer buffer, uint64_t offset,
                                                       const void *source, uint64_t size) {
    VERNON_DISPATCH_STATUS(device, uploadBuffer, buffer, offset, source, size);
}

extern "C" VernonRhiStatus vernonRhiDeviceDownloadBuffer(VernonRhiDevice device, VernonRhiBuffer buffer,
                                                         uint64_t offset, void *destination, uint64_t size) {
    VERNON_DISPATCH_STATUS(device, downloadBuffer, buffer, offset, destination, size);
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyBuffer(VernonRhiDevice device, VernonRhiBuffer buffer) {
    VERNON_DISPATCH_STATUS(device, destroyBuffer, buffer);
}

extern "C" uint32_t vernonRhiDeviceIsBufferValid(VernonRhiDevice device, VernonRhiBuffer buffer) {
    const auto *backend = dispatch(device);
    return backend && backend->isBufferValid ? backend->isBufferValid(device, buffer) : 0;
}

extern "C" VernonRhiStatus vernonRhiDeviceGetBufferNativeHandle(VernonRhiDevice device, VernonRhiBuffer buffer,
                                                                void **output) {
    VERNON_DISPATCH_STATUS(device, getBufferNativeHandle, buffer, output);
}

extern "C" VernonRhiStatus
vernonRhiDeviceCreateImage(VernonRhiDevice device, const VernonRhiImageDescriptor *descriptor, VernonRhiImage *output) {
    VERNON_DISPATCH_STATUS(device, createImage, descriptor, output);
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateImageView(VernonRhiDevice device,
                                                          const VernonRhiImageViewDescriptor *descriptor,
                                                          VernonRhiImageView *output) {
    VERNON_DISPATCH_STATUS(device, createImageView, descriptor, output);
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyImageView(VernonRhiDevice device, VernonRhiImageView imageView) {
    VERNON_DISPATCH_STATUS(device, destroyImageView, imageView);
}

extern "C" VernonRhiStatus vernonRhiDeviceGetImageViewNativeHandle(VernonRhiDevice device, VernonRhiImageView imageView,
                                                                   uint64_t *output) {
    VERNON_DISPATCH_STATUS(device, getImageViewNativeHandle, imageView, output);
}

extern "C" VernonRhiStatus vernonRhiDeviceSetImageSampler(VernonRhiDevice device, VernonRhiImage image,
                                                          const VernonRhiSamplerDescriptor *descriptor) {
    VERNON_DISPATCH_STATUS(device, setImageSampler, image, descriptor);
}

extern "C" VernonRhiStatus vernonRhiDeviceUploadImage(VernonRhiDevice device, VernonRhiImage image,
                                                      const VernonRhiImageUploadDescriptor *uploads,
                                                      size_t uploadCount) {
    VERNON_DISPATCH_STATUS(device, uploadImage, image, uploads, uploadCount);
}

extern "C" VernonRhiStatus vernonRhiDeviceDownloadImage(VernonRhiDevice device, VernonRhiImage image, void *destination,
                                                        size_t size) {
    VERNON_DISPATCH_STATUS(device, downloadImage, image, destination, size);
}

extern "C" VernonRhiStatus vernonRhiDeviceGenerateImageMipmaps(VernonRhiDevice device, VernonRhiImage image) {
    VERNON_DISPATCH_STATUS(device, generateImageMipmaps, image);
}

extern "C" VernonRhiStatus vernonRhiDeviceBindImage(VernonRhiDevice device, VernonRhiImage image,
                                                    uint32_t textureUnit) {
    VERNON_DISPATCH_STATUS(device, bindImage, image, textureUnit);
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyImage(VernonRhiDevice device, VernonRhiImage image) {
    VERNON_DISPATCH_STATUS(device, destroyImage, image);
}

extern "C" uint32_t vernonRhiDeviceIsImageValid(VernonRhiDevice device, VernonRhiImage image) {
    const auto *backend = dispatch(device);
    return backend && backend->isImageValid ? backend->isImageValid(device, image) : 0;
}

extern "C" VernonRhiStatus vernonRhiDeviceGetImageNativeHandle(VernonRhiDevice device, VernonRhiImage image,
                                                               uint64_t *output) {
    VERNON_DISPATCH_STATUS(device, getImageNativeHandle, image, output);
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateSampler(VernonRhiDevice device,
                                                        const VernonRhiSamplerDescriptor *descriptor,
                                                        VernonRhiSampler *output) {
    VERNON_DISPATCH_STATUS(device, createSampler, descriptor, output);
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroySampler(VernonRhiDevice device, VernonRhiSampler sampler) {
    VERNON_DISPATCH_STATUS(device, destroySampler, sampler);
}

extern "C" uint32_t vernonRhiDeviceIsSamplerValid(VernonRhiDevice device, VernonRhiSampler sampler) {
    const auto *backend = dispatch(device);
    return backend && backend->isSamplerValid ? backend->isSamplerValid(device, sampler) : 0;
}

#undef VERNON_DISPATCH_STATUS

uint64_t vernon::rhi::bufferResource(VernonRhiDevice device, VernonRhiBuffer buffer) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->bufferResource ? backend->bufferResource(device, buffer) : 0;
}

uint64_t vernon::rhi::imageResource(VernonRhiDevice device, VernonRhiImage image) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->imageResource ? backend->imageResource(device, image) : 0;
}

uint64_t vernon::rhi::samplerResource(VernonRhiDevice device, VernonRhiSampler sampler) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->samplerResource ? backend->samplerResource(device, sampler) : 0;
}

bool vernon::rhi::retainResource(VernonRhiDevice device, ResourceKind kind, uint64_t key) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->retainResource && backend->retainResource(device, kind, key);
}

uint64_t vernon::rhi::resolveResource(VernonRhiDevice device, ResourceKind kind, uint64_t key) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->resolveResource ? backend->resolveResource(device, kind, key) : 0;
}

void vernon::rhi::releaseResource(VernonRhiDevice device, ResourceKind kind, uint64_t key) {
    if (const BackendDispatch *backend = dispatch(device))
        if (backend->releaseResource)
            backend->releaseResource(device, kind, key);
}

bool vernon::rhi::beginCommandRecording(VernonRhiDevice device, uint64_t &native, VernonRhiBackend &backendKind) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->beginCommands && backend->beginCommands(device, native, backendKind);
}

bool vernon::rhi::submitCommandRecording(VernonRhiDevice device, uint64_t native, bool computeWrites, bool &completed) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->submitCommands && backend->submitCommands(device, native, computeWrites, completed);
}

void vernon::rhi::completeBorrowedCommandRecording(VernonRhiDevice device, uint64_t native) {
    if (const BackendDispatch *backend = dispatch(device))
        if (backend->completeBorrowedCommands)
            backend->completeBorrowedCommands(device, native);
}

void vernon::rhi::abandonCommandRecording(VernonRhiDevice device, uint64_t native) {
    if (const BackendDispatch *backend = dispatch(device))
        if (backend->abandonCommands)
            backend->abandonCommands(device, native);
}

VernonRhiStatus vernon::rhi::recordBarriers(VernonRhiDevice device, uint64_t encoderKey, uint64_t native,
                                            const VernonRhiBarrier *barriers, size_t barrierCount) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!backend->recordBarriers)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    return backend->recordBarriers(device, encoderKey, native, barriers, barrierCount)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

bool vernon::rhi::endCommandRendering(VernonRhiDevice device, uint64_t native, VernonRhiBackend backendKind,
                                      uint32_t renderingKind, uint32_t colorDiscardMask, uint32_t depthStencilDiscard,
                                      const uint64_t *colorResources, size_t colorCount, uint64_t depthResource,
                                      uint64_t renderingObject) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->endRendering &&
           backend->endRendering(device, native, backendKind, renderingKind, colorDiscardMask, depthStencilDiscard,
                                 colorResources, colorCount, depthResource, renderingObject);
}

VernonRhiStatus vernon::rhi::clearCommandColor(VernonRhiDevice device, uint64_t native, VernonRhiBackend backendKind,
                                               uint32_t renderingKind, int32_t x, int32_t y, uint32_t width,
                                               uint32_t height, uint32_t layers, uint64_t target, uint32_t location,
                                               const float color[4]) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!backend->clearColor)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    return backend->clearColor(device, native, backendKind, renderingKind, x, y, width, height, layers, target,
                               location, color)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus vernon::rhi::clearCommandDepthStencil(VernonRhiDevice device, uint64_t native,
                                                      VernonRhiBackend backendKind, uint32_t renderingKind, int32_t x,
                                                      int32_t y, uint32_t width, uint32_t height, uint32_t layers,
                                                      uint64_t target, float depth, uint32_t stencil,
                                                      uint32_t aspects) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!backend->clearDepthStencil)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    return backend->clearDepthStencil(device, native, backendKind, renderingKind, x, y, width, height, layers, target,
                                      depth, stencil, aspects)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}
