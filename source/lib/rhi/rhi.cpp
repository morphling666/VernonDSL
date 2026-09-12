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
    if (const BackendDispatch *backend = dispatch(device))
        (void)backend->destroyDevice(device);
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

vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError>
vernon::rhi::commandState(VernonRhiDevice device) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend || !backend->commandState)
        return Result<CommandDeviceStateRef, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"command_state", device.generation, device.index}})};
    return backend->commandState(device);
}

vernon::Result<vernon::ChildLease, vernon::RhiError> vernon::rhi::retainDeviceLease(VernonRhiDevice device) noexcept {
    auto state = commandState(device);
    if (state.isErr())
        return Result<ChildLease, RhiError>{err(std::move(state).error())};
    return state.value().retainDeviceLease();
}

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

extern "C" VernonRhiStatus vernonRhiDeviceUploadBufferRanges(VernonRhiDevice device, VernonRhiBuffer buffer,
                                                             const VernonRhiBufferUploadRange *ranges,
                                                             size_t rangeCount) {
    VERNON_DISPATCH_STATUS(device, uploadBufferRanges, buffer, ranges, rangeCount);
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

extern "C" VernonRhiStatus vernonRhiDeviceDownloadImage(VernonRhiDevice device, VernonRhiImage image,
                                                        const VernonRhiImageDownloadDescriptor *descriptor,
                                                        void *destination, size_t size) {
    VERNON_DISPATCH_STATUS(device, downloadImage, image, descriptor, destination, size);
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

uint64_t vernon::rhi::imageViewResource(VernonRhiDevice device, VernonRhiImageView view) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->imageViewResource ? backend->imageViewResource(device, view) : 0;
}

uint64_t vernon::rhi::samplerResource(VernonRhiDevice device, VernonRhiSampler sampler) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->samplerResource ? backend->samplerResource(device, sampler) : 0;
}

vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>
vernon::rhi::retainResource(VernonRhiDevice device, ResourceKind kind, uint64_t key) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<RetainedRhiResourceLease, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"retain_resource", key, static_cast<uint32_t>(kind)}})};
    if (!backend->retainResource)
        return Result<RetainedRhiResourceLease, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"retain_resource", key, static_cast<uint32_t>(kind)}})};
    auto retained = backend->retainResource(device, kind, key);
    if (retained.isErr())
        return retained;
    if (!retained.value().bindOwner(device.index, device.generation, static_cast<uint32_t>(kind), key)) {
        auto released = retained.value().release();
        if (released.isErr())
            resultContractViolation();
        return Result<RetainedRhiResourceLease, RhiError>{
            err(RhiError{RhiErrorCode::LifecycleFailure, {"retain_resource", key, static_cast<uint32_t>(kind)}})};
    }
    return retained;
}

vernon::Result<uint64_t, vernon::RhiError> vernon::rhi::resolveResource(VernonRhiDevice device, ResourceKind kind,
                                                                        uint64_t key,
                                                                        RetainedRhiResourceLease &lease) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<uint64_t, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_resource", key, static_cast<uint32_t>(kind)}})};
    if (!backend->resolveRetainedResource)
        return Result<uint64_t, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"resolve_resource", key, static_cast<uint32_t>(kind)}})};
    if (!lease.authorizes(device.index, device.generation, static_cast<uint32_t>(kind), key))
        return Result<uint64_t, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_resource", key, static_cast<uint32_t>(kind)}})};
    auto pinned = lease.pin();
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    return backend->resolveRetainedResource(device, kind, key);
}

vernon::Result<uint64_t, vernon::RhiError> vernon::rhi::resolvePinnedResource(VernonRhiDevice device, ResourceKind kind,
                                                                              uint64_t key) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<uint64_t, RhiError>{err(
            RhiError{RhiErrorCode::InvalidArgument, {"resolve_pinned_resource", key, static_cast<uint32_t>(kind)}})};
    if (!backend->resolveRetainedResource)
        return Result<uint64_t, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"resolve_pinned_resource", key, static_cast<uint32_t>(kind)}})};
    return backend->resolveRetainedResource(device, kind, key);
}

vernon::Result<void, vernon::RhiError>
vernon::rhi::describeImageResource(VernonRhiDevice device, uint64_t key,
                                   VernonRhiImageDescriptor &descriptor) noexcept {
    auto retained = retainResource(device, ResourceKind::Image, key);
    if (retained.isErr())
        return Result<void, RhiError>{err(std::move(retained).error())};
    auto described = describeImageResource(device, key, descriptor, retained.value());
    auto released = retained.value().release();
    if (described.isErr())
        return described;
    return released;
}

vernon::Result<void, vernon::RhiError> vernon::rhi::describeImageResource(VernonRhiDevice device, uint64_t key,
                                                                          VernonRhiImageDescriptor &descriptor,
                                                                          RetainedRhiResourceLease &lease) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"describe_image_resource", key, 0}})};
    if (!backend->describeImageResource)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"describe_image_resource", key, 0}})};
    if (!lease.authorizes(device.index, device.generation, static_cast<uint32_t>(ResourceKind::Image), key))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"describe_image_resource", key, 0}})};
    auto pinned = lease.pin();
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    return backend->describeImageResource(device, key, &descriptor);
}

vernon::Result<void, vernon::RhiError> vernon::rhi::describeImageViewResource(VernonRhiDevice device, uint64_t key,
                                                                              VernonRhiImageViewDescriptor &view,
                                                                              VernonRhiImageDescriptor &image,
                                                                              uint64_t &parentKey) noexcept {
    auto retained = retainResource(device, ResourceKind::ImageView, key);
    if (retained.isErr())
        return Result<void, RhiError>{err(std::move(retained).error())};
    auto described = describeImageViewResource(device, key, view, image, parentKey, retained.value());
    auto released = retained.value().release();
    if (described.isErr())
        return described;
    return released;
}

vernon::Result<void, vernon::RhiError>
vernon::rhi::describeImageViewResource(VernonRhiDevice device, uint64_t key, VernonRhiImageViewDescriptor &view,
                                       VernonRhiImageDescriptor &image, uint64_t &parentKey,
                                       RetainedRhiResourceLease &lease) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"describe_image_view_resource", key, 0}})};
    if (!backend->describeImageViewResource)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"describe_image_view_resource", key, 0}})};
    if (!lease.authorizes(device.index, device.generation, static_cast<uint32_t>(ResourceKind::ImageView), key))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"describe_image_view_resource", key, 0}})};
    auto pinned = lease.pin();
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    return backend->describeImageViewResource(device, key, &view, &image, &parentKey);
}

bool vernon::rhi::beginCommandRecording(VernonRhiDevice device, uint64_t &native, VernonRhiBackend &backendKind) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->beginCommands && backend->beginCommands(device, native, backendKind);
}

uint32_t vernon::rhi::deviceCommandCapabilities(VernonRhiDevice device) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return 0;
    return backend->commandCapabilitiesForDevice ? backend->commandCapabilitiesForDevice(device)
                                                 : backend->commandCapabilities;
}

bool vernon::rhi::submitCommandRecording(VernonRhiDevice device, uint64_t native, bool computeWrites, bool &completed,
                                         bool &externalCompletion) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->submitCommands &&
           backend->submitCommands(device, native, computeWrites, completed, externalCompletion);
}

bool vernon::rhi::pollCommandRecording(VernonRhiDevice device, uint64_t native, bool &completed, bool &succeeded) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->pollCommands && backend->pollCommands(device, native, completed, succeeded);
}

bool vernon::rhi::completeCommandRecording(VernonRhiDevice device, uint64_t native) {
    const BackendDispatch *backend = dispatch(device);
    return backend && backend->completeBorrowedCommands && backend->completeBorrowedCommands(device, native);
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

VernonRhiStatus vernon::rhi::recordBufferCopy(VernonRhiDevice device, uint64_t native, VernonRhiBuffer source,
                                              uint64_t sourceOffset, VernonRhiBuffer destination,
                                              uint64_t destinationOffset, uint64_t size) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!backend->recordBufferCopy)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    return backend->recordBufferCopy(device, native, source, sourceOffset, destination, destinationOffset, size)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus vernon::rhi::recordImageCopy(VernonRhiDevice device, uint64_t encoderKey, uint64_t native,
                                             VernonRhiImage source, VernonRhiImage destination,
                                             const VernonRhiImageCopyRegion *regions, size_t regionCount) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!backend->recordImageCopy || !backend->supportsImageCopy || !backend->supportsImageCopy(device))
        return VERNON_RHI_STATUS_UNSUPPORTED;
    return backend->recordImageCopy(device, encoderKey, native, source, destination, regions, regionCount)
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
                                               uint32_t renderingKind, uint64_t renderingObject, int32_t x, int32_t y,
                                               uint32_t width, uint32_t height, uint32_t layers, uint64_t target,
                                               uint32_t location, const float color[4]) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!backend->clearColor)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    return backend->clearColor(device, native, backendKind, renderingKind, renderingObject, x, y, width, height, layers,
                               target, location, color)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus vernon::rhi::clearCommandDepthStencil(VernonRhiDevice device, uint64_t native,
                                                      VernonRhiBackend backendKind, uint32_t renderingKind,
                                                      uint64_t renderingObject, int32_t x, int32_t y, uint32_t width,
                                                      uint32_t height, uint32_t layers, uint64_t target, float depth,
                                                      uint32_t stencil, uint32_t aspects) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!backend->clearDepthStencil)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    return backend->clearDepthStencil(device, native, backendKind, renderingKind, renderingObject, x, y, width, height,
                                      layers, target, depth, stencil, aspects)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}
