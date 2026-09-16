#include "public_c_boundary.h"
#include "rhi_internal.h"
#include "rhi_test_hooks.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>
#include <utility>

namespace {

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }
thread_local std::array<char, 256> creationError{};
thread_local size_t creationErrorSize{};

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

vernon::Result<VernonRhiDevice, vernon::RhiError>
vernon::rhi::createDeviceImpl(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    creationErrorSize = 0;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->flags & ~static_cast<uint32_t>(VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE)) ||
        (descriptor->backend != VERNON_RHI_BACKEND_DIRECTX12 && descriptor->flags))
        return Result<VernonRhiDevice, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"create_device", 0, 0}})};
    const BackendDispatch *backend = dispatch(descriptor->backend);
    if (!backend) {
        setDeviceCreationError("requested RHI backend is unavailable in this build");
        return Result<VernonRhiDevice, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"create_device", static_cast<uint64_t>(descriptor->backend), 0}})};
    }
    return backend->createOwnedDevice(descriptor);
}

void vernon::rhi::setDeviceCreationError(std::string_view error) noexcept {
    creationErrorSize = std::min(error.size(), creationError.size() - 1);
    if (creationErrorSize)
        std::memcpy(creationError.data(), error.data(), creationErrorSize);
    creationError[creationErrorSize] = '\0';
}

VernonStringView vernon::rhi::deviceCreationError() noexcept { return {creationError.data(), creationErrorSize}; }

vernon::Result<void, vernon::RhiError> vernon::rhi::destroyDeviceImpl(VernonRhiDevice device) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"destroy_device", device.generation, device.index}})};
    return backend->destroyDevice(device);
}

vernon::Result<vernon::Option<VernonStringView>, vernon::RhiError>
vernon::rhi::deviceLastError(VernonRhiDevice device) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<Option<VernonStringView>, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"device_last_error", device.generation, device.index}})};
    return Result<Option<VernonStringView>, RhiError>{ok(backend->lastError(device))};
}

vernon::Result<void, vernon::RhiError> vernon::rhi::synchronizeDevice(VernonRhiDevice device) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"synchronize_device", device.generation, device.index}})};
    return backend->synchronize(device);
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
    if (!backend || !backend->trackedBufferState)
        return UINT64_MAX;
    auto state = backend->trackedBufferState.value()(device, buffer);
    return state.isOk() ? state.value() : UINT64_MAX;
}

vernon::Result<void *, vernon::RhiError> vernon::rhi::deviceState(VernonRhiDevice device, VernonRhiBackend expected) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend || (backend->backend != expected &&
                     !(backend->backend == VERNON_RHI_BACKEND_OPENGL && expected == VERNON_RHI_BACKEND_OPENGL_ES)))
        return Result<void *, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"device_state", device.generation, device.index}})};
    return backend->deviceState(device);
}

template <typename Function>
vernon::Result<void, vernon::RhiError> dispatchStatusImpl(VernonRhiDevice device, const char *operation,
                                                          Function &&function) {
    const auto *backend = dispatch(device);
    if (!backend)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {operation, 0, 0}})};
    return std::forward<Function>(function)(*backend);
}

#define VERNON_REQUIRED_RESULT_IMPL(name, operation, parameters, arguments)                                            \
    static vernon::Result<void, vernon::RhiError> name##Impl parameters {                                              \
        return dispatchStatusImpl(device, #operation, [&](const vernon::rhi::BackendDispatch &backend) {               \
            return backend.operation arguments;                                                                        \
        });                                                                                                            \
    }                                                                                                                  \
    extern "C" VernonRhiStatus name parameters {                                                                       \
        return vernon::rhi::publicStatusBoundary([&] { return name##Impl arguments; });                                \
    }

#define VERNON_OPTIONAL_RESULT_IMPL(name, operation, parameters, arguments)                                            \
    static vernon::Result<void, vernon::RhiError> name##Impl parameters {                                              \
        return dispatchStatusImpl(device, #operation, [&](const vernon::rhi::BackendDispatch &backend) {               \
            if (!backend.operation)                                                                                    \
                return vernon::Result<void, vernon::RhiError>{                                                         \
                    vernon::err(vernon::RhiError{vernon::RhiErrorCode::Unsupported, {#operation, 0, 0}})};             \
            return backend.operation.value() arguments;                                                                \
        });                                                                                                            \
    }                                                                                                                  \
    extern "C" VernonRhiStatus name parameters {                                                                       \
        return vernon::rhi::publicStatusBoundary([&] { return name##Impl arguments; });                                \
    }

static vernon::Result<void, vernon::RhiError>
vernonRhiDeviceCreateBufferImpl(VernonRhiDevice device, const VernonRhiBufferDescriptor *descriptor,
                                VernonRhiBuffer *output) {
    if (!output)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"createBuffer", 0, 0}})};
    return dispatchStatusImpl(device, "createBuffer", [&](const vernon::rhi::BackendDispatch &backend) {
        auto created = backend.createBuffer(device, descriptor);
        if (created.isErr())
            return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(created).error())};
        *output = created.value();
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    });
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateBuffer(VernonRhiDevice device,
                                                       const VernonRhiBufferDescriptor *descriptor,
                                                       VernonRhiBuffer *output) {
    return vernon::rhi::publicStatusBoundary(
        [&] { return vernonRhiDeviceCreateBufferImpl(device, descriptor, output); });
}

VERNON_REQUIRED_RESULT_IMPL(vernonRhiDeviceUploadBuffer, uploadBuffer,
                            (VernonRhiDevice device, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size),
                            (device, buffer, offset, source, size))
VERNON_REQUIRED_RESULT_IMPL(vernonRhiDeviceUploadBufferRanges, uploadBufferRanges,
                            (VernonRhiDevice device, VernonRhiBuffer buffer, const VernonRhiBufferUploadRange *ranges,
                             size_t rangeCount),
                            (device, buffer, ranges, rangeCount))
VERNON_REQUIRED_RESULT_IMPL(vernonRhiDeviceDownloadBufferRanges, downloadBufferRanges,
                            (VernonRhiDevice device, VernonRhiBuffer buffer, const VernonRhiBufferDownloadRange *ranges,
                             size_t rangeCount),
                            (device, buffer, ranges, rangeCount))
VERNON_REQUIRED_RESULT_IMPL(vernonRhiDeviceDownloadBuffer, downloadBuffer,
                            (VernonRhiDevice device, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                             uint64_t size),
                            (device, buffer, offset, destination, size))
VERNON_REQUIRED_RESULT_IMPL(vernonRhiDeviceDestroyBuffer, destroyBuffer,
                            (VernonRhiDevice device, VernonRhiBuffer buffer), (device, buffer))

extern "C" uint32_t vernonRhiDeviceIsBufferValid(VernonRhiDevice device, VernonRhiBuffer buffer) {
    return vernon::rhi::publicQueryBoundary<uint32_t>([&] {
        const auto *backend = dispatch(device);
        if (!backend)
            return 0u;
        auto valid = backend->isBufferValid(device, buffer);
        return valid.isOk() && valid.value() ? 1u : 0u;
    });
}

template <typename Handle, typename Descriptor, typename Member>
vernon::Result<void, vernon::RhiError> createOptionalResource(VernonRhiDevice device, const Descriptor *descriptor,
                                                              Handle *output, const char *operation, Member member) {
    if (!output)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {operation, 0, 0}})};
    return dispatchStatusImpl(device, operation, [&](const vernon::rhi::BackendDispatch &backend) {
        const auto &callback = backend.*member;
        if (!callback)
            return vernon::Result<void, vernon::RhiError>{
                vernon::err(vernon::RhiError{vernon::RhiErrorCode::Unsupported, {operation, 0, 0}})};
        auto created = callback.value()(device, descriptor);
        if (created.isErr())
            return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(created).error())};
        *output = created.value();
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    });
}

extern "C" VernonRhiStatus
vernonRhiDeviceCreateImage(VernonRhiDevice device, const VernonRhiImageDescriptor *descriptor, VernonRhiImage *output) {
    return vernon::rhi::publicStatusBoundary([&] {
        return createOptionalResource(device, descriptor, output, "createImage",
                                      &vernon::rhi::BackendDispatch::createImage);
    });
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateImageView(VernonRhiDevice device,
                                                          const VernonRhiImageViewDescriptor *descriptor,
                                                          VernonRhiImageView *output) {
    return vernon::rhi::publicStatusBoundary([&] {
        return createOptionalResource(device, descriptor, output, "createImageView",
                                      &vernon::rhi::BackendDispatch::createImageView);
    });
}

VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceDestroyImageView, destroyImageView,
                            (VernonRhiDevice device, VernonRhiImageView imageView), (device, imageView))

static vernon::Result<void, vernon::RhiError>
vernonRhiDeviceGetImageViewNativeHandleImpl(VernonRhiDevice device, VernonRhiImageView imageView, uint64_t *output) {
    if (!output)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"getImageViewNativeHandle", 0, 0}})};
    return dispatchStatusImpl(device, "getImageViewNativeHandle", [&](const vernon::rhi::BackendDispatch &backend) {
        if (!backend.getImageViewNativeHandle)
            return vernon::Result<void, vernon::RhiError>{
                vernon::err(vernon::RhiError{vernon::RhiErrorCode::Unsupported, {"getImageViewNativeHandle", 0, 0}})};
        auto native = backend.getImageViewNativeHandle.value()(device, imageView);
        if (native.isErr())
            return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(native).error())};
        *output = native.value();
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    });
}

extern "C" VernonRhiStatus vernonRhiDeviceGetImageViewNativeHandle(VernonRhiDevice device, VernonRhiImageView imageView,
                                                                   uint64_t *output) {
    return vernon::rhi::publicStatusBoundary(
        [&] { return vernonRhiDeviceGetImageViewNativeHandleImpl(device, imageView, output); });
}

VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceSetImageSampler, setImageSampler,
                            (VernonRhiDevice device, VernonRhiImage image,
                             const VernonRhiSamplerDescriptor *descriptor),
                            (device, image, descriptor))
VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceUploadImage, uploadImage,
                            (VernonRhiDevice device, VernonRhiImage image,
                             const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount),
                            (device, image, uploads, uploadCount))
VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceDownloadImage, downloadImage,
                            (VernonRhiDevice device, VernonRhiImage image,
                             const VernonRhiImageDownloadDescriptor *descriptor, void *destination, size_t size),
                            (device, image, descriptor, destination, size))
VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceDownloadImageBatch, downloadImageBatch,
                            (VernonRhiDevice device, VernonRhiImage image, const VernonRhiImageDownload *downloads,
                             size_t downloadCount),
                            (device, image, downloads, downloadCount))
VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceGenerateImageMipmaps, generateImageMipmaps,
                            (VernonRhiDevice device, VernonRhiImage image), (device, image))
VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceBindImage, bindImage,
                            (VernonRhiDevice device, VernonRhiImage image, uint32_t textureUnit),
                            (device, image, textureUnit))
VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceDestroyImage, destroyImage, (VernonRhiDevice device, VernonRhiImage image),
                            (device, image))

extern "C" uint32_t vernonRhiDeviceIsImageValid(VernonRhiDevice device, VernonRhiImage image) {
    return vernon::rhi::publicQueryBoundary<uint32_t>([&] {
        const auto *backend = dispatch(device);
        if (!backend || !backend->isImageValid)
            return 0u;
        auto valid = backend->isImageValid.value()(device, image);
        return valid.isOk() && valid.value() ? 1u : 0u;
    });
}

static vernon::Result<void, vernon::RhiError>
vernonRhiDeviceGetImageNativeHandleImpl(VernonRhiDevice device, VernonRhiImage image, uint64_t *output) {
    if (!output)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"getImageNativeHandle", 0, 0}})};
    return dispatchStatusImpl(device, "getImageNativeHandle", [&](const vernon::rhi::BackendDispatch &backend) {
        if (!backend.getImageNativeHandle)
            return vernon::Result<void, vernon::RhiError>{
                vernon::err(vernon::RhiError{vernon::RhiErrorCode::Unsupported, {"getImageNativeHandle", 0, 0}})};
        auto native = backend.getImageNativeHandle.value()(device, image);
        if (native.isErr())
            return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(native).error())};
        *output = native.value();
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    });
}

extern "C" VernonRhiStatus vernonRhiDeviceGetImageNativeHandle(VernonRhiDevice device, VernonRhiImage image,
                                                               uint64_t *output) {
    return vernon::rhi::publicStatusBoundary(
        [&] { return vernonRhiDeviceGetImageNativeHandleImpl(device, image, output); });
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateSampler(VernonRhiDevice device,
                                                        const VernonRhiSamplerDescriptor *descriptor,
                                                        VernonRhiSampler *output) {
    return vernon::rhi::publicStatusBoundary([&] {
        return createOptionalResource(device, descriptor, output, "createSampler",
                                      &vernon::rhi::BackendDispatch::createSampler);
    });
}

VERNON_OPTIONAL_RESULT_IMPL(vernonRhiDeviceDestroySampler, destroySampler,
                            (VernonRhiDevice device, VernonRhiSampler sampler), (device, sampler))

extern "C" uint32_t vernonRhiDeviceIsSamplerValid(VernonRhiDevice device, VernonRhiSampler sampler) {
    return vernon::rhi::publicQueryBoundary<uint32_t>([&] {
        const auto *backend = dispatch(device);
        if (!backend || !backend->isSamplerValid)
            return 0u;
        auto valid = backend->isSamplerValid.value()(device, sampler);
        return valid.isOk() && valid.value() ? 1u : 0u;
    });
}

static vernon::Result<void, vernon::RhiError>
vernonRhiDeviceGetBufferNativeHandleImpl(VernonRhiDevice device, VernonRhiBuffer buffer, void **output) {
    if (!output)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"getBufferNativeHandle", 0, 0}})};
    return dispatchStatusImpl(device, "getBufferNativeHandle", [&](const vernon::rhi::BackendDispatch &backend) {
        auto native = backend.getBufferNativeHandle(device, buffer);
        if (native.isErr())
            return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(native).error())};
        *output = native.value();
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    });
}

extern "C" VernonRhiStatus vernonRhiDeviceGetBufferNativeHandle(VernonRhiDevice device, VernonRhiBuffer buffer,
                                                                void **output) {
    return vernon::rhi::publicStatusBoundary(
        [&] { return vernonRhiDeviceGetBufferNativeHandleImpl(device, buffer, output); });
}

#undef VERNON_OPTIONAL_RESULT_IMPL
#undef VERNON_REQUIRED_RESULT_IMPL

vernon::Result<uint64_t, vernon::RhiError> vernon::rhi::bufferResource(VernonRhiDevice device,
                                                                       VernonRhiBuffer buffer) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend || !backend->bufferResource)
        return Result<uint64_t, RhiError>{
            err(RhiError{backend ? RhiErrorCode::Unsupported : RhiErrorCode::InvalidArgument,
                         {"buffer_resource", buffer.generation, buffer.index}})};
    return backend->bufferResource(device, buffer);
}

vernon::Result<uint64_t, vernon::RhiError> vernon::rhi::imageResource(VernonRhiDevice device,
                                                                      VernonRhiImage image) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend || !backend->imageResource)
        return Result<uint64_t, RhiError>{
            err(RhiError{backend ? RhiErrorCode::Unsupported : RhiErrorCode::InvalidArgument,
                         {"image_resource", image.generation, image.index}})};
    return backend->imageResource.value()(device, image);
}

vernon::Result<uint64_t, vernon::RhiError> vernon::rhi::imageViewResource(VernonRhiDevice device,
                                                                          VernonRhiImageView view) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend || !backend->imageViewResource)
        return Result<uint64_t, RhiError>{
            err(RhiError{backend ? RhiErrorCode::Unsupported : RhiErrorCode::InvalidArgument,
                         {"image_view_resource", view.generation, view.index}})};
    return backend->imageViewResource.value()(device, view);
}

vernon::Result<uint64_t, vernon::RhiError> vernon::rhi::samplerResource(VernonRhiDevice device,
                                                                        VernonRhiSampler sampler) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend || !backend->samplerResource)
        return Result<uint64_t, RhiError>{
            err(RhiError{backend ? RhiErrorCode::Unsupported : RhiErrorCode::InvalidArgument,
                         {"sampler_resource", sampler.generation, sampler.index}})};
    return backend->samplerResource.value()(device, sampler);
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
    auto described = backend->describeImageResource.value()(device, key);
    if (described.isErr())
        return Result<void, RhiError>{err(std::move(described).error())};
    descriptor = described.value().image;
    return Result<void, RhiError>{ok()};
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
    auto described = backend->describeImageViewResource.value()(device, key);
    if (described.isErr())
        return Result<void, RhiError>{err(std::move(described).error())};
    view = described.value().view;
    image = described.value().image;
    parentKey = described.value().parentKey;
    return Result<void, RhiError>{ok()};
}

vernon::Result<vernon::rhi::CommandRecording, vernon::RhiError>
vernon::rhi::beginCommandRecording(VernonRhiDevice device) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<CommandRecording, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"begin_command_recording", device.generation, device.index}})};
    if (!backend->beginCommands)
        return Result<CommandRecording, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"begin_command_recording", device.generation, device.index}})};
    return backend->beginCommands(device);
}

uint32_t vernon::rhi::deviceCommandCapabilities(VernonRhiDevice device) {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return 0;
    return backend->commandCapabilitiesForDevice ? backend->commandCapabilitiesForDevice.value()(device)
                                                 : backend->commandCapabilities;
}

vernon::Result<vernon::rhi::CommandSubmission, vernon::RhiError>
vernon::rhi::submitCommandRecording(VernonRhiDevice device, uint64_t native, bool computeWrites) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<CommandSubmission, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"submit_command_recording", native, 0}})};
    if (!backend->submitCommands)
        return Result<CommandSubmission, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"submit_command_recording", native, 0}})};
    return backend->submitCommands(device, native, computeWrites);
}

vernon::Result<vernon::rhi::CommandPoll, vernon::RhiError> vernon::rhi::pollCommandRecording(VernonRhiDevice device,
                                                                                             uint64_t native) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<CommandPoll, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"poll_command_recording", native, 0}})};
    if (!backend->pollCommands)
        return Result<CommandPoll, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"poll_command_recording", native, 0}})};
    return backend->pollCommands.value()(device, native);
}

vernon::Result<void, vernon::RhiError> vernon::rhi::completeCommandRecording(VernonRhiDevice device,
                                                                             uint64_t native) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"complete_command_recording", native, 0}})};
    if (!backend->completeBorrowedCommands)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"complete_command_recording", native, 0}})};
    return backend->completeBorrowedCommands.value()(device, native);
}

void vernon::rhi::abandonCommandRecording(VernonRhiDevice device, uint64_t native) noexcept {
    if (const BackendDispatch *backend = dispatch(device))
        if (backend->abandonCommands)
            backend->abandonCommands.value()(device, native);
}

vernon::Result<void, vernon::RhiError> vernon::rhi::recordBarriers(VernonRhiDevice device, uint64_t encoderKey,
                                                                   uint64_t native, const VernonRhiBarrier *barriers,
                                                                   size_t barrierCount) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"record_barriers", native, 0}})};
    if (!backend->recordBarriers)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"record_barriers", native, 0}})};
    return backend->recordBarriers.value()(device, encoderKey, native, barriers, barrierCount);
}

vernon::Result<void, vernon::RhiError>
vernon::rhi::recordBufferCopy(VernonRhiDevice device, uint64_t native, VernonRhiBuffer source, uint64_t sourceOffset,
                              VernonRhiBuffer destination, uint64_t destinationOffset, uint64_t size) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"record_buffer_copy", native, 0}})};
    if (!backend->recordBufferCopy)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"record_buffer_copy", native, 0}})};
    return backend->recordBufferCopy.value()(device, native, source, sourceOffset, destination, destinationOffset,
                                             size);
}

vernon::Result<void, vernon::RhiError> vernon::rhi::recordImageCopy(VernonRhiDevice device, uint64_t encoderKey,
                                                                    uint64_t native, VernonRhiImage source,
                                                                    VernonRhiImage destination,
                                                                    const VernonRhiImageCopyRegion *regions,
                                                                    size_t regionCount) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"record_image_copy", native, 0}})};
    if (!backend->recordImageCopy)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"record_image_copy", native, 0}})};
    return backend->recordImageCopy.value()(device, encoderKey, native, source, destination, regions, regionCount);
}

vernon::Result<void, vernon::RhiError>
vernon::rhi::endCommandRendering(VernonRhiDevice device, uint64_t native, VernonRhiBackend backendKind,
                                 uint32_t renderingKind, uint32_t colorDiscardMask, uint32_t depthStencilDiscard,
                                 const uint64_t *colorResources, size_t colorCount, uint64_t depthResource,
                                 uint64_t renderingObject) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"end_rendering", native, 0}})};
    if (!backend->endRendering)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"end_rendering", native, 0}})};
    return backend->endRendering.value()(device, native, backendKind, renderingKind, colorDiscardMask,
                                         depthStencilDiscard, colorResources, colorCount, depthResource,
                                         renderingObject);
}

vernon::Result<void, vernon::RhiError>
vernon::rhi::clearCommandColor(VernonRhiDevice device, uint64_t native, VernonRhiBackend backendKind,
                               uint32_t renderingKind, uint64_t renderingObject, int32_t x, int32_t y, uint32_t width,
                               uint32_t height, uint32_t layers, uint64_t target, uint32_t location,
                               const float color[4]) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"clear_color", native, 0}})};
    if (!backend->clearColor)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"clear_color", native, 0}})};
    return backend->clearColor.value()(device, native, backendKind, renderingKind, renderingObject, x, y, width, height,
                                       layers, target, location, color);
}

vernon::Result<void, vernon::RhiError>
vernon::rhi::clearCommandDepthStencil(VernonRhiDevice device, uint64_t native, VernonRhiBackend backendKind,
                                      uint32_t renderingKind, uint64_t renderingObject, int32_t x, int32_t y,
                                      uint32_t width, uint32_t height, uint32_t layers, uint64_t target, float depth,
                                      uint32_t stencil, uint32_t aspects) noexcept {
    const BackendDispatch *backend = dispatch(device);
    if (!backend)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"clear_depth_stencil", native, 0}})};
    if (!backend->clearDepthStencil)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"clear_depth_stencil", native, 0}})};
    return backend->clearDepthStencil.value()(device, native, backendKind, renderingKind, renderingObject, x, y, width,
                                              height, layers, target, depth, stencil, aspects);
}
