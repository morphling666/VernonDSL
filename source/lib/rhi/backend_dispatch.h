#ifndef VERNON_RHI_BACKEND_DISPATCH_H
#define VERNON_RHI_BACKEND_DISPATCH_H

#include "rhi_command_state.h"
#include "rhi_lifecycle.h"

#include <array>
#include <string_view>
#include <utility>

namespace vernon::rhi {

enum class ResourceKind : uint32_t { Buffer = 1, Image = 2, Sampler = 3, ImageView = 4 };

inline RhiError invalidArgument(const char *operation, uint64_t value = 0, uint32_t detail = 0) noexcept {
    return {RhiErrorCode::InvalidArgument, {operation, value, detail}};
}

inline RhiError unsupported(const char *operation, uint64_t value = 0, uint32_t detail = 0) noexcept {
    return {RhiErrorCode::Unsupported, {operation, value, detail}};
}

inline RhiError backendFailure(const char *operation, uint64_t value = 0, uint32_t detail = 0) noexcept {
    return {RhiErrorCode::BackendFailure, {operation, value, detail}};
}

inline Result<void, RhiError> invalidResult(const char *operation, uint64_t value = 0, uint32_t detail = 0) noexcept {
    return Result<void, RhiError>{err(invalidArgument(operation, value, detail))};
}

inline Result<void, RhiError> unsupportedResult(const char *operation, uint64_t value = 0,
                                                uint32_t detail = 0) noexcept {
    return Result<void, RhiError>{err(unsupported(operation, value, detail))};
}

inline Result<void, RhiError> backendResult(const char *operation, uint64_t value = 0, uint32_t detail = 0) noexcept {
    return Result<void, RhiError>{err(backendFailure(operation, value, detail))};
}

enum CommandRenderingKind : uint32_t {
    CommandRenderingDynamic = 1,
    CommandRenderingRenderPass = 2,
    CommandRenderingStateless = 3
};
struct RenderingColorAttachment {
    uint64_t view{};
    VernonRhiFormat format{};
    uint32_t location{};
    VernonRhiLoadOperation load{VERNON_RHI_LOAD_PRESERVE};
    VernonRhiStoreOperation store{VERNON_RHI_STORE_PRESERVE};
    VernonClearColor clear{};
};

struct RenderingDepthAttachment {
    uint64_t view{};
    VernonRhiFormat format{};
    bool stencil{};
    VernonRhiLoadOperation depthLoad{VERNON_RHI_LOAD_PRESERVE};
    VernonRhiStoreOperation depthStore{VERNON_RHI_STORE_PRESERVE};
    VernonRhiLoadOperation stencilLoad{VERNON_RHI_LOAD_DISCARD};
    VernonRhiStoreOperation stencilStore{VERNON_RHI_STORE_DISCARD};
    float clearDepth{1.0f};
    uint32_t clearStencil{};
};

// Attachment views are image view resource keys, as `imageViewResource` returns them. `BeginRendering` validates
// them against the render area, so backends open the scope without revalidating; `format` is the view's format and
// `stencil` whether it has a stencil aspect.
struct RenderingScope {
    std::array<RenderingColorAttachment, VERNON_RHI_MAX_COLOR_ATTACHMENTS> colors{};
    size_t colorCount{};
    RenderingDepthAttachment depth{};
    bool hasDepth{};
    int32_t x{};
    int32_t y{};
    uint32_t width{};
    uint32_t height{};
    uint32_t layers{};

    const RenderingColorAttachment *colorAt(uint32_t location) const noexcept {
        for (size_t index = 0; index < colorCount; ++index)
            if (colors[index].location == location)
                return &colors[index];
        return nullptr;
    }
    RenderingColorAttachment *colorAt(uint32_t location) noexcept {
        return const_cast<RenderingColorAttachment *>(std::as_const(*this).colorAt(location));
    }
    // Attachments are indexed colors first, then depth.
    size_t attachmentCount() const noexcept { return colorCount + (hasDepth ? 1u : 0u); }
    uint64_t attachmentView(size_t index) const noexcept {
        return index < colorCount ? colors[index].view : depth.view;
    }
};

// An image or image view, as a resource key, that the draws of a rendering scope sample.
struct RenderingRead {
    ResourceKind kind{};
    uint64_t key{};
};

// A backend's open native scope; `kind` is nonzero. Backends place native color attachments at their locations and
// store in the location-indexed `colorTargets`/`colorResources` and in `depthTarget`/`depthResource` what their clear
// and end callbacks address.
struct OpenedRendering {
    uint32_t kind{};
    uint64_t object{};
    std::array<uint64_t, VERNON_RHI_MAX_COLOR_ATTACHMENTS> colorTargets{};
    std::array<uint64_t, VERNON_RHI_MAX_COLOR_ATTACHMENTS> colorResources{};
    uint64_t depthTarget{};
    uint64_t depthResource{};
};

enum BackendCommandCapabilityBits : uint32_t {
    BackendCommandIndependentRecording = 1u << 0,
    // Submission, polling and completion may be called from any thread. Without it, every call on the device must come
    // from the thread that owns the device's native context.
    BackendCommandConcurrentSubmission = 1u << 1,
    BackendCommandTimelineCompletion = 1u << 2,
    // The device records timers; without it the RHI accepts timers and records nothing.
    BackendCommandTimers = 1u << 3,
    BackendCommandExplicitComputeDependencies = 1u << 4,
    // The device's host executes its submissions, as on a borrowed device. The RHI cannot wait for them, so it rejects
    // operations whose results the host reads.
    BackendCommandExternalExecution = 1u << 5,
};

struct CommandRecording {
    uint64_t native{};
    VernonRhiBackend backend{VERNON_RHI_BACKEND_CUDA};
};

struct CommandSubmission {
    bool completed{};
    bool externalCompletion{};
    // Names the submitted work to pollCommands and completeCommands. It differs from the recording's native handle on
    // backends that reuse that handle before the work completes.
    uint64_t retirement{};
};

struct CommandPoll {
    bool completed{};
    bool succeeded{};
};

struct ImageResourceDescription {
    VernonRhiImageDescriptor image{};
};

struct ImageViewResourceDescription {
    VernonRhiImageViewDescriptor view{};
    VernonRhiImageDescriptor image{};
    uint64_t parentKey{};
};

using CommandCapabilitiesCallback = uint32_t (*)(VernonRhiDevice);
using CreateImageCallback = Result<VernonRhiImage, RhiError> (*)(VernonRhiDevice, const VernonRhiImageDescriptor *);
using ImageOperationCallback = Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiImage);
using SetImageSamplerCallback = Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiImage,
                                                           const VernonRhiSamplerDescriptor *);
using UploadImageCallback = Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiImage,
                                                       const VernonRhiImageUploadDescriptor *, size_t);
using DownloadImageCallback = Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiImage,
                                                         const VernonRhiImageDownloadDescriptor *, void *, size_t);
using DownloadImageBatchCallback = Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiImage,
                                                              const VernonRhiImageDownload *, size_t);
using BindImageCallback = Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiImage, uint32_t);
using ImageValidityCallback = Result<bool, RhiError> (*)(VernonRhiDevice, VernonRhiImage);
using ImageNativeHandleCallback = Result<uint64_t, RhiError> (*)(VernonRhiDevice, VernonRhiImage);
using ImageResourceCallback = Result<uint64_t, RhiError> (*)(VernonRhiDevice, VernonRhiImage) noexcept;
using DescribeImageResourceCallback = Result<ImageResourceDescription, RhiError> (*)(VernonRhiDevice,
                                                                                     uint64_t) noexcept;
using PollCommandsCallback = Result<CommandPoll, RhiError> (*)(VernonRhiDevice, uint64_t) noexcept;
using CompleteCommandsCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t) noexcept;
using AbandonCommandsCallback = void (*)(VernonRhiDevice, uint64_t) noexcept;
using RecordBarriersCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, uint64_t, const VernonRhiBarrier *,
                                                          size_t) noexcept;
using RecordBufferCopyCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, VernonRhiBuffer, uint64_t,
                                                            VernonRhiBuffer, uint64_t, uint64_t) noexcept;
using RecordImageCopyCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, uint64_t, VernonRhiImage,
                                                           VernonRhiImage, const VernonRhiImageCopyRegion *,
                                                           size_t) noexcept;
// Opens the native scope of `scope` on the encoder, applying its load operations, after making `reads` readable by
// shaders. Attachments and reads are retained and transitioned through the encoder, so a failure leaves rollbacks that
// restore their state.
using BeginRenderingCallback = Result<OpenedRendering, RhiError> (*)(VernonRhiDevice, uint64_t encoderKey,
                                                                     uint64_t native, const RenderingScope &,
                                                                     const RenderingRead *reads,
                                                                     size_t readCount) noexcept;
// Ends the native scope that `BeginRenderingCallback` opened for `scope`, applying its store operations. It runs under
// the encoder's lock, so it must not call encoder services.
using EndRenderingCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t native, const RenderingScope &,
                                                        const OpenedRendering &) noexcept;
// Clear callbacks clear the render area of the open scope's attachment at `location`, or of its depth attachment.
// They run under the encoder's lock.
using ClearColorCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t native, const RenderingScope &,
                                                      const OpenedRendering &, uint32_t location,
                                                      const VernonClearColor &color) noexcept;
using ClearDepthStencilCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t native, const RenderingScope &,
                                                             const OpenedRendering &, float depth, uint32_t stencil,
                                                             uint32_t aspects) noexcept;
enum class DebugCommand : uint32_t { PushGroup, PopGroup, InsertMarker };
// Records a debug command on the native recording. `rendering` is the open native scope for commands inside a
// rendering scope and null outside one; the label is empty for `PopGroup`. It runs under the encoder's lock.
using RecordDebugCommandCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t native,
                                                              const OpenedRendering *rendering, DebugCommand,
                                                              std::string_view label) noexcept;
enum class TimerEdge : uint32_t { Begin, End };
// Records the begin or end of timer `timer` on the native recording outside rendering scopes, under the encoder's lock.
// Timers are numbered in begin order. `storage` holds the recording's timer storage, 0 until the first call creates
// it; the storage outlives the recording until `ReleaseTimersCallback`.
using RecordTimerCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t native, uint64_t &storage,
                                                       uint32_t timer, TimerEdge) noexcept;
// Writes each timer's begin and end in nanoseconds on one device clock with an arbitrary origin, after the storage's
// submission succeeded. `Unsupported` reports a measurement the device invalidated.
using ReadTimersCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t storage, uint32_t count,
                                                      VernonRhiTimerSpan *spans) noexcept;
// Releases timer storage after its recording was abandoned or its submission retired.
using ReleaseTimersCallback = void (*)(VernonRhiDevice, uint64_t storage) noexcept;
using TrackedBufferStateCallback = Result<uint64_t, RhiError> (*)(VernonRhiDevice, VernonRhiBuffer);
using CreateImageViewCallback = Result<VernonRhiImageView, RhiError> (*)(VernonRhiDevice,
                                                                         const VernonRhiImageViewDescriptor *);
using ImageViewOperationCallback = Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiImageView);
using ImageViewNativeHandleCallback = Result<uint64_t, RhiError> (*)(VernonRhiDevice, VernonRhiImageView);
using ImageViewResourceCallback = Result<uint64_t, RhiError> (*)(VernonRhiDevice, VernonRhiImageView) noexcept;
using DescribeImageViewResourceCallback = Result<ImageViewResourceDescription, RhiError> (*)(VernonRhiDevice,
                                                                                             uint64_t) noexcept;

struct BackendDispatch {
    VernonRhiBackend backend;
    uint32_t commandCapabilities;
    Option<CommandCapabilitiesCallback> commandCapabilitiesForDevice;
    bool (*ownsDevice)(VernonRhiDevice);
    Result<VernonRhiDevice, RhiError> (*createOwnedDevice)(const VernonRhiOwnedDeviceDescriptor *);
    Result<void, RhiError> (*destroyDevice)(VernonRhiDevice) noexcept;
    Option<VernonStringView> (*lastError)(VernonRhiDevice);
    Result<void *, RhiError> (*deviceState)(VernonRhiDevice);
    Result<CommandDeviceStateRef, RhiError> (*commandState)(VernonRhiDevice) noexcept;

    Result<VernonRhiBuffer, RhiError> (*createBuffer)(VernonRhiDevice, const VernonRhiBufferDescriptor *);
    Result<void, RhiError> (*uploadBuffer)(VernonRhiDevice, VernonRhiBuffer, uint64_t, const void *, uint64_t);
    Result<void, RhiError> (*uploadBufferRanges)(VernonRhiDevice, VernonRhiBuffer, const VernonRhiBufferUploadRange *,
                                                 size_t);
    Result<void, RhiError> (*downloadBufferRanges)(VernonRhiDevice, VernonRhiBuffer,
                                                   const VernonRhiBufferDownloadRange *, size_t);
    Result<void, RhiError> (*downloadBuffer)(VernonRhiDevice, VernonRhiBuffer, uint64_t, void *, uint64_t);
    Result<void, RhiError> (*destroyBuffer)(VernonRhiDevice, VernonRhiBuffer);
    Result<bool, RhiError> (*isBufferValid)(VernonRhiDevice, VernonRhiBuffer);
    Result<void *, RhiError> (*getBufferNativeHandle)(VernonRhiDevice, VernonRhiBuffer);

    Option<CreateImageCallback> createImage;
    Option<SetImageSamplerCallback> setImageSampler;
    Option<UploadImageCallback> uploadImage;
    Option<DownloadImageCallback> downloadImage;
    Option<DownloadImageBatchCallback> downloadImageBatch;
    Option<ImageOperationCallback> generateImageMipmaps;
    Option<BindImageCallback> bindImage;
    Option<ImageOperationCallback> destroyImage;
    Option<ImageValidityCallback> isImageValid;
    Option<ImageNativeHandleCallback> getImageNativeHandle;
    // Device capability bits for a format in the table; the front end intersects them with the table baseline.
    Option<Result<uint32_t, RhiError> (*)(VernonRhiDevice, VernonRhiFormat)> formatCapabilities;

    Option<Result<VernonRhiSampler, RhiError> (*)(VernonRhiDevice, const VernonRhiSamplerDescriptor *)> createSampler;
    Option<Result<void, RhiError> (*)(VernonRhiDevice, VernonRhiSampler)> destroySampler;
    Option<Result<bool, RhiError> (*)(VernonRhiDevice, VernonRhiSampler)> isSamplerValid;

    Result<uint64_t, RhiError> (*bufferResource)(VernonRhiDevice, VernonRhiBuffer) noexcept;
    Option<ImageResourceCallback> imageResource;
    Option<Result<uint64_t, RhiError> (*)(VernonRhiDevice, VernonRhiSampler) noexcept> samplerResource;
    Result<RetainedRhiResourceLease, RhiError> (*retainResource)(VernonRhiDevice, ResourceKind, uint64_t) noexcept;
    Result<uint64_t, RhiError> (*resolveRetainedResource)(VernonRhiDevice, ResourceKind, uint64_t) noexcept;
    Option<DescribeImageResourceCallback> describeImageResource;

    Result<CommandRecording, RhiError> (*beginCommands)(VernonRhiDevice) noexcept;
    Result<CommandSubmission, RhiError> (*submitCommands)(VernonRhiDevice, uint64_t, bool) noexcept;
    Option<PollCommandsCallback> pollCommands;
    Option<CompleteCommandsCallback> completeCommands;
    Option<AbandonCommandsCallback> abandonCommands;
    Option<RecordBarriersCallback> recordBarriers;
    Option<RecordBufferCopyCallback> recordBufferCopy;
    Option<RecordImageCopyCallback> recordImageCopy;
    Option<BeginRenderingCallback> beginRendering;
    Option<EndRenderingCallback> endRendering;
    Option<ClearColorCallback> clearColor;
    Option<ClearDepthStencilCallback> clearDepthStencil;
    Option<RecordDebugCommandCallback> recordDebugCommand;
    Option<RecordTimerCallback> recordTimer;
    Option<ReadTimersCallback> readTimers;
    Option<ReleaseTimersCallback> releaseTimers;
    Option<TrackedBufferStateCallback> trackedBufferState;

    Option<CreateImageViewCallback> createImageView;
    Option<ImageViewOperationCallback> destroyImageView;
    Option<ImageViewNativeHandleCallback> getImageViewNativeHandle;
    Option<ImageViewResourceCallback> imageViewResource;
    Option<DescribeImageViewResourceCallback> describeImageViewResource;
};

VERNON_RHI_CAPI void setDeviceCreationError(std::string_view error) noexcept;
VERNON_RHI_CAPI VernonStringView deviceCreationError() noexcept;
const BackendDispatch &openGLBackendDispatch();
// Encoder services for backends and adapters. Resources retained through the encoder `key` stay alive until it
// completes; cleanups run on completion or abandonment, rollbacks only on abandonment.
VERNON_RHI_CAPI Result<void, RhiError> retainCommandResource(VernonRhiDevice device, uint64_t key, ResourceKind kind,
                                                             uint64_t resource_key) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> resolveCommandResource(VernonRhiDevice device, uint64_t key,
                                                                  ResourceKind kind, uint64_t resource_key) noexcept;
inline Result<uint64_t, RhiError> retainResolvedCommandResource(VernonRhiDevice device, uint64_t key, ResourceKind kind,
                                                                uint64_t resource_key) noexcept {
    auto retained = retainCommandResource(device, key, kind, resource_key);
    if (retained.isErr())
        return Result<uint64_t, RhiError>{err(std::move(retained).error())};
    return resolveCommandResource(device, key, kind, resource_key);
}
VERNON_RHI_CAPI Result<void, RhiError> deferCommandCleanup(VernonRhiDevice device, uint64_t key, void *context,
                                                           uint64_t object, void (*cleanup)(void *, uint64_t)) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> deferCommandRollback(VernonRhiDevice device, uint64_t key, void *context,
                                                            uint64_t object,
                                                            void (*rollback)(void *, uint64_t)) noexcept;
#if defined(VERNON_HAS_CUDA_RHI)
const BackendDispatch &cudaBackendDispatch();
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
const BackendDispatch &directX12BackendDispatch();
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
const BackendDispatch &vulkanBackendDispatch();
#endif
#if defined(VERNON_HAS_METAL_RHI)
const BackendDispatch &metalBackendDispatch();
#endif

} // namespace vernon::rhi

#endif
