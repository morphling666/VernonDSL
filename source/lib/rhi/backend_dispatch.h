#ifndef VERNON_RHI_BACKEND_DISPATCH_H
#define VERNON_RHI_BACKEND_DISPATCH_H

#include "rhi_command_state.h"
#include "rhi_lifecycle.h"

#include <string_view>

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
enum BackendCommandCapabilityBits : uint32_t {
    BackendCommandIndependentRecording = 1u << 0,
    BackendCommandConcurrentSubmission = 1u << 1,
    BackendCommandTimelineCompletion = 1u << 2,
    BackendCommandGpuTimestamps = 1u << 3,
    BackendCommandExplicitComputeDependencies = 1u << 4,
};

struct CommandRecording {
    uint64_t native{};
    VernonRhiBackend backend{VERNON_RHI_BACKEND_CUDA};
};

struct CommandSubmission {
    bool completed{};
    bool externalCompletion{};
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
using CompleteBorrowedCommandsCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t) noexcept;
using AbandonCommandsCallback = void (*)(VernonRhiDevice, uint64_t) noexcept;
using RecordBarriersCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, uint64_t, const VernonRhiBarrier *,
                                                          size_t) noexcept;
using RecordBufferCopyCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, VernonRhiBuffer, uint64_t,
                                                            VernonRhiBuffer, uint64_t, uint64_t) noexcept;
using RecordImageCopyCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, uint64_t, VernonRhiImage,
                                                           VernonRhiImage, const VernonRhiImageCopyRegion *,
                                                           size_t) noexcept;
using EndRenderingCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, VernonRhiBackend, uint32_t, uint32_t,
                                                        uint32_t, const uint64_t *, size_t, uint64_t,
                                                        uint64_t) noexcept;
using ClearColorCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, VernonRhiBackend, uint32_t, uint64_t,
                                                      int32_t, int32_t, uint32_t, uint32_t, uint32_t, uint64_t,
                                                      uint32_t, const float[4]) noexcept;
using ClearDepthStencilCallback = Result<void, RhiError> (*)(VernonRhiDevice, uint64_t, VernonRhiBackend, uint32_t,
                                                             uint64_t, int32_t, int32_t, uint32_t, uint32_t, uint32_t,
                                                             uint64_t, float, uint32_t, uint32_t) noexcept;
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
    Result<void, RhiError> (*synchronize)(VernonRhiDevice);
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
    Option<CompleteBorrowedCommandsCallback> completeBorrowedCommands;
    Option<AbandonCommandsCallback> abandonCommands;
    Option<RecordBarriersCallback> recordBarriers;
    Option<RecordBufferCopyCallback> recordBufferCopy;
    Option<RecordImageCopyCallback> recordImageCopy;
    Option<EndRenderingCallback> endRendering;
    Option<ClearColorCallback> clearColor;
    Option<ClearDepthStencilCallback> clearDepthStencil;
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
VERNON_RHI_CAPI Result<void, RhiError> deferCommandRollback(VernonRhiDevice device, uint64_t encoderKey, void *context,
                                                            uint64_t object,
                                                            void (*rollback)(void *, uint64_t)) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> deferCommandCleanup(VernonRhiDevice device, uint64_t encoderKey, void *context,
                                                           uint64_t object, void (*cleanup)(void *, uint64_t)) noexcept;
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
