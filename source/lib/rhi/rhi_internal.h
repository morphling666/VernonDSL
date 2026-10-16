#ifndef VERNON_RHI_INTERNAL_H
#define VERNON_RHI_INTERNAL_H

#include "VernonTextureTypes.h"
#include "backend_dispatch.h"

namespace vernon::rhi {

VERNON_RHI_CAPI Result<VernonRhiDevice, RhiError> createDeviceImpl(const VernonRhiOwnedDeviceDescriptor *descriptor);
VERNON_RHI_CAPI VernonStringView deviceCreationError() noexcept;
VERNON_RHI_CAPI Result<void, RhiError> destroyDeviceImpl(VernonRhiDevice device);
VERNON_RHI_CAPI Result<Option<VernonStringView>, RhiError> deviceLastError(VernonRhiDevice device);
VERNON_RHI_CAPI bool deviceExists(VernonRhiDevice device);
VERNON_RHI_CAPI uint32_t deviceCommandCapabilities(VernonRhiDevice device);
VERNON_RHI_CAPI bool deviceHasActiveCommandEncoder(VernonRhiDevice device);
VERNON_RHI_CAPI Result<CommandDeviceStateRef, RhiError> commandState(VernonRhiDevice device) noexcept;
VERNON_RHI_CAPI Result<ChildLease, RhiError> retainDeviceLease(VernonRhiDevice device) noexcept;
VERNON_RHI_CAPI Result<void *, RhiError> deviceState(VernonRhiDevice device, VernonRhiBackend backend);
VERNON_RHI_CAPI Result<uint64_t, RhiError> bufferResource(VernonRhiDevice device, VernonRhiBuffer buffer) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> imageResource(VernonRhiDevice device, VernonRhiImage image) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> imageViewResource(VernonRhiDevice device, VernonRhiImageView view) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> samplerResource(VernonRhiDevice device, VernonRhiSampler sampler) noexcept;
VERNON_RHI_CAPI Result<RetainedRhiResourceLease, RhiError> retainResource(VernonRhiDevice device, ResourceKind kind,
                                                                          uint64_t key) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> resolveResource(VernonRhiDevice device, ResourceKind kind, uint64_t key,
                                                           RetainedRhiResourceLease &lease) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> resolvePinnedResource(VernonRhiDevice device, ResourceKind kind,
                                                                 uint64_t key) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> describeImageResource(VernonRhiDevice device, uint64_t key,
                                                             VernonRhiImageDescriptor &descriptor) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> describeImageResource(VernonRhiDevice device, uint64_t key,
                                                             VernonRhiImageDescriptor &descriptor,
                                                             RetainedRhiResourceLease &lease) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> describeImageViewResource(VernonRhiDevice device, uint64_t key,
                                                                 VernonRhiImageViewDescriptor &view,
                                                                 VernonRhiImageDescriptor &image,
                                                                 uint64_t &parentKey) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> describeImageViewResource(VernonRhiDevice device, uint64_t key,
                                                                 VernonRhiImageViewDescriptor &view,
                                                                 VernonRhiImageDescriptor &image, uint64_t &parentKey,
                                                                 RetainedRhiResourceLease &lease) noexcept;
VERNON_RHI_CAPI Result<CommandRecording, RhiError> beginCommandRecording(VernonRhiDevice device) noexcept;
VERNON_RHI_CAPI Result<CommandSubmission, RhiError> submitCommandRecording(VernonRhiDevice device, uint64_t native,
                                                                           bool compute_writes) noexcept;
VERNON_RHI_CAPI Result<CommandPoll, RhiError> pollCommandRecording(VernonRhiDevice device, uint64_t native) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> completeCommandRecording(VernonRhiDevice device, uint64_t native) noexcept;
VERNON_RHI_CAPI void abandonCommandRecording(VernonRhiDevice device, uint64_t native) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordBarriers(VernonRhiDevice device, uint64_t encoder_key, uint64_t native,
                                                      const VernonRhiBarrier *barriers, size_t barrier_count) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordBufferCopy(VernonRhiDevice device, uint64_t native, VernonRhiBuffer source,
                                                        uint64_t source_offset, VernonRhiBuffer destination,
                                                        uint64_t destination_offset, uint64_t size) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordImageCopy(VernonRhiDevice device, uint64_t encoder_key, uint64_t native,
                                                       VernonRhiImage source, VernonRhiImage destination,
                                                       const VernonRhiImageCopyRegion *regions,
                                                       size_t region_count) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> commandEncoderKey(VernonRhiDevice device,
                                                             VernonRhiCommandEncoder encoder) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> commandEncoderNative(VernonRhiDevice device, uint64_t key,
                                                                VernonRhiBackend backend) noexcept;
VERNON_RHI_CAPI Result<bool, RhiError> commandEncoderRendering(VernonRhiDevice device, uint64_t key) noexcept;
// The rendering scope of `vernonRhiCommandEncoderBeginRendering`/`EndRendering` for an encoder key, with attachments
// as image view resource keys; `beginEncoderRendering` fills their formats and stencil flags. `reads` are the sampled
// images of the scope's draws, which the native scope makes readable when it opens.
VERNON_RHI_CAPI Result<void, RhiError> beginEncoderRendering(VernonRhiDevice device, uint64_t key, RenderingScope scope,
                                                             const RenderingRead *reads, size_t readCount) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> endEncoderRendering(VernonRhiDevice device, uint64_t key) noexcept;
// Opens the native scope of the encoder's rendering scope unless it is open, and returns it. Draws open it before
// recording; `EndRendering` opens a scope that no draw opened so that its load operations apply.
VERNON_RHI_CAPI Result<OpenedRendering, RhiError> openCommandRendering(VernonRhiDevice device, uint64_t key) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordProviderCommand(VernonRhiDevice device, uint64_t key, bool draw) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordCommandWriteResource(VernonRhiDevice device, uint64_t key,
                                                                  ResourceKind kind, uint64_t resource_key) noexcept;
VERNON_RHI_CAPI Result<OpenedRendering, RhiError> beginCommandRendering(VernonRhiDevice device, uint64_t key,
                                                                        uint64_t native, const RenderingScope &scope,
                                                                        const RenderingRead *reads,
                                                                        size_t readCount) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> endCommandRendering(VernonRhiDevice device, uint64_t native,
                                                           const RenderingScope &scope,
                                                           const OpenedRendering &opened) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> clearCommandColor(VernonRhiDevice device, uint64_t native,
                                                         const RenderingScope &scope, const OpenedRendering &opened,
                                                         uint32_t location, const VernonClearColor &color) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> clearCommandDepthStencil(VernonRhiDevice device, uint64_t native,
                                                                const RenderingScope &scope,
                                                                const OpenedRendering &opened, float depth,
                                                                uint32_t stencil, uint32_t aspects) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordDebugCommand(VernonRhiDevice device, uint64_t native,
                                                          const OpenedRendering *rendering, DebugCommand command,
                                                          std::string_view label) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordCommandTimer(VernonRhiDevice device, uint64_t native, uint64_t &storage,
                                                          uint32_t timer, TimerEdge edge) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> readCommandTimers(VernonRhiDevice device, uint64_t storage, uint32_t count,
                                                         VernonRhiTimerSpan *spans) noexcept;
VERNON_RHI_CAPI void releaseCommandTimers(VernonRhiDevice device, uint64_t storage) noexcept;
#if defined(VERNON_HAS_METAL_RHI) || defined(VERNON_HAS_METAL_RUNTIME)
VERNON_RHI_CAPI uint32_t metalTexturePixelFormat(VernonTextureFormat format);
#endif

} // namespace vernon::rhi

#endif
