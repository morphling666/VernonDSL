#ifndef VERNON_RHI_INTERNAL_H
#define VERNON_RHI_INTERNAL_H

#include "VernonTextureTypes.h"
#include "backend_dispatch.h"

#include <vector>

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
// Writes `size` bytes for the commands of encoder `key` into memory it owns until it completes or is abandoned; see
// WriteTransientCallback. A failure fails the encoder.
VERNON_RHI_CAPI Result<CommandTransient, RhiError> writeCommandTransient(VernonRhiDevice device, uint64_t key,
                                                                         const void *data, uint64_t size,
                                                                         uint64_t alignment) noexcept;
VERNON_RHI_CAPI Result<CommandTransient, RhiError> writeRecordingTransient(VernonRhiDevice device, uint64_t key,
                                                                           uint64_t native, const void *data,
                                                                           uint64_t size, uint64_t alignment) noexcept;
// Reserves transient memory of encoder `key` for the host to fill in place; see ReserveTransientCallback. A failure
// fails the encoder.
VERNON_RHI_CAPI Result<CommandTransientReservation, RhiError>
reserveCommandTransient(VernonRhiDevice device, uint64_t key, uint64_t size, uint64_t alignment) noexcept;
VERNON_RHI_CAPI Result<CommandTransientReservation, RhiError> reserveRecordingTransient(VernonRhiDevice device,
                                                                                        uint64_t key, uint64_t native,
                                                                                        uint64_t size,
                                                                                        uint64_t alignment) noexcept;
// Places bytes in the device's persistent memory until `releasePersistentBlock`; see PlacePersistentCallback.
VERNON_RHI_CAPI Result<PersistentBlock, RhiError> placePersistentBlock(VernonRhiDevice device, const void *data,
                                                                       uint64_t size, uint64_t alignment) noexcept;
VERNON_RHI_CAPI void releasePersistentBlock(VernonRhiDevice device, uint64_t key) noexcept;
// Device memory that encoder-ordered writes update; see PlaceUpdatableCallback and WriteUpdatableCallback.
VERNON_RHI_CAPI Result<PersistentBlock, RhiError> placeUpdatableBlock(VernonRhiDevice device, uint64_t size,
                                                                      uint64_t alignment) noexcept;
// A failure fails encoder `key`.
VERNON_RHI_CAPI Result<void, RhiError> writeUpdatableBlock(VernonRhiDevice device, uint64_t key,
                                                           const PersistentBlock &block, const void *data,
                                                           uint64_t size) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> writeRecordingUpdatable(VernonRhiDevice device, uint64_t key, uint64_t native,
                                                               const PersistentBlock &block, const void *data,
                                                               uint64_t size) noexcept;
VERNON_RHI_CAPI void releaseUpdatableBlock(VernonRhiDevice device, uint64_t key) noexcept;
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
// A rendering scope checked against the descriptions of its attachments and reads, which never change for a key, so
// it stays valid while its owner retains those resources. Its clear values are the begin's.
struct PreparedRendering {
    RenderingScope scope;
    std::vector<RenderingRead> reads;
    std::vector<uint64_t> attachmentImages;
    // The images whose contents the scope reads: preserved attachments and sampled reads.
    std::vector<uint64_t> readImages;
};
VERNON_RHI_CAPI Result<PreparedRendering, RhiError> prepareRendering(VernonRhiDevice device,
                                                                     const RenderingScope &scope,
                                                                     const RenderingRead *reads,
                                                                     size_t readCount) noexcept;
// Begins `prepared` on encoder `key` with the clear values of `clears`, whose attachments match `prepared.scope`.
VERNON_RHI_CAPI Result<void, RhiError> beginPreparedEncoderRendering(VernonRhiDevice device, uint64_t key,
                                                                     const PreparedRendering &prepared,
                                                                     const RenderingScope &clears) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> endEncoderRendering(VernonRhiDevice device, uint64_t key) noexcept;
// An acquired surface image's contents are undefined until a submitted command writes it: a rendering scope that
// loads it with CLEAR or DISCARD, or a copy into it. Until then a rendering scope that preserves it, a read of it in a
// rendering scope and a copy from it are rejected unless an earlier command of the same encoder wrote it, and a
// download of it is rejected. Keys are image resource keys. Every surface holds one reservation from creation to
// destruction, which marking its acquired image uses; present forgets the image whether or not it was written.
VERNON_RHI_CAPI Result<void, RhiError> reserveUndefinedImage(VernonRhiDevice device) noexcept;
VERNON_RHI_CAPI void releaseUndefinedImage(VernonRhiDevice device) noexcept;
VERNON_RHI_CAPI void markImageContentsUndefined(VernonRhiDevice device, uint64_t imageKey) noexcept;
VERNON_RHI_CAPI void forgetUndefinedImage(VernonRhiDevice device, uint64_t imageKey) noexcept;
VERNON_RHI_CAPI bool imageContentsUndefined(VernonRhiDevice device, uint64_t imageKey) noexcept;
// The two above for a submission's images at once.
VERNON_RHI_CAPI void forgetUndefinedImages(VernonRhiDevice device, const uint64_t *imageKeys, size_t count) noexcept;
VERNON_RHI_CAPI bool anyImageContentsUndefined(VernonRhiDevice device, const uint64_t *imageKeys,
                                               size_t count) noexcept;
// Opens the native scope of the encoder's rendering scope unless it is open, and returns it. Draws open it before
// recording; `EndRendering` opens a scope that no draw opened so that its load operations apply.
VERNON_RHI_CAPI Result<OpenedRendering, RhiError> openCommandRendering(VernonRhiDevice device, uint64_t key) noexcept;
// A resource that commands write, by kind and resource key.
struct CommandWrite {
    ResourceKind kind{};
    uint64_t key{};
};
// Records the draws, dispatches and written resources of a provider packet batch that encoder `key` encoded outside a
// rendering scope.
VERNON_RHI_CAPI Result<void, RhiError> recordProviderCommands(VernonRhiDevice device, uint64_t key, uint32_t draws,
                                                              uint32_t dispatches, const CommandWrite *writes,
                                                              size_t write_count) noexcept;
// Whether the device emits debug groups and markers to the native API.
VERNON_RHI_CAPI Result<bool, RhiError> deviceEmitsLabels(VernonRhiDevice device) noexcept;
// `vernonRhiCommandEncoderPushDebugGroup`, `PopDebugGroup` and `InsertDebugMarker` for an encoder key.
VERNON_RHI_CAPI Result<void, RhiError> recordEncoderDebugCommand(VernonRhiDevice device, uint64_t key,
                                                                 DebugCommand command, std::string_view label) noexcept;
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
