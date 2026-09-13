#ifndef VERNON_RHI_INTERNAL_H
#define VERNON_RHI_INTERNAL_H

#include "VernonTextureTypes.h"
#include "backend_dispatch.h"

namespace vernon::rhi {

enum class CommandRenderingClaim : uint8_t { Existing, Acquired };

VERNON_RHI_CAPI Result<VernonRhiDevice, RhiError> createDeviceImpl(const VernonRhiOwnedDeviceDescriptor *descriptor);
VERNON_RHI_CAPI VernonStringView deviceCreationError() noexcept;
VERNON_RHI_CAPI Result<void, RhiError> destroyDeviceImpl(VernonRhiDevice device);
VERNON_RHI_CAPI Result<Option<VernonStringView>, RhiError> deviceLastError(VernonRhiDevice device);
VERNON_RHI_CAPI Result<void, RhiError> synchronizeDevice(VernonRhiDevice device);
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
VERNON_RHI_CAPI Result<bool, RhiError> commandEncoderHasRenderingDescriptor(VernonRhiDevice device,
                                                                            uint64_t key) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> commandColorOperations(VernonRhiDevice device, uint64_t key, size_t index,
                                                              VernonRhiLoadOperation &load,
                                                              VernonRhiStoreOperation &store, float clear[4]) noexcept;
VERNON_RHI_CAPI Result<void, RhiError>
commandDepthOperations(VernonRhiDevice device, uint64_t key, VernonRhiLoadOperation &depth_load,
                       VernonRhiStoreOperation &depth_store, VernonRhiLoadOperation &stencil_load,
                       VernonRhiStoreOperation &stencil_store, float &clear_depth, uint32_t &clear_stencil) noexcept;
VERNON_RHI_CAPI Result<CommandRenderingClaim, RhiError> claimCommandRendering(VernonRhiDevice device, uint64_t key,
                                                                              uint32_t backend_kind) noexcept;
VERNON_RHI_CAPI void rollbackCommandRenderingClaim(VernonRhiDevice device, uint64_t key,
                                                   uint32_t backend_kind) noexcept;
VERNON_RHI_CAPI Result<Option<uint64_t>, RhiError> commandRenderingObject(VernonRhiDevice device, uint64_t key,
                                                                          Option<uint64_t> candidate) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> installCommandRenderingObject(VernonRhiDevice device, uint64_t key,
                                                                     uint64_t candidate, const uint64_t *colors,
                                                                     const uint64_t *resources, size_t color_count,
                                                                     uint64_t depth, uint64_t depth_resource) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> beginProviderRendering(VernonRhiDevice device,
                                                              VernonRhiCommandEncoder encoder) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordProviderCommand(VernonRhiDevice device, uint64_t key, bool draw) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> recordCommandWriteResource(VernonRhiDevice device, uint64_t key,
                                                                  ResourceKind kind, uint64_t resource_key) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> retainCommandResource(VernonRhiDevice device, uint64_t key, ResourceKind kind,
                                                             uint64_t resource_key) noexcept;
VERNON_RHI_CAPI Result<uint64_t, RhiError> resolveCommandResource(VernonRhiDevice device, uint64_t key,
                                                                  ResourceKind kind, uint64_t resource_key) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> deferCommandCleanup(VernonRhiDevice device, uint64_t key, void *context,
                                                           uint64_t object, void (*cleanup)(void *, uint64_t)) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> deferCommandRollback(VernonRhiDevice device, uint64_t key, void *context,
                                                            uint64_t object,
                                                            void (*rollback)(void *, uint64_t)) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> setCommandRenderingTargets(VernonRhiDevice device, uint64_t key,
                                                                  const uint64_t *colors, const uint64_t *resources,
                                                                  size_t color_count, uint64_t depth,
                                                                  uint64_t depth_resource) noexcept;
VERNON_RHI_CAPI Result<void, RhiError> endCommandRendering(VernonRhiDevice device, uint64_t native,
                                                           VernonRhiBackend backend, uint32_t backend_kind,
                                                           uint32_t color_discard_mask, uint32_t depth_stencil_discard,
                                                           const uint64_t *color_resources, size_t color_count,
                                                           uint64_t depth_resource, uint64_t rendering_object) noexcept;
VERNON_RHI_CAPI Result<void, RhiError>
clearCommandColor(VernonRhiDevice device, uint64_t native, VernonRhiBackend backend, uint32_t backend_kind,
                  uint64_t rendering_object, int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers,
                  uint64_t target, uint32_t location, const float color[4]) noexcept;
VERNON_RHI_CAPI Result<void, RhiError>
clearCommandDepthStencil(VernonRhiDevice device, uint64_t native, VernonRhiBackend backend, uint32_t backend_kind,
                         uint64_t rendering_object, int32_t x, int32_t y, uint32_t width, uint32_t height,
                         uint32_t layers, uint64_t target, float depth, uint32_t stencil, uint32_t aspects) noexcept;
#if defined(VERNON_HAS_METAL_RHI) || defined(VERNON_HAS_METAL_RUNTIME)
VERNON_RHI_CAPI uint32_t metalTexturePixelFormat(VernonTextureFormat format);
#endif

} // namespace vernon::rhi

#endif
