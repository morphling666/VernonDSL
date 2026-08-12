#ifndef VERNON_RHI_INTERNAL_H
#define VERNON_RHI_INTERNAL_H

#include "VernonTextureTypes.h"
#include "backend_dispatch.h"
#include "image_descriptor_validation.h"

namespace vernon::rhi {

VERNON_RHI_CAPI VernonRhiDevice createDevice(const VernonRhiOwnedDeviceDescriptor *descriptor);
VERNON_RHI_CAPI VernonStringView deviceCreationError();
VERNON_RHI_CAPI void destroyDevice(VernonRhiDevice device);
VERNON_RHI_CAPI VernonStringView deviceLastError(VernonRhiDevice device);
VERNON_RHI_CAPI VernonRhiStatus synchronizeDevice(VernonRhiDevice device);
VERNON_RHI_CAPI bool deviceExists(VernonRhiDevice device);
VERNON_RHI_CAPI bool deviceHasActiveCommandEncoder(VernonRhiDevice device);
VERNON_RHI_CAPI void drainDeviceCompletions(VernonRhiDevice device);
VERNON_RHI_CAPI void *deviceState(VernonRhiDevice device, VernonRhiBackend backend);
VERNON_RHI_CAPI uint64_t bufferResource(VernonRhiDevice device, VernonRhiBuffer buffer);
VERNON_RHI_CAPI uint64_t imageResource(VernonRhiDevice device, VernonRhiImage image);
VERNON_RHI_CAPI uint64_t imageViewResource(VernonRhiDevice device, VernonRhiImageView view);
VERNON_RHI_CAPI uint64_t samplerResource(VernonRhiDevice device, VernonRhiSampler sampler);
VERNON_RHI_CAPI bool retainResource(VernonRhiDevice device, ResourceKind kind, uint64_t key);
VERNON_RHI_CAPI void releaseResource(VernonRhiDevice device, ResourceKind kind, uint64_t key);
VERNON_RHI_CAPI uint64_t resolveResource(VernonRhiDevice device, ResourceKind kind, uint64_t key);
VERNON_RHI_CAPI bool describeImageResource(VernonRhiDevice device, uint64_t key, VernonRhiImageDescriptor &descriptor);
VERNON_RHI_CAPI bool describeImageViewResource(VernonRhiDevice device, uint64_t key, VernonRhiImageViewDescriptor &view,
                                               VernonRhiImageDescriptor &image, uint64_t &parentKey);
VERNON_RHI_CAPI bool beginCommandRecording(VernonRhiDevice device, uint64_t &native, VernonRhiBackend &backend);
VERNON_RHI_CAPI bool submitCommandRecording(VernonRhiDevice device, uint64_t native, bool compute_writes,
                                            bool &completed, bool &external_completion);
VERNON_RHI_CAPI void completeBorrowedCommandRecording(VernonRhiDevice device, uint64_t native);
VERNON_RHI_CAPI void abandonCommandRecording(VernonRhiDevice device, uint64_t native);
VERNON_RHI_CAPI VernonRhiStatus recordBarriers(VernonRhiDevice device, uint64_t encoder_key, uint64_t native,
                                               const VernonRhiBarrier *barriers, size_t barrier_count);
VERNON_RHI_CAPI uint64_t commandEncoderKey(VernonRhiDevice device, VernonRhiCommandEncoder encoder);
VERNON_RHI_CAPI uint64_t commandEncoderNative(VernonRhiDevice device, uint64_t key, VernonRhiBackend backend);
VERNON_RHI_CAPI bool commandEncoderRendering(VernonRhiDevice device, uint64_t key);
VERNON_RHI_CAPI bool commandEncoderHasRenderingDescriptor(VernonRhiDevice device, uint64_t key);
VERNON_RHI_CAPI bool commandColorOperations(VernonRhiDevice device, uint64_t key, size_t index,
                                            VernonRhiLoadOperation &load, VernonRhiStoreOperation &store,
                                            float clear[4]);
VERNON_RHI_CAPI bool commandDepthOperations(VernonRhiDevice device, uint64_t key, VernonRhiLoadOperation &depth_load,
                                            VernonRhiStoreOperation &depth_store, VernonRhiLoadOperation &stencil_load,
                                            VernonRhiStoreOperation &stencil_store, float &clear_depth,
                                            uint32_t &clear_stencil);
VERNON_RHI_CAPI int claimCommandRendering(VernonRhiDevice device, uint64_t key, uint32_t backend_kind);
VERNON_RHI_CAPI uint64_t commandRenderingObject(VernonRhiDevice device, uint64_t key, uint64_t candidate);
VERNON_RHI_CAPI bool beginProviderRendering(VernonRhiDevice device, VernonRhiCommandEncoder encoder);
VERNON_RHI_CAPI bool recordProviderCommand(VernonRhiDevice device, uint64_t key, bool draw);
VERNON_RHI_CAPI bool recordCommandWriteResource(VernonRhiDevice device, uint64_t key, ResourceKind kind,
                                                uint64_t resource_key);
VERNON_RHI_CAPI bool retainCommandResource(VernonRhiDevice device, uint64_t key, ResourceKind kind,
                                           uint64_t resource_key);
VERNON_RHI_CAPI bool deferCommandCleanup(VernonRhiDevice device, uint64_t key, void *context, uint64_t object,
                                         void (*cleanup)(void *, uint64_t));
VERNON_RHI_CAPI bool deferCommandRollback(VernonRhiDevice device, uint64_t key, void *context, uint64_t object,
                                          void (*rollback)(void *, uint64_t));
VERNON_RHI_CAPI bool setCommandRenderingTargets(VernonRhiDevice device, uint64_t key, const uint64_t *colors,
                                                const uint64_t *resources, size_t color_count, uint64_t depth,
                                                uint64_t depth_resource);
VERNON_RHI_CAPI bool endCommandRendering(VernonRhiDevice device, uint64_t native, VernonRhiBackend backend,
                                         uint32_t backend_kind, uint32_t color_discard_mask,
                                         uint32_t depth_stencil_discard, const uint64_t *color_resources,
                                         size_t color_count, uint64_t depth_resource, uint64_t rendering_object);
VERNON_RHI_CAPI VernonRhiStatus clearCommandColor(VernonRhiDevice device, uint64_t native, VernonRhiBackend backend,
                                                  uint32_t backend_kind, int32_t x, int32_t y, uint32_t width,
                                                  uint32_t height, uint32_t layers, uint64_t target, uint32_t location,
                                                  const float color[4]);
VERNON_RHI_CAPI VernonRhiStatus clearCommandDepthStencil(VernonRhiDevice device, uint64_t native,
                                                         VernonRhiBackend backend, uint32_t backend_kind, int32_t x,
                                                         int32_t y, uint32_t width, uint32_t height, uint32_t layers,
                                                         uint64_t target, float depth, uint32_t stencil,
                                                         uint32_t aspects);
#if defined(VERNON_HAS_METAL_RHI) || defined(VERNON_HAS_METAL_RUNTIME)
VERNON_RHI_CAPI uint32_t metalTexturePixelFormat(VernonTextureFormat format);
#endif

} // namespace vernon::rhi

#endif
