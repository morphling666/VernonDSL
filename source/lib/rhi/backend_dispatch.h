#ifndef VERNON_RHI_BACKEND_DISPATCH_H
#define VERNON_RHI_BACKEND_DISPATCH_H

#include "VernonRHI.h"

#include <string>

namespace vernon::rhi {

enum class ResourceKind : uint32_t { Buffer = 1, Image = 2, Sampler = 3 };
enum CommandRenderingKind : uint32_t {
    CommandRenderingDynamic = 1,
    CommandRenderingRenderPass = 2,
    CommandRenderingStateless = 3
};

struct BackendDispatch {
    // Optional operations are null when the backend does not expose that
    // resource or command capability; the entry layer reports unsupported.
    VernonRhiBackend backend;
    bool (*ownsDevice)(VernonRhiDevice);
    VernonRhiDevice (*createOwnedDevice)(const VernonRhiOwnedDeviceDescriptor *);
    void (*destroyDevice)(VernonRhiDevice);
    VernonStringView (*lastError)(VernonRhiDevice);
    VernonRhiStatus (*synchronize)(VernonRhiDevice);
    void *(*deviceState)(VernonRhiDevice);

    VernonRhiStatus (*createBuffer)(VernonRhiDevice, const VernonRhiBufferDescriptor *, VernonRhiBuffer *);
    VernonRhiStatus (*uploadBuffer)(VernonRhiDevice, VernonRhiBuffer, uint64_t, const void *, uint64_t);
    VernonRhiStatus (*downloadBuffer)(VernonRhiDevice, VernonRhiBuffer, uint64_t, void *, uint64_t);
    VernonRhiStatus (*destroyBuffer)(VernonRhiDevice, VernonRhiBuffer);
    uint32_t (*isBufferValid)(VernonRhiDevice, VernonRhiBuffer);
    VernonRhiStatus (*getBufferNativeHandle)(VernonRhiDevice, VernonRhiBuffer, void **);

    VernonRhiStatus (*createImage)(VernonRhiDevice, const VernonRhiImageDescriptor *, VernonRhiImage *);
    VernonRhiStatus (*setImageSampler)(VernonRhiDevice, VernonRhiImage, const VernonRhiSamplerDescriptor *);
    VernonRhiStatus (*uploadImage)(VernonRhiDevice, VernonRhiImage, const VernonRhiImageUploadDescriptor *, size_t);
    VernonRhiStatus (*downloadImage)(VernonRhiDevice, VernonRhiImage, const VernonRhiImageDownloadDescriptor *, void *,
                                     size_t);
    VernonRhiStatus (*generateImageMipmaps)(VernonRhiDevice, VernonRhiImage);
    VernonRhiStatus (*bindImage)(VernonRhiDevice, VernonRhiImage, uint32_t);
    VernonRhiStatus (*destroyImage)(VernonRhiDevice, VernonRhiImage);
    uint32_t (*isImageValid)(VernonRhiDevice, VernonRhiImage);
    VernonRhiStatus (*getImageNativeHandle)(VernonRhiDevice, VernonRhiImage, uint64_t *);

    VernonRhiStatus (*createSampler)(VernonRhiDevice, const VernonRhiSamplerDescriptor *, VernonRhiSampler *);
    VernonRhiStatus (*destroySampler)(VernonRhiDevice, VernonRhiSampler);
    uint32_t (*isSamplerValid)(VernonRhiDevice, VernonRhiSampler);

    uint64_t (*bufferResource)(VernonRhiDevice, VernonRhiBuffer);
    uint64_t (*imageResource)(VernonRhiDevice, VernonRhiImage);
    uint64_t (*samplerResource)(VernonRhiDevice, VernonRhiSampler);
    bool (*retainResource)(VernonRhiDevice, ResourceKind, uint64_t);
    uint64_t (*resolveResource)(VernonRhiDevice, ResourceKind, uint64_t);
    bool (*describeImageResource)(VernonRhiDevice, uint64_t, VernonRhiImageDescriptor *);
    void (*releaseResource)(VernonRhiDevice, ResourceKind, uint64_t);

    bool (*beginCommands)(VernonRhiDevice, uint64_t &, VernonRhiBackend &);
    bool (*submitCommands)(VernonRhiDevice, uint64_t, bool, bool &);
    void (*completeBorrowedCommands)(VernonRhiDevice, uint64_t);
    void (*abandonCommands)(VernonRhiDevice, uint64_t);
    bool (*recordBarriers)(VernonRhiDevice, uint64_t, uint64_t, const VernonRhiBarrier *, size_t);
    bool (*endRendering)(VernonRhiDevice, uint64_t, VernonRhiBackend, uint32_t, uint32_t, uint32_t, const uint64_t *,
                         size_t, uint64_t, uint64_t);
    bool (*clearColor)(VernonRhiDevice, uint64_t, VernonRhiBackend, uint32_t, int32_t, int32_t, uint32_t, uint32_t,
                       uint32_t, uint64_t, uint32_t, const float[4]);
    bool (*clearDepthStencil)(VernonRhiDevice, uint64_t, VernonRhiBackend, uint32_t, int32_t, int32_t, uint32_t,
                              uint32_t, uint32_t, uint64_t, float, uint32_t, uint32_t);
    uint64_t (*trackedBufferState)(VernonRhiDevice, VernonRhiBuffer);

    VernonRhiStatus (*createImageView)(VernonRhiDevice, const VernonRhiImageViewDescriptor *, VernonRhiImageView *);
    VernonRhiStatus (*destroyImageView)(VernonRhiDevice, VernonRhiImageView);
    VernonRhiStatus (*getImageViewNativeHandle)(VernonRhiDevice, VernonRhiImageView, uint64_t *);
};

void setDeviceCreationError(std::string error);
const BackendDispatch &openGLBackendDispatch();
VERNON_RHI_CAPI bool deferCommandRollback(VernonRhiDevice device, uint64_t encoderKey, void *context, uint64_t object,
                                          void (*rollback)(void *, uint64_t));
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
