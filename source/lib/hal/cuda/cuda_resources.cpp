#include "hal/cuda/cuda_device.h"

#include "hal/native_buffers.h"
#include "hal/resources.h"

#include "VernonFormats.hpp"

#if defined(VERNON_HAS_CUDA_RUNTIME)

#include "rhi/cuda_backend.h"
#include "rhi/rhi_internal.h"

#include <limits>
#include <new>
#include <string>
#include <vector>

namespace vernon::hal {
namespace {

using rhi::cuda::DevicePointer;
using rhi::cuda::DeviceState;

template <typename T = void> Done<T> cudaFailure(rhi::cuda::Result status, const char *what) {
    return fail<T>(VERNON_HAL_INTERNAL_ERROR, rhi::cuda::describeResult(status, what));
}

struct CudaHalBuffer final : VernonHalBuffer {
    CudaHalBuffer(DeviceState &state, DevicePointer pointer) : state(state), pointer(pointer) {}
    CudaHalBuffer(const CudaHalBuffer &) = delete;
    CudaHalBuffer &operator=(const CudaHalBuffer &) = delete;
    // DeviceState::free keeps the allocation for the next one of its size, which the stream orders after every
    // submission that read it.
    ~CudaHalBuffer() override { (void)state.free(pointer); }

    DeviceState &state;
    DevicePointer pointer;
};

Done<VernonRuntimeProviderResourceReference> unsupported(const char *what) {
    return fail<VernonRuntimeProviderResourceReference>(VERNON_HAL_UNSUPPORTED, what);
}

class CudaResources final : public Resources {
public:
    explicit CudaResources(DeviceState &state) : state_(state) {}

    Done<uint32_t> formatCapabilities(VernonTextureFormat format) override {
        if (!formats::find(format))
            return fail<uint32_t>(VERNON_HAL_INVALID_ARGUMENT, "the format is unknown");
        return Done<uint32_t>{ok(0u)};
    }

    Done<std::unique_ptr<VernonHalBuffer>> createBuffer(const VernonHalBufferDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalBuffer>>;
        if (descriptor.size > (std::numeric_limits<size_t>::max)())
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "the buffer is too large");
        DevicePointer pointer{};
        const rhi::cuda::Result status = state_.allocate(pointer, static_cast<size_t>(descriptor.size));
        if (status != rhi::cuda::kSuccess)
            return cudaFailure<std::unique_ptr<VernonHalBuffer>>(status, "cuMemAlloc");
        auto buffer = std::unique_ptr<CudaHalBuffer>(new (std::nothrow) CudaHalBuffer(state_, pointer));
        if (!buffer) {
            (void)state_.free(pointer);
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a buffer");
        }
        return Created{ok(std::unique_ptr<VernonHalBuffer>(std::move(buffer)))};
    }

    Done<void> write(VernonHalBuffer &buffer, const VernonHalBufferWrite *writes, size_t count) override {
        std::vector<VernonRhiBufferUploadRange> ranges(count);
        for (size_t index = 0; index < count; ++index)
            ranges[index] = {writes[index].offset, writes[index].source, writes[index].size};
        const rhi::cuda::Result status =
            state_.uploadRanges(static_cast<CudaHalBuffer &>(buffer).pointer, ranges.data(), ranges.size());
        if (status != rhi::cuda::kSuccess)
            return cudaFailure(status, "cannot write the buffer");
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalBuffer &buffer, const VernonHalBufferRead *reads, size_t count) override {
        std::vector<VernonRhiBufferDownloadRange> ranges(count);
        for (size_t index = 0; index < count; ++index)
            ranges[index] = {reads[index].offset, reads[index].destination, reads[index].size};
        const rhi::cuda::Result status =
            state_.downloadRanges(static_cast<CudaHalBuffer &>(buffer).pointer, ranges.data(), ranges.size());
        if (status != rhi::cuda::kSuccess)
            return cudaFailure(status, "cannot read the buffer");
        return Done<void>{ok()};
    }

    Done<void *> address(const VernonHalBuffer &buffer) override {
        return Done<void *>{
            ok(reinterpret_cast<void *>(static_cast<uintptr_t>(static_cast<const CudaHalBuffer &>(buffer).pointer)))};
    }

    Done<std::unique_ptr<VernonHalImage>> createImage(const VernonImageDescriptor &) override {
        return fail<std::unique_ptr<VernonHalImage>>(VERNON_HAL_UNSUPPORTED, "the CUDA device has no images");
    }
    Done<void> write(VernonHalImage &, const VernonHalImageWrite *, size_t) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CUDA device has no images");
    }
    Done<void> read(VernonHalImage &, const VernonHalImageRead *, size_t) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CUDA device has no images");
    }
    Done<void> generateMipmaps(VernonHalImage &) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CUDA device has no images");
    }
    Done<std::unique_ptr<VernonHalImageView>> createView(VernonHalImage &, const VernonImageViewDescriptor &) override {
        return fail<std::unique_ptr<VernonHalImageView>>(VERNON_HAL_UNSUPPORTED, "the CUDA device has no images");
    }
    Done<std::unique_ptr<VernonHalSampler>> createSampler(const VernonHalSamplerDescriptor &) override {
        return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_UNSUPPORTED, "the CUDA device has no samplers");
    }
    Done<std::unique_ptr<VernonHalSurface>> createSurface(const VernonHalSurfaceDescriptor &) override {
        return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_UNSUPPORTED, "the CUDA device cannot present");
    }
    Done<VernonHalSurfaceCapabilities> capabilities(VernonHalSurface &) override {
        return fail<VernonHalSurfaceCapabilities>(VERNON_HAL_UNSUPPORTED, "the CUDA device cannot present");
    }
    Done<void> configure(VernonHalSurface &, const VernonHalSurfaceConfiguration &) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CUDA device cannot present");
    }
    Done<VernonHalSurfaceImage> acquire(VernonHalSurface &) override {
        return fail<VernonHalSurfaceImage>(VERNON_HAL_UNSUPPORTED, "the CUDA device cannot present");
    }
    Done<void> present(VernonHalSurface &) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CUDA device cannot present");
    }

    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalBuffer &buffer,
                                                           uint64_t offset, uint64_t size) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeBufferIdentity(&state_);
        reference.resource.value = static_cast<const CudaHalBuffer &>(buffer).pointer;
        reference.offset = offset;
        reference.size = size;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalImageView &) override {
        return unsupported("the CUDA device has no images");
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalSampler &) override {
        return unsupported("the CUDA device has no samplers");
    }

private:
    DeviceState &state_;
};

} // namespace

std::unique_ptr<Resources> createCudaResources(VernonRhiDevice device) {
    auto state = rhi::deviceState(device, VERNON_RHI_BACKEND_CUDA);
    if (state.isErr() || !state.value())
        return nullptr;
    return std::unique_ptr<Resources>(new (std::nothrow) CudaResources(*static_cast<DeviceState *>(state.value())));
}

} // namespace vernon::hal

#else

namespace vernon::hal {

std::unique_ptr<Resources> createCudaResources(VernonRhiDevice) { return nullptr; }

} // namespace vernon::hal

#endif
