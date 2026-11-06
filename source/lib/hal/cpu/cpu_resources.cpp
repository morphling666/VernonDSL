#include "hal/resources.h"

#include "VernonFormats.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

namespace vernon::hal {
namespace {

constexpr uintptr_t kAlignment = 256;
constexpr uint32_t kMaxExtent = 1u << 16;
constexpr uint32_t kMaxLayers = 2048;

uint32_t mipExtent(uint32_t extent, uint32_t mip) { return std::max<uint32_t>(1, extent >> mip); }

struct CpuBuffer final : VernonHalBuffer {
    std::unique_ptr<uint8_t[]> storage;
    // `storage` aligned to kAlignment.
    uint8_t *bytes{};
};

struct CpuImage final : VernonHalImage {
    const formats::Format *format{};
    std::vector<uint8_t> texels;
    // The byte offset of each subresource in `texels`, at mip * array_layer_count + layer.
    std::vector<size_t> offsets;

    uint8_t *texel(const VernonHalImageRegion &region, uint32_t x, uint32_t y, uint32_t z) {
        const uint32_t width = mipExtent(descriptor.extent.width, region.mip_level);
        const uint32_t height = mipExtent(descriptor.extent.height, region.mip_level);
        const size_t subresource =
            offsets[size_t{region.mip_level} * descriptor.array_layer_count + region.array_layer];
        return texels.data() + subresource + ((size_t{z} * height + y) * width + x) * format->pixelBytes;
    }
};

struct CpuImageView final : VernonHalImageView {};
struct CpuSampler final : VernonHalSampler {};

Done<void> checkDimension(const VernonImageDescriptor &image) {
    const VernonImageExtent &extent = image.extent;
    const uint32_t layers = image.array_layer_count;
    bool valid = true;
    switch (image.dimension) {
    case VERNON_TEXTURE_2D:
        valid = extent.depth == 1 && layers == 1;
        break;
    case VERNON_TEXTURE_2D_ARRAY:
        valid = extent.depth == 1;
        break;
    case VERNON_TEXTURE_3D:
        valid = layers == 1;
        break;
    case VERNON_TEXTURE_CUBE:
        valid = extent.depth == 1 && extent.width == extent.height && layers == 6;
        break;
    case VERNON_TEXTURE_CUBE_ARRAY:
        valid = extent.depth == 1 && extent.width == extent.height && layers % 6 == 0;
        break;
    default:
        valid = false;
    }
    if (!valid)
        return fail(VERNON_HAL_INVALID_ARGUMENT, "the image's extent and layers do not fit its dimension");
    return Done<void>{ok()};
}

// A CPU transfer copies whole texels in the format's own layout, so it converts nothing.
Done<const formats::Format *> checkLayout(const CpuImage &image, const VernonHalImageRegion &region) {
    const formats::Format &format = *image.format;
    if (region.aspect != format.aspects() || static_cast<uint32_t>(region.data_format) != format.dataFormat ||
        static_cast<uint32_t>(region.data_type) != format.dataType)
        return fail<const formats::Format *>(
            VERNON_HAL_UNSUPPORTED, "the CPU device transfers images only in their format's whole-texel layout");
    return Done<const formats::Format *>{ok(&format)};
}

class CpuResources final : public Resources {
public:
    Done<uint32_t> formatCapabilities(VernonTextureFormat format) override {
        const formats::Format *entry = formats::find(format);
        if (!entry)
            return fail<uint32_t>(VERNON_HAL_INVALID_ARGUMENT, "the format is unknown");
        return Done<uint32_t>{ok(
            entry->baseline & (VERNON_HAL_FORMAT_FILTERABLE | VERNON_HAL_FORMAT_STORAGE | VERNON_HAL_FORMAT_COPYABLE))};
    }

    Done<std::unique_ptr<VernonHalBuffer>> createBuffer(const VernonHalBufferDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalBuffer>>;
        auto buffer = std::unique_ptr<CpuBuffer>(new (std::nothrow) CpuBuffer());
        if (!buffer || descriptor.size > UINT64_MAX - kAlignment)
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a buffer");
        buffer->storage.reset(new (std::nothrow) uint8_t[static_cast<size_t>(descriptor.size + kAlignment)]());
        if (!buffer->storage)
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a buffer");
        const auto base = reinterpret_cast<uintptr_t>(buffer->storage.get());
        buffer->bytes = buffer->storage.get() + ((kAlignment - base % kAlignment) % kAlignment);
        return Created{ok(std::unique_ptr<VernonHalBuffer>(std::move(buffer)))};
    }

    Done<void> write(VernonHalBuffer &buffer, const VernonHalBufferWrite *writes, size_t count) override {
        uint8_t *bytes = static_cast<CpuBuffer &>(buffer).bytes;
        for (size_t index = 0; index < count; ++index)
            if (writes[index].size)
                std::memcpy(bytes + writes[index].offset, writes[index].source, writes[index].size);
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalBuffer &buffer, const VernonHalBufferRead *reads, size_t count) override {
        const uint8_t *bytes = static_cast<CpuBuffer &>(buffer).bytes;
        for (size_t index = 0; index < count; ++index)
            if (reads[index].size)
                std::memcpy(reads[index].destination, bytes + reads[index].offset, reads[index].size);
        return Done<void>{ok()};
    }

    Done<void *> address(const VernonHalBuffer &buffer) override {
        return Done<void *>{ok(static_cast<void *>(static_cast<const CpuBuffer &>(buffer).bytes))};
    }

    Done<std::unique_ptr<VernonHalImage>> createImage(const VernonImageDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalImage>>;
        const formats::Format *format = formats::find(descriptor.format);
        if (!format)
            return fail<std::unique_ptr<VernonHalImage>>(VERNON_HAL_INVALID_ARGUMENT, "the image format is unknown");
        if (descriptor.sample_count != 1 ||
            (descriptor.usage & (VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT)))
            return fail<std::unique_ptr<VernonHalImage>>(VERNON_HAL_UNSUPPORTED,
                                                         "the CPU device has no attachments or multisampled images");
        if ((descriptor.usage & VERNON_IMAGE_STORAGE) && !format->allows(VERNON_HAL_FORMAT_STORAGE))
            return fail<std::unique_ptr<VernonHalImage>>(VERNON_HAL_UNSUPPORTED, "the format has no storage images");
        const VernonImageExtent &extent = descriptor.extent;
        const uint32_t largest = std::max({extent.width, extent.height, extent.depth});
        uint32_t mips = 0;
        for (uint32_t size = largest; size; size >>= 1)
            ++mips;
        if (!extent.width || !extent.height || !extent.depth || largest > kMaxExtent || !descriptor.mip_level_count ||
            descriptor.mip_level_count > mips || !descriptor.array_layer_count ||
            descriptor.array_layer_count > kMaxLayers)
            return fail<std::unique_ptr<VernonHalImage>>(VERNON_HAL_INVALID_ARGUMENT,
                                                         "the image's extent, mip levels or layers are out of range");
        if (auto checked = checkDimension(descriptor); checked.isErr())
            return Created{err(std::move(checked).error())};
        auto image = std::unique_ptr<CpuImage>(new (std::nothrow) CpuImage());
        if (!image)
            return fail<std::unique_ptr<VernonHalImage>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate an image");
        image->format = format;
        try {
            size_t bytes = 0;
            for (uint32_t mip = 0; mip < descriptor.mip_level_count; ++mip)
                for (uint32_t layer = 0; layer < descriptor.array_layer_count; ++layer) {
                    image->offsets.push_back(bytes);
                    bytes += size_t{mipExtent(extent.width, mip)} * mipExtent(extent.height, mip) *
                             mipExtent(extent.depth, mip) * format->pixelBytes;
                }
            image->texels.assign(bytes, 0);
        } catch (const std::bad_alloc &) {
            return fail<std::unique_ptr<VernonHalImage>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate an image");
        }
        return Created{ok(std::unique_ptr<VernonHalImage>(std::move(image)))};
    }

    Done<void> write(VernonHalImage &image, const VernonHalImageWrite *writes, size_t count) override {
        auto &cpu = static_cast<CpuImage &>(image);
        for (size_t index = 0; index < count; ++index)
            if (auto checked = checkLayout(cpu, writes[index].region); checked.isErr())
                return Done<void>{err(std::move(checked).error())};
        for (size_t index = 0; index < count; ++index) {
            const VernonHalImageRegion &region = writes[index].region;
            const size_t row = size_t{region.width} * cpu.format->pixelBytes;
            const auto *source = static_cast<const uint8_t *>(writes[index].source);
            for (uint32_t z = 0; z < region.depth; ++z)
                for (uint32_t y = 0; y < region.height; ++y, source += row)
                    std::memcpy(cpu.texel(region, region.offset_x, region.offset_y + y, region.offset_z + z), source,
                                row);
        }
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalImage &image, const VernonHalImageRead *reads, size_t count) override {
        auto &cpu = static_cast<CpuImage &>(image);
        for (size_t index = 0; index < count; ++index) {
            const VernonHalImageRegion &region = reads[index].region;
            if (auto checked = checkLayout(cpu, region); checked.isErr())
                return Done<void>{err(std::move(checked).error())};
            if (reads[index].size < size_t{region.width} * region.height * region.depth * cpu.format->pixelBytes)
                return fail(VERNON_HAL_INVALID_ARGUMENT, "the read's destination is smaller than its region");
        }
        for (size_t index = 0; index < count; ++index) {
            const VernonHalImageRegion &region = reads[index].region;
            const size_t row = size_t{region.width} * cpu.format->pixelBytes;
            auto *destination = static_cast<uint8_t *>(reads[index].destination);
            for (uint32_t z = 0; z < region.depth; ++z)
                for (uint32_t y = 0; y < region.height; ++y, destination += row)
                    std::memcpy(destination,
                                cpu.texel(region, region.offset_x, region.offset_y + y, region.offset_z + z), row);
        }
        return Done<void>{ok()};
    }

    Done<void> generateMipmaps(VernonHalImage &) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CPU device generates no mipmaps");
    }

    Done<std::unique_ptr<VernonHalImageView>> createView(VernonHalImage &image,
                                                         const VernonImageViewDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalImageView>>;
        const VernonImageDescriptor &source = image.descriptor;
        const VernonImageSubresourceRange &range = descriptor.subresources;
        const uint32_t aspects = formats::aspects(source.format);
        if (!formats::viewCompatible(source.format, descriptor.format))
            return fail<std::unique_ptr<VernonHalImageView>>(VERNON_HAL_INVALID_ARGUMENT,
                                                             "the view's format cannot reinterpret the image's");
        if (!range.mip_level_count || range.base_mip_level >= source.mip_level_count ||
            range.mip_level_count > source.mip_level_count - range.base_mip_level || !range.array_layer_count ||
            range.base_array_layer >= source.array_layer_count ||
            range.array_layer_count > source.array_layer_count - range.base_array_layer || !range.aspects ||
            (range.aspects & ~aspects))
            return fail<std::unique_ptr<VernonHalImageView>>(VERNON_HAL_INVALID_ARGUMENT,
                                                             "the view's subresources are outside the image");
        const bool volume = source.dimension == VERNON_TEXTURE_3D;
        bool fits = false;
        switch (descriptor.dimension) {
        case VERNON_TEXTURE_2D:
            fits = !volume && range.array_layer_count == 1;
            break;
        case VERNON_TEXTURE_2D_ARRAY:
            fits = !volume;
            break;
        case VERNON_TEXTURE_3D:
            fits = volume;
            break;
        case VERNON_TEXTURE_CUBE:
            fits = !volume && range.array_layer_count == 6 && source.extent.width == source.extent.height;
            break;
        case VERNON_TEXTURE_CUBE_ARRAY:
            fits = !volume && range.array_layer_count % 6 == 0 && source.extent.width == source.extent.height;
            break;
        }
        if (!fits)
            return fail<std::unique_ptr<VernonHalImageView>>(VERNON_HAL_INVALID_ARGUMENT,
                                                             "the view's dimension does not fit its layers");
        auto view = std::unique_ptr<CpuImageView>(new (std::nothrow) CpuImageView());
        if (!view)
            return fail<std::unique_ptr<VernonHalImageView>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a view");
        return Created{ok(std::unique_ptr<VernonHalImageView>(std::move(view)))};
    }

    Done<std::unique_ptr<VernonHalSampler>> createSampler(const VernonHalSamplerDescriptor &) override {
        using Created = Done<std::unique_ptr<VernonHalSampler>>;
        auto sampler = std::unique_ptr<CpuSampler>(new (std::nothrow) CpuSampler());
        if (!sampler)
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a sampler");
        return Created{ok(std::unique_ptr<VernonHalSampler>(std::move(sampler)))};
    }

    Done<std::unique_ptr<VernonHalSurface>> createSurface(const VernonHalSurfaceDescriptor &) override {
        return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_UNSUPPORTED, "the CPU device cannot present");
    }
    Done<VernonHalSurfaceCapabilities> capabilities(VernonHalSurface &) override {
        return fail<VernonHalSurfaceCapabilities>(VERNON_HAL_UNSUPPORTED, "the CPU device cannot present");
    }
    Done<void> configure(VernonHalSurface &, const VernonHalSurfaceConfiguration &) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CPU device cannot present");
    }
    Done<VernonHalSurfaceImage> acquire(VernonHalSurface &) override {
        return fail<VernonHalSurfaceImage>(VERNON_HAL_UNSUPPORTED, "the CPU device cannot present");
    }
    Done<void> present(VernonHalSurface &) override {
        return fail(VERNON_HAL_UNSUPPORTED, "the CPU device cannot present");
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalBuffer &, uint64_t,
                                                           uint64_t) override {
        return unreferenced();
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalImageView &) override {
        return unreferenced();
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalSampler &) override {
        return unreferenced();
    }

private:
    static Done<VernonRuntimeProviderResourceReference> unreferenced() {
        return fail<VernonRuntimeProviderResourceReference>(VERNON_HAL_UNSUPPORTED,
                                                            "CPU schedule arguments are host tensors");
    }
};

} // namespace

std::unique_ptr<Resources> createCpuResources() {
    return std::unique_ptr<Resources>(new (std::nothrow) CpuResources());
}

} // namespace vernon::hal
