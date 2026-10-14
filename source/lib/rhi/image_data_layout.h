#ifndef VERNON_RHI_IMAGE_DATA_LAYOUT_H
#define VERNON_RHI_IMAGE_DATA_LAYOUT_H

#include "VernonFormats.hpp"
#include "VernonRHI.h"

#include <cstring>
#include <limits>
#include <optional>

namespace vernon::rhi {

inline constexpr size_t packedDepthStencilPixelSize = 8;

inline constexpr size_t imageFormatPixelSize(VernonRhiFormat format) {
    const formats::Format *entry = formats::find(format);
    return entry ? entry->pixelBytes : 0;
}

inline constexpr uint32_t imageMipExtent(uint32_t extent, uint32_t mipLevel) {
    return (extent >> mipLevel) ? extent >> mipLevel : 1;
}

inline std::optional<size_t> regionByteSize(size_t pixelSize, uint32_t width, uint32_t height, uint32_t depth) {
    if (!pixelSize || !width || !height || !depth || width > (std::numeric_limits<size_t>::max)() / height)
        return std::nullopt;
    size_t pixels = static_cast<size_t>(width) * height;
    if (pixels > (std::numeric_limits<size_t>::max)() / depth)
        return std::nullopt;
    pixels *= depth;
    if (pixels > (std::numeric_limits<size_t>::max)() / pixelSize)
        return std::nullopt;
    return pixels * pixelSize;
}

inline std::optional<size_t> imageRegionByteSize(VernonRhiFormat format, uint32_t width, uint32_t height,
                                                 uint32_t depth = 1) {
    return regionByteSize(imageFormatPixelSize(format), width, height, depth);
}

inline std::optional<size_t> imageMipByteSize(const VernonRhiImageDescriptor &descriptor, uint32_t mipLevel) {
    if (mipLevel >= descriptor.mip_levels)
        return std::nullopt;
    return imageRegionByteSize(
        descriptor.format, imageMipExtent(descriptor.width, mipLevel), imageMipExtent(descriptor.height, mipLevel),
        descriptor.dimension == VERNON_RHI_IMAGE_3D ? imageMipExtent(descriptor.depth, mipLevel) : descriptor.depth);
}

inline std::optional<size_t> imageByteSize(const VernonRhiImageDescriptor &descriptor) {
    size_t layer = 0;
    for (uint32_t mipLevel = 0; mipLevel < descriptor.mip_levels; ++mipLevel) {
        const std::optional<size_t> mip = imageMipByteSize(descriptor, mipLevel);
        if (!mip || *mip > (std::numeric_limits<size_t>::max)() - layer)
            return std::nullopt;
        layer += *mip;
    }
    const size_t layers = descriptor.array_layers ? descriptor.array_layers : 1;
    if (!layer || layer > (std::numeric_limits<size_t>::max)() / layers)
        return std::nullopt;
    return layer * layers;
}

inline constexpr bool uploadLayoutMatches(VernonRhiFormat destination, VernonRhiImageDataFormat sourceFormat,
                                          VernonRhiImageDataType sourceType) {
    const formats::Format *entry = formats::find(destination);
    return entry && entry->dataFormat == sourceFormat && entry->dataType == sourceType;
}

inline constexpr bool imageTransferAspectMatches(VernonRhiFormat format, uint32_t aspect,
                                                 VernonRhiImageDataFormat dataFormat, VernonRhiImageDataType dataType) {
    if (!uploadLayoutMatches(format, dataFormat, dataType)) {
        if (!formats::hasDepth(format) || !formats::hasStencil(format))
            return false;
        if (aspect == VERNON_RHI_IMAGE_ASPECT_DEPTH)
            return dataFormat == VERNON_RHI_IMAGE_DATA_DEPTH && dataType == VERNON_RHI_IMAGE_DATA_FLOAT32;
        if (aspect == VERNON_RHI_IMAGE_ASPECT_STENCIL)
            return dataFormat == VERNON_RHI_IMAGE_DATA_STENCIL && dataType == VERNON_RHI_IMAGE_DATA_UINT8;
        return false;
    }
    return aspect == formats::aspects(format);
}

inline constexpr size_t imageTransferPixelSize(VernonRhiFormat format, uint32_t aspect) {
    if (formats::hasDepth(format) && formats::hasStencil(format)) {
        if (aspect == VERNON_RHI_IMAGE_ASPECT_DEPTH)
            return sizeof(float);
        if (aspect == VERNON_RHI_IMAGE_ASPECT_STENCIL)
            return sizeof(uint8_t);
    }
    return imageFormatPixelSize(format);
}

inline std::optional<size_t> imageTransferRegionByteSize(VernonRhiFormat format, uint32_t aspect, uint32_t width,
                                                         uint32_t height, uint32_t depth = 1) {
    return regionByteSize(imageTransferPixelSize(format, aspect), width, height, depth);
}

inline std::optional<size_t> imageDownloadByteSize(const VernonRhiImageDescriptor &image,
                                                   const VernonRhiImageDownloadDescriptor &download) {
    if (download.mip_level >= image.mip_levels || download.array_layer >= image.array_layers ||
        !imageTransferAspectMatches(image.format, download.aspect, download.destination_format,
                                    download.destination_type))
        return std::nullopt;
    const uint32_t mipWidth = imageMipExtent(image.width, download.mip_level);
    const uint32_t mipHeight = imageMipExtent(image.height, download.mip_level);
    const uint32_t mipDepth =
        image.dimension == VERNON_RHI_IMAGE_3D ? imageMipExtent(image.depth, download.mip_level) : image.depth;
    if (!download.width || !download.height || !download.depth || download.offset_x >= mipWidth ||
        download.offset_y >= mipHeight || download.offset_z >= mipDepth ||
        download.width > mipWidth - download.offset_x || download.height > mipHeight - download.offset_y ||
        download.depth > mipDepth - download.offset_z ||
        (image.dimension != VERNON_RHI_IMAGE_3D && (download.offset_z != 0 || download.depth != 1)))
        return std::nullopt;
    return imageTransferRegionByteSize(image.format, download.aspect, download.width, download.height, download.depth);
}

inline void storePackedDepthStencil(uint8_t *destination, float depth, uint8_t stencil) {
    std::memcpy(destination, &depth, sizeof(depth));
    destination[sizeof(depth)] = stencil;
    std::memset(destination + sizeof(depth) + sizeof(stencil), 0,
                packedDepthStencilPixelSize - sizeof(depth) - sizeof(stencil));
}

inline void loadPackedDepthStencil(const uint8_t *source, float &depth, uint8_t &stencil) {
    std::memcpy(&depth, source, sizeof(depth));
    stencil = source[sizeof(depth)];
}

} // namespace vernon::rhi

#endif
