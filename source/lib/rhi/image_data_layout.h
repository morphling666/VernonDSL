#ifndef VERNON_RHI_IMAGE_DATA_LAYOUT_H
#define VERNON_RHI_IMAGE_DATA_LAYOUT_H

#include "VernonRHI.h"

#include <cstring>
#include <limits>
#include <optional>

namespace vernon::rhi {

inline constexpr size_t packedDepthStencilPixelSize = 8;

inline constexpr size_t imageFormatPixelSize(VernonRhiFormat format) {
    switch (format) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        return 1;
    case VERNON_RHI_FORMAT_RG8_UNORM:
    case VERNON_RHI_FORMAT_R16_FLOAT:
        return 2;
    case VERNON_RHI_FORMAT_RGB8_UNORM:
        return 3;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
    case VERNON_RHI_FORMAT_RGBA8_SRGB:
    case VERNON_RHI_FORMAT_R32_FLOAT:
    case VERNON_RHI_FORMAT_R11G11B10_FLOAT:
    case VERNON_RHI_FORMAT_D32_FLOAT:
        return 4;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
    case VERNON_RHI_FORMAT_RG32_FLOAT:
    case VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT:
        return 8;
    case VERNON_RHI_FORMAT_RGB32_FLOAT:
        return 12;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        return 16;
    default:
        return 0;
    }
}

inline constexpr uint32_t imageMipExtent(uint32_t extent, uint32_t mipLevel) {
    return (extent >> mipLevel) ? extent >> mipLevel : 1;
}

inline std::optional<size_t> imageRegionByteSize(VernonRhiFormat format, uint32_t width, uint32_t height,
                                                 uint32_t depth = 1) {
    const size_t pixelSize = imageFormatPixelSize(format);
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

inline std::optional<size_t> imageMipByteSize(const VernonRhiImageDescriptor &descriptor, uint32_t mipLevel) {
    if (mipLevel >= descriptor.mip_levels)
        return std::nullopt;
    return imageRegionByteSize(
        descriptor.format, imageMipExtent(descriptor.width, mipLevel), imageMipExtent(descriptor.height, mipLevel),
        descriptor.dimension == VERNON_RHI_IMAGE_3D ? imageMipExtent(descriptor.depth, mipLevel) : descriptor.depth);
}

inline constexpr bool uploadLayoutMatches(VernonRhiFormat destination, VernonRhiImageDataFormat sourceFormat,
                                          VernonRhiImageDataType sourceType) {
    switch (destination) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RED && sourceType == VERNON_RHI_IMAGE_DATA_UINT8;
    case VERNON_RHI_FORMAT_RG8_UNORM:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RG && sourceType == VERNON_RHI_IMAGE_DATA_UINT8;
    case VERNON_RHI_FORMAT_RGB8_UNORM:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RGB && sourceType == VERNON_RHI_IMAGE_DATA_UINT8;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
    case VERNON_RHI_FORMAT_RGBA8_SRGB:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RGBA && sourceType == VERNON_RHI_IMAGE_DATA_UINT8;
    case VERNON_RHI_FORMAT_R16_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RED && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT16;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RGBA && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT16;
    case VERNON_RHI_FORMAT_R32_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RED && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    case VERNON_RHI_FORMAT_RG32_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RG && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    case VERNON_RHI_FORMAT_RGB32_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RGB && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RGBA && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    case VERNON_RHI_FORMAT_R11G11B10_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_RGB && sourceType == VERNON_RHI_IMAGE_DATA_UINT32;
    case VERNON_RHI_FORMAT_D32_FLOAT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_DEPTH && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    case VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT:
        return sourceFormat == VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL && sourceType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    default:
        return false;
    }
}

inline std::optional<size_t> imageDownloadByteSize(const VernonRhiImageDescriptor &image,
                                                   const VernonRhiImageDownloadDescriptor &download) {
    if (download.mip_level >= image.mip_levels || download.array_layer >= image.array_layers ||
        !uploadLayoutMatches(image.format, download.destination_format, download.destination_type))
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
    return imageRegionByteSize(image.format, download.width, download.height, download.depth);
}

inline void storePackedDepthStencil(uint8_t *destination, float depth, uint8_t stencil) {
    std::memcpy(destination, &depth, sizeof(depth));
    destination[sizeof(depth)] = stencil;
    std::memset(destination + sizeof(depth) + sizeof(stencil), 0,
                packedDepthStencilPixelSize - sizeof(depth) - sizeof(stencil));
}

} // namespace vernon::rhi

#endif
