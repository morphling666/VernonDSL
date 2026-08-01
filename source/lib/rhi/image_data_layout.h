#ifndef VERNON_RHI_IMAGE_DATA_LAYOUT_H
#define VERNON_RHI_IMAGE_DATA_LAYOUT_H

#include "VernonRHI.h"

#include <cstring>
#include <limits>
#include <optional>

namespace vernon::rhi {

inline constexpr size_t packedDepthStencilPixelSize = 8;

inline std::optional<size_t> simpleImageDownloadSize(const VernonRhiImageDescriptor &descriptor) {
    if (descriptor.dimension != VERNON_RHI_IMAGE_2D ||
        (descriptor.format != VERNON_RHI_FORMAT_RGBA8_UNORM && descriptor.format != VERNON_RHI_FORMAT_D32_FLOAT &&
         descriptor.format != VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT) ||
        descriptor.depth != 1 || descriptor.mip_levels != 1 || descriptor.array_layers != 1 ||
        descriptor.width > (std::numeric_limits<size_t>::max)() / descriptor.height)
        return std::nullopt;
    const size_t pixels = static_cast<size_t>(descriptor.width) * descriptor.height;
    const size_t pixelSize = descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT ? packedDepthStencilPixelSize : 4;
    if (pixels > (std::numeric_limits<size_t>::max)() / pixelSize)
        return std::nullopt;
    return pixels * pixelSize;
}

inline void storePackedDepthStencil(uint8_t *destination, float depth, uint8_t stencil) {
    std::memcpy(destination, &depth, sizeof(depth));
    destination[sizeof(depth)] = stencil;
    std::memset(destination + sizeof(depth) + sizeof(stencil), 0,
                packedDepthStencilPixelSize - sizeof(depth) - sizeof(stencil));
}

} // namespace vernon::rhi

#endif
