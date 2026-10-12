#ifndef VERNON_RHI_IMAGE_DESCRIPTOR_VALIDATION_H
#define VERNON_RHI_IMAGE_DESCRIPTOR_VALIDATION_H

#include "VernonRHI.h"

namespace vernon::rhi {

inline constexpr uint32_t cubeFaceCount = 6;

constexpr bool knownImageDimension(VernonRhiImageDimension dimension) {
    return dimension == VERNON_RHI_IMAGE_2D || dimension == VERNON_RHI_IMAGE_3D || dimension == VERNON_RHI_IMAGE_CUBE ||
           dimension == VERNON_RHI_IMAGE_2D_ARRAY || dimension == VERNON_RHI_IMAGE_CUBE_ARRAY;
}

constexpr bool cubeImageDimension(VernonRhiImageDimension dimension) {
    return dimension == VERNON_RHI_IMAGE_CUBE || dimension == VERNON_RHI_IMAGE_CUBE_ARRAY;
}

constexpr bool layeredImageDimension(VernonRhiImageDimension dimension) {
    return dimension == VERNON_RHI_IMAGE_2D_ARRAY || cubeImageDimension(dimension);
}

// The extent, depth and layer rules of each dimension; format, usage and mip rules stay with the caller.
// Header-only so the backend-agnostic runtime core shares these rules without linking the RHI.
constexpr bool validImageShape(const VernonRhiImageDescriptor &image) {
    if (image.width == 0 || image.height == 0 || image.depth == 0 || image.array_layers == 0)
        return false;
    switch (image.dimension) {
    case VERNON_RHI_IMAGE_2D:
        return image.depth == 1 && image.array_layers == 1;
    case VERNON_RHI_IMAGE_2D_ARRAY:
        return image.depth == 1;
    case VERNON_RHI_IMAGE_3D:
        return image.array_layers == 1;
    case VERNON_RHI_IMAGE_CUBE:
        return image.width == image.height && image.depth == 1 && image.array_layers == cubeFaceCount;
    case VERNON_RHI_IMAGE_CUBE_ARRAY:
        return image.width == image.height && image.depth == 1 && image.array_layers % cubeFaceCount == 0;
    }
    return false;
}

// A view reinterprets layers, never the 3D/layered split: 2D and 2D_ARRAY views select layers of any non-3D image,
// and cube views select whole cubes, which start at a multiple of 6 layers. Range bounds stay with the caller.
constexpr bool compatibleViewDimension(VernonRhiImageDimension image, VernonRhiImageDimension view, uint32_t baseLayer,
                                       uint32_t layerCount) {
    switch (view) {
    case VERNON_RHI_IMAGE_2D:
        return image != VERNON_RHI_IMAGE_3D && layerCount == 1;
    case VERNON_RHI_IMAGE_2D_ARRAY:
        return image != VERNON_RHI_IMAGE_3D;
    case VERNON_RHI_IMAGE_3D:
        return image == VERNON_RHI_IMAGE_3D && baseLayer == 0 && layerCount == 1;
    case VERNON_RHI_IMAGE_CUBE:
        return cubeImageDimension(image) && baseLayer % cubeFaceCount == 0 && layerCount == cubeFaceCount;
    case VERNON_RHI_IMAGE_CUBE_ARRAY:
        return cubeImageDimension(image) && baseLayer % cubeFaceCount == 0 && layerCount % cubeFaceCount == 0;
    }
    return false;
}

bool validImageViewDescriptor(const VernonRhiImageDescriptor &image, const VernonRhiImageViewDescriptor &view);
uint32_t imageFormatAspects(VernonRhiFormat format);

} // namespace vernon::rhi

#endif
