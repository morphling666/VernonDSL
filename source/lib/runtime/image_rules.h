#ifndef VERNON_RUNTIME_IMAGE_RULES_H
#define VERNON_RUNTIME_IMAGE_RULES_H

// The extent, layer and view-dimension rules of each image dimension, which every HAL device and the graphics planner
// share; format, usage and mip-range rules stay with the caller.

#include "VernonTextureTypes.h"

#include <cstdint>

namespace vernon::runtime {

inline constexpr uint32_t kCubeFaces = 6;

constexpr bool knownImageDimension(VernonTextureDimension dimension) {
    return dimension == VERNON_TEXTURE_2D || dimension == VERNON_TEXTURE_3D || dimension == VERNON_TEXTURE_CUBE ||
           dimension == VERNON_TEXTURE_2D_ARRAY || dimension == VERNON_TEXTURE_CUBE_ARRAY;
}

constexpr bool cubeImageDimension(VernonTextureDimension dimension) {
    return dimension == VERNON_TEXTURE_CUBE || dimension == VERNON_TEXTURE_CUBE_ARRAY;
}

constexpr bool validImageShape(const VernonImageDescriptor &image) {
    const VernonImageExtent &extent = image.extent;
    if (!extent.width || !extent.height || !extent.depth || !image.array_layer_count)
        return false;
    switch (image.dimension) {
    case VERNON_TEXTURE_2D:
        return extent.depth == 1 && image.array_layer_count == 1;
    case VERNON_TEXTURE_2D_ARRAY:
        return extent.depth == 1;
    case VERNON_TEXTURE_3D:
        return image.array_layer_count == 1;
    case VERNON_TEXTURE_CUBE:
        return extent.width == extent.height && extent.depth == 1 && image.array_layer_count == kCubeFaces;
    case VERNON_TEXTURE_CUBE_ARRAY:
        return extent.width == extent.height && extent.depth == 1 && image.array_layer_count % kCubeFaces == 0;
    }
    return false;
}

// The levels of a full mip chain, down to a 1x1x1 level.
constexpr uint32_t maximumMipLevels(const VernonImageDescriptor &image) {
    uint32_t extent = image.extent.width > image.extent.height ? image.extent.width : image.extent.height;
    if (image.dimension == VERNON_TEXTURE_3D && image.extent.depth > extent)
        extent = image.extent.depth;
    uint32_t levels = 1;
    for (; extent > 1; extent >>= 1)
        ++levels;
    return levels;
}

// A view reinterprets layers, never the 3D/layered split: 2D and 2D_ARRAY views select layers of any non-3D image,
// and cube views select whole cubes, which start at a multiple of 6 layers. Range bounds stay with the caller.
constexpr bool compatibleViewDimension(VernonTextureDimension image, VernonTextureDimension view, uint32_t baseLayer,
                                       uint32_t layerCount) {
    switch (view) {
    case VERNON_TEXTURE_2D:
        return image != VERNON_TEXTURE_3D && layerCount == 1;
    case VERNON_TEXTURE_2D_ARRAY:
        return image != VERNON_TEXTURE_3D;
    case VERNON_TEXTURE_3D:
        return image == VERNON_TEXTURE_3D && baseLayer == 0 && layerCount == 1;
    case VERNON_TEXTURE_CUBE:
        return cubeImageDimension(image) && baseLayer % kCubeFaces == 0 && layerCount == kCubeFaces;
    case VERNON_TEXTURE_CUBE_ARRAY:
        return cubeImageDimension(image) && baseLayer % kCubeFaces == 0 && layerCount % kCubeFaces == 0;
    }
    return false;
}

} // namespace vernon::runtime

#endif
