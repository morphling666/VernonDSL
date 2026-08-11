#ifndef VERNON_RUNTIME_PROVIDER_IMAGE_DESCRIPTION_H
#define VERNON_RUNTIME_PROVIDER_IMAGE_DESCRIPTION_H

#include "VernonRuntimeProvider.h"

namespace vernon::runtime {

inline bool providerImageDescriptionIsCanonical(const VernonRuntimeProviderImageDescription &description) {
    const auto &image = description.image;
    const auto &view = description.view;
    const auto &range = view.subresources;
    constexpr uint32_t allUsages = VERNON_IMAGE_SAMPLED | VERNON_IMAGE_STORAGE | VERNON_IMAGE_COLOR_ATTACHMENT |
                                   VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE |
                                   VERNON_IMAGE_TRANSFER_DESTINATION;
    const uint32_t expectedAspects = view.format == VERNON_TEXTURE_D32_FLOAT ? VERNON_IMAGE_ASPECT_DEPTH
                                     : view.format == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                                         ? VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL
                                         : VERNON_IMAGE_ASPECT_COLOR;
    const bool formatCompatible =
        view.format == image.format ||
        ((view.format == VERNON_TEXTURE_RGBA8_UNORM || view.format == VERNON_TEXTURE_RGBA8_SRGB) &&
         (image.format == VERNON_TEXTURE_RGBA8_UNORM || image.format == VERNON_TEXTURE_RGBA8_SRGB));
    const bool dimensionCompatible = view.dimension == image.dimension ||
                                     (view.dimension == VERNON_TEXTURE_2D && image.dimension == VERNON_TEXTURE_CUBE);
    return description.struct_size >= sizeof(description) && description.parent_identity != 0 &&
           description.resource_kind <= VERNON_RUNTIME_PROVIDER_IMAGE_VIEW && image.dimension <= VERNON_TEXTURE_CUBE &&
           view.dimension <= VERNON_TEXTURE_CUBE && image.format <= VERNON_TEXTURE_D32_FLOAT_S8_UINT &&
           view.format <= VERNON_TEXTURE_D32_FLOAT_S8_UINT && formatCompatible && dimensionCompatible &&
           image.extent.width != 0 && image.extent.height != 0 && image.extent.depth != 0 &&
           image.mip_level_count != 0 && image.array_layer_count != 0 && image.sample_count != 0 && image.usage != 0 &&
           (image.usage & ~allUsages) == 0 && range.mip_level_count != 0 && range.array_layer_count != 0 &&
           range.aspects != 0 && (range.aspects & ~expectedAspects) == 0 &&
           range.base_mip_level < image.mip_level_count &&
           range.mip_level_count <= image.mip_level_count - range.base_mip_level &&
           range.base_array_layer < image.array_layer_count &&
           range.array_layer_count <= image.array_layer_count - range.base_array_layer &&
           (image.dimension != VERNON_TEXTURE_2D || image.extent.depth == 1) &&
           (image.dimension != VERNON_TEXTURE_3D || image.array_layer_count == 1) &&
           (image.dimension != VERNON_TEXTURE_CUBE ||
            (image.extent.width == image.extent.height && image.extent.depth == 1 && image.array_layer_count == 6)) &&
           (view.dimension != VERNON_TEXTURE_3D || (range.base_array_layer == 0 && range.array_layer_count == 1)) &&
           (view.dimension != VERNON_TEXTURE_CUBE || range.array_layer_count == 6);
}

} // namespace vernon::runtime

#endif
