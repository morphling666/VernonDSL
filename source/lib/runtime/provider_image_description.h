#ifndef VERNON_RUNTIME_PROVIDER_IMAGE_DESCRIPTION_H
#define VERNON_RUNTIME_PROVIDER_IMAGE_DESCRIPTION_H

#include "VernonFormats.hpp"
#include "VernonRuntimeProvider.h"
#include "rhi/image_descriptor_validation.h"

static_assert(static_cast<int>(VERNON_TEXTURE_2D) == VERNON_RHI_IMAGE_2D &&
                  static_cast<int>(VERNON_TEXTURE_3D) == VERNON_RHI_IMAGE_3D &&
                  static_cast<int>(VERNON_TEXTURE_CUBE) == VERNON_RHI_IMAGE_CUBE &&
                  static_cast<int>(VERNON_TEXTURE_2D_ARRAY) == VERNON_RHI_IMAGE_2D_ARRAY &&
                  static_cast<int>(VERNON_TEXTURE_CUBE_ARRAY) == VERNON_RHI_IMAGE_CUBE_ARRAY,
              "VernonTextureDimension mirrors VernonRhiImageDimension");

namespace vernon::runtime {

inline bool providerImageDescriptionIsCanonical(const VernonRuntimeProviderImageDescription &description) {
    const auto &image = description.image;
    const auto &view = description.view;
    const auto &range = view.subresources;
    constexpr uint32_t allUsages = VERNON_IMAGE_SAMPLED | VERNON_IMAGE_STORAGE | VERNON_IMAGE_COLOR_ATTACHMENT |
                                   VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE |
                                   VERNON_IMAGE_TRANSFER_DESTINATION;
    const uint32_t expectedAspects = formats::aspects(view.format);
    const auto imageDimension = static_cast<VernonRhiImageDimension>(image.dimension);
    const auto viewDimension = static_cast<VernonRhiImageDimension>(view.dimension);
    VernonRhiImageDescriptor shape{};
    shape.dimension = imageDimension;
    shape.width = image.extent.width;
    shape.height = image.extent.height;
    shape.depth = image.extent.depth;
    shape.array_layers = image.array_layer_count;
    return description.struct_size >= sizeof(description) && description.parent_identity != 0 &&
           description.resource_kind <= VERNON_RUNTIME_PROVIDER_IMAGE_VIEW &&
           rhi::knownImageDimension(imageDimension) && rhi::knownImageDimension(viewDimension) &&
           formats::viewCompatible(image.format, view.format) && rhi::validImageShape(shape) &&
           image.mip_level_count != 0 && image.sample_count != 0 && image.usage != 0 &&
           (image.usage & ~allUsages) == 0 && range.mip_level_count != 0 && range.array_layer_count != 0 &&
           range.aspects != 0 && (range.aspects & ~expectedAspects) == 0 &&
           range.base_mip_level < image.mip_level_count &&
           range.mip_level_count <= image.mip_level_count - range.base_mip_level &&
           range.base_array_layer < image.array_layer_count &&
           range.array_layer_count <= image.array_layer_count - range.base_array_layer &&
           rhi::compatibleViewDimension(imageDimension, viewDimension, range.base_array_layer, range.array_layer_count);
}

// The subresources of the image `parent` that a described image or image view covers.
struct ProviderImageSubresources {
    uint64_t parent{};
    VernonImageSubresourceRange range{};
};

inline ProviderImageSubresources
providerImageSubresources(const VernonRuntimeProviderImageDescription &description) noexcept {
    return {description.parent_identity, description.view.subresources};
}

inline bool providerImageSubresourcesOverlap(const ProviderImageSubresources &left,
                                             const ProviderImageSubresources &right) noexcept {
    const auto &a = left.range;
    const auto &b = right.range;
    return left.parent == right.parent && (a.aspects & b.aspects) != 0 &&
           a.base_mip_level < uint64_t{b.base_mip_level} + b.mip_level_count &&
           b.base_mip_level < uint64_t{a.base_mip_level} + a.mip_level_count &&
           a.base_array_layer < uint64_t{b.base_array_layer} + b.array_layer_count &&
           b.base_array_layer < uint64_t{a.base_array_layer} + a.array_layer_count;
}

} // namespace vernon::runtime

#endif
