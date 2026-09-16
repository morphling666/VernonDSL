#include "image_descriptor_validation.h"

namespace vernon::rhi {

uint32_t imageFormatAspects(VernonRhiFormat format) {
    if (format == VERNON_RHI_FORMAT_D32_FLOAT)
        return VERNON_RHI_IMAGE_ASPECT_DEPTH;
    if (format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT)
        return VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL;
    return VERNON_RHI_IMAGE_ASPECT_COLOR;
}

bool validImageViewDescriptor(const VernonRhiImageDescriptor &image, const VernonRhiImageViewDescriptor &view) {
    const bool compatibleFormat =
        view.format == image.format ||
        ((view.format == VERNON_RHI_FORMAT_RGBA8_UNORM || view.format == VERNON_RHI_FORMAT_RGBA8_SRGB) &&
         (image.format == VERNON_RHI_FORMAT_RGBA8_UNORM || image.format == VERNON_RHI_FORMAT_RGBA8_SRGB));
    const uint32_t availableAspects = imageFormatAspects(image.format);
    const bool compatibleDimension = view.dimension == image.dimension || (view.dimension == VERNON_RHI_IMAGE_2D &&
                                                                           image.dimension == VERNON_RHI_IMAGE_CUBE);
    return compatibleFormat && compatibleDimension && view.aspects != 0 && (view.aspects & ~availableAspects) == 0 &&
           view.mip_level_count != 0 && view.array_layer_count != 0 && view.base_mip_level < image.mip_levels &&
           view.mip_level_count <= image.mip_levels - view.base_mip_level &&
           view.base_array_layer < image.array_layers &&
           view.array_layer_count <= image.array_layers - view.base_array_layer &&
           (view.dimension != VERNON_RHI_IMAGE_3D || (view.base_array_layer == 0 && view.array_layer_count == 1)) &&
           (view.dimension != VERNON_RHI_IMAGE_CUBE || view.array_layer_count == 6);
}

} // namespace vernon::rhi
