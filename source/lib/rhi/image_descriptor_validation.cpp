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
    return compatibleFormat && view.aspects != 0 && (view.aspects & ~availableAspects) == 0 &&
           view.mip_level_count != 0 && view.array_layer_count != 0 && view.base_mip_level < image.mip_levels &&
           view.mip_level_count <= image.mip_levels - view.base_mip_level &&
           view.base_array_layer < image.array_layers &&
           view.array_layer_count <= image.array_layers - view.base_array_layer &&
           compatibleViewDimension(image.dimension, view.dimension, view.base_array_layer, view.array_layer_count);
}

} // namespace vernon::rhi
