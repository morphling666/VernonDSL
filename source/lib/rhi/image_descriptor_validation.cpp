#include "image_descriptor_validation.h"

#include "VernonFormats.hpp"

namespace vernon::rhi {

bool validImageViewDescriptor(const VernonRhiImageDescriptor &image, const VernonRhiImageViewDescriptor &view) {
    const uint32_t availableAspects = formats::aspects(image.format);
    return formats::viewCompatible(image.format, view.format) && view.aspects != 0 &&
           (view.aspects & ~availableAspects) == 0 && view.mip_level_count != 0 && view.array_layer_count != 0 &&
           view.base_mip_level < image.mip_levels && view.mip_level_count <= image.mip_levels - view.base_mip_level &&
           view.base_array_layer < image.array_layers &&
           view.array_layer_count <= image.array_layers - view.base_array_layer &&
           compatibleViewDimension(image.dimension, view.dimension, view.base_array_layer, view.array_layer_count);
}

} // namespace vernon::rhi
