#ifndef VERNON_RHI_IMAGE_DESCRIPTOR_VALIDATION_H
#define VERNON_RHI_IMAGE_DESCRIPTOR_VALIDATION_H

#include "VernonRHI.h"

namespace vernon::rhi {

bool validImageViewDescriptor(const VernonRhiImageDescriptor &image, const VernonRhiImageViewDescriptor &view);
uint32_t imageFormatAspects(VernonRhiFormat format);

} // namespace vernon::rhi

#endif
