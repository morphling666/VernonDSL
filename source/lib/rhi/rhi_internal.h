#ifndef VERNON_RHI_INTERNAL_H
#define VERNON_RHI_INTERNAL_H

#include "VernonRHI.h"

namespace vernon::rhi {

VERNON_RHI_CAPI VernonRhiDevice createDevice(const VernonRhiOwnedDeviceDescriptor *descriptor);
VERNON_RHI_CAPI void destroyDevice(VernonRhiDevice device);
VERNON_RHI_CAPI VernonStringView deviceLastError(VernonRhiDevice device);
VERNON_RHI_CAPI VernonRhiStatus synchronizeDevice(VernonRhiDevice device);
VERNON_RHI_CAPI void *deviceState(VernonRhiDevice device, VernonRhiBackend backend);
VERNON_RHI_CAPI uint64_t bufferResource(VernonRhiDevice device, VernonRhiBuffer buffer);
VERNON_RHI_CAPI uint64_t imageResource(VernonRhiDevice device, VernonRhiImage image);
VERNON_RHI_CAPI uint64_t samplerResource(VernonRhiDevice device, VernonRhiSampler sampler);

} // namespace vernon::rhi

#endif
