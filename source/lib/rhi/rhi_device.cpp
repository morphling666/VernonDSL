#include "VernonRHI.h"

#include "rhi_internal.h"

extern "C" uint32_t vernonRhiGetApiVersion(void) { return VERNON_PIPELINE_VERSION; }

extern "C" VernonRhiDevice vernonRhiCreateDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    return vernon::rhi::createDevice(descriptor);
}

extern "C" void vernonRhiDestroyDevice(VernonRhiDevice device) { vernon::rhi::destroyDevice(device); }

extern "C" VernonStringView vernonRhiDeviceGetLastError(VernonRhiDevice device) {
    return vernon::rhi::deviceLastError(device);
}

extern "C" VernonRhiStatus vernonRhiDeviceSynchronize(VernonRhiDevice device) {
    return vernon::rhi::synchronizeDevice(device);
}
