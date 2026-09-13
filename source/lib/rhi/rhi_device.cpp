#include "VernonRHI.h"

#include "public_c_boundary.h"
#include "rhi_internal.h"

namespace {
vernon::Result<VernonStringView, vernon::RhiError> deviceLastErrorImpl(VernonRhiDevice device) {
    auto result = vernon::rhi::deviceLastError(device);
    if (result.isErr())
        return vernon::Result<VernonStringView, vernon::RhiError>{vernon::err(std::move(result).error())};
    return vernon::Result<VernonStringView, vernon::RhiError>{
        vernon::ok(result.value() ? result.value().value() : VernonStringView{})};
}
} // namespace

extern "C" uint32_t vernonRhiGetApiVersion(void) { return VERNON_PROGRAM_VERSION; }

extern "C" VernonRhiDevice vernonRhiCreateDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    const VernonRhiDevice invalid{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    return vernon::rhi::publicHandleBoundary([&] { return vernon::rhi::createDeviceImpl(descriptor); }, invalid);
}

extern "C" void vernonRhiDestroyDevice(VernonRhiDevice device) {
    vernon::rhi::publicDestroyBoundary([&] { return vernon::rhi::destroyDeviceImpl(device); });
}

extern "C" VernonStringView vernonRhiDeviceGetLastError(VernonRhiDevice device) {
    return vernon::rhi::publicViewBoundary<VernonStringView>([&] { return deviceLastErrorImpl(device); });
}
