#ifndef VERNON_RHI_COMMAND_STATE_H
#define VERNON_RHI_COMMAND_STATE_H

#include "device_registry.h"

namespace vernon::rhi {

class CommandDeviceStateControl;

class [[nodiscard]] CommandDeviceStateRef {
public:
    CommandDeviceStateRef() noexcept;
    CommandDeviceStateRef(const CommandDeviceStateRef &) = delete;
    CommandDeviceStateRef &operator=(const CommandDeviceStateRef &) = delete;
    CommandDeviceStateRef(CommandDeviceStateRef &&other) noexcept;
    CommandDeviceStateRef &operator=(CommandDeviceStateRef &&other) noexcept;
    ~CommandDeviceStateRef() noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] Result<CommandDeviceStateRef, RhiError> retain() const noexcept;
    [[nodiscard]] Result<ChildLease, RhiError> retainDeviceLease() const noexcept;

private:
    explicit CommandDeviceStateRef(CheckedIntrusiveRef<CommandDeviceStateControl> control) noexcept;

    Option<CheckedIntrusiveRef<CommandDeviceStateControl>> control_;

    friend Result<CommandDeviceStateRef, RhiError> createCommandDeviceState(OwnerRef) noexcept;
    friend CommandDeviceStateControl &commandDeviceState(CommandDeviceStateRef &) noexcept;
};

[[nodiscard]] Result<CommandDeviceStateRef, RhiError> createCommandDeviceState(OwnerRef owner) noexcept;
[[nodiscard]] CommandDeviceStateControl &commandDeviceState(CommandDeviceStateRef &state) noexcept;

} // namespace vernon::rhi

#endif
