#ifndef VERNON_RHI_COMMAND_STATE_H
#define VERNON_RHI_COMMAND_STATE_H

#include "VernonRHI.h"
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

    friend Result<CommandDeviceStateRef, RhiError> createCommandDeviceState(OwnerRef, uint32_t) noexcept;
    friend CommandDeviceStateControl &commandDeviceState(CommandDeviceStateRef &) noexcept;
};

constexpr uint32_t deviceFlagMask = VERNON_RHI_DEVICE_VALIDATION | VERNON_RHI_DEVICE_LABELS;
constexpr uint32_t openGLDeviceFlagMask = deviceFlagMask | VERNON_RHI_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT;

// Validation implies labels.
[[nodiscard]] constexpr uint32_t effectiveDeviceFlags(uint32_t flags) noexcept {
    return flags & VERNON_RHI_DEVICE_VALIDATION ? flags | VERNON_RHI_DEVICE_LABELS : flags;
}

// `flags` are VernonRhiDeviceFlagBits, which the backend has checked.
[[nodiscard]] Result<CommandDeviceStateRef, RhiError> createCommandDeviceState(OwnerRef owner, uint32_t flags) noexcept;
[[nodiscard]] CommandDeviceStateControl &commandDeviceState(CommandDeviceStateRef &state) noexcept;

} // namespace vernon::rhi

#endif
