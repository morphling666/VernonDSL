#ifndef VERNON_HAL_METAL_METAL_DEVICE_H
#define VERNON_HAL_METAL_METAL_DEVICE_H

// The Metal device of linked schedules (specs/runtime/architecture_v2.md section 16.3): it plans each command buffer's
// passes into a command list once, and encodes the list into a fresh MTLCommandBuffer every call, whose bytes and
// argument buffers live in one of the recording's frame slots.

#include "hal/hal.h"

#include <memory>

struct VernonRuntimeContext;

namespace vernon::hal {

// Null unless the context is a Metal context with its device state.
std::unique_ptr<Device> createMetalDevice(VernonRuntimeContext &context);

} // namespace vernon::hal

#endif
