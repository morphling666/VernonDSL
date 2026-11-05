#ifndef VERNON_HAL_VULKAN_DEVICE_H
#define VERNON_HAL_VULKAN_DEVICE_H

// The Vulkan device of linked schedules (specs/runtime/architecture_v2.md section 16.3): it records each command
// buffer's passes into native primary command buffers once per frame slot and submits a call as one queue submission.

#include "hal/hal.h"

#include <memory>

struct VernonRuntimeContext;

namespace vernon::hal {

// Null when the context's RHI device is not Vulkan or borrows its native objects, whose host submits for it.
std::unique_ptr<Device> createVulkanDevice(VernonRuntimeContext &context);

} // namespace vernon::hal

#endif
