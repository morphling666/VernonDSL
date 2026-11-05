#ifndef VERNON_HAL_CPU_DEVICE_H
#define VERNON_HAL_CPU_DEVICE_H

#include "hal/hal.h"

#include <memory>

namespace vernon::runtime {
class CpuWorkgroupScheduler;
}

namespace vernon::hal {

// The CPU as a HAL device: memory is unified, a command buffer is a tape of dispatches, and the queue is the
// scheduler's worker pool, which a submission runs to completion before it returns.
std::unique_ptr<Device> createCpuDevice(runtime::CpuWorkgroupScheduler &scheduler);

} // namespace vernon::hal

#endif
