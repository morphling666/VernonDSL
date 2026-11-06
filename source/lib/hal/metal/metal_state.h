#ifndef VERNON_HAL_METAL_METAL_STATE_H
#define VERNON_HAL_METAL_METAL_STATE_H

// The Metal device a Metal HAL device owns: its MTLDevice and command queue, the command buffers it committed,
// numbered in commit order, and the objects it keeps until the work committed before their release completes. Its
// resources, its runtime context and its HAL device share it; `mutex` guards everything but the native objects, which
// Metal makes thread safe. Only Objective-C++ sees the native objects (hal/metal/metal_native.h), so C++ translation
// units include this header.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace vernon::hal::metal {

struct Native;

struct DeviceState {
    DeviceState();
    DeviceState(const DeviceState &) = delete;
    DeviceState &operator=(const DeviceState &) = delete;
    ~DeviceState();

    bool initialize(uint32_t deviceIndex, uint32_t flags, std::string &error);

    // Command buffers commit in value order on one queue, which runs them in that order.
    bool waitForValue(uint64_t value, std::string &error);
    bool completed(uint64_t value, bool &done, std::string &error);
    bool synchronize(std::string &error);

    std::mutex mutex;
    std::unique_ptr<Native> native;
    bool labels{};
    // A committed command buffer failed; every later wait fails.
    bool lost{};
    std::string lostReason;
    uint64_t submittedValue{};
    uint64_t completedValue{};
};

} // namespace vernon::hal::metal

#endif
