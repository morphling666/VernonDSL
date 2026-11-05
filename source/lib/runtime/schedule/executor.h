#ifndef VERNON_RUNTIME_SCHEDULE_EXECUTOR_H
#define VERNON_RUNTIME_SCHEDULE_EXECUTOR_H

// The executor of specs/runtime/architecture_v2.md section 7: records a linked schedule once, then each call checks
// its extents and grid, places its transients in the arena, and submits the recorded commands with the caller's
// record. A call completes as the device's submit does and returns the timeline point its submission signals.

#include "VernonResult.hpp"
#include "hal/hal.h"
#include "runtime/schedule/linker.h"

#include <memory>
#include <string>
#include <vector>

namespace vernon::schedule {

struct Recording {
    std::vector<std::unique_ptr<hal::Executable>> executables;
    std::unique_ptr<hal::CommandBuffer> commands;
    // The state of the calls, which run one at a time: the record a call submits, the arena, and the capacity class
    // in bytes of each transient it was packed for, with the offsets it gave them.
    std::vector<uint8_t> record;
    std::unique_ptr<hal::Buffer> arena;
    uint64_t arenaSize{};
    // The schedule's constants, uploaded when it is recorded.
    std::unique_ptr<hal::Buffer> constants;
    std::vector<uint64_t> classes;
    std::vector<uint64_t> offsets;
    hal::TimelinePoint lastPoint{};
};

// Extent `dimension` of the argument `record` passes in `slot`: an ARGUMENT slot's is its Tensor's (-1 for none), any
// other slot's is its extent field.
int64_t slotExtent(const LinkedSchedule &schedule, const uint8_t *record, uint32_t slot, uint32_t dimension);

[[nodiscard]] Result<Recording, std::string> record(hal::Device &device, const LinkedSchedule &schedule);

[[nodiscard]] Result<hal::TimelinePoint, std::string> call(hal::Device &device, const LinkedSchedule &schedule,
                                                           Recording &recording, const uint8_t *record, size_t size);

} // namespace vernon::schedule

#endif
