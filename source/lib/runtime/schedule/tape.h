#ifndef VERNON_RUNTIME_SCHEDULE_TAPE_H
#define VERNON_RUNTIME_SCHEDULE_TAPE_H

// The VernonScheduleTape of a TAPE slot: what a forward dispatch wrote for its backward dispatch to replay. A writing
// dispatch rewrites it, so a tape holds only its last forward call's.

#include "hal/hal.h"
#include "runtime/autodiff/host_tape_allocator.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace vernon::schedule {

// The carriers of a GPU kernel's tape, in the order of its reflected autodiff roles: the tape words, one replay
// segment per workgroup, and the replay status a forward kernel reports overflow in.
enum class DeviceTapeCarrier : uint8_t { None, Data, Segments, Status };

// A GPU forward dispatch's tape: `stride` bytes per invocation in `data`, the segments that place each workgroup's
// lanes there, and the status its lanes raise when one needs more than `stride` (gpu::BatchSummary). A call that
// needs more than the carriers hold retires them rather than freeing them, since a tape the forward calls of a VJP
// share may still be written by an earlier submission.
struct DeviceTape {
    std::unique_ptr<hal::Buffer> data;
    std::unique_ptr<hal::Buffer> segments;
    std::unique_ptr<hal::Buffer> status;
    uint64_t dataBytes{};
    uint64_t segmentBytes{};
    uint64_t stride{};
    uint64_t dataCapacity{};
    uint64_t segmentCapacity{};
    std::vector<std::unique_ptr<hal::Buffer>> retired;
    // The i32 element counts of the data, segments and status, and their byte stride, which the carrier arguments of
    // a graphics pass point at.
    uint64_t words[3]{};
    int64_t wordStride{sizeof(int32_t)};
};

} // namespace vernon::schedule

struct VernonScheduleTape {
    std::shared_ptr<vernon::runtime::ad::HostStaticTapeBatch> batch;
    vernon::schedule::DeviceTape device;
    // The bytes per invocation a GPU forward dispatch gives the tape, when more than its Value's minimum stride.
    uint64_t requestedStride{};
};

#endif
