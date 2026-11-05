#ifndef VERNON_RUNTIME_SCHEDULE_PROGRAM_ARGUMENTS_H
#define VERNON_RUNTIME_SCHEDULE_PROGRAM_ARGUMENTS_H

// The calling convention of a compute Program linked from its executable (specs/runtime/architecture_v2.md section
// 16.2): the caller's record holds a VernonProgramArgument per parameter, as a graphics node's does, and each call
// writes them into the slots and fields of the linked record. A VJP's backward schedule also reads arguments of the
// forward call again, and both bind residuals from the call's residual block.

#include "VernonResult.hpp"
#include "VernonSchedule.h"
#include "runtime/schedule/linker.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace vernon::schedule {

struct ProgramArguments {
    struct Site {
        // The ABI boundary slot of the argument, and its ARGUMENT slot's offset in the record it is read from.
        uint32_t boundary{};
        uint64_t offset{};
        // The linked slot the argument writes; kNoSlot for an output that shares an input's slot, whose argument the
        // input's already wrote.
        uint32_t slot{kNoSlot};
    };

    // The caller's arguments, an ARGUMENT slot of `slots` each; consecutive sites of one boundary share their slot.
    std::vector<Site> sites;
    // Arguments read from the record of an earlier call: a backward schedule's from its forward call.
    std::vector<Site> captures;
    // The linked slots bound from the call's residual block: a forward schedule's in block order, which allocates the
    // block's entries; a backward schedule's bind the entry `blockEntries` names, or the one at their index.
    std::vector<uint32_t> blockSlots;
    std::vector<uint32_t> blockEntries;
    // The caller's record.
    std::vector<VernonScheduleSlot> slots;
    uint64_t recordSize{};
    // Per linked slot, the indices of its fields in the linked schedule's fields.
    std::vector<std::vector<uint32_t>> fieldsOfSlot;
};

// The caller's arguments of a schedule whose `boundarySlots` pair a Program boundary slot with the linked slot it
// binds, or kNoSlot: a site per pair that `parameter` admits, in boundary slot order.
[[nodiscard]] Result<ProgramArguments, std::string>
programArguments(const LinkedSchedule &linked, std::vector<std::pair<uint32_t, uint32_t>> boundarySlots,
                 const std::function<bool(uint32_t)> &parameter);

// Fails unless the sites, captures and block slots write every linked slot.
[[nodiscard]] Result<void, std::string> checkCovered(const ProgramArguments &arguments, const LinkedSchedule &linked);

// Where a TensorView's allocation starts and its bytes (the view's byte_size); byte_offset counts from `address`.
struct BufferRange {
    uint8_t *address{};
    uint64_t bytes{};
};

using ResolveBuffer = Result<BufferRange, std::string> (*)(void *context, const VernonTensorView &view);

// Writes the arguments `sites` name in `record` into `linked`, a copy of the linked record template. Checks each
// against its slot's kind and element size; the call checks extents. An ARGUMENT slot takes the argument as it is,
// which the graphics device's pass checks.
[[nodiscard]] Result<void, std::string> writeProgramArguments(const ProgramArguments &arguments,
                                                              const std::vector<ProgramArguments::Site> &sites,
                                                              const LinkedSchedule &schedule, const uint8_t *record,
                                                              uint8_t *linked, ResolveBuffer resolve, void *context);

// Binds buffer slot `slot` to a dense row-major buffer of `shape` at `address`.
[[nodiscard]] Result<void, std::string> writeDenseBuffer(const ProgramArguments &arguments,
                                                         const LinkedSchedule &schedule, uint32_t slot,
                                                         uint8_t *address, const std::vector<int64_t> &shape,
                                                         uint8_t *linked);

} // namespace vernon::schedule

#endif
