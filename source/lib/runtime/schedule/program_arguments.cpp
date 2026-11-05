#include "runtime/schedule/program_arguments.h"

#include "runtime/tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace vernon::schedule {
namespace {

using Written = Result<void, std::string>;

Written failWrite(std::string message) { return Written{err(std::move(message))}; }

Written writeInteger(uint8_t *linked, const VernonScheduleField &field, int64_t value) {
    if (field.size == sizeof(int64_t)) {
        std::memcpy(linked + field.offset, &value, sizeof(value));
        return Written{ok()};
    }
    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
        return failWrite("a Program argument's metadata exceeds its 32-bit field");
    const auto narrowed = static_cast<int32_t>(value);
    std::memcpy(linked + field.offset, &narrowed, sizeof(narrowed));
    return Written{ok()};
}

// A buffer as its slot's fields describe it, in elements of the slot.
struct BufferView {
    uint8_t *address{};
    uint64_t bytes{};
    int64_t offset{};
    const int64_t *shape{};
    const int64_t *strides{};
};

Written writeBuffer(const ProgramArguments &arguments, const LinkedSchedule &schedule, uint32_t slot,
                    const BufferView &buffer, uint8_t *linked) {
    const uint64_t element = schedule.slots[slot].element_size;
    std::memcpy(linked + schedule.slots[slot].offset, &buffer.address, sizeof(buffer.address));
    for (uint32_t index : arguments.fieldsOfSlot[slot]) {
        const VernonScheduleField &field = schedule.fields[index];
        int64_t value = 0;
        switch (field.kind) {
        case VERNON_SCHEDULE_FIELD_OFFSET:
            value = buffer.offset;
            break;
        case VERNON_SCHEDULE_FIELD_EXTENT:
            value = buffer.shape[field.dimension];
            break;
        case VERNON_SCHEDULE_FIELD_STRIDE:
            value = buffer.strides[field.dimension];
            break;
        case VERNON_SCHEDULE_FIELD_ADDRESS:
            std::memcpy(linked + field.offset, &buffer.address, sizeof(buffer.address));
            continue;
        case VERNON_SCHEDULE_FIELD_CAPACITY:
            value = static_cast<int64_t>(buffer.bytes / element);
            break;
        case VERNON_SCHEDULE_FIELD_BYTES:
            value = static_cast<int64_t>(buffer.bytes);
            break;
        default:
            return failWrite("a buffer slot has a field a Program argument cannot write");
        }
        if (auto wrote = writeInteger(linked, field, value); wrote.isErr())
            return wrote;
    }
    return Written{ok()};
}

constexpr uint32_t kMaxRank = 8;

// The bytes a view's elements cover, as addresses; empty for a view of no elements.
struct TouchedSpan {
    uintptr_t begin{};
    uintptr_t end{};
    bool writes{};
    uint32_t boundary{};
};

Option<TouchedSpan> touchedSpan(const VernonTensorView &view, const uint8_t *address, uint64_t element, bool writes,
                                uint32_t boundary) {
    int64_t low = static_cast<int64_t>(view.byte_offset);
    int64_t high = low;
    for (uint32_t dimension = 0; dimension < view.rank; ++dimension) {
        if (!view.shape[dimension])
            return none;
        const int64_t reach = static_cast<int64_t>(view.shape[dimension] - 1) * view.byte_strides[dimension];
        (reach < 0 ? low : high) += reach;
    }
    const auto base = reinterpret_cast<uintptr_t>(address);
    return Option<TouchedSpan>{some(TouchedSpan{base + static_cast<uintptr_t>(low),
                                                base + static_cast<uintptr_t>(high) + element, writes, boundary})};
}

// Linked entries may assume their buffers do not alias (CPU entries mark them `noalias`), so a call fails when a
// written argument's elements overlap another argument's. Sorted by start, a span overlaps an earlier one exactly
// when it starts before the furthest end so far.
Written checkDisjointWrites(std::vector<TouchedSpan> &spans) {
    std::sort(spans.begin(), spans.end(),
              [](const TouchedSpan &left, const TouchedSpan &right) { return left.begin < right.begin; });
    const TouchedSpan *furthest = nullptr;
    const TouchedSpan *furthestWritten = nullptr;
    for (const TouchedSpan &span : spans) {
        const TouchedSpan *earlier = span.writes ? furthest : furthestWritten;
        if (earlier && span.begin < earlier->end)
            return failWrite("Program parameters " + std::to_string(earlier->boundary) + " and " +
                             std::to_string(span.boundary) + " overlap where one is written");
        if (!furthest || span.end > furthest->end)
            furthest = &span;
        if (span.writes && (!furthestWritten || span.end > furthestWritten->end))
            furthestWritten = &span;
    }
    return Written{ok()};
}

} // namespace

Result<ProgramArguments, std::string> programArguments(const LinkedSchedule &linked,
                                                       std::vector<std::pair<uint32_t, uint32_t>> boundarySlots,
                                                       const std::function<bool(uint32_t)> &parameter) {
    using Built = Result<ProgramArguments, std::string>;
    std::sort(boundarySlots.begin(), boundarySlots.end());
    ProgramArguments arguments;
    std::vector<bool> written(linked.slots.size(), false);
    for (const auto &[boundary, slot] : boundarySlots) {
        if (!parameter(boundary))
            continue;
        if (slot != kNoSlot && slot >= linked.slots.size())
            return Built{err(std::string("Program boundary slot names no slot of its schedule"))};
        ProgramArguments::Site site;
        site.boundary = boundary;
        site.offset = arguments.recordSize;
        if (slot != kNoSlot && !written[slot]) {
            site.slot = slot;
            written[slot] = true;
        }
        // A boundary that feeds several nodes of a graph is one argument written into each of their slots.
        if (!arguments.sites.empty() && arguments.sites.back().boundary == boundary) {
            site.offset = arguments.sites.back().offset;
            arguments.sites.push_back(site);
            continue;
        }
        VernonScheduleSlot outer{};
        outer.kind = VERNON_SCHEDULE_SLOT_ARGUMENT;
        outer.access = slot != kNoSlot ? linked.slots[slot].access : VERNON_SCHEDULE_ACCESS_READ;
        outer.offset = site.offset;
        outer.size = sizeof(VernonProgramArgument);
        arguments.slots.push_back(outer);
        arguments.recordSize = (site.offset + sizeof(VernonProgramArgument) + 15) / 16 * 16;
        arguments.sites.push_back(site);
    }
    arguments.fieldsOfSlot.resize(linked.slots.size());
    for (size_t index = 0; index < linked.fields.size(); ++index)
        arguments.fieldsOfSlot[linked.fields[index].slot].push_back(static_cast<uint32_t>(index));
    return Built{ok(std::move(arguments))};
}

Written checkCovered(const ProgramArguments &arguments, const LinkedSchedule &linked) {
    std::vector<uint32_t> writes(linked.slots.size(), 0);
    for (const auto *sites : {&arguments.sites, &arguments.captures})
        for (const ProgramArguments::Site &site : *sites)
            if (site.slot != kNoSlot)
                ++writes[site.slot];
    for (uint32_t slot : arguments.blockSlots) {
        if (slot >= linked.slots.size() || (linked.slots[slot].kind != VERNON_SCHEDULE_SLOT_BUFFER &&
                                            linked.slots[slot].kind != VERNON_SCHEDULE_SLOT_ARGUMENT &&
                                            linked.slots[slot].kind != VERNON_SCHEDULE_SLOT_TAPE))
            return failWrite("a residual block slot must be a buffer, argument or tape slot");
        ++writes[slot];
    }
    for (size_t slot = 0; slot < linked.slots.size(); ++slot) {
        if (writes[slot] != 1)
            return failWrite("schedule slot " + std::to_string(slot) +
                             (writes[slot] ? " has several Program arguments" : " takes no Program argument"));
        if (linked.slots[slot].kind == VERNON_SCHEDULE_SLOT_BUFFER && linked.slots[slot].rank > kMaxRank)
            return failWrite("schedule slot " + std::to_string(slot) + " has too many dimensions");
    }
    return Written{ok()};
}

Written writeProgramArguments(const ProgramArguments &arguments, const std::vector<ProgramArguments::Site> &sites,
                              const LinkedSchedule &schedule, const uint8_t *record, uint8_t *linked,
                              ResolveBuffer resolve, void *context) {
    std::vector<TouchedSpan> spans;
    std::vector<uint64_t> spanOffsets;
    for (const ProgramArguments::Site &site : sites) {
        if (site.slot == kNoSlot)
            continue;
        VernonProgramArgument argument;
        std::memcpy(&argument, record + site.offset, sizeof(argument));
        const auto name = [&] { return "Program parameter " + std::to_string(site.boundary); };
        const VernonScheduleSlot &slot = schedule.slots[site.slot];
        // A graphics device's pass reads the argument itself.
        if (slot.kind == VERNON_SCHEDULE_SLOT_ARGUMENT) {
            std::memcpy(linked + slot.offset, &argument, sizeof(argument));
            for (uint32_t index : arguments.fieldsOfSlot[site.slot])
                std::memcpy(linked + schedule.fields[index].offset, &argument, sizeof(argument));
            continue;
        }
        if (argument.kind != VERNON_PROGRAM_TENSOR)
            return failWrite(name() + " takes a Tensor");
        const VernonTensorView &view = argument.tensor;
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
            if (view.storage != VERNON_TENSOR_HOST || !view.host_data || view.byte_offset > view.byte_size ||
                view.byte_size - view.byte_offset < slot.size)
                return failWrite(name() + " takes " + std::to_string(slot.size) + " host bytes");
            const uint8_t *bytes = static_cast<const uint8_t *>(view.host_data) + view.byte_offset;
            std::memcpy(linked + slot.offset, bytes, slot.size);
            for (uint32_t index : arguments.fieldsOfSlot[site.slot])
                std::memcpy(linked + schedule.fields[index].offset, bytes, slot.size);
            continue;
        }
        if (slot.kind != VERNON_SCHEDULE_SLOT_BUFFER)
            return failWrite(name() + " has a slot a Program argument cannot write");
        if (view.rank != slot.rank || (view.rank && (!view.shape || !view.byte_strides)) ||
            !runtime::tensorFitsAllocation(view))
            return failWrite(name() + " takes a TensorView of rank " + std::to_string(slot.rank) +
                             " within its allocation");
        auto range = resolve(context, view);
        if (range.isErr())
            return failWrite(name() + ": " + range.error());
        const auto element = static_cast<int64_t>(slot.element_size);
        if (!element || view.byte_offset % slot.element_size || range.value().bytes % slot.element_size)
            return failWrite(name() + " is not aligned to its elements");
        int64_t shape[kMaxRank];
        int64_t strides[kMaxRank];
        for (uint32_t dimension = 0; dimension < view.rank; ++dimension) {
            if (view.byte_strides[dimension] % element)
                return failWrite(name() + " has a stride that is not a whole number of elements");
            shape[dimension] = static_cast<int64_t>(view.shape[dimension]);
            strides[dimension] = view.byte_strides[dimension] / element;
        }
        const BufferView buffer{range.value().address, range.value().bytes,
                                static_cast<int64_t>(view.byte_offset / slot.element_size), shape, strides};
        if (auto wrote = writeBuffer(arguments, schedule, site.slot, buffer, linked); wrote.isErr())
            return wrote;
        // One argument feeding the slots of several nodes is one span, written when any of the slots writes it.
        const bool writes = slot.access != VERNON_SCHEDULE_ACCESS_READ;
        if (!spans.empty() && spanOffsets.back() == site.offset) {
            spans.back().writes = spans.back().writes || writes;
        } else if (auto span = touchedSpan(view, range.value().address, slot.element_size, writes, site.boundary);
                   span.hasValue()) {
            spans.push_back(span.value());
            spanOffsets.push_back(site.offset);
        }
    }
    return checkDisjointWrites(spans);
}

Written writeDenseBuffer(const ProgramArguments &arguments, const LinkedSchedule &schedule, uint32_t slot,
                         uint8_t *address, const std::vector<int64_t> &shape, uint8_t *linked) {
    std::vector<int64_t> strides(shape.size(), 1);
    int64_t elements = shape.empty() ? 1 : shape.back();
    for (size_t dimension = shape.size(); dimension > 1; --dimension) {
        strides[dimension - 2] = strides[dimension - 1] * shape[dimension - 1];
        elements *= shape[dimension - 2];
    }
    const BufferView buffer{address, static_cast<uint64_t>(elements) * schedule.slots[slot].element_size, 0,
                            shape.data(), strides.data()};
    return writeBuffer(arguments, schedule, slot, buffer, linked);
}

} // namespace vernon::schedule
