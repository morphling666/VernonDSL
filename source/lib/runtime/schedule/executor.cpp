#include "runtime/schedule/executor.h"

#include "runtime/autodiff/runtime_gpu_replay.h"
#include "runtime/schedule/tape.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace vernon::schedule {
namespace {

using Called = Result<hal::TimelinePoint, std::string>;

constexpr uint64_t kArenaAlignment = 256;

Called failCall(std::string message) { return Called{err(std::move(message))}; }

int64_t readField(const uint8_t *record, const VernonScheduleField &field) {
    if (field.size == sizeof(int32_t)) {
        int32_t value = 0;
        std::memcpy(&value, record + field.offset, sizeof(value));
        return value;
    }
    int64_t value = 0;
    std::memcpy(&value, record + field.offset, sizeof(value));
    return value;
}

bool writeField(uint8_t *record, const VernonScheduleField &field, int64_t value) {
    if (field.size == sizeof(int32_t)) {
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
            return false;
        const auto narrowed = static_cast<int32_t>(value);
        std::memcpy(record + field.offset, &narrowed, sizeof(narrowed));
        return true;
    }
    std::memcpy(record + field.offset, &value, sizeof(value));
    return true;
}

int64_t extentValue(const LinkedSchedule &schedule, const uint8_t *record, const Extent &extent) {
    return extent.slot == kNoSlot ? extent.value : slotExtent(schedule, record, extent.slot, extent.dimension);
}

// The next power of two of `bytes`, at least the arena's alignment.
uint64_t capacityClass(uint64_t bytes) {
    uint64_t capacity = kArenaAlignment;
    while (capacity < bytes)
        capacity <<= 1;
    return capacity;
}

// Gives a writing GPU dispatch's tape carriers for this call's grid: tape words of the tape's stride per invocation,
// one replay segment per workgroup and a cleared status. The segments and status are uploaded in submission order, so
// carriers large enough are rewritten in place after the earlier submissions that use them.
Result<void, std::string> allocateDeviceTape(hal::Device &device, const TapeUse &use, const uint32_t grid[3],
                                             VernonScheduleTape &tape) {
    using Allocated = Result<void, std::string>;
    namespace gpu = runtime::ad::gpu;
    size_t stride = 0;
    if (!gpu::normalizeTapeStride(static_cast<size_t>(std::max(use.stride, tape.requestedStride)), stride))
        return Allocated{err(std::string("autodiff tape stride overflows"))};
    const VernonLaunchSize launch{grid[0], grid[1], grid[2]};
    const VernonLaunchSize workgroup{use.workgroup[0], use.workgroup[1], use.workgroup[2]};
    const uint64_t groups = uint64_t{grid[0]} * grid[1] * grid[2];
    const uint64_t volume = uint64_t{workgroup.x} * workgroup.y * workgroup.z;
    if (!groups || !volume || stride > std::numeric_limits<uint32_t>::max() / volume ||
        groups > std::numeric_limits<uint32_t>::max() / (stride * volume))
        return Allocated{err(std::string("autodiff tape exceeds 4 GiB"))};
    const uint64_t dataBytes = stride * volume * groups;
    std::vector<gpu::Segment> segments(static_cast<size_t>(groups));
    for (size_t group = 0; group < segments.size(); ++group)
        if (!gpu::initializeBatchSegment(launch, workgroup, group, group, stride, static_cast<size_t>(volume),
                                         static_cast<size_t>(dataBytes), segments[group]))
            return Allocated{err(std::string("autodiff tape segments overflow"))};
    DeviceTape &carriers = tape.device;
    const uint64_t segmentBytes = segments.size() * sizeof(gpu::Segment);
    const auto replace = [&](std::unique_ptr<hal::Buffer> &buffer, uint64_t &capacity,
                             uint64_t bytes) -> Result<void, std::string> {
        if (buffer && capacity >= bytes)
            return Result<void, std::string>{ok()};
        auto allocated = device.allocate(bytes);
        if (allocated.isErr())
            return Result<void, std::string>{err(std::move(allocated).error())};
        if (buffer)
            carriers.retired.push_back(std::move(buffer));
        buffer = std::move(allocated).value();
        capacity = bytes;
        return Result<void, std::string>{ok()};
    };
    if (auto data = replace(carriers.data, carriers.dataCapacity, dataBytes); data.isErr())
        return data;
    if (auto placed = replace(carriers.segments, carriers.segmentCapacity, segmentBytes); placed.isErr())
        return placed;
    if (auto copied = device.enqueueUpload(*carriers.segments, segments.data(), segmentBytes); copied.isErr())
        return copied;
    const bool reports = std::any_of(use.fields.begin(), use.fields.end(), [](const DeviceTapeField &field) {
        return field.carrier == DeviceTapeCarrier::Status;
    });
    if (reports) {
        if (!carriers.status) {
            auto status = device.allocate(sizeof(gpu::BatchSummary));
            if (status.isErr())
                return Allocated{err(std::move(status).error())};
            carriers.status = std::move(status).value();
        }
        const gpu::BatchSummary clear{};
        if (auto copied = device.enqueueUpload(*carriers.status, &clear, sizeof(clear)); copied.isErr())
            return copied;
    }
    carriers.dataBytes = dataBytes;
    carriers.segmentBytes = segmentBytes;
    carriers.stride = stride;
    carriers.words[0] = dataBytes / sizeof(int32_t);
    carriers.words[1] = segmentBytes / sizeof(int32_t);
    carriers.words[2] = sizeof(gpu::BatchSummary) / sizeof(int32_t);
    return Allocated{ok()};
}

// Writes the carrier fields of a GPU dispatch's tape into `scratch`: each carrier is a dense i32 TensorView.
Result<void, std::string> writeDeviceTape(const TapeUse &use, const DeviceTape &carriers, uint8_t *scratch) {
    using Written = Result<void, std::string>;
    for (const DeviceTapeField &field : use.fields) {
        const hal::Buffer *buffer = field.carrier == DeviceTapeCarrier::Data       ? carriers.data.get()
                                    : field.carrier == DeviceTapeCarrier::Segments ? carriers.segments.get()
                                                                                   : carriers.status.get();
        const uint64_t bytes = field.carrier == DeviceTapeCarrier::Data       ? carriers.dataBytes
                               : field.carrier == DeviceTapeCarrier::Segments ? carriers.segmentBytes
                                                                              : sizeof(runtime::ad::gpu::BatchSummary);
        if (!buffer)
            return Written{err(std::string("autodiff tape carrier was never written"))};
        if (field.field.kind == kTapeFieldArgument) {
            const size_t carrier = static_cast<size_t>(field.carrier) - static_cast<size_t>(DeviceTapeCarrier::Data);
            VernonProgramArgument argument{};
            argument.kind = VERNON_PROGRAM_TENSOR;
            argument.tensor.struct_size = sizeof(argument.tensor);
            argument.tensor.storage = VERNON_TENSOR_DEVICE_RESOURCE;
            argument.tensor.resource = buffer->reference(0, bytes);
            argument.tensor.access = VERNON_ACCESS_READ_WRITE;
            argument.tensor.rank = 1;
            argument.tensor.shape = &carriers.words[carrier];
            argument.tensor.byte_strides = &carriers.wordStride;
            argument.tensor.byte_size = bytes;
            if (!argument.tensor.resource.identity)
                return Written{err(std::string("autodiff tape carrier has no buffer reference"))};
            std::memcpy(scratch + field.field.offset, &argument, sizeof(argument));
            continue;
        }
        int64_t value = 0;
        switch (field.field.kind) {
        case VERNON_SCHEDULE_FIELD_ADDRESS:
            value = static_cast<int64_t>(reinterpret_cast<uintptr_t>(buffer->address()));
            break;
        case VERNON_SCHEDULE_FIELD_EXTENT:
        case VERNON_SCHEDULE_FIELD_CAPACITY:
            value = static_cast<int64_t>(bytes / sizeof(int32_t));
            break;
        case VERNON_SCHEDULE_FIELD_STRIDE:
            value = 1;
            break;
        default:
            break;
        }
        if (!writeField(scratch, field.field, value))
            return Written{err(std::string("autodiff tape carrier exceeds its metadata field"))};
    }
    return Written{ok()};
}

} // namespace

int64_t slotExtent(const LinkedSchedule &schedule, const uint8_t *record, uint32_t slot, uint32_t dimension) {
    if (schedule.slots[slot].kind == VERNON_SCHEDULE_SLOT_ARGUMENT) {
        VernonProgramArgument argument;
        std::memcpy(&argument, record + schedule.slots[slot].offset, sizeof(argument));
        if (argument.kind != VERNON_PROGRAM_TENSOR || dimension >= argument.tensor.rank || !argument.tensor.shape)
            return -1;
        return static_cast<int64_t>(argument.tensor.shape[dimension]);
    }
    return readField(record, schedule.extentFields[slot][dimension]);
}

Result<Recording, std::string> record(hal::Device &device, const LinkedSchedule &schedule) {
    using Recorded = Result<Recording, std::string>;
    Recording recording;
    std::vector<hal::DispatchCommand> commands;
    try {
        for (const hal::ExecutableSource &source : schedule.executables) {
            auto executable = device.createExecutable(source);
            if (executable.isErr())
                return Recorded{err(std::move(executable).error())};
            recording.executables.push_back(std::move(executable).value());
        }
        uint32_t node = 0;
        for (size_t index = 0; index < schedule.operations.size(); ++index) {
            const LinkedDispatch &operation = schedule.operations[index];
            while (node + 1 < schedule.nodeOperationEnds.size() && index >= schedule.nodeOperationEnds[node])
                ++node;
            if (operation.kind == OperationKind::Fill) {
                commands.push_back(
                    {hal::CommandKind::Fill, nullptr, operation.argumentsOffset, operation.argumentsSize, 0});
                continue;
            }
            if (operation.kind == OperationKind::Copy) {
                hal::DispatchCommand copy{hal::CommandKind::Copy, nullptr, operation.argumentsOffset,
                                          operation.argumentsSize};
                copy.copy = operation.copy;
                copy.pass = operation.pass;
                commands.push_back(std::move(copy));
                continue;
            }
            if (operation.executable >= recording.executables.size())
                return Recorded{err(std::string("linked dispatch names no executable"))};
            if (operation.kind == OperationKind::Pass) {
                hal::DispatchCommand pass{hal::CommandKind::Pass, recording.executables[operation.executable].get()};
                pass.pass = operation.pass;
                pass.node = node;
                if (node < schedule.nodeLabels.size())
                    pass.label = schedule.nodeLabels[node];
                commands.push_back(std::move(pass));
                continue;
            }
            hal::DispatchTape tape;
            if (operation.tape.present()) {
                if (operation.tape.slot >= schedule.slots.size() ||
                    schedule.slots[operation.tape.slot].kind != VERNON_SCHEDULE_SLOT_TAPE)
                    return Recorded{err(std::string("linked dispatch names no autodiff tape slot"))};
                if (!operation.tape.device)
                    tape = {schedule.slots[operation.tape.slot].offset, operation.tape.allocatorOffset,
                            operation.tape.rootOffset, operation.tape.stride, operation.tape.writes};
            }
            hal::DispatchCommand dispatch{hal::CommandKind::Dispatch, recording.executables[operation.executable].get(),
                                          operation.argumentsOffset,  operation.argumentsSize,
                                          operation.gridOffset,       tape};
            dispatch.resultsSize = operation.resultsSize;
            for (const ResultUse &result : operation.results) {
                if (result.slot >= schedule.slots.size() ||
                    schedule.slots[result.slot].kind != VERNON_SCHEDULE_SLOT_BUFFER)
                    return Recorded{err(std::string("linked dispatch commits a result to no buffer slot"))};
                if (result.reduction)
                    dispatch.resultSums.push_back(
                        {result.frameOffset, result.size, static_cast<uint32_t>(*result.reduction)});
                for (const runtime::ValueByteCopy &copy : result.copies)
                    dispatch.results.push_back(
                        {schedule.slots[result.slot].offset, copy.physicalOffset, copy.canonicalOffset, copy.size});
            }
            dispatch.recordAddresses = operation.recordAddresses;
            dispatch.node = node;
            if (node < schedule.nodeLabels.size())
                dispatch.label = schedule.nodeLabels[node];
            commands.push_back(std::move(dispatch));
        }
        recording.record.resize(schedule.recordSize);
        recording.classes.assign(schedule.transients.size(), 0);
    } catch (const std::bad_alloc &) {
        return Recorded{err(std::string("cannot allocate a recording"))};
    }
    auto recorded = device.record(commands);
    if (recorded.isErr())
        return Recorded{err(std::move(recorded).error())};
    recording.commands = std::move(recorded).value();
    if (!schedule.constants.empty()) {
        auto constants = device.allocate(schedule.constants.size());
        if (constants.isErr())
            return Recorded{err(std::move(constants).error())};
        recording.constants = std::move(constants).value();
        if (auto uploaded = device.upload(*recording.constants, schedule.constants.data(), schedule.constants.size());
            uploaded.isErr())
            return Recorded{err(std::move(uploaded).error())};
    }
    return Recorded{ok(std::move(recording))};
}

Result<hal::TimelinePoint, std::string> call(hal::Device &device, const LinkedSchedule &schedule, Recording &recording,
                                             const uint8_t *record, size_t size) {
    if (!record || size != schedule.recordSize || recording.record.size() != size)
        return failCall("schedule record has the wrong size");
    for (size_t slot = 0; slot < schedule.slots.size(); ++slot) {
        for (uint32_t dimension = 0; dimension < schedule.slots[slot].rank; ++dimension) {
            const Extent &required = schedule.extents[slot][dimension];
            if (required.slot == slot && required.dimension == dimension)
                continue;
            if (slotExtent(schedule, record, static_cast<uint32_t>(slot), dimension) !=
                extentValue(schedule, record, required))
                return failCall("schedule argument " + std::to_string(slot) + " has extent " +
                                std::to_string(dimension) + " other than its Program requires");
        }
    }
    if (schedule.callerGrid) {
        for (const LinkedDispatch &operation : schedule.operations) {
            uint32_t grid[3];
            std::memcpy(grid, record + operation.gridOffset, sizeof(grid));
            for (unsigned axis = 0; axis < 3; ++axis) {
                if (!grid[axis])
                    return failCall(std::string("schedule grid axis ") + "xyz"[axis] + " must be nonzero");
                if (operation.unitGridAxes[axis] && grid[axis] != 1)
                    return failCall(std::string("schedule grid axis ") + "xyz"[axis] + " must equal 1");
            }
        }
    }
    for (const LinkedDispatch &operation : schedule.operations)
        for (const auto &[axis, slot] : operation.unitGridSlots) {
            uint32_t count = 0;
            std::memcpy(&count, record + schedule.slots[slot].offset, sizeof(count));
            if (count != 1)
                return failCall("compute dispatch grid axis " + std::to_string(axis) + " must equal 1");
        }
    const bool elementGrids =
        std::any_of(schedule.operations.begin(), schedule.operations.end(), [](const LinkedDispatch &operation) {
            return !operation.gridElements.empty() || !operation.invocationExtents.empty() || operation.tape.device;
        });
    if (schedule.transients.empty() && !elementGrids && schedule.constantAddresses.empty()) {
        auto submitted = device.submit(*recording.commands, record, size);
        if (submitted.isErr())
            return failCall(std::move(submitted).error());
        recording.lastPoint = submitted.value();
        return Called{ok(submitted.value())};
    }

    // Each transient's extents and bytes at this call. The arena is packed again only when a capacity class changes.
    const size_t count = schedule.transients.size();
    std::vector<uint64_t> elements(count, 1);
    std::vector<uint64_t> classes(count);
    bool repack = count && !recording.arena;
    for (size_t index = 0; index < count; ++index) {
        const Transient &transient = schedule.transients[index];
        for (const Extent &extent : transient.extents) {
            const int64_t value = extentValue(schedule, record, extent);
            if (value < 0 || (value && elements[index] > std::numeric_limits<uint64_t>::max() /
                                                             static_cast<uint64_t>(value) / transient.elementSize))
                return failCall("schedule allocation size overflows");
            elements[index] *= static_cast<uint64_t>(value);
        }
        classes[index] = capacityClass(elements[index] * transient.elementSize);
        repack = repack || classes[index] != recording.classes[index];
    }
    if (repack) {
        ArenaLayout layout = packArena(schedule.transients, classes, kArenaAlignment);
        if (!recording.arena || layout.size > recording.arenaSize) {
            // The previous arena may still be read by an earlier submission.
            if (recording.arena) {
                if (auto waited = device.wait(recording.lastPoint); waited.isErr())
                    return failCall(std::move(waited).error());
                recording.arena.reset();
            }
            auto arena = device.allocate(layout.size);
            if (arena.isErr())
                return failCall(std::move(arena).error());
            recording.arena = std::move(arena).value();
            recording.arenaSize = layout.size;
        }
        recording.classes = classes;
        recording.offsets = std::move(layout.offsets);
    }

    uint8_t *scratch = recording.record.data();
    std::memcpy(scratch, record, size);
    for (const LinkedDispatch &operation : schedule.operations) {
        if (operation.tape.device) {
            VernonScheduleTape *tape = nullptr;
            std::memcpy(&tape, record + schedule.slots[operation.tape.slot].offset, sizeof(tape));
            if (!tape)
                return failCall("schedule call passes no autodiff tape");
            if (operation.tape.writes) {
                uint32_t grid[3];
                if (operation.kind == OperationKind::Pass) {
                    auto counts = hal::gridCounts(operation.pass.grid, record);
                    if (counts.isErr())
                        return failCall("autodiff tape pass: " + counts.error());
                    std::copy(counts.value().begin(), counts.value().end(), grid);
                } else {
                    std::memcpy(grid, record + operation.gridOffset, sizeof(grid));
                }
                if (auto allocated = allocateDeviceTape(device, operation.tape, grid, *tape); allocated.isErr())
                    return failCall(std::move(allocated).error());
            }
            if (auto written = writeDeviceTape(operation.tape, tape->device, scratch); written.isErr())
                return failCall(std::move(written).error());
        }
        if (!operation.invocationExtents.empty()) {
            uint32_t grid[3];
            std::memcpy(grid, record + operation.gridOffset, sizeof(grid));
            for (const InvocationExtent &extent : operation.invocationExtents) {
                const uint64_t invocations = uint64_t{grid[0]} * grid[1] * grid[2] * extent.workgroupVolume;
                if (extent.size == sizeof(uint32_t) && invocations > std::numeric_limits<uint32_t>::max())
                    return failCall("schedule dispatch invocations exceed 32 bits");
                const uint32_t narrow = static_cast<uint32_t>(invocations);
                std::memcpy(scratch + extent.offset,
                            extent.size == sizeof(uint32_t) ? static_cast<const void *>(&narrow) : &invocations,
                            extent.size);
            }
        }
        if (operation.gridElements.empty())
            continue;
        uint64_t invocations = 1;
        for (const Extent &extent : operation.gridElements) {
            const int64_t value = extentValue(schedule, record, extent);
            if (value < 0 ||
                (value && invocations > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(value)))
                return failCall("schedule built-in grid exceeds 32 bits");
            invocations *= static_cast<uint64_t>(value);
        }
        // Built-ins have no bounds guard, so an empty one cannot run as a grid of one.
        if (!invocations)
            return failCall("schedule built-in over an empty TensorView is not linked yet");
        const uint32_t grid[3]{static_cast<uint32_t>(invocations), 1, 1};
        std::memcpy(scratch + operation.gridOffset, grid, sizeof(grid));
    }
    const auto base = count ? reinterpret_cast<uintptr_t>(recording.arena->address()) : uintptr_t{0};
    for (const VernonScheduleField &field : schedule.transientFields) {
        const Transient &transient = schedule.transients[field.slot];
        int64_t value = 0;
        switch (field.kind) {
        case VERNON_SCHEDULE_FIELD_ADDRESS:
            value = static_cast<int64_t>(base + recording.offsets[field.slot]);
            break;
        case VERNON_SCHEDULE_FIELD_EXTENT:
            value = extentValue(schedule, record, transient.extents[field.dimension]);
            break;
        case VERNON_SCHEDULE_FIELD_STRIDE:
            // Row-major in elements.
            value = 1;
            for (size_t dimension = field.dimension + 1; dimension < transient.extents.size(); ++dimension)
                value *= extentValue(schedule, record, transient.extents[dimension]);
            break;
        case VERNON_SCHEDULE_FIELD_CAPACITY:
            value = static_cast<int64_t>(recording.classes[field.slot] / transient.elementSize);
            break;
        case kTransientFieldBytes:
            value = static_cast<int64_t>(elements[field.slot] * transient.elementSize);
            break;
        default:
            break;
        }
        if (!writeField(scratch, field, value))
            return failCall("schedule allocation metadata exceeds its field");
    }
    for (const ConstantAddress &address : schedule.constantAddresses) {
        const uint64_t value = reinterpret_cast<uintptr_t>(recording.constants->address()) + address.constant;
        std::memcpy(scratch + address.recordOffset, &value, sizeof(value));
    }
    auto submitted = device.submit(*recording.commands, scratch, size);
    if (submitted.isErr())
        return failCall(std::move(submitted).error());
    recording.lastPoint = submitted.value();
    return Called{ok(submitted.value())};
}

} // namespace vernon::schedule
