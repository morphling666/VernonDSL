#include "runtime/schedule/fragment.h"

#include "runtime/autodiff/tape_allocator_abi.h"
#include "runtime/pipeline_metadata.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace vernon::schedule {
namespace {

using Built = Result<Fragment, std::string>;
using Checked = Result<void, std::string>;

// How many unslotted Storages a Storage extent may resolve through; a Program's descriptors never cycle, so this only
// bounds malformed input.
constexpr unsigned kMaximumExtentDepth = 8;

Built failBuild(std::string message) { return Built{err(std::move(message))}; }
Checked failCheck(std::string message) { return Checked{err(std::move(message))}; }

std::string elementType(const VernonScheduleSlot &slot) {
    return std::to_string(slot.dtype) + "/" + std::to_string(slot.element_size);
}

uint64_t scalarBytes(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
    case VERNON_DATA_U8:
        return 1;
    case VERNON_DATA_F16:
        return 2;
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_F32:
        return 4;
    case VERNON_DATA_F64:
        return 8;
    }
    return 0;
}

bool inRecord(uint64_t offset, uint64_t size, uint64_t recordSize) {
    return offset <= recordSize && size <= recordSize - offset;
}

uint32_t parseAccess(const std::string &access) {
    if (access == "read")
        return VERNON_SCHEDULE_ACCESS_READ;
    if (access == "write")
        return VERNON_SCHEDULE_ACCESS_WRITE;
    if (access == "read_write")
        return VERNON_SCHEDULE_ACCESS_READ_WRITE;
    return 0;
}

uint32_t fieldKind(runtime::MetadataFieldKind kind) {
    switch (kind) {
    case runtime::MetadataFieldKind::Offset:
        return VERNON_SCHEDULE_FIELD_OFFSET;
    case runtime::MetadataFieldKind::Extent:
        return VERNON_SCHEDULE_FIELD_EXTENT;
    case runtime::MetadataFieldKind::Stride:
        return VERNON_SCHEDULE_FIELD_STRIDE;
    }
    return 0;
}

} // namespace

namespace {

struct Range {
    uint64_t begin;
    uint64_t end;
};

// Counts a TensorView's metadata fields by kind: the offset, extents and strides by dimension, addresses, capacity
// and a fill's byte count. Every binding has one offset and one extent and stride per dimension, so the offsets count
// the bindings.
struct MetadataCounts {
    explicit MetadataCounts(uint32_t rank) : rank(rank), counts(5 + 2 * size_t{rank}, 0) {}

    bool add(const VernonScheduleField &field) {
        size_t counter = 0;
        if (field.kind == VERNON_SCHEDULE_FIELD_OFFSET && field.dimension == 0)
            counter = 0;
        else if (field.kind == VERNON_SCHEDULE_FIELD_EXTENT && field.dimension < rank)
            counter = 1 + field.dimension;
        else if (field.kind == VERNON_SCHEDULE_FIELD_STRIDE && field.dimension < rank)
            counter = 1 + rank + field.dimension;
        else if (field.kind == VERNON_SCHEDULE_FIELD_ADDRESS && field.dimension == 0 && field.size == 8)
            counter = 1 + 2 * size_t{rank};
        else if (field.kind == VERNON_SCHEDULE_FIELD_CAPACITY && field.dimension == 0)
            counter = 2 + 2 * size_t{rank};
        else if ((field.kind == kTransientFieldBytes || field.kind == VERNON_SCHEDULE_FIELD_BYTES) &&
                 field.dimension == 0 && field.size == 8)
            counter = 3 + 2 * size_t{rank};
        else
            return false;
        ++counts[counter];
        return true;
    }

    uint32_t bindings() const { return counts[0]; }
    uint32_t addresses() const { return counts[1 + 2 * size_t{rank}]; }
    uint32_t capacities() const { return counts[2 + 2 * size_t{rank}]; }
    uint32_t fills() const { return counts[3 + 2 * size_t{rank}]; }
    bool perBinding() const {
        return std::all_of(counts.begin() + 1, counts.begin() + 1 + 2 * size_t{rank},
                           [&](uint32_t value) { return value == counts[0]; });
    }

    uint32_t rank;
    std::vector<uint32_t> counts;
};

Checked checkExtents(const std::vector<Extent> &extents, const std::vector<VernonScheduleSlot> &slots) {
    for (const Extent &extent : extents) {
        if (extent.slot == kNoSlot) {
            if (extent.value < 0)
                return failCheck("fragment extent is negative");
        } else if (extent.slot >= slots.size() || (slots[extent.slot].kind == VERNON_SCHEDULE_SLOT_BUFFER
                                                       ? extent.dimension >= slots[extent.slot].rank
                                                       : slots[extent.slot].kind != VERNON_SCHEDULE_SLOT_ARGUMENT)) {
            // An ARGUMENT slot's rank is the call's argument's, which the call checks.
            return failCheck("fragment extent names no TensorView dimension");
        }
    }
    return Checked{ok()};
}

} // namespace

Result<void, std::string> Fragment::validate() const {
    if (shapes.size() != slots.size() || extents.size() != slots.size())
        return failCheck("fragment shapes or extents do not match its slots");
    if (recordTemplate.size() != recordSize)
        return failCheck("fragment record template does not cover its record");
    std::vector<Range> ranges;
    for (size_t index = 0; index < slots.size(); ++index) {
        const VernonScheduleSlot &slot = slots[index];
        if (!inRecord(slot.offset, slot.size, recordSize) || !slot.size)
            return failCheck("fragment slot lies outside its record");
        if (slot.kind == VERNON_SCHEDULE_SLOT_BUFFER) {
            if (slot.size != sizeof(void *) || slot.rank != shapes[index].size() ||
                slot.rank != extents[index].size() || !slot.element_size || slot.access < VERNON_SCHEDULE_ACCESS_READ ||
                slot.access > VERNON_SCHEDULE_ACCESS_READ_WRITE)
                return failCheck("fragment buffer slot is malformed");
        } else if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
            if (slot.rank || slot.access || !shapes[index].empty() || !extents[index].empty() ||
                slot.element_size != slot.size)
                return failCheck("fragment scalar slot is malformed");
        } else if (slot.kind == VERNON_SCHEDULE_SLOT_TAPE) {
            if (slot.size != sizeof(void *) || slot.rank || !shapes[index].empty() || !extents[index].empty() ||
                (slot.access != VERNON_SCHEDULE_ACCESS_READ && slot.access != VERNON_SCHEDULE_ACCESS_WRITE))
                return failCheck("fragment tape slot is malformed");
        } else if (graphicsSlotSize(slot.kind)) {
            // An ARGUMENT slot of a residual describes the shape the residual block checks and sizes.
            const bool shaped = slot.kind == VERNON_SCHEDULE_SLOT_ARGUMENT;
            if (slot.size != graphicsSlotSize(slot.kind) || (!shaped && slot.rank) ||
                slot.rank != shapes[index].size() || slot.rank != extents[index].size() ||
                slot.access < VERNON_SCHEDULE_ACCESS_READ || slot.access > VERNON_SCHEDULE_ACCESS_READ_WRITE)
                return failCheck("fragment graphics slot is malformed");
        } else {
            return failCheck("fragment slot has an unknown kind");
        }
        if (auto checked = checkExtents(extents[index], slots); checked.isErr())
            return checked;
        ranges.push_back({slot.offset, slot.offset + slot.size});
    }
    std::vector<MetadataCounts> counts;
    counts.reserve(slots.size());
    for (const VernonScheduleSlot &slot : slots)
        counts.emplace_back(slot.rank);
    for (const VernonScheduleField &field : fields) {
        if (field.slot >= slots.size())
            return failCheck("fragment field names no slot");
        if (!inRecord(field.offset, field.size, recordSize))
            return failCheck("fragment field lies outside its record");
        ranges.push_back({field.offset, field.offset + field.size});
        const VernonScheduleSlot &slot = slots[field.slot];
        if (slot.kind == VERNON_SCHEDULE_SLOT_TAPE)
            return failCheck("fragment field repeats a tape slot");
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR || graphicsSlotSize(slot.kind)) {
            if (field.kind != VERNON_SCHEDULE_FIELD_VALUE || field.size != slot.size || field.dimension)
                return failCheck("fragment scalar or graphics field is not a repeated value");
            continue;
        }
        if ((field.size != 4 && field.size != 8) || field.kind == kTransientFieldBytes ||
            !counts[field.slot].add(field))
            return failCheck("fragment field has an invalid kind, size or dimension");
    }
    for (size_t index = 0; index < slots.size(); ++index) {
        if (slots[index].kind != VERNON_SCHEDULE_SLOT_BUFFER)
            continue;
        // The slot's own offset is a binding's first address; a CUDA memref repeats it in every binding, and a Struct
        // view's binding has one memref, each with its capacity, per storage leaf.
        const MetadataCounts &count = counts[index];
        const uint32_t descriptors = std::max(count.bindings(), count.capacities());
        if (!count.bindings() || !count.perBinding() || count.addresses() > 2 * descriptors - 1 + count.fills() ||
            count.fills() > 1)
            return failCheck("fragment TensorView metadata is incomplete or duplicated");
    }
    for (const ConstantAddress &address : constantAddresses) {
        if (!inRecord(address.recordOffset, sizeof(uint64_t), recordSize) || address.constant >= constants.size())
            return failCheck("fragment constant address lies outside its record or constants");
        ranges.push_back({address.recordOffset, address.recordOffset + sizeof(uint64_t)});
    }
    std::vector<MetadataCounts> transientCounts;
    transientCounts.reserve(transients.size());
    for (const Transient &transient : transients) {
        if (!transient.elementSize || transient.first > transient.last || transient.last >= operations.size())
            return failCheck("fragment transient is malformed");
        if (auto checked = checkExtents(transient.extents, slots); checked.isErr())
            return checked;
        transientCounts.emplace_back(static_cast<uint32_t>(transient.extents.size()));
    }
    for (const VernonScheduleField &field : transientFields) {
        if (field.slot >= transients.size() || !inRecord(field.offset, field.size, recordSize) ||
            (field.size != 4 && field.size != 8) || !transientCounts[field.slot].add(field))
            return failCheck("fragment transient field is malformed");
        ranges.push_back({field.offset, field.offset + field.size});
    }
    for (size_t index = 0; index < transients.size(); ++index) {
        const MetadataCounts &count = transientCounts[index];
        if (!count.perBinding() || count.addresses() < count.bindings() + count.fills() ||
            count.fills() != (transients[index].zeroed ? 1U : 0U))
            return failCheck("fragment transient metadata is incomplete");
    }
    std::vector<uint64_t> scalarWords;
    for (const VernonScheduleSlot &slot : slots)
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR && slot.size == sizeof(uint32_t))
            scalarWords.push_back(slot.offset);
    for (const VernonScheduleField &field : fields)
        if (slots[field.slot].kind == VERNON_SCHEDULE_SLOT_SCALAR && field.size == sizeof(uint32_t))
            scalarWords.push_back(field.offset);
    std::sort(scalarWords.begin(), scalarWords.end());
    // Each offset of a pass is the offset of a slot of its kind, or a field repeating one.
    std::vector<std::pair<uint64_t, uint32_t>> graphicsSites;
    for (const VernonScheduleSlot &slot : slots)
        if (graphicsSlotSize(slot.kind))
            graphicsSites.emplace_back(slot.offset, slot.kind);
    for (const VernonScheduleField &field : fields)
        if (graphicsSlotSize(slots[field.slot].kind))
            graphicsSites.emplace_back(field.offset, slots[field.slot].kind);
    std::sort(graphicsSites.begin(), graphicsSites.end());
    const auto graphicsSite = [&](uint64_t offset, uint32_t kind) {
        return std::binary_search(graphicsSites.begin(), graphicsSites.end(), std::make_pair(offset, kind));
    };
    // A parameter reads an ARGUMENT slot or its pass's tape field, an internal buffer and no record bytes, or an
    // attachment of a PASS slot.
    const auto parametersBound = [&](const Dispatch &dispatch) {
        const hal::PassOffsets &pass = dispatch.pass;
        const auto tapeField = [&](uint64_t offset) {
            return std::any_of(dispatch.tape.fields.begin(), dispatch.tape.fields.end(),
                               [&](const DeviceTapeField &field) {
                                   return field.field.kind == kTapeFieldArgument && field.field.offset == offset;
                               });
        };
        if ((!pass.internals.empty() && pass.internals.size() != pass.parameters.size()) ||
            (!pass.internalExtents.empty() && pass.internalExtents.size() != pass.parameters.size()) ||
            (!pass.attachments.empty() && pass.attachments.size() != pass.parameters.size()) ||
            (!pass.zeroed.empty() && pass.zeroed.size() != pass.parameters.size()) ||
            (!pass.launches.empty() && pass.launches.size() != pass.parameters.size()))
            return false;
        for (size_t parameter = 0; parameter < pass.parameters.size(); ++parameter) {
            const hal::InternalBuffer *internal = pass.internals.empty() ? nullptr : pass.internals[parameter].get();
            const uint64_t attachment = pass.attachments.empty() ? UINT64_MAX : pass.attachments[parameter].renderPass;
            if (internal && attachment != UINT64_MAX)
                return false;
            if (!pass.launches.empty() && pass.launches[parameter]) {
                const hal::ComputeGrid &launch = *pass.launches[parameter];
                if (internal || attachment != UINT64_MAX || pass.parameters[parameter] != UINT64_MAX ||
                    launch.linearized || (!pass.zeroed.empty() && pass.zeroed[parameter]))
                    return false;
                for (size_t axis = 0; axis < 3; ++axis)
                    if (launch.argument[axis] != UINT64_MAX &&
                        (!graphicsSite(launch.argument[axis], VERNON_SCHEDULE_SLOT_ARGUMENT) ||
                         (launch.scalarBytes[axis] != 4 && launch.scalarBytes[axis] != 8)))
                        return false;
                continue;
            }
            // A zeroed parameter reads its buffer from an ARGUMENT slot.
            if (!pass.zeroed.empty() && pass.zeroed[parameter] && (internal || attachment != UINT64_MAX))
                return false;
            if (!pass.internalExtents.empty() && !pass.internalExtents[parameter].empty()) {
                const std::vector<hal::InternalExtent> &extents = pass.internalExtents[parameter];
                if (!internal || extents.size() != internal->shape.size() ||
                    std::any_of(extents.begin(), extents.end(), [&](const hal::InternalExtent &extent) {
                        return extent.argument != UINT64_MAX &&
                               !graphicsSite(extent.argument, VERNON_SCHEDULE_SLOT_ARGUMENT);
                    }))
                    return false;
            }
            if (internal ? pass.parameters[parameter] != UINT64_MAX || !internal->bytes ||
                               internal->shape.size() != internal->strides.size()
                : attachment != UINT64_MAX
                    ? pass.parameters[parameter] != UINT64_MAX || !graphicsSite(attachment, VERNON_SCHEDULE_SLOT_PASS)
                    : !graphicsSite(pass.parameters[parameter], VERNON_SCHEDULE_SLOT_ARGUMENT) &&
                          !tapeField(pass.parameters[parameter]))
                return false;
        }
        return true;
    };
    for (const Dispatch &dispatch : operations) {
        for (const ResultUse &result : dispatch.results) {
            if (result.slot != kNoSlot &&
                (result.slot >= slots.size() || slots[result.slot].kind != VERNON_SCHEDULE_SLOT_BUFFER))
                return failCheck("fragment result commits to no buffer slot");
            for (const runtime::ValueByteCopy &copy : result.copies)
                if (copy.physicalOffset > dispatch.resultsSize ||
                    copy.size > dispatch.resultsSize - copy.physicalOffset)
                    return failCheck("fragment result copies outside its results frame");
        }
        if (dispatch.kind == OperationKind::Pass && dispatch.pass.compute) {
            const hal::PassOffsets &pass = dispatch.pass;
            const auto argument = [&](uint64_t offset) { return graphicsSite(offset, VERNON_SCHEDULE_SLOT_ARGUMENT); };
            if (dispatch.executable >= executables.size() || !executables[dispatch.executable].graphics ||
                pass.renderPass != UINT64_MAX || !parametersBound(dispatch))
                return failCheck("fragment compute pass is malformed");
            for (size_t axis = 0; axis < 3; ++axis)
                if (pass.grid.argument[axis] != UINT64_MAX &&
                    (!argument(pass.grid.argument[axis]) ||
                     (pass.grid.scalarBytes[axis] != 4 && pass.grid.scalarBytes[axis] != 8)))
                    return failCheck("fragment compute pass reads its grid from no scalar argument");
        } else if (dispatch.kind == OperationKind::Pass) {
            const hal::PassOffsets &pass = dispatch.pass;
            if (dispatch.executable >= executables.size() || !executables[dispatch.executable].graphics ||
                !graphicsSite(pass.renderPass, VERNON_SCHEDULE_SLOT_PASS) ||
                !graphicsSite(pass.draw, VERNON_SCHEDULE_SLOT_DRAW) ||
                !graphicsSite(pass.dynamic, VERNON_SCHEDULE_SLOT_DYNAMIC) ||
                !graphicsSite(pass.stream, VERNON_SCHEDULE_SLOT_STREAM) ||
                !graphicsSite(pass.table, VERNON_SCHEDULE_SLOT_TABLE) || !parametersBound(dispatch))
                return failCheck("fragment pass is malformed");
        } else if (dispatch.kind == OperationKind::Fill) {
            if (dispatch.argumentsSize != 2 * sizeof(uint64_t) ||
                !inRecord(dispatch.argumentsOffset, dispatch.argumentsSize, recordSize) ||
                dispatch.accesses.size() != 1 || dispatch.accesses[0].access != VERNON_SCHEDULE_ACCESS_WRITE)
                return failCheck("fragment fill is malformed");
        } else if (dispatch.kind == OperationKind::Copy && !dispatch.pass.parameters.empty()) {
            // A graphics device's: the Storage's ARGUMENT slot, then the carrier's internal buffer.
            const hal::LeafCopy &copy = dispatch.copy;
            const hal::PassOffsets &pass = dispatch.pass;
            if (pass.parameters.size() != 2 || pass.compute || pass.renderPass != UINT64_MAX ||
                pass.internals.size() != 2 || pass.internals[0] || !pass.internals[1] ||
                !graphicsSite(pass.parameters[0], VERNON_SCHEDULE_SLOT_ARGUMENT) || !parametersBound(dispatch) ||
                dispatch.argumentsSize || dispatch.accesses.size() != 1 || dispatch.accesses[0].transient ||
                dispatch.accesses[0].access !=
                    (copy.scatter ? VERNON_SCHEDULE_ACCESS_WRITE : VERNON_SCHEDULE_ACCESS_READ) ||
                !copy.bytes || copy.storageOffset > copy.storageElement ||
                copy.bytes > copy.storageElement - copy.storageOffset || copy.carrierOffset > copy.carrierElement ||
                copy.bytes > copy.carrierElement - copy.carrierOffset)
                return failCheck("fragment graphics leaf copy is malformed");
        } else if (dispatch.kind == OperationKind::Copy) {
            const hal::LeafCopy &copy = dispatch.copy;
            const uint32_t storage = copy.scatter ? VERNON_SCHEDULE_ACCESS_WRITE : VERNON_SCHEDULE_ACCESS_READ;
            const uint32_t carrier = copy.scatter ? VERNON_SCHEDULE_ACCESS_READ : VERNON_SCHEDULE_ACCESS_WRITE;
            if (dispatch.argumentsSize != copy.argumentsSize() ||
                !inRecord(dispatch.argumentsOffset, dispatch.argumentsSize, recordSize) ||
                dispatch.accesses.size() != 2 || dispatch.accesses[0].access != storage ||
                !dispatch.accesses[1].transient || dispatch.accesses[1].access != carrier || !copy.bytes ||
                copy.storageOffset > copy.storageElement || copy.bytes > copy.storageElement - copy.storageOffset ||
                copy.carrierOffset > copy.carrierElement || copy.bytes > copy.carrierElement - copy.carrierOffset)
                return failCheck("fragment leaf copy is malformed");
        } else if (dispatch.executable >= executables.size() ||
                   !inRecord(dispatch.argumentsOffset, dispatch.argumentsSize, recordSize) ||
                   !inRecord(dispatch.gridOffset, 3 * sizeof(uint32_t), recordSize)) {
            return failCheck("fragment dispatch lies outside its record");
        } else {
            // A grid axis the caller passes as a Value is a scalar slot or field at its word, already a range.
            for (uint64_t axis = 0; axis < 3; ++axis) {
                const uint64_t word = dispatch.gridOffset + axis * sizeof(uint32_t);
                if (!std::binary_search(scalarWords.begin(), scalarWords.end(), word))
                    ranges.push_back({word, word + sizeof(uint32_t)});
            }
            if (auto checked = checkExtents(dispatch.gridElements, slots); checked.isErr())
                return checked;
        }
        // A record address lies in the dispatch's arguments and points at a scalar slot's bytes or a repeat of them.
        for (const hal::RecordAddress &address : dispatch.recordAddresses) {
            // A by-pointer Value's bytes start at a scalar; a kernel's launch metadata holds one in its three counts.
            const auto holdsScalar = [&](uint64_t offset) {
                return offset >= address.target && offset - address.target < 3 * sizeof(uint32_t);
            };
            const bool pointsAtScalar =
                std::any_of(slots.begin(), slots.end(),
                            [&](const VernonScheduleSlot &slot) {
                                return slot.kind == VERNON_SCHEDULE_SLOT_SCALAR && holdsScalar(slot.offset);
                            }) ||
                std::any_of(fields.begin(), fields.end(), [&](const VernonScheduleField &field) {
                    return field.kind == VERNON_SCHEDULE_FIELD_VALUE && holdsScalar(field.offset);
                });
            if (dispatch.kind != OperationKind::Dispatch || !pointsAtScalar || address.at < dispatch.argumentsOffset ||
                address.at - dispatch.argumentsOffset > dispatch.argumentsSize ||
                sizeof(uint64_t) > dispatch.argumentsSize - (address.at - dispatch.argumentsOffset))
                return failCheck("fragment record address is malformed");
        }
        for (const InvocationExtent &extent : dispatch.invocationExtents)
            if (dispatch.kind != OperationKind::Dispatch || !extent.workgroupVolume ||
                (extent.size != sizeof(uint32_t) && extent.size != sizeof(uint64_t)) ||
                extent.offset < dispatch.argumentsOffset ||
                extent.offset - dispatch.argumentsOffset > dispatch.argumentsSize ||
                extent.size > dispatch.argumentsSize - (extent.offset - dispatch.argumentsOffset))
                return failCheck("fragment invocation extent is malformed");
        if (dispatch.tape.present()) {
            const TapeUse &tape = dispatch.tape;
            const auto inArguments = [&](uint64_t offset, uint64_t bytes) {
                return offset >= dispatch.argumentsOffset &&
                       offset - dispatch.argumentsOffset <= dispatch.argumentsSize &&
                       bytes <= dispatch.argumentsSize - (offset - dispatch.argumentsOffset);
            };
            const bool pass = dispatch.kind == OperationKind::Pass;
            const bool fields = std::all_of(tape.fields.begin(), tape.fields.end(), [&](const DeviceTapeField &field) {
                if (pass)
                    return field.carrier != DeviceTapeCarrier::None && field.field.kind == kTapeFieldArgument &&
                           field.field.size == sizeof(VernonProgramArgument) &&
                           inRecord(field.field.offset, field.field.size, recordSize) &&
                           std::count(dispatch.pass.parameters.begin(), dispatch.pass.parameters.end(),
                                      field.field.offset) == 1;
                return field.carrier != DeviceTapeCarrier::None && inArguments(field.field.offset, field.field.size) &&
                       (field.field.size == sizeof(uint32_t) || field.field.size == sizeof(uint64_t));
            });
            if ((dispatch.kind != OperationKind::Dispatch &&
                 (!pass || !dispatch.pass.compute || !tape.device || dispatch.pass.grid.linearized)) ||
                (tape.device ? !fields || tape.allocatorOffset != UINT64_MAX || !tape.workgroup[0] ||
                                   !tape.workgroup[1] || !tape.workgroup[2]
                             : !inArguments(tape.allocatorOffset, sizeof(void *)) || !tape.fields.empty()) ||
                (tape.rootOffset != UINT64_MAX && !inArguments(tape.rootOffset, sizeof(VernonAdRegionHandle))) ||
                (tape.slot != kNoSlot &&
                 (tape.slot >= slots.size() || slots[tape.slot].kind != VERNON_SCHEDULE_SLOT_TAPE || !tape.stride ||
                  slots[tape.slot].access !=
                      (tape.writes ? VERNON_SCHEDULE_ACCESS_WRITE : VERNON_SCHEDULE_ACCESS_READ))))
                return failCheck("fragment dispatch's autodiff tape is malformed");
        }
        for (const BufferAccess &access : dispatch.accesses) {
            if (access.transient ? access.index >= transients.size()
                                 : access.index >= slots.size() ||
                                       (slots[access.index].kind != VERNON_SCHEDULE_SLOT_BUFFER &&
                                        slots[access.index].kind != VERNON_SCHEDULE_SLOT_ARGUMENT &&
                                        slots[access.index].kind != VERNON_SCHEDULE_SLOT_PASS) ||
                                       (access.access & ~slots[access.index].access))
                return failCheck("fragment operation " + std::to_string(&dispatch - operations.data()) + " (kind " +
                                 std::to_string(static_cast<int>(dispatch.kind)) + ") accesses " +
                                 (access.transient ? "transient " : "slot ") + std::to_string(access.index) + " as " +
                                 std::to_string(access.access) + ", which it may not");
            if (access.access < VERNON_SCHEDULE_ACCESS_READ || access.access > VERNON_SCHEDULE_ACCESS_READ_WRITE)
                return failCheck("fragment operation access is invalid");
        }
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const Range &left, const Range &right) { return left.begin < right.begin; });
    for (size_t index = 1; index < ranges.size(); ++index)
        if (ranges[index].begin < ranges[index - 1].end)
            return failCheck("fragment record ranges overlap");
    return Checked{ok()};
}

namespace {

// What both backends read from a kernel's reflection: its slots, in argument order, at their CPU frame offsets.
struct ReflectedKernel {
    runtime::ReflectedEntry parsed;
    Fragment fragment;
    Dispatch dispatch;
    std::vector<uint32_t> slotOfArgument;
    std::vector<uint32_t> bindingOfSlot;
    std::vector<uint64_t> alignmentOfSlot;
    // Per slot, the element size of a scalar slot CUDA takes by pointer, or zero.
    std::vector<uint64_t> byPointerElement;
    // Per slot, the bindings it takes: one per storage leaf of a TensorView, else one.
    std::vector<uint32_t> leavesOfSlot;
};

Result<ReflectedKernel, std::string> reflectKernel(runtime::ReflectedEntry reflected, const std::string &entry,
                                                   VernonRuntimeBackend backend) {
    using Reflected = Result<ReflectedKernel, std::string>;
    const auto fail = [](std::string message) { return Reflected{err(std::move(message))}; };
    ReflectedKernel kernel;
    kernel.parsed = std::move(reflected);
    const runtime::ReflectedEntry &parsed = kernel.parsed;
    if (parsed.packedResults && parsed.packedResults->size && backend != VERNON_RUNTIME_CPU)
        return fail("GPU kernels with results are not linked yet");
    if (parsed.dispatchContract.requiresUnitWorkgroup &&
        (parsed.workgroup[0] != 1 || parsed.workgroup[1] != 1 || parsed.workgroup[2] != 1))
        return fail("kernel requires a unit workgroup it does not have");

    Fragment &fragment = kernel.fragment;
    // The Program runtime recognizes built-ins by entry name too (target_binding_plan.cpp).
    fragment.elementGrid =
        entry.find("_program_copy_") != std::string::npos || entry.find("_program_add_") != std::string::npos;
    Dispatch &dispatch = kernel.dispatch;
    std::copy_n(parsed.dispatchContract.unitGridAxes, 3, dispatch.unitGridAxes);
    std::vector<uint32_t> &slotOfArgument = kernel.slotOfArgument;
    uint32_t argumentEnd = 0;
    for (const runtime::ReflectedArgument &argument : parsed.arguments)
        if (argument.index != UINT32_MAX && !argument.result)
            argumentEnd = std::max(argumentEnd, argument.index + 1);
    slotOfArgument.assign(argumentEnd, UINT32_MAX);
    for (const runtime::ReflectedArgument &argument : parsed.arguments) {
        if (argument.result) {
            if (argument.resultCopies.empty() || !parsed.packedResults ||
                argument.physical.offset > parsed.packedResults->size ||
                argument.physical.size > parsed.packedResults->size - argument.physical.offset)
                return fail("kernel result '" + argument.sourceName + "' has no layout a schedule copies");
            ResultUse result{argument.index, kNoSlot, argument.physical.offset, argument.physical.size};
            result.copies = argument.resultCopies;
            for (runtime::ValueByteCopy &copy : result.copies)
                copy.physicalOffset += argument.physical.offset;
            // As the Stage runtime does, a gradient result sums the lanes (prepared_binding_plan.cpp).
            if (argument.autodiffRole == "gradient") {
                if (argument.dtype != VERNON_DATA_F32 && argument.dtype != VERNON_DATA_F64)
                    return fail("kernel gradient result '" + argument.sourceName + "' is not f32 or f64");
                result.reduction = argument.dtype;
            }
            dispatch.results.push_back(std::move(result));
            dispatch.resultsSize = parsed.packedResults->size;
            continue;
        }
        if (argument.kind == "builtin") {
            const bool allocator = argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN;
            if (!allocator && argument.builtin != VERNON_AD_TAPE_ROOT_REGION_BUILTIN)
                continue;
            if (backend != VERNON_RUNTIME_CPU)
                return fail("GPU kernels with autodiff tapes are not linked yet");
            if (argument.physical.size != (allocator ? sizeof(void *) : sizeof(VernonAdRegionHandle)))
                return fail("kernel autodiff tape builtin has the wrong size");
            (allocator ? dispatch.tape.allocatorOffset : dispatch.tape.rootOffset) = argument.physical.offset;
            continue;
        }
        // A CUDA kernel takes a Struct, Tuple or ranked Tensor Value by pointer (program_execution_backend.cpp).
        const bool byPointer = argument.kind == "buffer value" && backend == VERNON_RUNTIME_CUDA;
        if (argument.kind != "tensor" && argument.kind != "scalar" && !byPointer)
            return fail("kernels with " + argument.kind + " arguments are not linked yet");
        if (!argument.dtype || argument.index >= slotOfArgument.size())
            return fail("kernel argument '" + argument.sourceName + "' has no scalar element type");
        VernonScheduleSlot slot{};
        slot.dtype = static_cast<uint32_t>(*argument.dtype);
        slot.offset = argument.physical.offset;
        slot.size = argument.physical.size;
        uint64_t alignment = std::max<uint64_t>(argument.physical.alignment, 1);
        const auto slotIndex = static_cast<uint32_t>(fragment.slots.size());
        // Every leaf of a TensorView indexes from the view's buffer start in its own scalars, so the leaves share
        // the view's address and their metadata is the view's, in whole elements (VernonAggregateStorage.cpp).
        uint32_t leafCount = 1;
        if (argument.kind == "tensor") {
            if (!argument.tensorElementSize)
                return fail("kernel TensorView '" + argument.sourceName + "' has no element size");
            leafCount = static_cast<uint32_t>(std::max<size_t>(argument.storageLeaves.size(), 1));
            slot.kind = VERNON_SCHEDULE_SLOT_BUFFER;
            slot.size = sizeof(void *);
            alignment = alignof(void *);
            slot.rank = static_cast<uint32_t>(argument.sourceShape.size());
            slot.element_size = argument.tensorElementSize;
            slot.access = parseAccess(argument.access);
            if (!slot.access)
                return fail("kernel TensorView '" + argument.sourceName + "' has no access");
            dispatch.accesses.push_back({false, slotIndex, slot.access});
            fragment.shapes.push_back(argument.sourceShape);
            std::vector<Extent> extents;
            for (size_t dimension = 0; dimension < argument.sourceShape.size(); ++dimension) {
                const int64_t declared = argument.sourceShape[dimension];
                extents.push_back(declared >= 0 ? Extent{declared, kNoSlot, 0}
                                                : Extent{0, slotIndex, static_cast<uint32_t>(dimension)});
            }
            fragment.extents.push_back(std::move(extents));
        } else {
            if (!argument.canonicalBytes)
                return fail("value arguments in a non-canonical physical layout are not linked yet");
            slot.kind = VERNON_SCHEDULE_SLOT_SCALAR;
            slot.element_size = slot.size;
            fragment.shapes.emplace_back();
            fragment.extents.emplace_back();
        }
        fragment.slotArguments.push_back(argument.index);
        fragment.slotLaunchMetadata.push_back(argument.autodiffRole == "launch_metadata");
        const DeviceTapeCarrier tapeCarrier = argument.autodiffRole == "tape"             ? DeviceTapeCarrier::Data
                                              : argument.autodiffRole == "replay_segment" ? DeviceTapeCarrier::Segments
                                              : argument.autodiffRole == "replay_status"  ? DeviceTapeCarrier::Status
                                                                                          : DeviceTapeCarrier::None;
        if (tapeCarrier != DeviceTapeCarrier::None) {
            if (backend == VERNON_RUNTIME_CPU || argument.kind != "tensor" || slot.rank != 1 ||
                argument.dtype != VERNON_DATA_I32)
                return fail("kernel autodiff tape carrier '" + argument.sourceName + "' is not a GPU i32 TensorView");
            dispatch.tape.device = true;
            std::copy_n(parsed.workgroup, 3, dispatch.tape.workgroup);
        }
        fragment.slotTapeCarrier.push_back(tapeCarrier);
        fragment.slotLeafOffset.push_back(argument.storageLeaves.empty() ? 0
                                                                         : argument.storageLeaves.front().byteOffset);
        const std::string &carrier = argument.autodiffCarrier;
        if (!carrier.empty() && (carrier != "invocation_linear" || argument.kind != "tensor" || slot.rank < 1))
            return fail("kernel argument '" + argument.sourceName + "' has an autodiff carrier that is not linked yet");
        fragment.slotInvocationCarrier.push_back(!carrier.empty());
        if (byPointer && (!argument.tensorElementSize || slot.size % argument.tensorElementSize))
            return fail("kernel Value '" + argument.sourceName + "' is not a whole number of its elements");
        kernel.byPointerElement.push_back(byPointer ? argument.tensorElementSize : 0);
        kernel.leavesOfSlot.push_back(leafCount);
        slotOfArgument[argument.index] = slotIndex;
        fragment.slots.push_back(slot);
        kernel.bindingOfSlot.push_back(argument.binding);
        kernel.alignmentOfSlot.push_back(alignment);
    }
    return Reflected{ok(std::move(kernel))};
}

// Places the carrier's members as fields of the TensorViews they describe, from `base` in the record.
Result<void, std::string> addMetadataFields(ReflectedKernel &kernel, const runtime::MetadataCarrier &carrier,
                                            uint64_t base) {
    for (const runtime::PhysicalMetadataMember &member : carrier.members) {
        if (member.semanticOrdinal >= carrier.fields.size())
            return failCheck("kernel metadata member names no field");
        const runtime::MetadataFieldIdentity &identity = carrier.fields[member.semanticOrdinal];
        if (identity.argument >= kernel.slotOfArgument.size() || kernel.slotOfArgument[identity.argument] == UINT32_MAX)
            return failCheck("kernel metadata field names no TensorView");
        VernonScheduleField field{};
        field.slot = kernel.slotOfArgument[identity.argument];
        field.kind = fieldKind(identity.kind);
        field.dimension = identity.dimension.value_or(0);
        field.size = static_cast<uint32_t>(member.byteSize);
        field.offset = base + member.byteOffset;
        kernel.fragment.fields.push_back(field);
    }
    return Checked{ok()};
}

uint64_t alignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

// Puts the grid after the arguments and completes the fragment.
Built finish(ReflectedKernel &kernel, hal::ExecutableSource source) {
    Fragment &fragment = kernel.fragment;
    Dispatch &dispatch = kernel.dispatch;
    std::copy_n(kernel.parsed.workgroup, 3, source.workgroup);
    fragment.executables.push_back(std::move(source));
    dispatch.gridOffset = alignUp(dispatch.argumentsSize, alignof(uint32_t));
    fragment.recordSize = dispatch.gridOffset + 3 * sizeof(uint32_t);
    fragment.recordTemplate.resize(fragment.recordSize, 0);
    fragment.callerGrid = true;
    fragment.operations.push_back(std::move(dispatch));
    if (auto valid = fragment.validate(); valid.isErr())
        return failBuild(std::move(valid).error());
    return Built{ok(std::move(fragment))};
}

} // namespace

Result<Fragment, std::string> cpuKernelFragment(const nlohmann::json &reflection, const std::string &entry,
                                                VernonCpuEntryPoint cpuEntry, VernonCpuChunkEntryPoint cpuChunkEntry) {
    runtime::ReflectedEntry parsed;
    std::string error;
    if (!runtime::parseReflection(reflection, entry, parsed, VERNON_RUNTIME_CPU, error))
        return failBuild(std::move(error));
    return cpuKernelFragment(std::move(parsed), entry, cpuEntry, cpuChunkEntry);
}

Result<Fragment, std::string> cpuKernelFragment(runtime::ReflectedEntry reflection, const std::string &entry,
                                                VernonCpuEntryPoint cpuEntry, VernonCpuChunkEntryPoint cpuChunkEntry) {
    auto reflected = reflectKernel(std::move(reflection), entry, VERNON_RUNTIME_CPU);
    if (reflected.isErr())
        return failBuild(std::move(reflected).error());
    ReflectedKernel kernel = std::move(reflected).value();
    if (!kernel.parsed.packedArguments)
        return failBuild("CPU kernel reflection has no packed arguments");
    kernel.dispatch.argumentsSize = kernel.parsed.packedArguments->size;
    if (kernel.parsed.metadataCarrier) {
        const runtime::MetadataCarrier &carrier = *kernel.parsed.metadataCarrier;
        if (carrier.carrier != "cpu_call_frame")
            return failBuild("CPU kernel metadata is not carried in its call frame");
        if (auto added = addMetadataFields(kernel, carrier, carrier.interfacePlan.frameOffset); added.isErr())
            return failBuild(std::move(added).error());
    }
    hal::ExecutableSource source;
    source.cpuEntry = cpuEntry;
    source.cpuChunkEntry = cpuChunkEntry;
    return finish(kernel, std::move(source));
}

Result<Fragment, std::string> cudaKernelFragment(const nlohmann::json &reflection, const std::string &entry,
                                                 std::string ptx) {
    runtime::ReflectedEntry parsed;
    std::string error;
    if (!runtime::parseReflection(reflection, entry, parsed, VERNON_RUNTIME_CUDA, error))
        return failBuild(std::move(error));
    return cudaKernelFragment(std::move(parsed), entry, std::move(ptx));
}

Result<Fragment, std::string> cudaKernelFragment(runtime::ReflectedEntry reflection, const std::string &entry,
                                                 std::string ptx) {
    auto reflected = reflectKernel(std::move(reflection), entry, VERNON_RUNTIME_CUDA);
    if (reflected.isErr())
        return failBuild(std::move(reflected).error());
    ReflectedKernel kernel = std::move(reflected).value();
    Fragment &fragment = kernel.fragment;
    const size_t slotCount = fragment.slots.size();

    // Kernel parameters follow the arguments' bindings, which number them contiguously.
    std::vector<uint32_t> order(slotCount);
    for (uint32_t index = 0; index < slotCount; ++index)
        order[index] = index;
    std::sort(order.begin(), order.end(),
              [&](uint32_t left, uint32_t right) { return kernel.bindingOfSlot[left] < kernel.bindingOfSlot[right]; });
    for (uint32_t index = 0, binding = 0; index < slotCount; binding += kernel.leavesOfSlot[order[index++]])
        if (kernel.bindingOfSlot[order[index]] != binding)
            return failBuild("CUDA kernel bindings are not contiguous");

    hal::ExecutableSource source;
    uint64_t end = 0;
    const auto place = [&](uint64_t size, uint64_t alignment) {
        const uint64_t at = alignUp(end, alignment);
        end = at + size;
        return at;
    };
    struct Constant {
        uint64_t offset;
        uint64_t value;
    };
    std::vector<Constant> constants;
    // The memref descriptor of each by-pointer Value, whose bytes follow every parameter.
    std::vector<std::pair<uint32_t, uint64_t>> byPointer;
    for (uint32_t slotIndex : order) {
        VernonScheduleSlot &slot = fragment.slots[slotIndex];
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR && kernel.byPointerElement[slotIndex]) {
            const uint64_t base = place(5 * sizeof(uint64_t), alignof(uint64_t));
            for (uint64_t member = 0; member < 5; ++member)
                source.parameterOffsets.push_back(base + member * sizeof(uint64_t));
            constants.push_back({base + 2 * sizeof(uint64_t), 0});
            constants.push_back({base + 3 * sizeof(uint64_t), slot.size / kernel.byPointerElement[slotIndex]});
            constants.push_back({base + 4 * sizeof(uint64_t), 1});
            byPointer.emplace_back(slotIndex, base);
            continue;
        }
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
            slot.offset = place(slot.size, kernel.alignmentOfSlot[slotIndex]);
            source.parameterOffsets.push_back(slot.offset);
            continue;
        }
        // A memref descriptor: allocated and aligned addresses, offset, size and stride, each 8 bytes. The view's own
        // offset, extents and strides travel in the metadata, so the descriptor spans the buffer in elements.
        // A Struct view's further leaves repeat the descriptor; the kernels read no descriptor size.
        for (uint32_t leaf = 0; leaf < kernel.leavesOfSlot[slotIndex]; ++leaf) {
            const uint64_t base = place(5 * sizeof(uint64_t), alignof(uint64_t));
            for (uint64_t member = 0; member < 5; ++member)
                source.parameterOffsets.push_back(base + member * sizeof(uint64_t));
            if (leaf == 0)
                slot.offset = base + sizeof(uint64_t);
            else
                fragment.fields.push_back(
                    {slotIndex, VERNON_SCHEDULE_FIELD_ADDRESS, 0, sizeof(uint64_t), base + sizeof(uint64_t)});
            fragment.fields.push_back({slotIndex, VERNON_SCHEDULE_FIELD_ADDRESS, 0, sizeof(uint64_t), base});
            fragment.fields.push_back(
                {slotIndex, VERNON_SCHEDULE_FIELD_CAPACITY, 0, sizeof(uint64_t), base + 3 * sizeof(uint64_t)});
            constants.push_back({base + 2 * sizeof(uint64_t), 0});
            constants.push_back({base + 4 * sizeof(uint64_t), 1});
        }
    }
    if (kernel.parsed.metadataCarrier) {
        const runtime::MetadataCarrier &carrier = *kernel.parsed.metadataCarrier;
        if (carrier.carrier != "kernel_parameter" || carrier.parameterOrdinal != source.parameterOffsets.size() ||
            !carrier.size)
            return failBuild("CUDA kernel metadata is not its last kernel parameter");
        const uint64_t base = place(carrier.size, std::max<uint64_t>(carrier.alignment, 1));
        source.parameterOffsets.push_back(base);
        if (auto added = addMetadataFields(kernel, carrier, base); added.isErr())
            return failBuild(std::move(added).error());
    }
    for (const auto &[slotIndex, base] : byPointer) {
        VernonScheduleSlot &slot = fragment.slots[slotIndex];
        slot.offset = place(slot.size, std::max<uint64_t>(kernel.alignmentOfSlot[slotIndex], 16));
        kernel.dispatch.recordAddresses.push_back({base, slot.offset});
        kernel.dispatch.recordAddresses.push_back({base + sizeof(uint64_t), slot.offset});
    }
    kernel.dispatch.argumentsSize = end;
    source.artifact = std::move(ptx);
    source.entry = entry;
    auto built = finish(kernel, std::move(source));
    if (built.isErr())
        return built;
    Fragment result = std::move(built).value();
    for (const Constant &constant : constants)
        std::memcpy(result.recordTemplate.data() + constant.offset, &constant.value, sizeof(constant.value));
    return Built{ok(std::move(result))};
}

namespace {

const nlohmann::json *member(const nlohmann::json &object, const char *name) {
    if (!object.is_object())
        return nullptr;
    const auto found = object.find(name);
    return found == object.end() ? nullptr : &*found;
}

bool readUint(const nlohmann::json *value, uint32_t &out) {
    if (!value || !value->is_number_unsigned() || value->get<uint64_t>() > UINT32_MAX)
        return false;
    out = static_cast<uint32_t>(value->get<uint64_t>());
    return true;
}

std::string readString(const nlohmann::json *value) {
    return value && value->is_string() ? value->get<std::string>() : std::string();
}

// A node's argument, resolved: the slot or transient it binds.
// What a kernel slot of a node binds: a slot of the fragment, a transient, or (`constant`) the bytes at `index` of the
// fragment's constants.
struct Bound {
    bool transient{};
    uint32_t index{};
    bool constant{};
    // A constant whose bytes lie in the record at `index` rather than in the constant buffer, as a kernel's launch
    // metadata does when the caller passes its grid; the dispatch takes their device address (hal::RecordAddress).
    bool record{};
    // A carrier of the node's autodiff tape, whose fields the executor writes from the tape (TapeUse::fields).
    DeviceTapeCarrier tapeCarrier{};
};

// A kernel argument that takes one leaf of a Struct Storage in an element layout of its own, which strides in the
// kernel's elements cannot address in the Storage's, binds a dense carrier transient in its layout instead, as the
// Program runtime's endpoint projection does. A copy gathers the leaf into the carrier before the node, or zeroes the
// carrier when the Storage is one the graph may only write; a kernel that writes the carrier has it scattered back
// after the node.
struct Carrier {
    Bound storage;
    uint32_t transient{};
    bool gather{};
    bool scatter{};
    hal::LeafCopy copy;
};

// Builds one graph of a Program into a fragment, in node order. Each node's kernel record is copied to the next
// 16-byte boundary of the record, and its slots, fields and dispatch move with it.
class ProgramFragmentBuilder {
public:
    ProgramFragmentBuilder(const nlohmann::json &program, const std::map<std::string, Fragment> &stages)
        : program_(program), stages_(stages) {}

    Built build(const std::string &graphName, const std::vector<uint32_t> &zeroed,
                const std::vector<uint32_t> &exposed) {
        const nlohmann::json *graphs = member(program_, "graphs");
        values_ = member(program_, "values");
        storages_ = member(program_, "storages");
        const nlohmann::json *abi = member(program_, "abi");
        if (!graphs || !graphs->is_array() || !values_ || !values_->is_array() || !storages_ ||
            !storages_->is_array() || !abi)
            return failBuild("Program has no graphs, values, storages or ABI");
        for (size_t index = 0; index < values_->size(); ++index) {
            uint32_t id = 0;
            if (!readUint(member((*values_)[index], "id"), id) || id != index)
                return failBuild("Program values are not numbered in order");
        }
        for (size_t index = 0; index < storages_->size(); ++index) {
            uint32_t id = 0;
            if (!readUint(member((*storages_)[index], "id"), id) || id != index)
                return failBuild("Program storages are not numbered in order");
        }
        // Tape plans need no check of their own: a tape Storage is opaque, so a kernel binding one fails in
        // bindKernelSlot unless it binds the tape's launch metadata, which is a constant. A kernel keeping a tape
        // reaches it through the tape slot of the Storage its node initializes or reads (bindTape).
        const nlohmann::json *graph = nullptr;
        for (const nlohmann::json &candidate : *graphs)
            if (readString(member(candidate, "name")) == graphName)
                graph = &candidate;
        if (!graph)
            return failBuild("Program has no graph '" + graphName + "'");

        storageSlot_.assign(storages_->size(), kNoSlot);
        storageTransient_.assign(storages_->size(), kNoSlot);
        allocationRank_.assign(storages_->size(), kNoSlot);
        outputStorage_.assign(storages_->size(), false);
        valueSlot_.assign(values_->size(), kNoSlot);
        resultSlot_.assign(values_->size(), kNoSlot);
        resultLeaves_.assign(values_->size(), {});
        if (auto slots = addSlots(*abi, *graph, graphName, exposed); slots.isErr())
            return failBuild(std::move(slots).error());
        // A captured Value without a slot is one no node binds: binding a Storage that is neither a slot nor an
        // allocation fails in bindKernelSlot.
        if (auto inputs = addInputs(*graph); inputs.isErr())
            return failBuild(std::move(inputs).error());
        if (auto owned = resolveOwned(zeroed); owned.isErr())
            return failBuild(std::move(owned).error());
        // A zeroed slot is a buffer the caller allocates for this call, so nothing reads it before the schedule
        // starts; its fills all come first, their arguments side by side, so a device may run them as one launch.
        for (uint32_t slot = 0; slot < slotZeroed_.size(); ++slot)
            addSlotFill(slot);
        const nlohmann::json *nodes = member(*graph, "nodes");
        if (!nodes || !nodes->is_array())
            return failBuild("Program graph has no nodes");
        for (const nlohmann::json &node : *nodes) {
            if (auto added = addNode(node); added.isErr())
                return failBuild(std::move(added).error());
            fragment_.nodeOperationEnds.push_back(static_cast<uint32_t>(fragment_.operations.size()));
        }
        if (auto outputs = checkOutputs(*graph); outputs.isErr())
            return failBuild(std::move(outputs).error());
        for (const auto &[slot, valueId] : exposedCaptures_)
            slotValue_[slot] = valueId;
        for (uint32_t index = 0; index < placed_.size(); ++index) {
            if (placed_[index])
                continue;
            if (slotValue_[index] == kNoSlot)
                return failBuild("Program slot has neither a binding nor a Value");
            if (auto placed = placeUnbound(index, slotValue_[index]); placed.isErr())
                return failBuild(std::move(placed).error());
        }
        for (size_t index = 0; index < touched_.size(); ++index)
            if (!touched_[index])
                return failBuild("Programs with unused allocations are not linked yet");
        fragment_.recordSize = alignUp(end_, 16);
        image_.resize(fragment_.recordSize, 0);
        fragment_.recordTemplate = std::move(image_);
        if (auto valid = fragment_.validate(); valid.isErr())
            return failBuild(std::move(valid).error());
        return Built{ok(std::move(fragment_))};
    }

private:
    const nlohmann::json *value(uint32_t id) const { return id < values_->size() ? &(*values_)[id] : nullptr; }
    const nlohmann::json *storage(uint32_t id) const { return id < storages_->size() ? &(*storages_)[id] : nullptr; }

    // A slot per ABI boundary slot of the graph, then per exposed Storage, then per captured scalar Value a node binds,
    // in capture order. An input's extents are its own or static; an output's are what its Storage's byte length names.
    Checked addSlots(const nlohmann::json &abi, const nlohmann::json &graph, const std::string &graphName,
                     const std::vector<uint32_t> &exposed) {
        const nlohmann::json *boundary = member(abi, "boundary_slots");
        if (!boundary || !boundary->is_array())
            return failCheck("Program ABI has no boundary slots");
        std::vector<const nlohmann::json *> graphRows;
        for (size_t index = 0; index < boundary->size(); ++index) {
            const nlohmann::json &row = (*boundary)[index];
            uint32_t id = 0;
            uint32_t valueId = 0;
            if (!readUint(member(row, "id"), id) || id != index || !readUint(member(row, "value"), valueId) ||
                !value(valueId))
                return failCheck("Program boundary slots are malformed");
            const nlohmann::json *origin = member(*value(valueId), "origin");
            if (origin && readString(member(*origin, "graph")) == graphName)
                graphRows.push_back(&row);
        }
        // An output that publishes the Storage of an input view (a kernel rewriting its read_write argument) is that
        // input's slot: the caller passes the view once.
        std::map<uint32_t, size_t> inputOfStorage;
        std::vector<const nlohmann::json *> rows;
        std::vector<std::pair<uint32_t, uint32_t>> aliases;
        for (const nlohmann::json *row : graphRows) {
            uint32_t storageId = 0;
            const bool view = readUint(member(*row, "storage_id"), storageId);
            const bool output = readString(member(*row, "direction")) == "output";
            const auto shared = view && output ? inputOfStorage.find(storageId) : inputOfStorage.end();
            if (shared != inputOfStorage.end()) {
                aliases.push_back({member(*row, "id")->get<uint32_t>(), static_cast<uint32_t>(shared->second)});
                continue;
            }
            if (view && !output)
                inputOfStorage.emplace(storageId, rows.size());
            fragment_.boundarySlots.push_back(
                {member(*row, "id")->get<uint32_t>(), static_cast<uint32_t>(rows.size())});
            rows.push_back(row);
        }
        fragment_.boundarySlots.insert(fragment_.boundarySlots.end(), aliases.begin(), aliases.end());
        // Captured scalars a node binds as a value or takes a workgroup count from, in capture order; a VJP may capture
        // Values its kernels never read. planVjp passes the same ones.
        std::vector<bool> boundValues(values_->size(), false);
        const auto bind = [&](const nlohmann::json *reference) {
            uint32_t valueId = 0;
            if (reference && readUint(reference, valueId) && valueId < boundValues.size())
                boundValues[valueId] = true;
        };
        if (const nlohmann::json *nodes = member(graph, "nodes"); nodes && nodes->is_array())
            for (const nlohmann::json &node : *nodes) {
                if (const nlohmann::json *bindings = member(node, "bindings"); bindings && bindings->is_array())
                    for (const nlohmann::json &binding : *bindings) {
                        const nlohmann::json *projections = member(binding, "projections");
                        if (readString(member(binding, "tag")) != "value" || !projections || !projections->is_array())
                            continue;
                        for (const nlohmann::json &projection : *projections)
                            bind(member(projection, "value"));
                    }
                const nlohmann::json *operation = member(node, "operation");
                if (const nlohmann::json *workgroups = operation ? member(*operation, "workgroups") : nullptr;
                    workgroups && workgroups->is_array())
                    for (const nlohmann::json &axis : *workgroups)
                        if (const nlohmann::json *control = axis.is_object() ? member(axis, "control") : nullptr)
                            bind(member(*control, "value"));
            }
        std::vector<uint32_t> scalarCaptures;
        if (const nlohmann::json *captures = member(graph, "captures"); captures && captures->is_array())
            for (const nlohmann::json &capture : *captures) {
                uint32_t valueId = 0;
                if (!readUint(member(capture, "value"), valueId) || !value(valueId) || !boundValues[valueId])
                    continue;
                const nlohmann::json *stored = member(*value(valueId), "storage");
                if (!stored || stored->is_null())
                    scalarCaptures.push_back(valueId);
            }
        const size_t count = rows.size() + exposed.size() + scalarCaptures.size();
        fragment_.slots.resize(count);
        fragment_.shapes.resize(count);
        fragment_.extents.resize(count);
        placed_.assign(count, false);
        typed_.assign(count, false);
        slotValue_.assign(count, kNoSlot);
        for (size_t index = 0; index < rows.size(); ++index) {
            const nlohmann::json &row = *rows[index];
            uint32_t valueId = 0;
            (void)readUint(member(row, "value"), valueId);
            slotValue_[index] = valueId;
            const std::string category = readString(member(row, "category"));
            const bool output = readString(member(row, "direction")) == "output";
            if (category == "value") {
                if (output) {
                    if (auto added = addResultSlot(index, valueId, row); added.isErr())
                        return added;
                    continue;
                }
                fragment_.slots[index].kind = VERNON_SCHEDULE_SLOT_SCALAR;
                valueSlot_[valueId] = static_cast<uint32_t>(index);
                continue;
            }
            if (category != "storage_view")
                return failCheck("boundary slots of category '" + category + "' are not linked yet");
            uint32_t storageId = 0;
            if (!readUint(member(row, "storage_id"), storageId) || !storage(storageId))
                return failCheck("Program boundary slot names no Storage");
            uint32_t access = parseAccess(readString(member(row, "access")));
            // An owned output is an allocation the caller creates and initializes per call, so the Program may also
            // read it (accumulating into a zeroed output); the boundary records only that it is published.
            if (output && readString(member(*storage(storageId), "ownership")) == "owned")
                access |= VERNON_SCHEDULE_ACCESS_READ;
            if (auto added = addBufferSlot(index, storageId, access, member(row, "outer_shape"), output); added.isErr())
                return added;
        }
        for (size_t index = 0; index < exposed.size(); ++index) {
            const uint32_t storageId = exposed[index];
            if (!storage(storageId))
                return failCheck("exposed Storage does not exist");
            if (isTape(storageId)) {
                if (auto added = addTapeSlot(rows.size() + index, storageId, graphName); added.isErr())
                    return added;
                continue;
            }
            // An allocation of the graph is a residual it exports; any other exposed Storage is a Value it captures.
            const nlohmann::json *shape = nullptr;
            bool exported = false;
            if (const nlohmann::json *inputs = member(graph, "inputs"); inputs && inputs->is_array())
                for (const nlohmann::json &input : *inputs) {
                    uint32_t inputStorage = 0;
                    uint32_t valueId = 0;
                    if (readString(member(input, "tag")) == "allocation" &&
                        readUint(member(input, "storage"), inputStorage) && inputStorage == storageId &&
                        readUint(member(input, "value"), valueId) && value(valueId)) {
                        shape = member(*value(valueId), "shape");
                        exported = true;
                    }
                }
            if (const nlohmann::json *captures = member(graph, "captures"); !exported && captures)
                for (const nlohmann::json &capture : *captures) {
                    uint32_t valueId = 0;
                    uint32_t captured = 0;
                    if (readUint(member(capture, "value"), valueId) && value(valueId) &&
                        readUint(member(*value(valueId), "storage"), captured) && captured == storageId) {
                        shape = member(*value(valueId), "shape");
                        exposedCaptures_.push_back({static_cast<uint32_t>(rows.size() + index), valueId});
                    }
                }
            if (!shape)
                return failCheck("exposed Storage is neither an allocation nor a capture of the graph");
            const uint32_t access =
                exported ? VERNON_SCHEDULE_ACCESS_READ | VERNON_SCHEDULE_ACCESS_WRITE : VERNON_SCHEDULE_ACCESS_READ;
            if (auto added = addBufferSlot(rows.size() + index, storageId, access, shape, exported); added.isErr())
                return added;
        }
        for (size_t index = 0; index < scalarCaptures.size(); ++index) {
            const size_t slot = rows.size() + exposed.size() + index;
            fragment_.slots[slot].kind = VERNON_SCHEDULE_SLOT_SCALAR;
            valueSlot_[scalarCaptures[index]] = static_cast<uint32_t>(slot);
            slotValue_[slot] = scalarCaptures[index];
        }
        return Checked{ok()};
    }

    // A gradient of a scalar or Struct Value: a kernel result, which the caller passes as a buffer of one element, its
    // Value's canonical bytes, that the result's dispatch copies its frame to (ResultUse).
    Checked addResultSlot(size_t index, uint32_t valueId, const nlohmann::json &row) {
        if (readString(member(row, "role")) != "gradient")
            return failCheck("scalar Program outputs are not linked yet");
        const nlohmann::json *layout = member(*value(valueId), "value_layout");
        const nlohmann::json *leaves = layout ? member(*layout, "leaves") : nullptr;
        uint32_t elementSize = 0;
        if (!leaves || !leaves->is_array() || leaves->empty() || !readUint(member(*layout, "byte_size"), elementSize) ||
            !elementSize)
            return failCheck("scalar Program gradient has no value layout");
        const std::optional<VernonDataType> dtype =
            runtime::pipelineDataType(readString(member(leaves->front(), "dtype")));
        if (!dtype)
            return failCheck("scalar Program gradient has an unknown element type");
        VernonScheduleSlot &slot = fragment_.slots[index];
        slot.kind = VERNON_SCHEDULE_SLOT_BUFFER;
        slot.size = sizeof(void *);
        slot.dtype = static_cast<uint32_t>(*dtype);
        slot.element_size = elementSize;
        slot.access = VERNON_SCHEDULE_ACCESS_WRITE;
        typed_[index] = true;
        resultSlot_[valueId] = static_cast<uint32_t>(index);
        return Checked{ok()};
    }

    bool isTape(uint32_t storageId) const {
        const nlohmann::json *descriptor = storage(storageId) ? member(*storage(storageId), "descriptor") : nullptr;
        return descriptor && readString(member(*descriptor, "tag")) == "opaque" &&
               readString(member(*descriptor, "contract")) == "vernon.ad_tape";
    }

    // A tape Storage's slot: written in the graph whose node initializes its Value, read in the graph that captures it.
    // Its record bytes are placed with the node that uses it.
    Checked addTapeSlot(size_t index, uint32_t storageId, const std::string &graphName) {
        if (storageSlot_[storageId] != kNoSlot)
            return failCheck("slots sharing a Storage are not linked yet");
        uint32_t initial = 0;
        const nlohmann::json *origin = readUint(member(*storage(storageId), "initial_value"), initial) && value(initial)
                                           ? member(*value(initial), "origin")
                                           : nullptr;
        if (!origin || readString(member(*origin, "tag")) != "node_result")
            return failCheck("autodiff tape has no forward node");
        storageSlot_[storageId] = static_cast<uint32_t>(index);
        VernonScheduleSlot &slot = fragment_.slots[index];
        slot.kind = VERNON_SCHEDULE_SLOT_TAPE;
        slot.size = sizeof(void *);
        slot.access = readString(member(*origin, "graph")) == graphName ? VERNON_SCHEDULE_ACCESS_WRITE
                                                                        : VERNON_SCHEDULE_ACCESS_READ;
        typed_[index] = true;
        return Checked{ok()};
    }

    // Binds a tape kernel's dispatch to the tape Storage its node initializes or reads, which must be a tape slot, and
    // places the slot's pointer in the record.
    Checked bindTape(const nlohmann::json &node, TapeUse &tape) {
        const nlohmann::json *accesses = member(node, "accesses");
        const nlohmann::json *found = nullptr;
        if (accesses && accesses->is_array())
            for (const nlohmann::json &access : *accesses) {
                uint32_t storageId = 0;
                if (readUint(member(access, "storage"), storageId) && storage(storageId) && isTape(storageId)) {
                    if (found)
                        return failCheck("Program node uses two autodiff tapes");
                    found = &access;
                }
            }
        uint32_t storageId = 0;
        if (!found || !readUint(member(*found, "storage"), storageId))
            return failCheck("Program node's kernel keeps an autodiff tape the node names no Storage for");
        const std::string tag = readString(member(*found, "tag"));
        if (tag != "initialize" && tag != "read")
            return failCheck("Program node accesses its autodiff tape other than to initialize or read it");
        const uint32_t slotIndex = storageSlot_[storageId];
        if (slotIndex == kNoSlot || fragment_.slots[slotIndex].kind != VERNON_SCHEDULE_SLOT_TAPE)
            return failCheck("autodiff tapes that are not exposed are not linked yet");
        tape.writes = tag == "initialize";
        if (fragment_.slots[slotIndex].access !=
            (tape.writes ? VERNON_SCHEDULE_ACCESS_WRITE : VERNON_SCHEDULE_ACCESS_READ))
            return failCheck("Program node's autodiff tape access disagrees with its graph");
        if (placed_[slotIndex])
            return failCheck("autodiff tapes several nodes use are not linked yet");
        uint32_t valueId = 0;
        const nlohmann::json *row =
            readUint(member(*found, tape.writes ? "after" : "value"), valueId) ? value(valueId) : nullptr;
        uint64_t stride = 0;
        if (!row || !member(*row, "minimum_tape_stride_bytes") ||
            !member(*row, "minimum_tape_stride_bytes")->is_number_unsigned() ||
            !(stride = member(*row, "minimum_tape_stride_bytes")->get<uint64_t>()))
            return failCheck("autodiff tape Value has no stride");
        tape.stride = stride;
        tape.slot = slotIndex;
        VernonScheduleSlot &slot = fragment_.slots[slotIndex];
        slot.offset = alignUp(end_, alignof(void *));
        end_ = slot.offset + sizeof(void *);
        image_.resize(end_, 0);
        placed_[slotIndex] = true;
        return Checked{ok()};
    }

    Checked addBufferSlot(size_t index, uint32_t storageId, uint32_t access, const nlohmann::json *shape, bool output) {
        if (storageSlot_[storageId] != kNoSlot)
            return failCheck("slots sharing a Storage are not linked yet");
        if (auto checked = checkStorage(storageId); checked.isErr())
            return checked;
        const auto slotIndex = static_cast<uint32_t>(index);
        storageSlot_[storageId] = slotIndex;
        VernonScheduleSlot &slot = fragment_.slots[index];
        slot.kind = VERNON_SCHEDULE_SLOT_BUFFER;
        slot.size = sizeof(void *);
        slot.access = access;
        if (!slot.access)
            return failCheck("Program slot has no access");
        if (!shape || !shape->is_array())
            return failCheck("Program slot has no shape");
        for (const nlohmann::json &extent : *shape) {
            if (!extent.is_number_integer() || extent.get<int64_t>() < -1)
                return failCheck("Program slot shape is malformed");
            fragment_.shapes[index].push_back(extent.get<int64_t>());
        }
        slot.rank = static_cast<uint32_t>(fragment_.shapes[index].size());
        for (uint32_t dimension = 0; dimension < slot.rank; ++dimension) {
            const int64_t declared = fragment_.shapes[index][dimension];
            fragment_.extents[index].push_back(declared >= 0 ? Extent{declared, kNoSlot, 0}
                                                             : Extent{0, slotIndex, dimension});
        }
        if (output)
            outputStorage_[storageId] = true;
        return Checked{ok()};
    }

    // Owned Storages in id order, which is capture order, so the Storage an extent names is resolved before it: an
    // output takes the extents its byte length names, and an allocation no output publishes becomes a transient.
    Checked resolveOwned(const std::vector<uint32_t> &zeroed) {
        // A zeroed Storage the caller binds, such as a fresh output, is zeroed in its whole buffer. A slot the graph
        // only reads, such as a forward Storage a backward graph captures, holds what its producer wrote.
        slotZeroed_.assign(fragment_.slots.size(), false);
        slotFilled_.assign(fragment_.slots.size(), false);
        for (uint32_t id : zeroed)
            if (id < storageSlot_.size() && storageSlot_[id] != kNoSlot &&
                (fragment_.slots[storageSlot_[id]].access & VERNON_SCHEDULE_ACCESS_WRITE))
                slotZeroed_[storageSlot_[id]] = true;
        for (uint32_t id = 0; id < storages_->size(); ++id) {
            if (outputStorage_[id]) {
                const uint32_t slotIndex = storageSlot_[id];
                auto extents = storageExtents(id, fragment_.slots[slotIndex].rank);
                if (extents.isErr())
                    return Checked{err(std::move(extents).error())};
                for (uint32_t dimension = 0; dimension < fragment_.slots[slotIndex].rank; ++dimension) {
                    const Extent &extent = extents.value()[dimension];
                    const int64_t declared = fragment_.shapes[slotIndex][dimension];
                    if (extent.slot == kNoSlot && declared >= 0 && declared != extent.value)
                        return failCheck("Program output extent disagrees with its shape");
                    if (declared < 0 || extent.slot == kNoSlot)
                        fragment_.extents[slotIndex][dimension] = extent;
                }
                continue;
            }
            if (allocationRank_[id] == kNoSlot)
                continue;
            if (auto checked = checkStorage(id); checked.isErr())
                return checked;
            auto extents = storageExtents(id, allocationRank_[id]);
            if (extents.isErr())
                return Checked{err(std::move(extents).error())};
            Transient transient;
            transient.extents = std::move(extents).value();
            transient.zeroed = std::find(zeroed.begin(), zeroed.end(), id) != zeroed.end();
            storageTransient_[id] = static_cast<uint32_t>(fragment_.transients.size());
            fragment_.transients.push_back(std::move(transient));
            touched_.push_back(false);
            transientTyped_.push_back(false);
        }
        return Checked{ok()};
    }

    Checked checkStorage(uint32_t id) const {
        const nlohmann::json &row = *storage(id);
        if (readString(member(row, "lifetime")) != "invocation")
            return failCheck("Storages that outlive an invocation are not linked yet");
        const nlohmann::json *descriptor = member(row, "descriptor");
        if (!descriptor || readString(member(*descriptor, "tag")) != "buffer")
            return failCheck("Storages other than buffers are not linked yet");
        return Checked{ok()};
    }

    // An owned Storage's extents: per axis a constant or an axis of a Value, which resolves to the slot or transient
    // the Value lives in. A static byte length takes its extents from the Storage's initial Value.
    Result<std::vector<Extent>, std::string> storageExtents(uint32_t id, uint32_t rank, unsigned depth = 0) const {
        using Extents = Result<std::vector<Extent>, std::string>;
        const auto fail = [](std::string message) { return Extents{err(std::move(message))}; };
        const nlohmann::json &row = *storage(id);
        const nlohmann::json *length = member(*member(row, "descriptor"), "byte_length");
        std::vector<Extent> extents;
        if (length && length->is_array()) {
            for (const nlohmann::json &component : *length) {
                if (component.is_number_unsigned()) {
                    extents.push_back({component.get<int64_t>(), kNoSlot, 0});
                    continue;
                }
                const nlohmann::json *dimension = member(component, "dimension");
                uint32_t source = 0;
                uint32_t axis = 0;
                if (!dimension || !readUint(member(*dimension, "axis"), axis) ||
                    !readUint(member(*member(*dimension, "control"), "value"), source) || !value(source))
                    return fail("Storage extents computed from Values are not linked yet");
                uint32_t sourceStorage = 0;
                if (!readUint(member(*value(source), "storage"), sourceStorage) || !storage(sourceStorage))
                    return fail("Storage extent names a Value without a Storage");
                if (storageSlot_[sourceStorage] != kNoSlot) {
                    const uint32_t slot = storageSlot_[sourceStorage];
                    if (axis >= fragment_.slots[slot].rank)
                        return fail("Storage extent names no axis of its source");
                    extents.push_back(fragment_.extents[slot][axis]);
                } else if (storageTransient_[sourceStorage] != kNoSlot) {
                    const Transient &transient = fragment_.transients[storageTransient_[sourceStorage]];
                    if (axis >= transient.extents.size())
                        return fail("Storage extent names no axis of its source");
                    extents.push_back(transient.extents[axis]);
                } else if (const nlohmann::json *sourceDescriptor = member(*storage(sourceStorage), "descriptor");
                           depth < kMaximumExtentDepth && sourceDescriptor &&
                           member(*sourceDescriptor, "byte_length") &&
                           member(*sourceDescriptor, "byte_length")->is_array() &&
                           axis < member(*sourceDescriptor, "byte_length")->size()) {
                    // A Storage the graph only captures (a forward output a VJP keeps but no node binds) has no slot;
                    // its own byte length names the slots its extents come from.
                    auto resolved = storageExtents(
                        sourceStorage, static_cast<uint32_t>(member(*sourceDescriptor, "byte_length")->size()),
                        depth + 1);
                    if (resolved.isErr())
                        return resolved;
                    extents.push_back(std::move(resolved).value()[axis]);
                } else {
                    return fail("Storage extent of Storage " + std::to_string(id) + " names Storage " +
                                std::to_string(sourceStorage) +
                                ", which the graph neither imports nor defines before it");
                }
            }
        } else {
            uint32_t initial = 0;
            const nlohmann::json *shape = readUint(member(row, "initial_value"), initial) && value(initial)
                                              ? member(*value(initial), "shape")
                                              : nullptr;
            if (!shape || !shape->is_array())
                return fail("Storage has no static shape");
            for (const nlohmann::json &extent : *shape) {
                if (!extent.is_number_unsigned())
                    return fail("Storage has no static shape");
                extents.push_back({extent.get<int64_t>(), kNoSlot, 0});
            }
        }
        if (extents.size() != rank)
            return fail("Storage extents disagree with its rank");
        return Extents{ok(std::move(extents))};
    }

    // Records each allocation that no output publishes, with its rank; resolveOwned makes it a transient, whose
    // element type comes from its first binding.
    Checked addInputs(const nlohmann::json &graph) {
        const nlohmann::json *inputs = member(graph, "inputs");
        if (!inputs || !inputs->is_array())
            return failCheck("Program graph has no inputs");
        for (const nlohmann::json &input : *inputs) {
            const std::string tag = readString(member(input, "tag"));
            if (tag == "user_input")
                continue;
            if (tag != "allocation")
                return failCheck("graph inputs of kind '" + tag + "' are not linked yet");
            uint32_t storageId = 0;
            uint32_t valueId = 0;
            if (!readUint(member(input, "storage"), storageId) || !storage(storageId) ||
                !readUint(member(input, "value"), valueId) || !value(valueId))
                return failCheck("Program allocation is malformed");
            if (storageSlot_[storageId] != kNoSlot)
                continue;
            const nlohmann::json *shape = member(*value(valueId), "shape");
            if (!shape || !shape->is_array())
                return failCheck("Program allocation has no shape");
            allocationRank_[storageId] = static_cast<uint32_t>(shape->size());
        }
        return Checked{ok()};
    }

    // A kernel that declares a static extent where the Program's is free makes the caller's extent static.
    Checked tighten(Extent &extent, int64_t declared) {
        if (declared < 0)
            return Checked{ok()};
        if (extent.slot == kNoSlot)
            return extent.value == declared ? Checked{ok()}
                                            : failCheck("kernel extent disagrees with its Program argument");
        std::vector<Extent> &source = fragment_.extents[extent.slot];
        if (source[extent.dimension].slot != extent.slot || source[extent.dimension].dimension != extent.dimension)
            return failCheck("kernel extent constrains a derived Program extent");
        const int64_t previous = fragment_.shapes[extent.slot][extent.dimension];
        if (previous >= 0 && previous != declared)
            return failCheck("kernel extent disagrees with its Program argument");
        fragment_.shapes[extent.slot][extent.dimension] = declared;
        source[extent.dimension] = {declared, kNoSlot, 0};
        extent = {declared, kNoSlot, 0};
        return Checked{ok()};
    }

    // Where leaf `leaf` lies in the element of the Storage `access` names, which `element` takes the type of, and in
    // the element a kernel argument declares.
    Checked projectedLeaf(const nlohmann::json &access, uint32_t leaf, const Fragment &kernel, size_t kernelSlot,
                          VernonScheduleSlot &element, hal::LeafCopy &copy) const {
        uint32_t valueId = 0;
        if (!readUint(member(access, "value"), valueId) && !readUint(member(access, "after"), valueId))
            return failCheck("Program node access names no Value");
        const nlohmann::json *layout = value(valueId) ? member(*value(valueId), "value_layout") : nullptr;
        const nlohmann::json *leaves = layout ? member(*layout, "leaves") : nullptr;
        uint32_t elementSize = 0;
        uint32_t offset = 0;
        uint32_t count = 0;
        if (!leaves || !leaves->is_array() || leaf >= leaves->size() ||
            readString(member(*layout, "scope")) != "element" || !readUint(member(*layout, "byte_size"), elementSize) ||
            !readUint(member((*leaves)[leaf], "byte_offset"), offset) ||
            !readUint(member((*leaves)[leaf], "scalar_count"), count))
            return failCheck("Program node binds a leaf of a Value without an element layout");
        const std::optional<VernonDataType> first =
            runtime::pipelineDataType(readString(member(leaves->front(), "dtype")));
        const std::optional<VernonDataType> scalar =
            runtime::pipelineDataType(readString(member((*leaves)[leaf], "dtype")));
        if (!first || !scalar)
            return failCheck("Program node binds a leaf of an unknown element type");
        element.dtype = static_cast<uint32_t>(*first);
        element.element_size = elementSize;
        copy.storageElement = elementSize;
        copy.storageOffset = offset;
        copy.carrierElement = kernel.slots[kernelSlot].element_size;
        copy.carrierOffset = kernelSlot < kernel.slotLeafOffset.size() ? kernel.slotLeafOffset[kernelSlot] : 0;
        copy.bytes = uint64_t{count} * scalarBytes(*scalar);
        if (!copy.bytes || copy.storageOffset > copy.storageElement ||
            copy.bytes > copy.storageElement - copy.storageOffset || copy.carrierOffset > copy.carrierElement ||
            copy.bytes > copy.carrierElement - copy.carrierOffset)
            return failCheck("kernel leaf does not fit its Program Value's element or its own");
        return Checked{ok()};
    }

    // A GPU kernel sums the lanes' share of a scalar Program gradient into a buffer of one element: the gradient's
    // slot, which the call zeroes before the first node that adds to it. Several nodes may add to one gradient.
    Checked bindGradientBuffer(const VernonScheduleSlot &declared, const nlohmann::json &binding, Bound &bound) {
        const nlohmann::json *projections = member(binding, "projections");
        uint32_t valueId = 0;
        if (!projections || !projections->is_array() || projections->size() != 1 ||
            readString(member((*projections)[0], "direction")) != "result" ||
            !readUint(member((*projections)[0], "value"), valueId) || valueId >= resultSlot_.size() ||
            resultSlot_[valueId] == kNoSlot)
            return failCheck("kernel results other than scalar Program gradients are not linked yet");
        const uint32_t slotIndex = resultSlot_[valueId];
        VernonScheduleSlot &slot = fragment_.slots[slotIndex];
        if (declared.rank != 0 || declared.dtype != slot.dtype || declared.element_size != slot.element_size)
            return failCheck("kernel gradient buffer disagrees with its scalar Program gradient");
        slot.access = VERNON_SCHEDULE_ACCESS_READ_WRITE;
        std::vector<bool> &leaves = resultLeaves_[valueId];
        leaves.assign(member(*member(*value(valueId), "value_layout"), "leaves")->size(), true);
        slotZeroed_[slotIndex] = true;
        addSlotFill(slotIndex);
        bound = {false, slotIndex};
        return Checked{ok()};
    }

    Checked bindKernelSlot(const Fragment &kernel, size_t kernelSlot, const nlohmann::json &node, Bound &bound,
                           std::vector<Carrier> &carriers) {
        const VernonScheduleSlot &declared = kernel.slots[kernelSlot];
        const uint32_t argument = kernel.slotArguments[kernelSlot];
        const nlohmann::json *bindings = member(node, "bindings");
        const nlohmann::json *binding = nullptr;
        if (bindings && bindings->is_array())
            for (const nlohmann::json &candidate : *bindings) {
                uint32_t index = 0;
                if (readUint(member(candidate, "index"), index) && index == argument &&
                    readString(member(candidate, "interface")) == "argument")
                    binding = &candidate;
            }
        if (!binding)
            return failCheck("Program node leaves a kernel argument unbound");
        const std::string tag = readString(member(*binding, "tag"));
        if (tag == "value" && declared.kind == VERNON_SCHEDULE_SLOT_BUFFER)
            return bindGradientBuffer(declared, *binding, bound);
        if (declared.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
            const nlohmann::json *projections = member(*binding, "projections");
            uint32_t valueId = 0;
            if (tag != "value" || !projections || !projections->is_array() || projections->size() != 1 ||
                !readUint(member((*projections)[0], "value"), valueId) || valueId >= valueSlot_.size() ||
                valueSlot_[valueId] == kNoSlot)
                return failCheck("node scalars other than Program arguments are not linked yet");
            bound = {false, valueSlot_[valueId]};
            VernonScheduleSlot &slot = fragment_.slots[bound.index];
            if (!typed_[bound.index]) {
                slot.dtype = declared.dtype;
                slot.size = declared.size;
                slot.element_size = declared.element_size;
                typed_[bound.index] = true;
            } else if (slot.dtype != declared.dtype || slot.size != declared.size) {
                return failCheck("Program scalar argument is bound with two types");
            }
            return Checked{ok()};
        }
        const nlohmann::json *accesses = member(node, "accesses");
        uint32_t accessIndex = 0;
        uint32_t storageId = 0;
        if (tag != "resource" || !readUint(member(*binding, "access"), accessIndex) || !accesses ||
            !accesses->is_array() || accessIndex >= accesses->size() ||
            !readUint(member((*accesses)[accessIndex], "storage"), storageId) || !storage(storageId))
            return failCheck("Program node binds a TensorView to no Storage");
        if (kernelSlot < kernel.slotTapeCarrier.size() &&
            kernel.slotTapeCarrier[kernelSlot] != DeviceTapeCarrier::None) {
            if (!isTape(storageId))
                return failCheck("Program node binds a kernel's autodiff tape carrier to a Storage that is no tape");
            bound.tapeCarrier = kernel.slotTapeCarrier[kernelSlot];
            return Checked{ok()};
        }
        const std::string accessTag = readString(member((*accesses)[accessIndex], "tag"));
        if ((declared.access & VERNON_SCHEDULE_ACCESS_WRITE) ? accessTag != "write" : accessTag != "read")
            return failCheck("Program node access disagrees with its kernel");
        if (kernel.slotLaunchMetadata[kernelSlot])
            return bindLaunchMetadata(declared, (*accesses)[accessIndex], storageId, bound);
        // A carrier's leading axis is the dispatch's (addNode), so the Program's view has one axis fewer.
        const uint32_t carrier = kernel.slotInvocationCarrier[kernelSlot] ? 1 : 0;
        const uint32_t rank = declared.rank - carrier;
        const std::vector<int64_t> kernelShape(kernel.shapes[kernelSlot].begin() + carrier,
                                               kernel.shapes[kernelSlot].end());
        // The Storage's element, which a kernel taking one leaf of it does not declare.
        VernonScheduleSlot element = declared;
        hal::LeafCopy copy;
        uint32_t leaf = 0;
        const bool projected = readUint(member(*binding, "leaf"), leaf);
        if (projected)
            if (auto read = projectedLeaf((*accesses)[accessIndex], leaf, kernel, kernelSlot, element, copy);
                read.isErr())
                return read;
        if (storageSlot_[storageId] != kNoSlot) {
            bound = {false, storageSlot_[storageId]};
            VernonScheduleSlot &slot = fragment_.slots[bound.index];
            if (slot.rank != rank || (declared.access & ~slot.access))
                return failCheck("Program node binds an argument its kernel cannot take");
            if (!typed_[bound.index]) {
                slot.dtype = element.dtype;
                slot.element_size = element.element_size;
                typed_[bound.index] = true;
            } else if (slot.dtype != element.dtype || slot.element_size != element.element_size) {
                return failCheck("Program argument is bound with two element types (" + elementType(slot) + " and " +
                                 elementType(element) + ")");
            }
            for (uint32_t dimension = 0; dimension < slot.rank; ++dimension) {
                Extent extent = fragment_.extents[bound.index][dimension];
                if (auto checked = tighten(extent, kernelShape[dimension]); checked.isErr())
                    return checked;
            }
        } else {
            if (storageTransient_[storageId] == kNoSlot)
                return failCheck("Program node binds a Storage that is neither an argument nor an allocation");
            bound = {true, storageTransient_[storageId]};
            Transient &transient = fragment_.transients[bound.index];
            if (transient.extents.size() != rank)
                return failCheck("Program node binds an allocation its kernel cannot take");
            if (!transientTyped_[bound.index]) {
                transient.dtype = element.dtype;
                transient.elementSize = element.element_size;
                transientTyped_[bound.index] = true;
            } else if (transient.dtype != element.dtype || transient.elementSize != element.element_size) {
                return failCheck("Program allocation is bound with two element types (" +
                                 std::to_string(transient.dtype) + "/" + std::to_string(transient.elementSize) +
                                 " and " + elementType(element) + ")");
            }
            for (uint32_t dimension = 0; dimension < rank; ++dimension)
                if (auto checked = tighten(transient.extents[dimension], kernelShape[dimension]); checked.isErr())
                    return checked;
        }
        if (!projected || (copy.carrierElement == copy.storageElement && copy.carrierOffset == copy.storageOffset))
            return Checked{ok()};
        if (carrier)
            return failCheck("autodiff carriers of Struct leaves are not linked yet");
        copy.rank = rank;
        Carrier added{bound, static_cast<uint32_t>(fragment_.transients.size())};
        added.gather = bound.transient || (fragment_.slots[bound.index].access & VERNON_SCHEDULE_ACCESS_READ);
        added.scatter = (declared.access & VERNON_SCHEDULE_ACCESS_WRITE) != 0;
        if (!added.gather && (declared.access & VERNON_SCHEDULE_ACCESS_READ))
            return failCheck("kernel reads a Struct leaf of a Storage the graph may only write");
        added.copy = copy;
        Transient packed;
        packed.dtype = declared.dtype;
        packed.elementSize = declared.element_size;
        packed.extents = bound.transient ? fragment_.transients[bound.index].extents : fragment_.extents[bound.index];
        packed.zeroed = !added.gather;
        fragment_.transients.push_back(std::move(packed));
        touched_.push_back(false);
        transientTyped_.push_back(true);
        bound = {true, added.transient};
        carriers.push_back(std::move(added));
        return Checked{ok()};
    }

    // A constant's view is the whole of its static shape, so its metadata goes into the record template; its address
    // is the executor's.
    Checked placeConstantField(const VernonScheduleField &field, const std::vector<int64_t> &shape,
                               std::vector<hal::RecordAddress> *recordAddresses = nullptr) {
        int64_t value = 0;
        switch (field.kind) {
        case VERNON_SCHEDULE_FIELD_ADDRESS:
            if (field.size != sizeof(uint64_t))
                return failCheck("constant address field is not 8 bytes");
            if (recordAddresses)
                recordAddresses->push_back({field.offset, field.slot});
            else
                fragment_.constantAddresses.push_back({field.offset, field.slot});
            return Checked{ok()};
        case VERNON_SCHEDULE_FIELD_OFFSET:
            value = 0;
            break;
        case VERNON_SCHEDULE_FIELD_EXTENT:
            if (field.dimension >= shape.size() || shape[field.dimension] < 0)
                return failCheck("constant has no static shape");
            value = shape[field.dimension];
            break;
        case VERNON_SCHEDULE_FIELD_STRIDE:
            value = 1;
            for (size_t dimension = field.dimension + 1; dimension < shape.size(); ++dimension)
                value *= shape[dimension];
            break;
        case VERNON_SCHEDULE_FIELD_CAPACITY:
            value = 1;
            for (int64_t extent : shape)
                value *= extent;
            break;
        default:
            return failCheck("constant has a field of an unknown kind");
        }
        return writeImage(field, value);
    }

    // Writes a field whose value the link fixes into the record template.
    Checked writeImage(const VernonScheduleField &field, int64_t value) {
        if (field.offset + field.size > image_.size())
            return failCheck("fixed field lies outside its record");
        if (field.size == sizeof(int32_t)) {
            if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
                return failCheck("fixed field value exceeds its field");
            const auto narrowed = static_cast<int32_t>(value);
            std::memcpy(image_.data() + field.offset, &narrowed, sizeof(narrowed));
        } else if (field.size == sizeof(int64_t)) {
            std::memcpy(image_.data() + field.offset, &value, sizeof(value));
        } else {
            return failCheck("fixed field is neither 4 nor 8 bytes");
        }
        return Checked{ok()};
    }

    // A GPU backward kernel replays its forward's invocation ids from the forward grid, which a Program fixes per
    // node; the tape Storage the forward node initializes carries it, so it becomes a constant of three u32 counts.
    Checked bindLaunchMetadata(const VernonScheduleSlot &declared, const nlohmann::json &access, uint32_t storageId,
                               Bound &bound) {
        const nlohmann::json *descriptor = member(*storage(storageId), "descriptor");
        if (!descriptor || readString(member(*descriptor, "tag")) != "opaque" ||
            readString(member(*descriptor, "contract")) != "vernon.ad_tape")
            return failCheck("kernel launch metadata binds no autodiff tape");
        if (declared.rank != 1 || declared.element_size != sizeof(uint32_t) ||
            (declared.access & VERNON_SCHEDULE_ACCESS_WRITE))
            return failCheck("kernel launch metadata is not three read-only 32-bit counts");
        uint32_t valueId = 0;
        uint32_t nodeId = 0;
        const nlohmann::json *origin =
            readUint(member(access, "value"), valueId) && value(valueId) ? member(*value(valueId), "origin") : nullptr;
        if (!origin || readString(member(*origin, "tag")) != "node_result" ||
            !readUint(member(*origin, "node"), nodeId))
            return failCheck("autodiff tape has no forward node");
        const std::string graphName = readString(member(*origin, "graph"));
        const nlohmann::json *workgroups = nullptr;
        for (const nlohmann::json &graph : *member(program_, "graphs")) {
            if (readString(member(graph, "name")) != graphName)
                continue;
            if (const nlohmann::json *nodes = member(graph, "nodes"); nodes && nodes->is_array())
                for (const nlohmann::json &node : *nodes) {
                    uint32_t id = 0;
                    if (readUint(member(node, "id"), id) && id == nodeId)
                        workgroups = member(*member(node, "operation"), "workgroups");
                }
        }
        uint32_t grid[3]{};
        uint32_t gridSlot[3]{kNoSlot, kNoSlot, kNoSlot};
        if (!workgroups || !workgroups->is_array() || workgroups->size() != 3)
            return failCheck("autodiff tape's forward node has no grid");
        for (unsigned axis = 0; axis < 3; ++axis) {
            if (readUint(&(*workgroups)[axis], grid[axis]))
                continue;
            const nlohmann::json *control = member((*workgroups)[axis], "control");
            uint32_t source = 0;
            if (!control || !readUint(member(*control, "value"), source) || source >= valueSlot_.size() ||
                valueSlot_[source] == kNoSlot)
                return failCheck("launch metadata of grids computed from Values other than Program arguments is not "
                                 "linked yet");
            gridSlot[axis] = valueSlot_[source];
            VernonScheduleSlot &slot = fragment_.slots[gridSlot[axis]];
            if (!typed_[gridSlot[axis]]) {
                slot.dtype = VERNON_DATA_U32;
                slot.size = slot.element_size = sizeof(uint32_t);
                typed_[gridSlot[axis]] = true;
            } else if (slot.size != sizeof(uint32_t)) {
                return failCheck("Program grid argument is not a 32-bit integer");
            }
        }
        if (gridSlot[0] == kNoSlot && gridSlot[1] == kNoSlot && gridSlot[2] == kNoSlot) {
            const uint64_t at = alignUp(fragment_.constants.size(), 16);
            fragment_.constants.resize(at + 16, 0);
            std::memcpy(fragment_.constants.data() + at, grid, sizeof(grid));
            bound = {false, static_cast<uint32_t>(at), true};
            return Checked{ok()};
        }
        // The caller's grid changes per call, so the counts live in the record: a fixed axis in its template, a
        // caller's axis as its grid slot's bytes.
        const uint64_t at = alignUp(end_, 16);
        end_ = at + 16;
        image_.resize(end_, 0);
        std::memcpy(image_.data() + at, grid, sizeof(grid));
        for (unsigned axis = 0; axis < 3; ++axis) {
            const uint32_t slot = gridSlot[axis];
            if (slot == kNoSlot)
                continue;
            const uint64_t count = at + axis * sizeof(uint32_t);
            if (!placed_[slot]) {
                fragment_.slots[slot].offset = count;
                placed_[slot] = true;
            } else {
                fragment_.fields.push_back(
                    {slot, VERNON_SCHEDULE_FIELD_VALUE, 0, static_cast<uint32_t>(sizeof(uint32_t)), count});
            }
        }
        bound = {false, static_cast<uint32_t>(at), true, true};
        return Checked{ok()};
    }

    // A slot no node binds still takes its argument, and a captured Storage still sizes the allocations and gradients
    // whose extents name it (a VJP keeps a weight whose gradient needs only its shape), so it gets a site of its own
    // that no operation reads. A scalar's site is its bytes; a view's is an address, then its offset, extents and
    // strides, 8 bytes each, which a caller writes as for any binding. Its types are its Value's, as a kernel's
    // reflection names them: the first leaf's scalar type and the whole element's size.
    Checked placeUnbound(uint32_t index, uint32_t valueId) {
        if (placed_[index])
            return Checked{ok()};
        VernonScheduleSlot &slot = fragment_.slots[index];
        if (!typed_[index]) {
            const nlohmann::json *layout = member(*value(valueId), "value_layout");
            const nlohmann::json *leaves = layout ? member(*layout, "leaves") : nullptr;
            uint32_t elementSize = 0;
            if (!leaves || !leaves->is_array() || leaves->empty() ||
                !readUint(member(*layout, "byte_size"), elementSize) || !elementSize)
                return failCheck("unbound Program slot has no element layout");
            const std::optional<VernonDataType> dtype =
                runtime::pipelineDataType(readString(member(leaves->front(), "dtype")));
            if (!dtype)
                return failCheck("unbound Program slot has an unknown element type");
            slot.dtype = static_cast<uint32_t>(*dtype);
            slot.element_size = elementSize;
            if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR)
                slot.size = elementSize;
            typed_[index] = true;
        }
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
            slot.offset = alignUp(end_, 16);
            end_ = slot.offset + slot.size;
            image_.resize(end_, 0);
            placed_[index] = true;
            return Checked{ok()};
        }
        if (slot.kind != VERNON_SCHEDULE_SLOT_BUFFER)
            return failCheck("unbound Program slot is neither a scalar nor a view");
        uint64_t at = alignUp(end_, alignof(uint64_t));
        slot.offset = at;
        at += sizeof(uint64_t);
        fragment_.fields.push_back({index, VERNON_SCHEDULE_FIELD_OFFSET, 0, sizeof(uint64_t), at});
        at += sizeof(uint64_t);
        for (uint32_t dimension = 0; dimension < slot.rank; ++dimension, at += sizeof(uint64_t))
            fragment_.fields.push_back({index, VERNON_SCHEDULE_FIELD_EXTENT, dimension, sizeof(uint64_t), at});
        for (uint32_t dimension = 0; dimension < slot.rank; ++dimension, at += sizeof(uint64_t))
            fragment_.fields.push_back({index, VERNON_SCHEDULE_FIELD_STRIDE, dimension, sizeof(uint64_t), at});
        end_ = at;
        image_.resize(end_, 0);
        placed_[index] = true;
        return Checked{ok()};
    }

    void touch(uint32_t transient) {
        const auto operation = static_cast<uint32_t>(fragment_.operations.size());
        Transient &row = fragment_.transients[transient];
        if (!touched_[transient])
            row.first = operation;
        row.last = operation;
        touched_[transient] = true;
    }

    // Binds a kernel result to scalar gradient slots. A kernel reflects a result either as one entry copying the whole
    // Value its binding projects into, or as one entry per physical leaf (physical leaf `ordinal` of `count` entries
    // sharing the result's index); each leaf's projection names the Value and the leaf it lands in.
    Checked bindResult(const nlohmann::json &node, ResultUse &result, size_t ordinal, size_t count) {
        const nlohmann::json *bindings = member(node, "bindings");
        const nlohmann::json *projections = nullptr;
        if (bindings && bindings->is_array())
            for (const nlohmann::json &candidate : *bindings) {
                uint32_t index = 0;
                if (readUint(member(candidate, "index"), index) && index == result.result &&
                    readString(member(candidate, "interface")) == "result" &&
                    readString(member(candidate, "tag")) == "value")
                    projections = member(candidate, "projections");
            }
        if (!projections || !projections->is_array() || projections->empty())
            return failCheck("Program node leaves a kernel result unbound");
        const auto leavesOf = [&](uint32_t valueId) -> const nlohmann::json * {
            const nlohmann::json *layout = value(valueId) ? member(*value(valueId), "value_layout") : nullptr;
            const nlohmann::json *leaves = layout ? member(*layout, "leaves") : nullptr;
            return leaves && leaves->is_array() && !leaves->empty() ? leaves : nullptr;
        };
        const auto gradientSlot = [&](uint32_t valueId) {
            return valueId < resultSlot_.size() ? resultSlot_[valueId] : kNoSlot;
        };
        const auto claim = [&](uint32_t valueId, uint32_t leaf) {
            std::vector<bool> &bound = resultLeaves_[valueId];
            if (bound.empty())
                bound.assign(leavesOf(valueId)->size(), false);
            if (leaf >= bound.size() || bound[leaf])
                return false;
            bound[leaf] = true;
            return true;
        };
        const auto fits = [&](uint64_t begin, uint64_t end) {
            for (const runtime::ValueByteCopy &copy : result.copies)
                if (copy.canonicalOffset < begin || copy.canonicalOffset > end ||
                    copy.size > end - copy.canonicalOffset)
                    return false;
            return true;
        };

        if (count == 1 && projections->size() > 1) {
            uint32_t valueId = kNoSlot;
            for (const nlohmann::json &projection : *projections) {
                uint32_t projected = 0;
                if (!readUint(member(projection, "value"), projected) || (valueId != kNoSlot && projected != valueId))
                    return failCheck("a kernel result copied whole projects into several Values");
                valueId = projected;
            }
            if (gradientSlot(valueId) == kNoSlot || !leavesOf(valueId))
                return failCheck("kernel results other than scalar Program gradients are not linked yet");
            if (leavesOf(valueId)->size() != projections->size())
                return failCheck("a kernel result copied whole writes part of its Value");
            for (uint32_t leaf = 0; leaf < projections->size(); ++leaf)
                if (!claim(valueId, leaf))
                    return failCheck("two kernel results write one leaf of a scalar Program gradient");
            result.slot = gradientSlot(valueId);
            if (!fits(0, fragment_.slots[result.slot].element_size))
                return failCheck("kernel result is larger than its Program gradient");
            return Checked{ok()};
        }

        if (count != projections->size())
            return failCheck("a kernel result's leaves disagree with its projections");
        const nlohmann::json *projection = nullptr;
        for (const nlohmann::json &candidate : *projections) {
            uint32_t physical = 0;
            if (readUint(member(candidate, "physical_leaf"), physical) && physical == ordinal)
                projection = &candidate;
        }
        uint32_t valueId = 0;
        uint32_t leaf = 0;
        if (!projection || !readUint(member(*projection, "value"), valueId) ||
            (member(*projection, "leaf") && !readUint(member(*projection, "leaf"), leaf)))
            return failCheck("a kernel result leaf has no projection");
        if (gradientSlot(valueId) == kNoSlot || !leavesOf(valueId))
            return failCheck("kernel results other than scalar Program gradients are not linked yet");
        const nlohmann::json *leaves = leavesOf(valueId);
        if (leaf >= leaves->size())
            return failCheck("a kernel result projects into no leaf of its Value");
        const nlohmann::json &row = (*leaves)[leaf];
        uint32_t byteOffset = 0;
        uint32_t scalars = 0;
        const std::optional<VernonDataType> dtype = runtime::pipelineDataType(readString(member(row, "dtype")));
        if (!readUint(member(row, "byte_offset"), byteOffset) || !readUint(member(row, "scalar_count"), scalars) ||
            !dtype)
            return failCheck("a scalar Program gradient leaf has no layout");
        if (!claim(valueId, leaf))
            return failCheck("two kernel results write one leaf of a scalar Program gradient");
        result.slot = gradientSlot(valueId);
        // A leaf entry's copies are in the leaf's own bytes; they land at the leaf's offset in its Value.
        if (!fits(0, uint64_t{scalars} * scalarBytes(*dtype)))
            return failCheck("kernel result is larger than its Program gradient leaf");
        for (runtime::ValueByteCopy &copy : result.copies)
            copy.canonicalOffset += byteOffset;
        if (!fits(0, fragment_.slots[result.slot].element_size))
            return failCheck("kernel result is larger than its Program gradient");
        return Checked{ok()};
    }

    Checked addNode(const nlohmann::json &node) {
        const nlohmann::json *operation = member(node, "operation");
        if (!operation || readString(member(*operation, "tag")) != "compute")
            return failCheck("graphics nodes are not linked yet");
        const nlohmann::json *workgroups = member(*operation, "workgroups");
        uint32_t grid[3]{};
        // Per axis, the scalar slot whose u32 the caller passes as the axis's workgroup count, or kNoSlot.
        uint32_t gridSlot[3]{kNoSlot, kNoSlot, kNoSlot};
        if (!workgroups || !workgroups->is_array() || workgroups->size() != 3)
            return failCheck("Program node has no grid");
        for (unsigned axis = 0; axis < 3; ++axis) {
            if (readUint(&(*workgroups)[axis], grid[axis]) && grid[axis])
                continue;
            const nlohmann::json *control = member((*workgroups)[axis], "control");
            uint32_t valueId = 0;
            if (!control || !readUint(member(*control, "value"), valueId) || valueId >= valueSlot_.size() ||
                valueSlot_[valueId] == kNoSlot)
                return failCheck("grids computed from Values other than Program arguments are not linked yet");
            gridSlot[axis] = valueSlot_[valueId];
            VernonScheduleSlot &slot = fragment_.slots[gridSlot[axis]];
            if (!typed_[gridSlot[axis]]) {
                slot.dtype = VERNON_DATA_U32;
                slot.size = slot.element_size = sizeof(uint32_t);
                typed_[gridSlot[axis]] = true;
            } else if (slot.size != sizeof(uint32_t) ||
                       (slot.dtype != VERNON_DATA_U32 && slot.dtype != VERNON_DATA_I32)) {
                return failCheck("Program grid argument is not a 32-bit integer");
            }
        }
        const bool callerGrid = gridSlot[0] != kNoSlot || gridSlot[1] != kNoSlot || gridSlot[2] != kNoSlot;
        const std::string stage = readString(member(node, "stage"));
        const auto found = stages_.find(stage);
        if (found == stages_.end())
            return failCheck("Program Stage '" + stage + "' has no kernel");
        const Fragment &kernel = found->second;
        if (kernel.operations.size() != 1 || !kernel.callerGrid || kernel.executables.size() != 1 ||
            kernel.slotArguments.size() != kernel.slots.size())
            return failCheck("Program Stage '" + stage + "' is not one kernel");

        std::vector<Bound> bound(kernel.slots.size());
        std::vector<Carrier> carriers;
        for (size_t index = 0; index < kernel.slots.size(); ++index)
            if (auto checked = bindKernelSlot(kernel, index, node, bound[index], carriers); checked.isErr())
                return checked;

        // A zeroed transient is filled just before the node that first uses it, a carrier's Storage before the gather.
        std::vector<Bound> filled = bound;
        for (const Carrier &carrier : carriers)
            filled.push_back(carrier.storage);
        for (const Bound &target : filled) {
            if (!target.transient || touched_[target.index] || !fragment_.transients[target.index].zeroed)
                continue;
            const uint64_t at = alignUp(end_, alignof(uint64_t));
            end_ = at + 2 * sizeof(uint64_t);
            Dispatch fill;
            fill.kind = OperationKind::Fill;
            fill.argumentsOffset = at;
            fill.argumentsSize = 2 * sizeof(uint64_t);
            fill.accesses.push_back({true, target.index, VERNON_SCHEDULE_ACCESS_WRITE});
            fragment_.transientFields.push_back({target.index, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, at});
            fragment_.transientFields.push_back({target.index, kTransientFieldBytes, 0, 8, at + 8});
            touch(target.index);
            fragment_.operations.push_back(std::move(fill));
        }
        for (const Carrier &carrier : carriers)
            if (carrier.gather)
                addLeafCopy(carrier, false);
        TapeUse tape = kernel.operations.front().tape;
        if (tape.present())
            if (auto bound = bindTape(node, tape); bound.isErr())
                return bound;
        const uint64_t base = alignUp(end_, 16);
        end_ = base + kernel.recordSize;
        image_.resize(end_, 0);
        std::copy(kernel.recordTemplate.begin(), kernel.recordTemplate.end(), image_.begin() + base);
        std::vector<hal::RecordAddress> recordAddresses;
        std::vector<InvocationExtent> invocationExtents;
        for (size_t index = 0; index < kernel.slots.size(); ++index) {
            const VernonScheduleSlot &declared = kernel.slots[index];
            const uint64_t at = base + declared.offset;
            const Bound &target = bound[index];
            if (target.tapeCarrier != DeviceTapeCarrier::None) {
                tape.fields.push_back(
                    {target.tapeCarrier, {0, VERNON_SCHEDULE_FIELD_ADDRESS, 0, sizeof(uint64_t), at}});
                continue;
            }
            if (target.record) {
                recordAddresses.push_back({at, target.index});
                continue;
            }
            if (target.constant) {
                fragment_.constantAddresses.push_back({at, target.index});
                continue;
            }
            if (target.transient) {
                fragment_.transientFields.push_back({target.index, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, at});
                continue;
            }
            VernonScheduleSlot &slot = fragment_.slots[target.index];
            if (!placed_[target.index]) {
                slot.offset = at;
                placed_[target.index] = true;
            } else {
                const uint32_t kind = slot.kind == VERNON_SCHEDULE_SLOT_SCALAR ? VERNON_SCHEDULE_FIELD_VALUE
                                                                               : VERNON_SCHEDULE_FIELD_ADDRESS;
                fragment_.fields.push_back({target.index, kind, 0, static_cast<uint32_t>(slot.size), at});
            }
        }
        for (const VernonScheduleField &field : kernel.fields) {
            const Bound &target = bound[field.slot];
            VernonScheduleField moved = field;
            moved.slot = target.index;
            moved.offset += base;
            if (target.tapeCarrier != DeviceTapeCarrier::None) {
                moved.slot = 0;
                tape.fields.push_back({target.tapeCarrier, moved});
                continue;
            }
            if (target.constant) {
                if (auto placed = placeConstantField(moved, kernel.shapes[field.slot],
                                                     target.record ? &recordAddresses : nullptr);
                    placed.isErr())
                    return placed;
                continue;
            }
            if (kernel.slotInvocationCarrier[field.slot] &&
                (field.kind == VERNON_SCHEDULE_FIELD_EXTENT || field.kind == VERNON_SCHEDULE_FIELD_STRIDE)) {
                if (field.dimension == 0) {
                    const uint32_t *workgroup = kernel.executables.front().workgroup;
                    const uint64_t volume = uint64_t{workgroup[0]} * workgroup[1] * workgroup[2];
                    if (callerGrid && field.kind == VERNON_SCHEDULE_FIELD_EXTENT) {
                        if (moved.size != sizeof(uint32_t) && moved.size != sizeof(uint64_t))
                            return failCheck("autodiff carrier extent is neither 4 nor 8 bytes");
                        invocationExtents.push_back({moved.offset, moved.size, volume});
                        continue;
                    }
                    const uint64_t invocations = uint64_t{grid[0]} * grid[1] * grid[2] * volume;
                    const int64_t value =
                        field.kind == VERNON_SCHEDULE_FIELD_EXTENT ? static_cast<int64_t>(invocations) : 0;
                    if (auto written = writeImage(moved, value); written.isErr())
                        return written;
                    continue;
                }
                --moved.dimension;
            }
            (target.transient ? fragment_.transientFields : fragment_.fields).push_back(moved);
        }

        Dispatch dispatch = kernel.operations.front();
        for (ResultUse &result : dispatch.results) {
            const auto sameResult = [&](const ResultUse &other) { return other.result == result.result; };
            // Physical leaves are numbered in frame order, which need not be the order the kernel reflects them in.
            const auto ordinal = static_cast<size_t>(
                std::count_if(dispatch.results.begin(), dispatch.results.end(), [&](const ResultUse &other) {
                    return sameResult(other) && other.frameOffset < result.frameOffset;
                }));
            const auto count =
                static_cast<size_t>(std::count_if(dispatch.results.begin(), dispatch.results.end(), sameResult));
            if (auto bound = bindResult(node, result, ordinal, count); bound.isErr())
                return bound;
        }
        // A constant is only read and never written, so it orders nothing; a tape orders its forward and backward
        // dispatches by schedule order alone, as a CPU tape does.
        dispatch.accesses.erase(std::remove_if(dispatch.accesses.begin(), dispatch.accesses.end(),
                                               [&](const BufferAccess &access) {
                                                   return bound[access.index].constant ||
                                                          bound[access.index].tapeCarrier != DeviceTapeCarrier::None;
                                               }),
                                dispatch.accesses.end());
        for (BufferAccess &access : dispatch.accesses) {
            const Bound &target = bound[access.index];
            access.transient = target.transient;
            access.index = target.index;
            if (target.transient)
                touch(target.index);
        }
        for (const ResultUse &result : dispatch.results)
            dispatch.accesses.push_back({false, result.slot, VERNON_SCHEDULE_ACCESS_WRITE});
        if (kernel.elementGrid) {
            size_t first = kernel.slots.size();
            for (size_t index = 0; index < kernel.slots.size(); ++index)
                if (kernel.slots[index].kind == VERNON_SCHEDULE_SLOT_BUFFER &&
                    (first == kernel.slots.size() || kernel.slotArguments[index] < kernel.slotArguments[first]))
                    first = index;
            if (first == kernel.slots.size())
                return failCheck("Program built-in has no TensorView argument");
            const Bound &target = bound[first];
            dispatch.gridElements =
                target.transient ? fragment_.transients[target.index].extents : fragment_.extents[target.index];
            grid[0] = grid[1] = grid[2] = 1;
        }
        for (unsigned axis = 0; axis < 3; ++axis) {
            // The kernel runs one workgroup on a unit axis, so a Value there (a 1-D kernel's `__grid_y`) must be 1.
            if (dispatch.unitGridAxes[axis] && gridSlot[axis] != kNoSlot) {
                grid[axis] = 1;
                dispatch.unitGridSlots.emplace_back(axis, gridSlot[axis]);
            }
            if (dispatch.unitGridAxes[axis] && grid[axis] != 1)
                return failCheck("Program node grid axis must equal 1");
        }
        auto executable = executableOfStage_.find(stage);
        if (executable == executableOfStage_.end()) {
            executable = executableOfStage_.emplace(stage, static_cast<uint32_t>(fragment_.executables.size())).first;
            fragment_.executables.push_back(kernel.executables.front());
        }
        dispatch.executable = executable->second;
        dispatch.argumentsOffset += base;
        dispatch.gridOffset += base;
        for (hal::RecordAddress &address : dispatch.recordAddresses) {
            address.at += base;
            address.target += base;
        }
        dispatch.recordAddresses.insert(dispatch.recordAddresses.end(), recordAddresses.begin(), recordAddresses.end());
        dispatch.invocationExtents = std::move(invocationExtents);
        if (tape.present()) {
            if (tape.device && std::none_of(tape.fields.begin(), tape.fields.end(), [](const DeviceTapeField &field) {
                    return field.carrier == DeviceTapeCarrier::Data;
                }))
                return failCheck("GPU kernel's autodiff tape has no tape words");
            tape.allocatorOffset = tape.allocatorOffset == UINT64_MAX ? UINT64_MAX : tape.allocatorOffset + base;
            if (tape.rootOffset != UINT64_MAX)
                tape.rootOffset += base;
            dispatch.tape = tape;
        }
        std::memcpy(image_.data() + dispatch.gridOffset, grid, sizeof(grid));
        for (unsigned axis = 0; axis < 3; ++axis) {
            if (gridSlot[axis] == kNoSlot)
                continue;
            if (dispatch.unitGridAxes[axis]) {
                if (!placed_[gridSlot[axis]]) {
                    fragment_.slots[gridSlot[axis]].offset = alignUp(end_, alignof(uint32_t));
                    end_ = fragment_.slots[gridSlot[axis]].offset + sizeof(uint32_t);
                    image_.resize(end_, 0);
                    placed_[gridSlot[axis]] = true;
                }
                continue;
            }
            const uint64_t at = dispatch.gridOffset + axis * sizeof(uint32_t);
            if (!placed_[gridSlot[axis]]) {
                fragment_.slots[gridSlot[axis]].offset = at;
                placed_[gridSlot[axis]] = true;
            } else {
                fragment_.fields.push_back(
                    {gridSlot[axis], VERNON_SCHEDULE_FIELD_VALUE, 0, static_cast<uint32_t>(sizeof(uint32_t)), at});
            }
        }
        fragment_.operations.push_back(std::move(dispatch));
        for (const Carrier &carrier : carriers)
            if (carrier.scatter)
                addLeafCopy(carrier, true);
        return Checked{ok()};
    }

    // A copy's arguments (hal::LeafCopy): the Storage's address, the carrier's, then the Storage view's offset,
    // extents and strides, each a field of the slot or transient as a kernel binding's are.
    void addLeafCopy(const Carrier &carrier, bool scatter) {
        Dispatch copy;
        copy.kind = OperationKind::Copy;
        copy.copy = carrier.copy;
        copy.copy.scatter = scatter;
        const uint64_t at = alignUp(end_, alignof(uint64_t));
        end_ = at + copy.copy.argumentsSize();
        image_.resize(end_, 0);
        copy.argumentsOffset = at;
        copy.argumentsSize = copy.copy.argumentsSize();
        const Bound &storage = carrier.storage;
        const uint32_t rank = carrier.copy.rank;
        std::vector<VernonScheduleField> view{{storage.index, VERNON_SCHEDULE_FIELD_OFFSET, 0, 8, at + 16}};
        for (uint32_t dimension = 0; dimension < rank; ++dimension) {
            view.push_back(
                {storage.index, VERNON_SCHEDULE_FIELD_EXTENT, dimension, 8, at + 24 + 8 * uint64_t{dimension}});
            view.push_back({storage.index, VERNON_SCHEDULE_FIELD_STRIDE, dimension, 8,
                            at + 24 + 8 * (uint64_t{rank} + dimension)});
        }
        if (storage.transient) {
            fragment_.transientFields.push_back({storage.index, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, at});
            fragment_.transientFields.insert(fragment_.transientFields.end(), view.begin(), view.end());
            touch(storage.index);
        } else {
            if (!placed_[storage.index]) {
                fragment_.slots[storage.index].offset = at;
                placed_[storage.index] = true;
            } else {
                fragment_.fields.push_back({storage.index, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, at});
            }
            fragment_.fields.insert(fragment_.fields.end(), view.begin(), view.end());
        }
        fragment_.transientFields.push_back({carrier.transient, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, at + 8});
        touch(carrier.transient);
        const uint32_t read = VERNON_SCHEDULE_ACCESS_READ;
        const uint32_t write = VERNON_SCHEDULE_ACCESS_WRITE;
        copy.accesses.push_back({storage.transient, storage.index, scatter ? write : read});
        copy.accesses.push_back({true, carrier.transient, scatter ? read : write});
        fragment_.operations.push_back(std::move(copy));
    }

    // Zeroes a zeroed slot's buffer once.
    void addSlotFill(uint32_t slot) {
        if (slot >= slotZeroed_.size() || !slotZeroed_[slot] || slotFilled_[slot])
            return;
        const uint64_t at = alignUp(end_, alignof(uint64_t));
        end_ = at + 2 * sizeof(uint64_t);
        Dispatch fill;
        fill.kind = OperationKind::Fill;
        fill.argumentsOffset = at;
        fill.argumentsSize = 2 * sizeof(uint64_t);
        fill.accesses.push_back({false, slot, VERNON_SCHEDULE_ACCESS_WRITE});
        fragment_.fields.push_back({slot, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, at});
        fragment_.fields.push_back({slot, VERNON_SCHEDULE_FIELD_BYTES, 0, 8, at + 8});
        slotFilled_[slot] = true;
        fragment_.operations.push_back(std::move(fill));
    }

    Checked checkOutputs(const nlohmann::json &graph) const {
        const nlohmann::json *outputs = member(graph, "outputs");
        if (!outputs || !outputs->is_array())
            return failCheck("Program graph has no outputs");
        for (const nlohmann::json &output : *outputs) {
            uint32_t valueId = 0;
            uint32_t storageId = 0;
            if (readString(member(output, "tag")) != "user_output" || !readUint(member(output, "value"), valueId) ||
                !value(valueId))
                return failCheck("Program outputs other than published Storages are not linked yet");
            if (resultSlot_[valueId] != kNoSlot) {
                const std::vector<bool> &bound = resultLeaves_[valueId];
                if (bound.empty())
                    return failCheck("no kernel result writes a scalar Program gradient");
                if (std::find(bound.begin(), bound.end(), false) != bound.end())
                    return failCheck("kernel results write part of a scalar Program gradient");
                continue;
            }
            if (!readUint(member(*value(valueId), "storage"), storageId) || storageId >= storageSlot_.size() ||
                storageSlot_[storageId] == kNoSlot)
                return failCheck("Program outputs other than published Storages are not linked yet");
        }
        return Checked{ok()};
    }

    const nlohmann::json &program_;
    const std::map<std::string, Fragment> &stages_;
    const nlohmann::json *values_{};
    const nlohmann::json *storages_{};
    Fragment fragment_;
    std::vector<uint8_t> image_;
    uint64_t end_{};
    std::vector<uint32_t> storageSlot_;
    std::vector<uint32_t> storageTransient_;
    std::vector<uint32_t> allocationRank_;
    std::vector<bool> outputStorage_;
    std::vector<uint32_t> valueSlot_;
    // Per Value, the slot of the scalar gradient kernel results commit to, or kNoSlot, and which of its leaves they
    // write (empty until one does).
    std::vector<uint32_t> resultSlot_;
    std::vector<std::vector<bool>> resultLeaves_;
    std::vector<bool> placed_;
    std::vector<bool> typed_;
    // Per slot, the Value it carries, or kNoSlot for an exported allocation or a tape.
    std::vector<uint32_t> slotValue_;
    std::vector<bool> touched_;
    std::vector<bool> transientTyped_;
    // Per slot, whether its buffer starts as zeros, and whether a fill already zeroes it.
    std::vector<bool> slotZeroed_;
    std::vector<bool> slotFilled_;
    // The slot and the captured Value of each exposed Storage the graph captures rather than allocates.
    std::vector<std::pair<uint32_t, uint32_t>> exposedCaptures_;
    std::map<std::string, uint32_t> executableOfStage_;
};

} // namespace

Result<Fragment, std::string> programFragment(const nlohmann::json &program, const std::string &graph,
                                              const std::map<std::string, Fragment> &stages,
                                              const std::vector<uint32_t> &zeroed,
                                              const std::vector<uint32_t> &exposed) {
    return ProgramFragmentBuilder(program, stages).build(graph, zeroed, exposed);
}

} // namespace vernon::schedule
