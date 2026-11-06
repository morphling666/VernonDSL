#include "VernonSchedule.h"

#include "hal/cpu/cpu_device.h"
#include "hal/cuda/cuda_device.h"
#include "hal/native_buffers.h"
#include "hal/opengl/opengl_device.h"
#if VERNON_HAS_VULKAN_RUNTIME
#include "hal/vulkan/vulkan_device.h"
#endif
#include "runtime/autodiff/runtime_gpu_replay.h"
#include "runtime/backend_cpu.h"
#include "runtime/capi/boundary.h"
#include "runtime/capi/program_state.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_python_bridge.h"
#include "runtime/runtime_state.h"
#include "runtime/schedule/executor.h"
#include "runtime/schedule/fragment.h"
#include "runtime/schedule/linker.h"
#include "runtime/schedule/program_arguments.h"
#include "runtime/schedule/tape.h"
#include "runtime/schedule/vjp_plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct VernonLinkedSchedule {
    VernonRuntimeContext *context{};
    std::shared_ptr<vernon::hal::Device> device;
    vernon::schedule::LinkedSchedule linked;
    vernon::schedule::Recording recording;
    uint64_t gridOffset{};
    // Set for a compute Program linked from its executable: the caller's record holds its arguments, which each call
    // writes into `inner`, a record of `linked`. The slot queries then describe the caller's record.
    std::optional<vernon::schedule::ProgramArguments> arguments;
    std::vector<uint8_t> argumentTemplate;
    std::vector<uint8_t> inner;
    // Per slot of the caller's record, the Program boundary slot it takes, or UINT32_MAX.
    std::vector<uint32_t> slotBoundaries;
    // Per ARGUMENT slot, the linked slot it writes with the union of its sites' accesses, and that slot's declared
    // extents (vernonScheduleArgumentSlot).
    std::vector<VernonScheduleSlot> argumentSlots;
    std::vector<std::vector<int64_t>> argumentShapes;
    // Whether a GPU kernel writes an autodiff tape, and the largest per-lane stride a pullback found its forward
    // needed; later forward calls size their tapes with it.
    bool writesDeviceTapes{};
    std::atomic<uint64_t> tapeStride{0};
    // A VJP forward's boundary slots that are outputs, which a replay of the forward writes to scratch.
    std::set<uint32_t> outputBoundaries;
};

// The residuals of one linked VJP call in one buffer, with the forward call's record: the forward schedule writes each
// residual at its entry's offset and the backward schedule reads it there, and reads the forward arguments again. An
// autodiff tape residual is a VernonScheduleTape rather than bytes in it. An entry the block does not hold is one a
// checkpointed call wrote where nothing reads it, or one a recompute's backward reads from the forward call's block.
struct VernonScheduleResiduals {
    struct Entry {
        std::vector<int64_t> shape;
        uint64_t elementSize{};
        // Into the block's bytes, or into `tapes` for a tape.
        uint64_t offset{};
        uint8_t *address{};
        // On a graphics device, which names the entry by reference: the argument's extents and dense byte strides.
        std::vector<uint64_t> extents;
        std::vector<int64_t> byteStrides;
        bool tape{};
        bool held{true};
    };

    static constexpr uint64_t kAlignment = 256;

    VernonScheduleResiduals() = default;
    VernonScheduleResiduals(const VernonScheduleResiduals &) = delete;
    VernonScheduleResiduals &operator=(const VernonScheduleResiduals &) = delete;
    // A block outlives every call that reads it. Waiting here observes no result, so a failed call reports nothing
    // through it; the caller's wait on the call's point does.
    ~VernonScheduleResiduals() {
        if (halDevice && lastPoint.load())
            (void)halDevice->wait(lastPoint.load());
    }

    void readBy(const std::shared_ptr<vernon::hal::Device> &by, uint64_t point) const {
        {
            std::lock_guard<std::mutex> lock(readers);
            if (!halDevice)
                halDevice = by;
        }
        uint64_t last = lastPoint.load();
        while (last < point && !lastPoint.compare_exchange_weak(last, point)) {
        }
    }

    std::vector<uint8_t> forwardRecord;
    // What the arguments of `forwardRecord` a backward call reads point at: shapes, strides and scalar bytes.
    std::vector<std::unique_ptr<uint8_t[]>> captured;
    std::vector<Entry> entries;
    std::vector<std::shared_ptr<VernonScheduleTape>> tapes;
    VernonScheduleCheckpoint checkpoint{sizeof(VernonScheduleCheckpoint)};
    uint8_t *base{};
    std::vector<uint8_t> host;
    // On a GPU, freed once the destructor has waited for the calls that read it.
    std::unique_ptr<vernon::hal::Buffer> device;
    mutable std::mutex readers;
    mutable std::shared_ptr<vernon::hal::Device> halDevice;
    mutable std::atomic<uint64_t> lastPoint{0};
    // The forward outputs of a replay, which nothing reads back, and on a graphics device the extents and dense byte
    // strides of the arguments that name them.
    std::vector<std::unique_ptr<vernon::hal::Buffer>> scratch;
    std::vector<Entry> scratchViews;
    // Set by the first backward call once the forward's device tapes held every lane; `replayed` is the forward run
    // again with larger tapes when one did not, and the backward calls read it instead. Its forward record points
    // into `captured`, so it is declared after it.
    mutable std::mutex checking;
    mutable bool checked{};
    mutable std::unique_ptr<VernonScheduleResiduals> replayed;
};

struct VernonScheduleVjp {
    std::unique_ptr<VernonLinkedSchedule> forward;
    std::unique_ptr<VernonLinkedSchedule> backward;
    // Per ARGUMENT slot of `backward`, whether it takes a cotangent or a gradient.
    std::vector<VernonProgramBoundaryRole> backwardRoles;
    // Per forward node, the residual block entries of what it left there (vernonScheduleVjpGetPass).
    struct Pass {
        uint32_t node{};
        std::string name;
        std::vector<uint32_t> entries;
    };
    std::vector<Pass> passes;
    // Set by vernonScheduleLinkCheckpointedVjp.
    struct Checkpointing {
        VernonScheduleCheckpointOptions options{};
        // Per residual block entry, the position in forward node order of the node that produces it.
        std::vector<uint32_t> entryNodes;
        // The forward nodes up to the last that records an autodiff tape.
        uint32_t tapeNodes{};
        // Per prefix length k, the forward schedule's operations of nodes 0 through k - 1, with its calling convention.
        std::map<uint32_t, std::unique_ptr<VernonLinkedSchedule>> recomputes;
        // Where calls write residuals nothing reads: what a forward call does not keep and what a recompute did not
        // produce. Each entry has its own range, so calls in flight together write garbage only where nothing reads.
        std::unique_ptr<vernon::hal::Buffer> discard;
        uint64_t discardCapacity{};
        std::vector<std::unique_ptr<vernon::hal::Buffer>> retired;
        std::vector<std::shared_ptr<VernonScheduleTape>> discardTapes;
    };
    std::optional<Checkpointing> checkpointing;
};

namespace {

using vernon::runtime::capi::fail;

std::atomic<size_t> failedScheduleWaits{0};

bool consumeFailedScheduleWait() {
    size_t remaining = failedScheduleWaits.load();
    while (remaining && !failedScheduleWaits.compare_exchange_weak(remaining, remaining - 1)) {
    }
    return remaining != 0;
}

bool linksComputeExecutables(const VernonRuntimeContext &context) {
    return context.backend == VERNON_RUNTIME_CPU || context.backend == VERNON_RUNTIME_CUDA;
}

vernon::Result<vernon::schedule::BufferRange, std::string> resolveBuffer(void *opaque, const VernonTensorView &view) {
    using Resolved = vernon::Result<vernon::schedule::BufferRange, std::string>;
    if (view.storage == VERNON_TENSOR_HOST) {
        if (!view.host_data)
            return Resolved{vernon::err(std::string("host Tensor has no data"))};
        return Resolved{vernon::ok(vernon::schedule::BufferRange{
            const_cast<uint8_t *>(static_cast<const uint8_t *>(view.host_data)), view.byte_size})};
    }
    const VernonRuntimeProviderResourceReference &resource = view.resource;
    if (view.storage != VERNON_TENSOR_DEVICE_RESOURCE || !resource.resource.value || view.byte_size > resource.size)
        return Resolved{vernon::err(std::string("device Tensor view is invalid"))};
    const auto &context = *static_cast<const VernonRuntimeContext *>(opaque);
    const auto base = static_cast<uintptr_t>(resource.resource.value);
    if (context.backend == VERNON_RUNTIME_CUDA && !vernon::hal::isNativeBufferReference(resource.identity))
        return Resolved{vernon::err(std::string("CUDA Tensor names no CUDA HAL buffer"))};
    return Resolved{
        vernon::ok(vernon::schedule::BufferRange{reinterpret_cast<uint8_t *>(base + resource.offset), view.byte_size})};
}

std::shared_ptr<vernon::hal::Device> contextDevice(VernonRuntimeContext &context, std::string &error) {
    std::lock_guard<std::mutex> lock(context.halDeviceMutex);
    if (context.halDevice)
        return context.halDevice;
    if (context.backend == VERNON_RUNTIME_CPU) {
        context.halDevice = std::shared_ptr<vernon::hal::Device>(
            vernon::hal::createCpuDevice(vernon::runtime::cpuWorkgroupScheduler(context)).release());
    } else if (context.backend == VERNON_RUNTIME_CUDA) {
        error = "a CUDA context is created over its HAL device";
        return nullptr;
    } else if (vernon::runtime::isOpenGLBackend(context.backend)) {
        context.halDevice = std::shared_ptr<vernon::hal::Device>(vernon::hal::createOpenGLDevice(context).release());
    } else if (context.backend == VERNON_RUNTIME_VULKAN) {
#if VERNON_HAS_VULKAN_RUNTIME
        context.halDevice = std::shared_ptr<vernon::hal::Device>(vernon::hal::createVulkanDevice(context).release());
#endif
        if (!context.halDevice)
            error = "linked schedules do not run on a Vulkan device whose host owns the native objects";
    } else {
        error = "linked schedules run on the CPU, CUDA, Vulkan and OpenGL only yet";
        return nullptr;
    }
    if (!context.halDevice && error.empty())
        error = "cannot create the HAL device";
    return context.halDevice;
}

vernon::Result<vernon::schedule::Fragment, std::string>
kernelFragment(const VernonRuntimeContext &context, const VernonScheduleKernelDescriptor &descriptor) {
    using Built = vernon::Result<vernon::schedule::Fragment, std::string>;
    if (descriptor.struct_size < sizeof(descriptor) || !descriptor.reflection.data || !descriptor.entry.data)
        return Built{vernon::err(std::string("schedule kernel descriptor is invalid"))};
    const nlohmann::json reflection = nlohmann::json::parse(
        descriptor.reflection.data, descriptor.reflection.data + descriptor.reflection.size, nullptr, false);
    if (reflection.is_discarded())
        return Built{vernon::err(std::string("schedule kernel reflection is invalid JSON"))};
    const std::string entry(descriptor.entry.data, descriptor.entry.size);
    if (context.backend == VERNON_RUNTIME_CUDA)
        return vernon::schedule::cudaKernelFragment(
            reflection, entry,
            descriptor.cuda_artifact
                ? std::string(static_cast<const char *>(descriptor.cuda_artifact), descriptor.cuda_artifact_size)
                : std::string());
    return vernon::schedule::cpuKernelFragment(reflection, entry, descriptor.cpu_entry, descriptor.cpu_chunk_entry);
}

// The fragment of graph `graph` of a resolved Program on the context's device: kernel fragments on the CPU and CUDA,
// passes on a graphics device.
vernon::Result<vernon::schedule::Fragment, std::string> executableFragment(const VernonRuntimeContext &context,
                                                                           VernonProgramExecutable &executable,
                                                                           const std::string &graph,
                                                                           const std::vector<uint32_t> &exposed) {
    if (executable.composition)
        return vernon::Result<vernon::schedule::Fragment, std::string>{
            vernon::err(std::string("a composed Program graph links as a whole, not as a node of another graph"))};
    if (linksComputeExecutables(context))
        return vernon::schedule::computeProgramFragment(executable, graph, exposed);
    return vernon::schedule::graphicsProgramFragment(executable, graph, exposed);
}

vernon::Result<vernon::schedule::Fragment, std::string>
programFragment(const VernonRuntimeContext &context, const VernonScheduleProgramDescriptor &descriptor) {
    using Built = vernon::Result<vernon::schedule::Fragment, std::string>;
    if (descriptor.struct_size >= sizeof(descriptor) && descriptor.executable) {
        if (!descriptor.graph.data || (descriptor.exposed_storage_count && !descriptor.exposed_storages))
            return Built{vernon::err(std::string("schedule Program descriptor is invalid"))};
        const std::string graph(descriptor.graph.data, descriptor.graph.size);
        return executableFragment(
            context, *descriptor.executable, graph,
            std::vector<uint32_t>(descriptor.exposed_storages,
                                  descriptor.exposed_storages + descriptor.exposed_storage_count));
    }
    if (descriptor.struct_size < sizeof(descriptor) || !descriptor.program.data || !descriptor.graph.data ||
        (descriptor.stage_count && !descriptor.stages) ||
        (descriptor.zeroed_storage_count && !descriptor.zeroed_storages) ||
        (descriptor.exposed_storage_count && !descriptor.exposed_storages))
        return Built{vernon::err(std::string("schedule Program descriptor is invalid"))};
    const nlohmann::json program = nlohmann::json::parse(
        descriptor.program.data, descriptor.program.data + descriptor.program.size, nullptr, false);
    if (program.is_discarded())
        return Built{vernon::err(std::string("schedule Program is invalid JSON"))};
    std::map<std::string, vernon::schedule::Fragment> stages;
    for (size_t index = 0; index < descriptor.stage_count; ++index) {
        const VernonScheduleStage &stage = descriptor.stages[index];
        if (!stage.name.data)
            return Built{vernon::err(std::string("schedule Program Stage has no name"))};
        auto fragment = kernelFragment(context, stage.kernel);
        if (fragment.isErr())
            return fragment;
        stages.emplace(std::string(stage.name.data, stage.name.size), std::move(fragment).value());
    }
    const std::vector<uint32_t> zeroed(descriptor.zeroed_storages,
                                       descriptor.zeroed_storages + descriptor.zeroed_storage_count);
    const std::vector<uint32_t> exposed(descriptor.exposed_storages,
                                        descriptor.exposed_storages + descriptor.exposed_storage_count);
    return vernon::schedule::programFragment(program, std::string(descriptor.graph.data, descriptor.graph.size), stages,
                                             zeroed, exposed);
}

// Links and records the graph of `fragments` on the context's device.
VernonLinkedSchedule *linkFragments(VernonRuntimeContext *context,
                                    const std::vector<const vernon::schedule::Fragment *> &fragments,
                                    const std::vector<vernon::schedule::Connection> &connections) {
    std::string error;
    std::shared_ptr<vernon::hal::Device> device = contextDevice(*context, error);
    if (!device) {
        (void)fail(context, error, VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    auto linked = vernon::schedule::link(fragments, connections);
    if (linked.isErr()) {
        (void)fail(context, linked.error(), VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    auto recording = vernon::schedule::record(*device, linked.value());
    if (recording.isErr()) {
        (void)fail(context, recording.error(), VERNON_STATUS_INTERNAL_ERROR);
        return nullptr;
    }
    auto schedule = std::make_unique<VernonLinkedSchedule>();
    schedule->context = context;
    schedule->device = std::move(device);
    schedule->linked = std::move(linked).value();
    schedule->recording = std::move(recording).value();
    schedule->gridOffset = schedule->linked.callerGrid ? schedule->linked.operations.front().gridOffset : UINT64_MAX;
    schedule->writesDeviceTapes = std::any_of(schedule->linked.operations.begin(), schedule->linked.operations.end(),
                                              [](const vernon::schedule::LinkedDispatch &operation) {
                                                  return operation.tape.device && operation.tape.writes;
                                              });
    return schedule.release();
}

VernonLinkedSchedule *linkFragment(VernonRuntimeContext *context, const vernon::schedule::Fragment &fragment) {
    return linkFragments(context, {&fragment}, {});
}

// Makes `arguments` the calling convention of `schedule` once they write every slot.
vernon::Result<void, std::string> adoptArguments(VernonLinkedSchedule &schedule,
                                                 vernon::schedule::ProgramArguments arguments) {
    if (auto covered = vernon::schedule::checkCovered(arguments, schedule.linked); covered.isErr())
        return covered;
    schedule.slotBoundaries.clear();
    schedule.argumentSlots.clear();
    schedule.argumentShapes.clear();
    for (size_t index = 0; index < arguments.sites.size(); ++index) {
        const vernon::schedule::ProgramArguments::Site &site = arguments.sites[index];
        if (!index || arguments.sites[index - 1].offset != site.offset) {
            schedule.slotBoundaries.push_back(site.boundary);
            schedule.argumentSlots.emplace_back();
            schedule.argumentShapes.emplace_back();
        }
        if (site.slot == vernon::schedule::kNoSlot)
            continue;
        VernonScheduleSlot &described = schedule.argumentSlots.back();
        const uint32_t access = described.access | schedule.linked.slots[site.slot].access;
        if (!described.kind) {
            described = schedule.linked.slots[site.slot];
            schedule.argumentShapes.back() = schedule.linked.shapes[site.slot];
        }
        described.access = access;
    }
    schedule.argumentTemplate.assign(arguments.recordSize, 0);
    schedule.inner = schedule.linked.recordTemplate;
    schedule.arguments = std::move(arguments);
    schedule.gridOffset = UINT64_MAX;
    return vernon::Result<void, std::string>{vernon::ok()};
}

// The extent `extent` names in `record`: static, or the one the caller wrote for another slot.
int64_t extentIn(const vernon::schedule::LinkedSchedule &linked, const vernon::schedule::Extent &extent,
                 const uint8_t *record) {
    return extent.slot == vernon::schedule::kNoSlot
               ? extent.value
               : vernon::schedule::slotExtent(linked, record, extent.slot, extent.dimension);
}

uint64_t alignedResidual(uint64_t bytes) {
    return (bytes + VernonScheduleResiduals::kAlignment - 1) / VernonScheduleResiduals::kAlignment *
           VernonScheduleResiduals::kAlignment;
}

// The shape a residual block slot of `schedule` takes in the call whose linked record `record` already holds.
vernon::Result<std::vector<int64_t>, std::string> residualShape(const vernon::schedule::LinkedSchedule &linked,
                                                                uint32_t slot, const uint8_t *record) {
    std::vector<int64_t> shape;
    for (size_t dimension = 0; dimension < linked.extents[slot].size(); ++dimension) {
        const int64_t value = extentIn(linked, linked.extents[slot][dimension], record);
        const int64_t declared = linked.shapes[slot][dimension];
        if (value < 0 || (declared >= 0 && value != declared))
            return vernon::Result<std::vector<int64_t>, std::string>{
                vernon::err(std::string("a residual's extent contradicts its declared one"))};
        shape.push_back(value);
    }
    return vernon::Result<std::vector<int64_t>, std::string>{vernon::ok(std::move(shape))};
}

uint64_t elementCount(const std::vector<int64_t> &shape) {
    uint64_t elements = 1;
    for (int64_t extent : shape)
        elements *= static_cast<uint64_t>(extent);
    return elements;
}

// Where a call writes the residual block entries the block does not hold: per entry an address, or a tape for a tape
// entry. An entry with neither, or past their end, the block allocates.
struct ResidualSources {
    std::vector<uint8_t *> addresses;
    std::vector<std::shared_ptr<VernonScheduleTape>> tapes;
};

// Sizes and allocates the residual block of a forward call whose arguments `record` already holds.
vernon::Result<void, std::string> allocateResiduals(const VernonLinkedSchedule &schedule, const uint8_t *record,
                                                    VernonScheduleResiduals &block,
                                                    const ResidualSources *sources = nullptr) {
    using Allocated = vernon::Result<void, std::string>;
    const vernon::schedule::LinkedSchedule &linked = schedule.linked;
    uint64_t size = 0;
    for (uint32_t slot : schedule.arguments->blockSlots) {
        const size_t index = block.entries.size();
        VernonScheduleResiduals::Entry &entry = block.entries.emplace_back();
        if (linked.slots[slot].kind == VERNON_SCHEDULE_SLOT_TAPE) {
            entry.tape = true;
            entry.offset = block.tapes.size();
            if (sources && index < sources->tapes.size() && sources->tapes[index]) {
                entry.held = false;
                block.tapes.push_back(sources->tapes[index]);
            } else {
                block.tapes.push_back(std::make_shared<VernonScheduleTape>());
            }
            block.tapes.back()->requestedStride =
                std::max(block.tapes.back()->requestedStride, schedule.tapeStride.load());
            continue;
        }
        entry.elementSize = linked.slots[slot].element_size;
        auto shape = residualShape(linked, slot, record);
        if (shape.isErr())
            return Allocated{vernon::err(std::move(shape).error())};
        entry.shape = std::move(shape).value();
        if (sources && index < sources->addresses.size() && sources->addresses[index]) {
            entry.held = false;
            entry.address = sources->addresses[index];
            continue;
        }
        entry.offset = size;
        size += alignedResidual(elementCount(entry.shape) * entry.elementSize);
    }
    size = std::max(size, VernonScheduleResiduals::kAlignment);
    const auto place = [&]() {
        for (VernonScheduleResiduals::Entry &entry : block.entries)
            if (!entry.tape && entry.held)
                entry.address = block.base + entry.offset;
        return Allocated{vernon::ok()};
    };
    if (schedule.context->backend == VERNON_RUNTIME_CPU) {
        block.host.resize(static_cast<size_t>(size + VernonScheduleResiduals::kAlignment));
        const auto address = reinterpret_cast<uintptr_t>(block.host.data());
        block.base =
            block.host.data() + (VernonScheduleResiduals::kAlignment - address % VernonScheduleResiduals::kAlignment) %
                                    VernonScheduleResiduals::kAlignment;
        return place();
    }
    auto allocated = schedule.device->allocate(size);
    if (allocated.isErr())
        return Allocated{vernon::err("cannot allocate a residual block: " + allocated.error())};
    block.device = std::move(allocated).value();
    block.base = static_cast<uint8_t *>(block.device->address());
    if (block.base)
        return place();
    for (VernonScheduleResiduals::Entry &entry : block.entries) {
        if (entry.tape)
            continue;
        if (!entry.held)
            return Allocated{vernon::err(std::string("a graphics device's residuals live in their own block"))};
        entry.extents.assign(entry.shape.begin(), entry.shape.end());
        entry.byteStrides.assign(entry.shape.size(), 0);
        auto stride = static_cast<int64_t>(entry.elementSize);
        for (size_t dimension = entry.shape.size(); dimension-- > 0;) {
            entry.byteStrides[dimension] = stride;
            stride *= entry.shape[dimension];
        }
    }
    return Allocated{vernon::ok()};
}

// Binds each block slot of `schedule` to its residual, which must have the slot's element size and rank.
vernon::Result<void, std::string> bindResiduals(VernonLinkedSchedule &schedule, const VernonScheduleResiduals &block) {
    using Bound = vernon::Result<void, std::string>;
    const vernon::schedule::ProgramArguments &arguments = *schedule.arguments;
    if (arguments.blockEntries.empty() && block.entries.size() != arguments.blockSlots.size())
        return Bound{vernon::err(std::string("the residual block belongs to another linked VJP"))};
    for (size_t index = 0; index < arguments.blockSlots.size(); ++index) {
        const uint32_t slot = arguments.blockSlots[index];
        const size_t at = arguments.blockEntries.empty() ? index : arguments.blockEntries[index];
        if (at >= block.entries.size())
            return Bound{vernon::err(std::string("the residual block belongs to another linked VJP"))};
        const VernonScheduleResiduals::Entry &entry = block.entries[at];
        const VernonScheduleSlot &described = schedule.linked.slots[slot];
        if (entry.tape != (described.kind == VERNON_SCHEDULE_SLOT_TAPE))
            return Bound{vernon::err(std::string("the residual block belongs to another linked VJP"))};
        if (entry.tape) {
            VernonScheduleTape *tape = block.tapes[entry.offset].get();
            std::memcpy(schedule.inner.data() + described.offset, &tape, sizeof(tape));
            continue;
        }
        if (entry.elementSize != described.element_size || entry.shape.size() != described.rank)
            return Bound{vernon::err(std::string("a residual does not match its backward slot"))};
        if (described.kind == VERNON_SCHEDULE_SLOT_ARGUMENT) {
            if (!block.device || block.base || entry.extents.size() != entry.shape.size())
                return Bound{vernon::err(std::string("the residual block belongs to another device"))};
            const uint64_t bytes = elementCount(entry.shape) * entry.elementSize;
            VernonProgramArgument argument{};
            argument.kind = VERNON_PROGRAM_TENSOR;
            VernonTensorView &tensor = argument.tensor;
            tensor.struct_size = sizeof(tensor);
            tensor.storage = VERNON_TENSOR_DEVICE_RESOURCE;
            tensor.resource = block.device->reference(entry.offset, bytes);
            tensor.access =
                described.access & VERNON_SCHEDULE_ACCESS_WRITE ? VERNON_ACCESS_READ_WRITE : VERNON_ACCESS_READ;
            tensor.rank = described.rank;
            tensor.shape = entry.extents.empty() ? nullptr : entry.extents.data();
            tensor.byte_strides = entry.byteStrides.empty() ? nullptr : entry.byteStrides.data();
            tensor.byte_size = bytes;
            std::memcpy(schedule.inner.data() + described.offset, &argument, sizeof(argument));
            for (uint32_t field : arguments.fieldsOfSlot[slot])
                std::memcpy(schedule.inner.data() + schedule.linked.fields[field].offset, &argument, sizeof(argument));
            continue;
        }
        if (auto wrote = vernon::schedule::writeDenseBuffer(arguments, schedule.linked, slot, entry.address,
                                                            entry.shape, schedule.inner.data());
            wrote.isErr())
            return wrote;
    }
    return Bound{vernon::ok()};
}

// One call of a schedule linked from an executable: writes the caller's `record`, then a backward schedule's arguments
// it reads again from the record of the forward call that produced `block`, and the residuals of `block`, or of
// `produced` once a forward call allocates them there, at the sources `place` picks from the linked record.
using PlaceResiduals = std::function<vernon::Result<const ResidualSources *, std::string>(const uint8_t *)>;

VernonStatus callProgram(VernonLinkedSchedule &schedule, const void *record, size_t size,
                         const VernonScheduleResiduals *block, VernonScheduleResiduals *produced, uint64_t *point,
                         const PlaceResiduals &place = {}) {
    const vernon::schedule::ProgramArguments &arguments = *schedule.arguments;
    if (!record || size < arguments.recordSize)
        return fail(schedule.context, "schedule call record is smaller than its Program arguments");
    if (produced)
        block = produced;
    if ((!arguments.blockSlots.empty() || !arguments.captures.empty()) && !block)
        return fail(schedule.context, "a linked VJP schedule runs through its VJP");
    uint8_t *inner = schedule.inner.data();
    std::memcpy(inner, schedule.linked.recordTemplate.data(), schedule.inner.size());
    const auto write = [&]() -> vernon::Result<void, std::string> {
        if (auto wrote = vernon::schedule::writeProgramArguments(arguments, arguments.sites, schedule.linked,
                                                                 static_cast<const uint8_t *>(record), inner,
                                                                 resolveBuffer, schedule.context);
            wrote.isErr())
            return wrote;
        if (!arguments.captures.empty())
            if (auto wrote = vernon::schedule::writeProgramArguments(arguments, arguments.captures, schedule.linked,
                                                                     block->forwardRecord.data(), inner, resolveBuffer,
                                                                     schedule.context);
                wrote.isErr())
                return wrote;
        // A checkpointed forward call plans what it keeps even when it keeps no residuals, so it reports its plan.
        const ResidualSources *sources = nullptr;
        if (produced && place) {
            auto placed = place(inner);
            if (placed.isErr())
                return vernon::Result<void, std::string>{vernon::err(std::move(placed).error())};
            sources = placed.value();
        }
        if (arguments.blockSlots.empty())
            return vernon::Result<void, std::string>{vernon::ok()};
        if (produced)
            if (auto allocated = allocateResiduals(schedule, inner, *produced, sources); allocated.isErr())
                return allocated;
        return bindResiduals(schedule, *block);
    };
    if (auto written = write(); written.isErr())
        return fail(schedule.context, written.error());
    auto called =
        vernon::schedule::call(*schedule.device, schedule.linked, schedule.recording, inner, schedule.inner.size());
    if (called.isErr())
        return fail(schedule.context, called.error());
    if (block)
        block->readBy(schedule.device, called.value());
    if (point)
        *point = called.value();
    return VERNON_STATUS_OK;
}

// Makes the arguments of `sites` in the block's forward record point at copies the block owns: an argument's shape,
// strides and a scalar's host bytes outlive only the forward call. A view's buffer is the caller's to keep.
vernon::Result<void, std::string> keepArguments(VernonScheduleResiduals &block,
                                                const std::vector<vernon::schedule::ProgramArguments::Site> &sites,
                                                const vernon::schedule::LinkedSchedule &linked) {
    using Kept = vernon::Result<void, std::string>;
    static constexpr size_t kAlign = alignof(std::max_align_t);
    const auto aligned = [](size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; };
    struct Copy {
        size_t offset;
        VernonProgramArgument argument;
        size_t rank, leaves, hash, scalar;
    };
    std::vector<Copy> kept;
    kept.reserve(sites.size());
    size_t total = 0;
    for (const vernon::schedule::ProgramArguments::Site &site : sites) {
        if (site.slot == vernon::schedule::kNoSlot ||
            block.forwardRecord.size() < site.offset + sizeof(VernonProgramArgument))
            continue;
        Copy entry{site.offset, {}, 0, 0, 0, 0};
        std::memcpy(&entry.argument, block.forwardRecord.data() + site.offset, sizeof(entry.argument));
        const VernonTensorView &view = entry.argument.tensor;
        if (entry.argument.kind != VERNON_PROGRAM_TENSOR)
            continue;
        const VernonValueLayoutView &layout = view.element_layout;
        entry.rank = view.shape && view.byte_strides ? view.rank : 0;
        entry.leaves = layout.leaves ? layout.leaf_count : 0;
        entry.hash = layout.layout_hash.data ? layout.layout_hash.size : 0;
        const VernonScheduleSlot &slot = linked.slots[site.slot];
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
            if (view.storage != VERNON_TENSOR_HOST || !view.host_data || view.byte_offset > view.byte_size ||
                view.byte_size - view.byte_offset < slot.size)
                return Kept{vernon::err("Program parameter " + std::to_string(site.boundary) + " takes " +
                                        std::to_string(slot.size) + " host bytes")};
            entry.scalar = slot.size;
        } else if (slot.kind == VERNON_SCHEDULE_SLOT_ARGUMENT && view.storage == VERNON_TENSOR_HOST && view.host_data &&
                   view.byte_offset <= view.byte_size) {
            entry.scalar = view.byte_size - view.byte_offset;
        }
        total += aligned(entry.rank * (sizeof(uint64_t) + sizeof(int64_t)) +
                         entry.leaves * sizeof(VernonValueLeafView) + entry.hash) +
                 aligned(entry.scalar);
        kept.push_back(entry);
    }
    if (kept.empty())
        return Kept{vernon::ok()};
    // One allocation per call: a forward call keeps an argument per captured site.
    auto copies = std::make_unique<uint8_t[]>(total);
    uint8_t *at = copies.get();
    for (Copy &entry : kept) {
        VernonTensorView &view = entry.argument.tensor;
        VernonValueLayoutView &layout = view.element_layout;
        auto *shape = reinterpret_cast<uint64_t *>(at);
        auto *strides = reinterpret_cast<int64_t *>(shape + entry.rank);
        auto *leafCopy = reinterpret_cast<VernonValueLeafView *>(strides + entry.rank);
        auto *hashCopy = reinterpret_cast<char *>(leafCopy + entry.leaves);
        at += aligned(entry.rank * (sizeof(uint64_t) + sizeof(int64_t)) + entry.leaves * sizeof(VernonValueLeafView) +
                      entry.hash);
        if (entry.rank) {
            view.shape = std::copy_n(view.shape, entry.rank, shape) - entry.rank;
            view.byte_strides = std::copy_n(view.byte_strides, entry.rank, strides) - entry.rank;
        }
        if (entry.leaves)
            layout.leaves = std::copy_n(layout.leaves, entry.leaves, leafCopy) - entry.leaves;
        if (entry.hash)
            layout.layout_hash.data = std::copy_n(layout.layout_hash.data, entry.hash, hashCopy) - entry.hash;
        if (entry.scalar) {
            std::memcpy(at, static_cast<const uint8_t *>(view.host_data) + view.byte_offset, entry.scalar);
            view.host_data = at;
            view.byte_offset = 0;
            view.byte_size = entry.scalar;
            at += aligned(entry.scalar);
        }
        std::memcpy(block.forwardRecord.data() + entry.offset, &entry.argument, sizeof(entry.argument));
    }
    block.captured.push_back(std::move(copies));
    return Kept{vernon::ok()};
}

// The largest per-lane stride a device tape of `block` reported needing beyond its own, or zero once every lane fit.
// Waits for the calls that wrote the block.
vernon::Result<uint64_t, std::string> overflowedStride(const VernonScheduleResiduals &block) {
    using Required = vernon::Result<uint64_t, std::string>;
    uint64_t required = 0;
    bool waited = false;
    for (const std::shared_ptr<VernonScheduleTape> &tape : block.tapes) {
        if (!tape->device.status)
            continue;
        if (!waited) {
            if (!block.halDevice)
                return Required{vernon::err(std::string("an autodiff tape's forward call never ran"))};
            if (auto finished = block.halDevice->wait(block.lastPoint.load()); finished.isErr())
                return Required{vernon::err(std::move(finished).error())};
            waited = true;
        }
        vernon::runtime::ad::gpu::BatchSummary summary{};
        if (auto read = block.halDevice->download(*tape->device.status, &summary, sizeof(summary)); read.isErr())
            return Required{vernon::err(std::move(read).error())};
        if (summary.status == 0)
            continue;
        if (summary.status != 1 || summary.requiredBytes <= tape->device.stride)
            return Required{vernon::err(std::string("a GPU autodiff tape reported an invalid status"))};
        required = std::max<uint64_t>(required, summary.requiredBytes);
    }
    return Required{vernon::ok(required)};
}

// Runs the forward call that produced `block` again into a block of its own, from the inputs its record names, which
// copy-on-write kept as they were. Its outputs go to scratch buffers: the caller's outputs are never rewritten.
vernon::Result<std::unique_ptr<VernonScheduleResiduals>, std::string>
replayForward(VernonLinkedSchedule &forward, const VernonScheduleResiduals &block,
              const ResidualSources *sources = nullptr) {
    using Replayed = vernon::Result<std::unique_ptr<VernonScheduleResiduals>, std::string>;
    const vernon::schedule::ProgramArguments &arguments = *forward.arguments;
    const vernon::schedule::LinkedSchedule &linked = forward.linked;
    auto replay = std::make_unique<VernonScheduleResiduals>();
    replay->forwardRecord = block.forwardRecord;
    std::vector<vernon::schedule::ProgramArguments::Site> inputs;
    std::vector<vernon::schedule::ProgramArguments::Site> outputs;
    for (const vernon::schedule::ProgramArguments::Site &site : arguments.sites) {
        if (site.slot == vernon::schedule::kNoSlot)
            continue;
        const uint32_t access = linked.slots[site.slot].access;
        if (forward.outputBoundaries.count(site.boundary) || access == VERNON_SCHEDULE_ACCESS_WRITE) {
            outputs.push_back(site);
            continue;
        }
        if (access & VERNON_SCHEDULE_ACCESS_WRITE)
            return Replayed{vernon::err(std::string("a forward call that reads and writes one of its arguments "
                                                    "cannot run again"))};
        inputs.push_back(site);
    }
    uint8_t *inner = forward.inner.data();
    std::memcpy(inner, linked.recordTemplate.data(), forward.inner.size());
    if (auto wrote = vernon::schedule::writeProgramArguments(arguments, inputs, linked, replay->forwardRecord.data(),
                                                             inner, resolveBuffer, forward.context);
        wrote.isErr())
        return Replayed{vernon::err(std::move(wrote).error())};
    // Reserved, since the arguments of a graphics device's outputs point into the views.
    replay->scratchViews.reserve(outputs.size());
    for (const vernon::schedule::ProgramArguments::Site &site : outputs) {
        VernonProgramArgument argument;
        std::memcpy(&argument, replay->forwardRecord.data() + site.offset, sizeof(argument));
        const VernonTensorView &view = argument.tensor;
        if (argument.kind != VERNON_PROGRAM_TENSOR || (view.rank && !view.shape))
            return Replayed{vernon::err(std::string("a forward output has no shape to run again with"))};
        std::vector<int64_t> shape(view.shape, view.shape + view.rank);
        uint64_t elements = 1;
        for (int64_t extent : shape)
            elements *= static_cast<uint64_t>(extent);
        const VernonScheduleSlot &slot = linked.slots[site.slot];
        // A graphics output's slot has no element size: its scratch takes the layout of the caller's view.
        const bool viewed = slot.kind == VERNON_SCHEDULE_SLOT_ARGUMENT;
        if (viewed && view.rank && !view.byte_strides)
            return Replayed{vernon::err(std::string("a forward output has no strides to run again with"))};
        const uint64_t bytes = std::max<uint64_t>(viewed ? view.byte_size : elements * slot.element_size, 1);
        auto buffer = forward.device->allocate(bytes);
        if (buffer.isErr())
            return Replayed{vernon::err(std::move(buffer).error())};
        auto *address = static_cast<uint8_t *>(buffer.value()->address());
        replay->scratch.push_back(std::move(buffer).value());
        if (viewed) {
            VernonScheduleResiduals::Entry &scratch = replay->scratchViews.emplace_back();
            scratch.extents.assign(view.shape, view.shape + view.rank);
            scratch.byteStrides.assign(view.byte_strides, view.byte_strides + view.rank);
            VernonProgramArgument output = argument;
            output.tensor.storage = VERNON_TENSOR_DEVICE_RESOURCE;
            output.tensor.resource = replay->scratch.back()->reference(0, bytes);
            output.tensor.shape = scratch.extents.empty() ? nullptr : scratch.extents.data();
            output.tensor.byte_strides = scratch.byteStrides.empty() ? nullptr : scratch.byteStrides.data();
            std::memcpy(inner + slot.offset, &output, sizeof(output));
            for (uint32_t field : arguments.fieldsOfSlot[site.slot])
                std::memcpy(inner + linked.fields[field].offset, &output, sizeof(output));
            continue;
        }
        if (auto wrote = vernon::schedule::writeDenseBuffer(arguments, linked, site.slot, address, shape, inner);
            wrote.isErr())
            return Replayed{vernon::err(std::move(wrote).error())};
    }
    if (auto allocated = allocateResiduals(forward, inner, *replay, sources); allocated.isErr())
        return Replayed{vernon::err(std::move(allocated).error())};
    if (auto bound = bindResiduals(forward, *replay); bound.isErr())
        return Replayed{vernon::err(std::move(bound).error())};
    auto called = vernon::schedule::call(*forward.device, linked, forward.recording, inner, forward.inner.size());
    if (called.isErr())
        return Replayed{vernon::err(std::move(called).error())};
    replay->readBy(forward.device, called.value());
    return Replayed{vernon::ok(std::move(replay))};
}

// Sources in the VJP's discard buffer for the residual block entries `discarded` marks, of `bytes` each (zero for a
// tape). The buffer grows to the call's entries and keeps what it outgrew, which calls in flight may still write.
vernon::Result<ResidualSources, std::string> discardSources(VernonScheduleVjp &vjp, const std::vector<uint64_t> &bytes,
                                                            const std::vector<bool> &discarded) {
    using Placed = vernon::Result<ResidualSources, std::string>;
    VernonScheduleVjp::Checkpointing &checkpointing = *vjp.checkpointing;
    std::vector<uint64_t> offsets;
    uint64_t size = 0;
    for (uint64_t entry : bytes) {
        offsets.push_back(size);
        size += alignedResidual(entry);
    }
    if (size > checkpointing.discardCapacity) {
        auto allocated = vjp.forward->device->allocate(size);
        if (allocated.isErr())
            return Placed{vernon::err(std::move(allocated).error())};
        if (checkpointing.discard)
            checkpointing.retired.push_back(std::move(checkpointing.discard));
        checkpointing.discard = std::move(allocated).value();
        checkpointing.discardCapacity = size;
    }
    ResidualSources sources;
    sources.addresses.assign(bytes.size(), nullptr);
    sources.tapes.resize(bytes.size());
    checkpointing.discardTapes.resize(std::max(checkpointing.discardTapes.size(), bytes.size()));
    auto *base = static_cast<uint8_t *>(checkpointing.discard ? checkpointing.discard->address() : nullptr);
    for (size_t entry = 0; entry < bytes.size(); ++entry) {
        if (!discarded[entry])
            continue;
        const uint32_t slot = vjp.forward->arguments->blockSlots[entry];
        if (vjp.forward->linked.slots[slot].kind != VERNON_SCHEDULE_SLOT_TAPE) {
            sources.addresses[entry] = base + offsets[entry];
            continue;
        }
        std::shared_ptr<VernonScheduleTape> &tape = checkpointing.discardTapes[entry];
        if (!tape)
            tape = std::make_shared<VernonScheduleTape>();
        sources.tapes[entry] = tape;
    }
    return Placed{vernon::ok(std::move(sources))};
}

// The forward schedule of `vjp` truncated to the operations of its first `nodes` nodes, with the forward's calling
// convention, linked once per prefix length.
vernon::Result<VernonLinkedSchedule *, std::string> recomputeSchedule(VernonScheduleVjp &vjp, uint32_t nodes) {
    using Linked = vernon::Result<VernonLinkedSchedule *, std::string>;
    std::unique_ptr<VernonLinkedSchedule> &found = vjp.checkpointing->recomputes[nodes];
    if (found)
        return Linked{vernon::ok(found.get())};
    const VernonLinkedSchedule &forward = *vjp.forward;
    auto schedule = std::make_unique<VernonLinkedSchedule>();
    schedule->context = forward.context;
    schedule->device = forward.device;
    schedule->linked = forward.linked;
    schedule->linked.operations.resize(forward.linked.nodeOperationEnds[nodes - 1]);
    schedule->linked.nodeOperationEnds.resize(nodes);
    schedule->linked.nodeLabels.resize(nodes);
    auto recording = vernon::schedule::record(*schedule->device, schedule->linked);
    if (recording.isErr())
        return Linked{vernon::err(std::move(recording).error())};
    schedule->recording = std::move(recording).value();
    schedule->gridOffset = UINT64_MAX;
    schedule->arguments = forward.arguments;
    schedule->argumentTemplate = forward.argumentTemplate;
    schedule->inner = schedule->linked.recordTemplate;
    schedule->slotBoundaries = forward.slotBoundaries;
    schedule->argumentSlots = forward.argumentSlots;
    schedule->argumentShapes = forward.argumentShapes;
    schedule->outputBoundaries = forward.outputBoundaries;
    schedule->writesDeviceTapes = std::any_of(schedule->linked.operations.begin(), schedule->linked.operations.end(),
                                              [](const vernon::schedule::LinkedDispatch &operation) {
                                                  return operation.tape.device && operation.tape.writes;
                                              });
    found = std::move(schedule);
    return Linked{vernon::ok(found.get())};
}

// Picks what a checkpointed forward call keeps once its linked record `record` holds the call's extents, records it
// in the block's checkpoint, and returns where the call writes the rest, or null when it keeps every residual.
vernon::Result<const ResidualSources *, std::string> planCheckpoint(VernonScheduleVjp &vjp, const uint8_t *record,
                                                                    VernonScheduleResiduals &block,
                                                                    ResidualSources &sources) {
    using Planned = vernon::Result<const ResidualSources *, std::string>;
    const VernonScheduleVjp::Checkpointing &checkpointing = *vjp.checkpointing;
    const VernonLinkedSchedule &forward = *vjp.forward;
    const vernon::schedule::LinkedSchedule &linked = forward.linked;
    const auto nodes = static_cast<uint32_t>(linked.nodeOperationEnds.size());
    uint64_t initial = 0;
    for (const vernon::schedule::ProgramArguments::Site &site : forward.arguments->sites) {
        if (site.slot == vernon::schedule::kNoSlot || forward.outputBoundaries.count(site.boundary))
            continue;
        const VernonScheduleSlot &slot = linked.slots[site.slot];
        if (slot.kind == VERNON_SCHEDULE_SLOT_SCALAR) {
            initial += slot.size;
            continue;
        }
        auto shape = residualShape(linked, site.slot, record);
        if (shape.isErr())
            return Planned{vernon::err(std::move(shape).error())};
        initial += elementCount(shape.value()) * slot.element_size;
    }
    const std::vector<uint32_t> &blockSlots = forward.arguments->blockSlots;
    std::vector<uint64_t> bytes(blockSlots.size(), 0);
    // fromNode[n] is what the residuals of nodes n and later take.
    std::vector<uint64_t> fromNode(nodes + 1, 0);
    for (size_t entry = 0; entry < blockSlots.size(); ++entry) {
        const VernonScheduleSlot &slot = linked.slots[blockSlots[entry]];
        if (slot.kind == VERNON_SCHEDULE_SLOT_TAPE)
            continue;
        auto shape = residualShape(linked, blockSlots[entry], record);
        if (shape.isErr())
            return Planned{vernon::err(std::move(shape).error())};
        bytes[entry] = elementCount(shape.value()) * slot.element_size;
        fromNode[checkpointing.entryNodes[entry]] += bytes[entry];
    }
    for (uint32_t node = nodes; node-- > 0;)
        fromNode[node] += fromNode[node + 1];
    const VernonScheduleCheckpointOptions &options = checkpointing.options;
    // With no residuals, running nodes again would keep nothing less.
    uint32_t recomputed = blockSlots.empty() ? 0 : nodes;
    if (options.policy != VERNON_SCHEDULE_CHECKPOINT_MIN_MEMORY && recomputed)
        for (uint32_t node = checkpointing.tapeNodes; node < nodes; ++node)
            if (initial + fromNode[node] <= options.memory_budget) {
                recomputed = node;
                break;
            }
    const uint64_t held = initial + fromNode[recomputed];
    if (held > options.memory_budget)
        return Planned{vernon::err("the checkpoint memory budget of " + std::to_string(options.memory_budget) +
                                   " bytes is less than the " + std::to_string(held) +
                                   " bytes a forward call holds at the least, its inputs and kept residuals")};
    VernonScheduleCheckpoint &plan = block.checkpoint;
    plan.present = 1;
    plan.policy = options.policy;
    plan.memory_budget = options.memory_budget;
    plan.held_bytes = held;
    plan.initial_state_bytes = initial;
    plan.kept_bytes = fromNode[recomputed];
    plan.recomputed_bytes = fromNode[0] - fromNode[recomputed];
    plan.recomputed_nodes = recomputed;
    plan.forward_nodes = nodes;
    if (!recomputed)
        return Planned{vernon::ok(static_cast<const ResidualSources *>(nullptr))};
    if (auto linkedRecompute = recomputeSchedule(vjp, recomputed); linkedRecompute.isErr())
        return Planned{vernon::err(std::move(linkedRecompute).error())};
    std::vector<bool> discarded(blockSlots.size());
    for (size_t entry = 0; entry < discarded.size(); ++entry)
        discarded[entry] = checkpointing.entryNodes[entry] < recomputed;
    auto placed = discardSources(vjp, bytes, discarded);
    if (placed.isErr())
        return Planned{vernon::err(std::move(placed).error())};
    sources = std::move(placed).value();
    return Planned{vernon::ok(static_cast<const ResidualSources *>(&sources))};
}

// What the first backward call of a forward call does before reading its block, once. A checkpointed call's pullback
// runs the forward nodes it did not keep again into `replayed`, which then also names the entries the forward call
// kept. A forward that writes device tapes never waits for its lanes, so the pullback reads what they reported and,
// when one outgrew its stride, replays the forward with the stride it asked for.
vernon::Result<void, std::string> prepareBackward(VernonScheduleVjp &vjp, const VernonScheduleResiduals &block) {
    using Prepared = vernon::Result<void, std::string>;
    const uint32_t recomputed = block.checkpoint.present ? block.checkpoint.recomputed_nodes : 0;
    if (!recomputed && !vjp.forward->writesDeviceTapes)
        return Prepared{vernon::ok()};
    std::lock_guard<std::mutex> lock(block.checking);
    if (block.checked)
        return Prepared{vernon::ok()};
    VernonLinkedSchedule &forward = recomputed ? *vjp.checkpointing->recomputes.at(recomputed) : *vjp.forward;
    ResidualSources sources;
    if (recomputed) {
        std::vector<uint64_t> bytes;
        std::vector<bool> discarded;
        for (size_t entry = 0; entry < block.entries.size(); ++entry) {
            const VernonScheduleResiduals::Entry &kept = block.entries[entry];
            bytes.push_back(kept.tape ? 0 : elementCount(kept.shape) * kept.elementSize);
            discarded.push_back(vjp.checkpointing->entryNodes[entry] >= recomputed);
        }
        auto placed = discardSources(vjp, bytes, discarded);
        if (placed.isErr())
            return Prepared{vernon::err(std::move(placed).error())};
        sources = std::move(placed).value();
    }
    const auto replay = [&]() -> vernon::Result<uint64_t, std::string> {
        auto replayed = replayForward(forward, block, recomputed ? &sources : nullptr);
        if (replayed.isErr())
            return vernon::Result<uint64_t, std::string>{vernon::err(std::move(replayed).error())};
        VernonScheduleResiduals &run = *replayed.value();
        auto required = overflowedStride(run);
        for (size_t entry = 0; recomputed && entry < block.entries.size(); ++entry) {
            if (vjp.checkpointing->entryNodes[entry] < recomputed)
                continue;
            run.entries[entry] = block.entries[entry];
            run.entries[entry].held = false;
            if (block.entries[entry].tape) {
                run.entries[entry].offset = run.tapes.size();
                run.tapes.push_back(block.tapes[block.entries[entry].offset]);
            }
        }
        block.replayed = std::move(replayed).value();
        return required;
    };
    auto required = recomputed ? replay() : overflowedStride(block);
    if (required.isErr())
        return Prepared{vernon::err(std::move(required).error())};
    // A lane stops at its first record that does not fit, so what it reports is a lower bound: the forward runs again
    // until every lane fits. Each report exceeds the stride it ran with, and a tape is at most 4 GiB, so this ends.
    for (uint64_t stride = required.value(); stride;) {
        uint64_t learned = forward.tapeStride.load();
        while (learned < stride && !forward.tapeStride.compare_exchange_weak(learned, stride)) {
        }
        auto again = replay();
        if (again.isErr())
            return Prepared{vernon::err(std::move(again).error())};
        stride = again.value();
    }
    block.checked = true;
    return Prepared{vernon::ok()};
}

using vernon::schedule::Fragment;
using vernon::schedule::kNoSlot;
using BoundarySlots = std::vector<std::pair<uint32_t, uint32_t>>;

// The fragment slot of boundary slot `boundary`: the first row naming it, since alias rows come last.
uint32_t fragmentSlot(const Fragment &fragment, uint32_t boundary) {
    for (const auto &[row, slot] : fragment.boundarySlots)
        if (row == boundary)
            return slot;
    return kNoSlot;
}

// The schedule slot of exposed Storage or captured scalar `index` of graph node `node`: a fragment's slots are its
// boundary slots, then its exposed Storages, then its captured scalars.
uint32_t exposedSlot(const VernonLinkedSchedule &schedule, size_t node, const Fragment &fragment, size_t index) {
    uint32_t rows = 0;
    for (const auto &entry : fragment.boundarySlots)
        rows = std::max(rows, entry.second + 1);
    return schedule.linked.nodeSlots[node][rows + index];
}

// The boundary slots of node `node` of `schedule` with the schedule slots they bind.
BoundarySlots scheduleBoundarySlots(const VernonLinkedSchedule &schedule, size_t node, const Fragment &fragment) {
    BoundarySlots slots;
    for (const auto &[boundary, slot] : fragment.boundarySlots)
        slots.push_back({boundary, schedule.linked.nodeSlots[node][slot]});
    return slots;
}

// Links `fragments`, which run in order, with `connections`; a boundary row that no connection covers and no argument
// of `arguments` (node, fragment slot) binds is a buffer only the graph uses.
VernonLinkedSchedule *linkComposed(VernonRuntimeContext *context, const std::vector<Fragment> &fragments,
                                   std::vector<vernon::schedule::Connection> connections,
                                   std::set<std::pair<uint32_t, uint32_t>> covered) {
    for (const vernon::schedule::Connection &connection : connections)
        for (const vernon::schedule::NodeSlot &site : connection.sites)
            covered.insert({site.node, site.slot});
    for (uint32_t node = 0; node < fragments.size(); ++node)
        for (const auto &row : fragments[node].boundarySlots)
            if (covered.insert({node, row.second}).second)
                connections.push_back({{{node, row.second}}, true});
    std::vector<const Fragment *> nodes;
    for (const Fragment &fragment : fragments)
        nodes.push_back(&fragment);
    return linkFragments(context, nodes, connections);
}

// A composed graph's forward graphs linked as one schedule node per composition node.
struct ComposedForward {
    std::vector<Fragment> fragments;
    std::unique_ptr<VernonLinkedSchedule> schedule;
    // Per composition connection, its schedule slot, or kNoSlot when it is internal.
    std::vector<uint32_t> connectionSlots;
    // Per export, the composed boundary slot and the schedule slot it binds.
    BoundarySlots boundarySlots;
};

// Links the forward graphs of `composition`'s nodes, node `n` exposing `exposed[n]`. A connection the graph exports
// or that `kept` names, a residual a VJP's backward graph reads, is a slot of the schedule; any other is internal.
// Fails with the context's last error set.
bool linkComposedForward(VernonRuntimeContext *context, const ProgramGraphComposition &composition,
                         const std::vector<std::vector<uint32_t>> &exposed, const std::vector<bool> &kept,
                         ComposedForward &forward) {
    const auto unsupported = [&](const std::string &message) {
        (void)fail(context, message, VERNON_STATUS_UNSUPPORTED_TARGET);
        return false;
    };
    for (size_t node = 0; node < composition.nodes.size(); ++node) {
        auto fragment = executableFragment(*context, *composition.nodes[node], "forward", exposed[node]);
        if (fragment.isErr())
            return unsupported("node " + std::to_string(node) + ": " + fragment.error());
        const std::string &label = composition.labels[node];
        std::vector<std::string> &labels = fragment.value().nodeLabels;
        if (!label.empty())
            for (std::string &name : labels)
                name = labels.size() == 1 ? label : label + "/" + name;
        forward.fragments.push_back(std::move(fragment).value());
    }
    // A pass fragment has no slot for an image a pass renders to, which its PASS slot names; on a graphics device such
    // a boundary is no argument of the composed schedule, and connecting it shares nothing the schedule allocates.
    const auto rendered = [&](const ProgramGraphComposition::Site &site) {
        if (linksComputeExecutables(*context))
            return false;
        const vernon::runtime::program::BoundarySlot &boundary =
            vernon::runtime::capi::executableProgram(*composition.nodes[site.node])->abi.boundarySlots[site.slot];
        return boundary.storage &&
               boundary.storage->descriptorKind == vernon::runtime::program::StorageDescriptorKind::Image;
    };
    const auto boundaryName = [&](const ProgramGraphComposition::Site &site) {
        return "node " + std::to_string(site.node) + " boundary '" +
               vernon::runtime::capi::executableProgram(*composition.nodes[site.node])
                   ->abi.boundarySlots[site.slot]
                   .path +
               "'";
    };
    std::set<std::pair<uint32_t, uint32_t>> exported;
    std::set<std::pair<uint32_t, uint32_t>> arguments;
    for (const ProgramGraphComposition::Export &entry : composition.exports) {
        const uint32_t slot = fragmentSlot(forward.fragments[entry.site.node], entry.site.slot);
        if (slot == kNoSlot && rendered(entry.site))
            continue;
        if (slot == kNoSlot)
            return unsupported("exported " + boundaryName(entry.site) + " is no slot of its node's schedule");
        exported.insert({entry.site.node, entry.site.slot});
        arguments.insert({entry.site.node, slot});
    }
    std::vector<vernon::schedule::Connection> connections;
    // Per composition connection, its index in `connections`, or SIZE_MAX when none of its sites has a slot.
    std::vector<size_t> linkedConnections;
    for (size_t index = 0; index < composition.connections.size(); ++index) {
        vernon::schedule::Connection connection;
        bool exports = false;
        for (const ProgramGraphComposition::Site &site : composition.connections[index].sites) {
            const uint32_t slot = fragmentSlot(forward.fragments[site.node], site.slot);
            if (slot == kNoSlot && rendered(site))
                continue;
            if (slot == kNoSlot)
                return unsupported("connected " + boundaryName(site) + " is no slot of its node's schedule");
            connection.sites.push_back({site.node, slot});
            exports = exports || exported.count({site.node, site.slot}) != 0;
        }
        connection.internal = !exports && !kept[index];
        linkedConnections.push_back(connection.sites.empty() ? SIZE_MAX : connections.size());
        if (!connection.sites.empty())
            connections.push_back(std::move(connection));
    }
    forward.schedule.reset(linkComposed(context, forward.fragments, connections, arguments));
    if (!forward.schedule)
        return false;
    const std::vector<std::vector<uint32_t>> &nodeSlots = forward.schedule->linked.nodeSlots;
    for (const size_t linked : linkedConnections) {
        const vernon::schedule::Connection *connection = linked == SIZE_MAX ? nullptr : &connections[linked];
        forward.connectionSlots.push_back(
            !connection || connection->internal
                ? kNoSlot
                : nodeSlots[connection->sites.front().node][connection->sites.front().slot]);
    }
    for (const ProgramGraphComposition::Export &entry : composition.exports) {
        const uint32_t slot = fragmentSlot(forward.fragments[entry.site.node], entry.site.slot);
        if (slot != kNoSlot)
            forward.boundarySlots.push_back({entry.boundary, nodeSlots[entry.site.node][slot]});
    }
    return true;
}

VernonLinkedSchedule *linkComposedProgram(VernonRuntimeContext *context, VernonProgramExecutable &executable,
                                          const std::string &graph) {
    if (graph != "forward") {
        (void)fail(context, "a composed Program graph has only a forward graph", VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    const ProgramGraphComposition &composition = *executable.composition;
    ComposedForward forward;
    if (!linkComposedForward(context, composition, std::vector<std::vector<uint32_t>>(composition.nodes.size()),
                             std::vector<bool>(composition.connections.size(), false), forward))
        return nullptr;
    if (!linksComputeExecutables(*context)) {
        VernonLinkedSchedule &schedule = *forward.schedule;
        schedule.slotBoundaries.assign(schedule.linked.slots.size(), UINT32_MAX);
        for (const auto &[boundary, slot] : forward.boundarySlots)
            if (slot != kNoSlot)
                schedule.slotBoundaries[slot] = boundary;
        return forward.schedule.release();
    }
    const std::unordered_set<uint32_t> &parameters = executable.bindableParameterSlots;
    auto arguments = vernon::schedule::programArguments(forward.schedule->linked, forward.boundarySlots,
                                                        [&](uint32_t boundary) { return parameters.count(boundary); });
    if (arguments.isErr()) {
        (void)fail(context, arguments.error(), VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    if (auto adopted = adoptArguments(*forward.schedule, std::move(arguments).value()); adopted.isErr()) {
        (void)fail(context, adopted.error(), VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    }
    return forward.schedule.release();
}

// The derivative boundary slots of a Program's ABI by the primal boundary slot each differentiates.
struct Derivatives {
    std::map<uint32_t, uint32_t> cotangent;
    std::map<uint32_t, uint32_t> gradient;
};

vernon::Result<Derivatives, std::string> derivatives(const VernonProgramExecutable &executable) {
    using Found = vernon::Result<Derivatives, std::string>;
    namespace program = vernon::runtime::program;
    const program::Program *source = vernon::runtime::capi::executableProgram(executable);
    if (!source)
        return Found{vernon::err(std::string("the Program executable has no ABI"))};
    // The projections of a Struct's fields name the boundary slot of the whole Value's derivative, one per field.
    Derivatives found;
    for (const program::DerivativeProjection &projection : source->abi.derivativeProjections) {
        const uint32_t derivative = projection.derivative.slot;
        if (derivative >= source->abi.boundarySlots.size())
            return Found{vernon::err(std::string("a derivative projection names no boundary slot"))};
        const program::BoundaryRole role = source->abi.boundarySlots[derivative].role;
        if (role != program::BoundaryRole::Cotangent && role != program::BoundaryRole::Gradient)
            return Found{vernon::err(std::string("a derivative projection names no cotangent or gradient"))};
        std::map<uint32_t, uint32_t> &derivatives =
            role == program::BoundaryRole::Cotangent ? found.cotangent : found.gradient;
        const auto [entry, inserted] = derivatives.emplace(projection.primal.slot, derivative);
        if (!inserted && entry->second != derivative)
            return Found{vernon::err(std::string("a primal boundary slot has two derivative boundary slots"))};
    }
    return Found{vernon::ok(std::move(found))};
}

// Renames the backward schedule's ARGUMENT slots from derivative boundary slots to the primal ones they
// differentiate, and records which take cotangents.
vernon::Result<void, std::string> namePrimals(VernonScheduleVjp &vjp, const Derivatives &found) {
    std::map<uint32_t, std::pair<uint32_t, VernonProgramBoundaryRole>> primal;
    for (const auto &[slot, derivative] : found.cotangent)
        primal[derivative] = {slot, VERNON_PROGRAM_BOUNDARY_COTANGENT};
    for (const auto &[slot, derivative] : found.gradient)
        primal[derivative] = {slot, VERNON_PROGRAM_BOUNDARY_GRADIENT};
    for (uint32_t &boundary : vjp.backward->slotBoundaries) {
        const auto named = primal.find(boundary);
        if (named == primal.end())
            return vernon::Result<void, std::string>{
                vernon::err(std::string("the backward graph takes an argument that is no derivative"))};
        boundary = named->second.first;
        vjp.backwardRoles.push_back(named->second.second);
    }
    return vernon::Result<void, std::string>{vernon::ok()};
}

// Adds the passes of `plan`, whose exposed Storages are the residual block's entries from `base` on.
void addPasses(VernonScheduleVjp &vjp, const vernon::schedule::VjpPlan &plan, uint32_t base) {
    for (const vernon::schedule::VjpPlan::Pass &planned : plan.passes) {
        VernonScheduleVjp::Pass &pass = vjp.passes.emplace_back();
        pass.node = planned.node;
        pass.name = planned.name;
        for (uint32_t exposed : planned.exposed)
            pass.entries.push_back(base + exposed);
    }
}

std::set<uint32_t> outputBoundaries(const nlohmann::json &program) {
    std::set<uint32_t> outputs;
    const nlohmann::json *abi = program.contains("abi") ? &program.at("abi") : nullptr;
    if (abi && abi->contains("boundary_slots"))
        for (const nlohmann::json &row : abi->at("boundary_slots"))
            if (row.value("direction", std::string()) == "output")
                outputs.insert(row.at("id").get<uint32_t>());
    return outputs;
}

// The forward argument site of boundary slot `boundary`, from which a backward schedule reads it again.
const vernon::schedule::ProgramArguments::Site *forwardSite(const VernonScheduleVjp &vjp, uint32_t boundary) {
    for (const vernon::schedule::ProgramArguments::Site &site : vjp.forward->arguments->sites)
        if (site.boundary == boundary)
            return &site;
    return nullptr;
}

VernonScheduleVjp *linkBundleVjp(VernonRuntimeContext *context, VernonProgramExecutable &executable) {
    const auto unsupported = [&](const std::string &message) -> VernonScheduleVjp * {
        (void)fail(context, message, VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    };
    auto planned = vernon::schedule::planVjp(*executable.programSource, context->backend == VERNON_RUNTIME_CPU);
    if (planned.isErr())
        return unsupported(planned.error());
    const vernon::schedule::VjpPlan &plan = planned.value();
    auto found = derivatives(executable);
    if (found.isErr())
        return unsupported(found.error());
    std::vector<uint32_t> forwardExposed = plan.residuals;
    forwardExposed.insert(forwardExposed.end(), plan.tapes.begin(), plan.tapes.end());
    std::vector<uint32_t> backwardExposed = plan.captured;
    backwardExposed.insert(backwardExposed.end(), plan.tapes.begin(), plan.tapes.end());
    auto forwardFragment = executableFragment(*context, executable, "forward", forwardExposed);
    if (forwardFragment.isErr())
        return unsupported("forward graph: " + forwardFragment.error());
    auto backwardFragment = executableFragment(*context, executable, "backward", backwardExposed);
    if (backwardFragment.isErr())
        return unsupported("backward graph: " + backwardFragment.error());
    const Fragment &forward = forwardFragment.value();
    const Fragment &backward = backwardFragment.value();
    auto vjp = std::make_unique<VernonScheduleVjp>();
    vjp->forward.reset(linkFragment(context, forward));
    vjp->backward.reset(linkFragment(context, backward));
    if (!vjp->forward || !vjp->backward)
        return nullptr;

    const std::unordered_set<uint32_t> &parameters = executable.bindableParameterSlots;
    auto forwardArguments =
        vernon::schedule::programArguments(vjp->forward->linked, scheduleBoundarySlots(*vjp->forward, 0, forward),
                                           [&](uint32_t boundary) { return parameters.count(boundary) != 0; });
    if (forwardArguments.isErr())
        return unsupported(forwardArguments.error());
    for (size_t index = 0; index < forwardExposed.size(); ++index)
        forwardArguments.value().blockSlots.push_back(exposedSlot(*vjp->forward, 0, forward, index));
    if (auto adopted = adoptArguments(*vjp->forward, std::move(forwardArguments).value()); adopted.isErr())
        return unsupported("forward graph: " + adopted.error());
    vjp->forward->outputBoundaries = outputBoundaries(*executable.programSource);

    auto backwardArguments = vernon::schedule::programArguments(
        vjp->backward->linked, scheduleBoundarySlots(*vjp->backward, 0, backward), [](uint32_t) { return true; });
    if (backwardArguments.isErr())
        return unsupported(backwardArguments.error());
    vernon::schedule::ProgramArguments &arguments = backwardArguments.value();
    const auto capture = [&](uint32_t boundary, uint32_t slot) {
        const vernon::schedule::ProgramArguments::Site *site = forwardSite(*vjp, boundary);
        if (site)
            arguments.captures.push_back({boundary, site->offset, slot});
        return site != nullptr;
    };
    for (size_t index = 0; index < plan.captured.size(); ++index) {
        const uint32_t slot = exposedSlot(*vjp->backward, 0, backward, index);
        if (plan.capturedBoundary[index] == UINT32_MAX)
            arguments.blockSlots.push_back(slot);
        else if (!capture(plan.capturedBoundary[index], slot))
            return unsupported("the backward graph reads a forward argument that is no parameter");
    }
    for (size_t index = 0; index < plan.tapes.size(); ++index)
        arguments.blockSlots.push_back(exposedSlot(*vjp->backward, 0, backward, plan.captured.size() + index));
    for (size_t index = 0; index < plan.scalarBoundaries.size(); ++index)
        if (!capture(plan.scalarBoundaries[index],
                     exposedSlot(*vjp->backward, 0, backward, backwardExposed.size() + index)))
            return unsupported("the backward graph reads a forward scalar that is no parameter");
    if (auto adopted = adoptArguments(*vjp->backward, std::move(arguments)); adopted.isErr())
        return unsupported("backward graph: " + adopted.error());
    if (auto named = namePrimals(*vjp, found.value()); named.isErr())
        return unsupported(named.error());
    addPasses(*vjp, plan, 0);
    return vjp.release();
}

// A composed graph's VJP: its nodes' forward graphs in order, then their backward graphs in reverse, where a node's
// input gradient is the cotangent of the output that feeds it. A connection a backward graph reads outlives the
// forward call as a residual.
VernonScheduleVjp *linkComposedVjp(VernonRuntimeContext *context, VernonProgramExecutable &executable) {
    const auto unsupported = [&](const std::string &message) -> VernonScheduleVjp * {
        (void)fail(context, message, VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
    };
    const ProgramGraphComposition &composition = *executable.composition;
    const auto count = static_cast<uint32_t>(composition.nodes.size());
    struct Node {
        vernon::schedule::VjpPlan plan;
        Derivatives derivatives;
        std::vector<uint32_t> forwardExposed;
        std::vector<uint32_t> backwardExposed;
    };
    std::vector<Node> nodes(count);
    for (uint32_t index = 0; index < count; ++index) {
        const VernonProgramExecutable &child = *composition.nodes[index];
        const std::string name = "node " + std::to_string(index) + ": ";
        if (!child.programSource)
            return unsupported(name + "a linked VJP runs Programs resolved from a bundle");
        auto planned = vernon::schedule::planVjp(*child.programSource, context->backend == VERNON_RUNTIME_CPU);
        if (planned.isErr())
            return unsupported(name + planned.error());
        auto found = derivatives(child);
        if (found.isErr())
            return unsupported(name + found.error());
        Node &node = nodes[index];
        node.plan = std::move(planned).value();
        node.derivatives = std::move(found).value();
        node.forwardExposed = node.plan.residuals;
        node.forwardExposed.insert(node.forwardExposed.end(), node.plan.tapes.begin(), node.plan.tapes.end());
        node.backwardExposed = node.plan.captured;
        node.backwardExposed.insert(node.backwardExposed.end(), node.plan.tapes.begin(), node.plan.tapes.end());
    }
    using Key = std::pair<uint32_t, uint32_t>;
    std::map<Key, uint32_t> exportOf;
    for (const ProgramGraphComposition::Export &entry : composition.exports)
        exportOf.emplace(Key{entry.site.node, entry.site.slot}, entry.boundary);
    std::map<Key, size_t> connectionOf;
    std::vector<bool> kept(composition.connections.size(), false);
    for (size_t index = 0; index < composition.connections.size(); ++index) {
        const ProgramGraphComposition::Connection &connection = composition.connections[index];
        if (connection.storage)
            return unsupported("a composed VJP does not link Storage chains yet");
        bool exported = false;
        for (const ProgramGraphComposition::Site &site : connection.sites) {
            connectionOf[{site.node, site.slot}] = index;
            exported = exported || exportOf.count({site.node, site.slot}) != 0;
        }
        for (const ProgramGraphComposition::Site &site : connection.sites) {
            const vernon::schedule::VjpPlan &plan = nodes[site.node].plan;
            const auto reads = [&](const std::vector<uint32_t> &boundaries) {
                return std::find(boundaries.begin(), boundaries.end(), site.slot) != boundaries.end();
            };
            if (!exported && reads(plan.scalarBoundaries))
                return unsupported("a backward graph reads a scalar another node computes, which is not linked yet");
            kept[index] = kept[index] || (!exported && reads(plan.capturedBoundary));
        }
    }

    std::vector<std::vector<uint32_t>> forwardExposed;
    for (const Node &node : nodes)
        forwardExposed.push_back(node.forwardExposed);
    ComposedForward forward;
    if (!linkComposedForward(context, composition, forwardExposed, kept, forward))
        return nullptr;
    auto vjp = std::make_unique<VernonScheduleVjp>();
    vjp->forward = std::move(forward.schedule);
    const std::unordered_set<uint32_t> &parameters = executable.bindableParameterSlots;
    auto forwardArguments = vernon::schedule::programArguments(
        vjp->forward->linked, forward.boundarySlots, [&](uint32_t boundary) { return parameters.count(boundary); });
    if (forwardArguments.isErr())
        return unsupported(forwardArguments.error());
    vernon::schedule::ProgramArguments &forwardBlock = forwardArguments.value();
    // The residual block's entries: each node's residuals and tapes, then the connections a backward graph reads.
    std::vector<std::map<uint32_t, uint32_t>> residualEntry(count);
    std::vector<uint32_t> tapeEntry(count);
    std::map<size_t, uint32_t> connectionEntry;
    for (uint32_t index = 0; index < count; ++index) {
        const Node &node = nodes[index];
        addPasses(*vjp, node.plan, static_cast<uint32_t>(forwardBlock.blockSlots.size()));
        for (size_t exposed = 0; exposed < node.forwardExposed.size(); ++exposed) {
            if (exposed < node.plan.residuals.size())
                residualEntry[index][node.plan.residuals[exposed]] =
                    static_cast<uint32_t>(forwardBlock.blockSlots.size());
            else if (exposed == node.plan.residuals.size())
                tapeEntry[index] = static_cast<uint32_t>(forwardBlock.blockSlots.size());
            forwardBlock.blockSlots.push_back(exposedSlot(*vjp->forward, index, forward.fragments[index], exposed));
        }
    }
    for (size_t index = 0; index < kept.size(); ++index)
        if (kept[index]) {
            connectionEntry[index] = static_cast<uint32_t>(forwardBlock.blockSlots.size());
            forwardBlock.blockSlots.push_back(forward.connectionSlots[index]);
        }
    if (auto adopted = adoptArguments(*vjp->forward, std::move(forwardBlock)); adopted.isErr())
        return unsupported("forward graph: " + adopted.error());
    {
        std::vector<std::set<uint32_t>> nodeOutputs;
        for (const std::shared_ptr<VernonProgramExecutable> &child : composition.nodes)
            nodeOutputs.push_back(outputBoundaries(*child->programSource));
        for (const ProgramGraphComposition::Export &entry : composition.exports)
            if (nodeOutputs[entry.site.node].count(entry.site.slot))
                vjp->forward->outputBoundaries.insert(entry.boundary);
    }

    // Backward node `count - 1 - n` is the backward graph of node n.
    const auto backwardNode = [&](uint32_t node) { return count - 1 - node; };
    std::vector<Fragment> fragments;
    for (uint32_t index = count; index-- > 0;) {
        auto fragment =
            executableFragment(*context, *composition.nodes[index], "backward", nodes[index].backwardExposed);
        if (fragment.isErr())
            return unsupported("node " + std::to_string(index) + " backward graph: " + fragment.error());
        fragments.push_back(std::move(fragment).value());
    }
    const auto derivativeSlot = [&](uint32_t node, uint32_t derivative) {
        return fragmentSlot(fragments[backwardNode(node)], derivative);
    };
    std::vector<vernon::schedule::Connection> connections;
    for (size_t index = 0; index < composition.connections.size(); ++index) {
        const std::vector<ProgramGraphComposition::Site> &sites = composition.connections[index].sites;
        const ProgramGraphComposition::Site &source = sites.front();
        const std::map<uint32_t, uint32_t> &cotangents = nodes[source.node].derivatives.cotangent;
        const auto cotangent = cotangents.find(source.slot);
        bool differentiated = false;
        for (size_t site = 1; site < sites.size(); ++site)
            differentiated = differentiated || nodes[sites[site].node].derivatives.gradient.count(sites[site].slot);
        if (!differentiated) {
            if (cotangent != cotangents.end() && !exportOf.count({source.node, source.slot}))
                return unsupported("an output whose cotangent is zero is not linked yet");
            continue;
        }
        if (sites.size() != 2 || exportOf.count({source.node, source.slot}))
            return unsupported("a Value several nodes or the caller read needs a sum of its cotangents, which is "
                               "not linked yet");
        const ProgramGraphComposition::Site &destination = sites.back();
        vernon::schedule::Connection &connection = connections.emplace_back();
        connection.internal = true;
        connection.sites.push_back(
            {backwardNode(destination.node),
             derivativeSlot(destination.node, nodes[destination.node].derivatives.gradient.at(destination.slot))});
        if (cotangent != cotangents.end())
            connection.sites.push_back({backwardNode(source.node), derivativeSlot(source.node, cotangent->second)});
    }
    struct Argument {
        uint32_t boundary{};
        uint32_t node{};
        uint32_t slot{};
        VernonProgramBoundaryRole role{};
    };
    std::vector<Argument> arguments;
    std::set<Key> covered;
    std::set<uint32_t> gradients;
    for (const ProgramGraphComposition::Export &entry : composition.exports) {
        const Derivatives &found = nodes[entry.site.node].derivatives;
        Argument argument{entry.boundary, backwardNode(entry.site.node)};
        if (const auto cotangent = found.cotangent.find(entry.site.slot); cotangent != found.cotangent.end()) {
            argument.slot = derivativeSlot(entry.site.node, cotangent->second);
            argument.role = VERNON_PROGRAM_BOUNDARY_COTANGENT;
        } else if (const auto gradient = found.gradient.find(entry.site.slot); gradient != found.gradient.end()) {
            if (!gradients.insert(entry.boundary).second)
                return unsupported("an input several nodes read needs a sum of its gradients, which is not linked yet");
            argument.slot = derivativeSlot(entry.site.node, gradient->second);
            argument.role = VERNON_PROGRAM_BOUNDARY_GRADIENT;
        } else {
            continue;
        }
        covered.insert({argument.node, argument.slot});
        arguments.push_back(argument);
    }
    for (const vernon::schedule::Connection &connection : connections)
        for (const vernon::schedule::NodeSlot &site : connection.sites)
            covered.insert({site.node, site.slot});
    for (uint32_t index = 0; index < count; ++index)
        for (const auto &[primal, derivative] : nodes[index].derivatives.cotangent)
            if (!covered.count({backwardNode(index), derivativeSlot(index, derivative)}))
                return unsupported("an output no node reads and the graph does not export has a cotangent, which is "
                                   "not linked yet");
    std::set<Key> argumentSlots;
    for (const Argument &argument : arguments)
        argumentSlots.insert({argument.node, argument.slot});
    vjp->backward.reset(linkComposed(context, fragments, connections, argumentSlots));
    if (!vjp->backward)
        return nullptr;

    BoundarySlots boundarySlots;
    std::map<uint32_t, VernonProgramBoundaryRole> roleOf;
    for (const Argument &argument : arguments) {
        boundarySlots.push_back({argument.boundary, vjp->backward->linked.nodeSlots[argument.node][argument.slot]});
        roleOf[argument.boundary] = argument.role;
    }
    auto backwardArguments =
        vernon::schedule::programArguments(vjp->backward->linked, boundarySlots, [](uint32_t) { return true; });
    if (backwardArguments.isErr())
        return unsupported(backwardArguments.error());
    vernon::schedule::ProgramArguments &backward = backwardArguments.value();
    const auto block = [&](uint32_t slot, uint32_t entry) {
        backward.blockSlots.push_back(slot);
        backward.blockEntries.push_back(entry);
    };
    const auto capture = [&](uint32_t boundary, uint32_t slot) {
        const vernon::schedule::ProgramArguments::Site *site = forwardSite(*vjp, boundary);
        if (site)
            backward.captures.push_back({boundary, site->offset, slot});
        return site != nullptr;
    };
    for (uint32_t index = 0; index < count; ++index) {
        const Node &node = nodes[index];
        const Fragment &fragment = fragments[backwardNode(index)];
        const auto exposed = [&](size_t at) { return exposedSlot(*vjp->backward, backwardNode(index), fragment, at); };
        for (size_t at = 0; at < node.plan.captured.size(); ++at) {
            const uint32_t boundary = node.plan.capturedBoundary[at];
            if (boundary == UINT32_MAX) {
                const auto entry = residualEntry[index].find(node.plan.captured[at]);
                if (entry == residualEntry[index].end())
                    return unsupported("a backward graph reads a forward Storage that is no residual");
                block(exposed(at), entry->second);
            } else if (const auto exported = exportOf.find({index, boundary}); exported != exportOf.end()) {
                if (!capture(exported->second, exposed(at)))
                    return unsupported("a backward graph reads a forward argument that is no parameter");
            } else if (const auto connection = connectionOf.find({index, boundary});
                       connection != connectionOf.end() && connectionEntry.count(connection->second)) {
                block(exposed(at), connectionEntry.at(connection->second));
            } else {
                return unsupported("a backward graph reads a forward boundary the graph neither passes nor keeps");
            }
        }
        for (size_t tape = 0; tape < node.plan.tapes.size(); ++tape)
            block(exposed(node.plan.captured.size() + tape), tapeEntry[index] + static_cast<uint32_t>(tape));
        for (size_t at = 0; at < node.plan.scalarBoundaries.size(); ++at) {
            const auto exported = exportOf.find({index, node.plan.scalarBoundaries[at]});
            if (exported == exportOf.end() || !capture(exported->second, exposed(node.backwardExposed.size() + at)))
                return unsupported("a backward graph reads a forward scalar the graph does not pass");
        }
    }
    if (auto adopted = adoptArguments(*vjp->backward, std::move(backward)); adopted.isErr())
        return unsupported("backward graph: " + adopted.error());
    for (uint32_t boundary : vjp->backward->slotBoundaries)
        vjp->backwardRoles.push_back(roleOf.at(boundary));
    return vjp.release();
}

// Readies a bundle VJP's forward calls to keep only what `options` lets them hold: each residual block entry is
// credited to the forward node that produces it, or to the last node when none is credited.
vernon::Result<void, std::string> checkpointVjp(VernonScheduleVjp &vjp,
                                                const VernonScheduleCheckpointOptions &options) {
    using Checkpointed = vernon::Result<void, std::string>;
    const VernonLinkedSchedule &forward = *vjp.forward;
    const auto nodes = static_cast<uint32_t>(forward.linked.nodeOperationEnds.size());
    if (!nodes || vjp.passes.size() != nodes)
        return Checkpointed{vernon::err(std::string("a checkpointed VJP's forward graph has no nodes to run again"))};
    for (const vernon::schedule::ProgramArguments::Site &site : forward.arguments->sites)
        if (site.slot != kNoSlot && !forward.outputBoundaries.count(site.boundary) &&
            forward.linked.slots[site.slot].access == VERNON_SCHEDULE_ACCESS_READ_WRITE)
            return Checkpointed{vernon::err(std::string("a checkpointed VJP's forward graph reads and writes one of "
                                                        "its arguments, which its pullback reads again"))};
    VernonScheduleVjp::Checkpointing checkpointing;
    checkpointing.options = options;
    checkpointing.entryNodes.assign(forward.arguments->blockSlots.size(), nodes - 1);
    for (uint32_t node = 0; node < nodes; ++node)
        for (uint32_t entry : vjp.passes[node].entries)
            checkpointing.entryNodes[entry] = node;
    for (size_t entry = 0; entry < checkpointing.entryNodes.size(); ++entry)
        if (forward.linked.slots[forward.arguments->blockSlots[entry]].kind == VERNON_SCHEDULE_SLOT_TAPE)
            checkpointing.tapeNodes = std::max(checkpointing.tapeNodes, checkpointing.entryNodes[entry] + 1);
    vjp.checkpointing = std::move(checkpointing);
    return Checkpointed{vernon::ok()};
}

} // namespace

extern "C" {

VernonLinkedSchedule *vernonScheduleLinkKernel(VernonRuntimeContext *context,
                                               const VernonScheduleKernelDescriptor *descriptor) {
    return vernon::runtime::capi::runtimePointerBoundary<VernonLinkedSchedule *>(
        context, [&]() -> VernonLinkedSchedule * {
            if (!context || !descriptor) {
                (void)fail(context, "schedule kernel descriptor is invalid");
                return nullptr;
            }
            auto fragment = kernelFragment(*context, *descriptor);
            if (fragment.isErr()) {
                (void)fail(context, fragment.error(), VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            return linkFragment(context, fragment.value());
        });
}

VernonLinkedSchedule *vernonScheduleLinkProgram(VernonRuntimeContext *context,
                                                const VernonScheduleProgramDescriptor *descriptor) {
    return vernon::runtime::capi::runtimePointerBoundary<VernonLinkedSchedule *>(
        context, [&]() -> VernonLinkedSchedule * {
            if (!context || !descriptor) {
                (void)fail(context, "schedule Program descriptor is invalid");
                return nullptr;
            }
            if (descriptor->struct_size >= sizeof(*descriptor) && descriptor->executable &&
                descriptor->executable->composition) {
                if (!descriptor->graph.data) {
                    (void)fail(context, "schedule Program descriptor is invalid");
                    return nullptr;
                }
                return linkComposedProgram(context, *descriptor->executable,
                                           std::string(descriptor->graph.data, descriptor->graph.size));
            }
            auto fragment = programFragment(*context, *descriptor);
            if (fragment.isErr()) {
                (void)fail(context, fragment.error(), VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            std::unique_ptr<VernonLinkedSchedule> schedule(linkFragment(context, fragment.value()));
            if (!schedule || !descriptor->executable)
                return schedule.release();
            const std::vector<std::pair<uint32_t, uint32_t>> &boundarySlots = fragment.value().boundarySlots;
            if (!linksComputeExecutables(*context)) {
                const std::vector<uint32_t> &nodeSlots = schedule->linked.nodeSlots.front();
                schedule->slotBoundaries.assign(schedule->linked.slots.size(), UINT32_MAX);
                for (const auto &[boundary, slot] : boundarySlots)
                    if (slot < nodeSlots.size() && nodeSlots[slot] != vernon::schedule::kNoSlot)
                        schedule->slotBoundaries[nodeSlots[slot]] = boundary;
                return schedule.release();
            }
            const std::unordered_set<uint32_t> &parameters = descriptor->executable->bindableParameterSlots;
            auto arguments = vernon::schedule::programArguments(
                schedule->linked, scheduleBoundarySlots(*schedule, 0, fragment.value()),
                [&](uint32_t boundary) { return parameters.count(boundary) != 0; });
            if (arguments.isErr()) {
                (void)fail(context, arguments.error(), VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            if (auto adopted = adoptArguments(*schedule, std::move(arguments).value()); adopted.isErr()) {
                (void)fail(context, adopted.error(), VERNON_STATUS_UNSUPPORTED_TARGET);
                return nullptr;
            }
            return schedule.release();
        });
}

VernonLinkedSchedule *vernonScheduleLinkGraph(VernonRuntimeContext *context,
                                              const VernonScheduleGraphDescriptor *descriptor) {
    return vernon::runtime::capi::runtimePointerBoundary<VernonLinkedSchedule *>(
        context, [&]() -> VernonLinkedSchedule * {
            if (!context || !descriptor || descriptor->struct_size < sizeof(*descriptor) ||
                (descriptor->node_count && !descriptor->nodes) ||
                (descriptor->connection_count && !descriptor->connections)) {
                (void)fail(context, "schedule graph descriptor is invalid");
                return nullptr;
            }
            std::vector<vernon::schedule::Fragment> fragments;
            fragments.reserve(descriptor->node_count);
            for (size_t index = 0; index < descriptor->node_count; ++index) {
                auto fragment = programFragment(*context, descriptor->nodes[index]);
                if (fragment.isErr()) {
                    (void)fail(context, "graph node " + std::to_string(index) + ": " + fragment.error(),
                               VERNON_STATUS_UNSUPPORTED_TARGET);
                    return nullptr;
                }
                fragments.push_back(std::move(fragment).value());
            }
            std::vector<vernon::schedule::Connection> connections(descriptor->connection_count);
            for (size_t index = 0; index < descriptor->connection_count; ++index) {
                const VernonScheduleConnection &connection = descriptor->connections[index];
                if (connection.site_count && !connection.sites) {
                    (void)fail(context, "schedule graph connection is invalid");
                    return nullptr;
                }
                for (size_t site = 0; site < connection.site_count; ++site)
                    connections[index].sites.push_back({connection.sites[site].node, connection.sites[site].slot});
                connections[index].internal = connection.internal != 0;
            }
            std::vector<const vernon::schedule::Fragment *> nodes;
            for (const vernon::schedule::Fragment &fragment : fragments)
                nodes.push_back(&fragment);
            return linkFragments(context, nodes, connections);
        });
}

size_t vernonScheduleNodeSlotCount(const VernonLinkedSchedule *schedule, size_t node) {
    if (schedule && schedule->arguments)
        return node == 0 ? schedule->arguments->slots.size() : 0;
    return schedule && node < schedule->linked.nodeSlots.size() ? schedule->linked.nodeSlots[node].size() : 0;
}

uint32_t vernonScheduleNodeSlot(const VernonLinkedSchedule *schedule, size_t node, size_t slot) {
    if (schedule && schedule->arguments)
        return node == 0 && slot < schedule->arguments->slots.size() ? static_cast<uint32_t>(slot) : UINT32_MAX;
    if (!schedule || node >= schedule->linked.nodeSlots.size() || slot >= schedule->linked.nodeSlots[node].size())
        return UINT32_MAX;
    return schedule->linked.nodeSlots[node][slot];
}

uint32_t vernonScheduleSlotBoundary(const VernonLinkedSchedule *schedule, size_t slot) {
    return schedule && slot < schedule->slotBoundaries.size() ? schedule->slotBoundaries[slot] : UINT32_MAX;
}

void vernonScheduleDestroy(VernonLinkedSchedule *schedule) { delete schedule; }

size_t vernonScheduleRecordSize(const VernonLinkedSchedule *schedule) {
    if (schedule && schedule->arguments)
        return static_cast<size_t>(schedule->arguments->recordSize);
    return schedule ? static_cast<size_t>(schedule->linked.recordSize) : 0;
}

const uint8_t *vernonScheduleRecordTemplate(const VernonLinkedSchedule *schedule) {
    if (schedule && schedule->arguments)
        return schedule->argumentTemplate.data();
    return schedule ? schedule->linked.recordTemplate.data() : nullptr;
}

uint64_t vernonScheduleGridOffset(const VernonLinkedSchedule *schedule) { return schedule ? schedule->gridOffset : 0; }

size_t vernonScheduleSlotCount(const VernonLinkedSchedule *schedule) {
    if (schedule && schedule->arguments)
        return schedule->arguments->slots.size();
    return schedule ? schedule->linked.slots.size() : 0;
}

const VernonScheduleSlot *vernonScheduleSlots(const VernonLinkedSchedule *schedule) {
    if (schedule && schedule->arguments)
        return schedule->arguments->slots.data();
    return schedule ? schedule->linked.slots.data() : nullptr;
}

const int64_t *vernonScheduleSlotShape(const VernonLinkedSchedule *schedule, size_t slot) {
    if (schedule && schedule->arguments)
        return slot < schedule->argumentShapes.size() ? schedule->argumentShapes[slot].data() : nullptr;
    return schedule && slot < schedule->linked.shapes.size() ? schedule->linked.shapes[slot].data() : nullptr;
}

const VernonScheduleSlot *vernonScheduleArgumentSlot(const VernonLinkedSchedule *schedule, size_t slot) {
    return schedule && schedule->arguments && slot < schedule->argumentSlots.size() ? &schedule->argumentSlots[slot]
                                                                                    : nullptr;
}

const VernonScheduleExtent *vernonScheduleSlotExtents(const VernonLinkedSchedule *schedule, size_t slot) {
    if (!schedule || schedule->arguments)
        return nullptr;
    return slot < schedule->linked.extents.size() ? schedule->linked.extents[slot].data() : nullptr;
}

size_t vernonScheduleFieldCount(const VernonLinkedSchedule *schedule) {
    return schedule && !schedule->arguments ? schedule->linked.fields.size() : 0;
}

const VernonScheduleField *vernonScheduleFields(const VernonLinkedSchedule *schedule) {
    return schedule && !schedule->arguments ? schedule->linked.fields.data() : nullptr;
}

VernonStatus vernonScheduleCall(VernonLinkedSchedule *schedule, const void *record, size_t size, uint64_t *point) {
    if (!schedule)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return vernon::runtime::capi::runtimeStatusBoundary(schedule->context, [&]() {
        if (schedule->arguments)
            return callProgram(*schedule, record, size, nullptr, nullptr, point);
        auto called = vernon::schedule::call(*schedule->device, schedule->linked, schedule->recording,
                                             static_cast<const uint8_t *>(record), size);
        if (called.isErr())
            return fail(schedule->context, called.error());
        if (point)
            *point = called.value();
        return VERNON_STATUS_OK;
    });
}

VernonScheduleVjp *vernonScheduleLinkVjp(VernonRuntimeContext *context, VernonProgramExecutable *executable) {
    return vernon::runtime::capi::runtimePointerBoundary<VernonScheduleVjp *>(context, [&]() -> VernonScheduleVjp * {
        if (!context || !executable) {
            (void)fail(context, "linked VJP takes a resolved Program");
            return nullptr;
        }
        const auto unsupported = [&](const std::string &message) -> VernonScheduleVjp * {
            (void)fail(context, message, VERNON_STATUS_UNSUPPORTED_TARGET);
            return nullptr;
        };
        if (executable->composition)
            return linkComposedVjp(context, *executable);
        if (!executable->programSource)
            return unsupported("linked VJPs run Programs resolved from a bundle or composed graphs of Programs that "
                               "each have a backward graph");
        return linkBundleVjp(context, *executable);
    });
}

VernonScheduleVjp *vernonScheduleLinkCheckpointedVjp(VernonRuntimeContext *context, VernonProgramExecutable *executable,
                                                     const VernonScheduleCheckpointOptions *options) {
    return vernon::runtime::capi::runtimePointerBoundary<VernonScheduleVjp *>(context, [&]() -> VernonScheduleVjp * {
        if (!context || !executable || !options || options->struct_size < sizeof(*options) ||
            options->policy > VERNON_SCHEDULE_CHECKPOINT_MIN_RUNTIME) {
            (void)fail(context, "checkpointed VJP takes a resolved Program and checkpoint options");
            return nullptr;
        }
        const auto unsupported = [&](const std::string &message) -> VernonScheduleVjp * {
            (void)fail(context, message, VERNON_STATUS_UNSUPPORTED_TARGET);
            return nullptr;
        };
        if (!linksComputeExecutables(*context))
            return unsupported("checkpointed VJPs run on the CPU and CUDA");
        if (!executable->programSource)
            return unsupported("checkpointed VJPs run Programs resolved from a bundle");
        std::unique_ptr<VernonScheduleVjp> vjp(linkBundleVjp(context, *executable));
        if (!vjp)
            return nullptr;
        if (auto checkpointed = checkpointVjp(*vjp, *options); checkpointed.isErr())
            return unsupported(checkpointed.error());
        return vjp.release();
    });
}

VernonProgramBoundaryRole vernonScheduleVjpSlotRole(const VernonScheduleVjp *vjp, size_t slot) {
    return vjp && slot < vjp->backwardRoles.size() ? vjp->backwardRoles[slot] : VERNON_PROGRAM_BOUNDARY_INPUT;
}

void vernonScheduleVjpDestroy(VernonScheduleVjp *vjp) { delete vjp; }

const VernonLinkedSchedule *vernonScheduleVjpSchedule(const VernonScheduleVjp *vjp, uint32_t backward) {
    if (!vjp)
        return nullptr;
    return backward ? vjp->backward.get() : vjp->forward.get();
}

VernonStatus vernonScheduleVjpForward(VernonScheduleVjp *vjp, const void *record, size_t size,
                                      VernonScheduleResiduals **residuals, uint64_t *point) {
    if (!vjp || !residuals)
        return VERNON_STATUS_INVALID_ARGUMENT;
    *residuals = nullptr;
    return vernon::runtime::capi::runtimeStatusBoundary(vjp->forward->context, [&]() {
        auto block = std::make_unique<VernonScheduleResiduals>();
        if (record && size >= vjp->forward->arguments->recordSize)
            block->forwardRecord.assign(static_cast<const uint8_t *>(record),
                                        static_cast<const uint8_t *>(record) + vjp->forward->arguments->recordSize);
        // A forward that writes device tapes or is checkpointed may run again at the pullback, from all of its
        // arguments.
        if (auto kept = keepArguments(*block, vjp->backward->arguments->captures, vjp->backward->linked); kept.isErr())
            return fail(vjp->forward->context, kept.error());
        if (vjp->forward->writesDeviceTapes || vjp->checkpointing)
            if (auto kept = keepArguments(*block, vjp->forward->arguments->sites, vjp->forward->linked); kept.isErr())
                return fail(vjp->forward->context, kept.error());
        ResidualSources sources;
        PlaceResiduals place;
        if (vjp->checkpointing)
            place = [&](const uint8_t *inner) { return planCheckpoint(*vjp, inner, *block, sources); };
        const VernonStatus status = callProgram(*vjp->forward, record, size, nullptr, block.get(), point, place);
        if (status == VERNON_STATUS_OK)
            *residuals = block.release();
        return status;
    });
}

VernonStatus vernonScheduleVjpBackward(VernonScheduleVjp *vjp, const void *record, size_t size,
                                       const VernonScheduleResiduals *residuals, uint64_t *point) {
    if (!vjp || !residuals)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return vernon::runtime::capi::runtimeStatusBoundary(vjp->backward->context, [&]() {
        if (auto prepared = prepareBackward(*vjp, *residuals); prepared.isErr())
            return fail(vjp->backward->context, prepared.error(), VERNON_STATUS_INTERNAL_ERROR);
        const VernonScheduleResiduals *block = residuals->replayed ? residuals->replayed.get() : residuals;
        return callProgram(*vjp->backward, record, size, block, nullptr, point);
    });
}

void vernonScheduleResidualsDestroy(VernonScheduleResiduals *residuals) { delete residuals; }

uint64_t vernonScheduleResidualsGetByteSize(const VernonScheduleResiduals *residuals) {
    uint64_t bytes = 0;
    if (!residuals)
        return bytes;
    for (const VernonScheduleResiduals::Entry &entry : residuals->entries)
        bytes += entry.tape || !entry.held ? 0 : elementCount(entry.shape) * entry.elementSize;
    return bytes;
}

VernonStatus vernonScheduleResidualsGetCheckpoint(const VernonScheduleResiduals *residuals,
                                                  VernonScheduleCheckpoint *checkpoint) {
    if (!residuals || !checkpoint || checkpoint->struct_size < sizeof(*checkpoint))
        return VERNON_STATUS_INVALID_ARGUMENT;
    *checkpoint = residuals->checkpoint;
    checkpoint->struct_size = sizeof(*checkpoint);
    return VERNON_STATUS_OK;
}

size_t vernonScheduleVjpPassCount(const VernonScheduleVjp *vjp) { return vjp ? vjp->passes.size() : 0; }

VernonStatus vernonScheduleVjpGetPass(const VernonScheduleVjp *vjp, const VernonScheduleResiduals *residuals,
                                      size_t pass, VernonScheduleVjpPass *output) {
    if (!vjp || !residuals || pass >= vjp->passes.size() || !output || output->struct_size < sizeof(*output))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonScheduleResiduals *taped = residuals->replayed ? residuals->replayed.get() : residuals;
    const VernonScheduleVjp::Pass &found = vjp->passes[pass];
    VernonScheduleVjpPass result{};
    result.struct_size = sizeof(result);
    result.node = found.node;
    result.name = {found.name.data(), found.name.size()};
    result.recomputed = residuals->checkpoint.present && pass < residuals->checkpoint.recomputed_nodes ? 1U : 0U;
    for (uint32_t at : found.entries) {
        if (at >= residuals->entries.size() || at >= taped->entries.size())
            return VERNON_STATUS_INVALID_ARGUMENT;
        const VernonScheduleResiduals::Entry &entry = residuals->entries[at];
        if (!entry.tape) {
            result.residual_bytes += entry.held ? elementCount(entry.shape) * entry.elementSize : 0;
            continue;
        }
        const VernonScheduleResiduals::Entry &taping = taped->entries[at];
        const VernonScheduleTape *tape =
            taping.offset < taped->tapes.size() ? taped->tapes[taping.offset].get() : nullptr;
        if (tape && tape->device.data) {
            ++result.tape_count;
            result.tape_logical_bytes += tape->device.dataBytes;
            result.tape_resident_bytes += tape->device.dataBytes;
            result.tape_allocated_bytes += tape->device.dataBytes;
            continue;
        }
        if (!tape || !tape->batch)
            continue;
        const vernon::runtime::ad::HostStaticTapeBatch &batch = *tape->batch;
        ++result.tape_count;
        result.dynamic_tape |= batch.hasDynamicLanes() ? 1U : 0U;
        result.control_history |= batch.hasControlHistory() ? 1U : 0U;
        result.tape_logical_bytes += batch.logicalBytes();
        result.tape_resident_bytes += batch.isCompacted() ? batch.logicalBytes() : batch.residentBytes();
        result.tape_allocated_bytes += batch.allocatedBytes();
    }
    *output = result;
    return VERNON_STATUS_OK;
}

VernonStatus vernonScheduleGetGraphicsTelemetry(const VernonLinkedSchedule *schedule,
                                                VernonScheduleGraphicsTelemetry *telemetry) {
    if (!schedule || !telemetry || telemetry->struct_size < sizeof(*telemetry))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const vernon::hal::GraphicsTelemetry drawn =
        schedule->recording.commands ? schedule->recording.commands->telemetry() : vernon::hal::GraphicsTelemetry{};
    telemetry->render_scopes = drawn.renderScopes;
    telemetry->draws = drawn.draws;
    return VERNON_STATUS_OK;
}

VernonStatus vernonScheduleSetTimings(VernonLinkedSchedule *schedule, uint32_t enabled) {
    if (!schedule)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (schedule->recording.commands)
        schedule->recording.commands->setTimed(enabled != 0);
    return VERNON_STATUS_OK;
}

VernonStatus vernonScheduleGetScopeTimings(const VernonLinkedSchedule *schedule, uint64_t point,
                                           VernonScopeTiming *scopes, size_t capacity, size_t *count) {
    if (!schedule || !count)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return vernon::runtime::capi::runtimeStatusBoundary(schedule->context, [&]() {
        *count = 0;
        const vernon::hal::SubmissionTimings timings = schedule->device->timings(point);
        if (!timings.retired)
            return fail(schedule->context, "the schedule call was not waited for", VERNON_STATUS_NOT_READY);
        if (!timings.scopes)
            return fail(schedule->context, "the schedule call has no timings", VERNON_STATUS_UNSUPPORTED_TARGET);
        *count = timings.scopes->size();
        if (!scopes)
            return VERNON_STATUS_OK;
        if (capacity < timings.scopes->size())
            return fail(schedule->context, "the scope timings do not fit");
        for (size_t index = 0; index < timings.scopes->size(); ++index) {
            const vernon::hal::ScopeTiming &scope = (*timings.scopes)[index];
            scopes[index] = {scope.firstNode, scope.nodeCount, {}, scope.begin, scope.end};
        }
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonScheduleWait(VernonRuntimeContext *context, uint64_t point) {
    if (!context)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return vernon::runtime::capi::runtimeStatusBoundary(context, [&]() {
        std::shared_ptr<vernon::hal::Device> device;
        {
            std::lock_guard<std::mutex> lock(context->halDeviceMutex);
            device = context->halDevice;
        }
        if (!point || !device)
            return VERNON_STATUS_OK;
        auto waited = device->wait(point);
        if (waited.isOk() && consumeFailedScheduleWait())
            return fail(context, "schedule wait failure injected for testing", VERNON_STATUS_INTERNAL_ERROR);
        return waited.isOk() ? VERNON_STATUS_OK : fail(context, waited.error(), VERNON_STATUS_INTERNAL_ERROR);
    });
}

void vernonRuntimePrivateFailScheduleWaits(size_t count) { failedScheduleWaits.store(count); }

VernonScheduleTape *vernonScheduleTapeCreate(void) { return new (std::nothrow) VernonScheduleTape(); }

void vernonScheduleTapeDestroy(VernonScheduleTape *tape) { delete tape; }

} // extern "C"
