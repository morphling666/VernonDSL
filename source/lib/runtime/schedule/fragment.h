#ifndef VERNON_RUNTIME_SCHEDULE_FRAGMENT_H
#define VERNON_RUNTIME_SCHEDULE_FRAGMENT_H

// Schedule fragments (specs/runtime/architecture_v2.md section 4.1): what one Program decides on its own. A fragment
// has no barriers, queues or addresses; the linker adds those for the graph it belongs to.

#include "VernonResult.hpp"
#include "VernonSchedule.h"
#include "hal/hal.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/schedule/tape.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vernon::schedule {

constexpr uint32_t kNoSlot = UINT32_MAX;

// The record bytes of a graphics slot kind (ARGUMENT, PASS, DRAW, DYNAMIC), or zero for any other kind.
inline uint64_t graphicsSlotSize(uint32_t kind) {
    switch (kind) {
    case VERNON_SCHEDULE_SLOT_ARGUMENT:
        return sizeof(VernonProgramArgument);
    case VERNON_SCHEDULE_SLOT_PASS:
        return sizeof(const VernonRenderPass *);
    case VERNON_SCHEDULE_SLOT_DRAW:
        return sizeof(VernonDrawCommand);
    case VERNON_SCHEDULE_SLOT_DYNAMIC:
        return sizeof(VernonDynamicState);
    case VERNON_SCHEDULE_SLOT_STREAM:
        return sizeof(VernonScheduleDrawStream);
    case VERNON_SCHEDULE_SLOT_TABLE:
        return sizeof(const VernonSchedulePassTable *);
    default:
        return 0;
    }
}

// Which buffer an operation reads or writes: a slot the caller binds, or a transient of the arena.
struct BufferAccess {
    bool transient{};
    uint32_t index{};
    uint32_t access{};
};

enum class OperationKind : uint8_t {
    Dispatch,
    // Zeroes a transient or a slot's whole buffer. Its arguments are the address and the byte count, 8 bytes each,
    // which the executor writes for a transient and the caller for a slot.
    Fill,
    // Gathers one leaf of a Struct Storage into a carrier transient or scatters it back (hal::LeafCopy). Its accesses
    // are the Storage then the carrier. A graphics fragment's carrier is an internal buffer its PassOffsets name, and
    // its one access is the Storage's.
    Copy,
    // Draws a graphics node: its executable is the node's pipeline, and its PassOffsets name its slots.
    Pass,
};

// A field of a GPU kernel's tape carrier argument, which the executor writes on every call from the carrier's buffer:
// its address, its capacity and extent in i32 words, a stride of 1 and an offset of 0.
struct DeviceTapeField {
    DeviceTapeCarrier carrier{};
    VernonScheduleField field{};
};

// A kernel's autodiff tape and the TAPE slot that is the node's tape Storage; a lone kernel's fragment has no such
// slot, so it does not link. On the CPU, where in its arguments each lane's allocator descriptor and root region go.
// On a GPU, the fields of its carrier arguments and its workgroup, which with the grid size the carriers.
struct TapeUse {
    uint64_t allocatorOffset{UINT64_MAX};
    uint64_t rootOffset{UINT64_MAX};
    uint32_t slot{kNoSlot};
    uint64_t stride{};
    bool writes{};
    bool device{};
    std::vector<DeviceTapeField> fields;
    uint32_t workgroup[3]{1, 1, 1};

    bool present() const { return allocatorOffset != UINT64_MAX || device; }
    void shift(uint64_t base) {
        if (allocatorOffset != UINT64_MAX)
            allocatorOffset += base;
        if (rootOffset != UINT64_MAX)
            rootOffset += base;
        for (DeviceTapeField &field : fields)
            field.field.offset += base;
    }
};

// A CPU kernel's result at `frameOffset` of the dispatch's zeroed results frame: a gradient (`reduction` its scalar
// type) is the sum of every lane's, anything else the last lane's. After the dispatch its `copies` move it to the
// buffer of `slot`, the Program output the node binds it to: `result` is its endpoint index, and once a node binds it,
// `slot` is set.
struct ResultUse {
    uint32_t result{};
    uint32_t slot{kNoSlot};
    uint64_t frameOffset{};
    uint64_t size{};
    std::optional<VernonDataType> reduction;
    std::vector<runtime::ValueByteCopy> copies;
};

// A record field each call sets to its dispatch's invocation count: the caller's grid times `workgroupVolume`, as
// the leading extent of a GPU backward kernel's invocation-linear cotangent carrier.
struct InvocationExtent {
    uint64_t offset{};
    uint32_t size{};
    uint64_t workgroupVolume{};
};

// An operation's arguments and grid are ranges of the fragment's record.
struct Dispatch {
    OperationKind kind{OperationKind::Dispatch};
    uint32_t executable{};
    uint64_t argumentsOffset{};
    uint64_t argumentsSize{};
    uint64_t gridOffset{};
    bool unitGridAxes[3]{};
    std::vector<BufferAccess> accesses;
    // When not empty, the executor writes the grid on every call: x is the product of these extents, one invocation
    // per element of a Program built-in's output, and y and z are 1.
    std::vector<VernonScheduleExtent> gridElements;
    // The (axis, scalar slot) pairs of a Program node's grid Values on axes its kernel runs one workgroup on: each
    // call fails unless the slot's u32 is 1.
    std::vector<std::pair<uint32_t, uint32_t>> unitGridSlots;
    std::vector<InvocationExtent> invocationExtents;
    TapeUse tape;
    uint64_t resultsSize{};
    std::vector<ResultUse> results;
    hal::PassOffsets pass;
    // Record fields the device sets to the device address of record bytes: a CUDA by-pointer Value's descriptor,
    // pointing at the bytes of its scalar slot.
    std::vector<hal::RecordAddress> recordAddresses;
    hal::LeafCopy copy;
};

using Extent = VernonScheduleExtent;

// A buffer the fragment allocates. Its view is the whole buffer, contiguous in row-major order.
struct Transient {
    uint32_t dtype{};
    uint64_t elementSize{};
    std::vector<Extent> extents;
    bool zeroed{};
    // The first and last operations that access it, in fragment order.
    uint32_t first{};
    uint32_t last{};
};

// The executor writes these fields of a transient, whose index `slot` holds, on every call: an address, its view's
// metadata, its capacity in elements, and a fill's byte count.
constexpr uint32_t kTransientFieldBytes = 100;
// The tape field of a graphics pass, whose devices bind buffers by reference rather than address: a
// VernonProgramArgument the executor writes on every call, referencing the carrier, which the pass reads as the
// parameter at its offset.
constexpr uint32_t kTapeFieldArgument = 101;

// A record field holding the device address of `constant` bytes into the schedule's constants, 8 bytes wide.
struct ConstantAddress {
    uint64_t recordOffset{};
    uint64_t constant{};
};

struct Fragment {
    std::vector<hal::ExecutableSource> executables;
    std::vector<Dispatch> operations;
    // For a Program's fragment: per graph node in order, the number of operations up to and including its own, so
    // the operations before `nodeOperationEnds[n]` run graph nodes 0 through n.
    std::vector<uint32_t> nodeOperationEnds;
    // The record's arguments, which a caller writes; `shapes[i]` is slot i's declared shape, -1 for a dynamic extent,
    // and `extents[i]` its extents as the caller must supply them. A slot's first binding is its offset; every
    // further binding repeats its bytes as an ADDRESS or VALUE field, and each binding of a TensorView has its own
    // offset, extent and stride fields.
    std::vector<VernonScheduleSlot> slots;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<Extent>> extents;
    std::vector<VernonScheduleField> fields;
    // For a Program's fragment: the slot of each ABI boundary slot of its graph, as (boundary slot id, slot). An output
    // that publishes an input view's Storage shares that input's slot.
    std::vector<std::pair<uint32_t, uint32_t>> boundarySlots;
    std::vector<Transient> transients;
    std::vector<VernonScheduleField> transientFields;
    // Device bytes fixed at link time, which the executor uploads once and kernels only read: the launch metadata
    // of a backward kernel, which is the grid of the forward node it differentiates. A constant's view metadata is
    // in the record template; only its address is written per call.
    std::vector<uint8_t> constants;
    std::vector<ConstantAddress> constantAddresses;
    uint64_t recordSize{};
    // The record's initial bytes, recordSize of them: constants of the ABI that no slot or field writes.
    std::vector<uint8_t> recordTemplate;
    // The caller writes the grid of the one dispatch of a kernel's fragment; a Program's grids are in its template.
    bool callerGrid{};
    // For a kernel's fragment: whether it is a Program built-in (copy, add) that runs one invocation per element of
    // its first TensorView argument, whatever grid its node records.
    bool elementGrid{};
    // For a kernel's fragment: the reflection argument index of each slot, and whether the slot is a backward
    // kernel's GPU launch metadata (`__vernon_ad_launch`, three u32 workgroup counts of the forward grid).
    std::vector<uint32_t> slotArguments;
    std::vector<bool> slotLaunchMetadata;
    // Per slot, the byte offset in its element of a TensorView's first storage leaf.
    std::vector<uint64_t> slotLeafOffset;
    // Whether the slot is a GPU backward kernel's invocation-linear cotangent carrier: the Program's view with a
    // leading axis of one element per invocation of the dispatch, with stride 0.
    std::vector<bool> slotInvocationCarrier;
    // Per slot, which carrier of a GPU kernel's autodiff tape it is, which a Program node binds to its tape Storage.
    std::vector<DeviceTapeCarrier> slotTapeCarrier;

    // Checks that every range lies in the record and no two overlap; that each TensorView slot has, per binding, one
    // offset field and one extent and stride field per dimension, and at most two addresses and one capacity; that
    // transients have the same per binding; that extents name buffer slots or are static; that the template covers
    // the record; and that operations name existing executables, slots and transients.
    [[nodiscard]] Result<void, std::string> validate() const;
};

// The fragment of one compiled kernel Stage on the CPU, from the Stage's reflection: one dispatch whose packed frame
// is the start of the record, followed by its grid. Fails for kernels that use a feature fragments do not carry yet
// (results, textures, Struct elements, Struct or by-value Tensor arguments). A kernel with an autodiff tape gets a
// fragment, which links only as a node of a Program graph that names its tape.
[[nodiscard]] Result<Fragment, std::string> cpuKernelFragment(const nlohmann::json &reflection,
                                                              const std::string &entry, VernonCpuEntryPoint cpuEntry,
                                                              VernonCpuChunkEntryPoint cpuChunkEntry = nullptr);
// The same from a reflection already parsed, its arguments numbered as the kernel numbers them.
[[nodiscard]] Result<Fragment, std::string> cpuKernelFragment(runtime::ReflectedEntry reflection,
                                                              const std::string &entry, VernonCpuEntryPoint cpuEntry,
                                                              VernonCpuChunkEntryPoint cpuChunkEntry = nullptr);

// The fragment of one compiled kernel Stage on CUDA: one dispatch whose arguments are the kernel's parameters in
// order, each at its natural alignment, followed by its grid. A TensorView passes as a memref descriptor, a scalar as
// itself, and the TensorView metadata as one struct after them. Fails as cpuKernelFragment does.
[[nodiscard]] Result<Fragment, std::string> cudaKernelFragment(const nlohmann::json &reflection,
                                                               const std::string &entry, std::string ptx);
[[nodiscard]] Result<Fragment, std::string> cudaKernelFragment(runtime::ReflectedEntry reflection,
                                                               const std::string &entry, std::string ptx);

// The fragment of one graph of a Program variant (specs/runtime/architecture_v2.md section 16.1), from the variant's
// `program` JSON and the kernel fragment of each logical Stage its nodes run. Slots are the graph's ABI boundary slots
// in order, then the `exposed` Storages (VernonScheduleProgramDescriptor); `zeroed` names the owned Storages that start
// as zeros. Fails for what fragments do not carry yet, and the Program then runs as a Program.
[[nodiscard]] Result<Fragment, std::string> programFragment(const nlohmann::json &program, const std::string &graph,
                                                            const std::map<std::string, Fragment> &stages,
                                                            const std::vector<uint32_t> &zeroed,
                                                            const std::vector<uint32_t> &exposed);

// The fragment of one graph of graphics nodes of a resolved Program (VernonScheduleProgramDescriptor::executable): a
// pass per node, whose executable draws or dispatches with the Stage the Program prepared. Its ARGUMENT slots are laid
// out as programFragment's slots are: the graph's ABI boundary slots, then the `exposed` Storages, then the captured
// scalars its nodes read. Fails for nodes whose Stage parameters do not each read one whole ARGUMENT slot or owned
// Storage of static size.
[[nodiscard]] Result<Fragment, std::string> graphicsProgramFragment(VernonProgramExecutable &executable,
                                                                    const std::string &graph,
                                                                    const std::vector<uint32_t> &exposed);

// The fragment of one compute graph of a Program loaded from a bundle, on the CPU or CUDA: programFragment over the
// variant's JSON and the kernel fragment of each Stage the executable loaded, with every owned Storage zeroed. A
// composed Program graph has no variant JSON; it links as a graph of its nodes.
[[nodiscard]] Result<Fragment, std::string> computeProgramFragment(VernonProgramExecutable &executable,
                                                                   const std::string &graph,
                                                                   const std::vector<uint32_t> &exposed);

} // namespace vernon::schedule

#endif
