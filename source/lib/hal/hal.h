#ifndef VERNON_HAL_HAL_H
#define VERNON_HAL_HAL_H

// The HAL of specs/runtime/architecture_v2.md section 6: a thin layer over one device, below the linker and the
// executor. It records commands once and submits them many times; everything a submission varies lives in the
// argument record it is given.

#include "VernonCommon.h"
#include "VernonResult.hpp"
#include "VernonRuntimeProvider.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vernon::hal {

// A graphics node of a resolved Program, which only a graphics device draws (runtime/schedule/pass_plan.h).
struct GraphicsSource;

struct ExecutableSource {
    VernonCpuEntryPoint cpuEntry{};
    uint32_t workgroup[3]{1, 1, 1};
    // Optional; a CPU dispatch runs spans of whole workgroups through it instead of one range per workgroup.
    VernonCpuChunkEntryPoint cpuChunkEntry{};
    // A CUDA kernel: its PTX, its entry, and where each kernel parameter starts in its packed arguments.
    std::string artifact;
    std::string entry;
    std::vector<uint64_t> parameterOffsets;
    std::shared_ptr<const GraphicsSource> graphics;
};

class Executable {
public:
    virtual ~Executable() = default;
};

enum class CommandKind : uint8_t {
    // Reads its packed arguments and its three uint32_t workgroup counts from the submitted record.
    Dispatch,
    // Zeroes memory; its arguments are the address and the byte count, 8 bytes each.
    Fill,
    // Copies one leaf of every element of a view between its Storage and a dense carrier (LeafCopy). On a graphics
    // device its PassOffsets name the two buffers instead of its arguments.
    Copy,
    // Draws a graphics node from the ARGUMENT, PASS, DRAW, DYNAMIC, STREAM and TABLE slots its PassOffsets name, or
    // dispatches a compute node of a graphics device from its ARGUMENT slots.
    Pass,
};

// The workgroup counts of a compute pass. Per axis a constant, or the scalar host Tensor of the ARGUMENT slot at
// `argument` (`scalarBytes` 4 or 8, negative values rejected when signed); a linearized grid is instead the element
// count of the first Stage argument that is a Tensor with a shape.
struct ComputeGrid {
    uint64_t constant[3]{};
    uint64_t argument[3]{UINT64_MAX, UINT64_MAX, UINT64_MAX};
    uint8_t scalarBytes[3]{};
    bool scalarSigned[3]{};
    bool linearized{};
};

// The workgroup counts of `grid` in `record`, before a linearized grid takes its count from the pass's arguments.
[[nodiscard]] Result<std::array<uint32_t, 3>, std::string> gridCounts(const ComputeGrid &grid, const uint8_t *record);

// A buffer the pass's Program owns for one call, such as `vd.zeros` scratch, which no caller binds: its bytes and
// the row-major element shape and byte strides of the Value that owns it. The passes of one fragment that read the
// same Storage share one; a device allocates it on first use and zeroes it at the start of every call.
struct InternalBuffer {
    uint64_t bytes{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

// An extent of an internal buffer that follows the call (a `vd.empty_like` of an input): extent `axis` of the Tensor
// in the ARGUMENT slot at record offset `argument`, or the buffer's static extent when `argument` is UINT64_MAX. Such a
// buffer's `bytes` is its element size, and its strides are row-major over the extents of each call.
struct InternalExtent {
    uint64_t argument{UINT64_MAX};
    uint32_t axis{};
};

// Where a pass reads its inputs in the submitted record: one VernonProgramArgument per Stage parameter, in Stage
// parameter order, then for a graphics pass a `const VernonRenderPass *`, a VernonDrawCommand, a VernonDynamicState,
// a VernonScheduleDrawStream and a `const VernonSchedulePassTable *`, whose offsets a compute pass leaves UINT64_MAX.
// A parameter that reads an internal buffer has the offset UINT64_MAX and its buffer in `internals`, which is empty
// or has one entry per parameter. A parameter that reads an attachment of an earlier pass's RenderPass (a color or
// depth output a later pass samples or a dispatch rewrites) has the offset UINT64_MAX and its attachment in
// `attachments`, which is empty or has one entry per parameter.
struct AttachmentRead {
    // The record offset of the RenderPass slot, or UINT64_MAX for a parameter that reads no attachment.
    uint64_t renderPass{UINT64_MAX};
    // The index in the RenderPass's color attachments it reads, or UINT32_MAX for the depth attachment.
    uint32_t color{};
};

struct PassOffsets {
    std::vector<uint64_t> parameters;
    std::vector<std::shared_ptr<const InternalBuffer>> internals;
    // Empty, or per parameter the extents, one per axis, of an internal buffer whose size follows the call, else none.
    std::vector<std::vector<InternalExtent>> internalExtents;
    std::vector<AttachmentRead> attachments;
    // Empty, or per parameter whether the call zeroes the buffer range its argument names before its first pass: a
    // residual Storage the Program owns, which the caller allocates and the pass reads from its argument.
    std::vector<uint8_t> zeroed;
    // Empty, or per parameter the grid of the forward dispatch whose autodiff tape launch metadata the parameter reads
    // (its three uint32_t workgroup counts, passed as a host Tensor); such a parameter has the offset UINT64_MAX.
    std::vector<std::optional<ComputeGrid>> launches;
    uint64_t renderPass{UINT64_MAX};
    uint64_t draw{UINT64_MAX};
    uint64_t dynamic{UINT64_MAX};
    uint64_t stream{UINT64_MAX};
    uint64_t table{UINT64_MAX};
    bool compute{};
    ComputeGrid grid;

    bool present() const { return renderPass != UINT64_MAX || compute; }
};

// The autodiff tape of a CPU dispatch: the record offset of its VernonScheduleTape pointer, and the offsets in the
// dispatch's arguments of the allocator descriptor and root region each lane gets. A writing dispatch fills the tape
// anew, one lane per invocation, and seals it; a reading one replays it.
struct DispatchTape {
    uint64_t tapeOffset{UINT64_MAX};
    uint64_t allocatorOffset{UINT64_MAX};
    uint64_t rootOffset{UINT64_MAX};
    // The bytes each lane's static tape holds before it grows (a tape Value's minimum_tape_stride_bytes).
    uint64_t stride{};
    bool writes{};

    bool present() const { return tapeOffset != UINT64_MAX; }
};

// After a CPU dispatch with a results frame, `size` bytes from `frameOffset` of the frame go to `bufferOffset` of the
// buffer whose address is the record's 8 bytes at `addressOffset`.
struct ResultCommit {
    uint64_t addressOffset{};
    uint64_t frameOffset{};
    uint64_t bufferOffset{};
    uint64_t size{};
};

// A range of a CPU results frame that is a gradient: each lane writes a frame of its own, and the range is the sum of
// the lanes' ranges, in lane order, of `dtype` (f32 or f64) scalars.
struct ResultSum {
    uint64_t frameOffset{};
    uint64_t size{};
    uint32_t dtype{};
};

// A record field a device sets before a dispatch reads its arguments: at `at`, 8 bytes, the device address of the
// record's bytes from `target`, which stay on the device until the dispatch completes. A CUDA kernel takes a Struct,
// Tuple or ranked Tensor Value by pointer to such bytes; devices that run nothing such reject it.
struct RecordAddress {
    uint64_t at{};
    uint64_t target{};
};

// A Copy's layouts. Its arguments are the Storage's address, the carrier's address, then the view's offset, `rank`
// extents and `rank` strides, 8 bytes each, offset and strides in Storage elements. Element i of the view, at Storage
// element `offset + sum(index * stride)`, has `bytes` at `storageOffset` of its `storageElement` bytes; the carrier
// holds the view's elements densely in row-major order, the leaf at `carrierOffset` of `carrierElement` bytes. A
// gather copies Storage to carrier, a scatter the reverse. A graphics device's Copy has no arguments and ignores
// `rank`: its PassOffsets have two parameters, the Storage's argument, whose view it copies, and the carrier's
// internal buffer.
struct LeafCopy {
    uint32_t rank{};
    uint64_t storageElement{};
    uint64_t storageOffset{};
    uint64_t carrierElement{};
    uint64_t carrierOffset{};
    uint64_t bytes{};
    bool scatter{};

    uint64_t argumentsSize() const { return (3 + 2 * uint64_t{rank}) * sizeof(uint64_t); }
};

struct DispatchCommand {
    CommandKind kind{CommandKind::Dispatch};
    const Executable *executable{};
    uint64_t argumentsOffset{};
    uint64_t argumentsSize{};
    uint64_t gridOffset{};
    DispatchTape tape{};
    // A CPU kernel with results writes them to a zeroed frame of `resultsSize` bytes, committed after the dispatch.
    uint64_t resultsSize{};
    std::vector<ResultSum> resultSums;
    std::vector<ResultCommit> results;
    PassOffsets pass{};
    std::vector<RecordAddress> recordAddresses;
    LeafCopy copy{};
    // The graph node it runs, which a timed dispatch's ScopeTiming names.
    uint32_t node{};
};

// Device memory with one home. On a CPU or CUDA device its address is what kernels and records name it by; a graphics
// device's buffer has none, and an argument names `size` bytes of it from `offset` by reference.
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual void *address() const = 0;
    virtual VernonRuntimeProviderResourceReference reference(uint64_t, uint64_t) const { return {}; }
};

// What the last submission of a command buffer drew; zero on devices that draw nothing.
struct GraphicsTelemetry {
    uint64_t renderScopes{};
    uint64_t draws{};
};

class CommandBuffer {
public:
    virtual ~CommandBuffer() = default;
    virtual GraphicsTelemetry telemetry() const { return {}; }
    // Whether later submissions time their scopes, which Device::timings reads; devices that time nothing ignore it.
    virtual void setTimed(bool) {}
};

// A point on the device queue's timeline. Each submission signals the next point; zero precedes every submission.
using TimelinePoint = uint64_t;

// The device time of a render scope of `nodeCount` graphics nodes from Program node `firstNode`, or of one compute
// node, from the begin of the submission's first timed scope.
struct ScopeTiming {
    uint32_t firstNode{};
    uint32_t nodeCount{};
    uint64_t begin{};
    uint64_t end{};
};

// What a device measured of a submission: nothing until it retired it in a wait, and no scopes for one it did not
// time or could not.
struct SubmissionTimings {
    bool retired{};
    std::optional<std::vector<ScopeTiming>> scopes;
};

class Device {
public:
    virtual ~Device() = default;

    virtual Result<std::unique_ptr<Executable>, std::string> createExecutable(const ExecutableSource &source) = 0;
    // At least `size` bytes, aligned to 256. Releasing a buffer waits for nothing: its owner first waits for the
    // submissions that use it.
    virtual Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) = 0;
    // Copies host bytes to the start of a buffer, after every earlier submission and before any later one; for data
    // written once, such as a schedule's constants.
    virtual Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) = 0;
    // The same without waiting: the copy is ordered after every earlier submission and before any later one, and the
    // host bytes are free on return. For data each call writes, such as an autodiff tape's replay segments.
    virtual Result<void, std::string> enqueueUpload(Buffer &, const void *, size_t) {
        return Result<void, std::string>{err(std::string("this device does not enqueue uploads"))};
    }
    // Copies the start of a buffer to the host; the submissions that write it must have been waited for.
    virtual Result<void, std::string> download(const Buffer &, void *, size_t) {
        return Result<void, std::string>{err(std::string("this device does not read buffers back"))};
    }
    // The executables must outlive the command buffer.
    virtual Result<std::unique_ptr<CommandBuffer>, std::string>
    record(const std::vector<DispatchCommand> &commands) = 0;
    // `record` must hold every offset the command buffer reads; the device is done with it when submit returns. A CPU
    // device has also completed the work by then; a GPU device has only enqueued it.
    virtual Result<TimelinePoint, std::string> submit(const CommandBuffer &commands, const uint8_t *record,
                                                      size_t size) = 0;
    virtual Result<void, std::string> wait(TimelinePoint point) = 0;
    // A device keeps the timings of its few latest timed submissions.
    virtual SubmissionTimings timings(TimelinePoint) { return {true, std::nullopt}; }
};

} // namespace vernon::hal

#endif
