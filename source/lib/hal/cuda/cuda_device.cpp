#include "hal/cuda/cuda_device.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)

#include "hal/cuda/cuda_state.h"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <utility>

namespace vernon::hal {
namespace {

using cuda::DeviceState;
using cuda::PreparedFunction;

struct CudaExecutable final : Executable {
    explicit CudaExecutable(DeviceState &state) : state(state) {}
    ~CudaExecutable() override { function.destroy(state); }

    DeviceState &state;
    PreparedFunction function;
    // The compiler's `<entry>__vernon_indirect` twin, when its parameter layout is the record's.
    cuda::Function indirect{};
    uint32_t workgroup[3]{};
    std::vector<uint64_t> parameterOffsets;
};

nvtxDomainHandle_t debugDomain() noexcept {
    static const nvtxDomainHandle_t domain = nvtxDomainCreateA("Vernon");
    return domain;
}

// An NVTX range around one dispatch's launch. Ranges are started and ended rather than pushed and popped because a
// push/pop range belongs to its thread, and submissions of one device may come from several threads.
class DebugRange {
public:
    DebugRange(bool enabled, const std::string &label) noexcept {
        if (!enabled || label.empty())
            return;
        nvtxEventAttributes_t attributes{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = label.c_str();
        range_ = nvtxDomainRangeStartEx(debugDomain(), &attributes);
        open_ = true;
    }
    DebugRange(const DebugRange &) = delete;
    DebugRange &operator=(const DebugRange &) = delete;
    ~DebugRange() {
        if (open_)
            nvtxDomainRangeEnd(debugDomain(), range_);
    }

private:
    nvtxRangeId_t range_{};
    bool open_{};
};

// The fills of a graph replay: row y of the grid zeroes the byte count at `arguments + 16y + 8` from the address at
// `arguments + 16y`, both read from the record in device memory like an indirect twin's, in 16-byte stores when both
// are 16-byte aligned. A grid-stride loop lets one grid serve any size, so a fill node never changes.
constexpr const char kFillKernel[] = R"(.version 6.0
.target sm_50
.address_size 64

.visible .entry vernon_fill(.param .u64 vernon_fill_arguments)
{
    .reg .pred %p<4>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<13>;

    ld.param.u64 %rd1, [vernon_fill_arguments];
    mov.u32 %r6, %ctaid.y;
    mul.wide.u32 %rd12, %r6, 16;
    add.u64 %rd1, %rd1, %rd12;
    cvta.to.global.u64 %rd1, %rd1;
    ld.global.nc.u64 %rd2, [%rd1];
    ld.global.nc.u64 %rd3, [%rd1+8];
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %nctaid.x;
    mul.wide.u32 %rd4, %r1, %r2;
    cvt.u64.u32 %rd5, %r3;
    add.u64 %rd4, %rd4, %rd5;
    mul.wide.u32 %rd6, %r4, %r2;
    mov.u32 %r5, 0;
    mov.u16 %rs1, 0;
    or.b64 %rd7, %rd2, %rd3;
    and.b64 %rd7, %rd7, 15;
    setp.ne.u64 %p1, %rd7, 0;
    @%p1 bra BYTES;
    shr.u64 %rd8, %rd3, 4;
WORDS:
    setp.ge.u64 %p2, %rd4, %rd8;
    @%p2 bra DONE;
    shl.b64 %rd9, %rd4, 4;
    add.u64 %rd10, %rd2, %rd9;
    st.global.v4.u32 [%rd10], {%r5, %r5, %r5, %r5};
    add.u64 %rd4, %rd4, %rd6;
    bra.uni WORDS;
BYTES:
    setp.ge.u64 %p3, %rd4, %rd3;
    @%p3 bra DONE;
    add.u64 %rd11, %rd2, %rd4;
    st.global.u8 [%rd11], %rs1;
    add.u64 %rd4, %rd4, %rd6;
    bra.uni BYTES;
DONE:
    ret;
}
)";
constexpr uint32_t kFillWorkgroup = 256;

// The parameters of one leaf copy (LeafCopy), passed by value. Element i of the carrier is the i-th element of the
// view in row-major order; an axis past the view's rank has extent 1.
struct LeafCopyParams {
    static constexpr uint32_t kAxes = 8;
    uint64_t storage{};
    uint64_t carrier{};
    int64_t offset{};
    uint64_t total{};
    int64_t extents[kAxes]{};
    int64_t strides[kAxes]{};
    uint64_t storageElement{};
    uint64_t storageOffset{};
    uint64_t carrierElement{};
    uint64_t carrierOffset{};
    uint64_t bytes{};
    uint64_t scatter{};
};
static_assert(sizeof(LeafCopyParams) == 208, "kLeafCopyKernel reads LeafCopyParams at fixed offsets");

// One lane per element over a grid-stride loop: it splits its index into the view's axes, last fastest, and copies
// the leaf's bytes between the Storage element and its carrier element, in the direction `scatter` names.
constexpr const char kLeafCopyKernel[] = R"(.version 6.0
.target sm_50
.address_size 64

.visible .entry vernon_leaf_copy(.param .align 8 .b8 p[208])
{
    .reg .pred %p<4>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<30>;

    ld.param.u64 %rd1, [p];
    ld.param.u64 %rd2, [p+8];
    ld.param.u64 %rd3, [p+16];
    ld.param.u64 %rd4, [p+24];
    ld.param.u64 %rd5, [p+160];
    ld.param.u64 %rd6, [p+168];
    ld.param.u64 %rd7, [p+176];
    ld.param.u64 %rd8, [p+184];
    ld.param.u64 %rd9, [p+192];
    ld.param.u64 %rd10, [p+200];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %nctaid.x;
    mul.wide.u32 %rd11, %r1, %r2;
    cvt.u64.u32 %rd12, %r3;
    add.u64 %rd11, %rd11, %rd12;
    mul.wide.u32 %rd13, %r4, %r2;
    setp.ne.u64 %p3, %rd10, 0;
LANE:
    setp.ge.u64 %p1, %rd11, %rd4;
    @%p1 bra DONE;
    mov.u64 %rd14, %rd11;
    mov.u64 %rd15, %rd3;
)"
#define VERNON_LEAF_COPY_AXIS(EXTENT, STRIDE)                                                                          \
    "    ld.param.u64 %rd16, [p+" #EXTENT "];\n"                                                                       \
    "    ld.param.u64 %rd17, [p+" #STRIDE "];\n"                                                                       \
    "    div.u64 %rd18, %rd14, %rd16;\n"                                                                               \
    "    mul.lo.u64 %rd19, %rd18, %rd16;\n"                                                                            \
    "    sub.u64 %rd19, %rd14, %rd19;\n"                                                                               \
    "    mad.lo.u64 %rd15, %rd19, %rd17, %rd15;\n"                                                                     \
    "    mov.u64 %rd14, %rd18;\n"
    VERNON_LEAF_COPY_AXIS(88, 152) VERNON_LEAF_COPY_AXIS(80, 144) VERNON_LEAF_COPY_AXIS(72, 136)
        VERNON_LEAF_COPY_AXIS(64, 128) VERNON_LEAF_COPY_AXIS(56, 120) VERNON_LEAF_COPY_AXIS(48, 112)
            VERNON_LEAF_COPY_AXIS(40, 104) VERNON_LEAF_COPY_AXIS(32, 96)
#undef VERNON_LEAF_COPY_AXIS
                R"(    mad.lo.u64 %rd20, %rd15, %rd5, %rd6;
    add.u64 %rd20, %rd1, %rd20;
    mad.lo.u64 %rd21, %rd11, %rd7, %rd8;
    add.u64 %rd21, %rd2, %rd21;
    selp.b64 %rd22, %rd21, %rd20, %p3;
    selp.b64 %rd23, %rd20, %rd21, %p3;
    mov.u64 %rd24, 0;
BYTE:
    setp.ge.u64 %p2, %rd24, %rd9;
    @%p2 bra NEXT;
    add.u64 %rd25, %rd22, %rd24;
    ld.global.u8 %rs1, [%rd25];
    add.u64 %rd26, %rd23, %rd24;
    st.global.u8 [%rd26], %rs1;
    add.u64 %rd24, %rd24, 1;
    bra.uni BYTE;
NEXT:
    add.u64 %rd11, %rd11, %rd13;
    bra.uni LANE;
DONE:
    ret;
}
)";
constexpr uint32_t kLeafCopyWorkgroup = 256;

// Whether the direct entry's PTX parameters, packed each at the next offset its alignment allows, lie at `offsets`.
// This is the layout the indirect twin loads its parameters from. A `.align` before the type is the parameter's own
// alignment; one after `.ptr` is the pointee's.
bool packedParameters(const std::string &ptx, const std::string &entry, const std::vector<uint64_t> &offsets) {
    const std::string signature = ".entry " + entry + "(";
    const size_t start = ptx.find(signature);
    if (start == std::string::npos)
        return false;
    const size_t close = ptx.find(')', start + signature.size());
    if (close == std::string::npos)
        return false;
    std::vector<std::string> tokens;
    size_t at = start + signature.size();
    while (at < close) {
        while (at < close && (std::isspace(static_cast<unsigned char>(ptx[at])) || ptx[at] == ','))
            ++at;
        const size_t begin = at;
        while (at < close && !std::isspace(static_cast<unsigned char>(ptx[at])) && ptx[at] != ',')
            ++at;
        if (at > begin)
            tokens.emplace_back(ptx, begin, at - begin);
    }
    uint64_t end = 0;
    size_t parameter = 0;
    for (size_t index = 0; index < tokens.size();) {
        if (tokens[index] != ".param")
            return false;
        ++index;
        uint64_t alignment = 0;
        if (index + 1 < tokens.size() && tokens[index] == ".align") {
            alignment = std::strtoull(tokens[index + 1].c_str(), nullptr, 10);
            index += 2;
        }
        if (index >= tokens.size() || tokens[index].size() < 3)
            return false;
        const uint64_t element = std::strtoull(tokens[index].c_str() + 2, nullptr, 10) / 8;
        uint64_t size = element;
        while (++index < tokens.size() && tokens[index] != ".param") {
            const size_t bracket = tokens[index].find('[');
            if (bracket != std::string::npos)
                size = element * std::strtoull(tokens[index].c_str() + bracket + 1, nullptr, 10);
        }
        if (!alignment)
            alignment = element;
        if (!size || !alignment || parameter >= offsets.size())
            return false;
        const uint64_t offset = (end + alignment - 1) / alignment * alignment;
        if (offset != offsets[parameter++])
            return false;
        end = offset + size;
    }
    return parameter == offsets.size();
}

class CudaBuffer final : public Buffer {
public:
    CudaBuffer(DeviceState &state, cuda::DevicePointer pointer) : state_(state), pointer_(pointer) {}
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;
    ~CudaBuffer() override { (void)state_.free(pointer_); }

    void *address() const override { return reinterpret_cast<void *>(static_cast<uintptr_t>(pointer_)); }

private:
    DeviceState &state_;
    cuda::DevicePointer pointer_;
};

// A command buffer of several commands whose kernels all have indirect twins replays as a CUDA graph: each stream
// launch costs several microseconds of host time, a graph launch about one. The twins read their parameters from the
// record in device memory, so a call copies its record into a slot of a ring and launches the graph instance bound to
// that slot; the instance's nodes change only when a grid or a fill does, which the slot's last record shows. A slot
// is rewritten once the event recorded after its last launch has completed, as that launch still reads it. Every
// mutable member is touched only under the owning device's lock.
struct CudaCommandBuffer final : CommandBuffer {
    static constexpr size_t kSlots = 4;

    struct Slot {
        cuda::GraphExec exec{};
        cuda::Event done{};
        bool pending{};
        // The record `exec` was last launched with, whose grids and fills its nodes hold.
        std::vector<uint8_t> record;
    };

    explicit CudaCommandBuffer(DeviceState &state) : state(state) {}
    CudaCommandBuffer(const CudaCommandBuffer &) = delete;
    CudaCommandBuffer &operator=(const CudaCommandBuffer &) = delete;
    ~CudaCommandBuffer() override {
        if (!graph && !ring && !staging)
            return;
        cuda::Driver &api = cuda::driver();
        if (state.makeCurrent() != cuda::kSuccess)
            return;
        // An in-flight graph instance is freed when it completes; the ring and the staging are not, so the stream
        // drains first.
        (void)state.synchronize();
        for (Slot &slot : slots) {
            if (slot.exec)
                api.graphExecDestroy(slot.exec);
            if (slot.done)
                api.eventDestroy(slot.done);
        }
        if (graph)
            api.graphDestroy(graph);
        if (ring)
            (void)state.free(ring);
        if (staging)
            api.hostFree(staging);
    }

    DeviceState &state;
    std::vector<DispatchCommand> commands;
    // Room for the largest launch's parameter pointers, which submissions fill under the device's lock.
    mutable std::vector<void *> parameters;
    // Whether every dispatch has an indirect twin, so the buffer may replay as a graph.
    bool indirect{};
    // Whether a dispatch takes record addresses, so even a stream launch stages its record in the ring.
    bool addresses{};
    mutable cuda::Graph graph{};
    // Per graph node, its first command and how many it runs: one dispatch, or a run of fills whose arguments lie
    // side by side in the record.
    struct Group {
        size_t command;
        size_t count;
    };
    mutable std::vector<Group> groups;
    mutable std::vector<cuda::GraphNode> nodes;
    // `kSlots` records of `stride` bytes on the device, and their pinned host staging.
    mutable cuda::DevicePointer ring{};
    mutable uint8_t *staging{};
    mutable size_t stride{};
    mutable Slot slots[kSlots];
    mutable size_t cursor{};
    // Set once the graph cannot be built or updated, so the buffer stays on stream launches.
    mutable bool streamOnly{};
    // A timed submission launches on the stream, with an event pair around each dispatch.
    std::atomic<bool> timed{false};

    void setTimed(bool enabled) override { timed.store(enabled, std::memory_order_relaxed); }
};

class CudaDevice final : public Device {
public:
    CudaDevice(std::shared_ptr<DeviceState> state, bool labels)
        : owner_(std::move(state)), state_(*owner_), labels_(labels) {}
    CudaDevice(const CudaDevice &) = delete;
    CudaDevice &operator=(const CudaDevice &) = delete;
    ~CudaDevice() override {
        if (!timings_.empty() && state_.makeCurrent() == cuda::kSuccess) {
            (void)state_.synchronize();
            for (TimedCall &call : timings_)
                destroyEvents(call);
        }
        (void)fill_.destroy(state_);
        (void)leafCopy_.destroy(state_);
    }

    Result<std::unique_ptr<Executable>, std::string> createExecutable(const ExecutableSource &source) override {
        using Created = Result<std::unique_ptr<Executable>, std::string>;
        if (source.artifact.empty() || source.entry.empty())
            return Created{err(std::string("CUDA executable has no PTX or entry"))};
        auto executable = std::unique_ptr<CudaExecutable>(new (std::nothrow) CudaExecutable(state_));
        if (!executable)
            return Created{err(std::string("cannot allocate a CUDA executable"))};
        const cuda::Result status =
            executable->function.create(state_, source.artifact.data(), source.artifact.size(), source.entry.c_str());
        if (status != cuda::kSuccess)
            return Created{err(cuda::describeResult(status, "CUDA kernel load"))};
        std::memcpy(executable->workgroup, source.workgroup, sizeof(executable->workgroup));
        try {
            executable->parameterOffsets = source.parameterOffsets;
            const std::string twin = source.entry + "__vernon_indirect";
            if (cuda::driver().hasGraphs() &&
                packedParameters(source.artifact, source.entry, source.parameterOffsets) &&
                cuda::driver().moduleGetFunction(&executable->indirect, executable->function.module, twin.c_str()) !=
                    cuda::kSuccess)
                executable->indirect = nullptr;
        } catch (const std::bad_alloc &) {
            return Created{err(std::string("cannot allocate a CUDA executable"))};
        }
        return Created{ok(std::unique_ptr<Executable>(std::move(executable)))};
    }

    Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) override {
        using Allocated = Result<std::unique_ptr<Buffer>, std::string>;
        cuda::DevicePointer pointer{};
        const cuda::Result status = state_.allocate(pointer, static_cast<size_t>(std::max<uint64_t>(size, 1)));
        if (status != cuda::kSuccess)
            return Allocated{err(cuda::describeResult(status, "CUDA arena allocation"))};
        auto buffer = std::unique_ptr<CudaBuffer>(new (std::nothrow) CudaBuffer(state_, pointer));
        if (!buffer) {
            (void)state_.free(pointer);
            return Allocated{err(std::string("cannot allocate a CUDA buffer"))};
        }
        return Allocated{ok(std::unique_ptr<Buffer>(std::move(buffer)))};
    }

    // Enqueued on the stream behind every submission so far, then drained, so the host bytes are free on return.
    Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) override {
        if (!size)
            return Result<void, std::string>{ok()};
        std::lock_guard<std::mutex> lock(mutex_);
        const auto target = static_cast<cuda::DevicePointer>(reinterpret_cast<uintptr_t>(buffer.address()));
        cuda::Result status = state_.makeCurrent();
        if (status == cuda::kSuccess)
            status = cuda::driver().copyHostToDeviceAsync(target, bytes, size, state_.stream);
        if (status == cuda::kSuccess)
            status = state_.synchronize();
        if (status != cuda::kSuccess)
            return Result<void, std::string>{err(cuda::describeResult(status, "CUDA constant upload"))};
        completed_ = submitted_;
        return Result<void, std::string>{ok()};
    }

    // Staged through pinned memory, which the stream releases once the copy has run.
    Result<void, std::string> enqueueUpload(Buffer &buffer, const void *bytes, size_t size) override {
        if (!size)
            return Result<void, std::string>{ok()};
        std::lock_guard<std::mutex> lock(mutex_);
        const auto target = static_cast<cuda::DevicePointer>(reinterpret_cast<uintptr_t>(buffer.address()));
        const cuda::Result status = state_.upload(target, bytes, size);
        if (status != cuda::kSuccess)
            return Result<void, std::string>{err(cuda::describeResult(status, "CUDA upload"))};
        return Result<void, std::string>{ok()};
    }

    Result<void, std::string> download(const Buffer &buffer, void *bytes, size_t size) override {
        if (!size)
            return Result<void, std::string>{ok()};
        std::lock_guard<std::mutex> lock(mutex_);
        const auto source = static_cast<cuda::DevicePointer>(reinterpret_cast<uintptr_t>(buffer.address()));
        const cuda::Result status = state_.download(bytes, source, size);
        if (status != cuda::kSuccess)
            return Result<void, std::string>{err(cuda::describeResult(status, "CUDA readback"))};
        return Result<void, std::string>{ok()};
    }

    Result<std::unique_ptr<CommandBuffer>, std::string> record(const std::vector<DispatchCommand> &commands) override {
        using Recorded = Result<std::unique_ptr<CommandBuffer>, std::string>;
        auto buffer = std::unique_ptr<CudaCommandBuffer>(new (std::nothrow) CudaCommandBuffer(state_));
        if (!buffer)
            return Recorded{err(std::string("cannot allocate a CUDA command buffer"))};
        size_t parameters = 0;
        buffer->indirect = commands.size() > 1 && cuda::driver().hasGraphs();
        bool copies = false;
        for (const DispatchCommand &command : commands) {
            if (command.kind == CommandKind::Fill)
                continue;
            if (command.kind == CommandKind::Copy) {
                const LeafCopy &copy = command.copy;
                if (command.argumentsSize != copy.argumentsSize() || copy.rank > LeafCopyParams::kAxes || !copy.bytes ||
                    copy.storageOffset > copy.storageElement || copy.bytes > copy.storageElement - copy.storageOffset ||
                    copy.carrierOffset > copy.carrierElement || copy.bytes > copy.carrierElement - copy.carrierOffset)
                    return Recorded{err(std::string("CUDA leaf copy is malformed"))};
                copies = true;
                continue;
            }
            if (command.tape.present())
                return Recorded{err(std::string("CUDA kernels with autodiff tapes are not linked yet"))};
            const auto *executable = dynamic_cast<const CudaExecutable *>(command.executable);
            if (!executable)
                return Recorded{err(std::string("CUDA command buffer names an executable of another device"))};
            for (uint64_t offset : executable->parameterOffsets)
                if (offset >= command.argumentsSize)
                    return Recorded{err(std::string("CUDA kernel parameter lies outside its arguments"))};
            parameters = std::max(parameters, executable->parameterOffsets.size());
            buffer->indirect = buffer->indirect && executable->indirect;
            buffer->addresses = buffer->addresses || !command.recordAddresses.empty();
        }
        try {
            buffer->commands = commands;
            buffer->parameters.resize(parameters);
        } catch (const std::bad_alloc &) {
            return Recorded{err(std::string("cannot allocate a CUDA command buffer"))};
        }
        // A graph node holds its parameters, and a copy's are the record's addresses and extents, so a command buffer
        // with copies launches on the stream.
        if (copies) {
            buffer->indirect = false;
            std::lock_guard<std::mutex> lock(mutex_);
            if (!leafCopy_.function && leafCopy_.create(state_, kLeafCopyKernel, sizeof(kLeafCopyKernel),
                                                        "vernon_leaf_copy") != cuda::kSuccess) {
                leafCopy_ = {};
                return Recorded{err(std::string("cannot load the CUDA leaf copy kernel"))};
            }
        }
        const bool fills = std::any_of(commands.begin(), commands.end(), [](const DispatchCommand &command) {
            return command.kind == CommandKind::Fill;
        });
        if (buffer->indirect && fills) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!fillAttempted_) {
                fillAttempted_ = true;
                if (fill_.create(state_, kFillKernel, sizeof(kFillKernel), "vernon_fill") != cuda::kSuccess)
                    fill_ = {};
            }
            buffer->indirect = fill_.function != nullptr;
        }
        return Recorded{ok(std::unique_ptr<CommandBuffer>(std::move(buffer)))};
    }

    Result<TimelinePoint, std::string> submit(const CommandBuffer &commands, const uint8_t *record,
                                              size_t size) override {
        using Submitted = Result<TimelinePoint, std::string>;
        const auto &tape = static_cast<const CudaCommandBuffer &>(commands);
        std::lock_guard<std::mutex> lock(mutex_);
        const bool timed = tape.timed.load(std::memory_order_relaxed);
        // An empty grid has no graph node, so such a record goes to the stream, as do labelled launches, whose ranges
        // a graph launch would merge into one.
        bool graphable = tape.indirect && !tape.streamOnly && !timed && !labels_;
        for (const DispatchCommand &command : tape.commands) {
            if (command.kind == CommandKind::Fill) {
                if (command.argumentsOffset > size || 2 * sizeof(uint64_t) > size - command.argumentsOffset)
                    return Submitted{err(std::string("CUDA fill reads outside its record"))};
                continue;
            }
            if (command.kind == CommandKind::Copy) {
                if (command.argumentsOffset > size || command.argumentsSize > size - command.argumentsOffset)
                    return Submitted{err(std::string("CUDA leaf copy reads outside its record"))};
                continue;
            }
            if (command.argumentsOffset > size || command.argumentsSize > size - command.argumentsOffset ||
                command.gridOffset > size || 3 * sizeof(uint32_t) > size - command.gridOffset)
                return Submitted{err(std::string("CUDA dispatch reads outside its record"))};
            for (const RecordAddress &address : command.recordAddresses)
                if (address.at > size || sizeof(uint64_t) > size - address.at || address.target >= size)
                    return Submitted{err(std::string("CUDA record address lies outside its record"))};
            uint32_t grid[3];
            std::memcpy(grid, record + command.gridOffset, sizeof(grid));
            graphable = graphable && grid[0] && grid[1] && grid[2];
        }
        cuda::Result status = state_.makeCurrent();
        if (status != cuda::kSuccess)
            return Submitted{err(cuda::describeResult(status, "CUDA submission"))};
        if (graphable) {
            bool launched = false;
            status = launchGraph(tape, record, size, launched);
            if (status != cuda::kSuccess)
                return Submitted{err(cuda::describeResult(status, "CUDA graph launch"))};
            if (launched)
                return Submitted{ok(++submitted_)};
        }
        cuda::Driver &api = cuda::driver();
        // Record addresses name the device copy of the record, so a stream launch reads its parameters from the
        // staged copy whose addresses are set.
        size_t staged = CudaCommandBuffer::kSlots;
        if (tape.addresses) {
            if (!ensureRing(tape, size) || size > tape.stride)
                return Submitted{err(std::string("cannot stage a CUDA record in device memory"))};
            staged = tape.cursor;
            status = waitSlot(tape, staged);
            if (status == cuda::kSuccess)
                status = stage(tape, record, size, staged);
            if (status != cuda::kSuccess)
                return Submitted{err(cuda::describeResult(status, "CUDA record staging"))};
            record = tape.staging + staged * tape.stride;
        }
        TimedCall call;
        // Destroys the events of a submission that fails before its timings are kept.
        struct Unkept {
            CudaDevice &device;
            TimedCall &call;
            ~Unkept() { device.destroyEvents(call); }
        } unkept{*this, call};
        for (const DispatchCommand &command : tape.commands) {
            if (command.kind == CommandKind::Fill) {
                const Fill fill = fillParams(record, command);
                if (!fill.bytes)
                    continue;
                status = api.memsetD8Async(fill.address, 0, static_cast<size_t>(fill.bytes), state_.stream);
                if (status != cuda::kSuccess)
                    return Submitted{err(cuda::describeResult(status, "CUDA fill"))};
                continue;
            }
            if (command.kind == CommandKind::Copy) {
                LeafCopyParams params = copyParams(command.copy, record + command.argumentsOffset);
                if (!params.total)
                    continue;
                void *arguments[1]{&params};
                const auto groups = static_cast<unsigned>(
                    std::min<uint64_t>((params.total + kLeafCopyWorkgroup - 1) / kLeafCopyWorkgroup, 65535));
                status = api.launchKernel(leafCopy_.function, groups, 1, 1, kLeafCopyWorkgroup, 1, 1, 0, state_.stream,
                                          arguments, nullptr);
                if (status != cuda::kSuccess)
                    return Submitted{err(cuda::describeResult(status, "CUDA leaf copy"))};
                continue;
            }
            // cuLaunchKernel copies the parameters it is pointed at, so the record is free once the launch returns.
            const cuda::KernelNodeParams launch = kernelParams(tape, record, command);
            const DebugRange range(labels_, command.label);
            if (timed) {
                if (status = recordEvent(call); status != cuda::kSuccess)
                    return Submitted{err(cuda::describeResult(status, "CUDA timer"))};
            }
            status =
                api.launchKernel(launch.function, launch.gridDimX, launch.gridDimY, launch.gridDimZ, launch.blockDimX,
                                 launch.blockDimY, launch.blockDimZ, 0, state_.stream, launch.kernelParams, nullptr);
            if (status != cuda::kSuccess)
                return Submitted{err(cuda::describeResult(status, "CUDA kernel launch"))};
            if (timed) {
                if (status = recordEvent(call); status != cuda::kSuccess)
                    return Submitted{err(cuda::describeResult(status, "CUDA timer"))};
                try {
                    call.nodes.push_back(command.node);
                } catch (const std::bad_alloc &) {
                    return Submitted{err(std::string("cannot allocate CUDA timers"))};
                }
            }
        }
        if (staged != CudaCommandBuffer::kSlots) {
            status = api.eventRecord(tape.slots[staged].done, state_.stream);
            if (status != cuda::kSuccess)
                return Submitted{err(cuda::describeResult(status, "CUDA record staging"))};
            tape.slots[staged].pending = true;
            tape.cursor = (staged + 1) % CudaCommandBuffer::kSlots;
        }
        ++submitted_;
        if (timed) {
            call.point = submitted_;
            try {
                if (timings_.size() == kKeptTimings) {
                    destroyEvents(timings_.front());
                    timings_.pop_front();
                }
                timings_.push_back(std::move(call));
            } catch (const std::bad_alloc &) {
            }
            call = {};
        }
        return Submitted{ok(submitted_)};
    }

    // The spans of a timed call's dispatches, from the first one's begin event, once a wait completed the call.
    SubmissionTimings timings(TimelinePoint point) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (point > completed_)
            return {false, std::nullopt};
        const auto found =
            std::find_if(timings_.begin(), timings_.end(), [&](const TimedCall &call) { return call.point == point; });
        if (found == timings_.end())
            return {true, std::nullopt};
        TimedCall &call = *found;
        if (!call.events.empty()) {
            if (state_.makeCurrent() != cuda::kSuccess)
                return {true, std::nullopt};
            cuda::Driver &api = cuda::driver();
            const auto since = [&](cuda::Event event, uint64_t &nanoseconds) {
                float milliseconds = 0.0f;
                if (api.eventElapsedTime(&milliseconds, call.events.front(), event) != cuda::kSuccess)
                    return false;
                nanoseconds = static_cast<uint64_t>(static_cast<double>(milliseconds) * 1.0e6);
                return true;
            };
            std::vector<ScopeTiming> scopes(call.nodes.size());
            bool measured = true;
            for (size_t index = 0; measured && index < call.nodes.size(); ++index) {
                scopes[index].firstNode = call.nodes[index];
                scopes[index].nodeCount = 1;
                measured = since(call.events[2 * index], scopes[index].begin) &&
                           since(call.events[2 * index + 1], scopes[index].end);
            }
            destroyEvents(call);
            if (measured)
                call.scopes = std::move(scopes);
        }
        return {true, call.scopes};
    }

    Result<void, std::string> wait(TimelinePoint point) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (point > submitted_)
            return Result<void, std::string>{err(std::string("CUDA timeline point was never submitted"))};
        if (point <= completed_)
            return Result<void, std::string>{ok()};
        // The stream is in order, so draining it completes every point submitted so far.
        const cuda::Result status = state_.synchronize();
        if (status != cuda::kSuccess)
            return Result<void, std::string>{err(cuda::describeResult(status, "CUDA stream synchronize"))};
        completed_ = submitted_;
        return Result<void, std::string>{ok()};
    }

private:
    struct Fill {
        cuda::DevicePointer address{};
        uint64_t bytes{};
    };

    // A timed call's event pair per dispatch and the node each ran, turned into scopes on the first read.
    struct TimedCall {
        TimelinePoint point{};
        std::vector<uint32_t> nodes;
        std::vector<cuda::Event> events;
        std::optional<std::vector<ScopeTiming>> scopes;
    };
    static constexpr size_t kKeptTimings = 8;

    cuda::Result recordEvent(TimedCall &call) {
        cuda::Driver &api = cuda::driver();
        cuda::Event event{};
        cuda::Result status = api.eventCreate(&event, 0);
        if (status != cuda::kSuccess)
            return status;
        try {
            call.events.push_back(event);
        } catch (const std::bad_alloc &) {
            api.eventDestroy(event);
            return cuda::kErrorOutOfMemory;
        }
        return api.eventRecord(event, state_.stream);
    }

    static void destroyEvents(TimedCall &call) {
        for (cuda::Event event : call.events)
            cuda::driver().eventDestroy(event);
        call.events.clear();
    }

    // The arguments are the Storage and carrier addresses, the view's first element, then its extents and element
    // strides (LeafCopy::argumentsSize). A view with an empty axis copies nothing (total 0).
    static LeafCopyParams copyParams(const LeafCopy &copy, const uint8_t *arguments) {
        LeafCopyParams params;
        std::memcpy(&params.storage, arguments, sizeof(params.storage));
        std::memcpy(&params.carrier, arguments + sizeof(uint64_t), sizeof(params.carrier));
        std::memcpy(&params.offset, arguments + 2 * sizeof(uint64_t), sizeof(params.offset));
        params.total = 1;
        for (uint32_t axis = 0; axis < LeafCopyParams::kAxes; ++axis) {
            params.extents[axis] = 1;
            if (axis >= copy.rank)
                continue;
            std::memcpy(&params.extents[axis], arguments + (3 + axis) * sizeof(uint64_t), sizeof(int64_t));
            std::memcpy(&params.strides[axis], arguments + (3 + copy.rank + axis) * sizeof(uint64_t), sizeof(int64_t));
            params.total = params.extents[axis] > 0 ? params.total * static_cast<uint64_t>(params.extents[axis]) : 0;
        }
        params.storageElement = copy.storageElement;
        params.storageOffset = copy.storageOffset;
        params.carrierElement = copy.carrierElement;
        params.carrierOffset = copy.carrierOffset;
        params.bytes = copy.bytes;
        params.scatter = copy.scatter ? 1 : 0;
        return params;
    }

    static Fill fillParams(const uint8_t *record, const DispatchCommand &command) {
        Fill fill;
        std::memcpy(&fill.address, record + command.argumentsOffset, sizeof(fill.address));
        std::memcpy(&fill.bytes, record + command.argumentsOffset + sizeof(uint64_t), sizeof(fill.bytes));
        return fill;
    }

    // Points the tape's parameter array into the record; the result is valid until the next call.
    static cuda::KernelNodeParams kernelParams(const CudaCommandBuffer &tape, const uint8_t *record,
                                               const DispatchCommand &command) {
        const auto &executable = static_cast<const CudaExecutable &>(*command.executable);
        uint8_t *arguments = const_cast<uint8_t *>(record + command.argumentsOffset);
        for (size_t index = 0; index < executable.parameterOffsets.size(); ++index)
            tape.parameters[index] = arguments + executable.parameterOffsets[index];
        uint32_t grid[3];
        std::memcpy(grid, record + command.gridOffset, sizeof(grid));
        return cuda::KernelNodeParams{executable.function.function,
                                      grid[0],
                                      grid[1],
                                      grid[2],
                                      executable.workgroup[0],
                                      executable.workgroup[1],
                                      executable.workgroup[2],
                                      0,
                                      tape.parameters.data(),
                                      nullptr,
                                      nullptr,
                                      nullptr};
    }

    // The one parameter of an indirect twin: the device address of its arguments, and the parameter array the driver
    // reads it through.
    struct IndirectArgument {
        uint64_t address{};
        void *parameters[1]{&address};

        IndirectArgument() = default;
        IndirectArgument(const IndirectArgument &) = delete;
        IndirectArgument &operator=(const IndirectArgument &) = delete;
    };

    // The graph node of a group reading its arguments from slot `slot`: a dispatch's indirect twin, or the fill kernel
    // with a row per fill over a grid sized for the largest at the first record, whose loop covers any later size.
    // `argument` must outlive the use of the result.
    cuda::KernelNodeParams nodeParams(const CudaCommandBuffer &tape, const uint8_t *record,
                                      const CudaCommandBuffer::Group &group, size_t slot,
                                      IndirectArgument &argument) const {
        const DispatchCommand &command = tape.commands[group.command];
        argument.address = tape.ring + slot * tape.stride + command.argumentsOffset;
        cuda::KernelNodeParams params{};
        params.kernelParams = argument.parameters;
        params.gridDimY = params.gridDimZ = params.blockDimY = params.blockDimZ = 1;
        if (command.kind == CommandKind::Fill) {
            uint64_t bytes = 0;
            for (size_t index = 0; index < group.count; ++index)
                bytes = std::max(bytes, fillParams(record, tape.commands[group.command + index]).bytes);
            params.function = fill_.function;
            params.gridDimX = static_cast<unsigned>(std::min<uint64_t>(std::max<uint64_t>(bytes / 4096, 1), 1024));
            params.gridDimY = static_cast<unsigned>(group.count);
            params.blockDimX = kFillWorkgroup;
            return params;
        }
        const auto &executable = static_cast<const CudaExecutable &>(*command.executable);
        uint32_t grid[3];
        std::memcpy(grid, record + command.gridOffset, sizeof(grid));
        params.function = executable.indirect;
        params.gridDimX = grid[0];
        params.gridDimY = grid[1];
        params.gridDimZ = grid[2];
        params.blockDimX = executable.workgroup[0];
        params.blockDimY = executable.workgroup[1];
        params.blockDimZ = executable.workgroup[2];
        return params;
    }

    // Whether a node built from `left` serves `right`: a dispatch's node holds its grid; a fill's holds nothing that
    // varies.
    static bool sameNode(const uint8_t *left, const uint8_t *right, const DispatchCommand &command) {
        return command.kind == CommandKind::Fill ||
               !std::memcmp(left + command.gridOffset, right + command.gridOffset, 3 * sizeof(uint32_t));
    }

    // Builds the graph on first use, then copies `record` into the next slot and launches that slot's instance.
    // `launched` stays false when the graph cannot take this record, and the caller launches on the stream instead.
    cuda::Result launchGraph(const CudaCommandBuffer &tape, const uint8_t *record, size_t size, bool &launched) {
        cuda::Driver &api = cuda::driver();
        if (!tape.graph && !buildGraph(tape, record, size)) {
            tape.streamOnly = true;
            return cuda::kSuccess;
        }
        if (size > tape.stride || tape.slots[0].record.size() != size)
            return cuda::kSuccess;
        const size_t index = tape.cursor;
        CudaCommandBuffer::Slot &slot = tape.slots[index];
        if (const cuda::Result status = waitSlot(tape, index); status != cuda::kSuccess)
            return status;
        for (size_t node = 0; node < tape.groups.size(); ++node) {
            if (sameNode(slot.record.data(), record, tape.commands[tape.groups[node].command]))
                continue;
            IndirectArgument argument;
            const cuda::KernelNodeParams launch = nodeParams(tape, record, tape.groups[node], index, argument);
            if (api.graphExecKernelNodeSetParams(slot.exec, tape.nodes[node], &launch) != cuda::kSuccess) {
                tape.streamOnly = true;
                return cuda::kSuccess;
            }
        }
        std::memcpy(slot.record.data(), record, size);
        cuda::Result status = stage(tape, record, size, index);
        if (status == cuda::kSuccess)
            status = api.graphLaunch(slot.exec, state_.stream);
        if (status == cuda::kSuccess)
            status = api.eventRecord(slot.done, state_.stream);
        if (status != cuda::kSuccess)
            return status;
        slot.pending = true;
        tape.cursor = (index + 1) % CudaCommandBuffer::kSlots;
        launched = true;
        return cuda::kSuccess;
    }

    // One instance per slot, its dispatches reading that slot. Nodes run in command order, each depending on the one
    // before, as on the stream; fills in a row with adjacent arguments share a node, as they write distinct buffers.
    // The ring of `kSlots` device records of at least `size` bytes, their pinned staging, and an event per slot, made
    // on first use.
    bool ensureRing(const CudaCommandBuffer &tape, size_t size) {
        if (tape.staging)
            return true;
        cuda::Driver &api = cuda::driver();
        if (!tape.ring) {
            tape.stride = (size + 255) / 256 * 256;
            if (state_.allocate(tape.ring, tape.stride * CudaCommandBuffer::kSlots) != cuda::kSuccess) {
                tape.ring = 0;
                return false;
            }
        }
        for (CudaCommandBuffer::Slot &slot : tape.slots)
            if (!slot.done && api.eventCreate(&slot.done, cuda::kEventDisableTiming) != cuda::kSuccess) {
                slot.done = nullptr;
                return false;
            }
        void *host = nullptr;
        if (api.hostAllocate(&host, tape.stride * CudaCommandBuffer::kSlots, 0) != cuda::kSuccess)
            return false;
        tape.staging = static_cast<uint8_t *>(host);
        return true;
    }

    // A slot is rewritten once the launches that read it have completed.
    static cuda::Result waitSlot(const CudaCommandBuffer &tape, size_t index) {
        CudaCommandBuffer::Slot &slot = tape.slots[index];
        if (!slot.pending)
            return cuda::kSuccess;
        if (const cuda::Result status = cuda::driver().eventSynchronize(slot.done); status != cuda::kSuccess)
            return status;
        slot.pending = false;
        return cuda::kSuccess;
    }

    // Copies `record` into slot `index`'s staging with each record address set into the slot's device copy, then
    // enqueues the copy to the device.
    cuda::Result stage(const CudaCommandBuffer &tape, const uint8_t *record, size_t size, size_t index) {
        uint8_t *host = tape.staging + index * tape.stride;
        std::memcpy(host, record, size);
        const cuda::DevicePointer device = tape.ring + index * tape.stride;
        for (const DispatchCommand &command : tape.commands)
            for (const RecordAddress &address : command.recordAddresses) {
                const uint64_t target = device + address.target;
                std::memcpy(host + address.at, &target, sizeof(target));
            }
        return cuda::driver().copyHostToDeviceAsync(device, host, size, state_.stream);
    }

    bool buildGraph(const CudaCommandBuffer &tape, const uint8_t *record, size_t size) {
        cuda::Driver &api = cuda::driver();
        constexpr size_t kMaximumFills = 65535;
        try {
            tape.groups.clear();
            for (size_t index = 0; index < tape.commands.size(); ++index) {
                const DispatchCommand &command = tape.commands[index];
                if (command.kind == CommandKind::Fill && !tape.groups.empty()) {
                    CudaCommandBuffer::Group &last = tape.groups.back();
                    const DispatchCommand &first = tape.commands[last.command];
                    if (first.kind == CommandKind::Fill && last.count < kMaximumFills &&
                        command.argumentsOffset == first.argumentsOffset + 2 * sizeof(uint64_t) * last.count) {
                        ++last.count;
                        continue;
                    }
                }
                tape.groups.push_back({index, 1});
            }
            tape.nodes.assign(tape.groups.size(), nullptr);
            for (CudaCommandBuffer::Slot &slot : tape.slots)
                slot.record.assign(record, record + size);
        } catch (const std::bad_alloc &) {
            return false;
        }
        if (!ensureRing(tape, size))
            return false;
        if (api.graphCreate(&tape.graph, 0) != cuda::kSuccess) {
            tape.graph = nullptr;
            return false;
        }
        for (size_t node = 0; node < tape.groups.size(); ++node) {
            const cuda::GraphNode *dependency = node ? &tape.nodes[node - 1] : nullptr;
            IndirectArgument argument;
            const cuda::KernelNodeParams launch = nodeParams(tape, record, tape.groups[node], 0, argument);
            if (api.graphAddKernelNode(&tape.nodes[node], tape.graph, dependency, node ? 1 : 0, &launch) !=
                cuda::kSuccess)
                return false;
        }
        for (size_t index = 0; index < CudaCommandBuffer::kSlots; ++index) {
            CudaCommandBuffer::Slot &slot = tape.slots[index];
            if (api.graphInstantiate(&slot.exec, tape.graph, 0) != cuda::kSuccess) {
                slot.exec = nullptr;
                return false;
            }
            for (size_t node = 0; index && node < tape.groups.size(); ++node) {
                IndirectArgument argument;
                const cuda::KernelNodeParams launch = nodeParams(tape, record, tape.groups[node], index, argument);
                if (api.graphExecKernelNodeSetParams(slot.exec, tape.nodes[node], &launch) != cuda::kSuccess)
                    return false;
            }
        }
        return true;
    }

    std::shared_ptr<DeviceState> owner_;
    DeviceState &state_;
    std::mutex mutex_;
    const bool labels_;
    TimelinePoint submitted_{};
    TimelinePoint completed_{};
    std::deque<TimedCall> timings_;
    // The fill kernel of graph replays, loaded with the first command buffer that needs it.
    PreparedFunction fill_;
    bool fillAttempted_{};
    // The leaf copy kernel, loaded with the first command buffer that copies a leaf.
    PreparedFunction leafCopy_;
};

} // namespace

std::unique_ptr<Device> createCudaDevice(std::shared_ptr<cuda::DeviceState> state, bool labels) {
    if (!state)
        return nullptr;
    return std::unique_ptr<Device>(new (std::nothrow) CudaDevice(std::move(state), labels));
}

} // namespace vernon::hal

#else

namespace vernon::hal {

std::unique_ptr<Device> createCudaDevice(std::shared_ptr<cuda::DeviceState>, bool) { return nullptr; }

} // namespace vernon::hal

#endif
