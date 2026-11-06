#include "hal/metal/metal_device.h"
#include "hal/metal/metal_native.h"
#include "hal/metal/metal_resources.h"
#include "hal/metal/metal_state.h"
#include "hal/native_buffers.h"
#include "hal/native_images.h"

#include "VernonFormats.hpp"
#include "VernonSchedule.h"
#include "runtime/backend_metal.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/dispatch_contract.h"
#include "runtime/prepared_binding_plan.h"
#include "runtime/prepared_graphics_draw.h"
#include "runtime/runtime_state.h"
#include "runtime/schedule/pass_plan.h"
#include "runtime/vertex_attribute_capabilities.h"

#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vernon::hal {
namespace {

using runtime::PreparedComputeBindingPlan;
using runtime::PreparedGraphicsBindingPlan;

constexpr uint32_t kNone = UINT32_MAX;
constexpr uint64_t kNoOffset = UINT64_MAX;
constexpr size_t kMaxColors = VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS;
constexpr uint64_t kRecordBytes = sizeof(VernonRuntimeProviderDrawRecord);
static_assert(sizeof(VernonRuntimeProviderDrawRecord) == sizeof(MTLDrawIndexedPrimitivesIndirectArguments),
              "a draw record is the indirect arguments of one indexed draw");
// Recordings one command buffer keeps: a call reuses one whose passes are its own.
constexpr size_t kRecordings = 8;
// The frame slots of one recording. A slot is written again only once its last submission completed.
constexpr size_t kSlots = 3;
// Descriptor set g is argument buffer g, for g below 8, and push constants are buffer 15 (compiler_spirv_cross.cpp);
// vertex buffer binding b is buffer 16 + b, below Metal's 31 buffers.
constexpr uint32_t kMaxSets = 8;
constexpr uint32_t kFirstVertexBuffer = 16;
constexpr uint32_t kMaxVertexBuffers = 31 - kFirstVertexBuffer;
// macOS binds constant address space buffers at multiples of 256 bytes.
constexpr uint64_t kConstantAlignment = 256;
// Storage buffers bind at the offsets they bind at on DirectX 12, which fit every MSL type's alignment.
constexpr uint64_t kStorageAlignment = 16;
constexpr uint32_t kStreamGroup = 64;

template <typename T> Made<T> failure(std::string message) { return Made<T>{err(std::move(message))}; }

uint64_t align(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

uint64_t roundUp(uint64_t value, uint64_t minimum) {
    uint64_t result = minimum;
    while (result < value)
        result *= 2;
    return result;
}

// macOS blits buffer ranges in multiples of 4 bytes.
bool copyable(uint64_t offset, uint64_t size) {
#if TARGET_OS_OSX
    return offset % 4 == 0 && size % 4 == 0;
#else
    (void)offset;
    (void)size;
    return true;
#endif
}

Made<bool> valueCompleted(metal::DeviceState &state, uint64_t value) {
    std::lock_guard<std::mutex> lock(state.mutex);
    bool done = false;
    std::string error;
    if (!state.completed(value, done, error))
        return Made<bool>{err(std::move(error))};
    return Made<bool>{ok(done)};
}

Made<void> waitForValue(metal::DeviceState &state, uint64_t value) {
    std::lock_guard<std::mutex> lock(state.mutex);
    std::string error;
    if (!state.waitForValue(value, error))
        return failure<void>(std::move(error));
    return Made<void>{ok()};
}

// The reference kinds of hal/native_buffers.h a Metal device resolves.
enum class Kind : uint64_t {
    Buffer = kNativeBufferReference,
    View = kNativeViewReference,
    Sampler = kNativeSamplerReference,
};

// `kind` is the reference's identity kind: serials of different kinds may be equal.
struct ResolvedKey {
    uint64_t resource{};
    uint64_t kind{};
    bool operator==(const ResolvedKey &other) const { return resource == other.resource && kind == other.kind; }
};

struct ResolvedKeyHash {
    size_t operator()(const ResolvedKey &key) const {
        return std::hash<uint64_t>()(key.resource * 0x9E3779B97F4A7C15ull + key.kind);
    }
};

// The native objects one build resolved, which the plans it records keep alive until it ends.
using ResolvedCache = std::unordered_map<ResolvedKey, uintptr_t, ResolvedKeyHash>;

// The native object a reference of `state`'s device names: a metal::Buffer, metal::View or metal::Sampler. The
// recording that names it stays valid only while the resource lives, which the keys of the plans it records
// guarantee.
template <typename T>
Made<T *> resolveNative(const metal::DeviceState &state, const VernonRuntimeProviderResourceReference &reference,
                        Kind kind, ResolvedCache *cache = nullptr) {
    if (!reference.resource.value)
        return failure<T *>("a resource reference is empty");
    if (reference.identity != nativeIdentity(&state, static_cast<uint64_t>(kind)))
        return failure<T *>("a resource reference names another device's resource or one of another kind");
    const ResolvedKey key{reference.resource.value, static_cast<uint64_t>(kind)};
    if (cache)
        if (const auto found = cache->find(key); found != cache->end())
            return Made<T *>{ok(reinterpret_cast<T *>(found->second))};
    void *native = nullptr;
    switch (kind) {
    case Kind::Buffer:
        native = metalBuffer(reference.resource.value);
        break;
    case Kind::View:
        if (const NativeView *view = nativeViews().find(reference.resource.value))
            native = view->native;
        break;
    case Kind::Sampler:
        native = nativeSamplers().find(reference.resource.value);
        break;
    }
    if (!native)
        return failure<T *>("a resource is destroyed");
    if (cache)
        cache->emplace(key, reinterpret_cast<uintptr_t>(native));
    return Made<T *>{ok(static_cast<T *>(native))};
}

bool packedBytes(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE || entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
           (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
            entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM);
}

// The graphics state enums have Vulkan's values, which Metal's compare functions, stencil operations and cull modes
// share.
MTLBlendFactor blendFactor(uint32_t value) {
    constexpr MTLBlendFactor values[]{MTLBlendFactorZero,
                                      MTLBlendFactorOne,
                                      MTLBlendFactorSourceColor,
                                      MTLBlendFactorOneMinusSourceColor,
                                      MTLBlendFactorDestinationColor,
                                      MTLBlendFactorOneMinusDestinationColor,
                                      MTLBlendFactorSourceAlpha,
                                      MTLBlendFactorOneMinusSourceAlpha,
                                      MTLBlendFactorDestinationAlpha,
                                      MTLBlendFactorOneMinusDestinationAlpha};
    return value < std::size(values) ? values[value] : MTLBlendFactorZero;
}

MTLBlendOperation blendOperation(uint32_t value) {
    constexpr MTLBlendOperation values[]{MTLBlendOperationAdd, MTLBlendOperationSubtract,
                                         MTLBlendOperationReverseSubtract, MTLBlendOperationMin, MTLBlendOperationMax};
    return value < std::size(values) ? values[value] : MTLBlendOperationAdd;
}

MTLCompareFunction compareFunction(uint32_t value) {
    return value <= MTLCompareFunctionAlways ? static_cast<MTLCompareFunction>(value) : MTLCompareFunctionNever;
}

MTLStencilOperation stencilOperation(uint32_t value) {
    return value <= MTLStencilOperationDecrementWrap ? static_cast<MTLStencilOperation>(value)
                                                     : MTLStencilOperationKeep;
}

// Vulkan's color write mask has red in bit 0, Metal's in bit 3.
MTLColorWriteMask writeMask(uint32_t mask) {
    MTLColorWriteMask result = MTLColorWriteMaskNone;
    if (mask & 1u)
        result |= MTLColorWriteMaskRed;
    if (mask & 2u)
        result |= MTLColorWriteMaskGreen;
    if (mask & 4u)
        result |= MTLColorWriteMaskBlue;
    if (mask & 8u)
        result |= MTLColorWriteMaskAlpha;
    return result;
}

MTLVertexFormat vertexFormat(uint32_t dtype, uint32_t components) {
    static constexpr MTLVertexFormat formats[][4] = {
        {MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4},
        {MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4},
        {MTLVertexFormatHalf, MTLVertexFormatHalf2, MTLVertexFormatHalf3, MTLVertexFormatHalf4},
        {MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4},
    };
    const int row = dtype == VERNON_RUNTIME_PROVIDER_I32   ? 0
                    : dtype == VERNON_RUNTIME_PROVIDER_U32 ? 1
                    : dtype == VERNON_RUNTIME_PROVIDER_F16 ? 2
                    : dtype == VERNON_RUNTIME_PROVIDER_F32 ? 3
                                                           : -1;
    return row < 0 || components == 0 || components > 4 ? MTLVertexFormatInvalid : formats[row][components - 1];
}

uint32_t vertexFormatBytes(uint32_t dtype, uint32_t components) {
    return (dtype == VERNON_RUNTIME_PROVIDER_F16 ? 2u : 4u) * components;
}

MTLPrimitiveType primitiveType(uint32_t topology) {
    return topology == 1 ? MTLPrimitiveTypeLine : topology == 2 ? MTLPrimitiveTypePoint : MTLPrimitiveTypeTriangle;
}

MTLLoadAction loadAction(VernonRuntimeProviderLoadOperation operation) {
    return operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR      ? MTLLoadActionClear
           : operation == VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE ? MTLLoadActionLoad
                                                                : MTLLoadActionDontCare;
}

MTLStoreAction storeAction(VernonRuntimeProviderStoreOperation operation) {
    return operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD ? MTLStoreActionDontCare : MTLStoreActionStore;
}

MTLRenderStages renderStages(uint32_t role) {
    return role == runtime::kMetalVertex ? MTLRenderStageVertex : MTLRenderStageFragment;
}

// Packed values a stage reads from the slot arena: push values as its buffer 15, the rest through argument buffer
// members, uniform blocks in the constant address space.
enum class SlotKind : uint8_t { Push, Inline, Buffer, Image, Sampler };

// Where a push value of a binding set goes in its stage's push constants: graphics stage 0 is vertex, 1 fragment.
struct PushPlacement {
    uint32_t stage{};
    uint32_t dataOffset{};
    uint32_t offset{};
    uint32_t size{};
};

// How one prepared layout slot of a Stage binds: as member `member` of its stage's argument buffer `set`, or as a push
// value. A slot whose stage does not read it, which SPIRV-Cross dropped from the MSL, binds nowhere.
struct SlotPlan {
    SlotKind kind{};
    VernonRuntimeProviderBindingKind providerKind{};
    uint32_t set{};
    uint32_t size{};
    // A push value's offset in its set's push data.
    uint32_t dataOffset{};
    uint32_t stage{};
    uint32_t member{kNone};
    bool read{};
    bool written{};
    bool constant{};
    VernonTextureFormat storageFormat{};
};

// Descriptor set `set` of a Stage: its slots, the argument owners that decide its values, and per stage role the
// encoder of its argument buffer, which is null where the stage's MSL declares none.
struct SetPlan {
    std::vector<uint32_t> slots;
    std::vector<uint32_t> parameters;
    bool command{};
    uint32_t pushBytes{};
    std::vector<PushPlacement> placements;
    std::array<id<MTLArgumentEncoder>, 3> encoders{};
    bool arguments() const { return encoders[0] || encoders[1] || encoders[2]; }
};

// The HAL's native objects of one Stage, which live in its MetalPipelineState; the variants are made on first use.
// Command buffers retain the pipelines they encode, so nothing outlives the Stage's last submission by hand.
struct StageObjects {
    // A pipeline and the rasterization state Metal sets on the encoder.
    struct Variant {
        runtime::GraphicsVariantKey key;
        id<MTLRenderPipelineState> pipeline;
        id<MTLDepthStencilState> depthStencil;
        MTLCullMode cull{MTLCullModeNone};
        MTLWinding winding{MTLWindingCounterClockwise};
        MTLDepthClipMode clip{MTLDepthClipModeClip};
        float depthBias{};
        float slopeBias{};
    };

    bool graphics{};
    std::vector<SlotPlan> slots;
    std::vector<SetPlan> sets;
    id<MTLComputePipelineState> compute;
    MTLSize workgroup{1, 1, 1};
    id<MTLFunction> vertex;
    id<MTLFunction> fragment;
    std::array<uint32_t, 2> pushSizes{};
    std::vector<VernonRuntimeProviderVertexBinding> vertexBindings;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    std::mutex mutex;
    // A deque, so that the variants recordings point at stay where they are.
    std::deque<Variant> variants;
    // MTLArgumentEncoder is not thread safe; slots of every recording of the Stage encode through it.
    std::mutex encoding;
};

enum class Op : uint8_t {
    Fill,
    CopyRegions,
    DebugPush,
    DebugPop,
    DebugMarker,
    Dispatch,
    // Expands a counted device draw stream into the slot's draw arguments.
    ExpandStream,
    BeginScope,
    EndScope,
    BindGraphics,
    Sets,
    Push,
    VertexBuffers,
    Viewport,
    Scissor,
    Stencil,
    Draw,
    DrawIndexed,
    DrawIndirect,
    DrawHost,
};

// One command of a recording; its arguments depend on `op`. A null source names the slot's draw arguments.
struct Command {
    Op op{};
    uint32_t a{};
    uint32_t b{};
    uint32_t c{};
    uint32_t d{};
    uint32_t e{};
    MTLPrimitiveType primitive{MTLPrimitiveTypeTriangle};
    id<MTLBuffer> index;
    uint64_t indexOffset{};
    id<MTLBuffer> source;
    id<MTLBuffer> destination;
    uint64_t sourceOffset{};
    uint64_t destinationOffset{};
    uint64_t size{};
    const StageObjects *stage{};
    const StageObjects::Variant *variant{};
};

// A resource an argument buffer names, which an encoder makes resident for the stages that read it.
struct Use {
    id<MTLResource> resource;
    MTLResourceUsage usage{};
    MTLRenderStages stages{};
};

// A render scope's pass and the resources its draws' argument buffers name.
struct Rendering {
    MTLRenderPassDescriptor *pass;
    uint32_t firstUse{};
    uint32_t uses{};
};

// One member of an argument buffer. A null buffer names the slot's arena.
struct Member {
    uint32_t index{};
    SlotKind kind{};
    id<MTLBuffer> buffer;
    uint64_t offset{};
    id<MTLTexture> texture;
    id<MTLSamplerState> sampler;
};

// The argument buffers of one descriptor set of a recording, at `arguments` of the slot arena per stage role, and the
// push data of its set at `pushOffset` of the recording's: one per binding group, so draws whose groups' owners agree
// share it.
struct SetInstance {
    const StageObjects *stage{};
    uint32_t set{};
    std::array<uint64_t, 3> arguments{kNoOffset, kNoOffset, kNoOffset};
    std::array<std::vector<Member>, 3> members;
    std::vector<Use> uses;
    uint32_t pushOffset{kNone};
};

// Inline bytes a call fills again: of a graphics draw's binding plan, or of compute pass `pass`'s launch. Push bytes
// go to the recording's push data, others to the arena.
struct Site {
    const PlannedDraw *draw{};
    const PreparedGraphicsBindingPlan *graphics{};
    uint32_t pass{kNone};
    uint32_t slot{};
    uint32_t size{};
    uint64_t offset{};
    bool push{};
};

// The push constants of one draw's stages, which a call composes in the arena from the push data of the set
// instances pushSets[first, first + count).
struct PushBlock {
    const StageObjects *stage{};
    uint32_t first{};
    uint32_t count{};
    std::array<uint64_t, 2> offsets{};
};

// A host draw stream, whose records a call draws directly.
struct HostStream {
    const PlannedDraw *draw{};
    uint64_t indexCapacity{};
};

struct VertexBuffer {
    id<MTLBuffer> buffer;
    uint64_t offset{};
};

// One frame of a recording: its arena, which holds the argument buffers, the inline values and the push constants,
// and the draw arguments device draw streams expand to. A pooled slot keeps both.
struct Slot {
    id<MTLBuffer> arena;
    uint64_t arenaCapacity{};
    uint8_t *mapped{};
    id<MTLBuffer> arguments;
    uint64_t argumentCapacity{};
    uint64_t serial{};
    // The argument buffers and fixed bytes are written.
    bool written{};
};

// The slots of dropped recordings, kept for later ones, since a pass whose plan changes every call builds a recording
// every call. A slot is handed out again only once its last submission completed.
class SlotPool {
public:
    explicit SlotPool(std::shared_ptr<metal::DeviceState> state) : state_(std::move(state)) {}

    void keep(std::vector<std::unique_ptr<Slot>> &slots) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::unique_ptr<Slot> &slot : slots) {
            slot->written = false;
            if (slots_.size() < kPooledSlots)
                slots_.push_back(std::move(slot));
        }
        slots.clear();
    }

    std::unique_ptr<Slot> take() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto found = slots_.begin(); found != slots_.end(); ++found) {
            if ((*found)->serial) {
                auto completed = valueCompleted(*state_, (*found)->serial);
                if (completed.isErr() || !completed.value())
                    continue;
            }
            std::unique_ptr<Slot> slot = std::move(*found);
            slots_.erase(found);
            slot->serial = 0;
            return slot;
        }
        return nullptr;
    }

    metal::DeviceState &state() const { return *state_; }
    const std::shared_ptr<metal::DeviceState> &owner() const { return state_; }

private:
    static constexpr size_t kPooledSlots = 32;

    std::shared_ptr<metal::DeviceState> state_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

// The commands of one call's passes, as planned: a later call with the same key, the same plans of the same passes,
// encodes them again after it wrote the bytes that vary.
struct Recording {
    std::shared_ptr<SlotPool> pool;
    std::vector<uint64_t> key;
    std::vector<uint64_t> plans;
    std::vector<std::shared_ptr<StageObjects>> stages;
    std::vector<Command> commands;
    // The regions of the CopyRegions commands.
    std::vector<CopyRegion> copies;
    std::vector<std::string> labels;
    std::vector<Rendering> renderings;
    std::vector<VertexBuffer> vertexBuffers;
    // (set index, instance) of the push data a PushBlock composes.
    std::vector<std::pair<uint32_t, uint32_t>> pushSets;
    std::vector<PushBlock> pushBlocks;
    std::vector<uint8_t> push;
    std::vector<SetInstance> sets;
    // The set instances each dispatch binds.
    std::vector<uint32_t> dispatchSets;
    std::vector<Use> uses;
    std::vector<Site> sites;
    // Arena bytes that never change, written once per slot.
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> fixed;
    std::vector<HostStream> streams;
    uint64_t arenaBytes{};
    uint64_t argumentBytes{};
    // Serials of the images the commands read before writing them, and of the attachments they write.
    std::vector<uint64_t> reads;
    std::vector<uint64_t> writes;
    std::vector<std::unique_ptr<Slot>> slots;

    Recording() = default;
    Recording(const Recording &) = delete;
    Recording &operator=(const Recording &) = delete;
    ~Recording() {
        if (pool)
            pool->keep(slots);
    }
};

class MetalCommandBuffer final : public PassCommandBuffer {
public:
    using PassCommandBuffer::PassCommandBuffer;
    // Guarded by the pass cache's mutex; the most recently used last.
    mutable std::deque<std::unique_ptr<Recording>> recordings;
};

class MetalDevice;

// Builds the recording of one call's planned passes.
class Builder {
public:
    Builder(MetalDevice &device, Recording &recording, const PlannedPasses &planned,
            const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Made<void> build(const std::vector<const InternalBuffers::Internal *> &internals);

private:
    struct DrawPlan {
        const PlannedDraw *draw{};
        StageObjects *stage{};
        const StageObjects::Variant *variant{};
        std::array<uint32_t, kMaxSets> sets{};
        uint32_t vertexFirst{};
        uint32_t vertexCount{};
        runtime::DrawInvocation invocation{};
        id<MTLBuffer> index;
        uint64_t indexOffset{};
        // A device stream's records and counts, and where a counted stream expands to in the slot's draw arguments.
        id<MTLBuffer> records;
        uint64_t recordsOffset{};
        id<MTLBuffer> counts;
        uint64_t countsOffset{};
        uint64_t arguments{kNoOffset};
    };
    using FillValue = std::function<bool(size_t slot, VernonRuntimeProviderBindingValue &value,
                                         std::vector<uint8_t> &storage, std::string &error)>;
    using Uses = std::map<void *, size_t>;

    void add(const Command &command) { recording_.commands.push_back(command); }
    uint64_t arena(uint64_t size, uint64_t alignment) {
        const uint64_t offset = align(recording_.arenaBytes, alignment);
        recording_.arenaBytes = offset + size;
        return offset;
    }
    void debug(uint32_t kind, const std::string &label);
    void merge(std::vector<Use> &into, Uses &indices, const std::vector<Use> &uses);
    Made<StageObjects *> stage(VernonStageExecutable &stage, bool graphics);
    template <typename T> Made<T *> nativeOf(const VernonRuntimeProviderResourceReference &reference, Kind kind) {
        return resolveNative<T>(state_, reference, kind, &resolved_);
    }
    Made<uint32_t> instance(const StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                            const Site &source);
    Made<void> zero(const VernonRuntimeProviderResourceReference &reference, uint64_t offset, uint64_t size);
    Made<void> buildCompute(size_t at);
    Made<void> buildCopy(const PlannedCopy &copy);
    Made<void> buildScope(size_t begin, size_t end);
    Made<DrawPlan> planDraw(const PlannedPass &pass, const PlannedDraw &draw);
    Made<void> emitDraw(const DrawPlan &plan);
    Made<Rendering> rendering(const runtime::GraphicsRenderScope &scope);

    MetalDevice &device_;
    metal::DeviceState &state_;
    Recording &recording_;
    const PlannedPasses &planned_;
    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches_;
    std::map<std::vector<const void *>, uint32_t> instances_;
    ResolvedCache resolved_;
    std::unordered_map<const VernonStageExecutable *, StageObjects *> stages_;
    std::unordered_set<uint64_t> written_;
    // The attachment extent of the open render scope, which scissors stay within.
    uint32_t width_{};
    uint32_t height_{};
    // What the commands bound last within the open render scope.
    const StageObjects::Variant *variant_{};
    std::array<std::array<uint32_t, kMaxSets>, 2> sets_{};
    std::array<uint32_t, kMaxSets> pushed_{};
    const StageObjects *pushStage_{};
    std::optional<std::array<uint32_t, 4>> viewport_;
    std::optional<std::array<uint32_t, 4>> scissor_;
    std::optional<uint32_t> stencil_;
};

class MetalDevice final : public Device {
public:
    MetalDevice(VernonRuntimeContext &context, std::shared_ptr<metal::DeviceState> state)
        : context_(context), state_(*state), pool_(std::make_shared<SlotPool>(state)),
          buffers_(createMetalDeviceBuffers(state)), internals_(buffers_.get()) {}

    ~MetalDevice() override {
        std::lock_guard<std::mutex> lock(state_.mutex);
        std::string ignored;
        (void)state_.synchronize(ignored);
    }

    metal::DeviceState &state() const { return state_; }
    const std::shared_ptr<metal::DeviceState> &owner() const { return pool_->owner(); }

    Made<std::unique_ptr<Executable>> createExecutable(const ExecutableSource &source) override {
        if (!source.graphics || !source.graphics->resolved || !source.graphics->resolved->stage)
            return failure<std::unique_ptr<Executable>>("Metal devices run Program nodes only");
        return Made<std::unique_ptr<Executable>>{ok(std::unique_ptr<Executable>(new PassExecutable(source.graphics)))};
    }

    Made<std::unique_ptr<Buffer>> allocate(uint64_t size) override {
        if (!buffers_)
            return failure<std::unique_ptr<Buffer>>("the Metal device allocates no buffers");
        return buffers_->allocate(size);
    }

    Made<void> upload(Buffer &buffer, const void *bytes, size_t size) override {
        return buffers_->upload(buffer, bytes, size);
    }

    // An upload stages the bytes and copies them before any later submission, as an enqueued one must.
    Made<void> enqueueUpload(Buffer &buffer, const void *bytes, size_t size) override {
        return buffers_->upload(buffer, bytes, size);
    }

    Made<void> download(const Buffer &buffer, void *bytes, size_t size) override {
        return buffers_->download(buffer, bytes, size);
    }

    Made<std::unique_ptr<CommandBuffer>> record(const std::vector<DispatchCommand> &commands) override {
        if (auto checked = checkPassCommands(commands, "Metal"); checked.isErr())
            return Made<std::unique_ptr<CommandBuffer>>{err(std::move(checked).error())};
        return Made<std::unique_ptr<CommandBuffer>>{
            ok(std::unique_ptr<CommandBuffer>(new MetalCommandBuffer(commands)))};
    }

    Made<TimelinePoint> submit(const CommandBuffer &commands, const uint8_t *record, size_t) override;

    // Metal devices measure no scopes.
    SubmissionTimings timings(TimelinePoint point) override {
        retireCompleted();
        std::lock_guard<std::mutex> lock(mutex_);
        if (point >= next_ || (!pending_.empty() && pending_.front().point <= point))
            return {};
        return {true, std::nullopt};
    }

    Made<void> wait(TimelinePoint point) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (point >= next_)
            return failure<void>("the timeline point was not signalled yet");
        bool failed = false;
        while (!pending_.empty() && pending_.front().point <= point) {
            const Pending retired = pending_.front();
            pending_.pop_front();
            if (waitForValue(state_, retired.value).isErr())
                failed = true;
        }
        if (failed)
            return failure<void>("a pass submission failed on the device");
        return Made<void>{ok()};
    }

    // The objects of a Stage this device dispatches or draws with, made on first use.
    Made<std::shared_ptr<StageObjects>> stageObjects(VernonStageExecutable &stage, bool graphics);
    Made<const StageObjects::Variant *> variant(StageObjects &stage, const runtime::GraphicsVariantKey &key);

private:
    struct Pending {
        TimelinePoint point{};
        uint64_t value{};
    };

    Made<std::unique_ptr<Recording>> build(const MetalCommandBuffer &commands, const PlannedPasses &planned,
                                           const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Made<void> planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
                          std::vector<uint64_t> &key);
    Made<Slot *> acquire(Recording &recording);
    Made<void> createSlot(Recording &recording, Slot &slot);
    Made<void> fill(Recording &recording, Slot &slot, const PlannedPasses &planned,
                    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    void writeArguments(Recording &recording, Slot &slot);
    Made<void> encode(const Recording &recording, Slot &slot, id<MTLCommandBuffer> commands,
                      const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);

    // Retires the submissions that completed, oldest first.
    void retireCompleted() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty()) {
            auto completed = valueCompleted(state_, pending_.front().value);
            if (completed.isErr() || !completed.value())
                return;
            pending_.pop_front();
        }
    }

    VernonRuntimeContext &context_;
    metal::DeviceState &state_;
    std::shared_ptr<SlotPool> pool_;
    std::unique_ptr<DeviceBuffers> buffers_;
    InternalBuffers internals_;
    std::mutex mutex_;
    std::deque<Pending> pending_;
    TimelinePoint next_{1};
};

Made<id<MTLFunction>> compileFunction(id<MTLDevice> device, const runtime::MetalShaderCode &code) {
    using Compiled = Made<id<MTLFunction>>;
    if (code.source.empty())
        return failure<id<MTLFunction>>("the Stage has no MSL");
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion2_4;
    // As on every other backend, float operations keep IEEE semantics.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(code.source.c_str()) options:options error:&error];
    if (!library)
        return failure<id<MTLFunction>>("cannot compile the Stage's MSL: " + metal::describe(error));
    // SPIRV-Cross renames an entry point that MSL reserves, such as main, by appending 0.
    id<MTLFunction> function = [library newFunctionWithName:@(code.entry.c_str())];
    if (!function)
        function = [library newFunctionWithName:@((code.entry + "0").c_str())];
    if (!function && library.functionNames.count == 1)
        function = [library newFunctionWithName:library.functionNames.firstObject];
    if (!function)
        return failure<id<MTLFunction>>("the Stage's MSL has no function '" + code.entry + "'");
    return Compiled{ok(function)};
}

Made<std::shared_ptr<StageObjects>> buildStageObjects(metal::DeviceState &device, VernonStageExecutable &stage,
                                                      const runtime::MetalPipelineState &state, bool graphics) {
    using Built = Made<std::shared_ptr<StageObjects>>;
    auto objects = std::make_shared<StageObjects>();
    objects->graphics = graphics;
    const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layouts =
        graphics ? state.graphicsPlan.layouts : state.computePlan.layouts;
    const std::vector<runtime::SlotDependencies> dependencies =
        graphics ? runtime::preparedGraphicsSlotDependencies(state.graphicsPlan)
                 : runtime::preparedComputeSlotDependencies(state.computePlan, stage.bindingProjection);
    if (dependencies.size() != layouts.size() || state.slots.size() != layouts.size())
        return failure<std::shared_ptr<StageObjects>>("the Stage's binding dependencies do not match its layouts");
    uint32_t setCount = 0;
    for (uint32_t role = 0; role < 3; ++role)
        for (uint32_t set = 0; set < kMaxSets; ++set)
            if (state.argumentBuffers[role] & (1u << set))
                setCount = (std::max)(setCount, set + 1);
    for (const VernonRuntimeProviderBindingLayoutEntry &layout : layouts) {
        if (layout.set >= kMaxSets || layout.array_count != 1)
            return failure<std::shared_ptr<StageObjects>>("the Stage has a binding Metal devices do not bind");
        setCount = (std::max)(setCount, layout.set + 1);
    }
    objects->sets.resize(setCount);
    objects->slots.resize(layouts.size());
    for (size_t index = 0; index < layouts.size(); ++index) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = layouts[index];
        SetPlan &set = objects->sets[layout.set];
        SlotPlan &slot = objects->slots[index];
        set.slots.push_back(static_cast<uint32_t>(index));
        set.parameters.insert(set.parameters.end(), dependencies[index].parameters.begin(),
                              dependencies[index].parameters.end());
        set.command = set.command || dependencies[index].command;
        slot.providerKind = layout.kind;
        slot.set = layout.set;
        slot.size = layout.element_size;
        slot.stage = !graphics                                                   ? runtime::kMetalCompute
                     : layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ? runtime::kMetalVertex
                                                                                 : runtime::kMetalFragment;
        slot.member = state.slots[index].member;
        if (graphics && layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
            const uint32_t alignment = layout.element_alignment;
            if (!alignment || (alignment & (alignment - 1)) != 0 || layout.element_size == 0)
                return failure<std::shared_ptr<StageObjects>>("an inline value's alignment is invalid");
            slot.kind = SlotKind::Push;
            slot.read = state.slots[index].directBuffer == runtime::kMetalPushConstantBuffer;
            slot.dataOffset = set.pushBytes;
            set.pushBytes += layout.element_size;
            continue;
        }
        slot.read = slot.member != kNone;
        slot.written = (layout.access & 2u) != 0;
        switch (layout.kind) {
        case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
            slot.kind = SlotKind::Inline;
            slot.constant = true;
            break;
        case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
            // Compute inline values are read storage buffers.
            slot.kind = SlotKind::Inline;
            break;
        case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
            slot.kind = packedBytes(layout) ? SlotKind::Inline : SlotKind::Buffer;
            if (slot.kind == SlotKind::Inline && slot.written)
                return failure<std::shared_ptr<StageObjects>>("the Stage writes a uniform block");
            break;
        case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
            slot.kind = SlotKind::Image;
            break;
        case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
            slot.kind = SlotKind::Image;
            slot.written = true;
            slot.storageFormat = static_cast<VernonTextureFormat>(layout.storage_image_format);
            if (metal::pixelFormat(slot.storageFormat) == MTLPixelFormatInvalid)
                return failure<std::shared_ptr<StageObjects>>("a storage image's format has no Metal format");
            break;
        case VERNON_RUNTIME_PROVIDER_SAMPLER:
            slot.kind = SlotKind::Sampler;
            break;
        default:
            return failure<std::shared_ptr<StageObjects>>("the Stage has a binding Metal devices do not bind");
        }
        if (slot.kind == SlotKind::Inline && slot.size == 0)
            return failure<std::shared_ptr<StageObjects>>("an inline binding has no bytes");
    }
    for (SetPlan &set : objects->sets) {
        std::sort(set.parameters.begin(), set.parameters.end());
        set.parameters.erase(std::unique(set.parameters.begin(), set.parameters.end()), set.parameters.end());
    }
    // Each stage's push values form its push constants sorted by entry argument order, which the members of the
    // block the compiler declares follow, at offsets aligned to each value's alignment.
    std::vector<uint32_t> pushSlots;
    for (uint32_t index = 0; index < objects->slots.size(); ++index)
        if (objects->slots[index].kind == SlotKind::Push && objects->slots[index].read)
            pushSlots.push_back(index);
    std::stable_sort(pushSlots.begin(), pushSlots.end(), [&](uint32_t left, uint32_t right) {
        return layouts[left].argument_index < layouts[right].argument_index;
    });
    for (uint32_t index : pushSlots) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = layouts[index];
        const SlotPlan &slot = objects->slots[index];
        const uint32_t stageIndex = slot.stage == runtime::kMetalVertex ? 0 : 1;
        const auto offset = static_cast<uint32_t>(align(objects->pushSizes[stageIndex], layout.element_alignment));
        objects->sets[slot.set].placements.push_back({stageIndex, slot.dataOffset, offset, slot.size});
        objects->pushSizes[stageIndex] = offset + slot.size;
    }
    id<MTLDevice> native = device.native->device;
    @autoreleasepool {
        const auto encoders = [&](id<MTLFunction> function, uint32_t role) -> Made<void> {
            for (uint32_t set = 0; set < setCount; ++set) {
                if (!(state.argumentBuffers[role] & (1u << set)))
                    continue;
                @try {
                    objects->sets[set].encoders[role] = [function newArgumentEncoderWithBufferIndex:set];
                } @catch (NSException *exception) {
                    return failure<void>("cannot encode argument buffer " + std::to_string(set) + ": " +
                                         std::string(exception.reason.UTF8String ?: "no reason given"));
                }
                if (!objects->sets[set].encoders[role])
                    return failure<void>("cannot encode argument buffer " + std::to_string(set));
            }
            return Made<void>{ok()};
        };
        if (!graphics) {
            auto function = compileFunction(native, state.compute);
            if (function.isErr())
                return Built{err(std::move(function).error())};
            NSError *error = nil;
            objects->compute = [native newComputePipelineStateWithFunction:function.value() error:&error];
            if (!objects->compute)
                return failure<std::shared_ptr<StageObjects>>("cannot create the Stage's pipeline: " +
                                                              metal::describe(error));
            objects->workgroup = MTLSizeMake(stage.workgroupSize.x, stage.workgroupSize.y, stage.workgroupSize.z);
            if (uint64_t{stage.workgroupSize.x} * stage.workgroupSize.y * stage.workgroupSize.z >
                objects->compute.maxTotalThreadsPerThreadgroup)
                return failure<std::shared_ptr<StageObjects>>("the Stage's workgroup exceeds its pipeline's threads");
            if (auto made = encoders(function.value(), runtime::kMetalCompute); made.isErr())
                return Built{err(std::move(made).error())};
            return Built{ok(std::move(objects))};
        }
        auto vertex = compileFunction(native, state.vertex);
        if (vertex.isErr())
            return Built{err(std::move(vertex).error())};
        auto fragment = compileFunction(native, state.fragment);
        if (fragment.isErr())
            return Built{err(std::move(fragment).error())};
        objects->vertex = vertex.value();
        objects->fragment = fragment.value();
        if (auto made = encoders(objects->vertex, runtime::kMetalVertex); made.isErr())
            return Built{err(std::move(made).error())};
        if (auto made = encoders(objects->fragment, runtime::kMetalFragment); made.isErr())
            return Built{err(std::move(made).error())};
    }
    objects->vertexBindings = state.graphicsPlan.vertexBindings;
    objects->vertexAttributes = state.graphicsPlan.vertexAttributes;
    for (const VernonRuntimeProviderVertexBinding &binding : objects->vertexBindings)
        if (binding.binding >= kMaxVertexBuffers)
            return failure<std::shared_ptr<StageObjects>>("a vertex binding exceeds Metal's buffer indices");
    for (const VernonRuntimeProviderVertexAttribute &attribute : objects->vertexAttributes) {
        std::string diagnostic;
        if (!runtime::validateVertexAttributeCapability(runtime::VertexAttributeBackend::Metal, attribute,
                                                        kMaxVertexBuffers, false, diagnostic))
            return failure<std::shared_ptr<StageObjects>>(std::move(diagnostic));
        if (vertexFormat(attribute.dtype, attribute.component_count) == MTLVertexFormatInvalid)
            return failure<std::shared_ptr<StageObjects>>("a vertex attribute's format is unsupported");
    }
    return Built{ok(std::move(objects))};
}

Made<std::shared_ptr<StageObjects>> MetalDevice::stageObjects(VernonStageExecutable &stage, bool graphics) {
    using Found = Made<std::shared_ptr<StageObjects>>;
    if (!stage.backendState)
        return failure<std::shared_ptr<StageObjects>>("the Stage has no Metal pipeline");
    auto &state = ::runtimeBackendState<runtime::MetalPipelineState>(stage);
    std::lock_guard<std::mutex> lock(state.halMutex);
    if (!state.hal) {
        auto built = buildStageObjects(state_, stage, state, graphics);
        if (built.isErr())
            return Found{err(std::move(built).error())};
        state.hal = std::move(built).value();
    }
    auto objects = std::static_pointer_cast<StageObjects>(state.hal);
    if (objects->graphics != graphics)
        return failure<std::shared_ptr<StageObjects>>("the Stage is not a " +
                                                      std::string(graphics ? "graphics" : "compute") + " Stage");
    return Found{ok(std::move(objects))};
}

Made<const StageObjects::Variant *> MetalDevice::variant(StageObjects &stage, const runtime::GraphicsVariantKey &key) {
    using Found = Made<const StageObjects::Variant *>;
    std::lock_guard<std::mutex> lock(stage.mutex);
    for (const StageObjects::Variant &variant : stage.variants)
        if (runtime::graphicsVariantKeysEqual(variant.key, key))
            return Found{ok(&variant)};
    if (key.sampleCount != 1 || key.colorFormats.size() > kMaxColors ||
        (key.depthStencilFormat && key.depthStencilFormat != MTLPixelFormatDepth32Float &&
         key.depthStencilFormat != MTLPixelFormatDepth32Float_Stencil8) ||
        key.colorBlends.size() != key.colorFormats.size())
        return failure<const StageObjects::Variant *>(
            "the graphics pipeline uses an unsupported attachment configuration");
    const bool hasDepth = key.depthStencilFormat != MTLPixelFormatInvalid;
    const bool hasStencil = key.depthStencilFormat == MTLPixelFormatDepth32Float_Stencil8;
    const VernonRasterizationState &rasterization = key.rasterization;
    if (rasterization.cull_mode > 2 || !std::isfinite(rasterization.depth_bias_constant) ||
        !std::isfinite(rasterization.depth_bias_slope))
        return failure<const StageObjects::Variant *>("the graphics pipeline has an unsupported rasterization state");
    const VernonDepthStencilState &depthState = key.depthStencil;
    if ((!hasDepth && (depthState.depth_test || depthState.depth_write)) || (!hasStencil && depthState.stencil_test))
        return failure<const StageObjects::Variant *>("the graphics pipeline has an invalid depth stencil state");
    StageObjects::Variant made;
    made.key = key;
    @autoreleasepool {
        MTLRenderPipelineDescriptor *native = [[MTLRenderPipelineDescriptor alloc] init];
        native.vertexFunction = stage.vertex;
        native.fragmentFunction = stage.fragment;
        if (!stage.vertexAttributes.empty()) {
            MTLVertexDescriptor *vertex = [MTLVertexDescriptor vertexDescriptor];
            for (const VernonRuntimeProviderVertexBinding &binding : stage.vertexBindings) {
                if (binding.binding >= key.vertexStrides.size())
                    return failure<const StageObjects::Variant *>("a vertex binding has no stride");
                MTLVertexBufferLayoutDescriptor *layout = vertex.layouts[kFirstVertexBuffer + binding.binding];
                const uint32_t stride = key.vertexStrides[binding.binding];
                if (stride == 0) {
                    // Every vertex and instance reads the binding's first element, which the stride spans.
                    uint32_t extent = 4;
                    for (const VernonRuntimeProviderVertexAttribute &attribute : stage.vertexAttributes)
                        if (attribute.binding == binding.binding)
                            extent =
                                (std::max)(extent, attribute.relative_offset +
                                                       vertexFormatBytes(attribute.dtype, attribute.component_count));
                    layout.stride = align(extent, 4);
                    layout.stepFunction = MTLVertexStepFunctionConstant;
                    layout.stepRate = 0;
                } else {
                    layout.stride = stride;
                    layout.stepFunction =
                        binding.divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
                    layout.stepRate = binding.divisor ? binding.divisor : 1;
                }
            }
            for (const VernonRuntimeProviderVertexAttribute &attribute : stage.vertexAttributes) {
                MTLVertexAttributeDescriptor *target = vertex.attributes[attribute.location];
                target.format = vertexFormat(attribute.dtype, attribute.component_count);
                target.offset = attribute.relative_offset;
                target.bufferIndex = kFirstVertexBuffer + attribute.binding;
            }
            native.vertexDescriptor = vertex;
        }
        for (size_t index = 0; index < key.colorFormats.size(); ++index) {
            const VernonColorBlendState &source = key.colorBlends[index];
            MTLRenderPipelineColorAttachmentDescriptor *target = native.colorAttachments[index];
            target.pixelFormat = static_cast<MTLPixelFormat>(key.colorFormats[index]);
            target.blendingEnabled = source.blend_enabled ? YES : NO;
            target.sourceRGBBlendFactor = blendFactor(source.source_color_factor);
            target.destinationRGBBlendFactor = blendFactor(source.destination_color_factor);
            target.rgbBlendOperation = blendOperation(source.color_operation);
            target.sourceAlphaBlendFactor = blendFactor(source.source_alpha_factor);
            target.destinationAlphaBlendFactor = blendFactor(source.destination_alpha_factor);
            target.alphaBlendOperation = blendOperation(source.alpha_operation);
            target.writeMask = writeMask(source.write_mask);
        }
        native.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(key.depthStencilFormat);
        if (hasStencil)
            native.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        native.inputPrimitiveTopology = key.topology == 1   ? MTLPrimitiveTopologyClassLine
                                        : key.topology == 2 ? MTLPrimitiveTopologyClassPoint
                                                            : MTLPrimitiveTopologyClassTriangle;
        native.rasterSampleCount = 1;
        NSError *error = nil;
        made.pipeline = [state_.native->device newRenderPipelineStateWithDescriptor:native error:&error];
        if (!made.pipeline)
            return failure<const StageObjects::Variant *>("cannot create the graphics pipeline: " +
                                                          metal::describe(error));
        if (hasDepth) {
            // As on Vulkan, a draw without the depth test writes no depth.
            MTLDepthStencilDescriptor *depth = [[MTLDepthStencilDescriptor alloc] init];
            depth.depthCompareFunction =
                depthState.depth_test ? compareFunction(depthState.depth_compare) : MTLCompareFunctionAlways;
            depth.depthWriteEnabled = depthState.depth_test && depthState.depth_write ? YES : NO;
            if (depthState.stencil_test) {
                const auto face = [&](const VernonStencilFaceState &source) {
                    MTLStencilDescriptor *stencil = [[MTLStencilDescriptor alloc] init];
                    stencil.stencilCompareFunction = compareFunction(source.compare);
                    stencil.stencilFailureOperation = stencilOperation(source.stencil_fail);
                    stencil.depthFailureOperation = stencilOperation(source.depth_fail);
                    stencil.depthStencilPassOperation = stencilOperation(source.pass);
                    stencil.readMask = depthState.stencil_read_mask;
                    stencil.writeMask = depthState.stencil_write_mask;
                    return stencil;
                };
                depth.frontFaceStencil = face(depthState.front);
                depth.backFaceStencil = face(depthState.back);
            }
            made.depthStencil = [state_.native->device newDepthStencilStateWithDescriptor:depth];
            if (!made.depthStencil)
                return failure<const StageObjects::Variant *>("cannot create the depth stencil state");
        }
    }
    made.cull = static_cast<MTLCullMode>(rasterization.cull_mode);
    made.winding = rasterization.front_face == 0 ? MTLWindingCounterClockwise : MTLWindingClockwise;
    made.clip = rasterization.depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
    made.depthBias = rasterization.depth_bias_enabled ? rasterization.depth_bias_constant : 0.0f;
    made.slopeBias = rasterization.depth_bias_enabled ? rasterization.depth_bias_slope : 0.0f;
    stage.variants.push_back(std::move(made));
    return Found{ok(&stage.variants.back())};
}

Builder::Builder(MetalDevice &device, Recording &recording, const PlannedPasses &planned,
                 const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches)
    : device_(device), state_(device.state()), recording_(recording), planned_(planned), launches_(launches) {}

void Builder::debug(uint32_t kind, const std::string &label) {
    if (!state_.labels)
        return;
    Command command;
    command.op = kind == VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP  ? Op::DebugPush
                 : kind == VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP ? Op::DebugPop
                                                                   : Op::DebugMarker;
    command.a = static_cast<uint32_t>(recording_.labels.size());
    recording_.labels.push_back(label);
    add(command);
}

void Builder::merge(std::vector<Use> &into, Uses &indices, const std::vector<Use> &uses) {
    for (const Use &use : uses) {
        const auto [found, inserted] = indices.emplace((__bridge void *)use.resource, into.size());
        if (inserted) {
            into.push_back(use);
            continue;
        }
        into[found->second].usage |= use.usage;
        into[found->second].stages |= use.stages;
    }
}

Made<StageObjects *> Builder::stage(VernonStageExecutable &stage, bool graphics) {
    if (const auto found = stages_.find(&stage); found != stages_.end())
        return Made<StageObjects *>{ok(found->second)};
    auto objects = device_.stageObjects(stage, graphics);
    if (objects.isErr())
        return failure<StageObjects *>(std::move(objects).error());
    StageObjects *pointer = objects.value().get();
    recording_.stages.push_back(std::move(objects).value());
    stages_.emplace(&stage, pointer);
    return Made<StageObjects *>{ok(pointer)};
}

// Makes the argument buffers and push data of `set` from the values `fill` gives. Its inline bytes are sites the call
// fills again from `source`, unless the instance is `persistent`: its owners' values, and so its bytes, stay as they
// are across calls.
Made<uint32_t> Builder::instance(const StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                                 const Site &source) {
    const SetPlan &plan = stage.sets[set];
    const auto at = static_cast<uint32_t>(recording_.sets.size());
    SetInstance made;
    made.stage = &stage;
    made.set = set;
    if (plan.pushBytes) {
        made.pushOffset = static_cast<uint32_t>(recording_.push.size());
        recording_.push.resize(recording_.push.size() + plan.pushBytes);
    }
    std::string error;
    std::vector<uint8_t> storage;
    for (const uint32_t slotIndex : plan.slots) {
        const SlotPlan &slot = stage.slots[slotIndex];
        if (!slot.read || (slot.kind != SlotKind::Push && !plan.encoders[slot.stage]))
            continue;
        VernonRuntimeProviderBindingValue value{};
        if (!fill(slotIndex, value, storage, error))
            return failure<uint32_t>("cannot fill a binding: " + error);
        if (value.kind != slot.providerKind)
            return failure<uint32_t>("a binding value has the wrong kind");
        Site site = source;
        site.slot = slotIndex;
        site.size = slot.size;
        std::vector<Member> &members = made.members[slot.stage];
        const MTLRenderStages stages = stage.graphics ? renderStages(slot.stage) : MTLRenderStages{};
        switch (slot.kind) {
        case SlotKind::Push: {
            if (!value.payload.inline_value.data || value.payload.inline_value.size != slot.size)
                return failure<uint32_t>("an inline binding value has the wrong size");
            site.offset = made.pushOffset + slot.dataOffset;
            site.push = true;
            std::memcpy(recording_.push.data() + site.offset, value.payload.inline_value.data, slot.size);
            if (!persistent)
                recording_.sites.push_back(site);
            break;
        }
        case SlotKind::Inline: {
            if (!value.payload.inline_value.data || value.payload.inline_value.size != slot.size)
                return failure<uint32_t>("an inline binding value has the wrong size");
            site.offset = arena(align(slot.size, 4), slot.constant ? kConstantAlignment : kStorageAlignment);
            const auto *bytes = static_cast<const uint8_t *>(value.payload.inline_value.data);
            if (persistent)
                recording_.fixed.emplace_back(site.offset, std::vector<uint8_t>(bytes, bytes + slot.size));
            else
                recording_.sites.push_back(site);
            Member member;
            member.index = slot.member;
            member.kind = SlotKind::Inline;
            member.offset = site.offset;
            members.push_back(member);
            break;
        }
        case SlotKind::Buffer: {
            const VernonRuntimeProviderResourceReference &resource = value.payload.buffer.resource;
            if (resource.size == 0)
                return failure<uint32_t>("a buffer binding is empty");
            if (resource.offset % kStorageAlignment != 0)
                return failure<uint32_t>("Metal binds storage buffers at offsets that are multiples of 16 bytes");
            auto buffer = nativeOf<metal::Buffer>(resource, Kind::Buffer);
            if (buffer.isErr())
                return failure<uint32_t>(std::move(buffer).error());
            Member member;
            member.index = slot.member;
            member.kind = SlotKind::Buffer;
            member.buffer = buffer.value()->buffer;
            member.offset = resource.offset;
            members.push_back(member);
            made.uses.push_back({buffer.value()->buffer,
                                 slot.written ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead,
                                 stages});
            break;
        }
        case SlotKind::Image: {
            if (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE)
                return failure<uint32_t>("an image binding has no image");
            auto view = nativeOf<metal::View>(value.payload.image.view, Kind::View);
            if (view.isErr())
                return failure<uint32_t>(std::move(view).error());
            const bool storageImage = slot.providerKind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
            if (!(view.value()->usage & (storageImage ? VERNON_IMAGE_STORAGE : VERNON_IMAGE_SAMPLED)))
                return failure<uint32_t>(storageImage ? "a storage image's view has no storage usage"
                                                      : "a sampled image's view has no sampled usage");
            if (storageImage && view.value()->format != slot.storageFormat)
                return failure<uint32_t>("a storage image's format is not the one its kernel declares");
            Member member;
            member.index = slot.member;
            member.kind = SlotKind::Image;
            member.texture = view.value()->texture;
            members.push_back(member);
            made.uses.push_back({view.value()->texture,
                                 storageImage ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead,
                                 stages});
            break;
        }
        case SlotKind::Sampler: {
            Member member;
            member.index = slot.member;
            member.kind = SlotKind::Sampler;
            if (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) {
                member.sampler = state_.native->defaultSampler;
            } else {
                auto sampler = nativeOf<metal::Sampler>(value.payload.sampler.resource, Kind::Sampler);
                if (sampler.isErr())
                    return failure<uint32_t>(std::move(sampler).error());
                member.sampler = sampler.value()->sampler;
            }
            members.push_back(member);
            break;
        }
        }
    }
    for (uint32_t role = 0; role < 3; ++role)
        if (id<MTLArgumentEncoder> encoder = plan.encoders[role])
            made.arguments[role] = arena(encoder.encodedLength, (std::max<uint64_t>)(encoder.alignment, 16));
    recording_.sets.push_back(std::move(made));
    return Made<uint32_t>{ok(at)};
}

Made<void> Builder::zero(const VernonRuntimeProviderResourceReference &reference, uint64_t offset, uint64_t size) {
    auto buffer = nativeOf<metal::Buffer>(reference, Kind::Buffer);
    if (buffer.isErr())
        return Made<void>{err(std::move(buffer).error())};
    if (!copyable(offset, size))
        return failure<void>("Metal on macOS zeroes buffer ranges in multiples of 4 bytes");
    Command fill{Op::Fill};
    fill.destination = buffer.value()->buffer;
    fill.destinationOffset = offset;
    fill.size = size;
    add(fill);
    return Made<void>{ok()};
}

Made<void> Builder::build(const std::vector<const InternalBuffers::Internal *> &internals) {
    for (const InternalBuffers::Internal *internal : internals)
        if (auto zeroed = zero(internal->reference, 0, internal->bytes); zeroed.isErr())
            return failure<void>("an internal buffer: " + zeroed.error());
    for (const VernonRuntimeProviderResourceReference &range : planned_.zeroed)
        if (range.size)
            if (auto zeroed = zero(range, range.offset, range.size); zeroed.isErr())
                return failure<void>("a zeroed argument: " + zeroed.error());
    const std::vector<PlannedPass *> &passes = planned_.passes;
    for (size_t at = 0; at < passes.size();) {
        const PlannedPass &pass = *passes[at];
        if (pass.copy) {
            if (auto built = buildCopy(*pass.copy); built.isErr())
                return built;
            ++at;
            continue;
        }
        if (pass.compute) {
            debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, pass.command->label);
            if (auto built = buildCompute(at); built.isErr())
                return built;
            debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
            ++at;
            continue;
        }
        size_t end = at + 1;
        while (end < passes.size() && !passes[end]->compute && !passes[end]->copy && passes[end]->scope == pass.scope)
            ++end;
        if (auto built = buildScope(at, end); built.isErr())
            return built;
        at = end;
    }
    return Made<void>{ok()};
}

Made<void> Builder::buildCompute(size_t at) {
    const PlannedPass &pass = *planned_.passes[at];
    if (!launches_[at])
        return Made<void>{ok()};
    const runtime::PlannedComputeLaunch &launch = *launches_[at];
    VernonStageExecutable &executable = *pass.executable->source().resolved->stage;
    auto objects = stage(executable, false);
    if (objects.isErr())
        return Made<void>{err(std::move(objects).error())};
    const StageObjects &stageObjects = *objects.value();
    const auto &state = ::runtimeBackendState<runtime::MetalPipelineState>(executable);
    const PreparedComputeBindingPlan &plan = state.computePlan;
    const std::string name = "pass '" + pass.executable->source().node->name + "': ";
    const FillValue fill = [&](size_t slot, VernonRuntimeProviderBindingValue &value, std::vector<uint8_t> &,
                               std::string &error) {
        return runtime::fillPreparedComputeBindingValue(plan, executable.bindingProjection, launch, slot, value, error);
    };
    Site source;
    source.pass = static_cast<uint32_t>(at);
    Command dispatch{Op::Dispatch};
    dispatch.a = static_cast<uint32_t>(at);
    dispatch.b = static_cast<uint32_t>(recording_.uses.size());
    dispatch.d = static_cast<uint32_t>(recording_.dispatchSets.size());
    dispatch.stage = &stageObjects;
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        if (!stageObjects.sets[set].encoders[runtime::kMetalCompute])
            continue;
        auto made = instance(stageObjects, set, false, fill, source);
        if (made.isErr())
            return Made<void>{err(name + made.error())};
        recording_.dispatchSets.push_back(made.value());
        const std::vector<Use> &uses = recording_.sets[made.value()].uses;
        recording_.uses.insert(recording_.uses.end(), uses.begin(), uses.end());
    }
    dispatch.c = static_cast<uint32_t>(recording_.uses.size()) - dispatch.b;
    dispatch.e = static_cast<uint32_t>(recording_.dispatchSets.size()) - dispatch.d;
    add(dispatch);
    return Made<void>{ok()};
}

Made<void> Builder::buildCopy(const PlannedCopy &copy) {
    if (copy.regions.empty())
        return Made<void>{ok()};
    auto source = nativeOf<metal::Buffer>(copy.source, Kind::Buffer);
    if (source.isErr())
        return failure<void>("a leaf copy's source: " + source.error());
    auto destination = nativeOf<metal::Buffer>(copy.destination, Kind::Buffer);
    if (destination.isErr())
        return failure<void>("a leaf copy's destination: " + destination.error());
    if (source.value()->buffer == destination.value()->buffer)
        return failure<void>("Metal devices copy between distinct buffers only");
    for (const CopyRegion &region : copy.regions)
        if (!copyable(region.source, region.size) || !copyable(region.destination, region.size))
            return failure<void>("Metal on macOS copies buffer ranges in multiples of 4 bytes");
    Command regions{Op::CopyRegions};
    regions.source = source.value()->buffer;
    regions.destination = destination.value()->buffer;
    regions.a = static_cast<uint32_t>(recording_.copies.size());
    regions.b = static_cast<uint32_t>(copy.regions.size());
    recording_.copies.insert(recording_.copies.end(), copy.regions.begin(), copy.regions.end());
    add(regions);
    return Made<void>{ok()};
}

Made<Builder::DrawPlan> Builder::planDraw(const PlannedPass &pass, const PlannedDraw &draw) {
    using Planned = Made<DrawPlan>;
    DrawPlan result;
    result.draw = &draw;
    result.sets.fill(kNone);
    auto &state = ::runtimeBackendState<runtime::MetalPipelineState>(*draw.stage);
    if (state.vertex.source.empty())
        return Planned{ok(result)};
    auto objects = stage(*draw.stage, true);
    if (objects.isErr())
        return Planned{err(std::move(objects).error())};
    StageObjects &stageObjects = *objects.value();
    result.stage = &stageObjects;
    const PreparedGraphicsBindingPlan &plan = state.graphicsPlan;
    const runtime::GraphicsSlotInputs inputs{draw.argumentPointers.data(), draw.draw.resolution};
    std::string error;
    std::vector<uint32_t> strides(plan.vertexInputs.size());
    result.vertexFirst = static_cast<uint32_t>(recording_.vertexBuffers.size());
    result.vertexCount = static_cast<uint32_t>(plan.vertexInputs.size());
    for (size_t binding = 0; binding < plan.vertexInputs.size(); ++binding) {
        VernonRuntimeProviderResourceReference buffer{};
        if (!runtime::preparedGraphicsVertexBuffer(plan, inputs, binding, buffer, strides[binding], error))
            return failure<DrawPlan>(error);
        auto native = nativeOf<metal::Buffer>(buffer, Kind::Buffer);
        if (native.isErr())
            return failure<DrawPlan>("a vertex buffer: " + native.error());
        recording_.vertexBuffers.push_back({native.value()->buffer, buffer.offset});
    }
    std::vector<uint32_t> formats;
    for (VernonTextureFormat format : pass.targets.attachmentFormats) {
        const MTLPixelFormat native = metal::pixelFormat(format);
        if (native == MTLPixelFormatInvalid)
            return failure<DrawPlan>("Metal color attachment format is unsupported");
        formats.push_back(static_cast<uint32_t>(native));
    }
    uint32_t depthFormat = 0;
    if (pass.targets.depthAttachment) {
        depthFormat = static_cast<uint32_t>(metal::pixelFormat(pass.targets.depthFormat));
        if (!depthFormat)
            return failure<DrawPlan>("Metal depth attachment format is unsupported");
    }
    runtime::GraphicsVariantKey key;
    if (!runtime::prepareGraphicsVariantKey(*draw.state, pass.targets, std::move(formats), depthFormat, strides, key,
                                            error))
        return failure<DrawPlan>(error);
    auto variant = device_.variant(stageObjects, key);
    if (variant.isErr())
        return Planned{err(std::move(variant).error())};
    result.variant = variant.value();
    const FillValue fill = [&](size_t slot, VernonRuntimeProviderBindingValue &value, std::vector<uint8_t> &storage,
                               std::string &message) {
        return runtime::fillPreparedGraphicsBindingValue(plan, inputs, slot, value, storage, message);
    };
    Site source;
    source.draw = &draw;
    source.graphics = &plan;
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        const SetPlan &setPlan = stageObjects.sets[set];
        if (!setPlan.arguments() && !setPlan.pushBytes)
            continue;
        // A group is found by the owners of what decides it, and by the draw when it also depends on the command. A
        // group whose owners share one persistent store keeps its bytes across calls.
        std::vector<const void *> found{&stageObjects, reinterpret_cast<const void *>(uintptr_t{set})};
        runtime::GroupStore *owner =
            setPlan.parameters.empty() ? nullptr : draw.owners[setPlan.parameters.front()].groups;
        for (uint32_t parameter : setPlan.parameters) {
            found.push_back(draw.owners[parameter].identity);
            if (draw.owners[parameter].groups != owner)
                owner = nullptr;
        }
        found.push_back(setPlan.command || setPlan.parameters.empty() ? &draw : nullptr);
        const bool persistent = owner && !setPlan.command;
        if (const auto existing = instances_.find(found); existing != instances_.end()) {
            result.sets[set] = existing->second;
            continue;
        }
        auto created = instance(stageObjects, set, persistent, fill, source);
        if (created.isErr())
            return Planned{err(std::move(created).error())};
        result.sets[set] = created.value();
        instances_.emplace(std::move(found), created.value());
    }
    if (!runtime::prepareGraphicsDrawInvocation(draw.draw, draw.state->topology, result.invocation, error))
        return failure<DrawPlan>(error);
    if (result.invocation.indexCount || draw.streamed) {
        if (!result.invocation.indexCount)
            return failure<DrawPlan>("a draw stream needs an index binding");
        auto index = nativeOf<metal::Buffer>(result.invocation.indexBuffer, Kind::Buffer);
        if (index.isErr())
            return failure<DrawPlan>("an index buffer: " + index.error());
        if (result.invocation.indexBuffer.offset % 4 != 0)
            return failure<DrawPlan>("an index buffer's offset is not a multiple of 4 bytes");
        result.index = index.value()->buffer;
        result.indexOffset = result.invocation.indexBuffer.offset;
    }
    const runtime::GraphicsDrawStream &stream = draw.stream;
    if (draw.streamed && stream.records.resource.value && stream.maxRecords) {
        const bool counted = stream.counts.resource.value != 0;
        if (uint64_t{stream.firstRecord} + stream.maxRecords > stream.records.size / kRecordBytes ||
            (counted && stream.countIndex >= stream.counts.size / sizeof(uint32_t)))
            return failure<DrawPlan>("a device draw stream exceeds its buffers");
        auto records = nativeOf<metal::Buffer>(stream.records, Kind::Buffer);
        if (records.isErr())
            return failure<DrawPlan>("a draw stream's records: " + records.error());
        if (stream.records.offset % 4 != 0)
            return failure<DrawPlan>("a draw stream's records are not at a multiple of 4 bytes");
        result.records = records.value()->buffer;
        result.recordsOffset = stream.records.offset + uint64_t{stream.firstRecord} * kRecordBytes;
        if (counted) {
            auto counts = nativeOf<metal::Buffer>(stream.counts, Kind::Buffer);
            if (counts.isErr())
                return failure<DrawPlan>("a draw stream's counts: " + counts.error());
            result.counts = counts.value()->buffer;
            result.countsOffset = stream.counts.offset + uint64_t{stream.countIndex} * sizeof(uint32_t);
            result.arguments = recording_.argumentBytes;
            recording_.argumentBytes += uint64_t{stream.maxRecords} * kRecordBytes;
        }
    }
    return Planned{ok(result)};
}

Made<void> Builder::emitDraw(const DrawPlan &plan) {
    if (!plan.stage)
        return Made<void>{ok()};
    const StageObjects &stage = *plan.stage;
    const PlannedDraw &draw = *plan.draw;
    if (variant_ != plan.variant) {
        Command bind{Op::BindGraphics};
        bind.variant = plan.variant;
        add(bind);
        variant_ = plan.variant;
    }
    bool push = pushStage_ != &stage;
    for (uint32_t set = 0; set < stage.sets.size(); ++set) {
        const uint32_t instance = plan.sets[set];
        if (instance == kNone)
            continue;
        const SetPlan &setPlan = stage.sets[set];
        if (setPlan.pushBytes && pushed_[set] != instance)
            push = true;
        const SetInstance &made = recording_.sets[instance];
        const bool vertex = made.arguments[runtime::kMetalVertex] != kNoOffset && sets_[0][set] != instance;
        const bool fragment = made.arguments[runtime::kMetalFragment] != kNoOffset && sets_[1][set] != instance;
        if (!vertex && !fragment)
            continue;
        add({Op::Sets, instance, vertex ? 1u : 0u, fragment ? 1u : 0u});
        sets_[0][set] = instance;
        sets_[1][set] = instance;
    }
    if (push && (stage.pushSizes[0] || stage.pushSizes[1])) {
        PushBlock block;
        block.stage = &stage;
        block.first = static_cast<uint32_t>(recording_.pushSets.size());
        for (uint32_t set = 0; set < stage.sets.size(); ++set)
            if (stage.sets[set].pushBytes && plan.sets[set] != kNone) {
                recording_.pushSets.emplace_back(set, plan.sets[set]);
                pushed_[set] = plan.sets[set];
            }
        block.count = static_cast<uint32_t>(recording_.pushSets.size()) - block.first;
        for (uint32_t index = 0; index < 2; ++index)
            if (stage.pushSizes[index])
                block.offsets[index] = arena(align(stage.pushSizes[index], 16), kConstantAlignment);
        add({Op::Push, static_cast<uint32_t>(recording_.pushBlocks.size())});
        recording_.pushBlocks.push_back(block);
    }
    pushStage_ = &stage;
    if (plan.vertexCount)
        add({Op::VertexBuffers, plan.vertexFirst, plan.vertexCount});
    const runtime::DrawInvocation &invocation = plan.invocation;
    const std::array<uint32_t, 4> viewport{invocation.viewport[0], invocation.viewport[1], invocation.viewport[2],
                                           invocation.viewport[3]};
    if (viewport_ != viewport) {
        add({Op::Viewport, viewport[0], viewport[1], viewport[2], viewport[3]});
        viewport_ = viewport;
    }
    // Metal rejects a scissor beyond the attachments.
    const uint32_t x = (std::min)(invocation.scissor[0], width_);
    const uint32_t y = (std::min)(invocation.scissor[1], height_);
    const std::array<uint32_t, 4> scissor{x, y, (std::min)(invocation.scissor[2], width_ - x),
                                          (std::min)(invocation.scissor[3], height_ - y)};
    if (scissor_ != scissor) {
        add({Op::Scissor, scissor[0], scissor[1], scissor[2], scissor[3]});
        scissor_ = scissor;
    }
    if (stencil_ != invocation.stencilReference) {
        add({Op::Stencil, invocation.stencilReference});
        stencil_ = invocation.stencilReference;
    }
    const MTLPrimitiveType primitive = primitiveType(invocation.topology);
    if (!draw.streamed) {
        Command drawn{plan.index ? Op::DrawIndexed : Op::Draw};
        drawn.primitive = primitive;
        drawn.a = plan.index ? invocation.indexCount : invocation.vertexCount;
        drawn.b = invocation.instanceCount;
        drawn.c = invocation.firstIndex;
        drawn.d = static_cast<uint32_t>(invocation.baseVertex);
        drawn.e = invocation.firstInstance;
        drawn.index = plan.index;
        drawn.indexOffset = plan.indexOffset;
        add(drawn);
        return Made<void>{ok()};
    }
    const runtime::GraphicsDrawStream &stream = draw.stream;
    if (stream.maxRecords == 0)
        return Made<void>{ok()};
    if (stream.records.resource.value) {
        Command indirect{Op::DrawIndirect};
        indirect.primitive = primitive;
        indirect.a = stream.maxRecords;
        indirect.index = plan.index;
        indirect.indexOffset = plan.indexOffset;
        // A counted stream draws from its expansion, an uncounted one from its records.
        indirect.source = plan.counts ? nil : plan.records;
        indirect.sourceOffset = plan.counts ? plan.arguments : plan.recordsOffset;
        add(indirect);
        return Made<void>{ok()};
    }
    if (!stream.hostRecords || !stream.hostCounts)
        return failure<void>("a host draw stream has no records");
    HostStream host;
    host.draw = &draw;
    host.indexCapacity = invocation.indexBuffer.size / sizeof(uint32_t);
    Command drawn{Op::DrawHost};
    drawn.primitive = primitive;
    drawn.a = static_cast<uint32_t>(recording_.streams.size());
    drawn.index = plan.index;
    drawn.indexOffset = plan.indexOffset;
    recording_.streams.push_back(host);
    add(drawn);
    return Made<void>{ok()};
}

// The pass of a render scope. Metal clears whole attachments only.
Made<Rendering> Builder::rendering(const runtime::GraphicsRenderScope &scope) {
    auto checked = checkRenderScope(scope, nativeIdentity(&state_, kNativeViewReference));
    if (checked.isErr())
        return failure<Rendering>("the render scope is invalid: " + checked.error());
    // An image the recording writes before reading it needs no defined contents when a call begins.
    for (const uint64_t image : checked.value().readImages)
        if (!written_.count(image))
            recording_.reads.push_back(image);
    for (const uint64_t image : checked.value().attachmentImages)
        if (written_.insert(image).second)
            recording_.writes.push_back(image);
    const bool whole =
        !scope.area[2] || !scope.area[3] ||
        (!scope.area[0] && !scope.area[1] && scope.area[2] >= scope.width && scope.area[3] >= scope.height);
    Rendering result;
    result.pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (size_t index = 0; index < scope.colors.size(); ++index) {
        const VernonRuntimeProviderColorAttachment &color = scope.colors[index];
        auto native = nativeOf<metal::View>(color.view, Kind::View);
        if (native.isErr())
            return failure<Rendering>("a color attachment: " + native.error());
        const metal::View &view = *native.value();
        if (color.location >= kMaxColors)
            return failure<Rendering>("a color attachment's location is out of range");
        if (!(view.usage & VERNON_IMAGE_COLOR_ATTACHMENT))
            return failure<Rendering>("a color attachment's view has no color attachment usage");
        const bool clear = color.load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        if (clear && !whole)
            return failure<Rendering>("Metal clears whole color attachments only");
        const VernonTextureFormat format = index < scope.colorFormats.size() ? scope.colorFormats[index] : view.format;
        MTLRenderPassColorAttachmentDescriptor *target = result.pass.colorAttachments[color.location];
        target.texture = view.texture;
        target.loadAction = loadAction(color.load_operation);
        target.storeAction = storeAction(color.store_operation);
        const VernonClearColor &value = color.clear_color;
        const formats::Format *entry = formats::find(format);
        if (entry && entry->componentClass == formats::ComponentClass::Uint)
            target.clearColor = MTLClearColorMake(value.u32[0], value.u32[1], value.u32[2], value.u32[3]);
        else if (entry && entry->componentClass == formats::ComponentClass::Sint)
            target.clearColor = MTLClearColorMake(value.i32[0], value.i32[1], value.i32[2], value.i32[3]);
        else
            target.clearColor = MTLClearColorMake(value.f32[0], value.f32[1], value.f32[2], value.f32[3]);
    }
    if (scope.depth) {
        const VernonRuntimeProviderDepthStencilAttachment &depth = *scope.depth;
        auto native = nativeOf<metal::View>(depth.view, Kind::View);
        if (native.isErr())
            return failure<Rendering>("the depth attachment: " + native.error());
        const metal::View &view = *native.value();
        if (!(view.usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT))
            return failure<Rendering>("the depth attachment's view has no depth stencil attachment usage");
        const bool stencil = checked.value().stencil;
        if (!whole && (depth.depth_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR ||
                       (stencil && depth.stencil_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR)))
            return failure<Rendering>("Metal clears whole depth attachments only");
        MTLRenderPassDepthAttachmentDescriptor *target = result.pass.depthAttachment;
        target.texture = view.texture;
        target.loadAction = loadAction(depth.depth_load_operation);
        target.storeAction = storeAction(depth.depth_store_operation);
        target.clearDepth = depth.clear_depth;
        // A pipeline of a depth-stencil format renders with its stencil, which a scope that leaves it alone keeps.
        if (formats::hasStencil(view.format)) {
            MTLRenderPassStencilAttachmentDescriptor *planes = result.pass.stencilAttachment;
            planes.texture = view.texture;
            planes.loadAction = stencil ? loadAction(depth.stencil_load_operation) : MTLLoadActionLoad;
            planes.storeAction = stencil ? storeAction(depth.stencil_store_operation) : MTLStoreActionStore;
            planes.clearStencil = depth.clear_stencil;
        }
    }
    result.pass.renderTargetWidth = scope.width;
    result.pass.renderTargetHeight = scope.height;
    return Made<Rendering>{ok(result)};
}

// Plans every draw of passes [begin, end), so that the device draw streams expand before the scope begins.
Made<void> Builder::buildScope(size_t begin, size_t end) {
    const std::vector<PlannedPass *> &passes = planned_.passes;
    const runtime::GraphicsRenderScope &scope = planned_.scopes[passes[begin]->scope];
    std::vector<DrawPlan> draws;
    for (size_t at = begin; at < end; ++at)
        for (const PlannedDraw &draw : passes[at]->draws) {
            auto planned = planDraw(*passes[at], draw);
            if (planned.isErr())
                return failure<void>("pass '" + passes[at]->executable->source().node->name + "': " + planned.error());
            draws.push_back(planned.value());
        }
    for (const DrawPlan &plan : draws) {
        if (!plan.counts)
            continue;
        Command expand{Op::ExpandStream};
        expand.a = plan.draw->stream.maxRecords;
        expand.source = plan.records;
        expand.sourceOffset = plan.recordsOffset;
        expand.destination = plan.counts;
        expand.destinationOffset = plan.countsOffset;
        expand.size = plan.arguments;
        add(expand);
    }
    auto opened = rendering(scope);
    if (opened.isErr())
        return failure<void>("pass '" + passes[begin]->executable->source().node->name + "': " + opened.error());
    Rendering made = opened.value();
    Uses indices;
    made.firstUse = static_cast<uint32_t>(recording_.uses.size());
    std::vector<Use> uses;
    for (const DrawPlan &plan : draws)
        for (const uint32_t instance : plan.sets)
            if (instance != kNone)
                merge(uses, indices, recording_.sets[instance].uses);
    recording_.uses.insert(recording_.uses.end(), uses.begin(), uses.end());
    made.uses = static_cast<uint32_t>(uses.size());
    add({Op::BeginScope, static_cast<uint32_t>(recording_.renderings.size())});
    recording_.renderings.push_back(made);
    width_ = scope.width;
    height_ = scope.height;
    // A render encoder begins without any state.
    variant_ = nullptr;
    pushStage_ = nullptr;
    pushed_.fill(kNone);
    for (auto &bound : sets_)
        bound.fill(kNone);
    viewport_.reset();
    scissor_.reset();
    stencil_.reset();
    std::string label;
    for (size_t at = begin; at < end; ++at)
        label += (at == begin ? "" : "+") + passes[at]->command->label;
    debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, label);
    const bool memberGroups = end - begin > 1;
    size_t next = 0;
    for (size_t at = begin; at < end; ++at) {
        const PlannedPass &pass = *passes[at];
        if (memberGroups)
            debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, pass.command->label);
        for (const PlannedDraw &draw : pass.draws) {
            for (const PlannedDebug &command : draw.debug)
                debug(command.kind, command.label);
            if (auto emitted = emitDraw(draws[next++]); emitted.isErr())
                return failure<void>("pass '" + pass.executable->source().node->name + "': " + emitted.error());
        }
        for (const PlannedDebug &command : pass.trailingDebug)
            debug(command.kind, command.label);
        if (memberGroups)
            debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
    }
    debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
    add({Op::EndScope});
    return Made<void>{ok()};
}

Made<std::unique_ptr<Recording>>
MetalDevice::build(const MetalCommandBuffer &commands, const PlannedPasses &planned,
                   const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches) {
    auto recording = std::make_unique<Recording>();
    recording->pool = pool_;
    auto used = internals_.used(commands.commands());
    if (used.isErr())
        return failure<std::unique_ptr<Recording>>(std::move(used).error());
    Builder builder(*this, *recording, planned, launches);
    if (auto built = builder.build(used.value()); built.isErr())
        return failure<std::unique_ptr<Recording>>(std::move(built).error());
    return Made<std::unique_ptr<Recording>>{ok(std::move(recording))};
}

// Plans a compute pass's launch, whose resources key the recording that dispatches it; the call encodes its group
// counts, so only whether it dispatches at all does.
Made<void> MetalDevice::planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
                                   std::vector<uint64_t> &key) {
    const bool empty = !pass.grid.x || !pass.grid.y || !pass.grid.z;
    key.insert(key.end(), {1, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pass.executable)), empty ? 0u : 1u});
    if (empty)
        return Made<void>{ok()};
    const std::string name = "pass '" + pass.executable->source().node->name + "': ";
    VernonStageExecutable &stage = *pass.executable->source().resolved->stage;
    const uint32_t grid[3]{pass.grid.x, pass.grid.y, pass.grid.z};
    const uint32_t workgroup[3]{stage.workgroupSize.x, stage.workgroupSize.y, stage.workgroupSize.z};
    std::string error;
    if (!runtime::validateDispatchContract(stage.dispatchContract, grid, workgroup, error))
        return failure<void>(name + error);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = pass.arguments.data();
    invocation.argument_count = pass.arguments.size();
    invocation.compute_grid = pass.grid;
    launch.emplace();
    if (!runtime::planComputeInvocation(stage.bindingProjection, invocation, *launch, error))
        return failure<void>(name + error);
    if (!stage.backendState)
        return failure<void>(name + "the Stage has no Metal pipeline");
    const auto &state = ::runtimeBackendState<runtime::MetalPipelineState>(stage);
    const PreparedComputeBindingPlan &plan = state.computePlan;
    for (size_t slot = 0; slot < plan.size(); ++slot) {
        VernonRuntimeProviderBindingValue value{};
        if (!runtime::fillPreparedComputeBindingValue(plan, stage.bindingProjection, *launch, slot, value, error))
            return failure<void>(name + error);
        key.push_back(value.kind);
        if (packedBytes(plan.layouts[slot])) {
            key.push_back(value.payload.inline_value.size);
            continue;
        }
        const VernonRuntimeProviderResourceReference &reference =
            value.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? value.payload.buffer.resource
            : value.kind == VERNON_RUNTIME_PROVIDER_SAMPLER      ? value.payload.sampler.resource
                                                                 : value.payload.image.view;
        key.push_back(value.flags);
        addReference(key, reference);
    }
    return Made<void>{ok()};
}

// Fills a new or pooled slot for `recording`, keeping what of a pooled slot is large enough. Capacities are rounded
// up so that the slots of a pass rebuilt every call fit the recordings after it.
Made<void> MetalDevice::createSlot(Recording &recording, Slot &slot) {
    id<MTLDevice> device = state_.native->device;
    if (!slot.arena || slot.arenaCapacity < recording.arenaBytes) {
        const uint64_t capacity = roundUp(recording.arenaBytes, 4096);
        slot.arena = [device newBufferWithLength:capacity
                                         options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        if (!slot.arena)
            return failure<void>("cannot create a recording's arena");
        slot.mapped = static_cast<uint8_t *>(slot.arena.contents);
        slot.arenaCapacity = capacity;
        slot.written = false;
    }
    if (recording.argumentBytes && slot.argumentCapacity < recording.argumentBytes) {
        const uint64_t capacity = roundUp(recording.argumentBytes, 4096);
        slot.arguments = [device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
        if (!slot.arguments)
            return failure<void>("cannot create a recording's draw arguments");
        slot.argumentCapacity = capacity;
    }
    return Made<void>{ok()};
}

// A slot whose last submission completed; a new one while the recording has fewer than kSlots, else the oldest once
// it completes.
Made<Slot *> MetalDevice::acquire(Recording &recording) {
    for (const std::unique_ptr<Slot> &slot : recording.slots) {
        if (!slot->serial)
            return Made<Slot *>{ok(slot.get())};
        auto completed = valueCompleted(state_, slot->serial);
        if (completed.isErr())
            return failure<Slot *>("the device was lost: " + completed.error());
        if (completed.value())
            return Made<Slot *>{ok(slot.get())};
    }
    if (recording.slots.size() < kSlots) {
        std::unique_ptr<Slot> slot = pool_->take();
        if (!slot)
            slot = std::make_unique<Slot>();
        Slot *made = slot.get();
        recording.slots.push_back(std::move(slot));
        if (auto created = createSlot(recording, *made); created.isErr())
            return failure<Slot *>(std::move(created).error());
        return Made<Slot *>{ok(made)};
    }
    Slot *oldest = recording.slots.front().get();
    for (const std::unique_ptr<Slot> &slot : recording.slots)
        if (slot->serial < oldest->serial)
            oldest = slot.get();
    if (auto waited = waitForValue(state_, oldest->serial); waited.isErr())
        return failure<Slot *>("a pass submission failed on the device: " + waited.error());
    return Made<Slot *>{ok(oldest)};
}

// Encodes the argument buffers of the recording's set instances into the slot's arena.
void MetalDevice::writeArguments(Recording &recording, Slot &slot) {
    for (const SetInstance &set : recording.sets)
        for (uint32_t role = 0; role < 3; ++role) {
            if (set.arguments[role] == kNoOffset)
                continue;
            auto &stage = const_cast<StageObjects &>(*set.stage);
            std::lock_guard<std::mutex> lock(stage.encoding);
            id<MTLArgumentEncoder> encoder = stage.sets[set.set].encoders[role];
            [encoder setArgumentBuffer:slot.arena offset:set.arguments[role]];
            for (const Member &member : set.members[role])
                switch (member.kind) {
                case SlotKind::Inline:
                    [encoder setBuffer:slot.arena offset:member.offset atIndex:member.index];
                    break;
                case SlotKind::Buffer:
                    [encoder setBuffer:member.buffer offset:member.offset atIndex:member.index];
                    break;
                case SlotKind::Image:
                    [encoder setTexture:member.texture atIndex:member.index];
                    break;
                case SlotKind::Sampler:
                    [encoder setSamplerState:member.sampler atIndex:member.index];
                    break;
                case SlotKind::Push:
                    break;
                }
        }
}

// Writes the bytes a call varies: the sites' inline values and the push constants.
Made<void> MetalDevice::fill(Recording &recording, Slot &slot, const PlannedPasses &planned,
                             const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches) {
    VernonRuntimeProviderBindingValue value{};
    std::vector<uint8_t> storage;
    std::string error;
    for (const Site &site : recording.sites) {
        bool filled = false;
        if (site.draw) {
            filled = runtime::fillPreparedGraphicsBindingValue(
                *site.graphics, {site.draw->argumentPointers.data(), site.draw->draw.resolution}, site.slot, value,
                storage, error);
        } else {
            const PlannedPass &pass = *planned.passes[site.pass];
            VernonStageExecutable &stage = *pass.executable->source().resolved->stage;
            const auto &state = ::runtimeBackendState<runtime::MetalPipelineState>(stage);
            filled = runtime::fillPreparedComputeBindingValue(state.computePlan, stage.bindingProjection,
                                                              *launches[site.pass], site.slot, value, error);
        }
        if (!filled)
            return failure<void>("cannot fill a binding: " + error);
        if (!value.payload.inline_value.data || value.payload.inline_value.size != site.size)
            return failure<void>("an inline binding value has the wrong size");
        std::memcpy(site.push ? recording.push.data() + site.offset : slot.mapped + site.offset,
                    value.payload.inline_value.data, site.size);
    }
    if (!slot.written) {
        for (const auto &[offset, bytes] : recording.fixed)
            std::memcpy(slot.mapped + offset, bytes.data(), bytes.size());
        writeArguments(recording, slot);
        slot.written = true;
    }
    for (const PushBlock &block : recording.pushBlocks)
        for (uint32_t index = block.first; index < block.first + block.count; ++index) {
            const auto &[set, instance] = recording.pushSets[index];
            const uint8_t *data = recording.push.data() + recording.sets[instance].pushOffset;
            for (const PushPlacement &placement : block.stage->sets[set].placements)
                std::memcpy(slot.mapped + block.offsets[placement.stage] + placement.offset,
                            data + placement.dataOffset, placement.size);
        }
    return Made<void>{ok()};
}

// The encoder the next command needs, ending the open one when it is of another kind.
class Encoders {
public:
    explicit Encoders(id<MTLCommandBuffer> commands) : commands_(commands) {}

    id<MTLComputeCommandEncoder> compute() {
        if (!compute_) {
            end();
            compute_ = [commands_ computeCommandEncoder];
            pipeline_ = nil;
        }
        return compute_;
    }
    id<MTLBlitCommandEncoder> blit() {
        if (!blit_) {
            end();
            blit_ = [commands_ blitCommandEncoder];
        }
        return blit_;
    }
    id<MTLRenderCommandEncoder> render(MTLRenderPassDescriptor *pass) {
        end();
        render_ = [commands_ renderCommandEncoderWithDescriptor:pass];
        return render_;
    }
    id<MTLRenderCommandEncoder> render() const { return render_; }
    // Debug groups outside render scopes belong to the compute encoder, which a compute pass dispatches on.
    id<MTLCommandEncoder> debugged() {
        if (render_)
            return render_;
        return compute();
    }
    void pipeline(id<MTLComputePipelineState> pipeline) {
        if (pipeline_ == pipeline)
            return;
        [compute() setComputePipelineState:pipeline];
        pipeline_ = pipeline;
    }
    void end() {
        if (compute_)
            [compute_ endEncoding];
        if (blit_)
            [blit_ endEncoding];
        if (render_)
            [render_ endEncoding];
        compute_ = nil;
        blit_ = nil;
        render_ = nil;
    }

private:
    id<MTLCommandBuffer> commands_;
    id<MTLComputeCommandEncoder> compute_;
    id<MTLBlitCommandEncoder> blit_;
    id<MTLRenderCommandEncoder> render_;
    id<MTLComputePipelineState> pipeline_;
};

void useResources(id<MTLRenderCommandEncoder> encoder, const Use *uses, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (@available(macOS 13.0, iOS 16.0, *))
            [encoder useResource:uses[index].resource usage:uses[index].usage stages:uses[index].stages];
        else
            [encoder useResource:uses[index].resource usage:uses[index].usage];
    }
}

// Encodes the recording's commands for one call, with its group counts and host draw streams.
Made<void> MetalDevice::encode(const Recording &recording, Slot &slot, id<MTLCommandBuffer> commands,
                               const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches) {
    Encoders encoders(commands);
    id<MTLBuffer> arena = slot.arena;
    for (const Command &op : recording.commands) {
        switch (op.op) {
        case Op::Fill:
            [encoders.blit() fillBuffer:op.destination range:NSMakeRange(op.destinationOffset, op.size) value:0];
            break;
        case Op::CopyRegions: {
            id<MTLBlitCommandEncoder> blit = encoders.blit();
            for (uint32_t index = op.a; index < op.a + op.b; ++index) {
                const CopyRegion &region = recording.copies[index];
                [blit copyFromBuffer:op.source
                         sourceOffset:region.source
                             toBuffer:op.destination
                    destinationOffset:region.destination
                                 size:region.size];
            }
            break;
        }
        case Op::DebugPush:
            [encoders.debugged() pushDebugGroup:@(recording.labels[op.a].c_str())];
            break;
        case Op::DebugPop:
            [encoders.debugged() popDebugGroup];
            break;
        case Op::DebugMarker:
            [encoders.debugged() insertDebugSignpost:@(recording.labels[op.a].c_str())];
            break;
        case Op::Dispatch: {
            id<MTLComputeCommandEncoder> encoder = encoders.compute();
            encoders.pipeline(op.stage->compute);
            for (uint32_t index = op.d; index < op.d + op.e; ++index) {
                const SetInstance &set = recording.sets[recording.dispatchSets[index]];
                [encoder setBuffer:arena offset:set.arguments[runtime::kMetalCompute] atIndex:set.set];
            }
            for (uint32_t index = op.b; index < op.b + op.c; ++index)
                [encoder useResource:recording.uses[index].resource usage:recording.uses[index].usage];
            const VernonLaunchSize &grid = launches[op.a]->grid;
            [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
                    threadsPerThreadgroup:op.stage->workgroup];
            break;
        }
        case Op::ExpandStream: {
            id<MTLComputeCommandEncoder> encoder = encoders.compute();
            encoders.pipeline(state_.native->streams);
            const uint32_t total = op.a;
            [encoder setBuffer:op.source offset:op.sourceOffset atIndex:0];
            [encoder setBuffer:op.destination offset:op.destinationOffset atIndex:1];
            [encoder setBuffer:slot.arguments offset:op.size atIndex:2];
            [encoder setBytes:&total length:sizeof(total) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((total + kStreamGroup - 1) / kStreamGroup, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kStreamGroup, 1, 1)];
            break;
        }
        case Op::BeginScope: {
            const Rendering &rendering = recording.renderings[op.a];
            id<MTLRenderCommandEncoder> encoder = encoders.render(rendering.pass);
            if (!encoder)
                return failure<void>("cannot begin a render scope");
            useResources(encoder, recording.uses.data() + rendering.firstUse, rendering.uses);
            break;
        }
        case Op::EndScope:
            encoders.end();
            break;
        case Op::BindGraphics: {
            id<MTLRenderCommandEncoder> encoder = encoders.render();
            const StageObjects::Variant &variant = *op.variant;
            [encoder setRenderPipelineState:variant.pipeline];
            if (variant.depthStencil)
                [encoder setDepthStencilState:variant.depthStencil];
            [encoder setCullMode:variant.cull];
            [encoder setFrontFacingWinding:variant.winding];
            [encoder setDepthClipMode:variant.clip];
            [encoder setDepthBias:variant.depthBias slopeScale:variant.slopeBias clamp:0];
            break;
        }
        case Op::Sets: {
            id<MTLRenderCommandEncoder> encoder = encoders.render();
            const SetInstance &set = recording.sets[op.a];
            if (op.b)
                [encoder setVertexBuffer:arena offset:set.arguments[runtime::kMetalVertex] atIndex:set.set];
            if (op.c)
                [encoder setFragmentBuffer:arena offset:set.arguments[runtime::kMetalFragment] atIndex:set.set];
            break;
        }
        case Op::Push: {
            id<MTLRenderCommandEncoder> encoder = encoders.render();
            const PushBlock &block = recording.pushBlocks[op.a];
            if (block.stage->pushSizes[0])
                [encoder setVertexBuffer:arena offset:block.offsets[0] atIndex:runtime::kMetalPushConstantBuffer];
            if (block.stage->pushSizes[1])
                [encoder setFragmentBuffer:arena offset:block.offsets[1] atIndex:runtime::kMetalPushConstantBuffer];
            break;
        }
        case Op::VertexBuffers: {
            id<MTLRenderCommandEncoder> encoder = encoders.render();
            for (uint32_t binding = 0; binding < op.b; ++binding) {
                const VertexBuffer &vertex = recording.vertexBuffers[op.a + binding];
                [encoder setVertexBuffer:vertex.buffer offset:vertex.offset atIndex:kFirstVertexBuffer + binding];
            }
            break;
        }
        case Op::Viewport:
            [encoders.render() setViewport:MTLViewport{static_cast<double>(op.a), static_cast<double>(op.b),
                                                       static_cast<double>(op.c), static_cast<double>(op.d), 0.0, 1.0}];
            break;
        case Op::Scissor:
            [encoders.render() setScissorRect:MTLScissorRect{op.a, op.b, op.c, op.d}];
            break;
        case Op::Stencil:
            [encoders.render() setStencilReferenceValue:op.a];
            break;
        case Op::Draw:
            if (op.a && op.b)
                [encoders.render() drawPrimitives:op.primitive
                                      vertexStart:0
                                      vertexCount:op.a
                                    instanceCount:op.b
                                     baseInstance:op.e];
            break;
        case Op::DrawIndexed:
            if (op.a && op.b)
                [encoders.render() drawIndexedPrimitives:op.primitive
                                              indexCount:op.a
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:op.index
                                       indexBufferOffset:op.indexOffset + uint64_t{op.c} * sizeof(uint32_t)
                                           instanceCount:op.b
                                              baseVertex:static_cast<int32_t>(op.d)
                                            baseInstance:op.e];
            break;
        case Op::DrawIndirect: {
            id<MTLRenderCommandEncoder> encoder = encoders.render();
            id<MTLBuffer> arguments = op.source ? op.source : slot.arguments;
            for (uint32_t record = 0; record < op.a; ++record)
                [encoder drawIndexedPrimitives:op.primitive
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:op.index
                             indexBufferOffset:op.indexOffset
                                indirectBuffer:arguments
                          indirectBufferOffset:op.sourceOffset + uint64_t{record} * kRecordBytes];
            break;
        }
        case Op::DrawHost: {
            id<MTLRenderCommandEncoder> encoder = encoders.render();
            const HostStream &host = recording.streams[op.a];
            const runtime::GraphicsDrawStream &stream = host.draw->stream;
            const VernonRuntimeProviderDrawRecord *records = stream.hostRecords + stream.firstRecord;
            const uint32_t count = (std::min)(stream.hostCounts[stream.countIndex], stream.maxRecords);
            // Checked as the call reads them: O(drawn records).
            for (uint32_t record = 0; record < count; ++record) {
                const VernonRuntimeProviderDrawRecord &source = records[record];
                if (uint64_t{source.first_index} + source.index_count > host.indexCapacity) {
                    encoders.end();
                    return failure<void>("a host draw stream's record exceeds its index buffer");
                }
                if (!source.index_count || !source.instance_count)
                    continue;
                [encoder drawIndexedPrimitives:op.primitive
                                    indexCount:source.index_count
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:op.index
                             indexBufferOffset:op.indexOffset + uint64_t{source.first_index} * sizeof(uint32_t)
                                 instanceCount:source.instance_count
                                    baseVertex:source.base_vertex
                                  baseInstance:source.first_instance];
            }
            break;
        }
        }
    }
    encoders.end();
    return Made<void>{ok()};
}

Made<TimelinePoint> MetalDevice::submit(const CommandBuffer &commands, const uint8_t *record, size_t) {
    const auto &buffer = static_cast<const MetalCommandBuffer &>(commands);
    retireCompleted();
    // The lock keeps every reused plan, and the recordings of it, as they are until the call is submitted.
    std::lock_guard<std::mutex> cacheLock(buffer.passCache().mutex);
    PlannedPasses planned;
    PassPlanner planner(context_, internals_);
    auto made = planner.plan(buffer, record, planned);
    for (const uint64_t evicted : planned.evicted) {
        const auto stale =
            std::remove_if(buffer.recordings.begin(), buffer.recordings.end(), [&](const auto &recording) {
                return std::find(recording->plans.begin(), recording->plans.end(), evicted) != recording->plans.end();
            });
        buffer.recordings.erase(stale, buffer.recordings.end());
    }
    if (made.isErr())
        return failure<TimelinePoint>(std::move(made).error());
    std::vector<std::optional<runtime::PlannedComputeLaunch>> launches(planned.passes.size());
    std::vector<uint64_t> key;
    std::vector<uint64_t> plans;
    bool cacheable = true;
    for (size_t at = 0; at < planned.passes.size(); ++at) {
        const PlannedPass &pass = *planned.passes[at];
        if (pass.copy) {
            addCopyKey(key, *pass.copy);
            continue;
        }
        if (pass.compute) {
            if (auto launched = planLaunch(pass, launches[at], key); launched.isErr())
                return failure<TimelinePoint>(std::move(launched).error());
            continue;
        }
        cacheable = cacheable && pass.plan != 0;
        key.insert(key.end(), {2, pass.plan});
        plans.push_back(pass.plan);
    }
    for (const runtime::GraphicsRenderScope &scope : planned.scopes)
        addScopeKey(key, scope);
    for (const VernonRuntimeProviderResourceReference &range : planned.zeroed)
        addReference(key, range);
    Recording *recording = nullptr;
    std::unique_ptr<Recording> once;
    if (cacheable)
        for (auto found = buffer.recordings.begin(); found != buffer.recordings.end(); ++found)
            if ((*found)->key == key) {
                std::unique_ptr<Recording> used = std::move(*found);
                buffer.recordings.erase(found);
                buffer.recordings.push_back(std::move(used));
                recording = buffer.recordings.back().get();
                break;
            }
    if (!recording) {
        auto built = build(buffer, planned, launches);
        if (built.isErr())
            return failure<TimelinePoint>(std::move(built).error());
        std::unique_ptr<Recording> fresh = std::move(built).value();
        fresh->key = std::move(key);
        fresh->plans = std::move(plans);
        if (cacheable) {
            if (buffer.recordings.size() >= kRecordings)
                buffer.recordings.pop_front();
            buffer.recordings.push_back(std::move(fresh));
            recording = buffer.recordings.back().get();
        } else {
            once = std::move(fresh);
            recording = once.get();
        }
    }
    auto acquired = acquire(*recording);
    if (acquired.isErr())
        return failure<TimelinePoint>(std::move(acquired).error());
    Slot &slot = *acquired.value();
    if (auto filled = fill(*recording, slot, planned, launches); filled.isErr())
        return failure<TimelinePoint>(std::move(filled).error());
    if (anyUndefined(recording->reads.data(), recording->reads.size()))
        return failure<TimelinePoint>("a pass reads an image whose contents are undefined");
    uint64_t value = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> native = [state_.native->queue commandBuffer];
        if (!native)
            return failure<TimelinePoint>("cannot create a Metal command buffer");
        if (auto encoded = encode(*recording, slot, native, launches); encoded.isErr())
            return failure<TimelinePoint>("cannot encode the passes: " + encoded.error());
        std::lock_guard<std::mutex> lock(state_.mutex);
        if (state_.lost)
            return failure<TimelinePoint>("the device was lost: " + state_.lostReason);
        value = metal::commit(state_, native);
    }
    slot.serial = value;
    forgetUndefined(recording->writes.data(), recording->writes.size());
    uint64_t draws = 0;
    for (const PlannedPass *pass : planned.passes)
        draws += pass->draws.size();
    buffer.setTelemetry(planned.scopes.size(), draws);
    std::lock_guard<std::mutex> lock(mutex_);
    const TimelinePoint point = next_++;
    pending_.push_back({point, value});
    return Made<TimelinePoint>{ok(point)};
}

} // namespace

std::unique_ptr<Device> createMetalDevice(VernonRuntimeContext &context) {
    if (context.backend != VERNON_RUNTIME_METAL)
        return nullptr;
    const std::shared_ptr<metal::DeviceState> &state = runtime::metalState(context).device;
    if (!state)
        return nullptr;
    return std::make_unique<MetalDevice>(context, state);
}

} // namespace vernon::hal
