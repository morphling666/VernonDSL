#include "hal/d3d12/d3d12_device.h"
#include "hal/d3d12/d3d12_resources.h"
#include "hal/d3d12/d3d12_state.h"
#include "hal/native_buffers.h"
#include "hal/native_images.h"

#include "VernonFormats.hpp"
#include "VernonSchedule.h"
#include "runtime/backend_directx12.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/dispatch_contract.h"
#include "runtime/interface_names.h"
#include "runtime/prepared_binding_plan.h"
#include "runtime/prepared_graphics_draw.h"
#include "runtime/runtime_state.h"
#include "runtime/schedule/pass_plan.h"
#include "runtime/vertex_attribute_capabilities.h"

#if VERNON_DIRECTX12_DRAW_STREAMS
#include "d3d12_draw_stream.h"
#endif

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

namespace dx = d3d12;
using runtime::PreparedComputeBindingPlan;
using runtime::PreparedGraphicsBindingPlan;

// The states a recording needs an image's subresources in when it begins executing and leaves them in, indexed like
// d3d12::Image::states; kUnused marks the subresources it does not use.
struct StateUse {
    std::vector<D3D12_RESOURCE_STATES> initial;
    std::vector<D3D12_RESOURCE_STATES> current;
};

const auto kUnused = static_cast<D3D12_RESOURCE_STATES>(-1);
constexpr D3D12_RESOURCE_STATES kShaderRead =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
constexpr uint32_t kNone = UINT32_MAX;
constexpr size_t kMaxSets = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;
constexpr size_t kMaxColors = VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS;
constexpr uint64_t kRecordBytes = sizeof(VernonRuntimeProviderDrawRecord);
// A draw's indirect arguments: the vertex stage's draw offsets {base vertex, first instance}, then the draw.
struct DrawArguments {
    int32_t baseVertex;
    uint32_t firstInstance;
    D3D12_DRAW_INDEXED_ARGUMENTS draw;
};
static_assert(sizeof(DrawArguments) == 28, "d3d12_draw_stream.hlsl writes 28-byte arguments");
// Recordings one command buffer keeps: a call reuses one whose passes and timing are its own.
constexpr size_t kRecordings = 8;
// The frame slots of one recording. A slot is written and submitted again only once its last submission completed.
constexpr size_t kSlots = 3;
// D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT.
constexpr uint64_t kRawAlignment = 16;
constexpr uint64_t kConstantAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
constexpr UINT kMaxSamplerDescriptors = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
// PIX_EVENT_ANSI_VERSION: the event's data is a null-terminated narrow string.
constexpr UINT kAnsiEvent = 1;

template <typename T> Made<T> failure(std::string message) { return Made<T>{err(std::move(message))}; }

Made<void> call(HRESULT result, const char *operation) {
    std::string error;
    if (dx::check(result, operation, error))
        return Made<void>{ok()};
    return failure<void>(std::move(error));
}

uint64_t align(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

uint64_t roundUp(uint64_t value, uint64_t minimum) {
    uint64_t result = minimum;
    while (result < value)
        result *= 2;
    return result;
}

D3D12_RESOURCE_BARRIER transitionBarrier(ID3D12Resource *resource, UINT subresource, D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = subresource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

D3D12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CreationNodeMask = 1;
    result.VisibleNodeMask = 1;
    return result;
}

Made<ID3D12Resource *> createBuffer(ID3D12Device *device, D3D12_HEAP_TYPE type, uint64_t size,
                                    D3D12_RESOURCE_STATES state,
                                    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC description{};
    description.Flags = flags;
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = (std::max<uint64_t>)(size, 1);
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES heap = heapProperties(type);
    ID3D12Resource *resource = nullptr;
    if (auto made = call(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description, state, nullptr,
                                                         IID_PPV_ARGS(&resource)),
                         "ID3D12Device::CreateCommittedResource");
        made.isErr())
        return Made<ID3D12Resource *>{err(std::move(made).error())};
    return Made<ID3D12Resource *>{ok(resource)};
}

Made<ID3D12RootSignature *> createRootSignature(ID3D12Device *device, const D3D12_ROOT_SIGNATURE_DESC &description) {
    ID3DBlob *serialized = nullptr;
    ID3DBlob *errors = nullptr;
    const HRESULT result =
        D3D12SerializeRootSignature(&description, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    std::string detail;
    if (errors)
        detail.assign(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize());
    dx::release(errors);
    if (FAILED(result)) {
        dx::release(serialized);
        return failure<ID3D12RootSignature *>("cannot serialize a root signature: " + detail);
    }
    ID3D12RootSignature *root = nullptr;
    auto created = call(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                                    IID_PPV_ARGS(&root)),
                        "ID3D12Device::CreateRootSignature");
    dx::release(serialized);
    if (created.isErr())
        return Made<ID3D12RootSignature *>{err(std::move(created).error())};
    return Made<ID3D12RootSignature *>{ok(root)};
}

Made<bool> valueCompleted(dx::DeviceState &state, uint64_t value) {
    std::lock_guard<std::mutex> lock(state.mutex);
    bool done = false;
    std::string error;
    if (!state.completed(value, done, error))
        return Made<bool>{err(std::move(error))};
    return Made<bool>{ok(done)};
}

Made<void> waitForValue(dx::DeviceState &state, uint64_t value) {
    std::lock_guard<std::mutex> lock(state.mutex);
    std::string error;
    if (!state.waitForValue(value, error))
        return failure<void>(std::move(error));
    return Made<void>{ok()};
}

struct Slot;

// Releases native objects once every submission made before they were retired completed. The device, the Stages'
// pipelines and the recordings share it, and any of them may be destroyed first. It also keeps the slots of dropped
// recordings for later ones, since a pass whose plan changes every call builds a recording every call.
class Retirer {
public:
    explicit Retirer(std::shared_ptr<dx::DeviceState> state) : state_(std::move(state)) {}
    Retirer(const Retirer &) = delete;
    Retirer &operator=(const Retirer &) = delete;
    ~Retirer() { drain(); }

    // Keeps an idle slot, or destroys it when enough are kept.
    void pool(std::unique_ptr<Slot> slot);
    std::unique_ptr<Slot> pooled();

    dx::DeviceState &state() const { return *state_; }
    const std::shared_ptr<dx::DeviceState> &owner() const { return state_; }

    void submitted(uint64_t value) {
        std::lock_guard<std::mutex> lock(mutex_);
        submitted_ = (std::max)(submitted_, value);
    }

    void retire(std::function<void()> destroy) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.push_back({submitted_, std::move(destroy)});
    }

    void collect() {
        std::vector<std::function<void()>> ready;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (!pending_.empty()) {
                if (const uint64_t value = pending_.front().value) {
                    auto completed = valueCompleted(*state_, value);
                    if (completed.isErr() || !completed.value())
                        break;
                }
                ready.push_back(std::move(pending_.front().destroy));
                pending_.pop_front();
            }
        }
        for (const std::function<void()> &destroy : ready)
            destroy();
    }

    // Waits for every submission, then destroys everything retired.
    void drain() {
        // What an entry releases, such as the last pipelines of a Stage, may retire more.
        for (;;) {
            std::deque<Pending> pending;
            uint64_t last = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                last = submitted_;
                pending.swap(pending_);
            }
            if (pending.empty())
                break;
            if (last)
                (void)waitForValue(*state_, last);
            for (Pending &entry : pending)
                entry.destroy();
        }
        std::vector<std::unique_ptr<Slot>> slots;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slots.swap(slots_);
        }
        for (std::unique_ptr<Slot> &slot : slots)
            destroy(*slot);
    }

    void destroy(Slot &slot);

private:
    struct Pending {
        uint64_t value{};
        std::function<void()> destroy;
    };

    static constexpr size_t kPooledSlots = 32;

    std::shared_ptr<dx::DeviceState> state_;
    std::mutex mutex_;
    std::deque<Pending> pending_;
    uint64_t submitted_{};
    std::vector<std::unique_ptr<Slot>> slots_;
};

// The reference kinds of hal/native_buffers.h a DirectX 12 device resolves.
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

// The native object a reference of `state`'s device names: a d3d12::Buffer, d3d12::View or d3d12::Sampler. The
// recording that names it stays valid only while the resource lives, which the keys of the plans it records
// guarantee.
template <typename T>
Made<T *> resolveNative(const dx::DeviceState &state, const VernonRuntimeProviderResourceReference &reference,
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
        native = directX12Buffer(reference.resource.value);
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

// The graphics state enums have Vulkan's values.
D3D12_BLEND blendFactor(uint32_t value) {
    constexpr D3D12_BLEND values[]{D3D12_BLEND_ZERO,          D3D12_BLEND_ONE,           D3D12_BLEND_SRC_COLOR,
                                   D3D12_BLEND_INV_SRC_COLOR, D3D12_BLEND_DEST_COLOR,    D3D12_BLEND_INV_DEST_COLOR,
                                   D3D12_BLEND_SRC_ALPHA,     D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA,
                                   D3D12_BLEND_INV_DEST_ALPHA};
    return value < std::size(values) ? values[value] : D3D12_BLEND_ZERO;
}

D3D12_BLEND_OP blendOperation(uint32_t value) {
    constexpr D3D12_BLEND_OP values[]{D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_SUBTRACT, D3D12_BLEND_OP_REV_SUBTRACT,
                                      D3D12_BLEND_OP_MIN, D3D12_BLEND_OP_MAX};
    return value < std::size(values) ? values[value] : D3D12_BLEND_OP_ADD;
}

D3D12_COMPARISON_FUNC compareOperation(uint32_t value) {
    constexpr D3D12_COMPARISON_FUNC values[]{D3D12_COMPARISON_FUNC_NEVER,         D3D12_COMPARISON_FUNC_LESS,
                                             D3D12_COMPARISON_FUNC_EQUAL,         D3D12_COMPARISON_FUNC_LESS_EQUAL,
                                             D3D12_COMPARISON_FUNC_GREATER,       D3D12_COMPARISON_FUNC_NOT_EQUAL,
                                             D3D12_COMPARISON_FUNC_GREATER_EQUAL, D3D12_COMPARISON_FUNC_ALWAYS};
    return value < std::size(values) ? values[value] : D3D12_COMPARISON_FUNC_NEVER;
}

D3D12_STENCIL_OP stencilOperation(uint32_t value) {
    constexpr D3D12_STENCIL_OP values[]{D3D12_STENCIL_OP_KEEP,     D3D12_STENCIL_OP_ZERO,     D3D12_STENCIL_OP_REPLACE,
                                        D3D12_STENCIL_OP_INCR_SAT, D3D12_STENCIL_OP_DECR_SAT, D3D12_STENCIL_OP_INVERT,
                                        D3D12_STENCIL_OP_INCR,     D3D12_STENCIL_OP_DECR};
    return value < std::size(values) ? values[value] : D3D12_STENCIL_OP_KEEP;
}

DXGI_FORMAT vertexFormat(uint32_t dtype, uint32_t components) {
    static constexpr DXGI_FORMAT formats[][4] = {
        {DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32G32_SINT, DXGI_FORMAT_R32G32B32_SINT, DXGI_FORMAT_R32G32B32A32_SINT},
        {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32B32_UINT, DXGI_FORMAT_R32G32B32A32_UINT},
        {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_FLOAT},
        {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT},
    };
    const int row = dtype == VERNON_RUNTIME_PROVIDER_I32   ? 0
                    : dtype == VERNON_RUNTIME_PROVIDER_U32 ? 1
                    : dtype == VERNON_RUNTIME_PROVIDER_F16 ? 2
                    : dtype == VERNON_RUNTIME_PROVIDER_F32 ? 3
                                                           : -1;
    return row < 0 || components == 0 || components > 4 ? DXGI_FORMAT_UNKNOWN : formats[row][components - 1];
}

D3D12_RECT rectangle(const uint32_t area[4]) {
    return {static_cast<LONG>(area[0]), static_cast<LONG>(area[1]), static_cast<LONG>(area[0] + area[2]),
            static_cast<LONG>(area[1] + area[3])};
}

// Push values are graphics inline values, which the shaders read from one constant buffer per stage at
// kDirectXRootConstantRegister; the other packed values live in the slot arena.
enum class SlotKind : uint8_t { Push, Inline, Buffer, Image, Sampler };

// Where a push value of a binding set goes in its stage's constant buffer: graphics stage 0 is vertex, 1 fragment.
struct PushPlacement {
    uint32_t stage{};
    uint32_t dataOffset{};
    uint32_t offset{};
    uint32_t size{};
};

// How one prepared layout slot of a Stage binds: at `binding` of its range type in register space `set` + 1, through
// descriptor `descriptor` of its set's resource or sampler table. Packed values live in the slot arena: uniform
// blocks as constant buffers, the rest as raw buffers. The compiler declares read storage buffers NonWritable, which
// HLSL makes t-register ByteAddressBuffers, so only written storage buffers are UAVs.
struct SlotPlan {
    SlotKind kind{};
    VernonRuntimeProviderBindingKind providerKind{};
    D3D12_DESCRIPTOR_RANGE_TYPE range{};
    uint32_t set{};
    uint32_t binding{};
    uint32_t size{};
    // A push value's offset in its set's push data.
    uint32_t dataOffset{};
    uint32_t descriptor{kNone};
    // False for a slot repeating an earlier slot's register: the same value read by another stage, whose descriptor
    // the earlier slot writes.
    bool owner{};
    DXGI_FORMAT storageFormat{DXGI_FORMAT_UNKNOWN};
};

// Descriptor set `set` of a Stage: its slots, its descriptor counts, the root parameters of its tables, and the
// argument owners that decide its values.
struct SetPlan {
    std::vector<uint32_t> slots;
    uint32_t resources{};
    uint32_t samplers{};
    uint32_t resourceTable{kNone};
    uint32_t samplerTable{kNone};
    std::vector<uint32_t> parameters;
    bool command{};
    uint32_t pushBytes{};
    std::vector<PushPlacement> placements;
};

// The HAL's native objects of one Stage, which live in its DirectX12PipelineState; the variants are made on first use.
class StageObjects {
public:
    explicit StageObjects(std::shared_ptr<Retirer> retirer) : retirer_(std::move(retirer)) {}
    StageObjects(const StageObjects &) = delete;
    StageObjects &operator=(const StageObjects &) = delete;

    ~StageObjects() {
        std::vector<ID3D12PipelineState *> pipelines;
        for (const Variant &variant : variants)
            pipelines.push_back(variant.pipeline);
        retirer_->retire([root = root, compute = compute, drawSignature = drawSignature, pipelines]() mutable {
            for (ID3D12PipelineState *pipeline : pipelines)
                dx::release(pipeline);
            dx::release(drawSignature);
            dx::release(compute);
            dx::release(root);
        });
    }

    struct Variant {
        runtime::GraphicsVariantKey key;
        ID3D12PipelineState *pipeline{};
    };

    bool graphics{};
    std::vector<SlotPlan> slots;
    std::vector<SetPlan> sets;
    ID3D12RootSignature *root{};
    ID3D12PipelineState *compute{};
    // Graphics only: the root parameters of each stage's push constant buffer and of the vertex stage's draw offsets,
    // and the signature of indirect draws that set the offsets.
    std::array<uint32_t, 2> pushParameters{kNone, kNone};
    std::array<uint32_t, 2> pushSizes{};
    uint32_t offsetsParameter{kNone};
    ID3D12CommandSignature *drawSignature{};
    runtime::DirectX12ShaderCode vertex;
    runtime::DirectX12ShaderCode fragment;
    std::vector<VernonRuntimeProviderVertexBinding> vertexBindings;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    std::mutex mutex;
    std::vector<Variant> variants;

private:
    std::shared_ptr<Retirer> retirer_;
};

enum class Op : uint8_t {
    Heaps,
    UavBarrier,
    Barriers,
    CopyBuffer,
    CopyRegions,
    Timestamp,
    DebugPush,
    DebugPop,
    DebugMarker,
    BindCompute,
    ComputeTable,
    Dispatch,
    // Expands a device draw stream into the slot's draw arguments, whose states ArgumentStates moves.
    ExpandStream,
    ArgumentStates,
    ClearInteger,
    BeginScope,
    BindGraphics,
    GraphicsTable,
    Push,
    VertexBuffers,
    IndexBuffer,
    Topology,
    Viewport,
    Scissor,
    Stencil,
    Draw,
    DrawIndexed,
    DrawIndirect,
};

// One command of a recording; its arguments depend on `op`. A null buffer names the slot's arena.
struct Command {
    Op op{};
    uint32_t a{};
    uint32_t b{};
    uint32_t c{};
    uint32_t d{};
    uint32_t e{};
    ID3D12Resource *source{};
    ID3D12Resource *destination{};
    uint64_t sourceOffset{};
    uint64_t destinationOffset{};
    uint64_t size{};
    const StageObjects *stage{};
    ID3D12PipelineState *pipeline{};
};

// The attachments a render scope binds and the clears it begins with, colors by location.
struct Rendering {
    uint32_t colors{};
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, kMaxColors> targets{};
    std::array<bool, kMaxColors> clearColor{};
    std::array<VernonClearColor, kMaxColors> clears{};
    D3D12_CPU_DESCRIPTOR_HANDLE depth{};
    D3D12_CLEAR_FLAGS depthClears{};
    float clearDepth{};
    uint8_t clearStencil{};
    D3D12_RECT area{};
};

// An integer color attachment's clear through its UAV, whose descriptor set instance `set` places in the slot's heap.
struct IntegerClear {
    ID3D12Resource *resource{};
    D3D12_CPU_DESCRIPTOR_HANDLE view{};
    uint32_t set{};
    VernonClearColor color{};
    D3D12_RECT area{};
};

// The push constant buffers of one draw's stage, which a call composes in the arena from the push data of the set
// instances pushSets[first, first + count).
struct PushBlock {
    const StageObjects *stage{};
    uint32_t first{};
    uint32_t count{};
    std::array<uint64_t, 2> offsets{};
};

// The records of a host draw stream, which a call expands into the arena with their count.
struct HostStream {
    const PlannedDraw *draw{};
    uint64_t records{};
    uint64_t count{};
    uint64_t indexCapacity{};
};

// One descriptor of a set instance's resource table. A null buffer names the slot's arena.
struct Write {
    enum class Type : uint8_t { ConstantBuffer, RawRead, RawWrite, Copy };
    Type type{};
    ID3D12Resource *buffer{};
    uint64_t offset{};
    uint64_t size{};
    D3D12_CPU_DESCRIPTOR_HANDLE source{};
};

// The tables of one descriptor set of a recording, at `resourceBase` and `samplerBase` of its slots' heaps, and the
// push data of its set at `pushOffset` of the recording's: one per binding group, so draws whose groups' owners agree
// share it.
struct SetInstance {
    std::vector<Write> resources;
    std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> samplers;
    uint32_t resourceBase{};
    uint32_t samplerBase{};
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

struct Queries {
    ID3D12QueryHeap *heap{};
    ID3D12Resource *readback{};
    const uint64_t *ticks{};
    Queries() = default;
    Queries(const Queries &) = delete;
    Queries &operator=(const Queries &) = delete;
    ~Queries() {
        if (readback && ticks) {
            const D3D12_RANGE written{0, 0};
            readback->Unmap(0, &written);
        }
        dx::release(readback);
        dx::release(heap);
    }
};

// The scope timings of one timed submission, read once it completed and before its slot records again.
struct TimingCapture {
    std::shared_ptr<Queries> queries;
    std::vector<ScopeTiming> scopes;
    bool read{};
    bool valid{};
};

// One frame of a recording: its command list, descriptor heaps, arena and timestamps. A pooled slot keeps its
// command list, arena and heaps.
struct Slot {
    ID3D12CommandAllocator *allocator{};
    ID3D12GraphicsCommandList *list{};
    ID3D12DescriptorHeap *resources{};
    ID3D12DescriptorHeap *samplers{};
    uint32_t resourceCapacity{};
    uint32_t samplerCapacity{};
    // Upload memory, which stays in GENERIC_READ: constant and raw buffer views and indirect arguments read it.
    ID3D12Resource *arena{};
    uint64_t arenaCapacity{};
    uint8_t *mapped{};
    // The draw arguments device draw streams expand to, in device memory.
    ID3D12Resource *arguments{};
    uint64_t argumentCapacity{};
    std::shared_ptr<Queries> queries;
    std::shared_ptr<TimingCapture> timing;
    uint64_t serial{};
    bool recorded{};
};

void Retirer::destroy(Slot &slot) {
    if (slot.arena && slot.mapped) {
        const D3D12_RANGE written{0, 0};
        slot.arena->Unmap(0, &written);
    }
    dx::release(slot.arena);
    dx::release(slot.arguments);
    dx::release(slot.resources);
    dx::release(slot.samplers);
    dx::release(slot.list);
    dx::release(slot.allocator);
    slot.mapped = nullptr;
}

void Retirer::pool(std::unique_ptr<Slot> slot) {
    // A capture of the slot's last submission may still read its queries, which a later recording would overwrite.
    slot->queries.reset();
    slot->timing.reset();
    slot->serial = 0;
    slot->recorded = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slots_.size() < kPooledSlots) {
            slots_.push_back(std::move(slot));
            return;
        }
    }
    destroy(*slot);
}

std::unique_ptr<Slot> Retirer::pooled() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slots_.empty())
        return nullptr;
    std::unique_ptr<Slot> slot = std::move(slots_.back());
    slots_.pop_back();
    return slot;
}

// The native commands of one call's passes, as planned: a later call with the same key, the same plans of the same
// passes, submits them again after it wrote the bytes that vary. `states` are the images' states as the commands
// begin and leave them; buffers decay to COMMON after every submission, so the commands begin them there.
struct Recording {
    std::shared_ptr<Retirer> retirer;
    std::vector<uint64_t> key;
    std::vector<uint64_t> plans;
    std::vector<std::shared_ptr<StageObjects>> stages;
    std::vector<Command> commands;
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    // The regions of the CopyRegions commands.
    std::vector<CopyRegion> copies;
    std::vector<std::string> labels;
    std::vector<Rendering> renderings;
    std::vector<IntegerClear> clears;
    std::vector<D3D12_VERTEX_BUFFER_VIEW> vertexViews;
    // (set index, instance) of the push data a PushBlock composes.
    std::vector<std::pair<uint32_t, uint32_t>> pushSets;
    std::vector<PushBlock> pushBlocks;
    std::vector<uint8_t> push;
    std::vector<SetInstance> sets;
    std::vector<Site> sites;
    // Arena bytes that never change, written once per slot.
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> fixed;
    std::vector<HostStream> streams;
    // (compute pass, arena offset) of each dispatch's group counts.
    std::vector<std::pair<uint32_t, uint64_t>> dispatches;
    uint64_t arenaBytes{};
    uint64_t argumentBytes{};
    uint32_t resourceDescriptors{};
    uint32_t samplerDescriptors{};
    std::unordered_map<dx::Image *, StateUse> states;
    // Serials of the images the commands read before writing them, and of the attachments they write.
    std::vector<uint64_t> reads;
    std::vector<uint64_t> writes;
    bool timed{};
    std::vector<ScopeTiming> timers;
    ID3D12Resource *zeros{};
    std::vector<std::unique_ptr<Slot>> slots;

    Recording() = default;
    Recording(const Recording &) = delete;
    Recording &operator=(const Recording &) = delete;

    ~Recording() {
        if (!retirer)
            return;
        auto owned = std::make_shared<std::vector<std::unique_ptr<Slot>>>(std::move(slots));
        Retirer *keeper = retirer.get();
        // The Retirer runs what it retires before it is destroyed, so it outlives this.
        retirer->retire([keeper, owned, zeros = zeros, stages = stages]() mutable {
            for (std::unique_ptr<Slot> &slot : *owned)
                keeper->pool(std::move(slot));
            dx::release(zeros);
        });
    }
};

class DirectX12CommandBuffer final : public PassCommandBuffer {
public:
    using PassCommandBuffer::PassCommandBuffer;
    // Guarded by the pass cache's mutex; the most recently used last.
    mutable std::deque<std::unique_ptr<Recording>> recordings;
};

class DirectX12Device;

// Builds the recording of one call's planned passes.
class Builder {
public:
    Builder(DirectX12Device &device, Recording &recording, const PlannedPasses &planned,
            const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Made<void> build(const std::vector<const InternalBuffers::Internal *> &internals);

private:
    struct ImageUse {
        const dx::View *view{};
        D3D12_RESOURCE_STATES state{};
    };
    struct DrawPlan {
        const PlannedDraw *draw{};
        StageObjects *stage{};
        ID3D12PipelineState *pipeline{};
        std::array<uint32_t, kMaxSets> sets{};
        uint32_t vertexFirst{};
        uint32_t vertexCount{};
        runtime::DrawInvocation invocation{};
        D3D12_INDEX_BUFFER_VIEW index{};
        // A device stream's records and counts, and where the stream expands to in the slot's draw arguments.
        ID3D12Resource *records{};
        uint64_t recordsOffset{};
        ID3D12Resource *counts{};
        uint64_t countsOffset{};
        uint64_t arguments{};
    };
    // The state each buffer a command reads or writes needs: a buffer it writes is a UAV however else it reads it.
    using BufferUses = std::map<ID3D12Resource *, D3D12_RESOURCE_STATES>;
    using FillValue = std::function<bool(size_t slot, VernonRuntimeProviderBindingValue &value,
                                         std::vector<uint8_t> &storage, std::string &error)>;

    void add(Command command) { recording_.commands.push_back(command); }
    uint64_t arena(uint64_t size, uint64_t alignment) {
        const uint64_t offset = align(recording_.arenaBytes, alignment);
        recording_.arenaBytes = offset + size;
        return offset;
    }
    void debug(uint32_t kind, const std::string &label);
    void timer();
    void use(BufferUses &uses, ID3D12Resource *buffer, D3D12_RESOURCE_STATES state);
    void transition(ID3D12Resource *buffer, D3D12_RESOURCE_STATES target);
    void transition(const dx::View &view, D3D12_RESOURCE_STATES target);
    void flushBarriers();
    void argumentStates(D3D12_RESOURCE_STATES target);
    Made<StageObjects *> stage(VernonStageExecutable &stage, bool graphics);
    template <typename T> Made<T *> nativeOf(const VernonRuntimeProviderResourceReference &reference, Kind kind) {
        return resolveNative<T>(state_, reference, kind, &resolved_);
    }
    Made<uint32_t> instance(const StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                            const Site &source, BufferUses &buffers, std::vector<ImageUse> &images);
    Made<void> buildCompute(size_t at);
    Made<void> buildCopy(const PlannedCopy &copy);
    Made<void> buildScope(size_t begin, size_t end);
    Made<DrawPlan> planDraw(const PlannedPass &pass, const PlannedDraw &draw, BufferUses &buffers,
                            std::vector<ImageUse> &images);
    Made<void> emitDraw(const DrawPlan &plan);
    Made<Rendering> rendering(const runtime::GraphicsRenderScope &scope);

    DirectX12Device &device_;
    dx::DeviceState &state_;
    Recording &recording_;
    const PlannedPasses &planned_;
    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches_;
    std::map<std::vector<const void *>, uint32_t> instances_;
    // Sampler heap offsets by table contents: a sampler heap holds only 2048 descriptors, and draws whose sets differ
    // in resources mostly share their samplers.
    std::map<std::vector<SIZE_T>, uint32_t> samplerTables_;
    ResolvedCache resolved_;
    std::unordered_map<const VernonStageExecutable *, StageObjects *> stages_;
    std::vector<D3D12_RESOURCE_BARRIER> barriers_;
    // The states of the buffers the commands so far used, which begin in COMMON, as the slot's draw arguments do.
    std::unordered_map<ID3D12Resource *, D3D12_RESOURCE_STATES> buffers_;
    D3D12_RESOURCE_STATES arguments_{D3D12_RESOURCE_STATE_COMMON};
    std::unordered_set<uint64_t> written_;
    uint32_t queries_{};
    // What the commands bound last: the compute Stage, and within a render scope the graphics state.
    const StageObjects *bound_{};
    const StageObjects *graphicsRoot_{};
    ID3D12PipelineState *pipeline_{};
    std::array<uint32_t, kMaxSets> tables_{};
    std::array<uint32_t, kMaxSets> pushed_{};
    const StageObjects *pushStage_{};
    std::optional<uint32_t> topology_;
    std::optional<std::array<uint32_t, 4>> viewport_;
    std::optional<std::array<uint32_t, 4>> scissor_;
    std::optional<uint32_t> stencil_;
    std::optional<std::pair<D3D12_GPU_VIRTUAL_ADDRESS, uint64_t>> index_;
};

class DirectX12Device final : public Device {
public:
    DirectX12Device(VernonRuntimeContext &context, std::shared_ptr<Retirer> retirer)
        : context_(context), retirer_(std::move(retirer)), state_(retirer_->state()),
          buffers_(createDirectX12DeviceBuffers(retirer_->owner())), internals_(buffers_.get()) {}

    ~DirectX12Device() override {
        retirer_->drain();
        dx::release(dispatchSignature_);
        dx::release(streamPipeline_);
        dx::release(streamRoot_);
        std::lock_guard<std::mutex> lock(state_.mutex);
        state_.samplerDescriptors.free(defaultSampler_);
        state_.renderTargets.free(nullTarget_);
    }

    // The objects every recording of the device uses.
    Made<void> initialize() {
        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC signature{};
        signature.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        signature.NumArgumentDescs = 1;
        signature.pArgumentDescs = &argument;
        if (auto made =
                call(state_.device->CreateCommandSignature(&signature, nullptr, IID_PPV_ARGS(&dispatchSignature_)),
                     "ID3D12Device::CreateCommandSignature");
            made.isErr())
            return made;
        // Bound where a pass binds no sampler, as on every backend: trilinear, repeating and sampling the base level.
        D3D12_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.MaxAnisotropy = 1;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler.MinLOD = sampler.MaxLOD = 0;
        std::string error;
        {
            std::lock_guard<std::mutex> lock(state_.mutex);
            defaultSampler_ = state_.samplerDescriptors.allocate(error);
        }
        if (!defaultSampler_.ptr)
            return failure<void>("cannot create the default sampler: " + error);
        state_.device->CreateSampler(&sampler, defaultSampler_);
        // Bound at the color locations a render scope leaves without an attachment, which keep their locations.
        {
            std::lock_guard<std::mutex> lock(state_.mutex);
            nullTarget_ = state_.renderTargets.allocate(error);
        }
        if (!nullTarget_.ptr)
            return failure<void>("cannot create the null render target: " + error);
        D3D12_RENDER_TARGET_VIEW_DESC target{};
        target.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        target.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        state_.device->CreateRenderTargetView(nullptr, &target, nullTarget_);
#if VERNON_DIRECTX12_DRAW_STREAMS
        std::array<D3D12_ROOT_PARAMETER, 3> parameters{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants = {0, 0, 1};
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[1].Descriptor = {0, 0};
        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[2].Descriptor = {0, 0};
        const D3D12_ROOT_SIGNATURE_DESC root{static_cast<UINT>(parameters.size()), parameters.data(), 0, nullptr,
                                             D3D12_ROOT_SIGNATURE_FLAG_NONE};
        auto created = createRootSignature(state_.device, root);
        if (created.isErr())
            return failure<void>("cannot create the draw stream expansion: " + created.error());
        streamRoot_ = created.value();
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = streamRoot_;
        pipeline.CS = {vernon_d3d12_draw_stream, vernon_d3d12_draw_stream_size};
        if (auto made = call(state_.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&streamPipeline_)),
                             "ID3D12Device::CreateComputePipelineState");
            made.isErr())
            return made;
#endif
        return Made<void>{ok()};
    }

    dx::DeviceState &state() const { return state_; }
    D3D12_CPU_DESCRIPTOR_HANDLE defaultSampler() const { return defaultSampler_; }
    D3D12_CPU_DESCRIPTOR_HANDLE nullTarget() const { return nullTarget_; }
    bool expandsStreams() const { return streamPipeline_ != nullptr; }

    Made<std::unique_ptr<Executable>> createExecutable(const ExecutableSource &source) override {
        if (!source.graphics || !source.graphics->resolved || !source.graphics->resolved->stage)
            return failure<std::unique_ptr<Executable>>("DirectX 12 devices run Program nodes only");
        return Made<std::unique_ptr<Executable>>{ok(std::unique_ptr<Executable>(new PassExecutable(source.graphics)))};
    }

    Made<std::unique_ptr<Buffer>> allocate(uint64_t size) override {
        if (!buffers_)
            return failure<std::unique_ptr<Buffer>>("the DirectX 12 device allocates no buffers");
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
        if (auto checked = checkPassCommands(commands, "DirectX 12"); checked.isErr())
            return Made<std::unique_ptr<CommandBuffer>>{err(std::move(checked).error())};
        return Made<std::unique_ptr<CommandBuffer>>{
            ok(std::unique_ptr<CommandBuffer>(new DirectX12CommandBuffer(commands)))};
    }

    Made<TimelinePoint> submit(const CommandBuffer &commands, const uint8_t *record, size_t) override;

    SubmissionTimings timings(TimelinePoint point) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (point >= next_ || (!pending_.empty() && pending_.front().point <= point))
            return {};
        for (const auto &[at, scopes] : timings_)
            if (at == point)
                return {true, scopes};
        return {true, std::nullopt};
    }

    Made<void> wait(TimelinePoint point) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (point >= next_)
            return failure<void>("the timeline point was not signalled yet");
        bool failed = false;
        while (!pending_.empty() && pending_.front().point <= point) {
            Pending retired = std::move(pending_.front());
            pending_.pop_front();
            if (waitForValue(state_, retired.value).isErr())
                failed = true;
            else
                keepTimings(retired);
        }
        if (failed)
            return failure<void>("a pass submission failed on the device");
        return Made<void>{ok()};
    }

    // The objects of a Stage this device dispatches or draws with, made on first use.
    Made<std::shared_ptr<StageObjects>> stageObjects(VernonStageExecutable &stage, bool graphics);
    Made<ID3D12PipelineState *> variant(StageObjects &stage, const runtime::GraphicsVariantKey &key);

private:
    struct Pending {
        TimelinePoint point{};
        uint64_t value{};
        std::shared_ptr<TimingCapture> timing;
    };

    static constexpr size_t kKeptTimings = 32;

    Made<std::unique_ptr<Recording>> build(const DirectX12CommandBuffer &commands, const PlannedPasses &planned,
                                           const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches,
                                           bool timed);
    Made<void> planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
                          std::vector<uint64_t> &key);
    Made<Slot *> acquire(Recording &recording);
    Made<void> createSlot(Recording &recording, Slot &slot);
    Made<void> fill(Recording &recording, Slot &slot, const PlannedPasses &planned,
                    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Made<void> recordCommands(Recording &recording, Slot &slot);
    Made<uint64_t> execute(const Recording &recording, Slot &slot);

    // Called with mutex_ held, or before the slot of the capture records again.
    void capture(TimingCapture &timing) {
        if (timing.read)
            return;
        timing.read = true;
        if (!timing.queries || !timing.queries->ticks || !state_.timestampFrequency)
            return;
        const uint64_t *ticks = timing.queries->ticks;
        const auto nanoseconds = [&](uint64_t value) {
            return static_cast<uint64_t>(static_cast<double>(value) * 1e9 /
                                         static_cast<double>(state_.timestampFrequency));
        };
        const uint64_t origin = timing.scopes.empty() ? 0 : nanoseconds(ticks[0]);
        for (size_t index = 0; index < timing.scopes.size(); ++index) {
            ScopeTiming &scope = timing.scopes[index];
            scope.begin = nanoseconds(ticks[2 * index]);
            scope.end = nanoseconds(ticks[2 * index + 1]);
            scope.begin -= (std::min)(origin, scope.begin);
            scope.end -= (std::min)(origin, scope.end);
            scope.end = (std::max)(scope.end, scope.begin);
        }
        timing.valid = true;
    }

    // Called with mutex_ held.
    void keepTimings(const Pending &retired) {
        if (!retired.timing)
            return;
        std::optional<std::vector<ScopeTiming>> scopes;
        if (retired.timing->queries) {
            capture(*retired.timing);
            if (retired.timing->valid)
                scopes = retired.timing->scopes;
        }
        if (timings_.size() == kKeptTimings)
            timings_.pop_front();
        timings_.emplace_back(retired.point, std::move(scopes));
    }

    // Retires the submissions that completed, oldest first.
    void retireCompleted() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty()) {
            auto completed = valueCompleted(state_, pending_.front().value);
            if (completed.isErr() || !completed.value())
                return;
            keepTimings(pending_.front());
            pending_.pop_front();
        }
    }

    VernonRuntimeContext &context_;
    std::shared_ptr<Retirer> retirer_;
    dx::DeviceState &state_;
    std::unique_ptr<DeviceBuffers> buffers_;
    InternalBuffers internals_;
    ID3D12CommandSignature *dispatchSignature_{};
    ID3D12RootSignature *streamRoot_{};
    ID3D12PipelineState *streamPipeline_{};
    D3D12_CPU_DESCRIPTOR_HANDLE defaultSampler_{};
    D3D12_CPU_DESCRIPTOR_HANDLE nullTarget_{};
    std::mutex mutex_;
    std::deque<Pending> pending_;
    std::deque<std::pair<TimelinePoint, std::optional<std::vector<ScopeTiming>>>> timings_;
    TimelinePoint next_{1};
};

Made<std::shared_ptr<StageObjects>> buildStageObjects(const std::shared_ptr<Retirer> &retirer,
                                                      VernonStageExecutable &stage,
                                                      const runtime::DirectX12PipelineState &state, bool graphics) {
    using Built = Made<std::shared_ptr<StageObjects>>;
    ID3D12Device *device = retirer->state().device;
    auto objects = std::make_shared<StageObjects>(retirer);
    objects->graphics = graphics;
    const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layouts =
        graphics ? state.graphicsPlan.layouts : state.computePlan.layouts;
    const std::vector<runtime::SlotDependencies> dependencies =
        graphics ? runtime::preparedGraphicsSlotDependencies(state.graphicsPlan)
                 : runtime::preparedComputeSlotDependencies(state.computePlan, stage.bindingProjection);
    if (dependencies.size() != layouts.size())
        return failure<std::shared_ptr<StageObjects>>("the Stage's binding dependencies do not match its layouts");
    uint32_t setCount = 0;
    for (const VernonRuntimeProviderBindingLayoutEntry &layout : layouts) {
        if (layout.set >= kMaxSets || layout.array_count != 1)
            return failure<std::shared_ptr<StageObjects>>("the Stage has a binding DirectX 12 devices do not bind");
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
        slot.binding = layout.binding;
        slot.size = layout.element_size;
        if (graphics && layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
            layout.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) {
            const uint32_t alignment = layout.element_alignment;
            if (!alignment || (alignment & (alignment - 1)) != 0 || layout.element_size == 0)
                return failure<std::shared_ptr<StageObjects>>("an inline value's alignment is invalid");
            if (layout.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_VERTEX &&
                layout.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
                return failure<std::shared_ptr<StageObjects>>("an inline value is read by several stages");
            slot.kind = SlotKind::Push;
            slot.dataOffset = set.pushBytes;
            set.pushBytes += layout.element_size;
            continue;
        }
        const bool writes = (layout.access & 2u) != 0;
        switch (layout.kind) {
        case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
            slot.kind = SlotKind::Inline;
            slot.range = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            break;
        case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
            // Compute inline values are read storage buffers.
            slot.kind = SlotKind::Inline;
            slot.range = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            break;
        case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
            slot.kind = packedBytes(layout) ? SlotKind::Inline : SlotKind::Buffer;
            slot.range = writes ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            if (slot.kind == SlotKind::Inline && writes)
                return failure<std::shared_ptr<StageObjects>>("the Stage writes a uniform block");
            break;
        case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
            slot.kind = SlotKind::Image;
            slot.range = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            break;
        case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
            slot.kind = SlotKind::Image;
            slot.range = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            slot.storageFormat = dx::dxgiFormat(layout.storage_image_format);
            if (slot.storageFormat == DXGI_FORMAT_UNKNOWN)
                return failure<std::shared_ptr<StageObjects>>("a storage image's format has no DirectX 12 format");
            break;
        case VERNON_RUNTIME_PROVIDER_SAMPLER:
            slot.kind = SlotKind::Sampler;
            slot.range = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            break;
        default:
            return failure<std::shared_ptr<StageObjects>>("the Stage has a binding DirectX 12 devices do not bind");
        }
        if (slot.kind == SlotKind::Inline && slot.size == 0)
            return failure<std::shared_ptr<StageObjects>>("an inline binding has no bytes");
    }
    // Root parameters point into `ranges`, which therefore never reallocates.
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    ranges.reserve(layouts.size());
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    for (uint32_t index = 0; index < setCount; ++index) {
        SetPlan &set = objects->sets[index];
        for (const bool samplers : {false, true}) {
            const size_t first = ranges.size();
            for (uint32_t slotIndex : set.slots) {
                SlotPlan &slot = objects->slots[slotIndex];
                if (slot.kind == SlotKind::Push || (slot.range == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER) != samplers)
                    continue;
                const auto existing =
                    std::find_if(ranges.begin() + static_cast<std::ptrdiff_t>(first), ranges.end(),
                                 [&](const D3D12_DESCRIPTOR_RANGE &range) {
                                     return range.RangeType == slot.range && range.BaseShaderRegister == slot.binding;
                                 });
                slot.owner = existing == ranges.end();
                if (!slot.owner) {
                    slot.descriptor = existing->OffsetInDescriptorsFromTableStart;
                    continue;
                }
                uint32_t &count = samplers ? set.samplers : set.resources;
                slot.descriptor = count++;
                ranges.push_back({slot.range, 1, slot.binding,
                                  index + ::vernon::runtime::kDirectXDescriptorSetRegisterSpaceOffset,
                                  slot.descriptor});
            }
            if (ranges.size() == first)
                continue;
            (samplers ? set.samplerTable : set.resourceTable) = static_cast<uint32_t>(parameters.size());
            D3D12_ROOT_PARAMETER &parameter = parameters.emplace_back();
            parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.DescriptorTable = {static_cast<UINT>(ranges.size() - first), ranges.data() + first};
            parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
    }
    for (SetPlan &set : objects->sets) {
        std::sort(set.parameters.begin(), set.parameters.end());
        set.parameters.erase(std::unique(set.parameters.begin(), set.parameters.end()), set.parameters.end());
    }
    // Each stage's push values form its constant buffer sorted by entry argument order, which the members of the
    // block the compiler declares follow, at offsets aligned to each value's alignment.
    std::vector<uint32_t> pushSlots;
    for (uint32_t index = 0; index < objects->slots.size(); ++index)
        if (objects->slots[index].kind == SlotKind::Push)
            pushSlots.push_back(index);
    std::stable_sort(pushSlots.begin(), pushSlots.end(), [&](uint32_t left, uint32_t right) {
        return layouts[left].argument_index < layouts[right].argument_index;
    });
    for (uint32_t index : pushSlots) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = layouts[index];
        const SlotPlan &slot = objects->slots[index];
        const uint32_t stageIndex = layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ? 0 : 1;
        const auto offset = static_cast<uint32_t>(align(objects->pushSizes[stageIndex], layout.element_alignment));
        objects->sets[slot.set].placements.push_back({stageIndex, slot.dataOffset, offset, slot.size});
        objects->pushSizes[stageIndex] = offset + slot.size;
    }
    uint32_t cost = static_cast<uint32_t>(parameters.size());
    for (uint32_t stageIndex = 0; stageIndex < 2; ++stageIndex) {
        if (!objects->pushSizes[stageIndex])
            continue;
        if (objects->pushSizes[stageIndex] > D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16)
            return failure<std::shared_ptr<StageObjects>>("a stage's inline values exceed a constant buffer");
        objects->pushParameters[stageIndex] = static_cast<uint32_t>(parameters.size());
        D3D12_ROOT_PARAMETER &parameter = parameters.emplace_back();
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameter.Descriptor = {::vernon::runtime::kDirectXRootConstantRegister,
                                ::vernon::runtime::kDirectXRootConstantRegisterSpace};
        parameter.ShaderVisibility = stageIndex == 0 ? D3D12_SHADER_VISIBILITY_VERTEX : D3D12_SHADER_VISIBILITY_PIXEL;
        cost += 2;
    }
    if (graphics) {
        objects->offsetsParameter = static_cast<uint32_t>(parameters.size());
        D3D12_ROOT_PARAMETER &parameter = parameters.emplace_back();
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameter.Constants = {::vernon::runtime::kDirectXDrawOffsetsRegister,
                               ::vernon::runtime::kDirectXRootConstantRegisterSpace, 2};
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        cost += 2;
    }
    if (cost > D3D12_MAX_ROOT_COST)
        return failure<std::shared_ptr<StageObjects>>("the Stage's root signature exceeds 64 DWORDs");
    const bool inputs = graphics && !state.graphicsPlan.vertexAttributes.empty();
    const D3D12_ROOT_SIGNATURE_DESC root{
        static_cast<UINT>(parameters.size()), parameters.empty() ? nullptr : parameters.data(), 0, nullptr,
        inputs ? D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT : D3D12_ROOT_SIGNATURE_FLAG_NONE};
    auto created = createRootSignature(device, root);
    if (created.isErr())
        return failure<std::shared_ptr<StageObjects>>("the Stage: " + created.error());
    objects->root = created.value();
    if (!graphics) {
        if (state.compute.bytecode.empty())
            return failure<std::shared_ptr<StageObjects>>("the Stage has no DXIL");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = objects->root;
        pipeline.CS = {state.compute.bytecode.data(), state.compute.bytecode.size()};
        if (auto made = call(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&objects->compute)),
                             "ID3D12Device::CreateComputePipelineState");
            made.isErr())
            return Built{err(std::move(made).error())};
        return Built{ok(std::move(objects))};
    }
    if (state.vertex.bytecode.empty() || state.fragment.bytecode.empty())
        return failure<std::shared_ptr<StageObjects>>("the Stage has no DXIL");
    objects->vertex = state.vertex;
    objects->fragment = state.fragment;
    objects->vertexBindings = state.graphicsPlan.vertexBindings;
    objects->vertexAttributes = state.graphicsPlan.vertexAttributes;
    for (const VernonRuntimeProviderVertexAttribute &attribute : objects->vertexAttributes) {
        std::string diagnostic;
        if (!runtime::validateVertexAttributeCapability(runtime::VertexAttributeBackend::DirectX12, attribute,
                                                        D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, false, diagnostic))
            return failure<std::shared_ptr<StageObjects>>(std::move(diagnostic));
        if (vertexFormat(attribute.dtype, attribute.component_count) == DXGI_FORMAT_UNKNOWN)
            return failure<std::shared_ptr<StageObjects>>("a vertex attribute's format is unsupported");
    }
    std::array<D3D12_INDIRECT_ARGUMENT_DESC, 2> arguments{};
    arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    arguments[0].Constant = {objects->offsetsParameter, 0, 2};
    arguments[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    D3D12_COMMAND_SIGNATURE_DESC signature{};
    signature.ByteStride = sizeof(DrawArguments);
    signature.NumArgumentDescs = static_cast<UINT>(arguments.size());
    signature.pArgumentDescs = arguments.data();
    if (auto made =
            call(device->CreateCommandSignature(&signature, objects->root, IID_PPV_ARGS(&objects->drawSignature)),
                 "ID3D12Device::CreateCommandSignature");
        made.isErr())
        return Built{err(std::move(made).error())};
    return Built{ok(std::move(objects))};
}

Made<std::shared_ptr<StageObjects>> DirectX12Device::stageObjects(VernonStageExecutable &stage, bool graphics) {
    using Found = Made<std::shared_ptr<StageObjects>>;
    if (!stage.backendState)
        return failure<std::shared_ptr<StageObjects>>("the Stage has no DirectX 12 pipeline");
    auto &state = ::runtimeBackendState<runtime::DirectX12PipelineState>(stage);
    std::lock_guard<std::mutex> lock(state.halMutex);
    if (!state.hal) {
        auto built = buildStageObjects(retirer_, stage, state, graphics);
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

Made<ID3D12PipelineState *> DirectX12Device::variant(StageObjects &stage, const runtime::GraphicsVariantKey &key) {
    std::lock_guard<std::mutex> lock(stage.mutex);
    for (const StageObjects::Variant &variant : stage.variants)
        if (runtime::graphicsVariantKeysEqual(variant.key, key))
            return Made<ID3D12PipelineState *>{ok(variant.pipeline)};
    if (key.sampleCount != 1 || key.colorFormats.size() > kMaxColors ||
        (key.depthStencilFormat && key.depthStencilFormat != DXGI_FORMAT_D16_UNORM &&
         key.depthStencilFormat != DXGI_FORMAT_D32_FLOAT &&
         key.depthStencilFormat != DXGI_FORMAT_D32_FLOAT_S8X24_UINT) ||
        key.colorBlends.size() != key.colorFormats.size())
        return failure<ID3D12PipelineState *>("the graphics pipeline uses an unsupported attachment configuration");
    const bool hasDepth = key.depthStencilFormat != DXGI_FORMAT_UNKNOWN;
    const bool hasStencil = key.depthStencilFormat == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    // SPIRV-Cross names a vertex input at location N TEXCOORD N.
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputs;
    for (const VernonRuntimeProviderVertexAttribute &attribute : stage.vertexAttributes) {
        const auto binding = std::find_if(stage.vertexBindings.begin(), stage.vertexBindings.end(),
                                          [&](const VernonRuntimeProviderVertexBinding &candidate) {
                                              return candidate.binding == attribute.binding;
                                          });
        if (binding == stage.vertexBindings.end())
            return failure<ID3D12PipelineState *>("a vertex attribute has no vertex binding");
        inputs.push_back({"TEXCOORD", attribute.location, vertexFormat(attribute.dtype, attribute.component_count),
                          attribute.binding, attribute.relative_offset,
                          binding->divisor ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                           : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                          binding->divisor});
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC native{};
    native.pRootSignature = stage.root;
    native.VS = {stage.vertex.bytecode.data(), stage.vertex.bytecode.size()};
    native.PS = {stage.fragment.bytecode.data(), stage.fragment.bytecode.size()};
    native.BlendState.IndependentBlendEnable = key.colorBlends.size() > 1;
    for (size_t index = 0; index < key.colorBlends.size(); ++index) {
        const VernonColorBlendState &source = key.colorBlends[index];
        D3D12_RENDER_TARGET_BLEND_DESC &target = native.BlendState.RenderTarget[index];
        target.BlendEnable = source.blend_enabled ? TRUE : FALSE;
        target.SrcBlend = blendFactor(source.source_color_factor);
        target.DestBlend = blendFactor(source.destination_color_factor);
        target.BlendOp = blendOperation(source.color_operation);
        target.SrcBlendAlpha = blendFactor(source.source_alpha_factor);
        target.DestBlendAlpha = blendFactor(source.destination_alpha_factor);
        target.BlendOpAlpha = blendOperation(source.alpha_operation);
        target.LogicOp = D3D12_LOGIC_OP_NOOP;
        target.RenderTargetWriteMask = static_cast<UINT8>(source.write_mask);
    }
    native.SampleMask = UINT_MAX;
    const VernonRasterizationState &rasterization = key.rasterization;
    if (rasterization.cull_mode > 2 || rasterization.depth_clamp || !std::isfinite(rasterization.depth_bias_constant) ||
        !std::isfinite(rasterization.depth_bias_slope) ||
        std::fabs(rasterization.depth_bias_constant) > static_cast<float>((std::numeric_limits<INT>::max)()))
        return failure<ID3D12PipelineState *>("the graphics pipeline has an unsupported rasterization state");
    constexpr D3D12_CULL_MODE culls[]{D3D12_CULL_MODE_NONE, D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_BACK};
    native.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    native.RasterizerState.CullMode = culls[rasterization.cull_mode];
    // Without a flipped viewport, counter-clockwise on the render target is counter-clockwise in clip space.
    native.RasterizerState.FrontCounterClockwise = rasterization.front_face == 0;
    native.RasterizerState.DepthBias =
        rasterization.depth_bias_enabled ? static_cast<INT>(rasterization.depth_bias_constant) : 0;
    native.RasterizerState.SlopeScaledDepthBias =
        rasterization.depth_bias_enabled ? rasterization.depth_bias_slope : 0.0f;
    native.RasterizerState.DepthClipEnable = TRUE;
    const VernonDepthStencilState &depthState = key.depthStencil;
    if ((!hasDepth && (depthState.depth_test || depthState.depth_write)) || (!hasStencil && depthState.stencil_test))
        return failure<ID3D12PipelineState *>("the graphics pipeline has an invalid depth stencil state");
    // As on Vulkan, a draw without the depth test writes no depth.
    native.DepthStencilState.DepthEnable = depthState.depth_test ? TRUE : FALSE;
    native.DepthStencilState.DepthWriteMask =
        depthState.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    native.DepthStencilState.DepthFunc =
        depthState.depth_test ? compareOperation(depthState.depth_compare) : D3D12_COMPARISON_FUNC_ALWAYS;
    native.DepthStencilState.StencilEnable = depthState.stencil_test ? TRUE : FALSE;
    native.DepthStencilState.StencilReadMask = static_cast<UINT8>(depthState.stencil_read_mask);
    native.DepthStencilState.StencilWriteMask = static_cast<UINT8>(depthState.stencil_write_mask);
    const auto face = [](const VernonStencilFaceState &source) {
        return D3D12_DEPTH_STENCILOP_DESC{stencilOperation(source.stencil_fail), stencilOperation(source.depth_fail),
                                          stencilOperation(source.pass), compareOperation(source.compare)};
    };
    native.DepthStencilState.FrontFace = face(depthState.front);
    native.DepthStencilState.BackFace = face(depthState.back);
    native.InputLayout = {inputs.empty() ? nullptr : inputs.data(), static_cast<UINT>(inputs.size())};
    native.PrimitiveTopologyType = key.topology == 1   ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                   : key.topology == 2 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                                                       : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    native.NumRenderTargets = static_cast<UINT>(key.colorFormats.size());
    for (size_t index = 0; index < key.colorFormats.size(); ++index)
        native.RTVFormats[index] = static_cast<DXGI_FORMAT>(key.colorFormats[index]);
    native.DSVFormat = static_cast<DXGI_FORMAT>(key.depthStencilFormat);
    native.SampleDesc.Count = 1;
    ID3D12PipelineState *pipeline = nullptr;
    if (auto made = call(state_.device->CreateGraphicsPipelineState(&native, IID_PPV_ARGS(&pipeline)),
                         "ID3D12Device::CreateGraphicsPipelineState");
        made.isErr())
        return Made<ID3D12PipelineState *>{err(std::move(made).error())};
    stage.variants.push_back({key, pipeline});
    return Made<ID3D12PipelineState *>{ok(pipeline)};
}

Builder::Builder(DirectX12Device &device, Recording &recording, const PlannedPasses &planned,
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

void Builder::timer() {
    if (!recording_.timed || !state_.timestampFrequency)
        return;
    Command command{Op::Timestamp};
    command.a = queries_++;
    add(command);
}

void Builder::use(BufferUses &uses, ID3D12Resource *buffer, D3D12_RESOURCE_STATES state) {
    const auto [found, inserted] = uses.emplace(buffer, state);
    if (inserted)
        return;
    found->second =
        found->second == D3D12_RESOURCE_STATE_UNORDERED_ACCESS || state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            : found->second | state;
}

void Builder::transition(ID3D12Resource *buffer, D3D12_RESOURCE_STATES target) {
    D3D12_RESOURCE_STATES &current = buffers_.try_emplace(buffer, D3D12_RESOURCE_STATE_COMMON).first->second;
    if (current == target)
        return;
    barriers_.push_back(transitionBarrier(buffer, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, current, target));
    current = target;
}

// Moves the subresources `view` covers to `target` at this point of the recording. A subresource's first use defines
// the state the recording begins it in, to which submission transitions it.
void Builder::transition(const dx::View &view, D3D12_RESOURCE_STATES target) {
    dx::Image &image = *view.owner;
    StateUse &use = recording_.states[&image];
    if (use.current.empty()) {
        use.initial.assign(image.states.size(), kUnused);
        use.current.assign(image.states.size(), kUnused);
    }
    uint32_t firstPlane = 0;
    uint32_t planes = image.planes;
    if (image.planes > 1 && view.aspects == VERNON_IMAGE_ASPECT_DEPTH)
        planes = 1;
    else if (image.planes > 1 && view.aspects == VERNON_IMAGE_ASPECT_STENCIL)
        firstPlane = 1;
    const uint32_t lastLayer = (std::min)(view.baseLayer + (std::max)(view.layers, 1u), image.layers);
    for (uint32_t plane = firstPlane; plane < planes; ++plane)
        for (uint32_t layer = view.baseLayer; layer < lastLayer; ++layer)
            for (uint32_t mip = view.baseMip; mip < view.baseMip + view.mips && mip < image.mipLevels; ++mip) {
                const uint32_t subresource = image.subresource(mip, layer, plane);
                if (subresource >= use.current.size())
                    continue;
                D3D12_RESOURCE_STATES &current = use.current[subresource];
                if (current == kUnused)
                    use.initial[subresource] = target;
                else if (current != target)
                    barriers_.push_back(transitionBarrier(image.resource, subresource, current, target));
                current = target;
            }
}

void Builder::flushBarriers() {
    if (barriers_.empty())
        return;
    Command command{Op::Barriers};
    command.a = static_cast<uint32_t>(recording_.barriers.size());
    command.b = static_cast<uint32_t>(barriers_.size());
    recording_.barriers.insert(recording_.barriers.end(), barriers_.begin(), barriers_.end());
    add(command);
    barriers_.clear();
}

void Builder::argumentStates(D3D12_RESOURCE_STATES target) {
    if (arguments_ == target)
        return;
    Command command{Op::ArgumentStates};
    command.a = static_cast<uint32_t>(arguments_);
    command.b = static_cast<uint32_t>(target);
    add(command);
    arguments_ = target;
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

// Makes the tables and push data of `set` from the values `fill` gives, and adds what they bind to `buffers` and
// `images`. Its inline bytes are sites the call fills again from `source`, unless the instance is `persistent`: its
// owners' values, and so its bytes, stay as they are across calls.
Made<uint32_t> Builder::instance(const StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                                 const Site &source, BufferUses &buffers, std::vector<ImageUse> &images) {
    const SetPlan &plan = stage.sets[set];
    const auto at = static_cast<uint32_t>(recording_.sets.size());
    SetInstance made;
    made.resources.resize(plan.resources);
    made.samplers.resize(plan.samplers);
    made.resourceBase = recording_.resourceDescriptors;
    made.samplerBase = recording_.samplerDescriptors;
    if (plan.pushBytes) {
        made.pushOffset = static_cast<uint32_t>(recording_.push.size());
        recording_.push.resize(recording_.push.size() + plan.pushBytes);
    }
    std::string error;
    std::vector<uint8_t> storage;
    for (const uint32_t slotIndex : plan.slots) {
        const SlotPlan &slot = stage.slots[slotIndex];
        if (slot.kind != SlotKind::Push && !slot.owner)
            continue;
        VernonRuntimeProviderBindingValue value{};
        if (!fill(slotIndex, value, storage, error))
            return failure<uint32_t>("cannot fill a binding: " + error);
        if (value.kind != slot.providerKind)
            return failure<uint32_t>("a binding value has the wrong kind");
        Site site = source;
        site.slot = slotIndex;
        site.size = slot.size;
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
            const bool constant = slot.range == D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            // A constant buffer view spans whole 256-byte blocks.
            site.offset = constant ? arena(align(slot.size, kConstantAlignment), kConstantAlignment)
                                   : arena(align(slot.size, 4), kRawAlignment);
            const auto *bytes = static_cast<const uint8_t *>(value.payload.inline_value.data);
            if (persistent)
                recording_.fixed.emplace_back(site.offset, std::vector<uint8_t>(bytes, bytes + slot.size));
            else
                recording_.sites.push_back(site);
            made.resources[slot.descriptor] = {constant ? Write::Type::ConstantBuffer : Write::Type::RawRead, nullptr,
                                               site.offset, slot.size};
            break;
        }
        case SlotKind::Buffer: {
            const VernonRuntimeProviderResourceReference &resource = value.payload.buffer.resource;
            if (resource.size == 0)
                return failure<uint32_t>("a buffer binding is empty");
            if (resource.offset % kRawAlignment != 0)
                return failure<uint32_t>("DirectX 12 binds storage buffers at offsets that are multiples of 16 bytes");
            auto buffer = nativeOf<dx::Buffer>(resource, Kind::Buffer);
            if (buffer.isErr())
                return failure<uint32_t>(std::move(buffer).error());
            const bool written = slot.range == D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            made.resources[slot.descriptor] = {written ? Write::Type::RawWrite : Write::Type::RawRead,
                                               buffer.value()->resource, resource.offset, resource.size};
            use(buffers, buffer.value()->resource, written ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : kShaderRead);
            break;
        }
        case SlotKind::Image: {
            if (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE)
                return failure<uint32_t>("an image binding has no image");
            auto view = nativeOf<dx::View>(value.payload.image.view, Kind::View);
            if (view.isErr())
                return failure<uint32_t>(std::move(view).error());
            const bool storage = slot.range == D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            const D3D12_CPU_DESCRIPTOR_HANDLE handle = storage ? view.value()->uav : view.value()->srv;
            if (!handle.ptr)
                return failure<uint32_t>(storage ? "a storage image's view has no storage usage"
                                                 : "a sampled image's view has no sampled usage");
            if (storage && view.value()->format != slot.storageFormat)
                return failure<uint32_t>("a storage image's format is not the one its kernel declares");
            images.push_back({view.value(), storage ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : kShaderRead});
            made.resources[slot.descriptor] = {Write::Type::Copy, nullptr, 0, 0, handle};
            break;
        }
        case SlotKind::Sampler: {
            if (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) {
                made.samplers[slot.descriptor] = device_.defaultSampler();
                break;
            }
            auto sampler = nativeOf<dx::Sampler>(value.payload.sampler.resource, Kind::Sampler);
            if (sampler.isErr())
                return failure<uint32_t>(std::move(sampler).error());
            made.samplers[slot.descriptor] = sampler.value()->cpu;
            break;
        }
        }
    }
    recording_.resourceDescriptors += plan.resources;
    if (!made.samplers.empty()) {
        std::vector<SIZE_T> key;
        key.reserve(made.samplers.size());
        for (const D3D12_CPU_DESCRIPTOR_HANDLE sampler : made.samplers)
            key.push_back(sampler.ptr);
        const auto table = samplerTables_.emplace(std::move(key), made.samplerBase);
        if (table.second) {
            recording_.samplerDescriptors += plan.samplers;
        } else {
            made.samplerBase = table.first->second;
            made.samplers.clear();
        }
    }
    recording_.sets.push_back(std::move(made));
    return Made<uint32_t>{ok(at)};
}

Made<void> Builder::build(const std::vector<const InternalBuffers::Internal *> &internals) {
    add({Op::Heaps});
    // Ordered after all work submitted before it, whose shader writes a UAV barrier makes visible.
    add({Op::UavBarrier});
    if (!internals.empty() || !planned_.zeroed.empty()) {
        uint64_t bytes = 0;
        for (const InternalBuffers::Internal *internal : internals)
            bytes = (std::max)(bytes, internal->bytes);
        for (const VernonRuntimeProviderResourceReference &range : planned_.zeroed)
            bytes = (std::max)(bytes, range.size);
        auto zeros = createBuffer(state_.device, D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
        if (zeros.isErr())
            return failure<void>("cannot create the internal buffers' zeros: " + zeros.error());
        recording_.zeros = zeros.value();
        void *mapped = nullptr;
        const D3D12_RANGE none{0, 0};
        if (auto made = call(recording_.zeros->Map(0, &none, &mapped), "ID3D12Resource::Map"); made.isErr())
            return made;
        std::memset(mapped, 0, bytes);
        recording_.zeros->Unmap(0, nullptr);
        std::vector<Command> copies;
        const auto zero = [&](const VernonRuntimeProviderResourceReference &reference, uint64_t offset,
                              uint64_t size) -> Made<void> {
            auto buffer = nativeOf<dx::Buffer>(reference, Kind::Buffer);
            if (buffer.isErr())
                return Made<void>{err(std::move(buffer).error())};
            transition(buffer.value()->resource, D3D12_RESOURCE_STATE_COPY_DEST);
            Command copy{Op::CopyBuffer};
            copy.source = recording_.zeros;
            copy.destination = buffer.value()->resource;
            copy.destinationOffset = offset;
            copy.size = size;
            copies.push_back(copy);
            return Made<void>{ok()};
        };
        for (const InternalBuffers::Internal *internal : internals)
            if (auto zeroed = zero(internal->reference, 0, internal->bytes); zeroed.isErr())
                return failure<void>("an internal buffer: " + zeroed.error());
        for (const VernonRuntimeProviderResourceReference &range : planned_.zeroed)
            if (range.size)
                if (auto zeroed = zero(range, range.offset, range.size); zeroed.isErr())
                    return failure<void>("a zeroed argument: " + zeroed.error());
        flushBarriers();
        recording_.commands.insert(recording_.commands.end(), copies.begin(), copies.end());
    }
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
            if (recording_.timed)
                recording_.timers.push_back({pass.command->node, 1, 0, 0});
            timer();
            debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, pass.command->label);
            if (auto built = buildCompute(at); built.isErr())
                return built;
            debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
            timer();
            ++at;
            continue;
        }
        size_t end = at + 1;
        while (end < passes.size() && !passes[end]->compute && !passes[end]->copy && passes[end]->scope == pass.scope)
            ++end;
        if (recording_.timed)
            recording_.timers.push_back({pass.command->node, static_cast<uint32_t>(end - at), 0, 0});
        timer();
        if (auto built = buildScope(at, end); built.isErr())
            return built;
        timer();
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
    const auto &state = ::runtimeBackendState<runtime::DirectX12PipelineState>(executable);
    const PreparedComputeBindingPlan &plan = state.computePlan;
    const std::string name = "pass '" + pass.executable->source().node->name + "': ";
    const FillValue fill = [&](size_t slot, VernonRuntimeProviderBindingValue &value, std::vector<uint8_t> &,
                               std::string &error) {
        return runtime::fillPreparedComputeBindingValue(plan, executable.bindingProjection, launch, slot, value, error);
    };
    Site source;
    source.pass = static_cast<uint32_t>(at);
    std::array<uint32_t, kMaxSets> sets{};
    sets.fill(kNone);
    BufferUses buffers;
    std::vector<ImageUse> images;
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        if (stageObjects.sets[set].slots.empty())
            continue;
        auto made = instance(stageObjects, set, false, fill, source, buffers, images);
        if (made.isErr())
            return Made<void>{err(name + made.error())};
        sets[set] = made.value();
    }
    for (const auto &[buffer, needed] : buffers)
        transition(buffer, needed);
    for (const ImageUse &use : images)
        transition(*use.view, use.state);
    flushBarriers();
    if (bound_ != &stageObjects) {
        Command bind{Op::BindCompute};
        bind.stage = &stageObjects;
        add(bind);
        bound_ = &stageObjects;
    }
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        if (sets[set] == kNone)
            continue;
        const SetPlan &setPlan = stageObjects.sets[set];
        const SetInstance &instance = recording_.sets[sets[set]];
        if (setPlan.resourceTable != kNone)
            add({Op::ComputeTable, setPlan.resourceTable, 0, instance.resourceBase});
        if (setPlan.samplerTable != kNone)
            add({Op::ComputeTable, setPlan.samplerTable, 1, instance.samplerBase});
    }
    // Indirect, so that a call whose grid changed submits the recording again after it wrote its group counts.
    Command dispatch{Op::Dispatch};
    dispatch.sourceOffset = arena(sizeof(D3D12_DISPATCH_ARGUMENTS), sizeof(uint32_t));
    recording_.dispatches.emplace_back(static_cast<uint32_t>(at), dispatch.sourceOffset);
    add(dispatch);
    // What follows in the same state reads what the dispatch wrote; other states transition.
    add({Op::UavBarrier});
    return Made<void>{ok()};
}

Made<void> Builder::buildCopy(const PlannedCopy &copy) {
    if (copy.regions.empty())
        return Made<void>{ok()};
    auto source = nativeOf<dx::Buffer>(copy.source, Kind::Buffer);
    if (source.isErr())
        return failure<void>("a leaf copy's source: " + source.error());
    auto destination = nativeOf<dx::Buffer>(copy.destination, Kind::Buffer);
    if (destination.isErr())
        return failure<void>("a leaf copy's destination: " + destination.error());
    if (source.value()->resource == destination.value()->resource)
        return failure<void>("DirectX 12 devices copy between distinct buffers only");
    transition(source.value()->resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(destination.value()->resource, D3D12_RESOURCE_STATE_COPY_DEST);
    flushBarriers();
    Command regions{Op::CopyRegions};
    regions.source = source.value()->resource;
    regions.destination = destination.value()->resource;
    regions.a = static_cast<uint32_t>(recording_.copies.size());
    regions.b = static_cast<uint32_t>(copy.regions.size());
    recording_.copies.insert(recording_.copies.end(), copy.regions.begin(), copy.regions.end());
    add(regions);
    return Made<void>{ok()};
}

Made<Builder::DrawPlan> Builder::planDraw(const PlannedPass &pass, const PlannedDraw &draw, BufferUses &buffers,
                                          std::vector<ImageUse> &images) {
    using Planned = Made<DrawPlan>;
    DrawPlan result;
    result.draw = &draw;
    result.sets.fill(kNone);
    auto &state = ::runtimeBackendState<runtime::DirectX12PipelineState>(*draw.stage);
    if (state.vertex.bytecode.empty())
        return Planned{ok(result)};
    auto objects = stage(*draw.stage, true);
    if (objects.isErr())
        return Planned{err(std::move(objects).error())};
    StageObjects &stageObjects = *objects.value();
    result.stage = &stageObjects;
    const PreparedGraphicsBindingPlan &plan = state.graphicsPlan;
    const runtime::GraphicsSlotInputs inputs{draw.argumentPointers.data(), draw.draw.resolution};
    std::string error;
    const auto view = [&](const VernonRuntimeProviderResourceReference &reference,
                          D3D12_RESOURCE_STATES needed) -> Made<std::pair<D3D12_GPU_VIRTUAL_ADDRESS, UINT>> {
        using Viewed = Made<std::pair<D3D12_GPU_VIRTUAL_ADDRESS, UINT>>;
        auto buffer = nativeOf<dx::Buffer>(reference, Kind::Buffer);
        if (buffer.isErr())
            return Viewed{err(std::move(buffer).error())};
        const dx::Buffer &native = *buffer.value();
        const uint64_t size =
            reference.size ? reference.size : native.size - (std::min)(native.size, uint64_t{reference.offset});
        if (size > (std::numeric_limits<UINT>::max)())
            return Viewed{err(std::string("a vertex or index buffer exceeds 4 GiB"))};
        use(buffers, native.resource, needed);
        return Viewed{
            ok(std::make_pair(native.resource->GetGPUVirtualAddress() + reference.offset, static_cast<UINT>(size)))};
    };
    std::vector<uint32_t> strides(plan.vertexInputs.size());
    result.vertexFirst = static_cast<uint32_t>(recording_.vertexViews.size());
    result.vertexCount = static_cast<uint32_t>(plan.vertexInputs.size());
    for (size_t binding = 0; binding < plan.vertexInputs.size(); ++binding) {
        VernonRuntimeProviderResourceReference buffer{};
        if (!runtime::preparedGraphicsVertexBuffer(plan, inputs, binding, buffer, strides[binding], error))
            return failure<DrawPlan>(error);
        auto viewed = view(buffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        if (viewed.isErr())
            return failure<DrawPlan>("a vertex buffer: " + viewed.error());
        recording_.vertexViews.push_back({viewed.value().first, viewed.value().second, strides[binding]});
    }
    std::vector<uint32_t> formats;
    for (VernonTextureFormat format : pass.targets.attachmentFormats) {
        const DXGI_FORMAT native = dx::dxgiFormat(format);
        if (native == DXGI_FORMAT_UNKNOWN)
            return failure<DrawPlan>("DirectX 12 color attachment format is unsupported");
        formats.push_back(static_cast<uint32_t>(native));
    }
    uint32_t depthFormat = 0;
    if (pass.targets.depthAttachment) {
        depthFormat = static_cast<uint32_t>(dx::dxgiFormat(pass.targets.depthFormat));
        if (!depthFormat)
            return failure<DrawPlan>("DirectX 12 depth attachment format is unsupported");
    }
    runtime::GraphicsVariantKey key;
    if (!runtime::prepareGraphicsVariantKey(*draw.state, pass.targets, std::move(formats), depthFormat, strides, key,
                                            error))
        return failure<DrawPlan>(error);
    auto pipeline = device_.variant(stageObjects, key);
    if (pipeline.isErr())
        return Planned{err(std::move(pipeline).error())};
    result.pipeline = pipeline.value();
    const FillValue fill = [&](size_t slot, VernonRuntimeProviderBindingValue &value, std::vector<uint8_t> &storage,
                               std::string &message) {
        return runtime::fillPreparedGraphicsBindingValue(plan, inputs, slot, value, storage, message);
    };
    Site source;
    source.draw = &draw;
    source.graphics = &plan;
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        const SetPlan &setPlan = stageObjects.sets[set];
        if (setPlan.slots.empty())
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
        auto created = instance(stageObjects, set, persistent, fill, source, buffers, images);
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
        auto viewed = view(result.invocation.indexBuffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        if (viewed.isErr())
            return failure<DrawPlan>("an index buffer: " + viewed.error());
        result.index = {viewed.value().first, viewed.value().second, DXGI_FORMAT_R32_UINT};
    }
    const runtime::GraphicsDrawStream &stream = draw.stream;
    if (draw.streamed && stream.records.resource.value && stream.maxRecords) {
        if (!device_.expandsStreams())
            return failure<DrawPlan>("the device draws no device draw streams");
        const bool counted = stream.counts.resource.value != 0;
        if (uint64_t{stream.firstRecord} + stream.maxRecords > stream.records.size / kRecordBytes ||
            (counted && stream.countIndex >= stream.counts.size / sizeof(uint32_t)))
            return failure<DrawPlan>("a device draw stream exceeds its buffers");
        auto records = nativeOf<dx::Buffer>(stream.records, Kind::Buffer);
        if (records.isErr())
            return failure<DrawPlan>("a draw stream's records: " + records.error());
        result.records = records.value()->resource;
        result.recordsOffset = stream.records.offset + uint64_t{stream.firstRecord} * kRecordBytes;
        use(buffers, result.records, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (counted) {
            auto counts = nativeOf<dx::Buffer>(stream.counts, Kind::Buffer);
            if (counts.isErr())
                return failure<DrawPlan>("a draw stream's counts: " + counts.error());
            result.counts = counts.value()->resource;
            result.countsOffset = stream.counts.offset + uint64_t{stream.countIndex} * sizeof(uint32_t);
            use(buffers, result.counts, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        result.arguments = recording_.argumentBytes;
        recording_.argumentBytes += uint64_t{stream.maxRecords} * sizeof(DrawArguments);
    }
    return Planned{ok(result)};
}

Made<void> Builder::emitDraw(const DrawPlan &plan) {
    if (!plan.stage)
        return Made<void>{ok()};
    const StageObjects &stage = *plan.stage;
    const PlannedDraw &draw = *plan.draw;
    if (graphicsRoot_ != &stage || pipeline_ != plan.pipeline) {
        Command bind{Op::BindGraphics};
        bind.stage = graphicsRoot_ != &stage ? &stage : nullptr;
        bind.pipeline = plan.pipeline;
        add(bind);
        if (graphicsRoot_ != &stage) {
            graphicsRoot_ = &stage;
            tables_.fill(kNone);
            pushStage_ = nullptr;
        }
        pipeline_ = plan.pipeline;
    }
    bool push = pushStage_ != &stage;
    for (uint32_t set = 0; set < stage.sets.size(); ++set) {
        const uint32_t instance = plan.sets[set];
        if (instance == kNone)
            continue;
        const SetPlan &setPlan = stage.sets[set];
        if (setPlan.pushBytes && pushed_[set] != instance)
            push = true;
        if (tables_[set] == instance)
            continue;
        const SetInstance &made = recording_.sets[instance];
        if (setPlan.resourceTable != kNone)
            add({Op::GraphicsTable, setPlan.resourceTable, 0, made.resourceBase});
        if (setPlan.samplerTable != kNone)
            add({Op::GraphicsTable, setPlan.samplerTable, 1, made.samplerBase});
        tables_[set] = instance;
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
                block.offsets[index] = arena(align(stage.pushSizes[index], kConstantAlignment), kConstantAlignment);
        add({Op::Push, static_cast<uint32_t>(recording_.pushBlocks.size())});
        recording_.pushBlocks.push_back(block);
        pushStage_ = &stage;
    }
    if (plan.vertexCount)
        add({Op::VertexBuffers, plan.vertexFirst, plan.vertexCount});
    const runtime::DrawInvocation &invocation = plan.invocation;
    if (topology_ != invocation.topology) {
        add({Op::Topology, invocation.topology});
        topology_ = invocation.topology;
    }
    const std::array<uint32_t, 4> viewport{invocation.viewport[0], invocation.viewport[1], invocation.viewport[2],
                                           invocation.viewport[3]};
    if (viewport_ != viewport) {
        add({Op::Viewport, viewport[0], viewport[1], viewport[2], viewport[3]});
        viewport_ = viewport;
    }
    const std::array<uint32_t, 4> scissor{invocation.scissor[0], invocation.scissor[1], invocation.scissor[2],
                                          invocation.scissor[3]};
    if (scissor_ != scissor) {
        add({Op::Scissor, scissor[0], scissor[1], scissor[2], scissor[3]});
        scissor_ = scissor;
    }
    if (stencil_ != invocation.stencilReference) {
        add({Op::Stencil, invocation.stencilReference});
        stencil_ = invocation.stencilReference;
    }
    if (plan.index.BufferLocation) {
        const std::pair<D3D12_GPU_VIRTUAL_ADDRESS, uint64_t> index{plan.index.BufferLocation, plan.index.SizeInBytes};
        if (index_ != index) {
            Command bind{Op::IndexBuffer};
            bind.sourceOffset = index.first;
            bind.size = index.second;
            add(bind);
            index_ = index;
        }
    }
    if (!draw.streamed) {
        Command drawn{plan.index.BufferLocation ? Op::DrawIndexed : Op::Draw};
        drawn.stage = &stage;
        if (plan.index.BufferLocation) {
            drawn.a = invocation.indexCount;
            drawn.b = invocation.instanceCount;
            drawn.c = invocation.firstIndex;
            drawn.d = static_cast<uint32_t>(invocation.baseVertex);
            drawn.e = invocation.firstInstance;
        } else {
            drawn.a = invocation.vertexCount;
            drawn.b = invocation.instanceCount;
            drawn.e = invocation.firstInstance;
        }
        add(drawn);
        return Made<void>{ok()};
    }
    const runtime::GraphicsDrawStream &stream = draw.stream;
    if (stream.maxRecords == 0)
        return Made<void>{ok()};
    Command indirect{Op::DrawIndirect};
    indirect.stage = &stage;
    indirect.a = stream.maxRecords;
    if (stream.records.resource.value) {
        indirect.b = 1;
        indirect.c = plan.counts ? 1 : 0;
        indirect.sourceOffset = plan.arguments;
        indirect.destination = plan.counts;
        indirect.destinationOffset = plan.countsOffset;
        add(indirect);
        return Made<void>{ok()};
    }
    if (!stream.hostRecords || !stream.hostCounts)
        return failure<void>("a host draw stream has no records");
    HostStream host;
    host.draw = &draw;
    host.indexCapacity = invocation.indexBuffer.size / sizeof(uint32_t);
    host.records = arena(uint64_t{stream.maxRecords} * sizeof(DrawArguments), sizeof(uint32_t));
    host.count = arena(sizeof(uint32_t), sizeof(uint32_t));
    recording_.streams.push_back(host);
    indirect.c = 1;
    indirect.sourceOffset = host.records;
    indirect.destinationOffset = host.count;
    add(indirect);
    return Made<void>{ok()};
}

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
    for (const VernonRuntimeProviderResourceReference &image : scope.reads.images) {
        auto native = nativeOf<dx::View>(image, Kind::View);
        if (native.isErr())
            return failure<Rendering>("a sampled image: " + native.error());
        transition(*native.value(), kShaderRead);
    }
    Rendering result;
    const uint32_t area[4]{scope.area[0], scope.area[1], scope.area[2], scope.area[3]};
    result.area = rectangle(area);
    for (size_t index = 0; index < scope.colors.size(); ++index) {
        const VernonRuntimeProviderColorAttachment &color = scope.colors[index];
        auto native = nativeOf<dx::View>(color.view, Kind::View);
        if (native.isErr())
            return failure<Rendering>("a color attachment: " + native.error());
        const dx::View &view = *native.value();
        if (color.location >= kMaxColors)
            return failure<Rendering>("a color attachment's location is out of range");
        if (!view.rtv.ptr)
            return failure<Rendering>("a color attachment's view has no color attachment usage");
        const bool clear = color.load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        const VernonTextureFormat format =
            index < scope.colorFormats.size() ? scope.colorFormats[index] : view.owner->vernonFormat;
        if (clear && formats::isInteger(format)) {
            if (!view.clear.ptr)
                return failure<Rendering>("an integer color attachment cannot be cleared");
            transition(view, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            flushBarriers();
            IntegerClear integer;
            integer.resource = view.owner->resource;
            integer.view = view.clear;
            integer.set = static_cast<uint32_t>(recording_.sets.size());
            integer.color = color.clear_color;
            // ClearUnorderedAccessViewUint saturates each value to the component's unsigned range, so a negative
            // value of a narrower signed format clears to its two's-complement bits instead.
            const formats::Format &entry = *formats::find(format);
            const uint32_t bits = 8 * entry.pixelBytes / entry.components();
            if (entry.componentClass == formats::ComponentClass::Sint && bits < 32)
                for (uint32_t &component : integer.color.u32)
                    component &= (1u << bits) - 1;
            integer.area = result.area;
            SetInstance descriptor;
            descriptor.resources.push_back({Write::Type::Copy, nullptr, 0, 0, view.clear});
            descriptor.resourceBase = recording_.resourceDescriptors++;
            descriptor.samplerBase = recording_.samplerDescriptors;
            recording_.sets.push_back(std::move(descriptor));
            add({Op::ClearInteger, static_cast<uint32_t>(recording_.clears.size())});
            recording_.clears.push_back(integer);
        }
        transition(view, D3D12_RESOURCE_STATE_RENDER_TARGET);
        result.colors = (std::max)(result.colors, color.location + 1);
        result.targets[color.location] = view.rtv;
        result.clearColor[color.location] = clear && !formats::isInteger(format);
        result.clears[color.location] = color.clear_color;
    }
    if (scope.depth) {
        const VernonRuntimeProviderDepthStencilAttachment &depth = *scope.depth;
        auto native = nativeOf<dx::View>(depth.view, Kind::View);
        if (native.isErr())
            return failure<Rendering>("the depth attachment: " + native.error());
        if (!native.value()->dsv.ptr)
            return failure<Rendering>("the depth attachment's view has no depth stencil attachment usage");
        transition(*native.value(), D3D12_RESOURCE_STATE_DEPTH_WRITE);
        result.depth = native.value()->dsv;
        if (depth.depth_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR)
            result.depthClears |= D3D12_CLEAR_FLAG_DEPTH;
        if (checked.value().stencil && depth.stencil_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR)
            result.depthClears |= D3D12_CLEAR_FLAG_STENCIL;
        result.clearDepth = depth.clear_depth;
        result.clearStencil = static_cast<uint8_t>(depth.clear_stencil);
    }
    return Made<Rendering>{ok(result)};
}

// Begins a render scope with every draw of passes [begin, end) planned, so that the states the draws need are set
// before it: buffers and images, and the device draw streams expanded into the slot's draw arguments.
Made<void> Builder::buildScope(size_t begin, size_t end) {
    const std::vector<PlannedPass *> &passes = planned_.passes;
    const runtime::GraphicsRenderScope &scope = planned_.scopes[passes[begin]->scope];
    std::string label;
    for (size_t at = begin; at < end; ++at)
        label += (at == begin ? "" : "+") + passes[at]->command->label;
    debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, label);
    std::vector<DrawPlan> draws;
    BufferUses buffers;
    std::vector<ImageUse> images;
    for (size_t at = begin; at < end; ++at)
        for (const PlannedDraw &draw : passes[at]->draws) {
            auto planned = planDraw(*passes[at], draw, buffers, images);
            if (planned.isErr())
                return failure<void>("pass '" + passes[at]->executable->source().node->name + "': " + planned.error());
            draws.push_back(planned.value());
        }
    for (const auto &[buffer, needed] : buffers)
        transition(buffer, needed);
    for (const ImageUse &use : images)
        transition(*use.view, use.state);
    flushBarriers();
    const bool expands =
        std::any_of(draws.begin(), draws.end(), [](const DrawPlan &plan) { return plan.records != nullptr; });
    if (expands) {
        argumentStates(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (const DrawPlan &plan : draws) {
            if (!plan.records)
                continue;
            Command expand{Op::ExpandStream};
            expand.a = plan.draw->stream.maxRecords;
            expand.source = plan.records;
            expand.sourceOffset = plan.recordsOffset;
            expand.destinationOffset = plan.arguments;
            add(expand);
        }
        add({Op::UavBarrier});
        argumentStates(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        bound_ = nullptr;
    }
    auto opened = rendering(scope);
    if (opened.isErr())
        return failure<void>("pass '" + passes[begin]->executable->source().node->name + "': " + opened.error());
    flushBarriers();
    add({Op::BeginScope, static_cast<uint32_t>(recording_.renderings.size())});
    recording_.renderings.push_back(opened.value());
    // Compute work between scopes replaces the pipeline, and the root signature with it.
    graphicsRoot_ = nullptr;
    pipeline_ = nullptr;
    pushStage_ = nullptr;
    topology_.reset();
    viewport_.reset();
    scissor_.reset();
    stencil_.reset();
    index_.reset();
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
    bound_ = nullptr;
    // What the draws wrote as UAVs is visible to what reads it next in the same state.
    add({Op::UavBarrier});
    return Made<void>{ok()};
}

Made<std::unique_ptr<Recording>>
DirectX12Device::build(const DirectX12CommandBuffer &commands, const PlannedPasses &planned,
                       const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches, bool timed) {
    auto recording = std::make_unique<Recording>();
    recording->retirer = retirer_;
    recording->timed = timed;
    auto used = internals_.used(commands.commands());
    if (used.isErr())
        return failure<std::unique_ptr<Recording>>(std::move(used).error());
    Builder builder(*this, *recording, planned, launches);
    if (auto built = builder.build(used.value()); built.isErr())
        return failure<std::unique_ptr<Recording>>(std::move(built).error());
    return Made<std::unique_ptr<Recording>>{ok(std::move(recording))};
}

// Plans a compute pass's launch, whose resources key the recording that dispatches it; the recording reads its group
// counts from the call, so only whether it dispatches at all does.
Made<void> DirectX12Device::planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
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
        return failure<void>(name + "the Stage has no DirectX 12 pipeline");
    const auto &state = ::runtimeBackendState<runtime::DirectX12PipelineState>(stage);
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
Made<void> DirectX12Device::createSlot(Recording &recording, Slot &slot) {
    ID3D12Device *device = state_.device;
    if (!slot.allocator) {
        if (auto made =
                call(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&slot.allocator)),
                     "ID3D12Device::CreateCommandAllocator");
            made.isErr())
            return made;
        if (auto made = call(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot.allocator, nullptr,
                                                       IID_PPV_ARGS(&slot.list)),
                             "ID3D12Device::CreateCommandList");
            made.isErr())
            return made;
        if (auto closed = call(slot.list->Close(), "ID3D12GraphicsCommandList::Close"); closed.isErr())
            return closed;
    }
    if (slot.arenaCapacity < recording.arenaBytes || !slot.arena) {
        if (slot.arena && slot.mapped) {
            const D3D12_RANGE written{0, 0};
            slot.arena->Unmap(0, &written);
        }
        dx::release(slot.arena);
        slot.mapped = nullptr;
        slot.arenaCapacity = 0;
        const uint64_t capacity = roundUp(recording.arenaBytes, 4096);
        auto arena = createBuffer(device, D3D12_HEAP_TYPE_UPLOAD, capacity, D3D12_RESOURCE_STATE_GENERIC_READ);
        if (arena.isErr())
            return failure<void>("cannot create a recording's arena: " + arena.error());
        slot.arena = arena.value();
        void *mapped = nullptr;
        const D3D12_RANGE none{0, 0};
        if (auto made = call(slot.arena->Map(0, &none, &mapped), "ID3D12Resource::Map"); made.isErr())
            return made;
        slot.mapped = static_cast<uint8_t *>(mapped);
        slot.arenaCapacity = capacity;
    }
    if (recording.argumentBytes && slot.argumentCapacity < recording.argumentBytes) {
        dx::release(slot.arguments);
        slot.argumentCapacity = 0;
        const uint64_t capacity = roundUp(recording.argumentBytes, 4096);
        auto arguments = createBuffer(device, D3D12_HEAP_TYPE_DEFAULT, capacity, D3D12_RESOURCE_STATE_COMMON,
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (arguments.isErr())
            return failure<void>("cannot create a recording's draw arguments: " + arguments.error());
        slot.arguments = arguments.value();
        slot.argumentCapacity = capacity;
    }
    const auto heap = [&](D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t needed, ID3D12DescriptorHeap *&made,
                          uint32_t &capacity) -> Made<void> {
        if (!needed || (made && capacity >= needed))
            return Made<void>{ok()};
        dx::release(made);
        capacity = 0;
        D3D12_DESCRIPTOR_HEAP_DESC description{};
        description.Type = type;
        description.NumDescriptors = static_cast<UINT>(roundUp(needed, 64));
        if (type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
            description.NumDescriptors = (std::min)(description.NumDescriptors, kMaxSamplerDescriptors);
        description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (auto created = call(device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&made)),
                                "ID3D12Device::CreateDescriptorHeap");
            created.isErr())
            return created;
        capacity = description.NumDescriptors;
        return Made<void>{ok()};
    };
    if (recording.samplerDescriptors > kMaxSamplerDescriptors)
        return failure<void>("the passes bind more samplers than a DirectX 12 sampler heap holds");
    if (auto made = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, recording.resourceDescriptors, slot.resources,
                         slot.resourceCapacity);
        made.isErr())
        return made;
    if (auto made =
            heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, recording.samplerDescriptors, slot.samplers, slot.samplerCapacity);
        made.isErr())
        return made;
    const UINT resourceStep = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT samplerStep = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    const D3D12_GPU_VIRTUAL_ADDRESS arenaAddress = slot.arena->GetGPUVirtualAddress();
    for (const SetInstance &set : recording.sets) {
        for (size_t index = 0; index < set.resources.size(); ++index) {
            const Write &write = set.resources[index];
            const D3D12_CPU_DESCRIPTOR_HANDLE target{slot.resources->GetCPUDescriptorHandleForHeapStart().ptr +
                                                     SIZE_T{set.resourceBase + index} * resourceStep};
            ID3D12Resource *buffer = write.buffer ? write.buffer : slot.arena;
            const auto words = static_cast<UINT>((std::max<uint64_t>)(1, (write.size + 3) / 4));
            switch (write.type) {
            case Write::Type::ConstantBuffer: {
                const D3D12_CONSTANT_BUFFER_VIEW_DESC view{arenaAddress + write.offset,
                                                           static_cast<UINT>(align(write.size, kConstantAlignment))};
                device->CreateConstantBufferView(&view, target);
                break;
            }
            case Write::Type::RawRead: {
                D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                view.Format = DXGI_FORMAT_R32_TYPELESS;
                view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                view.Buffer.FirstElement = write.offset / 4;
                view.Buffer.NumElements = words;
                view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
                device->CreateShaderResourceView(buffer, &view, target);
                break;
            }
            case Write::Type::RawWrite: {
                D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                view.Format = DXGI_FORMAT_R32_TYPELESS;
                view.Buffer.FirstElement = write.offset / 4;
                view.Buffer.NumElements = words;
                view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                device->CreateUnorderedAccessView(buffer, nullptr, &view, target);
                break;
            }
            case Write::Type::Copy:
                device->CopyDescriptorsSimple(1, target, write.source, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                break;
            }
        }
        for (size_t index = 0; index < set.samplers.size(); ++index) {
            const D3D12_CPU_DESCRIPTOR_HANDLE target{slot.samplers->GetCPUDescriptorHandleForHeapStart().ptr +
                                                     SIZE_T{set.samplerBase + index} * samplerStep};
            device->CopyDescriptorsSimple(1, target, set.samplers[index], D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        }
    }
    if (!recording.timers.empty() && state_.timestampFrequency) {
        auto queries = std::make_shared<Queries>();
        const auto count = static_cast<UINT>(recording.timers.size() * 2);
        D3D12_QUERY_HEAP_DESC description{};
        description.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        description.Count = count;
        if (auto made = call(device->CreateQueryHeap(&description, IID_PPV_ARGS(&queries->heap)),
                             "ID3D12Device::CreateQueryHeap");
            made.isErr())
            return made;
        auto readback = createBuffer(device, D3D12_HEAP_TYPE_READBACK, uint64_t{count} * sizeof(uint64_t),
                                     D3D12_RESOURCE_STATE_COPY_DEST);
        if (readback.isErr())
            return failure<void>(readback.error());
        queries->readback = readback.value();
        void *mapped = nullptr;
        const D3D12_RANGE read{0, SIZE_T{count} * sizeof(uint64_t)};
        if (auto made = call(queries->readback->Map(0, &read, &mapped), "ID3D12Resource::Map"); made.isErr())
            return made;
        queries->ticks = static_cast<const uint64_t *>(mapped);
        slot.queries = std::move(queries);
    }
    return Made<void>{ok()};
}

// A slot whose last submission completed, after reading its timings; a new one while the recording has fewer than
// kSlots, else the oldest once it completes.
Made<Slot *> DirectX12Device::acquire(Recording &recording) {
    const auto reuse = [&](Slot &slot) {
        if (slot.timing) {
            std::lock_guard<std::mutex> lock(mutex_);
            capture(*slot.timing);
        }
        return Made<Slot *>{ok(&slot)};
    };
    for (const std::unique_ptr<Slot> &slot : recording.slots) {
        if (!slot->serial)
            return reuse(*slot);
        auto completed = valueCompleted(state_, slot->serial);
        if (completed.isErr())
            return failure<Slot *>("the device was lost: " + completed.error());
        if (completed.value())
            return reuse(*slot);
    }
    if (recording.slots.size() < kSlots) {
        std::unique_ptr<Slot> slot = retirer_->pooled();
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
    return reuse(*oldest);
}

// Writes the bytes a call varies: the sites' inline values and the dispatches' group counts.
Made<void> DirectX12Device::fill(Recording &recording, Slot &slot, const PlannedPasses &planned,
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
            const auto &state = ::runtimeBackendState<runtime::DirectX12PipelineState>(stage);
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
    if (!slot.recorded)
        for (const auto &[offset, bytes] : recording.fixed)
            std::memcpy(slot.mapped + offset, bytes.data(), bytes.size());
    for (const auto &[pass, offset] : recording.dispatches) {
        const VernonLaunchSize &grid = launches[pass]->grid;
        const D3D12_DISPATCH_ARGUMENTS groups{grid.x, grid.y, grid.z};
        std::memcpy(slot.mapped + offset, &groups, sizeof(groups));
    }
    for (const PushBlock &block : recording.pushBlocks)
        for (uint32_t index = block.first; index < block.first + block.count; ++index) {
            const auto &[set, instance] = recording.pushSets[index];
            const uint8_t *data = recording.push.data() + recording.sets[instance].pushOffset;
            for (const PushPlacement &placement : block.stage->sets[set].placements)
                std::memcpy(slot.mapped + block.offsets[placement.stage] + placement.offset,
                            data + placement.dataOffset, placement.size);
        }
    for (const HostStream &host : recording.streams) {
        const runtime::GraphicsDrawStream &stream = host.draw->stream;
        const VernonRuntimeProviderDrawRecord *records = stream.hostRecords + stream.firstRecord;
        const uint32_t count = (std::min)(stream.hostCounts[stream.countIndex], stream.maxRecords);
        auto *arguments = reinterpret_cast<DrawArguments *>(slot.mapped + host.records);
        // Checked as the call reads them: O(drawn records).
        for (uint32_t record = 0; record < count; ++record) {
            const VernonRuntimeProviderDrawRecord &source = records[record];
            if (uint64_t{source.first_index} + source.index_count > host.indexCapacity)
                return failure<void>("a host draw stream's record exceeds its index buffer");
            arguments[record] = {source.base_vertex,
                                 source.first_instance,
                                 {source.index_count, source.instance_count, source.first_index, source.base_vertex,
                                  source.first_instance}};
        }
        std::memcpy(slot.mapped + host.count, &count, sizeof(count));
    }
    return Made<void>{ok()};
}

Made<void> DirectX12Device::recordCommands(Recording &recording, Slot &slot) {
    if (auto reset = call(slot.allocator->Reset(), "ID3D12CommandAllocator::Reset"); reset.isErr())
        return reset;
    if (auto reset = call(slot.list->Reset(slot.allocator, nullptr), "ID3D12GraphicsCommandList::Reset"); reset.isErr())
        return reset;
    ID3D12GraphicsCommandList *list = slot.list;
    const UINT resourceStep = state_.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT samplerStep = state_.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    for (const Command &op : recording.commands) {
        switch (op.op) {
        case Op::Heaps: {
            std::array<ID3D12DescriptorHeap *, 2> heaps{};
            UINT count = 0;
            if (recording.resourceDescriptors)
                heaps[count++] = slot.resources;
            if (recording.samplerDescriptors)
                heaps[count++] = slot.samplers;
            if (count)
                list->SetDescriptorHeaps(count, heaps.data());
            break;
        }
        case Op::UavBarrier: {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            list->ResourceBarrier(1, &barrier);
            break;
        }
        case Op::Barriers:
            list->ResourceBarrier(op.b, recording.barriers.data() + op.a);
            break;
        case Op::CopyBuffer:
            list->CopyBufferRegion(op.destination, op.destinationOffset, op.source, op.sourceOffset, op.size);
            break;
        case Op::CopyRegions:
            for (uint32_t index = op.a; index < op.a + op.b; ++index) {
                const CopyRegion &region = recording.copies[index];
                list->CopyBufferRegion(op.destination, region.destination, op.source, region.source, region.size);
            }
            break;
        case Op::Timestamp:
            if (slot.queries)
                list->EndQuery(slot.queries->heap, D3D12_QUERY_TYPE_TIMESTAMP, op.a);
            break;
        case Op::DebugPush:
        case Op::DebugMarker: {
            const std::string &label = recording.labels[op.a];
            const auto size = static_cast<UINT>(label.size() + 1);
            if (op.op == Op::DebugPush)
                list->BeginEvent(kAnsiEvent, label.c_str(), size);
            else
                list->SetMarker(kAnsiEvent, label.c_str(), size);
            break;
        }
        case Op::DebugPop:
            list->EndEvent();
            break;
        case Op::BindCompute:
            list->SetComputeRootSignature(op.stage->root);
            list->SetPipelineState(op.stage->compute);
            break;
        case Op::ComputeTable: {
            D3D12_GPU_DESCRIPTOR_HANDLE table =
                (op.b ? slot.samplers : slot.resources)->GetGPUDescriptorHandleForHeapStart();
            table.ptr += UINT64{op.c} * (op.b ? samplerStep : resourceStep);
            list->SetComputeRootDescriptorTable(op.a, table);
            break;
        }
        case Op::Dispatch:
            list->ExecuteIndirect(dispatchSignature_, 1, slot.arena, op.sourceOffset, nullptr, 0);
            break;
        case Op::ExpandStream:
            list->SetComputeRootSignature(streamRoot_);
            list->SetPipelineState(streamPipeline_);
            list->SetComputeRoot32BitConstant(0, op.a, 0);
            list->SetComputeRootShaderResourceView(1, op.source->GetGPUVirtualAddress() + op.sourceOffset);
            list->SetComputeRootUnorderedAccessView(2, slot.arguments->GetGPUVirtualAddress() + op.destinationOffset);
            list->Dispatch((op.a + 63) / 64, 1, 1);
            break;
        case Op::ArgumentStates: {
            const D3D12_RESOURCE_BARRIER barrier =
                transitionBarrier(slot.arguments, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                  static_cast<D3D12_RESOURCE_STATES>(op.a), static_cast<D3D12_RESOURCE_STATES>(op.b));
            list->ResourceBarrier(1, &barrier);
            break;
        }
        case Op::ClearInteger: {
            const IntegerClear &clear = recording.clears[op.a];
            D3D12_GPU_DESCRIPTOR_HANDLE visible = slot.resources->GetGPUDescriptorHandleForHeapStart();
            visible.ptr += UINT64{recording.sets[clear.set].resourceBase} * resourceStep;
            const bool whole = clear.area.right <= clear.area.left || clear.area.bottom <= clear.area.top;
            list->ClearUnorderedAccessViewUint(visible, clear.view, clear.resource, clear.color.u32, whole ? 0 : 1,
                                               whole ? nullptr : &clear.area);
            break;
        }
        case Op::BeginScope: {
            const Rendering &rendering = recording.renderings[op.a];
            std::array<D3D12_CPU_DESCRIPTOR_HANDLE, kMaxColors> targets{};
            for (uint32_t location = 0; location < rendering.colors; ++location)
                targets[location] = rendering.targets[location].ptr ? rendering.targets[location] : nullTarget_;
            list->OMSetRenderTargets(rendering.colors, rendering.colors ? targets.data() : nullptr, FALSE,
                                     rendering.depth.ptr ? &rendering.depth : nullptr);
            const bool whole =
                rendering.area.right <= rendering.area.left || rendering.area.bottom <= rendering.area.top;
            for (uint32_t location = 0; location < rendering.colors; ++location)
                if (rendering.clearColor[location])
                    list->ClearRenderTargetView(targets[location], rendering.clears[location].f32, whole ? 0 : 1,
                                                whole ? nullptr : &rendering.area);
            if (rendering.depthClears)
                list->ClearDepthStencilView(rendering.depth, rendering.depthClears, rendering.clearDepth,
                                            rendering.clearStencil, whole ? 0 : 1, whole ? nullptr : &rendering.area);
            break;
        }
        case Op::BindGraphics:
            if (op.stage)
                list->SetGraphicsRootSignature(op.stage->root);
            list->SetPipelineState(op.pipeline);
            break;
        case Op::GraphicsTable: {
            D3D12_GPU_DESCRIPTOR_HANDLE table =
                (op.b ? slot.samplers : slot.resources)->GetGPUDescriptorHandleForHeapStart();
            table.ptr += UINT64{op.c} * (op.b ? samplerStep : resourceStep);
            list->SetGraphicsRootDescriptorTable(op.a, table);
            break;
        }
        case Op::Push: {
            const PushBlock &block = recording.pushBlocks[op.a];
            for (uint32_t index = 0; index < 2; ++index)
                if (block.stage->pushParameters[index] != kNone)
                    list->SetGraphicsRootConstantBufferView(block.stage->pushParameters[index],
                                                            slot.arena->GetGPUVirtualAddress() + block.offsets[index]);
            break;
        }
        case Op::VertexBuffers:
            list->IASetVertexBuffers(0, op.b, recording.vertexViews.data() + op.a);
            break;
        case Op::IndexBuffer: {
            const D3D12_INDEX_BUFFER_VIEW view{op.sourceOffset, static_cast<UINT>(op.size), DXGI_FORMAT_R32_UINT};
            list->IASetIndexBuffer(&view);
            break;
        }
        case Op::Topology:
            list->IASetPrimitiveTopology(op.a == 1   ? D3D_PRIMITIVE_TOPOLOGY_LINELIST
                                         : op.a == 2 ? D3D_PRIMITIVE_TOPOLOGY_POINTLIST
                                                     : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            break;
        case Op::Viewport: {
            const D3D12_VIEWPORT viewport{static_cast<float>(op.a),
                                          static_cast<float>(op.b),
                                          static_cast<float>(op.c),
                                          static_cast<float>(op.d),
                                          0,
                                          1};
            list->RSSetViewports(1, &viewport);
            break;
        }
        case Op::Scissor: {
            const uint32_t area[4]{op.a, op.b, op.c, op.d};
            const D3D12_RECT scissor = rectangle(area);
            list->RSSetScissorRects(1, &scissor);
            break;
        }
        case Op::Stencil:
            list->OMSetStencilRef(op.a);
            break;
        case Op::Draw:
        case Op::DrawIndexed: {
            const uint32_t offsets[2]{op.op == Op::DrawIndexed ? op.d : 0, op.e};
            list->SetGraphicsRoot32BitConstants(op.stage->offsetsParameter, 2, offsets, 0);
            if (op.op == Op::DrawIndexed)
                list->DrawIndexedInstanced(op.a, op.b, op.c, static_cast<INT>(op.d), op.e);
            else
                list->DrawInstanced(op.a, op.b, 0, op.e);
            break;
        }
        case Op::DrawIndirect:
            list->ExecuteIndirect(op.stage->drawSignature, op.a, op.b ? slot.arguments : slot.arena, op.sourceOffset,
                                  op.c ? (op.destination ? op.destination : slot.arena) : nullptr,
                                  op.destinationOffset);
            break;
        }
    }
    if (slot.queries)
        list->ResolveQueryData(slot.queries->heap, D3D12_QUERY_TYPE_TIMESTAMP, 0,
                               static_cast<UINT>(recording.timers.size() * 2), slot.queries->readback, 0);
    if (auto closed = call(list->Close(), "ID3D12GraphicsCommandList::Close"); closed.isErr())
        return closed;
    slot.recorded = true;
    return Made<void>{ok()};
}

// Executes the slot's commands after moving the images they use from their tracked states to the ones the commands
// begin in, and tracks the states the commands leave them in.
Made<uint64_t> DirectX12Device::execute(const Recording &recording, Slot &slot) {
    std::lock_guard<std::mutex> lock(state_.mutex);
    std::vector<D3D12_RESOURCE_BARRIER> prologue;
    for (const auto &[image, use] : recording.states)
        for (size_t subresource = 0; subresource < use.initial.size(); ++subresource)
            if (use.initial[subresource] != kUnused && image->states[subresource] != use.initial[subresource])
                prologue.push_back(transitionBarrier(image->resource, static_cast<UINT>(subresource),
                                                     image->states[subresource], use.initial[subresource]));
    std::string error;
    uint64_t value = 0;
    if (!prologue.empty()) {
        dx::CommandFrame *frame = nullptr;
        if (!state_.beginCommands(frame, error))
            return Made<uint64_t>{err(std::move(error))};
        frame->commands->ResourceBarrier(static_cast<UINT>(prologue.size()), prologue.data());
        if (!state_.submitCommands(frame, value, error))
            return Made<uint64_t>{err(std::move(error))};
    }
    ID3D12CommandList *lists[] = {slot.list};
    if (!state_.execute(lists, 1, value, error))
        return Made<uint64_t>{err(std::move(error))};
    for (const auto &[image, use] : recording.states)
        for (size_t subresource = 0; subresource < use.current.size(); ++subresource)
            if (use.current[subresource] != kUnused)
                image->states[subresource] = use.current[subresource];
    return Made<uint64_t>{ok(value)};
}

Made<TimelinePoint> DirectX12Device::submit(const CommandBuffer &commands, const uint8_t *record, size_t) {
    const auto &buffer = static_cast<const DirectX12CommandBuffer &>(commands);
    retirer_->collect();
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
    const bool timed = buffer.timed();
    key.push_back(timed ? 1 : 0);
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
        auto built = build(buffer, planned, launches, timed);
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
    if (!slot.recorded)
        if (auto recorded = recordCommands(*recording, slot); recorded.isErr()) {
            slot.recorded = false;
            return failure<TimelinePoint>(std::move(recorded).error());
        }
    if (anyUndefined(recording->reads.data(), recording->reads.size()))
        return failure<TimelinePoint>("a pass reads an image whose contents are undefined");
    auto submitted = execute(*recording, slot);
    if (submitted.isErr())
        return failure<TimelinePoint>("cannot submit the passes: " + submitted.error());
    const uint64_t value = submitted.value();
    retirer_->submitted(value);
    slot.serial = value;
    forgetUndefined(recording->writes.data(), recording->writes.size());
    std::shared_ptr<TimingCapture> timing;
    if (timed) {
        timing = std::make_shared<TimingCapture>();
        timing->queries = slot.queries;
        timing->scopes = recording->timers;
    }
    slot.timing = timing;
    uint64_t draws = 0;
    for (const PlannedPass *pass : planned.passes)
        draws += pass->draws.size();
    buffer.setTelemetry(planned.scopes.size(), draws);
    std::lock_guard<std::mutex> lock(mutex_);
    const TimelinePoint point = next_++;
    pending_.push_back({point, value, std::move(timing)});
    return Made<TimelinePoint>{ok(point)};
}

} // namespace

std::unique_ptr<Device> createDirectX12Device(VernonRuntimeContext &context) {
    if (context.backend != VERNON_RUNTIME_DIRECTX12)
        return nullptr;
    const std::shared_ptr<dx::DeviceState> &state = runtime::directX12State(context).device;
    if (!state)
        return nullptr;
    auto device = std::make_unique<DirectX12Device>(context, std::make_shared<Retirer>(state));
    if (device->initialize().isErr())
        return nullptr;
    return device;
}

} // namespace vernon::hal
