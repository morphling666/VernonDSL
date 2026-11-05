#include "hal/opengl/opengl_device.h"
#include "hal/native_buffers.h"
#include "hal/native_images.h"
#include "hal/opengl/opengl_resources.h"

#include "VernonFormats.hpp"
#include "VernonSchedule.h"
#include "rhi/opengl_access.h"
#include "rhi/rhi_internal.h"
#include "runtime/backend_opengl.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/dispatch_contract.h"
#include "runtime/interface_names.h"
#include "runtime/prepared_binding_plan.h"
#include "runtime/prepared_graphics_draw.h"
#include "runtime/runtime_state.h"
#include "runtime/schedule/pass_plan.h"
#include "runtime/vertex_attribute_capabilities.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
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

namespace gl = rhi::opengl;
using runtime::PreparedComputeBindingPlan;
using runtime::PreparedGraphicsBindingPlan;

constexpr uint32_t kNone = UINT32_MAX;
constexpr size_t kMaxSets = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;
constexpr uint32_t kMaxColors = VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS;
constexpr uint32_t kMaxVertexBindings = VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS;
// Vertex attribute locations are enabled by a 64-bit mask.
constexpr uint32_t kMaxLocations = 64;
// Recordings one command buffer keeps: a call reuses one whose passes, attachments and timing are its own, such as
// one per swapchain image.
constexpr size_t kRecordings = 8;
// The arenas of one recording, used in turn so that a call rarely rewrites bytes the previous calls' commands read;
// OpenGL orders the write after them either way, so no call waits for one.
constexpr size_t kSlots = 3;
constexpr uint64_t kRecordBytes = sizeof(VernonRuntimeProviderDrawRecord);
// Buffer names that a recording's commands name before a call picks their buffers.
constexpr gl::Uint kArena = UINT32_MAX;
constexpr gl::Uint kZeros = UINT32_MAX - 1;
constexpr gl::Enum kPolygonOffsetFill = 0x8037;
constexpr gl::Enum kDynamicDraw = 0x88E8;
constexpr unsigned kAllBarriers = 0xFFFFFFFFu;
constexpr gl::Enum kTopologies[]{gl::kTriangles, gl::kLines, gl::kPoints};

template <typename T> Made<T> failure(std::string message) { return Made<T>{err(std::move(message))}; }

uint64_t align(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

uint64_t roundUp(uint64_t value, uint64_t minimum) {
    uint64_t result = minimum;
    while (result < value)
        result *= 2;
    return result;
}

uint32_t floatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bitsFloat(uint32_t bits) {
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t blendFactor(uint32_t value) {
    constexpr uint32_t factors[]{0, 1, 0x0300, 0x0301, 0x0306, 0x0307, 0x0302, 0x0303, 0x0304, 0x0305};
    return value < std::size(factors) ? factors[value] : 0;
}

uint32_t blendOperation(uint32_t value) {
    constexpr uint32_t operations[]{0x8006, 0x800A, 0x800B, 0x8007, 0x8008};
    return value < std::size(operations) ? operations[value] : 0;
}

uint32_t stencilOperation(uint32_t value) {
    constexpr uint32_t operations[]{0x1E00, 0, 0x1E01, 0x1E02, 0x1E03, 0x150A, 0x8507, 0x8508};
    return value < std::size(operations) ? operations[value] : 0;
}

// What the device's context can do, read once.
struct Caps {
    bool desktop{};
    bool compute{};
    bool separateFormats{};
    bool deviceStreams{};
    bool countedStreams{};
    bool labels{};
    bool timers{};
    uint64_t uniformAlignment{256};
    uint64_t storageAlignment{256};
    uint32_t maxRelativeOffset{};
    uint32_t maxLocations{};
};

struct Slot;

// What the device, the Stages' programs and the recordings share, any of which may be destroyed first: the context's
// state and mutex, a lease that keeps the RHI device alive, and the arenas of dropped recordings for later ones.
class Context {
public:
    Context(VernonRhiDevice device, gl::DeviceAccess access, ChildLease lease)
        : device_(device), access_(access), lease_(std::move(lease)) {}
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    ~Context();

    VernonRhiDevice device() const { return device_; }
    gl::DeviceState &state() const { return *access_.state; }
    gl::Driver &driver() const { return access_.state->driver; }

    // Holds the context's mutex with the context current.
    std::unique_lock<std::mutex> lock() const {
        std::unique_lock<std::mutex> held(*access_.mutex);
        access_.state->makeCurrent();
        return held;
    }

    // Keeps an idle slot, or destroys it when enough are kept. Called without the context's mutex.
    void pool(std::unique_ptr<Slot> slot);
    std::unique_ptr<Slot> pooled();

private:
    static constexpr size_t kPooledSlots = 32;

    VernonRhiDevice device_;
    gl::DeviceAccess access_;
    ChildLease lease_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

// The arena of one recording's calls: the bytes that the commands read from a buffer, as last uploaded.
struct Slot {
    gl::Uint arena{};
    uint64_t capacity{};
    std::vector<uint8_t> uploaded;
};

Context::~Context() {
    if (slots_.empty())
        return;
    auto held = lock();
    for (const std::unique_ptr<Slot> &slot : slots_)
        if (slot->arena)
            driver().deleteBuffers(1, &slot->arena);
}

void Context::pool(std::unique_ptr<Slot> slot) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slots_.size() < kPooledSlots) {
            slots_.push_back(std::move(slot));
            return;
        }
    }
    auto held = lock();
    if (slot->arena)
        driver().deleteBuffers(1, &slot->arena);
}

std::unique_ptr<Slot> Context::pooled() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slots_.empty())
        return nullptr;
    std::unique_ptr<Slot> slot = std::move(slots_.back());
    slots_.pop_back();
    return slot;
}

// How an entry reaches the program: context bindings are shared by every program, named uniforms belong to one.
enum class Application : uint8_t { UniformBuffer, StorageBuffer, SampledImage, Sampler, StorageImage, NamedUniform };

std::optional<Application> entryApplication(const VernonRuntimeProviderBindingLayoutEntry &source) {
    const bool nativeNumericType = source.numeric_type == VERNON_RUNTIME_PROVIDER_F32 ||
                                   source.numeric_type == VERNON_RUNTIME_PROVIDER_I32 ||
                                   source.numeric_type == VERNON_RUNTIME_PROVIDER_U32;
    const bool nativeUniformShape =
        source.vector_count != 0 &&
        ((source.vector_count == 1 && source.element_count >= 1 && source.element_count <= 4) ||
         (source.numeric_type == VERNON_RUNTIME_PROVIDER_F32 && source.vector_count >= 2 && source.vector_count <= 4 &&
          source.element_count % source.vector_count == 0 && source.element_count / source.vector_count >= 2 &&
          source.element_count / source.vector_count <= 4));
    const bool compute = source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
    const bool named = source.name.data && source.name.size != 0;
    const bool bound = source.binding != UINT32_MAX;
    switch (source.kind) {
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
        if (compute && bound && source.element_size != 0)
            return Application::StorageBuffer;
        if (!compute && source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && named &&
            nativeNumericType && nativeUniformShape && source.element_size == source.element_count * sizeof(uint32_t))
            return Application::NamedUniform;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
        if ((compute || source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
             source.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
            (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE ||
             source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) &&
            bound && source.element_size != 0)
            return Application::StorageBuffer;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && bound && source.element_size != 0)
            return Application::UniformBuffer;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE && bound && (compute || named))
            return Application::SampledImage;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE && bound && source.access != 0)
            return Application::StorageImage;
        return std::nullopt;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        if (source.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE && bound)
            return Application::Sampler;
        return std::nullopt;
    }
    return std::nullopt;
}

bool packedBytes(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE || entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
           (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
            entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM);
}

// How one prepared layout slot of a Stage binds. Packed bytes of a buffer entry live in the arena; a named uniform's
// bytes are the recording's values, which the program holds once uploaded.
struct SlotPlan {
    Application application{};
    VernonRuntimeProviderBindingKind providerKind{};
    uint32_t set{};
    uint32_t binding{};
    uint32_t size{};
    bool packed{};
    // Whether the program reads the entry's context binding: an inactive uniform block, sampled image or sampler unit
    // is left as the context holds it.
    bool reads{true};
    gl::Int location{-1};
    uint32_t valueCount{};
    uint32_t columnCount{};
    uint32_t numericType{};
    gl::Enum imageAccess{};
    VernonTextureFormat storageFormat{};
    std::string name;
};

// Binding set `set` of a Stage: the slots in it, and what decides its values as PipelineGroups groups them.
struct SetPlan {
    std::vector<uint32_t> slots;
    std::vector<uint32_t> parameters;
    bool command{};
};

struct VertexAttribute {
    VernonRuntimeProviderVertexAttribute layout{};
    uint32_t divisor{};
};

// The HAL's program of one Stage, which lives in its OpenGLPipelineState, and what the device's calls left in it.
class StageObjects {
public:
    explicit StageObjects(std::shared_ptr<Context> context) : context_(std::move(context)) {}
    StageObjects(const StageObjects &) = delete;
    StageObjects &operator=(const StageObjects &) = delete;

    ~StageObjects() {
        if (!program)
            return;
        auto held = context_->lock();
        context_->state().destroyProgram(program);
    }

    bool graphics{};
    gl::Uint program{};
    std::vector<SlotPlan> slots;
    std::vector<SetPlan> sets;
    // SPIRV-Cross adds the first instance to gl_InstanceID through this uniform when the driver lacks
    // GL_ARB_shader_draw_parameters; -1 when the program reads gl_BaseInstanceARB or no instance index.
    gl::Int baseInstanceLocation{-1};
    std::vector<VertexAttribute> attributes;
    // By vertex binding; zero for a per-vertex binding.
    std::array<uint32_t, kMaxVertexBindings> divisors{};
    uint32_t bindingMask{};
    uint64_t attributeMask{};
    // Whether separate attribute formats can specify every attribute: their relative offsets are bounded.
    bool separate{};
    // Guarded by the context's mutex. Uniform values are program state that no other code changes, so the bytes each
    // named uniform holds and the base instance stay valid across calls.
    std::vector<std::vector<uint8_t>> uniforms;
    std::optional<uint32_t> baseInstance;

private:
    std::shared_ptr<Context> context_;
};

enum class Op : uint8_t {
    UseProgram,
    Enable,
    Disable,
    Enablei,
    Disablei,
    CullFace,
    FrontFace,
    PolygonOffset,
    DepthFunc,
    DepthMask,
    StencilFunc,
    StencilOp,
    StencilMask,
    BlendFunc,
    BlendFunci,
    BlendEquation,
    BlendEquationi,
    ColorMask,
    ColorMaski,
    Viewport,
    Scissor,
    BindFramebuffer,
    ClearColor,
    ClearDepthStencil,
    Invalidate,
    BindVertexArray,
    BindBuffer,
    BindBufferRange,
    ActiveTexture,
    BindTexture,
    BindSampler,
    BindImage,
    AttribMask,
    AttribFormat,
    AttribBinding,
    BindingDivisor,
    VertexBuffer,
    AttribPointer,
    AttribDivisor,
    Uniform,
    BaseInstance,
    DrawArrays,
    DrawElements,
    MultiDraw,
    MultiDrawCount,
    HostDraws,
    Dispatch,
    Barrier,
    CopyBuffer,
    Timestamp,
    DebugPush,
    DebugPop,
    DebugMarker,
};

// One OpenGL call of a recording; its arguments depend on `op`. Buffer names may be kArena or kZeros.
struct Command {
    Op op{};
    uint32_t a{};
    uint32_t b{};
    uint32_t c{};
    uint32_t d{};
    uint32_t e{};
    uint32_t f{};
    uint64_t offset{};
    uint64_t size{};
    // Where a CopyBuffer reads.
    uint64_t readOffset{};
    StageObjects *stage{};
};

// One binding of a set instance.
struct Binding {
    uint32_t slot{};
    gl::Uint name{};
    uint64_t offset{};
    uint64_t range{};
    gl::Enum target{};
    gl::Enum format{};
    bool layered{};
    bool transpose{};
};

// A set instance of a recording: one per binding group as PipelineGroups finds it, so draws whose groups' owners agree
// share it. A named uniform's binding holds the offset of its bytes in the recording's values.
struct SetInstance {
    std::vector<Binding> bindings;
};

// Inline bytes a call fills again: of a graphics draw's binding plan, or of compute pass `pass`'s launch; in the
// recording's values for a named uniform, else in its arena.
struct Site {
    const PlannedDraw *draw{};
    const PreparedGraphicsBindingPlan *graphics{};
    uint32_t pass{kNone};
    uint32_t slot{};
    uint32_t size{};
    uint64_t offset{};
    bool values{};
};

// A per-instance vertex binding that a host stream's draws without base instances move to each record's first
// instance.
struct InstancedBinding {
    uint32_t binding{};
    uint32_t divisor{};
    gl::Uint name{};
    uint64_t offset{};
    uint32_t stride{};
};

// The records of a host draw stream. A multi-draw stream copies them into the arena, with an indirect count when
// `counted`, else zeroing the records past the count; the others draw them one by one as the call runs.
struct HostStream {
    const PlannedDraw *draw{};
    StageObjects *stage{};
    uint64_t records{};
    uint64_t count{};
    uint64_t indexCapacity{};
    uint64_t indexOffset{};
    gl::Enum topology{};
    bool indirect{};
    bool counted{};
    bool baseInstances{};
    bool separate{};
    std::vector<InstancedBinding> instanced;
};

// The scope timings of one timed submission and the timestamp queries it wrote, read once it completed.
struct TimingCapture {
    std::vector<gl::Uint> queries;
    std::vector<ScopeTiming> scopes;
};

// The OpenGL calls of one call's passes, as planned: a later call with the same key, the same plans of the same passes
// on the same attachments, runs them again after it wrote the bytes that vary.
struct Recording {
    std::shared_ptr<Context> context;
    std::vector<uint64_t> key;
    std::vector<uint64_t> plans;
    std::vector<std::shared_ptr<StageObjects>> stages;
    std::vector<Command> commands;
    std::vector<std::string> labels;
    std::vector<VernonClearColor> clears;
    std::vector<gl::Enum> discards;
    std::vector<SetInstance> sets;
    std::vector<Site> sites;
    // The arena's bytes, which a call uploads when they differ from what its slot holds.
    std::vector<uint8_t> arena;
    std::vector<uint8_t> values;
    // The compute pass and group counts of each dispatch.
    std::vector<uint32_t> dispatchPasses;
    std::vector<std::array<uint32_t, 3>> groups;
    std::vector<HostStream> streams;
    // Image keys the commands read before writing them, and the attachments they write.
    std::vector<uint64_t> reads;
    std::vector<uint64_t> writes;
    bool timed{};
    std::vector<ScopeTiming> timers;
    uint32_t queries{};
    std::vector<std::unique_ptr<Slot>> slots;
    size_t nextSlot{};

    Recording() = default;
    Recording(const Recording &) = delete;
    Recording &operator=(const Recording &) = delete;

    ~Recording() {
        for (std::unique_ptr<Slot> &slot : slots)
            context->pool(std::move(slot));
    }
};

class OpenGLCommandBuffer final : public PassCommandBuffer {
public:
    using PassCommandBuffer::PassCommandBuffer;
    // Guarded by the pass cache's mutex; the most recently used last.
    mutable std::deque<std::unique_ptr<Recording>> recordings;
};

// The OpenGL state that the commands of a recording leave, as it builds them: each entry holds what its last command
// set. Other OpenGL code runs between calls, so nothing is known as a recording begins.
class Tracker {
public:
    using Value = std::array<uint64_t, 4>;

    enum Kind : uint64_t {
        Program = 1,
        Capability,
        CullFace,
        FrontFace,
        PolygonOffset,
        DepthFunc,
        DepthMask,
        StencilFunc,
        StencilOp,
        StencilMask,
        // By color location; the global calls set every location.
        BlendEnable,
        BlendFunc,
        BlendEquation,
        ColorMask,
        Viewport,
        Scissor,
        Framebuffer,
        VertexArray,
        ActiveTexture,
        Texture,
        Sampler,
        Image,
        IndexedBuffer,
        BufferTarget,
        AttribMask,
        AttribFormat,
        AttribBinding,
        BindingDivisor,
        VertexBuffer,
        AttribPointer,
        AttribDivisor,
    };

    static uint64_t id(Kind kind, uint64_t index) { return (static_cast<uint64_t>(kind) << 40) | index; }

    // Whether `value` differs from what `state` holds, which it then holds.
    bool change(Kind kind, uint64_t index, const Value &value) {
        auto [found, inserted] = values_.try_emplace(id(kind, index), value);
        if (inserted)
            return true;
        if (found->second == value)
            return false;
        found->second = value;
        return true;
    }
    bool holds(Kind kind, uint64_t index, const Value &value) const {
        const auto found = values_.find(id(kind, index));
        return found != values_.end() && found->second == value;
    }
    void forget(Kind kind, uint64_t index) { values_.erase(id(kind, index)); }

private:
    std::unordered_map<uint64_t, Value> values_;
};

// `kind` is the reference's identity kind: an RHI key and a HAL buffer serial may be equal.
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

struct ImageInfo {
    gl::Uint name{};
    VernonTextureDimension dimension{};
    VernonTextureFormat format{};
};

// The reference kinds of hal/native_buffers.h an OpenGL device resolves.
enum class Kind : uint64_t {
    Buffer = kNativeBufferReference,
    View = kNativeViewReference,
    Sampler = kNativeSamplerReference,
};

// glInvalidateFramebuffer discards whole attachments, so a store discard applies only when the render area covers the
// attachment.
bool renderAreaCovers(const NativeView &view, const runtime::GraphicsRenderScope &scope) {
    const uint32_t mip = view.view.subresources.base_mip_level;
    return scope.width == std::max(view.image.extent.width >> mip, 1u) &&
           scope.height == std::max(view.image.extent.height >> mip, 1u);
}

// Attaches mip 0 and layer 0 of the view's texture: a texture view starts at the view's subresources, and an identity
// view reaches rendering only when it has one mip and one layer.
void attachView(gl::Driver &driver, gl::Enum attachment, const OpenGLViewTexture *view) {
    if (!view)
        driver.framebufferTexture2D(gl::kFramebuffer, attachment, gl::kTexture2D, 0, 0);
    else if (view->target == gl::kTexture2D)
        driver.framebufferTexture2D(gl::kFramebuffer, attachment, gl::kTexture2D, view->name, 0);
    else if (view->target == gl::kTextureCubeMap)
        driver.framebufferTexture2D(gl::kFramebuffer, attachment, gl::kTextureCubeMapPositiveX, view->name, 0);
    else
        driver.framebufferTextureLayer(gl::kFramebuffer, attachment, view->name, 0, 0);
}

void uploadNamedUniform(gl::Driver &driver, gl::Int location, const SlotPlan &slot, bool transpose,
                        const uint8_t *bytes) {
    const uint32_t count = slot.valueCount;
    if (slot.columnCount <= 1) {
        if (slot.numericType == VERNON_RUNTIME_PROVIDER_I32) {
            std::array<gl::Int, 4> value{};
            std::memcpy(value.data(), bytes, count * sizeof(value[0]));
            (count == 1   ? driver.uniform1iv
             : count == 2 ? driver.uniform2iv
             : count == 3 ? driver.uniform3iv
                          : driver.uniform4iv)(location, 1, value.data());
        } else if (slot.numericType == VERNON_RUNTIME_PROVIDER_U32) {
            std::array<gl::Uint, 4> value{};
            std::memcpy(value.data(), bytes, count * sizeof(value[0]));
            (count == 1   ? driver.uniform1uiv
             : count == 2 ? driver.uniform2uiv
             : count == 3 ? driver.uniform3uiv
                          : driver.uniform4uiv)(location, 1, value.data());
        } else {
            std::array<float, 4> value{};
            std::memcpy(value.data(), bytes, count * sizeof(value[0]));
            (count == 1   ? driver.uniform1fv
             : count == 2 ? driver.uniform2fv
             : count == 3 ? driver.uniform3fv
                          : driver.uniform4fv)(location, 1, value.data());
        }
        return;
    }
    std::array<float, 16> value{};
    std::memcpy(value.data(), bytes, count * sizeof(value[0]));
    const uint32_t columns = slot.columnCount;
    const uint32_t rows = count / columns;
    const auto matrix = columns == 2   ? (rows == 2   ? driver.uniformMatrix2fv
                                          : rows == 3 ? driver.uniformMatrix2x3fv
                                                      : driver.uniformMatrix2x4fv)
                        : columns == 3 ? (rows == 2   ? driver.uniformMatrix3x2fv
                                          : rows == 3 ? driver.uniformMatrix3fv
                                                      : driver.uniformMatrix3x4fv)
                                       : (rows == 2   ? driver.uniformMatrix4x2fv
                                          : rows == 3 ? driver.uniformMatrix4x3fv
                                                      : driver.uniformMatrix4fv);
    matrix(location, 1, transpose ? 1 : 0, value.data());
}

// The format of an attribute, packed for the commands: its component count, type and entry point (0 float, 1 integer,
// 2 double).
struct AttributeFormat {
    gl::Int components{};
    gl::Enum type{};
    uint32_t kind{};
};

AttributeFormat attributeFormat(const VernonRuntimeProviderVertexAttribute &attribute) {
    const auto components = static_cast<gl::Int>(attribute.component_count);
    switch (attribute.dtype) {
    case VERNON_RUNTIME_PROVIDER_I32:
        return {components, gl::kInt, 1};
    case VERNON_RUNTIME_PROVIDER_U32:
        return {components, gl::kUnsignedInt, 1};
    case VERNON_RUNTIME_PROVIDER_F64:
        return {components, gl::kDouble, 2};
    case VERNON_RUNTIME_PROVIDER_F16:
        return {components, gl::kHalfFloat, 0};
    default:
        return {components, gl::kFloat, 0};
    }
}

void attribPointer(gl::Driver &driver, uint32_t location, gl::Int components, gl::Enum type, uint32_t kind,
                   gl::Size stride, uint64_t offset) {
    const auto *pointer = reinterpret_cast<const void *>(static_cast<uintptr_t>(offset));
    if (kind == 1)
        driver.vertexAttribIPointer(location, components, type, stride, pointer);
    else if (kind == 2)
        driver.vertexAttribLPointer(location, components, type, stride, pointer);
    else
        driver.vertexAttribPointer(location, components, type, 0, stride, pointer);
}

class OpenGLDevice;

// Builds the recording of one call's planned passes.
class Builder {
public:
    Builder(OpenGLDevice &device, Recording &recording, const PlannedPasses &planned,
            const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Made<void> build(const std::vector<const InternalBuffers::Internal *> &internals);

private:
    using Value = Tracker::Value;
    using FillValue = std::function<bool(size_t slot, VernonRuntimeProviderBindingValue &value,
                                         std::vector<uint8_t> &storage, std::string &error)>;
    struct VertexSource {
        gl::Uint name{};
        uint64_t offset{};
        uint32_t stride{};
    };

    void add(const Command &command) { recording_.commands.push_back(command); }
    // Adds `command` unless the state it sets already holds `value`.
    void set(Tracker::Kind kind, uint64_t index, const Value &value, const Command &command) {
        if (tracker_.change(kind, index, value))
            add(command);
    }
    // A call that sets the state of every color location.
    void setAll(Tracker::Kind kind, const Value &value, const Command &command) {
        bool held = true;
        for (uint32_t location = 0; held && location < kMaxColors; ++location)
            held = tracker_.holds(kind, location, value);
        if (held)
            return;
        for (uint32_t location = 0; location < kMaxColors; ++location)
            tracker_.change(kind, location, value);
        add(command);
    }
    void capability(gl::Enum capability, bool enabled) {
        set(Tracker::Capability, capability, {enabled ? 1u : 0u}, {enabled ? Op::Enable : Op::Disable, capability});
    }
    void bindBuffer(gl::Enum target, gl::Uint name) {
        set(Tracker::BufferTarget, target, {name}, {Op::BindBuffer, target, name});
    }
    uint64_t arena(uint64_t size, uint64_t alignment) {
        const uint64_t offset = align(recording_.arena.size(), alignment);
        recording_.arena.resize(offset + size);
        return offset;
    }
    void debug(uint32_t kind, const std::string &label);
    void timer(bool begin);
    Made<StageObjects *> stage(VernonStageExecutable &stage, bool graphics);
    Made<gl::Uint> nameOf(const VernonRuntimeProviderResourceReference &reference, Kind kind);
    Made<ImageInfo> imageOf(const VernonRuntimeProviderResourceReference &reference);
    Made<uint32_t> instance(StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                            const Site &source);
    void bindings(StageObjects &stage, const std::array<uint32_t, kMaxSets> &sets);
    Made<void> fixedFunction(const runtime::GraphicsVariantKey &key, uint32_t stencilReference);
    void vertexInputs(const StageObjects &stage, const std::vector<VertexSource> &sources, uint32_t firstInstance);
    Made<void> buildCompute(size_t at);
    Made<void> buildCopy(const PlannedCopy &copy);
    Made<void> buildScope(size_t begin, size_t end);
    Made<void> draw(const PlannedPass &pass, const PlannedDraw &draw);
    Made<ScopeImages> beginRendering(const runtime::GraphicsRenderScope &scope);
    Made<gl::Uint> scopeFramebuffer(const runtime::GraphicsRenderScope &scope, const ScopeImages &images);

    OpenGLDevice &device_;
    gl::DeviceState &state_;
    const Caps &caps_;
    Recording &recording_;
    const PlannedPasses &planned_;
    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches_;
    Tracker tracker_;
    std::map<std::vector<const void *>, uint32_t> instances_;
    std::unordered_map<ResolvedKey, gl::Uint, ResolvedKeyHash> resolved_;
    std::unordered_map<uint64_t, ImageInfo> images_;
    std::unordered_map<const VernonStageExecutable *, StageObjects *> stages_;
    // By Stage, the value offset plus one of the bytes each named uniform was last given, and the base instance.
    std::unordered_map<const StageObjects *, std::vector<uint64_t>> uniforms_;
    std::unordered_map<const StageObjects *, uint32_t> baseInstances_;
    std::unordered_set<uint64_t> written_;
    bool graphicsBefore_{};
};

class OpenGLDevice final : public Device {
public:
    OpenGLDevice(VernonRuntimeContext &context, std::shared_ptr<Context> shared, const Caps &caps, gl::Uint vertexArray)
        : context_(context), shared_(std::move(shared)), state_(shared_->state()), caps_(caps),
          vertexArray_(vertexArray), buffers_(createOpenGLDeviceBuffers(context.rhiDevice)),
          internals_(buffers_.get()) {}

    ~OpenGLDevice() override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto held = shared_->lock();
        for (Pending &pending : pending_) {
            state_.driver.clientWaitSync(pending.fence, gl::kSyncFlushCommandsBit, gl::kTimeoutIgnored);
            state_.driver.deleteSync(pending.fence);
            if (pending.timing)
                release(*pending.timing);
        }
        (void)state_.takeErrors();
        if (vertexArray_)
            state_.destroyVertexArray(vertexArray_);
        if (zeros_)
            state_.driver.deleteBuffers(1, &zeros_);
    }

    gl::DeviceState &state() const { return state_; }
    const Caps &caps() const { return caps_; }
    const std::shared_ptr<Context> &shared() const { return shared_; }
    VernonRhiDevice rhiDevice() const { return context_.rhiDevice; }
    gl::Uint vertexArray() const { return vertexArray_; }

    Made<std::unique_ptr<Executable>> createExecutable(const ExecutableSource &source) override {
        if (!source.graphics || !source.graphics->resolved || !source.graphics->resolved->stage)
            return failure<std::unique_ptr<Executable>>("OpenGL devices run Program nodes only");
        return Made<std::unique_ptr<Executable>>{ok(std::unique_ptr<Executable>(new PassExecutable(source.graphics)))};
    }

    Made<std::unique_ptr<Buffer>> allocate(uint64_t size) override {
        if (!buffers_)
            return failure<std::unique_ptr<Buffer>>("the OpenGL device allocates no buffers");
        return buffers_->allocate(size);
    }

    Made<void> upload(Buffer &buffer, const void *bytes, size_t size) override {
        return buffers_->upload(buffer, bytes, size);
    }

    // A buffer write is ordered in the GL command stream after the submissions before it, as an enqueued upload must.
    Made<void> enqueueUpload(Buffer &buffer, const void *bytes, size_t size) override {
        return buffers_->upload(buffer, bytes, size);
    }

    Made<void> download(const Buffer &buffer, void *bytes, size_t size) override {
        return buffers_->download(buffer, bytes, size);
    }

    Made<std::unique_ptr<CommandBuffer>> record(const std::vector<DispatchCommand> &commands) override {
        if (auto checked = checkPassCommands(commands, "OpenGL"); checked.isErr())
            return Made<std::unique_ptr<CommandBuffer>>{err(std::move(checked).error())};
        return Made<std::unique_ptr<CommandBuffer>>{
            ok(std::unique_ptr<CommandBuffer>(new OpenGLCommandBuffer(commands)))};
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
        if (!pending_.empty() && pending_.front().point <= point) {
            auto held = shared_->lock();
            while (!pending_.empty() && pending_.front().point <= point)
                retire(gl::kTimeoutIgnored);
        }
        const auto failed = std::find_if(failed_.begin(), failed_.end(), [&](TimelinePoint at) { return at <= point; });
        if (failed == failed_.end())
            return Made<void>{ok()};
        failed_.erase(std::remove_if(failed_.begin(), failed_.end(), [&](TimelinePoint at) { return at <= point; }),
                      failed_.end());
        return failure<void>("a pass submission failed on the device");
    }

    // The program of a Stage this device draws or dispatches with, made on first use.
    Made<std::shared_ptr<StageObjects>> stageObjects(VernonStageExecutable &stage, bool graphics);
    // A buffer of at least `bytes` zeros that internal buffers are cleared from.
    Made<void> reserveZeros(uint64_t bytes);

private:
    struct Pending {
        TimelinePoint point{};
        gl::Sync fence{};
        std::shared_ptr<TimingCapture> timing;
    };

    static constexpr size_t kKeptTimings = 32;

    Made<std::unique_ptr<Recording>> build(const OpenGLCommandBuffer &commands, const PlannedPasses &planned,
                                           const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches,
                                           bool timed);
    Made<void> planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
                          std::vector<uint64_t> &key);
    Made<void> fill(Recording &recording, const PlannedPasses &planned,
                    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Slot &acquire(Recording &recording);
    // Called with the context's mutex held.
    void execute(const Recording &recording, const Slot &slot, const std::vector<gl::Uint> &queries);
    void hostDraws(const Recording &recording, const HostStream &host);

    // Called with mutex_ and the context's mutex held.
    void release(TimingCapture &timing) {
        if (!timing.queries.empty())
            state_.releaseQueries(timing.queries);
        timing.queries.clear();
    }

    // Called with mutex_ and the context's mutex held: retires the oldest pending submission once its fence signals
    // within `timeout`. Returns whether it did.
    bool retire(gl::Uint64 timeout) {
        Pending &pending = pending_.front();
        const gl::Enum status = state_.driver.clientWaitSync(pending.fence, gl::kSyncFlushCommandsBit, timeout);
        if (status == gl::kTimeoutExpired)
            return false;
        state_.driver.deleteSync(pending.fence);
        if ((status != gl::kAlreadySignaled && status != gl::kConditionSatisfied) || state_.takeErrors())
            failed_.push_back(pending.point);
        std::optional<std::vector<ScopeTiming>> scopes;
        if (pending.timing) {
            scopes = readTimings(*pending.timing);
            release(*pending.timing);
        }
        if (pending.timing) {
            if (timings_.size() == kKeptTimings)
                timings_.pop_front();
            timings_.emplace_back(pending.point, std::move(scopes));
        }
        pending_.pop_front();
        return true;
    }

    std::optional<std::vector<ScopeTiming>> readTimings(const TimingCapture &timing) {
        std::vector<uint64_t> ticks(timing.queries.size());
        for (size_t index = 0; index < ticks.size(); ++index)
            state_.driver.getQueryObjectui64v(timing.queries[index], gl::kQueryResult, &ticks[index]);
        gl::Int disjoint = 0;
        if (state_.disjointQueryable)
            state_.driver.getIntegerv(gl::kGpuDisjoint, &disjoint);
        if (disjoint || ticks.size() != timing.scopes.size() * 2)
            return std::nullopt;
        std::vector<ScopeTiming> scopes = timing.scopes;
        const uint64_t origin = ticks.empty() ? 0 : ticks[0];
        for (size_t index = 0; index < scopes.size(); ++index) {
            ScopeTiming &scope = scopes[index];
            scope.begin = ticks[2 * index] - std::min(origin, ticks[2 * index]);
            scope.end = ticks[2 * index + 1] - std::min(origin, ticks[2 * index + 1]);
            scope.end = std::max(scope.end, scope.begin);
        }
        return scopes;
    }

    // Retires the submissions that completed, oldest first.
    void retireCompleted() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_.empty())
            return;
        auto held = shared_->lock();
        while (!pending_.empty() && retire(0)) {
        }
    }

    VernonRuntimeContext &context_;
    std::shared_ptr<Context> shared_;
    gl::DeviceState &state_;
    Caps caps_;
    gl::Uint vertexArray_{};
    // Guarded by the context's mutex.
    gl::Uint zeros_{};
    uint64_t zeroBytes_{};
    // Guarded by the context's mutex: the vertex attribute locations the vertex array enables, which only the device's
    // commands change.
    uint64_t enabled_{};
    std::unique_ptr<DeviceBuffers> buffers_;
    InternalBuffers internals_;
    // Taken before the context's mutex.
    std::mutex mutex_;
    std::deque<Pending> pending_;
    std::vector<TimelinePoint> failed_;
    std::deque<std::pair<TimelinePoint, std::optional<std::vector<ScopeTiming>>>> timings_;
    TimelinePoint next_{1};
};

Made<std::shared_ptr<StageObjects>> buildStageObjects(const std::shared_ptr<Context> &context, const Caps &caps,
                                                      VernonStageExecutable &stage, runtime::OpenGLPipelineState &state,
                                                      bool graphics) {
    using Built = Made<std::shared_ptr<StageObjects>>;
    auto objects = std::make_shared<StageObjects>(context);
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
        if (layout.set >= kMaxSets || layout.array_count != 1 || !entryApplication(layout))
            return failure<std::shared_ptr<StageObjects>>("the Stage has a binding OpenGL devices do not bind");
        setCount = std::max(setCount, layout.set + 1);
    }
    gl::Driver &driver = context->driver();
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
        slot.application = *entryApplication(layout);
        slot.providerKind = layout.kind;
        slot.set = layout.set;
        slot.binding = layout.binding;
        slot.size = layout.element_size;
        slot.packed = packedBytes(layout);
        slot.valueCount = layout.element_count ? layout.element_count : layout.element_size / 4;
        slot.columnCount = layout.vector_count;
        slot.numericType = layout.numeric_type;
        slot.imageAccess = layout.access == 1 ? gl::kReadOnly : layout.access == 2 ? gl::kWriteOnly : gl::kReadWrite;
        slot.storageFormat = layout.storage_image_format;
        if (layout.name.data && layout.name.size)
            slot.name.assign(layout.name.data, layout.name.size);
        if (slot.application == Application::StorageImage && !driver.bindImageTexture)
            return failure<std::shared_ptr<StageObjects>>("the device binds no storage images");
    }
    for (SetPlan &set : objects->sets) {
        std::sort(set.parameters.begin(), set.parameters.end());
        set.parameters.erase(std::unique(set.parameters.begin(), set.parameters.end()), set.parameters.end());
    }
    objects->uniforms.resize(objects->slots.size());
    if (graphics) {
        const PreparedGraphicsBindingPlan &plan = state.graphicsPlan;
        for (const VernonRuntimeProviderVertexBinding &binding : plan.vertexBindings) {
            if (binding.binding >= kMaxVertexBindings)
                return failure<std::shared_ptr<StageObjects>>("a vertex binding is out of range");
            objects->divisors[binding.binding] = binding.divisor;
            objects->bindingMask |= 1u << binding.binding;
        }
        objects->separate = caps.separateFormats;
        for (const VernonRuntimeProviderVertexAttribute &attribute : plan.vertexAttributes) {
            std::string diagnostic;
            if (attribute.binding >= kMaxVertexBindings)
                return failure<std::shared_ptr<StageObjects>>("a vertex attribute's binding is out of range");
            if (!runtime::validateVertexAttributeCapability(runtime::VertexAttributeBackend::OpenGL, attribute,
                                                            caps.maxLocations, driver.vertexAttribLPointer != nullptr,
                                                            diagnostic))
                return failure<std::shared_ptr<StageObjects>>(std::move(diagnostic));
            objects->attributeMask |= uint64_t{1} << attribute.location;
            objects->attributes.push_back({attribute, objects->divisors[attribute.binding]});
            if (attribute.relative_offset > caps.maxRelativeOffset ||
                (attribute.dtype == VERNON_RUNTIME_PROVIDER_F64 && !driver.vertexAttribLFormat))
                objects->separate = false;
        }
    }
    std::vector<std::pair<gl::Enum, const std::string *>> sources;
    if (graphics)
        sources = {{gl::kVertexShader, &state.vertex}, {gl::kFragmentShader, &state.fragment}};
    else
        sources = {{gl::kComputeShader, &state.compute}};
    auto held = context->lock();
    gl::DeviceState &device = context->state();
    std::vector<gl::Uint> compiled;
    std::string error;
    for (const auto &[kind, source] : sources) {
        const gl::Uint shader = source->empty() ? 0 : device.compileShader(kind, *source, error);
        if (!shader) {
            for (gl::Uint made : compiled)
                driver.deleteShader(made);
            return failure<std::shared_ptr<StageObjects>>(source->empty() ? "the Stage has no GLSL" : error);
        }
        compiled.push_back(shader);
    }
    objects->program = device.linkProgram(compiled, error);
    if (!objects->program)
        return failure<std::shared_ptr<StageObjects>>(error);
    // A block left unassigned silently reads binding point 0, so every active block must have a planned binding.
    std::vector<gl::Uint> assignedBlocks;
    for (SlotPlan &slot : objects->slots) {
        if (slot.providerKind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER)
            continue;
        const std::string blockName = runtime::openGLUniformBlockName(slot.binding);
        const gl::Uint block = driver.getUniformBlockIndex(objects->program, blockName.c_str());
        if (block == gl::kInvalidIndex) {
            slot.reads = false;
            continue;
        }
        driver.uniformBlockBinding(objects->program, block, slot.binding);
        if (std::find(assignedBlocks.begin(), assignedBlocks.end(), block) == assignedBlocks.end())
            assignedBlocks.push_back(block);
    }
    gl::Int activeBlocks = 0;
    driver.getProgramiv(objects->program, gl::kActiveUniformBlocks, &activeBlocks);
    if (activeBlocks < 0 || assignedBlocks.size() < static_cast<size_t>(activeBlocks))
        return failure<std::shared_ptr<StageObjects>>("a uniform block of the Stage has no planned binding");
    // Compute GLSL declares each sampler's texture unit as its binding; graphics samplers receive it here, once.
    if (!graphics)
        return Built{ok(std::move(objects))};
    driver.useProgram(objects->program);
    ++device.drawStateGeneration;
    objects->baseInstanceLocation = driver.getUniformLocation(objects->program, "SPIRV_Cross_BaseInstance");
    // A separate sampler serves the texture unit of its binding, which only active sampled images read.
    std::vector<uint32_t> sampledUnits;
    for (SlotPlan &slot : objects->slots) {
        if (slot.application != Application::NamedUniform && slot.application != Application::SampledImage)
            continue;
        const gl::Int location = driver.getUniformLocation(objects->program, slot.name.c_str());
        if (location < 0) {
            slot.reads = false;
            continue;
        }
        if (slot.application == Application::SampledImage) {
            driver.uniform1i(location, static_cast<gl::Int>(slot.binding));
            sampledUnits.push_back(slot.binding);
        } else {
            slot.location = location;
        }
    }
    for (SlotPlan &slot : objects->slots)
        if (slot.application == Application::Sampler)
            slot.reads = std::find(sampledUnits.begin(), sampledUnits.end(), slot.binding) != sampledUnits.end();
    return Built{ok(std::move(objects))};
}

Made<std::shared_ptr<StageObjects>> OpenGLDevice::stageObjects(VernonStageExecutable &stage, bool graphics) {
    using Found = Made<std::shared_ptr<StageObjects>>;
    if (!stage.backendState)
        return failure<std::shared_ptr<StageObjects>>("the Stage has no OpenGL pipeline");
    auto &state = ::runtimeBackendState<runtime::OpenGLPipelineState>(stage);
    std::lock_guard<std::mutex> lock(state.halMutex);
    if (!state.hal) {
        auto built = buildStageObjects(shared_, caps_, stage, state, graphics);
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

Made<void> OpenGLDevice::reserveZeros(uint64_t bytes) {
    auto held = shared_->lock();
    if (zeroBytes_ >= bytes)
        return Made<void>{ok()};
    if (zeros_)
        state_.driver.deleteBuffers(1, &zeros_);
    zeros_ = 0;
    zeroBytes_ = 0;
    state_.driver.genBuffers(1, &zeros_);
    if (!zeros_)
        return failure<void>("cannot create the internal buffers' zeros");
    const std::vector<uint8_t> zeros(bytes);
    state_.driver.bindBuffer(gl::kCopyWriteBuffer, zeros_);
    state_.driver.bufferData(gl::kCopyWriteBuffer, static_cast<gl::SizePtr>(bytes), zeros.data(), kDynamicDraw);
    zeroBytes_ = bytes;
    ++state_.drawStateGeneration;
    return Made<void>{ok()};
}

Builder::Builder(OpenGLDevice &device, Recording &recording, const PlannedPasses &planned,
                 const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches)
    : device_(device), state_(device.state()), caps_(device.caps()), recording_(recording), planned_(planned),
      launches_(launches) {}

void Builder::debug(uint32_t kind, const std::string &label) {
    if (!caps_.labels)
        return;
    Command command;
    command.op = kind == VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP  ? Op::DebugPush
                 : kind == VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP ? Op::DebugPop
                                                                   : Op::DebugMarker;
    command.a = static_cast<uint32_t>(recording_.labels.size());
    recording_.labels.push_back(label);
    add(command);
}

void Builder::timer(bool) {
    if (!recording_.timed || !caps_.timers)
        return;
    add({Op::Timestamp, recording_.queries++});
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

// The OpenGL name of a buffer, view or sampler reference of this device. The recording that names it stays valid only
// while the resource lives, which the keys of the plans it records guarantee.
Made<gl::Uint> Builder::nameOf(const VernonRuntimeProviderResourceReference &reference, Kind kind) {
    if (!reference.resource.value)
        return failure<gl::Uint>("a resource reference is empty");
    if (reference.identity != nativeIdentity(&state_, static_cast<uint64_t>(kind)))
        return failure<gl::Uint>("a resource reference names another device's resource or one of another kind");
    const ResolvedKey key{reference.resource.value, static_cast<uint64_t>(kind)};
    if (const auto found = resolved_.find(key); found != resolved_.end())
        return Made<gl::Uint>{ok(found->second)};
    gl::Uint name = 0;
    switch (kind) {
    case Kind::Buffer:
        if (const gl::Buffer *buffer = openGLBuffer(reference.resource.value))
            name = buffer->name;
        break;
    case Kind::View:
        if (const NativeView *view = nativeViews().find(reference.resource.value))
            name = static_cast<const OpenGLViewTexture *>(view->native)->name;
        break;
    case Kind::Sampler:
        if (const auto *sampler = static_cast<const gl::Sampler *>(nativeSamplers().find(reference.resource.value)))
            name = sampler->name;
        break;
    }
    if (!name)
        return failure<gl::Uint>("a resource is destroyed");
    resolved_.emplace(key, name);
    return Made<gl::Uint>{ok(name)};
}

Made<ImageInfo> Builder::imageOf(const VernonRuntimeProviderResourceReference &reference) {
    if (const auto found = images_.find(reference.resource.value); found != images_.end())
        return Made<ImageInfo>{ok(found->second)};
    auto name = nameOf(reference, Kind::View);
    if (name.isErr())
        return failure<ImageInfo>(std::move(name).error());
    const NativeView &view = *nativeViews().find(reference.resource.value);
    const ImageInfo info{name.value(), view.view.dimension, view.view.format};
    images_.emplace(reference.resource.value, info);
    return Made<ImageInfo>{ok(info)};
}

// Makes a set instance of `set` from the values `fill` gives. A persistent instance's bytes are written once; the
// others are sites the call fills again from `source`.
Made<uint32_t> Builder::instance(StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                                 const Site &source) {
    const SetPlan &plan = stage.sets[set];
    const auto at = static_cast<uint32_t>(recording_.sets.size());
    SetInstance made;
    VernonRuntimeProviderBindingValue value{};
    std::vector<uint8_t> storage;
    std::string error;
    for (const uint32_t slotIndex : plan.slots) {
        const SlotPlan &slot = stage.slots[slotIndex];
        if (!fill(slotIndex, value, storage, error))
            return failure<uint32_t>("cannot fill a binding: " + error);
        if (value.kind != slot.providerKind)
            return failure<uint32_t>("a binding value has the wrong kind");
        Binding binding;
        binding.slot = slotIndex;
        Site site = source;
        site.slot = slotIndex;
        site.size = slot.size;
        if (slot.packed) {
            const auto *bytes = static_cast<const uint8_t *>(value.payload.inline_value.data);
            if (!bytes || value.payload.inline_value.size != slot.size)
                return failure<uint32_t>("an inline binding value has the wrong size");
            binding.transpose = (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE) != 0;
            if (slot.application == Application::NamedUniform) {
                site.values = true;
                site.offset = recording_.values.size();
                recording_.values.insert(recording_.values.end(), bytes, bytes + slot.size);
            } else {
                site.offset = arena(slot.size, slot.application == Application::UniformBuffer ? caps_.uniformAlignment
                                                                                              : caps_.storageAlignment);
                std::memcpy(recording_.arena.data() + site.offset, bytes, slot.size);
                binding.name = kArena;
                binding.range = slot.size;
            }
            binding.offset = site.offset;
            if (!persistent)
                recording_.sites.push_back(site);
            made.bindings.push_back(binding);
            continue;
        }
        const bool defaultResource = (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
        binding.target = gl::kTexture2D;
        switch (slot.application) {
        case Application::UniformBuffer:
        case Application::StorageBuffer: {
            const VernonRuntimeProviderResourceReference &resource = value.payload.buffer.resource;
            if (resource.size == 0)
                return failure<uint32_t>("a buffer binding is empty");
            auto name = nameOf(resource, Kind::Buffer);
            if (name.isErr())
                return failure<uint32_t>(std::move(name).error());
            const uint64_t alignment =
                slot.application == Application::UniformBuffer ? caps_.uniformAlignment : caps_.storageAlignment;
            if (resource.offset % alignment != 0)
                return failure<uint32_t>("a buffer binding's offset is misaligned");
            binding.name = name.value();
            binding.offset = resource.offset;
            binding.range = resource.size;
            break;
        }
        case Application::SampledImage:
        case Application::StorageImage: {
            if (defaultResource) {
                if (slot.application == Application::StorageImage)
                    return failure<uint32_t>("a storage image binding has no image");
                break;
            }
            auto image = imageOf(value.payload.image.view);
            if (image.isErr())
                return failure<uint32_t>(std::move(image).error());
            binding.name = image.value().name;
            binding.target = openGLTarget(image.value().dimension);
            binding.layered = image.value().dimension != VERNON_TEXTURE_2D;
            if (slot.application == Application::StorageImage) {
                if (image.value().format != slot.storageFormat)
                    return failure<uint32_t>("a storage image's format is not the one its kernel declares");
                const vernon::formats::Format *format = vernon::formats::find(image.value().format);
                if (!format || !format->allows(VERNON_RHI_FORMAT_STORAGE))
                    return failure<uint32_t>("a storage image's format cannot be stored to");
                binding.format = format->glInternalFormat;
            }
            break;
        }
        case Application::Sampler: {
            if (defaultResource)
                break;
            auto name = nameOf(value.payload.sampler.resource, Kind::Sampler);
            if (name.isErr())
                return failure<uint32_t>(std::move(name).error());
            binding.name = name.value();
            break;
        }
        case Application::NamedUniform:
            return failure<uint32_t>("a named uniform has no bytes");
        }
        made.bindings.push_back(binding);
    }
    recording_.sets.push_back(std::move(made));
    return Made<uint32_t>{ok(at)};
}

// The context bindings and named uniforms of a draw's or dispatch's set instances, as the program reads them.
void Builder::bindings(StageObjects &stage, const std::array<uint32_t, kMaxSets> &sets) {
    std::vector<uint64_t> &uniforms = uniforms_[&stage];
    uniforms.resize(stage.slots.size());
    for (uint32_t index = 0; index < stage.sets.size(); ++index) {
        if (sets[index] == kNone)
            continue;
        for (const Binding &binding : recording_.sets[sets[index]].bindings) {
            const SlotPlan &slot = stage.slots[binding.slot];
            if (!slot.reads)
                continue;
            switch (slot.application) {
            case Application::UniformBuffer:
            case Application::StorageBuffer: {
                const gl::Enum target =
                    slot.application == Application::UniformBuffer ? gl::kUniformBuffer : gl::kShaderStorageBuffer;
                // A resource bound from offset 0 binds the whole buffer, as the Core adapter binds it.
                const uint64_t range = binding.name != kArena && binding.offset == 0 ? 0 : binding.range;
                Command bind{Op::BindBufferRange, target, slot.binding, binding.name};
                bind.offset = binding.offset;
                bind.size = range;
                set(Tracker::IndexedBuffer, (uint64_t{target} << 16) | slot.binding,
                    {binding.name, binding.offset, range}, bind);
                break;
            }
            case Application::SampledImage:
                if (tracker_.change(Tracker::Texture, slot.binding, {binding.target, binding.name})) {
                    set(Tracker::ActiveTexture, 0, {slot.binding}, {Op::ActiveTexture, slot.binding});
                    add({Op::BindTexture, binding.target, binding.name});
                }
                break;
            case Application::Sampler:
                set(Tracker::Sampler, slot.binding, {binding.name}, {Op::BindSampler, slot.binding, binding.name});
                break;
            case Application::StorageImage:
                set(Tracker::Image, slot.binding,
                    {binding.name, binding.layered ? 1u : 0u, slot.imageAccess, binding.format},
                    {Op::BindImage, slot.binding, binding.name, binding.layered ? 1u : 0u, slot.imageAccess,
                     binding.format});
                break;
            case Application::NamedUniform: {
                if (slot.location < 0 || uniforms[binding.slot] == binding.offset + 1)
                    break;
                Command upload{Op::Uniform, binding.slot, binding.transpose ? 1u : 0u};
                upload.stage = &stage;
                upload.offset = binding.offset;
                add(upload);
                uniforms[binding.slot] = binding.offset + 1;
                break;
            }
            }
        }
    }
}

Made<void> Builder::fixedFunction(const runtime::GraphicsVariantKey &key, uint32_t stencilReference) {
    const VernonRasterizationState &rasterization = key.rasterization;
    const VernonDepthStencilState &depthStencil = key.depthStencil;
    const gl::Driver &driver = state_.driver;
    const bool stencilFormat = key.depthStencilFormat == gl::kDepth32fStencil8;
    if (key.sampleCount != 1 || key.colorFormats.size() > kMaxColors ||
        key.colorBlends.size() != key.colorFormats.size() || rasterization.cull_mode > VERNON_RHI_CULL_BACK ||
        rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE || rasterization.depth_clamp ||
        !std::isfinite(rasterization.depth_bias_constant) || !std::isfinite(rasterization.depth_bias_slope) ||
        depthStencil.depth_compare > VERNON_RHI_COMPARE_ALWAYS ||
        (key.depthStencilFormat && key.depthStencilFormat != gl::kDepthComponent32f && !stencilFormat) ||
        (!key.depthStencilFormat && (depthStencil.depth_test || depthStencil.depth_write)) ||
        (depthStencil.stencil_test && !stencilFormat) || depthStencil.stencil_read_mask > 0xff ||
        depthStencil.stencil_write_mask > 0xff ||
        (key.colorBlends.size() > 1 && (!driver.enablei || !driver.disablei || !driver.blendFuncSeparatei ||
                                        !driver.blendEquationSeparatei || !driver.colorMaski)))
        return failure<void>("the graphics pipeline uses state OpenGL devices do not draw with");
    const bool cull = rasterization.cull_mode != VERNON_RHI_CULL_NONE;
    capability(gl::kCullFace, cull);
    if (cull) {
        const gl::Enum face = rasterization.cull_mode == VERNON_RHI_CULL_FRONT ? gl::kFront : gl::kBack;
        set(Tracker::CullFace, 0, {face}, {Op::CullFace, face});
    }
    // GL vertex stages negate y so that image row 0 is the top, which mirrors every primitive's window-space winding.
    const gl::Enum front =
        rasterization.front_face == VERNON_RHI_FRONT_FACE_CLOCKWISE ? gl::kCounterClockwise : gl::kClockwise;
    set(Tracker::FrontFace, 0, {front}, {Op::FrontFace, front});
    capability(kPolygonOffsetFill, rasterization.depth_bias_enabled != 0);
    if (rasterization.depth_bias_enabled) {
        const uint32_t slope = floatBits(rasterization.depth_bias_slope);
        const uint32_t constant = floatBits(rasterization.depth_bias_constant);
        set(Tracker::PolygonOffset, 0, {slope, constant}, {Op::PolygonOffset, slope, constant});
    }
    capability(gl::kDepthTest, depthStencil.depth_test != 0);
    if (depthStencil.depth_test) {
        const gl::Enum function = 0x0200 + depthStencil.depth_compare;
        set(Tracker::DepthFunc, 0, {function}, {Op::DepthFunc, function});
    }
    set(Tracker::DepthMask, 0, {depthStencil.depth_write ? 1u : 0u},
        {Op::DepthMask, depthStencil.depth_write ? 1u : 0u});
    capability(gl::kStencilTest, depthStencil.stencil_test != 0);
    if (depthStencil.stencil_test) {
        const std::array<std::pair<gl::Enum, const VernonStencilFaceState *>, 2> faces{
            {{gl::kFront, &depthStencil.front}, {gl::kBack, &depthStencil.back}}};
        for (const auto &[face, faceState] : faces) {
            if (faceState->stencil_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
                faceState->depth_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
                faceState->pass > VERNON_RHI_STENCIL_DECREMENT_WRAP || faceState->compare > VERNON_RHI_COMPARE_ALWAYS)
                return failure<void>("the graphics pipeline has an invalid stencil state");
            const gl::Enum function = 0x0200 + faceState->compare;
            set(Tracker::StencilFunc, face, {function, stencilReference, depthStencil.stencil_read_mask},
                {Op::StencilFunc, face, function, stencilReference, depthStencil.stencil_read_mask});
            const uint32_t fail = stencilOperation(faceState->stencil_fail);
            const uint32_t depthFail = stencilOperation(faceState->depth_fail);
            const uint32_t pass = stencilOperation(faceState->pass);
            set(Tracker::StencilOp, face, {fail, depthFail, pass}, {Op::StencilOp, face, fail, depthFail, pass});
            set(Tracker::StencilMask, face, {depthStencil.stencil_write_mask},
                {Op::StencilMask, face, depthStencil.stencil_write_mask});
        }
    }
    for (uint32_t index = 0; index < key.colorBlends.size(); ++index) {
        const VernonColorBlendState &blend = key.colorBlends[index];
        if (blend.blend_enabled > 1 || blend.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            blend.color_operation > VERNON_RHI_BLEND_MAXIMUM || blend.alpha_operation > VERNON_RHI_BLEND_MAXIMUM ||
            (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
            return failure<void>("the graphics pipeline has an invalid blend state");
        const Value function{blendFactor(blend.source_color_factor), blendFactor(blend.destination_color_factor),
                             blendFactor(blend.source_alpha_factor), blendFactor(blend.destination_alpha_factor)};
        const Value equation{blendOperation(blend.color_operation), blendOperation(blend.alpha_operation)};
        const auto f = [&](size_t at) { return static_cast<uint32_t>(function[at]); };
        if (key.colorBlends.size() == 1) {
            setAll(Tracker::BlendEnable, {blend.blend_enabled},
                   {blend.blend_enabled ? Op::Enable : Op::Disable, gl::kBlend});
            setAll(Tracker::BlendFunc, function, {Op::BlendFunc, f(0), f(1), f(2), f(3)});
            setAll(Tracker::BlendEquation, equation,
                   {Op::BlendEquation, static_cast<uint32_t>(equation[0]), static_cast<uint32_t>(equation[1])});
            setAll(Tracker::ColorMask, {blend.write_mask}, {Op::ColorMask, blend.write_mask});
            continue;
        }
        set(Tracker::BlendEnable, index, {blend.blend_enabled},
            {blend.blend_enabled ? Op::Enablei : Op::Disablei, gl::kBlend, index});
        set(Tracker::BlendFunc, index, function, {Op::BlendFunci, f(0), f(1), f(2), f(3), index});
        set(Tracker::BlendEquation, index, equation,
            {Op::BlendEquationi, static_cast<uint32_t>(equation[0]), static_cast<uint32_t>(equation[1]), 0, 0, index});
        set(Tracker::ColorMask, index, {blend.write_mask}, {Op::ColorMaski, blend.write_mask, 0, 0, 0, index});
    }
    return Made<void>{ok()};
}

// Specifies the vertex array's attributes for `stage` reading `sources`, with each per-instance binding of divisor d
// moved forward by firstInstance / d records, which emulates a first instance on draws without a base instance.
// Attribute pointers and separate formats describe the same vertex array state: a pointer sets its location's format,
// binding and buffer, and a divisor its location's binding and divisor, so each forgets what the other mode set.
void Builder::vertexInputs(const StageObjects &stage, const std::vector<VertexSource> &sources,
                           uint32_t firstInstance) {
    // A zero stride packs a binding's attributes tightly for attribute pointers only.
    bool separate = stage.separate;
    for (uint32_t binding = 0; separate && binding < kMaxVertexBindings; ++binding)
        separate = (stage.bindingMask & (1u << binding)) == 0 || sources[binding].stride != 0;
    std::array<uint64_t, kMaxVertexBindings> offsets{};
    for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
        if (stage.bindingMask & (1u << binding))
            offsets[binding] =
                sources[binding].offset +
                (stage.divisors[binding] ? uint64_t{firstInstance / stage.divisors[binding]} * sources[binding].stride
                                         : 0);
    Command mask{Op::AttribMask};
    mask.offset = stage.attributeMask;
    set(Tracker::AttribMask, 0, {stage.attributeMask}, mask);
    if (separate) {
        for (const VertexAttribute &attribute : stage.attributes) {
            const uint32_t location = attribute.layout.location;
            const AttributeFormat format = attributeFormat(attribute.layout);
            if (tracker_.change(Tracker::AttribFormat, location,
                                {static_cast<uint64_t>(format.components), format.type, format.kind,
                                 attribute.layout.relative_offset})) {
                add({Op::AttribFormat, location, static_cast<uint32_t>(format.components), format.type,
                     attribute.layout.relative_offset, format.kind});
                tracker_.forget(Tracker::AttribPointer, location);
            }
            if (tracker_.change(Tracker::AttribBinding, location, {attribute.layout.binding})) {
                add({Op::AttribBinding, location, attribute.layout.binding});
                tracker_.forget(Tracker::AttribPointer, location);
                tracker_.forget(Tracker::AttribDivisor, location);
            }
        }
        for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding) {
            if ((stage.bindingMask & (1u << binding)) == 0)
                continue;
            if (tracker_.change(Tracker::BindingDivisor, binding, {stage.divisors[binding]})) {
                add({Op::BindingDivisor, binding, stage.divisors[binding]});
                tracker_.forget(Tracker::AttribDivisor, binding);
            }
            const VertexSource &source = sources[binding];
            if (tracker_.change(Tracker::VertexBuffer, binding, {source.name, offsets[binding], source.stride})) {
                Command bind{Op::VertexBuffer, binding, source.name, source.stride};
                bind.offset = offsets[binding];
                add(bind);
                tracker_.forget(Tracker::AttribPointer, binding);
            }
        }
        return;
    }
    for (const VertexAttribute &attribute : stage.attributes) {
        const uint32_t location = attribute.layout.location;
        const uint32_t binding = attribute.layout.binding;
        const VertexSource &source = sources[binding];
        const AttributeFormat format = attributeFormat(attribute.layout);
        const uint64_t offset = offsets[binding] + attribute.layout.relative_offset;
        if (tracker_.change(Tracker::AttribPointer, location,
                            {source.name, offset, source.stride,
                             (uint64_t{format.type} << 8) | (uint64_t{format.kind} << 4) |
                                 static_cast<uint64_t>(format.components)})) {
            bindBuffer(gl::kArrayBuffer, source.name);
            Command pointer{Op::AttribPointer, location,      static_cast<uint32_t>(format.components),
                            format.type,       source.stride, format.kind};
            pointer.offset = offset;
            add(pointer);
            tracker_.forget(Tracker::AttribFormat, location);
            tracker_.forget(Tracker::AttribBinding, location);
            tracker_.forget(Tracker::VertexBuffer, location);
        }
        if (tracker_.change(Tracker::AttribDivisor, location, {attribute.divisor})) {
            add({Op::AttribDivisor, location, attribute.divisor});
            tracker_.forget(Tracker::AttribBinding, location);
            tracker_.forget(Tracker::BindingDivisor, location);
        }
    }
}

Made<void> Builder::build(const std::vector<const InternalBuffers::Internal *> &internals) {
    if (!internals.empty() || !planned_.zeroed.empty()) {
        uint64_t bytes = 0;
        for (const InternalBuffers::Internal *internal : internals)
            bytes = std::max(bytes, internal->bytes);
        for (const VernonRuntimeProviderResourceReference &range : planned_.zeroed)
            bytes = std::max(bytes, range.size);
        if (auto reserved = device_.reserveZeros(bytes); reserved.isErr())
            return reserved;
        for (const InternalBuffers::Internal *internal : internals) {
            auto buffer = nameOf(internal->reference, Kind::Buffer);
            if (buffer.isErr())
                return failure<void>("an internal buffer: " + buffer.error());
            Command copy{Op::CopyBuffer, kZeros, buffer.value()};
            copy.size = internal->bytes;
            add(copy);
        }
        for (const VernonRuntimeProviderResourceReference &range : planned_.zeroed) {
            if (!range.size)
                continue;
            auto buffer = nameOf(range, Kind::Buffer);
            if (buffer.isErr())
                return failure<void>("a zeroed argument: " + buffer.error());
            Command copy{Op::CopyBuffer, kZeros, buffer.value()};
            copy.offset = range.offset;
            copy.size = range.size;
            add(copy);
        }
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
            if (recording_.timed && caps_.timers)
                recording_.timers.push_back({pass.executable->source().node->id, 1, 0, 0});
            timer(true);
            debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, nodeLabel(*pass.executable->source().node));
            if (auto built = buildCompute(at); built.isErr())
                return built;
            debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
            timer(false);
            ++at;
            continue;
        }
        size_t end = at + 1;
        while (end < passes.size() && !passes[end]->compute && passes[end]->scope == pass.scope)
            ++end;
        if (recording_.timed && caps_.timers)
            recording_.timers.push_back({pass.executable->source().node->id, static_cast<uint32_t>(end - at), 0, 0});
        timer(true);
        if (auto built = buildScope(at, end); built.isErr())
            return built;
        timer(false);
        at = end;
    }
    // Image copies blit through framebuffers and must stay raw, and other OpenGL code expects the default.
    if (tracker_.holds(Tracker::Capability, gl::kFramebufferSrgb, {1}))
        add({Op::Disable, gl::kFramebufferSrgb});
    return Made<void>{ok()};
}

Made<void> Builder::buildCompute(size_t at) {
    const PlannedPass &pass = *planned_.passes[at];
    if (!launches_[at])
        return Made<void>{ok()};
    if (!caps_.compute)
        return failure<void>("the OpenGL device dispatches no compute passes");
    const runtime::PlannedComputeLaunch &launch = *launches_[at];
    VernonStageExecutable &executable = *pass.executable->source().resolved->stage;
    auto objects = stage(executable, false);
    if (objects.isErr())
        return Made<void>{err(std::move(objects).error())};
    StageObjects &stageObjects = *objects.value();
    const auto &state = ::runtimeBackendState<runtime::OpenGLPipelineState>(executable);
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
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        if (stageObjects.sets[set].slots.empty())
            continue;
        auto made = instance(stageObjects, set, false, fill, source);
        if (made.isErr())
            return Made<void>{err(name + made.error())};
        sets[set] = made.value();
    }
    if (graphicsBefore_) {
        add({Op::Barrier, kAllBarriers});
        graphicsBefore_ = false;
    }
    set(Tracker::Program, 0, {stageObjects.program}, {Op::UseProgram, stageObjects.program});
    bindings(stageObjects, sets);
    add({Op::Dispatch, static_cast<uint32_t>(recording_.groups.size())});
    recording_.dispatchPasses.push_back(static_cast<uint32_t>(at));
    recording_.groups.push_back({});
    // What follows reads what the dispatch wrote: as vertices, indices, indirect records, images or in shaders.
    add({Op::Barrier, kAllBarriers});
    return Made<void>{ok()};
}

// A dispatch is followed by a barrier already; shaders that read what a copy wrote need none.
Made<void> Builder::buildCopy(const PlannedCopy &copy) {
    if (copy.regions.empty())
        return Made<void>{ok()};
    auto source = nameOf(copy.source, Kind::Buffer);
    if (source.isErr())
        return failure<void>("a leaf copy's source: " + source.error());
    auto destination = nameOf(copy.destination, Kind::Buffer);
    if (destination.isErr())
        return failure<void>("a leaf copy's destination: " + destination.error());
    if (graphicsBefore_) {
        add({Op::Barrier, kAllBarriers});
        graphicsBefore_ = false;
    }
    for (const CopyRegion &region : copy.regions) {
        Command command{Op::CopyBuffer, source.value(), destination.value()};
        command.readOffset = region.source;
        command.offset = region.destination;
        command.size = region.size;
        add(command);
    }
    return Made<void>{ok()};
}

Made<void> Builder::draw(const PlannedPass &pass, const PlannedDraw &draw) {
    auto &state = ::runtimeBackendState<runtime::OpenGLPipelineState>(*draw.stage);
    if (state.vertex.empty())
        return Made<void>{ok()};
    auto objects = stage(*draw.stage, true);
    if (objects.isErr())
        return Made<void>{err(std::move(objects).error())};
    StageObjects &stageObjects = *objects.value();
    const PreparedGraphicsBindingPlan &plan = state.graphicsPlan;
    const runtime::GraphicsSlotInputs inputs{draw.argumentPointers.data(), draw.draw.resolution};
    std::string error;
    std::vector<uint32_t> strides(plan.vertexInputs.size());
    std::vector<VertexSource> sources(kMaxVertexBindings);
    if (plan.vertexInputs.size() > kMaxVertexBindings)
        return failure<void>("the draw has too many vertex buffers");
    for (size_t binding = 0; binding < plan.vertexInputs.size(); ++binding) {
        VernonRuntimeProviderResourceReference buffer{};
        if (!runtime::preparedGraphicsVertexBuffer(plan, inputs, binding, buffer, strides[binding], error))
            return failure<void>(error);
        auto name = nameOf(buffer, Kind::Buffer);
        if (name.isErr())
            return failure<void>("a vertex buffer: " + name.error());
        sources[binding] = {name.value(), buffer.offset, strides[binding]};
    }
    uint32_t depthFormat = 0;
    if (pass.targets.depthAttachment) {
        const vernon::formats::Format *format = vernon::formats::find(pass.targets.depthFormat);
        if (!format)
            return failure<void>("OpenGL depth attachment format is unsupported");
        depthFormat = format->glInternalFormat;
    }
    std::vector<uint32_t> formats;
    for (VernonTextureFormat format : pass.targets.attachmentFormats)
        formats.push_back(static_cast<uint32_t>(format));
    runtime::GraphicsVariantKey key;
    if (!runtime::prepareGraphicsVariantKey(*draw.state, pass.targets, std::move(formats), depthFormat, strides, key,
                                            error))
        return failure<void>(error);
    const FillValue fill = [&](size_t slot, VernonRuntimeProviderBindingValue &value, std::vector<uint8_t> &storage,
                               std::string &message) {
        return runtime::fillPreparedGraphicsBindingValue(plan, inputs, slot, value, storage, message);
    };
    Site source;
    source.draw = &draw;
    source.graphics = &plan;
    std::array<uint32_t, kMaxSets> sets{};
    sets.fill(kNone);
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        const SetPlan &setPlan = stageObjects.sets[set];
        if (setPlan.slots.empty())
            continue;
        // As PipelineGroups finds a group: by the owners of what decides it, and by the draw when it also depends
        // on the command. A group whose owners share one persistent store keeps its bytes across calls.
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
            sets[set] = existing->second;
            continue;
        }
        auto created = instance(stageObjects, set, persistent, fill, source);
        if (created.isErr())
            return Made<void>{err(std::move(created).error())};
        sets[set] = created.value();
        instances_.emplace(std::move(found), created.value());
    }
    VernonRuntimeCoreDrawInvocation invocation{};
    if (!runtime::prepareGraphicsDrawInvocation(draw.draw, draw.state->topology, invocation, error))
        return failure<void>(error);
    if (invocation.topology > 2)
        return failure<void>("the draw's topology is unsupported");
    const gl::Enum topology = kTopologies[invocation.topology];
    const bool indexed = invocation.index_count != 0 || draw.streamed;
    gl::Uint indexBuffer = 0;
    if (indexed) {
        if (!invocation.index_count)
            return failure<void>("a draw stream needs an index binding");
        auto index = nameOf(invocation.index_buffer, Kind::Buffer);
        if (index.isErr())
            return failure<void>("an index buffer: " + index.error());
        indexBuffer = index.value();
    }
    const gl::Driver &driver = state_.driver;
    const runtime::GraphicsDrawStream &stream = draw.stream;
    const bool deviceStream = draw.streamed && stream.records.resource.value != 0;
    const bool counted = deviceStream && stream.counts.resource.value != 0;
    if (draw.streamed) {
        if (deviceStream && (!caps_.deviceStreams || (counted && !caps_.countedStreams)))
            return failure<void>("the device draws no device draw streams");
        if (!deviceStream && (!stream.hostRecords || !stream.hostCounts))
            return failure<void>("a host draw stream has no records");
        if (stream.maxRecords == 0)
            return Made<void>{ok()};
    }
    const uint32_t firstInstance = draw.streamed ? 0 : invocation.first_instance;
    const bool baseInstanceDraw =
        firstInstance != 0 && (indexed ? driver.drawElementsInstancedBaseVertexBaseInstance != nullptr
                                       : driver.drawArraysInstancedBaseInstance != nullptr);
    if (firstInstance != 0 && !baseInstanceDraw)
        // An attribute of divisor d reads record (first + i) / d, which an offset reaches only when d divides first.
        for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
            if (stageObjects.divisors[binding] > 1 && firstInstance % stageObjects.divisors[binding] != 0)
                return failure<void>("the draw's first instance is not a multiple of a vertex divisor");
    if (indexed && !draw.streamed && invocation.base_vertex != 0 && !driver.drawElementsInstancedBaseVertex)
        return failure<void>("the device draws no base vertices");

    set(Tracker::Program, 0, {stageObjects.program}, {Op::UseProgram, stageObjects.program});
    if (auto applied = fixedFunction(key, invocation.stencil_reference); applied.isErr())
        return applied;
    set(Tracker::VertexArray, 0, {device_.vertexArray()}, {Op::BindVertexArray, device_.vertexArray()});
    vertexInputs(stageObjects, sources, baseInstanceDraw ? 0 : firstInstance);
    bindings(stageObjects, sets);
    if (stageObjects.baseInstanceLocation >= 0 && !draw.streamed) {
        const auto found = baseInstances_.find(&stageObjects);
        if (found == baseInstances_.end() || found->second != firstInstance) {
            Command base{Op::BaseInstance, firstInstance};
            base.stage = &stageObjects;
            add(base);
            baseInstances_[&stageObjects] = firstInstance;
        }
    }
    const std::array<uint32_t, 4> viewport{invocation.viewport[0], invocation.viewport[1], invocation.viewport[2],
                                           invocation.viewport[3]};
    set(Tracker::Viewport, 0, {viewport[0], viewport[1], viewport[2], viewport[3]},
        {Op::Viewport, viewport[0], viewport[1], viewport[2], viewport[3]});
    capability(gl::kScissorTest, true);
    const std::array<uint32_t, 4> scissor{invocation.scissor[0], invocation.scissor[1], invocation.scissor[2],
                                          invocation.scissor[3]};
    set(Tracker::Scissor, 0, {scissor[0], scissor[1], scissor[2], scissor[3]},
        {Op::Scissor, scissor[0], scissor[1], scissor[2], scissor[3]});
    if (indexed)
        bindBuffer(gl::kElementArrayBuffer, indexBuffer);
    const uint64_t indexOffset = invocation.index_buffer.offset;
    if (!draw.streamed) {
        if (indexed) {
            Command command{Op::DrawElements,
                            topology,
                            invocation.index_count,
                            invocation.instance_count,
                            static_cast<uint32_t>(invocation.base_vertex),
                            firstInstance};
            command.f = baseInstanceDraw ? 2 : invocation.base_vertex != 0 ? 1 : 0;
            command.offset = indexOffset + uint64_t{invocation.first_index} * sizeof(uint32_t);
            add(command);
        } else {
            Command command{Op::DrawArrays, topology, invocation.vertex_count, invocation.instance_count,
                            firstInstance};
            command.f = baseInstanceDraw ? 2 : invocation.instance_count == 1 ? 0 : 1;
            add(command);
        }
        return Made<void>{ok()};
    }
    if (deviceStream) {
        // The records are DrawElementsIndirectCommands, whose base instance offsets the per-instance attributes. One
        // uniform cannot give each draw its base instance, nor can an index binding with an offset reach their first
        // indices, which count from the start of the bound buffer.
        if (stageObjects.baseInstanceLocation >= 0)
            return failure<void>("device draw streams need native base instances");
        if (indexOffset != 0)
            return failure<void>("device draw streams need an index binding at offset zero");
        if (uint64_t{stream.firstRecord} + stream.maxRecords > stream.records.size / kRecordBytes ||
            (counted && stream.countIndex >= stream.counts.size / sizeof(uint32_t)))
            return failure<void>("a device draw stream exceeds its buffers");
        auto records = nameOf(stream.records, Kind::Buffer);
        if (records.isErr())
            return failure<void>("a draw stream's records: " + records.error());
        bindBuffer(gl::kDrawIndirectBuffer, records.value());
        Command indirect{counted ? Op::MultiDrawCount : Op::MultiDraw, topology, stream.maxRecords};
        indirect.offset = stream.records.offset + uint64_t{stream.firstRecord} * kRecordBytes;
        if (counted) {
            auto counts = nameOf(stream.counts, Kind::Buffer);
            if (counts.isErr())
                return failure<void>("a draw stream's counts: " + counts.error());
            bindBuffer(gl::kParameterBuffer, counts.value());
            indirect.size = stream.counts.offset + uint64_t{stream.countIndex} * sizeof(uint32_t);
        }
        add(indirect);
        return Made<void>{ok()};
    }
    if (indexOffset % sizeof(uint32_t) != 0)
        return failure<void>("a host draw stream's index binding is misaligned");
    HostStream host;
    host.draw = &draw;
    host.stage = &stageObjects;
    host.indexCapacity = invocation.index_buffer.size / sizeof(uint32_t);
    host.indexOffset = indexOffset;
    host.topology = topology;
    host.indirect = caps_.deviceStreams && stageObjects.baseInstanceLocation < 0;
    const auto index = static_cast<uint32_t>(recording_.streams.size());
    if (host.indirect) {
        host.counted = caps_.countedStreams;
        host.records = arena(uint64_t{stream.maxRecords} * kRecordBytes, sizeof(uint32_t));
        if (host.counted)
            host.count = arena(sizeof(uint32_t), sizeof(uint32_t));
        recording_.streams.push_back(host);
        bindBuffer(gl::kDrawIndirectBuffer, kArena);
        Command indirect{host.counted ? Op::MultiDrawCount : Op::MultiDraw, topology, stream.maxRecords};
        indirect.offset = host.records;
        if (host.counted) {
            bindBuffer(gl::kParameterBuffer, kArena);
            indirect.size = host.count;
        }
        add(indirect);
        return Made<void>{ok()};
    }
    host.baseInstances = driver.drawElementsInstancedBaseVertexBaseInstance != nullptr;
    bool separate = stageObjects.separate;
    for (uint32_t binding = 0; separate && binding < kMaxVertexBindings; ++binding)
        separate = (stageObjects.bindingMask & (1u << binding)) == 0 || sources[binding].stride != 0;
    host.separate = separate;
    if (!host.baseInstances)
        for (uint32_t binding = 0; binding < kMaxVertexBindings; ++binding)
            if ((stageObjects.bindingMask & (1u << binding)) && stageObjects.divisors[binding])
                host.instanced.push_back({binding, stageObjects.divisors[binding], sources[binding].name,
                                          sources[binding].offset, sources[binding].stride});
    recording_.streams.push_back(host);
    add({Op::HostDraws, index});
    // The draws move the per-instance bindings and set the base instance as each record needs.
    for (const InstancedBinding &instanced : host.instanced) {
        tracker_.forget(Tracker::VertexBuffer, instanced.binding);
        for (const VertexAttribute &attribute : stageObjects.attributes)
            if (attribute.layout.binding == instanced.binding)
                tracker_.forget(Tracker::AttribPointer, attribute.layout.location);
    }
    if (!host.instanced.empty() && !separate)
        tracker_.forget(Tracker::BufferTarget, gl::kArrayBuffer);
    baseInstances_.erase(&stageObjects);
    return Made<void>{ok()};
}

// The complete framebuffer that attaches the scope's views, made and completeness-checked on first use. Each lives in
// the context's cache until one of the textures it attaches is destroyed.
Made<gl::Uint> Builder::scopeFramebuffer(const runtime::GraphicsRenderScope &scope, const ScopeImages &images) {
    std::array<const OpenGLViewTexture *, kMaxColors> colors{};
    gl::FramebufferAttachments attachments{};
    uint32_t colorSlots = 0;
    for (size_t index = 0; index < scope.colors.size(); ++index) {
        const uint32_t location = scope.colors[index].location;
        colors[location] = static_cast<const OpenGLViewTexture *>(images.colors[index]->native);
        attachments.colors[location] = colors[location]->name;
        colorSlots = std::max(colorSlots, location + 1);
    }
    const auto *depth = images.depth ? static_cast<const OpenGLViewTexture *>(images.depth->native) : nullptr;
    if (depth) {
        attachments.depth = depth->name;
        attachments.stencil = images.stencil;
    }
    auto held = device_.shared()->lock();
    ++state_.drawStateGeneration;
    gl::Driver &driver = state_.driver;
    auto &framebuffers = state_.attachmentFramebuffers;
    const auto cached = std::find_if(framebuffers.begin(), framebuffers.end(),
                                     [&](const auto &framebuffer) { return framebuffer.attachments == attachments; });
    if (cached != framebuffers.end())
        return Made<gl::Uint>{ok(cached->name)};
    gl::Uint name = 0;
    std::string error;
    if (!state_.createFramebuffer(name, error))
        return failure<gl::Uint>("cannot create the render scope's framebuffer: " + error);
    driver.bindFramebuffer(gl::kFramebuffer, name);
    std::array<gl::Enum, kMaxColors> drawBuffers{};
    for (uint32_t location = 0; location < kMaxColors; ++location) {
        if (colors[location])
            attachView(driver, gl::kColorAttachment0 + location, colors[location]);
        drawBuffers[location] = colors[location] ? gl::kColorAttachment0 + location : gl::kNone;
    }
    if (depth)
        attachView(driver, attachments.stencil ? gl::kDepthStencilAttachment : gl::kDepthAttachment, depth);
    driver.drawBuffers(static_cast<gl::Size>(std::max(1u, colorSlots)), drawBuffers.data());
    driver.readBuffer(colorSlots ? drawBuffers[0] : gl::kNone);
    const bool complete = driver.checkFramebufferStatus(gl::kFramebuffer) == gl::kFramebufferComplete;
    driver.bindFramebuffer(gl::kFramebuffer, 0);
    if (!complete) {
        state_.destroyFramebuffer(name);
        return failure<gl::Uint>("the render scope's framebuffer is incomplete");
    }
    framebuffers.push_back({attachments, name});
    return Made<gl::Uint>{ok(name)};
}

// Binds the scope's framebuffer and applies its clears, and returns the scope's checked views. glClearBuffer honors
// the write masks and the scissor test.
Made<ScopeImages> Builder::beginRendering(const runtime::GraphicsRenderScope &scope) {
    for (const VernonRuntimeProviderColorAttachment &color : scope.colors)
        if (color.location >= kMaxColors)
            return failure<ScopeImages>("a color attachment's location is out of range");
    auto checked = checkRenderScope(scope, nativeIdentity(&state_, kNativeViewReference));
    if (checked.isErr())
        return failure<ScopeImages>("the render scope is invalid: " + checked.error());
    const ScopeImages &images = checked.value();
    // An image the recording writes before reading it needs no defined contents when a call begins.
    for (const uint64_t image : images.readImages)
        if (!written_.count(image))
            recording_.reads.push_back(image);
    for (const uint64_t image : images.attachmentImages)
        if (written_.insert(image).second)
            recording_.writes.push_back(image);
    auto framebuffer = scopeFramebuffer(scope, images);
    if (framebuffer.isErr())
        return failure<ScopeImages>(std::move(framebuffer).error());
    set(Tracker::Framebuffer, 0, {framebuffer.value()}, {Op::BindFramebuffer, framebuffer.value()});
    // Desktop GL encodes sRGB attachment writes and clears only while GL_FRAMEBUFFER_SRGB is enabled; OpenGL ES always
    // encodes.
    if (caps_.desktop)
        capability(gl::kFramebufferSrgb, true);
    const VernonRuntimeProviderDepthStencilAttachment *depth = scope.depth ? &*scope.depth : nullptr;
    const bool depthClear = depth && depth->depth_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    const bool stencilClear =
        depth && images.stencil && depth->stencil_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    bool colorClear = false;
    for (const VernonRuntimeProviderColorAttachment &color : scope.colors)
        colorClear = colorClear || color.load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
    if (colorClear || depthClear || stencilClear) {
        setAll(Tracker::ColorMask, {VERNON_RHI_COLOR_WRITE_ALL}, {Op::ColorMask, VERNON_RHI_COLOR_WRITE_ALL});
        set(Tracker::DepthMask, 0, {1}, {Op::DepthMask, 1});
        set(Tracker::StencilMask, gl::kFront, {UINT32_MAX}, {Op::StencilMask, gl::kFront, UINT32_MAX});
        set(Tracker::StencilMask, gl::kBack, {UINT32_MAX}, {Op::StencilMask, gl::kBack, UINT32_MAX});
        capability(gl::kScissorTest, true);
        set(Tracker::Scissor, 0, {0, 0, scope.width, scope.height}, {Op::Scissor, 0, 0, scope.width, scope.height});
        for (size_t index = 0; index < scope.colors.size(); ++index) {
            const VernonRuntimeProviderColorAttachment &color = scope.colors[index];
            if (color.load_operation != VERNON_RUNTIME_PROVIDER_LOAD_CLEAR)
                continue;
            const vernon::formats::Format *entry = vernon::formats::find(images.colors[index]->view.format);
            const auto component = entry ? entry->componentClass : vernon::formats::ComponentClass::Float;
            // glClearBufferfv on an integer color buffer is undefined, so each component class uses its own call.
            const uint32_t kind = component == vernon::formats::ComponentClass::Uint   ? 2
                                  : component == vernon::formats::ComponentClass::Sint ? 1
                                                                                       : 0;
            add({Op::ClearColor, color.location, kind, static_cast<uint32_t>(recording_.clears.size())});
            recording_.clears.push_back(color.clear_color);
        }
        if (depthClear || stencilClear)
            add({Op::ClearDepthStencil, depthClear ? 1u : 0u, stencilClear ? 1u : 0u, floatBits(depth->clear_depth),
                 depth->clear_stencil});
    }
    return Made<ScopeImages>{ok(std::move(checked).value())};
}

Made<void> Builder::buildScope(size_t begin, size_t end) {
    const std::vector<PlannedPass *> &passes = planned_.passes;
    const runtime::GraphicsRenderScope &scope = planned_.scopes[passes[begin]->scope];
    std::string label;
    for (size_t at = begin; at < end; ++at)
        label += (at == begin ? "" : "+") + nodeLabel(*passes[at]->executable->source().node);
    debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, label);
    auto begun = beginRendering(scope);
    if (begun.isErr())
        return failure<void>("pass '" + passes[begin]->executable->source().node->name + "': " + begun.error());
    const ScopeImages opened = std::move(begun).value();
    const bool memberGroups = end - begin > 1;
    for (size_t at = begin; at < end; ++at) {
        const PlannedPass &pass = *passes[at];
        if (memberGroups)
            debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, nodeLabel(*pass.executable->source().node));
        for (const PlannedDraw &planned : pass.draws) {
            for (const PlannedDebug &command : planned.debug)
                debug(command.kind, command.label);
            if (auto drawn = draw(pass, planned); drawn.isErr())
                return failure<void>("pass '" + pass.executable->source().node->name + "': " + drawn.error());
        }
        for (const PlannedDebug &command : pass.trailingDebug)
            debug(command.kind, command.label);
        if (memberGroups)
            debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
    }
    if (state_.driver.invalidateFramebuffer) {
        const auto first = static_cast<uint32_t>(recording_.discards.size());
        for (size_t index = 0; index < scope.colors.size(); ++index)
            if (scope.colors[index].store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD &&
                renderAreaCovers(*opened.colors[index], scope))
                recording_.discards.push_back(gl::kColorAttachment0 + scope.colors[index].location);
        if (scope.depth && renderAreaCovers(*opened.depth, scope)) {
            if (scope.depth->depth_store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD)
                recording_.discards.push_back(gl::kDepthAttachment);
            if (opened.stencil && scope.depth->stencil_store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD)
                recording_.discards.push_back(gl::kStencilAttachment);
        }
        const auto count = static_cast<uint32_t>(recording_.discards.size()) - first;
        if (count)
            add({Op::Invalidate, first, count});
    }
    debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
    graphicsBefore_ = true;
    return Made<void>{ok()};
}

Made<std::unique_ptr<Recording>>
OpenGLDevice::build(const OpenGLCommandBuffer &commands, const PlannedPasses &planned,
                    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches, bool timed) {
    auto recording = std::make_unique<Recording>();
    recording->context = shared_;
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
Made<void> OpenGLDevice::planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
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
        return failure<void>(name + "the Stage has no OpenGL pipeline");
    const auto &state = ::runtimeBackendState<runtime::OpenGLPipelineState>(stage);
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

// Writes the bytes a call varies: the sites' inline values, the dispatches' group counts and the host draw streams'
// records.
Made<void> OpenGLDevice::fill(Recording &recording, const PlannedPasses &planned,
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
            const auto &state = ::runtimeBackendState<runtime::OpenGLPipelineState>(stage);
            filled = runtime::fillPreparedComputeBindingValue(state.computePlan, stage.bindingProjection,
                                                              *launches[site.pass], site.slot, value, error);
        }
        if (!filled)
            return failure<void>("cannot fill a binding: " + error);
        if (!value.payload.inline_value.data || value.payload.inline_value.size != site.size)
            return failure<void>("an inline binding value has the wrong size");
        std::memcpy((site.values ? recording.values.data() : recording.arena.data()) + site.offset,
                    value.payload.inline_value.data, site.size);
    }
    for (size_t index = 0; index < recording.groups.size(); ++index) {
        const VernonLaunchSize &grid = launches[recording.dispatchPasses[index]]->grid;
        recording.groups[index] = {grid.x, grid.y, grid.z};
    }
    const gl::Driver &driver = state_.driver;
    for (const HostStream &host : recording.streams) {
        const runtime::GraphicsDrawStream &stream = host.draw->stream;
        const VernonRuntimeProviderDrawRecord *records = stream.hostRecords + stream.firstRecord;
        const uint32_t count = std::min(stream.hostCounts[stream.countIndex], stream.maxRecords);
        // Checked as the call reads them: O(drawn records).
        for (uint32_t index = 0; index < count; ++index) {
            const VernonRuntimeProviderDrawRecord &record = records[index];
            if (uint64_t{record.first_index} + record.index_count > host.indexCapacity)
                return failure<void>("a host draw stream's record exceeds its index buffer");
            if (host.indirect || !record.index_count || !record.instance_count)
                continue;
            if (record.base_vertex != 0 && !driver.drawElementsInstancedBaseVertex)
                return failure<void>("the device draws no base vertices");
            for (const InstancedBinding &instanced : host.instanced)
                if (record.first_instance % instanced.divisor != 0)
                    return failure<void>("a record's first instance is not a multiple of a vertex divisor");
        }
        if (!host.indirect)
            continue;
        auto *target = reinterpret_cast<VernonRuntimeProviderDrawRecord *>(recording.arena.data() + host.records);
        std::memcpy(target, records, count * kRecordBytes);
        // The records' first indices count from the start of the bound buffer.
        const auto shift = static_cast<uint32_t>(host.indexOffset / sizeof(uint32_t));
        if (shift)
            for (uint32_t index = 0; index < count; ++index)
                target[index].first_index += shift;
        if (host.counted)
            std::memcpy(recording.arena.data() + host.count, &count, sizeof(count));
        else
            std::memset(target + count, 0, (stream.maxRecords - count) * kRecordBytes);
    }
    return Made<void>{ok()};
}

// The recording's slots in turn; a new one while it has fewer than kSlots.
Slot &OpenGLDevice::acquire(Recording &recording) {
    if (recording.slots.size() < kSlots) {
        std::unique_ptr<Slot> slot = shared_->pooled();
        if (!slot)
            slot = std::make_unique<Slot>();
        recording.slots.push_back(std::move(slot));
        return *recording.slots.back();
    }
    Slot &slot = *recording.slots[recording.nextSlot];
    recording.nextSlot = (recording.nextSlot + 1) % recording.slots.size();
    return slot;
}

// A host stream's records drawn one by one. Without base-instance draws, its per-instance bindings move to each
// record's first instance.
void OpenGLDevice::hostDraws(const Recording &, const HostStream &host) {
    gl::Driver &driver = state_.driver;
    const runtime::GraphicsDrawStream &stream = host.draw->stream;
    const VernonRuntimeProviderDrawRecord *records = stream.hostRecords + stream.firstRecord;
    const uint32_t count = std::min(stream.hostCounts[stream.countIndex], stream.maxRecords);
    StageObjects &stage = *host.stage;
    uint32_t applied = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const VernonRuntimeProviderDrawRecord &record = records[index];
        if (!record.index_count || !record.instance_count)
            continue;
        if (!host.instanced.empty() && applied != record.first_instance) {
            for (const InstancedBinding &instanced : host.instanced) {
                const uint64_t offset =
                    instanced.offset + uint64_t{record.first_instance / instanced.divisor} * instanced.stride;
                if (host.separate) {
                    driver.bindVertexBuffer(instanced.binding, instanced.name, static_cast<gl::IntPtr>(offset),
                                            static_cast<gl::Size>(instanced.stride));
                    continue;
                }
                driver.bindBuffer(gl::kArrayBuffer, instanced.name);
                for (const VertexAttribute &attribute : stage.attributes)
                    if (attribute.layout.binding == instanced.binding) {
                        const AttributeFormat format = attributeFormat(attribute.layout);
                        attribPointer(driver, attribute.layout.location, format.components, format.type, format.kind,
                                      static_cast<gl::Size>(instanced.stride),
                                      offset + attribute.layout.relative_offset);
                    }
            }
            applied = record.first_instance;
        }
        if (stage.baseInstanceLocation >= 0 && stage.baseInstance != record.first_instance) {
            driver.uniform1i(stage.baseInstanceLocation, static_cast<gl::Int>(record.first_instance));
            stage.baseInstance = record.first_instance;
        }
        const auto *indices = reinterpret_cast<const void *>(
            static_cast<uintptr_t>(host.indexOffset + uint64_t{record.first_index} * sizeof(uint32_t)));
        const auto indexCount = static_cast<gl::Size>(record.index_count);
        const auto instances = static_cast<gl::Size>(record.instance_count);
        if (record.first_instance && host.baseInstances)
            driver.drawElementsInstancedBaseVertexBaseInstance(host.topology, indexCount, gl::kUnsignedInt, indices,
                                                               instances, record.base_vertex, record.first_instance);
        else if (record.base_vertex)
            driver.drawElementsInstancedBaseVertex(host.topology, indexCount, gl::kUnsignedInt, indices, instances,
                                                   record.base_vertex);
        else
            driver.drawElementsInstanced(host.topology, indexCount, gl::kUnsignedInt, indices, instances);
    }
}

void OpenGLDevice::execute(const Recording &recording, const Slot &slot, const std::vector<gl::Uint> &queries) {
    gl::Driver &driver = state_.driver;
    const auto buffer = [&](gl::Uint name) { return name == kArena ? slot.arena : name == kZeros ? zeros_ : name; };
    const auto pointer = [](uint64_t offset) { return reinterpret_cast<const void *>(static_cast<uintptr_t>(offset)); };
    for (const Command &op : recording.commands) {
        switch (op.op) {
        case Op::UseProgram:
            driver.useProgram(op.a);
            break;
        case Op::Enable:
            driver.enable(op.a);
            break;
        case Op::Disable:
            driver.disable(op.a);
            break;
        case Op::Enablei:
            driver.enablei(op.a, op.b);
            break;
        case Op::Disablei:
            driver.disablei(op.a, op.b);
            break;
        case Op::CullFace:
            driver.cullFace(op.a);
            break;
        case Op::FrontFace:
            driver.frontFace(op.a);
            break;
        case Op::PolygonOffset:
            driver.polygonOffset(bitsFloat(op.a), bitsFloat(op.b));
            break;
        case Op::DepthFunc:
            driver.depthFunc(op.a);
            break;
        case Op::DepthMask:
            driver.depthMask(static_cast<gl::Boolean>(op.a));
            break;
        case Op::StencilFunc:
            driver.stencilFuncSeparate(op.a, op.b, static_cast<gl::Int>(op.c), op.d);
            break;
        case Op::StencilOp:
            driver.stencilOpSeparate(op.a, op.b, op.c, op.d);
            break;
        case Op::StencilMask:
            driver.stencilMaskSeparate(op.a, op.b);
            break;
        case Op::BlendFunc:
            driver.blendFuncSeparate(op.a, op.b, op.c, op.d);
            break;
        case Op::BlendFunci:
            driver.blendFuncSeparatei(op.e, op.a, op.b, op.c, op.d);
            break;
        case Op::BlendEquation:
            driver.blendEquationSeparate(op.a, op.b);
            break;
        case Op::BlendEquationi:
            driver.blendEquationSeparatei(op.e, op.a, op.b);
            break;
        case Op::ColorMask:
            driver.colorMask((op.a & VERNON_RHI_COLOR_WRITE_RED) != 0, (op.a & VERNON_RHI_COLOR_WRITE_GREEN) != 0,
                             (op.a & VERNON_RHI_COLOR_WRITE_BLUE) != 0, (op.a & VERNON_RHI_COLOR_WRITE_ALPHA) != 0);
            break;
        case Op::ColorMaski:
            driver.colorMaski(op.e, (op.a & VERNON_RHI_COLOR_WRITE_RED) != 0,
                              (op.a & VERNON_RHI_COLOR_WRITE_GREEN) != 0, (op.a & VERNON_RHI_COLOR_WRITE_BLUE) != 0,
                              (op.a & VERNON_RHI_COLOR_WRITE_ALPHA) != 0);
            break;
        case Op::Viewport:
            driver.viewport(static_cast<gl::Int>(op.a), static_cast<gl::Int>(op.b), static_cast<gl::Size>(op.c),
                            static_cast<gl::Size>(op.d));
            break;
        case Op::Scissor:
            driver.scissor(static_cast<gl::Int>(op.a), static_cast<gl::Int>(op.b), static_cast<gl::Size>(op.c),
                           static_cast<gl::Size>(op.d));
            break;
        case Op::BindFramebuffer:
            driver.bindFramebuffer(gl::kFramebuffer, op.a);
            break;
        case Op::ClearColor: {
            const VernonClearColor &color = recording.clears[op.c];
            const auto location = static_cast<gl::Int>(op.a);
            if (op.b == 2)
                driver.clearBufferuiv(gl::kColor, location, color.u32);
            else if (op.b == 1)
                driver.clearBufferiv(gl::kColor, location, color.i32);
            else
                driver.clearBufferfv(gl::kColor, location, color.f32);
            break;
        }
        case Op::ClearDepthStencil: {
            const float depth = bitsFloat(op.c);
            const auto stencil = static_cast<gl::Int>(op.d);
            if (op.a && op.b)
                driver.clearBufferfi(gl::kDepthStencil, 0, depth, stencil);
            else if (op.a)
                driver.clearBufferfv(gl::kDepth, 0, &depth);
            else
                driver.clearBufferiv(gl::kStencil, 0, &stencil);
            break;
        }
        case Op::Invalidate:
            driver.invalidateFramebuffer(gl::kFramebuffer, static_cast<gl::Size>(op.b),
                                         recording.discards.data() + op.a);
            break;
        case Op::BindVertexArray:
            driver.bindVertexArray(op.a);
            break;
        case Op::BindBuffer:
            driver.bindBuffer(op.a, buffer(op.b));
            break;
        case Op::BindBufferRange:
            if (op.size == 0)
                driver.bindBufferBase(op.a, op.b, buffer(op.c));
            else
                driver.bindBufferRange(op.a, op.b, buffer(op.c), static_cast<gl::IntPtr>(op.offset),
                                       static_cast<gl::SizePtr>(op.size));
            break;
        case Op::ActiveTexture:
            driver.activeTexture(gl::kTexture0 + op.a);
            break;
        case Op::BindTexture:
            driver.bindTexture(op.a, op.b);
            break;
        case Op::BindSampler:
            driver.bindSampler(op.a, op.b);
            break;
        case Op::BindImage:
            driver.bindImageTexture(op.a, op.b, 0, static_cast<unsigned char>(op.c), 0, op.d, op.e);
            break;
        case Op::AttribMask: {
            uint64_t changed = enabled_ ^ op.offset;
            for (gl::Uint location = 0; changed; ++location, changed >>= 1)
                if (changed & 1) {
                    if (op.offset & (uint64_t{1} << location))
                        driver.enableVertexAttribArray(location);
                    else
                        driver.disableVertexAttribArray(location);
                }
            enabled_ = op.offset;
            break;
        }
        case Op::AttribFormat:
            if (op.e == 1)
                driver.vertexAttribIFormat(op.a, static_cast<gl::Int>(op.b), op.c, op.d);
            else if (op.e == 2)
                driver.vertexAttribLFormat(op.a, static_cast<gl::Int>(op.b), op.c, op.d);
            else
                driver.vertexAttribFormat(op.a, static_cast<gl::Int>(op.b), op.c, 0, op.d);
            break;
        case Op::AttribBinding:
            driver.vertexAttribBinding(op.a, op.b);
            break;
        case Op::BindingDivisor:
            driver.vertexBindingDivisor(op.a, op.b);
            break;
        case Op::VertexBuffer:
            driver.bindVertexBuffer(op.a, op.b, static_cast<gl::IntPtr>(op.offset), static_cast<gl::Size>(op.c));
            break;
        case Op::AttribPointer:
            attribPointer(driver, op.a, static_cast<gl::Int>(op.b), op.c, op.e, static_cast<gl::Size>(op.d), op.offset);
            break;
        case Op::AttribDivisor:
            driver.vertexAttribDivisor(op.a, op.b);
            break;
        case Op::Uniform: {
            StageObjects &stage = *op.stage;
            const SlotPlan &plan = stage.slots[op.a];
            const uint8_t *bytes = recording.values.data() + op.offset;
            std::vector<uint8_t> &held = stage.uniforms[op.a];
            if (held.size() == plan.size && std::memcmp(held.data(), bytes, plan.size) == 0)
                break;
            uploadNamedUniform(driver, plan.location, plan, op.b != 0, bytes);
            held.assign(bytes, bytes + plan.size);
            break;
        }
        case Op::BaseInstance:
            if (op.stage->baseInstance != op.a) {
                driver.uniform1i(op.stage->baseInstanceLocation, static_cast<gl::Int>(op.a));
                op.stage->baseInstance = op.a;
            }
            break;
        case Op::DrawArrays:
            if (op.f == 2)
                driver.drawArraysInstancedBaseInstance(op.a, 0, static_cast<gl::Size>(op.b),
                                                       static_cast<gl::Size>(op.c), op.d);
            else if (op.f == 1)
                driver.drawArraysInstanced(op.a, 0, static_cast<gl::Size>(op.b), static_cast<gl::Size>(op.c));
            else
                driver.drawArrays(op.a, 0, static_cast<gl::Size>(op.b));
            break;
        case Op::DrawElements:
            if (op.f == 2)
                driver.drawElementsInstancedBaseVertexBaseInstance(op.a, static_cast<gl::Size>(op.b), gl::kUnsignedInt,
                                                                   pointer(op.offset), static_cast<gl::Size>(op.c),
                                                                   static_cast<gl::Int>(op.d), op.e);
            else if (op.f == 1)
                driver.drawElementsInstancedBaseVertex(op.a, static_cast<gl::Size>(op.b), gl::kUnsignedInt,
                                                       pointer(op.offset), static_cast<gl::Size>(op.c),
                                                       static_cast<gl::Int>(op.d));
            else
                driver.drawElementsInstanced(op.a, static_cast<gl::Size>(op.b), gl::kUnsignedInt, pointer(op.offset),
                                             static_cast<gl::Size>(op.c));
            break;
        case Op::MultiDraw:
            driver.multiDrawElementsIndirect(op.a, gl::kUnsignedInt, pointer(op.offset), static_cast<gl::Size>(op.b),
                                             static_cast<gl::Size>(kRecordBytes));
            break;
        case Op::MultiDrawCount:
            driver.multiDrawElementsIndirectCount(op.a, gl::kUnsignedInt, pointer(op.offset),
                                                  static_cast<gl::IntPtr>(op.size), static_cast<gl::Size>(op.b),
                                                  static_cast<gl::Size>(kRecordBytes));
            break;
        case Op::HostDraws:
            hostDraws(recording, recording.streams[op.a]);
            break;
        case Op::Dispatch: {
            const std::array<uint32_t, 3> &groups = recording.groups[op.a];
            driver.dispatchCompute(groups[0], groups[1], groups[2]);
            break;
        }
        case Op::Barrier:
            driver.memoryBarrier(op.a);
            break;
        case Op::CopyBuffer:
            driver.bindBuffer(gl::kCopyReadBuffer, buffer(op.a));
            driver.bindBuffer(gl::kCopyWriteBuffer, buffer(op.b));
            driver.copyBufferSubData(gl::kCopyReadBuffer, gl::kCopyWriteBuffer, static_cast<gl::IntPtr>(op.readOffset),
                                     static_cast<gl::IntPtr>(op.offset), static_cast<gl::SizePtr>(op.size));
            break;
        case Op::Timestamp:
            driver.queryCounter(queries[op.a], gl::kTimestamp);
            break;
        case Op::DebugPush:
            state_.pushDebugGroup(recording.labels[op.a]);
            break;
        case Op::DebugPop:
            state_.popDebugGroup();
            break;
        case Op::DebugMarker:
            state_.insertDebugMarker(recording.labels[op.a]);
            break;
        }
    }
}

Made<TimelinePoint> OpenGLDevice::submit(const CommandBuffer &commands, const uint8_t *record, size_t) {
    const auto &buffer = static_cast<const OpenGLCommandBuffer &>(commands);
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
        std::unique_ptr<Recording> recorded = std::move(built).value();
        recorded->key = std::move(key);
        recorded->plans = std::move(plans);
        if (cacheable) {
            if (buffer.recordings.size() >= kRecordings)
                buffer.recordings.pop_front();
            buffer.recordings.push_back(std::move(recorded));
            recording = buffer.recordings.back().get();
        } else {
            once = std::move(recorded);
            recording = once.get();
        }
    }
    if (auto filled = fill(*recording, planned, launches); filled.isErr())
        return failure<TimelinePoint>(std::move(filled).error());
    if (anyUndefined(recording->reads.data(), recording->reads.size()))
        return failure<TimelinePoint>("a pass reads an image whose contents are undefined");
    Slot &slot = acquire(*recording);
    auto timing = recording->queries ? std::make_shared<TimingCapture>() : nullptr;
    gl::Sync fence{};
    {
        auto held = shared_->lock();
        gl::Driver &driver = state_.driver;
        const std::vector<uint8_t> &bytes = recording->arena;
        if (!bytes.empty()) {
            if (slot.capacity < bytes.size() || !slot.arena) {
                if (slot.arena)
                    driver.deleteBuffers(1, &slot.arena);
                slot.arena = 0;
                slot.capacity = 0;
                slot.uploaded.clear();
                driver.genBuffers(1, &slot.arena);
                if (!slot.arena)
                    return failure<TimelinePoint>("cannot create a recording's arena");
                slot.capacity = roundUp(bytes.size(), 4096);
                driver.bindBuffer(gl::kCopyWriteBuffer, slot.arena);
                driver.bufferData(gl::kCopyWriteBuffer, static_cast<gl::SizePtr>(slot.capacity), nullptr, kDynamicDraw);
            }
            if (slot.uploaded != bytes) {
                driver.bindBuffer(gl::kCopyWriteBuffer, slot.arena);
                driver.bufferSubData(gl::kCopyWriteBuffer, 0, static_cast<gl::SizePtr>(bytes.size()), bytes.data());
                slot.uploaded = bytes;
            }
        }
        std::vector<gl::Uint> queries;
        if (timing) {
            for (uint32_t query = 0; query < recording->queries; ++query)
                queries.push_back(state_.acquireQuery());
            // Reading the disjoint flag clears it, so that only disjoint events during these timers invalidate them.
            if (state_.disjointQueryable) {
                gl::Int disjoint = 0;
                driver.getIntegerv(gl::kGpuDisjoint, &disjoint);
            }
        }
        // Errors that other OpenGL code left pending would otherwise be reported as the passes'.
        if (state_.validation)
            for (int drained = 0; drained < 16 && state_.readError() != 0; ++drained) {
            }
        execute(*recording, slot, queries);
        ++state_.drawStateGeneration;
        if (state_.validation && state_.readError() != 0) {
            if (!queries.empty())
                state_.releaseQueries(queries);
            return failure<TimelinePoint>("the passes raised an OpenGL error");
        }
        fence = driver.fenceSync(gl::kSyncGpuCommandsComplete, 0);
        if (!fence) {
            if (!queries.empty())
                state_.releaseQueries(queries);
            return failure<TimelinePoint>("cannot fence the passes");
        }
        if (timing) {
            timing->queries = std::move(queries);
            timing->scopes = recording->timers;
        }
    }
    forgetUndefined(recording->writes.data(), recording->writes.size());
    uint64_t draws = 0;
    for (const PlannedPass *pass : planned.passes)
        draws += pass->draws.size();
    buffer.setTelemetry(planned.scopes.size(), draws);
    std::lock_guard<std::mutex> lock(mutex_);
    const TimelinePoint point = next_++;
    pending_.push_back({point, fence, std::move(timing)});
    return Made<TimelinePoint>{ok(point)};
}

} // namespace

std::unique_ptr<Device> createOpenGLDevice(VernonRuntimeContext &context) {
    if (context.backend != VERNON_RUNTIME_OPENGL && context.backend != VERNON_RUNTIME_OPENGL_ES)
        return nullptr;
    auto access = gl::deviceAccess(context.rhiDevice);
    if (access.isErr())
        return nullptr;
    auto lease = rhi::retainDeviceLease(context.rhiDevice);
    if (lease.isErr())
        return nullptr;
    auto labels = rhi::deviceEmitsLabels(context.rhiDevice);
    auto shared = std::make_shared<Context>(context.rhiDevice, access.value(), std::move(lease).value());
    Caps caps;
    gl::Uint vertexArray = 0;
    {
        auto held = shared->lock();
        gl::DeviceState &state = shared->state();
        const gl::Driver &driver = state.driver;
        caps.desktop = !state.embeddedProfile;
        caps.compute = state.supportsCompute() && driver.dispatchCompute && driver.memoryBarrier;
        caps.separateFormats = (caps.desktop ? state.apiVersionAtLeast(4, 3) : state.apiVersionAtLeast(3, 1)) &&
                               driver.bindVertexBuffer && driver.vertexAttribFormat && driver.vertexAttribIFormat &&
                               driver.vertexAttribBinding && driver.vertexBindingDivisor;
        // Device streams are multi-draws, core in OpenGL 4.3, counted in 4.6; OpenGL ES has none.
        caps.deviceStreams = caps.desktop && state.apiVersionAtLeast(4, 3) && driver.multiDrawElementsIndirect;
        caps.countedStreams =
            caps.deviceStreams && state.apiVersionAtLeast(4, 6) && driver.multiDrawElementsIndirectCount;
        caps.labels = labels.isOk() && labels.value();
        caps.timers = driver.queryCounter != nullptr;
        gl::Int value = 0;
        driver.getIntegerv(gl::kUniformBufferOffsetAlignment, &value);
        caps.uniformAlignment = value > 0 ? static_cast<uint64_t>(value) : 256;
        if (state.supportsCompute()) {
            value = 0;
            driver.getIntegerv(gl::kShaderStorageBufferOffsetAlignment, &value);
            caps.storageAlignment = value > 0 ? static_cast<uint64_t>(value) : 256;
        }
        if (caps.separateFormats) {
            value = 0;
            driver.getIntegerv(gl::kMaxVertexAttribRelativeOffset, &value);
            caps.maxRelativeOffset = static_cast<uint32_t>(std::max(value, 0));
        }
        value = 0;
        driver.getIntegerv(gl::kMaxVertexAttribs, &value);
        caps.maxLocations = std::min(static_cast<uint32_t>(std::max(value, 0)), kMaxLocations);
        std::string error;
        if (!state.createVertexArray(vertexArray, error))
            return nullptr;
    }
    return std::make_unique<OpenGLDevice>(context, std::move(shared), caps, vertexArray);
}

} // namespace vernon::hal
