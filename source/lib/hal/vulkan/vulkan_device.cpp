#include "hal/vulkan/vulkan_device.h"
#include "hal/native_buffers.h"
#include "hal/native_images.h"
#include "hal/vulkan/vulkan_resources.h"

#include "VernonSchedule.h"
#include "rhi/rhi_internal.h"
#include "rhi/vulkan_backend.h"
#include "runtime/backend_vulkan.h"
#include "runtime/compute_launch_planner.h"
#include "runtime/dispatch_contract.h"
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

namespace rv = rhi::vulkan;
using LayoutUse = rv::DeviceState::ImageLayoutUse;
using runtime::PreparedComputeBindingPlan;
using runtime::PreparedGraphicsBindingPlan;

constexpr VkImageLayout kUnusedLayout = rv::DeviceState::kUnusedLayout;
constexpr uint32_t kNone = UINT32_MAX;
constexpr size_t kMaxSets = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;
constexpr size_t kMaxColors = VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS;
// Recordings one command buffer keeps: a call reuses one whose passes, attachments and timing are its own, such as
// one per swapchain image.
constexpr size_t kRecordings = 8;
// The frame slots of one recording. A slot is written and submitted again only once its last submission completed.
constexpr size_t kSlots = 3;
constexpr uint64_t kRecordBytes = sizeof(VernonRuntimeProviderDrawRecord);

template <typename T> Made<T> failure(std::string message) { return Made<T>{err(std::move(message))}; }

Made<void> vulkanCall(VkResult result, const char *operation) {
    if (result == VK_SUCCESS)
        return Made<void>{ok()};
    return failure<void>(std::string(operation) + " failed (VkResult " + std::to_string(static_cast<int>(result)) +
                         ")");
}

std::string rhiMessage(const char *what, const RhiError &error) {
    return error.context.operation ? std::string(what) + " (" + error.context.operation + ")" : std::string(what);
}

uint64_t align(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

uint64_t roundUp(uint64_t value, uint64_t minimum) {
    uint64_t result = minimum;
    while (result < value)
        result *= 2;
    return result;
}

struct Slot;

// Destroys native objects once every submission made before they were retired completed. The device, the Stages'
// pipelines and the recordings share it, and any of them may be destroyed first. It also keeps the slots of dropped
// recordings for later ones, since a pass whose plan changes every call builds a recording every call.
class Retirer {
public:
    Retirer(VernonRhiDevice device, rv::DeviceState &state, ChildLease lease)
        : device_(device), state_(state), lease_(std::move(lease)) {}
    Retirer(const Retirer &) = delete;
    Retirer &operator=(const Retirer &) = delete;
    ~Retirer() { drain(); }

    // Keeps an idle slot, or destroys it when enough are kept.
    void pool(std::unique_ptr<Slot> slot);
    std::unique_ptr<Slot> pooled();

    VernonRhiDevice device() const { return device_; }
    rv::DeviceState &state() const { return state_; }

    void submitted(uint64_t serial) {
        std::lock_guard<std::mutex> lock(mutex_);
        submitted_ = std::max(submitted_, serial);
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
                if (const uint64_t serial = pending_.front().serial) {
                    auto completed = rv::submissionCompleted(device_, serial);
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
                (void)rv::waitForSubmission(device_, last);
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
        uint64_t serial{};
        std::function<void()> destroy;
    };

    static constexpr size_t kPooledSlots = 32;

    VernonRhiDevice device_;
    rv::DeviceState &state_;
    ChildLease lease_;
    std::mutex mutex_;
    std::deque<Pending> pending_;
    uint64_t submitted_{};
    std::vector<std::unique_ptr<Slot>> slots_;
};

VkShaderStageFlags shaderStages(uint32_t mask) {
    VkShaderStageFlags result = 0;
    if (mask & VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
        result |= VK_SHADER_STAGE_COMPUTE_BIT;
    if (mask & VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
        result |= VK_SHADER_STAGE_VERTEX_BIT;
    if (mask & VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
        result |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return result;
}

VkDescriptorType descriptorType(VernonRuntimeProviderBindingKind kind) {
    switch (kind) {
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
        // Compute inline values live in a storage buffer; graphics inline values are push constants.
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

bool packedBytes(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE || entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
           (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
            entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM);
}

VkFormat vertexFormat(uint32_t dtype, uint32_t components) {
    static constexpr VkFormat formats[][4] = {
        {VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT},
        {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT},
        {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
        {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
        {VK_FORMAT_R64_SFLOAT, VK_FORMAT_R64G64_SFLOAT, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED},
    };
    const int row = dtype == VERNON_RUNTIME_PROVIDER_I32   ? 0
                    : dtype == VERNON_RUNTIME_PROVIDER_U32 ? 1
                    : dtype == VERNON_RUNTIME_PROVIDER_F16 ? 2
                    : dtype == VERNON_RUNTIME_PROVIDER_F32 ? 3
                    : dtype == VERNON_RUNTIME_PROVIDER_F64 ? 4
                                                           : -1;
    return row < 0 || components == 0 || components > 4 ? VK_FORMAT_UNDEFINED : formats[row][components - 1];
}

std::optional<VkFormat> attachmentFormat(VernonTextureFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, rhi, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,  \
                      vulkan, ...)                                                                                     \
    case VERNON_TEXTURE_##NAME:                                                                                        \
        return vulkan == VK_FORMAT_UNDEFINED ? std::nullopt : std::optional<VkFormat>(static_cast<VkFormat>(vulkan));
#include "VernonFormats.def"
    }
    return std::nullopt;
}

// Shaders read their push constants at offsets from zero; a stage whose block the pipeline layout places at `delta`
// has every member offset of its push constant block moved by it.
bool relocatePushConstantOffsets(std::vector<uint32_t> &words, uint32_t delta) {
    if (delta == 0)
        return true;
    constexpr uint16_t kOpTypePointer = 32;
    constexpr uint16_t kOpVariable = 59;
    constexpr uint16_t kOpMemberDecorate = 72;
    constexpr uint32_t kPushConstantStorageClass = 9;
    constexpr uint32_t kOffsetDecoration = 35;
    uint32_t pointerType = 0;
    uint32_t block = 0;
    for (size_t index = 5; index < words.size();) {
        const auto count = static_cast<uint16_t>(words[index] >> 16);
        const auto opcode = static_cast<uint16_t>(words[index]);
        if (count == 0 || index + count > words.size())
            return false;
        if (opcode == kOpVariable && count >= 4 && words[index + 3] == kPushConstantStorageClass)
            pointerType = words[index + 1];
        index += count;
    }
    if (pointerType == 0)
        return true;
    for (size_t index = 5; index < words.size();) {
        const auto count = static_cast<uint16_t>(words[index] >> 16);
        const auto opcode = static_cast<uint16_t>(words[index]);
        if (opcode == kOpTypePointer && count >= 4 && words[index + 1] == pointerType &&
            words[index + 2] == kPushConstantStorageClass) {
            block = words[index + 3];
            break;
        }
        index += count;
    }
    if (block == 0)
        return false;
    for (size_t index = 5; index < words.size();) {
        const auto count = static_cast<uint16_t>(words[index] >> 16);
        const auto opcode = static_cast<uint16_t>(words[index]);
        if (opcode == kOpMemberDecorate && count >= 5 && words[index + 1] == block &&
            words[index + 3] == kOffsetDecoration)
            words[index + 4] += delta;
        index += count;
    }
    return true;
}

void imageDependency(VkImageLayout layout, VkPipelineStageFlags &stages, VkAccessFlags &access) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        access = 0;
        return;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        access = VK_ACCESS_TRANSFER_READ_BIT;
        return;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        access = VK_ACCESS_TRANSFER_WRITE_BIT;
        return;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        access = VK_ACCESS_SHADER_READ_BIT;
        return;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        return;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        return;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        stages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        access = VK_ACCESS_MEMORY_READ_BIT;
        return;
    default:
        stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        return;
    }
}

VkAttachmentLoadOp loadOperation(VernonRuntimeProviderLoadOperation operation) {
    switch (operation) {
    case VERNON_RUNTIME_PROVIDER_LOAD_CLEAR:
        return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE:
        return VK_ATTACHMENT_LOAD_OP_LOAD;
    default:
        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }
}

VkAttachmentStoreOp storeOperation(VernonRuntimeProviderStoreOperation operation) {
    return operation == VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ? VK_ATTACHMENT_STORE_OP_STORE
                                                               : VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

// The image tracking the layouts of `image`, an image or a view, and the subresources of it that `image` covers.
std::pair<rv::Image *, rv::ImageSubresources> layoutSubresources(rv::Image &image) {
    if (image.layoutOwner)
        return {image.layoutOwner, image.subresources};
    const uint32_t layers =
        image.mipLevels ? static_cast<uint32_t>(image.subresourceLayouts.size() / image.mipLevels) : 0;
    return {&image, {0, image.mipLevels, 0, layers}};
}

// The reference kinds of hal/native_buffers.h a Vulkan device resolves.
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

// The native object a reference of `state`'s device names: an rv::Buffer, a view's rv::Image or an rv::Sampler. The
// recording that names it stays valid only while the resource lives, which the keys of the plans it records
// guarantee.
template <typename T>
Made<T *> resolveNative(const rv::DeviceState &state, const VernonRuntimeProviderResourceReference &reference,
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
        native = vulkanBuffer(reference.resource.value);
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

struct PushRange {
    VkShaderStageFlags stages{};
    uint32_t offset{};
    uint32_t size{};
};

// Where a push value of a binding set goes in the pipeline's push constants.
struct PushPlacement {
    uint32_t dataOffset{};
    uint32_t offset{};
    uint32_t size{};
};

enum class SlotKind : uint8_t { Push, Inline, Buffer, Image, Sampler, CombinedSampler };

// How one prepared layout slot of a Stage binds. Graphics inline values are push constants placed at `dataOffset` of
// their set's push data; other packed values live in the slot arena; a sampler that shares its sampled image's
// binding is that image's combined image sampler, named by the image's `sampler`.
struct SlotPlan {
    SlotKind kind{};
    VernonRuntimeProviderBindingKind providerKind{};
    VkDescriptorType type{};
    uint32_t set{};
    uint32_t binding{};
    uint32_t size{};
    uint32_t dataOffset{};
    uint32_t sampler{kNone};
    VkFormat storageFormat{VK_FORMAT_UNDEFINED};
};

// Descriptor set `set` of a Stage: the slots in it, and what decides its values as PipelineGroups groups them.
struct SetPlan {
    VkDescriptorSetLayout layout{};
    std::vector<uint32_t> slots;
    std::vector<uint32_t> parameters;
    bool command{};
    bool descriptors{};
    uint32_t pushBytes{};
    std::vector<PushPlacement> placements;
};

// The HAL's native objects of one Stage, which live in its VulkanPipelineState; the variants are made on first use.
class StageObjects {
public:
    explicit StageObjects(std::shared_ptr<Retirer> retirer) : retirer_(std::move(retirer)) {}
    StageObjects(const StageObjects &) = delete;
    StageObjects &operator=(const StageObjects &) = delete;

    ~StageObjects() {
        std::vector<VkPipeline> pipelines;
        for (const Variant &variant : variants)
            pipelines.push_back(variant.pipeline);
        if (compute)
            pipelines.push_back(compute);
        std::vector<VkDescriptorSetLayout> layouts;
        for (const SetPlan &set : sets)
            layouts.push_back(set.layout);
        VkDevice device = retirer_->state().device;
        retirer_->retire([device, pipelines, layouts, modules = modules, layout = layout] {
            auto &api = rv::driver();
            for (VkPipeline pipeline : pipelines)
                api.destroyPipeline(device, pipeline, nullptr);
            if (layout)
                api.destroyPipelineLayout(device, layout, nullptr);
            for (VkDescriptorSetLayout set : layouts)
                if (set)
                    api.destroyDescriptorSetLayout(device, set, nullptr);
            for (VkShaderModule module : modules)
                if (module)
                    api.destroyShaderModule(device, module, nullptr);
        });
    }

    struct Variant {
        runtime::GraphicsVariantKey key;
        VkPipeline pipeline{};
    };

    bool graphics{};
    std::vector<SlotPlan> slots;
    std::vector<SetPlan> sets;
    VkPipelineLayout layout{};
    std::vector<PushRange> pushRanges;
    // Vertex and fragment, or compute.
    std::vector<VkShaderModule> modules;
    std::string vertexEntry;
    std::string fragmentEntry;
    VkPipeline compute{};
    std::vector<VernonRuntimeProviderVertexBinding> vertexBindings;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    std::mutex mutex;
    std::vector<Variant> variants;

private:
    std::shared_ptr<Retirer> retirer_;
};

struct RenderPassKey {
    std::vector<VkFormat> colors;
    std::vector<VkAttachmentLoadOp> loads;
    std::vector<VkAttachmentStoreOp> stores;
    VkFormat depth{VK_FORMAT_UNDEFINED};
    bool stencil{};
    VkAttachmentLoadOp depthLoad{VK_ATTACHMENT_LOAD_OP_DONT_CARE};
    VkAttachmentStoreOp depthStore{VK_ATTACHMENT_STORE_OP_DONT_CARE};
    VkAttachmentLoadOp stencilLoad{VK_ATTACHMENT_LOAD_OP_DONT_CARE};
    VkAttachmentStoreOp stencilStore{VK_ATTACHMENT_STORE_OP_DONT_CARE};

    std::vector<uint64_t> words() const {
        std::vector<uint64_t> result;
        for (size_t index = 0; index < colors.size(); ++index)
            result.insert(result.end(), {static_cast<uint64_t>(colors[index]), static_cast<uint64_t>(loads[index]),
                                         static_cast<uint64_t>(stores[index])});
        result.insert(result.end(), {static_cast<uint64_t>(depth), stencil ? 1u : 0u, static_cast<uint64_t>(depthLoad),
                                     static_cast<uint64_t>(depthStore), static_cast<uint64_t>(stencilLoad),
                                     static_cast<uint64_t>(stencilStore)});
        return result;
    }
};

enum class Op : uint8_t {
    QueueOrder,
    MemoryBarrier,
    ImageBarriers,
    CopyBuffer,
    CopyRegions,
    ResetQueries,
    Timestamp,
    DebugPush,
    DebugPop,
    DebugMarker,
    BeginRendering,
    EndRendering,
    BindPipeline,
    BindSet,
    Push,
    BindVertexBuffers,
    BindIndexBuffer,
    Viewport,
    Scissor,
    Stencil,
    Draw,
    DrawIndexed,
    DrawIndirect,
    DrawIndirectCount,
    HostDraws,
    Dispatch,
};

// One command of a recording; its arguments depend on `op`. A null buffer names the slot's arena.
struct Command {
    Op op{};
    uint32_t a{};
    uint32_t b{};
    uint32_t c{};
    uint32_t d{};
    uint32_t e{};
    VkBuffer buffer{};
    VkBuffer countBuffer{};
    uint64_t offset{};
    uint64_t countOffset{};
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
    const StageObjects *stage{};
};

struct Rendering {
    uint32_t colors{};
    std::array<VkImageView, kMaxColors> views{};
    std::array<VkAttachmentLoadOp, kMaxColors> loads{};
    std::array<VkAttachmentStoreOp, kMaxColors> stores{};
    // Colors by location, then depth.
    std::array<VkClearValue, kMaxColors + 1> clears{};
    VkImageView depthView{};
    bool stencil{};
    VkAttachmentLoadOp depthLoad{};
    VkAttachmentStoreOp depthStore{};
    VkAttachmentLoadOp stencilLoad{};
    VkAttachmentStoreOp stencilStore{};
    VkRect2D area{};
    // Without dynamic rendering.
    VkRenderPass renderPass{};
    VkFramebuffer framebuffer{};
};

// One descriptor of a set instance.
struct Write {
    uint32_t binding{};
    VkDescriptorType type{};
    VkBuffer buffer{};
    VkDeviceSize offset{};
    VkDeviceSize range{};
    VkImageView view{};
    VkImageLayout imageLayout{};
    VkSampler sampler{};
};

// A descriptor set of a recording and the push data of its set: one per binding group as PipelineGroups finds it, so
// draws whose groups' owners agree share it.
struct SetInstance {
    VkDescriptorSetLayout layout{};
    std::vector<Write> writes;
    uint32_t pushOffset{kNone};
};

// Inline bytes a call fills again: of a graphics draw's binding plan, or of compute pass `pass`'s launch.
struct Site {
    const PlannedDraw *draw{};
    const PreparedGraphicsBindingPlan *graphics{};
    uint32_t pass{kNone};
    uint32_t slot{};
    uint32_t size{};
    uint64_t offset{};
    bool push{};
};

// The records of a host draw stream, which a call copies into the arena; `counted` draws them with an indirect
// count, else the records past the count are zeroed.
struct HostStream {
    const PlannedDraw *draw{};
    uint64_t records{};
    uint64_t count{};
    uint64_t indexCapacity{};
    bool counted{};
};

struct QueryPool {
    VkDevice device{};
    VkQueryPool pool{};
    ~QueryPool() {
        if (pool)
            rv::driver().destroyQueryPool(device, pool, nullptr);
    }
};

// The scope timings of one timed submission, read once it completed and before its slot records again.
struct TimingCapture {
    std::shared_ptr<QueryPool> queries;
    std::vector<ScopeTiming> scopes;
    bool read{};
    bool valid{};
};

// One frame of a recording: its command buffer, descriptor sets, arena and timestamps. A pooled slot keeps its
// command pool, arena and descriptor pool, with what they hold.
struct Slot {
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkDescriptorPool descriptors{};
    uint32_t setCapacity{};
    std::map<VkDescriptorType, uint32_t> descriptorCapacity;
    std::vector<VkDescriptorSet> sets;
    rv::Buffer arena;
    uint64_t arenaCapacity{};
    uint8_t *mapped{};
    std::shared_ptr<QueryPool> queries;
    std::shared_ptr<TimingCapture> timing;
    uint64_t serial{};
    bool recorded{};
    // The push data the command buffer was recorded with.
    std::vector<uint8_t> push;
};

void Retirer::destroy(Slot &slot) {
    auto &api = rv::driver();
    if (slot.pool)
        api.destroyCommandPool(state_.device, slot.pool, nullptr);
    if (slot.descriptors)
        api.destroyDescriptorPool(state_.device, slot.descriptors, nullptr);
    state_.destroyBuffer(slot.arena);
}

void Retirer::pool(std::unique_ptr<Slot> slot) {
    // A capture of the slot's last submission may still read its queries, which a later recording would reset.
    slot->queries.reset();
    slot->timing.reset();
    slot->sets.clear();
    slot->serial = 0;
    slot->recorded = false;
    slot->push.clear();
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
// passes on the same attachments, submits them again after it wrote the bytes that vary. Layouts are the images'
// layouts as the commands begin and leave them.
struct Recording {
    std::shared_ptr<Retirer> retirer;
    std::vector<uint64_t> key;
    std::vector<uint64_t> plans;
    std::vector<std::shared_ptr<StageObjects>> stages;
    std::vector<Command> commands;
    std::vector<VkImageMemoryBarrier> imageBarriers;
    // The regions of the CopyRegions commands.
    std::vector<VkBufferCopy> copies;
    std::vector<std::string> labels;
    std::vector<Rendering> renderings;
    // (set index, instance) of the push data a Push command composes.
    std::vector<std::pair<uint32_t, uint32_t>> pushSets;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceSize> vertexOffsets;
    std::vector<SetInstance> sets;
    std::vector<Site> sites;
    // Arena bytes that never change, written once per slot.
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> fixed;
    std::vector<HostStream> streams;
    // (compute pass, arena offset) of each dispatch's group counts.
    std::vector<std::pair<uint32_t, uint64_t>> dispatches;
    std::vector<uint8_t> push;
    uint64_t arenaBytes{};
    std::unordered_map<rv::Image *, LayoutUse> layouts;
    // Serials of the images the commands read before writing them, and of the attachments they write.
    std::vector<uint64_t> reads;
    std::vector<uint64_t> writes;
    bool timed{};
    std::vector<ScopeTiming> timers;
    // Host draws recorded from the records themselves, so every call records again.
    bool volatileCommands{};
    std::map<VkDescriptorType, uint32_t> descriptorCounts;
    uint32_t descriptorSets{};
    rv::Buffer zeros;
    std::vector<VkFramebuffer> framebuffers;
    std::vector<std::unique_ptr<Slot>> slots;

    Recording() = default;
    Recording(const Recording &) = delete;
    Recording &operator=(const Recording &) = delete;

    ~Recording() {
        if (!retirer)
            return;
        rv::DeviceState &state = retirer->state();
        auto owned = std::make_shared<std::vector<std::unique_ptr<Slot>>>(std::move(slots));
        auto buffer = std::make_shared<rv::Buffer>(zeros);
        Retirer *keeper = retirer.get();
        // The Retirer runs what it retires before it is destroyed, so it outlives this.
        retirer->retire([&state, keeper, owned, buffer, framebuffers = framebuffers, stages = stages] {
            auto &api = rv::driver();
            for (std::unique_ptr<Slot> &slot : *owned)
                keeper->pool(std::move(slot));
            for (VkFramebuffer framebuffer : framebuffers)
                api.destroyFramebuffer(state.device, framebuffer, nullptr);
            state.destroyBuffer(*buffer);
        });
    }
};

class VulkanCommandBuffer final : public PassCommandBuffer {
public:
    using PassCommandBuffer::PassCommandBuffer;
    // Guarded by the pass cache's mutex; the most recently used last.
    mutable std::deque<std::unique_ptr<Recording>> recordings;
};

class VulkanDevice;

// Builds the recording of one call's planned passes.
class Builder {
public:
    Builder(VulkanDevice &device, Recording &recording, const PlannedPasses &planned,
            const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Made<void> build(const std::vector<const InternalBuffers::Internal *> &internals);

private:
    struct DrawPlan {
        const PlannedDraw *draw{};
        StageObjects *stage{};
        VkPipeline pipeline{};
        std::array<uint32_t, kMaxSets> sets{};
        uint32_t vertexFirst{};
        uint32_t vertexCount{};
        VernonRuntimeCoreDrawInvocation invocation{};
        VkBuffer indexBuffer{};
        uint64_t indexOffset{};
    };
    struct ImageUse {
        rv::Image *image{};
        VkImageLayout layout{};
    };
    using FillValue = std::function<bool(size_t slot, VernonRuntimeProviderBindingValue &value,
                                         std::vector<uint8_t> &storage, std::string &error)>;

    void add(Command command) { recording_.commands.push_back(command); }
    uint64_t arena(uint64_t size, uint64_t alignment) {
        const uint64_t offset = align(recording_.arenaBytes, alignment);
        recording_.arenaBytes = offset + size;
        return offset;
    }
    void debug(uint32_t kind, const std::string &label);
    void timer(bool begin);
    void transition(rv::Image &image, VkImageLayout target);
    void flushBarriers();
    Made<StageObjects *> stage(VernonStageExecutable &stage, bool graphics);
    template <typename T> Made<T *> nativeOf(const VernonRuntimeProviderResourceReference &reference, Kind kind) {
        return resolveNative<T>(state_, reference, kind, &resolved_);
    }
    Made<uint32_t> instance(StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                            const Site &source, std::vector<ImageUse> &images);
    Made<void> buildCompute(size_t at);
    Made<void> buildCopy(const PlannedCopy &copy);
    Made<void> buildScope(size_t begin, size_t end);
    Made<DrawPlan> planDraw(const PlannedPass &pass, const PlannedDraw &draw, std::vector<ImageUse> &images);
    Made<void> emitDraw(const DrawPlan &plan);
    Made<Rendering> rendering(const runtime::GraphicsRenderScope &scope);

    VulkanDevice &device_;
    rv::DeviceState &state_;
    Recording &recording_;
    const PlannedPasses &planned_;
    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches_;
    std::map<std::vector<const void *>, uint32_t> instances_;
    ResolvedCache resolved_;
    std::unordered_map<const VernonStageExecutable *, StageObjects *> stages_;
    std::vector<VkImageMemoryBarrier> barriers_;
    VkPipelineStageFlags sourceStages_{};
    VkPipelineStageFlags destinationStages_{};
    std::unordered_set<uint64_t> written_;
    uint32_t queries_{};
    bool graphicsBefore_{};
    // What the commands bound last, per bind point: graphics 0, compute 1.
    std::array<VkPipeline, 2> pipelines_{};
    std::array<VkPipelineLayout, 2> layouts_{};
    std::array<std::array<uint32_t, kMaxSets>, 2> bound_{};
    const StageObjects *pushStage_{};
    std::array<uint32_t, kMaxSets> pushed_{};
    std::optional<std::array<uint32_t, 4>> viewport_;
    std::optional<std::array<uint32_t, 4>> scissor_;
    std::optional<uint32_t> stencil_;
    VkBuffer indexBuffer_{};
    uint64_t indexOffset_{};
};

class VulkanDevice final : public Device {
public:
    VulkanDevice(VernonRuntimeContext &context, std::shared_ptr<Retirer> retirer)
        : context_(context), retirer_(std::move(retirer)), state_(retirer_->state()),
          buffers_(createVulkanDeviceBuffers(context.rhiDevice)), internals_(buffers_.get()) {}

    ~VulkanDevice() override {
        retirer_->drain();
        for (const auto &[key, pass] : renderPasses_)
            rv::driver().destroyRenderPass(state_.device, pass, nullptr);
    }

    rv::DeviceState &state() const { return state_; }
    VernonRhiDevice rhiDevice() const { return context_.rhiDevice; }
    const std::shared_ptr<Retirer> &retirer() const { return retirer_; }

    Made<std::unique_ptr<Executable>> createExecutable(const ExecutableSource &source) override {
        if (!source.graphics || !source.graphics->resolved || !source.graphics->resolved->stage)
            return failure<std::unique_ptr<Executable>>("Vulkan devices run Program nodes only");
        return Made<std::unique_ptr<Executable>>{ok(std::unique_ptr<Executable>(new PassExecutable(source.graphics)))};
    }

    Made<std::unique_ptr<Buffer>> allocate(uint64_t size) override {
        if (!buffers_)
            return failure<std::unique_ptr<Buffer>>("the Vulkan device allocates no buffers");
        return buffers_->allocate(size);
    }

    Made<void> upload(Buffer &buffer, const void *bytes, size_t size) override {
        return buffers_->upload(buffer, bytes, size);
    }

    // An upload copies the bytes to staging and the buffer in queue order, as an enqueued one must.
    Made<void> enqueueUpload(Buffer &buffer, const void *bytes, size_t size) override {
        return buffers_->upload(buffer, bytes, size);
    }

    Made<void> download(const Buffer &buffer, void *bytes, size_t size) override {
        return buffers_->download(buffer, bytes, size);
    }

    Made<std::unique_ptr<CommandBuffer>> record(const std::vector<DispatchCommand> &commands) override {
        if (auto checked = checkPassCommands(commands, "Vulkan"); checked.isErr())
            return Made<std::unique_ptr<CommandBuffer>>{err(std::move(checked).error())};
        return Made<std::unique_ptr<CommandBuffer>>{
            ok(std::unique_ptr<CommandBuffer>(new VulkanCommandBuffer(commands)))};
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
            if (rv::waitForSubmission(context_.rhiDevice, retired.serial).isErr())
                failed = true;
            else
                keepTimings(retired);
        }
        if (failed)
            return failure<void>("a pass submission failed on the device");
        return Made<void>{ok()};
    }

    // The objects of a Stage this device draws or dispatches with, made on first use.
    Made<std::shared_ptr<StageObjects>> stageObjects(VernonStageExecutable &stage, bool graphics);
    Made<VkPipeline> variant(StageObjects &stage, const runtime::GraphicsVariantKey &key);
    Made<VkRenderPass> renderPass(const RenderPassKey &key);

private:
    struct Pending {
        TimelinePoint point{};
        uint64_t serial{};
        std::shared_ptr<TimingCapture> timing;
    };

    static constexpr size_t kKeptTimings = 32;

    Made<std::unique_ptr<Recording>> build(const VulkanCommandBuffer &commands, const PlannedPasses &planned,
                                           const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches,
                                           bool timed);
    Made<void> planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
                          std::vector<uint64_t> &key);
    Made<Slot *> acquire(Recording &recording);
    Made<void> createSlot(Recording &recording, Slot &slot);
    Made<void> fill(Recording &recording, Slot &slot, const PlannedPasses &planned,
                    const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches);
    Made<void> recordCommands(Recording &recording, Slot &slot);

    // Called with mutex_ held, or before the slot of the capture records again.
    void capture(TimingCapture &timing) {
        if (timing.read)
            return;
        timing.read = true;
        const auto queries = static_cast<uint32_t>(timing.scopes.size() * 2);
        std::vector<uint64_t> ticks(queries);
        if (queries &&
            rv::driver().getQueryPoolResults(state_.device, timing.queries->pool, 0, queries,
                                             ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
            return;
        const uint64_t mask =
            state_.timestampValidBits >= 64 ? ~uint64_t{0} : (uint64_t{1} << state_.timestampValidBits) - 1;
        const auto nanoseconds = [&](uint64_t value) {
            return static_cast<uint64_t>(static_cast<double>(value & mask) * state_.timestampPeriod);
        };
        const uint64_t origin = queries ? nanoseconds(ticks[0]) : 0;
        for (size_t index = 0; index < timing.scopes.size(); ++index) {
            ScopeTiming &scope = timing.scopes[index];
            scope.begin = nanoseconds(ticks[2 * index]);
            scope.end = nanoseconds(ticks[2 * index + 1]);
            scope.begin -= std::min(origin, scope.begin);
            scope.end -= std::min(origin, scope.end);
            // Drivers write begin and end at different pipeline stages, so an empty scope may end before it begins.
            scope.end = std::max(scope.end, scope.begin);
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
            auto completed = rv::submissionCompleted(context_.rhiDevice, pending_.front().serial);
            if (completed.isErr() || !completed.value())
                return;
            keepTimings(pending_.front());
            pending_.pop_front();
        }
    }

    VernonRuntimeContext &context_;
    std::shared_ptr<Retirer> retirer_;
    rv::DeviceState &state_;
    std::unique_ptr<DeviceBuffers> buffers_;
    InternalBuffers internals_;
    std::mutex mutex_;
    std::deque<Pending> pending_;
    std::deque<std::pair<TimelinePoint, std::optional<std::vector<ScopeTiming>>>> timings_;
    TimelinePoint next_{1};
    std::mutex renderPassMutex_;
    std::map<std::vector<uint64_t>, VkRenderPass> renderPasses_;
};

Made<std::shared_ptr<StageObjects>> buildStageObjects(const std::shared_ptr<Retirer> &retirer,
                                                      VernonStageExecutable &stage, runtime::VulkanPipelineState &state,
                                                      bool graphics) {
    using Built = Made<std::shared_ptr<StageObjects>>;
    auto &api = rv::driver();
    rv::DeviceState &device = retirer->state();
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
        if (layout.set >= kMaxSets || layout.array_count != 1 ||
            descriptorType(layout.kind) == VK_DESCRIPTOR_TYPE_MAX_ENUM)
            return failure<std::shared_ptr<StageObjects>>("the Stage has a binding Vulkan devices do not bind");
        setCount = std::max(setCount, layout.set + 1);
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
        slot.type = descriptorType(layout.kind);
        slot.set = layout.set;
        slot.binding = layout.binding;
        slot.size = layout.element_size;
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
            const std::optional<VkFormat> format = attachmentFormat(layout.storage_image_format);
            if (!format)
                return failure<std::shared_ptr<StageObjects>>("a storage image's format has no Vulkan format");
            slot.storageFormat = *format;
        }
        if (graphics && layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
            layout.stage_mask != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE) {
            const uint32_t alignment = layout.element_alignment;
            if (!alignment || (alignment & (alignment - 1)) != 0 || layout.element_size == 0)
                return failure<std::shared_ptr<StageObjects>>("an inline value's alignment is invalid");
            slot.kind = SlotKind::Push;
            slot.dataOffset = set.pushBytes;
            set.pushBytes += layout.element_size;
        } else if (packedBytes(layout)) {
            slot.kind = SlotKind::Inline;
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            slot.kind = SlotKind::Buffer;
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) {
            slot.kind = SlotKind::Sampler;
        } else {
            slot.kind = SlotKind::Image;
        }
    }
    for (SlotPlan &image : objects->slots) {
        if (image.providerKind != VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)
            continue;
        for (uint32_t index = 0; index < objects->slots.size(); ++index) {
            SlotPlan &sampler = objects->slots[index];
            if ((sampler.kind == SlotKind::Sampler || sampler.kind == SlotKind::CombinedSampler) &&
                sampler.set == image.set && sampler.binding == image.binding) {
                image.sampler = index;
                image.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                sampler.kind = SlotKind::CombinedSampler;
            }
        }
    }
    std::vector<VkDescriptorSetLayout> setLayouts;
    for (SetPlan &set : objects->sets) {
        std::sort(set.parameters.begin(), set.parameters.end());
        set.parameters.erase(std::unique(set.parameters.begin(), set.parameters.end()), set.parameters.end());
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t index : set.slots) {
            const SlotPlan &slot = objects->slots[index];
            if (slot.kind == SlotKind::Push)
                continue;
            const VkShaderStageFlags stages = shaderStages(layouts[index].stage_mask);
            const VkDescriptorType type =
                slot.kind == SlotKind::CombinedSampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : slot.type;
            const auto existing = std::find_if(bindings.begin(), bindings.end(),
                                               [&](const auto &binding) { return binding.binding == slot.binding; });
            // The compiler links one descriptor per Program Value across a pipeline, so a repeated binding of the
            // same type is one value read by several stages.
            if (existing == bindings.end())
                bindings.push_back({slot.binding, type, 1, stages, nullptr});
            else if (existing->descriptorType == type)
                existing->stageFlags |= stages;
            else
                return failure<std::shared_ptr<StageObjects>>("the Stage binds two descriptors at one binding");
        }
        set.descriptors = !bindings.empty();
        const VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
                                                   static_cast<uint32_t>(bindings.size()), bindings.data()};
        if (auto created = vulkanCall(api.createDescriptorSetLayout(device.device, &info, nullptr, &set.layout),
                                      "vkCreateDescriptorSetLayout");
            created.isErr())
            return Built{err(std::move(created).error())};
        setLayouts.push_back(set.layout);
    }
    // Push values are placed sorted by stage, then by entry argument order, which shader push constant members
    // follow, while each stage's block is relocated to its offset.
    std::vector<uint32_t> pushSlots;
    for (uint32_t index = 0; index < objects->slots.size(); ++index)
        if (objects->slots[index].kind == SlotKind::Push)
            pushSlots.push_back(index);
    std::stable_sort(pushSlots.begin(), pushSlots.end(), [&](uint32_t left, uint32_t right) {
        if (layouts[left].stage_mask != layouts[right].stage_mask)
            return layouts[left].stage_mask < layouts[right].stage_mask;
        return layouts[left].argument_index < layouts[right].argument_index;
    });
    uint32_t pushSize = 0;
    std::map<uint32_t, uint32_t> stageOffsets;
    for (uint32_t index : pushSlots) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = layouts[index];
        const SlotPlan &slot = objects->slots[index];
        const uint32_t offset = static_cast<uint32_t>(align(pushSize, layout.element_alignment));
        objects->sets[slot.set].placements.push_back({slot.dataOffset, offset, slot.size});
        pushSize = offset + slot.size;
        const auto found = stageOffsets.find(layout.stage_mask);
        if (found == stageOffsets.end())
            stageOffsets.emplace(layout.stage_mask, offset);
        const VkShaderStageFlags stages = shaderStages(layout.stage_mask);
        if (objects->pushRanges.empty() || objects->pushRanges.back().stages != stages)
            objects->pushRanges.push_back({stages, offset, slot.size});
        else
            objects->pushRanges.back().size = offset + slot.size - objects->pushRanges.back().offset;
    }
    if (pushSize > device.maxPushConstantsSize)
        return failure<std::shared_ptr<StageObjects>>("the Stage's push constants exceed the device limit");
    std::vector<VkPushConstantRange> ranges;
    for (const PushRange &range : objects->pushRanges)
        ranges.push_back({range.stages, range.offset, range.size});
    const VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                nullptr,
                                                0,
                                                static_cast<uint32_t>(setLayouts.size()),
                                                setLayouts.data(),
                                                static_cast<uint32_t>(ranges.size()),
                                                ranges.data()};
    if (auto created = vulkanCall(api.createPipelineLayout(device.device, &layoutInfo, nullptr, &objects->layout),
                                  "vkCreatePipelineLayout");
        created.isErr())
        return Built{err(std::move(created).error())};
    const auto module = [&](const runtime::VulkanShaderCode &code, uint32_t offset) -> Made<void> {
        if (code.words.empty())
            return failure<void>("the Stage has no SPIR-V");
        std::vector<uint32_t> words = code.words;
        if (!relocatePushConstantOffsets(words, offset))
            return failure<void>("cannot relocate the Stage's push constant offsets");
        const VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                            words.size() * sizeof(uint32_t), words.data()};
        VkShaderModule created{};
        if (auto made =
                vulkanCall(api.createShaderModule(device.device, &info, nullptr, &created), "vkCreateShaderModule");
            made.isErr())
            return made;
        objects->modules.push_back(created);
        return Made<void>{ok()};
    };
    const auto stageOffset = [&](uint32_t mask) {
        const auto found = stageOffsets.find(mask);
        return found == stageOffsets.end() ? 0u : found->second;
    };
    if (!graphics) {
        if (auto made = module(state.compute, 0); made.isErr())
            return Built{err(std::move(made).error())};
        const VkPipelineShaderStageCreateInfo shader{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                     nullptr,
                                                     0,
                                                     VK_SHADER_STAGE_COMPUTE_BIT,
                                                     objects->modules[0],
                                                     state.compute.entry.c_str(),
                                                     nullptr};
        const VkComputePipelineCreateInfo info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, shader, objects->layout, {}, 0};
        if (auto created =
                vulkanCall(api.createComputePipelines(device.device, {}, 1, &info, nullptr, &objects->compute),
                           "vkCreateComputePipelines");
            created.isErr())
            return Built{err(std::move(created).error())};
        return Built{ok(std::move(objects))};
    }
    if (auto made = module(state.vertex, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)); made.isErr())
        return Built{err(std::move(made).error())};
    if (auto made = module(state.fragment, stageOffset(VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)); made.isErr())
        return Built{err(std::move(made).error())};
    objects->vertexEntry = state.vertex.entry;
    objects->fragmentEntry = state.fragment.entry;
    objects->vertexBindings = state.graphicsPlan.vertexBindings;
    objects->vertexAttributes = state.graphicsPlan.vertexAttributes;
    for (const VernonRuntimeProviderVertexAttribute &attribute : objects->vertexAttributes) {
        std::string diagnostic;
        if (!runtime::validateVertexAttributeCapability(runtime::VertexAttributeBackend::Vulkan, attribute,
                                                        device.maxVertexInputAttributes, true, diagnostic))
            return failure<std::shared_ptr<StageObjects>>(std::move(diagnostic));
        const VkFormat format = vertexFormat(attribute.dtype, attribute.component_count);
        VkFormatProperties properties{};
        if (format != VK_FORMAT_UNDEFINED)
            api.getPhysicalDeviceFormatProperties(device.physicalDevice, format, &properties);
        if (!(properties.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT))
            return failure<std::shared_ptr<StageObjects>>("a vertex attribute's format is unsupported");
    }
    return Built{ok(std::move(objects))};
}

Made<std::shared_ptr<StageObjects>> VulkanDevice::stageObjects(VernonStageExecutable &stage, bool graphics) {
    using Found = Made<std::shared_ptr<StageObjects>>;
    if (!stage.backendState)
        return failure<std::shared_ptr<StageObjects>>("the Stage has no Vulkan pipeline");
    auto &state = ::runtimeBackendState<runtime::VulkanPipelineState>(stage);
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

Made<VkRenderPass> VulkanDevice::renderPass(const RenderPassKey &key) {
    std::vector<uint64_t> words = key.words();
    std::lock_guard<std::mutex> lock(renderPassMutex_);
    if (const auto found = renderPasses_.find(words); found != renderPasses_.end())
        return Made<VkRenderPass>{ok(found->second)};
    std::array<VkAttachmentDescription, kMaxColors + 1> attachments{};
    std::array<VkAttachmentReference, kMaxColors> colors{};
    VkAttachmentReference depth{};
    for (size_t index = 0; index < key.colors.size(); ++index) {
        VkAttachmentDescription &attachment = attachments[index];
        attachment.format = key.colors[index];
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = key.loads[index];
        attachment.storeOp = key.stores[index];
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colors[index] = {static_cast<uint32_t>(index), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    const auto count = static_cast<uint32_t>(key.colors.size());
    if (key.depth != VK_FORMAT_UNDEFINED) {
        VkAttachmentDescription &attachment = attachments[count];
        attachment.format = key.depth;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = key.depthLoad;
        attachment.storeOp = key.depthStore;
        attachment.stencilLoadOp = key.stencil ? key.stencilLoad : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = key.stencil ? key.stencilStore : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth = {count, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    }
    const bool hasDepth = key.depth != VK_FORMAT_UNDEFINED;
    const VkSubpassDescription subpass{0,       VK_PIPELINE_BIND_POINT_GRAPHICS, 0, nullptr, count, colors.data(),
                                       nullptr, hasDepth ? &depth : nullptr,     0, nullptr};
    const VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                      nullptr,
                                      0,
                                      count + (hasDepth ? 1u : 0u),
                                      attachments.data(),
                                      1,
                                      &subpass,
                                      0,
                                      nullptr};
    VkRenderPass created{};
    if (auto made =
            vulkanCall(rv::driver().createRenderPass(state_.device, &info, nullptr, &created), "vkCreateRenderPass");
        made.isErr())
        return Made<VkRenderPass>{err(std::move(made).error())};
    renderPasses_.emplace(std::move(words), created);
    return Made<VkRenderPass>{ok(created)};
}

Made<VkPipeline> VulkanDevice::variant(StageObjects &stage, const runtime::GraphicsVariantKey &key) {
    std::lock_guard<std::mutex> lock(stage.mutex);
    for (const StageObjects::Variant &variant : stage.variants)
        if (runtime::graphicsVariantKeysEqual(variant.key, key))
            return Made<VkPipeline>{ok(variant.pipeline)};
    if (key.sampleCount != 1 || key.colorFormats.size() > kMaxColors ||
        (key.depthStencilFormat && key.depthStencilFormat != VK_FORMAT_D32_SFLOAT &&
         key.depthStencilFormat != VK_FORMAT_D32_SFLOAT_S8_UINT) ||
        key.colorBlends.size() != key.colorFormats.size())
        return failure<VkPipeline>("the graphics pipeline uses an unsupported attachment configuration");
    const bool hasDepth = key.depthStencilFormat != 0;
    const bool hasStencil = key.depthStencilFormat == VK_FORMAT_D32_SFLOAT_S8_UINT;
    std::array<VkPipelineShaderStageCreateInfo, 2> shaders{};
    shaders[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr,
                  0,
                  VK_SHADER_STAGE_VERTEX_BIT,
                  stage.modules[0],
                  stage.vertexEntry.c_str(),
                  nullptr};
    shaders[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr,
                  0,
                  VK_SHADER_STAGE_FRAGMENT_BIT,
                  stage.modules[1],
                  stage.fragmentEntry.c_str(),
                  nullptr};
    std::vector<VkVertexInputBindingDescription> bindings;
    for (const VernonRuntimeProviderVertexBinding &binding : stage.vertexBindings) {
        if (binding.binding >= key.vertexStrides.size())
            return failure<VkPipeline>("the graphics pipeline is missing a vertex stride");
        bindings.push_back({binding.binding, key.vertexStrides[binding.binding],
                            binding.divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
    }
    std::vector<VkVertexInputAttributeDescription> attributes;
    for (const VernonRuntimeProviderVertexAttribute &attribute : stage.vertexAttributes)
        attributes.push_back({attribute.location, attribute.binding,
                              vertexFormat(attribute.dtype, attribute.component_count), attribute.relative_offset});
    const VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                                           nullptr,
                                                           0,
                                                           static_cast<uint32_t>(bindings.size()),
                                                           bindings.data(),
                                                           static_cast<uint32_t>(attributes.size()),
                                                           attributes.data()};
    const VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                          nullptr, 0,
                                                          key.topology == 1   ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                                                          : key.topology == 2 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                                                                              : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                          VK_FALSE};
    const VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
    const VernonRasterizationState &rasterization = key.rasterization;
    if (rasterization.depth_clamp || !std::isfinite(rasterization.depth_bias_constant) ||
        !std::isfinite(rasterization.depth_bias_slope))
        return failure<VkPipeline>("the graphics pipeline has an unsupported rasterization state");
    const VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                        nullptr,
                                                        0,
                                                        VK_FALSE,
                                                        VK_FALSE,
                                                        VK_POLYGON_MODE_FILL,
                                                        static_cast<VkCullModeFlags>(rasterization.cull_mode),
                                                        rasterization.front_face ? VK_FRONT_FACE_CLOCKWISE
                                                                                 : VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                        rasterization.depth_bias_enabled ? VK_TRUE : VK_FALSE,
                                                        rasterization.depth_bias_constant,
                                                        0,
                                                        rasterization.depth_bias_slope,
                                                        1};
    const VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                           nullptr,
                                                           0,
                                                           VK_SAMPLE_COUNT_1_BIT,
                                                           VK_FALSE,
                                                           0,
                                                           nullptr,
                                                           VK_FALSE,
                                                           VK_FALSE};
    const VernonDepthStencilState &depthState = key.depthStencil;
    if ((!hasDepth && (depthState.depth_test || depthState.depth_write)) || (!hasStencil && depthState.stencil_test))
        return failure<VkPipeline>("the graphics pipeline has an invalid depth stencil state");
    const auto face = [&](const VernonStencilFaceState &source) {
        return VkStencilOpState{static_cast<VkStencilOp>(source.stencil_fail),
                                static_cast<VkStencilOp>(source.pass),
                                static_cast<VkStencilOp>(source.depth_fail),
                                static_cast<VkCompareOp>(source.compare),
                                depthState.stencil_read_mask,
                                depthState.stencil_write_mask,
                                0};
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        nullptr,
        0,
        depthState.depth_test ? VK_TRUE : VK_FALSE,
        depthState.depth_write ? VK_TRUE : VK_FALSE,
        depthState.depth_test ? static_cast<VkCompareOp>(depthState.depth_compare) : VK_COMPARE_OP_ALWAYS,
        VK_FALSE,
        depthState.stencil_test ? VK_TRUE : VK_FALSE,
        face(depthState.front),
        face(depthState.back),
        0,
        1};
    std::array<VkPipelineColorBlendAttachmentState, kMaxColors> blends{};
    for (size_t index = 0; index < key.colorBlends.size(); ++index) {
        const VernonColorBlendState &source = key.colorBlends[index];
        VkPipelineColorBlendAttachmentState &target = blends[index];
        target.blendEnable = source.blend_enabled ? VK_TRUE : VK_FALSE;
        target.srcColorBlendFactor = static_cast<VkBlendFactor>(source.source_color_factor);
        target.dstColorBlendFactor = static_cast<VkBlendFactor>(source.destination_color_factor);
        target.colorBlendOp = static_cast<VkBlendOp>(source.color_operation);
        target.srcAlphaBlendFactor = static_cast<VkBlendFactor>(source.source_alpha_factor);
        target.dstAlphaBlendFactor = static_cast<VkBlendFactor>(source.destination_alpha_factor);
        target.alphaBlendOp = static_cast<VkBlendOp>(source.alpha_operation);
        target.colorWriteMask = source.write_mask;
    }
    const VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                    nullptr,
                                                    0,
                                                    VK_FALSE,
                                                    VK_LOGIC_OP_COPY,
                                                    static_cast<uint32_t>(key.colorBlends.size()),
                                                    blends.data(),
                                                    {}};
    const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                         VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    const VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 3,
                                                   dynamicStates};
    std::vector<VkFormat> colorFormats;
    for (uint32_t format : key.colorFormats)
        colorFormats.push_back(static_cast<VkFormat>(format));
    const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  static_cast<uint32_t>(colorFormats.size()),
                                                  colorFormats.data(),
                                                  static_cast<VkFormat>(key.depthStencilFormat),
                                                  hasStencil ? static_cast<VkFormat>(key.depthStencilFormat)
                                                             : VK_FORMAT_UNDEFINED};
    VkRenderPass compatible{};
    if (!state_.dynamicRendering) {
        // Defines render pass compatibility only; a scope's own render pass carries its load and store operations.
        RenderPassKey passKey;
        passKey.colors = colorFormats;
        passKey.loads.assign(colorFormats.size(), VK_ATTACHMENT_LOAD_OP_CLEAR);
        passKey.stores.assign(colorFormats.size(), VK_ATTACHMENT_STORE_OP_STORE);
        passKey.depth = static_cast<VkFormat>(key.depthStencilFormat);
        passKey.stencil = hasStencil;
        passKey.depthLoad = passKey.stencilLoad = VK_ATTACHMENT_LOAD_OP_CLEAR;
        passKey.depthStore = passKey.stencilStore = VK_ATTACHMENT_STORE_OP_STORE;
        auto made = renderPass(passKey);
        if (made.isErr())
            return Made<VkPipeline>{err(std::move(made).error())};
        compatible = made.value();
    }
    const VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                            state_.dynamicRendering ? &rendering : nullptr,
                                            0,
                                            static_cast<uint32_t>(shaders.size()),
                                            shaders.data(),
                                            &vertexInput,
                                            &assembly,
                                            nullptr,
                                            &viewport,
                                            &raster,
                                            &multisample,
                                            hasDepth ? &depthStencil : nullptr,
                                            &blend,
                                            &dynamic,
                                            stage.layout,
                                            compatible,
                                            0,
                                            {},
                                            0};
    VkPipeline pipeline{};
    if (auto made = vulkanCall(rv::driver().createGraphicsPipelines(state_.device, {}, 1, &info, nullptr, &pipeline),
                               "vkCreateGraphicsPipelines");
        made.isErr())
        return Made<VkPipeline>{err(std::move(made).error())};
    stage.variants.push_back({key, pipeline});
    return Made<VkPipeline>{ok(pipeline)};
}

Builder::Builder(VulkanDevice &device, Recording &recording, const PlannedPasses &planned,
                 const std::vector<std::optional<runtime::PlannedComputeLaunch>> &launches)
    : device_(device), state_(device.state()), recording_(recording), planned_(planned), launches_(launches) {
    for (auto &sets : bound_)
        sets.fill(kNone);
    pushed_.fill(kNone);
}

void Builder::debug(uint32_t kind, const std::string &label) {
    if (!state_.cmdBeginDebugUtilsLabel)
        return;
    Command command;
    command.op = kind == VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP  ? Op::DebugPush
                 : kind == VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP ? Op::DebugPop
                                                                   : Op::DebugMarker;
    command.a = static_cast<uint32_t>(recording_.labels.size());
    recording_.labels.push_back(label);
    add(command);
}

void Builder::timer(bool begin) {
    if (!recording_.timed || !state_.timestampValidBits)
        return;
    Command command;
    command.op = Op::Timestamp;
    command.a = queries_++;
    command.b = begin ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    add(command);
}

// Moves `image`'s subresources to `target` at this point of the recording. A subresource's first use defines the
// layout the recording begins it in, to which submission transitions it.
void Builder::transition(rv::Image &image, VkImageLayout target) {
    const auto [owner, range] = layoutSubresources(image);
    if (!owner || !range.mipLevelCount || !range.arrayLayerCount)
        return;
    LayoutUse &use = recording_.layouts[owner];
    if (use.current.empty()) {
        use.initial.assign(owner->subresourceLayouts.size(), kUnusedLayout);
        use.current.assign(owner->subresourceLayouts.size(), kUnusedLayout);
    }
    const auto emit = [&](VkImageLayout source, uint32_t mip, uint32_t mips, uint32_t layer, uint32_t layers) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = source;
        barrier.newLayout = target;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = owner->image;
        barrier.subresourceRange = {rv::imageAspectMask(owner->format), mip, mips, layer, layers};
        VkPipelineStageFlags sourceStage{};
        VkPipelineStageFlags destinationStage{};
        imageDependency(source, sourceStage, barrier.srcAccessMask);
        imageDependency(target, destinationStage, barrier.dstAccessMask);
        sourceStages_ |= sourceStage;
        destinationStages_ |= destinationStage;
        barriers_.push_back(barrier);
    };
    std::vector<VkImageLayout> sources;
    for (uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.arrayLayerCount; ++layer)
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.mipLevelCount; ++mip) {
            const size_t subresource = mip + size_t{layer} * owner->mipLevels;
            if (subresource >= use.current.size())
                continue;
            VkImageLayout &current = use.current[subresource];
            if (current == kUnusedLayout)
                use.initial[subresource] = target;
            sources.push_back(current == kUnusedLayout ? target : current);
            current = target;
        }
    if (sources.empty())
        return;
    if (std::all_of(sources.begin(), sources.end(), [&](VkImageLayout layout) { return layout == sources.front(); })) {
        if (sources.front() != target)
            emit(sources.front(), range.baseMipLevel, range.mipLevelCount, range.baseArrayLayer, range.arrayLayerCount);
        return;
    }
    for (size_t index = 0; index < sources.size(); ++index)
        if (sources[index] != target)
            emit(sources[index], range.baseMipLevel + static_cast<uint32_t>(index % range.mipLevelCount), 1,
                 range.baseArrayLayer + static_cast<uint32_t>(index / range.mipLevelCount), 1);
}

void Builder::flushBarriers() {
    if (barriers_.empty())
        return;
    Command command;
    command.op = Op::ImageBarriers;
    command.a = static_cast<uint32_t>(recording_.imageBarriers.size());
    command.b = static_cast<uint32_t>(barriers_.size());
    command.c = sourceStages_;
    command.d = destinationStages_;
    recording_.imageBarriers.insert(recording_.imageBarriers.end(), barriers_.begin(), barriers_.end());
    add(command);
    barriers_.clear();
    sourceStages_ = destinationStages_ = 0;
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

// Makes a set instance of `set` from the values `fill` gives. A persistent instance's bytes are written once; the
// others are sites the call fills again from `source`.
Made<uint32_t> Builder::instance(StageObjects &stage, uint32_t set, bool persistent, const FillValue &fill,
                                 const Site &source, std::vector<ImageUse> &images) {
    const SetPlan &plan = stage.sets[set];
    std::vector<VernonRuntimeProviderBindingValue> values(plan.slots.size());
    std::vector<std::vector<uint8_t>> storage(plan.slots.size());
    std::string error;
    for (size_t index = 0; index < plan.slots.size(); ++index) {
        const SlotPlan &slot = stage.slots[plan.slots[index]];
        if (!fill(plan.slots[index], values[index], storage[index], error))
            return failure<uint32_t>("cannot fill a binding: " + error);
        if (values[index].kind != slot.providerKind)
            return failure<uint32_t>("a binding value has the wrong kind");
        if ((slot.kind == SlotKind::Push || slot.kind == SlotKind::Inline) &&
            (!values[index].payload.inline_value.data || values[index].payload.inline_value.size != slot.size))
            return failure<uint32_t>("an inline binding value has the wrong size");
    }
    const auto at = static_cast<uint32_t>(recording_.sets.size());
    SetInstance &made = recording_.sets.emplace_back();
    made.layout = plan.descriptors ? plan.layout : VK_NULL_HANDLE;
    if (plan.pushBytes) {
        made.pushOffset = static_cast<uint32_t>(recording_.push.size());
        recording_.push.resize(recording_.push.size() + plan.pushBytes);
    }
    const auto samplerOf = [&](uint32_t slot) -> Made<VkSampler> {
        const auto position = std::find(plan.slots.begin(), plan.slots.end(), slot) - plan.slots.begin();
        const VernonRuntimeProviderBindingValue &value = values[position];
        if (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE)
            return Made<VkSampler>{ok(state_.defaultImplicitSampler.sampler)};
        auto sampler = nativeOf<rv::Sampler>(value.payload.sampler.resource, Kind::Sampler);
        if (sampler.isErr())
            return Made<VkSampler>{err(std::move(sampler).error())};
        return Made<VkSampler>{ok(sampler.value()->sampler)};
    };
    for (size_t index = 0; index < plan.slots.size(); ++index) {
        const uint32_t slotIndex = plan.slots[index];
        const SlotPlan &slot = stage.slots[slotIndex];
        const VernonRuntimeProviderBindingValue &value = values[index];
        Site site = source;
        site.slot = slotIndex;
        site.size = slot.size;
        switch (slot.kind) {
        case SlotKind::Push: {
            site.offset = made.pushOffset + slot.dataOffset;
            site.push = true;
            std::memcpy(recording_.push.data() + site.offset, value.payload.inline_value.data, slot.size);
            if (!persistent)
                recording_.sites.push_back(site);
            break;
        }
        case SlotKind::Inline: {
            site.offset = arena(slot.size, std::max<uint64_t>(state_.descriptorBufferOffsetAlignment, 4));
            const auto *bytes = static_cast<const uint8_t *>(value.payload.inline_value.data);
            if (persistent)
                recording_.fixed.emplace_back(site.offset, std::vector<uint8_t>(bytes, bytes + slot.size));
            else
                recording_.sites.push_back(site);
            made.writes.push_back({slot.binding, slot.type, VK_NULL_HANDLE, site.offset, slot.size});
            break;
        }
        case SlotKind::Buffer: {
            const VernonRuntimeProviderResourceReference &resource = value.payload.buffer.resource;
            if (resource.size == 0)
                return failure<uint32_t>("a buffer binding is empty");
            auto buffer = nativeOf<rv::Buffer>(resource, Kind::Buffer);
            if (buffer.isErr())
                return failure<uint32_t>(std::move(buffer).error());
            made.writes.push_back({slot.binding, slot.type, buffer.value()->buffer, resource.offset, resource.size});
            break;
        }
        case SlotKind::Image: {
            if (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE)
                return failure<uint32_t>("an image binding has no image");
            auto image = nativeOf<rv::Image>(value.payload.image.view, Kind::View);
            if (image.isErr())
                return failure<uint32_t>(std::move(image).error());
            if (slot.providerKind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE &&
                image.value()->format != slot.storageFormat)
                return failure<uint32_t>("a storage image's format is not the one its kernel declares");
            const VkImageLayout layout = slot.providerKind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
                                             ? VK_IMAGE_LAYOUT_GENERAL
                                             : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            images.push_back({image.value(), layout});
            Write write{slot.binding, slot.type};
            write.view = image.value()->view;
            write.imageLayout = layout;
            if (slot.sampler != kNone) {
                auto sampler = samplerOf(slot.sampler);
                if (sampler.isErr())
                    return failure<uint32_t>(std::move(sampler).error());
                write.sampler = sampler.value();
            }
            made.writes.push_back(write);
            break;
        }
        case SlotKind::Sampler: {
            auto sampler = samplerOf(slotIndex);
            if (sampler.isErr())
                return failure<uint32_t>(std::move(sampler).error());
            Write write{slot.binding, slot.type};
            write.sampler = sampler.value();
            made.writes.push_back(write);
            break;
        }
        case SlotKind::CombinedSampler:
            break;
        }
    }
    if (made.layout) {
        ++recording_.descriptorSets;
        for (const Write &write : made.writes)
            ++recording_.descriptorCounts[write.type];
    }
    return Made<uint32_t>{ok(at)};
}

Made<void> Builder::build(const std::vector<const InternalBuffers::Internal *> &internals) {
    auto &api = rv::driver();
    add({Op::QueueOrder});
    const size_t reset = recording_.commands.size();
    add({Op::ResetQueries});
    if (!internals.empty() || !planned_.zeroed.empty()) {
        uint64_t bytes = 0;
        for (const InternalBuffers::Internal *internal : internals)
            bytes = std::max(bytes, internal->bytes);
        for (const VernonRuntimeProviderResourceReference &range : planned_.zeroed)
            bytes = std::max(bytes, range.size);
        std::string error;
        if (!state_.createBuffer(recording_.zeros, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, error))
            return failure<void>("cannot create the internal buffers' zeros: " + error);
        void *mapped = nullptr;
        if (auto mappedMemory =
                vulkanCall(api.mapMemory(state_.device, recording_.zeros.memory, 0, bytes, 0, &mapped), "vkMapMemory");
            mappedMemory.isErr())
            return mappedMemory;
        std::memset(mapped, 0, bytes);
        api.unmapMemory(state_.device, recording_.zeros.memory);
        for (const InternalBuffers::Internal *internal : internals) {
            auto buffer = nativeOf<rv::Buffer>(internal->reference, Kind::Buffer);
            if (buffer.isErr())
                return failure<void>("an internal buffer: " + buffer.error());
            Command copy{Op::CopyBuffer};
            copy.buffer = recording_.zeros.buffer;
            copy.countBuffer = buffer.value()->buffer;
            copy.offset = internal->bytes;
            add(copy);
        }
        for (const VernonRuntimeProviderResourceReference &range : planned_.zeroed) {
            if (!range.size)
                continue;
            auto buffer = nativeOf<rv::Buffer>(range, Kind::Buffer);
            if (buffer.isErr())
                return failure<void>("a zeroed argument: " + buffer.error());
            Command copy{Op::CopyBuffer};
            copy.buffer = recording_.zeros.buffer;
            copy.countBuffer = buffer.value()->buffer;
            copy.countOffset = range.offset;
            copy.offset = range.size;
            add(copy);
        }
        Command ordered{Op::MemoryBarrier};
        ordered.a = VK_PIPELINE_STAGE_TRANSFER_BIT;
        ordered.b = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        ordered.c = VK_ACCESS_TRANSFER_WRITE_BIT;
        ordered.d = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        add(ordered);
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
        if (recording_.timed)
            recording_.timers.push_back({pass.executable->source().node->id, static_cast<uint32_t>(end - at), 0, 0});
        timer(true);
        if (auto built = buildScope(at, end); built.isErr())
            return built;
        timer(false);
        at = end;
    }
    recording_.commands[reset].a = queries_;
    if (!queries_)
        recording_.commands.erase(recording_.commands.begin() + static_cast<std::ptrdiff_t>(reset));
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
    StageObjects &stageObjects = *objects.value();
    const auto &state = ::runtimeBackendState<runtime::VulkanPipelineState>(executable);
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
    std::vector<ImageUse> images;
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        if (stageObjects.sets[set].slots.empty())
            continue;
        auto made = instance(stageObjects, set, false, fill, source, images);
        if (made.isErr())
            return Made<void>{err(name + made.error())};
        sets[set] = made.value();
    }
    if (graphicsBefore_) {
        Command ordered{Op::MemoryBarrier};
        ordered.a = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        ordered.b = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        ordered.c = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        ordered.d = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        add(ordered);
        graphicsBefore_ = false;
    }
    for (const ImageUse &use : images)
        transition(*use.image, use.layout);
    flushBarriers();
    if (pipelines_[1] != stageObjects.compute) {
        Command bind{Op::BindPipeline};
        bind.a = VK_PIPELINE_BIND_POINT_COMPUTE;
        bind.pipeline = stageObjects.compute;
        add(bind);
        pipelines_[1] = stageObjects.compute;
    }
    if (layouts_[1] != stageObjects.layout) {
        layouts_[1] = stageObjects.layout;
        bound_[1].fill(kNone);
    }
    for (uint32_t set = 0; set < stageObjects.sets.size(); ++set) {
        if (sets[set] == kNone || !recording_.sets[sets[set]].layout || bound_[1][set] == sets[set])
            continue;
        Command bind{Op::BindSet};
        bind.a = VK_PIPELINE_BIND_POINT_COMPUTE;
        bind.b = set;
        bind.c = sets[set];
        bind.layout = stageObjects.layout;
        add(bind);
        bound_[1][set] = sets[set];
    }
    // Indirect, so that a call whose grid changed submits the recording again after it wrote its group counts.
    Command dispatch{Op::Dispatch};
    dispatch.offset = arena(sizeof(uint32_t) * 3, sizeof(uint32_t));
    recording_.dispatches.emplace_back(static_cast<uint32_t>(at), dispatch.offset);
    add(dispatch);
    // What follows reads what the dispatch wrote: as vertices, indices, indirect records or in shaders.
    Command ordered{Op::MemoryBarrier};
    ordered.a = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    ordered.b = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    ordered.c = VK_ACCESS_SHADER_WRITE_BIT;
    ordered.d = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_TRANSFER_READ_BIT;
    add(ordered);
    return Made<void>{ok()};
}

Made<void> Builder::buildCopy(const PlannedCopy &copy) {
    if (copy.regions.empty())
        return Made<void>{ok()};
    auto source = nativeOf<rv::Buffer>(copy.source, Kind::Buffer);
    if (source.isErr())
        return failure<void>("a leaf copy's source: " + source.error());
    auto destination = nativeOf<rv::Buffer>(copy.destination, Kind::Buffer);
    if (destination.isErr())
        return failure<void>("a leaf copy's destination: " + destination.error());
    flushBarriers();
    Command before{Op::MemoryBarrier};
    before.a = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    before.b = VK_PIPELINE_STAGE_TRANSFER_BIT;
    before.c = VK_ACCESS_MEMORY_WRITE_BIT;
    before.d = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    add(before);
    graphicsBefore_ = false;
    Command regions{Op::CopyRegions};
    regions.buffer = source.value()->buffer;
    regions.countBuffer = destination.value()->buffer;
    regions.a = static_cast<uint32_t>(recording_.copies.size());
    regions.b = static_cast<uint32_t>(copy.regions.size());
    for (const CopyRegion &region : copy.regions)
        recording_.copies.push_back({region.source, region.destination, region.size});
    add(regions);
    Command after{Op::MemoryBarrier};
    after.a = VK_PIPELINE_STAGE_TRANSFER_BIT;
    after.b = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    after.c = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.d = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    add(after);
    return Made<void>{ok()};
}

Made<Builder::DrawPlan> Builder::planDraw(const PlannedPass &pass, const PlannedDraw &draw,
                                          std::vector<ImageUse> &images) {
    using Planned = Made<DrawPlan>;
    DrawPlan result;
    result.draw = &draw;
    result.sets.fill(kNone);
    auto &state = ::runtimeBackendState<runtime::VulkanPipelineState>(*draw.stage);
    if (state.vertex.words.empty())
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
        auto native = nativeOf<rv::Buffer>(buffer, Kind::Buffer);
        if (native.isErr())
            return failure<DrawPlan>("a vertex buffer: " + native.error());
        recording_.vertexBuffers.push_back(native.value()->buffer);
        recording_.vertexOffsets.push_back(buffer.offset);
    }
    std::vector<uint32_t> formats;
    for (VernonTextureFormat format : pass.targets.attachmentFormats) {
        const std::optional<VkFormat> native = attachmentFormat(format);
        if (!native)
            return failure<DrawPlan>("Vulkan color attachment format is unsupported");
        formats.push_back(static_cast<uint32_t>(*native));
    }
    uint32_t depthFormat = 0;
    if (pass.targets.depthAttachment) {
        const std::optional<VkFormat> native = attachmentFormat(pass.targets.depthFormat);
        if (!native)
            return failure<DrawPlan>("Vulkan depth attachment format is unsupported");
        depthFormat = static_cast<uint32_t>(*native);
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
            result.sets[set] = existing->second;
            continue;
        }
        std::vector<ImageUse> made;
        auto created = instance(stageObjects, set, persistent, fill, source, made);
        if (created.isErr())
            return Planned{err(std::move(created).error())};
        result.sets[set] = created.value();
        instances_.emplace(std::move(found), created.value());
        images.insert(images.end(), made.begin(), made.end());
    }
    if (!runtime::prepareGraphicsDrawInvocation(draw.draw, draw.state->topology, result.invocation, error))
        return failure<DrawPlan>(error);
    if (result.invocation.index_count || draw.streamed) {
        if (!result.invocation.index_count)
            return failure<DrawPlan>("a draw stream needs an index binding");
        auto index = nativeOf<rv::Buffer>(result.invocation.index_buffer, Kind::Buffer);
        if (index.isErr())
            return failure<DrawPlan>("an index buffer: " + index.error());
        result.indexBuffer = index.value()->buffer;
        result.indexOffset = result.invocation.index_buffer.offset;
    }
    return Planned{ok(result)};
}

Made<void> Builder::emitDraw(const DrawPlan &plan) {
    if (!plan.stage)
        return Made<void>{ok()};
    const StageObjects &stage = *plan.stage;
    const PlannedDraw &draw = *plan.draw;
    if (pipelines_[0] != plan.pipeline) {
        Command bind{Op::BindPipeline};
        bind.a = VK_PIPELINE_BIND_POINT_GRAPHICS;
        bind.pipeline = plan.pipeline;
        add(bind);
        pipelines_[0] = plan.pipeline;
    }
    if (layouts_[0] != stage.layout) {
        layouts_[0] = stage.layout;
        bound_[0].fill(kNone);
    }
    bool push = pushStage_ != &stage;
    for (uint32_t set = 0; set < stage.sets.size(); ++set) {
        const uint32_t instance = plan.sets[set];
        if (instance == kNone)
            continue;
        if (stage.sets[set].pushBytes && pushed_[set] != instance)
            push = true;
        if (!recording_.sets[instance].layout || bound_[0][set] == instance)
            continue;
        Command bind{Op::BindSet};
        bind.a = VK_PIPELINE_BIND_POINT_GRAPHICS;
        bind.b = set;
        bind.c = instance;
        bind.layout = stage.layout;
        add(bind);
        bound_[0][set] = instance;
    }
    if (push && !stage.pushRanges.empty()) {
        Command pushed{Op::Push};
        pushed.stage = &stage;
        pushed.layout = stage.layout;
        pushed.a = static_cast<uint32_t>(recording_.pushSets.size());
        for (uint32_t set = 0; set < stage.sets.size(); ++set)
            if (stage.sets[set].pushBytes && plan.sets[set] != kNone) {
                recording_.pushSets.emplace_back(set, plan.sets[set]);
                pushed_[set] = plan.sets[set];
            }
        pushed.b = static_cast<uint32_t>(recording_.pushSets.size()) - pushed.a;
        add(pushed);
        pushStage_ = &stage;
    }
    if (plan.vertexCount) {
        Command bind{Op::BindVertexBuffers};
        bind.a = plan.vertexFirst;
        bind.b = plan.vertexCount;
        add(bind);
    }
    const VernonRuntimeCoreDrawInvocation &invocation = plan.invocation;
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
    if (stencil_ != invocation.stencil_reference) {
        add({Op::Stencil, invocation.stencil_reference});
        stencil_ = invocation.stencil_reference;
    }
    if (plan.indexBuffer && (indexBuffer_ != plan.indexBuffer || indexOffset_ != plan.indexOffset)) {
        Command bind{Op::BindIndexBuffer};
        bind.buffer = plan.indexBuffer;
        bind.offset = plan.indexOffset;
        add(bind);
        indexBuffer_ = plan.indexBuffer;
        indexOffset_ = plan.indexOffset;
    }
    auto &api = rv::driver();
    if (!draw.streamed) {
        if (plan.indexBuffer)
            add({Op::DrawIndexed, invocation.index_count, invocation.instance_count, invocation.first_index,
                 static_cast<uint32_t>(invocation.base_vertex), invocation.first_instance});
        else
            add({Op::Draw, invocation.vertex_count, invocation.instance_count, invocation.first_instance});
        return Made<void>{ok()};
    }
    const runtime::GraphicsDrawStream &stream = draw.stream;
    const bool indirectCount = state_.drawIndirectCount && api.cmdDrawIndexedIndirectCount;
    if (stream.records.resource.value) {
        const bool counted = stream.counts.resource.value != 0;
        if (!state_.multiDrawIndirect || (counted && !indirectCount))
            return failure<void>("the device draws no device draw streams");
        if (stream.maxRecords == 0)
            return Made<void>{ok()};
        if (uint64_t{stream.firstRecord} + stream.maxRecords > stream.records.size / kRecordBytes ||
            (counted && stream.countIndex >= stream.counts.size / sizeof(uint32_t)))
            return failure<void>("a device draw stream exceeds its buffers");
        auto records = nativeOf<rv::Buffer>(stream.records, Kind::Buffer);
        if (records.isErr())
            return failure<void>("a draw stream's records: " + records.error());
        Command indirect{counted ? Op::DrawIndirectCount : Op::DrawIndirect};
        indirect.buffer = records.value()->buffer;
        indirect.offset = stream.records.offset + uint64_t{stream.firstRecord} * kRecordBytes;
        indirect.a = stream.maxRecords;
        if (counted) {
            auto counts = nativeOf<rv::Buffer>(stream.counts, Kind::Buffer);
            if (counts.isErr())
                return failure<void>("a draw stream's counts: " + counts.error());
            indirect.countBuffer = counts.value()->buffer;
            indirect.countOffset = stream.counts.offset + uint64_t{stream.countIndex} * sizeof(uint32_t);
        }
        add(indirect);
        return Made<void>{ok()};
    }
    if (!stream.hostRecords || !stream.hostCounts)
        return failure<void>("a host draw stream has no records");
    if (stream.maxRecords == 0)
        return Made<void>{ok()};
    HostStream host;
    host.draw = &draw;
    host.indexCapacity = invocation.index_buffer.size / sizeof(uint32_t);
    const auto index = static_cast<uint32_t>(recording_.streams.size());
    if (stream.maxRecords > 1 && !state_.multiDrawIndirect) {
        recording_.volatileCommands = true;
        recording_.streams.push_back(host);
        add({Op::HostDraws, index});
        return Made<void>{ok()};
    }
    host.counted = indirectCount;
    host.records = arena(uint64_t{stream.maxRecords} * kRecordBytes, 16);
    if (host.counted)
        host.count = arena(sizeof(uint32_t), sizeof(uint32_t));
    recording_.streams.push_back(host);
    Command indirect{host.counted ? Op::DrawIndirectCount : Op::DrawIndirect};
    indirect.offset = host.records;
    indirect.countOffset = host.count;
    indirect.a = stream.maxRecords;
    add(indirect);
    return Made<void>{ok()};
}

Made<Rendering> Builder::rendering(const runtime::GraphicsRenderScope &scope) {
    using Made = Made<Rendering>;
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
        auto native = nativeOf<rv::Image>(image, Kind::View);
        if (native.isErr())
            return failure<Rendering>("a sampled image: " + native.error());
        transition(*native.value(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    Rendering result;
    result.area = {{0, 0}, {scope.width, scope.height}};
    RenderPassKey key;
    for (const VernonRuntimeProviderColorAttachment &color : scope.colors) {
        auto native = nativeOf<rv::Image>(color.view, Kind::View);
        if (native.isErr())
            return failure<Rendering>("a color attachment: " + native.error());
        if (color.location >= kMaxColors)
            return failure<Rendering>("a color attachment's location is out of range");
        transition(*native.value(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        result.colors = std::max(result.colors, color.location + 1);
        result.views[color.location] = native.value()->view;
        result.loads[color.location] = loadOperation(color.load_operation);
        result.stores[color.location] = storeOperation(color.store_operation);
        std::memcpy(&result.clears[color.location].color, &color.clear_color, sizeof(VkClearColorValue));
        key.colors.push_back(native.value()->format);
        key.loads.push_back(result.loads[color.location]);
        key.stores.push_back(result.stores[color.location]);
    }
    if (scope.depth) {
        const VernonRuntimeProviderDepthStencilAttachment &depth = *scope.depth;
        auto native = nativeOf<rv::Image>(depth.view, Kind::View);
        if (native.isErr())
            return failure<Rendering>("the depth attachment: " + native.error());
        transition(*native.value(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        result.depthView = native.value()->view;
        result.stencil = checked.value().stencil;
        result.depthLoad = loadOperation(depth.depth_load_operation);
        result.depthStore = storeOperation(depth.depth_store_operation);
        result.stencilLoad =
            result.stencil ? loadOperation(depth.stencil_load_operation) : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        result.stencilStore =
            result.stencil ? storeOperation(depth.stencil_store_operation) : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        result.clears[kMaxColors].depthStencil = {depth.clear_depth, depth.clear_stencil};
        key.depth = native.value()->format;
        key.stencil = result.stencil;
        key.depthLoad = result.depthLoad;
        key.depthStore = result.depthStore;
        key.stencilLoad = result.stencilLoad;
        key.stencilStore = result.stencilStore;
    }
    if (state_.dynamicRendering)
        return Made{ok(result)};
    auto pass = device_.renderPass(key);
    if (pass.isErr())
        return failure<Rendering>(std::move(pass).error());
    result.renderPass = pass.value();
    std::array<VkImageView, kMaxColors + 1> views{};
    uint32_t count = 0;
    for (uint32_t location = 0; location < result.colors; ++location)
        if (result.views[location])
            views[count++] = result.views[location];
    if (result.depthView)
        views[count++] = result.depthView;
    VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    info.renderPass = result.renderPass;
    info.attachmentCount = count;
    info.pAttachments = views.data();
    info.width = scope.width;
    info.height = scope.height;
    info.layers = 1;
    if (auto created = vulkanCall(rv::driver().createFramebuffer(state_.device, &info, nullptr, &result.framebuffer),
                                  "vkCreateFramebuffer");
        created.isErr())
        return failure<Rendering>(std::move(created).error());
    recording_.framebuffers.push_back(result.framebuffer);
    return Made{ok(result)};
}

// Begins a render scope with every draw of passes [begin, end) planned, since barriers cannot be recorded inside it:
// the images the draws sample and write move to their layouts first.
Made<void> Builder::buildScope(size_t begin, size_t end) {
    const std::vector<PlannedPass *> &passes = planned_.passes;
    const runtime::GraphicsRenderScope &scope = planned_.scopes[passes[begin]->scope];
    std::string label;
    for (size_t at = begin; at < end; ++at)
        label += (at == begin ? "" : "+") + nodeLabel(*passes[at]->executable->source().node);
    debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, label);
    std::vector<DrawPlan> draws;
    std::vector<ImageUse> images;
    for (size_t at = begin; at < end; ++at)
        for (const PlannedDraw &draw : passes[at]->draws) {
            auto planned = planDraw(*passes[at], draw, images);
            if (planned.isErr())
                return failure<void>("pass '" + passes[at]->executable->source().node->name + "': " + planned.error());
            draws.push_back(planned.value());
        }
    for (const ImageUse &use : images)
        transition(*use.image, use.layout);
    auto opened = rendering(scope);
    if (opened.isErr())
        return failure<void>("pass '" + passes[begin]->executable->source().node->name + "': " + opened.error());
    flushBarriers();
    Command beginCommand{Op::BeginRendering};
    beginCommand.a = static_cast<uint32_t>(recording_.renderings.size());
    recording_.renderings.push_back(opened.value());
    add(beginCommand);
    const bool memberGroups = end - begin > 1;
    size_t next = 0;
    for (size_t at = begin; at < end; ++at) {
        const PlannedPass &pass = *passes[at];
        if (memberGroups)
            debug(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, nodeLabel(*pass.executable->source().node));
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
    add({Op::EndRendering, state_.dynamicRendering ? 1u : 0u});
    debug(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP, {});
    graphicsBefore_ = true;
    return Made<void>{ok()};
}

Made<std::unique_ptr<Recording>>
VulkanDevice::build(const VulkanCommandBuffer &commands, const PlannedPasses &planned,
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
Made<void> VulkanDevice::planLaunch(const PlannedPass &pass, std::optional<runtime::PlannedComputeLaunch> &launch,
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
        return failure<void>(name + "the Stage has no Vulkan pipeline");
    const auto &state = ::runtimeBackendState<runtime::VulkanPipelineState>(stage);
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
Made<void> VulkanDevice::createSlot(Recording &recording, Slot &slot) {
    auto &api = rv::driver();
    if (!slot.pool) {
        const VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, state_.queueFamily};
        if (auto made =
                vulkanCall(api.createCommandPool(state_.device, &poolInfo, nullptr, &slot.pool), "vkCreateCommandPool");
            made.isErr())
            return made;
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = slot.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        if (auto made = vulkanCall(api.allocateCommandBuffers(state_.device, &allocation, &slot.command),
                                   "vkAllocateCommandBuffers");
            made.isErr())
            return made;
    }
    if (slot.arenaCapacity < recording.arenaBytes || !slot.arena.buffer) {
        state_.destroyBuffer(slot.arena);
        slot.arena = {};
        slot.mapped = nullptr;
        slot.arenaCapacity = roundUp(recording.arenaBytes, 4096);
        std::string error;
        if (!state_.createBuffer(slot.arena, slot.arenaCapacity,
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, error,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            slot.arenaCapacity = 0;
            return failure<void>("cannot create a recording's arena: " + error);
        }
        void *mapped = nullptr;
        if (auto made = vulkanCall(api.mapMemory(state_.device, slot.arena.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
                                   "vkMapMemory");
            made.isErr())
            return made;
        slot.arena.mapped = slot.mapped = static_cast<uint8_t *>(mapped);
    }
    for (const auto &[offset, bytes] : recording.fixed)
        std::memcpy(slot.mapped + offset, bytes.data(), bytes.size());
    if (recording.descriptorSets) {
        bool fits = slot.descriptors && slot.setCapacity >= recording.descriptorSets;
        for (const auto &[type, count] : recording.descriptorCounts) {
            const auto found = slot.descriptorCapacity.find(type);
            fits = fits && found != slot.descriptorCapacity.end() && found->second >= count;
        }
        if (fits) {
            if (auto reset =
                    vulkanCall(api.resetDescriptorPool(state_.device, slot.descriptors, 0), "vkResetDescriptorPool");
                reset.isErr())
                return reset;
        } else {
            if (slot.descriptors)
                api.destroyDescriptorPool(state_.device, slot.descriptors, nullptr);
            slot.descriptors = VK_NULL_HANDLE;
            slot.setCapacity = static_cast<uint32_t>(roundUp(std::max(recording.descriptorSets, slot.setCapacity), 8));
            for (const auto &[type, count] : recording.descriptorCounts)
                slot.descriptorCapacity[type] =
                    static_cast<uint32_t>(roundUp(std::max(count, slot.descriptorCapacity[type]), 8));
            std::vector<VkDescriptorPoolSize> sizes;
            for (const auto &[type, count] : slot.descriptorCapacity)
                sizes.push_back({type, count});
            const VkDescriptorPoolCreateInfo info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,     0, slot.setCapacity,
                static_cast<uint32_t>(sizes.size()),           sizes.data()};
            if (auto made = vulkanCall(api.createDescriptorPool(state_.device, &info, nullptr, &slot.descriptors),
                                       "vkCreateDescriptorPool");
                made.isErr())
                return made;
        }
        std::vector<VkDescriptorSetLayout> layouts;
        for (const SetInstance &set : recording.sets)
            if (set.layout)
                layouts.push_back(set.layout);
        std::vector<VkDescriptorSet> allocated(layouts.size());
        const VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                  slot.descriptors, static_cast<uint32_t>(layouts.size()),
                                                  layouts.data()};
        if (auto made = vulkanCall(api.allocateDescriptorSets(state_.device, &setInfo, allocated.data()),
                                   "vkAllocateDescriptorSets");
            made.isErr())
            return made;
        slot.sets.assign(recording.sets.size(), VK_NULL_HANDLE);
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> buffers;
        std::vector<VkDescriptorImageInfo> images;
        size_t descriptors = 0;
        for (const SetInstance &set : recording.sets)
            if (set.layout)
                descriptors += set.writes.size();
        writes.reserve(descriptors);
        buffers.reserve(descriptors);
        images.reserve(descriptors);
        size_t next = 0;
        for (size_t index = 0; index < recording.sets.size(); ++index) {
            const SetInstance &set = recording.sets[index];
            if (!set.layout)
                continue;
            slot.sets[index] = allocated[next++];
            for (const Write &write : set.writes) {
                VkWriteDescriptorSet native{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                native.dstSet = slot.sets[index];
                native.dstBinding = write.binding;
                native.descriptorCount = 1;
                native.descriptorType = write.type;
                if (write.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                    write.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                    buffers.push_back({write.buffer ? write.buffer : slot.arena.buffer, write.offset, write.range});
                    native.pBufferInfo = &buffers.back();
                } else {
                    images.push_back({write.sampler, write.view, write.imageLayout});
                    native.pImageInfo = &images.back();
                }
                writes.push_back(native);
            }
        }
        if (!writes.empty())
            api.updateDescriptorSets(state_.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
    if (!recording.timers.empty() && state_.timestampValidBits) {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = static_cast<uint32_t>(recording.timers.size() * 2);
        auto queries = std::make_shared<QueryPool>();
        queries->device = state_.device;
        if (auto made =
                vulkanCall(api.createQueryPool(state_.device, &info, nullptr, &queries->pool), "vkCreateQueryPool");
            made.isErr())
            return made;
        slot.queries = std::move(queries);
    }
    return Made<void>{ok()};
}

// A slot whose last submission completed, after reading its timings; a new one while the recording has fewer than
// kSlots, else the oldest once it completes.
Made<Slot *> VulkanDevice::acquire(Recording &recording) {
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
        auto completed = rv::submissionCompleted(context_.rhiDevice, slot->serial);
        if (completed.isErr())
            return failure<Slot *>(rhiMessage("the device was lost", completed.error()));
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
    if (auto waited = rv::waitForSubmission(context_.rhiDevice, oldest->serial); waited.isErr())
        return failure<Slot *>(rhiMessage("a pass submission failed on the device", waited.error()));
    return reuse(*oldest);
}

// Writes the bytes a call varies: the sites' inline values and push data, and the host draw streams' records.
Made<void> VulkanDevice::fill(Recording &recording, Slot &slot, const PlannedPasses &planned,
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
            const auto &state = ::runtimeBackendState<runtime::VulkanPipelineState>(stage);
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
    for (const auto &[pass, offset] : recording.dispatches) {
        const VernonLaunchSize &grid = launches[pass]->grid;
        const uint32_t groups[3]{grid.x, grid.y, grid.z};
        std::memcpy(slot.mapped + offset, groups, sizeof(groups));
    }
    for (const HostStream &host : recording.streams) {
        const runtime::GraphicsDrawStream &stream = host.draw->stream;
        const VernonRuntimeProviderDrawRecord *records = stream.hostRecords + stream.firstRecord;
        const uint32_t count = std::min(stream.hostCounts[stream.countIndex], stream.maxRecords);
        // Checked as the call reads them: O(drawn records).
        for (uint32_t record = 0; record < count; ++record)
            if (uint64_t{records[record].first_index} + records[record].index_count > host.indexCapacity)
                return failure<void>("a host draw stream's record exceeds its index buffer");
        if (recording.volatileCommands && !host.records && !host.count)
            continue;
        std::memcpy(slot.mapped + host.records, records, count * kRecordBytes);
        if (host.counted)
            std::memcpy(slot.mapped + host.count, &count, sizeof(count));
        else
            std::memset(slot.mapped + host.records + count * kRecordBytes, 0,
                        (stream.maxRecords - count) * kRecordBytes);
    }
    return Made<void>{ok()};
}

Made<void> VulkanDevice::recordCommands(Recording &recording, Slot &slot) {
    auto &api = rv::driver();
    if (auto reset = vulkanCall(api.resetCommandBuffer(slot.command, 0), "vkResetCommandBuffer"); reset.isErr())
        return reset;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (auto begun = vulkanCall(api.beginCommandBuffer(slot.command, &begin), "vkBeginCommandBuffer"); begun.isErr())
        return begun;
    const VkCommandBuffer command = slot.command;
    const auto buffer = [&](VkBuffer named) { return named ? named : slot.arena.buffer; };
    std::vector<uint8_t> composed(state_.maxPushConstantsSize);
    for (const Command &op : recording.commands) {
        switch (op.op) {
        case Op::QueueOrder: {
            // Submissions do not wait for each other on the host, so the recording orders itself after all work
            // submitted before it.
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            api.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                   1, &barrier, 0, nullptr, 0, nullptr);
            break;
        }
        case Op::MemoryBarrier: {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = op.c;
            barrier.dstAccessMask = op.d;
            api.cmdPipelineBarrier(command, op.a, op.b, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            break;
        }
        case Op::ImageBarriers:
            api.cmdPipelineBarrier(command, op.c, op.d, 0, 0, nullptr, 0, nullptr, op.b,
                                   recording.imageBarriers.data() + op.a);
            break;
        case Op::CopyBuffer: {
            const VkBufferCopy region{0, op.countOffset, op.offset};
            api.cmdCopyBuffer(command, op.buffer, op.countBuffer, 1, &region);
            break;
        }
        case Op::CopyRegions:
            api.cmdCopyBuffer(command, op.buffer, op.countBuffer, op.b, recording.copies.data() + op.a);
            break;
        case Op::ResetQueries:
            if (slot.queries)
                api.cmdResetQueryPool(command, slot.queries->pool, 0, op.a);
            break;
        case Op::Timestamp:
            if (slot.queries)
                api.cmdWriteTimestamp(command, static_cast<VkPipelineStageFlagBits>(op.b), slot.queries->pool, op.a);
            break;
        case Op::DebugPush:
        case Op::DebugMarker: {
            VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
            label.pLabelName = recording.labels[op.a].c_str();
            if (op.op == Op::DebugPush)
                state_.cmdBeginDebugUtilsLabel(command, &label);
            else
                state_.cmdInsertDebugUtilsLabel(command, &label);
            break;
        }
        case Op::DebugPop:
            state_.cmdEndDebugUtilsLabel(command);
            break;
        case Op::BeginRendering: {
            const Rendering &rendering = recording.renderings[op.a];
            if (!rendering.renderPass) {
                std::array<VkRenderingAttachmentInfo, kMaxColors> colors{};
                for (uint32_t location = 0; location < rendering.colors; ++location) {
                    VkRenderingAttachmentInfo &attachment = colors[location];
                    attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    attachment.imageView = rendering.views[location];
                    attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    attachment.loadOp = rendering.loads[location];
                    attachment.storeOp = rendering.stores[location];
                    attachment.clearValue = rendering.clears[location];
                }
                VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                depth.imageView = rendering.depthView;
                depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depth.loadOp = rendering.depthLoad;
                depth.storeOp = rendering.depthStore;
                depth.clearValue = rendering.clears[kMaxColors];
                VkRenderingAttachmentInfo stencil = depth;
                stencil.loadOp = rendering.stencilLoad;
                stencil.storeOp = rendering.stencilStore;
                VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
                info.renderArea = rendering.area;
                info.layerCount = 1;
                info.colorAttachmentCount = rendering.colors;
                info.pColorAttachments = colors.data();
                info.pDepthAttachment = rendering.depthView ? &depth : nullptr;
                info.pStencilAttachment = rendering.depthView && rendering.stencil ? &stencil : nullptr;
                api.cmdBeginRendering(command, &info);
                break;
            }
            std::array<VkClearValue, kMaxColors + 1> clears{};
            uint32_t count = 0;
            for (uint32_t location = 0; location < rendering.colors; ++location)
                if (rendering.views[location])
                    clears[count++] = rendering.clears[location];
            if (rendering.depthView)
                clears[count++] = rendering.clears[kMaxColors];
            VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            info.renderPass = rendering.renderPass;
            info.framebuffer = rendering.framebuffer;
            info.renderArea = rendering.area;
            info.clearValueCount = count;
            info.pClearValues = clears.data();
            api.cmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
            break;
        }
        case Op::EndRendering:
            if (op.a)
                api.cmdEndRendering(command);
            else
                api.cmdEndRenderPass(command);
            break;
        case Op::BindPipeline:
            api.cmdBindPipeline(command, static_cast<VkPipelineBindPoint>(op.a), op.pipeline);
            break;
        case Op::BindSet:
            api.cmdBindDescriptorSets(command, static_cast<VkPipelineBindPoint>(op.a), op.layout, op.b, 1,
                                      &slot.sets[op.c], 0, nullptr);
            break;
        case Op::Push: {
            const StageObjects &stage = *op.stage;
            for (uint32_t index = op.a; index < op.a + op.b; ++index) {
                const auto &[set, instance] = recording.pushSets[index];
                const uint8_t *data = recording.push.data() + recording.sets[instance].pushOffset;
                for (const PushPlacement &placement : stage.sets[set].placements)
                    std::memcpy(composed.data() + placement.offset, data + placement.dataOffset, placement.size);
            }
            for (const PushRange &range : stage.pushRanges)
                api.cmdPushConstants(command, op.layout, range.stages, range.offset, range.size,
                                     composed.data() + range.offset);
            break;
        }
        case Op::BindVertexBuffers:
            api.cmdBindVertexBuffers(command, 0, op.b, recording.vertexBuffers.data() + op.a,
                                     recording.vertexOffsets.data() + op.a);
            break;
        case Op::BindIndexBuffer:
            api.cmdBindIndexBuffer(command, op.buffer, op.offset, VK_INDEX_TYPE_UINT32);
            break;
        case Op::Viewport: {
            // Flipped, so that clip space y points up as on every other backend.
            const VkViewport viewport{static_cast<float>(op.a),
                                      static_cast<float>(op.b + op.d),
                                      static_cast<float>(op.c),
                                      -static_cast<float>(op.d),
                                      0,
                                      1};
            api.cmdSetViewport(command, 0, 1, &viewport);
            break;
        }
        case Op::Scissor: {
            const VkRect2D scissor{{static_cast<int32_t>(op.a), static_cast<int32_t>(op.b)}, {op.c, op.d}};
            api.cmdSetScissor(command, 0, 1, &scissor);
            break;
        }
        case Op::Stencil:
            api.cmdSetStencilReference(command, VK_STENCIL_FACE_FRONT_AND_BACK, op.a);
            break;
        case Op::Draw:
            api.cmdDraw(command, op.a, op.b, 0, op.c);
            break;
        case Op::DrawIndexed:
            api.cmdDrawIndexed(command, op.a, op.b, op.c, static_cast<int32_t>(op.d), op.e);
            break;
        case Op::DrawIndirect:
            api.cmdDrawIndexedIndirect(command, buffer(op.buffer), op.offset, op.a,
                                       static_cast<uint32_t>(kRecordBytes));
            break;
        case Op::DrawIndirectCount:
            api.cmdDrawIndexedIndirectCount(command, buffer(op.buffer), op.offset, buffer(op.countBuffer),
                                            op.countOffset, op.a, static_cast<uint32_t>(kRecordBytes));
            break;
        case Op::HostDraws: {
            const runtime::GraphicsDrawStream &stream = recording.streams[op.a].draw->stream;
            const VernonRuntimeProviderDrawRecord *records = stream.hostRecords + stream.firstRecord;
            const uint32_t count = std::min(stream.hostCounts[stream.countIndex], stream.maxRecords);
            for (uint32_t record = 0; record < count; ++record)
                if (records[record].index_count && records[record].instance_count)
                    api.cmdDrawIndexed(command, records[record].index_count, records[record].instance_count,
                                       records[record].first_index, records[record].base_vertex,
                                       records[record].first_instance);
            break;
        }
        case Op::Dispatch:
            api.cmdDispatchIndirect(command, slot.arena.buffer, op.offset);
            break;
        }
    }
    if (auto ended = vulkanCall(api.endCommandBuffer(command), "vkEndCommandBuffer"); ended.isErr())
        return ended;
    slot.push = recording.push;
    slot.recorded = true;
    return Made<void>{ok()};
}

Made<TimelinePoint> VulkanDevice::submit(const CommandBuffer &commands, const uint8_t *record, size_t) {
    const auto &buffer = static_cast<const VulkanCommandBuffer &>(commands);
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
            if (auto planned = planLaunch(pass, launches[at], key); planned.isErr())
                return failure<TimelinePoint>(std::move(planned).error());
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
        std::unique_ptr<Recording> made = std::move(built).value();
        made->key = std::move(key);
        made->plans = std::move(plans);
        if (cacheable) {
            if (buffer.recordings.size() >= kRecordings)
                buffer.recordings.pop_front();
            buffer.recordings.push_back(std::move(made));
            recording = buffer.recordings.back().get();
        } else {
            once = std::move(made);
            recording = once.get();
        }
    }
    auto acquired = acquire(*recording);
    if (acquired.isErr())
        return failure<TimelinePoint>(std::move(acquired).error());
    Slot &slot = *acquired.value();
    if (auto filled = fill(*recording, slot, planned, launches); filled.isErr())
        return failure<TimelinePoint>(std::move(filled).error());
    if (!slot.recorded || recording->volatileCommands || slot.push != recording->push)
        if (auto recorded = recordCommands(*recording, slot); recorded.isErr()) {
            slot.recorded = false;
            return failure<TimelinePoint>(std::move(recorded).error());
        }
    if (anyUndefined(recording->reads.data(), recording->reads.size()))
        return failure<TimelinePoint>("a pass reads an image whose contents are undefined");
    auto submitted = rv::submitRecordedCommands(context_.rhiDevice, slot.command, recording->layouts);
    if (submitted.isErr())
        return failure<TimelinePoint>(rhiMessage("cannot submit the passes", submitted.error()));
    const uint64_t serial = submitted.value();
    retirer_->submitted(serial);
    slot.serial = serial;
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
    pending_.push_back({point, serial, std::move(timing)});
    return Made<TimelinePoint>{ok(point)};
}

} // namespace

std::unique_ptr<Device> createVulkanDevice(VernonRuntimeContext &context) {
    if (context.backend != VERNON_RUNTIME_VULKAN)
        return nullptr;
    auto state = rv::deviceStateOf(context.rhiDevice);
    if (state.isErr() || !state.value() || state.value()->nativeObjectsBorrowed)
        return nullptr;
    auto lease = rhi::retainDeviceLease(context.rhiDevice);
    if (lease.isErr())
        return nullptr;
    auto retirer = std::make_shared<Retirer>(context.rhiDevice, *state.value(), std::move(lease).value());
    return std::make_unique<VulkanDevice>(context, std::move(retirer));
}

} // namespace vernon::hal
