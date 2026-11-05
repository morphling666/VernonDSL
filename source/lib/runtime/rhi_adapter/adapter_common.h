#ifndef VERNON_RUNTIME_RHI_ADAPTER_COMMON_H
#define VERNON_RUNTIME_RHI_ADAPTER_COMMON_H

#include "VernonError.hpp"
#include "VernonRuntimeRHIAdapter.h"
#include "rhi/rhi_internal.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

template <typename T> using RhiAdapterResult = vernon::Result<T, vernon::ProviderError>;

struct RhiAdapterBackendOps {
    void (*destroy)(void *) noexcept;
    RhiAdapterResult<void> (*synchronize)(void *, std::string &) noexcept;
    uint64_t (*resourceIdentity)(const void *) noexcept;
};

struct RhiAdapterBackendStorage {
    RhiAdapterBackendStorage() = default;
    ~RhiAdapterBackendStorage();
    RhiAdapterBackendStorage(const RhiAdapterBackendStorage &) = delete;
    RhiAdapterBackendStorage &operator=(const RhiAdapterBackendStorage &) = delete;
    RhiAdapterBackendStorage(RhiAdapterBackendStorage &&other) noexcept;
    RhiAdapterBackendStorage &operator=(RhiAdapterBackendStorage &&other) noexcept;

    [[nodiscard]] RhiAdapterResult<void> adopt(void *newState, const RhiAdapterBackendOps *newOps) noexcept;

    void *state{};
    const RhiAdapterBackendOps *ops{};
};

struct VernonRuntimeRhiAdapter {
    VernonRuntimeRhiAdapter() = default;
    VernonRuntimeRhiAdapter(const VernonRuntimeRhiAdapter &) = delete;
    VernonRuntimeRhiAdapter &operator=(const VernonRuntimeRhiAdapter &) = delete;

    VernonRhiDevice rhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiBackend rhiBackend{};
    vernon::Option<vernon::ChildLease> rhiDeviceLease;
    RhiAdapterBackendStorage backend;
    VernonRuntimeDeviceProvider provider{};
    std::string error;
    std::atomic<size_t> shaderPreparations{};
    std::atomic<size_t> layoutPreparations{};
    std::atomic<size_t> pipelinePreparations{};
    std::atomic<size_t> groupCreations{};
    std::atomic<size_t> dispatches{};
    std::atomic<size_t> livePreparedPipelines{};
    std::atomic<uint32_t> lastStencilReference{};
    std::atomic<bool> lastDrawIndexed{};
};

namespace vernon::runtime::rhi_adapter {

inline constexpr uint64_t kRhiResourceKindMask = 7;
inline constexpr uint64_t kRhiImageResource = 1;
inline constexpr uint64_t kRhiSamplerResource = 2;
inline constexpr uint64_t kRhiBufferResource = 3;
inline constexpr uint64_t kRhiImageViewResource = 4;

inline bool isRhiImageReference(VernonRuntimeProviderResourceReference resource) {
    const uint64_t kind = resource.identity & kRhiResourceKindMask;
    return kind == kRhiImageResource || kind == kRhiImageViewResource;
}

inline bool packedUniformBytes(VernonRuntimeProviderBindingKind kind,
                               VernonRuntimeProviderBindingInterface interfaceKind) {
    return kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE || kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
           (kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
            interfaceKind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM);
}

template <typename Object> VernonRuntimeProviderObject toHandle(Object *object) {
    return {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(object))};
}

template <typename Object> Object *fromHandle(VernonRuntimeProviderObject handle) {
    return reinterpret_cast<Object *>(static_cast<uintptr_t>(handle.value));
}

void setBackendError(std::string &error, const char *message) noexcept;
vernon::ProviderError providerError(vernon::RhiError error) noexcept;
void recordProviderError(VernonRuntimeRhiAdapter &adapter, vernon::ProviderError error,
                         const char *diagnostic) noexcept;
VernonStatus providerStatus(VernonRuntimeRhiAdapter &adapter, RhiAdapterResult<void> result,
                            const char *diagnostic) noexcept;
VernonStatus providerStatus(VernonRuntimeRhiAdapter &adapter, vernon::ProviderError error,
                            const char *diagnostic) noexcept;
RhiAdapterResult<vernon::rhi::ResourceKind> resourceKind(const VernonRuntimeRhiAdapter &adapter,
                                                         VernonRuntimeProviderResourceReference resource) noexcept;
struct RhiImageBinding {
    VernonRhiImageDimension dimension{VERNON_RHI_IMAGE_2D};
    VernonRhiFormat format{VERNON_RHI_FORMAT_UNDEFINED};
};

RhiAdapterResult<void> describeProviderImage(VernonRuntimeRhiAdapter &adapter,
                                             VernonRuntimeProviderResourceReference resource,
                                             VernonRuntimeProviderImageDescription &description) noexcept;
VernonStatus describeProviderImageCallback(void *data, VernonRuntimeProviderResourceReference resource,
                                           VernonRuntimeProviderImageDescription *description);
const VernonRuntimeProviderResourceReference *providerBindingResource(const VernonRuntimeProviderBindingValue &value);
RhiAdapterResult<uint64_t> nativeCommandEncoder(VernonRuntimeRhiAdapter &adapter,
                                                VernonRuntimeProviderObject encoder) noexcept;
RhiAdapterResult<bool> commandEncoderRendering(VernonRuntimeRhiAdapter &adapter,
                                               VernonRuntimeProviderObject encoder) noexcept;
// Begin and end the RHI rendering scope of the encoder.
RhiAdapterResult<void> beginEncoderRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             const VernonRuntimeProviderRenderingDescriptor &descriptor) noexcept;
RhiAdapterResult<void> endEncoderRendering(VernonRuntimeRhiAdapter &adapter,
                                           VernonRuntimeProviderObject encoder) noexcept;
// Opens the encoder's native rendering scope unless it is open; draws open it before recording.
RhiAdapterResult<vernon::rhi::OpenedRendering> openCommandRendering(VernonRuntimeRhiAdapter &adapter,
                                                                    VernonRuntimeProviderObject encoder) noexcept;
// Writes bytes for the encoder's commands into memory its RHI recording owns until the encoder retires.
RhiAdapterResult<vernon::rhi::CommandTransient> writeCommandTransient(VernonRuntimeRhiAdapter &adapter,
                                                                      VernonRuntimeProviderObject encoder,
                                                                      const void *data, uint64_t size,
                                                                      uint64_t alignment) noexcept;
// Reserves such memory for the adapter to fill in place, on backends whose transient memory is host-visible.
RhiAdapterResult<vernon::rhi::CommandTransientReservation> reserveCommandTransient(VernonRuntimeRhiAdapter &adapter,
                                                                                   VernonRuntimeProviderObject encoder,
                                                                                   uint64_t size,
                                                                                   uint64_t alignment) noexcept;
RhiAdapterResult<void> deferCommandCleanup(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                           void *context, uint64_t object, void (*cleanup)(void *, uint64_t)) noexcept;
RhiAdapterResult<void> deferCommandRollback(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            void *context, uint64_t object,
                                            void (*rollback)(void *, uint64_t)) noexcept;

// A provider object that commands which used it keep alive until their command encoder retires.
struct RhiRetainedObject {
    RhiRetainedObject() = default;
    RhiRetainedObject(const RhiRetainedObject &) = delete;
    RhiRetainedObject &operator=(const RhiRetainedObject &) = delete;
    virtual ~RhiRetainedObject() = default;

    std::atomic<uint32_t> references{1};
};

void releaseRetainedObject(RhiRetainedObject *object) noexcept;
RhiAdapterResult<void> retainForCommands(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                         RhiRetainedObject &object) noexcept;

// Layout entry access is a mask: bit 0 reads, bit 1 writes.
inline bool writesBinding(const VernonRuntimeProviderBindingLayoutEntry &entry) noexcept {
    return (entry.access & 2u) != 0;
}

// A group keeps every resource its values name alive for as long as it exists.
// A block of the device's persistent memory that a group holds. Destroying the group returns it, which happens only
// after every encoder that bound the group has completed, since binding retains the group for its encoder.
class PersistentBlockLease {
public:
    PersistentBlockLease(VernonRhiDevice device, uint64_t key, bool updatable) noexcept
        : device_(device), key_(key), updatable_(updatable) {}
    PersistentBlockLease(PersistentBlockLease &&other) noexcept
        : device_(other.device_), key_(other.key_), updatable_(other.updatable_),
          active_(std::exchange(other.active_, false)) {}
    PersistentBlockLease(const PersistentBlockLease &) = delete;
    PersistentBlockLease &operator=(const PersistentBlockLease &) = delete;
    PersistentBlockLease &operator=(PersistentBlockLease &&) = delete;
    ~PersistentBlockLease() {
        if (!active_)
            return;
        if (updatable_)
            vernon::rhi::releaseUpdatableBlock(device_, key_);
        else
            vernon::rhi::releasePersistentBlock(device_, key_);
    }

private:
    VernonRhiDevice device_;
    uint64_t key_;
    bool updatable_;
    bool active_{true};
};

struct RhiGroup : RhiRetainedObject {
    std::vector<vernon::rhi::RetainedRhiResourceLease> leases;
    std::vector<PersistentBlockLease> blocks;
    // Resources the group's writable bindings name; every command that binds the group writes them.
    std::vector<VernonRuntimeProviderResourceReference> writes;
};

// Retains the resource for the group and returns its backend object; `image` receives an image's view description.
RhiAdapterResult<uint64_t> acquireGroupResource(VernonRuntimeRhiAdapter &adapter, RhiGroup &group,
                                                VernonRuntimeProviderResourceReference resource,
                                                RhiImageBinding *image = nullptr) noexcept;
// Places bytes in the device's persistent memory for `group`, which holds the block until it is destroyed; see
// PlacePersistentCallback.
RhiAdapterResult<vernon::rhi::PersistentBlock> placePersistent(VernonRuntimeRhiAdapter &adapter, RhiGroup &group,
                                                               const void *data, uint64_t size,
                                                               uint64_t alignment) noexcept;
// Places device memory for `group` that only `writeUpdatable` writes; see PlaceUpdatableCallback.
RhiAdapterResult<vernon::rhi::PersistentBlock> placeUpdatable(VernonRuntimeRhiAdapter &adapter, RhiGroup &group,
                                                              uint64_t size, uint64_t alignment) noexcept;
// Writes `data` into `block` in the order of the encoder's commands; see WriteUpdatableCallback.
RhiAdapterResult<void> writeUpdatable(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                      const vernon::rhi::PersistentBlock &block, const void *data,
                                      uint64_t size) noexcept;

// The command state every RHI provider keeps between begin_commands and end_commands.
struct RhiCommands {
    VernonRuntimeProviderObject encoder{};
    // The backend's command encoder: a VkCommandBuffer, ID3D12GraphicsCommandList, or the OpenGL device.
    uint64_t native{};
    bool rendering{};
};

RhiAdapterResult<void> beginRhiCommands(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                        RhiCommands &commands) noexcept;

inline RhiAdapterResult<void> providerFailure(vernon::ProviderErrorCode code, const char *operation, uint64_t value = 0,
                                              uint32_t detail = 0) noexcept {
    return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{code, {operation, value, detail}})};
}

// A provider packet batch with its objects and resources resolved once; see stored_plans.md section 11.3. Encoding it
// retains it for the encoder, so it keeps them alive until every encoder that encoded it retires.
struct RhiPacketBatch : RhiRetainedObject {
    RhiPacketBatch() = default;
    ~RhiPacketBatch() override;

    // A scope checked once when the batch is built; its clear values come from the payload.
    struct Rendering {
        // The attachment views and sampled images that `prepared` was checked against.
        std::vector<vernon::rhi::RetainedRhiResourceLease> leases;
        vernon::rhi::PreparedRendering prepared;
    };

    // The debug packets are dropped when the device does not emit labels.
    std::vector<uint32_t> packets;
    // Retained: the pipeline or group each object operand names.
    std::vector<RhiRetainedObject *> objects;
    std::vector<VernonRuntimeProviderResourceReference> resources;
    // The backend object of each resource, which `leases` keeps alive.
    std::vector<uint64_t> natives;
    std::vector<vernon::rhi::RetainedRhiResourceLease> leases;
    std::vector<Rendering> renderings;
    // The resources the batch's groups write.
    std::vector<vernon::rhi::CommandWrite> writes;
    size_t payloadWords{};
    uint32_t draws{};
    uint32_t dispatches{};
};

// The retained objects behind a backend's pipeline and group handles.
struct RhiPacketObjects {
    RhiRetainedObject *(*pipeline)(VernonRuntimeProviderObject) noexcept;
    RhiGroup *(*group)(VernonRuntimeProviderObject) noexcept;
};

RhiAdapterResult<void> createRhiPacketBatch(VernonRuntimeRhiAdapter &adapter,
                                            const VernonRuntimeProviderPacketBatchDescriptor *descriptor,
                                            RhiPacketObjects objects, VernonRuntimeProviderObject *output) noexcept;
void destroyRhiPacketBatch(void *, VernonRuntimeProviderObject batch);

// Begins the RHI scope of a BEGIN_RENDERING packet with the clear values at `clears`.
RhiAdapterResult<void> beginBatchRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                           const RhiPacketBatch::Rendering &rendering, const uint32_t *clears) noexcept;
RhiAdapterResult<void> recordPacketDebugCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                uint32_t opcode, const uint32_t *operands) noexcept;

// Records the batch's draws, dispatches and writes on the encoder after its packets were decoded. The last draw's
// stencil reference and indexing count only when the batch issued a draw.
RhiAdapterResult<void> finishRhiPacketBatch(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            const RhiPacketBatch &batch, bool drew, uint32_t lastStencilReference,
                                            bool lastDrawIndexed) noexcept;

/*
 * Decodes `batch` with `payload` on `encoder`, outside a rendering scope. The RHI sees the batch once, the rendering
 * scopes once each and the debug packets; `decoder` issues the native work: beginRendering(), endRendering() after the
 * RHI began and ended the scope,
 *   bindPipeline(RhiRetainedObject &)
 *   bindGroup(uint32_t index, RhiGroup &)
 *   bindVertexBuffers(uint32_t first, const uint32_t *resources, size_t count)
 *   draw(uint32_t topology, uint32_t indexResource or UINT32_MAX, const VernonRuntimeProviderDrawPayload &,
 *        const vernon::rhi::OpenedRendering &)
 *   drawStream(uint32_t topology, uint32_t indexResource, uint32_t recordsResource, uint32_t countsResource,
 *              const VernonRuntimeProviderStreamPayload &, const vernon::rhi::OpenedRendering &), whose records and
 *              counts resources are UINT32_MAX for a host stream
 *   dispatch(const uint32_t groupCount[3])
 * each returning RhiAdapterResult<void>.
 */
template <typename Decoder>
RhiAdapterResult<void> decodeRhiPackets(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                        RhiPacketBatch &batch, const uint32_t *payload, size_t payloadWords,
                                        Decoder &decoder) {
    if (payloadWords < batch.payloadWords || (batch.payloadWords != 0 && !payload))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "rhi_packet_payload_is_too_short",
                               payloadWords);
    if (auto retained = retainForCommands(adapter, encoder, batch); !retained)
        return retained;

    vernon::rhi::OpenedRendering opened{};
    bool open = false;
    bool drew = false;
    uint32_t stencilReference = 0;
    bool indexed = false;
    const auto decode = [&](uint32_t opcode, uint32_t count, const uint32_t *operands) -> RhiAdapterResult<void> {
        switch (opcode) {
        case VERNON_RUNTIME_PROVIDER_PACKET_BEGIN_RENDERING:
            open = false;
            if (auto begun =
                    beginBatchRendering(adapter, encoder, batch.renderings[operands[0]], payload + operands[1]);
                !begun)
                return begun;
            return decoder.beginRendering();
        case VERNON_RUNTIME_PROVIDER_PACKET_END_RENDERING:
            if (auto ended = endEncoderRendering(adapter, encoder); !ended)
                return ended;
            return decoder.endRendering();
        case VERNON_RUNTIME_PROVIDER_PACKET_BIND_PIPELINE:
            return decoder.bindPipeline(*batch.objects[operands[0]]);
        case VERNON_RUNTIME_PROVIDER_PACKET_BIND_GROUP:
            return decoder.bindGroup(operands[0], static_cast<RhiGroup &>(*batch.objects[operands[1]]));
        case VERNON_RUNTIME_PROVIDER_PACKET_BIND_VERTEX_BUFFERS:
            return decoder.bindVertexBuffers(operands[0], operands + 1, count - 2);
        case VERNON_RUNTIME_PROVIDER_PACKET_DRAW:
        case VERNON_RUNTIME_PROVIDER_PACKET_DRAW_INDEXED: {
            const bool drawIndexed = opcode == VERNON_RUNTIME_PROVIDER_PACKET_DRAW_INDEXED;
            VernonRuntimeProviderDrawPayload draw;
            std::memcpy(&draw, payload + operands[drawIndexed ? 2 : 1], sizeof(draw));
            if (!draw.instance_count)
                return RhiAdapterResult<void>{vernon::ok()};
            if (!open) {
                auto rendering = openCommandRendering(adapter, encoder);
                if (!rendering)
                    return RhiAdapterResult<void>{vernon::err(std::move(rendering).error())};
                opened = rendering.value();
                open = true;
            }
            drew = true;
            indexed = drawIndexed;
            stencilReference = draw.stencil_reference;
            return decoder.draw(operands[0], indexed ? operands[1] : UINT32_MAX, draw, opened);
        }
        case VERNON_RUNTIME_PROVIDER_PACKET_DRAW_INDEXED_STREAM: {
            VernonRuntimeProviderStreamPayload stream;
            std::memcpy(&stream, payload + operands[4], sizeof(stream));
            if (!stream.max_records)
                return RhiAdapterResult<void>{vernon::ok()};
            if (!open) {
                auto rendering = openCommandRendering(adapter, encoder);
                if (!rendering)
                    return RhiAdapterResult<void>{vernon::err(std::move(rendering).error())};
                opened = rendering.value();
                open = true;
            }
            drew = true;
            indexed = true;
            stencilReference = stream.stencil_reference;
            return decoder.drawStream(operands[0], operands[1], operands[2], operands[3], stream, opened);
        }
        case VERNON_RUNTIME_PROVIDER_PACKET_DISPATCH:
            return decoder.dispatch(payload + operands[0]);
        default:
            return recordPacketDebugCommand(adapter, encoder, opcode, operands);
        }
    };
    const uint32_t *word = batch.packets.data();
    const uint32_t *const end = word + batch.packets.size();
    while (word != end) {
        const uint32_t count = *word >> 16;
        if (auto decoded = decode(*word & 0xffffu, count, word + 1); !decoded)
            return decoded;
        word += count;
    }
    return finishRhiPacketBatch(adapter, encoder, batch, drew, stencilReference, indexed);
}

} // namespace vernon::runtime::rhi_adapter

#endif
