#ifndef VERNON_RUNTIME_RHI_ADAPTER_COMMON_H
#define VERNON_RUNTIME_RHI_ADAPTER_COMMON_H

#include "VernonError.hpp"
#include "VernonRuntimeRHIAdapter.h"
#include "rhi/rhi_internal.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
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
RhiAdapterResult<uint64_t> resolveCommandRhiResource(VernonRuntimeRhiAdapter &adapter,
                                                     VernonRuntimeProviderObject encoder,
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
RhiAdapterResult<void> recordProviderCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             bool draw) noexcept;
RhiAdapterResult<void> recordCommandWriteResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  VernonRuntimeProviderResourceReference resource) noexcept;
RhiAdapterResult<void> retainCommandResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             VernonRuntimeProviderResourceReference resource) noexcept;
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
    PersistentBlockLease(VernonRhiDevice device, uint64_t key) noexcept : device_(device), key_(key) {}
    PersistentBlockLease(PersistentBlockLease &&other) noexcept
        : device_(other.device_), key_(other.key_), active_(std::exchange(other.active_, false)) {}
    PersistentBlockLease(const PersistentBlockLease &) = delete;
    PersistentBlockLease &operator=(const PersistentBlockLease &) = delete;
    PersistentBlockLease &operator=(PersistentBlockLease &&) = delete;
    ~PersistentBlockLease() {
        if (active_)
            vernon::rhi::releasePersistentBlock(device_, key_);
    }

private:
    VernonRhiDevice device_;
    uint64_t key_;
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
// Records the group's written resources against the encoder of a command that bound it.
RhiAdapterResult<void> recordGroupWrites(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                         const RhiGroup &group) noexcept;

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

} // namespace vernon::runtime::rhi_adapter

#endif
