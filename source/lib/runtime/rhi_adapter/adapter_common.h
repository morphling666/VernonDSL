#ifndef VERNON_RUNTIME_RHI_ADAPTER_COMMON_H
#define VERNON_RUNTIME_RHI_ADAPTER_COMMON_H

#include "VernonError.hpp"
#include "VernonRuntimeRHIAdapter.h"
#include "rhi/backend_dispatch.h"

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
    ~VernonRuntimeRhiAdapter();
    VernonRuntimeRhiAdapter(const VernonRuntimeRhiAdapter &) = delete;
    VernonRuntimeRhiAdapter &operator=(const VernonRuntimeRhiAdapter &) = delete;

    VernonRhiDevice rhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiBackend rhiBackend{};
    vernon::Option<vernon::ChildLease> rhiDeviceLease;
    RhiAdapterBackendStorage backend;
    VernonRuntimeDeviceProvider provider{};
    std::string error;
    struct RetainedLeaseEntry {
        RetainedLeaseEntry(vernon::rhi::ResourceKind newKind, uint64_t newKey,
                           vernon::rhi::RetainedRhiResourceLease newLease) noexcept
            : kind(newKind), key(newKey), lease(std::move(newLease)) {}

        vernon::rhi::ResourceKind kind;
        uint64_t key;
        vernon::rhi::RetainedRhiResourceLease lease;
        std::unique_ptr<RetainedLeaseEntry> next;
    };
    mutable std::mutex retainedLeaseMutex;
    std::unique_ptr<RetainedLeaseEntry> retainedLeaseHead;
    std::atomic<size_t> shaderPreparations{};
    std::atomic<size_t> layoutPreparations{};
    std::atomic<size_t> pipelinePreparations{};
    std::atomic<size_t> bindingCreations{};
    std::atomic<size_t> bindingSnapshotCreations{};
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

inline bool isRhiImageViewReference(VernonRuntimeProviderResourceReference resource) {
    return (resource.identity & kRhiResourceKindMask) == kRhiImageViewResource;
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
RhiAdapterResult<void> retainRhiResource(VernonRuntimeRhiAdapter &adapter,
                                         VernonRuntimeProviderResourceReference resource) noexcept;
RhiAdapterResult<void> releaseRetainedRhiResource(VernonRuntimeRhiAdapter &adapter,
                                                  VernonRuntimeProviderResourceReference resource) noexcept;
RhiAdapterResult<uint64_t> resolveRhiResource(VernonRuntimeRhiAdapter &adapter,
                                              VernonRuntimeProviderResourceReference resource) noexcept;
RhiAdapterResult<uint64_t> resolveCommandRhiResource(VernonRuntimeRhiAdapter &adapter,
                                                     VernonRuntimeProviderObject encoder,
                                                     VernonRuntimeProviderResourceReference resource) noexcept;
RhiAdapterResult<void> describeRhiImage(VernonRuntimeRhiAdapter &adapter,
                                        VernonRuntimeProviderResourceReference resource,
                                        VernonRhiImageDescriptor &descriptor) noexcept;
RhiAdapterResult<void> describeProviderImage(VernonRuntimeRhiAdapter &adapter,
                                             VernonRuntimeProviderResourceReference resource,
                                             VernonRuntimeProviderImageDescription &description) noexcept;
VernonStatus retainRhiResourceCallback(void *data, VernonRuntimeProviderResourceReference resource);
void releaseRhiResourceCallback(void *data, VernonRuntimeProviderResourceReference resource);
VernonStatus describeProviderImageCallback(void *data, VernonRuntimeProviderResourceReference resource,
                                           VernonRuntimeProviderImageDescription *description);
const VernonRuntimeProviderResourceReference *providerBindingResource(const VernonRuntimeProviderBindingValue &value);
RhiAdapterResult<uint64_t> nativeCommandEncoder(VernonRuntimeRhiAdapter &adapter,
                                                VernonRuntimeProviderObject encoder) noexcept;
bool commandEncoderRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
bool commandEncoderHasRenderingDescriptor(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
RhiAdapterResult<void> commandColorOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                              size_t index, VernonRhiLoadOperation &load,
                                              VernonRhiStoreOperation &store, float clear[4]) noexcept;
RhiAdapterResult<void> commandDepthOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                              VernonRhiLoadOperation &depthLoad, VernonRhiStoreOperation &depthStore,
                                              VernonRhiLoadOperation &stencilLoad,
                                              VernonRhiStoreOperation &stencilStore, float &clearDepth,
                                              uint32_t &clearStencil) noexcept;
RhiAdapterResult<int> claimCommandRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            uint32_t backendKind) noexcept;
RhiAdapterResult<uint64_t> commandRenderingObject(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  uint64_t candidate) noexcept;
RhiAdapterResult<void> recordProviderCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             bool draw) noexcept;
RhiAdapterResult<void> recordCommandWriteResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  VernonRuntimeProviderResourceReference resource) noexcept;
RhiAdapterResult<void> retainCommandResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             VernonRuntimeProviderResourceReference resource) noexcept;
bool validCommonDrawDescriptor(const VernonRuntimeProviderDrawDescriptor *descriptor);
RhiAdapterResult<void> deferCommandCleanup(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                           void *context, uint64_t object, void (*cleanup)(void *, uint64_t)) noexcept;
RhiAdapterResult<void> deferCommandRollback(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            void *context, uint64_t object,
                                            void (*rollback)(void *, uint64_t)) noexcept;
RhiAdapterResult<void> setCommandRenderingTargets(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  const uint64_t *colors, const uint64_t *resources, size_t colorCount,
                                                  uint64_t depth, uint64_t depthResource) noexcept;

} // namespace vernon::runtime::rhi_adapter

#endif
