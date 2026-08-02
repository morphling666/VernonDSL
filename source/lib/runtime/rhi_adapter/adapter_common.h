#ifndef VERNON_RUNTIME_RHI_ADAPTER_COMMON_H
#define VERNON_RUNTIME_RHI_ADAPTER_COMMON_H

#include "VernonRuntimeRHIAdapter.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

struct RhiAdapterBackendOps {
    void (*destroy)(void *) noexcept;
    VernonStatus (*synchronize)(void *, std::string &) noexcept;
    uint64_t (*resourceIdentity)(const void *) noexcept;
    void (*invalidate)(void *) noexcept;
};

struct RhiAdapterBackendStorage {
    RhiAdapterBackendStorage() = default;
    ~RhiAdapterBackendStorage();
    RhiAdapterBackendStorage(const RhiAdapterBackendStorage &) = delete;
    RhiAdapterBackendStorage &operator=(const RhiAdapterBackendStorage &) = delete;
    RhiAdapterBackendStorage(RhiAdapterBackendStorage &&other) noexcept;
    RhiAdapterBackendStorage &operator=(RhiAdapterBackendStorage &&other) noexcept;

    bool adopt(void *newState, const RhiAdapterBackendOps *newOps) noexcept;

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
    RhiAdapterBackendStorage backend;
    VernonRuntimeDeviceProvider provider{};
    std::string error;
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

inline constexpr uint64_t kRhiResourceKindMask = 3;
inline constexpr uint64_t kRhiImageResource = 1;
inline constexpr uint64_t kRhiSamplerResource = 2;
inline constexpr uint64_t kRhiBufferResource = 3;

template <typename Object> VernonRuntimeProviderObject toHandle(Object *object) {
    return {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(object))};
}

template <typename Object> Object *fromHandle(VernonRuntimeProviderObject handle) {
    return reinterpret_cast<Object *>(static_cast<uintptr_t>(handle.value));
}

VernonStatus fail(VernonRuntimeRhiAdapter &adapter, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT);
void setBackendError(std::string &error, const char *message) noexcept;
bool retainRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource);
void releaseRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource);
uint64_t resolveRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource);
uint64_t nativeCommandEncoder(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
bool commandEncoderRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
bool commandEncoderHasRenderingDescriptor(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
bool commandColorOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, size_t index,
                            VernonRhiLoadOperation &load, VernonRhiStoreOperation &store, float clear[4]);
bool commandDepthOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                            VernonRhiLoadOperation &depthLoad, VernonRhiStoreOperation &depthStore,
                            VernonRhiLoadOperation &stencilLoad, VernonRhiStoreOperation &stencilStore,
                            float &clearDepth, uint32_t &clearStencil);
int claimCommandRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, uint32_t backendKind);
uint64_t commandRenderingObject(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                uint64_t candidate);
bool recordProviderCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, bool draw);
bool recordCommandWriteResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                VernonRuntimeProviderResourceReference resource);
bool retainCommandResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                           VernonRuntimeProviderResourceReference resource);
bool deferCommandCleanup(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, void *context,
                         uint64_t object, void (*cleanup)(void *, uint64_t));
bool deferCommandRollback(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, void *context,
                          uint64_t object, void (*rollback)(void *, uint64_t));
bool setCommandRenderingTargets(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                const uint64_t *colors, const uint64_t *resources, size_t colorCount, uint64_t depth,
                                uint64_t depthResource);

} // namespace vernon::runtime::rhi_adapter

#endif
