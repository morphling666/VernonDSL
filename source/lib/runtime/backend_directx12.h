#ifndef VERNON_RUNTIME_BACKEND_DIRECTX12_H
#define VERNON_RUNTIME_BACKEND_DIRECTX12_H

#include "VernonRuntimeCore.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "../rhi/directx12_backend.h"
#include <d3d12.h>
#endif

#include <string>
#include <vector>

struct VernonRuntimeRhiAdapter;

namespace vernon::runtime {

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
struct DirectX12ContextState : rhi::directx12::DeviceState {
    VernonRuntimeRhiAdapter *adapter{};
};
using DirectX12BufferState = rhi::directx12::Buffer;
using DirectX12TextureState = rhi::directx12::Image;
using DirectX12SamplerState = rhi::directx12::Sampler;

struct DirectX12PipelineState {
    struct GraphicsBinding {
        enum Source { EXTERNAL_VERTEX, EXTERNAL_TEXTURE, EXTERNAL_SAMPLER, IMPLICIT_SAMPLER };
        Source source{};
        uint32_t externalSlot{};
    };

    VernonRuntimeCorePipeline *rhiComputePipeline{};
    VernonRuntimeCoreBindings *rhiComputeBindings{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiComputeLayout;
    std::vector<VernonRuntimeProviderBindingValue> rhiComputeValues;
    std::vector<uint64_t> rhiComputeResourceOffsets;
    uint32_t rhiComputeWorkgroup[3]{1, 1, 1};
    VernonRuntimeCorePipeline *rhiGraphicsPipeline{};
    VernonRuntimeCoreBindings *rhiGraphicsBindings{};
    VernonRuntimeCoreGraphicsVariant *rhiGraphicsVariant{};
    std::vector<uint32_t> rhiGraphicsFormats;
    uint64_t rhiGraphicsVertexLayoutIdentity{};
    uint32_t rhiGraphicsTopology{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiGraphicsLayout;
    std::vector<VernonRuntimeProviderBindingValue> rhiGraphicsValues;
    std::vector<GraphicsBinding> rhiGraphicsBindingsPlan;
};

inline DirectX12ContextState &directX12State(VernonRuntimeContext &context) {
    return runtimeBackendState<DirectX12ContextState>(context);
}
inline const DirectX12ContextState &directX12State(const VernonRuntimeContext &context) {
    return runtimeBackendState<DirectX12ContextState>(context);
}
inline DirectX12BufferState &directX12BufferState(VernonDeviceBuffer &buffer) {
    return runtimeBackendState<DirectX12BufferState>(buffer);
}
inline const DirectX12BufferState &directX12BufferState(const VernonDeviceBuffer &buffer) {
    return runtimeBackendState<DirectX12BufferState>(buffer);
}
inline DirectX12TextureState &directX12TextureState(VernonDeviceTexture &texture) {
    return runtimeBackendState<DirectX12TextureState>(texture);
}
inline const DirectX12TextureState &directX12TextureState(const VernonDeviceTexture &texture) {
    return runtimeBackendState<DirectX12TextureState>(texture);
}
#endif

bool probeDirectX12(std::string &diagnostic);
bool initializeDirectX12Context(VernonRuntimeContext &context, uint32_t deviceIndex);
void destroyDirectX12Context(VernonRuntimeContext &context);
VernonStatus synchronizeDirectX12(VernonRuntimeContext &context);

bool createDirectX12Buffer(VernonDeviceBuffer &buffer);
void destroyDirectX12Buffer(VernonDeviceBuffer &buffer);
VernonStatus copyToDirectX12Buffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromDirectX12Buffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

bool createDirectX12Texture(VernonDeviceTexture &texture);
void destroyDirectX12Texture(VernonDeviceTexture &texture);
VernonStatus copyToDirectX12Texture(VernonDeviceTexture &texture, const void *source, size_t size);
VernonStatus copyFromDirectX12Texture(const VernonDeviceTexture &texture, void *destination, size_t size);
bool createDirectX12Sampler(VernonDeviceSampler &sampler);
void destroyDirectX12Sampler(VernonDeviceSampler &sampler);

} // namespace vernon::runtime

#endif
