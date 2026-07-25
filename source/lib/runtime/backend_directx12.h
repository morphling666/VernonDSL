#ifndef VERNON_RUNTIME_BACKEND_DIRECTX12_H
#define VERNON_RUNTIME_BACKEND_DIRECTX12_H

#include "pipeline_metadata.h"
#include "runtime_state.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#endif

#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::runtime {

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
struct DirectX12ContextState {
    IDXGIFactory6 *factory{};
    IDXGIAdapter1 *adapter{};
    ID3D12Device *device{};
    ID3D12CommandQueue *queue{};
    ID3D12CommandAllocator *allocator{};
    ID3D12GraphicsCommandList *commands{};
    ID3D12Fence *fence{};
    HANDLE fenceEvent{};
    uint64_t fenceValue{};
    D3D_FEATURE_LEVEL featureLevel{D3D_FEATURE_LEVEL_11_0};
    D3D_SHADER_MODEL shaderModel{D3D_SHADER_MODEL_6_0};
    D3D_ROOT_SIGNATURE_VERSION rootSignatureVersion{D3D_ROOT_SIGNATURE_VERSION_1_0};
    D3D12_RESOURCE_BINDING_TIER resourceBindingTier{D3D12_RESOURCE_BINDING_TIER_1};
    uint32_t maxComputeWorkGroupSize[3]{1024, 1024, 64};
    uint32_t maxComputeInvocations{1024};
};

struct DirectX12BufferState {
    ID3D12Resource *resource{};
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
};

struct DirectX12TextureState {
    ID3D12Resource *resource{};
    DXGI_FORMAT format{DXGI_FORMAT_UNKNOWN};
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
};

struct DirectX12SamplerState {};

struct DirectX12KernelState {
    ID3D12RootSignature *rootSignature{};
    ID3D12PipelineState *pipeline{};
    uint32_t descriptorCount{};
};

struct DirectX12PipelineState {
    VernonLoadedKernel *kernel{};
    std::vector<uint8_t> vertexDxil;
    std::vector<uint8_t> fragmentDxil;
    std::unordered_map<std::string, ID3D12PipelineState *> graphicsPipelines;
    size_t graphicsPipelineCreations{};
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
#else
struct DirectX12KernelState {};
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

bool loadDirectX12Kernel(VernonRuntimeContext &context, const void *artifact, size_t artifactSize,
                         const char *reflection, size_t reflectionSize, const char *entry, size_t entrySize,
                         DirectX12KernelState &state, ReflectedEntry &metadata);
void destroyDirectX12Kernel(VernonRuntimeContext &context, DirectX12KernelState &state);
VernonStatus launchDirectX12Kernel(VernonRuntimeContext &context, const DirectX12KernelState &state,
                                   const ReflectedEntry &metadata, VernonLaunchSize globalSize,
                                   const VernonLaunchArgument *arguments, size_t argumentCount);

} // namespace vernon::runtime

#endif
