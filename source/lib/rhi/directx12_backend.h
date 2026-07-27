#ifndef VERNON_RHI_DIRECTX12_BACKEND_H
#define VERNON_RHI_DIRECTX12_BACKEND_H

#include "VernonRHI.h"

#if defined(_WIN32)
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vernon::rhi::directx12 {

#if defined(_WIN32)

struct Buffer {
    ID3D12Resource *resource{};
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
    bool owned{true};
};

struct Image {
    ID3D12Resource *resource{};
    DXGI_FORMAT format{DXGI_FORMAT_UNKNOWN};
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
    bool owned{true};
};

struct Sampler {
    D3D12_SAMPLER_DESC descriptor{};
};

struct VERNON_RHI_CAPI DeviceState {
    struct CommandFrame {
        ID3D12CommandAllocator *allocator{};
        ID3D12GraphicsCommandList *commands{};
        uint64_t completionValue{};
    };

    struct StagingRing {
        Buffer buffer;
        uint8_t *mapped{};
        size_t capacity{};
        size_t cursor{};
    };

    struct DescriptorRing {
        ID3D12DescriptorHeap *heap{};
        uint32_t capacity{};
        uint32_t cursor{};
        std::vector<ID3D12DescriptorHeap *> retiredHeaps;
    };

    ~DeviceState();

    bool initialize(uint32_t deviceIndex, bool forceWarp, std::string &error);
    bool initializeBorrowed(ID3D12Device *borrowedDevice, ID3D12CommandQueue *borrowedQueue,
                            ID3D12GraphicsCommandList *borrowedCommands, std::string &error);
    void shutdown();
    bool synchronize(std::string &error);
    bool beginCommands(std::string &error, ID3D12PipelineState *initialState = nullptr);
    bool submitCommands(std::string &error);
    void recycleCommandStorage();
    ID3D12GraphicsCommandList *commandList() const;
    bool createBuffer(Buffer &buffer, size_t size, bool unorderedAccess, D3D12_HEAP_TYPE heapType,
                      D3D12_RESOURCE_STATES initialState, std::string &error);
    void destroyBuffer(Buffer &buffer);
    bool createImage(Image &image, const D3D12_RESOURCE_DESC &descriptor, DXGI_FORMAT format,
                     D3D12_RESOURCE_STATES initialState, std::string &error);
    void destroyImage(Image &image);
    bool acquireStaging(bool upload, size_t size, size_t alignment, ID3D12Resource *&resource, size_t &offset,
                        uint8_t *&mapped, std::string &error);
    bool acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE type, bool shaderVisible, uint32_t count,
                            ID3D12DescriptorHeap *&heap, D3D12_CPU_DESCRIPTOR_HANDLE &cpu,
                            D3D12_GPU_DESCRIPTOR_HANDLE &gpu, std::string &error);

    IDXGIFactory6 *factory{};
    IDXGIAdapter1 *adapter{};
    ID3D12Device *device{};
    ID3D12CommandQueue *queue{};
    CommandFrame frame;
    ID3D12GraphicsCommandList *borrowedCommands{};
    ID3D12Fence *fence{};
    HANDLE fenceEvent{};
    uint64_t fenceValue{};
    D3D_FEATURE_LEVEL featureLevel{D3D_FEATURE_LEVEL_11_0};
    D3D_SHADER_MODEL shaderModel{D3D_SHADER_MODEL_6_0};
    D3D_ROOT_SIGNATURE_VERSION rootSignatureVersion{D3D_ROOT_SIGNATURE_VERSION_1_0};
    D3D12_RESOURCE_BINDING_TIER resourceBindingTier{D3D12_RESOURCE_BINDING_TIER_1};
    uint32_t maxComputeWorkGroupSize[3]{1024, 1024, 64};
    uint32_t maxComputeInvocations{1024};
    StagingRing uploadRing;
    StagingRing readbackRing;
    DescriptorRing resourceDescriptorRing;
    DescriptorRing samplerDescriptorRing;
    DescriptorRing rtvDescriptorRing;
    DescriptorRing dsvDescriptorRing;
    bool nativeObjectsBorrowed{};
};

VERNON_RHI_CAPI bool check(HRESULT result, const char *operation, std::string &error);
VERNON_RHI_CAPI void transition(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource,
                                D3D12_RESOURCE_STATES &current, D3D12_RESOURCE_STATES target);

#endif

} // namespace vernon::rhi::directx12

#endif
