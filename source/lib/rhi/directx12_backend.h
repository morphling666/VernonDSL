#ifndef VERNON_RHI_DIRECTX12_BACKEND_H
#define VERNON_RHI_DIRECTX12_BACKEND_H

#include "VernonError.hpp"
#include "VernonRHI.h"
#include "VernonResult.hpp"
#include "staging_pool.h"

#if defined(_WIN32)
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#endif

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::rhi::directx12 {

#if defined(_WIN32)

struct Buffer {
    ID3D12Resource *resource{};
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
    bool owned{true};
};

struct Image {
    struct StateJournal {
        D3D12_RESOURCE_STATES state{};
        std::vector<D3D12_RESOURCE_STATES> subresources;
    };

    ID3D12Resource *resource{};
    DXGI_FORMAT format{DXGI_FORMAT_UNKNOWN};
    VernonRhiImageDimension dimension{VERNON_RHI_IMAGE_2D};
    // An image tracks the state of each of its subresources in `subresourceStates`, indexed by the D3D12 subresource
    // index `mip + layer * mipLevels + plane * mipLevels * layers`; `state` is the state they last shared. A view
    // tracks nothing itself: it covers the subresources of `stateOwner` that `view` selects, so every view of an image
    // observes the same states.
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
    std::vector<D3D12_RESOURCE_STATES> subresourceStates;
    std::unordered_map<uint64_t, StateJournal> stateJournals;
    uint32_t mipLevels{};
    uint32_t planeCount{1};
    Image *stateOwner{};
    bool owned{true};
    VernonRhiImageViewDescriptor view{};
};

struct Sampler {
    D3D12_SAMPLER_DESC descriptor{};
};

struct VERNON_RHI_CAPI DeviceState {
    struct StagingChunk {
        Buffer buffer;
        uint8_t *mapped{};
        size_t capacity{};
    };

    struct DescriptorRing {
        ID3D12DescriptorHeap *heap{};
        uint32_t capacity{};
        uint32_t cursor{};
        std::vector<ID3D12DescriptorHeap *> retiredHeaps;
    };

    // Everything a recording allocates for the GPU to read lives in its frame and is reused only after the fence
    // reaches the frame's `completionValue`.
    struct CommandFrame {
        ID3D12CommandAllocator *allocator{};
        ID3D12GraphicsCommandList *commands{};
        uint64_t completionValue{};
        StagingArena<StagingChunk> upload;
        StagingArena<StagingChunk> readback;
        DescriptorRing resourceDescriptors;
        // CPU-only resource views, for calls such as ClearUnorderedAccessViewUint that read a view from a heap the
        // shader cannot see.
        DescriptorRing stagedResourceDescriptors;
        DescriptorRing samplerDescriptors;
        DescriptorRing rtvDescriptors;
        DescriptorRing dsvDescriptors;
    };

    struct DeferredRelease {
        uint64_t value{};
        ID3D12Resource *resource{};
    };

    static constexpr size_t maxFramesInFlight = 64;

    ~DeviceState();

    bool initialize(uint32_t deviceIndex, bool forceWarp, std::string &error);
    bool initializeBorrowed(ID3D12Device *borrowedDevice, ID3D12CommandQueue *borrowedQueue,
                            ID3D12GraphicsCommandList *borrowedCommands, std::string &error);
    void shutdown();
    bool synchronize(std::string &error);
    // A device records one command list at a time; `commandList()` names it until it is submitted or abandoned.
    bool beginCommands(std::string &error, ID3D12PipelineState *initialState = nullptr);
    // Submits without waiting and returns the fence value that marks its completion. A borrowed list is executed by
    // its owner; its submission only numbers what borrowedFrame holds and moves it to inFlightFrames.
    bool submitCommands(uint64_t &value, std::string &error);
    bool waitForValue(uint64_t value, std::string &error);
    bool pollValue(uint64_t value, bool &completed, std::string &error);
    // Recycles the frames of borrowed submission `value` and of those before it, once the host reports that the
    // submission executed. The host executes the borrowed list in recording order, but later submissions may not
    // have executed yet.
    void retireBorrowedCommands(uint64_t value);
    void abandonCommands();
    ID3D12GraphicsCommandList *commandList() const;
    bool createBuffer(Buffer &buffer, size_t size, bool unorderedAccess, D3D12_HEAP_TYPE heapType,
                      D3D12_RESOURCE_STATES initialState, std::string &error);
    // Owned resources are released once the work submitted before the call completes.
    void destroyBuffer(Buffer &buffer);
    bool createImage(Image &image, const D3D12_RESOURCE_DESC &descriptor, DXGI_FORMAT format,
                     D3D12_RESOURCE_STATES initialState, std::string &error);
    void destroyImage(Image &image);
    // Staging and descriptors belong to the current recording.
    bool acquireStaging(bool upload, size_t size, size_t alignment, ID3D12Resource *&resource, size_t &offset,
                        uint8_t *&mapped, std::string &error);
    bool acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE type, bool shaderVisible, uint32_t count,
                            ID3D12DescriptorHeap *&heap, D3D12_CPU_DESCRIPTOR_HANDLE &cpu,
                            D3D12_GPU_DESCRIPTOR_HANDLE &gpu, std::string &error);

    IDXGIFactory6 *factory{};
    IDXGIAdapter1 *adapter{};
    ID3D12Device *device{};
    ID3D12CommandQueue *queue{};
    ID3D12GraphicsCommandList *borrowedCommands{};
    ID3D12Fence *fence{};
    HANDLE fenceEvent{};
    D3D_FEATURE_LEVEL featureLevel{D3D_FEATURE_LEVEL_11_0};
    D3D_SHADER_MODEL shaderModel{D3D_SHADER_MODEL_6_0};
    D3D_ROOT_SIGNATURE_VERSION rootSignatureVersion{D3D_ROOT_SIGNATURE_VERSION_1_0};
    D3D12_RESOURCE_BINDING_TIER resourceBindingTier{D3D12_RESOURCE_BINDING_TIER_1};
    uint32_t maxComputeWorkGroupSize[3]{1024, 1024, 64};
    uint32_t maxComputeInvocations{1024};
    std::vector<CommandFrame> availableFrames;
    CommandFrame recordingFrame;
    bool recording{};
    std::deque<CommandFrame> inFlightFrames;
    CommandFrame borrowedFrame;
    std::vector<DeferredRelease> deferredReleases;
    StagingPool<StagingChunk> uploadStaging;
    StagingPool<StagingChunk> readbackStaging;
    uint64_t submittedValue{};
    uint64_t completedValue{};
    bool deviceLost{};
    bool nativeObjectsBorrowed{};

private:
    CommandFrame &currentFrame();
    bool createCommandFrame(CommandFrame &frame, std::string &error);
    void destroyCommandFrame(CommandFrame &frame);
    void recycleCommandFrame(CommandFrame frame);
    // Keeps the storage of a frame whose work completed for its next recording.
    void resetFrameStorage(CommandFrame &frame);
    bool createStagingChunk(bool upload, size_t capacity, StagingChunk &chunk, std::string &error);
    bool retireCommandFrames(uint64_t through, std::string &error);
    void releaseDeferred(bool all);
    void releaseAfterSubmittedWork(ID3D12Resource *resource);
};

VERNON_RHI_CAPI bool check(HRESULT result, const char *operation, std::string &error);
VERNON_RHI_CAPI void transition(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource,
                                D3D12_RESOURCE_STATES &current, D3D12_RESOURCE_STATES target);

struct ImageSubresourceRange {
    uint32_t firstMip{};
    uint32_t mipCount{};
    uint32_t firstLayer{};
    uint32_t layerCount{};
};

// Raw images expose every mip and layer; image views carry their own subresource range.
VERNON_RHI_CAPI ImageSubresourceRange imageSubresourceRange(const Image &image);

// Transitions the subresources an image or image view `image` covers within the encoder `encoderKey`; abandoning the
// encoder restores the states its image had before the encoder's first transition of it.
VERNON_RHI_CAPI Result<void, RhiError> transitionCommandImage(VernonRhiDevice device, uint64_t encoderKey,
                                                              ID3D12GraphicsCommandList *commands, Image &image,
                                                              D3D12_RESOURCE_STATES target) noexcept;

#endif

} // namespace vernon::rhi::directx12

#endif
