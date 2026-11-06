#ifndef VERNON_HAL_D3D12_D3D12_STATE_H
#define VERNON_HAL_D3D12_D3D12_STATE_H

// The D3D12 device a DirectX 12 HAL device owns: its adapter, device and direct queue, the fence that numbers its
// submissions, the command frames of the work it records outside passes, and the objects it releases once the work
// submitted before their release completes. Its resources, its runtime context and its HAL device share it; `mutex`
// guards everything but the native objects, which D3D12 makes thread safe.

#include "VernonHal.h"
#include "VernonTextureTypes.h"
#include "hal/staging_pool.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vernon::hal::d3d12 {

template <typename T> void release(T *&object) {
    if (object)
        object->Release();
    object = nullptr;
}

bool check(HRESULT result, const char *operation, std::string &error);

struct Buffer {
    ID3D12Resource *resource{};
    uint64_t size{};
    VernonHalMemory memory{VERNON_HAL_MEMORY_DEVICE};
    // Mapped for the buffer's life in upload and readback memory.
    uint8_t *mapped{};
};

// An image's resource, which may have the typeless format of its family so that views reinterpret it, and the states
// of its subresources, indexed as D3D12 does: `mip + layer * mipLevels + plane * mipLevels * layers`.
struct Image {
    ID3D12Resource *resource{};
    DXGI_FORMAT format{DXGI_FORMAT_UNKNOWN};
    VernonTextureFormat vernonFormat{};
    VernonTextureDimension dimension{VERNON_TEXTURE_2D};
    uint32_t mipLevels{};
    uint32_t layers{};
    uint32_t planes{1};
    D3D12_RESOURCE_FLAGS flags{};
    std::vector<D3D12_RESOURCE_STATES> states;

    uint32_t subresource(uint32_t mip, uint32_t layer, uint32_t plane) const {
        return mip + (layer + plane * layers) * mipLevels;
    }
};

// Descriptors in a heap the shaders cannot see, which recordings copy into their own heaps.
class DescriptorAllocator {
public:
    DescriptorAllocator() = default;
    DescriptorAllocator(const DescriptorAllocator &) = delete;
    DescriptorAllocator &operator=(const DescriptorAllocator &) = delete;
    ~DescriptorAllocator();

    void initialize(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type);
    // Zero when the device cannot create another heap.
    D3D12_CPU_DESCRIPTOR_HANDLE allocate(std::string &error);
    void free(D3D12_CPU_DESCRIPTOR_HANDLE handle);

private:
    static constexpr uint32_t kPerHeap = 256;

    ID3D12Device *device_{};
    D3D12_DESCRIPTOR_HEAP_TYPE type_{};
    uint32_t increment_{};
    std::vector<ID3D12DescriptorHeap *> heaps_;
    std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> free_;
};

// What passes bind of an image view: the subresources of `owner` it covers and the descriptors it was made with,
// each zero where the image's usage has none.
struct View {
    Image *owner{};
    DXGI_FORMAT format{DXGI_FORMAT_UNKNOWN};
    VernonTextureDimension dimension{VERNON_TEXTURE_2D};
    uint32_t baseMip{};
    uint32_t mips{};
    uint32_t baseLayer{};
    uint32_t layers{};
    uint32_t aspects{};
    D3D12_CPU_DESCRIPTOR_HANDLE srv{};
    D3D12_CPU_DESCRIPTOR_HANDLE uav{};
    // An integer color attachment's UAV, which clears it: ClearRenderTargetView converts its float color to the format
    // and cannot represent every 32-bit value.
    D3D12_CPU_DESCRIPTOR_HANDLE clear{};
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
};

struct Sampler {
    D3D12_SAMPLER_DESC descriptor{};
    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
};

struct StagingChunk {
    ID3D12Resource *resource{};
    uint8_t *mapped{};
    size_t capacity{};
};

// A command list and the staging it reads or writes, reused once the fence passes `value`.
struct CommandFrame {
    ID3D12CommandAllocator *allocator{};
    ID3D12GraphicsCommandList *commands{};
    uint64_t value{};
    StagingArena<StagingChunk> upload;
    StagingArena<StagingChunk> readback;
    // Shader-visible heaps of the frame's dispatches, released with the frame.
    std::vector<ID3D12DescriptorHeap *> heaps;
};

struct DeviceState {
    DeviceState() = default;
    DeviceState(const DeviceState &) = delete;
    DeviceState &operator=(const DeviceState &) = delete;
    ~DeviceState();

    bool initialize(uint32_t deviceIndex, uint32_t flags, std::string &error);
    void shutdown();

    // A frame records work outside passes; submitting it executes it after every earlier submission without a host
    // wait. The upload frame instead gathers host writes, which run before anything submitted after them.
    bool beginCommands(CommandFrame *&frame, std::string &error);
    bool submitCommands(CommandFrame *frame, uint64_t &value, std::string &error);
    void abandonCommands(CommandFrame *frame);
    bool uploadCommands(CommandFrame *&frame, std::string &error);
    bool flushUploads(std::string &error);
    // Executes `lists` after the pending uploads, as one ExecuteCommandLists, and signals the value that marks them.
    bool execute(ID3D12CommandList *const *lists, uint32_t count, uint64_t &value, std::string &error);

    bool waitForValue(uint64_t value, std::string &error);
    bool completed(uint64_t value, bool &done, std::string &error);
    bool synchronize(std::string &error);

    bool acquireStaging(CommandFrame &frame, bool upload, size_t size, size_t alignment, ID3D12Resource *&resource,
                        size_t &offset, uint8_t *&mapped, std::string &error);
    // A shader-visible heap the frame holds until it completes.
    bool frameHeap(CommandFrame &frame, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t count, ID3D12DescriptorHeap *&heap,
                   std::string &error);

    bool createBuffer(Buffer &buffer, uint64_t size, uint32_t usage, VernonHalMemory memory, std::string &error);
    // Releases `object` once the work submitted before the call completes.
    void releaseAfterSubmittedWork(IUnknown *object);
    void destroyBuffer(Buffer &buffer);

    std::mutex mutex;
    IDXGIFactory6 *factory{};
    IDXGIAdapter1 *adapter{};
    ID3D12Device *device{};
    ID3D12CommandQueue *queue{};
    ID3D12Fence *fence{};
    HANDLE fenceEvent{};
    uint64_t timestampFrequency{};
    bool tearing{};
    bool labels{};
    DescriptorAllocator resourceDescriptors;
    DescriptorAllocator samplerDescriptors;
    DescriptorAllocator renderTargets;
    DescriptorAllocator depthStencils;
    // Mipmap pipelines over the embedded box filters; null where the build had no shader compiler.
    ID3D12RootSignature *mipmapRoot{};
    ID3D12PipelineState *mipmap2d{};
    ID3D12PipelineState *mipmap3d{};
    uint64_t submittedValue{};
    uint64_t completedValue{};
    bool lost{};

private:
    struct DeferredRelease {
        uint64_t value{};
        IUnknown *object{};
    };

    bool createFrame(std::unique_ptr<CommandFrame> &frame, std::string &error);
    void destroyFrame(CommandFrame &frame);
    void recycle(std::unique_ptr<CommandFrame> frame);
    bool retire(uint64_t through, std::string &error);
    void releaseDeferred(bool all);
    bool createMipmapPipelines(std::string &error);

    std::vector<std::unique_ptr<CommandFrame>> available_;
    // Recording frames, the upload frame among them; submitted ones wait for their value in `inFlight_`.
    std::vector<std::unique_ptr<CommandFrame>> recording_;
    std::deque<std::unique_ptr<CommandFrame>> inFlight_;
    CommandFrame *uploads_{};
    std::vector<DeferredRelease> deferred_;
    StagingPool<StagingChunk> uploadStaging_;
    StagingPool<StagingChunk> readbackStaging_;
};

// The DXGI format of `format`, or DXGI_FORMAT_UNKNOWN.
DXGI_FORMAT dxgiFormat(VernonTextureFormat format);

} // namespace vernon::hal::d3d12

#endif
