#include "directx12_backend.h"

#if defined(_WIN32)

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <utility>

namespace vernon::rhi::directx12 {
namespace {

template <typename T> void release(T *&value) {
    if (value)
        value->Release();
    value = nullptr;
}

D3D12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    result.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    result.CreationNodeMask = 1;
    result.VisibleNodeMask = 1;
    return result;
}

D3D12_RESOURCE_DESC bufferDescription(size_t size, bool unorderedAccess) {
    D3D12_RESOURCE_DESC result{};
    result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    result.Width = std::max<size_t>(4, (size + 3) & ~size_t{3});
    result.Height = 1;
    result.DepthOrArraySize = 1;
    result.MipLevels = 1;
    result.Format = DXGI_FORMAT_UNKNOWN;
    result.SampleDesc.Count = 1;
    result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    result.Flags = unorderedAccess ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    return result;
}

void destroyStagingChunk(DeviceState::StagingChunk &chunk) {
    if (chunk.buffer.resource && chunk.mapped)
        chunk.buffer.resource->Unmap(0, nullptr);
    release(chunk.buffer.resource);
}

void releaseDescriptorRing(DeviceState::DescriptorRing &ring) {
    release(ring.heap);
    for (ID3D12DescriptorHeap *heap : ring.retiredHeaps)
        release(heap);
    ring.retiredHeaps.clear();
    ring.capacity = 0;
    ring.cursor = 0;
}

std::array<DeviceState::DescriptorRing *, 5> descriptorRings(DeviceState::CommandFrame &frame) {
    return {&frame.resourceDescriptors, &frame.stagedResourceDescriptors, &frame.samplerDescriptors,
            &frame.rtvDescriptors, &frame.dsvDescriptors};
}

void queryCapabilities(DeviceState &device) {
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModelQuery{D3D_SHADER_MODEL_6_8};
    HRESULT shaderModelResult =
        device.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModelQuery, sizeof(shaderModelQuery));
    while (shaderModelResult == E_INVALIDARG && shaderModelQuery.HighestShaderModel > D3D_SHADER_MODEL_6_0) {
        shaderModelQuery.HighestShaderModel =
            static_cast<D3D_SHADER_MODEL>(static_cast<uint32_t>(shaderModelQuery.HighestShaderModel) - 1);
        shaderModelResult =
            device.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModelQuery, sizeof(shaderModelQuery));
    }
    device.shaderModel = FAILED(shaderModelResult) ? D3D_SHADER_MODEL_5_1 : shaderModelQuery.HighestShaderModel;
    D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignature{D3D_ROOT_SIGNATURE_VERSION_1_1};
    if (FAILED(device.device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &rootSignature, sizeof(rootSignature))))
        rootSignature.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    device.rootSignatureVersion = rootSignature.HighestVersion;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (SUCCEEDED(device.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))))
        device.resourceBindingTier = options.ResourceBindingTier;
}

} // namespace

bool check(HRESULT result, const char *operation, std::string &error) {
    if (SUCCEEDED(result))
        return true;
    error = std::string(operation) + " failed with HRESULT 0x";
    constexpr char digits[] = "0123456789abcdef";
    const uint32_t code = static_cast<uint32_t>(result);
    for (int shift = 28; shift >= 0; shift -= 4)
        error.push_back(digits[(code >> shift) & 0xf]);
    return false;
}

void transition(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource, D3D12_RESOURCE_STATES &current,
                D3D12_RESOURCE_STATES target) {
    if (current == target)
        return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = current;
    barrier.Transition.StateAfter = target;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commands->ResourceBarrier(1, &barrier);
    current = target;
}

DeviceState::~DeviceState() { shutdown(); }

bool DeviceState::initialize(uint32_t deviceIndex, bool forceWarp, std::string &error) {
    shutdown();
    if (!check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1", error))
        return false;
    if (forceWarp) {
        if (deviceIndex != 0 ||
            !check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "IDXGIFactory::EnumWarpAdapter", error)) {
            shutdown();
            return false;
        }
    }
    for (uint32_t index = 0; !adapter; ++index) {
        IDXGIAdapter1 *candidate = nullptr;
        if (factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 description{};
        candidate->GetDesc1(&description);
        if (!(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(candidate, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
            if (deviceIndex == 0) {
                adapter = candidate;
                break;
            }
            --deviceIndex;
        }
        release(candidate);
    }
    if (!adapter) {
        error = "requested D3D12 hardware adapter was not found";
        shutdown();
        return false;
    }
    static constexpr D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1,
                                                   D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_1,
                                                   D3D_FEATURE_LEVEL_11_0};
    for (D3D_FEATURE_LEVEL level : levels)
        if (SUCCEEDED(D3D12CreateDevice(adapter, level, IID_PPV_ARGS(&device)))) {
            featureLevel = level;
            break;
        }
    if (!device) {
        error = "D3D12CreateDevice failed for the selected adapter";
        shutdown();
        return false;
    }
    queryCapabilities(*this);
    D3D12_COMMAND_QUEUE_DESC queueDescription{};
    queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!check(device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&queue)), "ID3D12Device::CreateCommandQueue",
               error) ||
        !check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "ID3D12Device::CreateFence",
               error)) {
        shutdown();
        return false;
    }
    CommandFrame frame;
    if (!createCommandFrame(frame, error)) {
        shutdown();
        return false;
    }
    availableFrames.push_back(std::move(frame));
    fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent) {
        error = "CreateEventW failed for the D3D12 fence";
        shutdown();
        return false;
    }
    return true;
}

bool DeviceState::initializeBorrowed(ID3D12Device *borrowedDevice, ID3D12CommandQueue *borrowedQueue,
                                     ID3D12GraphicsCommandList *borrowedCommands, std::string &error) {
    shutdown();
    if (!borrowedDevice || !borrowedQueue || !borrowedCommands) {
        error = "borrowed D3D12 device, queue, and command list are required";
        return false;
    }
    device = borrowedDevice;
    queue = borrowedQueue;
    this->borrowedCommands = borrowedCommands;
    nativeObjectsBorrowed = true;
    queryCapabilities(*this);
    return true;
}

void DeviceState::shutdown() {
    if (!nativeObjectsBorrowed && queue && fence) {
        std::string ignored;
        (void)synchronize(ignored);
    }
    if (fenceEvent)
        CloseHandle(fenceEvent);
    fenceEvent = nullptr;
    releaseDeferred(true);
    for (CommandFrame &frame : availableFrames)
        destroyCommandFrame(frame);
    for (CommandFrame &frame : inFlightFrames)
        destroyCommandFrame(frame);
    destroyCommandFrame(recordingFrame);
    destroyCommandFrame(borrowedFrame);
    uploadStaging.clear(destroyStagingChunk);
    readbackStaging.clear(destroyStagingChunk);
    availableFrames.clear();
    inFlightFrames.clear();
    recording = false;
    submittedValue = 0;
    completedValue = 0;
    deviceLost = false;
    release(fence);
    borrowedCommands = nullptr;
    if (nativeObjectsBorrowed) {
        queue = nullptr;
        device = nullptr;
    } else {
        release(queue);
        release(device);
    }
    release(adapter);
    release(factory);
    nativeObjectsBorrowed = false;
}

bool DeviceState::synchronize(std::string &error) {
    if (nativeObjectsBorrowed) {
        error = "borrowed D3D12 queues are synchronized by their owner";
        return false;
    }
    if (!queue || !fence || !fenceEvent) {
        error = "D3D12 device synchronization state is incomplete";
        return false;
    }
    return retireCommandFrames(submittedValue, error);
}

bool DeviceState::createCommandFrame(CommandFrame &frame, std::string &error) {
    if (!check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame.allocator)),
               "ID3D12Device::CreateCommandAllocator", error) ||
        !check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frame.allocator, nullptr,
                                         IID_PPV_ARGS(&frame.commands)),
               "ID3D12Device::CreateCommandList", error) ||
        !check(frame.commands->Close(), "ID3D12GraphicsCommandList::Close", error)) {
        destroyCommandFrame(frame);
        return false;
    }
    return true;
}

void DeviceState::destroyCommandFrame(CommandFrame &frame) {
    uploadStaging.retire(frame.upload);
    readbackStaging.retire(frame.readback);
    for (DescriptorRing *ring : descriptorRings(frame))
        releaseDescriptorRing(*ring);
    release(frame.commands);
    release(frame.allocator);
    frame.completionValue = 0;
}

void DeviceState::recycleCommandFrame(CommandFrame frame) {
    resetFrameStorage(frame);
    availableFrames.push_back(std::move(frame));
}

void DeviceState::resetFrameStorage(CommandFrame &frame) {
    frame.completionValue = 0;
    uploadStaging.retire(frame.upload);
    readbackStaging.retire(frame.readback);
    for (DescriptorRing *ring : descriptorRings(frame)) {
        for (ID3D12DescriptorHeap *heap : ring->retiredHeaps)
            release(heap);
        ring->retiredHeaps.clear();
        ring->cursor = 0;
    }
}

DeviceState::CommandFrame &DeviceState::currentFrame() {
    return nativeObjectsBorrowed ? borrowedFrame : recordingFrame;
}

// Separate ExecuteCommandLists calls on one queue run in submission order, and their boundaries flush and invalidate
// GPU caches, so a recording needs no barrier to observe the writes of work submitted before it.
bool DeviceState::beginCommands(std::string &error, ID3D12PipelineState *initialState) {
    if (nativeObjectsBorrowed) {
        if (!borrowedCommands) {
            error = "borrowed D3D12 command list is unavailable";
            return false;
        }
        if (initialState)
            borrowedCommands->SetPipelineState(initialState);
        return true;
    }
    if (recording) {
        error = "D3D12 device is already recording a command list";
        return false;
    }
    if (!retireCommandFrames(0, error))
        return false;
    if (inFlightFrames.size() >= maxFramesInFlight &&
        !retireCommandFrames(inFlightFrames.front().completionValue, error))
        return false;
    CommandFrame frame;
    if (!availableFrames.empty()) {
        frame = std::move(availableFrames.back());
        availableFrames.pop_back();
    } else if (!createCommandFrame(frame, error)) {
        return false;
    }
    if (!check(frame.allocator->Reset(), "ID3D12CommandAllocator::Reset", error) ||
        !check(frame.commands->Reset(frame.allocator, initialState), "ID3D12GraphicsCommandList::Reset", error)) {
        recycleCommandFrame(std::move(frame));
        return false;
    }
    recordingFrame = std::move(frame);
    recording = true;
    return true;
}

bool DeviceState::submitCommands(uint64_t &value, std::string &error) {
    value = 0;
    if (nativeObjectsBorrowed) {
        if (!borrowedCommands) {
            error = "borrowed D3D12 command list is unavailable";
            return false;
        }
        CommandFrame next;
        if (!availableFrames.empty()) {
            next = std::move(availableFrames.back());
            availableFrames.pop_back();
        }
        borrowedFrame.completionValue = ++submittedValue;
        value = borrowedFrame.completionValue;
        inFlightFrames.push_back(std::exchange(borrowedFrame, std::move(next)));
        return true;
    }
    if (!recording) {
        error = "D3D12 device has no command list recording";
        return false;
    }
    CommandFrame frame = std::exchange(recordingFrame, {});
    recording = false;
    if (!check(frame.commands->Close(), "ID3D12GraphicsCommandList::Close", error)) {
        recycleCommandFrame(std::move(frame));
        return false;
    }
    ID3D12CommandList *lists[] = {frame.commands};
    queue->ExecuteCommandLists(1, lists);
    frame.completionValue = ++submittedValue;
    value = frame.completionValue;
    inFlightFrames.push_back(std::move(frame));
    if (!check(queue->Signal(fence, value), "ID3D12CommandQueue::Signal", error)) {
        deviceLost = true;
        return false;
    }
    return true;
}

bool DeviceState::retireCommandFrames(uint64_t through, std::string &error) {
    if (nativeObjectsBorrowed)
        return true;
    if (deviceLost) {
        error = "D3D12 device was removed";
        return false;
    }
    if (through > completedValue && fence->GetCompletedValue() < through) {
        if (!check(fence->SetEventOnCompletion(through, fenceEvent), "ID3D12Fence::SetEventOnCompletion", error))
            return false;
        WaitForSingleObject(fenceEvent, INFINITE);
    }
    // A removed device reports every fence value as reached.
    const uint64_t reached = fence->GetCompletedValue();
    if (reached == (std::numeric_limits<uint64_t>::max)()) {
        deviceLost = true;
        error = "D3D12 device was removed";
        return false;
    }
    completedValue = (std::max)(completedValue, reached);
    while (!inFlightFrames.empty() && inFlightFrames.front().completionValue <= completedValue) {
        recycleCommandFrame(std::move(inFlightFrames.front()));
        inFlightFrames.pop_front();
    }
    releaseDeferred(false);
    return true;
}

bool DeviceState::waitForValue(uint64_t value, std::string &error) {
    if (value > submittedValue) {
        error = "D3D12 fence value was never submitted";
        return false;
    }
    return retireCommandFrames(value, error);
}

bool DeviceState::pollValue(uint64_t value, bool &completed, std::string &error) {
    completed = false;
    if (value > submittedValue) {
        error = "D3D12 fence value was never submitted";
        return false;
    }
    if (!retireCommandFrames(0, error))
        return false;
    completed = value <= completedValue;
    return true;
}

void DeviceState::retireBorrowedCommands(uint64_t value) {
    while (!inFlightFrames.empty() && inFlightFrames.front().completionValue <= value) {
        recycleCommandFrame(std::move(inFlightFrames.front()));
        inFlightFrames.pop_front();
    }
}

void DeviceState::abandonCommands() {
    // The host still executes what the abandoned recording put in its command list, so that storage stays in
    // borrowedFrame and retires with the next borrowed submission.
    if (nativeObjectsBorrowed)
        return;
    if (!recording)
        return;
    CommandFrame frame = std::exchange(recordingFrame, {});
    recording = false;
    (void)frame.commands->Close();
    recycleCommandFrame(std::move(frame));
}

ID3D12GraphicsCommandList *DeviceState::commandList() const {
    return nativeObjectsBorrowed ? borrowedCommands : recording ? recordingFrame.commands : nullptr;
}

void DeviceState::releaseAfterSubmittedWork(ID3D12Resource *resource) {
    if (nativeObjectsBorrowed || submittedValue <= completedValue) {
        resource->Release();
        return;
    }
    deferredReleases.push_back({submittedValue, resource});
}

void DeviceState::releaseDeferred(bool all) {
    const auto retired =
        std::stable_partition(deferredReleases.begin(), deferredReleases.end(),
                              [&](const DeferredRelease &entry) { return !all && entry.value > completedValue; });
    for (auto entry = retired; entry != deferredReleases.end(); ++entry)
        entry->resource->Release();
    deferredReleases.erase(retired, deferredReleases.end());
}

bool DeviceState::createBuffer(Buffer &buffer, size_t size, bool unorderedAccess, D3D12_HEAP_TYPE heapType,
                               D3D12_RESOURCE_STATES initialState, std::string &error) {
    const D3D12_RESOURCE_DESC description = bufferDescription(size, unorderedAccess);
    const D3D12_HEAP_PROPERTIES heap = heapProperties(heapType);
    if (!check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description, initialState, nullptr,
                                               IID_PPV_ARGS(&buffer.resource)),
               "ID3D12Device::CreateCommittedResource", error))
        return false;
    buffer.state = initialState;
    buffer.owned = true;
    return true;
}

void DeviceState::destroyBuffer(Buffer &buffer) {
    if (buffer.resource && buffer.owned)
        releaseAfterSubmittedWork(buffer.resource);
    buffer.resource = nullptr;
    buffer.state = D3D12_RESOURCE_STATE_COMMON;
    buffer.owned = true;
}

bool DeviceState::createImage(Image &image, const D3D12_RESOURCE_DESC &descriptor, DXGI_FORMAT format,
                              D3D12_RESOURCE_STATES initialState, std::string &error) {
    const D3D12_HEAP_PROPERTIES heap = heapProperties(D3D12_HEAP_TYPE_DEFAULT);
    if (!check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &descriptor, initialState, nullptr,
                                               IID_PPV_ARGS(&image.resource)),
               "ID3D12Device::CreateCommittedResource", error))
        return false;
    image.format = format;
    image.state = initialState;
    image.owned = true;
    return true;
}

void DeviceState::destroyImage(Image &image) {
    if (image.resource && image.owned)
        releaseAfterSubmittedWork(image.resource);
    image.resource = nullptr;
    image.format = DXGI_FORMAT_UNKNOWN;
    image.state = D3D12_RESOURCE_STATE_COMMON;
    image.owned = true;
}

bool DeviceState::acquireStaging(bool upload, size_t size, size_t alignment, ID3D12Resource *&resource, size_t &offset,
                                 uint8_t *&mapped, std::string &error) {
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        size > (std::numeric_limits<size_t>::max)() - (alignment - 1)) {
        error = "D3D12 staging allocation request is invalid";
        return false;
    }
    if (!nativeObjectsBorrowed && !recording) {
        error = "D3D12 staging requires an active recording";
        return false;
    }
    StagingChunk *chunk = (upload ? uploadStaging : readbackStaging)
                              .acquire(upload ? currentFrame().upload : currentFrame().readback, size, alignment,
                                       offset, [&](size_t capacity, StagingChunk &created) {
                                           return createStagingChunk(upload, capacity, created, error);
                                       });
    if (!chunk)
        return false;
    resource = chunk->buffer.resource;
    mapped = chunk->mapped + offset;
    return true;
}

bool DeviceState::createStagingChunk(bool upload, size_t capacity, StagingChunk &chunk, std::string &error) {
    chunk.capacity = capacity;
    if (!createBuffer(chunk.buffer, capacity, false, upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK,
                      upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST, error))
        return false;
    const D3D12_RANGE readRange{0, upload ? 0 : capacity};
    if (!check(chunk.buffer.resource->Map(0, &readRange, reinterpret_cast<void **>(&chunk.mapped)),
               "ID3D12Resource::Map", error)) {
        release(chunk.buffer.resource);
        return false;
    }
    return true;
}

bool DeviceState::acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE type, bool shaderVisible, uint32_t count,
                                     ID3D12DescriptorHeap *&heap, D3D12_CPU_DESCRIPTOR_HANDLE &cpu,
                                     D3D12_GPU_DESCRIPTOR_HANDLE &gpu, std::string &error) {
    if (!nativeObjectsBorrowed && !recording) {
        error = "D3D12 descriptors require an active recording";
        return false;
    }
    CommandFrame &frame = currentFrame();
    DescriptorRing *ring = nullptr;
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && shaderVisible)
        ring = &frame.resourceDescriptors;
    else if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        ring = &frame.stagedResourceDescriptors;
    else if (type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && shaderVisible)
        ring = &frame.samplerDescriptors;
    else if (type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV && !shaderVisible)
        ring = &frame.rtvDescriptors;
    else if (type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV && !shaderVisible)
        ring = &frame.dsvDescriptors;
    if (!ring || count == 0) {
        error = "D3D12 descriptor ring request is unsupported";
        return false;
    }
    if (!ring->heap || count > ring->capacity - ring->cursor) {
        if (ring->heap) {
            try {
                ring->retiredHeaps.push_back(ring->heap);
            } catch (const std::bad_alloc &) {
                error = "D3D12 descriptor heap retention allocation failed";
                return false;
            }
            ring->heap = nullptr;
        }
        uint32_t capacity = (std::max)(64u, ring->capacity);
        while (capacity < count) {
            if (capacity > (std::numeric_limits<uint32_t>::max)() / 2) {
                error = "D3D12 descriptor ring capacity overflow";
                return false;
            }
            capacity *= 2;
        }
        if (ring->capacity && capacity <= ring->capacity) {
            if (capacity > (std::numeric_limits<uint32_t>::max)() / 2) {
                error = "D3D12 descriptor ring capacity overflow";
                return false;
            }
            capacity *= 2;
        }
        D3D12_DESCRIPTOR_HEAP_DESC description{};
        description.Type = type;
        description.NumDescriptors = capacity;
        description.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (!check(device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&ring->heap)),
                   "ID3D12Device::CreateDescriptorHeap", error))
            return false;
        ring->capacity = capacity;
        ring->cursor = 0;
    }
    const uint32_t increment = device->GetDescriptorHandleIncrementSize(type);
    cpu = ring->heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += static_cast<SIZE_T>(ring->cursor) * increment;
    gpu = {};
    if (shaderVisible) {
        gpu = ring->heap->GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += static_cast<UINT64>(ring->cursor) * increment;
    }
    heap = ring->heap;
    ring->cursor += count;
    return true;
}

} // namespace vernon::rhi::directx12

#endif
