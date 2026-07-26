#include "directx12_backend.h"

#if defined(_WIN32)

#include <algorithm>
#include <limits>

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

bool waitForFence(DeviceState &device, uint64_t value, std::string &error) {
    if (value == 0 || device.fence->GetCompletedValue() >= value)
        return true;
    if (!check(device.fence->SetEventOnCompletion(value, device.fenceEvent), "ID3D12Fence::SetEventOnCompletion",
               error))
        return false;
    WaitForSingleObject(device.fenceEvent, INFINITE);
    return true;
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
    for (CommandFrame &frame : frames) {
        if (!check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame.allocator)),
                   "ID3D12Device::CreateCommandAllocator", error) ||
            !check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frame.allocator, nullptr,
                                             IID_PPV_ARGS(&frame.commands)),
                   "ID3D12Device::CreateCommandList", error) ||
            !check(frame.commands->Close(), "ID3D12GraphicsCommandList::Close", error)) {
            shutdown();
            return false;
        }
    }
    allocator = frames[0].allocator;
    commands = frames[0].commands;
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
    commands = borrowedCommands;
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
    for (StagingRing *ring : {&uploadRing, &readbackRing}) {
        if (ring->buffer.resource && ring->mapped)
            ring->buffer.resource->Unmap(0, nullptr);
        ring->mapped = nullptr;
        destroyBuffer(ring->buffer);
        ring->capacity = 0;
        ring->cursor = 0;
    }
    for (DescriptorRing *ring :
         {&resourceDescriptorRing, &samplerDescriptorRing, &rtvDescriptorRing, &dsvDescriptorRing}) {
        release(ring->heap);
        ring->capacity = 0;
        ring->cursor = 0;
    }
    release(fence);
    commands = nullptr;
    allocator = nullptr;
    for (CommandFrame &frame : frames) {
        release(frame.commands);
        release(frame.allocator);
        frame.completionValue = 0;
    }
    currentFrame = frameCount - 1;
    if (nativeObjectsBorrowed) {
        queue = nullptr;
        device = nullptr;
    } else {
        release(queue);
        release(device);
    }
    release(adapter);
    release(factory);
    fenceValue = 0;
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
    const uint64_t value = ++fenceValue;
    if (!check(queue->Signal(fence, value), "ID3D12CommandQueue::Signal", error))
        return false;
    return waitForFence(*this, value, error);
}

bool DeviceState::beginCommands(std::string &error, ID3D12PipelineState *initialState) {
    if (nativeObjectsBorrowed) {
        if (!commands) {
            error = "borrowed D3D12 command list is unavailable";
            return false;
        }
        if (initialState)
            commands->SetPipelineState(initialState);
        return true;
    }
    currentFrame = (currentFrame + 1) % frameCount;
    CommandFrame &frame = frames[currentFrame];
    if (!waitForFence(*this, frame.completionValue, error))
        return false;
    allocator = frame.allocator;
    commands = frame.commands;
    return check(allocator->Reset(), "ID3D12CommandAllocator::Reset", error) &&
           check(commands->Reset(allocator, initialState), "ID3D12GraphicsCommandList::Reset", error);
}

bool DeviceState::submitCommands(std::string &error) {
    if (nativeObjectsBorrowed) {
        if (!commands) {
            error = "borrowed D3D12 command list is unavailable";
            return false;
        }
        return true;
    }
    if (!check(commands->Close(), "ID3D12GraphicsCommandList::Close", error))
        return false;
    ID3D12CommandList *lists[] = {commands};
    queue->ExecuteCommandLists(1, lists);
    const uint64_t value = ++fenceValue;
    if (!check(queue->Signal(fence, value), "ID3D12CommandQueue::Signal", error))
        return false;
    frames[currentFrame].completionValue = value;
    if (!waitForFence(*this, value, error))
        return false;
    uploadRing.cursor = 0;
    readbackRing.cursor = 0;
    return true;
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
        buffer.resource->Release();
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
        image.resource->Release();
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
    StagingRing &ring = upload ? uploadRing : readbackRing;
    const size_t alignedCursor = (ring.cursor + alignment - 1) & ~(alignment - 1);
    if (!ring.buffer.resource || alignedCursor > ring.capacity || size > ring.capacity - alignedCursor) {
        if (ring.buffer.resource && size <= ring.capacity) {
            ring.cursor = 0;
        } else {
            if (ring.buffer.resource) {
                if (!synchronize(error))
                    return false;
                if (ring.mapped)
                    ring.buffer.resource->Unmap(0, nullptr);
                ring.mapped = nullptr;
                destroyBuffer(ring.buffer);
            }
            size_t capacity = 64 * 1024;
            while (capacity < size) {
                if (capacity > (std::numeric_limits<size_t>::max)() / 2) {
                    error = "D3D12 staging ring capacity overflow";
                    return false;
                }
                capacity *= 2;
            }
            if (!createBuffer(ring.buffer, capacity, false, upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK,
                              upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST, error))
                return false;
            const D3D12_RANGE readRange{0, upload ? 0 : capacity};
            if (!check(ring.buffer.resource->Map(0, &readRange, reinterpret_cast<void **>(&ring.mapped)),
                       "ID3D12Resource::Map", error)) {
                destroyBuffer(ring.buffer);
                return false;
            }
            ring.capacity = capacity;
            ring.cursor = 0;
        }
    }
    offset = (ring.cursor + alignment - 1) & ~(alignment - 1);
    resource = ring.buffer.resource;
    mapped = ring.mapped + offset;
    ring.cursor = offset + size;
    return true;
}

bool DeviceState::acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE type, bool shaderVisible, uint32_t count,
                                     ID3D12DescriptorHeap *&heap, D3D12_CPU_DESCRIPTOR_HANDLE &cpu,
                                     D3D12_GPU_DESCRIPTOR_HANDLE &gpu, std::string &error) {
    DescriptorRing *ring = nullptr;
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && shaderVisible)
        ring = &resourceDescriptorRing;
    else if (type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && shaderVisible)
        ring = &samplerDescriptorRing;
    else if (type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV && !shaderVisible)
        ring = &rtvDescriptorRing;
    else if (type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV && !shaderVisible)
        ring = &dsvDescriptorRing;
    if (!ring || count == 0) {
        error = "D3D12 descriptor ring request is unsupported";
        return false;
    }
    if (ring->heap && count <= ring->capacity && count > ring->capacity - ring->cursor)
        ring->cursor = 0;
    if (!ring->heap || count > ring->capacity) {
        if (ring->heap) {
            if (!synchronize(error))
                return false;
            release(ring->heap);
        }
        uint32_t capacity = 64;
        while (capacity < count) {
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
