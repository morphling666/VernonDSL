#include "hal/d3d12/d3d12_state.h"

#if defined(VERNON_DIRECTX12_MIPMAPS)
#include "d3d12_mipmap_2d.h"
#include "d3d12_mipmap_3d.h"
#endif

#include <algorithm>
#include <limits>
#include <utility>

namespace vernon::hal::d3d12 {
namespace {

constexpr size_t kMaxFramesInFlight = 64;

D3D12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CreationNodeMask = 1;
    result.VisibleNodeMask = 1;
    return result;
}

D3D12_RESOURCE_DESC bufferDescription(uint64_t size, bool unorderedAccess) {
    D3D12_RESOURCE_DESC result{};
    result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    // Constant buffer views read whole 256-byte blocks, which must lie inside the resource.
    result.Width = (std::max<uint64_t>(size, 1) + D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1) &
                   ~uint64_t{D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1};
    result.Height = 1;
    result.DepthOrArraySize = 1;
    result.MipLevels = 1;
    result.SampleDesc.Count = 1;
    result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    result.Flags = unorderedAccess ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    return result;
}

void queryCapabilities(DeviceState &state) {
    IDXGIFactory5 *factory = nullptr;
    BOOL tearing = FALSE;
    if (SUCCEEDED(state.factory->QueryInterface(IID_PPV_ARGS(&factory))) &&
        SUCCEEDED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
        state.tearing = tearing != FALSE;
    release(factory);
}

} // namespace

bool check(HRESULT result, const char *operation, std::string &error) {
    if (SUCCEEDED(result))
        return true;
    error = std::string(operation) + " failed with HRESULT 0x";
    constexpr char digits[] = "0123456789abcdef";
    const auto code = static_cast<uint32_t>(result);
    for (int shift = 28; shift >= 0; shift -= 4)
        error.push_back(digits[(code >> shift) & 0xf]);
    return false;
}

DXGI_FORMAT dxgiFormat(VernonTextureFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,       \
                      vulkan, dxgi, ...)                                                                               \
    case VERNON_TEXTURE_##NAME:                                                                                        \
        return dxgi;
#include "VernonFormats.def"
    }
    return DXGI_FORMAT_UNKNOWN;
}

DescriptorAllocator::~DescriptorAllocator() {
    for (ID3D12DescriptorHeap *&heap : heaps_)
        release(heap);
}

void DescriptorAllocator::initialize(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type) {
    device_ = device;
    type_ = type;
    increment_ = device->GetDescriptorHandleIncrementSize(type);
}

D3D12_CPU_DESCRIPTOR_HANDLE DescriptorAllocator::allocate(std::string &error) {
    if (free_.empty()) {
        D3D12_DESCRIPTOR_HEAP_DESC description{};
        description.Type = type_;
        description.NumDescriptors = kPerHeap;
        ID3D12DescriptorHeap *heap = nullptr;
        if (!check(device_->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                   "ID3D12Device::CreateDescriptorHeap", error))
            return {};
        heaps_.push_back(heap);
        const D3D12_CPU_DESCRIPTOR_HANDLE start = heap->GetCPUDescriptorHandleForHeapStart();
        for (uint32_t index = kPerHeap; index-- > 0;)
            free_.push_back({start.ptr + SIZE_T{index} * increment_});
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE handle = free_.back();
    free_.pop_back();
    return handle;
}

void DescriptorAllocator::free(D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    if (handle.ptr)
        free_.push_back(handle);
}

DeviceState::~DeviceState() { shutdown(); }

bool DeviceState::initialize(uint32_t deviceIndex, uint32_t flags, std::string &error) {
    shutdown();
    labels = (flags & (VERNON_HAL_DEVICE_LABELS | VERNON_HAL_DEVICE_VALIDATION)) != 0;
    if (flags & VERNON_HAL_DEVICE_VALIDATION) {
        ID3D12Debug *debug = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->EnableDebugLayer();
        release(debug);
    }
    if (!check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2", error))
        return false;
    if (flags & VERNON_HAL_DEVICE_FORCE_SOFTWARE) {
        if (deviceIndex != 0) {
            error = "a software D3D12 device has index 0";
            shutdown();
            return false;
        }
        if (!check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "IDXGIFactory4::EnumWarpAdapter", error)) {
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
        error = "the requested D3D12 hardware adapter was not found";
        shutdown();
        return false;
    }
    static constexpr D3D_FEATURE_LEVEL kLevels[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1,
                                                    D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_1,
                                                    D3D_FEATURE_LEVEL_11_0};
    for (const D3D_FEATURE_LEVEL level : kLevels)
        if (SUCCEEDED(D3D12CreateDevice(adapter, level, IID_PPV_ARGS(&device))))
            break;
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
    if (FAILED(queue->GetTimestampFrequency(&timestampFrequency)))
        timestampFrequency = 0;
    fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent) {
        error = "CreateEventW failed for the D3D12 fence";
        shutdown();
        return false;
    }
    resourceDescriptors.initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    samplerDescriptors.initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    renderTargets.initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    depthStencils.initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    if (!createMipmapPipelines(error)) {
        shutdown();
        return false;
    }
    return true;
}

void DeviceState::shutdown() {
    if (queue && fence && fenceEvent) {
        std::string ignored;
        (void)flushUploads(ignored);
        (void)synchronize(ignored);
    }
    releaseDeferred(true);
    for (std::unique_ptr<CommandFrame> &frame : recording_)
        destroyFrame(*frame);
    for (std::unique_ptr<CommandFrame> &frame : inFlight_)
        destroyFrame(*frame);
    for (std::unique_ptr<CommandFrame> &frame : available_)
        destroyFrame(*frame);
    recording_.clear();
    inFlight_.clear();
    available_.clear();
    uploads_ = nullptr;
    const auto destroyChunk = [](StagingChunk &chunk) {
        if (chunk.resource && chunk.mapped)
            chunk.resource->Unmap(0, nullptr);
        release(chunk.resource);
    };
    uploadStaging_.clear(destroyChunk);
    readbackStaging_.clear(destroyChunk);
    release(mipmap2d);
    release(mipmap3d);
    release(mipmapRoot);
    if (fenceEvent)
        CloseHandle(fenceEvent);
    fenceEvent = nullptr;
    release(fence);
    release(queue);
    release(device);
    release(adapter);
    release(factory);
    submittedValue = 0;
    completedValue = 0;
    lost = false;
}

bool DeviceState::createMipmapPipelines(std::string &error) {
#if defined(VERNON_DIRECTX12_MIPMAPS)
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {2, ranges};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[1].Constants = {0, 0, 4};
    const D3D12_ROOT_SIGNATURE_DESC root{2, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ID3DBlob *serialized = nullptr;
    ID3DBlob *diagnostics = nullptr;
    const HRESULT made = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &diagnostics);
    release(diagnostics);
    if (!check(made, "D3D12SerializeRootSignature", error))
        return false;
    const bool created = check(device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                                           serialized->GetBufferSize(), IID_PPV_ARGS(&mipmapRoot)),
                               "ID3D12Device::CreateRootSignature", error);
    release(serialized);
    if (!created)
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = mipmapRoot;
    pipeline.CS = {vernon_d3d12_mipmap_2d, vernon_d3d12_mipmap_2d_size};
    if (!check(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&mipmap2d)),
               "ID3D12Device::CreateComputePipelineState", error))
        return false;
    pipeline.CS = {vernon_d3d12_mipmap_3d, vernon_d3d12_mipmap_3d_size};
    return check(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&mipmap3d)),
                 "ID3D12Device::CreateComputePipelineState", error);
#else
    (void)error;
    return true;
#endif
}

bool DeviceState::createFrame(std::unique_ptr<CommandFrame> &frame, std::string &error) {
    frame = std::make_unique<CommandFrame>();
    if (!check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame->allocator)),
               "ID3D12Device::CreateCommandAllocator", error) ||
        !check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frame->allocator, nullptr,
                                         IID_PPV_ARGS(&frame->commands)),
               "ID3D12Device::CreateCommandList", error) ||
        !check(frame->commands->Close(), "ID3D12GraphicsCommandList::Close", error)) {
        destroyFrame(*frame);
        frame.reset();
        return false;
    }
    return true;
}

void DeviceState::destroyFrame(CommandFrame &frame) {
    uploadStaging_.retire(frame.upload);
    readbackStaging_.retire(frame.readback);
    for (ID3D12DescriptorHeap *&heap : frame.heaps)
        release(heap);
    frame.heaps.clear();
    release(frame.commands);
    release(frame.allocator);
}

void DeviceState::recycle(std::unique_ptr<CommandFrame> frame) {
    uploadStaging_.retire(frame->upload);
    readbackStaging_.retire(frame->readback);
    for (ID3D12DescriptorHeap *&heap : frame->heaps)
        release(heap);
    frame->heaps.clear();
    frame->value = 0;
    available_.push_back(std::move(frame));
}

bool DeviceState::beginCommands(CommandFrame *&frame, std::string &error) {
    frame = nullptr;
    if (!retire(0, error))
        return false;
    if (inFlight_.size() >= kMaxFramesInFlight && !retire(inFlight_.front()->value, error))
        return false;
    std::unique_ptr<CommandFrame> made;
    if (!available_.empty()) {
        made = std::move(available_.back());
        available_.pop_back();
    } else if (!createFrame(made, error)) {
        return false;
    }
    if (!check(made->allocator->Reset(), "ID3D12CommandAllocator::Reset", error) ||
        !check(made->commands->Reset(made->allocator, nullptr), "ID3D12GraphicsCommandList::Reset", error)) {
        recycle(std::move(made));
        return false;
    }
    frame = made.get();
    recording_.push_back(std::move(made));
    return true;
}

void DeviceState::abandonCommands(CommandFrame *frame) {
    const auto found = std::find_if(recording_.begin(), recording_.end(),
                                    [&](const std::unique_ptr<CommandFrame> &owned) { return owned.get() == frame; });
    if (found == recording_.end())
        return;
    std::unique_ptr<CommandFrame> owned = std::move(*found);
    recording_.erase(found);
    (void)owned->commands->Close();
    if (uploads_ == frame)
        uploads_ = nullptr;
    recycle(std::move(owned));
}

bool DeviceState::submitCommands(CommandFrame *frame, uint64_t &value, std::string &error) {
    value = 0;
    if (!check(frame->commands->Close(), "ID3D12GraphicsCommandList::Close", error)) {
        abandonCommands(frame);
        return false;
    }
    ID3D12CommandList *lists[] = {frame->commands};
    // Taken out of the recording frames first, so that the uploads it follows do not close it again.
    const auto found = std::find_if(recording_.begin(), recording_.end(),
                                    [&](const std::unique_ptr<CommandFrame> &owned) { return owned.get() == frame; });
    std::unique_ptr<CommandFrame> owned = std::move(*found);
    recording_.erase(found);
    if (uploads_ == frame)
        uploads_ = nullptr;
    const bool executed = execute(lists, 1, value, error);
    owned->value = value ? value : submittedValue;
    inFlight_.push_back(std::move(owned));
    return executed;
}

bool DeviceState::uploadCommands(CommandFrame *&frame, std::string &error) {
    if (!uploads_ && !beginCommands(uploads_, error))
        return false;
    frame = uploads_;
    return true;
}

bool DeviceState::flushUploads(std::string &error) {
    if (!uploads_)
        return true;
    uint64_t value = 0;
    return submitCommands(uploads_, value, error);
}

bool DeviceState::execute(ID3D12CommandList *const *lists, uint32_t count, uint64_t &value, std::string &error) {
    value = 0;
    if (lost) {
        error = "the D3D12 device was removed";
        return false;
    }
    if (!flushUploads(error))
        return false;
    queue->ExecuteCommandLists(count, lists);
    value = ++submittedValue;
    if (!check(queue->Signal(fence, value), "ID3D12CommandQueue::Signal", error)) {
        lost = true;
        return false;
    }
    return true;
}

bool DeviceState::retire(uint64_t through, std::string &error) {
    if (lost) {
        error = "the D3D12 device was removed";
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
        lost = true;
        error = "the D3D12 device was removed";
        return false;
    }
    completedValue = (std::max)(completedValue, reached);
    while (!inFlight_.empty() && inFlight_.front()->value <= completedValue) {
        recycle(std::move(inFlight_.front()));
        inFlight_.pop_front();
    }
    releaseDeferred(false);
    return true;
}

bool DeviceState::waitForValue(uint64_t value, std::string &error) {
    if (value > submittedValue) {
        error = "the D3D12 fence value was never submitted";
        return false;
    }
    return retire(value, error);
}

bool DeviceState::completed(uint64_t value, bool &done, std::string &error) {
    done = false;
    if (value > submittedValue) {
        error = "the D3D12 fence value was never submitted";
        return false;
    }
    if (!retire(0, error))
        return false;
    done = value <= completedValue;
    return true;
}

bool DeviceState::synchronize(std::string &error) {
    if (!flushUploads(error))
        return false;
    return retire(submittedValue, error);
}

bool DeviceState::acquireStaging(CommandFrame &frame, bool upload, size_t size, size_t alignment,
                                 ID3D12Resource *&resource, size_t &offset, uint8_t *&mapped, std::string &error) {
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        error = "the D3D12 staging request is invalid";
        return false;
    }
    StagingChunk *chunk =
        (upload ? uploadStaging_ : readbackStaging_)
            .acquire(upload ? frame.upload : frame.readback, size, alignment, offset,
                     [&](size_t capacity, StagingChunk &created) {
                         const D3D12_RESOURCE_DESC description = bufferDescription(capacity, false);
                         const D3D12_HEAP_PROPERTIES heap =
                             heapProperties(upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK);
                         if (!check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
                                                                    upload ? D3D12_RESOURCE_STATE_GENERIC_READ
                                                                           : D3D12_RESOURCE_STATE_COPY_DEST,
                                                                    nullptr, IID_PPV_ARGS(&created.resource)),
                                    "ID3D12Device::CreateCommittedResource", error))
                             return false;
                         const D3D12_RANGE read{0, upload ? 0 : capacity};
                         if (!check(created.resource->Map(0, &read, reinterpret_cast<void **>(&created.mapped)),
                                    "ID3D12Resource::Map", error)) {
                             release(created.resource);
                             return false;
                         }
                         created.capacity = capacity;
                         return true;
                     });
    if (!chunk)
        return false;
    resource = chunk->resource;
    mapped = chunk->mapped + offset;
    return true;
}

bool DeviceState::frameHeap(CommandFrame &frame, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t count,
                            ID3D12DescriptorHeap *&heap, std::string &error) {
    D3D12_DESCRIPTOR_HEAP_DESC description{};
    description.Type = type;
    description.NumDescriptors = count;
    description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heap = nullptr;
    if (!check(device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)), "ID3D12Device::CreateDescriptorHeap",
               error))
        return false;
    frame.heaps.push_back(heap);
    return true;
}

bool DeviceState::createBuffer(Buffer &buffer, uint64_t size, uint32_t, VernonHalMemory memory, std::string &error) {
    buffer = {};
    const bool onDevice = memory == VERNON_HAL_MEMORY_DEVICE;
    const D3D12_RESOURCE_DESC description = bufferDescription(size, true);
    // Host-visible buffers are in custom heaps rather than the upload and readback ones, whose buffers can never be
    // written by shaders nor leave one state; so every buffer, as Vulkan's, begins each submission in COMMON, to
    // which buffers decay.
    D3D12_HEAP_PROPERTIES heap = heapProperties(onDevice ? D3D12_HEAP_TYPE_DEFAULT : D3D12_HEAP_TYPE_CUSTOM);
    if (!onDevice) {
        heap.CPUPageProperty = memory == VERNON_HAL_MEMORY_UPLOAD ? D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
                                                                  : D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        heap.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    }
    if (!check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&buffer.resource)),
               "ID3D12Device::CreateCommittedResource", error))
        return false;
    buffer.size = size;
    buffer.memory = memory;
    if (!onDevice) {
        const D3D12_RANGE read{0, memory == VERNON_HAL_MEMORY_READBACK ? static_cast<SIZE_T>(description.Width) : 0};
        if (!check(buffer.resource->Map(0, &read, reinterpret_cast<void **>(&buffer.mapped)), "ID3D12Resource::Map",
                   error)) {
            release(buffer.resource);
            return false;
        }
    }
    return true;
}

void DeviceState::releaseAfterSubmittedWork(IUnknown *object) {
    if (!object)
        return;
    // The pending uploads are the next submission.
    const uint64_t value = submittedValue + (uploads_ ? 1 : 0);
    if (value <= completedValue) {
        object->Release();
        return;
    }
    deferred_.push_back({value, object});
}

void DeviceState::destroyBuffer(Buffer &buffer) {
    if (buffer.resource && buffer.mapped)
        buffer.resource->Unmap(0, nullptr);
    if (buffer.resource)
        releaseAfterSubmittedWork(buffer.resource);
    buffer = {};
}

void DeviceState::releaseDeferred(bool all) {
    const auto retired = std::stable_partition(deferred_.begin(), deferred_.end(), [&](const DeferredRelease &entry) {
        return !all && entry.value > completedValue;
    });
    for (auto entry = retired; entry != deferred_.end(); ++entry)
        entry->object->Release();
    deferred_.erase(retired, deferred_.end());
}

} // namespace vernon::hal::d3d12
