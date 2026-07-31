#include "backend_dispatch.h"
#include "directx12_backend.h"
#include "logical_resource_record.h"
#include "rhi_test_hooks.h"
#include "sampler_filter.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
struct DirectX12BufferSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::directx12::Buffer buffer;
    VernonRhiDirectX12BorrowedBufferDescriptor descriptor{};
    VernonRhiBufferDescriptor ownedDescriptor{};
};

struct DirectX12ImageSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::directx12::Image image;
    VernonRhiDirectX12BorrowedImageDescriptor descriptor{};
    VernonRhiImageDescriptor ownedDescriptor{};
};

struct DirectX12DescriptorRangeSlot {
    VernonRhiDirectX12BorrowedDescriptorRangeDescriptor descriptor{};
    uint32_t generation{1};
    bool occupied{};
};

struct DirectX12SamplerSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::directx12::Sampler sampler;
};

struct DirectX12InteropDevice {
    vernon::rhi::directx12::DeviceState state;
    std::deque<DirectX12BufferSlot> buffers;
    std::deque<DirectX12ImageSlot> images;
    std::deque<DirectX12SamplerSlot> samplers;
    std::deque<DirectX12DescriptorRangeSlot> descriptorRanges;
    uint32_t queueCapabilities{};
    bool owned{};
    std::string error;
    std::mutex mutex;
};

struct DirectX12DeviceSlot {
    std::shared_ptr<DirectX12InteropDevice> device;
    uint32_t generation{1};
};

constexpr uint32_t directX12DeviceBit = uint32_t{1} << 31;
std::vector<DirectX12DeviceSlot> directX12Devices;
std::mutex deviceMutex;

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

bool decodeResourceKey(uint64_t key, uint32_t &index, uint32_t &generation) {
    const uint64_t encodedIndex = key & UINT32_MAX;
    generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return false;
    index = static_cast<uint32_t>(encodedIndex - 1);
    return true;
}

template <typename Slots> auto *lookupResourceRecord(Slots &slots, uint64_t key) {
    uint32_t index = 0;
    uint32_t generation = 0;
    if (!decodeResourceKey(key, index, generation) || index >= slots.size())
        return static_cast<typename Slots::value_type *>(nullptr);
    auto &slot = slots[index];
    return slot.isRetained(generation) ? &slot : nullptr;
}

template <typename Slot, typename = void> struct HasPublicAlive : std::false_type {};
template <typename Slot>
struct HasPublicAlive<Slot, std::void_t<decltype(std::declval<Slot &>().publicAlive)>> : std::true_type {};

template <typename Slot> bool publicAlive(const Slot &slot) {
    if constexpr (HasPublicAlive<Slot>::value)
        return slot.publicAlive;
    return true;
}

std::optional<size_t> rgba8Size(const VernonRhiImageDescriptor &descriptor) {
    if (descriptor.dimension != VERNON_RHI_IMAGE_2D || descriptor.format != VERNON_RHI_FORMAT_RGBA8_UNORM ||
        descriptor.depth != 1 || descriptor.mip_levels != 1 || descriptor.array_layers != 1 ||
        descriptor.width > (std::numeric_limits<size_t>::max)() / descriptor.height)
        return std::nullopt;
    const size_t pixels = static_cast<size_t>(descriptor.width) * descriptor.height;
    return pixels <= (std::numeric_limits<size_t>::max)() / 4 ? std::optional<size_t>{pixels * 4} : std::nullopt;
}

std::shared_ptr<DirectX12InteropDevice> lookupDirectX12Device(VernonRhiDevice handle) {
    if ((handle.index & directX12DeviceBit) == 0)
        return {};
    const uint32_t index = handle.index & ~directX12DeviceBit;
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (index >= directX12Devices.size())
        return {};
    DirectX12DeviceSlot &slot = directX12Devices[index];
    return slot.device && slot.generation == handle.generation ? slot.device
                                                               : std::shared_ptr<DirectX12InteropDevice>{};
}

template <typename Slots, typename Handle>
typename Slots::value_type *lookupDirectX12Slot(Slots &slots, Handle handle) {
    if (handle.index >= slots.size())
        return nullptr;
    auto &slot = slots[handle.index];
    return slot.occupied && publicAlive(slot) && slot.generation == handle.generation ? &slot : nullptr;
}

template <typename Slots, typename Handle>
typename Slots::value_type &allocateDirectX12Slot(Slots &slots, Handle &output) {
    uint32_t index = 0;
    while (index < slots.size() && slots[index].occupied)
        ++index;
    if (index == slots.size())
        slots.emplace_back();
    auto &slot = slots[index];
    if constexpr (std::is_base_of_v<vernon::rhi::LogicalResourceRecord, typename Slots::value_type>)
        slot.publish();
    else
        slot.occupied = true;
    output = {index, slot.generation};
    return slot;
}

template <typename Slot> void releaseDirectX12Slot(Slot &slot) {
    if constexpr (std::is_base_of_v<vernon::rhi::LogicalResourceRecord, Slot>) {
        if (slot.publicAlive)
            slot.destroyPublicOwner();
        slot.recycle();
    } else {
        slot.occupied = false;
        if (++slot.generation == 0)
            slot.generation = 1;
    }
}

D3D12_RESOURCE_STATES directX12ResourceState(VernonRhiResourceState state) {
    switch (state) {
    case VERNON_RHI_STATE_COMMON:
        return D3D12_RESOURCE_STATE_COMMON;
    case VERNON_RHI_STATE_TRANSFER_SOURCE:
        return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case VERNON_RHI_STATE_TRANSFER_DESTINATION:
        return D3D12_RESOURCE_STATE_COPY_DEST;
    case VERNON_RHI_STATE_SHADER_READ:
        return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    case VERNON_RHI_STATE_SHADER_WRITE:
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case VERNON_RHI_STATE_COLOR_ATTACHMENT:
        return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT:
        return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case VERNON_RHI_STATE_PRESENT:
        return D3D12_RESOURCE_STATE_PRESENT;
    case VERNON_RHI_STATE_UNDEFINED:
        return D3D12_RESOURCE_STATE_COMMON;
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

void restoreDirectX12BufferState(void *context, uint64_t state) {
    static_cast<vernon::rhi::directx12::Buffer *>(context)->state = static_cast<D3D12_RESOURCE_STATES>(state);
}

void restoreDirectX12ImageState(void *context, uint64_t state) {
    static_cast<vernon::rhi::directx12::Image *>(context)->state = static_cast<D3D12_RESOURCE_STATES>(state);
}

DXGI_FORMAT directX12Format(VernonRhiFormat format) {
    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_UNKNOWN,
        DXGI_FORMAT_R8_UNORM,
        DXGI_FORMAT_R8G8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        DXGI_FORMAT_R16_FLOAT,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_UNKNOWN,
        DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R32G32B32_FLOAT,
        DXGI_FORMAT_R11G11B10_FLOAT,
    };
    const uint32_t index = static_cast<uint32_t>(format);
    return index < sizeof(formats) / sizeof(formats[0]) ? formats[index] : DXGI_FORMAT_UNKNOWN;
}

template <typename Object> bool belongsToDirectX12Device(Object *object, ID3D12Device *expected) {
    ID3D12Device *actual = nullptr;
    if (FAILED(object->GetDevice(IID_PPV_ARGS(&actual))))
        return false;
    const bool matches = actual == expected;
    actual->Release();
    return matches;
}
} // namespace

namespace vernon::rhi::directx12_api {

extern "C" VERNON_RHI_CAPI VernonRhiDevice
vernonRhiCreateBorrowedDirectX12Device(const VernonRhiDirectX12BorrowedDeviceDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->device || !descriptor->queue ||
        !descriptor->command_list ||
        (descriptor->queue_capabilities &
         ~(VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS)) != 0)
        return invalidDevice();
    auto *queue = static_cast<ID3D12CommandQueue *>(descriptor->queue);
    auto *commands = static_cast<ID3D12GraphicsCommandList *>(descriptor->command_list);
    auto *nativeDevice = static_cast<ID3D12Device *>(descriptor->device);
    if (queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT ||
        commands->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT || !belongsToDirectX12Device(queue, nativeDevice) ||
        !belongsToDirectX12Device(commands, nativeDevice))
        return invalidDevice();
    auto device = std::shared_ptr<DirectX12InteropDevice>(new (std::nothrow) DirectX12InteropDevice());
    if (!device || !device->state.initializeBorrowed(nativeDevice, queue, commands, device->error))
        return invalidDevice();
    device->queueCapabilities = descriptor->queue_capabilities;
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < directX12Devices.size() && directX12Devices[index].device)
        ++index;
    if (index == directX12Devices.size()) {
        if (index >= directX12DeviceBit)
            return invalidDevice();
        directX12Devices.emplace_back();
    }
    directX12Devices[index].device = std::move(device);
    return {index | directX12DeviceBit, directX12Devices[index].generation};
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceGetBorrowedQueue(VernonRhiDevice handle,
                                                                                    void **output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.queue;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceGetBorrowedCommandList(VernonRhiDevice handle,
                                                                                          void **output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = device->state.commandList();
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedBuffer(
    VernonRhiDevice handle, const VernonRhiDirectX12BorrowedBufferDescriptor *descriptor, VernonRhiBuffer *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->resource ||
        !descriptor->size || descriptor->state == VERNON_RHI_STATE_UNDEFINED || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto *resource = static_cast<ID3D12Resource *>(descriptor->resource);
    if (!belongsToDirectX12Device(resource, device->state.device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const D3D12_RESOURCE_DESC native = resource->GetDesc();
    if (native.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || native.Width < descriptor->size)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot &slot = allocateDirectX12Slot(device->buffers, *output);
    slot.buffer.resource = resource;
    slot.buffer.state = directX12ResourceState(descriptor->state);
    slot.buffer.owned = false;
    slot.descriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedImage(
    VernonRhiDevice handle, const VernonRhiDirectX12BorrowedImageDescriptor *descriptor, VernonRhiImage *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->resource ||
        descriptor->image.struct_size < sizeof(descriptor->image) || descriptor->state == VERNON_RHI_STATE_UNDEFINED ||
        !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto *resource = static_cast<ID3D12Resource *>(descriptor->resource);
    if (!belongsToDirectX12Device(resource, device->state.device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const D3D12_RESOURCE_DESC native = resource->GetDesc();
    const DXGI_FORMAT format = directX12Format(descriptor->image.format);
    const D3D12_RESOURCE_DIMENSION dimension = descriptor->image.dimension == VERNON_RHI_IMAGE_3D
                                                   ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                                   : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    const uint32_t expectedDepthOrLayers =
        descriptor->image.dimension == VERNON_RHI_IMAGE_3D ? descriptor->image.depth : descriptor->image.array_layers;
    const bool validCube = descriptor->image.dimension != VERNON_RHI_IMAGE_CUBE ||
                           (descriptor->image.width == descriptor->image.height &&
                            descriptor->image.array_layers == 6 && descriptor->image.depth == 1);
    const bool validFormat =
        native.Format == format || (format == DXGI_FORMAT_D32_FLOAT && native.Format == DXGI_FORMAT_R32_TYPELESS &&
                                    (descriptor->image.usage & VERNON_RHI_IMAGE_SAMPLED) != 0 &&
                                    (descriptor->image.usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0);
    if (format == DXGI_FORMAT_UNKNOWN || native.Dimension != dimension || !validFormat ||
        native.Width != descriptor->image.width || native.Height != descriptor->image.height ||
        native.DepthOrArraySize != expectedDepthOrLayers || native.MipLevels != descriptor->image.mip_levels ||
        native.SampleDesc.Count != descriptor->image.sample_count || !validCube)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot &slot = allocateDirectX12Slot(device->images, *output);
    slot.image.resource = resource;
    slot.image.format = format;
    slot.image.dimension = descriptor->image.dimension;
    slot.image.state = directX12ResourceState(descriptor->state);
    slot.image.owned = false;
    slot.descriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus vernonRhiDirectX12DeviceImportBorrowedDescriptorRange(
    VernonRhiDevice handle, const VernonRhiDirectX12BorrowedDescriptorRangeDescriptor *descriptor,
    VernonRhiNativeDescriptorRange *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->heap ||
        !descriptor->cpu_handle || !descriptor->descriptor_count ||
        descriptor->heap_type > VERNON_RHI_NATIVE_DESCRIPTOR_DEPTH_STENCIL || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto *heap = static_cast<ID3D12DescriptorHeap *>(descriptor->heap);
    if (!belongsToDirectX12Device(heap, device->state.device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const D3D12_DESCRIPTOR_HEAP_DESC native = heap->GetDesc();
    const D3D12_DESCRIPTOR_HEAP_TYPE expected[] = {
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    };
    if (native.Type != expected[descriptor->heap_type] || descriptor->descriptor_count > native.NumDescriptors)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const bool shaderVisible = (native.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) != 0;
    if (shaderVisible != (descriptor->gpu_handle != 0))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t increment = device->state.device->GetDescriptorHandleIncrementSize(native.Type);
    const uint64_t cpuStart = heap->GetCPUDescriptorHandleForHeapStart().ptr;
    if (descriptor->cpu_handle < cpuStart || (descriptor->cpu_handle - cpuStart) % increment != 0 ||
        (descriptor->cpu_handle - cpuStart) / increment + descriptor->descriptor_count > native.NumDescriptors)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (shaderVisible) {
        const uint64_t gpuStart = heap->GetGPUDescriptorHandleForHeapStart().ptr;
        if (descriptor->gpu_handle < gpuStart || (descriptor->gpu_handle - gpuStart) % increment != 0 ||
            (descriptor->gpu_handle - gpuStart) / increment + descriptor->descriptor_count > native.NumDescriptors)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12DescriptorRangeSlot &slot = allocateDirectX12Slot(device->descriptorRanges, *output);
    slot.descriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus
vernonRhiDeviceDestroyNativeDescriptorRange(VernonRhiDevice handle, VernonRhiNativeDescriptorRange range) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12DescriptorRangeSlot *slot = lookupDirectX12Slot(device->descriptorRanges, range);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    releaseDirectX12Slot(*slot);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VERNON_RHI_CAPI VernonRhiStatus
vernonRhiDeviceGetNativeDescriptorRange(VernonRhiDevice handle, VernonRhiNativeDescriptorRange range,
                                        VernonRhiDirectX12BorrowedDescriptorRangeDescriptor *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !output || output->struct_size < sizeof(*output))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12DescriptorRangeSlot *slot = lookupDirectX12Slot(device->descriptorRanges, range);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = slot->descriptor;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiDevice createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {

    auto device = std::shared_ptr<DirectX12InteropDevice>(new (std::nothrow) DirectX12InteropDevice());
    const bool forceSoftware = (descriptor->flags & VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE) != 0;
    if (!device || !device->state.initialize(descriptor->device_index, forceSoftware, device->error))
        return invalidDevice();
    device->owned = true;
    device->queueCapabilities = VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < directX12Devices.size() && directX12Devices[index].device)
        ++index;
    if (index == directX12Devices.size())
        directX12Devices.emplace_back();
    directX12Devices[index].device = std::move(device);
    return {index | directX12DeviceBit, directX12Devices[index].generation};
}

void destroyDevice(VernonRhiDevice handle) {

    std::shared_ptr<DirectX12InteropDevice> device;
    {
        const uint32_t index = handle.index & ~directX12DeviceBit;
        std::lock_guard<std::mutex> guard(deviceMutex);
        if (index >= directX12Devices.size() || directX12Devices[index].generation != handle.generation)
            return;
        DirectX12DeviceSlot &slot = directX12Devices[index];
        device = std::move(slot.device);
        ++slot.generation;
        if (slot.generation == 0)
            slot.generation = 1;
    }
    if (device) {
        std::lock_guard<std::mutex> guard(device->mutex);
        for (DirectX12BufferSlot &buffer : device->buffers)
            if (buffer.occupied)
                device->state.destroyBuffer(buffer.buffer);
        for (DirectX12ImageSlot &image : device->images)
            if (image.occupied)
                device->state.destroyImage(image.image);
        device->state.shutdown();
    }
    return;
}

VernonStringView lastError(VernonRhiDevice handle) {
    auto device = lookupDirectX12Device(handle);
    return device ? VernonStringView{device->error.data(), device->error.size()} : VernonStringView{};
}

VernonRhiStatus synchronize(VernonRhiDevice handle) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return device->state.synchronize(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor,
                             VernonRhiBuffer *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot &slot = allocateDirectX12Slot(device->buffers, *output);
    const bool unorderedAccess = (descriptor->usage & VERNON_RHI_BUFFER_STORAGE) != 0;
    if (!device->state.createBuffer(slot.buffer, static_cast<size_t>(descriptor->size), unorderedAccess,
                                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, device->error)) {
        releaseDirectX12Slot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot.ownedDescriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!source)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
    if (!slot || offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    ID3D12Resource *upload = nullptr;
    size_t uploadOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(true, static_cast<size_t>(size), 256, upload, uploadOffset, mapped,
                                      device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    std::memcpy(mapped, source, static_cast<size_t>(size));
    if (!device->state.beginCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    vernon::rhi::directx12::transition(device->state.commandList(), slot->buffer.resource, slot->buffer.state,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
    device->state.commandList()->CopyBufferRegion(slot->buffer.resource, offset, upload, uploadOffset, size);
    vernon::rhi::directx12::transition(device->state.commandList(), slot->buffer.resource, slot->buffer.state,
                                       D3D12_RESOURCE_STATE_COMMON);
    return device->state.submitCommands(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                               uint64_t size) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
    if (!slot || offset > slot->ownedDescriptor.size || size > slot->ownedDescriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    ID3D12Resource *readback = nullptr;
    size_t readbackOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(false, static_cast<size_t>(size), 256, readback, readbackOffset, mapped,
                                      device->error) ||
        !device->state.beginCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    vernon::rhi::directx12::transition(device->state.commandList(), slot->buffer.resource, slot->buffer.state,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
    device->state.commandList()->CopyBufferRegion(readback, readbackOffset, slot->buffer.resource, offset, size);
    vernon::rhi::directx12::transition(device->state.commandList(), slot->buffer.resource, slot->buffer.state,
                                       D3D12_RESOURCE_STATE_COMMON);
    if (!device->state.submitCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    std::memcpy(destination, mapped, static_cast<size_t>(size));
    return VERNON_RHI_STATUS_OK;
}

uint32_t isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupDirectX12Slot(device->buffers, buffer) != nullptr;
}

VernonRhiStatus createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor,
                            VernonRhiImage *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }

    const DXGI_FORMAT format = descriptor ? directX12Format(descriptor->format) : DXGI_FORMAT_UNKNOWN;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || format == DXGI_FORMAT_UNKNOWN ||
        descriptor->width == 0 || descriptor->height == 0 || descriptor->depth == 0 || descriptor->mip_levels == 0 ||
        descriptor->sample_count != 1)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot &slot = allocateDirectX12Slot(device->images, *output);
    D3D12_RESOURCE_DESC native{};
    native.Dimension = descriptor->dimension == VERNON_RHI_IMAGE_3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                                                    : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    native.Width = descriptor->width;
    native.Height = descriptor->height;
    native.DepthOrArraySize = static_cast<UINT16>(
        descriptor->dimension == VERNON_RHI_IMAGE_3D ? descriptor->depth : descriptor->array_layers);
    native.MipLevels = static_cast<UINT16>(descriptor->mip_levels);
    const bool sampledDepth = format == DXGI_FORMAT_D32_FLOAT && (descriptor->usage & VERNON_RHI_IMAGE_SAMPLED) != 0 &&
                              (descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0;
    native.Format = sampledDepth ? DXGI_FORMAT_R32_TYPELESS : format;
    native.SampleDesc.Count = 1;
    native.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if ((descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0)
        native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if ((descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0)
        native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if ((descriptor->usage & VERNON_RHI_IMAGE_STORAGE) != 0)
        native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (!device->state.createImage(slot.image, native, format, D3D12_RESOURCE_STATE_COMMON, device->error)) {
        releaseDirectX12Slot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot.image.dimension = descriptor->dimension;
    slot.ownedDescriptor = *descriptor;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus setImageSampler(VernonRhiDevice handle, VernonRhiImage image,
                                const VernonRhiSamplerDescriptor *descriptor) {
    return VERNON_RHI_STATUS_UNSUPPORTED;
}

VernonRhiStatus uploadImage(VernonRhiDevice handle, VernonRhiImage image, const VernonRhiImageUploadDescriptor *uploads,
                            size_t uploadCount) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!uploads || uploadCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
    if (!slot || !slot->image.owned)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiImageDescriptor &descriptor = slot->ownedDescriptor;
    if ((descriptor.dimension != VERNON_RHI_IMAGE_2D && descriptor.dimension != VERNON_RHI_IMAGE_CUBE) ||
        descriptor.format != VERNON_RHI_FORMAT_RGBA8_UNORM || descriptor.depth != 1 || descriptor.mip_levels != 1)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto &state = device->state;
    const D3D12_RESOURCE_DESC native = slot->image.resource->GetDesc();
    struct UploadFootprint {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        UINT rows{};
        UINT64 rowBytes{};
        UINT subresource{};
    };
    std::vector<UploadFootprint> footprints(uploadCount);
    UINT64 required = 0;
    const size_t expectedLayerSize = static_cast<size_t>(descriptor.width) * descriptor.height * 4;
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const bool validLayer = descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                                    ? upload.array_layer < descriptor.array_layers
                                    : upload.array_layer == 0;
        if (upload.struct_size < sizeof(upload) || upload.mip_level != 0 || !validLayer ||
            upload.width != descriptor.width || upload.height != descriptor.height || upload.depth != 1 ||
            upload.source_format != VERNON_RHI_IMAGE_DATA_RGBA || upload.source_type != VERNON_RHI_IMAGE_DATA_UINT8 ||
            !upload.data)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        footprints[index].subresource = upload.array_layer;
        UINT64 nextRequired = 0;
        state.device->GetCopyableFootprints(&native, footprints[index].subresource, 1, required,
                                            &footprints[index].placed, &footprints[index].rows,
                                            &footprints[index].rowBytes, &nextRequired);
        if (static_cast<size_t>(footprints[index].rowBytes) * footprints[index].rows != expectedLayerSize)
            return VERNON_RHI_STATUS_UNSUPPORTED;
        // pTotalBytes is the footprint size for this call, independent of
        // BaseOffset. Advance past the aligned placed footprint instead of
        // reusing the same staging range for every array layer.
        if (nextRequired > (std::numeric_limits<UINT64>::max)() - footprints[index].placed.Offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        required = footprints[index].placed.Offset + nextRequired;
    }
    ID3D12Resource *staging = nullptr;
    size_t stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(true, static_cast<size_t>(required), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, staging,
                              stagingOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    for (size_t index = 0; index < uploadCount; ++index) {
        auto &footprint = footprints[index];
        footprint.placed.Offset += stagingOffset;
        for (UINT row = 0; row < footprint.rows; ++row)
            std::memcpy(mapped + (footprint.placed.Offset - stagingOffset) + row * footprint.placed.Footprint.RowPitch,
                        static_cast<const uint8_t *>(uploads[index].data) + row * footprint.rowBytes,
                        static_cast<size_t>(footprint.rowBytes));
    }
    if (!state.beginCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    vernon::rhi::directx12::transition(state.commandList(), slot->image.resource, slot->image.state,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
    for (size_t index = 0; index < uploadCount; ++index) {
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = slot->image.resource;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = footprints[index].subresource;
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = staging;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = footprints[index].placed;
        state.commandList()->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    }
    vernon::rhi::directx12::transition(state.commandList(), slot->image.resource, slot->image.state,
                                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    return state.submitCommands(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadImage(VernonRhiDevice handle, VernonRhiImage image, void *destination, size_t size) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
    if (!slot || !slot->image.owned)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const auto expected = rgba8Size(slot->ownedDescriptor);
    if (!expected || size != *expected)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto &state = device->state;
    const D3D12_RESOURCE_DESC native = slot->image.resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 rowBytes = 0;
    UINT64 required = 0;
    state.device->GetCopyableFootprints(&native, 0, 1, 0, &footprint, &rows, &rowBytes, &required);
    ID3D12Resource *staging = nullptr;
    size_t stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(false, static_cast<size_t>(required), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, staging,
                              stagingOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    state.device->GetCopyableFootprints(&native, 0, 1, stagingOffset, &footprint, &rows, &rowBytes, nullptr);
    if (!state.beginCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    vernon::rhi::directx12::transition(state.commandList(), slot->image.resource, slot->image.state,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION target{};
    target.pResource = staging;
    target.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    target.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = slot->image.resource;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    state.commandList()->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
    vernon::rhi::directx12::transition(state.commandList(), slot->image.resource, slot->image.state,
                                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    if (!state.submitCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    for (UINT row = 0; row < rows; ++row)
        std::memcpy(static_cast<uint8_t *>(destination) + row * rowBytes,
                    mapped + (footprint.Offset - stagingOffset) + row * footprint.Footprint.RowPitch,
                    static_cast<size_t>(rowBytes));
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    return VERNON_RHI_STATUS_UNSUPPORTED;
}

VernonRhiStatus bindImage(VernonRhiDevice handle, VernonRhiImage image, uint32_t textureUnit) {
    return VERNON_RHI_STATUS_UNSUPPORTED;
}

VernonRhiStatus destroyImage(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyImage(slot->image);
        releaseDirectX12Slot(*slot);
    }
    return VERNON_RHI_STATUS_OK;
}

uint32_t isImageValid(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupDirectX12Slot(device->images, image) != nullptr;
}

VernonRhiStatus createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor,
                              VernonRhiSampler *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    SamplerFilter filter;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        !decodeSamplerFilter(*descriptor, filter))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (filter.maxAnisotropy > 1.0f)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12SamplerSlot &slot = allocateDirectX12Slot(device->samplers, *output);
    auto address = [](uint32_t mode) {
        constexpr D3D12_TEXTURE_ADDRESS_MODE values[] = {
            D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
        return values[mode];
    };
    slot.sampler.descriptor.Filter = D3D12_ENCODE_BASIC_FILTER(
        filter.minLinear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
        filter.magLinear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
        filter.mipLinear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT, D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    slot.sampler.descriptor.AddressU = address(descriptor->address_u);
    slot.sampler.descriptor.AddressV = address(descriptor->address_v);
    slot.sampler.descriptor.AddressW = address(descriptor->address_w);
    slot.sampler.descriptor.MaxAnisotropy = 1;
    slot.sampler.descriptor.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    slot.sampler.descriptor.MinLOD = 0;
    slot.sampler.descriptor.MaxLOD = D3D12_FLOAT32_MAX;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12SamplerSlot *slot = lookupDirectX12Slot(device->samplers, sampler);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0)
        releaseDirectX12Slot(*slot);
    return VERNON_RHI_STATUS_OK;
}

uint32_t isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupDirectX12Slot(device->samplers, sampler) != nullptr;
}

VernonRhiStatus getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image, uint64_t *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = reinterpret_cast<uint64_t>(slot->image.resource);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyBuffer(slot->buffer);
        releaseDirectX12Slot(*slot);
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer, void **output) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = slot->buffer.resource;
    return VERNON_RHI_STATUS_OK;
}

bool ownsDevice(VernonRhiDevice handle) { return static_cast<bool>(lookupDirectX12Device(handle)); }

void *deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupDirectX12Device(handle);
    return device ? &device->state : nullptr;
}

bool beginCommands(VernonRhiDevice handle, uint64_t &native, VernonRhiBackend &backend) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (!device->state.beginCommands(device->error))
        return false;
    backend = VERNON_RHI_BACKEND_DIRECTX12;
    native = reinterpret_cast<uintptr_t>(device->state.commandList());
    return true;
}

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool computeWrites, bool &completed) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (native != reinterpret_cast<uintptr_t>(device->state.commandList()))
        return false;
    const bool submitted = device->state.submitCommands(device->error);
    completed = submitted && !device->state.nativeObjectsBorrowed;
    return submitted;
}

void completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return;

    std::lock_guard<std::mutex> guard(device->mutex);
    if (device->state.nativeObjectsBorrowed && native == reinterpret_cast<uintptr_t>(device->state.commandList()))
        device->state.recycleCommandStorage();
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return;

    std::lock_guard<std::mutex> guard(device->mutex);
    if (native == reinterpret_cast<uintptr_t>(device->state.commandList())) {
        if (!device->state.nativeObjectsBorrowed)
            device->state.commandList()->Close();
        device->state.recycleCommandStorage();
    }
    return;
}

bool recordBarriers(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native, const VernonRhiBarrier *barriers,
                    size_t barrierCount) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
    if (!commands || commands != device->state.commandList())
        return false;
    std::vector<D3D12_RESOURCE_BARRIER> nativeBarriers;
    try {
        nativeBarriers.reserve(barrierCount);
    } catch (const std::bad_alloc &) {
        return false;
    }
    for (size_t index = 0; index < barrierCount; ++index) {
        const D3D12_RESOURCE_STATES target = directX12ResourceState(barriers[index].new_state);
        D3D12_RESOURCE_STATES *current{};
        ID3D12Resource *resource{};
        void (*restore)(void *, uint64_t){};
        void *rollbackContext{};
        if (barriers[index].is_image) {
            auto *slot = lookupResourceRecord(device->images, resourceKey(barriers[index].image));
            if (!slot)
                return false;
            current = &slot->image.state;
            resource = slot->image.resource;
            restore = restoreDirectX12ImageState;
            rollbackContext = &slot->image;
        } else {
            auto *slot = lookupResourceRecord(device->buffers, resourceKey(barriers[index].buffer));
            if (!slot)
                return false;
            current = &slot->buffer.state;
            resource = slot->buffer.resource;
            restore = restoreDirectX12BufferState;
            rollbackContext = &slot->buffer;
        }
        if (!deferCommandRollback(handle, encoderKey, rollbackContext, static_cast<uint64_t>(*current), restore))
            return false;
        if (*current == target) {
            if (barriers[index].new_state == VERNON_RHI_STATE_SHADER_WRITE) {
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = resource;
                nativeBarriers.push_back(barrier);
            }
            continue;
        }
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.StateBefore = *current;
        barrier.Transition.StateAfter = target;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        nativeBarriers.push_back(barrier);
        *current = target;
    }
    if (!nativeBarriers.empty())
        commands->ResourceBarrier(static_cast<UINT>(nativeBarriers.size()), nativeBarriers.data());
    return true;
}

bool endRendering(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                  uint32_t colorDiscardMask, uint32_t depthStencilDiscard, const uint64_t *colorResources,
                  size_t colorCount, uint64_t depthResource) {
    auto device = lookupDirectX12Device(handle);
    auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
    if (!device || !commands || backend != VERNON_RHI_BACKEND_DIRECTX12 || backendKind != CommandRenderingStateless ||
        (colorCount && !colorResources))
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    for (size_t index = 0; index < colorCount; ++index)
        if ((colorDiscardMask & (uint32_t{1} << index)) && colorResources[index])
            commands->DiscardResource(reinterpret_cast<ID3D12Resource *>(colorResources[index]), nullptr);
    if (depthStencilDiscard && depthResource)
        commands->DiscardResource(reinterpret_cast<ID3D12Resource *>(depthResource), nullptr);
    return true;
}

bool clearColor(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind, int32_t x,
                int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target, uint32_t location,
                const float color[4]) {
    auto device = lookupDirectX12Device(handle);
    auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
    if (!device || !commands || !target || backend != VERNON_RHI_BACKEND_DIRECTX12 ||
        backendKind != CommandRenderingStateless)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    commands->ClearRenderTargetView({target}, color, 0, nullptr);
    return true;
}

bool clearDepthStencil(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                       int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target,
                       float depth, uint32_t stencil, uint32_t aspects) {
    auto device = lookupDirectX12Device(handle);
    auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
    if (!device || !commands || !target || backend != VERNON_RHI_BACKEND_DIRECTX12 ||
        backendKind != CommandRenderingStateless)
        return false;
    D3D12_CLEAR_FLAGS flags = static_cast<D3D12_CLEAR_FLAGS>(0);
    if (aspects & VERNON_RHI_ATTACHMENT_DEPTH)
        flags |= D3D12_CLEAR_FLAG_DEPTH;
    if (aspects & VERNON_RHI_ATTACHMENT_STENCIL)
        flags |= D3D12_CLEAR_FLAG_STENCIL;
    std::lock_guard<std::mutex> guard(device->mutex);
    commands->ClearDepthStencilView({target}, flags, depth, static_cast<UINT8>(stencil), 0, nullptr);
    return true;
}

uint64_t bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer))
        return resourceKey(buffer);
    return 0;
}

uint64_t imageResource(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image))
        return resourceKey(image);
    return 0;
}

uint64_t samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (DirectX12SamplerSlot *slot = lookupDirectX12Slot(device->samplers, sampler))
        return resourceKey(sampler);
    return 0;
}

bool retainResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        DirectX12BufferSlot *slot = lookupResourceRecord(device->buffers, key);
        return slot && slot->retain();
    }
    if (kind == ResourceKind::Image) {
        DirectX12ImageSlot *slot = lookupResourceRecord(device->images, key);
        return slot && slot->retain();
    }
    DirectX12SamplerSlot *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->retain();
}

uint64_t resolveResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        auto *slot = lookupResourceRecord(device->buffers, key);
        return slot && slot->occupied ? reinterpret_cast<uintptr_t>(&slot->buffer) : 0;
    }
    if (kind == ResourceKind::Image) {
        auto *slot = lookupResourceRecord(device->images, key);
        return slot && slot->occupied ? reinterpret_cast<uintptr_t>(&slot->image) : 0;
    }
    auto *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->occupied ? reinterpret_cast<uintptr_t>(&slot->sampler) : 0;
}

void releaseResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return;

    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        auto *slot = lookupResourceRecord(device->buffers, key);
        if (!slot || !slot->release())
            return;
        device->state.destroyBuffer(slot->buffer);
        releaseDirectX12Slot(*slot);
        return;
    }
    if (kind == ResourceKind::Image) {
        auto *slot = lookupResourceRecord(device->images, key);
        if (!slot || !slot->release())
            return;
        device->state.destroyImage(slot->image);
        releaseDirectX12Slot(*slot);
        return;
    }
    auto *slot = lookupResourceRecord(device->samplers, key);
    if (!slot || !slot->release())
        return;
    releaseDirectX12Slot(*slot);
    return;
}

uint64_t trackedBufferState(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return UINT64_MAX;
    std::lock_guard<std::mutex> guard(device->mutex);
    const auto *slot = lookupResourceRecord(device->buffers, resourceKey(buffer));
    return slot ? static_cast<uint64_t>(slot->buffer.state) : UINT64_MAX;
}

} // namespace vernon::rhi::directx12_api

const vernon::rhi::BackendDispatch &vernon::rhi::directX12BackendDispatch() {
    using namespace directx12_api;
    static const BackendDispatch dispatch{
        VERNON_RHI_BACKEND_DIRECTX12,
        ownsDevice,
        createOwnedDevice,
        destroyDevice,
        lastError,
        synchronize,
        deviceStateForBackend,
        createBuffer,
        uploadBuffer,
        downloadBuffer,
        destroyBuffer,
        isBufferValid,
        getBufferNativeHandle,
        createImage,
        setImageSampler,
        uploadImage,
        downloadImage,
        generateImageMipmaps,
        bindImage,
        destroyImage,
        isImageValid,
        getImageNativeHandle,
        createSampler,
        destroySampler,
        isSamplerValid,
        bufferResource,
        imageResource,
        samplerResource,
        retainResource,
        resolveResource,
        releaseResource,
        beginCommands,
        submitCommands,
        completeBorrowedCommands,
        abandonCommands,
        recordBarriers,
        endRendering,
        clearColor,
        clearDepthStencil,
        trackedBufferState,
    };
    return dispatch;
}
