#include "backend_dispatch.h"
#include "directx12_backend.h"
#include "directx12_mipmap_2d.h"
#include "directx12_mipmap_3d.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
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

namespace vernon::rhi {
bool deviceHasActiveCommandEncoder(VernonRhiDevice device);
}

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

struct DirectX12ImageViewSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::directx12::Image image;
    VernonRhiImageViewDescriptor descriptor{};
    uint64_t imageResource{};
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
    std::deque<DirectX12ImageViewSlot> imageViews;
    std::deque<DirectX12SamplerSlot> samplers;
    std::deque<DirectX12DescriptorRangeSlot> descriptorRanges;
    uint32_t queueCapabilities{};
    ID3D12RootSignature *mipmapRootSignature{};
    ID3D12PipelineState *mipmap2dPipeline{};
    ID3D12PipelineState *mipmap3dPipeline{};
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

template <typename Handle> uint64_t resourceKey(Handle handle) { return vernon::rhi::encodeResourceKey(handle); }

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

void restoreDirectX12ImageState(void *context, uint64_t encoderKey) {
    auto &image = *static_cast<vernon::rhi::directx12::Image *>(context);
    const auto found = image.stateJournals.find(encoderKey);
    if (found == image.stateJournals.end())
        return;
    image.state = found->second.state;
    image.subresourceStates = found->second.subresources;
}

void clearDirectX12ImageStateJournal(void *context, uint64_t encoderKey) {
    static_cast<vernon::rhi::directx12::Image *>(context)->stateJournals.erase(encoderKey);
}

void transitionDirectX12Image(ID3D12GraphicsCommandList *commands, vernon::rhi::directx12::Image &image,
                              D3D12_RESOURCE_STATES target) {
    if (image.subresourceStates.empty()) {
        vernon::rhi::directx12::transition(commands, image.resource, image.state, target);
        return;
    }
    const bool uniform =
        std::all_of(image.subresourceStates.begin(), image.subresourceStates.end(),
                    [&](D3D12_RESOURCE_STATES state) { return state == image.subresourceStates.front(); });
    if (uniform) {
        D3D12_RESOURCE_STATES source = image.subresourceStates.front();
        vernon::rhi::directx12::transition(commands, image.resource, source, target);
    } else {
        for (size_t subresource = 0; subresource < image.subresourceStates.size(); ++subresource) {
            const D3D12_RESOURCE_STATES source = image.subresourceStates[subresource];
            if (source == target)
                continue;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = image.resource;
            barrier.Transition.StateBefore = source;
            barrier.Transition.StateAfter = target;
            barrier.Transition.Subresource = static_cast<UINT>(subresource);
            commands->ResourceBarrier(1, &barrier);
        }
    }
    std::fill(image.subresourceStates.begin(), image.subresourceStates.end(), target);
    image.state = target;
}

uint32_t directX12PlaneCount(VernonRhiFormat format) { return format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT ? 2u : 1u; }

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
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    };
    const uint32_t index = static_cast<uint32_t>(format);
    return index < sizeof(formats) / sizeof(formats[0]) ? formats[index] : DXGI_FORMAT_UNKNOWN;
}

bool ensureMipmapPipelines(DirectX12InteropDevice &device) {
    if (device.mipmapRootSignature && device.mipmap2dPipeline && device.mipmap3dPipeline)
        return true;
    if (device.mipmap3dPipeline)
        device.mipmap3dPipeline->Release();
    if (device.mipmap2dPipeline)
        device.mipmap2dPipeline->Release();
    if (device.mipmapRootSignature)
        device.mipmapRootSignature->Release();
    device.mipmap3dPipeline = nullptr;
    device.mipmap2dPipeline = nullptr;
    device.mipmapRootSignature = nullptr;
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 0;
    D3D12_ROOT_PARAMETER parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {1, &ranges[0]};
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable = {1, &ranges[1]};
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[2].Constants = {0, 0, 4};
    parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const D3D12_ROOT_SIGNATURE_DESC root{3, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ID3DBlob *serialized{};
    ID3DBlob *diagnostics{};
    if (FAILED(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &diagnostics))) {
        device.error = diagnostics ? std::string(static_cast<const char *>(diagnostics->GetBufferPointer()),
                                                 diagnostics->GetBufferSize())
                                   : "D3D12 mipmap root signature serialization failed";
        if (diagnostics)
            diagnostics->Release();
        return false;
    }
    if (diagnostics)
        diagnostics->Release();
    if (FAILED(device.state.device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                                        IID_PPV_ARGS(&device.mipmapRootSignature)))) {
        serialized->Release();
        device.error = "D3D12 mipmap root signature creation failed";
        return false;
    }
    serialized->Release();
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = device.mipmapRootSignature;
    pipeline.CS = {vernon_directx12_mipmap_2d, vernon_directx12_mipmap_2d_size};
    if (FAILED(device.state.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&device.mipmap2dPipeline)))) {
        device.error = "D3D12 2D mipmap pipeline creation failed";
        return false;
    }
    pipeline.CS = {vernon_directx12_mipmap_3d, vernon_directx12_mipmap_3d_size};
    if (FAILED(device.state.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&device.mipmap3dPipeline)))) {
        device.error = "D3D12 3D mipmap pipeline creation failed";
        return false;
    }
    return true;
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
    try {
        slot.image.subresourceStates.assign(static_cast<size_t>(descriptor->image.mip_levels) *
                                                descriptor->image.array_layers *
                                                directX12PlaneCount(descriptor->image.format),
                                            slot.image.state);
    } catch (const std::bad_alloc &) {
        releaseDirectX12Slot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
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
        if (device->mipmap3dPipeline)
            device->mipmap3dPipeline->Release();
        if (device->mipmap2dPipeline)
            device->mipmap2dPipeline->Release();
        if (device->mipmapRootSignature)
            device->mipmapRootSignature->Release();
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

    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->size > (std::numeric_limits<size_t>::max)() ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot &slot = allocateDirectX12Slot(device->buffers, *output);
    const bool unorderedAccess =
        descriptor->memory_class == VERNON_RHI_MEMORY_DEVICE && (descriptor->usage & VERNON_RHI_BUFFER_STORAGE) != 0;
    const D3D12_HEAP_TYPE heapType = descriptor->memory_class == VERNON_RHI_MEMORY_UPLOAD     ? D3D12_HEAP_TYPE_UPLOAD
                                     : descriptor->memory_class == VERNON_RHI_MEMORY_READBACK ? D3D12_HEAP_TYPE_READBACK
                                                                                              : D3D12_HEAP_TYPE_DEFAULT;
    const D3D12_RESOURCE_STATES initialState =
        descriptor->memory_class == VERNON_RHI_MEMORY_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
        : descriptor->memory_class == VERNON_RHI_MEMORY_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                 : D3D12_RESOURCE_STATE_COMMON;
    if (!device->state.createBuffer(slot.buffer, static_cast<size_t>(descriptor->size), unorderedAccess, heapType,
                                    initialState, device->error)) {
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
    if (slot->ownedDescriptor.memory_class == VERNON_RHI_MEMORY_UPLOAD) {
        const D3D12_RANGE readRange{0, 0};
        void *mapped = nullptr;
        if (FAILED(slot->buffer.resource->Map(0, &readRange, &mapped))) {
            device->error = "DirectX 12 upload-buffer mapping failed";
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(static_cast<uint8_t *>(mapped) + offset, source, static_cast<size_t>(size));
        const D3D12_RANGE writtenRange{static_cast<SIZE_T>(offset), static_cast<SIZE_T>(offset + size)};
        slot->buffer.resource->Unmap(0, &writtenRange);
        return VERNON_RHI_STATUS_OK;
    }
    if (vernon::rhi::deviceHasActiveCommandEncoder(handle)) {
        device->error = "device-level DirectX 12 upload cannot submit while a command encoder is recording";
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
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

VernonRhiStatus uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                   const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    if (!ranges || rangeCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot *slot = lookupDirectX12Slot(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (slot->ownedDescriptor.memory_class == VERNON_RHI_MEMORY_UPLOAD) {
        uint64_t writtenBegin = slot->ownedDescriptor.size;
        uint64_t writtenEnd = 0;
        for (size_t index = 0; index < rangeCount; ++index) {
            const VernonRhiBufferUploadRange &range = ranges[index];
            if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
                range.offset > slot->ownedDescriptor.size || range.size > slot->ownedDescriptor.size - range.offset)
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            writtenBegin = std::min(writtenBegin, range.offset);
            writtenEnd = std::max(writtenEnd, range.offset + range.size);
        }
        const D3D12_RANGE readRange{0, 0};
        void *mapped = nullptr;
        if (FAILED(slot->buffer.resource->Map(0, &readRange, &mapped))) {
            device->error = "DirectX 12 upload-buffer mapping failed";
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        for (size_t index = 0; index < rangeCount; ++index)
            std::memcpy(static_cast<uint8_t *>(mapped) + ranges[index].offset, ranges[index].source,
                        static_cast<size_t>(ranges[index].size));
        const D3D12_RANGE writtenRange{static_cast<SIZE_T>(writtenBegin), static_cast<SIZE_T>(writtenEnd)};
        slot->buffer.resource->Unmap(0, &writtenRange);
        return VERNON_RHI_STATUS_OK;
    }
    if (vernon::rhi::deviceHasActiveCommandEncoder(handle)) {
        device->error = "device-level DirectX 12 upload cannot submit while a command encoder is recording";
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    std::vector<size_t> packedOffsets;
    packedOffsets.reserve(rangeCount);
    size_t packedSize = 0;
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot->ownedDescriptor.size || range.size > slot->ownedDescriptor.size - range.offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        if (packedSize > (std::numeric_limits<size_t>::max)() - 255u)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        packedSize = (packedSize + 255u) & ~size_t{255u};
        packedOffsets.push_back(packedSize);
        if (static_cast<size_t>(range.size) > (std::numeric_limits<size_t>::max)() - packedSize)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        packedSize += static_cast<size_t>(range.size);
    }
    ID3D12Resource *upload = nullptr;
    size_t uploadOffset = 0;
    uint8_t *mapped = nullptr;
    if (!device->state.acquireStaging(true, packedSize, 256, upload, uploadOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    for (size_t index = 0; index < rangeCount; ++index)
        std::memcpy(mapped + packedOffsets[index], ranges[index].source, static_cast<size_t>(ranges[index].size));
    if (!device->state.beginCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    vernon::rhi::directx12::transition(device->state.commandList(), slot->buffer.resource, slot->buffer.state,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
    for (size_t index = 0; index < rangeCount; ++index)
        device->state.commandList()->CopyBufferRegion(slot->buffer.resource, ranges[index].offset, upload,
                                                      uploadOffset + packedOffsets[index], ranges[index].size);
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
    const bool depthFormat = descriptor && (descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT ||
                                            descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || format == DXGI_FORMAT_UNKNOWN ||
        descriptor->width == 0 || descriptor->height == 0 || descriptor->depth == 0 || descriptor->mip_levels == 0 ||
        descriptor->sample_count != 1 || (depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)) ||
        (!depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)))
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
    const bool mutableRgba8 =
        descriptor->format == VERNON_RHI_FORMAT_RGBA8_UNORM || descriptor->format == VERNON_RHI_FORMAT_RGBA8_SRGB;
    native.Format = sampledDepth ? DXGI_FORMAT_R32_TYPELESS : mutableRgba8 ? DXGI_FORMAT_R8G8B8A8_TYPELESS : format;
    native.SampleDesc.Count = 1;
    native.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if ((descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0)
        native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if ((descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0)
        native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if ((descriptor->usage & VERNON_RHI_IMAGE_STORAGE) != 0)
        native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (descriptor->mip_levels > 1 && !depthFormat) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
            descriptor->format == VERNON_RHI_FORMAT_RGBA8_SRGB ? DXGI_FORMAT_R8G8B8A8_UNORM : format};
        if (SUCCEEDED(
                device->state.device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
            (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0)
            native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if (!device->state.createImage(slot.image, native, format, D3D12_RESOURCE_STATE_COMMON, device->error)) {
        releaseDirectX12Slot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot.image.dimension = descriptor->dimension;
    try {
        slot.image.subresourceStates.assign(static_cast<size_t>(descriptor->mip_levels) * descriptor->array_layers *
                                                directX12PlaneCount(descriptor->format),
                                            slot.image.state);
    } catch (const std::bad_alloc &) {
        device->state.destroyImage(slot.image);
        releaseDirectX12Slot(slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
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
    if ((descriptor.dimension != VERNON_RHI_IMAGE_2D && descriptor.dimension != VERNON_RHI_IMAGE_3D &&
         descriptor.dimension != VERNON_RHI_IMAGE_CUBE) ||
        (descriptor.dimension != VERNON_RHI_IMAGE_3D && descriptor.depth != 1))
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
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const bool validLayer = descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                                    ? upload.array_layer < descriptor.array_layers
                                    : upload.array_layer == 0;
        const bool validMip = upload.mip_level < descriptor.mip_levels;
        const uint32_t mipWidth = validMip ? vernon::rhi::imageMipExtent(descriptor.width, upload.mip_level) : 0;
        const uint32_t mipHeight = validMip ? vernon::rhi::imageMipExtent(descriptor.height, upload.mip_level) : 0;
        const uint32_t mipDepth = !validMip ? 0
                                  : descriptor.dimension == VERNON_RHI_IMAGE_3D
                                      ? vernon::rhi::imageMipExtent(descriptor.depth, upload.mip_level)
                                      : descriptor.depth;
        const size_t pixelSize = vernon::rhi::imageFormatPixelSize(descriptor.format);
        if (upload.struct_size < sizeof(upload) || !validMip || !validLayer || !upload.width || !upload.height ||
            !upload.depth || upload.offset_x >= mipWidth || upload.width > mipWidth - upload.offset_x ||
            upload.offset_y >= mipHeight || upload.height > mipHeight - upload.offset_y ||
            upload.offset_z >= mipDepth || upload.depth > mipDepth - upload.offset_z || !pixelSize ||
            !vernon::rhi::uploadLayoutMatches(descriptor.format, upload.source_format, upload.source_type) ||
            !upload.data)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        auto &footprint = footprints[index];
        footprint.subresource = upload.mip_level + upload.array_layer * descriptor.mip_levels;
        footprint.rowBytes = static_cast<UINT64>(upload.width) * pixelSize;
        footprint.rows = upload.height;
        footprint.placed.Offset = (required + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) &
                                  ~(static_cast<UINT64>(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) - 1);
        footprint.placed.Footprint.Format = native.Format;
        footprint.placed.Footprint.Width = upload.width;
        footprint.placed.Footprint.Height = upload.height;
        footprint.placed.Footprint.Depth = upload.depth;
        footprint.placed.Footprint.RowPitch =
            static_cast<UINT>((footprint.rowBytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                              ~(static_cast<UINT64>(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) - 1));
        const UINT64 footprintSize =
            static_cast<UINT64>(footprint.placed.Footprint.RowPitch) * upload.height * upload.depth;
        if (footprintSize > (std::numeric_limits<UINT64>::max)() - footprint.placed.Offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        required = footprint.placed.Offset + footprintSize;
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
        for (UINT slice = 0; slice < footprint.placed.Footprint.Depth; ++slice)
            for (UINT row = 0; row < footprint.rows; ++row)
                std::memcpy(mapped + (footprint.placed.Offset - stagingOffset) +
                                (static_cast<size_t>(slice) * footprint.rows + row) *
                                    footprint.placed.Footprint.RowPitch,
                            static_cast<const uint8_t *>(uploads[index].data) +
                                (static_cast<size_t>(slice) * footprint.rows + row) * footprint.rowBytes,
                            static_cast<size_t>(footprint.rowBytes));
    }
    if (!state.beginCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    transitionDirectX12Image(state.commandList(), slot->image, D3D12_RESOURCE_STATE_COPY_DEST);
    for (size_t index = 0; index < uploadCount; ++index) {
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = slot->image.resource;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = footprints[index].subresource;
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = staging;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = footprints[index].placed;
        state.commandList()->CopyTextureRegion(&destination, uploads[index].offset_x, uploads[index].offset_y,
                                               uploads[index].offset_z, &source, nullptr);
    }
    transitionDirectX12Image(state.commandList(), slot->image, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    return state.submitCommands(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                              const VernonRhiImageDownloadDescriptor *download, void *destination, size_t size) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
    if (!slot || !slot->image.owned || !download || download->struct_size < sizeof(*download) || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const auto expected = imageDownloadByteSize(slot->ownedDescriptor, *download);
    if (!expected || size != *expected)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto &state = device->state;
    const D3D12_RESOURCE_DESC native = slot->image.resource->GetDesc();
    const bool depthStencil = slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    const UINT planeCount = depthStencil ? 2u : 1u;
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, 2> footprints{};
    std::array<UINT, 2> rows{};
    std::array<UINT64, 2> rowBytes{};
    UINT64 required = 0;
    const UINT baseSubresource = download->mip_level + download->array_layer * slot->ownedDescriptor.mip_levels;
    const UINT planeStride = slot->ownedDescriptor.mip_levels * slot->ownedDescriptor.array_layers;
    for (UINT plane = 0; plane < planeCount; ++plane) {
        UINT64 planeRequired = 0;
        state.device->GetCopyableFootprints(&native, baseSubresource + plane * planeStride, 1, required,
                                            &footprints[plane], &rows[plane], &rowBytes[plane], &planeRequired);
        required = planeRequired;
    }
    ID3D12Resource *staging = nullptr;
    size_t stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(false, static_cast<size_t>(required), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, staging,
                              stagingOffset, mapped, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    required = stagingOffset;
    for (UINT plane = 0; plane < planeCount; ++plane) {
        UINT64 planeRequired = 0;
        state.device->GetCopyableFootprints(&native, baseSubresource + plane * planeStride, 1, required,
                                            &footprints[plane], &rows[plane], &rowBytes[plane], &planeRequired);
        required = planeRequired;
    }
    if (!state.beginCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    transitionDirectX12Image(state.commandList(), slot->image, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const D3D12_BOX sourceBox{download->offset_x,
                              download->offset_y,
                              download->offset_z,
                              download->offset_x + download->width,
                              download->offset_y + download->height,
                              download->offset_z + download->depth};
    for (UINT plane = 0; plane < planeCount; ++plane) {
        D3D12_TEXTURE_COPY_LOCATION target{};
        target.pResource = staging;
        target.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        target.PlacedFootprint = footprints[plane];
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = slot->image.resource;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = baseSubresource + plane * planeStride;
        state.commandList()->CopyTextureRegion(&target, 0, 0, 0, &source, &sourceBox);
    }
    transitionDirectX12Image(state.commandList(), slot->image,
                             depthStencil ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                                          : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    if (!state.submitCommands(device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    if (!depthStencil) {
        const size_t outputRowBytes =
            static_cast<size_t>(download->width) * vernon::rhi::imageFormatPixelSize(slot->ownedDescriptor.format);
        for (UINT slice = 0; slice < download->depth; ++slice)
            for (UINT row = 0; row < download->height; ++row)
                std::memcpy(static_cast<uint8_t *>(destination) +
                                (static_cast<size_t>(slice) * download->height + row) * outputRowBytes,
                            mapped + (footprints[0].Offset - stagingOffset) +
                                (static_cast<size_t>(slice) * footprints[0].Footprint.Height + row) *
                                    footprints[0].Footprint.RowPitch,
                            outputRowBytes);
    } else {
        auto *output = static_cast<uint8_t *>(destination);
        for (UINT row = 0; row < download->height; ++row) {
            const uint8_t *depthRow =
                mapped + (footprints[0].Offset - stagingOffset) + row * footprints[0].Footprint.RowPitch;
            const uint8_t *stencilRow =
                mapped + (footprints[1].Offset - stagingOffset) + row * footprints[1].Footprint.RowPitch;
            for (UINT column = 0; column < download->width; ++column) {
                const size_t pixel = static_cast<size_t>(row) * download->width + column;
                float depth{};
                std::memcpy(&depth, depthRow + column * sizeof(depth), sizeof(depth));
                vernon::rhi::storePackedDepthStencil(output + pixel * vernon::rhi::packedDepthStencilPixelSize, depth,
                                                     stencilRow[column]);
            }
        }
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot *slot = lookupDirectX12Slot(device->images, image);
    if (!slot || !slot->image.owned || slot->ownedDescriptor.mip_levels < 2 ||
        slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT ||
        slot->ownedDescriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT ||
        !(slot->ownedDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(slot->ownedDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const D3D12_RESOURCE_DESC resourceDescriptor = slot->image.resource->GetDesc();
    if ((resourceDescriptor.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    if (!ensureMipmapPipelines(*device))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    auto &state = device->state;
    ID3D12PipelineState *pipeline =
        slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D ? device->mipmap3dPipeline : device->mipmap2dPipeline;
    if (!state.beginCommands(device->error, pipeline))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    ID3D12GraphicsCommandList *commands = state.commandList();
    transitionDirectX12Image(commands, slot->image, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    commands->SetComputeRootSignature(device->mipmapRootSignature);
    const DXGI_FORMAT srvFormat = slot->image.format;
    const DXGI_FORMAT uavFormat =
        slot->ownedDescriptor.format == VERNON_RHI_FORMAT_RGBA8_SRGB ? DXGI_FORMAT_R8G8B8A8_UNORM : slot->image.format;
    const UINT descriptorSize = state.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const uint32_t layers =
        slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D ? 1 : slot->ownedDescriptor.array_layers;
    for (uint32_t level = 1; level < slot->ownedDescriptor.mip_levels; ++level) {
        const uint32_t sourceWidth = vernon::rhi::imageMipExtent(slot->ownedDescriptor.width, level - 1);
        const uint32_t sourceHeight = vernon::rhi::imageMipExtent(slot->ownedDescriptor.height, level - 1);
        const uint32_t sourceDepth = slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D
                                         ? vernon::rhi::imageMipExtent(slot->ownedDescriptor.depth, level - 1)
                                         : layers;
        std::vector<D3D12_RESOURCE_BARRIER> transitions;
        transitions.reserve(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = slot->image.resource;
            barrier.Transition.Subresource = level + layer * slot->ownedDescriptor.mip_levels;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            transitions.push_back(barrier);
        }
        commands->ResourceBarrier(static_cast<UINT>(transitions.size()), transitions.data());
        ID3D12DescriptorHeap *heap{};
        D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
        D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
        if (!state.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, 2, heap, cpu, gpu, device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = srvFormat;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D) {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srv.Texture3D.MostDetailedMip = level - 1;
            srv.Texture3D.MipLevels = 1;
        } else {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srv.Texture2DArray.MostDetailedMip = level - 1;
            srv.Texture2DArray.MipLevels = 1;
            srv.Texture2DArray.ArraySize = layers;
        }
        state.device->CreateShaderResourceView(slot->image.resource, &srv, cpu);
        cpu.ptr += descriptorSize;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = uavFormat;
        if (slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D) {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            uav.Texture3D.MipSlice = level;
            uav.Texture3D.WSize = vernon::rhi::imageMipExtent(slot->ownedDescriptor.depth, level);
        } else {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            uav.Texture2DArray.MipSlice = level;
            uav.Texture2DArray.ArraySize = layers;
        }
        state.device->CreateUnorderedAccessView(slot->image.resource, nullptr, &uav, cpu);
        commands->SetDescriptorHeaps(1, &heap);
        commands->SetComputeRootDescriptorTable(0, gpu);
        gpu.ptr += descriptorSize;
        commands->SetComputeRootDescriptorTable(1, gpu);
        const uint32_t constants[4] = {
            sourceWidth,
            sourceHeight,
            sourceDepth,
            slot->ownedDescriptor.format == VERNON_RHI_FORMAT_RGBA8_SRGB ? 1u : 0u,
        };
        commands->SetComputeRoot32BitConstants(2, 4, constants, 0);
        const uint32_t destinationWidth = vernon::rhi::imageMipExtent(sourceWidth, 1);
        const uint32_t destinationHeight = vernon::rhi::imageMipExtent(sourceHeight, 1);
        const bool is3d = slot->ownedDescriptor.dimension == VERNON_RHI_IMAGE_3D;
        commands->Dispatch((destinationWidth + (is3d ? 3 : 7)) / (is3d ? 4 : 8),
                           (destinationHeight + (is3d ? 3 : 7)) / (is3d ? 4 : 8),
                           is3d ? (vernon::rhi::imageMipExtent(sourceDepth, 1) + 3) / 4 : layers);
        D3D12_RESOURCE_BARRIER uavBarrier{};
        uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarrier.UAV.pResource = slot->image.resource;
        commands->ResourceBarrier(1, &uavBarrier);
        for (D3D12_RESOURCE_BARRIER &barrier : transitions) {
            std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        }
        commands->ResourceBarrier(static_cast<UINT>(transitions.size()), transitions.data());
    }
    return state.submitCommands(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
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

VernonRhiStatus createImageView(VernonRhiDevice handle, const VernonRhiImageViewDescriptor *descriptor,
                                VernonRhiImageView *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !output ||
        !descriptor->mip_level_count || !descriptor->array_layer_count || !descriptor->aspects)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageSlot *parent = lookupDirectX12Slot(device->images, descriptor->image);
    if (!parent)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiImageDescriptor &imageDescriptor =
        parent->image.owned ? parent->ownedDescriptor : parent->descriptor.image;
    if (!vernon::rhi::validImageViewDescriptor(imageDescriptor, *descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!parent->retain())
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    DirectX12ImageViewSlot &slot = allocateDirectX12Slot(device->imageViews, *output);
    slot.image = parent->image;
    slot.image.owned = false;
    slot.image.format = directX12Format(descriptor->format);
    slot.image.dimension = descriptor->dimension;
    slot.image.view = *descriptor;
    slot.descriptor = *descriptor;
    slot.imageResource = resourceKey(descriptor->image);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyImageView(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageViewSlot *slot = lookupDirectX12Slot(device->imageViews, view);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t parentKey = slot->imageResource;
    slot->destroyPublicOwner();
    if (slot->bindingReferences != 0)
        return VERNON_RHI_STATUS_OK;
    releaseDirectX12Slot(*slot);
    DirectX12ImageSlot *parent = lookupResourceRecord(device->images, parentKey);
    if (parent && parent->release()) {
        device->state.destroyImage(parent->image);
        releaseDirectX12Slot(*parent);
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView view, uint64_t *output) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12ImageViewSlot *slot = lookupDirectX12Slot(device->imageViews, view);
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

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool computeWrites, bool &completed,
                    bool &externalCompletion) {
    auto device = lookupDirectX12Device(handle);
    if (!device) {
        return false;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (native != reinterpret_cast<uintptr_t>(device->state.commandList()))
        return false;
    const bool submitted = device->state.submitCommands(device->error);
    completed = submitted && !device->state.nativeObjectsBorrowed;
    externalCompletion = submitted && device->state.nativeObjectsBorrowed;
    return submitted;
}

bool completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return false;

    std::lock_guard<std::mutex> guard(device->mutex);
    if (device->state.nativeObjectsBorrowed && native == reinterpret_cast<uintptr_t>(device->state.commandList()))
        device->state.recycleCommandStorage();
    return true;
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
    try {
        std::vector<D3D12_RESOURCE_BARRIER> nativeBarriers;
        nativeBarriers.reserve(barrierCount);
        for (size_t index = 0; index < barrierCount; ++index) {
            const D3D12_RESOURCE_STATES target = directX12ResourceState(barriers[index].new_state);
            D3D12_RESOURCE_STATES *current{};
            const VernonRhiImageDescriptor *imageDescriptor{};
            ID3D12Resource *resource{};
            void (*restore)(void *, uint64_t){};
            void *rollbackContext{};
            if (barriers[index].is_image) {
                auto *slot = lookupResourceRecord(device->images, resourceKey(barriers[index].image));
                if (!slot)
                    return false;
                imageDescriptor = slot->ownedDescriptor.struct_size ? &slot->ownedDescriptor : &slot->descriptor.image;
                resource = slot->image.resource;
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
            const auto &range = barriers[index].image_subresources;
            const uint32_t imageLayers = imageDescriptor ? imageDescriptor->array_layers : 0;
            const uint64_t mipEnd = imageDescriptor ? (range.mip_level_count == UINT32_MAX
                                                           ? imageDescriptor->mip_levels
                                                           : uint64_t{range.base_mip_level} + range.mip_level_count)
                                                    : 0;
            const uint64_t layerEnd = imageDescriptor
                                          ? (range.array_layer_count == UINT32_MAX
                                                 ? imageLayers
                                                 : uint64_t{range.base_array_layer} + range.array_layer_count)
                                          : 0;
            if (imageDescriptor &&
                (range.base_mip_level >= imageDescriptor->mip_levels || mipEnd > imageDescriptor->mip_levels ||
                 range.base_array_layer >= imageLayers || layerEnd > imageLayers ||
                 (range.aspects & ~vernon::rhi::imageFormatAspects(imageDescriptor->format)) != 0))
                return false;
            if (!imageDescriptor &&
                !deferCommandRollback(handle, encoderKey, rollbackContext, static_cast<uint64_t>(*current), restore))
                return false;
            if (imageDescriptor) {
                auto *image = static_cast<vernon::rhi::directx12::Image *>(rollbackContext);
                const size_t planeStride = static_cast<size_t>(imageDescriptor->mip_levels) * imageLayers;
                const uint32_t planeCount = directX12PlaneCount(imageDescriptor->format);
                const size_t subresourceCount = planeStride * planeCount;
                if (image->subresourceStates.size() != subresourceCount)
                    image->subresourceStates.assign(subresourceCount, image->state);
                auto journal = image->stateJournals.find(encoderKey);
                bool inserted = false;
                if (journal == image->stateJournals.end()) {
                    const auto result = image->stateJournals.emplace(
                        encoderKey,
                        vernon::rhi::directx12::Image::StateJournal{image->state, image->subresourceStates});
                    journal = result.first;
                    inserted = result.second;
                }
                if (inserted &&
                    (!deferCommandCleanup(handle, encoderKey, image, encoderKey, clearDirectX12ImageStateJournal) ||
                     !deferCommandRollback(handle, encoderKey, image, encoderKey, restoreDirectX12ImageState))) {
                    image->stateJournals.erase(journal);
                    return false;
                }
                bool needsUavBarrier = false;
                const uint32_t availableAspects = vernon::rhi::imageFormatAspects(imageDescriptor->format);
                for (uint32_t plane = 0; plane < planeCount; ++plane) {
                    const uint32_t planeAspect =
                        imageDescriptor->format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT
                            ? (plane == 0 ? VERNON_RHI_IMAGE_ASPECT_DEPTH : VERNON_RHI_IMAGE_ASPECT_STENCIL)
                            : availableAspects;
                    if ((range.aspects & planeAspect) == 0)
                        continue;
                    for (uint32_t layer = range.base_array_layer; layer < layerEnd; ++layer)
                        for (uint32_t mip = range.base_mip_level; mip < mipEnd; ++mip) {
                            const uint32_t subresource =
                                mip + layer * imageDescriptor->mip_levels + static_cast<uint32_t>(plane * planeStride);
                            const D3D12_RESOURCE_STATES source = image->subresourceStates[subresource];
                            if (source == target) {
                                needsUavBarrier |= barriers[index].new_state == VERNON_RHI_STATE_SHADER_WRITE;
                                continue;
                            }
                            D3D12_RESOURCE_BARRIER barrier{};
                            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                            barrier.Transition.pResource = resource;
                            barrier.Transition.StateBefore = source;
                            barrier.Transition.StateAfter = target;
                            barrier.Transition.Subresource = subresource;
                            nativeBarriers.push_back(barrier);
                            image->subresourceStates[subresource] = target;
                        }
                }
                if (needsUavBarrier) {
                    D3D12_RESOURCE_BARRIER barrier{};
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    barrier.UAV.pResource = resource;
                    nativeBarriers.push_back(barrier);
                }
                if (std::all_of(image->subresourceStates.begin(), image->subresourceStates.end(),
                                [target](D3D12_RESOURCE_STATES state) { return state == target; }))
                    image->state = target;
                continue;
            }
            const D3D12_RESOURCE_STATES source = *current;
            if (source == target) {
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
            barrier.Transition.StateBefore = source;
            barrier.Transition.StateAfter = target;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            nativeBarriers.push_back(barrier);
            *current = target;
        }
        if (!nativeBarriers.empty())
            commands->ResourceBarrier(static_cast<UINT>(nativeBarriers.size()), nativeBarriers.data());
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

bool recordBufferCopy(VernonRhiDevice handle, uint64_t native, VernonRhiBuffer source, uint64_t sourceOffset,
                      VernonRhiBuffer destination, uint64_t destinationOffset, uint64_t size) {
    auto device = lookupDirectX12Device(handle);
    auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
    if (!device || !commands || commands != device->state.commandList() || !size)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    DirectX12BufferSlot *sourceSlot = lookupDirectX12Slot(device->buffers, source);
    DirectX12BufferSlot *destinationSlot = lookupDirectX12Slot(device->buffers, destination);
    if (!sourceSlot || !destinationSlot || sourceOffset > sourceSlot->ownedDescriptor.size ||
        size > sourceSlot->ownedDescriptor.size - sourceOffset ||
        destinationOffset > destinationSlot->ownedDescriptor.size ||
        size > destinationSlot->ownedDescriptor.size - destinationOffset)
        return false;
    if (sourceSlot->ownedDescriptor.memory_class == VERNON_RHI_MEMORY_DEVICE)
        vernon::rhi::directx12::transition(commands, sourceSlot->buffer.resource, sourceSlot->buffer.state,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
    vernon::rhi::directx12::transition(commands, destinationSlot->buffer.resource, destinationSlot->buffer.state,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
    commands->CopyBufferRegion(destinationSlot->buffer.resource, destinationOffset, sourceSlot->buffer.resource,
                               sourceOffset, size);
    if (sourceSlot->ownedDescriptor.memory_class == VERNON_RHI_MEMORY_DEVICE)
        vernon::rhi::directx12::transition(commands, sourceSlot->buffer.resource, sourceSlot->buffer.state,
                                           D3D12_RESOURCE_STATE_COMMON);
    vernon::rhi::directx12::transition(commands, destinationSlot->buffer.resource, destinationSlot->buffer.state,
                                       D3D12_RESOURCE_STATE_COMMON);
    return true;
}

bool endRendering(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                  uint32_t colorDiscardMask, uint32_t depthStencilDiscard, const uint64_t *colorResources,
                  size_t colorCount, uint64_t depthResource, uint64_t) {
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

uint64_t imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupDirectX12Device(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupDirectX12Slot(device->imageViews, view) ? resourceKey(view) : 0;
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
    if (kind == ResourceKind::ImageView) {
        DirectX12ImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
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
    if (kind == ResourceKind::ImageView) {
        auto *slot = lookupResourceRecord(device->imageViews, key);
        return slot ? reinterpret_cast<uintptr_t>(&slot->image) : 0;
    }
    auto *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->occupied ? reinterpret_cast<uintptr_t>(&slot->sampler) : 0;
}

bool describeImageResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageDescriptor *descriptor) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !descriptor)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const DirectX12ImageSlot *slot = lookupResourceRecord(device->images, key);
    if (!slot || !slot->occupied)
        return false;
    *descriptor = slot->image.owned ? slot->ownedDescriptor : slot->descriptor.image;
    return true;
}

bool describeImageViewResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageViewDescriptor *view,
                               VernonRhiImageDescriptor *image, uint64_t *parentKey) {
    auto device = lookupDirectX12Device(handle);
    if (!device || !view || !image || !parentKey)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const DirectX12ImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
    if (!slot)
        return false;
    const DirectX12ImageSlot *parent = lookupResourceRecord(device->images, slot->imageResource);
    if (!parent)
        return false;
    *view = slot->descriptor;
    *image = parent->image.owned ? parent->ownedDescriptor : parent->descriptor.image;
    *parentKey = slot->imageResource;
    return true;
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
    if (kind == ResourceKind::ImageView) {
        auto *slot = lookupResourceRecord(device->imageViews, key);
        if (!slot || !slot->release())
            return;
        const uint64_t parentKey = slot->imageResource;
        releaseDirectX12Slot(*slot);
        auto *parent = lookupResourceRecord(device->images, parentKey);
        if (parent && parent->release()) {
            device->state.destroyImage(parent->image);
            releaseDirectX12Slot(*parent);
        }
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
        0,
        nullptr,
        ownsDevice,
        createOwnedDevice,
        destroyDevice,
        lastError,
        synchronize,
        deviceStateForBackend,
        createBuffer,
        uploadBuffer,
        uploadBufferRanges,
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
        describeImageResource,
        releaseResource,
        beginCommands,
        submitCommands,
        nullptr,
        completeBorrowedCommands,
        abandonCommands,
        recordBarriers,
        recordBufferCopy,
        endRendering,
        clearColor,
        clearDepthStencil,
        trackedBufferState,
        createImageView,
        destroyImageView,
        getImageViewNativeHandle,
        imageViewResource,
        describeImageViewResource,
    };
    return dispatch;
}
