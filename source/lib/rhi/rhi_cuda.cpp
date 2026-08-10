#include "backend_dispatch.h"
#include "cuda_backend.h"
#include "logical_resource_record.h"
#include "rhi_test_hooks.h"

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
struct CudaBufferSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::cuda::DevicePointer pointer{};
    VernonRhiBufferDescriptor descriptor{};
};

struct CudaDevice {
    vernon::rhi::cuda::DeviceState state;
    std::vector<CudaBufferSlot> buffers;
    std::string error;
    std::mutex mutex;
};

struct CudaDeviceSlot {
    std::shared_ptr<CudaDevice> device;
    uint32_t generation{1};
};

constexpr uint32_t cudaDeviceBit = uint32_t{1} << 29;
std::vector<CudaDeviceSlot> cudaDevices;
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
std::shared_ptr<CudaDevice> lookupCudaDevice(VernonRhiDevice handle) {
    if ((handle.index & cudaDeviceBit) == 0)
        return {};
    const uint32_t index = handle.index & ~cudaDeviceBit;
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (index >= cudaDevices.size())
        return {};
    CudaDeviceSlot &slot = cudaDevices[index];
    return slot.device && slot.generation == handle.generation ? slot.device : std::shared_ptr<CudaDevice>{};
}

CudaBufferSlot *lookupCudaBuffer(CudaDevice &device, VernonRhiBuffer handle) {
    if (handle.index >= device.buffers.size())
        return nullptr;
    CudaBufferSlot &slot = device.buffers[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}
} // namespace

namespace vernon::rhi::cuda_api {

VernonRhiDevice createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {

    auto device = std::shared_ptr<CudaDevice>(new (std::nothrow) CudaDevice());
    if (!device)
        return invalidDevice();
    const auto status = device->state.initialize(descriptor->device_index);
    if (status != vernon::rhi::cuda::kSuccess) {
        device->error = vernon::rhi::cuda::describeResult(status, "cuDevicePrimaryCtxRetain");
        return invalidDevice();
    }
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < cudaDevices.size() && cudaDevices[index].device)
        ++index;
    if (index == cudaDevices.size())
        cudaDevices.emplace_back();
    cudaDevices[index].device = std::move(device);
    return {index | cudaDeviceBit, cudaDevices[index].generation};
}

void destroyDevice(VernonRhiDevice handle) {

    std::shared_ptr<CudaDevice> device;
    {
        const uint32_t index = handle.index & ~cudaDeviceBit;
        std::lock_guard<std::mutex> guard(deviceMutex);
        if (index >= cudaDevices.size() || cudaDevices[index].generation != handle.generation)
            return;
        CudaDeviceSlot &slot = cudaDevices[index];
        device = std::move(slot.device);
        ++slot.generation;
        if (slot.generation == 0)
            slot.generation = 1;
    }
    if (device) {
        std::lock_guard<std::mutex> guard(device->mutex);
        for (CudaBufferSlot &buffer : device->buffers)
            if (buffer.occupied)
                device->state.free(buffer.pointer);
        device->state.shutdown();
    }
    return;
}

VernonStringView lastError(VernonRhiDevice handle) {
    auto device = lookupCudaDevice(handle);
    return device ? VernonStringView{device->error.data(), device->error.size()} : VernonStringView{};
}

VernonRhiStatus synchronize(VernonRhiDevice handle) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    const auto status = device->state.synchronize();
    if (status == vernon::rhi::cuda::kSuccess)
        return VERNON_RHI_STATUS_OK;
    device->error = vernon::rhi::cuda::describeResult(status, "cuStreamSynchronize");
    return VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor,
                             VernonRhiBuffer *output) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->size > (std::numeric_limits<size_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    uint32_t index = 0;
    while (index < device->buffers.size() && device->buffers[index].occupied)
        ++index;
    if (index == device->buffers.size())
        device->buffers.emplace_back();
    CudaBufferSlot &slot = device->buffers[index];
    const auto status = device->state.allocate(slot.pointer, static_cast<size_t>(descriptor->size));
    if (status != vernon::rhi::cuda::kSuccess) {
        device->error = vernon::rhi::cuda::describeResult(status, "cuMemAlloc");
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot.descriptor = *descriptor;
    slot.publish();
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!source || size > (std::numeric_limits<size_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const auto status = device->state.upload(slot->pointer + offset, source, static_cast<size_t>(size));
    if (status == vernon::rhi::cuda::kSuccess)
        return VERNON_RHI_STATUS_OK;
    device->error = vernon::rhi::cuda::describeResult(status, "cuMemcpyHtoDAsync");
    return VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                               uint64_t size) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    if (!destination || size > (std::numeric_limits<size_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const auto status = device->state.download(destination, slot->pointer + offset, static_cast<size_t>(size));
    if (status == vernon::rhi::cuda::kSuccess)
        return VERNON_RHI_STATUS_OK;
    device->error = vernon::rhi::cuda::describeResult(status, "cuMemcpyDtoHAsync");
    return VERNON_RHI_STATUS_INTERNAL_ERROR;
}

uint32_t isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupCudaBuffer(*device, buffer) != nullptr;
}

VernonRhiStatus destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        const auto status = device->state.free(slot->pointer);
        if (status != vernon::rhi::cuda::kSuccess) {
            slot->restorePublicOwner();
            device->error = vernon::rhi::cuda::describeResult(status, "cuMemFree");
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        slot->pointer = 0;
        slot->recycle();
    }
    return VERNON_RHI_STATUS_OK;
}

bool ownsDevice(VernonRhiDevice handle) { return static_cast<bool>(lookupCudaDevice(handle)); }

void *deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupCudaDevice(handle);
    return device ? &device->state : nullptr;
}

bool beginCommands(VernonRhiDevice handle, uint64_t &native, VernonRhiBackend &backend) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return false;
    }

    backend = VERNON_RHI_BACKEND_CUDA;
    native = reinterpret_cast<uintptr_t>(&device->state);
    return true;
}

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool computeWrites, bool &completed) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return false;
    }

    if (!native)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const auto status = device->state.synchronize();
    completed = status == vernon::rhi::cuda::kSuccess;
    if (!completed)
        device->error = vernon::rhi::cuda::describeResult(status, "cuStreamSynchronize");
    return completed;
}

bool recordBarriers(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native, const VernonRhiBarrier *barriers,
                    size_t barrierCount) {
    return static_cast<bool>(lookupCudaDevice(handle)) && native != 0;
}

uint64_t bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    if (CudaBufferSlot *slot = lookupCudaBuffer(*device, buffer))
        return resourceKey(buffer);
    return 0;
}

bool retainResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return false;
    }

    if (kind != ResourceKind::Buffer)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    CudaBufferSlot *slot = lookupResourceRecord(device->buffers, key);
    return slot && slot->retain();
}

uint64_t resolveResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupCudaDevice(handle);
    if (!device) {
        return 0;
    }

    std::lock_guard<std::mutex> guard(device->mutex);
    CudaBufferSlot *slot = kind == ResourceKind::Buffer ? lookupResourceRecord(device->buffers, key) : nullptr;
    return slot && slot->occupied ? slot->pointer : 0;
}

void releaseResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupCudaDevice(handle);
    if (!device)
        return;

    if (kind != ResourceKind::Buffer)
        return;
    std::lock_guard<std::mutex> guard(device->mutex);
    CudaBufferSlot *slot = lookupResourceRecord(device->buffers, key);
    if (!slot || !slot->release())
        return;
    const auto status = device->state.free(slot->pointer);
    if (status != vernon::rhi::cuda::kSuccess) {
        slot->retain();
        device->error = vernon::rhi::cuda::describeResult(status, "cuMemFree");
        return;
    }
    slot->pointer = 0;
    slot->recycle();
    return;
}

} // namespace vernon::rhi::cuda_api

const vernon::rhi::BackendDispatch &vernon::rhi::cudaBackendDispatch() {
    using namespace cuda_api;
    static const BackendDispatch dispatch{
        VERNON_RHI_BACKEND_CUDA,
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
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        bufferResource,
        nullptr,
        nullptr,
        retainResource,
        resolveResource,
        nullptr,
        releaseResource,
        beginCommands,
        submitCommands,
        nullptr,
        nullptr,
        recordBarriers,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
    return dispatch;
}
