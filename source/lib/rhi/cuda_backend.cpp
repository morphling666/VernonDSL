#include "cuda_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace vernon::rhi::cuda {
namespace {

constexpr Result kInvalidValue = 1;
constexpr int kComputeCapabilityMajorAttribute = 75;
constexpr int kComputeCapabilityMinorAttribute = 76;

} // namespace

DeviceState::~DeviceState() { shutdown(); }

Result DeviceState::initialize(uint32_t deviceIndex) {
    Driver &api = driver();
    if (!api.load())
        return kInvalidValue;
    Result status = api.init(0);
    int major = 0;
    int minor = 0;
    int version = 0;
    if (status == kSuccess)
        status = api.deviceGet(&device, static_cast<int>(deviceIndex));
    if (status == kSuccess)
        status = api.deviceGetAttribute(&major, kComputeCapabilityMajorAttribute, device);
    if (status == kSuccess)
        status = api.deviceGetAttribute(&minor, kComputeCapabilityMinorAttribute, device);
    if (status == kSuccess)
        status = api.driverGetVersion(&version);
    if (status == kSuccess)
        status = api.primaryContextRetain(&context, device);
    if (status == kSuccess)
        status = api.contextSetCurrent(context);
    if (status == kSuccess)
        status = api.streamCreate(&stream, 0);
    if (status != kSuccess) {
        shutdown();
        return status;
    }
    computeCapabilityMajor = static_cast<uint32_t>(major);
    computeCapabilityMinor = static_cast<uint32_t>(minor);
    driverVersion = static_cast<uint32_t>(version);
    return kSuccess;
}

void DeviceState::shutdown() {
    Driver &api = driver();
    if (context)
        api.contextSetCurrent(context);
    if (stream) {
        api.streamSynchronize(stream);
        api.streamDestroy(stream);
        stream = nullptr;
    }
    for (PinnedBlock &block : stagingPool)
        if (block.data)
            api.hostFree(block.data);
    stagingPool.clear();
    if (context)
        api.primaryContextRelease(device);
    context = nullptr;
    device = 0;
}

Result DeviceState::makeCurrent() const { return context ? driver().contextSetCurrent(context) : kInvalidValue; }

Result DeviceState::synchronize() const {
    Result status = makeCurrent();
    return status == kSuccess ? driver().streamSynchronize(stream) : status;
}

Result DeviceState::allocate(DevicePointer &pointer, size_t size) const {
    Result status = makeCurrent();
    return status == kSuccess && size != 0 ? driver().memoryAllocate(&pointer, size) : kInvalidValue;
}

Result DeviceState::free(DevicePointer pointer) const {
    if (!pointer)
        return kSuccess;
    Result status = makeCurrent();
    return status == kSuccess ? driver().memoryFree(pointer) : status;
}

DeviceState::PinnedBlock *DeviceState::acquirePinned(size_t size, Result &status) {
    std::lock_guard<std::mutex> guard(stagingMutex);
    auto available = std::min_element(
        stagingPool.begin(), stagingPool.end(), [size](const PinnedBlock &left, const PinnedBlock &right) {
            const size_t leftSize = !left.inUse && left.size >= size ? left.size : SIZE_MAX;
            const size_t rightSize = !right.inUse && right.size >= size ? right.size : SIZE_MAX;
            return leftSize < rightSize;
        });
    if (available != stagingPool.end() && !available->inUse && available->size >= size) {
        available->inUse = true;
        status = kSuccess;
        return &*available;
    }
    void *data = nullptr;
    status = driver().hostAllocate(&data, size, 0);
    if (status != kSuccess)
        return nullptr;
    try {
        stagingPool.push_back({data, size, true});
    } catch (const std::bad_alloc &) {
        driver().hostFree(data);
        status = kInvalidValue;
        return nullptr;
    }
    return &stagingPool.back();
}

void DeviceState::releasePinned(PinnedBlock &block) {
    std::lock_guard<std::mutex> guard(stagingMutex);
    block.inUse = false;
}

Result DeviceState::upload(DevicePointer destination, const void *source, size_t size) {
    if (!destination || !source || size == 0)
        return kInvalidValue;
    Result status = makeCurrent();
    if (status != kSuccess)
        return status;
    PinnedBlock *staging = acquirePinned(size, status);
    if (!staging)
        return status;
    std::memcpy(staging->data, source, size);
    status = driver().copyHostToDeviceAsync(destination, staging->data, size, stream);
    if (status == kSuccess)
        status = driver().streamSynchronize(stream);
    releasePinned(*staging);
    return status;
}

Result DeviceState::uploadRanges(DevicePointer destination, const VernonRhiBufferUploadRange *ranges,
                                 size_t rangeCount) {
    if (!destination || !ranges || rangeCount == 0)
        return kInvalidValue;
    size_t stagingSize = 0;
    for (size_t index = 0; index < rangeCount; ++index) {
        if (ranges[index].size > (std::numeric_limits<size_t>::max)() - stagingSize)
            return kInvalidValue;
        stagingSize += static_cast<size_t>(ranges[index].size);
    }
    Result status = makeCurrent();
    if (status != kSuccess)
        return status;
    PinnedBlock *staging = acquirePinned(stagingSize, status);
    if (!staging)
        return status;
    size_t stagingOffset = 0;
    for (size_t index = 0; index < rangeCount; ++index) {
        const size_t size = static_cast<size_t>(ranges[index].size);
        auto *source = static_cast<std::byte *>(staging->data) + stagingOffset;
        std::memcpy(source, ranges[index].source, size);
        status = driver().copyHostToDeviceAsync(destination + ranges[index].offset, source, size, stream);
        if (status != kSuccess)
            break;
        stagingOffset += size;
    }
    const Result synchronizationStatus = driver().streamSynchronize(stream);
    if (status == kSuccess)
        status = synchronizationStatus;
    releasePinned(*staging);
    return status;
}

Result DeviceState::download(void *destination, DevicePointer source, size_t size) {
    if (!destination || !source || size == 0)
        return kInvalidValue;
    Result status = makeCurrent();
    if (status != kSuccess)
        return status;
    PinnedBlock *staging = acquirePinned(size, status);
    if (!staging)
        return status;
    status = driver().copyDeviceToHostAsync(staging->data, source, size, stream);
    if (status == kSuccess)
        status = driver().streamSynchronize(stream);
    if (status == kSuccess)
        std::memcpy(destination, staging->data, size);
    releasePinned(*staging);
    return status;
}

Result PreparedFunction::create(DeviceState &device, const void *artifact, size_t artifactSize, const char *entry) {
    if (!artifact || artifactSize == 0 || !entry || !*entry)
        return kInvalidValue;
    Result status = device.makeCurrent();
    if (status != kSuccess)
        return status;
    std::vector<char> image;
    try {
        image.resize(artifactSize + 1);
    } catch (const std::bad_alloc &) {
        return kInvalidValue;
    }
    std::memcpy(image.data(), artifact, artifactSize);
    image.back() = '\0';
    status = driver().moduleLoadData(&module, image.data(), 0, nullptr, nullptr);
    if (status == kSuccess)
        status = driver().moduleGetFunction(&function, module, entry);
    if (status != kSuccess) {
        if (module)
            driver().moduleUnload(module);
        module = nullptr;
        function = nullptr;
    }
    return status;
}

Result PreparedFunction::destroy(DeviceState &device) {
    if (!module)
        return kSuccess;
    Result status = device.makeCurrent();
    if (status == kSuccess)
        status = driver().moduleUnload(module);
    if (status == kSuccess) {
        module = nullptr;
        function = nullptr;
    }
    return status;
}

Result PreparedFunction::launch(DeviceState &device, const uint32_t groupCount[3], const uint32_t blockSize[3],
                                void **parameters) const {
    Result status = device.makeCurrent();
    if (status != kSuccess)
        return status;
    return driver().launchKernel(function, groupCount[0], groupCount[1], groupCount[2], blockSize[0], blockSize[1],
                                 blockSize[2], 0, device.stream, parameters, nullptr);
}

std::string describeResult(Result result, const char *operation) {
    const char *name = nullptr;
    const char *description = nullptr;
    Driver &api = driver();
    if (api.errorName)
        api.errorName(result, &name);
    if (api.errorString)
        api.errorString(result, &description);
    if (!api.available && !api.error.empty())
        description = api.error.c_str();
    return std::string(operation) + " failed: " + (name ? name : "CUDA_ERROR") + " (" +
           (description ? description : "unknown") + ")";
}

} // namespace vernon::rhi::cuda
