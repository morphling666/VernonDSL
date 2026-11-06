#include "hal/cuda/cuda_state.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace vernon::hal::cuda {
namespace {

constexpr Result kInvalidValue = 1;
constexpr int kComputeCapabilityMajorAttribute = 75;
constexpr int kComputeCapabilityMinorAttribute = 76;

// Small allocations round to a power of two, larger ones to whole MiB, so a cached allocation serves any request of
// its rounded size and wastes at most half of a small one.
constexpr size_t kCacheGranule = size_t{1} << 20;
constexpr size_t kMaximumCachedBytes = size_t{1} << 30;

size_t roundedAllocation(size_t size) {
    if (size >= kCacheGranule)
        return size > std::numeric_limits<size_t>::max() - (kCacheGranule - 1)
                   ? size
                   : (size + kCacheGranule - 1) / kCacheGranule * kCacheGranule;
    size_t rounded = 256;
    while (rounded < size)
        rounded <<= 1;
    return rounded;
}

template <typename Range> bool stagingSize(const Range *ranges, size_t count, size_t &total) {
    total = 0;
    for (size_t index = 0; index < count; ++index) {
        if (ranges[index].size > (std::numeric_limits<size_t>::max)() - total)
            return false;
        total += static_cast<size_t>(ranges[index].size);
    }
    return true;
}

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
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        releaseCachedAllocations();
        allocationSizes.clear();
    }
    for (PinnedBlock &block : stagingPool) {
        if (block.retirement)
            api.eventDestroy(block.retirement);
        if (block.data)
            api.hostFree(block.data);
    }
    stagingPool.clear();
    for (Event event : eventPool)
        api.eventDestroy(event);
    eventPool.clear();
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

Result DeviceState::acquireEvent(Event &event) {
    {
        std::lock_guard<std::mutex> guard(poolMutex);
        if (!eventPool.empty()) {
            event = eventPool.back();
            eventPool.pop_back();
            return kSuccess;
        }
    }
    return driver().eventCreate(&event, kEventDisableTiming);
}

void DeviceState::recycleEvent(Event event, bool reusable) {
    if (reusable) {
        std::lock_guard<std::mutex> guard(poolMutex);
        try {
            eventPool.push_back(event);
            return;
        } catch (const std::bad_alloc &) {
        }
    }
    driver().eventDestroy(event);
}

Result DeviceState::allocate(DevicePointer &pointer, size_t size) const {
    Result status = makeCurrent();
    if (status != kSuccess || size == 0)
        return kInvalidValue;
    const size_t rounded = roundedAllocation(size);
    std::lock_guard<std::mutex> guard(cacheMutex);
    try {
        if (auto cached = cachedAllocations.find(rounded);
            cached != cachedAllocations.end() && !cached->second.empty()) {
            pointer = cached->second.back();
            cached->second.pop_back();
            cachedBytes -= rounded;
            allocationSizes[pointer] = rounded;
            return kSuccess;
        }
        status = driver().memoryAllocate(&pointer, rounded);
        if (status != kSuccess && cachedBytes) {
            releaseCachedAllocations();
            status = driver().memoryAllocate(&pointer, rounded);
        }
        if (status == kSuccess)
            allocationSizes[pointer] = rounded;
    } catch (const std::bad_alloc &) {
        if (status == kSuccess && pointer)
            driver().memoryFree(pointer);
        return kInvalidValue;
    }
    return status;
}

Result DeviceState::free(DevicePointer pointer) const {
    if (!pointer)
        return kSuccess;
    Result status = makeCurrent();
    if (status != kSuccess)
        return status;
    std::lock_guard<std::mutex> guard(cacheMutex);
    const auto found = allocationSizes.find(pointer);
    if (found == allocationSizes.end())
        return driver().memoryFree(pointer);
    const size_t rounded = found->second;
    allocationSizes.erase(found);
    if (cachedBytes + rounded <= kMaximumCachedBytes) {
        try {
            cachedAllocations[rounded].push_back(pointer);
            cachedBytes += rounded;
            return kSuccess;
        } catch (const std::bad_alloc &) {
        }
    }
    return driver().memoryFree(pointer);
}

// The caller holds `cacheMutex` and has made the context current.
void DeviceState::releaseCachedAllocations() const {
    for (auto &[size, pointers] : cachedAllocations)
        for (DevicePointer pointer : pointers)
            driver().memoryFree(pointer);
    cachedAllocations.clear();
    cachedBytes = 0;
}

DeviceState::PinnedBlock *DeviceState::acquirePinned(size_t size, Result &status) {
    std::lock_guard<std::mutex> guard(poolMutex);
    for (PinnedBlock &block : stagingPool) {
        if (!block.retirement)
            continue;
        const Result retired = driver().eventQuery(block.retirement);
        if (retired == kNotReady)
            continue;
        if (retired != kSuccess) {
            status = retired;
            return nullptr;
        }
        try {
            eventPool.push_back(block.retirement);
        } catch (const std::bad_alloc &) {
            driver().eventDestroy(block.retirement);
        }
        block.retirement = nullptr;
        block.inUse = false;
    }
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
    std::lock_guard<std::mutex> guard(poolMutex);
    block.inUse = false;
}

Result DeviceState::retirePinned(PinnedBlock &block) {
    Event retirement{};
    Result status = acquireEvent(retirement);
    if (status == kSuccess) {
        status = driver().eventRecord(retirement, stream);
        if (status != kSuccess)
            recycleEvent(retirement, false);
    }
    if (status != kSuccess) {
        driver().streamSynchronize(stream);
        releasePinned(block);
        return status;
    }
    std::lock_guard<std::mutex> guard(poolMutex);
    block.retirement = retirement;
    return kSuccess;
}

Result DeviceState::upload(DevicePointer destination, const void *source, size_t size) {
    if (!destination || !source || size == 0)
        return kInvalidValue;
    const VernonHalBufferWrite write{0, source, size};
    return uploadRanges(destination, &write, 1);
}

Result DeviceState::uploadRanges(DevicePointer destination, const VernonHalBufferWrite *writes, size_t count) {
    size_t total = 0;
    if (!destination || !writes || count == 0 || !stagingSize(writes, count, total))
        return kInvalidValue;
    Result status = makeCurrent();
    if (status != kSuccess)
        return status;
    PinnedBlock *staging = acquirePinned(total, status);
    if (!staging)
        return status;
    size_t stagingOffset = 0;
    for (size_t index = 0; index < count; ++index) {
        const size_t size = static_cast<size_t>(writes[index].size);
        auto *source = static_cast<std::byte *>(staging->data) + stagingOffset;
        std::memcpy(source, writes[index].source, size);
        status = driver().copyHostToDeviceAsync(destination + writes[index].offset, source, size, stream);
        if (status != kSuccess)
            break;
        stagingOffset += size;
    }
    if (status != kSuccess) {
        driver().streamSynchronize(stream);
        releasePinned(*staging);
        return status;
    }
    return retirePinned(*staging);
}

Result DeviceState::download(void *destination, DevicePointer source, size_t size) {
    if (!destination || !source || size == 0)
        return kInvalidValue;
    const VernonHalBufferRead read{0, destination, size};
    return downloadRanges(source, &read, 1);
}

Result DeviceState::downloadRanges(DevicePointer source, const VernonHalBufferRead *reads, size_t count) {
    size_t total = 0;
    if (!source || !reads || count == 0 || !stagingSize(reads, count, total))
        return kInvalidValue;
    Result status = makeCurrent();
    if (status != kSuccess)
        return status;
    PinnedBlock *staging = acquirePinned(total, status);
    if (!staging)
        return status;
    size_t stagingOffset = 0;
    for (size_t index = 0; index < count; ++index) {
        const size_t size = static_cast<size_t>(reads[index].size);
        auto *destination = static_cast<std::byte *>(staging->data) + stagingOffset;
        status = driver().copyDeviceToHostAsync(destination, source + reads[index].offset, size, stream);
        if (status != kSuccess)
            break;
        stagingOffset += size;
    }
    if (status == kSuccess)
        status = driver().streamSynchronize(stream);
    if (status == kSuccess) {
        stagingOffset = 0;
        for (size_t index = 0; index < count; ++index) {
            const size_t size = static_cast<size_t>(reads[index].size);
            std::memcpy(reads[index].destination, static_cast<std::byte *>(staging->data) + stagingOffset, size);
            stagingOffset += size;
        }
    }
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

} // namespace vernon::hal::cuda
