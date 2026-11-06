#ifndef VERNON_HAL_NATIVE_BUFFERS_H
#define VERNON_HAL_NATIVE_BUFFERS_H

// References to the buffers a HAL device allocates.

#include "hal/hal.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::hal {

// The low bits of a HAL reference's identity, over the address of the device's state, which name what its resource
// value is the serial of.
inline constexpr uint64_t kNativeBufferReference = 5;
inline constexpr uint64_t kNativeViewReference = 6;
inline constexpr uint64_t kNativeSamplerReference = 7;
inline constexpr uint64_t kReferenceKindMask = 7;

inline uint64_t nativeIdentity(const void *state, uint64_t kind) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state)) | kind;
}

inline uint64_t nativeBufferIdentity(const void *state) { return nativeIdentity(state, kNativeBufferReference); }

inline bool isNativeBufferReference(uint64_t identity) {
    return (identity & kReferenceKindMask) == kNativeBufferReference;
}

// Serials for the buffers, views or samplers of a device whose native handles the driver reuses. A reference names its
// object by serial, which no later object reuses: recordings and plan keys compare resource values, so a reused handle
// would let a cached recording bind a new object through stale descriptors.
template <typename Native> class Serials {
public:
    uint64_t add(Native *object) {
        std::lock_guard<std::mutex> guard(mutex_);
        const uint64_t serial = next_++;
        objects_.emplace(serial, object);
        return serial;
    }

    void remove(uint64_t serial) {
        std::lock_guard<std::mutex> guard(mutex_);
        objects_.erase(serial);
    }

    // Null once the object is destroyed. The caller keeps the object alive while it uses the result.
    Native *find(uint64_t serial) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto found = objects_.find(serial);
        return found == objects_.end() ? nullptr : found->second;
    }

private:
    std::mutex mutex_;
    std::unordered_map<uint64_t, Native *> objects_;
    uint64_t next_{1};
};

// The buffers a graphics device's Device::allocate returns. A released buffer goes back to the pool, and a later
// allocation of about its size takes it again, serial included: a schedule allocates residuals per call, and its
// cached recordings name them by serial, so reuse keeps the next calls' recordings cached. `Native` owns one device
// buffer, which it frees when destroyed, and has `uint64_t size() const` and
// `VernonRuntimeProviderResourceReference reference(uint64_t offset, uint64_t size) const`.
template <typename Native> class BufferPool : public std::enable_shared_from_this<BufferPool<Native>> {
public:
    using Create = std::function<Result<std::unique_ptr<Native>, std::string>(uint64_t)>;

    class Pooled final : public Buffer {
    public:
        Pooled(std::shared_ptr<BufferPool> pool, std::unique_ptr<Native> native)
            : pool_(std::move(pool)), native_(std::move(native)) {}
        Pooled(const Pooled &) = delete;
        Pooled &operator=(const Pooled &) = delete;
        ~Pooled() override { pool_->release(std::move(native_)); }

        void *address() const override { return nullptr; }
        VernonRuntimeProviderResourceReference reference(uint64_t offset, uint64_t size) const override {
            return native_->reference(offset, size);
        }
        Native &native() const { return *native_; }

    private:
        std::shared_ptr<BufferPool> pool_;
        std::unique_ptr<Native> native_;
    };

    explicit BufferPool(Create create) : create_(std::move(create)) {}

    Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) {
        using Allocated = Result<std::unique_ptr<Buffer>, std::string>;
        size = std::max<uint64_t>(size, 1);
        std::unique_ptr<Native> native;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto best = free_.end();
            for (auto at = free_.begin(); at != free_.end(); ++at)
                if ((*at)->size() >= size && (*at)->size() / 2 <= size &&
                    (best == free_.end() || (*at)->size() < (*best)->size()))
                    best = at;
            if (best != free_.end()) {
                native = std::move(*best);
                free_.erase(best);
            }
        }
        if (!native) {
            auto created = create_(size);
            if (created.isErr())
                return Allocated{err(std::move(created).error())};
            native = std::move(created).value();
        }
        return Allocated{ok(std::unique_ptr<Buffer>(new Pooled(this->shared_from_this(), std::move(native))))};
    }

private:
    static constexpr size_t kKept = 16;

    void release(std::unique_ptr<Native> native) {
        std::unique_ptr<Native> evicted;
        std::lock_guard<std::mutex> guard(mutex_);
        free_.push_back(std::move(native));
        if (free_.size() > kKept) {
            evicted = std::move(free_.front());
            free_.erase(free_.begin());
        }
    }

    Create create_;
    std::mutex mutex_;
    // The oldest release first.
    std::vector<std::unique_ptr<Native>> free_;
};

// What a graphics device's Device::allocate, upload and download do.
class DeviceBuffers {
public:
    virtual ~DeviceBuffers() = default;
    virtual Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) = 0;
    virtual Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) = 0;
    virtual Result<void, std::string> download(const Buffer &buffer, void *bytes, size_t size) = 0;
};

} // namespace vernon::hal

#endif
