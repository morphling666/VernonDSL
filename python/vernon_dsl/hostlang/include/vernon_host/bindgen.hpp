/* What a C++ library implements its generated Host bindings with (vernon_dsl.bindgen, level 1).
 *
 * Implementations return vh::Result<T> or vh::Status, and the generated C thunks unwrap them: a failure's status is
 * the C function's result and its message is the library's thread-local last error. Library failures are positive
 * statuses; negative ones belong to Host code (HostStatus), so a status passes through Host functions unchanged.
 * Objects behind handles live in a vh::HandleTable the thunks own.
 */
#ifndef VERNON_HOST_BINDGEN_HPP
#define VERNON_HOST_BINDGEN_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "vernon_host/reflect.h"

namespace vh {

/* A `list[T]` or `dict[K, V]` argument, borrowed for one call: the C++17 stand-in for std::span over a VhSpan. */
template <typename T> class Span {
public:
    Span() = default;
    Span(const T *data, size_t size) : data_(data), size_(size) {}
    explicit Span(VhSpan span) : data_(static_cast<const T *>(span.data)), size_(static_cast<size_t>(span.size)) {}

    const T *data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const T &operator[](size_t index) const { return data_[index]; }
    const T *begin() const { return data_; }
    const T *end() const { return data_ + size_; }

private:
    const T *data_ = nullptr;
    size_t size_ = 0;
};

/* One entry of a dict argument or result, `{K key; V value;}` as C lays it out. */
template <typename K, typename V> struct Entry {
    K key;
    V value;
};

/* Fills an extern's empty owned-result record with `count` items through the allocator the caller passed, as the
 * generated thunks do. Returns 0 or VH_STATUS_OUT_OF_MEMORY. */
inline int32_t assign_result(VhList *out, const VhAllocator *allocator, std::string_view text) {
    return vh_list_assign(out, text.data(), text.size(), 1, 1, allocator);
}

template <typename T> int32_t assign_result(VhList *out, const VhAllocator *allocator, const std::vector<T> &items) {
    static_assert(std::is_trivially_copyable<T>::value, "an owned result holds plain values");
    return vh_list_assign(out, items.data(), items.size(), sizeof(T), alignof(T), allocator);
}

/* Statuses the thunks report themselves; libraries may use these or any other positive status. */
inline constexpr int32_t kFailed = 1;
inline constexpr int32_t kInvalidHandle = 2;
inline constexpr int32_t kOutOfMemory = 3;

struct Error {
    int32_t status = kFailed;
    std::string message;
};

inline Error fail(std::string message, int32_t status = kFailed) { return Error{status, std::move(message)}; }

template <typename T> class [[nodiscard]] Result {
public:
    Result(T value) : state_(std::in_place_index<0>, std::move(value)) {}
    Result(Error error) : state_(std::in_place_index<1>, std::move(error)) {}

    bool ok() const { return state_.index() == 0; }
    T &value() { return std::get<0>(state_); }
    const T &value() const { return std::get<0>(state_); }
    Error &error() { return std::get<1>(state_); }
    const Error &error() const { return std::get<1>(state_); }

private:
    std::variant<T, Error> state_;
};

template <> class [[nodiscard]] Result<void> {
public:
    Result() = default;
    Result(Error error) : error_(std::move(error)) {}

    bool ok() const { return !error_.has_value(); }
    Error &error() { return *error_; }
    const Error &error() const { return *error_; }

private:
    std::optional<Error> error_;
};

using Status = Result<void>;

/* The live objects of one handle type. A handle's low 32 bits index a slot and its high 32 bits are the slot's
 * generation, which taking the object advances, so the bits of a destroyed handle never find the slot's next object.
 * Generations start at 1, so the zero handle is never live. An object of a handle declared `depends_on` another
 * records the bits of the parent it was created with, and take_children removes them together when the parent goes.
 * The table keeps its slots consistent across threads; whether two calls may use one object at once is the library's
 * own rule. */
template <typename T> class HandleTable {
public:
    uint64_t insert(std::unique_ptr<T> object, uint64_t parent = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint32_t index = 0;
        if (free_.empty()) {
            index = static_cast<uint32_t>(slots_.size());
            slots_.emplace_back();
        } else {
            index = free_.back();
            free_.pop_back();
        }
        Slot &slot = slots_[index];
        slot.object = std::move(object);
        slot.parent = parent;
        return (static_cast<uint64_t>(slot.generation) << 32) | index;
    }

    T *find(uint64_t bits) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const Slot *slot = live(bits);
        return slot == nullptr ? nullptr : slot->object.get();
    }

    /* Removes the object, which the caller then owns; null when `bits` is not a live handle. */
    std::unique_ptr<T> take(uint64_t bits) {
        std::lock_guard<std::mutex> lock(mutex_);
        Slot *slot = const_cast<Slot *>(live(bits));
        if (slot == nullptr) {
            return nullptr;
        }
        return release(static_cast<uint32_t>(bits & UINT32_MAX));
    }

    /* Removes every object created with the parent `bits`, with its handle's bits, in slot order. */
    std::vector<std::pair<uint64_t, std::unique_ptr<T>>> take_children(uint64_t parent) {
        return take_if([parent](const Slot &slot) { return slot.parent == parent; });
    }

    /* Removes every object, with its handle's bits, in slot order. */
    std::vector<std::pair<uint64_t, std::unique_ptr<T>>> take_all() {
        return take_if([](const Slot &) { return true; });
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_.size() - free_.size();
    }

private:
    struct Slot {
        std::unique_ptr<T> object;
        uint32_t generation = 1;
        // The bits of the parent handle the object was created with; 0 without one.
        uint64_t parent = 0;
    };

    template <typename Predicate> std::vector<std::pair<uint64_t, std::unique_ptr<T>>> take_if(Predicate predicate) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<uint64_t, std::unique_ptr<T>>> taken;
        for (size_t index = 0; index < slots_.size(); ++index) {
            const Slot &slot = slots_[index];
            if (slot.object != nullptr && predicate(slot)) {
                uint64_t bits = (static_cast<uint64_t>(slot.generation) << 32) | index;
                taken.emplace_back(bits, release(static_cast<uint32_t>(index)));
            }
        }
        return taken;
    }

    std::unique_ptr<T> release(uint32_t index) {
        Slot &slot = slots_[index];
        std::unique_ptr<T> object = std::move(slot.object);
        slot.generation = slot.generation == UINT32_MAX ? 1 : slot.generation + 1;
        slot.parent = 0;
        free_.push_back(index);
        return object;
    }

    const Slot *live(uint64_t bits) const {
        uint64_t index = bits & UINT32_MAX;
        if (index >= slots_.size()) {
            return nullptr;
        }
        const Slot &slot = slots_[index];
        if (slot.object == nullptr || slot.generation != (bits >> 32)) {
            return nullptr;
        }
        return &slot;
    }

    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    std::vector<uint32_t> free_;
};

/* Copies `message` into `buffer` as a NUL-terminated string cut to `capacity`, and returns its full length, as the
 * generated `<library>_last_error` functions do. */
inline uint64_t copy_message(const std::string &message, char *buffer, uint64_t capacity) {
    if (buffer != nullptr && capacity > 0) {
        size_t count = message.size() < capacity - 1 ? message.size() : static_cast<size_t>(capacity - 1);
        std::memcpy(buffer, message.data(), count);
        buffer[count] = '\0';
    }
    return message.size();
}

} // namespace vh

#endif // VERNON_HOST_BINDGEN_HPP
