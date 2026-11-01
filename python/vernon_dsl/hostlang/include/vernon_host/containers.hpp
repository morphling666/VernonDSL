/* Typed C++ access to the container fields of Host class instances.
 *
 * vh::Str, vh::List<T>, vh::Dict<K, V> and vh::TensorStorage<T, Rank> have the layout of reflect.h's VhList, VhDict
 * and VH_STORAGE records, and generated headers declare container fields with them in C++. Engine code reads and edits
 * an instance's containers as typed C++ while compiled Host code, Python mode and C see the same bytes. They change
 * records only through VernonRuntime's vernonRuntimeHost* functions, as compiled Host code does, so link VernonRuntime
 * to use them.
 *
 * Reads take no allocator. Operations that grow or free take a vh::Allocator, normally the module whose instance holds
 * the field (`vh::Allocator(module)`), because memory goes back to the allocator it came from. They report running out
 * of memory by returning false or null and leave the container unchanged. Records cannot be copied, since two owners
 * would free the memory twice; an instance's containers move to another instance through vh_migrate.
 */
#ifndef VERNON_HOST_CONTAINERS_HPP
#define VERNON_HOST_CONTAINERS_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

#include "VernonRuntime.h"
#include "vernon_host/reflect.h"
#include "vernon_host/tensor_view.h"

#define VH_STR ::vh::Str
#define VH_LIST(T) ::vh::List<T>
#define VH_DICT(K, V) ::vh::Dict<K, V>
#define VH_STORAGE(T, RANK) ::vh::TensorStorage<T, RANK>

namespace vh {

/* The allocator a container operation grows and frees through; null is VernonRuntime's default. */
class Allocator {
public:
    Allocator(std::nullptr_t) {}
    Allocator(const VhAllocator *allocator) : allocator_(allocator) {}
    Allocator(const VhModule *module) : allocator_(vh_get_allocator(module)) {}
    Allocator(const VhModule &module) : allocator_(vh_get_allocator(&module)) {}

    const VhAllocator *get() const { return allocator_; }

private:
    const VhAllocator *allocator_ = nullptr;
};

namespace detail {

inline uint64_t address(const void *pointer) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer)); }

template <typename T> T *pointer(uint64_t address) { return reinterpret_cast<T *>(static_cast<uintptr_t>(address)); }

/* The `{data, size}` prefix of a VhList, which the dict runtime takes as a str key. */
struct StrKey {
    uint64_t data;
    uint64_t size;
};

} // namespace detail

/* A `str` field: UTF-8 bytes, without a terminator. */
class Str {
public:
    Str() = default;
    Str(const Str &) = delete;
    Str &operator=(const Str &) = delete;

    std::string_view view() const {
        return {detail::pointer<const char>(record_.data), static_cast<size_t>(record_.size)};
    }
    operator std::string_view() const { return view(); }
    /* In bytes; Host code's len counts code points. */
    uint64_t size() const { return record_.size; }
    bool empty() const { return record_.size == 0; }

    [[nodiscard]] bool assign(Allocator allocator, std::string_view text) {
        // Growing never happens when `text` lies in this str, whose size is within its capacity, so the bytes stay
        // valid; memmove covers the overlap.
        if (text.size() > record_.capacity &&
            vernonRuntimeHostListReserve(&record_, text.size(), 1, 1, allocator.get()) != 0) {
            return false;
        }
        if (!text.empty()) {
            std::memmove(detail::pointer<char>(record_.data), text.data(), text.size());
        }
        record_.size = text.size();
        return true;
    }
    void release(Allocator allocator) { vernonRuntimeHostListRelease(&record_, 1, 1, allocator.get()); }

    VhList *record() { return &record_; }
    const VhList *record() const { return &record_; }

    friend bool operator==(const Str &left, std::string_view right) { return left.view() == right; }
    friend bool operator==(std::string_view left, const Str &right) { return left == right.view(); }
    friend bool operator==(const Str &left, const Str &right) { return left.view() == right.view(); }
    friend bool operator!=(const Str &left, std::string_view right) { return left.view() != right; }
    friend bool operator!=(std::string_view left, const Str &right) { return left != right.view(); }
    friend bool operator!=(const Str &left, const Str &right) { return left.view() != right.view(); }

private:
    VhList record_{};
};

/* A `list[T]` field. T is a Host value type (a scalar, handle, Struct, Tuple or Tensor, a C array) or a Host class.
 * Elements of a class are read and edited in place; adding, removing or releasing them copies or drops memory each
 * one owns, so those operations take value types only, and an instance's fini releases a list of classes. T may be
 * the class declaring the list, so the class body requires nothing of T. */
template <typename T> class List {
public:
    using value_type = T;

    List() = default;
    List(const List &) = delete;
    List &operator=(const List &) = delete;

    uint64_t size() const { return record_.size; }
    bool empty() const { return record_.size == 0; }
    T *data() { return detail::pointer<T>(record_.data); }
    const T *data() const { return detail::pointer<const T>(record_.data); }
    T *begin() { return data(); }
    T *end() { return data() + record_.size; }
    const T *begin() const { return data(); }
    const T *end() const { return data() + record_.size; }
    /* Unchecked. */
    T &operator[](uint64_t index) { return data()[index]; }
    const T &operator[](uint64_t index) const { return data()[index]; }
    /* Null when `index` is out of range. */
    T *get(uint64_t index) { return index < record_.size ? data() + index : nullptr; }
    const T *get(uint64_t index) const { return index < record_.size ? data() + index : nullptr; }

    [[nodiscard]] bool reserve(Allocator allocator, uint64_t capacity) {
        return capacity <= record_.capacity ||
               vernonRuntimeHostListReserve(&record_, capacity, sizeof(T), alignof(T), allocator.get()) == 0;
    }
    /* `value` may be an element of this list, which growing moves, so it is copied first. */
    [[nodiscard]] bool push_back(Allocator allocator, const T &value) {
        static_assert(std::is_trivially_copyable_v<T>, "only lists of Host value types grow from C++");
        T copy;
        std::memcpy(&copy, &value, sizeof(T));
        if (!reserve(allocator, record_.size + 1)) {
            return false;
        }
        std::memcpy(data() + record_.size, &copy, sizeof(T));
        ++record_.size;
        return true;
    }
    /* Requires a nonempty list. */
    void pop_back() {
        static_assert(std::is_trivially_copyable_v<T>, "only lists of Host value types shrink from C++");
        --record_.size;
    }
    /* Keeps the memory, as Host code's clear does. */
    void clear() {
        static_assert(std::is_trivially_copyable_v<T>, "only lists of Host value types shrink from C++");
        record_.size = 0;
    }
    void release(Allocator allocator) {
        static_assert(std::is_trivially_copyable_v<T>, "an instance's fini releases a list of classes");
        vernonRuntimeHostListRelease(&record_, sizeof(T), alignof(T), allocator.get());
    }

    VhList *record() { return &record_; }
    const VhList *record() const { return &record_; }

private:
    VhList record_{};
};

/* A `dict[K, V]` field, iterated in insertion order. K is an integer, bool, handle or vh::Str; V is a Host value type
 * or a Host class, whose values, like a List's class elements, are edited in place and released by an instance's
 * fini. A str-keyed dict takes and yields its keys as std::string_view. Inserting a new key may move every entry, so
 * it invalidates value pointers and iterators; assigning to an existing key does not. */
template <typename K, typename V> class Dict {
    static constexpr bool kStrKey = std::is_same_v<K, Str>;
    static_assert(kStrKey || (std::is_trivially_copyable_v<K> && !std::is_floating_point_v<K>),
                  "dict keys are integers, bools, handles or vh::Str");

    struct Entry {
        uint64_t hash;
        K key;
        V value;
    };

public:
    /* What the dict's operations take a key as, and what iteration yields it as. */
    using key_argument = std::conditional_t<kStrKey, std::string_view, const K &>;
    using mapped_type = V;

    template <typename Value> struct Item {
        key_argument key;
        Value &value;
    };

    template <typename Value, typename Pointer> class Iterator {
    public:
        Iterator(Pointer current, Pointer end) : current_(current), end_(end) { skip(); }
        Item<Value> operator*() const {
            if constexpr (kStrKey) {
                return {current_->key.view(), current_->value};
            } else {
                return {current_->key, current_->value};
            }
        }
        Iterator &operator++() {
            ++current_;
            skip();
            return *this;
        }
        bool operator==(const Iterator &other) const { return current_ == other.current_; }
        bool operator!=(const Iterator &other) const { return current_ != other.current_; }

    private:
        void skip() {
            while (current_ != end_ && current_->hash == VH_DICT_REMOVED) {
                ++current_;
            }
        }
        Pointer current_;
        Pointer end_;
    };
    using iterator = Iterator<V, Entry *>;
    using const_iterator = Iterator<const V, const Entry *>;

    /* The layout Host code passes the runtime for this dict type; vh_dict_layout gives the same offsets. */
    static VernonHostDictLayout layout() {
        return {sizeof(Entry), alignof(Entry),         offsetof(Entry, key),
                sizeof(K),     offsetof(Entry, value), kStrKey ? VERNON_HOST_DICT_KEY_STR : VERNON_HOST_DICT_KEY_BYTES};
    }

    Dict() = default;
    Dict(const Dict &) = delete;
    Dict &operator=(const Dict &) = delete;

    uint64_t size() const { return record_.size; }
    bool empty() const { return record_.size == 0; }

    iterator begin() { return {entries(), entries() + record_.used}; }
    iterator end() { return {entries() + record_.used, entries() + record_.used}; }
    const_iterator begin() const { return {entries(), entries() + record_.used}; }
    const_iterator end() const { return {entries() + record_.used, entries() + record_.used}; }

    /* The value of `key`, or null. */
    V *find(key_argument key) {
        const int64_t index = index_of(key);
        return index < 0 ? nullptr : &entries()[index].value;
    }
    const V *find(key_argument key) const {
        const int64_t index = index_of(key);
        return index < 0 ? nullptr : &entries()[index].value;
    }
    bool contains(key_argument key) const { return index_of(key) >= 0; }

    /* The value of `key`, inserted zeroed at the end of the order when absent; null when out of memory. */
    [[nodiscard]] V *emplace(Allocator allocator, key_argument key) {
        const Key passed(key);
        const VernonHostDictLayout entry = layout();
        uint64_t index = 0;
        if (vernonRuntimeHostDictInsert(&record_, passed.get(), &entry, allocator.get(), &index) != 0) {
            return nullptr;
        }
        return &entries()[index].value;
    }
    /* `value` may be a value of this dict, which inserting moves, so it is copied first. */
    [[nodiscard]] bool assign(Allocator allocator, key_argument key, const V &value) {
        static_assert(std::is_trivially_copyable_v<V>, "only dicts of Host value types are assigned from C++");
        V copy;
        std::memcpy(&copy, &value, sizeof(V));
        V *slot = emplace(allocator, key);
        if (slot == nullptr) {
            return false;
        }
        std::memcpy(slot, &copy, sizeof(V));
        return true;
    }
    /* Whether `key` was present. */
    bool erase(Allocator allocator, key_argument key) {
        static_assert(std::is_trivially_copyable_v<V>, "only dicts of Host value types shrink from C++");
        const Key passed(key);
        const VernonHostDictLayout entry = layout();
        return vernonRuntimeHostDictRemove(&record_, passed.get(), &entry, allocator.get()) != 0;
    }
    /* Keeps the memory, as Host code's clear does. */
    void clear(Allocator allocator) {
        static_assert(std::is_trivially_copyable_v<V>, "only dicts of Host value types shrink from C++");
        const VernonHostDictLayout entry = layout();
        vernonRuntimeHostDictClear(&record_, &entry, allocator.get());
    }
    void release(Allocator allocator) {
        static_assert(std::is_trivially_copyable_v<V>, "an instance's fini releases a dict of classes");
        const VernonHostDictLayout entry = layout();
        vernonRuntimeHostDictRelease(&record_, &entry, allocator.get());
    }

    VhDict *record() { return &record_; }
    const VhDict *record() const { return &record_; }

private:
    /* A key in the form the runtime reads: a str key's {data, size}, any other key's bytes. It points into itself, so
     * it lives in the calling frame. */
    class Key {
    public:
        explicit Key(key_argument key) {
            if constexpr (kStrKey) {
                text_ = {detail::address(key.data()), key.size()};
                address_ = &text_;
            } else {
                address_ = &key;
            }
        }
        Key(const Key &) = delete;
        Key &operator=(const Key &) = delete;
        const void *get() const { return address_; }

    private:
        detail::StrKey text_{};
        const void *address_ = nullptr;
    };

    int64_t index_of(key_argument key) const {
        const Key passed(key);
        const VernonHostDictLayout entry = layout();
        return vernonRuntimeHostDictFind(&record_, passed.get(), &entry);
    }

    Entry *entries() { return detail::pointer<Entry>(record_.entries); }
    const Entry *entries() const { return detail::pointer<const Entry>(record_.entries); }

    VhDict record_{};
};

namespace detail {

template <typename T> constexpr VernonDataType storage_dtype() {
    if constexpr (std::is_same_v<T, bool>) {
        return VERNON_DATA_BOOL;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return VERNON_DATA_I32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return VERNON_DATA_U32;
    } else if constexpr (std::is_same_v<T, float>) {
        return VERNON_DATA_F32;
    } else {
        static_assert(std::is_same_v<T, double>, "TensorStorage elements are bool, int32_t, uint32_t, float or double");
        return VERNON_DATA_F64;
    }
}

} // namespace detail

/* A `TensorStorage[T, (e0, ...)]` field: dense row-major elements over a shape of Rank axes, which Host code and
 * allocate() replace whole. Replacing or releasing the storage frees the elements, invalidating pointers and views
 * into them. */
template <typename T, uint32_t Rank> class TensorStorage {
    static_assert(Rank >= 1 && Rank <= VERNON_HOST_STORAGE_MAX_RANK, "a TensorStorage has 1 to 8 axes");
    static constexpr VernonDataType kDtype = detail::storage_dtype<T>();

public:
    using value_type = T;
    static constexpr uint32_t rank = Rank;

    TensorStorage() = default;
    TensorStorage(const TensorStorage &) = delete;
    TensorStorage &operator=(const TensorStorage &) = delete;

    const uint64_t *shape() const { return record_.shape; }
    uint64_t extent(uint32_t axis) const { return record_.shape[axis]; }
    /* The element count. */
    uint64_t size() const { return vh_storage_count(&record_, Rank); }
    bool empty() const { return size() == 0; }
    T *data() { return detail::pointer<T>(record_.data); }
    const T *data() const { return detail::pointer<const T>(record_.data); }

    /* Unchecked, one index per axis. */
    template <typename... Index> T &operator()(Index... index) {
        static_assert(sizeof...(Index) == Rank, "a TensorStorage takes one index per axis");
        return data()[offset({uint64_t(index)...})];
    }
    template <typename... Index> const T &operator()(Index... index) const {
        static_assert(sizeof...(Index) == Rank, "a TensorStorage takes one index per axis");
        return data()[offset({uint64_t(index)...})];
    }
    /* Null when an index is out of range. */
    T *get(const uint64_t (&index)[Rank]) { return in_range(index) ? data() + offset(index) : nullptr; }
    const T *get(const uint64_t (&index)[Rank]) const { return in_range(index) ? data() + offset(index) : nullptr; }

    /* Replaces the elements with `shape`'s, zeroed, as `zeros(dtype=T, shape=shape)` assigned in Host code does; on
     * failure, a negative extent or no memory, the storage is unchanged. */
    [[nodiscard]] bool allocate(Allocator allocator, const int64_t (&shape)[Rank]) {
        Record created{};
        if (vernonRuntimeHostStorageAllocate(&created, Rank, shape, sizeof(T), alignof(T), allocator.get()) != 0) {
            return false;
        }
        release(allocator);
        record_ = created;
        return true;
    }
    void fill(const T &value) { vernonRuntimeHostStorageFill(&record_, Rank, &value, sizeof(T)); }
    void release(Allocator allocator) {
        vernonRuntimeHostStorageRelease(&record_, Rank, sizeof(T), alignof(T), allocator.get());
    }

    /* The elements as a VernonTensorView for a Vernon program; it points into `byte_strides`, which must outlive it. */
    VernonTensorView view(int64_t (&byte_strides)[Rank], VernonValueAccess access = VERNON_ACCESS_READ_WRITE) const {
        return vh_storage_tensor_view(&record_, Rank, kDtype, byte_strides, access);
    }

    void *record() { return &record_; }
    const void *record() const { return &record_; }

private:
    struct Record {
        uint64_t data;
        uint64_t shape[Rank];
    };

    uint64_t offset(const uint64_t (&index)[Rank]) const {
        uint64_t flat = 0;
        for (uint32_t axis = 0; axis < Rank; ++axis) {
            flat = flat * record_.shape[axis] + index[axis];
        }
        return flat;
    }
    bool in_range(const uint64_t (&index)[Rank]) const {
        for (uint32_t axis = 0; axis < Rank; ++axis) {
            if (index[axis] >= record_.shape[axis]) {
                return false;
            }
        }
        return true;
    }

    Record record_{};
};

static_assert(sizeof(Str) == sizeof(VhList) && alignof(Str) == alignof(VhList) && std::is_standard_layout_v<Str>);
static_assert(sizeof(TensorStorage<float, 2>) == 3 * sizeof(uint64_t) &&
              std::is_standard_layout_v<TensorStorage<float, 2>>);
static_assert(sizeof(List<float>) == sizeof(VhList) && std::is_standard_layout_v<List<float>>);
static_assert(sizeof(Dict<int32_t, float>) == sizeof(VhDict) && std::is_standard_layout_v<Dict<int32_t, float>>);

} // namespace vh

#endif /* VERNON_HOST_CONTAINERS_HPP */
