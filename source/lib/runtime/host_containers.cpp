#include "VernonRuntime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// The records of `vernon_host/reflect.h`, which the packaged runtime sources do not include;
// runtime_c_api_test.cpp checks that both declarations agree.
struct VhList {
    uint64_t data;
    uint64_t size;
    uint64_t capacity;
};

struct VhDict {
    uint64_t entries;
    uint64_t size;
    uint64_t used;
    uint64_t capacity;
    uint64_t slots;
    uint64_t slot_count;
};

struct VhAllocator {
    uint32_t struct_size;
    void *user;
    void *(*allocate)(void *user, uint64_t size, uint64_t alignment);
    void (*release)(void *user, void *pointer, uint64_t size, uint64_t alignment);
};

namespace {

constexpr int32_t kOutOfMemory = -4;
// The hash of a removed entry, which no live key hashes to, and the slot of a removed entry. A slot is otherwise 0,
// empty, or one more than the index of its entry.
constexpr uint64_t kRemoved = std::numeric_limits<uint64_t>::max();

// Host value alignments never exceed 8, which malloc satisfies on every target.
void *allocate(const VhAllocator *allocator, uint64_t size, uint64_t alignment) {
    if (allocator != nullptr)
        return allocator->allocate(allocator->user, size, alignment);
    if (alignment > alignof(std::max_align_t) || size > std::numeric_limits<size_t>::max())
        return nullptr;
    return std::malloc(static_cast<size_t>(size));
}

void release(const VhAllocator *allocator, void *pointer, uint64_t size, uint64_t alignment) {
    if (allocator != nullptr)
        allocator->release(allocator->user, pointer, size, alignment);
    else
        std::free(pointer);
}

void *address(uint64_t data) { return reinterpret_cast<void *>(static_cast<uintptr_t>(data)); }

void *allocateDefault(void * /*user*/, uint64_t size, uint64_t alignment) { return allocate(nullptr, size, alignment); }

void releaseDefault(void * /*user*/, void *pointer, uint64_t size, uint64_t alignment) {
    release(nullptr, pointer, size, alignment);
}

constexpr VhAllocator kDefaultAllocator{sizeof(VhAllocator), nullptr, allocateDefault, releaseDefault};

} // namespace

const VhAllocator *vernonRuntimeHostDefaultAllocator() { return &kDefaultAllocator; }

int32_t vernonRuntimeHostListReserve(VhList *list, uint64_t capacity, uint64_t element_size, uint64_t alignment,
                                     const VhAllocator *allocator) {
    if (capacity <= list->capacity)
        return 0;
    uint64_t grown = list->capacity > std::numeric_limits<uint64_t>::max() / 2 ? capacity : list->capacity * 2;
    if (grown < capacity)
        grown = capacity;
    if (element_size != 0 && grown > std::numeric_limits<uint64_t>::max() / element_size)
        return kOutOfMemory;
    void *data = allocate(allocator, grown * element_size, alignment);
    if (data == nullptr)
        return kOutOfMemory;
    if (list->capacity != 0) {
        std::memcpy(data, address(list->data), static_cast<size_t>(list->size * element_size));
        release(allocator, address(list->data), list->capacity * element_size, alignment);
    }
    list->data = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data));
    list->capacity = grown;
    return 0;
}

void vernonRuntimeHostListRelease(VhList *list, uint64_t element_size, uint64_t alignment,
                                  const VhAllocator *allocator) {
    if (list->capacity != 0)
        release(allocator, address(list->data), list->capacity * element_size, alignment);
    *list = VhList{};
}

namespace {

struct Key {
    const unsigned char *data;
    uint64_t size;
};

Key keyAt(const void *key, const VernonHostDictLayout *layout) {
    if (layout->key_kind == VERNON_HOST_DICT_KEY_STR) {
        const auto *text = static_cast<const uint64_t *>(key);
        return {static_cast<const unsigned char *>(address(text[0])), text[1]};
    }
    return {static_cast<const unsigned char *>(key), layout->key_size};
}

// FNV-1a, then the murmur3 finalizer, which spreads FNV's weak low bits for the short keys dicts mostly hold.
uint64_t hashOf(Key key) {
    uint64_t hash = 0xCBF29CE484222325ull;
    for (uint64_t index = 0; index < key.size; ++index)
        hash = (hash ^ key.data[index]) * 0x100000001B3ull;
    hash ^= hash >> 33;
    hash *= 0xFF51AFD7ED558CCDull;
    hash ^= hash >> 33;
    hash *= 0xC4CEB9FE1A85EC53ull;
    hash ^= hash >> 33;
    return hash == kRemoved ? kRemoved - 1 : hash;
}

bool sameKey(Key left, Key right) {
    return left.size == right.size && (left.size == 0 || std::memcmp(left.data, right.data, left.size) == 0);
}

unsigned char *entryAt(const VhDict *dict, const VernonHostDictLayout *layout, uint64_t index) {
    return static_cast<unsigned char *>(address(dict->entries)) + index * layout->entry_size;
}

uint64_t *slotsOf(const VhDict *dict) { return static_cast<uint64_t *>(address(dict->slots)); }

uint64_t entryHash(const unsigned char *entry) {
    uint64_t hash;
    std::memcpy(&hash, entry, sizeof hash);
    return hash;
}

// The slot holding `key`, or the empty slot ending its probe sequence. The table is never more than half full of
// live and removed slots, so the sequence ends.
uint64_t *probe(const VhDict *dict, const VernonHostDictLayout *layout, Key key, uint64_t hash, bool *found) {
    uint64_t *slots = slotsOf(dict);
    const uint64_t mask = dict->slot_count - 1;
    for (uint64_t index = hash & mask;; index = (index + 1) & mask) {
        const uint64_t slot = slots[index];
        if (slot == 0) {
            *found = false;
            return &slots[index];
        }
        if (slot == kRemoved)
            continue;
        const unsigned char *entry = entryAt(dict, layout, slot - 1);
        if (entryHash(entry) == hash && sameKey(keyAt(entry + layout->key_offset, layout), key)) {
            *found = true;
            return &slots[index];
        }
    }
}

void releaseKey(unsigned char *entry, const VernonHostDictLayout *layout, const VhAllocator *allocator) {
    if (layout->key_kind != VERNON_HOST_DICT_KEY_STR)
        return;
    auto *text = reinterpret_cast<VhList *>(entry + layout->key_offset);
    if (text->capacity != 0)
        release(allocator, address(text->data), text->capacity, 1);
    *text = VhList{};
}

// Moves the live entries, in order, into new memory of `capacity` entries, dropping removed ones and rebuilding
// the slots. On failure the dict is unchanged.
int32_t rebuild(VhDict *dict, const VernonHostDictLayout *layout, const VhAllocator *allocator, uint64_t capacity) {
    uint64_t slot_count = 8;
    while (slot_count < capacity * 2)
        slot_count *= 2;
    if (capacity > std::numeric_limits<uint64_t>::max() / layout->entry_size ||
        slot_count > std::numeric_limits<uint64_t>::max() / sizeof(uint64_t))
        return kOutOfMemory;
    void *entries = allocate(allocator, capacity * layout->entry_size, layout->entry_alignment);
    if (entries == nullptr)
        return kOutOfMemory;
    void *slots = allocate(allocator, slot_count * sizeof(uint64_t), alignof(uint64_t));
    if (slots == nullptr) {
        release(allocator, entries, capacity * layout->entry_size, layout->entry_alignment);
        return kOutOfMemory;
    }
    std::memset(slots, 0, static_cast<size_t>(slot_count * sizeof(uint64_t)));
    VhDict built{static_cast<uint64_t>(reinterpret_cast<uintptr_t>(entries)), 0,         0, capacity,
                 static_cast<uint64_t>(reinterpret_cast<uintptr_t>(slots)),   slot_count};
    for (uint64_t index = 0; index < dict->used; ++index) {
        const unsigned char *entry = entryAt(dict, layout, index);
        const uint64_t hash = entryHash(entry);
        if (hash == kRemoved)
            continue;
        std::memcpy(entryAt(&built, layout, built.used), entry, static_cast<size_t>(layout->entry_size));
        bool found = false;
        *probe(&built, layout, keyAt(entry + layout->key_offset, layout), hash, &found) = ++built.used;
    }
    built.size = built.used;
    if (dict->capacity != 0) {
        release(allocator, address(dict->entries), dict->capacity * layout->entry_size, layout->entry_alignment);
        release(allocator, address(dict->slots), dict->slot_count * sizeof(uint64_t), alignof(uint64_t));
    }
    *dict = built;
    return 0;
}

} // namespace

int64_t vernonRuntimeHostDictFind(const VhDict *dict, const void *key, const VernonHostDictLayout *layout) {
    if (dict->size == 0)
        return -1;
    const Key wanted = keyAt(key, layout);
    bool found = false;
    const uint64_t *slot = probe(dict, layout, wanted, hashOf(wanted), &found);
    return found ? static_cast<int64_t>(*slot - 1) : -1;
}

int32_t vernonRuntimeHostDictInsert(VhDict *dict, const void *key, const VernonHostDictLayout *layout,
                                    const VhAllocator *allocator, uint64_t *entry) {
    const Key wanted = keyAt(key, layout);
    const uint64_t hash = hashOf(wanted);
    bool found = false;
    if (dict->slot_count != 0) {
        const uint64_t *slot = probe(dict, layout, wanted, hash, &found);
        if (found) {
            *entry = *slot - 1;
            return 0;
        }
    }
    if (dict->used == dict->capacity) {
        // Mostly removed entries compact in place; otherwise the dict doubles.
        const uint64_t capacity =
            dict->size + 1 > dict->capacity / 2 ? (dict->capacity < 4 ? 4 : dict->capacity * 2) : dict->capacity;
        if (rebuild(dict, layout, allocator, capacity) != 0)
            return kOutOfMemory;
    }
    void *text = nullptr;
    if (layout->key_kind == VERNON_HOST_DICT_KEY_STR && wanted.size != 0) {
        text = allocate(allocator, wanted.size, 1);
        if (text == nullptr)
            return kOutOfMemory;
        std::memcpy(text, wanted.data, static_cast<size_t>(wanted.size));
    }
    unsigned char *written = entryAt(dict, layout, dict->used);
    std::memset(written, 0, static_cast<size_t>(layout->entry_size));
    std::memcpy(written, &hash, sizeof hash);
    if (layout->key_kind == VERNON_HOST_DICT_KEY_STR) {
        const VhList owned{static_cast<uint64_t>(reinterpret_cast<uintptr_t>(text)), wanted.size, wanted.size};
        std::memcpy(written + layout->key_offset, &owned, sizeof owned);
    } else {
        std::memcpy(written + layout->key_offset, wanted.data, static_cast<size_t>(wanted.size));
    }
    *probe(dict, layout, wanted, hash, &found) = dict->used + 1;
    *entry = dict->used++;
    ++dict->size;
    return 0;
}

int32_t vernonRuntimeHostDictRemove(VhDict *dict, const void *key, const VernonHostDictLayout *layout,
                                    const VhAllocator *allocator) {
    if (dict->size == 0)
        return 0;
    const Key wanted = keyAt(key, layout);
    bool found = false;
    uint64_t *slot = probe(dict, layout, wanted, hashOf(wanted), &found);
    if (!found)
        return 0;
    unsigned char *entry = entryAt(dict, layout, *slot - 1);
    releaseKey(entry, layout, allocator);
    std::memcpy(entry, &kRemoved, sizeof kRemoved);
    *slot = kRemoved;
    if (--dict->size == 0) {
        std::memset(slotsOf(dict), 0, static_cast<size_t>(dict->slot_count * sizeof(uint64_t)));
        dict->used = 0;
    }
    return 1;
}

void vernonRuntimeHostDictClear(VhDict *dict, const VernonHostDictLayout *layout, const VhAllocator *allocator) {
    for (uint64_t index = 0; index < dict->used; ++index) {
        unsigned char *entry = entryAt(dict, layout, index);
        if (entryHash(entry) != kRemoved)
            releaseKey(entry, layout, allocator);
    }
    if (dict->slot_count != 0)
        std::memset(slotsOf(dict), 0, static_cast<size_t>(dict->slot_count * sizeof(uint64_t)));
    dict->size = 0;
    dict->used = 0;
}

void vernonRuntimeHostDictRelease(VhDict *dict, const VernonHostDictLayout *layout, const VhAllocator *allocator) {
    vernonRuntimeHostDictClear(dict, layout, allocator);
    if (dict->capacity != 0) {
        release(allocator, address(dict->entries), dict->capacity * layout->entry_size, layout->entry_alignment);
        release(allocator, address(dict->slots), dict->slot_count * sizeof(uint64_t), alignof(uint64_t));
    }
    *dict = VhDict{};
}

namespace {

uint64_t codePoints(const char *data, uint64_t size) {
    uint64_t count = 0;
    for (uint64_t index = 0; index < size; ++index)
        count += (static_cast<unsigned char>(data[index]) & 0xC0) != 0x80;
    return count;
}

// The UTF-8 bytes of a fill code point, at most 4.
uint32_t encodeFill(uint32_t code, char *out) {
    if (code == 0)
        code = ' ';
    if (code < 0x80) {
        out[0] = static_cast<char>(code);
        return 1;
    }
    if (code < 0x800) {
        out[0] = static_cast<char>(0xC0 | (code >> 6));
        out[1] = static_cast<char>(0x80 | (code & 0x3F));
        return 2;
    }
    if (code < 0x10000) {
        out[0] = static_cast<char>(0xE0 | (code >> 12));
        out[1] = static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        out[2] = static_cast<char>(0x80 | (code & 0x3F));
        return 3;
    }
    out[0] = static_cast<char>(0xF0 | (code >> 18));
    out[1] = static_cast<char>(0x80 | ((code >> 12) & 0x3F));
    out[2] = static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    out[3] = static_cast<char>(0x80 | (code & 0x3F));
    return 4;
}

// Appends `sign` (0 for none) and `body`, padded to the spec's width as Python's format() pads. `body` may lie in
// `str`, so its offset is kept across growth.
int32_t appendPadded(VhList *str, char sign, const char *body, uint64_t size, const VernonHostFormatSpec *spec,
                     uint8_t default_align, const VhAllocator *allocator) {
    const uint64_t length = codePoints(body, size) + (sign != 0);
    const uint64_t padding = spec != nullptr && spec->width > length ? spec->width - length : 0;
    char fill[4];
    const uint32_t fill_size = padding != 0 ? encodeFill(spec->fill, fill) : 0;
    const uint64_t total = size + (sign != 0) + padding * fill_size;
    const char *base = static_cast<const char *>(address(str->data));
    const bool inside = str->capacity != 0 && body >= base && body < base + str->capacity;
    const uint64_t offset = inside ? static_cast<uint64_t>(body - base) : 0;
    if (vernonRuntimeHostListReserve(str, str->size + total, 1, 1, allocator) != 0)
        return kOutOfMemory;
    if (inside)
        body = static_cast<const char *>(address(str->data)) + offset;
    const uint8_t align = spec != nullptr && spec->align != 0 ? spec->align : default_align;
    const uint64_t before = align == '<' ? 0 : align == '^' ? padding / 2 : padding;
    char *out = static_cast<char *>(address(str->data)) + str->size;
    auto pad = [&](uint64_t count) {
        for (uint64_t index = 0; index < count; ++index, out += fill_size)
            std::memcpy(out, fill, fill_size);
    };
    if (align == '=') {
        if (sign != 0)
            *out++ = sign;
        pad(before);
    } else {
        pad(before);
        if (sign != 0)
            *out++ = sign;
    }
    if (size != 0)
        std::memmove(out, body, static_cast<size_t>(size));
    out += size;
    pad(padding - before);
    str->size += total;
    return 0;
}

char signOf(bool negative, const VernonHostFormatSpec *spec) {
    if (negative)
        return '-';
    if (spec != nullptr && (spec->sign == '+' || spec->sign == ' '))
        return static_cast<char>(spec->sign);
    return 0;
}

// Python's float repr of a finite, nonnegative double into `out`, which holds at least 32 bytes: its shortest
// round-tripping digits, positional when the decimal exponent is in [-4, 16), with ".0" on integers, and
// scientific otherwise.
int reprFloat(double value, char *out) {
    char scientific[40];
    for (int digits = 1; digits <= 17; ++digits) {
        std::snprintf(scientific, sizeof scientific, "%.*e", digits - 1, value);
        if (std::strtod(scientific, nullptr) == value)
            break;
    }
    char mantissa[24];
    int count = 0;
    const char *cursor = scientific;
    for (; *cursor != 'e'; ++cursor) {
        if (*cursor != '.')
            mantissa[count++] = *cursor;
    }
    const int exponent = std::atoi(cursor + 1);
    while (count > 1 && mantissa[count - 1] == '0')
        --count;
    int size = 0;
    auto put = [&](const char *text, int length) {
        std::memcpy(out + size, text, static_cast<size_t>(length));
        size += length;
    };
    auto zeros = [&](int length) {
        for (int index = 0; index < length; ++index)
            out[size++] = '0';
    };
    if (exponent < -4 || exponent >= 16) {
        put(mantissa, 1);
        if (count > 1) {
            put(".", 1);
            put(mantissa + 1, count - 1);
        }
        char suffix[8];
        put(suffix, std::snprintf(suffix, sizeof suffix, "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent)));
    } else if (exponent < 0) {
        put("0.", 2);
        zeros(-exponent - 1);
        put(mantissa, count);
    } else if (exponent + 1 >= count) {
        put(mantissa, count);
        zeros(exponent + 1 - count);
        put(".0", 2);
    } else {
        put(mantissa, exponent + 1);
        put(".", 1);
        put(mantissa + exponent + 1, count - exponent - 1);
    }
    return size;
}

} // namespace

int32_t vernonRuntimeHostStrAppend(VhList *str, const char *data, uint64_t size, const VernonHostFormatSpec *spec,
                                   const VhAllocator *allocator) {
    return appendPadded(str, 0, data, size, spec, '<', allocator);
}

int32_t vernonRuntimeHostStrAppendInteger(VhList *str, uint64_t bits, int32_t is_signed,
                                          const VernonHostFormatSpec *spec, const VhAllocator *allocator) {
    const bool negative = is_signed != 0 && static_cast<int64_t>(bits) < 0;
    uint64_t magnitude = negative ? 0 - bits : bits;
    const uint8_t type = spec != nullptr ? spec->type : 0;
    const unsigned base = type == 'x' || type == 'X' ? 16 : type == 'o' ? 8 : type == 'b' ? 2 : 10;
    const char *digits = type == 'X' ? "0123456789ABCDEF" : "0123456789abcdef";
    char text[64];
    int start = 64;
    do {
        text[--start] = digits[magnitude % base];
        magnitude /= base;
    } while (magnitude != 0);
    return appendPadded(str, signOf(negative, spec), text + start, static_cast<uint64_t>(64 - start), spec, '>',
                        allocator);
}

int32_t vernonRuntimeHostStrAppendFloat(VhList *str, double value, const VernonHostFormatSpec *spec,
                                        const VhAllocator *allocator) {
    const uint8_t type = spec != nullptr ? spec->type : 0;
    const bool negative = std::signbit(value) && !std::isnan(value);
    const double magnitude = std::fabs(value);
    char text[512];
    int size = 0;
    if (std::isnan(value)) {
        size = std::snprintf(text, sizeof text, "%s", type == 'E' ? "NAN" : "nan");
    } else if (std::isinf(value)) {
        size = std::snprintf(text, sizeof text, "%s", type == 'E' ? "INF" : "inf");
    } else if (type == 0) {
        size = reprFloat(magnitude, text);
    } else {
        const int precision = spec->precision < 0 ? 6 : spec->precision > 64 ? 64 : spec->precision;
        const char *format = type == 'f' ? "%.*f" : type == 'e' ? "%.*e" : "%.*E";
        size = std::snprintf(text, sizeof text, format, precision, magnitude);
    }
    if (size < 0)
        return kOutOfMemory;
    return appendPadded(str, signOf(negative, spec), text, static_cast<uint64_t>(size), spec, '>', allocator);
}

namespace {

constexpr int32_t kInvalidShape = -7;

// The bytes of a storage's elements, or false when they do not fit in 64 bits.
bool storageBytes(const uint64_t *shape, uint32_t rank, uint64_t element_size, uint64_t *bytes) {
    uint64_t total = element_size;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] != 0 && total > std::numeric_limits<uint64_t>::max() / shape[axis])
            return false;
        total *= shape[axis];
    }
    *bytes = total;
    return true;
}

} // namespace

int32_t vernonRuntimeHostStorageAllocate(void *storage, uint32_t rank, const int64_t *shape, uint64_t element_size,
                                         uint64_t alignment, const VhAllocator *allocator) {
    auto *record = static_cast<uint64_t *>(storage);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] < 0)
            return kInvalidShape;
    }
    uint64_t extents[VERNON_HOST_STORAGE_MAX_RANK];
    if (rank > VERNON_HOST_STORAGE_MAX_RANK)
        return kInvalidShape;
    for (uint32_t axis = 0; axis < rank; ++axis)
        extents[axis] = static_cast<uint64_t>(shape[axis]);
    uint64_t bytes = 0;
    if (!storageBytes(extents, rank, element_size, &bytes))
        return kOutOfMemory;
    void *data = nullptr;
    if (bytes != 0) {
        data = allocate(allocator, bytes, alignment);
        if (data == nullptr)
            return kOutOfMemory;
        std::memset(data, 0, static_cast<size_t>(bytes));
    }
    record[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data));
    std::memcpy(record + 1, extents, rank * sizeof(uint64_t));
    return 0;
}

void vernonRuntimeHostStorageRelease(void *storage, uint32_t rank, uint64_t element_size, uint64_t alignment,
                                     const VhAllocator *allocator) {
    auto *record = static_cast<uint64_t *>(storage);
    uint64_t bytes = 0;
    if (record[0] != 0 && storageBytes(record + 1, rank, element_size, &bytes))
        release(allocator, address(record[0]), bytes, alignment);
    std::memset(record, 0, (rank + 1) * sizeof(uint64_t));
}

void vernonRuntimeHostStorageFill(void *storage, uint32_t rank, const void *value, uint64_t element_size) {
    const auto *record = static_cast<const uint64_t *>(storage);
    uint64_t bytes = 0;
    if (record[0] == 0 || !storageBytes(record + 1, rank, element_size, &bytes))
        return;
    auto *data = static_cast<unsigned char *>(address(record[0]));
    for (uint64_t offset = 0; offset < bytes; offset += element_size)
        std::memcpy(data + offset, value, static_cast<size_t>(element_size));
}
