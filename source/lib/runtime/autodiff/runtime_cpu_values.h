#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_VALUES_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_VALUES_H

#include "runtime_autodiff_internal.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vernon::runtime::ad::cpu {

constexpr size_t kCpuAdGradientDispatchLimit = 512u * 1024u * 1024u;
constexpr size_t kCpuAdInlineInvocationLimit = 16u * 1024u;

bool checkedAddBytes(size_t &total, size_t count, size_t bytes);

struct OwnedAdValue {
    std::string path;
    VernonDataType dtype{};
    std::shared_ptr<std::vector<uint8_t>> bytes;
    std::vector<uint64_t> shape;
    VernonAdValue value{};

    OwnedAdValue() = default;
    explicit OwnedAdValue(const VernonAdValue &source, bool copyBytes = true)
        : path(source.path.data, source.path.size), dtype(source.dtype),
          bytes(std::make_shared<std::vector<uint8_t>>(source.size)) {
        if (source.rank)
            shape.assign(source.shape, source.shape + source.rank);
        if (copyBytes && source.size)
            std::memcpy(bytes->data(), source.data, source.size);
        refresh();
    }
    OwnedAdValue(OwnedAdValue &&other) noexcept
        : path(std::move(other.path)), dtype(other.dtype), bytes(std::move(other.bytes)),
          shape(std::move(other.shape)) {
        refresh();
    }
    OwnedAdValue &operator=(OwnedAdValue &&other) noexcept {
        path = std::move(other.path);
        dtype = other.dtype;
        bytes = std::move(other.bytes);
        shape = std::move(other.shape);
        refresh();
        return *this;
    }
    OwnedAdValue(const OwnedAdValue &) = delete;
    OwnedAdValue &operator=(const OwnedAdValue &) = delete;

    void refresh() {
        value = {sizeof(VernonAdValue),
                 {path.data(), path.size()},
                 dtype,
                 !bytes || bytes->empty() ? nullptr : bytes->data(),
                 bytes ? bytes->size() : 0,
                 static_cast<uint32_t>(shape.size()),
                 shape.empty() ? nullptr : shape.data()};
    }
};

struct OwnedAdValueSet {
    std::vector<OwnedAdValue> storage;
    std::vector<VernonAdValue> views;
    VernonAdValueSet set{};

    OwnedAdValueSet() = default;
    explicit OwnedAdValueSet(const VernonAdValueSet &source, bool copyBytes = true) {
        storage.reserve(source.value_count);
        std::unordered_map<const void *, std::shared_ptr<std::vector<uint8_t>>> sharedBytes;
        for (size_t index = 0; index < source.value_count; ++index) {
            storage.emplace_back(source.values[index], copyBytes);
            if (copyBytes) {
                const VernonAdValue &value = source.values[index];
                const auto [found, inserted] = sharedBytes.emplace(value.data, storage.back().bytes);
                if (!inserted && found->second->size() == value.size)
                    storage.back().bytes = found->second;
            }
        }
        refresh();
    }
    OwnedAdValueSet(OwnedAdValueSet &&other) noexcept : storage(std::move(other.storage)) { refresh(); }
    OwnedAdValueSet &operator=(OwnedAdValueSet &&other) noexcept {
        storage = std::move(other.storage);
        refresh();
        return *this;
    }
    OwnedAdValueSet(const OwnedAdValueSet &) = delete;
    OwnedAdValueSet &operator=(const OwnedAdValueSet &) = delete;

    void refresh() {
        views.clear();
        views.reserve(storage.size());
        for (OwnedAdValue &value : storage) {
            value.refresh();
            views.push_back(value.value);
        }
        set = {sizeof(VernonAdValueSet), views.empty() ? nullptr : views.data(), views.size(), {}};
    }

    size_t retainedBytes() const {
        size_t result = 0;
        std::unordered_set<const void *> counted;
        for (const OwnedAdValue &value : storage)
            if ((value.bytes && counted.insert(value.bytes.get()).second &&
                 !checkedAddBytes(result, value.bytes->capacity(), sizeof(uint8_t))) ||
                !checkedAddBytes(result, value.shape.capacity(), sizeof(uint64_t)))
                return std::numeric_limits<size_t>::max();
        return result;
    }
};

} // namespace vernon::runtime::ad::cpu

#endif
