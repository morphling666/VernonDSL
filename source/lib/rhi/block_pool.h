#ifndef VERNON_RHI_BLOCK_POOL_H
#define VERNON_RHI_BLOCK_POOL_H

#include "staging_pool.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace vernon::rhi {

// Blocks of native memory that live until their holder releases them, such as the inline values of a persistent
// binding group. A block takes the power-of-two size class that covers its size and alignment, at least `minimumBlock`
// bytes, from a chunk serving only that class, so a released block serves the next block of its class and chunks never
// fragment. Chunks are kept until `clear`, so the pool holds the peak amount in use.
template <typename Chunk> class BlockPool {
public:
    using Size = decltype(Chunk::capacity);
    // A multiple of every uniform-buffer offset alignment the backends require.
    static constexpr Size minimumBlock = 256;
    static constexpr Size minimumCapacity = StagingPool<Chunk>::minimumCapacity;

    // Takes a block of `size` bytes at `offset` in the returned chunk, a multiple of `alignment`, a power of two; `key`
    // releases it. A class without a free block takes a chunk from `create(capacity, chunk)`, which sets the chunk's
    // capacity. Returns null when the request is invalid or `create` fails.
    template <typename Create> Chunk *acquire(Size size, Size alignment, Size &offset, uint64_t &key, Create &&create) {
        if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0)
            return nullptr;
        uint32_t sizeClass = 0;
        Size block = minimumBlock;
        while (block < size || block < alignment) {
            if (block > (std::numeric_limits<Size>::max)() / 2)
                return nullptr;
            block *= 2;
            ++sizeClass;
        }
        if (classes_.size() <= sizeClass)
            classes_.resize(sizeClass + 1);
        Class &blocks = classes_[sizeClass];
        if (blocks.free.empty()) {
            const Size capacity = block > minimumCapacity ? block : minimumCapacity;
            const uint64_t count = capacity / block;
            if (chunks_.size() > (std::numeric_limits<uint32_t>::max)())
                return nullptr;
            // Reserved for every block of the class, so that `release` never allocates.
            blocks.free.reserve(blocks.total + count);
            // Chunks own native memory without a destructor, so the entry exists before it takes ownership of one.
            Entry &entry = chunks_.emplace_back();
            if (!create(capacity, entry.chunk)) {
                chunks_.pop_back();
                return nullptr;
            }
            entry.sizeClass = sizeClass;
            blocks.total += count;
            const uint64_t index = chunks_.size() - 1;
            for (uint64_t slot = count; slot-- > 0;)
                blocks.free.push_back(index << kIndexShift | slot);
        }
        key = blocks.free.back();
        blocks.free.pop_back();
        offset = static_cast<Size>((key & kSlotMask) * block);
        return &chunks_[key >> kIndexShift].chunk;
    }

    // The block of `key` serves later blocks of its class at once.
    void release(uint64_t key) noexcept { classes_[chunks_[key >> kIndexShift].sizeClass].free.push_back(key); }

    template <typename Destroy> void clear(Destroy &&destroy) {
        for (Entry &entry : chunks_)
            destroy(entry.chunk);
        chunks_.clear();
        classes_.clear();
    }

private:
    static constexpr uint32_t kIndexShift = 32;
    static constexpr uint64_t kSlotMask = (uint64_t{1} << kIndexShift) - 1;

    struct Entry {
        Chunk chunk;
        uint32_t sizeClass{};
    };
    struct Class {
        std::vector<uint64_t> free;
        uint64_t total{};
    };

    std::vector<Entry> chunks_;
    std::vector<Class> classes_;
};

} // namespace vernon::rhi

#endif
