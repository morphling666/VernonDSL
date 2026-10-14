#ifndef VERNON_RHI_STAGING_POOL_H
#define VERNON_RHI_STAGING_POOL_H

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace vernon::rhi {

// The staging chunks one command frame bumps through. The frame holds them until its work completes.
template <typename Chunk> struct StagingArena {
    using Size = decltype(Chunk::capacity);

    std::vector<Chunk> chunks;
    // Offset into the last chunk.
    Size cursor{};
};

// Host-visible staging chunks of one kind that a device's command frames share. Retired frames return their chunks
// here, so staging memory follows the data in flight rather than the number of frames. Free chunks are kept until
// `clear`, so the pool holds the peak amount in flight.
template <typename Chunk> class StagingPool {
public:
    using Size = decltype(Chunk::capacity);
    // Modest because MoltenVK maps host-visible memory to Metal buffers, and virtualized CI devices reject 1 MiB
    // allocations even for uploads of a few bytes.
    static constexpr Size minimumCapacity = 64 * 1024;

    // Bumps `size` bytes aligned to `alignment`, a power of two, out of `arena`. When its last chunk is full the arena
    // takes the smallest free chunk that fits, or one that `create(capacity, chunk)` makes, at least twice as large as
    // the last so that one large recording needs few chunks. Returns null when `create` fails.
    template <typename Create>
    Chunk *acquire(StagingArena<Chunk> &arena, Size size, Size alignment, Size &offset, Create &&create) {
        if (!arena.chunks.empty()) {
            const Size aligned = (arena.cursor + alignment - 1) & ~(alignment - 1);
            const Chunk &last = arena.chunks.back();
            if (aligned <= last.capacity && size <= last.capacity - aligned) {
                offset = aligned;
                arena.cursor = aligned + size;
                return &arena.chunks.back();
            }
        }
        // Chunks own native memory without a destructor, so the arena grows before it takes ownership of one.
        arena.chunks.reserve(arena.chunks.size() + 1);
        auto best = free_.end();
        for (auto candidate = free_.begin(); candidate != free_.end(); ++candidate)
            if (candidate->capacity >= size && (best == free_.end() || candidate->capacity < best->capacity))
                best = candidate;
        Chunk chunk{};
        if (best != free_.end()) {
            chunk = std::move(*best);
            *best = std::move(free_.back());
            free_.pop_back();
        } else {
            Size capacity = arena.chunks.empty() ? minimumCapacity
                                                 : (std::max)(minimumCapacity, doubled(arena.chunks.back().capacity));
            while (capacity < size)
                capacity = doubled(capacity);
            if (!create(capacity, chunk))
                return nullptr;
        }
        arena.chunks.push_back(std::move(chunk));
        offset = 0;
        arena.cursor = size;
        return &arena.chunks.back();
    }

    // The work that read or wrote the chunks of `arena` has completed.
    void retire(StagingArena<Chunk> &arena) {
        free_.reserve(free_.size() + arena.chunks.size());
        for (Chunk &chunk : arena.chunks)
            free_.push_back(std::move(chunk));
        arena.chunks.clear();
        arena.cursor = 0;
    }

    template <typename Destroy> void clear(Destroy &&destroy) {
        for (Chunk &chunk : free_)
            destroy(chunk);
        free_.clear();
    }

private:
    static Size doubled(Size capacity) noexcept {
        return capacity > (std::numeric_limits<Size>::max)() / 2 ? (std::numeric_limits<Size>::max)() : capacity * 2;
    }

    std::vector<Chunk> free_;
};

} // namespace vernon::rhi

#endif
