#include "staging_pool.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace {

struct FakeChunk {
    int id{};
    size_t capacity{};
};

using Pool = vernon::rhi::StagingPool<FakeChunk>;
using Arena = vernon::rhi::StagingArena<FakeChunk>;

class StagingPoolTest : public testing::Test {
protected:
    FakeChunk *acquire(Arena &arena, size_t size, size_t alignment, size_t &offset) {
        return pool_.acquire(arena, size, alignment, offset, [this](size_t capacity, FakeChunk &chunk) {
            if (failCreation_)
                return false;
            chunk = {static_cast<int>(created_.size()), capacity};
            created_.push_back(capacity);
            return true;
        });
    }

    Pool pool_;
    std::vector<size_t> created_;
    bool failCreation_{};
};

TEST_F(StagingPoolTest, BumpsAlignedOffsetsThroughOneChunk) {
    Arena arena;
    size_t offset = 1;
    const FakeChunk *first = acquire(arena, 3, 1, offset);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(offset, 0u);
    const FakeChunk *second = acquire(arena, 4, 16, offset);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(offset, 16u);
    EXPECT_EQ(second->id, first->id);
    EXPECT_EQ(created_, (std::vector<size_t>{Pool::minimumCapacity}));
}

TEST_F(StagingPoolTest, RetiredChunksServeOtherFrames) {
    Arena uploading;
    Arena copying;
    size_t offset = 0;
    ASSERT_NE(acquire(uploading, 4, 4, offset), nullptr);
    pool_.retire(copying);
    pool_.retire(uploading);
    ASSERT_NE(acquire(copying, 4, 4, offset), nullptr);
    EXPECT_EQ(created_.size(), 1u);
    EXPECT_TRUE(uploading.chunks.empty());
}

TEST_F(StagingPoolTest, ChunksHeldByAnUnretiredFrameAreNotShared) {
    Arena first;
    Arena second;
    size_t offset = 0;
    const FakeChunk *held = acquire(first, 4, 4, offset);
    ASSERT_NE(held, nullptr);
    const int heldId = held->id;
    const FakeChunk *other = acquire(second, 4, 4, offset);
    ASSERT_NE(other, nullptr);
    EXPECT_NE(other->id, heldId);
    EXPECT_EQ(created_.size(), 2u);
}

TEST_F(StagingPoolTest, OutgrownArenaDoublesAndFreshArenasTakeTheSmallestFit) {
    Arena large;
    size_t offset = 0;
    ASSERT_NE(acquire(large, Pool::minimumCapacity, 1, offset), nullptr);
    ASSERT_NE(acquire(large, 1, 1, offset), nullptr);
    ASSERT_NE(acquire(large, 5 * Pool::minimumCapacity, 1, offset), nullptr);
    EXPECT_EQ(created_,
              (std::vector<size_t>{Pool::minimumCapacity, 2 * Pool::minimumCapacity, 8 * Pool::minimumCapacity}));
    pool_.retire(large);

    Arena small;
    const FakeChunk *smallest = acquire(small, 16, 1, offset);
    ASSERT_NE(smallest, nullptr);
    EXPECT_EQ(smallest->capacity, Pool::minimumCapacity);
    Arena medium;
    const FakeChunk *fit = acquire(medium, 3 * Pool::minimumCapacity, 1, offset);
    ASSERT_NE(fit, nullptr);
    EXPECT_EQ(fit->capacity, 8 * Pool::minimumCapacity);
    EXPECT_EQ(created_.size(), 3u);
}

TEST_F(StagingPoolTest, FailedCreationLeavesTheArenaUnchanged) {
    Arena arena;
    size_t offset = 0;
    failCreation_ = true;
    EXPECT_EQ(acquire(arena, 4, 4, offset), nullptr);
    EXPECT_TRUE(arena.chunks.empty());
    EXPECT_EQ(arena.cursor, 0u);
}

TEST_F(StagingPoolTest, ClearDestroysOnlyFreeChunks) {
    Arena held;
    Arena retired;
    size_t offset = 0;
    ASSERT_NE(acquire(held, 4, 4, offset), nullptr);
    ASSERT_NE(acquire(retired, 4, 4, offset), nullptr);
    pool_.retire(retired);
    std::vector<int> destroyed;
    pool_.clear([&](FakeChunk &chunk) { destroyed.push_back(chunk.id); });
    EXPECT_EQ(destroyed, (std::vector<int>{1}));
    EXPECT_EQ(held.chunks.size(), 1u);
}

} // namespace
