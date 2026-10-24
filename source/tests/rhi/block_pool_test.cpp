#include "block_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

struct FakeChunk {
    int id{};
    size_t capacity{};
};

using Pool = vernon::rhi::BlockPool<FakeChunk>;

class BlockPoolTest : public testing::Test {
protected:
    FakeChunk *acquire(size_t size, size_t alignment, size_t &offset, uint64_t &key) {
        return pool_.acquire(size, alignment, offset, key, [this](size_t capacity, FakeChunk &chunk) {
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

TEST_F(BlockPoolTest, BlocksOfOneClassShareAChunkAtDistinctAlignedOffsets) {
    size_t first = 1;
    size_t second = 1;
    uint64_t firstKey = 0;
    uint64_t secondKey = 0;
    const FakeChunk *firstChunk = acquire(4, 4, first, firstKey);
    const FakeChunk *secondChunk = acquire(Pool::minimumBlock, 16, second, secondKey);
    ASSERT_NE(firstChunk, nullptr);
    ASSERT_NE(secondChunk, nullptr);
    EXPECT_EQ(firstChunk->id, secondChunk->id);
    EXPECT_NE(first, second);
    EXPECT_NE(firstKey, secondKey);
    EXPECT_EQ(first % Pool::minimumBlock, 0u);
    EXPECT_EQ(second % Pool::minimumBlock, 0u);
    EXPECT_EQ(created_, (std::vector<size_t>{Pool::minimumCapacity}));
}

TEST_F(BlockPoolTest, ReleasedBlocksServeTheirClassWithoutNewChunks) {
    size_t offset = 0;
    uint64_t key = 0;
    for (int repetition = 0; repetition < 1000; ++repetition) {
        ASSERT_NE(acquire(100, 4, offset, key), nullptr);
        pool_.release(key);
    }
    EXPECT_EQ(created_.size(), 1u);
}

TEST_F(BlockPoolTest, AFullClassTakesAnotherChunk) {
    const size_t perChunk = Pool::minimumCapacity / Pool::minimumBlock;
    std::vector<size_t> offsets;
    size_t offset = 0;
    uint64_t key = 0;
    for (size_t block = 0; block < perChunk; ++block) {
        ASSERT_NE(acquire(1, 1, offset, key), nullptr);
        offsets.push_back(offset);
    }
    std::sort(offsets.begin(), offsets.end());
    for (size_t block = 0; block < perChunk; ++block)
        EXPECT_EQ(offsets[block], block * Pool::minimumBlock);
    EXPECT_EQ(created_.size(), 1u);
    const FakeChunk *next = acquire(1, 1, offset, key);
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(next->id, 1);
}

TEST_F(BlockPoolTest, ClassesFollowSizeAndAlignment) {
    size_t offset = 0;
    uint64_t key = 0;
    ASSERT_NE(acquire(Pool::minimumBlock + 1, 1, offset, key), nullptr);
    ASSERT_NE(acquire(1, 2 * Pool::minimumBlock, offset, key), nullptr);
    EXPECT_EQ(offset % (2 * Pool::minimumBlock), 0u);
    ASSERT_NE(acquire(3 * Pool::minimumCapacity, 1, offset, key), nullptr);
    EXPECT_EQ(offset, 0u);
    EXPECT_EQ(created_, (std::vector<size_t>{Pool::minimumCapacity, 4 * Pool::minimumCapacity}));
}

TEST_F(BlockPoolTest, InvalidRequestsAndFailedCreationTakeNothing) {
    size_t offset = 0;
    uint64_t key = 0;
    EXPECT_EQ(acquire(0, 1, offset, key), nullptr);
    EXPECT_EQ(acquire(4, 3, offset, key), nullptr);
    failCreation_ = true;
    EXPECT_EQ(acquire(4, 4, offset, key), nullptr);
    failCreation_ = false;
    ASSERT_NE(acquire(4, 4, offset, key), nullptr);
    EXPECT_EQ(created_.size(), 1u);
}

TEST_F(BlockPoolTest, ClearDestroysEveryChunk) {
    size_t offset = 0;
    uint64_t key = 0;
    ASSERT_NE(acquire(4, 4, offset, key), nullptr);
    ASSERT_NE(acquire(2 * Pool::minimumBlock, 4, offset, key), nullptr);
    pool_.release(key);
    std::vector<int> destroyed;
    pool_.clear([&](FakeChunk &chunk) { destroyed.push_back(chunk.id); });
    EXPECT_EQ(destroyed, (std::vector<int>{0, 1}));
    ASSERT_NE(acquire(4, 4, offset, key), nullptr);
    EXPECT_EQ(created_.size(), 3u);
}

} // namespace
