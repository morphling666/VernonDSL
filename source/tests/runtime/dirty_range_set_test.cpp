#include "runtime/dirty_range_set.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace {

using vernon::runtime::DirtyRangeSet;

TEST(DirtyRangeSet, CoalescesOverlappingAndAdjacentRanges) {
    DirtyRangeSet dirty(64);
    ASSERT_TRUE(dirty.mark({{16, 24}, {0, 8}, {8, 20}}, false).isOk());

    ASSERT_EQ(dirty.ranges().size(), 1u);
    const std::pair<size_t, size_t> expected{0, 24};
    EXPECT_EQ(dirty.ranges()[0], expected);
}

TEST(DirtyRangeSet, PromotesHalfAllocationToFullUpload) {
    DirtyRangeSet dirty(64);
    ASSERT_TRUE(dirty.mark({{0, 16}}, true).isOk());
    ASSERT_TRUE(dirty.mark({{32, 48}}, true).isOk());

    ASSERT_EQ(dirty.ranges().size(), 1u);
    const std::pair<size_t, size_t> expected{0, 64};
    EXPECT_EQ(dirty.ranges()[0], expected);
}

TEST(DirtyRangeSet, RejectsRangesOutsideAllocation) {
    DirtyRangeSet dirty(16);
    auto result = dirty.mark({{8, 17}}, false);
    ASSERT_TRUE(result.isErr());
    EXPECT_EQ(result.error(), vernon::runtime::DirtyRangeError::InvalidRange);
    EXPECT_TRUE(dirty.empty());

    auto promotion = dirty.shouldPromoteFull({{9, 8}});
    ASSERT_TRUE(promotion.isErr());
    EXPECT_EQ(promotion.error(), vernon::runtime::DirtyRangeError::InvalidRange);
}

} // namespace
