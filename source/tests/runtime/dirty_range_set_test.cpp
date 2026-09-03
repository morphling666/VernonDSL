#include "runtime/dirty_index_set.h"
#include "runtime/dirty_range_set.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using vernon::runtime::DirtyIndexSet;
using vernon::runtime::DirtyRangeSet;

TEST(DirtyIndexSet, UpdatesAndClearsSelectedIndices) {
    DirtyIndexSet dirty(4);
    dirty.update({0, 2, 3});
    dirty.difference({2});

    EXPECT_EQ(dirty.indices(), (std::vector<size_t>{0, 3}));
    EXPECT_TRUE(dirty.contains(0));
    EXPECT_FALSE(dirty.contains(1));
}

TEST(DirtyIndexSet, RejectsIndicesOutsideResource) {
    DirtyIndexSet dirty(2);
    EXPECT_THROW(dirty.add(2), std::out_of_range);
}

TEST(DirtyRangeSet, CoalescesOverlappingAndAdjacentRanges) {
    DirtyRangeSet dirty(64);
    dirty.mark({{16, 24}, {0, 8}, {8, 20}}, false);

    ASSERT_EQ(dirty.ranges().size(), 1u);
    const std::pair<size_t, size_t> expected{0, 24};
    EXPECT_EQ(dirty.ranges()[0], expected);
}

TEST(DirtyRangeSet, PromotesHalfAllocationToFullUpload) {
    DirtyRangeSet dirty(64);
    dirty.mark({{0, 16}}, true);
    dirty.mark({{32, 48}}, true);

    ASSERT_EQ(dirty.ranges().size(), 1u);
    const std::pair<size_t, size_t> expected{0, 64};
    EXPECT_EQ(dirty.ranges()[0], expected);
}

TEST(DirtyRangeSet, RejectsRangesOutsideAllocation) {
    DirtyRangeSet dirty(16);
    EXPECT_THROW(dirty.mark({{8, 17}}, false), std::invalid_argument);
}

} // namespace
