#include "runtime/dirty_range_set.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using vernon::runtime::DirtyRangeSet;

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
