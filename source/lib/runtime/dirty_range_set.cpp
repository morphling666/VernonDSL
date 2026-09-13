#include "dirty_range_set.h"

#include <algorithm>
#include <limits>

namespace vernon::runtime {
namespace {

constexpr size_t maxDirtyRanges = 4096;

} // namespace

DirtyRangeSet::DirtyRangeSet(size_t byteSize, bool dirty) : byteSize_(byteSize) {
    if (dirty)
        markAll();
}

Result<std::vector<std::pair<size_t, size_t>>, DirtyRangeError>
DirtyRangeSet::coalescedWith(const std::vector<std::pair<size_t, size_t>> &ranges) const {
    std::vector<std::pair<size_t, size_t>> combined = ranges_;
    combined.insert(combined.end(), ranges.begin(), ranges.end());
    std::sort(combined.begin(), combined.end());
    std::vector<std::pair<size_t, size_t>> result;
    for (const auto &[begin, end] : combined) {
        if (begin > end || end > byteSize_)
            return Result<std::vector<std::pair<size_t, size_t>>, DirtyRangeError>{err(DirtyRangeError::InvalidRange)};
        if (begin == end)
            continue;
        if (!result.empty() && begin <= result.back().second)
            result.back().second = std::max(result.back().second, end);
        else
            result.emplace_back(begin, end);
    }
    return Result<std::vector<std::pair<size_t, size_t>>, DirtyRangeError>{ok(std::move(result))};
}

bool DirtyRangeSet::shouldPromote(const std::vector<std::pair<size_t, size_t>> &ranges) const {
    if (ranges.size() > maxDirtyRanges)
        return true;
    size_t dirtyBytes = 0;
    for (const auto &[begin, end] : ranges) {
        const size_t length = end - begin;
        if (dirtyBytes > std::numeric_limits<size_t>::max() - length)
            return true;
        dirtyBytes += length;
    }
    return dirtyBytes >= (byteSize_ + 1) / 2;
}

Result<void, DirtyRangeError> DirtyRangeSet::mark(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                  bool allowFull) {
    if (ranges.empty())
        return Result<void, DirtyRangeError>{ok()};
    auto dirty = coalescedWith(ranges);
    if (dirty.isErr())
        return Result<void, DirtyRangeError>{err(dirty.error())};
    if (allowFull && shouldPromote(dirty.value())) {
        markAll();
        return Result<void, DirtyRangeError>{ok()};
    }
    ranges_ = std::move(dirty).value();
    return Result<void, DirtyRangeError>{ok()};
}

Result<bool, DirtyRangeError>
DirtyRangeSet::shouldPromoteFull(const std::vector<std::pair<size_t, size_t>> &ranges) const {
    auto dirty = coalescedWith(ranges);
    if (dirty.isErr())
        return Result<bool, DirtyRangeError>{err(dirty.error())};
    return Result<bool, DirtyRangeError>{ok(shouldPromote(dirty.value()))};
}

void DirtyRangeSet::markAll() {
    ranges_ =
        byteSize_ ? std::vector<std::pair<size_t, size_t>>{{0, byteSize_}} : std::vector<std::pair<size_t, size_t>>{};
}

} // namespace vernon::runtime
