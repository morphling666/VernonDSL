#ifndef VERNON_RUNTIME_DIRTY_RANGE_SET_H
#define VERNON_RUNTIME_DIRTY_RANGE_SET_H

#include <cstddef>
#include <utility>
#include <vector>

namespace vernon::runtime {

class DirtyRangeSet {
public:
    explicit DirtyRangeSet(size_t byteSize, bool dirty = false);

    const std::vector<std::pair<size_t, size_t>> &ranges() const { return ranges_; }
    void mark(const std::vector<std::pair<size_t, size_t>> &ranges, bool allowFull);
    bool shouldPromoteFull(const std::vector<std::pair<size_t, size_t>> &ranges) const;
    void markAll();
    void clear() { ranges_.clear(); }
    bool empty() const { return ranges_.empty(); }

private:
    std::vector<std::pair<size_t, size_t>> coalescedWith(const std::vector<std::pair<size_t, size_t>> &ranges) const;
    bool shouldPromote(const std::vector<std::pair<size_t, size_t>> &ranges) const;

    size_t byteSize_;
    std::vector<std::pair<size_t, size_t>> ranges_;
};

} // namespace vernon::runtime

#endif
