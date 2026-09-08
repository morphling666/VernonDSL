#ifndef VERNON_RUNTIME_DIRTY_INDEX_SET_H
#define VERNON_RUNTIME_DIRTY_INDEX_SET_H

#include <cstddef>
#include <vector>

namespace vernon::runtime {

class DirtyIndexSet {
public:
    explicit DirtyIndexSet(size_t count);

    std::vector<size_t> indices() const;
    bool contains(size_t index) const;
    void add(size_t index);
    void discard(size_t index);
    void update(const std::vector<size_t> &indices);
    void difference(const std::vector<size_t> &indices);
    void markAll();
    void clear();
    bool empty() const;

private:
    void requireIndex(size_t index) const;

    std::vector<bool> dirty_;
};

} // namespace vernon::runtime

#endif
