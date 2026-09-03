#include "dirty_index_set.h"

#include <algorithm>
#include <stdexcept>

namespace vernon::runtime {

DirtyIndexSet::DirtyIndexSet(size_t count) : dirty_(count) {}

std::vector<size_t> DirtyIndexSet::indices() const {
    std::vector<size_t> result;
    for (size_t index = 0; index < dirty_.size(); ++index)
        if (dirty_[index])
            result.push_back(index);
    return result;
}

void DirtyIndexSet::requireIndex(size_t index) const {
    if (index >= dirty_.size())
        throw std::out_of_range("dirty index exceeds its resource");
}

bool DirtyIndexSet::contains(size_t index) const {
    requireIndex(index);
    return dirty_[index];
}

void DirtyIndexSet::add(size_t index) {
    requireIndex(index);
    dirty_[index] = true;
}

void DirtyIndexSet::discard(size_t index) {
    requireIndex(index);
    dirty_[index] = false;
}

void DirtyIndexSet::update(const std::vector<size_t> &indices) {
    for (const size_t index : indices)
        add(index);
}

void DirtyIndexSet::difference(const std::vector<size_t> &indices) {
    for (const size_t index : indices)
        discard(index);
}

void DirtyIndexSet::markAll() { std::fill(dirty_.begin(), dirty_.end(), true); }

void DirtyIndexSet::clear() { std::fill(dirty_.begin(), dirty_.end(), false); }

bool DirtyIndexSet::empty() const {
    return std::none_of(dirty_.begin(), dirty_.end(), [](bool value) { return value; });
}

} // namespace vernon::runtime
