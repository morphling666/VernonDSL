#ifndef VERNON_RUNTIME_STORED_GRAPHICS_SCOPES_H
#define VERNON_RUNTIME_STORED_GRAPHICS_SCOPES_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vernon::runtime {

// The planned and encoded commands of one rendering scope of an invocation; see stored_plans.md section 12.
struct StoredGraphicsScope;

// The stored scopes of one Program instance, found by the Node that opens them. Their commands are immutable once
// published, so concurrent invocations of the instance share them; a replay rewrites only their updatable groups.
class StoredGraphicsScopes {
public:
    static constexpr size_t kScopesPerNode = 4;

    std::vector<std::shared_ptr<const StoredGraphicsScope>> find(uint64_t node) const {
        const std::lock_guard lock(mutex_);
        const auto found = scopes_.find(node);
        return found == scopes_.end() ? std::vector<std::shared_ptr<const StoredGraphicsScope>>{} : found->second;
    }

    // Drops the scopes of `node` that `drop` selects given `scope`, then the least recently published beyond
    // kScopesPerNode.
    template <typename Drop> void publish(uint64_t node, std::shared_ptr<const StoredGraphicsScope> scope, Drop drop) {
        const std::lock_guard lock(mutex_);
        std::vector<std::shared_ptr<const StoredGraphicsScope>> &scopes = scopes_[node];
        scopes.erase(std::remove_if(scopes.begin(), scopes.end(),
                                    [&](const std::shared_ptr<const StoredGraphicsScope> &stored) {
                                        return drop(*stored, *scope);
                                    }),
                     scopes.end());
        scopes.insert(scopes.begin(), std::move(scope));
        if (scopes.size() > kScopesPerNode)
            scopes.resize(kScopesPerNode);
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<const StoredGraphicsScope>>> scopes_;
};

} // namespace vernon::runtime

#endif
