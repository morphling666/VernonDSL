#ifndef VERNON_RHI_DEFERRED_ACTION_H
#define VERNON_RHI_DEFERRED_ACTION_H

#include "VernonError.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace vernon::rhi {

struct DeferredAction {
    void *context{};
    std::uint64_t object{};
    void (*function)(void *, std::uint64_t){};
};

[[nodiscard]] VERNON_RHI_CAPI Result<void, RhiError> invokeDeferredAction(const DeferredAction &action,
                                                                          const char *operation) noexcept;

template <typename Actions>
[[nodiscard]] Result<void, RhiError> drainDeferredActionsReverse(Actions &actions, const char *operation) noexcept {
    Option<RhiError> firstError;
    for (auto action = actions.rbegin(); action != actions.rend(); ++action) {
        auto invoked = invokeDeferredAction(*action, operation);
        if (invoked.isErr() && !firstError)
            firstError.emplace(std::move(invoked).error());
        if (invoked.isOk())
            action->function = nullptr;
    }
    actions.erase(std::remove_if(actions.begin(), actions.end(), [](const auto &action) { return !action.function; }),
                  actions.end());
    if (firstError)
        return Result<void, RhiError>{err(firstError.value())};
    return Result<void, RhiError>{ok()};
}

template <typename Rollbacks, typename Cleanups>
[[nodiscard]] Result<void, RhiError> drainDeferredRollbackAndCleanup(Rollbacks &rollbacks,
                                                                     Cleanups &cleanups) noexcept {
    auto rolledBack = drainDeferredActionsReverse(rollbacks, "command_rollback_callback");
    auto cleanedUp = drainDeferredActionsReverse(cleanups, "command_cleanup_callback");
    if (rolledBack.isErr())
        return rolledBack;
    return cleanedUp;
}

template <typename Resources>
[[nodiscard]] Result<void, RhiError> releaseRetainedResourcesReverse(Resources &resources) noexcept {
    Option<RhiError> firstError;
    for (auto resource = resources.rbegin(); resource != resources.rend(); ++resource) {
        if (!resource->active())
            continue;
        auto released = resource->release();
        if (released.isErr() && !firstError)
            firstError.emplace(std::move(released).error());
    }
    if (firstError)
        return Result<void, RhiError>{err(firstError.value())};
    return Result<void, RhiError>{ok()};
}

} // namespace vernon::rhi

#endif
