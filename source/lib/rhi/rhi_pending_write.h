#ifndef VERNON_RHI_PENDING_WRITE_H
#define VERNON_RHI_PENDING_WRITE_H

#include "backend_dispatch.h"

#include <new>

namespace vernon::rhi {

template <typename Resources>
void recordPendingWrite(Resources &resources, bool &unknownPendingWrites, ResourceKind kind,
                        uint64_t resource) noexcept {
    const bool wasUnknown = unknownPendingWrites;
    unknownPendingWrites = true;
    try {
        resources.insert(typename Resources::value_type{kind, resource});
    } catch (const std::bad_alloc &) {
        // Unknown state is a complete conservative representation: submission
        // must issue the backend's full write barrier instead of relying on the
        // per-resource set.
        return;
    }
    unknownPendingWrites = wasUnknown;
}

} // namespace vernon::rhi

#endif
