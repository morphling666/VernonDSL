#include "runtime/autodiff/runtime_gpu_telemetry.h"

#include "runtime/runtime_state.h"

#include <algorithm>
#include <limits>

namespace vernon::runtime::ad::gpu {
namespace {

bool add(size_t left, size_t right, size_t &result) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

} // namespace

bool retainedValueBytes(const DeviceValues &devices, const HostValues &hosts, size_t &bytes) {
    bytes = 0;
    for (const auto &[name, value] : devices) {
        (void)name;
        if (!add(bytes, value.buffer.size(), bytes))
            return false;
    }
    for (const auto &[name, value] : hosts) {
        (void)name;
        if (!add(bytes, value.bytes.size(), bytes))
            return false;
    }
    return true;
}

std::shared_ptr<AutodiffMemoryReservation> reserveRetainedValues(VernonRuntimeContext &context,
                                                                 const DeviceValues &devices, const HostValues &hosts,
                                                                 size_t &bytes) {
    if (!retainedValueBytes(devices, hosts, bytes) || !context.autodiffMemoryPolicy)
        return {};
    return AutodiffMemoryReservation::reserve(context.autodiffMemoryPolicy, bytes);
}

std::shared_ptr<AutodiffMemoryReservation> reserveApplyMemory(VernonRuntimeContext &context, size_t requested,
                                                              size_t &limit) {
    limit = requested;
    if (!context.autodiffMemoryPolicy)
        return {};
    const HostTapeMemoryUsage usage = context.autodiffMemoryPolicy->usage();
    if (usage.currentBytes >= context.autodiffMemoryPolicy->contextLimit()) {
        limit = 0;
        return {};
    }
    limit = std::min(limit, context.autodiffMemoryPolicy->contextLimit() - usage.currentBytes);
    return AutodiffMemoryReservation::reserve(context.autodiffMemoryPolicy, limit);
}

} // namespace vernon::runtime::ad::gpu
