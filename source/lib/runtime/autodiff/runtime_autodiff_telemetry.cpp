#include "runtime_autodiff_telemetry.h"

#include "runtime/runtime_state.h"
#include "runtime_autodiff_internal.h"

#include <algorithm>
#include <mutex>

vernon::runtime::AutodiffPullbackMemoryUsage
vernon::runtime::autodiffPullbackMemoryUsage(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    const ad::PullbackMemoryUsage usage = pullback->execution->memoryUsage();
    return {usage.logicalResidualBytes, usage.residentBytes, usage.allocatedBytes, usage.retainedAllocationBytes,
            usage.peakTemporaryBytes};
}

vernon::runtime::AutodiffPullbackControlPlaneUsage
vernon::runtime::autodiffPullbackControlPlaneUsage(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    const program_execution::ExecutionControlPlaneUsage usage = pullback->execution->controlPlaneUsage();
    return {usage.submissions,
            usage.waits,
            usage.readbacks,
            usage.atomicPublications,
            usage.temporaryAllocationBytes,
            usage.deviceWaitNanoseconds};
}

vernon::runtime::AutodiffPullbackCheckpointPlan
vernon::runtime::autodiffPullbackCheckpointPlan(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    return pullback->execution->checkpointPlan();
}

std::vector<vernon::runtime::AutodiffPullbackPassTelemetry>
vernon::runtime::autodiffPullbackPassTelemetry(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    return pullback->execution->passTelemetry();
}

uint64_t vernon::runtime::autodiffPullbackPeakRuntimeManagedBytes(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return 0;
    return pullback->execution->peakRuntimeManagedBytes();
}

size_t vernon::runtime::autodiffHostTapeContextLimit(const VernonRuntimeContext *context) {
    if (!context)
        return 0;
    std::lock_guard lock(context->autodiffMemoryPolicyMutex);
    return ad::autodiffMemoryContextLimit(context->autodiffMemoryPolicy);
}

VernonRhiDevice vernon::runtime::autodiffRhiDevice(const VernonRuntimeContext *context) {
    return context ? context->rhiDevice : VernonRhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}
