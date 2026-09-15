#include "runtime_python_bridge.h"

#include "autodiff/runtime_autodiff_telemetry.h"
#include "program_boundary_view.h"
#include "program_execution/failure_injection.h"

#include <limits>
#include <string>
#include <vector>

namespace {

VernonStringView viewOf(const std::string &value) { return {value.data(), value.size()}; }

} // namespace

extern "C" {

VernonStatus vernonRuntimePrivateGetAutodiffMemoryUsage(const VernonPullback *pullback,
                                                        VernonRuntimePrivateAutodiffMemoryUsage *output) {
    if (!output || output->struct_size < sizeof(*output))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const vernon::runtime::AutodiffPullbackMemoryUsage usage = vernon::runtime::autodiffPullbackMemoryUsage(pullback);
    *output = {sizeof(*output),      usage.logicalResidualBytes,    usage.residentBytes,
               usage.allocatedBytes, usage.retainedAllocationBytes, usage.peakTemporaryBytes};
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimePrivateGetAutodiffControlPlaneUsage(const VernonPullback *pullback,
                                                              VernonRuntimePrivateAutodiffControlPlaneUsage *output) {
    if (!output || output->struct_size < sizeof(*output))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const vernon::runtime::AutodiffPullbackControlPlaneUsage usage =
        vernon::runtime::autodiffPullbackControlPlaneUsage(pullback);
    *output = {sizeof(*output),
               usage.submissions,
               usage.waits,
               usage.readbacks,
               usage.atomicPublications,
               usage.temporaryAllocationBytes,
               usage.deviceWaitNanoseconds};
    return VERNON_STATUS_OK;
}

uint64_t vernonRuntimePrivateGetAutodiffHostTapeContextLimit(const VernonRuntimeContext *context) {
    return vernon::runtime::autodiffHostTapeContextLimit(context);
}

VernonRhiDevice vernonRuntimePrivateGetAutodiffRhiDevice(const VernonRuntimeContext *context) {
    return vernon::runtime::autodiffRhiDevice(context);
}

uint64_t vernonRuntimePrivateGetAutodiffPeakRuntimeManagedBytes(const VernonPullback *pullback) {
    return vernon::runtime::autodiffPullbackPeakRuntimeManagedBytes(pullback);
}

VernonStatus vernonRuntimePrivateVisitAutodiffCheckpointPlan(const VernonPullback *pullback,
                                                             VernonRuntimePrivateAutodiffCheckpointVisitor visitor,
                                                             void *userData) {
    if (!visitor)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        const vernon::runtime::AutodiffPullbackCheckpointPlan plan =
            vernon::runtime::autodiffPullbackCheckpointPlan(pullback);
        const VernonRuntimePrivateAutodiffCheckpointPlan view{
            sizeof(VernonRuntimePrivateAutodiffCheckpointPlan),
            static_cast<uint8_t>(plan.present),
            {},
            plan.peakBytes,
            plan.memoryBudget,
            plan.logicalResidualBytes,
            plan.retainedAllocationBytes,
            plan.initialStateBytes,
            plan.restorationBytes,
            plan.transactionBytes,
            plan.persistentCheckpointBytes,
            plan.backwardValueBytes,
            plan.replayCost,
            plan.recomputationCost,
            viewOf(plan.selectedPolicy),
        };
        visitor(userData, &view);
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

VernonStatus vernonRuntimePrivateVisitAutodiffPassTelemetry(const VernonPullback *pullback,
                                                            VernonRuntimePrivateAutodiffPassVisitor visitor,
                                                            void *userData) {
    if (!visitor)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        for (const vernon::runtime::AutodiffPullbackPassTelemetry &item :
             vernon::runtime::autodiffPullbackPassTelemetry(pullback)) {
            const VernonRuntimePrivateAutodiffPassTelemetry view{
                sizeof(VernonRuntimePrivateAutodiffPassTelemetry),
                item.scheduleOffset,
                viewOf(item.passName),
                viewOf(item.residualSourceKind),
                viewOf(item.controlHistoryKind),
                item.estimatedTapeBytes,
                item.logicalResidualBytes,
                item.residentTapeBytes,
                item.allocatedTapeBytes,
                item.retainedAllocationBytes,
                item.peakTemporaryTapeBytes,
                item.checkpointBytes,
                item.activeOperationCount,
                item.recomputationCost,
            };
            visitor(userData, &view);
        }
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

VernonStatus vernonRuntimePrivateVisitProgramBoundaries(const VernonProgramExecutable *executable,
                                                        VernonRuntimePrivateProgramBoundaryVisitor visitor,
                                                        void *userData) {
    if (!executable || !visitor)
        return VERNON_STATUS_INVALID_ARGUMENT;
    try {
        for (const vernon::runtime::ProgramBoundaryView &item : vernon::runtime::programBoundaryViews(*executable)) {
            const VernonRuntimePrivateProgramBoundary view{sizeof(VernonRuntimePrivateProgramBoundary),
                                                           item.slot,
                                                           item.value,
                                                           viewOf(item.path),
                                                           viewOf(item.role),
                                                           viewOf(item.category)};
            visitor(userData, &view);
        }
        return VERNON_STATUS_OK;
    } catch (...) {
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}

VernonStatus vernonRuntimePrivateSetFailureInjection(VernonRuntimePrivateFailureBoundary boundary, size_t occurrence) {
    using vernon::runtime::program_execution::FailureBoundary;
    if (boundary < VERNON_RUNTIME_PRIVATE_FAILURE_NONE || boundary > VERNON_RUNTIME_PRIVATE_FAILURE_COMMIT)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (boundary == VERNON_RUNTIME_PRIVATE_FAILURE_NONE) {
        vernon::runtime::program_execution::clearFailureInjectionForTesting();
        return VERNON_STATUS_OK;
    }
    if (!occurrence)
        return VERNON_STATUS_INVALID_ARGUMENT;
    vernon::runtime::program_execution::setFailureInjectionForTesting(static_cast<FailureBoundary>(boundary),
                                                                      occurrence);
    return VERNON_STATUS_OK;
}

} // extern "C"
