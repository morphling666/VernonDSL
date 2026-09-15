#ifndef VERNON_RUNTIME_PYTHON_BRIDGE_H
#define VERNON_RUNTIME_PYTHON_BRIDGE_H

#include "VernonRuntime.h"

#if defined(VERNON_RUNTIME_STATIC)
#define VERNON_RUNTIME_PRIVATE_CAPI
#elif defined(_WIN32) && defined(VERNON_RUNTIME_BUILD)
#define VERNON_RUNTIME_PRIVATE_CAPI __declspec(dllexport)
#elif defined(_WIN32)
#define VERNON_RUNTIME_PRIVATE_CAPI __declspec(dllimport)
#else
#define VERNON_RUNTIME_PRIVATE_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonRuntimePrivateAutodiffMemoryUsage {
    uint32_t struct_size;
    uint64_t logical_residual_bytes;
    uint64_t resident_bytes;
    uint64_t allocated_bytes;
    uint64_t retained_allocation_bytes;
    uint64_t peak_temporary_bytes;
} VernonRuntimePrivateAutodiffMemoryUsage;

typedef struct VernonRuntimePrivateAutodiffControlPlaneUsage {
    uint32_t struct_size;
    uint64_t submissions;
    uint64_t waits;
    uint64_t readbacks;
    uint64_t atomic_publications;
    uint64_t temporary_allocation_bytes;
    uint64_t device_wait_nanoseconds;
} VernonRuntimePrivateAutodiffControlPlaneUsage;

typedef struct VernonRuntimePrivateAutodiffCheckpointPlan {
    uint32_t struct_size;
    uint8_t present;
    uint8_t reserved[3];
    uint64_t peak_bytes;
    uint64_t memory_budget;
    uint64_t logical_residual_bytes;
    uint64_t retained_allocation_bytes;
    uint64_t initial_state_bytes;
    uint64_t restoration_bytes;
    uint64_t transaction_bytes;
    uint64_t persistent_checkpoint_bytes;
    uint64_t backward_value_bytes;
    uint64_t replay_cost;
    uint64_t recomputation_cost;
    VernonStringView selected_policy;
} VernonRuntimePrivateAutodiffCheckpointPlan;

typedef struct VernonRuntimePrivateAutodiffPassTelemetry {
    uint32_t struct_size;
    uint32_t schedule_offset;
    VernonStringView pass_name;
    VernonStringView residual_source_kind;
    VernonStringView control_history_kind;
    uint64_t estimated_tape_bytes;
    uint64_t logical_residual_bytes;
    uint64_t resident_tape_bytes;
    uint64_t allocated_tape_bytes;
    uint64_t retained_allocation_bytes;
    uint64_t peak_temporary_tape_bytes;
    uint64_t checkpoint_bytes;
    uint64_t active_operation_count;
    uint64_t recomputation_cost;
} VernonRuntimePrivateAutodiffPassTelemetry;

typedef void (*VernonRuntimePrivateAutodiffPassVisitor)(void *user_data,
                                                        const VernonRuntimePrivateAutodiffPassTelemetry *telemetry);
typedef void (*VernonRuntimePrivateAutodiffCheckpointVisitor)(void *user_data,
                                                              const VernonRuntimePrivateAutodiffCheckpointPlan *plan);

typedef struct VernonRuntimePrivateProgramBoundary {
    uint32_t struct_size;
    uint32_t slot;
    uint32_t value;
    VernonStringView path;
    VernonStringView role;
    VernonStringView category;
} VernonRuntimePrivateProgramBoundary;

typedef void (*VernonRuntimePrivateProgramBoundaryVisitor)(void *user_data,
                                                           const VernonRuntimePrivateProgramBoundary *boundary);

typedef enum VernonRuntimePrivateFailureBoundary {
    VERNON_RUNTIME_PRIVATE_FAILURE_NONE = 0,
    VERNON_RUNTIME_PRIVATE_FAILURE_PLANNING = 1,
    VERNON_RUNTIME_PRIVATE_FAILURE_ALLOCATION = 2,
    VERNON_RUNTIME_PRIVATE_FAILURE_TRANSFER = 3,
    VERNON_RUNTIME_PRIVATE_FAILURE_SUBMISSION = 4,
    VERNON_RUNTIME_PRIVATE_FAILURE_COMPLETION = 5,
    VERNON_RUNTIME_PRIVATE_FAILURE_TAPE_VALIDATION = 6,
    VERNON_RUNTIME_PRIVATE_FAILURE_READBACK = 7,
    VERNON_RUNTIME_PRIVATE_FAILURE_COMMIT = 8
} VernonRuntimePrivateFailureBoundary;

VERNON_RUNTIME_PRIVATE_CAPI VernonStatus vernonRuntimePrivateGetAutodiffMemoryUsage(
    const VernonPullback *pullback, VernonRuntimePrivateAutodiffMemoryUsage *output);
VERNON_RUNTIME_PRIVATE_CAPI VernonStatus vernonRuntimePrivateGetAutodiffControlPlaneUsage(
    const VernonPullback *pullback, VernonRuntimePrivateAutodiffControlPlaneUsage *output);
VERNON_RUNTIME_PRIVATE_CAPI uint64_t
vernonRuntimePrivateGetAutodiffHostTapeContextLimit(const VernonRuntimeContext *context);
VERNON_RUNTIME_PRIVATE_CAPI VernonRhiDevice
vernonRuntimePrivateGetAutodiffRhiDevice(const VernonRuntimeContext *context);
VERNON_RUNTIME_PRIVATE_CAPI uint64_t
vernonRuntimePrivateGetAutodiffPeakRuntimeManagedBytes(const VernonPullback *pullback);
VERNON_RUNTIME_PRIVATE_CAPI VernonStatus vernonRuntimePrivateVisitAutodiffCheckpointPlan(
    const VernonPullback *pullback, VernonRuntimePrivateAutodiffCheckpointVisitor visitor, void *user_data);
VERNON_RUNTIME_PRIVATE_CAPI VernonStatus vernonRuntimePrivateVisitAutodiffPassTelemetry(
    const VernonPullback *pullback, VernonRuntimePrivateAutodiffPassVisitor visitor, void *user_data);
VERNON_RUNTIME_PRIVATE_CAPI VernonStatus vernonRuntimePrivateVisitProgramBoundaries(
    const VernonProgramExecutable *executable, VernonRuntimePrivateProgramBoundaryVisitor visitor, void *user_data);
VERNON_RUNTIME_PRIVATE_CAPI VernonStatus
vernonRuntimePrivateSetFailureInjection(VernonRuntimePrivateFailureBoundary boundary, size_t occurrence);

#ifdef __cplusplus
}
#endif

#endif
