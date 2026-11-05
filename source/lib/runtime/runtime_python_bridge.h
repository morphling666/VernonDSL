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

VERNON_RUNTIME_PRIVATE_CAPI uint64_t
vernonRuntimePrivateGetAutodiffHostTapeContextLimit(const VernonRuntimeContext *context);
VERNON_RUNTIME_PRIVATE_CAPI VernonStatus vernonRuntimePrivateVisitProgramBoundaries(
    const VernonProgramExecutable *executable, VernonRuntimePrivateProgramBoundaryVisitor visitor, void *user_data);
/* Fails the next `count` vernonScheduleWait calls of a nonzero point, as a device failure would; zero clears it. */
VERNON_RUNTIME_PRIVATE_CAPI void vernonRuntimePrivateFailScheduleWaits(size_t count);

#ifdef __cplusplus
}
#endif

#endif
