#ifndef VERNON_C_COMMON_H
#define VERNON_C_COMMON_H

#include "VernonVersions.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonStringView {
    const char *data;
    size_t size;
} VernonStringView;

typedef enum VernonStatus {
    VERNON_STATUS_OK = 0,
    VERNON_STATUS_INVALID_ARGUMENT = 1,
    VERNON_STATUS_PARSE_ERROR = 2,
    VERNON_STATUS_VERIFICATION_ERROR = 3,
    VERNON_STATUS_UNSUPPORTED_TARGET = 4,
    VERNON_STATUS_INTERNAL_ERROR = 5
} VernonStatus;

// CPU entry points are emitted by the compiler but invoked by the standalone
// runtime, so their ABI belongs to neither library's API exclusively.
typedef struct VernonCpuTextureCallbacks {
    void *user_data;
    void (*sample_2d)(void *user_data, uintptr_t texture, float u, float v, float out_rgba[4]);
    void (*size_2d)(void *user_data, uintptr_t texture, int32_t level, int32_t out_size[2]);
} VernonCpuTextureCallbacks;

typedef struct VernonCpuInvocation {
    const void *arguments;
    size_t arguments_size;
    void *results;
    size_t results_size;
    const VernonCpuTextureCallbacks *textures;
} VernonCpuInvocation;

typedef VernonStatus (*VernonCpuEntryPoint)(const VernonCpuInvocation *invocation);

#ifdef __cplusplus
}
#endif

#endif
