#ifndef VERNON_OPENGL_CONTEXT_H
#define VERNON_OPENGL_CONTEXT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*VernonOpenGLMakeCurrentFn)(void *user_data);
typedef void *(*VernonOpenGLGetProcAddressFn)(void *user_data, const char *name);

/*
 * OpenGL context ownership is intentionally outside the RHI. Python context
 * owners and host engines provide the same callbacks; their origin is not
 * observable by the OpenGL backend.
 */
typedef struct VernonOpenGLContextCallbacks {
    uint32_t struct_size;
    void *user_data;
    VernonOpenGLMakeCurrentFn make_current;
    VernonOpenGLGetProcAddressFn get_proc_address;
    uint16_t api_version_major;
    uint16_t api_version_minor;
    uint32_t reserved[4];
} VernonOpenGLContextCallbacks;

#ifdef __cplusplus
}
#endif

#endif
