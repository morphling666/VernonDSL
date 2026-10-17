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
 *
 * A device created from these callbacks is bound to the thread that owns the
 * context: every call on the device, including submission and completion
 * queries, must come from that thread.
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

typedef void (*VernonOpenGLSwapIntervalFn)(void *user_data, int32_t interval);
typedef void (*VernonOpenGLSwapBuffersFn)(void *user_data);
typedef void (*VernonOpenGLDrawableSizeFn)(void *user_data, uint32_t *width, uint32_t *height);

/*
 * A window an OpenGL device presents to. `make_current` makes the device's own context current on the window, as
 * SDL_GL_MakeCurrent(window, device_context) does; framebuffer and vertex-array objects are not shared between
 * contexts, so the window must accept that context. `swap_interval` and `swap_buffers` act on the window while it is
 * current, and `drawable_size` reports its size in pixels. The callbacks run on the device's thread.
 */
typedef struct VernonOpenGLSurfaceCallbacks {
    uint32_t struct_size;
    void *user_data;
    VernonOpenGLMakeCurrentFn make_current;
    VernonOpenGLSwapIntervalFn swap_interval;
    VernonOpenGLSwapBuffersFn swap_buffers;
    VernonOpenGLDrawableSizeFn drawable_size;
    uint32_t reserved[4];
} VernonOpenGLSurfaceCallbacks;

#ifdef __cplusplus
}
#endif

#endif
