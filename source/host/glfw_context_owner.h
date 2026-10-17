#ifndef VERNON_HOST_GLFW_CONTEXT_OWNER_H
#define VERNON_HOST_GLFW_CONTEXT_OWNER_H

#include "VernonOpenGLContext.h"

#include <cstdint>

namespace vernon::host {

enum class GlfwContextApi {
    OpenGL,
    OpenGLES,
};

class GlfwContextOwner {
public:
    // The context's window is hidden; its size is the drawable size of `surfaceCallbacks()`.
    GlfwContextOwner(GlfwContextApi api, uint16_t major, uint16_t minor, uint32_t windowWidth = 1,
                     uint32_t windowHeight = 1);
    ~GlfwContextOwner();

    GlfwContextOwner(const GlfwContextOwner &) = delete;
    GlfwContextOwner &operator=(const GlfwContextOwner &) = delete;
    GlfwContextOwner(GlfwContextOwner &&) = delete;
    GlfwContextOwner &operator=(GlfwContextOwner &&) = delete;

    VernonOpenGLContextCallbacks callbacks() noexcept;
    // The context's own window as a surface of the device created from `callbacks()`.
    VernonOpenGLSurfaceCallbacks surfaceCallbacks() noexcept;
    uint16_t apiMajor() const noexcept { return major_; }
    uint16_t apiMinor() const noexcept { return minor_; }

private:
    static void makeCurrent(void *userData);
    static void *getProcAddress(void *userData, const char *name);
    static void swapInterval(void *userData, int32_t interval);
    static void swapBuffers(void *userData);
    static void drawableSize(void *userData, uint32_t *width, uint32_t *height);

    void *window_{};
    uint16_t major_{};
    uint16_t minor_{};
};

} // namespace vernon::host

#endif
