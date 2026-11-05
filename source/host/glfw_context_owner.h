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

// A hidden, undecorated window without a client API, for the surfaces of devices that present to native windows.
class GlfwWindowOwner {
public:
    GlfwWindowOwner(uint32_t width, uint32_t height);
    ~GlfwWindowOwner();

    GlfwWindowOwner(const GlfwWindowOwner &) = delete;
    GlfwWindowOwner &operator=(const GlfwWindowOwner &) = delete;
    GlfwWindowOwner(GlfwWindowOwner &&) = delete;
    GlfwWindowOwner &operator=(GlfwWindowOwner &&) = delete;

    // The HWND on Windows; null elsewhere.
    void *win32Window() const noexcept;
    // Resizes the window's client area and waits until the window system reports the new size.
    void resize(uint32_t width, uint32_t height);

private:
    void *window_{};
};

} // namespace vernon::host

#endif
