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
    GlfwContextOwner(GlfwContextApi api, uint16_t major, uint16_t minor);
    ~GlfwContextOwner();

    GlfwContextOwner(const GlfwContextOwner &) = delete;
    GlfwContextOwner &operator=(const GlfwContextOwner &) = delete;
    GlfwContextOwner(GlfwContextOwner &&) = delete;
    GlfwContextOwner &operator=(GlfwContextOwner &&) = delete;

    VernonOpenGLContextCallbacks callbacks() noexcept;
    uint16_t apiMajor() const noexcept { return major_; }
    uint16_t apiMinor() const noexcept { return minor_; }

private:
    static void makeCurrent(void *userData);
    static void *getProcAddress(void *userData, const char *name);

    void *window_{};
    uint16_t major_{};
    uint16_t minor_{};
};

} // namespace vernon::host

#endif
