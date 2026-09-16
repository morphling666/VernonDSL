#include "glfw_context_owner.h"

#include <GLFW/glfw3.h>

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vernon::host {
namespace {

std::mutex glfwMutex;
size_t glfwOwnerCount = 0;

std::string glfwError(const char *fallback) {
    const char *description = nullptr;
    glfwGetError(&description);
    return description ? description : fallback;
}

} // namespace

GlfwContextOwner::GlfwContextOwner(GlfwContextApi api, uint16_t major, uint16_t minor) : major_(major), minor_(minor) {
    if (major < 2)
        throw std::invalid_argument("invalid OpenGL context version");

    std::lock_guard<std::mutex> lock(glfwMutex);
    if (glfwOwnerCount == 0 && glfwInit() != GLFW_TRUE)
        throw std::runtime_error(glfwError("GLFW initialization failed"));

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, api == GlfwContextApi::OpenGLES ? GLFW_OPENGL_ES_API : GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    if (api == GlfwContextApi::OpenGL && major >= 3)
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    if (api == GlfwContextApi::OpenGL && major >= 3)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    GLFWwindow *window = glfwCreateWindow(1, 1, "VernonDSL", nullptr, nullptr);
    if (!window) {
        const std::string error = glfwError("GLFW context creation failed");
        if (glfwOwnerCount == 0)
            glfwTerminate();
        throw std::runtime_error(error);
    }
    window_ = window;
    ++glfwOwnerCount;
    glfwMakeContextCurrent(window);
}

GlfwContextOwner::~GlfwContextOwner() {
    if (!window_)
        return;
    std::lock_guard<std::mutex> lock(glfwMutex);
    auto *window = static_cast<GLFWwindow *>(window_);
    if (glfwGetCurrentContext() == window)
        glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(window);
    if (--glfwOwnerCount == 0)
        glfwTerminate();
}

VernonOpenGLContextCallbacks GlfwContextOwner::callbacks() noexcept {
    VernonOpenGLContextCallbacks result{};
    result.struct_size = sizeof(result);
    result.user_data = this;
    result.make_current = &makeCurrent;
    result.get_proc_address = &getProcAddress;
    result.api_version_major = major_;
    result.api_version_minor = minor_;
    return result;
}

void GlfwContextOwner::makeCurrent(void *userData) {
    auto *owner = static_cast<GlfwContextOwner *>(userData);
    glfwMakeContextCurrent(static_cast<GLFWwindow *>(owner->window_));
}

void *GlfwContextOwner::getProcAddress(void *, const char *name) {
    return reinterpret_cast<void *>(glfwGetProcAddress(name));
}

} // namespace vernon::host
