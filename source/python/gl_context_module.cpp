#include <GLFW/glfw3.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

std::mutex glfwMutex;
size_t glfwOwners = 0;

void retainGlfw() {
    std::lock_guard<std::mutex> lock(glfwMutex);
    if (!glfwOwners && glfwInit() != GLFW_TRUE) {
        const char *description = nullptr;
        glfwGetError(&description);
        throw std::runtime_error(description ? description : "GLFW initialization failed");
    }
    ++glfwOwners;
}

void releaseGlfw() {
    std::lock_guard<std::mutex> lock(glfwMutex);
    if (--glfwOwners == 0)
        glfwTerminate();
}

class Context {
public:
    Context(const std::string &backend, int major, int minor) : major_(major), minor_(minor) {
        if (backend != "opengl" && backend != "opengles")
            throw std::invalid_argument("backend must be opengl or opengles");
        if (major < 2 || minor < 0)
            throw std::invalid_argument("invalid OpenGL context version");

        retainGlfw();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CLIENT_API, backend == "opengles" ? GLFW_OPENGL_ES_API : GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
        if (backend == "opengl" && major >= 3)
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
        if (backend == "opengl" && major >= 3)
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
        window_ = glfwCreateWindow(1, 1, "VernonDSL", nullptr, nullptr);
        if (!window_) {
            const char *description = nullptr;
            glfwGetError(&description);
            releaseGlfw();
            throw std::runtime_error(description ? description : "GLFW context creation failed");
        }
        glfwMakeContextCurrent(window_);
    }

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    ~Context() {
        if (!window_)
            return;
        if (glfwGetCurrentContext() == window_)
            glfwMakeContextCurrent(nullptr);
        glfwDestroyWindow(window_);
        releaseGlfw();
    }

    uintptr_t userData() { return reinterpret_cast<uintptr_t>(this); }

    uintptr_t makeCurrentAddress() const { return reinterpret_cast<uintptr_t>(&makeCurrent); }

    uintptr_t getProcAddressAddress() const { return reinterpret_cast<uintptr_t>(&getProcAddress); }

    nb::tuple apiVersion() const { return nb::make_tuple(major_, minor_); }

private:
    static void makeCurrent(void *userData) {
        auto *context = static_cast<Context *>(userData);
        glfwMakeContextCurrent(context->window_);
    }

    static void *getProcAddress(void *, const char *name) { return reinterpret_cast<void *>(glfwGetProcAddress(name)); }

    GLFWwindow *window_{};
    int major_{};
    int minor_{};
};

} // namespace

NB_MODULE(_gl_context, module) {
    module.doc() = "Hidden GLFW context owner for VernonDSL OpenGL runtimes";
    nb::class_<Context>(module, "Context")
        .def(nb::init<const std::string &, int, int>(), nb::arg("backend"), nb::arg("major"), nb::arg("minor"))
        .def_prop_ro("user_data", &Context::userData)
        .def_prop_ro("make_current", &Context::makeCurrentAddress)
        .def_prop_ro("get_proc_address", &Context::getProcAddressAddress)
        .def_prop_ro("api_version", &Context::apiVersion);
}
