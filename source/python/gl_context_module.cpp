#include "glfw_context_owner.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

class Context {
public:
    Context(const std::string &backend, int major, int minor) {
        if (backend != "opengl" && backend != "opengles")
            throw std::invalid_argument("backend must be opengl or opengles");
        if (major < 2 || minor < 0)
            throw std::invalid_argument("invalid OpenGL context version");
        const auto api =
            backend == "opengles" ? vernon::host::GlfwContextApi::OpenGLES : vernon::host::GlfwContextApi::OpenGL;
        owner_ = std::make_unique<vernon::host::GlfwContextOwner>(api, static_cast<uint16_t>(major),
                                                                  static_cast<uint16_t>(minor));
    }

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    uintptr_t userData() { return reinterpret_cast<uintptr_t>(owner_->callbacks().user_data); }

    uintptr_t makeCurrentAddress() { return reinterpret_cast<uintptr_t>(owner_->callbacks().make_current); }

    uintptr_t getProcAddressAddress() { return reinterpret_cast<uintptr_t>(owner_->callbacks().get_proc_address); }

    nb::tuple apiVersion() const { return nb::make_tuple(owner_->apiMajor(), owner_->apiMinor()); }

private:
    std::unique_ptr<vernon::host::GlfwContextOwner> owner_;
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
