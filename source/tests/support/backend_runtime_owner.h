#pragma once

#include "backend_test_matrix.h"
#include "runtime_rhi_test_utils.h"

#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
#include "glfw_context_owner.h"
#endif

#include <memory>
#include <stdexcept>
#include <string>

namespace vernon::tests {

class BackendRuntimeOwner {
public:
    BackendProbeResult initialize(const BackendTestRow &backend, const BackendTestRequirements &requirements) {
        if (backend.runtime == VERNON_RUNTIME_OPENGL || backend.runtime == VERNON_RUNTIME_OPENGL_ES) {
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
            const auto api = backend.runtime == VERNON_RUNTIME_OPENGL ? host::GlfwContextApi::OpenGL
                                                                      : host::GlfwContextApi::OpenGLES;
            const uint16_t major = requirements.minimumApiMajor
                                       ? requirements.minimumApiMajor
                                       : static_cast<uint16_t>(backend.runtime == VERNON_RUNTIME_OPENGL ? 4 : 3);
            const uint16_t minor = requirements.minimumApiMajor
                                       ? requirements.minimumApiMinor
                                       : static_cast<uint16_t>(backend.runtime == VERNON_RUNTIME_OPENGL ? 1 : 0);
            try {
                openglContext_ = std::make_unique<host::GlfwContextOwner>(api, major, minor);
            } catch (const std::runtime_error &error) {
                return {BackendProbeKind::DeviceOrContextUnavailable,
                        std::string(backend.name) + " context is unavailable: " + error.what()};
            }
            openglCallbacks_ = openglContext_->callbacks();
            runtime_ = std::make_unique<OwnedRhiRuntime>(backend.runtime, &openglCallbacks_);
#else
            return {BackendProbeKind::PlatformNotBuilt, std::string(backend.name) + " GLFW context owner is not built"};
#endif
        } else {
            const RhiTestDevicePreference devicePreference = backend.runtime == VERNON_RUNTIME_DIRECTX12
                                                                 ? configuredDirectXTestDevicePreference()
                                                                 : RhiTestDevicePreference::Hardware;
            runtime_ = std::make_unique<OwnedRhiRuntime>(backend.runtime, nullptr, devicePreference);
        }
        return probeRuntimeBackend(backend, requirements, runtime_->runtime());
    }

    OwnedRhiRuntime &owned() { return *runtime_; }
    RhiRuntime &context() { return runtime_->context(); }

private:
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
    std::unique_ptr<host::GlfwContextOwner> openglContext_;
    VernonOpenGLContextCallbacks openglCallbacks_{};
#endif
    std::unique_ptr<OwnedRhiRuntime> runtime_;
};

} // namespace vernon::tests
