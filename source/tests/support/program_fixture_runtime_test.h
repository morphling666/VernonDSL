#pragma once

#include "backend_test_matrix.h"
#include "program_fixture_manifest_table.h"
#include "runtime_rhi_test_utils.h"

#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
#include "glfw_context_owner.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::tests {

inline std::vector<ProgramFixtureManifest> programFixtureCases(std::string_view fixtureId, bool includeCpu = false) {
    std::vector<ProgramFixtureManifest> result;
    for (const ProgramFixtureManifest &entry : programFixtureManifestTable)
        if (entry.fixtureId == fixtureId && (includeCpu || entry.runtime != VERNON_RUNTIME_CPU))
            result.push_back(entry);
    return result;
}

inline void setOpenGLComputeApiRequirement(BackendTestRequirements &requirements, VernonRuntimeBackend runtime) {
    if (runtime == VERNON_RUNTIME_OPENGL) {
        requirements.minimumApiMajor = 4;
        requirements.minimumApiMinor = 3;
    } else if (runtime == VERNON_RUNTIME_OPENGL_ES) {
        requirements.minimumApiMajor = 3;
        requirements.minimumApiMinor = 1;
    }
}

inline BackendTestRequirements computeFixtureRequirements(VernonRuntimeBackend runtime) {
    BackendTestRequirements requirements{};
    requirements.compute = true;
    requirements.storageBuffers = true;
    setOpenGLComputeApiRequirement(requirements, runtime);
    return requirements;
}

inline BackendTestRequirements graphicsImageCopyFixtureRequirements(VernonRuntimeBackend runtime) {
    BackendTestRequirements requirements{};
    requirements.graphics = true;
    if (runtime == VERNON_RUNTIME_OPENGL) {
        requirements.minimumApiMajor = 4;
        requirements.minimumApiMinor = 3;
    } else if (runtime == VERNON_RUNTIME_OPENGL_ES) {
        requirements.minimumApiMajor = 3;
        requirements.minimumApiMinor = 2;
    }
    return requirements;
}

inline BackendTestRequirements computeGraphicsFixtureRequirements(VernonRuntimeBackend runtime) {
    BackendTestRequirements requirements = computeFixtureRequirements(runtime);
    requirements.graphics = true;
    return requirements;
}

class ProgramFixtureRuntimeTest : public testing::TestWithParam<ProgramFixtureManifest> {
protected:
    void SetUp() override {
        const auto backend =
            std::find_if(backendTestMatrix.begin(), backendTestMatrix.end(),
                         [this](const BackendTestRow &row) { return row.runtime == GetParam().runtime; });
        ASSERT_NE(backend, backendTestMatrix.end());
        const BackendTestRequirements testRequirements = requirements();
        if (backend->runtime == VERNON_RUNTIME_OPENGL || backend->runtime == VERNON_RUNTIME_OPENGL_ES) {
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
            const auto api = backend->runtime == VERNON_RUNTIME_OPENGL ? host::GlfwContextApi::OpenGL
                                                                       : host::GlfwContextApi::OpenGLES;
            const uint16_t major = testRequirements.minimumApiMajor
                                       ? testRequirements.minimumApiMajor
                                       : static_cast<uint16_t>(backend->runtime == VERNON_RUNTIME_OPENGL ? 4 : 3);
            const uint16_t minor = testRequirements.minimumApiMajor
                                       ? testRequirements.minimumApiMinor
                                       : static_cast<uint16_t>(backend->runtime == VERNON_RUNTIME_OPENGL ? 1 : 0);
            try {
                openglContext_ = std::make_unique<host::GlfwContextOwner>(api, major, minor);
            } catch (const std::runtime_error &error) {
                GTEST_SKIP() << backend->name << " context is unavailable: " << error.what();
            }
            openglCallbacks_ = openglContext_->callbacks();
            owned_ = std::make_unique<OwnedRhiRuntime>(backend->runtime, &openglCallbacks_);
#else
            GTEST_SKIP() << backend->name << " GLFW context owner is not built";
#endif
        } else {
            owned_ = std::make_unique<OwnedRhiRuntime>(backend->runtime);
        }
        const BackendProbeResult probe = probeRuntimeBackend(*backend, testRequirements, owned_->runtime());
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
            return;
        }
    }

    virtual BackendTestRequirements requirements() const = 0;

    const ProgramFixtureManifest &fixture(std::string_view fixtureId) const {
        const ProgramFixtureManifest *result = findProgramFixtureManifest(fixtureId, GetParam().target);
        if (!result)
            throw std::logic_error("missing fixture '" + std::string(fixtureId) + "' for target '" +
                                   std::string(GetParam().target) + "'");
        return *result;
    }

    OwnedRhiRuntime &owned() { return *owned_; }
    RhiRuntime &runtime() { return owned_->context(); }

private:
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
    std::unique_ptr<host::GlfwContextOwner> openglContext_;
    VernonOpenGLContextCallbacks openglCallbacks_{};
#endif
    std::unique_ptr<OwnedRhiRuntime> owned_;
};

} // namespace vernon::tests
