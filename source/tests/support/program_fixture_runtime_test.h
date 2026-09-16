#pragma once

#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "program_fixture_manifest_table.h"

#include <gtest/gtest.h>

#include <algorithm>
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

inline BackendTestRequirements programVjpFixtureRequirements(VernonRuntimeBackend runtime) {
    BackendTestRequirements requirements = computeFixtureRequirements(runtime);
    requirements.programVjp = true;
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
        const BackendProbeResult probe = runtimeOwner_.initialize(*backend, testRequirements);
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

    OwnedRhiRuntime &owned() { return runtimeOwner_.owned(); }
    RhiRuntime &runtime() { return runtimeOwner_.context(); }

private:
    BackendRuntimeOwner runtimeOwner_;
};

} // namespace vernon::tests
