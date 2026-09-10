#pragma once

#include "backend_test_matrix.h"
#include "program_fixture_manifest_table.h"
#include "runtime_rhi_test_utils.h"

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

inline BackendTestRequirements computeFixtureRequirements(VernonRuntimeBackend runtime, bool graphics = false) {
    BackendTestRequirements requirements{};
    requirements.compute = true;
    requirements.graphics = graphics;
    requirements.storageBuffers = true;
    if (runtime == VERNON_RUNTIME_OPENGL) {
        requirements.minimumApiMajor = 4;
        requirements.minimumApiMinor = 3;
    } else if (runtime == VERNON_RUNTIME_OPENGL_ES) {
        requirements.minimumApiMajor = 3;
        requirements.minimumApiMinor = 1;
    }
    return requirements;
}

class ProgramFixtureRuntimeTest : public testing::TestWithParam<ProgramFixtureManifest> {
protected:
    void SetUp() override {
        const auto backend =
            std::find_if(backendTestMatrix.begin(), backendTestMatrix.end(),
                         [this](const BackendTestRow &row) { return row.runtime == GetParam().runtime; });
        ASSERT_NE(backend, backendTestMatrix.end());
        owned_ = std::make_unique<OwnedRhiRuntime>(backend->runtime);
        const BackendProbeResult probe = probeRuntimeBackend(*backend, requirements(), owned_->runtime());
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
    std::unique_ptr<OwnedRhiRuntime> owned_;
};

} // namespace vernon::tests
