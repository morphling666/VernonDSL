#ifndef VERNON_TESTS_SUPPORT_PROGRAM_FIXTURE_MANIFEST_TABLE_H
#define VERNON_TESTS_SUPPORT_PROGRAM_FIXTURE_MANIFEST_TABLE_H

#include "VernonRuntime.h"

#include <ostream>
#include <string_view>

#include "program_fixture_registration_declarations.inc"

namespace vernon::tests {

struct ProgramFixtureManifest {
    std::string_view fixtureId;
    std::string_view target;
    VernonRuntimeBackend runtime;
    std::string_view manifestPath;
    VernonStatus (*registerArtifacts)();

    VernonStatus prepare() const { return registerArtifacts ? registerArtifacts() : VERNON_STATUS_OK; }
};

inline void PrintTo(const ProgramFixtureManifest &fixture, std::ostream *stream) { *stream << fixture.target; }

inline constexpr ProgramFixtureManifest programFixtureManifestTable[]{
#include "program_fixture_manifest_rows.inc"
};

inline const ProgramFixtureManifest *findProgramFixtureManifest(std::string_view fixtureId, std::string_view target) {
    for (const ProgramFixtureManifest &entry : programFixtureManifestTable)
        if (entry.fixtureId == fixtureId && entry.target == target)
            return &entry;
    return nullptr;
}

inline const ProgramFixtureManifest *findProgramFixtureManifest(std::string_view fixtureId,
                                                                VernonRuntimeBackend runtime) {
    for (const ProgramFixtureManifest &entry : programFixtureManifestTable)
        if (entry.fixtureId == fixtureId && entry.runtime == runtime)
            return &entry;
    return nullptr;
}

} // namespace vernon::tests

#endif
