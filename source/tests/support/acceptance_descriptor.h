#pragma once

#include "VernonRuntime.h"

#include <cstddef>
#include <string_view>

namespace vernon::tests {

struct AcceptanceDescriptor {
    std::string_view suite;
    std::string_view fixtureId;
    bool compute;
    bool graphics;
    bool storageBuffers;
    bool storageTexture;
    bool deviceAtomics;
    bool f32AtomicAdd;
    bool textureSamplerOperations;
    bool programVjp;
    std::string_view oracle;
    std::string_view parameter;
    VernonLaunchSize grid;
    const double *expected;
    size_t expectedCount;
};

} // namespace vernon::tests
