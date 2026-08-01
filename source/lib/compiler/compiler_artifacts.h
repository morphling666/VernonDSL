#pragma once

#include "VernonCompiler.h"
#include "compiler_internal.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::compiler {

// Kept in the private compiler header, but exported so the separately linked
// compiler unit tests can exercise artifact publication on Windows.
VERNON_DSL_CAPI bool addArtifactTable(std::string &reflection, std::string &diagnostics,
                                      const std::vector<Artifact> &artifacts, VernonTarget target, uint32_t glslVersion,
                                      std::string_view cpuTargetTriple = {}, std::string_view cpu = {},
                                      std::string_view cpuFeatures = {}, uint32_t hlslShaderModel = 50,
                                      VernonMetalPlatform metalPlatform = VERNON_METAL_PLATFORM_MACOS,
                                      const std::vector<TargetResourceSlot> &targetResourceSlots = {});

} // namespace vernon::compiler
