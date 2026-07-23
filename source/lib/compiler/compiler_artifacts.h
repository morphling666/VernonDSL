#pragma once

#include "VernonCompiler.h"
#include "compiler_internal.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::compiler {

void addArtifactTable(std::string &reflection, const std::vector<Artifact> &artifacts, VernonTarget target,
                      uint32_t glslVersion, std::string_view cpuTargetTriple = {}, std::string_view cpu = {},
                      std::string_view cpuFeatures = {});

} // namespace vernon::compiler
