#pragma once

#include "VernonCompiler.h"
#include "compiler_internal.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::compiler {

bool crossCompileSpirv(std::vector<Artifact> &artifacts, std::string &diagnostics, VernonTarget target,
                       uint32_t glslVersion);

} // namespace vernon::compiler
