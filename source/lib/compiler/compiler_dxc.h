#pragma once

#include "compiler_internal.h"

#include <string>
#include <vector>

namespace vernon::compiler {

bool compileHlslToDxil(const std::vector<Artifact> &hlslArtifacts, uint32_t shaderModel,
                       std::vector<Artifact> &dxilArtifacts, std::string &diagnostics);

} // namespace vernon::compiler
