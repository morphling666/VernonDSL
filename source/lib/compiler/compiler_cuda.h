#pragma once

#include "compiler_internal.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vernon::compiler {

class PreparedModule;

bool compileCuda(PreparedModule &prepared, std::vector<Artifact> &artifacts, std::string &diagnostics);

} // namespace vernon::compiler
