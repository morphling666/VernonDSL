#pragma once

#include "compiler_internal.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vernon::compiler {

class PreparedModule;
struct TargetProfile;

bool compileCuda(PreparedModule &prepared, const TargetProfile &profile, std::vector<Artifact> &artifacts,
                 std::string &reflection, std::string &diagnostics);

} // namespace vernon::compiler
