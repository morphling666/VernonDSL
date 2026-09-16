#pragma once

#include "compiler_dispatch.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::compiler {

bool addArtifactTable(std::string &reflection, std::string &diagnostics, const std::vector<Artifact> &artifacts,
                      const CompileOptions &options, const std::vector<TargetResourceSlot> &targetResourceSlots = {});

} // namespace vernon::compiler
