#pragma once

#include "compiler_internal.h"

#include <cstddef>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace vernon::compiler {

bool compileCuda(mlir::MLIRContext &context, const char *source, size_t sourceSize, std::vector<Artifact> &artifacts,
                 std::string &diagnostics);

} // namespace vernon::compiler
