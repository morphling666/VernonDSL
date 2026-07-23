#pragma once

#include "VernonCompiler.h"
#include "compiler_internal.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace vernon::compiler {

bool compileCpu(mlir::MLIRContext &context, const char *source, size_t sourceSize, const CpuCodegenOptions &options,
                std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics,
                CpuExecutionStatePtr &execution);

bool linkHostObject(const void *object, size_t objectSize, Artifact &artifact, std::string &diagnostics);

VernonCpuEntryPoint findCpuEntry(const CpuExecutionState *execution, std::string_view name);

} // namespace vernon::compiler
