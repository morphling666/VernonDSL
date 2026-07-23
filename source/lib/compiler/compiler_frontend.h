#pragma once

#include "VernonCommon.h"
#include "compiler_internal.h"

#include <cstddef>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace vernon::compiler {

class CompilerFrontend;

CompilerFrontend *createCompilerFrontend();
void destroyCompilerFrontend(CompilerFrontend *frontend);
mlir::MLIRContext &compilerMlirContext(CompilerFrontend &frontend);

VernonStatus validateMlir(CompilerFrontend &frontend, const char *source, size_t sourceSize,
                          std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics);

} // namespace vernon::compiler
