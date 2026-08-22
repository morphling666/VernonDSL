#ifndef VERNON_COMPILER_KERNEL_BOOTSTRAP_H
#define VERNON_COMPILER_KERNEL_BOOTSTRAP_H

#include "VernonCommon.h"
#include "compiler_frontend.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vernon::compiler {

VernonStatus planComputeKernel(CompilerFrontend &frontend, const char *source, size_t sourceSize,
                               std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics);

} // namespace vernon::compiler

#endif
