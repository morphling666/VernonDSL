#pragma once

#include "VernonCompiler.h"
#include "VernonCpuAbiWrapper.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <vector>

namespace vernon::compiler {

mlir::FailureOr<std::string> buildReflection(mlir::ModuleOp module, mlir::ModuleOp sourceModule);

bool selectTargetPhysicalLayouts(std::string &reflection, VernonTarget target, std::string &diagnostics);

bool setCpuReflectionSymbols(std::string &reflection, const std::vector<vernon::CpuAbiWrapperMetadata> &metadata);

} // namespace vernon::compiler
