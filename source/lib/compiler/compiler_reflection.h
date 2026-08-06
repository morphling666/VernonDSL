#pragma once

#include "VernonCompiler.h"
#include "VernonCpuAbiWrapper.h"
#include "compiler_frontend.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <vector>

namespace vernon::compiler {

mlir::FailureOr<LogicalReflectionModel> buildLogicalReflectionModel(mlir::ModuleOp module);
std::vector<PhysicalEntryModel> buildPhysicalEntryModels(mlir::ModuleOp module,
                                                         const std::vector<PhysicalEntryProvenance> &provenance);

mlir::FailureOr<std::string> buildReflection(mlir::ModuleOp module, const LogicalReflectionModel &logical,
                                             const std::vector<PhysicalEntryModel> &physicalEntries,
                                             const std::vector<PhysicalEntryProvenance> &provenance);

bool selectTargetPhysicalLayouts(std::string &reflection, VernonTarget target, std::string &diagnostics);

bool setCpuReflectionSymbols(std::string &reflection, const std::vector<vernon::CpuAbiWrapperMetadata> &metadata);

} // namespace vernon::compiler
