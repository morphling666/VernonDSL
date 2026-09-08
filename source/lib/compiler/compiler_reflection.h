#pragma once

#include "VernonCompiler.h"
#include "VernonCpuAbiWrapper.h"
#include "compiler_frontend.h"

#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"

#include <string>
#include <vector>

namespace vernon::compiler {

mlir::FailureOr<LogicalReflectionModel> buildLogicalReflectionModel(mlir::ModuleOp module);
mlir::FailureOr<llvm::json::Object> reflectCanonicalValueLayout(mlir::ModuleOp module, mlir::Type type,
                                                                llvm::ArrayRef<llvm::StringRef> logicalDtypes = {});
llvm::json::Object reflectCanonicalValueLayout(const mlir::vernon::ValueAbiLayout &layout, llvm::StringRef logicalType);
std::vector<PhysicalEntryModel> buildPhysicalEntryModels(mlir::ModuleOp module,
                                                         const std::vector<PhysicalEntryProvenance> &provenance);

mlir::FailureOr<std::string> buildReflection(mlir::ModuleOp module, const LogicalReflectionModel &logical,
                                             const std::vector<PhysicalEntryModel> &physicalEntries,
                                             const std::vector<PhysicalEntryProvenance> &provenance);

bool selectTargetPhysicalLayouts(std::string &reflection, VernonTarget target, std::string &diagnostics);

bool setCpuReflectionSymbols(std::string &reflection, const std::vector<vernon::CpuAbiWrapperMetadata> &metadata);

} // namespace vernon::compiler
