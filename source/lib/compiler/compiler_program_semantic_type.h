#ifndef VERNON_COMPILER_PROGRAM_SEMANTIC_TYPE_H
#define VERNON_COMPILER_PROGRAM_SEMANTIC_TYPE_H

#include "VernonProgramSemanticTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace vernon::compiler {

mlir::FailureOr<vernon::program::SemanticType> programSemanticType(mlir::ModuleOp module, mlir::Type type,
                                                                   llvm::ArrayRef<llvm::StringRef> logicalDtypes = {});

} // namespace vernon::compiler

#endif
