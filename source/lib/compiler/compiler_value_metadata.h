#ifndef VERNON_COMPILER_VALUE_METADATA_H
#define VERNON_COMPILER_VALUE_METADATA_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "llvm/ADT/StringRef.h"

namespace vernon::compiler {

inline void copyValueMetadata(mlir::func::FuncOp source, unsigned sourceIndex, mlir::func::FuncOp target,
                              unsigned targetIndex) {
    for (llvm::StringRef name : {"vernon.source_name", "vernon.dtype", "vernon.source_shape", "vernon.abi_leaf_dtypes",
                                 "vernon.element_abi_leaf_dtypes"})
        if (mlir::Attribute value = source.getArgAttr(sourceIndex, name))
            target.setArgAttr(targetIndex, name, value);
    if (targetIndex < target.getFunctionType().getNumInputs() &&
        mlir::isa<mlir::vernon::TensorViewType>(target.getFunctionType().getInput(targetIndex)))
        if (mlir::Attribute elementDtypes = source.getArgAttr(sourceIndex, "vernon.element_abi_leaf_dtypes"))
            target.setArgAttr(targetIndex, "vernon.abi_leaf_dtypes", elementDtypes);
}

} // namespace vernon::compiler

#endif
