//===- VernonProgram.h - Program scheduling dialect -------------*- C++ -*-===//

#ifndef MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
#define MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

#include "mlir/Dialect/VernonProgram/IR/VernonProgramDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/VernonProgram/IR/VernonProgramOps.h.inc"

namespace mlir::vernon::program {

inline constexpr llvm::StringLiteral kCaptureForwardValueAttr = "vernon_program.capture_forward_value";
inline constexpr llvm::StringLiteral kOperandAccessesAttrName = "vernon_program.operand_accesses";

inline llvm::StringRef getProgramOperandAccess(Operation *operation, unsigned index) {
    auto accesses = operation->getAttrOfType<mlir::ArrayAttr>(kOperandAccessesAttrName);
    if (!accesses || index >= accesses.size())
        return {};
    if (auto name = mlir::dyn_cast<mlir::StringAttr>(accesses[index]))
        return name.getValue();
    return {};
}

/// Write-only DPS destinations are Storage objectives (kernel `outputs`), not
/// local VJP `wrt`. Nested kernel bind keeps the callee's parameter names.
inline bool isWriteOnlyProgramOperand(Operation *operation, unsigned index) {
    return getProgramOperandAccess(operation, index) == "write";
}

inline bool isProgramAllocIntrinsicName(llvm::StringRef name) {
    return name == "empty" || name == "empty_like" || name == "zeros" || name == "zeros_like" || name == "from_values";
}

inline bool isLikeAllocIntrinsicName(llvm::StringRef name) { return name == "empty_like" || name == "zeros_like"; }

inline bool isStorageAllocIntrinsic(Operation *operation) {
    auto intrinsic = dyn_cast<mlir::vernon::IntrinsicOp>(operation);
    if (!intrinsic || intrinsic.getNumResults() != 1)
        return false;
    if (!isa<mlir::vernon::TensorViewType>(intrinsic.getResult().getType()))
        return false;
    return isProgramAllocIntrinsicName(intrinsic.getName());
}

/// Language ABI attached to a Program SSA value. Allocations stay storage, not
/// compute nodes; this metadata is how reflection and kernel matching see them.
struct ProgramLanguageAbi {
    std::optional<StringRef> dtype;
    SmallVector<StringRef, 4> leaves;
};

inline void appendProgramLanguageLeaves(ArrayAttr dtypes, SmallVectorImpl<StringRef> &leaves) {
    if (!dtypes)
        return;
    for (Attribute attribute : dtypes)
        if (auto dtype = dyn_cast<StringAttr>(attribute))
            leaves.push_back(dtype.getValue());
}

inline ProgramLanguageAbi programLanguageAbiFromAttrs(DictionaryAttr attrs, Type type) {
    ProgramLanguageAbi abi;
    if (!attrs)
        return abi;
    if (isa<TensorViewType>(type))
        appendProgramLanguageLeaves(attrs.getAs<ArrayAttr>("vernon.element_abi_leaf_dtypes"), abi.leaves);
    if (abi.leaves.empty())
        appendProgramLanguageLeaves(attrs.getAs<ArrayAttr>("vernon.abi_leaf_dtypes"), abi.leaves);
    if (auto dtype = attrs.getAs<StringAttr>("vernon.dtype"); dtype && !dtype.getValue().empty())
        abi.dtype = dtype.getValue();
    if (abi.leaves.empty() && abi.dtype)
        abi.leaves.push_back(*abi.dtype);
    if (!abi.dtype && abi.leaves.size() == 1)
        abi.dtype = abi.leaves.front();
    return abi;
}

inline ProgramLanguageAbi getProgramValueLanguageAbi(Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp()))
            return programLanguageAbiFromAttrs(function.getArgAttrDict(argument.getArgNumber()), value.getType());
        return {};
    }
    Operation *operation = value.getDefiningOp();
    if (!operation)
        return {};
    const unsigned index = cast<OpResult>(value).getResultNumber();
    if (auto resultDtypes = operation->getAttrOfType<ArrayAttr>("vernon_program.result_abi_leaf_dtypes"))
        if (index < resultDtypes.size())
            if (auto leaves = dyn_cast<ArrayAttr>(resultDtypes[index])) {
                ProgramLanguageAbi abi;
                appendProgramLanguageLeaves(leaves, abi.leaves);
                if (abi.leaves.size() == 1)
                    abi.dtype = abi.leaves.front();
                if (!abi.leaves.empty())
                    return abi;
            }
    return programLanguageAbiFromAttrs(operation->getAttrDictionary(), value.getType());
}

inline ArrayAttr programLanguageAbiLeafArray(MLIRContext *context, const ProgramLanguageAbi &abi) {
    SmallVector<Attribute> leaves;
    for (StringRef dtype : abi.leaves)
        leaves.push_back(StringAttr::get(context, dtype));
    if (leaves.empty() && abi.dtype)
        leaves.push_back(StringAttr::get(context, *abi.dtype));
    return ArrayAttr::get(context, leaves);
}

inline void applyProgramLanguageAbi(Operation *operation, const ProgramLanguageAbi &abi) {
    ArrayAttr leaves = programLanguageAbiLeafArray(operation->getContext(), abi);
    if (leaves.empty())
        return;
    operation->setAttr("vernon.abi_leaf_dtypes", leaves);
    operation->setAttr("vernon_program.result_abi_leaf_dtypes",
                       ArrayAttr::get(operation->getContext(), ArrayRef<Attribute>{leaves}));
    if (operation->getNumResults() == 1 && isa<TensorViewType>(operation->getResult(0).getType()))
        operation->setAttr("vernon.element_abi_leaf_dtypes", leaves);
    if (leaves.size() == 1)
        operation->setAttr("vernon.dtype", leaves[0]);
}

inline void applyProgramLanguageAbi(func::FuncOp function, unsigned index, const ProgramLanguageAbi &abi, bool result) {
    ArrayAttr leaves = programLanguageAbiLeafArray(function.getContext(), abi);
    if (leaves.empty())
        return;
    const auto setAttr = [&](StringRef name, Attribute value) {
        if (result)
            function.setResultAttr(index, name, value);
        else
            function.setArgAttr(index, name, value);
    };
    setAttr("vernon.abi_leaf_dtypes", leaves);
    Type type = result ? function.getFunctionType().getResult(index) : function.getArgument(index).getType();
    if (isa<TensorViewType>(type))
        setAttr("vernon.element_abi_leaf_dtypes", leaves);
    if (leaves.size() == 1)
        setAttr("vernon.dtype", leaves[0]);
}

} // namespace mlir::vernon::program

#endif // MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
