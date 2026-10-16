#ifndef VERNON_LIB_DIALECT_VERNON_TRANSFORMS_VERNONTENSORELEMENTACCESS_H
#define VERNON_LIB_DIALECT_VERNON_TRANSFORMS_VERNONTENSORELEMENTACCESS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"

#include <cstdint>
#include <optional>

namespace mlir::vernon {

/// The row-major element index of `indices` into `shape` when every index is an in-bounds constant.
std::optional<int64_t> constantLinearIndex(ArrayRef<int64_t> shape, ValueRange indices);

/// The row-major element index of `indices` into `shape` as an index value; `indices` must be non-empty.
Value linearTensorIndex(Location location, ArrayRef<int64_t> shape, ValueRange indices, OpBuilder &builder);

/// Reads element `linear` of an `!llvm.array` register value through a function-local copy, because LLVM indexes an
/// aggregate in place only through a pointer; clang lowers an indexed rvalue array the same way. The copy is allocated
/// in the entry block of the function enclosing `op` so it is not re-allocated per loop iteration.
FailureOr<Value> extractLLVMArrayElement(Operation *op, Value array, Value linear, RewriterBase &rewriter);

} // namespace mlir::vernon

#endif
