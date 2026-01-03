//===- VernonTypes.cpp - Vernon Type Implementation -----------*- C++ -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::vernon;

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Vernon/IR/VernonTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TensorType Implementation
//===----------------------------------------------------------------------===//

unsigned TensorType::getRank() const { return getShape().size(); }

bool TensorType::hasStaticShape() const {
  return llvm::none_of(getShape(),
                       [](int64_t dim) { return ShapedType::isDynamic(dim); });
}

int64_t TensorType::getNumElements() const {
  if (!hasStaticShape())
    return ShapedType::kDynamic;
  int64_t numElements = 1;
  for (int64_t dim : getShape())
    numElements *= dim;
  return numElements;
}
