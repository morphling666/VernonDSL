//===- VernonDialect.cpp - Vernon Dialect Implementation -------*- C++ -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpBase.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::vernon;

#include "mlir/Dialect/Vernon/IR/VernonDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// VernonDialect Methods
//===----------------------------------------------------------------------===//

void VernonDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Vernon/IR/VernonTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Vernon/IR/VernonOps.cpp.inc"
      >();
}
