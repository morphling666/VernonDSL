//===- Vernon.h - Vernon dialect ---------------------------------*- C++
//-*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VERNON_IR_VERNON_H_
#define MLIR_DIALECT_VERNON_IR_VERNON_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Vernon Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vernon/IR/VernonDialect.h.inc"

//===----------------------------------------------------------------------===//
// Vernon Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Vernon/IR/VernonTypes.h.inc"

//===----------------------------------------------------------------------===//
// Vernon Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Vernon/IR/VernonOps.h.inc"

#endif // MLIR_DIALECT_VERNON_IR_VERNON_H_
