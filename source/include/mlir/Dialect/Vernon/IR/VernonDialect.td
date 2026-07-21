//===- VernonDialect.td - Vernon Dialect Definition ---------*- tablegen -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#ifndef VERNON_DIALECT
#define VERNON_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Vernon Dialect
//===----------------------------------------------------------------------===//

def Vernon_Dialect : Dialect {
  let name = "vernon";
  let cppNamespace = "::mlir::vernon";

  let description = [{
    The Vernon dialect defines shader stage and interface semantics around
    MLIR-native computation.

    The dialect provides:
    - Stable stage and interface annotations for function boundaries
    - Semantic attributes used by validation and future lowering
    - Standard tensor/vector/linalg types and operations for ordinary values
  }];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
    "vector::VectorDialect",
    "linalg::LinalgDialect"
  ];
}

#endif // VERNON_DIALECT
