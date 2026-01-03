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
    The Vernon dialect defines types and operations for tensor computations
    with support for n-dimensional shapes and swizzling accessors.

    The dialect provides:
    - A general tensor type with n-dimensional shape support
    - Type aliases for common vector and matrix types (vec2, vec3, vec4, mat2, mat3, mat4)
    - Operations for component access and swizzling (.r, .g, .b, .a, .rg, .rgb, etc.)
  }];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "arith::ArithDialect",
    "tensor::TensorDialect",
    "vector::VectorDialect",
    "linalg::LinalgDialect"
  ];
}

#endif // VERNON_DIALECT
