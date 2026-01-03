//===- VernonTypes.td - Vernon Type Definitions -----------*- tablegen -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#ifndef VERNON_TYPES
#define VERNON_TYPES

include "mlir/Dialect/Vernon/IR/VernonDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Vernon Tensor Type
//===----------------------------------------------------------------------===//

def Vernon_TensorType : TypeDef<Vernon_Dialect, "Tensor"> {
  let summary = "General n-dimensional tensor type";
  let description = [{
    A tensor type with support for n-dimensional shapes. This is the base
    tensor type that can represent tensors of any rank and shape.

    Example:
    ```mlir
    !vernon.tensor<2x3x4xf32>
    !vernon.tensor<?x?xf64>
    ```
  }];

  let mnemonic = "tensor";
  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape dimensions">:$shape,
    TypeParameter<"Type", "Element type">:$elementType
  );

  let assemblyFormat = "`<` $shape `x` $elementType `>`";
  
  let extraClassDeclaration = [{
    /// Get the rank of the tensor.
    unsigned getRank() const {
      return getShape().size();
    }

    /// Check if the tensor has a static shape (no dynamic dimensions).
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), 
        [](int64_t dim) { return ShapedType::isDynamic(dim); });
    }

    /// Get the total number of elements (if static shape).
    int64_t getNumElements() const {
      if (!hasStaticShape())
        return ShapedType::kDynamic;
      int64_t numElements = 1;
      for (int64_t dim : getShape())
        numElements *= dim;
      return numElements;
    }

  }];
}

//===----------------------------------------------------------------------===//
// Vector Type Aliases (1D tensors)
//===----------------------------------------------------------------------===//

def Vernon_Vec2Type : TypeDef<Vernon_Dialect, "Vec2"> {
  let summary = "2-element vector type alias";
  let description = [{
    A type alias for a 1-dimensional tensor with 2 elements.

    Example:
    ```mlir
    !vernon.vec2<f32>
    ```
  }];

  let mnemonic = "vec2";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

def Vernon_Vec3Type : TypeDef<Vernon_Dialect, "Vec3"> {
  let summary = "3-element vector type alias";
  let description = [{
    A type alias for a 1-dimensional tensor with 3 elements.

    Example:
    ```mlir
    !vernon.vec3<f32>
    ```
  }];

  let mnemonic = "vec3";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

def Vernon_Vec4Type : TypeDef<Vernon_Dialect, "Vec4"> {
  let summary = "4-element vector type alias";
  let description = [{
    A type alias for a 1-dimensional tensor with 4 elements.

    Example:
    ```mlir
    !vernon.vec4<f32>
    ```
  }];

  let mnemonic = "vec4";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Matrix Type Aliases (2D square tensors)
//===----------------------------------------------------------------------===//

def Vernon_Mat2Type : TypeDef<Vernon_Dialect, "Mat2"> {
  let summary = "2x2 matrix type alias";
  let description = [{
    A type alias for a 2-dimensional square tensor with 2x2 elements.

    Example:
    ```mlir
    !vernon.mat2<f32>
    ```
  }];

  let mnemonic = "mat2";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

def Vernon_Mat3Type : TypeDef<Vernon_Dialect, "Mat3"> {
  let summary = "3x3 matrix type alias";
  let description = [{
    A type alias for a 2-dimensional square tensor with 3x3 elements.

    Example:
    ```mlir
    !vernon.mat3<f32>
    ```
  }];

  let mnemonic = "mat3";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

def Vernon_Mat4Type : TypeDef<Vernon_Dialect, "Mat4"> {
  let summary = "4x4 matrix type alias";
  let description = [{
    A type alias for a 2-dimensional square tensor with 4x4 elements.

    Example:
    ```mlir
    !vernon.mat4<f32>
    ```
  }];

  let mnemonic = "mat4";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

#endif // VERNON_TYPES
