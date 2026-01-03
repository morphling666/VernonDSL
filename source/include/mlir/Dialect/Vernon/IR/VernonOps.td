//===- VernonOps.td - Vernon Operation Definitions ---------*- tablegen -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#ifndef VERNON_OPS
#define VERNON_OPS

include "mlir/Dialect/Vernon/IR/VernonDialect.td"
include "mlir/Dialect/Vernon/IR/VernonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Base operation class for Vernon dialect
//===----------------------------------------------------------------------===//

class Vernon_Op<string mnemonic, list<Trait> traits = []>
    : Op<Vernon_Dialect, mnemonic, traits>;
//class Vernon_Op<string mnemonic, list<Trait> traits = []>
//    : Op<Vernon_Dialect, mnemonic, traits> {
//  let dialect = Vernon_Dialect;
//}

//===----------------------------------------------------------------------===//
// Component Access Operations
//===----------------------------------------------------------------------===//

def Vernon_ExtractComponentOp : Vernon_Op<"extract_component", [Pure]> {
  let summary = "Extract a single component from a vector/matrix";
  let description = [{
    Extracts a single component from a vector or matrix type using a component
    name. Supported component names are:
    - For vectors: 'r'/'x' (0), 'g'/'y' (1), 'b'/'z' (2), 'a'/'w' (3)
    - For matrices: 'r'/'x' (row 0), 'g'/'y' (row 1), 'b'/'z' (row 2), 'a'/'w' (row 3)

    Example:
    ```mlir
    %x = vernon.extract_component %vec {component = "r"} : !vernon.vec4<f32> -> f32
    %y = vernon.extract_component %vec {component = "g"} : !vernon.vec4<f32> -> f32
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[Vernon_Vec2Type, Vernon_Vec3Type, Vernon_Vec4Type, 
               Vernon_Mat2Type, Vernon_Mat3Type, Vernon_Mat4Type, Vernon_TensorType]>:$source,
    StrAttr:$component
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];

  let extraClassDeclaration = [{
    /// Get the component index from the component name.
    /// Returns -1 if invalid component name.
    int getComponentIndex() const;

    /// Verify that the component name is valid for the source type.
    LogicalResult verifyComponent();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Swizzle Operations
//===----------------------------------------------------------------------===//

def Vernon_SwizzleOp : Vernon_Op<"swizzle", [Pure]> {
  let summary = "Swizzle multiple components from a vector/matrix";
  let description = [{
    Extracts and reorders multiple components from a vector or matrix type using
    a swizzle pattern. The pattern is a string of component names where each
    character represents a component to extract.

    Supported component names:
    - 'r'/'x' (component 0)
    - 'g'/'y' (component 1)
    - 'b'/'z' (component 2)
    - 'a'/'w' (component 3)

    The result type depends on the number of components in the swizzle pattern.
    For example:
    - ".r" or ".x" -> scalar (element type)
    - ".rg" or ".xy" -> !vernon.vec2<element_type>
    - ".rgb" or ".xyz" -> !vernon.vec3<element_type>
    - ".rgba" or ".xyzw" -> !vernon.vec4<element_type>

    Example:
    ```mlir
    %xy = vernon.swizzle %vec {pattern = "rg"} : !vernon.vec4<f32> -> !vernon.vec2<f32>
    %xyz = vernon.swizzle %vec {pattern = "rgb"} : !vernon.vec4<f32> -> !vernon.vec3<f32>
    %zyx = vernon.swizzle %vec {pattern = "bgr"} : !vernon.vec4<f32> -> !vernon.vec3<f32>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[Vernon_Vec2Type, Vernon_Vec3Type, Vernon_Vec4Type,
               Vernon_Mat2Type, Vernon_Mat3Type, Vernon_Mat4Type, Vernon_TensorType]>:$source,
    StrAttr:$pattern
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];

  let extraClassDeclaration = [{
    /// Get the number of components in the swizzle pattern.
    unsigned getNumComponents() const {
      return getPattern().size();
    }

    /// Get the component indices from the swizzle pattern.
    SmallVector<unsigned> getComponentIndices() const;

    /// Verify that the swizzle pattern is valid for the source type.
    LogicalResult verifyPattern();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Convenience Operations (with custom assembly format)
//===----------------------------------------------------------------------===//

def Vernon_SwizzleRGBAOp : Vernon_Op<"swizzle_rgba", [Pure, 
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Convenience operation for RGBA swizzling";
  let description = [{
    A convenience operation that supports direct swizzling using the dot notation
    syntax like .r, .g, .rg, .rgb, .rgba, etc. This operation is designed to
    be used with custom parsing/printing to support the natural syntax.

    Example (with custom parser):
    ```mlir
    %xy = %vec.rg : !vernon.vec4<f32> -> !vernon.vec2<f32>
    %xyz = %vec.rgb : !vernon.vec4<f32> -> !vernon.vec3<f32>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[Vernon_Vec2Type, Vernon_Vec3Type, Vernon_Vec4Type,
               Vernon_Mat2Type, Vernon_Mat3Type, Vernon_Mat4Type, Vernon_TensorType]>:$source,
    StrAttr:$pattern
  );

  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the number of components in the swizzle pattern.
    unsigned getNumComponents() const {
      return getPattern().size();
    }

    /// Get the component indices from the swizzle pattern.
    SmallVector<unsigned> getComponentIndices() const;

    /// Infer the result type from the source type and pattern.
    static Type inferResultType(Type sourceType, StringRef pattern);
  }];

  let hasVerifier = 1;
}

#endif // VERNON_OPS
