# Vernon MLIR Dialect

This directory contains the implementation of the Vernon MLIR dialect, which provides tensor types and operations with support for n-dimensional shapes and swizzling accessors.

## Structure

```
source/
├── include/mlir/Dialect/Vernon/IR/
│   ├── VernonDialect.td    # Dialect definition
│   ├── VernonTypes.td      # Type definitions (tensor, vec2-4, mat2-4)
│   ├── VernonOps.td        # Operation definitions (extract_component, swizzle)
│   ├── Vernon.h            # Main header file
│   └── CMakeLists.txt      # Tablegen configuration
├── lib/Dialect/Vernon/IR/
│   ├── VernonDialect.cpp   # Dialect initialization
│   ├── VernonTypes.cpp     # Type method implementations
│   ├── VernonOps.cpp       # Operation implementations and verifiers
│   └── CMakeLists.txt      # Library build configuration
└── CMakeLists.txt          # Top-level CMakeLists
```

## Types

### General Tensor Type
- `!vernon.tensor<shape x elementType>` - Supports n-dimensional shapes
- Example: `!vernon.tensor<2x3x4xf32>`, `!vernon.tensor<?x?xf64>`

### Vector Type Aliases
- `!vernon.vec2<elementType>` - 2-element vector
- `!vernon.vec3<elementType>` - 3-element vector
- `!vernon.vec4<elementType>` - 4-element vector

### Matrix Type Aliases
- `!vernon.mat2<elementType>` - 2x2 matrix
- `!vernon.mat3<elementType>` - 3x3 matrix
- `!vernon.mat4<elementType>` - 4x4 matrix

## Operations

### ExtractComponentOp
Extracts a single component from a vector/matrix:
```mlir
%x = vernon.extract_component %vec {component = "r"} : !vernon.vec4<f32> -> f32
```

### SwizzleOp
Extracts and reorders multiple components:
```mlir
%xy = vernon.swizzle %vec {pattern = "rg"} : !vernon.vec4<f32> -> !vernon.vec2<f32>
%xyz = vernon.swizzle %vec {pattern = "rgb"} : !vernon.vec4<f32> -> !vernon.vec3<f32>
```

### SwizzleRGBAOp
Convenience operation for RGBA swizzling with custom assembly format support.

## Building

To build this dialect, you need to:

1. Ensure MLIR is built and available
2. Add this directory to your CMake project:
   ```cmake
   add_subdirectory(source)
   ```
3. Link against the generated library:
   ```cmake
   target_link_libraries(your_target MLIRVernonDialect)
   ```

## Implementation Notes

- All type definitions use MLIR's TableGen framework
- Custom methods are implemented in the corresponding .cpp files
- Verifiers ensure type safety and correct component access
- The dialect depends on `MLIRArithDialect`, `MLIRIR`, and related interfaces

## Next Steps

1. Integrate this dialect into your MLIR build system
2. Add custom parsing/printing for the swizzle operations if desired
3. Implement lowering passes to convert Vernon operations to other dialects
4. Add more operations as needed for your use case
