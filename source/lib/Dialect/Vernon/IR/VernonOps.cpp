//===- VernonOps.cpp - Vernon Operation Implementation --------*- C++ -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::vernon;

#define GET_OP_CLASSES
#include "mlir/Dialect/Vernon/IR/VernonOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {
/// Map component name to index.
/// Returns -1 if invalid component name.
int getComponentIndex(char component) {
  switch (component) {
  case 'r':
  case 'x':
    return 0;
  case 'g':
  case 'y':
    return 1;
  case 'b':
  case 'z':
    return 2;
  case 'a':
  case 'w':
    return 3;
  default:
    return -1;
  }
}

/// Get the maximum number of components for a given type.
unsigned getMaxComponents(Type type) {
  if (isa<Vec2Type>(type))
    return 2;
  if (isa<Vec3Type>(type))
    return 3;
  if (isa<Vec4Type>(type))
    return 4;
  if (isa<Mat2Type>(type))
    return 2;
  if (isa<Mat3Type>(type))
    return 3;
  if (isa<Mat4Type>(type))
    return 4;
  if (auto tensorType = dyn_cast<TensorType>(type)) {
    if (tensorType.getRank() > 0)
      return tensorType.getShape()[0];
  }
  return 0;
}

/// Get the element type from a Vernon type.
Type getElementType(Type type) {
  if (auto vec2Type = dyn_cast<Vec2Type>(type))
    return vec2Type.getElementType();
  if (auto vec3Type = dyn_cast<Vec3Type>(type))
    return vec3Type.getElementType();
  if (auto vec4Type = dyn_cast<Vec4Type>(type))
    return vec4Type.getElementType();
  if (auto mat2Type = dyn_cast<Mat2Type>(type))
    return mat2Type.getElementType();
  if (auto mat3Type = dyn_cast<Mat3Type>(type))
    return mat3Type.getElementType();
  if (auto mat4Type = dyn_cast<Mat4Type>(type))
    return mat4Type.getElementType();
  if (auto tensorType = dyn_cast<TensorType>(type))
    return tensorType.getElementType();
  return Type();
}

} // namespace

//===----------------------------------------------------------------------===//
// ExtractComponentOp
//===----------------------------------------------------------------------===//

int ExtractComponentOp::getComponentIndex() const {
  StringRef componentStr = getComponent();
  if (componentStr.size() != 1)
    return -1;
  return ::getComponentIndex(componentStr[0]);
}

LogicalResult ExtractComponentOp::verifyComponent() {
  StringRef componentStr = getComponent();
  if (componentStr.size() != 1) {
    return emitOpError("component must be a single character");
  }

  int index = getComponentIndex();
  if (index < 0) {
    return emitOpError("invalid component name '")
           << componentStr << "', expected one of: r, g, b, a, x, y, z, w";
  }

  Type sourceType = getSource().getType();
  unsigned maxComponents = ::getMaxComponents(sourceType);
  if (static_cast<unsigned>(index) >= maxComponents) {
    return emitOpError("component index ")
           << index
           << " is out of range for source type (max: " << (maxComponents - 1)
           << ")";
  }

  Type resultType = getResult().getType();
  Type expectedType = ::getElementType(sourceType);
  if (resultType != expectedType) {
    return emitOpError("result type ")
           << resultType << " does not match element type " << expectedType;
  }

  return success();
}

LogicalResult ExtractComponentOp::verify() { return verifyComponent(); }

//===----------------------------------------------------------------------===//
// SwizzleOp
//===----------------------------------------------------------------------===//

SmallVector<unsigned> SwizzleOp::getComponentIndices() const {
  SmallVector<unsigned> indices;
  StringRef pattern = getPattern();
  for (char c : pattern) {
    int index = ::getComponentIndex(c);
    if (index >= 0)
      indices.push_back(index);
  }
  return indices;
}

LogicalResult SwizzleOp::verifyPattern() {
  StringRef pattern = getPattern();
  if (pattern.empty()) {
    return emitOpError("pattern cannot be empty");
  }

  Type sourceType = getSource().getType();
  unsigned maxComponents = ::getMaxComponents(sourceType);

  // Verify all characters in pattern are valid
  for (char c : pattern) {
    int index = ::getComponentIndex(c);
    if (index < 0) {
      return emitOpError("invalid component character '")
             << c << "' in pattern, expected one of: r, g, b, a, x, y, z, w";
    }
    if (static_cast<unsigned>(index) >= maxComponents) {
      return emitOpError("component index ")
             << index << " in pattern is out of range for source type (max: "
             << (maxComponents - 1) << ")";
    }
  }

  // Verify result type matches pattern length
  Type resultType = getResult().getType();
  Type elementType = ::getElementType(sourceType);
  unsigned patternLength = pattern.size();

  // Determine expected result type based on pattern length
  Type expectedType;
  if (patternLength == 1) {
    expectedType = elementType; // Scalar
  } else if (patternLength == 2) {
    if (auto vec2Type = Vec2Type::get(getContext(), elementType))
      expectedType = vec2Type;
  } else if (patternLength == 3) {
    if (auto vec3Type = Vec3Type::get(getContext(), elementType))
      expectedType = vec3Type;
  } else if (patternLength == 4) {
    if (auto vec4Type = Vec4Type::get(getContext(), elementType))
      expectedType = vec4Type;
  } else {
    // For longer patterns, create a tensor type
    SmallVector<int64_t> shape = {static_cast<int64_t>(patternLength)};
    expectedType = TensorType::get(getContext(), shape, elementType);
  }

  if (resultType != expectedType) {
    return emitOpError("result type ")
           << resultType << " does not match expected type " << expectedType
           << " for pattern of length " << patternLength;
  }

  return success();
}

LogicalResult SwizzleOp::verify() { return verifyPattern(); }

//===----------------------------------------------------------------------===//
// SwizzleRGBAOp
//===----------------------------------------------------------------------===//

SmallVector<unsigned> SwizzleRGBAOp::getComponentIndices() const {
  SmallVector<unsigned> indices;
  StringRef pattern = getPattern();
  for (char c : pattern) {
    int index = ::getComponentIndex(c);
    if (index >= 0)
      indices.push_back(index);
  }
  return indices;
}

Type SwizzleRGBAOp::inferResultType(Type sourceType, StringRef pattern) {
  if (pattern.empty())
    return Type();

  Type elementType = ::getElementType(sourceType);
  if (!elementType)
    return Type();

  unsigned patternLength = pattern.size();
  MLIRContext *context = sourceType.getContext();

  if (patternLength == 1) {
    return elementType; // Scalar
  } else if (patternLength == 2) {
    return Vec2Type::get(context, elementType);
  } else if (patternLength == 3) {
    return Vec3Type::get(context, elementType);
  } else if (patternLength == 4) {
    return Vec4Type::get(context, elementType);
  } else {
    // For longer patterns, create a tensor type
    SmallVector<int64_t> shape = {static_cast<int64_t>(patternLength)};
    return TensorType::get(context, shape, elementType);
  }
}

LogicalResult SwizzleRGBAOp::verify() {
  StringRef pattern = getPattern();
  if (pattern.empty()) {
    return emitOpError("pattern cannot be empty");
  }

  Type sourceType = getSource().getType();
  unsigned maxComponents = ::getMaxComponents(sourceType);

  // Verify all characters in pattern are valid
  for (char c : pattern) {
    int index = ::getComponentIndex(c);
    if (index < 0) {
      return emitOpError("invalid component character '")
             << c << "' in pattern, expected one of: r, g, b, a, x, y, z, w";
    }
    if (static_cast<unsigned>(index) >= maxComponents) {
      return emitOpError("component index ")
             << index << " in pattern is out of range for source type (max: "
             << (maxComponents - 1) << ")";
    }
  }

  // Verify result type matches inferred type
  Type resultType = getResult().getType();
  Type inferredType = inferResultType(sourceType, pattern);
  if (resultType != inferredType) {
    return emitOpError("result type ")
           << resultType << " does not match inferred type " << inferredType;
  }

  return success();
}
