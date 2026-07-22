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

std::optional<unsigned> mlir::vernon::decodeSwizzleComponent(char component) {
  switch (component) {
  case 'x':
  case 'r':
    return 0;
  case 'y':
  case 'g':
    return 1;
  case 'z':
  case 'b':
    return 2;
  case 'w':
  case 'a':
    return 3;
  default:
    return std::nullopt;
  }
}

LogicalResult SwizzleOp::verify() {
  StringRef mask = getMask();
  if (mask.empty())
    return emitOpError("requires a non-empty component mask");

  Type inputElementType;
  int64_t inputWidth = 0;
  if (auto tensor = dyn_cast<RankedTensorType>(getInput().getType())) {
    if (tensor.getRank() != 1)
      return emitOpError("requires a rank-one tensor or vector input");
    if (tensor.isDynamicDim(0))
      return emitOpError("requires a statically sized component axis");
    inputElementType = tensor.getElementType();
    inputWidth = tensor.getDimSize(0);
  } else if (auto vector = dyn_cast<VectorType>(getInput().getType())) {
    if (vector.getRank() != 1)
      return emitOpError("requires a rank-one tensor or vector input");
    inputElementType = vector.getElementType();
    inputWidth = vector.getDimSize(0);
  } else {
    return emitOpError("requires a rank-one tensor or vector input");
  }

  for (char component : mask) {
    std::optional<unsigned> index = decodeSwizzleComponent(component);
    if (!index)
      return emitOpError() << "contains invalid component '" << component
                           << "'; expected an XYZW or RGBA alias";
    if (*index >= static_cast<uint64_t>(inputWidth))
      return emitOpError() << "component '" << component
                           << "' is out of bounds for input width "
                           << inputWidth;
  }

  Type resultType = getResult().getType();
  if (mask.size() == 1) {
    if (resultType != inputElementType)
      return emitOpError()
             << "single-component result must have input element type "
             << inputElementType;
    return success();
  }

  Type resultElementType;
  int64_t resultWidth = 0;
  if (auto tensor = dyn_cast<RankedTensorType>(resultType)) {
    if (tensor.getRank() != 1 || tensor.isDynamicDim(0))
      return emitOpError(
          "multi-component result must be a statically sized rank-one tensor "
          "or vector");
    resultElementType = tensor.getElementType();
    resultWidth = tensor.getDimSize(0);
  } else if (auto vector = dyn_cast<VectorType>(resultType)) {
    if (vector.getRank() != 1)
      return emitOpError(
          "multi-component result must be a statically sized rank-one tensor "
          "or vector");
    resultElementType = vector.getElementType();
    resultWidth = vector.getDimSize(0);
  } else {
    return emitOpError(
        "multi-component result must be a statically sized rank-one tensor "
        "or vector");
  }
  if (resultWidth != static_cast<int64_t>(mask.size()))
    return emitOpError() << "result width " << resultWidth
                         << " does not match mask length " << mask.size();
  if (resultElementType != inputElementType)
    return emitOpError() << "result element type " << resultElementType
                         << " does not match input element type "
                         << inputElementType;
  return success();
}

LogicalResult IntrinsicOp::verify() {
  if (getNameAttr().getValue().empty())
    return emitOpError("requires a non-empty intrinsic name");
  if (getName() != "texture_sample" && getName() != "texture_size")
    return success();

  if (getNumResults() != 1)
    return emitOpError() << getName() << " requires exactly one result";
  if (getNumOperands() == 0)
    return emitOpError() << getName() << " requires a texture operand";

  auto texture = dyn_cast<TextureType>(getOperand(0).getType());
  if (!texture)
    return emitOpError() << getName() << " operand #0 must be a Vernon texture";

  auto shapedWidthAndElement =
      [](Type type) -> std::optional<std::pair<int64_t, Type>> {
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      if (tensor.getRank() != 1 || tensor.isDynamicDim(0))
        return std::nullopt;
      return std::pair<int64_t, Type>{tensor.getDimSize(0),
                                      tensor.getElementType()};
    }
    if (auto vector = dyn_cast<VectorType>(type)) {
      if (vector.getRank() != 1)
        return std::nullopt;
      return std::pair<int64_t, Type>{vector.getDimSize(0),
                                      vector.getElementType()};
    }
    return std::nullopt;
  };

  if (getName() == "texture_size") {
    if (getNumOperands() < 1 || getNumOperands() > 2)
      return emitOpError(
          "texture_size requires a texture and an optional integer lod");
    if (getNumOperands() == 2 && !getOperand(1).getType().isSignlessInteger(32))
      return emitOpError("texture_size lod must be a 32-bit integer scalar");
    const int64_t expectedResultWidth = texture.getDimension() == "3d" ? 3 : 2;
    auto result = shapedWidthAndElement(getResult().getType());
    if (!result || result->first != expectedResultWidth ||
        !result->second.isSignlessInteger(32))
      return emitOpError()
             << "texture_size result must be a statically sized rank-one "
             << expectedResultWidth << "-component i32 tensor or vector";
    return success();
  }

  if (getNumOperands() < 3 || getNumOperands() > 4)
    return emitOpError(
        "texture_sample requires texture, sampler, coordinates, and optional "
        "lod");
  if (!isa<SamplerType>(getOperand(1).getType()))
    return emitOpError("texture_sample operand #1 must be a Vernon sampler");

  std::optional<std::pair<int64_t, Type>> coordinates =
      shapedWidthAndElement(getOperand(2).getType());
  const int64_t expectedCoordinateWidth =
      texture.getDimension() == "2d" ? 2 : 3;
  if (!coordinates || coordinates->first != expectedCoordinateWidth ||
      coordinates->second != texture.getElementType() ||
      !isa<FloatType>(coordinates->second))
    return emitOpError()
           << "texture_sample operand #2 must be a statically sized rank-one "
           << expectedCoordinateWidth << "-component "
           << texture.getElementType() << " tensor or vector";
  if (getNumOperands() == 4 && !isa<FloatType>(getOperand(3).getType()))
    return emitOpError(
        "texture_sample operand #3 lod must be a floating-point scalar");

  std::optional<std::pair<int64_t, Type>> result =
      shapedWidthAndElement(getResult().getType());
  if (!result || result->first != 4 ||
      result->second != texture.getElementType())
    return emitOpError()
           << "texture_sample result must be a statically sized rank-one "
              "4-component "
           << texture.getElementType() << " tensor or vector";
  return success();
}
