//===- VernonTypes.cpp - Vernon Type Implementation -----------*- C++ -*-===//
//
// Part of the Vernon DSL Project
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::vernon;

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Vernon/IR/VernonTypes.cpp.inc"

namespace {
bool isStructurallyAbiStableValue(Type type) {
    if (type.isInteger(1) || type.isInteger(32) || type.isF16() || type.isF32() || type.isF64())
        return true;
    if (isa<StructType>(type))
        return true; // Named fields are resolved and cycle-checked by the module ABI planner.
    if (auto tuple = dyn_cast<TupleType>(type))
        return llvm::all_of(tuple.getTypes(), isStructurallyAbiStableValue);
    Type element;
    ArrayRef<int64_t> shape;
    if (auto tensor = dyn_cast<mlir::vernon::TensorType>(type)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else if (auto vector = dyn_cast<VectorType>(type)) {
        if (vector.isScalable())
            return false;
        element = vector.getElementType();
        shape = vector.getShape();
    } else {
        return false;
    }
    return !shape.empty() && llvm::all_of(shape, [](int64_t extent) { return extent > 0; }) &&
           isStructurallyAbiStableValue(element);
}
} // namespace

LogicalResult TensorViewType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType,
                                     ArrayRef<int64_t> shape, StringRef access, StringRef addressSpace) {
    if (!isStructurallyAbiStableValue(elementType) || shape.empty())
        return emitError() << "TensorView requires an ABI-stable Value element and non-empty shape";
    if (llvm::any_of(shape, [](int64_t extent) { return extent == 0 || extent < -1; }))
        return emitError() << "TensorView dimensions must be positive or -1 for dynamic";
    if (access != "read" && access != "write" && access != "read_write")
        return emitError() << "TensorView access must be read, write, or read_write";
    if (addressSpace != "device" && addressSpace != "workgroup" && addressSpace != "private")
        return emitError() << "TensorView address space must be device, workgroup, or private";
    if (addressSpace != "device" && llvm::is_contained(shape, int64_t{-1}))
        return emitError() << "workgroup and private TensorViews require static shapes";
    return success();
}

LogicalResult AdAdjointBufferType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType,
                                          ArrayRef<int64_t> shape, unsigned indexRank) {
    if (!isa<FloatType>(elementType))
        return emitError() << "adjoint buffer requires a floating scalar element type";
    if (shape.empty() || llvm::any_of(shape, [](int64_t extent) { return extent <= 0; }))
        return emitError() << "adjoint buffer requires a positive static flattened shape";
    if (indexRank == 0 || indexRank > shape.size())
        return emitError() << "adjoint buffer index rank must select a non-empty shape prefix";
    return success();
}

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
