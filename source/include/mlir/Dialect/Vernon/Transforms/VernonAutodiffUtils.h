#pragma once

#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace mlir::vernon {

// Scalar dtype policy: f16/f32 → f32, f64 → f64.
FailureOr<Type> getAutodiffDerivativeScalarType(Type scalarType);
// Value payload derivative (not TensorView). Shaped Values (Tensor, ranked
// tensor, vector) keep their constructor and map the element. Products and
// scalars map differentiable ABI leaves. `module` resolves named structs.
FailureOr<Type> getAutodiffDerivativeValueType(Type valueType, ModuleOp module = {});
// TensorView envelope around a payload derivative. Shape stays; `access` is the resource ABI.
Type wrapAutodiffDerivativeTensorView(TensorViewType view, Type payload, StringRef access);
// Any SSA type: TensorView resource or Value.
FailureOr<Type> getAutodiffDerivativeType(Type type, ModuleOp module);
FailureOr<ValueAbiLayout> getAutodiffDerivativeValueLayout(Type primalType, Type derivativeType, ModuleOp module,
                                                           ArrayRef<StringRef> logicalLeafDtypes = {});

inline SmallVector<SmallVector<int64_t>> enumerateStaticCoordinates(ArrayRef<int64_t> shape) {
    SmallVector<SmallVector<int64_t>> coordinates(1);
    for (int64_t extent : shape) {
        SmallVector<SmallVector<int64_t>> expanded;
        for (const SmallVector<int64_t> &prefix : coordinates)
            for (int64_t index = 0; index < extent; ++index) {
                SmallVector<int64_t> coordinate(prefix);
                coordinate.push_back(index);
                expanded.push_back(std::move(coordinate));
            }
        coordinates = std::move(expanded);
    }
    return coordinates;
}

} // namespace mlir::vernon
