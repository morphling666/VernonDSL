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
// TensorView envelope around a payload derivative. Shape stays; `access` is the
// resource ABI (`read` for cotangents, `write` for gradient dests).
Type wrapAutodiffDerivativeTensorView(TensorViewType view, Type payload, StringRef access);
// Any SSA type: TensorView resource or Value. The 2-argument overload keeps the
// primal TensorView access; the 3-argument form sets the AD resource ABI.
FailureOr<Type> getAutodiffDerivativeType(Type type, ModuleOp module);
FailureOr<Type> getAutodiffDerivativeType(Type type, ModuleOp module, StringRef tensorViewAccess);
// Canonical dotted spelling shared by analysis and Program boundary matching.
std::string appendValueAbiPath(StringRef root, ArrayRef<ValueAbiPathComponent> path);
// Project primal logical dtypes onto the differentiable ABI leaves, in the
// same order used by the derivative Value type.
FailureOr<SmallVector<StringRef>> getAutodiffDerivativeLogicalLeafDtypes(Type primalType, ModuleOp module,
                                                                         ArrayRef<StringRef> logicalLeafDtypes);
// Physical Value ABI of a derivative payload. Validates that `derivativeType` is
// a legal tangent of `primalType`. `layout_hash` uses the same pipeline Value ABI
// as ordinary values; AD pairing is origin/signature, not a `tangent<>` identity.
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
