#pragma once

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vernon {

struct StaticMatmulPlan {
    SmallVector<int64_t> leftShape;
    SmallVector<int64_t> rightShape;
    SmallVector<int64_t> batchShape;
    SmallVector<int64_t> resultShape;
    int64_t rows{};
    int64_t columns{};
    int64_t reduction{};
    bool leftVector{};
    bool rightVector{};
};

FailureOr<SmallVector<int64_t>> getStaticBroadcastShape(ArrayRef<int64_t> left, ArrayRef<int64_t> right);
FailureOr<int64_t> getStaticBroadcastLinearIndex(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> resultShape,
                                                 int64_t resultLinearIndex);
FailureOr<StaticMatmulPlan> getStaticMatmulPlan(ArrayRef<int64_t> left, ArrayRef<int64_t> right);
FailureOr<int64_t> getStaticMatmulLeftLinearIndex(const StaticMatmulPlan &plan, int64_t resultLinearIndex,
                                                  int64_t reductionIndex);
FailureOr<int64_t> getStaticMatmulRightLinearIndex(const StaticMatmulPlan &plan, int64_t resultLinearIndex,
                                                   int64_t reductionIndex);
FailureOr<int64_t> getStaticShapeElementCount(ArrayRef<int64_t> shape);

} // namespace mlir::vernon
