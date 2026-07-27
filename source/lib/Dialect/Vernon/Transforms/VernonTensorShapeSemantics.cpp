#include "mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h"

#include <limits>

namespace mlir::vernon {
namespace {

FailureOr<SmallVector<int64_t>> decodeLinearIndex(ArrayRef<int64_t> shape, int64_t linearIndex) {
    FailureOr<int64_t> count = getStaticShapeElementCount(shape);
    if (failed(count) || linearIndex < 0 || linearIndex >= *count)
        return failure();
    SmallVector<int64_t> coordinates(shape.size());
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        coordinates[dimension] = linearIndex % shape[dimension];
        linearIndex /= shape[dimension];
    }
    return coordinates;
}

FailureOr<int64_t> encodeLinearIndex(ArrayRef<int64_t> shape, ArrayRef<int64_t> coordinates) {
    if (shape.size() != coordinates.size())
        return failure();
    int64_t result = 0;
    for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
        const int64_t extent = shape[dimension];
        const int64_t coordinate = coordinates[dimension];
        if (extent <= 0 || coordinate < 0 || coordinate >= extent ||
            result > (std::numeric_limits<int64_t>::max() - coordinate) / extent)
            return failure();
        result = result * extent + coordinate;
    }
    return result;
}

FailureOr<SmallVector<int64_t>> mapBatchCoordinates(ArrayRef<int64_t> resultBatch, ArrayRef<int64_t> inputBatch,
                                                    ArrayRef<int64_t> resultCoordinates) {
    if (resultCoordinates.size() < resultBatch.size() || inputBatch.size() > resultBatch.size())
        return failure();
    SmallVector<int64_t> coordinates;
    coordinates.reserve(inputBatch.size());
    const size_t offset = resultBatch.size() - inputBatch.size();
    for (size_t dimension = 0; dimension < inputBatch.size(); ++dimension) {
        const int64_t extent = inputBatch[dimension];
        const int64_t coordinate = resultCoordinates[offset + dimension];
        if (extent != 1 && extent != resultBatch[offset + dimension])
            return failure();
        coordinates.push_back(extent == 1 ? 0 : coordinate);
    }
    return coordinates;
}

struct MatmulCoordinates {
    SmallVector<int64_t> result;
    int64_t row{};
    int64_t column{};
};

FailureOr<MatmulCoordinates> getMatmulCoordinates(const StaticMatmulPlan &plan, int64_t resultLinearIndex) {
    FailureOr<SmallVector<int64_t>> result = decodeLinearIndex(plan.resultShape, resultLinearIndex);
    if (failed(result))
        return failure();
    MatmulCoordinates coordinates{*result, 0, 0};
    size_t cursor = plan.batchShape.size();
    if (!plan.leftVector)
        coordinates.row = (*result)[cursor++];
    if (!plan.rightVector)
        coordinates.column = (*result)[cursor++];
    if (cursor != result->size())
        return failure();
    return coordinates;
}

} // namespace

FailureOr<int64_t> getStaticShapeElementCount(ArrayRef<int64_t> shape) {
    int64_t result = 1;
    for (int64_t extent : shape) {
        if (extent <= 0 || result > std::numeric_limits<int64_t>::max() / extent)
            return failure();
        result *= extent;
    }
    return result;
}

FailureOr<SmallVector<int64_t>> getStaticBroadcastShape(ArrayRef<int64_t> left, ArrayRef<int64_t> right) {
    const size_t rank = std::max(left.size(), right.size());
    SmallVector<int64_t> result(rank, 1);
    for (size_t reverse = 0; reverse < rank; ++reverse) {
        const int64_t leftExtent = reverse < left.size() ? left[left.size() - 1 - reverse] : 1;
        const int64_t rightExtent = reverse < right.size() ? right[right.size() - 1 - reverse] : 1;
        if (leftExtent <= 0 || rightExtent <= 0 || (leftExtent != rightExtent && leftExtent != 1 && rightExtent != 1))
            return failure();
        result[rank - 1 - reverse] = std::max(leftExtent, rightExtent);
    }
    return result;
}

FailureOr<int64_t> getStaticBroadcastLinearIndex(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> resultShape,
                                                 int64_t resultLinearIndex) {
    FailureOr<SmallVector<int64_t>> broadcast = getStaticBroadcastShape(inputShape, resultShape);
    FailureOr<SmallVector<int64_t>> coordinates = decodeLinearIndex(resultShape, resultLinearIndex);
    if (failed(broadcast) || failed(coordinates) || *broadcast != resultShape || inputShape.size() > resultShape.size())
        return failure();
    SmallVector<int64_t> inputCoordinates;
    const size_t offset = resultShape.size() - inputShape.size();
    for (size_t dimension = 0; dimension < inputShape.size(); ++dimension)
        inputCoordinates.push_back(inputShape[dimension] == 1 ? 0 : (*coordinates)[offset + dimension]);
    return encodeLinearIndex(inputShape, inputCoordinates);
}

FailureOr<StaticMatmulPlan> getStaticMatmulPlan(ArrayRef<int64_t> left, ArrayRef<int64_t> right) {
    if (left.empty() || right.empty())
        return failure();
    if (failed(getStaticShapeElementCount(left)) || failed(getStaticShapeElementCount(right)))
        return failure();

    StaticMatmulPlan plan;
    plan.leftShape.assign(left.begin(), left.end());
    plan.rightShape.assign(right.begin(), right.end());
    plan.leftVector = left.size() == 1;
    plan.rightVector = right.size() == 1;
    plan.rows = plan.leftVector ? 1 : left[left.size() - 2];
    plan.reduction = left.back();
    const int64_t rightReduction = plan.rightVector ? right.front() : right[right.size() - 2];
    plan.columns = plan.rightVector ? 1 : right.back();
    if (plan.reduction != rightReduction)
        return failure();

    ArrayRef<int64_t> leftBatch = plan.leftVector ? ArrayRef<int64_t>() : left.drop_back(2);
    ArrayRef<int64_t> rightBatch = plan.rightVector ? ArrayRef<int64_t>() : right.drop_back(2);
    FailureOr<SmallVector<int64_t>> batch = getStaticBroadcastShape(leftBatch, rightBatch);
    if (failed(batch))
        return failure();
    plan.batchShape = *batch;
    plan.resultShape = plan.batchShape;
    if (!plan.leftVector)
        plan.resultShape.push_back(plan.rows);
    if (!plan.rightVector)
        plan.resultShape.push_back(plan.columns);
    return plan;
}

FailureOr<int64_t> getStaticMatmulLeftLinearIndex(const StaticMatmulPlan &plan, int64_t resultLinearIndex,
                                                  int64_t reductionIndex) {
    if (reductionIndex < 0 || reductionIndex >= plan.reduction)
        return failure();
    FailureOr<MatmulCoordinates> result = getMatmulCoordinates(plan, resultLinearIndex);
    if (failed(result))
        return failure();
    if (plan.leftVector)
        return reductionIndex;
    ArrayRef<int64_t> leftBatch = ArrayRef<int64_t>(plan.leftShape).drop_back(2);
    FailureOr<SmallVector<int64_t>> coordinates = mapBatchCoordinates(plan.batchShape, leftBatch, result->result);
    if (failed(coordinates))
        return failure();
    coordinates->push_back(result->row);
    coordinates->push_back(reductionIndex);
    return encodeLinearIndex(plan.leftShape, *coordinates);
}

FailureOr<int64_t> getStaticMatmulRightLinearIndex(const StaticMatmulPlan &plan, int64_t resultLinearIndex,
                                                   int64_t reductionIndex) {
    if (reductionIndex < 0 || reductionIndex >= plan.reduction)
        return failure();
    FailureOr<MatmulCoordinates> result = getMatmulCoordinates(plan, resultLinearIndex);
    if (failed(result))
        return failure();
    if (plan.rightVector)
        return reductionIndex;
    ArrayRef<int64_t> rightBatch = ArrayRef<int64_t>(plan.rightShape).drop_back(2);
    FailureOr<SmallVector<int64_t>> coordinates = mapBatchCoordinates(plan.batchShape, rightBatch, result->result);
    if (failed(coordinates))
        return failure();
    coordinates->push_back(reductionIndex);
    coordinates->push_back(result->column);
    return encodeLinearIndex(plan.rightShape, *coordinates);
}

} // namespace mlir::vernon
