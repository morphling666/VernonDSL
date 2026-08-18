#include "operator/elementwise_operator.h"
#include "operator/operator_validation.h"

#include <utility>

namespace vernon::ops {
namespace {

bool overlaps(const TensorViewDescriptor &left, const TensorViewDescriptor &right) {
    if (left.ownerIdentity != right.ownerIdentity)
        return false;
    uint64_t leftBegin = 0;
    uint64_t leftEnd = 0;
    uint64_t rightBegin = 0;
    uint64_t rightEnd = 0;
    return tensorViewFootprint(left, leftBegin, leftEnd) && tensorViewFootprint(right, rightBegin, rightEnd) &&
           leftBegin < rightEnd && rightBegin < leftEnd;
}

} // namespace

bool planElementwiseAdd(TensorViewDescriptor left, TensorViewDescriptor right, TensorViewDescriptor output,
                        ElementwiseAddPlan &plan, std::string &error) {
    if (!validateTensorViewDescriptor(left, error) || !validateTensorViewDescriptor(right, error) ||
        !validateTensorViewDescriptor(output, error) || !tensorViewsElementwiseCompatible(left, right, error) ||
        !tensorViewsElementwiseCompatible(left, output, error))
        return false;

    plan = {std::move(left), std::move(right), std::move(output), true, {}};
    if (plan.output.dtype != VERNON_DATA_F32) {
        plan.device = false;
        plan.fallbackReason = "device elementwise Add currently supports f32 gradients";
    } else if (plan.output.shape.size() > 3) {
        plan.device = false;
        plan.fallbackReason = "device elementwise Add currently supports ranks zero through three";
    } else if (overlaps(plan.output, plan.left) || overlaps(plan.output, plan.right)) {
        plan.device = false;
        plan.fallbackReason = "device elementwise Add requires an invocation-private output owner";
    }
    return true;
}

} // namespace vernon::ops
