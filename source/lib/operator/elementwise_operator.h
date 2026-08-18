#ifndef VERNON_ELEMENTWISE_OPERATOR_H
#define VERNON_ELEMENTWISE_OPERATOR_H

#include "operator/operator_model.h"

#include <string>

namespace vernon::ops {

struct ElementwiseAddPlan {
    TensorViewDescriptor left;
    TensorViewDescriptor right;
    TensorViewDescriptor output;
    bool device{};
    std::string fallbackReason;
};

bool planElementwiseAdd(TensorViewDescriptor left, TensorViewDescriptor right, TensorViewDescriptor output,
                        ElementwiseAddPlan &plan, std::string &error);

} // namespace vernon::ops

#endif
