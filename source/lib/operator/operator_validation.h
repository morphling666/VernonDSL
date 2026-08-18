#ifndef VERNON_OPERATOR_VALIDATION_H
#define VERNON_OPERATOR_VALIDATION_H

#include "operator/operator_model.h"

#include <string>

namespace vernon::ops {

bool tensorViewFootprint(const TensorViewDescriptor &descriptor, uint64_t &begin, uint64_t &end);
bool validateTensorViewDescriptor(const TensorViewDescriptor &descriptor, std::string &error);
bool tensorViewsElementwiseCompatible(const TensorViewDescriptor &left, const TensorViewDescriptor &right,
                                      std::string &error);
bool validateOperatorDag(const OperatorDag &dag, std::string &error);

} // namespace vernon::ops

#endif
