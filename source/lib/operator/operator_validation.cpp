#include "operator/operator_validation.h"

#include <limits>

namespace vernon::ops {
namespace {

uint64_t dataTypeBytes(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
    case VERNON_DATA_U8:
        return 1;
    case VERNON_DATA_F16:
        return 2;
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_F32:
        return 4;
    case VERNON_DATA_F64:
        return 8;
    }
    return 0;
}

bool checkedAdd(uint64_t left, uint64_t right, uint64_t &result) {
    if (right > std::numeric_limits<uint64_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

bool checkedMultiply(uint64_t left, uint64_t right, uint64_t &result) {
    if (left && right > std::numeric_limits<uint64_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

} // namespace

bool tensorViewFootprint(const TensorViewDescriptor &descriptor, uint64_t &begin, uint64_t &end) {
    const uint64_t elementBytes = dataTypeBytes(descriptor.dtype);
    if (!elementBytes || descriptor.shape.size() != descriptor.byteStrides.size())
        return false;
    uint64_t beforeOffset = 0;
    uint64_t afterOffset = 0;
    for (size_t dimension = 0; dimension < descriptor.shape.size(); ++dimension) {
        if (!descriptor.shape[dimension])
            return false;
        const int64_t stride = descriptor.byteStrides[dimension];
        const uint64_t magnitude =
            stride < 0 ? static_cast<uint64_t>(-(stride + 1)) + 1 : static_cast<uint64_t>(stride);
        uint64_t extent = 0;
        if (!checkedMultiply(descriptor.shape[dimension] - 1, magnitude, extent))
            return false;
        uint64_t &side = stride < 0 ? beforeOffset : afterOffset;
        if (!checkedAdd(side, extent, side))
            return false;
    }
    if (descriptor.byteOffset < beforeOffset)
        return false;
    begin = descriptor.byteOffset - beforeOffset;
    if (!checkedAdd(descriptor.byteOffset, afterOffset, end) || !checkedAdd(end, elementBytes, end))
        return false;
    return true;
}

bool validateTensorViewDescriptor(const TensorViewDescriptor &descriptor, std::string &error) {
    uint64_t begin = 0;
    uint64_t end = 0;
    if (!descriptor.ownerIdentity) {
        error = "operator tensor view has no physical owner";
        return false;
    }
    if (descriptor.shape.size() != descriptor.byteStrides.size()) {
        error = "operator tensor view rank and strides do not match";
        return false;
    }
    if (!tensorViewFootprint(descriptor, begin, end) || end > descriptor.allocationBytes) {
        error = "operator tensor view exceeds its physical owner";
        return false;
    }
    return true;
}

bool tensorViewsElementwiseCompatible(const TensorViewDescriptor &left, const TensorViewDescriptor &right,
                                      std::string &error) {
    if (left.dtype != right.dtype) {
        error = "elementwise operator input dtypes do not match";
        return false;
    }
    if (left.shape != right.shape) {
        error = "elementwise operator input shapes do not match";
        return false;
    }
    return true;
}

bool validateOperatorDag(const OperatorDag &dag, std::string &error) {
    const auto &nodes = dag.nodes();
    for (uint32_t index = 0; index < nodes.size(); ++index) {
        const OperatorNode &node = nodes[index];
        if (!validateTensorViewDescriptor(node.output, error))
            return false;
        if (node.kind == OperatorKind::Leaf) {
            if (!node.inputs.empty()) {
                error = "operator leaf unexpectedly has inputs";
                return false;
            }
            continue;
        }
        if (node.kind != OperatorKind::Elementwise || node.elementwise != ElementwiseOperatorKind::Add ||
            node.inputs.size() != 2) {
            error = "operator DAG contains an unsupported node";
            return false;
        }
        for (uint32_t input : node.inputs) {
            if (input >= index) {
                error = "operator DAG is not topologically ordered";
                return false;
            }
            if (!tensorViewsElementwiseCompatible(nodes[input].output, node.output, error))
                return false;
        }
    }
    return true;
}

} // namespace vernon::ops
