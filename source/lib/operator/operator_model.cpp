#include "operator/operator_model.h"

#include <utility>

namespace vernon::ops {

uint32_t OperatorDag::addLeaf(TensorViewDescriptor descriptor) {
    const uint32_t index = static_cast<uint32_t>(nodes_.size());
    OperatorNode node;
    node.kind = OperatorKind::Leaf;
    node.output = std::move(descriptor);
    nodes_.push_back(std::move(node));
    return index;
}

uint32_t OperatorDag::addElementwise(ElementwiseOperatorKind kind, std::vector<uint32_t> inputs,
                                     TensorViewDescriptor output) {
    const uint32_t index = static_cast<uint32_t>(nodes_.size());
    OperatorNode node;
    node.kind = OperatorKind::Elementwise;
    node.elementwise = kind;
    node.inputs = std::move(inputs);
    node.output = std::move(output);
    nodes_.push_back(std::move(node));
    return index;
}

} // namespace vernon::ops
