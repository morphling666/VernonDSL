#ifndef VERNON_OPERATOR_MODEL_H
#define VERNON_OPERATOR_MODEL_H

#include "VernonRuntime.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::ops {

enum class OperatorKind : uint8_t { Leaf, Elementwise };
enum class ElementwiseOperatorKind : uint8_t { Add };

struct TensorViewDescriptor {
    VernonDataType dtype{VERNON_DATA_F32};
    std::vector<uint64_t> shape;
    std::vector<int64_t> byteStrides;
    uint64_t byteOffset{};
    uint64_t allocationBytes{};
    uintptr_t ownerIdentity{};
};

struct OperatorNode {
    OperatorKind kind{OperatorKind::Elementwise};
    ElementwiseOperatorKind elementwise{ElementwiseOperatorKind::Add};
    std::vector<uint32_t> inputs;
    TensorViewDescriptor output;
};

class OperatorDag {
public:
    uint32_t addLeaf(TensorViewDescriptor descriptor);
    uint32_t addElementwise(ElementwiseOperatorKind kind, std::vector<uint32_t> inputs, TensorViewDescriptor output);

    const std::vector<OperatorNode> &nodes() const { return nodes_; }

private:
    std::vector<OperatorNode> nodes_;
};

} // namespace vernon::ops

#endif
