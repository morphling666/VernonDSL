#ifndef VERNON_RUNTIME_TENSOR_BRIDGE_H
#define VERNON_RUNTIME_TENSOR_BRIDGE_H

#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace vernon::runtime {

struct TransportNode;

size_t dataTypeSize(VernonDataType dtype);

bool valueLayoutValid(const VernonValueLayoutView &layout);
bool valueLayoutsEqual(const VernonValueLayoutView &left, const VernonValueLayoutView &right);

const uint8_t *hostTensorData(const VernonTensorView &tensor);

std::optional<size_t> tensorElementCount(const VernonTensorView &tensor);
std::optional<size_t> tensorLogicalByteSize(const VernonTensorView &tensor);

bool tensorRequiredSpan(const VernonTensorView &tensor, size_t &span);
bool tensorFitsAllocation(const VernonTensorView &tensor);
bool tensorRelativeByteBounds(const VernonTensorView &tensor, size_t &before, size_t &after);
bool tensorByteLayoutInjective(const VernonTensorView &tensor);

enum class TensorPhysicalOverlap {
    Disjoint,
    Overlapping,
    Unknown,
};

TensorPhysicalOverlap tensorViewsPhysicalOverlap(const VernonTensorView &left, const VernonTensorView &right);
bool tensorViewsHaveWritableOverlap(const VernonTensorView &left, const VernonTensorView &right);
bool hostByteRangesHaveWritableOverlap(const void *leftData, size_t leftSize, bool leftWritable, const void *rightData,
                                       size_t rightSize, bool rightWritable);

bool isRowMajorContiguous(const VernonTensorView &tensor);

struct CopyOperation {
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

struct TensorCopyPlan {
    size_t elementSize{};
    std::vector<uint64_t> shape;
    std::vector<size_t> byteStrides;
    size_t byteSize{};
    std::vector<CopyOperation> operations;
};

std::optional<TensorCopyPlan> compileTensorCopyPlan(const VernonValueLayoutView &canonical,
                                                    const TransportNode &transport, std::vector<uint64_t> shape);

std::optional<std::vector<uint8_t>> packTensor(const VernonTensorView &tensor, const TensorCopyPlan &plan);

std::optional<std::vector<uint8_t>> packTensorRowMajor(const VernonTensorView &tensor);

} // namespace vernon::runtime

#endif
