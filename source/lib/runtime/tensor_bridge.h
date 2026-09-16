#ifndef VERNON_RUNTIME_TENSOR_BRIDGE_H
#define VERNON_RUNTIME_TENSOR_BRIDGE_H

#include "VernonResult.hpp"
#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vernon::runtime {

struct TransportNode;

enum class TensorBridgeError : uint8_t {
    InvalidDataType,
    InvalidShape,
    InvalidLayout,
    InvalidStorage,
    InvalidCopyPlan,
    IncompatibleRepresentation,
    OutOfBounds,
    Overflow,
};

template <typename T> using TensorBridgeResult = vernon::Result<T, TensorBridgeError>;

struct TensorRelativeByteBounds {
    size_t before{};
    size_t after{};
};

TensorBridgeResult<size_t> dataTypeSize(VernonDataType dtype);

bool valueLayoutValid(const VernonValueLayoutView &layout);
bool valueLayoutsEqual(const VernonValueLayoutView &left, const VernonValueLayoutView &right);

vernon::Option<const uint8_t *> hostTensorData(const VernonTensorView &tensor);

TensorBridgeResult<size_t> tensorElementCount(const VernonTensorView &tensor);
TensorBridgeResult<size_t> tensorLogicalByteSize(const VernonTensorView &tensor);

TensorBridgeResult<size_t> tensorRequiredSpan(const VernonTensorView &tensor);
bool tensorFitsAllocation(const VernonTensorView &tensor);
TensorBridgeResult<TensorRelativeByteBounds> tensorRelativeByteBounds(const VernonTensorView &tensor);
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

TensorBridgeResult<TensorCopyPlan> compileWholeValueCopyPlan(const VernonValueLayoutView &canonicalValue,
                                                             const TransportNode &physicalValue);
TensorBridgeResult<TensorCopyPlan> compileElementStreamCopyPlan(const VernonValueLayoutView &elementLayout,
                                                                std::vector<uint64_t> logicalShape,
                                                                const TransportNode &physicalStream);

TensorBridgeResult<std::vector<uint8_t>> packTensor(const VernonTensorView &tensor, const TensorCopyPlan &plan);
TensorBridgeResult<void> unpackTensor(const std::vector<uint8_t> &packed, const VernonTensorView &tensor,
                                      const TensorCopyPlan &plan);

TensorBridgeResult<std::vector<uint8_t>> packTensorRowMajor(const VernonTensorView &tensor);
const char *tensorBridgeErrorMessage(TensorBridgeError error) noexcept;

} // namespace vernon::runtime

#endif
