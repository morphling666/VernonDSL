#ifndef VERNON_RUNTIME_TENSOR_BRIDGE_H
#define VERNON_RUNTIME_TENSOR_BRIDGE_H

#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace vernon::runtime {

size_t dataTypeSize(VernonDataType dtype);

const uint8_t *hostTensorData(const VernonTensorView &tensor);

std::optional<size_t> tensorElementCount(const VernonTensorView &tensor);
std::optional<size_t> tensorLogicalByteSize(const VernonTensorView &tensor);

bool tensorRequiredSpan(const VernonTensorView &tensor, size_t &span);

bool isRowMajorContiguous(const VernonTensorView &tensor);

std::optional<std::vector<uint8_t>> packTensorRowMajor(const VernonTensorView &tensor);

} // namespace vernon::runtime

#endif
