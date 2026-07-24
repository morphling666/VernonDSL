#include "tensor_bridge.h"

#include <cstring>
#include <limits>

namespace vernon::runtime {
namespace {

uint64_t strideMagnitude(int64_t stride) {
    return stride < 0 ? static_cast<uint64_t>(-(stride + 1)) + 1 : static_cast<uint64_t>(stride);
}

bool tensorRelativeBounds(const VernonTensorView &tensor, size_t &before, size_t &after) {
    if (tensor.rank && (!tensor.shape || !tensor.byte_strides))
        return false;
    before = 0;
    after = 0;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (!tensor.shape[dimension])
            return true;
        const uint64_t steps = tensor.shape[dimension] - 1;
        const uint64_t magnitude = strideMagnitude(tensor.byte_strides[dimension]);
        if (magnitude > std::numeric_limits<size_t>::max() ||
            (steps && magnitude > std::numeric_limits<size_t>::max() / steps))
            return false;
        const size_t extent = static_cast<size_t>(steps * magnitude);
        size_t &bound = tensor.byte_strides[dimension] < 0 ? before : after;
        if (extent > std::numeric_limits<size_t>::max() - bound)
            return false;
        bound += extent;
    }
    return true;
}

} // namespace

size_t dataTypeSize(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
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

const uint8_t *hostTensorData(const VernonTensorView &tensor) {
    if (!tensor.host_data)
        return nullptr;
    return static_cast<const uint8_t *>(tensor.host_data) + tensor.byte_offset;
}

std::optional<size_t> tensorElementCount(const VernonTensorView &tensor) {
    if (tensor.rank && !tensor.shape)
        return std::nullopt;
    size_t count = 1;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (tensor.shape[dimension] > std::numeric_limits<size_t>::max() / count)
            return std::nullopt;
        count *= static_cast<size_t>(tensor.shape[dimension]);
    }
    return count;
}

std::optional<size_t> tensorLogicalByteSize(const VernonTensorView &tensor) {
    const size_t elementSize = dataTypeSize(tensor.dtype);
    const std::optional<size_t> elementCount = tensorElementCount(tensor);
    if (!elementSize || !elementCount || *elementCount > std::numeric_limits<size_t>::max() / elementSize)
        return std::nullopt;
    return *elementCount * elementSize;
}

bool tensorRequiredSpan(const VernonTensorView &tensor, size_t &span) {
    const size_t elementSize = dataTypeSize(tensor.dtype);
    const std::optional<size_t> elementCount = tensorElementCount(tensor);
    if (!elementSize || !elementCount)
        return false;
    if (!*elementCount) {
        span = 0;
        return true;
    }
    size_t before = 0;
    size_t after = 0;
    if (!tensorRelativeBounds(tensor, before, after) || after > std::numeric_limits<size_t>::max() - before ||
        elementSize > std::numeric_limits<size_t>::max() - before - after)
        return false;
    span = before + after + elementSize;
    return true;
}

bool tensorFitsAllocation(const VernonTensorView &tensor) {
    size_t span = 0;
    const std::optional<size_t> elementCount = tensorElementCount(tensor);
    if (!elementCount || !tensorRequiredSpan(tensor, span) || tensor.byte_offset > tensor.byte_size)
        return false;
    if (!*elementCount)
        return true;
    size_t before = 0;
    size_t after = 0;
    if (!tensorRelativeBounds(tensor, before, after) || before > tensor.byte_offset)
        return false;
    const size_t availableAfter = tensor.byte_size - tensor.byte_offset;
    const size_t elementSize = dataTypeSize(tensor.dtype);
    return after <= availableAfter && elementSize <= availableAfter - after;
}

bool isRowMajorContiguous(const VernonTensorView &tensor) {
    const size_t elementSize = dataTypeSize(tensor.dtype);
    if (!elementSize || (tensor.rank && (!tensor.shape || !tensor.byte_strides)))
        return false;
    size_t stride = elementSize;
    for (uint32_t dimension = tensor.rank; dimension-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
            tensor.byte_strides[dimension] != static_cast<int64_t>(stride))
            return false;
        if (dimension == 0)
            continue;
        if (tensor.shape[dimension] > std::numeric_limits<size_t>::max() / stride)
            return false;
        stride *= static_cast<size_t>(tensor.shape[dimension]);
    }
    return true;
}

std::optional<std::vector<uint8_t>> packTensorRowMajor(const VernonTensorView &tensor) {
    if (tensor.storage != VERNON_TENSOR_HOST)
        return std::nullopt;
    const size_t elementSize = dataTypeSize(tensor.dtype);
    const std::optional<size_t> packedSize = tensorLogicalByteSize(tensor);
    if (!packedSize)
        return std::nullopt;

    std::vector<uint8_t> packed(*packedSize);
    if (*packedSize == 0)
        return packed;
    if (!tensorFitsAllocation(tensor))
        return std::nullopt;
    const uint8_t *source = hostTensorData(tensor);
    if (!source)
        return std::nullopt;
    if (isRowMajorContiguous(tensor)) {
        if (*packedSize > tensor.byte_size - tensor.byte_offset)
            return std::nullopt;
        std::memcpy(packed.data(), source, *packedSize);
        return packed;
    }

    const size_t elementCount = *packedSize / elementSize;
    for (size_t linear = 0; linear < elementCount; ++linear) {
        size_t remainder = linear;
        size_t positiveOffset = 0;
        size_t negativeOffset = 0;
        for (uint32_t dimension = tensor.rank; dimension-- > 0;) {
            const size_t extent = static_cast<size_t>(tensor.shape[dimension]);
            const size_t index = remainder % extent;
            remainder /= extent;
            const size_t offset = index * static_cast<size_t>(strideMagnitude(tensor.byte_strides[dimension]));
            (tensor.byte_strides[dimension] < 0 ? negativeOffset : positiveOffset) += offset;
        }
        std::memcpy(packed.data() + linear * elementSize, source + positiveOffset - negativeOffset, elementSize);
    }
    return packed;
}

} // namespace vernon::runtime
