#include "tensor_bridge.h"

#include <cstring>
#include <limits>

namespace vernon::runtime {

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
    if (!elementSize || (tensor.rank && (!tensor.shape || !tensor.byte_strides)))
        return false;
    span = elementSize;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (!tensor.shape[dimension] || !tensor.byte_strides[dimension])
            return false;
        const uint64_t steps = tensor.shape[dimension] - 1;
        if (steps && tensor.byte_strides[dimension] > (std::numeric_limits<size_t>::max() - span) / steps)
            return false;
        span += static_cast<size_t>(steps * tensor.byte_strides[dimension]);
    }
    return true;
}

bool isRowMajorContiguous(const VernonTensorView &tensor) {
    const size_t elementSize = dataTypeSize(tensor.dtype);
    if (!elementSize || (tensor.rank && (!tensor.shape || !tensor.byte_strides)))
        return false;
    size_t stride = elementSize;
    for (uint32_t dimension = tensor.rank; dimension-- > 0;) {
        if (tensor.byte_strides[dimension] != stride)
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
    if (tensor.byte_offset > tensor.byte_size || (tensor.rank && !tensor.byte_strides))
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

    uint32_t contiguousBegin = tensor.rank;
    size_t blockBytes = elementSize;
    while (contiguousBegin != 0) {
        const uint32_t dimension = contiguousBegin - 1;
        if (tensor.byte_strides[dimension] != blockBytes)
            break;
        if (tensor.shape[dimension] > std::numeric_limits<size_t>::max() / blockBytes)
            return std::nullopt;
        blockBytes *= static_cast<size_t>(tensor.shape[dimension]);
        contiguousBegin = dimension;
    }

    const size_t outerCount = *packedSize / blockBytes;
    for (size_t outer = 0; outer < outerCount; ++outer) {
        size_t remainder = outer;
        size_t sourceOffset = 0;
        for (uint32_t dimension = contiguousBegin; dimension-- > 0;) {
            const size_t extent = static_cast<size_t>(tensor.shape[dimension]);
            const size_t index = remainder % extent;
            remainder /= extent;
            if (tensor.byte_strides[dimension] > std::numeric_limits<size_t>::max() ||
                (index && static_cast<size_t>(tensor.byte_strides[dimension]) >
                              (std::numeric_limits<size_t>::max() - sourceOffset) / index))
                return std::nullopt;
            sourceOffset += index * static_cast<size_t>(tensor.byte_strides[dimension]);
        }
        if (sourceOffset > tensor.byte_size - tensor.byte_offset ||
            blockBytes > tensor.byte_size - tensor.byte_offset - sourceOffset)
            return std::nullopt;
        std::memcpy(packed.data() + outer * blockBytes, source + sourceOffset, blockBytes);
    }
    return packed;
}

} // namespace vernon::runtime
