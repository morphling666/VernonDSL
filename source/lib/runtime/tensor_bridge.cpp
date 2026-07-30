#include "tensor_bridge.h"
#include "pipeline_manifest.h"

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

bool valueLayoutValid(const VernonValueLayoutView &layout) {
    if (layout.struct_size < sizeof(VernonValueLayoutView) || !layout.byte_size || !layout.alignment ||
        (layout.alignment & (layout.alignment - 1)) || !layout.layout_hash.data || !layout.layout_hash.size ||
        !layout.leaves || !layout.leaf_count)
        return false;
    for (size_t index = 0; index < layout.leaf_count; ++index) {
        const VernonValueLeafView &leaf = layout.leaves[index];
        const size_t scalarSize = dataTypeSize(static_cast<VernonDataType>(leaf.dtype));
        if (!scalarSize || !leaf.scalar_count || leaf.byte_offset >= layout.byte_size ||
            leaf.scalar_count > (layout.byte_size - leaf.byte_offset) / scalarSize)
            return false;
    }
    return true;
}

bool valueLayoutsEqual(const VernonValueLayoutView &left, const VernonValueLayoutView &right) {
    if (!valueLayoutValid(left) || !valueLayoutValid(right) || left.byte_size != right.byte_size ||
        left.alignment != right.alignment || left.layout_hash.size != right.layout_hash.size ||
        std::memcmp(left.layout_hash.data, right.layout_hash.data, left.layout_hash.size) != 0 ||
        left.leaf_count != right.leaf_count)
        return false;
    for (size_t index = 0; index < left.leaf_count; ++index)
        if (left.leaves[index].dtype != right.leaves[index].dtype ||
            left.leaves[index].scalar_count != right.leaves[index].scalar_count ||
            left.leaves[index].byte_offset != right.leaves[index].byte_offset)
            return false;
    return true;
}

bool tensorMatchesSpecialization(const VernonTensorView &tensor, const ParameterUse &use) {
    if (use.elementStrides.empty())
        return true;
    if (!use.elementOffset || tensor.rank != use.elementStrides.size() ||
        (!use.shape.empty() && tensor.rank != use.shape.size()) ||
        (tensor.rank && (!tensor.shape || !tensor.byte_strides)) || !valueLayoutValid(tensor.element_layout))
        return false;
    const uint64_t elementSize = tensor.element_layout.byte_size;
    if (*use.elementOffset > std::numeric_limits<size_t>::max() / elementSize ||
        tensor.byte_offset != *use.elementOffset * elementSize)
        return false;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (!use.shape.empty() && use.shape[dimension] && tensor.shape[dimension] != use.shape[dimension])
            return false;
        const int64_t elementStride = use.elementStrides[dimension];
        const uint64_t magnitude = strideMagnitude(elementStride);
        if (magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / elementSize)
            return false;
        if (tensor.byte_strides[dimension] != elementStride * static_cast<int64_t>(elementSize))
            return false;
    }
    return true;
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
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
    const std::optional<size_t> elementCount = tensorElementCount(tensor);
    if (!elementSize || !elementCount || *elementCount > std::numeric_limits<size_t>::max() / elementSize)
        return std::nullopt;
    return *elementCount * elementSize;
}

bool tensorRequiredSpan(const VernonTensorView &tensor, size_t &span) {
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
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
    const size_t elementSize = tensor.element_layout.byte_size;
    return after <= availableAfter && elementSize <= availableAfter - after;
}

bool isRowMajorContiguous(const VernonTensorView &tensor) {
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
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

std::optional<std::vector<uint8_t>> packTensor(const VernonTensorView &tensor, const TensorPackingLayout &layout) {
    if (tensor.storage != VERNON_TENSOR_HOST || !valueLayoutValid(tensor.element_layout) ||
        tensor.element_layout.byte_size != layout.elementSize || tensor.rank != layout.shape.size() ||
        layout.byteStrides.size() != layout.shape.size() || (tensor.rank && !tensor.shape) ||
        (!layout.elementLeafOffsets.empty() && layout.elementLeafOffsets.size() != tensor.element_layout.leaf_count))
        return std::nullopt;
    const size_t elementSize = tensor.element_layout.byte_size;
    if (!elementSize || !tensorFitsAllocation(tensor))
        return std::nullopt;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension)
        if (tensor.shape[dimension] != layout.shape[dimension])
            return std::nullopt;

    const std::optional<size_t> elementCount = tensorElementCount(tensor);
    if (!elementCount)
        return std::nullopt;
    size_t requiredSize = *elementCount ? elementSize : 0;
    if (*elementCount)
        for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
            const uint64_t steps = layout.shape[dimension] - 1;
            if (steps && layout.byteStrides[dimension] > (std::numeric_limits<size_t>::max() - requiredSize) / steps)
                return std::nullopt;
            requiredSize += static_cast<size_t>(steps) * layout.byteStrides[dimension];
        }
    if (requiredSize > layout.byteSize)
        return std::nullopt;

    std::vector<uint8_t> packed(layout.byteSize);
    if (!*elementCount)
        return packed;
    const uint8_t *source = hostTensorData(tensor);
    if (!source)
        return std::nullopt;

    for (size_t linear = 0; linear < *elementCount; ++linear) {
        size_t remainder = linear;
        size_t positiveOffset = 0;
        size_t negativeOffset = 0;
        size_t destinationOffset = 0;
        for (uint32_t dimension = tensor.rank; dimension-- > 0;) {
            const size_t extent = static_cast<size_t>(tensor.shape[dimension]);
            const size_t index = remainder % extent;
            remainder /= extent;
            const size_t offset = index * static_cast<size_t>(strideMagnitude(tensor.byte_strides[dimension]));
            (tensor.byte_strides[dimension] < 0 ? negativeOffset : positiveOffset) += offset;
            destinationOffset += index * layout.byteStrides[dimension];
        }
        const uint8_t *sourceElement = source + positiveOffset - negativeOffset;
        if (layout.elementLeafOffsets.empty()) {
            std::memcpy(packed.data() + destinationOffset, sourceElement, elementSize);
            continue;
        }
        for (size_t leafIndex = 0; leafIndex < tensor.element_layout.leaf_count; ++leafIndex) {
            const VernonValueLeafView &leaf = tensor.element_layout.leaves[leafIndex];
            const size_t scalarSize = dataTypeSize(static_cast<VernonDataType>(leaf.dtype));
            if (!scalarSize || leaf.scalar_count > std::numeric_limits<size_t>::max() / scalarSize)
                return std::nullopt;
            const size_t leafSize = leaf.scalar_count * scalarSize;
            const size_t physicalOffset = layout.elementLeafOffsets[leafIndex];
            if (destinationOffset > packed.size() || physicalOffset > packed.size() - destinationOffset ||
                leafSize > packed.size() - destinationOffset - physicalOffset)
                return std::nullopt;
            std::memcpy(packed.data() + destinationOffset + physicalOffset, sourceElement + leaf.byte_offset, leafSize);
        }
    }
    return packed;
}

std::optional<std::vector<uint8_t>> packTensorRowMajor(const VernonTensorView &tensor) {
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
    const std::optional<size_t> packedSize = tensorLogicalByteSize(tensor);
    if (!elementSize || !packedSize || (tensor.rank && !tensor.shape))
        return std::nullopt;
    TensorPackingLayout layout;
    layout.elementSize = elementSize;
    if (tensor.rank)
        layout.shape.assign(tensor.shape, tensor.shape + tensor.rank);
    layout.byteStrides.resize(tensor.rank);
    size_t stride = elementSize;
    for (uint32_t dimension = tensor.rank; dimension-- > 0;) {
        layout.byteStrides[dimension] = stride;
        if (layout.shape[dimension] && stride > std::numeric_limits<size_t>::max() / layout.shape[dimension])
            return std::nullopt;
        stride *= static_cast<size_t>(layout.shape[dimension]);
    }
    layout.byteSize = *packedSize;
    return packTensor(tensor, layout);
}

} // namespace vernon::runtime
