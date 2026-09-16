#include "tensor_bridge.h"
#include "transport_node.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <string_view>

namespace vernon::runtime {
namespace {

uint64_t strideMagnitude(int64_t stride) {
    return stride < 0 ? static_cast<uint64_t>(-(stride + 1)) + 1 : static_cast<uint64_t>(stride);
}

TensorBridgeResult<TensorRelativeByteBounds> computeTensorRelativeByteBounds(const VernonTensorView &tensor) {
    if (tensor.rank && (!tensor.shape || !tensor.byte_strides))
        return TensorBridgeResult<TensorRelativeByteBounds>{vernon::err(TensorBridgeError::InvalidShape)};
    TensorRelativeByteBounds bounds;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (!tensor.shape[dimension])
            return TensorBridgeResult<TensorRelativeByteBounds>{vernon::ok(bounds)};
        const uint64_t steps = tensor.shape[dimension] - 1;
        const uint64_t magnitude = strideMagnitude(tensor.byte_strides[dimension]);
        if (magnitude > std::numeric_limits<size_t>::max() ||
            (steps && magnitude > std::numeric_limits<size_t>::max() / steps))
            return TensorBridgeResult<TensorRelativeByteBounds>{vernon::err(TensorBridgeError::Overflow)};
        const size_t extent = static_cast<size_t>(steps * magnitude);
        size_t &bound = tensor.byte_strides[dimension] < 0 ? bounds.before : bounds.after;
        if (extent > std::numeric_limits<size_t>::max() - bound)
            return TensorBridgeResult<TensorRelativeByteBounds>{vernon::err(TensorBridgeError::Overflow)};
        bound += extent;
    }
    return TensorBridgeResult<TensorRelativeByteBounds>{vernon::ok(bounds)};
}

struct TransportScalar {
    size_t offset{};
    size_t size{};
    std::string representation;
};

bool collectTransportScalars(const TransportNode &node, size_t base, std::vector<TransportScalar> &scalars) {
    if (node.offset > std::numeric_limits<size_t>::max() - base)
        return false;
    const size_t offset = base + static_cast<size_t>(node.offset);
    if (node.kind == TransportNodeKind::Scalar) {
        if (node.size > std::numeric_limits<size_t>::max())
            return false;
        scalars.push_back({offset, static_cast<size_t>(node.size), node.representation});
        return true;
    }
    if (node.kind == TransportNodeKind::Array) {
        if (node.children.size() != 1 || node.shape.size() != node.byteStrides.size())
            return false;
        size_t count = 1;
        for (uint64_t extent : node.shape) {
            if (!extent || extent > std::numeric_limits<size_t>::max() / count)
                return false;
            count *= static_cast<size_t>(extent);
        }
        for (size_t linear = 0; linear < count; ++linear) {
            size_t remainder = linear;
            size_t elementOffset = 0;
            for (size_t dimension = node.shape.size(); dimension-- > 0;) {
                const size_t extent = static_cast<size_t>(node.shape[dimension]);
                const size_t index = remainder % extent;
                remainder /= extent;
                if (index && node.byteStrides[dimension] > (std::numeric_limits<size_t>::max() - elementOffset) / index)
                    return false;
                elementOffset += index * static_cast<size_t>(node.byteStrides[dimension]);
            }
            if (elementOffset > std::numeric_limits<size_t>::max() - offset ||
                !collectTransportScalars(node.children.front(), offset + elementOffset, scalars))
                return false;
        }
        return true;
    }
    for (const TransportNode &child : node.children)
        if (!collectTransportScalars(child, offset, scalars))
            return false;
    return true;
}

struct TensorPhysicalRange {
    uint64_t begin{};
    uint64_t end{};
    uint64_t base{};
    bool empty{};
};

bool addPhysicalOffset(uint64_t base, size_t offset, uint64_t &result) {
    if (offset > std::numeric_limits<uint64_t>::max() - base)
        return false;
    result = base + static_cast<uint64_t>(offset);
    return true;
}

bool tensorPhysicalRange(const VernonTensorView &tensor, TensorPhysicalRange &range) {
    auto count = tensorElementCount(tensor);
    if (count.isErr() || !tensorFitsAllocation(tensor))
        return false;
    if (!count.value()) {
        range.empty = true;
        return true;
    }

    auto bounds = tensorRelativeByteBounds(tensor);
    if (bounds.isErr())
        return false;
    const size_t localBegin = tensor.byte_offset - bounds.value().before;
    if (tensor.element_layout.byte_size >
        std::numeric_limits<size_t>::max() - tensor.byte_offset - bounds.value().after)
        return false;
    const size_t localEnd = tensor.byte_offset + bounds.value().after + tensor.element_layout.byte_size;

    if (tensor.storage == VERNON_TENSOR_HOST) {
        if (!tensor.host_data)
            return false;
        range.base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.host_data));
    } else if (tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
        if (!tensor.resource.identity || !tensor.resource.resource.value ||
            (tensor.resource.size && tensor.byte_size > tensor.resource.size) ||
            tensor.byte_size > std::numeric_limits<uint64_t>::max() - tensor.resource.offset)
            return false;
        range.base = tensor.resource.offset;
    } else {
        return false;
    }
    return addPhysicalOffset(range.base, localBegin, range.begin) && addPhysicalOffset(range.base, localEnd, range.end);
}

uint64_t tensorStrideLattice(const VernonTensorView &tensor) {
    uint64_t lattice = 0;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension)
        if (tensor.shape[dimension] > 1)
            lattice = std::gcd(lattice, strideMagnitude(tensor.byte_strides[dimension]));
    return lattice;
}

bool byteIntervalsOverlap(uint64_t left, size_t leftSize, uint64_t right, size_t rightSize) {
    if (leftSize > std::numeric_limits<uint64_t>::max() - left ||
        rightSize > std::numeric_limits<uint64_t>::max() - right)
        return true;
    return left < right + rightSize && right < left + leftSize;
}

bool latticeProvesDisjoint(uint64_t leftAddress, size_t leftSize, uint64_t leftLattice, uint64_t rightAddress,
                           size_t rightSize, uint64_t rightLattice) {
    const uint64_t lattice = std::gcd(leftLattice, rightLattice);
    if (!lattice)
        return false;
    if (leftSize >= lattice || rightSize >= lattice)
        return false;
    const uint64_t rightFromLeft = rightAddress >= leftAddress
                                       ? (rightAddress - leftAddress) % lattice
                                       : (lattice - (leftAddress - rightAddress) % lattice) % lattice;
    const uint64_t leftFromRight = (lattice - rightFromLeft) % lattice;
    return rightFromLeft >= leftSize && leftFromRight >= rightSize;
}

bool periodicSpan(const VernonTensorView &tensor, uint64_t address, uint64_t period, uint64_t &begin, uint64_t &end) {
    uint64_t before = 0;
    uint64_t after = 0;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (tensor.shape[dimension] <= 1)
            continue;
        const uint64_t stride = strideMagnitude(tensor.byte_strides[dimension]);
        if (stride >= period) {
            if (stride % period)
                return false;
            continue;
        }
        const uint64_t steps = tensor.shape[dimension] - 1;
        if (steps &&
            stride >
                (std::numeric_limits<uint64_t>::max() - (tensor.byte_strides[dimension] < 0 ? before : after)) / steps)
            return false;
        (tensor.byte_strides[dimension] < 0 ? before : after) += stride * steps;
    }
    const uint64_t residue = address % period;
    const uint64_t elementSize = tensor.element_layout.byte_size;
    if (before > residue || after > period - residue || elementSize > period - residue - after)
        return false;
    begin = residue - before;
    end = residue + after + elementSize;
    return true;
}

bool periodicSpansProveDisjoint(const VernonTensorView &left, uint64_t leftAddress, const VernonTensorView &right,
                                uint64_t rightAddress) {
    for (uint32_t leftDimension = 0; leftDimension < left.rank; ++leftDimension) {
        const uint64_t period = strideMagnitude(left.byte_strides[leftDimension]);
        if (left.shape[leftDimension] <= 1 || !period)
            continue;
        bool sharedPeriod = false;
        for (uint32_t rightDimension = 0; rightDimension < right.rank; ++rightDimension)
            sharedPeriod |=
                right.shape[rightDimension] > 1 && strideMagnitude(right.byte_strides[rightDimension]) == period;
        if (!sharedPeriod)
            continue;
        uint64_t leftBegin = 0;
        uint64_t leftEnd = 0;
        uint64_t rightBegin = 0;
        uint64_t rightEnd = 0;
        if (periodicSpan(left, leftAddress, period, leftBegin, leftEnd) &&
            periodicSpan(right, rightAddress, period, rightBegin, rightEnd) &&
            (leftEnd <= rightBegin || rightEnd <= leftBegin))
            return true;
    }
    return false;
}

} // namespace

TensorBridgeResult<size_t> dataTypeSize(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
    case VERNON_DATA_U8:
        return TensorBridgeResult<size_t>{vernon::ok(size_t{1})};
    case VERNON_DATA_F16:
        return TensorBridgeResult<size_t>{vernon::ok(size_t{2})};
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_F32:
        return TensorBridgeResult<size_t>{vernon::ok(size_t{4})};
    case VERNON_DATA_F64:
        return TensorBridgeResult<size_t>{vernon::ok(size_t{8})};
    }
    return TensorBridgeResult<size_t>{vernon::err(TensorBridgeError::InvalidDataType)};
}

namespace {

vernon::Option<std::string_view> dataTypeRepresentation(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
        return vernon::Option<std::string_view>{vernon::some(std::string_view{"bool"})};
    case VERNON_DATA_U8:
        return vernon::Option<std::string_view>{vernon::some(std::string_view{"u8"})};
    case VERNON_DATA_F16:
        return vernon::Option<std::string_view>{vernon::some(std::string_view{"f16"})};
    case VERNON_DATA_I32:
        return vernon::Option<std::string_view>{vernon::some(std::string_view{"i32"})};
    case VERNON_DATA_U32:
        return vernon::Option<std::string_view>{vernon::some(std::string_view{"u32"})};
    case VERNON_DATA_F32:
        return vernon::Option<std::string_view>{vernon::some(std::string_view{"f32"})};
    case VERNON_DATA_F64:
        return vernon::Option<std::string_view>{vernon::some(std::string_view{"f64"})};
    }
    return {};
}

} // namespace

bool valueLayoutValid(const VernonValueLayoutView &layout) {
    if (layout.struct_size < sizeof(VernonValueLayoutView) || !layout.byte_size || !layout.alignment ||
        (layout.alignment & (layout.alignment - 1)) || !layout.layout_hash.data || !layout.layout_hash.size ||
        !layout.leaves || !layout.leaf_count)
        return false;
    for (size_t index = 0; index < layout.leaf_count; ++index) {
        const VernonValueLeafView &leaf = layout.leaves[index];
        auto scalarSize = dataTypeSize(static_cast<VernonDataType>(leaf.dtype));
        if (scalarSize.isErr() || !leaf.scalar_count || leaf.byte_offset >= layout.byte_size ||
            leaf.scalar_count > (layout.byte_size - leaf.byte_offset) / scalarSize.value())
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

vernon::Option<const uint8_t *> hostTensorData(const VernonTensorView &tensor) {
    if (!tensor.host_data)
        return {};
    return vernon::Option<const uint8_t *>{
        vernon::some(static_cast<const uint8_t *>(tensor.host_data) + tensor.byte_offset)};
}

TensorBridgeResult<size_t> tensorElementCount(const VernonTensorView &tensor) {
    if (tensor.rank && !tensor.shape)
        return TensorBridgeResult<size_t>{vernon::err(TensorBridgeError::InvalidShape)};
    size_t count = 1;
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
        if (!count)
            return TensorBridgeResult<size_t>{vernon::ok(size_t{0})};
        if (tensor.shape[dimension] > std::numeric_limits<size_t>::max() / count)
            return TensorBridgeResult<size_t>{vernon::err(TensorBridgeError::Overflow)};
        count *= static_cast<size_t>(tensor.shape[dimension]);
    }
    return TensorBridgeResult<size_t>{vernon::ok(count)};
}

TensorBridgeResult<size_t> tensorLogicalByteSize(const VernonTensorView &tensor) {
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
    if (!elementSize)
        return TensorBridgeResult<size_t>{vernon::err(TensorBridgeError::InvalidLayout)};
    auto elementCount = tensorElementCount(tensor);
    if (elementCount.isErr())
        return TensorBridgeResult<size_t>{vernon::err(elementCount.error())};
    if (elementCount.value() > std::numeric_limits<size_t>::max() / elementSize)
        return TensorBridgeResult<size_t>{vernon::err(TensorBridgeError::Overflow)};
    return TensorBridgeResult<size_t>{vernon::ok(elementCount.value() * elementSize)};
}

TensorBridgeResult<TensorRelativeByteBounds> tensorRelativeByteBounds(const VernonTensorView &tensor) {
    return computeTensorRelativeByteBounds(tensor);
}

TensorBridgeResult<size_t> tensorRequiredSpan(const VernonTensorView &tensor) {
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
    if (!elementSize)
        return TensorBridgeResult<size_t>{vernon::err(TensorBridgeError::InvalidLayout)};
    auto elementCount = tensorElementCount(tensor);
    if (elementCount.isErr())
        return TensorBridgeResult<size_t>{vernon::err(elementCount.error())};
    if (!elementCount.value())
        return TensorBridgeResult<size_t>{vernon::ok(size_t{0})};
    auto bounds = tensorRelativeByteBounds(tensor);
    if (bounds.isErr())
        return TensorBridgeResult<size_t>{vernon::err(bounds.error())};
    if (bounds.value().after > std::numeric_limits<size_t>::max() - bounds.value().before ||
        elementSize > std::numeric_limits<size_t>::max() - bounds.value().before - bounds.value().after)
        return TensorBridgeResult<size_t>{vernon::err(TensorBridgeError::Overflow)};
    return TensorBridgeResult<size_t>{vernon::ok(bounds.value().before + bounds.value().after + elementSize)};
}

bool tensorFitsAllocation(const VernonTensorView &tensor) {
    auto elementCount = tensorElementCount(tensor);
    auto span = tensorRequiredSpan(tensor);
    if (elementCount.isErr() || span.isErr() || tensor.byte_offset > tensor.byte_size)
        return false;
    if (!elementCount.value())
        return true;
    auto bounds = tensorRelativeByteBounds(tensor);
    if (bounds.isErr() || bounds.value().before > tensor.byte_offset)
        return false;
    const size_t availableAfter = tensor.byte_size - tensor.byte_offset;
    const size_t elementSize = tensor.element_layout.byte_size;
    return bounds.value().after <= availableAfter && elementSize <= availableAfter - bounds.value().after;
}

bool tensorByteLayoutInjective(const VernonTensorView &tensor) {
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
    auto elementCount = tensorElementCount(tensor);
    if (!elementSize || elementCount.isErr() || !tensorFitsAllocation(tensor))
        return false;
    if (elementCount.value() < 2)
        return true;

    if (elementSize - 1 > std::numeric_limits<uint64_t>::max())
        return false;
    uint64_t coveredSpan = static_cast<uint64_t>(elementSize - 1);
    std::vector<std::pair<uint64_t, uint64_t>> dimensions;
    dimensions.reserve(tensor.rank);
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension)
        dimensions.emplace_back(strideMagnitude(tensor.byte_strides[dimension]), tensor.shape[dimension]);
    std::sort(dimensions.begin(), dimensions.end());
    for (auto [stride, extent] : dimensions) {
        if (extent <= 1)
            continue;
        if (!stride || stride <= coveredSpan)
            return false;
        const uint64_t steps = extent - 1;
        if (stride > (std::numeric_limits<uint64_t>::max() - coveredSpan) / steps)
            return false;
        coveredSpan += stride * steps;
    }
    return true;
}

TensorPhysicalOverlap tensorViewsPhysicalOverlap(const VernonTensorView &left, const VernonTensorView &right) {
    const auto supportedStorage = [](VernonTensorStorage storage) {
        return storage == VERNON_TENSOR_HOST || storage == VERNON_TENSOR_RHI_RESOURCE;
    };
    if (!supportedStorage(left.storage) || !supportedStorage(right.storage))
        return TensorPhysicalOverlap::Unknown;
    if (left.storage != right.storage)
        return TensorPhysicalOverlap::Disjoint;
    if (left.storage == VERNON_TENSOR_RHI_RESOURCE) {
        if (!left.resource.identity || !right.resource.identity || !left.resource.resource.value ||
            !right.resource.resource.value)
            return TensorPhysicalOverlap::Unknown;
        if (left.resource.identity != right.resource.identity ||
            left.resource.resource.value != right.resource.resource.value)
            return TensorPhysicalOverlap::Disjoint;
    }

    TensorPhysicalRange leftRange;
    TensorPhysicalRange rightRange;
    if (!tensorPhysicalRange(left, leftRange) || !tensorPhysicalRange(right, rightRange))
        return TensorPhysicalOverlap::Unknown;
    if (leftRange.empty || rightRange.empty || leftRange.begin >= rightRange.end || rightRange.begin >= leftRange.end)
        return TensorPhysicalOverlap::Disjoint;

    uint64_t leftFirst = 0;
    uint64_t rightFirst = 0;
    if (!addPhysicalOffset(leftRange.base, left.byte_offset, leftFirst) ||
        !addPhysicalOffset(rightRange.base, right.byte_offset, rightFirst))
        return TensorPhysicalOverlap::Unknown;
    const size_t leftSize = left.element_layout.byte_size;
    const size_t rightSize = right.element_layout.byte_size;
    if (byteIntervalsOverlap(leftFirst, leftSize, rightFirst, rightSize))
        return TensorPhysicalOverlap::Overlapping;
    if (isRowMajorContiguous(left) && isRowMajorContiguous(right))
        return TensorPhysicalOverlap::Overlapping;
    if (periodicSpansProveDisjoint(left, leftFirst, right, rightFirst))
        return TensorPhysicalOverlap::Disjoint;
    if (latticeProvesDisjoint(leftFirst, leftSize, tensorStrideLattice(left), rightFirst, rightSize,
                              tensorStrideLattice(right)))
        return TensorPhysicalOverlap::Disjoint;
    return TensorPhysicalOverlap::Unknown;
}

bool tensorViewsHaveWritableOverlap(const VernonTensorView &left, const VernonTensorView &right) {
    if (left.access == VERNON_ACCESS_READ && right.access == VERNON_ACCESS_READ)
        return false;
    return tensorViewsPhysicalOverlap(left, right) != TensorPhysicalOverlap::Disjoint;
}

bool hostByteRangesHaveWritableOverlap(const void *leftData, size_t leftSize, bool leftWritable, const void *rightData,
                                       size_t rightSize, bool rightWritable) {
    if ((!leftWritable && !rightWritable) || !leftSize || !rightSize)
        return false;
    if (!leftData || !rightData)
        return true;
    const uintptr_t leftBegin = reinterpret_cast<uintptr_t>(leftData);
    const uintptr_t rightBegin = reinterpret_cast<uintptr_t>(rightData);
    if (leftSize > std::numeric_limits<uintptr_t>::max() - leftBegin ||
        rightSize > std::numeric_limits<uintptr_t>::max() - rightBegin)
        return true;
    return leftBegin < rightBegin + rightSize && rightBegin < leftBegin + leftSize;
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

namespace {

TensorBridgeResult<TensorCopyPlan> compileValueCopyPlan(const VernonValueLayoutView &canonical,
                                                        const TransportNode &physicalValue, size_t byteSize) {
    if (!valueLayoutValid(canonical))
        return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::InvalidLayout)};
    std::vector<TransportScalar> physicalScalars;
    if (!collectTransportScalars(physicalValue, 0, physicalScalars))
        return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::InvalidCopyPlan)};

    TensorCopyPlan plan;
    plan.elementSize = canonical.byte_size;
    plan.byteSize = byteSize;
    size_t scalarIndex = 0;
    for (size_t leafIndex = 0; leafIndex < canonical.leaf_count; ++leafIndex) {
        const VernonValueLeafView &leaf = canonical.leaves[leafIndex];
        auto scalarSize = dataTypeSize(static_cast<VernonDataType>(leaf.dtype));
        if (scalarSize.isErr() || leaf.scalar_count > std::numeric_limits<size_t>::max() / scalarSize.value() ||
            scalarIndex > physicalScalars.size() || leaf.scalar_count > physicalScalars.size() - scalarIndex)
            return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::InvalidCopyPlan)};
        for (size_t lane = 0; lane < leaf.scalar_count; ++lane) {
            const TransportScalar &physical = physicalScalars[scalarIndex++];
            auto representation = dataTypeRepresentation(static_cast<VernonDataType>(leaf.dtype));
            const bool compatibleRepresentation =
                (representation && physical.representation == representation.value()) ||
                (leaf.dtype == VERNON_DATA_U32 &&
                 (physical.representation == "i32" || physical.representation == "u32"));
            if (physical.size != scalarSize.value() || !compatibleRepresentation)
                return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::IncompatibleRepresentation)};
            CopyOperation operation{leaf.byte_offset + lane * scalarSize.value(), physical.offset, scalarSize.value()};
            if (!plan.operations.empty()) {
                CopyOperation &previous = plan.operations.back();
                if (previous.sourceOffset + previous.size == operation.sourceOffset &&
                    previous.destinationOffset + previous.size == operation.destinationOffset) {
                    previous.size += operation.size;
                    continue;
                }
            }
            plan.operations.push_back(operation);
        }
    }
    if (scalarIndex != physicalScalars.size())
        return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::InvalidCopyPlan)};
    return TensorBridgeResult<TensorCopyPlan>{vernon::ok(std::move(plan))};
}

} // namespace

TensorBridgeResult<TensorCopyPlan> compileWholeValueCopyPlan(const VernonValueLayoutView &canonicalValue,
                                                             const TransportNode &physicalValue) {
    if (physicalValue.size > std::numeric_limits<size_t>::max())
        return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::Overflow)};
    return compileValueCopyPlan(canonicalValue, physicalValue, static_cast<size_t>(physicalValue.size));
}

TensorBridgeResult<TensorCopyPlan> compileElementStreamCopyPlan(const VernonValueLayoutView &elementLayout,
                                                                std::vector<uint64_t> logicalShape,
                                                                const TransportNode &physicalStream) {
    if (physicalStream.kind != TransportNodeKind::Array || physicalStream.children.size() != 1 ||
        physicalStream.byteStrides.size() != logicalShape.size() ||
        physicalStream.size > std::numeric_limits<size_t>::max())
        return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::InvalidCopyPlan)};
    auto plan =
        compileValueCopyPlan(elementLayout, physicalStream.children.front(), static_cast<size_t>(physicalStream.size));
    if (plan.isErr())
        return TensorBridgeResult<TensorCopyPlan>{vernon::err(plan.error())};
    plan.value().shape = std::move(logicalShape);
    for (uint64_t stride : physicalStream.byteStrides) {
        if (stride > std::numeric_limits<size_t>::max())
            return TensorBridgeResult<TensorCopyPlan>{vernon::err(TensorBridgeError::Overflow)};
        plan.value().byteStrides.push_back(static_cast<size_t>(stride));
    }
    return plan;
}

TensorBridgeResult<std::vector<uint8_t>> packWholeValue(const VernonTensorView &tensor, const TensorCopyPlan &layout) {
    if (tensor.storage != VERNON_TENSOR_HOST || !valueLayoutValid(tensor.element_layout) || layout.operations.empty() ||
        !layout.shape.empty() || !layout.byteStrides.empty() || layout.byteSize < layout.elementSize)
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::InvalidCopyPlan)};
    std::vector<uint8_t> logical;
    if (tensor.rank == 0 && tensor.element_layout.byte_size == layout.elementSize) {
        auto source = hostTensorData(tensor);
        if (!source || !tensorFitsAllocation(tensor))
            return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::OutOfBounds)};
        logical.assign(source.value(), source.value() + layout.elementSize);
    } else {
        auto packed = packTensorRowMajor(tensor);
        if (packed.isErr())
            return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(packed.error())};
        logical = std::move(packed).value();
        if (logical.size() != layout.elementSize)
            return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::InvalidLayout)};
    }
    std::vector<uint8_t> packed(layout.byteSize);
    for (const CopyOperation &operation : layout.operations) {
        if (operation.sourceOffset > logical.size() || operation.size > logical.size() - operation.sourceOffset ||
            operation.destinationOffset > packed.size() || operation.size > packed.size() - operation.destinationOffset)
            return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::OutOfBounds)};
        std::memcpy(packed.data() + operation.destinationOffset, logical.data() + operation.sourceOffset,
                    operation.size);
    }
    return TensorBridgeResult<std::vector<uint8_t>>{vernon::ok(std::move(packed))};
}

TensorBridgeResult<std::vector<uint8_t>> packTensor(const VernonTensorView &tensor, const TensorCopyPlan &layout) {
    if (layout.shape.empty() && layout.byteStrides.empty())
        return packWholeValue(tensor, layout);
    if (tensor.storage != VERNON_TENSOR_HOST || !valueLayoutValid(tensor.element_layout) ||
        tensor.element_layout.byte_size != layout.elementSize || tensor.rank != layout.shape.size() ||
        layout.byteStrides.size() != layout.shape.size() || (tensor.rank && !tensor.shape) || layout.operations.empty())
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::InvalidCopyPlan)};
    const size_t elementSize = tensor.element_layout.byte_size;
    if (!elementSize || !tensorFitsAllocation(tensor))
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::OutOfBounds)};
    for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension)
        if (tensor.shape[dimension] != layout.shape[dimension])
            return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::InvalidShape)};

    auto elementCount = tensorElementCount(tensor);
    if (elementCount.isErr())
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(elementCount.error())};
    size_t requiredSize = elementCount.value() ? elementSize : 0;
    if (elementCount.value())
        for (uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
            const uint64_t steps = layout.shape[dimension] - 1;
            if (steps && layout.byteStrides[dimension] > (std::numeric_limits<size_t>::max() - requiredSize) / steps)
                return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::Overflow)};
            requiredSize += static_cast<size_t>(steps) * layout.byteStrides[dimension];
        }
    if (requiredSize > layout.byteSize)
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::OutOfBounds)};

    std::vector<uint8_t> packed(layout.byteSize);
    if (!elementCount.value())
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::ok(std::move(packed))};
    auto source = hostTensorData(tensor);
    if (!source)
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::InvalidStorage)};

    for (size_t linear = 0; linear < elementCount.value(); ++linear) {
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
        const uint8_t *sourceElement = source.value() + positiveOffset - negativeOffset;
        for (const CopyOperation &operation : layout.operations) {
            if (destinationOffset > packed.size() || operation.destinationOffset > packed.size() - destinationOffset ||
                operation.size > packed.size() - destinationOffset - operation.destinationOffset ||
                operation.sourceOffset > elementSize || operation.size > elementSize - operation.sourceOffset)
                return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::OutOfBounds)};
            std::memcpy(packed.data() + destinationOffset + operation.destinationOffset,
                        sourceElement + operation.sourceOffset, operation.size);
        }
    }
    return TensorBridgeResult<std::vector<uint8_t>>{vernon::ok(std::move(packed))};
}

TensorBridgeResult<void> unpackTensor(const std::vector<uint8_t> &packed, const VernonTensorView &tensor,
                                      const TensorCopyPlan &layout) {
    if (tensor.storage != VERNON_TENSOR_HOST || tensor.rank != 0 || !layout.shape.empty() ||
        !layout.byteStrides.empty() || packed.size() != layout.byteSize ||
        tensor.element_layout.byte_size != layout.elementSize || !tensorFitsAllocation(tensor))
        return TensorBridgeResult<void>{vernon::err(TensorBridgeError::InvalidCopyPlan)};
    if (!tensor.host_data)
        return TensorBridgeResult<void>{vernon::err(TensorBridgeError::InvalidStorage)};
    uint8_t *destination = static_cast<uint8_t *>(const_cast<void *>(tensor.host_data)) + tensor.byte_offset;
    for (const CopyOperation &operation : layout.operations) {
        if (operation.destinationOffset > packed.size() ||
            operation.size > packed.size() - operation.destinationOffset ||
            operation.sourceOffset > layout.elementSize || operation.size > layout.elementSize - operation.sourceOffset)
            return TensorBridgeResult<void>{vernon::err(TensorBridgeError::OutOfBounds)};
        std::memcpy(destination + operation.sourceOffset, packed.data() + operation.destinationOffset, operation.size);
    }
    return TensorBridgeResult<void>{vernon::ok()};
}

TensorBridgeResult<std::vector<uint8_t>> packTensorRowMajor(const VernonTensorView &tensor) {
    const size_t elementSize = valueLayoutValid(tensor.element_layout) ? tensor.element_layout.byte_size : 0;
    auto packedSize = tensorLogicalByteSize(tensor);
    if (!elementSize)
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::InvalidLayout)};
    if (packedSize.isErr())
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(packedSize.error())};
    if (tensor.rank && !tensor.shape)
        return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::InvalidShape)};
    TensorCopyPlan layout;
    layout.elementSize = elementSize;
    if (tensor.rank)
        layout.shape.assign(tensor.shape, tensor.shape + tensor.rank);
    layout.byteStrides.resize(tensor.rank);
    size_t stride = elementSize;
    for (uint32_t dimension = tensor.rank; dimension-- > 0;) {
        layout.byteStrides[dimension] = stride;
        if (layout.shape[dimension] && stride > std::numeric_limits<size_t>::max() / layout.shape[dimension])
            return TensorBridgeResult<std::vector<uint8_t>>{vernon::err(TensorBridgeError::Overflow)};
        stride *= static_cast<size_t>(layout.shape[dimension]);
    }
    layout.byteSize = packedSize.value();
    layout.operations.push_back({0, 0, elementSize});
    return packTensor(tensor, layout);
}

const char *tensorBridgeErrorMessage(TensorBridgeError error) noexcept {
    switch (error) {
    case TensorBridgeError::InvalidDataType:
        return "Tensor data type is invalid";
    case TensorBridgeError::InvalidShape:
        return "Tensor shape metadata is invalid";
    case TensorBridgeError::InvalidLayout:
        return "Tensor value layout is invalid";
    case TensorBridgeError::InvalidStorage:
        return "Tensor storage is invalid";
    case TensorBridgeError::InvalidCopyPlan:
        return "Tensor copy plan is invalid";
    case TensorBridgeError::IncompatibleRepresentation:
        return "Tensor physical representation is incompatible";
    case TensorBridgeError::OutOfBounds:
        return "Tensor byte range is out of bounds";
    case TensorBridgeError::Overflow:
        return "Tensor byte arithmetic overflowed";
    }
    return "unknown Tensor bridge error";
}

} // namespace vernon::runtime
