#include "shape_layout.h"

#include <utility>

namespace vernon::runtime::shape {
namespace {

bool resolveProjectionShape(const ConcreteShape &ownerShape, const DeclaredShape &physicalShape,
                            ConcreteShape &projectedShape) {
    if (physicalShape.size() < ownerShape.size())
        return false;
    const DeclaredShape ownerDeclaration(physicalShape.begin(),
                                         physicalShape.begin() + static_cast<ptrdiff_t>(ownerShape.size()));
    if (!matches(ownerDeclaration, ownerShape))
        return false;
    const DeclaredShape payloadDeclaration(physicalShape.begin() + static_cast<ptrdiff_t>(ownerShape.size()),
                                           physicalShape.end());
    const std::optional<ConcreteShape> payloadShape = concrete(payloadDeclaration);
    if (!payloadShape)
        return false;
    projectedShape = ownerShape;
    projectedShape.insert(projectedShape.end(), payloadShape->begin(), payloadShape->end());
    return true;
}

} // namespace

ExtentExpression ExtentExpression::fixed(uint64_t value) {
    ExtentExpression result;
    result.kind_ = Kind::Static;
    result.staticValue_ = value;
    return result;
}

ExtentExpression ExtentExpression::dynamic() { return {}; }

ExtentExpression ExtentExpression::valueAxis(uint32_t value, uint32_t axis) {
    ExtentExpression result;
    result.kind_ = Kind::ValueAxis;
    result.valueAxis_ = ValueAxisExtent{value, axis};
    return result;
}

ExtentExpression ExtentExpression::symbol(std::string value) {
    ExtentExpression result;
    result.kind_ = Kind::Symbol;
    result.symbol_ = std::move(value);
    return result;
}

std::optional<Extent> decodeReflectedExtent(int64_t encoded) {
    if (encoded == -1)
        return Extent::dynamic();
    if (encoded < 0)
        return std::nullopt;
    return Extent::fixed(static_cast<uint64_t>(encoded));
}

std::optional<Extent> decodeRuntimeContractExtent(uint64_t encoded) {
    return encoded ? Extent::fixed(encoded) : Extent::dynamic();
}

std::optional<DeclaredShape> decodeReflectedShape(const std::vector<int64_t> &encoded) {
    DeclaredShape result;
    result.reserve(encoded.size());
    for (int64_t value : encoded) {
        std::optional<Extent> extent = decodeReflectedExtent(value);
        if (!extent)
            return std::nullopt;
        result.push_back(*extent);
    }
    return result;
}

DeclaredShape decodeRuntimeContractShape(const std::vector<uint64_t> &encoded) {
    DeclaredShape result;
    result.reserve(encoded.size());
    for (uint64_t value : encoded)
        result.push_back(*decodeRuntimeContractExtent(value));
    return result;
}

std::optional<int64_t> encodeReflectedExtent(Extent extent) {
    const std::optional<uint64_t> value = extent.staticValue();
    if (!value)
        return int64_t{-1};
    if (*value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return std::nullopt;
    return static_cast<int64_t>(*value);
}

uint64_t encodeRuntimeContractExtent(Extent extent) { return extent.staticValue().value_or(0); }

std::optional<std::vector<int64_t>> encodeReflectedShape(const DeclaredShape &shape) {
    std::vector<int64_t> result;
    result.reserve(shape.size());
    for (Extent extent : shape) {
        const std::optional<int64_t> encoded = encodeReflectedExtent(extent);
        if (!encoded)
            return std::nullopt;
        result.push_back(*encoded);
    }
    return result;
}

std::vector<uint64_t> encodeRuntimeContractShape(const DeclaredShape &shape) {
    std::vector<uint64_t> result;
    result.reserve(shape.size());
    for (Extent extent : shape)
        result.push_back(encodeRuntimeContractExtent(extent));
    return result;
}

bool isConcrete(const DeclaredShape &shape) {
    for (Extent extent : shape)
        if (extent.isDynamic())
            return false;
    return true;
}

std::optional<ConcreteShape> concrete(const DeclaredShape &shape) {
    ConcreteShape result;
    result.reserve(shape.size());
    for (Extent extent : shape) {
        const std::optional<uint64_t> value = extent.staticValue();
        if (!value)
            return std::nullopt;
        result.push_back(*value);
    }
    return result;
}

bool matches(const DeclaredShape &declared, const ConcreteShape &actual) {
    if (declared.size() != actual.size())
        return false;
    for (size_t axis = 0; axis < declared.size(); ++axis)
        if (const std::optional<uint64_t> value = declared[axis].staticValue(); value && *value != actual[axis])
            return false;
    return true;
}

bool checkedElementCount(const ConcreteShape &shape, size_t &count) {
    count = 1;
    for (uint64_t extent : shape) {
        if (extent > std::numeric_limits<size_t>::max() ||
            (extent && count > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent)))
            return false;
        count *= static_cast<size_t>(extent);
    }
    return true;
}

bool rowMajorByteStrides(const ConcreteShape &shape, size_t elementBytes, ByteStrides &strides) {
    if (!elementBytes)
        return false;
    strides.resize(shape.size());
    size_t stride = elementBytes;
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return false;
        strides[axis] = static_cast<int64_t>(stride);
        if (shape[axis] && stride > std::numeric_limits<size_t>::max() / shape[axis])
            return false;
        stride *= static_cast<size_t>(shape[axis]);
    }
    return true;
}

bool materializeLeafProjection(const ConcreteShape &ownerShape, const ByteStrides &ownerStrides,
                               const DeclaredShape &physicalShape, size_t leafElementBytes,
                               ConcreteShape &projectedShape, ByteStrides &projectedStrides) {
    if (ownerShape.size() != ownerStrides.size() || !resolveProjectionShape(ownerShape, physicalShape, projectedShape))
        return false;
    const ConcreteShape payloadShape(projectedShape.begin() + static_cast<ptrdiff_t>(ownerShape.size()),
                                     projectedShape.end());
    ByteStrides payloadStrides;
    if (!payloadShape.empty() && !rowMajorByteStrides(payloadShape, leafElementBytes, payloadStrides))
        return false;
    projectedStrides = ownerStrides;
    projectedStrides.insert(projectedStrides.end(), payloadStrides.begin(), payloadStrides.end());
    return true;
}

bool materializeCompactProjection(const ConcreteShape &ownerShape, const DeclaredShape &physicalShape,
                                  size_t elementBytes, ConcreteShape &projectedShape, ByteStrides &projectedStrides) {
    if (!resolveProjectionShape(ownerShape, physicalShape, projectedShape))
        return false;
    return rowMajorByteStrides(projectedShape, elementBytes, projectedStrides);
}

bool resolveSingleDynamicExtent(const DeclaredShape &declared, size_t elementBytes, size_t logicalBytes,
                                ConcreteShape &resolved) {
    if (!elementBytes || logicalBytes % elementBytes)
        return false;
    size_t dynamicAxis = declared.size();
    size_t staticElements = 1;
    resolved.resize(declared.size());
    for (size_t axis = 0; axis < declared.size(); ++axis) {
        const std::optional<uint64_t> extent = declared[axis].staticValue();
        if (!extent) {
            if (dynamicAxis != declared.size())
                return false;
            dynamicAxis = axis;
            continue;
        }
        resolved[axis] = *extent;
        if (*extent && staticElements > std::numeric_limits<size_t>::max() / *extent)
            return false;
        staticElements *= static_cast<size_t>(*extent);
    }
    const size_t elements = logicalBytes / elementBytes;
    if (dynamicAxis == declared.size())
        return staticElements == elements;
    if (!staticElements || elements % staticElements)
        return false;
    resolved[dynamicAxis] = elements / staticElements;
    return true;
}

} // namespace vernon::runtime::shape
