#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace vernon::runtime::shape {

class Extent {
public:
    static Extent dynamic() { return Extent(); }
    static Extent fixed(uint64_t value) { return Extent(value); }

    bool isDynamic() const { return !value_; }
    bool isStatic() const { return value_.has_value(); }
    std::optional<uint64_t> staticValue() const { return value_; }

    friend bool operator==(const Extent &left, const Extent &right) { return left.value_ == right.value_; }
    friend bool operator!=(const Extent &left, const Extent &right) { return !(left == right); }

private:
    Extent() = default;
    explicit Extent(uint64_t value) : value_(value) {}

    std::optional<uint64_t> value_;
};

using DeclaredShape = std::vector<Extent>;
using ConcreteShape = std::vector<uint64_t>;
using ByteStrides = std::vector<int64_t>;

struct ValueAxisExtent {
    uint32_t value{};
    uint32_t axis{};
};

struct SymbolExtent {
    std::string symbol;
};

class ExtentExpression {
public:
    enum class Kind {
        Static,
        Dynamic,
        ValueAxis,
        Symbol,
    };

    static ExtentExpression fixed(uint64_t value);
    static ExtentExpression dynamic();
    static ExtentExpression valueAxis(uint32_t value, uint32_t axis);
    static ExtentExpression symbol(std::string value);

    Kind kind() const { return kind_; }
    std::optional<uint64_t> staticValue() const { return staticValue_; }
    std::optional<ValueAxisExtent> valueAxis() const { return valueAxis_; }
    const std::string &symbol() const { return symbol_; }

private:
    Kind kind_{Kind::Dynamic};
    std::optional<uint64_t> staticValue_;
    std::optional<ValueAxisExtent> valueAxis_;
    std::string symbol_;
};

std::optional<Extent> decodeReflectedExtent(int64_t encoded);
std::optional<Extent> decodeRuntimeContractExtent(uint64_t encoded);
std::optional<DeclaredShape> decodeReflectedShape(const std::vector<int64_t> &encoded);
DeclaredShape decodeRuntimeContractShape(const std::vector<uint64_t> &encoded);
std::optional<int64_t> encodeReflectedExtent(Extent extent);
uint64_t encodeRuntimeContractExtent(Extent extent);
std::optional<std::vector<int64_t>> encodeReflectedShape(const DeclaredShape &shape);
std::vector<uint64_t> encodeRuntimeContractShape(const DeclaredShape &shape);

bool isConcrete(const DeclaredShape &shape);
std::optional<ConcreteShape> concrete(const DeclaredShape &shape);
bool matches(const DeclaredShape &declared, const ConcreteShape &actual);
bool checkedElementCount(const ConcreteShape &shape, size_t &count);
bool rowMajorByteStrides(const ConcreteShape &shape, size_t elementBytes, ByteStrides &strides);
bool materializeLeafProjection(const ConcreteShape &ownerShape, const ByteStrides &ownerStrides,
                               const DeclaredShape &physicalShape, size_t leafElementBytes,
                               ConcreteShape &projectedShape, ByteStrides &projectedStrides);
bool materializeCompactProjection(const ConcreteShape &ownerShape, const DeclaredShape &physicalShape,
                                  size_t elementBytes, ConcreteShape &projectedShape, ByteStrides &projectedStrides);
bool resolveSingleDynamicExtent(const DeclaredShape &declared, size_t elementBytes, size_t logicalBytes,
                                ConcreteShape &resolved);

} // namespace vernon::runtime::shape
