#pragma once

#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"

namespace mlir::vernon {

struct AttributeAbiLeaf {
    SmallVector<ValueAbiPathComponent> path;
    Type scalarType;
    std::string dtype;
    uint32_t locationOffset{};
    uint32_t componentCount{};
    uint64_t byteOffset{};
};

struct AttributeAbiLayout {
    ValueAbiLayout valueLayout;
    SmallVector<AttributeAbiLeaf> leaves;

    uint32_t getLocationSpan() const { return static_cast<uint32_t>(leaves.size()); }
};

FailureOr<AttributeAbiLayout> getAttributeAbiLayout(Type type, ModuleOp module,
                                                    ArrayRef<StringRef> logicalLeafDtypes = {});

} // namespace mlir::vernon
