#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::vernon {

struct ValueAbiPathComponent {
    std::optional<std::string> field;
    uint64_t index{};

    static ValueAbiPathComponent getField(StringRef name);
    static ValueAbiPathComponent getIndex(uint64_t index);
};

struct ValueAbiLeaf {
    SmallVector<ValueAbiPathComponent> path;
    SmallVector<uint64_t> shape;
    Type scalarType;
    std::string dtype;
    uint64_t byteOffset{};
    uint64_t scalarCount{};
};

struct ValueAbiLayout {
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<uint64_t> fieldOffsets;
    std::optional<uint64_t> elementStride;
    SmallVector<ValueAbiLeaf> leaves;
    std::string layoutHash;
};

FailureOr<ValueAbiLayout> getValueAbiLayout(Type type, ModuleOp module, ArrayRef<StringRef> logicalLeafDtypes = {});

} // namespace mlir::vernon
