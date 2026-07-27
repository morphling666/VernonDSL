#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::vernon {

struct StorageLeaf {
    Type type;
    uint64_t byteOffset{};
    uint64_t scalarCount{};
};

struct StorageLayout {
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<StorageLeaf> leaves;
};

inline FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> resolveStructFields(StructType structure,
                                                                                 ModuleOp module) {
    StructDeclOp declaration;
    for (StructDeclOp candidate : module.getOps<StructDeclOp>()) {
        if (candidate.getSymName() == structure.getName()) {
            if (declaration)
                return failure();
            declaration = candidate;
        }
    }
    if (!declaration)
        return failure();
    SmallVector<Type> fields;
    for (Attribute attribute : declaration.getFields()) {
        StringRef spelling = cast<StringAttr>(attribute).getValue();
        size_t separator = spelling.find(':');
        if (separator == StringRef::npos)
            return failure();
        Type field = parseType(spelling.drop_front(separator + 1), module.getContext());
        if (!field)
            return failure();
        fields.push_back(field);
    }
    return std::make_pair(declaration, std::move(fields));
}

inline FailureOr<StorageLayout> resolveStorageLayout(Type type, ModuleOp module) {
    FailureOr<ValueAbiLayout> valueLayout = getValueAbiLayout(type, module);
    if (failed(valueLayout))
        return failure();
    StorageLayout result;
    result.size = valueLayout->size;
    result.alignment = valueLayout->alignment;
    result.leaves.reserve(valueLayout->leaves.size());
    for (const ValueAbiLeaf &leaf : valueLayout->leaves)
        result.leaves.push_back({leaf.scalarType, leaf.byteOffset, leaf.scalarCount});
    return result;
}

} // namespace mlir::vernon

#endif
