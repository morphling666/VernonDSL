#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"

#include <limits>

namespace mlir::vernon {

struct StorageLeaf {
    Type type;
    uint64_t byteOffset{};
};

struct StorageLayout {
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<StorageLeaf> leaves;
};

inline FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> resolveStructFields(StructType structure,
                                                                                 ModuleOp module) {
    StructDeclOp declaration;
    for (StructDeclOp candidate : module.getOps<StructDeclOp>())
        if (candidate.getSymName() == structure.getName()) {
            declaration = candidate;
            break;
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

inline FailureOr<StorageLayout> resolveStorageLayout(Type type, ModuleOp module,
                                                     SmallVectorImpl<StringRef> &activeStructs) {
    if (type.isIntOrFloat()) {
        uint64_t size = std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
        return StorageLayout{size, size, {{type, 0}}};
    }

    auto productLayout = [&](TypeRange fields, ArrayRef<int64_t> explicitOffsets = {}) -> FailureOr<StorageLayout> {
        StorageLayout result;
        result.alignment = 1;
        uint64_t offset = 0;
        for (auto [index, field] : llvm::enumerate(fields)) {
            FailureOr<StorageLayout> layout = resolveStorageLayout(field, module, activeStructs);
            if (failed(layout))
                return failure();
            offset = explicitOffsets.empty() ? llvm::alignTo(offset, layout->alignment)
                                             : static_cast<uint64_t>(explicitOffsets[index]);
            for (StorageLeaf leaf : layout->leaves) {
                leaf.byteOffset += offset;
                result.leaves.push_back(leaf);
            }
            offset += layout->size;
            result.alignment = std::max(result.alignment, layout->alignment);
        }
        result.size = llvm::alignTo(offset, result.alignment);
        return result;
    };

    if (auto tuple = dyn_cast<TupleType>(type))
        return productLayout(tuple.getTypes());

    auto tensorLayout = [&](Type element, ArrayRef<int64_t> shape) -> FailureOr<StorageLayout> {
        FailureOr<StorageLayout> elementLayout = resolveStorageLayout(element, module, activeStructs);
        if (failed(elementLayout))
            return failure();
        uint64_t count = 1;
        for (int64_t dimension : shape) {
            if (dimension <= 0 || count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension))
                return failure();
            count *= static_cast<uint64_t>(dimension);
        }
        const uint64_t stride = llvm::alignTo(elementLayout->size, elementLayout->alignment);
        StorageLayout result;
        result.size = count * stride;
        result.alignment = elementLayout->alignment;
        result.leaves.reserve(elementLayout->leaves.size() * count);
        for (uint64_t index = 0; index < count; ++index)
            for (StorageLeaf leaf : elementLayout->leaves) {
                leaf.byteOffset += index * stride;
                result.leaves.push_back(leaf);
            }
        return result;
    };

    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape())
            return failure();
        return tensorLayout(tensor.getElementType(), tensor.getShape());
    }
    if (auto tensor = dyn_cast<TensorType>(type))
        return tensorLayout(tensor.getElementType(), tensor.getShape());

    auto structure = dyn_cast<StructType>(type);
    if (!structure)
        return failure();
    if (llvm::is_contained(activeStructs, structure.getName()))
        return failure();
    FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> resolved = resolveStructFields(structure, module);
    if (failed(resolved))
        return failure();
    StructDeclOp declaration = resolved->first;
    SmallVector<Type> &fields = resolved->second;
    auto offsets = declaration->getAttrOfType<DenseI64ArrayAttr>("abi_field_offsets");
    auto size = declaration->getAttrOfType<IntegerAttr>("abi_size");
    auto alignment = declaration->getAttrOfType<IntegerAttr>("abi_alignment");
    if (!offsets || offsets.size() != fields.size() || !size || !alignment)
        return failure();
    activeStructs.push_back(structure.getName());
    FailureOr<StorageLayout> result = productLayout(fields, offsets.asArrayRef());
    activeStructs.pop_back();
    if (failed(result))
        return failure();
    result->size = static_cast<uint64_t>(size.getInt());
    result->alignment = static_cast<uint64_t>(alignment.getInt());
    return result;
}

inline FailureOr<StorageLayout> resolveStorageLayout(Type type, ModuleOp module) {
    SmallVector<StringRef> activeStructs;
    return resolveStorageLayout(type, module, activeStructs);
}

} // namespace mlir::vernon

#endif
