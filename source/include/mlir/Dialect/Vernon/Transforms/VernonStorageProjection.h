#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::vernon {

struct ResolvedStructField {
    std::string name;
    Type type;
};

struct ResolvedStructFields {
    StructDeclOp declaration;
    SmallVector<ResolvedStructField> fields;
};

inline FailureOr<ResolvedStructFields> resolveNamedStructFields(StructType structure, ModuleOp module) {
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
    SmallVector<ResolvedStructField> fields;
    for (Attribute attribute : declaration.getFields()) {
        StringRef spelling = cast<StringAttr>(attribute).getValue();
        size_t separator = spelling.find(':');
        if (separator == StringRef::npos)
            return failure();
        Type field = parseType(spelling.drop_front(separator + 1), module.getContext());
        if (!field)
            return failure();
        fields.push_back({spelling.take_front(separator).str(), field});
    }
    return ResolvedStructFields{declaration, std::move(fields)};
}

inline FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> resolveStructFields(StructType structure,
                                                                                 ModuleOp module) {
    FailureOr<ResolvedStructFields> resolved = resolveNamedStructFields(structure, module);
    if (failed(resolved))
        return failure();
    SmallVector<Type> fields;
    fields.reserve(resolved->fields.size());
    for (const ResolvedStructField &field : resolved->fields)
        fields.push_back(field.type);
    return std::make_pair(resolved->declaration, std::move(fields));
}

} // namespace mlir::vernon

#endif
