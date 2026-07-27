#include "mlir/Dialect/Vernon/Transforms/VernonValueAbi.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir::vernon {
namespace {

struct PlannedLayout {
    ValueAbiLayout layout;
    std::string canonical;
};

struct StructField {
    std::string name;
    Type type;
};

FailureOr<SmallVector<StructField>> resolveStructFields(StructType structure, ModuleOp module) {
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

    SmallVector<StructField> fields;
    for (Attribute attribute : declaration.getFields()) {
        StringRef spelling = cast<StringAttr>(attribute).getValue();
        size_t separator = spelling.find(':');
        if (separator == StringRef::npos)
            return failure();
        Type type = parseType(spelling.drop_front(separator + 1), module.getContext());
        if (!type)
            return failure();
        fields.push_back({spelling.take_front(separator).str(), type});
    }
    return fields;
}

FailureOr<uint64_t> checkedAlign(uint64_t value, uint64_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<uint64_t>::max() - (alignment - 1))
        return failure();
    return llvm::alignTo(value, alignment);
}

FailureOr<uint64_t> checkedMultiply(uint64_t left, uint64_t right) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right)
        return failure();
    return left * right;
}

bool isCompatibleLogicalDtype(Type scalar, StringRef dtype) {
    if (dtype == "bool")
        return scalar.isInteger(1);
    if (dtype == "i32" || dtype == "u32")
        return scalar.isInteger(32);
    if (dtype == "f16")
        return scalar.isF16();
    if (dtype == "f32")
        return scalar.isF32();
    if (dtype == "f64")
        return scalar.isF64();
    return false;
}

void prependPath(ValueAbiLeaf &leaf, ArrayRef<ValueAbiPathComponent> prefix) {
    SmallVector<ValueAbiPathComponent> path(prefix.begin(), prefix.end());
    path.append(leaf.path.begin(), leaf.path.end());
    leaf.path = std::move(path);
}

FailureOr<PlannedLayout> planValue(Type type, ModuleOp module, SmallVectorImpl<StringRef> &activeStructs);

FailureOr<PlannedLayout> planProduct(ArrayRef<StructField> fields, StringRef kind, ModuleOp module,
                                     SmallVectorImpl<StringRef> &activeStructs) {
    PlannedLayout result;
    result.layout.alignment = 1;
    uint64_t offset = 0;
    std::string fieldCanonical;
    llvm::raw_string_ostream fieldStream(fieldCanonical);
    for (auto [index, field] : llvm::enumerate(fields)) {
        FailureOr<PlannedLayout> child = planValue(field.type, module, activeStructs);
        if (failed(child))
            return failure();
        FailureOr<uint64_t> aligned = checkedAlign(offset, child->layout.alignment);
        if (failed(aligned) || child->layout.size > std::numeric_limits<uint64_t>::max() - *aligned)
            return failure();
        offset = *aligned;
        result.layout.fieldOffsets.push_back(offset);
        for (ValueAbiLeaf leaf : child->layout.leaves) {
            leaf.byteOffset += offset;
            ValueAbiPathComponent component = kind.starts_with("struct") ? ValueAbiPathComponent::getField(field.name)
                                                                         : ValueAbiPathComponent::getIndex(index);
            prependPath(leaf, {component});
            result.layout.leaves.push_back(std::move(leaf));
        }
        fieldStream << (index == 0 ? "" : ";") << field.name << '@' << offset << ':' << child->canonical;
        offset += child->layout.size;
        result.layout.alignment = std::max(result.layout.alignment, child->layout.alignment);
    }
    FailureOr<uint64_t> size = checkedAlign(offset, result.layout.alignment);
    if (failed(size))
        return failure();
    result.layout.size = *size;
    fieldStream.flush();
    std::string canonical;
    llvm::raw_string_ostream stream(canonical);
    stream << kind << "(align=" << result.layout.alignment << ",size=" << result.layout.size << ';' << fieldCanonical
           << ')';
    stream.flush();
    result.canonical = std::move(canonical);
    return result;
}

FailureOr<PlannedLayout> planTensor(Type element, ArrayRef<int64_t> shape, ModuleOp module,
                                    SmallVectorImpl<StringRef> &activeStructs) {
    if (shape.empty())
        return failure();
    uint64_t count = 1;
    for (int64_t extent : shape) {
        if (extent <= 0)
            return failure();
        FailureOr<uint64_t> product = checkedMultiply(count, static_cast<uint64_t>(extent));
        if (failed(product))
            return failure();
        count = *product;
    }
    FailureOr<PlannedLayout> child = planValue(element, module, activeStructs);
    if (failed(child))
        return failure();
    FailureOr<uint64_t> stride = checkedAlign(child->layout.size, child->layout.alignment);
    FailureOr<uint64_t> size = succeeded(stride) ? checkedMultiply(*stride, count) : FailureOr<uint64_t>(failure());
    if (failed(size))
        return failure();

    PlannedLayout result;
    result.layout.size = *size;
    result.layout.alignment = child->layout.alignment;
    result.layout.elementStride = *stride;
    if (element.isIntOrFloat()) {
        SmallVector<uint64_t> leafShape;
        leafShape.reserve(shape.size());
        for (int64_t extent : shape)
            leafShape.push_back(static_cast<uint64_t>(extent));
        result.layout.leaves.push_back(
            {{}, std::move(leafShape), element, child->layout.leaves.front().dtype, 0, count});
    } else {
        for (uint64_t linear = 0; linear < count; ++linear) {
            SmallVector<ValueAbiPathComponent> coordinates(shape.size());
            uint64_t remaining = linear;
            for (size_t dimension = shape.size(); dimension-- > 0;) {
                coordinates[dimension] =
                    ValueAbiPathComponent::getIndex(remaining % static_cast<uint64_t>(shape[dimension]));
                remaining /= static_cast<uint64_t>(shape[dimension]);
            }
            for (ValueAbiLeaf leaf : child->layout.leaves) {
                leaf.byteOffset += linear * *stride;
                prependPath(leaf, coordinates);
                result.layout.leaves.push_back(std::move(leaf));
            }
        }
    }
    std::string canonical;
    llvm::raw_string_ostream stream(canonical);
    stream << "tensor([";
    for (auto [index, extent] : llvm::enumerate(shape))
        stream << (index == 0 ? "" : ",") << extent;
    stream << "],stride=" << *stride << ",size=" << *size << ",element=" << child->canonical << ')';
    stream.flush();
    result.canonical = std::move(canonical);
    return result;
}

FailureOr<PlannedLayout> planValue(Type type, ModuleOp module, SmallVectorImpl<StringRef> &activeStructs) {
    if (type.isIntOrFloat()) {
        uint64_t size = std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
        StringRef name;
        if (type.isInteger(1))
            name = "bool";
        else if (auto integer = dyn_cast<IntegerType>(type))
            name = integer.isUnsigned() ? "u32" : "i32";
        else if (type.isF16())
            name = "f16";
        else if (type.isF32())
            name = "f32";
        else if (type.isF64())
            name = "f64";
        else
            return failure();
        std::string canonical;
        llvm::raw_string_ostream stream(canonical);
        stream << "scalar(" << name << ',' << size << ',' << size << ')';
        stream.flush();
        return PlannedLayout{ValueAbiLayout{size, size, {}, std::nullopt, {{{}, {}, type, name.str(), 0, 1}}, {}},
                             std::move(canonical)};
    }
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape())
            return failure();
        return planTensor(tensor.getElementType(), tensor.getShape(), module, activeStructs);
    }
    if (auto vector = dyn_cast<VectorType>(type))
        return planTensor(vector.getElementType(), vector.getShape(), module, activeStructs);
    if (auto tensor = dyn_cast<TensorType>(type))
        return planTensor(tensor.getElementType(), tensor.getShape(), module, activeStructs);
    if (auto tuple = dyn_cast<TupleType>(type)) {
        SmallVector<StructField> fields;
        for (auto [index, element] : llvm::enumerate(tuple.getTypes()))
            fields.push_back({std::to_string(index), element});
        return planProduct(fields, "tuple", module, activeStructs);
    }
    auto structure = dyn_cast<StructType>(type);
    if (!structure || llvm::is_contained(activeStructs, structure.getName()))
        return failure();
    FailureOr<SmallVector<StructField>> fields = resolveStructFields(structure, module);
    if (failed(fields))
        return failure();
    activeStructs.push_back(structure.getName());
    std::string kind = ("struct(" + structure.getName() + ")").str();
    FailureOr<PlannedLayout> result = planProduct(*fields, kind, module, activeStructs);
    activeStructs.pop_back();
    if (succeeded(result)) {
        for (StructDeclOp declaration : module.getOps<StructDeclOp>()) {
            if (declaration.getSymName() != structure.getName())
                continue;
            auto dtypes = declaration->getAttrOfType<ArrayAttr>("abi_leaf_dtypes");
            if (!dtypes)
                continue;
            if (dtypes.size() != result->layout.leaves.size())
                return failure();
            for (auto [leaf, dtypeAttr] : llvm::zip_equal(result->layout.leaves, dtypes)) {
                auto dtype = dyn_cast<StringAttr>(dtypeAttr);
                if (!dtype || !llvm::is_contained({"bool", "i32", "u32", "f16", "f32", "f64"}, dtype.getValue()))
                    return failure();
                leaf.dtype = dtype.getValue().str();
            }
            break;
        }
    }
    return result;
}

} // namespace

ValueAbiPathComponent ValueAbiPathComponent::getField(StringRef name) { return ValueAbiPathComponent{name.str(), 0}; }

ValueAbiPathComponent ValueAbiPathComponent::getIndex(uint64_t index) {
    return ValueAbiPathComponent{std::nullopt, index};
}

FailureOr<ValueAbiLayout> getValueAbiLayout(Type type, ModuleOp module, ArrayRef<StringRef> logicalLeafDtypes) {
    SmallVector<StringRef> activeStructs;
    FailureOr<PlannedLayout> planned = planValue(type, module, activeStructs);
    if (failed(planned))
        return failure();
    if (!logicalLeafDtypes.empty()) {
        // Signless i32 cannot recover the source i32/u32 distinction. Apply
        // validated frontend metadata before hashing or exposing ABI leaves.
        if (logicalLeafDtypes.size() != planned->layout.leaves.size())
            return failure();
        for (auto [leaf, dtype] : llvm::zip_equal(planned->layout.leaves, logicalLeafDtypes)) {
            if (!isCompatibleLogicalDtype(leaf.scalarType, dtype))
                return failure();
            leaf.dtype = dtype.str();
        }
    }
    llvm::SHA256 hash;
    hash.update(planned->canonical);
    hash.update("|dtypes=");
    for (auto [index, leaf] : llvm::enumerate(planned->layout.leaves)) {
        if (index != 0)
            hash.update(",");
        hash.update(leaf.dtype);
    }
    planned->layout.layoutHash = llvm::toHex(hash.final(), true);
    return std::move(planned->layout);
}

} // namespace mlir::vernon
