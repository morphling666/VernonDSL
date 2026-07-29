#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir::vernon {

FailureOr<ResolvedStructFields> resolveNamedStructFields(StructType structure, ModuleOp module) {
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
        auto spellingAttribute = dyn_cast<StringAttr>(attribute);
        if (!spellingAttribute)
            return failure();
        StringRef spelling = spellingAttribute.getValue();
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

FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> resolveStructFields(StructType structure, ModuleOp module) {
    FailureOr<ResolvedStructFields> resolved = resolveNamedStructFields(structure, module);
    if (failed(resolved))
        return failure();
    SmallVector<Type> fields;
    fields.reserve(resolved->fields.size());
    for (const ResolvedStructField &field : resolved->fields)
        fields.push_back(field.type);
    return std::make_pair(resolved->declaration, std::move(fields));
}

namespace {

struct PlannedLayout {
    ValueAbiLayout layout;
    std::string canonical;
};

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

FailureOr<SmallVector<uint64_t>> rowMajorStrides(ArrayRef<int64_t> shape, uint64_t elementStride) {
    SmallVector<uint64_t> strides(shape.size());
    uint64_t stride = elementStride;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (shape[dimension] <= 0)
            return failure();
        strides[dimension] = stride;
        FailureOr<uint64_t> next = checkedMultiply(stride, static_cast<uint64_t>(shape[dimension]));
        if (failed(next))
            return failure();
        stride = *next;
    }
    return strides;
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

FailureOr<PhysicalValueAbiLayout> planCudaValue(Type type, ModuleOp module, SmallVectorImpl<StringRef> &activeStructs) {
    if (type.isIndex())
        return PhysicalValueAbiLayout{8, 8, {}, {0}};
    if (type.isIntOrFloat()) {
        const uint64_t size = std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
        return PhysicalValueAbiLayout{size, size, {}, {0}};
    }

    Type tensorElement;
    ArrayRef<int64_t> tensorShape;
    bool registerTensor = false;
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape() || !tensor.getElementType().isIntOrFloat())
            return failure();
        tensorElement = tensor.getElementType();
        tensorShape = tensor.getShape();
        registerTensor = tensor.getNumElements() <= 16;
    } else if (auto tensor = dyn_cast<TensorType>(type)) {
        tensorElement = tensor.getElementType();
        tensorShape = tensor.getShape();
    } else if (auto vector = dyn_cast<VectorType>(type)) {
        tensorElement = vector.getElementType();
        tensorShape = vector.getShape();
        registerTensor = true;
    }
    if (tensorElement) {
        FailureOr<PhysicalValueAbiLayout> element = planCudaValue(tensorElement, module, activeStructs);
        if (failed(element))
            return failure();
        FailureOr<SmallVector<uint64_t>> strides =
            rowMajorStrides(tensorShape, llvm::alignTo(element->size, element->alignment));
        if (failed(strides))
            return failure();
        uint64_t count = 1;
        for (int64_t extent : tensorShape) {
            if (extent <= 0)
                return failure();
            FailureOr<uint64_t> next = checkedMultiply(count, static_cast<uint64_t>(extent));
            if (failed(next))
                return failure();
            count = *next;
        }
        FailureOr<uint64_t> size = checkedMultiply(llvm::alignTo(element->size, element->alignment), count);
        if (failed(size))
            return failure();
        uint64_t alignment = element->alignment;
        if (registerTensor && count > 1) {
            alignment = element->size;
            while (alignment < *size && alignment < 16)
                alignment *= 2;
            alignment = std::min<uint64_t>(alignment, 16);
        }
        return PhysicalValueAbiLayout{*size, alignment, std::move(*strides), std::move(element->elementLeafOffsets)};
    }

    SmallVector<Type> fields;
    std::optional<StringRef> activeName;
    if (auto tuple = dyn_cast<TupleType>(type)) {
        fields.append(tuple.getTypes().begin(), tuple.getTypes().end());
    } else if (auto structure = dyn_cast<StructType>(type)) {
        if (llvm::is_contained(activeStructs, structure.getName()))
            return failure();
        FailureOr<ResolvedStructFields> resolved = resolveNamedStructFields(structure, module);
        if (failed(resolved))
            return failure();
        activeStructs.push_back(structure.getName());
        activeName = structure.getName();
        for (const ResolvedStructField &field : resolved->fields)
            fields.push_back(field.type);
    } else {
        return failure();
    }

    uint64_t size = 0;
    uint64_t alignment = 1;
    SmallVector<uint64_t> leafOffsets;
    for (Type field : fields) {
        FailureOr<PhysicalValueAbiLayout> child = planCudaValue(field, module, activeStructs);
        if (failed(child)) {
            if (activeName)
                activeStructs.pop_back();
            return failure();
        }
        FailureOr<uint64_t> offset = checkedAlign(size, child->alignment);
        if (failed(offset) || child->size > std::numeric_limits<uint64_t>::max() - *offset) {
            if (activeName)
                activeStructs.pop_back();
            return failure();
        }
        for (uint64_t leafOffset : child->elementLeafOffsets)
            leafOffsets.push_back(*offset + leafOffset);
        size = *offset + child->size;
        alignment = std::max(alignment, child->alignment);
    }
    if (activeName)
        activeStructs.pop_back();
    FailureOr<uint64_t> finalSize = checkedAlign(size, alignment);
    if (failed(finalSize))
        return failure();
    return PhysicalValueAbiLayout{*finalSize, alignment, {}, std::move(leafOffsets)};
}

FailureOr<PlannedLayout> planProduct(ArrayRef<ResolvedStructField> fields, StringRef kind, ModuleOp module,
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
        if (!(type.isInteger(1) || type.isInteger(32) || type.isF16() || type.isF32() || type.isF64()))
            return failure();
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
        SmallVector<ResolvedStructField> fields;
        for (auto [index, element] : llvm::enumerate(tuple.getTypes()))
            fields.push_back({std::to_string(index), element});
        return planProduct(fields, "tuple", module, activeStructs);
    }
    auto structure = dyn_cast<StructType>(type);
    if (!structure || llvm::is_contained(activeStructs, structure.getName()))
        return failure();
    FailureOr<ResolvedStructFields> fields = resolveNamedStructFields(structure, module);
    if (failed(fields))
        return failure();
    activeStructs.push_back(structure.getName());
    std::string kind = ("struct(" + structure.getName() + ")").str();
    FailureOr<PlannedLayout> result = planProduct(fields->fields, kind, module, activeStructs);
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

LogicalResult verifyValueAbiType(Type type, ModuleOp module) {
    return success(succeeded(getValueAbiLayout(type, module)));
}

FailureOr<PhysicalValueAbiLayout> planPhysicalBytes(Type type, ModuleOp module, PhysicalAbiProfile profile) {
    if (profile == PhysicalAbiProfile::Count)
        return failure();
    if (profile == PhysicalAbiProfile::HostValue) {
        if (type.isIndex())
            return PhysicalValueAbiLayout{8, 8};
        if (isa<TensorViewType, TextureType, SamplerType>(type))
            return PhysicalValueAbiLayout{8, 8};

        FailureOr<ValueAbiLayout> logical = getValueAbiLayout(type, module);
        if (failed(logical))
            return failure();
        PhysicalValueAbiLayout result{logical->size, logical->alignment};
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
            // CPU tensor values lower to LLVM vectors/aggregates. Keep the
            // wrapper packing alignment identical to that lowered ABI while
            // retaining the target-independent logical Value layout above.
            result.alignment = result.size >= 16 ? 16 : result.size >= 8 ? 8 : 4;
            FailureOr<ValueAbiLayout> element = getValueAbiLayout(tensor.getElementType(), module);
            if (failed(element))
                return failure();
            FailureOr<SmallVector<uint64_t>> strides =
                rowMajorStrides(tensor.getShape(), llvm::alignTo(element->size, element->alignment));
            if (failed(strides))
                return failure();
            result.byteStrides = std::move(*strides);
        } else if (auto tensor = dyn_cast<TensorType>(type)) {
            FailureOr<ValueAbiLayout> element = getValueAbiLayout(tensor.getElementType(), module);
            if (failed(element))
                return failure();
            FailureOr<SmallVector<uint64_t>> strides =
                rowMajorStrides(tensor.getShape(), llvm::alignTo(element->size, element->alignment));
            if (failed(strides))
                return failure();
            result.byteStrides = std::move(*strides);
        }
        return result;
    }

    if (profile == PhysicalAbiProfile::CudaKernelParameter) {
        SmallVector<StringRef> activeStructs;
        return planCudaValue(type, module, activeStructs);
    }

    if (profile == PhysicalAbiProfile::OpenGLNativeUniform) {
        if (type.isIntOrFloat()) {
            const uint64_t size = std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
            return PhysicalValueAbiLayout{size, size};
        }
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape() || tensor.getRank() > 2 || !tensor.getElementType().isIntOrFloat())
            return failure();
        const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
        if (tensor.getRank() == 0)
            return PhysicalValueAbiLayout{elementSize, elementSize};
        if (tensor.getRank() == 1) {
            const uint64_t count = static_cast<uint64_t>(tensor.getDimSize(0));
            return PhysicalValueAbiLayout{count * elementSize, elementSize, {elementSize}};
        }
        const uint64_t rows = static_cast<uint64_t>(tensor.getDimSize(0));
        const uint64_t columns = static_cast<uint64_t>(tensor.getDimSize(1));
        return PhysicalValueAbiLayout{rows * columns * elementSize, elementSize, {elementSize, rows * elementSize}};
    }

    if (profile == PhysicalAbiProfile::DirectXConstantBuffer) {
        if (auto tensor = dyn_cast<RankedTensorType>(type);
            tensor && tensor.hasStaticShape() && tensor.getRank() == 2 && tensor.getElementType().isIntOrFloat()) {
            const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
            const uint64_t rows = static_cast<uint64_t>(tensor.getDimSize(0));
            const uint64_t columns = static_cast<uint64_t>(tensor.getDimSize(1));
            const uint64_t rowStride = std::max<uint64_t>(columns * elementSize, 16);
            return PhysicalValueAbiLayout{rows * rowStride, 16, {rowStride, elementSize}};
        }
    }

    const bool std430 = profile == PhysicalAbiProfile::VulkanStd430StorageBuffer ||
                        profile == PhysicalAbiProfile::VulkanPushConstant ||
                        profile == PhysicalAbiProfile::MetalConstantBuffer;
    if (type.isIndex())
        return PhysicalValueAbiLayout{4, 4};
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape() || llvm::any_of(tensor.getShape(), [](int64_t extent) { return extent <= 0; }) ||
            !tensor.getElementType().isIntOrFloat())
            return failure();
        const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
        if (tensor.getRank() == 0)
            return PhysicalValueAbiLayout{elementSize, elementSize};
        if (tensor.getRank() == 1 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4) {
            const uint64_t count = static_cast<uint64_t>(tensor.getDimSize(0));
            const uint64_t alignment = (count == 2 ? 2 : 4) * elementSize;
            return PhysicalValueAbiLayout{count * elementSize, alignment, {elementSize}};
        }
        if (tensor.getRank() == 2 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4 &&
            tensor.getDimSize(1) >= 2 && tensor.getDimSize(1) <= 4) {
            const uint64_t rows = static_cast<uint64_t>(tensor.getDimSize(0));
            const uint64_t columns = static_cast<uint64_t>(tensor.getDimSize(1));
            const uint64_t vectorAlignment = (rows == 2 ? 2 : 4) * elementSize;
            const uint64_t columnAlignment = std430 ? vectorAlignment : std::max<uint64_t>(vectorAlignment, 16);
            const uint64_t stride = llvm::alignTo(rows * elementSize, columnAlignment);
            return PhysicalValueAbiLayout{stride * columns, columnAlignment, {elementSize, stride}};
        }

        uint64_t size = elementSize;
        uint64_t alignment = elementSize;
        SmallVector<uint64_t> reversedStrides;
        for (int64_t extent : llvm::reverse(tensor.getShape())) {
            if (!std430)
                alignment = std::max<uint64_t>(alignment, 16);
            FailureOr<uint64_t> stride = checkedAlign(size, alignment);
            FailureOr<uint64_t> next = succeeded(stride) ? checkedMultiply(*stride, static_cast<uint64_t>(extent))
                                                         : FailureOr<uint64_t>(failure());
            if (failed(next))
                return failure();
            reversedStrides.push_back(*stride);
            size = *next;
        }
        return PhysicalValueAbiLayout{size, alignment,
                                      SmallVector<uint64_t>(reversedStrides.rbegin(), reversedStrides.rend())};
    }

    FailureOr<ValueAbiLayout> logical = getValueAbiLayout(type, module);
    if (failed(logical))
        return failure();
    PhysicalValueAbiLayout result{logical->size, logical->alignment};
    if (auto tensor = dyn_cast<TensorType>(type)) {
        FailureOr<ValueAbiLayout> element = getValueAbiLayout(tensor.getElementType(), module);
        if (failed(element))
            return failure();
        FailureOr<SmallVector<uint64_t>> strides =
            rowMajorStrides(tensor.getShape(), llvm::alignTo(element->size, element->alignment));
        if (failed(strides))
            return failure();
        result.byteStrides = std::move(*strides);
    }
    return result;
}

FailureOr<PhysicalValueAbiPlan> getPhysicalValueAbiPlan(Type type, ModuleOp module, PhysicalAbiProfile profile) {
    if (auto view = dyn_cast<TensorViewType>(type)) {
        FailureOr<ValueAbiLayout> element = getValueAbiLayout(view.getElementType(), module);
        if (failed(element))
            return failure();
        switch (profile) {
        case PhysicalAbiProfile::HostValue:
            return PhysicalValueAbiPlan{
                PhysicalResourceAbiLayout{PhysicalResourceAbiKind::HostPointer, 8, 8, std::move(*element)}};
        case PhysicalAbiProfile::CudaKernelParameter:
            return PhysicalValueAbiPlan{
                PhysicalResourceAbiLayout{PhysicalResourceAbiKind::CudaStorageLeaves, 0, 0, std::move(*element)}};
        case PhysicalAbiProfile::VulkanStd140UniformBuffer:
        case PhysicalAbiProfile::VulkanStd430StorageBuffer:
        case PhysicalAbiProfile::VulkanPushConstant:
        case PhysicalAbiProfile::OpenGLNativeUniform:
        case PhysicalAbiProfile::DirectXConstantBuffer:
        case PhysicalAbiProfile::MetalConstantBuffer:
            return PhysicalValueAbiPlan{
                PhysicalResourceAbiLayout{PhysicalResourceAbiKind::GraphicsStorageLeaves, 0, 0, std::move(*element)}};
        case PhysicalAbiProfile::Count:
            return failure();
        }
    }
    if (isa<TextureType>(type)) {
        if (profile == PhysicalAbiProfile::HostValue)
            return PhysicalValueAbiPlan{PhysicalResourceAbiLayout{PhysicalResourceAbiKind::HostPointer, 8, 8}};
        if (profile != PhysicalAbiProfile::CudaKernelParameter)
            return PhysicalValueAbiPlan{PhysicalResourceAbiLayout{PhysicalResourceAbiKind::GraphicsTexture}};
        return PhysicalValueAbiPlan{UnsupportedPhysicalValueAbi{"texture_argument"}};
    }
    if (isa<SamplerType>(type)) {
        if (profile == PhysicalAbiProfile::HostValue)
            return PhysicalValueAbiPlan{PhysicalResourceAbiLayout{PhysicalResourceAbiKind::HostPointer, 8, 8}};
        if (profile != PhysicalAbiProfile::CudaKernelParameter)
            return PhysicalValueAbiPlan{PhysicalResourceAbiLayout{PhysicalResourceAbiKind::GraphicsSampler}};
        return PhysicalValueAbiPlan{UnsupportedPhysicalValueAbi{"sampler_argument"}};
    }
    FailureOr<PhysicalValueAbiLayout> bytes = planPhysicalBytes(type, module, profile);
    if (failed(bytes))
        return failure();
    return PhysicalValueAbiPlan{std::move(*bytes)};
}

FailureOr<PhysicalValueAbiLayout> getPhysicalValueAbiLayout(Type type, ModuleOp module, PhysicalAbiProfile profile) {
    FailureOr<PhysicalValueAbiPlan> plan = getPhysicalValueAbiPlan(type, module, profile);
    if (failed(plan))
        return failure();
    if (auto *bytes = std::get_if<PhysicalValueAbiLayout>(&*plan))
        return *bytes;
    if (auto *resource = std::get_if<PhysicalResourceAbiLayout>(&*plan); resource && resource->handleSize != 0)
        return PhysicalValueAbiLayout{resource->handleSize, resource->handleAlignment};
    return failure();
}

FailureOr<WorkgroupPhysicalStoragePlan> getWorkgroupPhysicalStoragePlan(TensorViewType view, ModuleOp module) {
    if (view.getAddressSpace() != "workgroup")
        return failure();
    if (view.getShape().empty() || llvm::any_of(view.getShape(), [](int64_t extent) { return extent <= 0; }))
        return failure();

    FailureOr<ValueAbiLayout> layout = getValueAbiLayout(view.getElementType(), module);
    if (failed(layout))
        return failure();

    uint64_t records = 1;
    for (int64_t extent : view.getShape()) {
        const uint64_t dimension = static_cast<uint64_t>(extent);
        if (records > std::numeric_limits<uint64_t>::max() / dimension)
            return failure();
        records *= dimension;
    }

    WorkgroupPhysicalStoragePlan plan;
    plan.elementType = view.getElementType();
    plan.layout = *layout;
    plan.recordCount = records;

    if (view.getElementType().isIntOrFloat()) {
        const uint64_t leafSize = std::max<uint64_t>(view.getElementType().getIntOrFloatBitWidth() / 8, 1);
        if (records > std::numeric_limits<uint64_t>::max() / leafSize)
            return failure();
        plan.leaves.push_back({view.getElementType(), records, records * leafSize, 0});
        plan.totalPhysicalBytes = records * leafSize;
        return plan;
    }

    if (layout->leaves.empty())
        return failure();

    for (auto [index, abiLeaf] : llvm::enumerate(layout->leaves)) {
        const uint64_t leafSize = std::max<uint64_t>(abiLeaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
        if (abiLeaf.scalarCount != 0 && records > std::numeric_limits<uint64_t>::max() / abiLeaf.scalarCount)
            return failure();
        const uint64_t scalarCount = records * abiLeaf.scalarCount;
        if (scalarCount != 0 && leafSize > std::numeric_limits<uint64_t>::max() / scalarCount)
            return failure();
        const uint64_t byteSize = scalarCount * leafSize;
        if (plan.totalPhysicalBytes > std::numeric_limits<uint64_t>::max() - byteSize)
            return failure();
        plan.leaves.push_back({abiLeaf.scalarType, scalarCount, byteSize, index});
        plan.totalPhysicalBytes += byteSize;
    }
    return plan;
}

} // namespace mlir::vernon
