#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir::vernon {

bool containsLogicalAutodiffHandle(Type type) {
    if (isa<AdTapeType, AdRegionHeaderType>(type))
        return true;
    if (auto tuple = dyn_cast<TupleType>(type))
        return llvm::any_of(tuple.getTypes(), containsLogicalAutodiffHandle);
    return false;
}

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

struct NodeProjection {
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<uint64_t> byteStrides;
    SmallVector<uint64_t> childOffsets;

    NodeProjection() = default;
    NodeProjection(uint64_t size, uint64_t alignment, SmallVector<uint64_t> byteStrides = {},
                   SmallVector<uint64_t> childOffsets = {})
        : size(size), alignment(alignment), byteStrides(std::move(byteStrides)), childOffsets(std::move(childOffsets)) {
    }
};

struct PlannedLayout {
    ValueAbiLayout layout;
    std::string canonical;
    std::shared_ptr<CanonicalAbiNode> root;
};

FailureOr<NodeProjection> projectNode(const CanonicalAbiNode &canonical, PhysicalAbiProfile profile);
std::shared_ptr<const ByteTransportNode> buildTransportTree(const CanonicalAbiNode &canonical,
                                                            const NodeProjection &physical, PhysicalAbiProfile profile,
                                                            uint64_t byteOffset);

std::string hashCanonical(StringRef canonical) {
    llvm::SHA256 hash;
    hash.update(canonical);
    return llvm::toHex(hash.final(), true);
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

FailureOr<SmallVector<uint64_t>> rowMajorStrides(ArrayRef<uint64_t> shape, uint64_t elementStride) {
    SmallVector<uint64_t> strides(shape.size());
    uint64_t stride = elementStride;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (shape[dimension] == 0)
            return failure();
        strides[dimension] = stride;
        FailureOr<uint64_t> next = checkedMultiply(stride, shape[dimension]);
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

FailureOr<PlannedLayout> planProduct(Type type, ArrayRef<ResolvedStructField> fields, StringRef kind, ModuleOp module,
                                     SmallVectorImpl<StringRef> &activeStructs) {
    PlannedLayout result;
    result.layout.alignment = 1;
    auto root = std::make_shared<CanonicalAbiNode>();
    root->kind = CanonicalAbiNodeKind::Product;
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
        root->children.push_back(child->root);
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
    root->type = type;
    root->size = result.layout.size;
    root->alignment = result.layout.alignment;
    root->childOffsets = result.layout.fieldOffsets;
    result.root = std::move(root);
    return result;
}

FailureOr<PlannedLayout> planTensor(Type container, Type element, ArrayRef<int64_t> shape, ModuleOp module,
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
    auto root = std::make_shared<CanonicalAbiNode>();
    root->kind = CanonicalAbiNodeKind::Array;
    root->type = container;
    root->size = result.layout.size;
    root->alignment = result.layout.alignment;
    root->shape.assign(shape.begin(), shape.end());
    root->elementStride = result.layout.elementStride;
    root->children.push_back(child->root);
    result.root = std::move(root);
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
        auto root = std::make_shared<CanonicalAbiNode>();
        root->kind = CanonicalAbiNodeKind::Scalar;
        root->type = type;
        root->representation = name.str();
        root->size = size;
        root->alignment = size;
        return PlannedLayout{ValueAbiLayout{size, size, {}, std::nullopt, {{{}, {}, type, name.str(), 0, 1}}, {}, {}},
                             std::move(canonical), std::move(root)};
    }
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape())
            return failure();
        return planTensor(type, tensor.getElementType(), tensor.getShape(), module, activeStructs);
    }
    if (auto vector = dyn_cast<VectorType>(type))
        return planTensor(type, vector.getElementType(), vector.getShape(), module, activeStructs);
    if (auto tensor = dyn_cast<TensorType>(type))
        return planTensor(type, tensor.getElementType(), tensor.getShape(), module, activeStructs);
    if (auto tuple = dyn_cast<TupleType>(type)) {
        SmallVector<ResolvedStructField> fields;
        for (auto [index, element] : llvm::enumerate(tuple.getTypes()))
            fields.push_back({std::to_string(index), element});
        return planProduct(type, fields, "tuple", module, activeStructs);
    }
    auto structure = dyn_cast<StructType>(type);
    if (!structure || llvm::is_contained(activeStructs, structure.getName()))
        return failure();
    FailureOr<ResolvedStructFields> fields = resolveNamedStructFields(structure, module);
    if (failed(fields))
        return failure();
    activeStructs.push_back(structure.getName());
    std::string kind = ("struct(" + structure.getName() + ")").str();
    FailureOr<PlannedLayout> result = planProduct(type, fields->fields, kind, module, activeStructs);
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
    planned->layout.tree = CanonicalAbiTree{planned->root};
    return std::move(planned->layout);
}

FailureOr<CpuCallPlan> getCpuCallPlan(Type type, ModuleOp module, ArrayRef<StringRef> logicalLeafDtypes) {
    FailureOr<ValueAbiLayout> layout = getValueAbiLayout(type, module, logicalLeafDtypes);
    if (failed(layout))
        return failure();

    CpuCallPlan plan;
    plan.layout = std::move(*layout);
    for (auto [leafIndex, leaf] : llvm::enumerate(plan.layout.leaves)) {
        const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
        if (leaf.scalarCount == 0 || leaf.byteOffset > plan.layout.size)
            return failure();
        FailureOr<uint64_t> leafSize = checkedMultiply(scalarSize, leaf.scalarCount);
        if (failed(leafSize) || *leafSize > plan.layout.size - leaf.byteOffset)
            return failure();
        for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex) {
            CpuCallLane lane;
            lane.leafIndex = leafIndex;
            lane.scalarIndex = scalarIndex;
            plan.lanes.push_back(std::move(lane));
        }
    }
    if (plan.lanes.empty() && plan.layout.size != 0)
        return failure();
    if (!plan.layout.tree.root)
        return failure();
    FailureOr<NodeProjection> projection = projectNode(*plan.layout.tree.root, PhysicalAbiProfile::HostValue);
    if (failed(projection))
        return failure();
    plan.root = buildTransportTree(*plan.layout.tree.root, *projection, PhysicalAbiProfile::HostValue, 0);
    if (!plan.root)
        return failure();
    return plan;
}

bool isCpuOpaqueAbiType(Type type) { return type.isIndex() || isa<TextureType, SamplerType>(type); }

LogicalResult verifyValueAbiType(Type type, ModuleOp module) {
    return success(succeeded(getValueAbiLayout(type, module)));
}

namespace {

FailureOr<NodeProjection> canonicalNodeProjection(const CanonicalAbiNode &canonical) {
    if (canonical.kind != CanonicalAbiNodeKind::Array)
        return NodeProjection{canonical.size, canonical.alignment, {}, canonical.childOffsets};
    if (!canonical.elementStride)
        return failure();
    FailureOr<SmallVector<uint64_t>> strides = rowMajorStrides(canonical.shape, *canonical.elementStride);
    if (failed(strides))
        return failure();
    return NodeProjection{canonical.size, canonical.alignment, std::move(*strides)};
}

FailureOr<NodeProjection> projectNode(const CanonicalAbiNode &canonical, PhysicalAbiProfile profile) {
    if (profile == PhysicalAbiProfile::Count)
        return failure();
    Type type = canonical.type;
    if (profile == PhysicalAbiProfile::HostValue)
        return canonicalNodeProjection(canonical);
    if (canonical.kind == CanonicalAbiNodeKind::Product) {
        // Aggregate carriers are lowered as scalar-lane records, so their
        // physical member offsets are the canonical scalar offsets. Container
        // nodes still apply profile-specific array and matrix strides.
        return NodeProjection{canonical.size, canonical.alignment, {}, canonical.childOffsets};
    }

    if (profile == PhysicalAbiProfile::CudaKernelParameter) {
        if (canonical.kind == CanonicalAbiNodeKind::Scalar)
            return NodeProjection{canonical.size, canonical.alignment};
        if (canonical.children.size() != 1)
            return failure();
        FailureOr<NodeProjection> element = projectNode(*canonical.children.front(), profile);
        if (failed(element))
            return failure();
        const uint64_t elementStride = llvm::alignTo(element->size, element->alignment);
        FailureOr<SmallVector<uint64_t>> strides = rowMajorStrides(canonical.shape, elementStride);
        if (failed(strides))
            return failure();
        uint64_t count = 1;
        for (uint64_t extent : canonical.shape) {
            FailureOr<uint64_t> next = checkedMultiply(count, extent);
            if (failed(next))
                return failure();
            count = *next;
        }
        FailureOr<uint64_t> size = checkedMultiply(elementStride, count);
        if (failed(size))
            return failure();
        uint64_t alignment = element->alignment;
        if ((isa<RankedTensorType, VectorType>(type)) && count > 1 && count <= 16) {
            alignment = element->size;
            while (alignment < *size && alignment < 16)
                alignment *= 2;
            alignment = std::min<uint64_t>(alignment, 16);
        }
        return NodeProjection{*size, alignment, std::move(*strides)};
    }

    if (profile == PhysicalAbiProfile::OpenGLNativeUniform) {
        if (type.isIntOrFloat()) {
            const uint64_t size = std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
            return NodeProjection{size, size};
        }
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape() || tensor.getRank() > 2 || !tensor.getElementType().isIntOrFloat())
            return failure();
        const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
        if (tensor.getRank() == 0)
            return NodeProjection{elementSize, elementSize};
        if (tensor.getRank() == 1) {
            const uint64_t count = static_cast<uint64_t>(tensor.getDimSize(0));
            return NodeProjection{count * elementSize, elementSize, {elementSize}};
        }
        const uint64_t rows = static_cast<uint64_t>(tensor.getDimSize(0));
        const uint64_t columns = static_cast<uint64_t>(tensor.getDimSize(1));
        return NodeProjection{rows * columns * elementSize, elementSize, {elementSize, rows * elementSize}};
    }

    if (profile == PhysicalAbiProfile::DirectXConstantBuffer) {
        if (auto tensor = dyn_cast<RankedTensorType>(type);
            tensor && tensor.hasStaticShape() && tensor.getRank() == 2 && tensor.getElementType().isIntOrFloat()) {
            const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
            const uint64_t rows = static_cast<uint64_t>(tensor.getDimSize(0));
            const uint64_t columns = static_cast<uint64_t>(tensor.getDimSize(1));
            const uint64_t rowStride = std::max<uint64_t>(columns * elementSize, 16);
            return NodeProjection{rows * rowStride, 16, {rowStride, elementSize}};
        }
    }

    const bool std430 = profile == PhysicalAbiProfile::VulkanStd430StorageBuffer ||
                        profile == PhysicalAbiProfile::VulkanPushConstant ||
                        profile == PhysicalAbiProfile::MetalConstantBuffer;
    if (type.isIndex())
        return NodeProjection{4, 4};
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape() || llvm::any_of(tensor.getShape(), [](int64_t extent) { return extent <= 0; }) ||
            !tensor.getElementType().isIntOrFloat())
            return failure();
        const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
        if (tensor.getRank() == 0)
            return NodeProjection{elementSize, elementSize};
        if (tensor.getRank() == 1 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4) {
            const uint64_t count = static_cast<uint64_t>(tensor.getDimSize(0));
            const uint64_t alignment = (count == 2 ? 2 : 4) * elementSize;
            return NodeProjection{count * elementSize, alignment, {elementSize}};
        }
        if (tensor.getRank() == 2 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4 &&
            tensor.getDimSize(1) >= 2 && tensor.getDimSize(1) <= 4) {
            const uint64_t rows = static_cast<uint64_t>(tensor.getDimSize(0));
            const uint64_t columns = static_cast<uint64_t>(tensor.getDimSize(1));
            const uint64_t vectorAlignment = (rows == 2 ? 2 : 4) * elementSize;
            const uint64_t columnAlignment = std430 ? vectorAlignment : std::max<uint64_t>(vectorAlignment, 16);
            const uint64_t stride = llvm::alignTo(rows * elementSize, columnAlignment);
            return NodeProjection{stride * columns, columnAlignment, {elementSize, stride}};
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
        return NodeProjection{size, alignment, SmallVector<uint64_t>(reversedStrides.rbegin(), reversedStrides.rend())};
    }

    NodeProjection result{canonical.size, canonical.alignment,
                          canonical.elementStride ? SmallVector<uint64_t>{*canonical.elementStride}
                                                  : SmallVector<uint64_t>{}};
    if (canonical.kind == CanonicalAbiNodeKind::Array) {
        if (canonical.children.size() != 1)
            return failure();
        FailureOr<NodeProjection> element = projectNode(*canonical.children.front(), profile);
        if (failed(element))
            return failure();
        FailureOr<SmallVector<uint64_t>> strides =
            rowMajorStrides(canonical.shape, llvm::alignTo(element->size, element->alignment));
        if (failed(strides))
            return failure();
        result.byteStrides = std::move(*strides);
        uint64_t count = 1;
        for (uint64_t extent : canonical.shape) {
            FailureOr<uint64_t> next = checkedMultiply(count, extent);
            if (failed(next))
                return failure();
            count = *next;
        }
        FailureOr<uint64_t> size = checkedMultiply(llvm::alignTo(element->size, element->alignment), count);
        if (failed(size))
            return failure();
        result.size = *size;
        result.alignment = element->alignment;
    }
    return result;
}

std::shared_ptr<const ByteTransportNode> buildTransportTree(const CanonicalAbiNode &canonical,
                                                            const NodeProjection &physical, PhysicalAbiProfile profile,
                                                            uint64_t byteOffset) {
    auto node = std::make_shared<ByteTransportNode>();
    node->kind = canonical.kind == CanonicalAbiNodeKind::Scalar    ? ByteTransportNodeKind::Scalar
                 : canonical.kind == CanonicalAbiNodeKind::Product ? ByteTransportNodeKind::Product
                                                                   : ByteTransportNodeKind::Array;
    node->representation = canonical.kind == CanonicalAbiNodeKind::Scalar && canonical.type.isIndex() &&
                                   profile != PhysicalAbiProfile::HostValue &&
                                   profile != PhysicalAbiProfile::CudaKernelParameter
                               ? "i32"
                               : canonical.representation;
    node->byteOffset = byteOffset;
    node->size = physical.size;
    node->alignment = physical.alignment;
    node->shape = canonical.shape;
    node->byteStrides = physical.byteStrides;
    if (canonical.kind == CanonicalAbiNodeKind::Product) {
        // The product node is a scalar-lane record; its descendants describe
        // that record's canonical member topology rather than nested native
        // vector or matrix carriers.
        for (auto [index, child] : llvm::enumerate(canonical.children)) {
            FailureOr<NodeProjection> childLayout = canonicalNodeProjection(*child);
            if (failed(childLayout))
                return {};
            node->children.push_back(buildTransportTree(*child, *childLayout, profile, physical.childOffsets[index]));
        }
    } else if (canonical.kind == CanonicalAbiNodeKind::Array && !canonical.children.empty()) {
        const CanonicalAbiNode &child = *canonical.children.front();
        FailureOr<NodeProjection> childLayout = projectNode(child, profile);
        if (failed(childLayout))
            return {};
        node->children.push_back(buildTransportTree(child, *childLayout, profile, 0));
    }
    return node;
}

} // namespace

FailureOr<BackendInterfaceAbiPlan> getBackendInterfaceAbiPlan(Type type, ModuleOp module, PhysicalAbiProfile profile,
                                                              ArrayRef<StringRef> logicalLeafDtypes) {
    if (auto view = dyn_cast<TensorViewType>(type)) {
        if (failed(getValueAbiLayout(view.getElementType(), module, logicalLeafDtypes)))
            return failure();
        switch (profile) {
        case PhysicalAbiProfile::HostValue:
            return BackendInterfaceAbiPlan{
                ResourceBindingPlan{PhysicalResourceAbiKind::TensorViewDescriptor,
                                    8 * (2 + 2 * static_cast<uint64_t>(view.getShape().size())), 8}};
        case PhysicalAbiProfile::CudaKernelParameter:
            return BackendInterfaceAbiPlan{ResourceBindingPlan{PhysicalResourceAbiKind::CudaStorageLeaves, 0, 0}};
        case PhysicalAbiProfile::VulkanStd140UniformBuffer:
        case PhysicalAbiProfile::VulkanStd430StorageBuffer:
        case PhysicalAbiProfile::VulkanPushConstant:
        case PhysicalAbiProfile::OpenGLNativeUniform:
        case PhysicalAbiProfile::DirectXConstantBuffer:
        case PhysicalAbiProfile::MetalConstantBuffer:
            return BackendInterfaceAbiPlan{ResourceBindingPlan{PhysicalResourceAbiKind::GraphicsStorageLeaves, 0, 0}};
        case PhysicalAbiProfile::Count:
            return failure();
        }
    }
    if (isa<TextureType>(type)) {
        if (profile == PhysicalAbiProfile::HostValue)
            return BackendInterfaceAbiPlan{ResourceBindingPlan{PhysicalResourceAbiKind::HostPointer, 8, 8}};
        if (profile != PhysicalAbiProfile::CudaKernelParameter)
            return BackendInterfaceAbiPlan{ResourceBindingPlan{PhysicalResourceAbiKind::GraphicsTexture, 0, 0}};
        return BackendInterfaceAbiPlan{UnsupportedBackendInterfaceAbi{"texture_argument"}};
    }
    if (isa<SamplerType>(type)) {
        if (profile == PhysicalAbiProfile::HostValue)
            return BackendInterfaceAbiPlan{ResourceBindingPlan{PhysicalResourceAbiKind::HostPointer, 8, 8}};
        if (profile != PhysicalAbiProfile::CudaKernelParameter)
            return BackendInterfaceAbiPlan{ResourceBindingPlan{PhysicalResourceAbiKind::GraphicsSampler, 0, 0}};
        return BackendInterfaceAbiPlan{UnsupportedBackendInterfaceAbi{"sampler_argument"}};
    }
    if (type.isIndex()) {
        CanonicalAbiNode canonical;
        canonical.kind = CanonicalAbiNodeKind::Scalar;
        canonical.type = type;
        canonical.representation = "index";
        canonical.size = 8;
        canonical.alignment = 8;
        const std::string layoutHash = hashCanonical("scalar(index,8,8)");
        FailureOr<NodeProjection> projection = projectNode(canonical, profile);
        if (failed(projection))
            return failure();
        std::shared_ptr<const ByteTransportNode> root = buildTransportTree(canonical, *projection, profile, 0);
        if (profile == PhysicalAbiProfile::CudaKernelParameter)
            return BackendInterfaceAbiPlan{KernelParameterPlan{layoutHash, std::move(root)}};
        return BackendInterfaceAbiPlan{ByteTransportPlan{profile, layoutHash, std::move(root)}};
    }
    FailureOr<ValueAbiLayout> canonical = getValueAbiLayout(type, module, logicalLeafDtypes);
    if (failed(canonical) || !canonical->tree.root)
        return failure();
    FailureOr<NodeProjection> projection = projectNode(*canonical->tree.root, profile);
    if (failed(projection))
        return failure();
    std::shared_ptr<const ByteTransportNode> root = buildTransportTree(*canonical->tree.root, *projection, profile, 0);
    if (!root)
        return failure();
    if (profile == PhysicalAbiProfile::CudaKernelParameter)
        return BackendInterfaceAbiPlan{KernelParameterPlan{canonical->layoutHash, std::move(root)}};
    if (profile == PhysicalAbiProfile::OpenGLNativeUniform)
        return BackendInterfaceAbiPlan{NativeUniformPlan{canonical->layoutHash, std::move(root)}};
    return BackendInterfaceAbiPlan{ByteTransportPlan{profile, canonical->layoutHash, std::move(root)}};
}

FailureOr<ByteTransportPlan> getByteTransportPlan(Type type, ModuleOp module, PhysicalAbiProfile profile) {
    FailureOr<BackendInterfaceAbiPlan> plan = getBackendInterfaceAbiPlan(type, module, profile);
    if (failed(plan))
        return failure();
    if (auto *bytes = std::get_if<ByteTransportPlan>(&*plan))
        return *bytes;
    return failure();
}

FailureOr<WorkgroupPhysicalStoragePlan> getWorkgroupPhysicalStoragePlan(TensorViewType view, ModuleOp module) {
    constexpr uint64_t allocationAlignment = 16;
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
    auto addAllocationFootprint = [&](uint64_t byteSize) {
        if (byteSize > std::numeric_limits<uint64_t>::max() - (allocationAlignment - 1))
            return failure();
        const uint64_t footprint = ((byteSize + allocationAlignment - 1) / allocationAlignment) * allocationAlignment;
        if (plan.totalPhysicalBytes > std::numeric_limits<uint64_t>::max() - footprint)
            return failure();
        plan.totalPhysicalBytes += footprint;
        return success();
    };

    if (view.getElementType().isIntOrFloat()) {
        const uint64_t leafSize = std::max<uint64_t>(view.getElementType().getIntOrFloatBitWidth() / 8, 1);
        if (records > std::numeric_limits<uint64_t>::max() / leafSize)
            return failure();
        const uint64_t byteSize = records * leafSize;
        plan.leaves.push_back({view.getElementType(), records, byteSize, 0});
        if (failed(addAllocationFootprint(byteSize)))
            return failure();
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
        plan.leaves.push_back({abiLeaf.scalarType, scalarCount, byteSize, index});
        if (failed(addAllocationFootprint(byteSize)))
            return failure();
    }
    return plan;
}

} // namespace mlir::vernon
