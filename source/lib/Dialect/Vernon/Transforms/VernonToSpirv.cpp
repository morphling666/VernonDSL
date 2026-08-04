#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAttributeAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSpirvMarkers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h"

#include "VernonSpirvMath.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <iterator>
#include <limits>

namespace mlir::vernon {
namespace {

FailureOr<Type> convertTensorType(RankedTensorType tensor, ModuleOp module) {
    if (!tensor.hasStaticShape() || llvm::any_of(tensor.getShape(), [](int64_t extent) { return extent <= 0; }) ||
        !tensor.getElementType().isIntOrFloat())
        return failure();
    FailureOr<ByteTransportPlan> layout =
        getByteTransportPlan(tensor, module, PhysicalAbiProfile::VulkanStd140UniformBuffer);
    if (failed(layout))
        return failure();
    Type elementType = tensor.getElementType();
    if (auto integer = dyn_cast<IntegerType>(elementType))
        elementType = IntegerType::get(tensor.getContext(), integer.getWidth());
    if (tensor.getRank() == 0)
        return elementType;
    if (tensor.getRank() == 1 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4)
        return VectorType::get(tensor.getShape(), elementType);
    if (tensor.getRank() == 2 && tensor.getDimSize(0) >= 2 && tensor.getDimSize(0) <= 4 && tensor.getDimSize(1) >= 2 &&
        tensor.getDimSize(1) <= 4) {
        auto columnType = VectorType::get({tensor.getShape()[0]}, elementType);
        return spirv::MatrixType::get(columnType, tensor.getShape()[1]);
    }
    Type result = elementType;
    for (size_t dimension = tensor.getRank(); dimension-- > 0;) {
        uint64_t stride = layout->root->byteStrides[dimension];
        if (stride > std::numeric_limits<unsigned>::max())
            return failure();
        result = spirv::ArrayType::get(result, static_cast<unsigned>(tensor.getDimSize(dimension)),
                                       static_cast<unsigned>(stride));
    }
    return result;
}

FailureOr<Type> convertStd140TensorInterfaceType(RankedTensorType tensor, ModuleOp module) {
    if (!tensor.hasStaticShape() || tensor.getRank() <= 2 ||
        llvm::any_of(tensor.getShape(), [](int64_t extent) { return extent <= 0; }) ||
        !tensor.getElementType().isIntOrFloat())
        return failure();
    FailureOr<ByteTransportPlan> layout =
        getByteTransportPlan(tensor, module, PhysicalAbiProfile::VulkanStd140UniformBuffer);
    if (failed(layout) || layout->root->byteStrides.size() != static_cast<size_t>(tensor.getRank()))
        return failure();
    Type elementType = tensor.getElementType();
    if (auto integer = dyn_cast<IntegerType>(elementType))
        elementType = IntegerType::get(tensor.getContext(), integer.getWidth());
    const unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    if (!bitWidth || bitWidth % 8 != 0)
        return failure();
    const uint64_t elementSize = bitWidth / 8;
    const uint64_t scalarStride = layout->root->byteStrides.back();
    if (scalarStride < elementSize || scalarStride % elementSize != 0)
        return failure();
    const uint64_t carrierComponents = scalarStride / elementSize;
    if (carrierComponents < 2 || carrierComponents > std::numeric_limits<uint32_t>::max())
        return failure();

    Type result;
    if (carrierComponents <= 4) {
        result = VectorType::get({static_cast<int64_t>(carrierComponents)}, elementType);
    } else {
        SmallVector<Type> fields;
        SmallVector<uint32_t> offsets;
        uint64_t remaining = carrierComponents;
        uint64_t offset = 0;
        while (remaining) {
            const uint64_t components = std::min<uint64_t>(remaining, 4);
            fields.push_back(components == 1 ? elementType
                                             : Type(VectorType::get({static_cast<int64_t>(components)}, elementType)));
            offsets.push_back(static_cast<uint32_t>(offset));
            offset += components * elementSize;
            remaining -= components;
        }
        result = spirv::StructType::get(fields, offsets);
    }
    for (size_t dimension = tensor.getRank(); dimension-- > 0;) {
        const uint64_t stride = layout->root->byteStrides[dimension];
        if (stride > std::numeric_limits<unsigned>::max())
            return failure();
        result = spirv::ArrayType::get(result, static_cast<unsigned>(tensor.getDimSize(dimension)),
                                       static_cast<unsigned>(stride));
    }
    return result;
}

FailureOr<Type> convertValueType(Type type, ModuleOp module = {}) {
    if (auto texture = dyn_cast<TextureType>(type)) {
        std::optional<spirv::Dim> dimension = llvm::StringSwitch<std::optional<spirv::Dim>>(texture.getDimension())
                                                  .Case("2d", spirv::Dim::Dim2D)
                                                  .Case("3d", spirv::Dim::Dim3D)
                                                  .Case("cube", spirv::Dim::Cube)
                                                  .Default(std::nullopt);
        if (!dimension)
            return failure();
        auto image = spirv::ImageType::get(texture.getElementType(), *dimension, spirv::ImageDepthInfo::NoDepth,
                                           spirv::ImageArrayedInfo::NonArrayed, spirv::ImageSamplingInfo::SingleSampled,
                                           spirv::ImageSamplerUseInfo::NeedSampler, spirv::ImageFormat::Unknown);
        return spirv::SampledImageType::get(image);
    }
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        return convertTensorType(tensor, module);
    }
    if (auto tuple = dyn_cast<TupleType>(type)) {
        SmallVector<Type> elements;
        for (Type element : tuple.getTypes()) {
            FailureOr<Type> converted = convertValueType(element, module);
            if (failed(converted))
                return failure();
            elements.push_back(*converted);
        }
        FailureOr<ValueAbiLayout> layout = getValueAbiLayout(type, module);
        if (failed(layout))
            return failure();
        SmallVector<uint32_t> offsets;
        for (uint64_t offset : layout->fieldOffsets) {
            if (offset > std::numeric_limits<uint32_t>::max())
                return failure();
            offsets.push_back(static_cast<uint32_t>(offset));
        }
        Type convertedStructure = spirv::StructType::get(elements, offsets);
        return convertedStructure;
    }
    if (auto tensor = dyn_cast<TensorType>(type)) {
        if (!module)
            return failure();
        FailureOr<Type> element = convertValueType(tensor.getElementType(), module);
        FailureOr<ByteTransportPlan> layout =
            getByteTransportPlan(type, module, PhysicalAbiProfile::VulkanStd140UniformBuffer);
        FailureOr<int64_t> count = getStaticShapeElementCount(tensor.getShape());
        if (failed(element) || failed(layout) || layout->root->byteStrides.empty() || failed(count) ||
            *count > static_cast<int64_t>(std::numeric_limits<unsigned>::max()))
            return failure();
        uint64_t stride = layout->root->byteStrides.back();
        if (stride > std::numeric_limits<unsigned>::max())
            return failure();
        return spirv::ArrayType::get(*element, static_cast<unsigned>(*count), static_cast<unsigned>(stride));
    }
    if (auto structure = dyn_cast<StructType>(type)) {
        if (!module)
            return failure();
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
        FailureOr<ValueAbiLayout> layout = getValueAbiLayout(type, module);
        if (failed(fields) || failed(layout))
            return failure();
        SmallVector<Type> elements;
        SmallVector<uint32_t> offsets;
        for (auto [field, offset] : llvm::zip_equal(fields->second, layout->fieldOffsets)) {
            FailureOr<Type> converted = convertValueType(field, module);
            if (failed(converted) || offset > std::numeric_limits<uint32_t>::max())
                return failure();
            elements.push_back(*converted);
            offsets.push_back(static_cast<uint32_t>(offset));
        }
        return spirv::StructType::get(elements, offsets);
    }
    if (type.isIndex())
        return IntegerType::get(type.getContext(), 32);
    if (auto integer = dyn_cast<IntegerType>(type))
        return IntegerType::get(type.getContext(), integer.getWidth());
    if (type.isIntOrFloat())
        return type;
    if (isa<VectorType>(type))
        return type;
    if (isa<spirv::ArrayType, spirv::MatrixType, spirv::StructType>(type))
        return type;
    return failure();
}

Type applyInterfaceIntegerSignedness(Type type, DictionaryAttr attributes) {
    auto dtype = attributes.getAs<StringAttr>("vernon.dtype");
    if (!dtype || (dtype.getValue() != "i32" && dtype.getValue() != "u32"))
        return type;
    const auto signedness = dtype.getValue() == "i32" ? IntegerType::SignednessSemantics::Signed
                                                      : IntegerType::SignednessSemantics::Unsigned;
    if (auto integer = dyn_cast<IntegerType>(type))
        return IntegerType::get(type.getContext(), integer.getWidth(), signedness);
    if (auto vector = dyn_cast<VectorType>(type))
        if (auto integer = dyn_cast<IntegerType>(vector.getElementType()))
            return VectorType::get(vector.getShape(),
                                   IntegerType::get(type.getContext(), integer.getWidth(), signedness));
    return type;
}

LogicalResult collectTransportScalarOffsets(const ByteTransportNode &node, uint64_t base,
                                            SmallVectorImpl<uint64_t> &offsets) {
    if (node.byteOffset > std::numeric_limits<uint64_t>::max() - base)
        return failure();
    const uint64_t offset = base + node.byteOffset;
    if (node.kind == ByteTransportNodeKind::Scalar) {
        offsets.push_back(offset);
        return success();
    }
    if (node.kind == ByteTransportNodeKind::Product) {
        for (const std::shared_ptr<const ByteTransportNode> &child : node.children)
            if (failed(collectTransportScalarOffsets(*child, offset, offsets)))
                return failure();
        return success();
    }
    if (node.children.size() != 1 || node.shape.size() != node.byteStrides.size())
        return failure();
    SmallVector<uint64_t> indices(node.shape.size());
    for (;;) {
        uint64_t elementOffset = offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >
                (std::numeric_limits<uint64_t>::max() - elementOffset) / node.byteStrides[dimension])
                return failure();
            elementOffset += indices[dimension] * node.byteStrides[dimension];
        }
        if (failed(collectTransportScalarOffsets(*node.children.front(), elementOffset, offsets)))
            return failure();
        size_t dimension = indices.size();
        while (dimension != 0) {
            --dimension;
            if (++indices[dimension] < node.shape[dimension])
                break;
            indices[dimension] = 0;
        }
        if (dimension == 0 && indices[0] == 0)
            break;
    }
    return success();
}

FailureOr<Type> convertAggregateTensorStorageType(TensorType tensor, ModuleOp module) {
    FailureOr<ValueAbiLayout> elementLayout = getValueAbiLayout(tensor.getElementType(), module);
    FailureOr<ByteTransportPlan> physicalLayout =
        getByteTransportPlan(tensor, module, PhysicalAbiProfile::VulkanStd430StorageBuffer);
    FailureOr<int64_t> count = getStaticShapeElementCount(tensor.getShape());
    if (failed(elementLayout) || failed(physicalLayout) || physicalLayout->root->byteStrides.empty() || failed(count) ||
        *count <= 0 || *count > static_cast<int64_t>(std::numeric_limits<unsigned>::max()))
        return failure();
    const ByteTransportNode &element = *physicalLayout->root->children.front();
    SmallVector<uint64_t> physicalOffsets;
    if (failed(collectTransportScalarOffsets(element, 0, physicalOffsets)))
        return failure();
    SmallVector<Type> members;
    for (const ValueAbiLeaf &leaf : elementLayout->leaves) {
        FailureOr<Type> scalar = convertValueType(leaf.scalarType, module);
        if (failed(scalar))
            return failure();
        for (uint64_t index = 0; index < leaf.scalarCount; ++index) {
            members.push_back(*scalar);
        }
    }
    if (members.size() != physicalOffsets.size() ||
        llvm::any_of(physicalOffsets, [](uint64_t offset) { return offset > std::numeric_limits<uint32_t>::max(); }))
        return failure();
    SmallVector<uint32_t> offsets;
    llvm::transform(physicalOffsets, std::back_inserter(offsets),
                    [](uint64_t offset) { return static_cast<uint32_t>(offset); });
    const uint64_t stride = physicalLayout->root->byteStrides.back();
    if (members.empty() || stride > std::numeric_limits<unsigned>::max())
        return failure();
    Type record = spirv::StructType::get(members, offsets);
    return spirv::ArrayType::get(record, static_cast<unsigned>(*count), static_cast<unsigned>(stride));
}

std::optional<spirv::ExecutionModel> parseExecutionModel(StringRef stage) {
    return llvm::StringSwitch<std::optional<spirv::ExecutionModel>>(stage)
        .Case("vertex", spirv::ExecutionModel::Vertex)
        .Case("fragment", spirv::ExecutionModel::Fragment)
        .Case("compute", spirv::ExecutionModel::GLCompute)
        .Default(std::nullopt);
}

std::optional<spirv::BuiltIn> parseBuiltIn(StringRef name) {
    return llvm::StringSwitch<std::optional<spirv::BuiltIn>>(name)
        .Case("position", spirv::BuiltIn::Position)
        .Case("frag_coord", spirv::BuiltIn::FragCoord)
        .Case("front_facing", spirv::BuiltIn::FrontFacing)
        .Case("vertex_index", spirv::BuiltIn::VertexIndex)
        .Case("instance_index", spirv::BuiltIn::InstanceIndex)
        .Case("global_invocation_id", spirv::BuiltIn::GlobalInvocationId)
        .Case("local_invocation_id", spirv::BuiltIn::LocalInvocationId)
        .Case("workgroup_id", spirv::BuiltIn::WorkgroupId)
        .Default(std::nullopt);
}

std::string sanitizeInterfaceName(StringRef sourceName, StringRef fallback, bool varying) {
    StringRef input = sourceName.empty() ? fallback : sourceName;
    std::string result;
    bool uppercaseNext = varying && input.contains('_');
    for (char character : input) {
        if (llvm::isAlnum(character)) {
            result.push_back(uppercaseNext ? llvm::toUpper(character) : character);
            uppercaseNext = false;
        } else if (character == '_' && varying) {
            uppercaseNext = true;
        } else {
            result.push_back('_');
            uppercaseNext = false;
        }
    }
    if (result.empty())
        result = "interface_value";
    if (!llvm::isAlpha(result.front()) && result.front() != '_')
        result.insert(result.begin(), '_');
    return result;
}

std::string uniqueInterfaceName(StringRef preferred, StringRef stage, llvm::StringSet<> &usedNames) {
    if (usedNames.insert(preferred).second)
        return preferred.str();
    std::string candidate = (preferred + "_" + stage).str();
    unsigned suffix = 2;
    while (!usedNames.insert(candidate).second)
        candidate = (preferred + "_" + stage + "_" + Twine(suffix++)).str();
    return candidate;
}

void appendMemberDecorations(OpBuilder &builder, uint32_t member, Type type, const ByteTransportNode &layout,
                             SmallVectorImpl<spirv::StructType::MemberDecorationInfo> &decorations) {
    if (isa<spirv::MatrixType>(type) && layout.byteStrides.size() == 2) {
        decorations.emplace_back(member, spirv::Decoration::MatrixStride,
                                 builder.getI32IntegerAttr(layout.byteStrides[1]));
        decorations.emplace_back(member, spirv::Decoration::ColMajor, builder.getUnitAttr());
    }
}

spirv::GlobalVariableOp createInterfaceVariable(OpBuilder &builder, Location location, StringRef name, Type valueType,
                                                spirv::StorageClass storageClass, DictionaryAttr attributes,
                                                const ByteTransportNode *layout = nullptr) {
    auto pointerType = spirv::PointerType::get(valueType, storageClass);
    if (storageClass == spirv::StorageClass::Uniform || storageClass == spirv::StorageClass::PushConstant ||
        storageClass == spirv::StorageClass::StorageBuffer) {
        SmallVector<spirv::StructType::MemberDecorationInfo> memberDecorations;
        if (layout)
            appendMemberDecorations(builder, 0, valueType, *layout, memberDecorations);
        SmallVector<spirv::StructType::StructDecorationInfo> structDecorations;
        structDecorations.emplace_back(spirv::Decoration::Block, builder.getUnitAttr());
        auto blockType = spirv::StructType::get({valueType}, {0}, memberDecorations, structDecorations);
        pointerType = spirv::PointerType::get(blockType, storageClass);
    }
    InterfaceAttrs interface = parseInterfaceAttrs(attributes);
    auto toI32 = [&](Attribute attribute) -> IntegerAttr {
        if (auto integer = dyn_cast_if_present<IntegerAttr>(attribute))
            return builder.getI32IntegerAttr(integer.getInt());
        return {};
    };
    IntegerAttr locationAttr = toI32(interface.location);
    IntegerAttr bindingAttr = toI32(interface.binding);
    IntegerAttr setAttr = toI32(interface.descriptorSet);
    StringAttr builtInAttr;
    if (auto spelling = dyn_cast_if_present<StringAttr>(interface.builtin)) {
        if (auto builtIn = parseBuiltIn(spelling.getValue()))
            builtInAttr = builder.getStringAttr(spirv::stringifyBuiltIn(*builtIn));
    }
    return spirv::GlobalVariableOp::create(builder, location, pointerType, name, FlatSymbolRefAttr(), locationAttr,
                                           bindingAttr, setAttr, builtInAttr, spirv::LinkageAttributesAttr());
}

void flattenComposite(Location location, Value value, OpBuilder &builder, SmallVectorImpl<Value> &leaves) {
    Type type = value.getType();
    if (auto array = dyn_cast<spirv::ArrayType>(type)) {
        for (unsigned index = 0; index < array.getNumElements(); ++index) {
            Value element = spirv::CompositeExtractOp::create(builder, location, value,
                                                              ArrayRef<int32_t>{static_cast<int32_t>(index)});
            flattenComposite(location, element, builder, leaves);
        }
        return;
    }
    if (auto matrix = dyn_cast<spirv::MatrixType>(type)) {
        auto columnType = cast<VectorType>(matrix.getColumnType());
        const unsigned rows = static_cast<unsigned>(columnType.getNumElements());
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned column = 0; column < matrix.getNumColumns(); ++column)
                leaves.push_back(spirv::CompositeExtractOp::create(
                    builder, location, value,
                    ArrayRef<int32_t>{static_cast<int32_t>(column), static_cast<int32_t>(row)}));
        return;
    }
    if (auto vector = dyn_cast<VectorType>(type)) {
        for (int64_t index = 0; index < vector.getNumElements(); ++index)
            leaves.push_back(spirv::CompositeExtractOp::create(builder, location, value,
                                                               ArrayRef<int32_t>{static_cast<int32_t>(index)}));
        return;
    }
    if (auto structure = dyn_cast<spirv::StructType>(type)) {
        for (unsigned index = 0; index < structure.getNumElements(); ++index) {
            Value element = spirv::CompositeExtractOp::create(builder, location, value,
                                                              ArrayRef<int32_t>{static_cast<int32_t>(index)});
            flattenComposite(location, element, builder, leaves);
        }
        return;
    }
    leaves.push_back(value);
}

FailureOr<Value> constructComposite(Location location, Type type, ArrayRef<Value> leaves, unsigned &cursor,
                                    OpBuilder &builder) {
    auto constructElements = [&](Type elementType, unsigned count) -> FailureOr<Value> {
        SmallVector<Value> elements;
        elements.reserve(count);
        for (unsigned index = 0; index < count; ++index) {
            FailureOr<Value> element = constructComposite(location, elementType, leaves, cursor, builder);
            if (failed(element))
                return failure();
            elements.push_back(*element);
        }
        return spirv::CompositeConstructOp::create(builder, location, type, elements).getResult();
    };
    if (auto array = dyn_cast<spirv::ArrayType>(type))
        return constructElements(array.getElementType(), array.getNumElements());
    if (auto matrix = dyn_cast<spirv::MatrixType>(type)) {
        auto columnType = cast<VectorType>(matrix.getColumnType());
        const unsigned rows = static_cast<unsigned>(columnType.getNumElements());
        const unsigned columns = matrix.getNumColumns();
        if (cursor > leaves.size() || leaves.size() - cursor < rows * columns)
            return failure();
        SmallVector<Value> valueColumns;
        for (unsigned column = 0; column < columns; ++column) {
            SmallVector<Value> columnElements;
            for (unsigned row = 0; row < rows; ++row)
                columnElements.push_back(leaves[cursor + row * columns + column]);
            valueColumns.push_back(spirv::CompositeConstructOp::create(builder, location, columnType, columnElements));
        }
        cursor += rows * columns;
        return spirv::CompositeConstructOp::create(builder, location, matrix, valueColumns).getResult();
    }
    if (auto vector = dyn_cast<VectorType>(type))
        return constructElements(vector.getElementType(), static_cast<unsigned>(vector.getNumElements()));
    if (auto structure = dyn_cast<spirv::StructType>(type)) {
        SmallVector<Value> fields;
        fields.reserve(structure.getNumElements());
        for (Type fieldType : structure.getElementTypes()) {
            FailureOr<Value> field = constructComposite(location, fieldType, leaves, cursor, builder);
            if (failed(field))
                return failure();
            fields.push_back(*field);
        }
        return spirv::CompositeConstructOp::create(builder, location, structure, fields).getResult();
    }
    if (cursor >= leaves.size() || leaves[cursor].getType() != type)
        return failure();
    return leaves[cursor++];
}

FailureOr<Value> constructComposite(Location location, Type type, ArrayRef<Value> leaves, OpBuilder &builder) {
    unsigned cursor = 0;
    FailureOr<Value> result = constructComposite(location, type, leaves, cursor, builder);
    return succeeded(result) && cursor == leaves.size() ? result : FailureOr<Value>(failure());
}

FailureOr<Value> unpackStd140TensorInterface(Location location, Value value, RankedTensorType sourceType,
                                             Type logicalType, OpBuilder &builder) {
    Type carrierType = value.getType();
    for (int64_t dimension = 0; dimension < sourceType.getRank(); ++dimension) {
        auto array = dyn_cast<spirv::ArrayType>(carrierType);
        if (!array)
            return failure();
        carrierType = array.getElementType();
    }
    SmallVector<int32_t> carrierIndices;
    while (carrierType != sourceType.getElementType()) {
        if (auto structure = dyn_cast<spirv::StructType>(carrierType)) {
            if (structure.getNumElements() == 0)
                return failure();
            carrierIndices.push_back(0);
            carrierType = structure.getElementType(0);
            continue;
        }
        if (auto vector = dyn_cast<VectorType>(carrierType)) {
            carrierIndices.push_back(0);
            carrierType = vector.getElementType();
            continue;
        }
        return failure();
    }

    SmallVector<Value> leaves;
    leaves.reserve(sourceType.getNumElements());
    for (int64_t linearIndex = 0; linearIndex < sourceType.getNumElements(); ++linearIndex) {
        SmallVector<int32_t> indices;
        int64_t remaining = linearIndex;
        for (int64_t dimension = sourceType.getRank() - 1; dimension >= 0; --dimension) {
            indices.push_back(static_cast<int32_t>(remaining % sourceType.getDimSize(dimension)));
            remaining /= sourceType.getDimSize(dimension);
        }
        std::reverse(indices.begin(), indices.end());
        indices.append(carrierIndices);
        leaves.push_back(spirv::CompositeExtractOp::create(builder, location, value, indices));
    }
    return constructComposite(location, logicalType, leaves, builder);
}

Value extractStaticTensorElement(Location location, Value value, RankedTensorType sourceType, int64_t linearIndex,
                                 OpBuilder &builder) {
    SmallVector<int32_t> indices;
    int64_t remaining = linearIndex;
    for (int64_t dimension = sourceType.getRank() - 1; dimension >= 0; --dimension) {
        indices.push_back(static_cast<int32_t>(remaining % sourceType.getDimSize(dimension)));
        remaining /= sourceType.getDimSize(dimension);
    }
    std::reverse(indices.begin(), indices.end());
    if (isa<spirv::MatrixType>(value.getType()))
        std::swap(indices[0], indices[1]);
    return indices.empty() ? value : spirv::CompositeExtractOp::create(builder, location, value, indices).getResult();
}

FailureOr<Value> lowerTensorExtract(tensor::ExtractOp extract, OpBuilder &builder, IRMapping &mapping) {
    auto sourceType = dyn_cast<RankedTensorType>(extract.getTensor().getType());
    Value input = mapping.lookupOrNull(extract.getTensor());
    if (!sourceType || !sourceType.hasStaticShape() || !input ||
        extract.getIndices().size() != static_cast<size_t>(sourceType.getRank()))
        return failure();
    if (sourceType.getRank() == 0)
        return input;
    Location location = extract.getLoc();
    int64_t constantLinear = 0;
    bool hasConstantIndices = true;
    for (auto [dimension, sourceIndex] : llvm::zip_equal(sourceType.getShape(), extract.getIndices())) {
        if (auto cast = sourceIndex.getDefiningOp<arith::IndexCastOp>())
            sourceIndex = cast.getIn();
        auto constant = sourceIndex.getDefiningOp<arith::ConstantOp>();
        auto index = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
        if (!index || index.getInt() < 0 || index.getInt() >= dimension) {
            hasConstantIndices = false;
            break;
        }
        constantLinear = constantLinear * dimension + index.getInt();
    }
    if (hasConstantIndices)
        return extractStaticTensorElement(location, input, sourceType, constantLinear, builder);

    Value linear = mapping.lookupOrNull(extract.getIndices().front());
    if (!linear)
        return failure();
    for (auto [extent, sourceIndex] :
         llvm::zip_equal(sourceType.getShape().drop_front(), extract.getIndices().drop_front())) {
        Value index = mapping.lookupOrNull(sourceIndex);
        if (!index)
            return failure();
        Value extentValue = spirv::ConstantOp::create(builder, location, linear.getType(),
                                                      builder.getIntegerAttr(linear.getType(), extent));
        linear = spirv::IMulOp::create(builder, location, linear, extentValue);
        linear = spirv::IAddOp::create(builder, location, linear, index);
    }
    Value selected = extractStaticTensorElement(location, input, sourceType, 0, builder);
    for (int64_t index = 1; index < sourceType.getNumElements(); ++index) {
        Value expected = spirv::ConstantOp::create(builder, location, linear.getType(),
                                                   builder.getIntegerAttr(linear.getType(), index));
        Value matches = spirv::IEqualOp::create(builder, location, linear, expected);
        Value candidate = extractStaticTensorElement(location, input, sourceType, index, builder);
        selected = spirv::SelectOp::create(builder, location, candidate.getType(), matches, candidate, selected);
    }
    return selected;
}

template <typename CreateLeaf>
FailureOr<Value> lowerCompositeElementwise(Location location, Type resultType, ArrayRef<Value> operands,
                                           OpBuilder &builder, CreateLeaf createLeaf) {
    if (auto array = dyn_cast<spirv::ArrayType>(resultType)) {
        SmallVector<Value> elements;
        for (unsigned index = 0; index < array.getNumElements(); ++index) {
            SmallVector<Value> extracted;
            for (Value operand : operands)
                extracted.push_back(spirv::CompositeExtractOp::create(builder, location, operand,
                                                                      ArrayRef<int32_t>{static_cast<int32_t>(index)}));
            FailureOr<Value> element =
                lowerCompositeElementwise(location, array.getElementType(), extracted, builder, createLeaf);
            if (failed(element))
                return failure();
            elements.push_back(*element);
        }
        return spirv::CompositeConstructOp::create(builder, location, resultType, elements).getResult();
    }
    if (auto matrix = dyn_cast<spirv::MatrixType>(resultType)) {
        SmallVector<Value> columns;
        for (unsigned column = 0; column < matrix.getNumColumns(); ++column) {
            SmallVector<Value> extracted;
            for (Value operand : operands)
                extracted.push_back(spirv::CompositeExtractOp::create(builder, location, operand,
                                                                      ArrayRef<int32_t>{static_cast<int32_t>(column)}));
            FailureOr<Value> value =
                lowerCompositeElementwise(location, matrix.getColumnType(), extracted, builder, createLeaf);
            if (failed(value))
                return failure();
            columns.push_back(*value);
        }
        return spirv::CompositeConstructOp::create(builder, location, resultType, columns).getResult();
    }
    return createLeaf(resultType, operands);
}

FailureOr<Value> translateOperation(Operation &operation, OpBuilder &builder, IRMapping &mapping) {
    Location location = operation.getLoc();
    ModuleOp sourceModule = operation.getParentOfType<ModuleOp>();
    auto mapped = [&](Value value) { return mapping.lookupOrNull(value); };

    if (auto construct = dyn_cast<spirv::CompositeConstructOp>(operation)) {
        SmallVector<Value> constituents;
        for (Value constituent : construct.getConstituents()) {
            Value converted = mapped(constituent);
            if (!converted)
                return failure();
            constituents.push_back(converted);
        }
        return spirv::CompositeConstructOp::create(builder, location, construct.getType(), constituents).getResult();
    }
    if (auto extract = dyn_cast<spirv::CompositeExtractOp>(operation)) {
        Value composite = mapped(extract.getComposite());
        if (!composite)
            return failure();
        SmallVector<int32_t> indices;
        for (Attribute index : extract.getIndices())
            indices.push_back(cast<IntegerAttr>(index).getInt());
        return spirv::CompositeExtractOp::create(builder, location, composite, indices).getResult();
    }

    if (auto extract = dyn_cast<TensorGetOp>(operation)) {
        auto sourceType = dyn_cast<TensorType>(extract.getInput().getType());
        if (!sourceType || extract.getIndices().size() != sourceType.getShape().size())
            return failure();
        int64_t linear = 0;
        bool hasConstantIndices = true;
        for (auto [dimension, sourceIndex] : llvm::zip_equal(sourceType.getShape(), extract.getIndices())) {
            if (auto cast = sourceIndex.getDefiningOp<arith::IndexCastOp>())
                sourceIndex = cast.getIn();
            auto constant = sourceIndex.getDefiningOp<arith::ConstantOp>();
            auto index = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
            if (!index || index.getInt() < 0 || index.getInt() >= dimension) {
                hasConstantIndices = false;
                break;
            }
            linear = linear * dimension + index.getInt();
        }
        if (auto construct = extract.getInput().getDefiningOp<IntrinsicOp>();
            hasConstantIndices && construct && construct.getName() == "construct" &&
            linear < static_cast<int64_t>(construct.getNumOperands())) {
            Value element = mapped(construct.getOperand(linear));
            if (element)
                return element;
        }
        Value input = mapped(extract.getInput());
        if (!input)
            return failure();
        if (auto pointerType = dyn_cast<spirv::PointerType>(input.getType());
            pointerType && pointerType.getStorageClass() == spirv::StorageClass::StorageBuffer) {
            Value blockMember =
                spirv::ConstantOp::create(builder, location, builder.getI32Type(), builder.getI32IntegerAttr(0));
            Value recordIndex;
            if (hasConstantIndices) {
                recordIndex = spirv::ConstantOp::create(builder, location, builder.getI32Type(),
                                                        builder.getI32IntegerAttr(linear));
            } else {
                recordIndex = mapped(extract.getIndices().front());
                if (!recordIndex)
                    return failure();
                for (auto [extent, sourceIndex] :
                     llvm::zip_equal(sourceType.getShape().drop_front(), extract.getIndices().drop_front())) {
                    Value index = mapped(sourceIndex);
                    if (!index)
                        return failure();
                    Value extentValue =
                        spirv::ConstantOp::create(builder, location, recordIndex.getType(),
                                                  builder.getIntegerAttr(recordIndex.getType(), extent));
                    recordIndex = spirv::IMulOp::create(builder, location, recordIndex, extentValue);
                    recordIndex = spirv::IAddOp::create(builder, location, recordIndex, index);
                }
            }
            SmallVector<Value> leaves;
            auto block = dyn_cast<spirv::StructType>(pointerType.getPointeeType());
            auto records = block && block.getNumElements() == 1 ? dyn_cast<spirv::ArrayType>(block.getElementType(0))
                                                                : spirv::ArrayType{};
            auto record = records ? dyn_cast<spirv::StructType>(records.getElementType()) : spirv::StructType{};
            if (!record)
                return failure();
            for (unsigned member = 0; member < record.getNumElements(); ++member) {
                Value memberIndex = spirv::ConstantOp::create(builder, location, builder.getI32Type(),
                                                              builder.getI32IntegerAttr(member));
                Value leafPointer = spirv::AccessChainOp::create(builder, location, input,
                                                                 ValueRange{blockMember, recordIndex, memberIndex});
                leaves.push_back(spirv::LoadOp::create(builder, location, leafPointer));
            }
            FailureOr<Type> elementType = convertValueType(sourceType.getElementType(), sourceModule);
            return failed(elementType) ? FailureOr<Value>(failure())
                                       : constructComposite(location, *elementType, leaves, builder);
        }
        if (!hasConstantIndices)
            return failure();
        return spirv::CompositeExtractOp::create(builder, location, input,
                                                 ArrayRef<int32_t>{static_cast<int32_t>(linear)})
            .getResult();
    }

    if (auto splat = dyn_cast<tensor::SplatOp>(operation)) {
        Value input = mapped(splat.getInput());
        FailureOr<Type> resultType = convertValueType(splat.getType(), sourceModule);
        if (!input || failed(resultType))
            return failure();
        SmallVector<Value> elements(splat.getType().getNumElements(), input);
        return constructComposite(location, *resultType, elements, builder);
    }

    if (auto fromElements = dyn_cast<tensor::FromElementsOp>(operation)) {
        FailureOr<Type> resultType = convertValueType(fromElements.getType(), sourceModule);
        SmallVector<Value> elements;
        for (Value element : fromElements.getElements()) {
            Value converted = mapped(element);
            if (!converted || failed(resultType))
                return failure();
            flattenComposite(location, converted, builder, elements);
        }
        return constructComposite(location, *resultType, elements, builder);
    }
    if (auto fromElements = dyn_cast<vector::FromElementsOp>(operation)) {
        FailureOr<Type> resultType = convertValueType(fromElements.getType(), sourceModule);
        SmallVector<Value> elements;
        for (Value element : fromElements.getElements()) {
            Value converted = mapped(element);
            if (!converted || failed(resultType))
                return failure();
            elements.push_back(converted);
        }
        return constructComposite(location, *resultType, elements, builder);
    }

    if (auto extract = dyn_cast<tensor::ExtractOp>(operation))
        return lowerTensorExtract(extract, builder, mapping);

    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        FailureOr<Type> type = convertValueType(constant.getType(), sourceModule);
        if (failed(type))
            return failure();
        Attribute value = constant.getValue();
        if (auto dense = dyn_cast<DenseElementsAttr>(value)) {
            SmallVector<Value> elements;
            for (Attribute element : dense.getValues<Attribute>()) {
                auto typed = cast<TypedAttr>(element);
                elements.push_back(spirv::ConstantOp::create(builder, location, typed.getType(), typed));
            }
            return constructComposite(location, *type, elements, builder);
        }
        if (constant.getType().isIndex()) {
            auto integer = cast<IntegerAttr>(value);
            value = builder.getIntegerAttr(*type, integer.getInt());
        }
        return spirv::ConstantOp::create(builder, location, *type, value).getResult();
    }

    if (auto swizzle = dyn_cast<SwizzleOp>(operation)) {
        Value input = mapped(swizzle.getInput());
        if (!input)
            return failure();
        SmallVector<int32_t> components;
        for (char component : swizzle.getMask()) {
            std::optional<unsigned> index = decodeSwizzleComponent(component);
            if (!index) {
                swizzle.emitError() << "cannot lower invalid swizzle component '" << component << "'";
                return failure();
            }
            components.push_back(static_cast<int32_t>(*index));
        }
        FailureOr<Type> resultType = convertValueType(swizzle.getResult().getType(), sourceModule);
        if (failed(resultType))
            return failure();
        if (components.size() == 1)
            return spirv::CompositeExtractOp::create(builder, location, input, components).getResult();
        SmallVector<Attribute> componentAttrs;
        for (int32_t component : components)
            componentAttrs.push_back(builder.getI32IntegerAttr(component));
        return spirv::VectorShuffleOp::create(builder, location, *resultType, input, input,
                                              builder.getArrayAttr(componentAttrs))
            .getResult();
    }

    if (auto tuple = dyn_cast<TupleCreateOp>(operation)) {
        SmallVector<Value> elements;
        for (Value element : tuple.getElements()) {
            Value converted = mapped(element);
            if (!converted)
                return failure();
            elements.push_back(converted);
        }
        FailureOr<Type> resultType = convertValueType(tuple.getResult().getType(), sourceModule);
        if (failed(resultType))
            return failure();
        return spirv::CompositeConstructOp::create(builder, location, *resultType, elements).getResult();
    }

    if (auto tuple = dyn_cast<TupleGetOp>(operation)) {
        Value input = mapped(tuple.getInput());
        if (!input)
            return failure();
        return spirv::CompositeExtractOp::create(builder, location, input,
                                                 ArrayRef<int32_t>{static_cast<int32_t>(tuple.getIndex())})
            .getResult();
    }

    if (auto structure = dyn_cast<StructCreateOp>(operation)) {
        SmallVector<Value> fields;
        for (Value field : structure.getFields()) {
            Value converted = mapped(field);
            if (!converted)
                return failure();
            fields.push_back(converted);
        }
        FailureOr<Type> resultType = convertValueType(structure.getResult().getType(), sourceModule);
        if (failed(resultType))
            return failure();
        return spirv::CompositeConstructOp::create(builder, location, *resultType, fields).getResult();
    }

    if (auto structure = dyn_cast<StructGetOp>(operation)) {
        Value input = mapped(structure.getInput());
        if (!input)
            return failure();
        return spirv::CompositeExtractOp::create(builder, location, input,
                                                 ArrayRef<int32_t>{static_cast<int32_t>(structure.getIndex())})
            .getResult();
    }

    if (auto intrinsic = dyn_cast<IntrinsicOp>(operation)) {
        if (intrinsic.getName() == "texture_sample") {
            Value sampledImage = mapped(intrinsic.getOperand(0));
            Value coordinates = mapped(intrinsic.getOperand(2));
            FailureOr<Type> resultType = convertValueType(intrinsic.getResult().getType());
            if (!sampledImage || !coordinates || failed(resultType))
                return failure();
            if (intrinsic.getNumOperands() == 4) {
                Value lod = mapped(intrinsic.getOperand(3));
                if (!lod)
                    return failure();
                auto imageOperands = spirv::ImageOperandsAttr::get(builder.getContext(), spirv::ImageOperands::Lod);
                return spirv::ImageSampleExplicitLodOp::create(builder, location, *resultType, sampledImage,
                                                               coordinates, imageOperands, ValueRange{lod})
                    .getResult();
            }
            return spirv::ImageSampleImplicitLodOp::create(builder, location, *resultType, sampledImage, coordinates,
                                                           spirv::ImageOperandsAttr(), ValueRange{})
                .getResult();
        }
        if (intrinsic.getName() == "texture_size") {
            Value sampledImage = mapped(intrinsic.getOperand(0));
            FailureOr<Type> resultType = convertValueType(intrinsic.getResult().getType());
            auto sampledImageType = sampledImage ? dyn_cast<spirv::SampledImageType>(sampledImage.getType()) : nullptr;
            auto vectorType = succeeded(resultType) ? dyn_cast<VectorType>(*resultType) : nullptr;
            if (!sampledImageType || !vectorType)
                return failure();

            Value lod;
            Type lodType = builder.getI32Type();
            if (intrinsic.getNumOperands() == 2) {
                lod = mapped(intrinsic.getOperand(1));
                if (!lod)
                    return failure();
                lodType = lod.getType();
            } else {
                lod = spirv::ConstantOp::create(builder, location, lodType, builder.getI32IntegerAttr(0));
            }
            Value lodMarker = spirv::ConstantOp::create(
                builder, location, lodType, builder.getIntegerAttr(lodType, APInt(32, kImageQueryLodMarker)));
            auto resultMarkerA = DenseElementsAttr::get(
                vectorType, IntegerAttr::get(vectorType.getElementType(), APInt(32, kImageQueryResultMarkerA)));
            auto resultMarkerB = DenseElementsAttr::get(
                vectorType, IntegerAttr::get(vectorType.getElementType(), APInt(32, kImageQueryResultMarkerB)));
            Value resultVectorMarkerA = spirv::ConstantOp::create(builder, location, vectorType, resultMarkerA);
            Value resultVectorMarkerB = spirv::ConstantOp::create(builder, location, vectorType, resultMarkerB);
            Type imageType = sampledImageType.getImageType();
            Value image = spirv::ImageOp::create(builder, location, imageType, sampledImage);
            (void)image;
            // The marker values are deliberately non-neutral. They must never
            // survive serialization fixup, which validates and replaces this exact
            // sequence before exposing the artifact.
            Value serializedLod = spirv::IAddOp::create(builder, location, lod, lodMarker).getResult();
            (void)serializedLod;
            return spirv::IAddOp::create(builder, location, resultVectorMarkerA, resultVectorMarkerB).getResult();
        }
        SmallVector<Value> operands;
        for (Value operand : intrinsic.getOperands()) {
            Value converted = mapped(operand);
            if (!converted)
                return failure();
            operands.push_back(converted);
        }
        FailureOr<Type> resultType = convertValueType(intrinsic.getResult().getType(), sourceModule);
        if (failed(resultType))
            return failure();
        StringRef name = intrinsic.getName();
        if (name == "construct") {
            SmallVector<Value> leaves;
            for (Value operand : operands)
                flattenComposite(location, operand, builder, leaves);
            return constructComposite(location, *resultType, leaves, builder);
        }
        if (name == "broadcast") {
            auto sourceType = dyn_cast<RankedTensorType>(intrinsic.getOperand(0).getType());
            auto resultTensorType = dyn_cast<RankedTensorType>(intrinsic.getResult().getType());
            if (!sourceType || !resultTensorType || operands.size() != 1)
                return failure();
            SmallVector<Value> leaves;
            for (int64_t resultIndex = 0; resultIndex < resultTensorType.getNumElements(); ++resultIndex) {
                FailureOr<int64_t> sourceIndex =
                    getStaticBroadcastLinearIndex(sourceType.getShape(), resultTensorType.getShape(), resultIndex);
                if (failed(sourceIndex))
                    return failure();
                leaves.push_back(extractStaticTensorElement(location, operands[0], sourceType, *sourceIndex, builder));
            }
            return constructComposite(location, *resultType, leaves, builder);
        }
        if (name == "matmul") {
            auto leftType = dyn_cast<RankedTensorType>(intrinsic.getOperand(0).getType());
            auto rightType = dyn_cast<RankedTensorType>(intrinsic.getOperand(1).getType());
            if (!leftType || !rightType || operands.size() != 2)
                return failure();
            if (leftType.getRank() == 2 && rightType.getRank() == 2 && isa<spirv::MatrixType>(operands[0].getType()) &&
                isa<spirv::MatrixType>(operands[1].getType()))
                return spirv::MatrixTimesMatrixOp::create(builder, location, *resultType, operands[0], operands[1])
                    .getResult();
            if (leftType.getRank() == 2 && rightType.getRank() == 1 && isa<spirv::MatrixType>(operands[0].getType()) &&
                isa<VectorType>(operands[1].getType()))
                return spirv::MatrixTimesVectorOp::create(builder, location, *resultType, operands[0], operands[1])
                    .getResult();

            FailureOr<StaticMatmulPlan> plan = getStaticMatmulPlan(leftType.getShape(), rightType.getShape());
            if (failed(plan) || !isa<FloatType>(leftType.getElementType()))
                return failure();
            FailureOr<int64_t> resultCount = getStaticShapeElementCount(plan->resultShape);
            if (failed(resultCount))
                return failure();
            SmallVector<Value> leaves;
            for (int64_t resultIndex = 0; resultIndex < *resultCount; ++resultIndex) {
                Value sum = spirv::ConstantOp::create(builder, location, leftType.getElementType(),
                                                      builder.getFloatAttr(leftType.getElementType(), 0.0));
                for (int64_t reduction = 0; reduction < plan->reduction; ++reduction) {
                    FailureOr<int64_t> leftIndex = getStaticMatmulLeftLinearIndex(*plan, resultIndex, reduction);
                    FailureOr<int64_t> rightIndex = getStaticMatmulRightLinearIndex(*plan, resultIndex, reduction);
                    if (failed(leftIndex) || failed(rightIndex))
                        return failure();
                    Value lhs = extractStaticTensorElement(location, operands[0], leftType, *leftIndex, builder);
                    Value rhs = extractStaticTensorElement(location, operands[1], rightType, *rightIndex, builder);
                    Value product = spirv::FMulOp::create(builder, location, lhs, rhs);
                    sum = spirv::FAddOp::create(builder, location, sum, product);
                }
                leaves.push_back(sum);
            }
            return constructComposite(location, *resultType, leaves, builder);
        }
        StringRef operationName = llvm::StringSwitch<StringRef>(name)
                                      .Case("dot", "spirv.Dot")
                                      .Case("cross", "spirv.GL.Cross")
                                      .Case("normalize", "spirv.GL.Normalize")
                                      .Case("reflect", "spirv.GL.Reflect")
                                      .Case("min", "spirv.GL.FMin")
                                      .Case("max", "spirv.GL.FMax")
                                      .Case("pow", "spirv.GL.Pow")
                                      .Case("clamp", "spirv.GL.FClamp")
                                      .Default("");
        if (operationName.empty())
            return failure();
        OperationState state(location, operationName);
        state.addOperands(operands);
        state.addTypes(*resultType);
        return builder.create(state)->getResult(0);
    }

    if (operation.getNumOperands() == 1 && operation.getNumResults() == 1) {
        Value operand = mapped(operation.getOperand(0));
        FailureOr<Type> resultType = convertValueType(operation.getResult(0).getType());
        if (!operand || failed(resultType))
            return failure();
        StringRef operationName = llvm::StringSwitch<StringRef>(operation.getName().getStringRef())
                                      .Case(math::AcosOp::getOperationName(), "spirv.GL.Acos")
                                      .Case(math::SinOp::getOperationName(), "spirv.GL.Sin")
                                      .Case(math::CosOp::getOperationName(), "spirv.GL.Cos")
                                      .Case(math::ExpOp::getOperationName(), "spirv.GL.Exp")
                                      .Case(math::FloorOp::getOperationName(), "spirv.GL.Floor")
                                      .Case(math::LogOp::getOperationName(), "spirv.GL.Log")
                                      .Case(math::SqrtOp::getOperationName(), "spirv.GL.Sqrt")
                                      .Case(math::AbsFOp::getOperationName(), "spirv.GL.FAbs")
                                      .Default("");
        if (!operationName.empty()) {
            OperationState state(location, operationName);
            state.addOperands(operand);
            state.addTypes(*resultType);
            return builder.create(state)->getResult(0);
        }
    }

    if (auto atan2 = dyn_cast<math::Atan2Op>(operation)) {
        Value y = mapped(atan2.getLhs());
        Value x = mapped(atan2.getRhs());
        FailureOr<Type> resultType = convertValueType(atan2.getType());
        if (!y || !x || failed(resultType))
            return failure();
        return lowerCompositeElementwise(location, *resultType, ArrayRef<Value>{y, x}, builder,
                                         [&](Type type, ArrayRef<Value> values) -> FailureOr<Value> {
                                             return lowerAtan2ToSpirv(location, type, values[0], values[1], builder);
                                         });
    }

    if (auto compare = dyn_cast<arith::CmpFOp>(operation)) {
        Value lhs = mapped(compare.getLhs());
        Value rhs = mapped(compare.getRhs());
        if (!lhs || !rhs)
            return failure();
        switch (compare.getPredicate()) {
        case arith::CmpFPredicate::OEQ:
            return spirv::FOrdEqualOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpFPredicate::ONE:
            return spirv::FOrdNotEqualOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpFPredicate::OLT:
            return spirv::FOrdLessThanOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpFPredicate::OLE:
            return spirv::FOrdLessThanEqualOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpFPredicate::OGT:
            return spirv::FOrdGreaterThanOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpFPredicate::OGE:
            return spirv::FOrdGreaterThanEqualOp::create(builder, location, lhs, rhs).getResult();
        default:
            return failure();
        }
    }

    if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
        Value lhs = mapped(compare.getLhs());
        Value rhs = mapped(compare.getRhs());
        if (!lhs || !rhs)
            return failure();
        switch (compare.getPredicate()) {
        case arith::CmpIPredicate::eq:
            if (lhs.getType().isInteger(1))
                return spirv::LogicalEqualOp::create(builder, location, lhs, rhs).getResult();
            return spirv::IEqualOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpIPredicate::ne:
            if (lhs.getType().isInteger(1))
                return spirv::LogicalNotEqualOp::create(builder, location, lhs, rhs).getResult();
            return spirv::INotEqualOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpIPredicate::slt:
            return spirv::SLessThanOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpIPredicate::sle:
            return spirv::SLessThanEqualOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpIPredicate::sgt:
            return spirv::SGreaterThanOp::create(builder, location, lhs, rhs).getResult();
        case arith::CmpIPredicate::sge:
            return spirv::SGreaterThanEqualOp::create(builder, location, lhs, rhs).getResult();
        default:
            return failure();
        }
    }

    if (auto cast = dyn_cast<arith::SIToFPOp>(operation)) {
        Value input = mapped(cast.getIn());
        FailureOr<Type> resultType = convertValueType(cast.getType());
        if (!input || failed(resultType))
            return failure();
        return lowerCompositeElementwise(
            location, *resultType, ArrayRef<Value>{input}, builder,
            [&](Type type, ArrayRef<Value> values) -> FailureOr<Value> {
                return spirv::ConvertSToFOp::create(builder, location, type, values[0]).getResult();
            });
    }
    if (auto cast = dyn_cast<arith::UIToFPOp>(operation)) {
        Value input = mapped(cast.getIn());
        FailureOr<Type> resultType = convertValueType(cast.getType());
        if (!input || failed(resultType))
            return failure();
        return lowerCompositeElementwise(
            location, *resultType, ArrayRef<Value>{input}, builder,
            [&](Type type, ArrayRef<Value> values) -> FailureOr<Value> {
                return spirv::ConvertUToFOp::create(builder, location, type, values[0]).getResult();
            });
    }

    if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        Value condition = mapped(select.getCondition());
        Value trueValue = mapped(select.getTrueValue());
        Value falseValue = mapped(select.getFalseValue());
        FailureOr<Type> resultType = convertValueType(select.getType());
        if (!condition || !trueValue || !falseValue || failed(resultType))
            return failure();
        return lowerCompositeElementwise(
            location, *resultType, ArrayRef<Value>{trueValue, falseValue}, builder,
            [&](Type type, ArrayRef<Value> values) -> FailureOr<Value> {
                return spirv::SelectOp::create(builder, location, type, condition, values[0], values[1]).getResult();
            });
    }

    if (auto cast = dyn_cast<arith::IndexCastOp>(operation)) {
        Value input = mapped(cast.getIn());
        FailureOr<Type> resultType = convertValueType(cast.getType());
        if (!input || failed(resultType) || input.getType() != *resultType)
            return failure();
        return input;
    }

    if (operation.getNumOperands() == 2 && operation.getNumResults() == 1) {
        Value lhs = mapped(operation.getOperand(0));
        Value rhs = mapped(operation.getOperand(1));
        if (!lhs || !rhs)
            return failure();
        StringRef name = operation.getName().getStringRef();
        FailureOr<Type> resultType = convertValueType(operation.getResult(0).getType());
        if (failed(resultType))
            return failure();
        auto lowerBinary = [&](auto create) {
            return lowerCompositeElementwise(location, *resultType, ArrayRef<Value>{lhs, rhs}, builder,
                                             [&](Type, ArrayRef<Value> values) -> FailureOr<Value> {
                                                 return create(values[0], values[1]).getResult();
                                             });
        };
        if (name == arith::AddFOp::getOperationName())
            return lowerBinary([&](Value a, Value b) { return spirv::FAddOp::create(builder, location, a, b); });
        if (name == arith::SubFOp::getOperationName())
            return lowerBinary([&](Value a, Value b) { return spirv::FSubOp::create(builder, location, a, b); });
        if (name == arith::MulFOp::getOperationName())
            return lowerBinary([&](Value a, Value b) { return spirv::FMulOp::create(builder, location, a, b); });
        if (name == arith::DivFOp::getOperationName())
            return lowerBinary([&](Value a, Value b) { return spirv::FDivOp::create(builder, location, a, b); });
        if (name == arith::AddIOp::getOperationName())
            return lowerBinary([&](Value a, Value b) { return spirv::IAddOp::create(builder, location, a, b); });
        if (name == arith::SubIOp::getOperationName())
            return lowerBinary([&](Value a, Value b) { return spirv::ISubOp::create(builder, location, a, b); });
        if (name == arith::MulIOp::getOperationName())
            return lowerBinary([&](Value a, Value b) { return spirv::IMulOp::create(builder, location, a, b); });
        if (name == arith::AndIOp::getOperationName()) {
            if (operation.getResult(0).getType().isInteger(1))
                return spirv::LogicalAndOp::create(builder, location, lhs, rhs).getResult();
            return spirv::BitwiseAndOp::create(builder, location, lhs, rhs).getResult();
        }
        if (name == arith::OrIOp::getOperationName()) {
            if (operation.getResult(0).getType().isInteger(1))
                return spirv::LogicalOrOp::create(builder, location, lhs, rhs).getResult();
            return spirv::BitwiseOrOp::create(builder, location, lhs, rhs).getResult();
        }
        if (name == arith::XOrIOp::getOperationName()) {
            if (operation.getResult(0).getType().isInteger(1))
                return spirv::LogicalNotEqualOp::create(builder, location, lhs, rhs).getResult();
            return spirv::BitwiseXorOp::create(builder, location, lhs, rhs).getResult();
        }
    }

    return failure();
}

LogicalResult translateStraightLineBlock(Block &source, OpBuilder &builder, Block *functionEntry, IRMapping &mapping,
                                         SmallVectorImpl<Value> &yieldedValues, Value *condition = nullptr);

LogicalResult lowerWhile(scf::WhileOp whileOp, OpBuilder &builder, Block *functionEntry, IRMapping &mapping,
                         SmallVectorImpl<Value> &results);

LogicalResult lowerIf(scf::IfOp ifOp, OpBuilder &builder, Block *functionEntry, IRMapping &mapping,
                      SmallVectorImpl<Value> &results) {
    Value condition = mapping.lookupOrNull(ifOp.getCondition());
    ModuleOp sourceModule = ifOp->getParentOfType<ModuleOp>();
    if (!condition || !ifOp.getThenRegion().hasOneBlock() ||
        (!ifOp.getElseRegion().empty() && !ifOp.getElseRegion().hasOneBlock()))
        return failure();

    SmallVector<Value> resultVariables;
    OpBuilder variableBuilder = OpBuilder::atBlockBegin(functionEntry);
    for (Type sourceType : ifOp.getResultTypes()) {
        FailureOr<Type> type = convertValueType(sourceType, sourceModule);
        if (failed(type))
            return failure();
        auto pointerType = spirv::PointerType::get(*type, spirv::StorageClass::Function);
        resultVariables.push_back(spirv::VariableOp::create(variableBuilder, ifOp.getLoc(), pointerType,
                                                            spirv::StorageClass::Function, /*initializer=*/nullptr));
    }

    auto selection = spirv::SelectionOp::create(builder, ifOp.getLoc(), spirv::SelectionControl::None);
    {
        OpBuilder::InsertionGuard guard(builder);
        Region &body = selection.getBody();
        Block *header = builder.createBlock(&body, body.end());
        Block *thenBlock = builder.createBlock(&body, body.end());
        Block *elseBlock = builder.createBlock(&body, body.end());
        Block *mergeBlock = builder.createBlock(&body, body.end());

        builder.setInsertionPointToEnd(header);
        spirv::BranchConditionalOp::create(builder, ifOp.getLoc(), condition, thenBlock, ValueRange{}, elseBlock,
                                           ValueRange{});

        SmallVector<Value> thenValues;
        OpBuilder thenBuilder = OpBuilder::atBlockBegin(thenBlock);
        if (failed(translateStraightLineBlock(ifOp.getThenRegion().front(), thenBuilder, functionEntry, mapping,
                                              thenValues)) ||
            thenValues.size() != resultVariables.size())
            return failure();
        for (auto [variable, value] : llvm::zip_equal(resultVariables, thenValues))
            spirv::StoreOp::create(thenBuilder, ifOp.getLoc(), variable, value);
        spirv::BranchOp::create(thenBuilder, ifOp.getLoc(), mergeBlock);

        SmallVector<Value> elseValues;
        OpBuilder elseBuilder = OpBuilder::atBlockBegin(elseBlock);
        if (ifOp.getElseRegion().empty()) {
            if (!resultVariables.empty())
                return failure();
        } else if (failed(translateStraightLineBlock(ifOp.getElseRegion().front(), elseBuilder, functionEntry, mapping,
                                                     elseValues)) ||
                   elseValues.size() != resultVariables.size()) {
            return failure();
        }
        for (auto [variable, value] : llvm::zip_equal(resultVariables, elseValues))
            spirv::StoreOp::create(elseBuilder, ifOp.getLoc(), variable, value);
        spirv::BranchOp::create(elseBuilder, ifOp.getLoc(), mergeBlock);

        builder.setInsertionPointToEnd(mergeBlock);
        spirv::MergeOp::create(builder, ifOp.getLoc());
    }

    for (Value variable : resultVariables)
        results.push_back(spirv::LoadOp::create(builder, ifOp.getLoc(), variable).getResult());
    return success();
}

LogicalResult translateStraightLineBlock(Block &source, OpBuilder &builder, Block *functionEntry, IRMapping &mapping,
                                         SmallVectorImpl<Value> &yieldedValues, Value *condition) {
    for (Operation &operation : source) {
        if (auto conditionOp = dyn_cast<scf::ConditionOp>(operation)) {
            if (!condition)
                return failure();
            *condition = mapping.lookupOrNull(conditionOp.getCondition());
            if (!*condition)
                return failure();
            for (Value operand : conditionOp.getArgs()) {
                Value mapped = mapping.lookupOrNull(operand);
                if (!mapped)
                    return failure();
                yieldedValues.push_back(mapped);
            }
            return success();
        }
        if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
            for (Value operand : yield.getOperands()) {
                Value mapped = mapping.lookupOrNull(operand);
                if (!mapped)
                    return failure();
                yieldedValues.push_back(mapped);
            }
            return success();
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
            SmallVector<Value> translatedResults;
            if (failed(lowerIf(ifOp, builder, functionEntry, mapping, translatedResults)))
                return failure();
            if (translatedResults.size() != ifOp.getNumResults())
                return failure();
            for (auto [sourceResult, translatedResult] : llvm::zip_equal(ifOp.getResults(), translatedResults))
                mapping.map(sourceResult, translatedResult);
            continue;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
            SmallVector<Value> translatedResults;
            if (failed(lowerWhile(whileOp, builder, functionEntry, mapping, translatedResults)) ||
                translatedResults.size() != whileOp.getNumResults())
                return failure();
            for (auto [sourceResult, translatedResult] : llvm::zip_equal(whileOp.getResults(), translatedResults))
                mapping.map(sourceResult, translatedResult);
            continue;
        }
        FailureOr<Value> translated = translateOperation(operation, builder, mapping);
        if (failed(translated) || operation.getNumResults() != 1)
            return operation.emitError() << "operation cannot be translated inside structured control flow: "
                                         << operation.getName();
        mapping.map(operation.getResult(0), *translated);
    }
    return success();
}

LogicalResult lowerWhile(scf::WhileOp whileOp, OpBuilder &builder, Block *functionEntry, IRMapping &mapping,
                         SmallVectorImpl<Value> &results) {
    if (!whileOp.getBefore().hasOneBlock() || !whileOp.getAfter().hasOneBlock())
        return whileOp.emitError("structured loop regions must contain one block");

    ModuleOp sourceModule = whileOp->getParentOfType<ModuleOp>();
    SmallVector<Value> resultVariables;
    OpBuilder variableBuilder = OpBuilder::atBlockBegin(functionEntry);
    for (Type sourceType : whileOp.getResultTypes()) {
        FailureOr<Type> type = convertValueType(sourceType, sourceModule);
        if (failed(type))
            return whileOp.emitError("cannot convert a structured loop result type");
        auto pointerType = spirv::PointerType::get(*type, spirv::StorageClass::Function);
        resultVariables.push_back(spirv::VariableOp::create(variableBuilder, whileOp.getLoc(), pointerType,
                                                            spirv::StorageClass::Function, /*initializer=*/nullptr));
    }

    auto loop = spirv::LoopOp::create(builder, whileOp.getLoc(), spirv::LoopControl::None);
    loop.addEntryAndMergeBlock(builder);
    {
        OpBuilder::InsertionGuard guard(builder);
        Region &body = loop.getBody();
        Block *entryBlock = loop.getEntryBlock();
        Block *mergeBlock = loop.getMergeBlock();
        Block *beforeBlock = builder.createBlock(&body, std::prev(body.end()));
        Block *afterBlock = builder.createBlock(&body, std::prev(body.end()));

        SmallVector<Value> initialValues;
        for (Value operand : whileOp.getInits()) {
            Value mapped = mapping.lookupOrNull(operand);
            if (!mapped)
                return whileOp.emitError("structured loop initial value is not mapped");
            initialValues.push_back(mapped);
        }
        for (auto [argument, value] : llvm::zip_equal(whileOp.getBefore().front().getArguments(), initialValues)) {
            FailureOr<Type> type = convertValueType(argument.getType(), sourceModule);
            if (failed(type))
                return whileOp.emitError("cannot convert a structured loop before argument type");
            BlockArgument targetArgument = beforeBlock->addArgument(*type, argument.getLoc());
            mapping.map(argument, targetArgument);
        }

        SmallVector<Value> conditionValues;
        Value condition;
        OpBuilder beforeBuilder = OpBuilder::atBlockBegin(beforeBlock);
        if (failed(translateStraightLineBlock(whileOp.getBefore().front(), beforeBuilder, functionEntry, mapping,
                                              conditionValues, &condition)) ||
            conditionValues.size() != whileOp.getAfter().front().getNumArguments())
            return whileOp.emitError("cannot translate a structured loop condition region");
        for (auto [argument, value] : llvm::zip_equal(whileOp.getAfter().front().getArguments(), conditionValues)) {
            FailureOr<Type> type = convertValueType(argument.getType(), sourceModule);
            if (failed(type) || value.getType() != *type)
                return whileOp.emitError("structured loop condition value type does not match its body argument");
            BlockArgument targetArgument = afterBlock->addArgument(*type, argument.getLoc());
            mapping.map(argument, targetArgument);
        }
        for (auto [variable, value] : llvm::zip_equal(resultVariables, conditionValues))
            spirv::StoreOp::create(beforeBuilder, whileOp.getLoc(), variable, value);
        spirv::BranchConditionalOp::create(beforeBuilder, whileOp.getLoc(), condition, afterBlock, conditionValues,
                                           mergeBlock, ValueRange{});

        SmallVector<Value> yieldedValues;
        OpBuilder afterBuilder = OpBuilder::atBlockBegin(afterBlock);
        if (failed(translateStraightLineBlock(whileOp.getAfter().front(), afterBuilder, functionEntry, mapping,
                                              yieldedValues)) ||
            yieldedValues.size() != beforeBlock->getNumArguments())
            return whileOp.emitError("cannot translate a structured loop body region");
        spirv::BranchOp::create(afterBuilder, whileOp.getLoc(), beforeBlock, yieldedValues);

        OpBuilder entryBuilder = OpBuilder::atBlockBegin(entryBlock);
        spirv::BranchOp::create(entryBuilder, whileOp.getLoc(), beforeBlock, initialValues);
    }

    for (Value variable : resultVariables)
        results.push_back(spirv::LoadOp::create(builder, whileOp.getLoc(), variable).getResult());
    return success();
}

LogicalResult lowerEntry(func::FuncOp source, spirv::ModuleOp target, OpBuilder &moduleBuilder,
                         llvm::StringSet<> &usedInterfaceNames, bool useVulkanInterfaceAbi,
                         const DenseMap<Value, std::pair<uint32_t, uint32_t>> &generatedBindings) {
    ModuleOp sourceModule = source->getParentOfType<ModuleOp>();
    auto stage = source->getAttrOfType<StringAttr>(kStageAttrName);
    if (!stage)
        return success();
    std::optional<spirv::ExecutionModel> executionModel = parseExecutionModel(stage.getValue());
    if (!executionModel)
        return source.emitError("unsupported SPIR-V execution model");

    SmallVector<SmallVector<spirv::GlobalVariableOp>> inputs;
    SmallVector<std::optional<uint32_t>> inputMembers;
    SmallVector<bool> inputAttributes;
    SmallVector<spirv::GlobalVariableOp> outputs;
    SmallVector<Attribute> interfaceSymbols;
    struct PushConstantMember {
        uint32_t argumentIndex;
        Type type;
        uint32_t offset;
        std::shared_ptr<const ByteTransportNode> layout;
    };
    SmallVector<PushConstantMember> pushConstantMembers;
    uint32_t pushConstantSize = 0;
    for (auto [index, type] : llvm::enumerate(source.getArgumentTypes())) {
        if (!useVulkanInterfaceAbi)
            break;
        InterfaceAttrs attrs = parseInterfaceAttrs(source.getArgAttrDict(index));
        auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
        if (!kind || kind.getValue() != "uniform" || attrs.binding ||
            generatedBindings.contains(source.getArgument(index)))
            continue;
        FailureOr<Type> converted = convertValueType(type, sourceModule);
        if (failed(converted))
            return source.emitError() << "cannot lower push-constant argument #" << index << " type " << type
                                      << " to SPIR-V";
        FailureOr<ByteTransportPlan> layout =
            getByteTransportPlan(type, sourceModule, PhysicalAbiProfile::VulkanPushConstant);
        if (failed(layout))
            return source.emitError() << "push-constant argument #" << index << " does not have a GPU interface layout";
        pushConstantSize = llvm::alignTo(pushConstantSize, layout->root->alignment);
        pushConstantMembers.push_back({static_cast<uint32_t>(index), *converted, pushConstantSize, layout->root});
        pushConstantSize += layout->root->size;
    }
    spirv::GlobalVariableOp pushConstantBlock;
    if (!pushConstantMembers.empty()) {
        SmallVector<Type> memberTypes;
        SmallVector<uint32_t> memberOffsets;
        SmallVector<spirv::StructType::MemberDecorationInfo> memberDecorations;
        for (auto [member, value] : llvm::enumerate(pushConstantMembers)) {
            memberTypes.push_back(value.type);
            memberOffsets.push_back(value.offset);
            appendMemberDecorations(moduleBuilder, static_cast<uint32_t>(member), value.type, *value.layout,
                                    memberDecorations);
        }
        SmallVector<spirv::StructType::StructDecorationInfo> structDecorations;
        structDecorations.emplace_back(spirv::Decoration::Block, moduleBuilder.getUnitAttr());
        auto blockType = spirv::StructType::get(memberTypes, memberOffsets, memberDecorations, structDecorations);
        std::string preferred = (source.getSymName() + "_push_constants").str();
        std::string name = uniqueInterfaceName(preferred, stage.getValue(), usedInterfaceNames);
        pushConstantBlock = spirv::GlobalVariableOp::create(
            moduleBuilder, source.getLoc(), spirv::PointerType::get(blockType, spirv::StorageClass::PushConstant), name,
            FlatSymbolRefAttr(), IntegerAttr(), IntegerAttr(), IntegerAttr(), StringAttr(),
            spirv::LinkageAttributesAttr());
        interfaceSymbols.push_back(SymbolRefAttr::get(source.getContext(), pushConstantBlock.getSymName()));
    }
    for (auto [index, type] : llvm::enumerate(source.getArgumentTypes())) {
        if (isa<SamplerType>(type)) {
            inputs.emplace_back();
            inputMembers.push_back(std::nullopt);
            inputAttributes.push_back(false);
            continue;
        }
        FailureOr<Type> converted = convertValueType(type, sourceModule);
        if (failed(converted))
            return source.emitError() << "cannot lower argument #" << index << " type " << type << " to SPIR-V";
        Type convertedType = *converted;
        NamedAttrList effectiveAttrs(source.getArgAttrDict(index));
        if (auto generated = generatedBindings.find(source.getArgument(index)); generated != generatedBindings.end()) {
            effectiveAttrs.set(kDescriptorSetAttrName, moduleBuilder.getI64IntegerAttr(generated->second.first));
            effectiveAttrs.set(kBindingAttrName, moduleBuilder.getI64IntegerAttr(generated->second.second));
        }
        DictionaryAttr argumentAttrs = effectiveAttrs.getDictionary(source.getContext());
        InterfaceAttrs attrs = parseInterfaceAttrs(argumentAttrs);
        convertedType = applyInterfaceIntegerSignedness(convertedType, argumentAttrs);
        auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
        if (!kind)
            return source.emitError() << "argument #" << index << " has no Vernon interface kind";
        PhysicalAbiProfile profile = kind.getValue() == "uniform" && isa<TensorType>(type)
                                         ? PhysicalAbiProfile::VulkanStd430StorageBuffer
                                     : kind.getValue() == "uniform" && useVulkanInterfaceAbi && !attrs.binding
                                         ? PhysicalAbiProfile::VulkanPushConstant
                                         : PhysicalAbiProfile::VulkanStd140UniformBuffer;
        FailureOr<ByteTransportPlan> physicalLayout = getByteTransportPlan(type, sourceModule, profile);
        spirv::StorageClass storageClass = isa<TextureType>(type)         ? spirv::StorageClass::UniformConstant
                                           : kind.getValue() == "input"   ? spirv::StorageClass::Input
                                           : kind.getValue() == "uniform" ? (attrs.binding || useVulkanInterfaceAbi
                                                                                 ? spirv::StorageClass::Uniform
                                                                                 : spirv::StorageClass::UniformConstant)
                                                                          : spirv::StorageClass::StorageBuffer;
        if (useVulkanInterfaceAbi && kind.getValue() == "uniform" && !attrs.binding) {
            auto member = llvm::find_if(pushConstantMembers,
                                        [&](const PushConstantMember &value) { return value.argumentIndex == index; });
            if (member == pushConstantMembers.end())
                return source.emitError("push-constant member layout is inconsistent");
            inputs.push_back({pushConstantBlock});
            inputMembers.push_back(static_cast<uint32_t>(std::distance(pushConstantMembers.begin(), member)));
            inputAttributes.push_back(false);
            continue;
        }
        std::string fallback = (source.getSymName() + "_arg_" + Twine(index)).str();
        auto sourceName = source.getArgAttrOfType<StringAttr>(index, "vernon.source_name");
        std::string preferred = sanitizeInterfaceName(sourceName ? sourceName.getValue() : StringRef(), fallback,
                                                      kind.getValue() == "input");
        std::string name = uniqueInterfaceName(preferred, stage.getValue(), usedInterfaceNames);
        if (auto tensor = dyn_cast<TensorType>(type); tensor && kind.getValue() == "uniform") {
            FailureOr<Type> storageType = convertAggregateTensorStorageType(tensor, sourceModule);
            if (failed(storageType) || !attrs.binding)
                return source.emitError() << "aggregate Tensor uniform #" << index
                                          << " requires a descriptor-backed canonical storage layout";
            auto global = createInterfaceVariable(moduleBuilder, source.getLoc(), name, *storageType,
                                                  spirv::StorageClass::StorageBuffer, argumentAttrs,
                                                  succeeded(physicalLayout) ? physicalLayout->root.get() : nullptr);
            inputs.push_back({global});
            inputMembers.push_back(std::nullopt);
            inputAttributes.push_back(false);
            interfaceSymbols.push_back(SymbolRefAttr::get(source.getContext(), global.getSymName()));
            continue;
        }
        SmallVector<spirv::GlobalVariableOp> globals;
        auto location = dyn_cast_if_present<IntegerAttr>(attrs.location);
        const bool vertexAttribute =
            stage.getValue() == "vertex" && kind.getValue() == "input" && !attrs.builtin && location;
        if (vertexAttribute) {
            SmallVector<StringRef> logicalDtypes;
            if (auto dtypes = argumentAttrs.getAs<ArrayAttr>("vernon.abi_leaf_dtypes"))
                for (Attribute dtype : dtypes) {
                    auto value = dyn_cast<StringAttr>(dtype);
                    logicalDtypes.push_back(value ? value.getValue() : StringRef());
                }
            FailureOr<AttributeAbiLayout> plan = getAttributeAbiLayout(type, sourceModule, logicalDtypes);
            if (failed(plan))
                return source.emitError() << "vertex input #" << index << " has no numeric attribute layout";
            for (const AttributeAbiLeaf &leaf : plan->leaves) {
                FailureOr<Type> scalarType = convertValueType(leaf.scalarType, sourceModule);
                if (failed(scalarType))
                    return source.emitError() << "vertex input #" << index << " has an unsupported scalar leaf";
                if (auto integer = dyn_cast<IntegerType>(*scalarType);
                    integer && (leaf.dtype == "i32" || leaf.dtype == "u32")) {
                    const auto signedness = leaf.dtype == "i32" ? IntegerType::SignednessSemantics::Signed
                                                                : IntegerType::SignednessSemantics::Unsigned;
                    scalarType = IntegerType::get(source.getContext(), integer.getWidth(), signedness);
                }
                Type leafType =
                    leaf.componentCount == 1 ? *scalarType : Type(VectorType::get({leaf.componentCount}, *scalarType));
                NamedAttrList leafAttrs(argumentAttrs);
                leafAttrs.set(kLocationAttrName,
                              moduleBuilder.getI64IntegerAttr(location.getInt() + leaf.locationOffset));
                std::string leafName = name + "_leaf_" + std::to_string(leaf.locationOffset);
                globals.push_back(createInterfaceVariable(moduleBuilder, source.getLoc(), leafName, leafType,
                                                          storageClass, leafAttrs.getDictionary(source.getContext())));
            }
        } else {
            Type interfaceType = convertedType;
            if (auto tensor = dyn_cast<RankedTensorType>(type); tensor && tensor.getRank() > 2 &&
                                                                kind.getValue() == "uniform" && attrs.binding &&
                                                                useVulkanInterfaceAbi) {
                FailureOr<Type> physicalType = convertStd140TensorInterfaceType(tensor, sourceModule);
                if (failed(physicalType))
                    return source.emitError() << "cannot lower rank-" << tensor.getRank()
                                              << " std140 Tensor interface argument #" << index;
                interfaceType = *physicalType;
            }
            globals.push_back(createInterfaceVariable(
                moduleBuilder, source.getLoc(), name, interfaceType, storageClass, argumentAttrs,
                succeeded(physicalLayout) ? physicalLayout->root.get() : nullptr));
        }
        inputs.push_back(globals);
        inputMembers.push_back(std::nullopt);
        inputAttributes.push_back(vertexAttribute);
        for (spirv::GlobalVariableOp global : globals)
            interfaceSymbols.push_back(SymbolRefAttr::get(source.getContext(), global.getSymName()));
    }
    for (auto [index, type] : llvm::enumerate(source.getResultTypes())) {
        FailureOr<Type> converted = convertValueType(type, sourceModule);
        if (failed(converted))
            return source.emitError() << "cannot lower result #" << index << " type " << type << " to SPIR-V";
        std::string fallback = (source.getSymName() + "_result_" + Twine(index)).str();
        auto sourceName = source.getResultAttrOfType<StringAttr>(index, "vernon.source_name");
        std::string preferred = sanitizeInterfaceName(sourceName ? sourceName.getValue() : StringRef(), fallback,
                                                      stage.getValue() == "vertex");
        std::string name = uniqueInterfaceName(preferred, stage.getValue(), usedInterfaceNames);
        auto global = createInterfaceVariable(moduleBuilder, source.getLoc(), name, *converted,
                                              spirv::StorageClass::Output, source.getResultAttrDict(index));
        outputs.push_back(global);
        interfaceSymbols.push_back(SymbolRefAttr::get(source.getContext(), global.getSymName()));
    }

    auto functionType = moduleBuilder.getFunctionType({}, {});
    auto function = spirv::FuncOp::create(moduleBuilder, source.getLoc(), source.getSymName(), functionType);
    Block *entry = function.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
    IRMapping mapping;
    for (auto [argument, globals, member, attribute] :
         llvm::zip_equal(source.getArguments(), inputs, inputMembers, inputAttributes)) {
        if (globals.empty())
            continue;
        Value pointer = spirv::AddressOfOp::create(bodyBuilder, source.getLoc(), globals.front());
        auto pointerType = cast<spirv::PointerType>(pointer.getType());
        if (isa<TensorType>(argument.getType()) &&
            pointerType.getStorageClass() == spirv::StorageClass::StorageBuffer) {
            mapping.map(argument, pointer);
            continue;
        }
        if (isa<spirv::StructType>(pointerType.getPointeeType())) {
            Value memberIndex = spirv::ConstantOp::create(bodyBuilder, source.getLoc(), bodyBuilder.getI32Type(),
                                                          bodyBuilder.getI32IntegerAttr(member.value_or(0)));
            pointer = spirv::AccessChainOp::create(bodyBuilder, source.getLoc(), pointer, memberIndex);
        }
        if (!attribute) {
            Value loaded = spirv::LoadOp::create(bodyBuilder, source.getLoc(), pointer);
            FailureOr<Type> logicalType = convertValueType(argument.getType(), sourceModule);
            if (failed(logicalType))
                return source.emitError("cannot reconstruct logical interface value");
            if (loaded.getType() != *logicalType) {
                if (auto tensor = dyn_cast<RankedTensorType>(argument.getType()); tensor && tensor.getRank() > 2) {
                    FailureOr<Value> unpacked =
                        unpackStd140TensorInterface(source.getLoc(), loaded, tensor, *logicalType, bodyBuilder);
                    if (failed(unpacked))
                        return source.emitError("cannot unpack rank-three-or-higher std140 Tensor interface");
                    loaded = *unpacked;
                } else {
                    loaded = spirv::BitcastOp::create(bodyBuilder, source.getLoc(), *logicalType, loaded);
                }
            }
            mapping.map(argument, loaded);
            continue;
        }
        SmallVector<Value> leaves;
        for (spirv::GlobalVariableOp global : globals) {
            Value leafPointer = spirv::AddressOfOp::create(bodyBuilder, source.getLoc(), global);
            Value leaf = spirv::LoadOp::create(bodyBuilder, source.getLoc(), leafPointer);
            flattenComposite(source.getLoc(), leaf, bodyBuilder, leaves);
        }
        for (Value &leaf : leaves) {
            auto integer = dyn_cast<IntegerType>(leaf.getType());
            if (!integer || integer.isSignless())
                continue;
            Type logicalInteger = IntegerType::get(source.getContext(), integer.getWidth());
            leaf = spirv::BitcastOp::create(bodyBuilder, source.getLoc(), logicalInteger, leaf);
        }
        FailureOr<Type> converted = convertValueType(argument.getType(), sourceModule);
        FailureOr<Value> value = failed(converted)
                                     ? FailureOr<Value>(failure())
                                     : constructComposite(source.getLoc(), *converted, leaves, bodyBuilder);
        if (failed(value))
            return source.emitError("cannot reconstruct logical vertex attribute Tensor");
        mapping.map(argument, *value);
    }

    for (Operation &operation : source.front()) {
        if (auto returnOp = dyn_cast<func::ReturnOp>(operation)) {
            for (auto [value, global] : llvm::zip_equal(returnOp.getOperands(), outputs)) {
                Value pointer = spirv::AddressOfOp::create(bodyBuilder, source.getLoc(), global);
                spirv::StoreOp::create(bodyBuilder, source.getLoc(), pointer, mapping.lookup(value));
            }
            spirv::ReturnOp::create(bodyBuilder, source.getLoc());
            continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
            SmallVector<Value> translatedResults;
            if (failed(lowerIf(ifOp, bodyBuilder, entry, mapping, translatedResults)) ||
                translatedResults.size() != ifOp.getNumResults())
                return operation.emitError("structured conditional is not supported by SPIR-V lowering");
            for (auto [sourceResult, translatedResult] : llvm::zip_equal(ifOp.getResults(), translatedResults))
                mapping.map(sourceResult, translatedResult);
            continue;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
            SmallVector<Value> translatedResults;
            if (failed(lowerWhile(whileOp, bodyBuilder, entry, mapping, translatedResults)) ||
                translatedResults.size() != whileOp.getNumResults())
                return operation.emitError("structured loop is not supported by SPIR-V lowering");
            for (auto [sourceResult, translatedResult] : llvm::zip_equal(whileOp.getResults(), translatedResults))
                mapping.map(sourceResult, translatedResult);
            continue;
        }
        if (auto construct = dyn_cast<IntrinsicOp>(operation);
            construct && construct.getName() == "construct" && isa<TensorType>(construct.getResult().getType()) &&
            llvm::all_of(construct.getResult().getUsers(), [](Operation *user) {
                auto get = dyn_cast<TensorGetOp>(user);
                return get && llvm::all_of(get.getIndices(), [](Value sourceIndex) {
                           if (auto cast = sourceIndex.getDefiningOp<arith::IndexCastOp>())
                               sourceIndex = cast.getIn();
                           return static_cast<bool>(sourceIndex.getDefiningOp<arith::ConstantOp>());
                       });
            })) {
            // Every use is scalarized by TensorGet translation, so materializing
            // the full composite would only create backend-local aggregate arrays.
            continue;
        }
        FailureOr<Value> translated = translateOperation(operation, bodyBuilder, mapping);
        if (failed(translated)) {
            if (auto intrinsic = dyn_cast<IntrinsicOp>(operation))
                return operation.emitError()
                       << "intrinsic '" << intrinsic.getName() << "' is not supported by SPIR-V lowering";
            return operation.emitError() << "operation '" << operation.getName()
                                         << "' is not supported by SPIR-V lowering";
        }
        mapping.map(operation.getResult(0), *translated);
    }

    spirv::EntryPointOp::create(moduleBuilder, source.getLoc(), *executionModel, function, interfaceSymbols);
    if (*executionModel == spirv::ExecutionModel::Fragment)
        spirv::ExecutionModeOp::create(moduleBuilder, source.getLoc(), function, spirv::ExecutionMode::OriginUpperLeft,
                                       ArrayRef<int32_t>{});
    if (*executionModel == spirv::ExecutionModel::GLCompute) {
        auto workgroup = source->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName);
        spirv::ExecutionModeOp::create(moduleBuilder, source.getLoc(), function, spirv::ExecutionMode::LocalSize,
                                       workgroup.asArrayRef());
    }
    return success();
}

struct VernonToSPIRVPass : public PassWrapper<VernonToSPIRVPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonToSPIRVPass)

    VernonToSPIRVPass() = default;
    explicit VernonToSPIRVPass(bool useVulkanInterfaceAbi) : useVulkanInterfaceAbi(useVulkanInterfaceAbi) {}

    StringRef getArgument() const final { return "vernon-to-spirv"; }
    StringRef getDescription() const final { return "Lower Vernon graphics and Vulkan compute entries to SPIR-V"; }
    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<spirv::SPIRVDialect>(); }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SmallVector<func::FuncOp> entries;
        for (func::FuncOp function : module.getOps<func::FuncOp>()) {
            auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
            if (function->hasAttr(kEntryAttrName) && stage && stage.getValue() != "compute")
                entries.push_back(function);
        }
        if (entries.empty())
            return;

        OpBuilder builder(module.getContext());
        builder.setInsertionPointToEnd(module.getBody());
        auto target = spirv::ModuleOp::create(builder, module.getLoc(), spirv::AddressingModel::Logical,
                                              spirv::MemoryModel::GLSL450);
        target->setAttr(spirv::getTargetEnvAttrName(), spirv::getDefaultTargetEnv(module.getContext()));
        OpBuilder moduleBuilder = OpBuilder::atBlockBegin(target.getBody());
        llvm::StringSet<> usedInterfaceNames;
        DenseMap<Value, std::pair<uint32_t, uint32_t>> generatedBindings;
        uint32_t nextGeneratedBinding = 0;
        for (func::FuncOp function : module.getOps<func::FuncOp>()) {
            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                InterfaceAttrs attrs = parseInterfaceAttrs(function.getArgAttrDict(index));
                auto descriptorSet = dyn_cast_if_present<IntegerAttr>(attrs.descriptorSet);
                auto binding = dyn_cast_if_present<IntegerAttr>(attrs.binding);
                if (binding && (!descriptorSet || descriptorSet.getInt() == 0))
                    nextGeneratedBinding = std::max(nextGeneratedBinding, static_cast<uint32_t>(binding.getInt() + 1));
            }
        }
        for (func::FuncOp function : entries) {
            uint64_t inlineSize = 0;
            for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
                InterfaceAttrs attrs = parseInterfaceAttrs(function.getArgAttrDict(index));
                auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
                if (!kind || kind.getValue() != "uniform" || attrs.binding)
                    continue;
                if (isa<TensorType>(type)) {
                    generatedBindings[function.getArgument(index)] = std::make_pair(0u, nextGeneratedBinding++);
                    continue;
                }
                FailureOr<ByteTransportPlan> layout =
                    getByteTransportPlan(type, module, PhysicalAbiProfile::VulkanPushConstant);
                if (failed(layout)) {
                    function.emitError() << "cannot plan graphics inline layout for argument #" << index;
                    target.erase();
                    return signalPassFailure();
                }
                uint64_t offset = llvm::alignTo(inlineSize, layout->root->alignment);
                bool spill = (isa<RankedTensorType>(type) && cast<RankedTensorType>(type).getRank() > 2) ||
                             offset > 128 || layout->root->size > 128 - std::min<uint64_t>(offset, 128);
                if (spill) {
                    generatedBindings[function.getArgument(index)] = std::make_pair(0u, nextGeneratedBinding++);
                } else {
                    inlineSize = offset + layout->root->size;
                }
            }
        }
        int64_t expectedImageQueryCount = 0;

        for (func::FuncOp function : entries) {
            function.walk([&](IntrinsicOp intrinsic) {
                if (intrinsic.getName() == "texture_size")
                    ++expectedImageQueryCount;
            });
            if (failed(lowerEntry(function, target, moduleBuilder, usedInterfaceNames, useVulkanInterfaceAbi,
                                  generatedBindings))) {
                target.erase();
                return signalPassFailure();
            }
        }
        target->setAttr(kImageQueryExpectedCountAttr, builder.getI64IntegerAttr(expectedImageQueryCount));
    }

    bool useVulkanInterfaceAbi = true;
};

} // namespace

std::unique_ptr<Pass> createVernonToSPIRVPass(bool useVulkanInterfaceAbi) {
    return std::make_unique<VernonToSPIRVPass>(useVulkanInterfaceAbi);
}

void registerVernonToSPIRVPass() { PassRegistration<VernonToSPIRVPass>(); }

} // namespace mlir::vernon
