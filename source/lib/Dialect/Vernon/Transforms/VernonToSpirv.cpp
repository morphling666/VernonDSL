#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSpirvMarkers.h"

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
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <iterator>

namespace mlir::vernon {
namespace {

FailureOr<Type> convertValueType(Type type) {
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
        if (!tensor.hasStaticShape())
            return failure();
        if (tensor.getRank() == 1)
            return VectorType::get(tensor.getShape(), tensor.getElementType());
        if (tensor.getRank() == 2) {
            auto columnType = VectorType::get({tensor.getShape()[0]}, tensor.getElementType());
            return spirv::MatrixType::get(columnType, tensor.getShape()[1]);
        }
        return failure();
    }
    if (auto tuple = dyn_cast<TupleType>(type)) {
        SmallVector<Type> elements;
        for (Type element : tuple.getTypes()) {
            FailureOr<Type> converted = convertValueType(element);
            if (failed(converted))
                return failure();
            elements.push_back(*converted);
        }
        return spirv::StructType::get(elements);
    }
    if (type.isIntOrIndexOrFloat())
        return type.isIndex() ? Type(IntegerType::get(type.getContext(), 32)) : type;
    if (isa<VectorType>(type))
        return type;
    return failure();
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

struct VulkanLayout {
    uint32_t alignment;
    uint32_t size;
    std::optional<uint32_t> matrixStride;
};

FailureOr<VulkanLayout> getVulkanLayout(Type type) {
    if (type.isIntOrFloat()) {
        uint32_t bitWidth = type.getIntOrFloatBitWidth();
        if (bitWidth < 8 || bitWidth % 8 != 0)
            return failure();
        uint32_t size = bitWidth / 8;
        return VulkanLayout{size, size, std::nullopt};
    }
    if (auto vector = dyn_cast<VectorType>(type)) {
        if (vector.getRank() != 1 || vector.getNumElements() < 2 || vector.getNumElements() > 4 ||
            !vector.getElementType().isIntOrFloat())
            return failure();
        uint32_t bitWidth = vector.getElementType().getIntOrFloatBitWidth();
        if (bitWidth < 8 || bitWidth % 8 != 0)
            return failure();
        uint32_t elementSize = bitWidth / 8;
        uint32_t count = static_cast<uint32_t>(vector.getNumElements());
        uint32_t alignment = (count == 2 ? 2 : 4) * elementSize;
        return VulkanLayout{alignment, count * elementSize, std::nullopt};
    }
    if (auto matrix = dyn_cast<spirv::MatrixType>(type)) {
        FailureOr<VulkanLayout> columnLayout = getVulkanLayout(matrix.getColumnType());
        if (failed(columnLayout))
            return failure();
        uint32_t stride = llvm::alignTo(columnLayout->size, columnLayout->alignment);
        return VulkanLayout{columnLayout->alignment, stride * static_cast<uint32_t>(matrix.getNumColumns()), stride};
    }
    return failure();
}

void appendMemberDecorations(OpBuilder &builder, uint32_t member, Type type,
                             SmallVectorImpl<spirv::StructType::MemberDecorationInfo> &decorations) {
    if (auto layout = getVulkanLayout(type); succeeded(layout) && layout->matrixStride) {
        decorations.emplace_back(member, spirv::Decoration::MatrixStride,
                                 builder.getI32IntegerAttr(*layout->matrixStride));
        decorations.emplace_back(member, spirv::Decoration::ColMajor, builder.getUnitAttr());
    }
}

spirv::GlobalVariableOp createInterfaceVariable(OpBuilder &builder, Location location, StringRef name, Type valueType,
                                                spirv::StorageClass storageClass, DictionaryAttr attributes) {
    auto pointerType = spirv::PointerType::get(valueType, storageClass);
    if (storageClass == spirv::StorageClass::Uniform || storageClass == spirv::StorageClass::PushConstant ||
        storageClass == spirv::StorageClass::StorageBuffer) {
        SmallVector<spirv::StructType::MemberDecorationInfo> memberDecorations;
        appendMemberDecorations(builder, 0, valueType, memberDecorations);
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

FailureOr<Value> translateOperation(Operation &operation, OpBuilder &builder, IRMapping &mapping) {
    Location location = operation.getLoc();
    auto mapped = [&](Value value) { return mapping.lookupOrNull(value); };

    if (auto splat = dyn_cast<tensor::SplatOp>(operation)) {
        Value input = mapped(splat.getInput());
        FailureOr<Type> resultType = convertValueType(splat.getType());
        if (!input || failed(resultType))
            return failure();
        auto vectorType = dyn_cast<VectorType>(*resultType);
        if (!vectorType)
            return failure();
        SmallVector<Value> elements(vectorType.getNumElements(), input);
        return spirv::CompositeConstructOp::create(builder, location, *resultType, elements).getResult();
    }

    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        FailureOr<Type> type = convertValueType(constant.getType());
        if (failed(type))
            return failure();
        Attribute value = constant.getValue();
        if (auto dense = dyn_cast<DenseElementsAttr>(value))
            value = dense.reshape(cast<ShapedType>(*type));
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
        FailureOr<Type> resultType = convertValueType(swizzle.getResult().getType());
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
        FailureOr<Type> resultType = convertValueType(tuple.getResult().getType());
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
        FailureOr<Type> resultType = convertValueType(intrinsic.getResult().getType());
        if (failed(resultType))
            return failure();
        StringRef name = intrinsic.getName();
        if (name == "construct")
            return spirv::CompositeConstructOp::create(builder, location, *resultType, operands).getResult();
        if (name == "matmul") {
            if (isa<spirv::MatrixType>(operands[1].getType()))
                return spirv::MatrixTimesMatrixOp::create(builder, location, *resultType, operands[0], operands[1])
                    .getResult();
            return spirv::MatrixTimesVectorOp::create(builder, location, *resultType, operands[0], operands[1])
                .getResult();
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
                                      .Case(math::SinOp::getOperationName(), "spirv.GL.Sin")
                                      .Case(math::CosOp::getOperationName(), "spirv.GL.Cos")
                                      .Case(math::ExpOp::getOperationName(), "spirv.GL.Exp")
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

    if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        Value condition = mapped(select.getCondition());
        Value trueValue = mapped(select.getTrueValue());
        Value falseValue = mapped(select.getFalseValue());
        FailureOr<Type> resultType = convertValueType(select.getType());
        if (!condition || !trueValue || !falseValue || failed(resultType))
            return failure();
        return spirv::SelectOp::create(builder, location, *resultType, condition, trueValue, falseValue).getResult();
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
        if (name == arith::AddFOp::getOperationName())
            return spirv::FAddOp::create(builder, location, lhs, rhs).getResult();
        if (name == arith::SubFOp::getOperationName())
            return spirv::FSubOp::create(builder, location, lhs, rhs).getResult();
        if (name == arith::MulFOp::getOperationName())
            return spirv::FMulOp::create(builder, location, lhs, rhs).getResult();
        if (name == arith::DivFOp::getOperationName())
            return spirv::FDivOp::create(builder, location, lhs, rhs).getResult();
        if (name == arith::AddIOp::getOperationName())
            return spirv::IAddOp::create(builder, location, lhs, rhs).getResult();
        if (name == arith::SubIOp::getOperationName())
            return spirv::ISubOp::create(builder, location, lhs, rhs).getResult();
        if (name == arith::MulIOp::getOperationName())
            return spirv::IMulOp::create(builder, location, lhs, rhs).getResult();
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
        if (name == arith::XOrIOp::getOperationName())
            return spirv::BitwiseXorOp::create(builder, location, lhs, rhs).getResult();
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
    if (!condition || !ifOp.getThenRegion().hasOneBlock() ||
        (!ifOp.getElseRegion().empty() && !ifOp.getElseRegion().hasOneBlock()))
        return failure();

    SmallVector<Value> resultVariables;
    OpBuilder variableBuilder = OpBuilder::atBlockBegin(functionEntry);
    for (Type sourceType : ifOp.getResultTypes()) {
        FailureOr<Type> type = convertValueType(sourceType);
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

    SmallVector<Value> resultVariables;
    OpBuilder variableBuilder = OpBuilder::atBlockBegin(functionEntry);
    for (Type sourceType : whileOp.getResultTypes()) {
        FailureOr<Type> type = convertValueType(sourceType);
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
            FailureOr<Type> type = convertValueType(argument.getType());
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
            FailureOr<Type> type = convertValueType(argument.getType());
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
                         llvm::StringSet<> &usedInterfaceNames, bool aggregatePushConstants) {
    auto stage = source->getAttrOfType<StringAttr>(kStageAttrName);
    if (!stage)
        return success();
    std::optional<spirv::ExecutionModel> executionModel = parseExecutionModel(stage.getValue());
    if (!executionModel)
        return source.emitError("unsupported SPIR-V execution model");

    SmallVector<spirv::GlobalVariableOp> inputs;
    SmallVector<std::optional<uint32_t>> inputMembers;
    SmallVector<spirv::GlobalVariableOp> outputs;
    SmallVector<Attribute> interfaceSymbols;
    struct PushConstantMember {
        uint32_t argumentIndex;
        Type type;
        uint32_t offset;
    };
    SmallVector<PushConstantMember> pushConstantMembers;
    uint32_t pushConstantSize = 0;
    for (auto [index, type] : llvm::enumerate(source.getArgumentTypes())) {
        if (!aggregatePushConstants)
            break;
        InterfaceAttrs attrs = parseInterfaceAttrs(source.getArgAttrDict(index));
        auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
        if (!kind || kind.getValue() != "uniform" || attrs.binding)
            continue;
        FailureOr<Type> converted = convertValueType(type);
        if (failed(converted))
            return source.emitError() << "cannot lower push-constant argument #" << index << " type " << type
                                      << " to SPIR-V";
        FailureOr<VulkanLayout> layout = getVulkanLayout(*converted);
        if (failed(layout))
            return source.emitError() << "push-constant argument #" << index
                                      << " does not have a Vulkan scalar/vector/matrix layout";
        pushConstantSize = llvm::alignTo(pushConstantSize, layout->alignment);
        pushConstantMembers.push_back({static_cast<uint32_t>(index), *converted, pushConstantSize});
        pushConstantSize += layout->size;
    }
    spirv::GlobalVariableOp pushConstantBlock;
    if (!pushConstantMembers.empty()) {
        SmallVector<Type> memberTypes;
        SmallVector<uint32_t> memberOffsets;
        SmallVector<spirv::StructType::MemberDecorationInfo> memberDecorations;
        for (auto [member, value] : llvm::enumerate(pushConstantMembers)) {
            memberTypes.push_back(value.type);
            memberOffsets.push_back(value.offset);
            appendMemberDecorations(moduleBuilder, static_cast<uint32_t>(member), value.type, memberDecorations);
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
            inputs.push_back({});
            inputMembers.push_back(std::nullopt);
            continue;
        }
        FailureOr<Type> converted = convertValueType(type);
        if (failed(converted))
            return source.emitError() << "cannot lower argument #" << index << " type " << type << " to SPIR-V";
        InterfaceAttrs attrs = parseInterfaceAttrs(source.getArgAttrDict(index));
        auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
        if (!kind)
            return source.emitError() << "argument #" << index << " has no Vernon interface kind";
        spirv::StorageClass storageClass = isa<TextureType>(type)         ? spirv::StorageClass::UniformConstant
                                           : kind.getValue() == "input"   ? spirv::StorageClass::Input
                                           : kind.getValue() == "uniform" ? (attrs.binding || aggregatePushConstants
                                                                                 ? spirv::StorageClass::Uniform
                                                                                 : spirv::StorageClass::UniformConstant)
                                                                          : spirv::StorageClass::StorageBuffer;
        if (aggregatePushConstants && kind.getValue() == "uniform" && !attrs.binding) {
            auto member = llvm::find_if(pushConstantMembers,
                                        [&](const PushConstantMember &value) { return value.argumentIndex == index; });
            if (member == pushConstantMembers.end())
                return source.emitError("push-constant member layout is inconsistent");
            inputs.push_back(pushConstantBlock);
            inputMembers.push_back(static_cast<uint32_t>(std::distance(pushConstantMembers.begin(), member)));
            continue;
        }
        std::string fallback = (source.getSymName() + "_arg_" + Twine(index)).str();
        auto sourceName = source.getArgAttrOfType<StringAttr>(index, "vernon.source_name");
        std::string preferred = sanitizeInterfaceName(sourceName ? sourceName.getValue() : StringRef(), fallback,
                                                      kind.getValue() == "input");
        std::string name = uniqueInterfaceName(preferred, stage.getValue(), usedInterfaceNames);
        auto global = createInterfaceVariable(moduleBuilder, source.getLoc(), name, *converted, storageClass,
                                              source.getArgAttrDict(index));
        inputs.push_back(global);
        inputMembers.push_back(std::nullopt);
        interfaceSymbols.push_back(SymbolRefAttr::get(source.getContext(), global.getSymName()));
    }
    for (auto [index, type] : llvm::enumerate(source.getResultTypes())) {
        FailureOr<Type> converted = convertValueType(type);
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
    for (auto [argument, global, member] : llvm::zip_equal(source.getArguments(), inputs, inputMembers)) {
        if (!global)
            continue;
        Value pointer = spirv::AddressOfOp::create(bodyBuilder, source.getLoc(), global);
        auto pointerType = cast<spirv::PointerType>(pointer.getType());
        if (isa<spirv::StructType>(pointerType.getPointeeType())) {
            Value memberIndex = spirv::ConstantOp::create(bodyBuilder, source.getLoc(), bodyBuilder.getI32Type(),
                                                          bodyBuilder.getI32IntegerAttr(member.value_or(0)));
            pointer = spirv::AccessChainOp::create(bodyBuilder, source.getLoc(), pointer, memberIndex);
        }
        Value value = spirv::LoadOp::create(bodyBuilder, source.getLoc(), pointer);
        mapping.map(argument, value);
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
        FailureOr<Value> translated = translateOperation(operation, bodyBuilder, mapping);
        if (failed(translated))
            return operation.emitError("operation is not supported by SPIR-V lowering");
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
    explicit VernonToSPIRVPass(bool aggregatePushConstants) : aggregatePushConstants(aggregatePushConstants) {}

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
        int64_t expectedImageQueryCount = 0;

        for (func::FuncOp function : entries) {
            function.walk([&](IntrinsicOp intrinsic) {
                if (intrinsic.getName() == "texture_size")
                    ++expectedImageQueryCount;
            });
            if (failed(lowerEntry(function, target, moduleBuilder, usedInterfaceNames, aggregatePushConstants))) {
                target.erase();
                return signalPassFailure();
            }
        }
        target->setAttr(kImageQueryExpectedCountAttr, builder.getI64IntegerAttr(expectedImageQueryCount));
    }

    bool aggregatePushConstants = true;
};

} // namespace

std::unique_ptr<Pass> createVernonToSPIRVPass(bool aggregatePushConstants) {
    return std::make_unique<VernonToSPIRVPass>(aggregatePushConstants);
}

void registerVernonToSPIRVPass() { PassRegistration<VernonToSPIRVPass>(); }

} // namespace mlir::vernon
