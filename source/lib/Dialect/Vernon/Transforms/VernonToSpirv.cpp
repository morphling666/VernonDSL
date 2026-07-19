#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::vernon {
namespace {

FailureOr<Type> convertValueType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (!tensor.hasStaticShape())
      return failure();
    if (tensor.getRank() == 1)
      return VectorType::get(tensor.getShape(), tensor.getElementType());
    if (tensor.getRank() == 2) {
      auto columnType =
          VectorType::get({tensor.getShape()[0]}, tensor.getElementType());
      return spirv::MatrixType::get(columnType, tensor.getShape()[1]);
    }
    return failure();
  }
  if (type.isIntOrIndexOrFloat())
    return type.isIndex() ? Type(IndexType::get(type.getContext())) : type;
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
      .Case("vertex_index", spirv::BuiltIn::VertexIndex)
      .Case("instance_index", spirv::BuiltIn::InstanceIndex)
      .Case("global_invocation_id", spirv::BuiltIn::GlobalInvocationId)
      .Case("local_invocation_id", spirv::BuiltIn::LocalInvocationId)
      .Case("workgroup_id", spirv::BuiltIn::WorkgroupId)
      .Default(std::nullopt);
}

spirv::GlobalVariableOp
createInterfaceVariable(OpBuilder &builder, Location location, StringRef name,
                        Type valueType, spirv::StorageClass storageClass,
                        DictionaryAttr attributes) {
  auto pointerType = spirv::PointerType::get(valueType, storageClass);
  if (storageClass == spirv::StorageClass::Uniform ||
      storageClass == spirv::StorageClass::PushConstant ||
      storageClass == spirv::StorageClass::StorageBuffer) {
    SmallVector<spirv::StructType::MemberDecorationInfo> memberDecorations;
    if (auto matrix = dyn_cast<spirv::MatrixType>(valueType)) {
      uint32_t elementBytes =
          matrix.getElementType().getIntOrFloatBitWidth() / 8;
      uint32_t rowCount = matrix.getNumRows();
      uint32_t alignment =
          rowCount >= 3 ? 4 * elementBytes : rowCount * elementBytes;
      uint32_t stride = llvm::alignTo(rowCount * elementBytes, alignment);
      memberDecorations.emplace_back(0, spirv::Decoration::MatrixStride,
                                     builder.getI32IntegerAttr(stride));
      memberDecorations.emplace_back(0, spirv::Decoration::ColMajor,
                                     builder.getUnitAttr());
    }
    SmallVector<spirv::StructType::StructDecorationInfo> structDecorations;
    structDecorations.emplace_back(spirv::Decoration::Block,
                                   builder.getUnitAttr());
    auto blockType = spirv::StructType::get({valueType}, {0}, memberDecorations,
                                            structDecorations);
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
  return spirv::GlobalVariableOp::create(
      builder, location, pointerType, name, FlatSymbolRefAttr(), locationAttr,
      bindingAttr, setAttr, builtInAttr, spirv::LinkageAttributesAttr());
}

FailureOr<Value> translateOperation(Operation &operation, OpBuilder &builder,
                                    IRMapping &mapping) {
  Location location = operation.getLoc();
  auto mapped = [&](Value value) { return mapping.lookupOrNull(value); };

  if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    FailureOr<Type> type = convertValueType(constant.getType());
    if (failed(type))
      return failure();
    Attribute value = constant.getValue();
    if (auto dense = dyn_cast<DenseElementsAttr>(value))
      value = dense.reshape(cast<ShapedType>(*type));
    return spirv::ConstantOp::create(builder, location, *type, value)
        .getResult();
  }

  if (auto swizzle = dyn_cast<SwizzleOp>(operation)) {
    Value input = mapped(swizzle.getInput());
    if (!input)
      return failure();
    SmallVector<int32_t> components;
    for (char component : swizzle.getMask())
      components.push_back(StringRef("xyzw").find(component));
    FailureOr<Type> resultType =
        convertValueType(swizzle.getResult().getType());
    if (failed(resultType))
      return failure();
    if (components.size() == 1)
      return spirv::CompositeExtractOp::create(builder, location, input,
                                               components)
          .getResult();
    SmallVector<Attribute> componentAttrs;
    for (int32_t component : components)
      componentAttrs.push_back(builder.getI32IntegerAttr(component));
    return spirv::VectorShuffleOp::create(builder, location, *resultType, input,
                                          input,
                                          builder.getArrayAttr(componentAttrs))
        .getResult();
  }

  if (auto intrinsic = dyn_cast<IntrinsicOp>(operation)) {
    SmallVector<Value> operands;
    for (Value operand : intrinsic.getOperands()) {
      Value converted = mapped(operand);
      if (!converted)
        return failure();
      operands.push_back(converted);
    }
    FailureOr<Type> resultType =
        convertValueType(intrinsic.getResult().getType());
    if (failed(resultType))
      return failure();
    StringRef name = intrinsic.getName();
    if (name == "construct")
      return spirv::CompositeConstructOp::create(builder, location, *resultType,
                                                 operands)
          .getResult();
    if (name == "matmul") {
      if (isa<spirv::MatrixType>(operands[1].getType()))
        return spirv::MatrixTimesMatrixOp::create(
                   builder, location, *resultType, operands[0], operands[1])
            .getResult();
      return spirv::MatrixTimesVectorOp::create(builder, location, *resultType,
                                                operands[0], operands[1])
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
  }

  return failure();
}

LogicalResult lowerEntry(func::FuncOp source, spirv::ModuleOp target,
                         OpBuilder &moduleBuilder) {
  auto stage = source->getAttrOfType<StringAttr>(kStageAttrName);
  if (!stage)
    return success();
  std::optional<spirv::ExecutionModel> executionModel =
      parseExecutionModel(stage.getValue());
  if (!executionModel)
    return source.emitError("unsupported SPIR-V execution model");

  SmallVector<spirv::GlobalVariableOp> inputs;
  SmallVector<spirv::GlobalVariableOp> outputs;
  SmallVector<Attribute> interfaceSymbols;
  for (auto [index, type] : llvm::enumerate(source.getArgumentTypes())) {
    FailureOr<Type> converted = convertValueType(type);
    if (failed(converted))
      return source.emitError() << "cannot lower argument #" << index
                                << " type " << type << " to SPIR-V";
    InterfaceAttrs attrs = parseInterfaceAttrs(source.getArgAttrDict(index));
    auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
    if (!kind)
      return source.emitError()
             << "argument #" << index << " has no Vernon interface kind";
    spirv::StorageClass storageClass =
        kind.getValue() == "input" ? spirv::StorageClass::Input
        : kind.getValue() == "uniform"
            ? (attrs.binding ? spirv::StorageClass::Uniform
                             : spirv::StorageClass::PushConstant)
            : spirv::StorageClass::StorageBuffer;
    std::string name = (source.getSymName() + "_arg_" + Twine(index)).str();
    auto global = createInterfaceVariable(moduleBuilder, source.getLoc(), name,
                                          *converted, storageClass,
                                          source.getArgAttrDict(index));
    inputs.push_back(global);
    interfaceSymbols.push_back(
        SymbolRefAttr::get(source.getContext(), global.getSymName()));
  }
  for (auto [index, type] : llvm::enumerate(source.getResultTypes())) {
    FailureOr<Type> converted = convertValueType(type);
    if (failed(converted))
      return source.emitError() << "cannot lower result #" << index << " type "
                                << type << " to SPIR-V";
    std::string name = (source.getSymName() + "_result_" + Twine(index)).str();
    auto global = createInterfaceVariable(
        moduleBuilder, source.getLoc(), name, *converted,
        spirv::StorageClass::Output, source.getResultAttrDict(index));
    outputs.push_back(global);
    interfaceSymbols.push_back(
        SymbolRefAttr::get(source.getContext(), global.getSymName()));
  }

  auto functionType = moduleBuilder.getFunctionType({}, {});
  auto function = spirv::FuncOp::create(moduleBuilder, source.getLoc(),
                                        source.getSymName(), functionType);
  Block *entry = function.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  IRMapping mapping;
  for (auto [argument, global] :
       llvm::zip_equal(source.getArguments(), inputs)) {
    Value pointer =
        spirv::AddressOfOp::create(bodyBuilder, source.getLoc(), global);
    auto pointerType = cast<spirv::PointerType>(pointer.getType());
    if (isa<spirv::StructType>(pointerType.getPointeeType())) {
      Value zero = spirv::ConstantOp::getZero(bodyBuilder.getI32Type(),
                                              source.getLoc(), bodyBuilder);
      pointer = spirv::AccessChainOp::create(bodyBuilder, source.getLoc(),
                                             pointer, zero);
    }
    Value value = spirv::LoadOp::create(bodyBuilder, source.getLoc(), pointer);
    mapping.map(argument, value);
  }

  for (Operation &operation : source.front()) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(operation)) {
      for (auto [value, global] :
           llvm::zip_equal(returnOp.getOperands(), outputs)) {
        Value pointer =
            spirv::AddressOfOp::create(bodyBuilder, source.getLoc(), global);
        spirv::StoreOp::create(bodyBuilder, source.getLoc(), pointer,
                               mapping.lookup(value));
      }
      spirv::ReturnOp::create(bodyBuilder, source.getLoc());
      continue;
    }
    FailureOr<Value> translated =
        translateOperation(operation, bodyBuilder, mapping);
    if (failed(translated))
      return operation.emitError(
          "operation is not supported by SPIR-V lowering");
    mapping.map(operation.getResult(0), *translated);
  }

  spirv::EntryPointOp::create(moduleBuilder, source.getLoc(), *executionModel,
                              function, interfaceSymbols);
  if (*executionModel == spirv::ExecutionModel::Fragment)
    spirv::ExecutionModeOp::create(moduleBuilder, source.getLoc(), function,
                                   spirv::ExecutionMode::OriginUpperLeft,
                                   ArrayRef<int32_t>{});
  if (*executionModel == spirv::ExecutionModel::GLCompute) {
    auto workgroup =
        source->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName);
    spirv::ExecutionModeOp::create(moduleBuilder, source.getLoc(), function,
                                   spirv::ExecutionMode::LocalSize,
                                   workgroup.asArrayRef());
  }
  return success();
}

struct VernonToSPIRVPass
    : public PassWrapper<VernonToSPIRVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonToSPIRVPass)

  StringRef getArgument() const final { return "vernon-to-spirv"; }
  StringRef getDescription() const final {
    return "Lower Vernon graphics and Vulkan compute entries to SPIR-V";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> entries;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
      if (function->hasAttr(kEntryAttrName) && stage &&
          stage.getValue() != "compute")
        entries.push_back(function);
    }
    if (entries.empty())
      return;

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    auto target = spirv::ModuleOp::create(builder, module.getLoc(),
                                          spirv::AddressingModel::Logical,
                                          spirv::MemoryModel::GLSL450);
    target->setAttr(spirv::getTargetEnvAttrName(),
                    spirv::getDefaultTargetEnv(module.getContext()));
    OpBuilder moduleBuilder = OpBuilder::atBlockBegin(target.getBody());

    for (func::FuncOp function : entries) {
      if (failed(lowerEntry(function, target, moduleBuilder))) {
        target.erase();
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVernonToSPIRVPass() {
  return std::make_unique<VernonToSPIRVPass>();
}

void registerVernonToSPIRVPass() { PassRegistration<VernonToSPIRVPass>(); }

} // namespace mlir::vernon
