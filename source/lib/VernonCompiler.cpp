#include "VernonCompiler.h"

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <vector>

#if defined(VERNON_HAS_SPIRV_CROSS)
#include "spirv_glsl.hpp"
#include "spirv_msl.hpp"
#endif

struct VernonCompilerContext {
  mlir::MLIRContext context;

  VernonCompilerContext() {
    static std::once_flag initializeLLVM;
    std::call_once(initializeLLVM, [] {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      llvm::InitializeNativeTargetAsmParser();
    });
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::registerAllToLLVMIRTranslations(registry);
    registry.insert<mlir::vernon::VernonDialect>();
    context.appendDialectRegistry(registry);
  }
};

struct VernonCompileResult {
  struct Artifact {
    std::string name;
    std::string data;
  };

  VernonStatus status{VERNON_STATUS_INTERNAL_ERROR};
  std::string diagnostics;
  std::vector<Artifact> artifacts;
  std::string reflection;
  std::unique_ptr<llvm::orc::LLJIT> cpuJit;
  std::map<std::string, VernonCpuEntryPoint> cpuEntries;
};

namespace {

struct KeepGpuModulesPass
    : public mlir::PassWrapper<KeepGpuModulesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KeepGpuModulesPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    for (mlir::Operation &operation :
         llvm::make_early_inc_range(module.getBody()->without_terminator()))
      if (!mlir::isa<mlir::gpu::GPUModuleOp>(operation))
        operation.erase();
  }
};

struct AttachSpirvTargetPass
    : public mlir::PassWrapper<AttachSpirvTargetPass,
                               mlir::OperationPass<mlir::spirv::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachSpirvTargetPass)

  void runOnOperation() override {
    mlir::spirv::ModuleOp module = getOperation();
    auto triple = mlir::spirv::VerCapExtAttr::get(
        mlir::spirv::Version::V_1_3, {mlir::spirv::Capability::Shader},
        llvm::ArrayRef<mlir::spirv::Extension>(), module.getContext());
    module->setAttr(
        mlir::spirv::getTargetEnvAttrName(),
        mlir::spirv::TargetEnvAttr::get(
            triple, mlir::spirv::getDefaultResourceLimits(module.getContext()),
            mlir::spirv::ClientAPI::Vulkan, mlir::spirv::Vendor::Unknown,
            mlir::spirv::DeviceType::Unknown,
            mlir::spirv::TargetEnvAttr::kUnknownDeviceID));
  }
};

VernonStringView viewOf(const std::string &value) {
  return VernonStringView{value.data(), value.size()};
}

void appendDiagnostic(std::string &output, mlir::Diagnostic &diagnostic) {
  llvm::raw_string_ostream stream(output);
  if (!output.empty())
    stream << '\n';
  stream << diagnostic.getLocation() << ": " << diagnostic;
}

llvm::json::Value attributeToJson(mlir::Attribute attribute) {
  if (!attribute)
    return nullptr;
  if (auto string = mlir::dyn_cast<mlir::StringAttr>(attribute))
    return string.getValue().str();
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
    return integer.getInt();

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  attribute.print(stream);
  return printed;
}

std::string buildReflection(mlir::ModuleOp module) {
  auto cpuTypeSize = [](mlir::Type type) -> uint64_t {
    if (type.isIntOrFloat())
      return std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
    if (type.isIndex())
      return sizeof(uint64_t);
    if (mlir::isa<mlir::vernon::BufferType, mlir::vernon::TextureType,
                  mlir::vernon::SamplerType>(type))
      return sizeof(uintptr_t);
    if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      if (!tensor.hasStaticShape())
        return 0;
      uint64_t count = 1;
      for (int64_t dimension : tensor.getShape())
        count *= dimension;
      return count *
             std::max<uint64_t>(
                 tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
    }
    return 0;
  };
  llvm::json::Array entries;
  std::set<std::string> requiredFeatures;
  module.walk([&](mlir::func::FuncOp function) {
    auto stage = function->getAttrOfType<mlir::StringAttr>("vernon.stage");
    if (!stage)
      return;
    if (stage.getValue() == "compute")
      requiredFeatures.insert("compute");

    llvm::json::Array arguments;
    uint64_t argumentOffset = 0;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      llvm::json::Object argument;
      argument["index"] = static_cast<int64_t>(index);

      std::string type;
      llvm::raw_string_ostream typeStream(type);
      function.getArgumentTypes()[index].print(typeStream);
      argument["type"] = std::move(type);
      uint64_t size = cpuTypeSize(function.getArgumentTypes()[index]);
      if (size) {
        uint64_t alignment = size >= 16 ? 16 : size >= 8 ? 8 : 4;
        argumentOffset = llvm::alignTo(argumentOffset, alignment);
        argument["cpu_offset"] = static_cast<int64_t>(argumentOffset);
        argument["cpu_size"] = static_cast<int64_t>(size);
        argumentOffset += size;
      }

      if (auto attrs = function.getArgAttrDict(index)) {
        for (mlir::NamedAttribute attr : attrs) {
          argument[attr.getName().strref().str()] =
              attributeToJson(attr.getValue());
          if (attr.getName().strref() == "vernon.instance_divisor")
            requiredFeatures.insert("instancing");
        }
      }
      mlir::Type argumentType = function.getArgumentTypes()[index];
      if (mlir::isa<mlir::vernon::BufferType>(argumentType))
        requiredFeatures.insert("buffers");
      if (mlir::isa<mlir::vernon::TextureType>(argumentType))
        requiredFeatures.insert("textures");
      arguments.emplace_back(std::move(argument));
    }

    llvm::json::Array results;
    uint64_t resultSize = 0;
    for (unsigned index = 0; index < function.getNumResults(); ++index) {
      llvm::json::Object output;
      output["index"] = static_cast<int64_t>(index);

      std::string type;
      llvm::raw_string_ostream typeStream(type);
      function.getResultTypes()[index].print(typeStream);
      output["type"] = std::move(type);
      resultSize = cpuTypeSize(function.getResultTypes()[index]);
      if (resultSize) {
        output["cpu_offset"] = int64_t{0};
        output["cpu_size"] = static_cast<int64_t>(resultSize);
      }

      if (auto attrs = function.getResultAttrDict(index)) {
        for (mlir::NamedAttribute attr : attrs)
          output[attr.getName().strref().str()] =
              attributeToJson(attr.getValue());
      }
      results.emplace_back(std::move(output));
    }

    llvm::json::Object entry;
    entry["name"] = function.getSymName().str();
    entry["stage"] = stage.getValue().str();
    entry["arguments"] = std::move(arguments);
    entry["results"] = std::move(results);
    entry["cpu_arguments_size"] = static_cast<int64_t>(argumentOffset);
    entry["cpu_results_size"] = static_cast<int64_t>(resultSize);
    if (auto workgroup = function->getAttr("vernon.workgroup_size"))
      entry["workgroup_size"] = attributeToJson(workgroup);
    entries.emplace_back(std::move(entry));
  });

  llvm::json::Object root;
  root["schema_version"] = int64_t{1};
  root["entries"] = std::move(entries);
  llvm::json::Array features;
  for (const std::string &feature : requiredFeatures)
    features.emplace_back(feature);
  root["required_features"] = std::move(features);
  std::string canonicalModule;
  llvm::raw_string_ostream moduleStream(canonicalModule);
  module.print(moduleStream, mlir::OpPrintingFlags().enableDebugInfo(false));
  root["module_hash"] = llvm::utohexstr(llvm::xxHash64(canonicalModule));

  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << llvm::json::Value(std::move(root));
  return output;
}

std::unique_ptr<VernonCompileResult> validate(VernonCompilerContext *context,
                                              const char *source,
                                              size_t sourceSize) {
  auto result = std::make_unique<VernonCompileResult>();
  if (!context || (!source && sourceSize != 0)) {
    result->status = VERNON_STATUS_INVALID_ARGUMENT;
    result->diagnostics = "context and source must be valid";
    return result;
  }

  mlir::ScopedDiagnosticHandler handler(
      &context->context, [&](mlir::Diagnostic &diagnostic) {
        appendDiagnostic(result->diagnostics, diagnostic);
      });

  llvm::StringRef text(source ? source : "", sourceSize);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context->context);
  if (!module) {
    result->status = VERNON_STATUS_PARSE_ERROR;
    return result;
  }
  if (mlir::failed(mlir::verify(*module))) {
    result->status = VERNON_STATUS_VERIFICATION_ERROR;
    return result;
  }

  mlir::PassManager passManager(&context->context);
  passManager.addPass(mlir::vernon::createVernonValidatePass());
  if (mlir::failed(passManager.run(*module))) {
    result->status = VERNON_STATUS_VERIFICATION_ERROR;
    return result;
  }

  std::string canonicalModule;
  llvm::raw_string_ostream artifactStream(canonicalModule);
  module->print(artifactStream, mlir::OpPrintingFlags().enableDebugInfo(false));
  result->artifacts.push_back(
      VernonCompileResult::Artifact{"module.mlir", std::move(canonicalModule)});
  result->reflection = buildReflection(*module);
  result->status = VERNON_STATUS_OK;
  return result;
}

class CpuLLVMEmitter {
public:
  CpuLLVMEmitter(llvm::Module &module)
      : module(module), context(module.getContext()) {}

  llvm::Expected<std::string> emit(mlir::ModuleOp source) {
    for (mlir::func::FuncOp function : source.getOps<mlir::func::FuncOp>()) {
      if (!function->hasAttr("vernon.entry"))
        continue;
      if (mlir::failed(emitFunction(function)))
        return llvm::createStringError(
            "unsupported operation in CPU entry '%s'",
            function.getSymName().str().c_str());
      entryNames.push_back(function.getSymName().str());
    }
    if (entryNames.empty())
      return llvm::createStringError("module has no CPU entry points");
    std::string text;
    llvm::raw_string_ostream stream(text);
    module.print(stream, nullptr);
    return text;
  }

  llvm::ArrayRef<std::string> getEntryNames() const { return entryNames; }

private:
  llvm::Type *convertType(mlir::Type type) {
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
      return llvm::IntegerType::get(context, integer.getWidth());
    if (type.isIndex())
      return llvm::Type::getInt64Ty(context);
    if (type.isF16())
      return llvm::Type::getHalfTy(context);
    if (type.isF32())
      return llvm::Type::getFloatTy(context);
    if (type.isF64())
      return llvm::Type::getDoubleTy(context);
    if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      if (!tensor.hasStaticShape())
        return nullptr;
      int64_t count = 1;
      for (int64_t dimension : tensor.getShape())
        count *= dimension;
      llvm::Type *element = convertType(tensor.getElementType());
      return element && count > 1 ? llvm::FixedVectorType::get(element, count)
                                  : element;
    }
    if (mlir::isa<mlir::vernon::BufferType>(type))
      return llvm::PointerType::get(context, 0);
    if (mlir::isa<mlir::vernon::TextureType, mlir::vernon::SamplerType>(type))
      return llvm::Type::getInt64Ty(context);
    return nullptr;
  }

  llvm::Value *splat(llvm::IRBuilder<> &builder, llvm::Value *value,
                     unsigned count) {
    llvm::Value *result = llvm::PoisonValue::get(
        llvm::FixedVectorType::get(value->getType(), count));
    for (unsigned index = 0; index < count; ++index)
      result = builder.CreateInsertElement(result, value, index);
    return result;
  }

  llvm::Value *dot(llvm::IRBuilder<> &builder, llvm::Value *lhs,
                   llvm::Value *rhs) {
    auto vectorType = llvm::cast<llvm::FixedVectorType>(lhs->getType());
    llvm::Value *product = builder.CreateFMul(lhs, rhs);
    llvm::Value *sum = llvm::ConstantFP::get(vectorType->getElementType(), 0.0);
    for (unsigned index = 0; index < vectorType->getNumElements(); ++index)
      sum =
          builder.CreateFAdd(sum, builder.CreateExtractElement(product, index));
    return sum;
  }

  llvm::Value *
  emitIntrinsic(mlir::vernon::IntrinsicOp intrinsic, llvm::IRBuilder<> &builder,
                llvm::DenseMap<mlir::Value, llvm::Value *> &values) {
    llvm::SmallVector<llvm::Value *> operands;
    for (mlir::Value operand : intrinsic.getOperands()) {
      llvm::Value *value = values.lookup(operand);
      if (!value)
        return nullptr;
      operands.push_back(value);
    }
    llvm::StringRef name = intrinsic.getName();
    if (name == "buffer_load") {
      llvm::Type *elementType = convertType(intrinsic.getResult().getType());
      llvm::Value *address =
          builder.CreateGEP(elementType, operands[0], operands[1]);
      llvm::LoadInst *load = builder.CreateLoad(elementType, address);
      load->setAlignment(llvm::Align(1));
      return load;
    }
    if (name == "buffer_store") {
      llvm::Value *address =
          builder.CreateGEP(operands[2]->getType(), operands[0], operands[1]);
      llvm::StoreInst *store = builder.CreateStore(operands[2], address);
      store->setAlignment(llvm::Align(1));
      return store;
    }
    if (name == "texture_sample" && textureCallbacks) {
      llvm::Type *pointerType = llvm::PointerType::get(context, 0);
      llvm::Value *sampleAddress = builder.CreateGEP(
          llvm::Type::getInt8Ty(context), textureCallbacks,
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 8));
      llvm::LoadInst *sampleFunction =
          builder.CreateLoad(pointerType, sampleAddress);
      sampleFunction->setAlignment(llvm::Align(1));
      llvm::LoadInst *userData =
          builder.CreateLoad(pointerType, textureCallbacks);
      userData->setAlignment(llvm::Align(1));
      llvm::Value *output = builder.CreateAlloca(
          llvm::ArrayType::get(llvm::Type::getFloatTy(context), 4));
      auto callbackType = llvm::FunctionType::get(
          llvm::Type::getVoidTy(context),
          {pointerType, llvm::Type::getInt64Ty(context),
           llvm::Type::getFloatTy(context), llvm::Type::getFloatTy(context),
           pointerType},
          false);
      builder.CreateCall(
          callbackType, sampleFunction,
          {userData, operands[0],
           builder.CreateExtractElement(operands[2], uint64_t{0}),
           builder.CreateExtractElement(operands[2], uint64_t{1}), output});
      llvm::LoadInst *sample = builder.CreateLoad(
          llvm::FixedVectorType::get(llvm::Type::getFloatTy(context), 4),
          output);
      sample->setAlignment(llvm::Align(1));
      return sample;
    }
    if (intrinsic.getNumResults() != 1)
      return nullptr;
    llvm::Type *resultType = convertType(intrinsic.getResult().getType());
    if (name == "construct") {
      auto outputType = llvm::dyn_cast<llvm::FixedVectorType>(resultType);
      if (!outputType)
        return nullptr;
      llvm::Value *result = llvm::PoisonValue::get(outputType);
      unsigned outputIndex = 0;
      for (llvm::Value *operand : operands) {
        if (auto inputType =
                llvm::dyn_cast<llvm::FixedVectorType>(operand->getType())) {
          for (unsigned index = 0; index < inputType->getNumElements(); ++index)
            result = builder.CreateInsertElement(
                result, builder.CreateExtractElement(operand, index),
                outputIndex++);
        } else {
          result = builder.CreateInsertElement(result, operand, outputIndex++);
        }
      }
      return result;
    }
    if (name == "dot")
      return dot(builder, operands[0], operands[1]);
    if (name == "normalize") {
      llvm::Value *lengthSquared = dot(builder, operands[0], operands[0]);
      llvm::Function *sqrt = llvm::Intrinsic::getOrInsertDeclaration(
          &module, llvm::Intrinsic::sqrt, {lengthSquared->getType()});
      llvm::Value *length = builder.CreateCall(sqrt, lengthSquared);
      auto vectorType =
          llvm::cast<llvm::FixedVectorType>(operands[0]->getType());
      return builder.CreateFDiv(
          operands[0], splat(builder, length, vectorType->getNumElements()));
    }
    if (name == "cross") {
      llvm::Value *result = llvm::PoisonValue::get(operands[0]->getType());
      constexpr unsigned lhsIndices[] = {1, 2, 0};
      constexpr unsigned rhsIndices[] = {2, 0, 1};
      constexpr unsigned lhsIndices2[] = {2, 0, 1};
      constexpr unsigned rhsIndices2[] = {1, 2, 0};
      for (unsigned index = 0; index < 3; ++index) {
        llvm::Value *first = builder.CreateFMul(
            builder.CreateExtractElement(operands[0], lhsIndices[index]),
            builder.CreateExtractElement(operands[1], rhsIndices[index]));
        llvm::Value *second = builder.CreateFMul(
            builder.CreateExtractElement(operands[0], lhsIndices2[index]),
            builder.CreateExtractElement(operands[1], rhsIndices2[index]));
        result = builder.CreateInsertElement(
            result, builder.CreateFSub(first, second), index);
      }
      return result;
    }
    if (name == "reflect") {
      llvm::Value *factor = builder.CreateFMul(
          llvm::ConstantFP::get(
              dot(builder, operands[1], operands[0])->getType(), 2.0),
          dot(builder, operands[1], operands[0]));
      auto vectorType =
          llvm::cast<llvm::FixedVectorType>(operands[0]->getType());
      return builder.CreateFSub(
          operands[0],
          builder.CreateFMul(operands[1], splat(builder, factor,
                                                vectorType->getNumElements())));
    }
    if (name == "matmul")
      return emitMatrixMultiply(intrinsic, builder, operands);
    llvm::Intrinsic::ID id = llvm::Intrinsic::not_intrinsic;
    if (name == "min")
      id = llvm::Intrinsic::minnum;
    else if (name == "max")
      id = llvm::Intrinsic::maxnum;
    else if (name == "pow")
      id = llvm::Intrinsic::pow;
    if (id != llvm::Intrinsic::not_intrinsic) {
      llvm::Function *function =
          llvm::Intrinsic::getOrInsertDeclaration(&module, id, {resultType});
      return builder.CreateCall(function, operands);
    }
    if (name == "clamp") {
      llvm::Function *maximum = llvm::Intrinsic::getOrInsertDeclaration(
          &module, llvm::Intrinsic::maxnum, {resultType});
      llvm::Function *minimum = llvm::Intrinsic::getOrInsertDeclaration(
          &module, llvm::Intrinsic::minnum, {resultType});
      return builder.CreateCall(
          minimum, {builder.CreateCall(maximum, {operands[0], operands[1]}),
                    operands[2]});
    }
    return nullptr;
  }

  llvm::Value *emitMatrixMultiply(mlir::vernon::IntrinsicOp intrinsic,
                                  llvm::IRBuilder<> &builder,
                                  llvm::ArrayRef<llvm::Value *> operands) {
    auto leftType =
        mlir::cast<mlir::RankedTensorType>(intrinsic.getOperand(0).getType());
    int64_t rows = leftType.getShape()[0];
    int64_t columns = leftType.getShape()[1];
    auto rightType =
        mlir::cast<mlir::RankedTensorType>(intrinsic.getOperand(1).getType());
    int64_t resultColumns =
        rightType.getRank() == 1 ? 1 : rightType.getShape()[1];
    auto resultType = llvm::cast<llvm::FixedVectorType>(
        convertType(intrinsic.getResult().getType()));
    llvm::Value *result = llvm::PoisonValue::get(resultType);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t column = 0; column < resultColumns; ++column) {
        llvm::Value *sum =
            llvm::ConstantFP::get(resultType->getElementType(), 0.0);
        for (int64_t inner = 0; inner < columns; ++inner) {
          llvm::Value *lhs =
              builder.CreateExtractElement(operands[0], row * columns + inner);
          int64_t rhsIndex =
              rightType.getRank() == 1 ? inner : inner * resultColumns + column;
          llvm::Value *rhs =
              builder.CreateExtractElement(operands[1], rhsIndex);
          sum = builder.CreateFAdd(sum, builder.CreateFMul(lhs, rhs));
        }
        result = builder.CreateInsertElement(result, sum,
                                             row * resultColumns + column);
      }
    }
    return result;
  }

  mlir::LogicalResult emitFunction(mlir::func::FuncOp source) {
    llvm::SmallVector<llvm::Type *> argumentTypes;
    for (mlir::Type type : source.getArgumentTypes()) {
      llvm::Type *converted = convertType(type);
      if (!converted)
        return mlir::failure();
      argumentTypes.push_back(converted);
    }
    argumentTypes.push_back(llvm::PointerType::get(context, 0));
    llvm::Type *resultType = llvm::Type::getVoidTy(context);
    if (source.getNumResults() == 1) {
      resultType = convertType(source.getResultTypes()[0]);
      if (!resultType)
        return mlir::failure();
    } else if (source.getNumResults() > 1) {
      return mlir::failure();
    }
    auto functionType =
        llvm::FunctionType::get(resultType, argumentTypes, false);
    llvm::Function *function =
        llvm::Function::Create(functionType, llvm::GlobalValue::InternalLinkage,
                               source.getSymName().str(), module);
    llvm::BasicBlock *block =
        llvm::BasicBlock::Create(context, "entry", function);
    llvm::IRBuilder<> builder(block);
    llvm::DenseMap<mlir::Value, llvm::Value *> values;
    for (auto [index, sourceArgument] : llvm::enumerate(source.getArguments()))
      values[sourceArgument] = function->getArg(index);
    textureCallbacks = function->getArg(source.getNumArguments());
    bool requiresTextureCallbacks = false;

    for (mlir::Operation &operation : source.front()) {
      llvm::Value *result = nullptr;
      if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation)) {
        if (auto floatValue =
                mlir::dyn_cast<mlir::FloatAttr>(constant.getValue()))
          result = llvm::ConstantFP::get(context, floatValue.getValue());
        else if (auto integer =
                     mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
          result = llvm::ConstantInt::get(convertType(constant.getType()),
                                          integer.getValue());
        else
          return mlir::failure();
      } else if (auto swizzle =
                     mlir::dyn_cast<mlir::vernon::SwizzleOp>(operation)) {
        llvm::Value *input = values.lookup(swizzle.getInput());
        llvm::SmallVector<int> mask;
        for (char component : swizzle.getMask())
          mask.push_back(llvm::StringRef("xyzw").find(component));
        result = mask.size() == 1
                     ? builder.CreateExtractElement(input, mask.front())
                     : builder.CreateShuffleVector(input, mask);
      } else if (auto intrinsic =
                     mlir::dyn_cast<mlir::vernon::IntrinsicOp>(operation)) {
        requiresTextureCallbacks |= intrinsic.getName() == "texture_sample";
        result = emitIntrinsic(intrinsic, builder, values);
      } else if (auto returnOp =
                     mlir::dyn_cast<mlir::func::ReturnOp>(operation)) {
        if (returnOp.getNumOperands() == 0)
          builder.CreateRetVoid();
        else
          builder.CreateRet(values.lookup(returnOp.getOperand(0)));
        continue;
      } else if (operation.getNumOperands() == 2 &&
                 operation.getNumResults() == 1) {
        llvm::Value *lhs = values.lookup(operation.getOperand(0));
        llvm::Value *rhs = values.lookup(operation.getOperand(1));
        llvm::StringRef name = operation.getName().getStringRef();
        if (name == "arith.addf")
          result = builder.CreateFAdd(lhs, rhs);
        else if (name == "arith.subf")
          result = builder.CreateFSub(lhs, rhs);
        else if (name == "arith.mulf")
          result = builder.CreateFMul(lhs, rhs);
        else if (name == "arith.divf")
          result = builder.CreateFDiv(lhs, rhs);
        else if (name == "arith.addi")
          result = builder.CreateAdd(lhs, rhs);
        else if (name == "arith.subi")
          result = builder.CreateSub(lhs, rhs);
        else if (name == "arith.muli")
          result = builder.CreateMul(lhs, rhs);
      }
      if (!result)
        return mlir::failure();
      if (operation.getNumResults() == 0)
        continue;
      if (operation.getNumResults() != 1)
        return mlir::failure();
      values[operation.getResult(0)] = result;
    }
    if (!block->getTerminator())
      return mlir::failure();
    emitWrapper(function, source.getNumArguments(), requiresTextureCallbacks);
    textureCallbacks = nullptr;
    return mlir::success();
  }

  void emitWrapper(llvm::Function *function, unsigned sourceArgumentCount,
                   bool requiresTextureCallbacks) {
    llvm::Type *pointerType = llvm::PointerType::get(context, 0);
    auto wrapperType = llvm::FunctionType::get(llvm::Type::getInt32Ty(context),
                                               {pointerType}, false);
    llvm::Function *wrapper =
        llvm::Function::Create(wrapperType, llvm::GlobalValue::ExternalLinkage,
                               "__vernon_cpu_" + function->getName(), module);
    llvm::BasicBlock *entryBlock =
        llvm::BasicBlock::Create(context, "entry", wrapper);
    llvm::BasicBlock *sizeBlock =
        llvm::BasicBlock::Create(context, "check_sizes", wrapper);
    llvm::BasicBlock *callBlock =
        llvm::BasicBlock::Create(context, "call", wrapper);
    llvm::BasicBlock *invalidBlock =
        llvm::BasicBlock::Create(context, "invalid", wrapper);

    llvm::SmallVector<uint64_t> argumentOffsets;
    uint64_t requiredArgumentsSize = 0;
    for (unsigned index = 0; index < sourceArgumentCount; ++index) {
      llvm::Argument &argument = *function->getArg(index);
      uint64_t size =
          module.getDataLayout().getTypeStoreSize(argument.getType());
      uint64_t alignment = size >= 16 ? 16 : size >= 8 ? 8 : 4;
      requiredArgumentsSize = llvm::alignTo(requiredArgumentsSize, alignment);
      argumentOffsets.push_back(requiredArgumentsSize);
      requiredArgumentsSize += size;
    }
    uint64_t requiredResultsSize =
        function->getReturnType()->isVoidTy()
            ? 0
            : module.getDataLayout().getTypeStoreSize(
                  function->getReturnType());

    llvm::IRBuilder<> builder(entryBlock);
    llvm::Value *invocation = wrapper->getArg(0);
    builder.CreateCondBr(
        builder.CreateICmpEQ(invocation,
                             llvm::ConstantPointerNull::get(
                                 llvm::cast<llvm::PointerType>(pointerType))),
        invalidBlock, sizeBlock);

    builder.SetInsertPoint(sizeBlock);
    llvm::LoadInst *argumentsLoad =
        builder.CreateLoad(pointerType, invocation, "arguments");
    argumentsLoad->setAlignment(llvm::Align(1));
    llvm::Value *arguments = argumentsLoad;
    llvm::Value *argumentsSizeAddress = builder.CreateGEP(
        llvm::Type::getInt8Ty(context), invocation,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 8));
    llvm::LoadInst *argumentsSize = builder.CreateLoad(
        llvm::Type::getInt64Ty(context), argumentsSizeAddress);
    argumentsSize->setAlignment(llvm::Align(1));
    llvm::Value *resultsAddress = builder.CreateGEP(
        llvm::Type::getInt8Ty(context), invocation,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 16));
    llvm::LoadInst *resultsLoad =
        builder.CreateLoad(pointerType, resultsAddress, "results");
    resultsLoad->setAlignment(llvm::Align(1));
    llvm::Value *results = resultsLoad;
    llvm::Value *resultsSizeAddress = builder.CreateGEP(
        llvm::Type::getInt8Ty(context), invocation,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 24));
    llvm::LoadInst *resultsSize =
        builder.CreateLoad(llvm::Type::getInt64Ty(context), resultsSizeAddress);
    resultsSize->setAlignment(llvm::Align(1));
    llvm::Value *validArguments =
        requiredArgumentsSize == 0
            ? llvm::ConstantInt::getTrue(context)
            : builder.CreateAnd(
                  builder.CreateICmpNE(
                      arguments,
                      llvm::ConstantPointerNull::get(
                          llvm::cast<llvm::PointerType>(pointerType))),
                  builder.CreateICmpUGE(
                      argumentsSize,
                      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context),
                                             requiredArgumentsSize)));
    llvm::Value *validResults =
        requiredResultsSize == 0
            ? llvm::ConstantInt::getTrue(context)
            : builder.CreateAnd(
                  builder.CreateICmpNE(
                      results, llvm::ConstantPointerNull::get(
                                   llvm::cast<llvm::PointerType>(pointerType))),
                  builder.CreateICmpUGE(
                      resultsSize,
                      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context),
                                             requiredResultsSize)));
    llvm::Value *texturesAddress = builder.CreateGEP(
        llvm::Type::getInt8Ty(context), invocation,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 32));
    llvm::LoadInst *textures =
        builder.CreateLoad(pointerType, texturesAddress, "textures");
    textures->setAlignment(llvm::Align(1));
    llvm::Value *validTextures =
        requiresTextureCallbacks
            ? builder.CreateICmpNE(
                  textures, llvm::ConstantPointerNull::get(
                                llvm::cast<llvm::PointerType>(pointerType)))
            : llvm::ConstantInt::getTrue(context);
    builder.CreateCondBr(
        builder.CreateAnd(builder.CreateAnd(validArguments, validResults),
                          validTextures),
        callBlock, invalidBlock);

    builder.SetInsertPoint(callBlock);
    llvm::SmallVector<llvm::Value *> argumentsToCall;
    for (unsigned index = 0; index < sourceArgumentCount; ++index) {
      llvm::Argument &argument = *function->getArg(index);
      uint64_t offset = argumentOffsets[index];
      llvm::Value *address = builder.CreateGEP(
          llvm::Type::getInt8Ty(context), arguments,
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), offset));
      llvm::LoadInst *load = builder.CreateLoad(argument.getType(), address);
      load->setAlignment(llvm::Align(1));
      argumentsToCall.push_back(load);
    }
    argumentsToCall.push_back(textures);
    llvm::CallInst *call = builder.CreateCall(function, argumentsToCall);
    if (!function->getReturnType()->isVoidTy()) {
      llvm::StoreInst *store = builder.CreateStore(call, results);
      store->setAlignment(llvm::Align(1));
    }
    builder.CreateRet(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0));

    builder.SetInsertPoint(invalidBlock);
    builder.CreateRet(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1));
  }

  llvm::Module &module;
  llvm::LLVMContext &context;
  llvm::Value *textureCallbacks{nullptr};
  llvm::SmallVector<std::string> entryNames;
};

bool compileVulkan(VernonCompilerContext *context, const char *source,
                   size_t sourceSize, VernonCompileResult &result) {
  mlir::ScopedDiagnosticHandler handler(
      &context->context, [&](mlir::Diagnostic &diagnostic) {
        appendDiagnostic(result.diagnostics, diagnostic);
      });
  llvm::StringRef text(source ? source : "", sourceSize);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context->context);
  if (!module)
    return false;

  mlir::PassManager passManager(&context->context);
  passManager.addPass(mlir::vernon::createVernonValidatePass());
  passManager.addPass(mlir::vernon::createVernonToGPUPass(true));
  passManager.addPass(mlir::createConvertGPUToSPIRVPass());
  passManager.addPass(mlir::vernon::createVernonToSPIRVPass());
  passManager.addNestedPass<mlir::spirv::ModuleOp>(
      std::make_unique<AttachSpirvTargetPass>());
  passManager.addNestedPass<mlir::spirv::ModuleOp>(
      mlir::spirv::createSPIRVUpdateVCEPass());
  if (mlir::failed(passManager.run(*module)))
    return false;

  llvm::SmallVector<mlir::spirv::ModuleOp> spirvModules;
  module->walk([&](mlir::spirv::ModuleOp spirvModule) {
    spirvModules.push_back(spirvModule);
  });
  if (spirvModules.empty()) {
    result.diagnostics =
        "module has no graphics or compute entry points for Vulkan";
    return false;
  }

  result.artifacts.clear();
  for (auto [index, spirvModule] : llvm::enumerate(spirvModules)) {
    llvm::SmallVector<uint32_t> words;
    if (mlir::failed(mlir::spirv::serialize(spirvModule, words)))
      return false;
    std::string binary(reinterpret_cast<const char *>(words.data()),
                       words.size() * sizeof(uint32_t));
    std::string name = spirvModules.size() == 1
                           ? "module.spv"
                           : "module_" + std::to_string(index) + ".spv";
    result.artifacts.push_back(
        VernonCompileResult::Artifact{std::move(name), std::move(binary)});
  }
  return true;
}

bool compileCuda(VernonCompilerContext *context, const char *source,
                 size_t sourceSize, VernonCompileResult &result) {
  mlir::ScopedDiagnosticHandler handler(
      &context->context, [&](mlir::Diagnostic &diagnostic) {
        appendDiagnostic(result.diagnostics, diagnostic);
      });
  llvm::StringRef text(source ? source : "", sourceSize);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context->context);
  if (!module)
    return false;

  mlir::PassManager passManager(&context->context);
  passManager.addPass(mlir::vernon::createVernonValidatePass());
  passManager.addPass(mlir::vernon::createVernonToGPUPass());
  passManager.addPass(std::make_unique<KeepGpuModulesPass>());
  mlir::gpu::GPUToNVVMPipelineOptions options;
  options.cubinFormat = "isa";
  mlir::gpu::buildLowerToNVVMPassPipeline(passManager, options);
  if (mlir::failed(passManager.run(*module)))
    return false;

  result.artifacts.clear();
  module->walk([&](mlir::gpu::BinaryOp binary) {
    for (mlir::Attribute attribute : binary.getObjects()) {
      auto object = mlir::dyn_cast<mlir::gpu::ObjectAttr>(attribute);
      if (!object)
        continue;
      mlir::StringAttr data = object.getObject();
      result.artifacts.push_back(VernonCompileResult::Artifact{
          binary.getSymName().str() + ".ptx", data.getValue().str()});
    }
  });
  if (result.artifacts.empty()) {
    result.diagnostics = "NVVM pipeline produced no PTX object";
    return false;
  }
  return true;
}

bool compileCpu(VernonCompilerContext *compilerContext, const char *source,
                size_t sourceSize, VernonCompileResult &result) {
  llvm::StringRef text(source ? source : "", sourceSize);
  mlir::OwningOpRef<mlir::ModuleOp> sourceModule =
      mlir::parseSourceString<mlir::ModuleOp>(text, &compilerContext->context);
  if (!sourceModule)
    return false;

  auto targetMachine = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!targetMachine) {
    result.diagnostics = llvm::toString(targetMachine.takeError());
    return false;
  }
  auto dataLayout = targetMachine->getDefaultDataLayoutForTarget();
  if (!dataLayout) {
    result.diagnostics = llvm::toString(dataLayout.takeError());
    return false;
  }
  auto jit = llvm::orc::LLJITBuilder()
                 .setJITTargetMachineBuilder(std::move(*targetMachine))
                 .setDataLayout(*dataLayout)
                 .create();
  if (!jit) {
    result.diagnostics = llvm::toString(jit.takeError());
    return false;
  }

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = std::make_unique<llvm::Module>("vernon_cpu", *llvmContext);
  llvmModule->setDataLayout(*dataLayout);
  llvmModule->setTargetTriple((*jit)->getTargetTriple());
  CpuLLVMEmitter emitter(*llvmModule);
  llvm::Expected<std::string> llvmIR = emitter.emit(*sourceModule);
  if (!llvmIR) {
    result.diagnostics = llvm::toString(llvmIR.takeError());
    return false;
  }
  if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
    result.diagnostics = "generated CPU LLVM IR failed verification";
    return false;
  }
  std::vector<std::string> entryNames(emitter.getEntryNames().begin(),
                                      emitter.getEntryNames().end());
  if (llvm::Error error = (*jit)->addIRModule(llvm::orc::ThreadSafeModule(
          std::move(llvmModule), std::move(llvmContext)))) {
    result.diagnostics = llvm::toString(std::move(error));
    return false;
  }
  for (const std::string &entryName : entryNames) {
    auto symbol = (*jit)->lookup("__vernon_cpu_" + entryName);
    if (!symbol) {
      result.diagnostics = llvm::toString(symbol.takeError());
      return false;
    }
    result.cpuEntries.emplace(entryName, symbol->toPtr<VernonCpuEntryPoint>());
  }
  result.artifacts.clear();
  result.artifacts.push_back(
      VernonCompileResult::Artifact{"module.ll", std::move(*llvmIR)});
  result.cpuJit = std::move(*jit);
  return true;
}

#if defined(VERNON_HAS_SPIRV_CROSS)
llvm::StringRef stageSuffix(spv::ExecutionModel model) {
  switch (model) {
  case spv::ExecutionModelVertex:
    return "vert";
  case spv::ExecutionModelFragment:
    return "frag";
  case spv::ExecutionModelGLCompute:
    return "comp";
  default:
    return "stage";
  }
}

bool crossCompile(VernonCompileResult &result, VernonTarget target) {
  std::vector<VernonCompileResult::Artifact> translated;
  try {
    for (const VernonCompileResult::Artifact &artifact : result.artifacts) {
      if (artifact.data.size() % sizeof(uint32_t) != 0)
        return false;
      std::vector<uint32_t> words(artifact.data.size() / sizeof(uint32_t));
      std::memcpy(words.data(), artifact.data.data(), artifact.data.size());
      spirv_cross::Compiler probe(words);
      for (const spirv_cross::EntryPoint &entry :
           probe.get_entry_points_and_stages()) {
        std::string source;
        std::string extension;
        if (target == VERNON_TARGET_METAL) {
          spirv_cross::CompilerMSL compiler(words);
          compiler.set_entry_point(entry.name, entry.execution_model);
          source = compiler.compile();
          extension = "metal";
        } else {
          spirv_cross::CompilerGLSL compiler(words);
          compiler.set_entry_point(entry.name, entry.execution_model);
          spirv_cross::CompilerGLSL::Options options;
          options.es = target == VERNON_TARGET_OPENGL_ES;
          options.version = options.es ? 310 : 450;
          compiler.set_common_options(options);
          source = compiler.compile();
          extension = options.es ? "gles" : "glsl";
        }
        std::string name = entry.name + "." +
                           stageSuffix(entry.execution_model).str() + "." +
                           extension;
        translated.push_back(
            VernonCompileResult::Artifact{std::move(name), std::move(source)});
      }
    }
  } catch (const std::exception &exception) {
    result.diagnostics = exception.what();
    return false;
  }
  if (translated.empty()) {
    result.diagnostics = "SPIRV-Cross found no shader entry points";
    return false;
  }
  result.artifacts = std::move(translated);
  return true;
}
#endif

} // namespace

extern "C" {

VernonCompilerContext *vernonCompilerCreate(void) {
  return new (std::nothrow) VernonCompilerContext();
}

void vernonCompilerDestroy(VernonCompilerContext *context) { delete context; }

VernonTargetCapabilities
vernonCompilerGetTargetCapabilities(const VernonCompilerContext *context,
                                    VernonTarget target) {
  if (!context || target < VERNON_TARGET_CPU || target > VERNON_TARGET_CUDA)
    return VernonTargetCapabilities{0, 0, 0, 0};

  if (target == VERNON_TARGET_CPU)
    return VernonTargetCapabilities{1, 1, 1, 0};
  if (target == VERNON_TARGET_VULKAN)
    return VernonTargetCapabilities{1, 1, 1, 0};
  if (target == VERNON_TARGET_CUDA)
    return VernonTargetCapabilities{1, 0, 1, 0};
#if defined(VERNON_HAS_SPIRV_CROSS)
  if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES ||
      target == VERNON_TARGET_METAL)
    return VernonTargetCapabilities{1, 1, 1, 0};
#endif

  // Targets become available only when their complete lowering pipeline is
  // registered. This prevents callers from mistaking an IR-only path for a
  // usable backend.
  return VernonTargetCapabilities{0, 0, 0, 0};
}

VernonCompileResult *vernonCompilerValidateMlir(VernonCompilerContext *context,
                                                const char *source,
                                                size_t sourceSize) {
  return validate(context, source, sourceSize).release();
}

VernonCompileResult *vernonCompilerCompileMlir(VernonCompilerContext *context,
                                               const char *source,
                                               size_t sourceSize,
                                               VernonTarget target) {
  auto result = validate(context, source, sourceSize);
  if (result->status != VERNON_STATUS_OK)
    return result.release();

  if (target < VERNON_TARGET_CPU || target > VERNON_TARGET_CUDA) {
    result->status = VERNON_STATUS_INVALID_ARGUMENT;
    result->diagnostics = "unknown compilation target";
    result->artifacts.clear();
    return result.release();
  }

  if (target == VERNON_TARGET_CPU) {
    result->diagnostics.clear();
    if (compileCpu(context, source, sourceSize, *result)) {
      result->status = VERNON_STATUS_OK;
      return result.release();
    }
    result->status = VERNON_STATUS_INTERNAL_ERROR;
    result->artifacts.clear();
    return result.release();
  }

  if (target == VERNON_TARGET_VULKAN || target == VERNON_TARGET_OPENGL ||
      target == VERNON_TARGET_OPENGL_ES || target == VERNON_TARGET_METAL) {
    result->diagnostics.clear();
    if (compileVulkan(context, source, sourceSize, *result)) {
#if defined(VERNON_HAS_SPIRV_CROSS)
      if (target != VERNON_TARGET_VULKAN && !crossCompile(*result, target)) {
        result->status = VERNON_STATUS_INTERNAL_ERROR;
        result->artifacts.clear();
        return result.release();
      }
#else
      if (target != VERNON_TARGET_VULKAN) {
        result->status = VERNON_STATUS_UNSUPPORTED_TARGET;
        result->diagnostics = "SPIRV-Cross support was disabled at build time";
        result->artifacts.clear();
        return result.release();
      }
#endif
      result->status = VERNON_STATUS_OK;
      return result.release();
    }
    result->status = VERNON_STATUS_INTERNAL_ERROR;
    result->artifacts.clear();
    return result.release();
  }

  if (target == VERNON_TARGET_CUDA) {
    result->diagnostics.clear();
    if (compileCuda(context, source, sourceSize, *result)) {
      result->status = VERNON_STATUS_OK;
      return result.release();
    }
    result->status = VERNON_STATUS_INTERNAL_ERROR;
    result->artifacts.clear();
    return result.release();
  }

  result->status = VERNON_STATUS_UNSUPPORTED_TARGET;
  result->diagnostics =
      "the requested target lowering pipeline is not available";
  result->artifacts.clear();
  return result.release();
}

void vernonCompileResultDestroy(VernonCompileResult *result) { delete result; }

VernonStatus vernonCompileResultGetStatus(const VernonCompileResult *result) {
  return result ? result->status : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStringView
vernonCompileResultGetDiagnostics(const VernonCompileResult *result) {
  return result ? viewOf(result->diagnostics) : VernonStringView{nullptr, 0};
}

VernonStringView
vernonCompileResultGetArtifact(const VernonCompileResult *result) {
  return result && !result->artifacts.empty()
             ? viewOf(result->artifacts.front().data)
             : VernonStringView{nullptr, 0};
}

size_t vernonCompileResultGetArtifactCount(const VernonCompileResult *result) {
  return result ? result->artifacts.size() : 0;
}

VernonStringView
vernonCompileResultGetArtifactName(const VernonCompileResult *result,
                                   size_t index) {
  return result && index < result->artifacts.size()
             ? viewOf(result->artifacts[index].name)
             : VernonStringView{nullptr, 0};
}

VernonStringView
vernonCompileResultGetArtifactData(const VernonCompileResult *result,
                                   size_t index) {
  return result && index < result->artifacts.size()
             ? viewOf(result->artifacts[index].data)
             : VernonStringView{nullptr, 0};
}

VernonStringView
vernonCompileResultGetReflection(const VernonCompileResult *result) {
  return result ? viewOf(result->reflection) : VernonStringView{nullptr, 0};
}

VernonCpuEntryPoint
vernonCompileResultGetCpuEntry(const VernonCompileResult *result,
                               const char *entryName, size_t entryNameSize) {
  if (!result || result->status != VERNON_STATUS_OK ||
      (!entryName && entryNameSize != 0))
    return nullptr;
  std::string name(entryName ? entryName : "", entryNameSize);
  auto entry = result->cpuEntries.find(name);
  return entry == result->cpuEntries.end() ? nullptr : entry->second;
}

} // extern "C"
