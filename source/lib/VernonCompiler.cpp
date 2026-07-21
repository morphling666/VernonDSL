#include "VernonCompiler.h"
#include "VernonCpuAbiWrapper.h"

#include "lld/Common/Driver.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonCpuPipeline.h"
#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCUDAMath.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
LLD_HAS_DRIVER(coff)
#elif defined(__APPLE__)
LLD_HAS_DRIVER(macho)
#else
LLD_HAS_DRIVER(elf)
#endif

#if defined(VERNON_HAS_SPIRV_CROSS)
#include "spirv_glsl.hpp"
#include "spirv_msl.hpp"
#endif

struct VernonCompilerContext {
  mlir::MLIRContext context;

  VernonCompilerContext() {
    static std::once_flag initializeLLVM;
    std::call_once(initializeLLVM, [] {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmPrinters();
      llvm::InitializeAllAsmParsers();
    });
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::vernon::registerVernonCpuPipelineDialects(registry);
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

llvm::StringRef targetName(VernonTarget target) {
  switch (target) {
  case VERNON_TARGET_CPU:
    return "cpu";
  case VERNON_TARGET_OPENGL:
    return "opengl";
  case VERNON_TARGET_OPENGL_ES:
    return "opengles";
  case VERNON_TARGET_VULKAN:
    return "vulkan";
  case VERNON_TARGET_METAL:
    return "metal";
  case VERNON_TARGET_DIRECTX:
    return "directx";
  case VERNON_TARGET_CUDA:
    return "cuda";
  }
  return "unknown";
}

llvm::StringRef artifactFormat(llvm::StringRef filename) {
  llvm::StringRef extension = llvm::sys::path::extension(filename);
  if (extension == ".spv")
    return "spirv";
  if (extension == ".glsl")
    return "glsl";
  if (extension == ".gles")
    return "gles";
  if (extension == ".metal")
    return "msl";
  if (extension == ".ptx")
    return "ptx";
  if (extension == ".ll")
    return "llvm_ir";
  if (extension == ".o" || extension == ".obj")
    return "relocatable_object";
  return "unknown";
}

void addArtifactTable(VernonCompileResult &result, VernonTarget target,
                      uint32_t glslVersion,
                      llvm::StringRef cpuTargetTriple = {},
                      llvm::StringRef cpu = {},
                      llvm::StringRef cpuFeatures = {}) {
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse(result.reflection);
  if (!parsed)
    return;
  llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return;

  (*root)["schema_version"] = int64_t{2};
  (*root)["target"] = targetName(target).str();
  llvm::json::Object targetOptions;
  if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES)
    targetOptions["glsl_version"] = static_cast<int64_t>(glslVersion);
  if (target == VERNON_TARGET_CPU) {
    targetOptions["target_triple"] = llvm::Triple::normalize(
        cpuTargetTriple.empty() ? llvm::sys::getDefaultTargetTriple()
                                : cpuTargetTriple);
    if (!cpu.empty())
      targetOptions["cpu"] = cpu.str();
    if (!cpuFeatures.empty())
      targetOptions["cpu_features"] = cpuFeatures.str();
  }
  (*root)["target_options"] = std::move(targetOptions);

  llvm::json::Array table;
  llvm::json::Array *entries = root->getArray("entries");
  if (entries) {
    for (auto [artifactIndex, artifact] : llvm::enumerate(result.artifacts)) {
      llvm::StringRef artifactName = artifact.name;
      for (auto [entryIndex, entryValue] : llvm::enumerate(*entries)) {
        llvm::json::Object *entry = entryValue.getAsObject();
        if (!entry)
          continue;
        std::optional<llvm::StringRef> entryName = entry->getString("name");
        std::optional<llvm::StringRef> stage = entry->getString("stage");
        if (!entryName || !stage)
          continue;

        // Cross-compiled artifacts carry the entry name. Binary module
        // artifacts are emitted in the same deterministic order as entries;
        // a single module may contain every entry.
        const bool namedArtifact =
            artifactName.starts_with(*entryName) &&
            artifactName.drop_front(entryName->size()).starts_with(".");
        const bool sharedArtifact = result.artifacts.size() == 1;
        const bool parallelArtifact =
            result.artifacts.size() == entries->size() &&
            artifactIndex == entryIndex;
        if (!namedArtifact && !sharedArtifact && !parallelArtifact)
          continue;

        llvm::json::Object row;
        row["entry_point"] = entryName->str();
        row["stage"] = stage->str();
        row["target"] = targetName(target).str();
        row["format"] = artifactFormat(artifactName).str();
        row["filename"] = artifact.name;
        table.emplace_back(std::move(row));
      }
    }
  }
  (*root)["artifacts"] = std::move(table);

  result.reflection.clear();
  llvm::raw_string_ostream stream(result.reflection);
  stream << llvm::json::Value(std::move(*root));
}

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

uint64_t cpuSourceTypeSize(mlir::Type type) {
  if (type.isIntOrFloat())
    return std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
  if (type.isIndex())
    return sizeof(uint64_t);
  if (mlir::isa<mlir::vernon::BufferType, mlir::vernon::TextureType,
                mlir::vernon::SamplerType>(type))
    return sizeof(uintptr_t);
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return 0;
  uint64_t count = 1;
  for (int64_t dimension : tensor.getShape())
    count *= static_cast<uint64_t>(dimension);
  return count * std::max<uint64_t>(
                     tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
}

enum class ResourceKind { None, Texture, Sampler };

ResourceKind resourceKind(mlir::Type type) {
  if (mlir::isa<mlir::vernon::TextureType>(type))
    return ResourceKind::Texture;
  if (mlir::isa<mlir::vernon::SamplerType>(type))
    return ResourceKind::Sampler;
  return ResourceKind::None;
}

struct SampledTextureBinding {
  int64_t descriptorSet;
  int64_t binding;

  bool operator<(const SampledTextureBinding &other) const {
    return std::tie(descriptorSet, binding) <
           std::tie(other.descriptorSet, other.binding);
  }
};

using ResourceOrigins = std::set<unsigned>;
using ResourceProvenance =
    llvm::DenseMap<mlir::Value, ResourceOrigins>;

mlir::FailureOr<std::map<unsigned, std::set<SampledTextureBinding>>>
analyzeSampledTextureBindings(mlir::func::FuncOp function) {
  struct ForwardingEdge {
    mlir::Value source;
    mlir::Value destination;
  };

  ResourceProvenance provenance;
  for (auto [index, argument] : llvm::enumerate(function.getArguments()))
    if (resourceKind(argument.getType()) != ResourceKind::None)
      provenance[argument].insert(index);

  llvm::SmallVector<ForwardingEdge> edges;
  auto addEdge = [&](mlir::Value source, mlir::Value destination) {
    ResourceKind sourceKind = resourceKind(source.getType());
    if (sourceKind != ResourceKind::None &&
        sourceKind == resourceKind(destination.getType()))
      edges.push_back({source, destination});
  };

  function.walk([&](mlir::Operation *operation) {
    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(operation)) {
      addEdge(select.getTrueValue(), select.getResult());
      addEdge(select.getFalseValue(), select.getResult());
    } else if (mlir::isa<mlir::UnrealizedConversionCastOp>(operation) &&
               operation->getNumOperands() == 1 &&
               operation->getNumResults() == 1) {
      addEdge(operation->getOperand(0), operation->getResult(0));
    } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
      auto thenYield =
          mlir::cast<mlir::scf::YieldOp>(ifOp.thenBlock()->getTerminator());
      auto elseYield =
          mlir::cast<mlir::scf::YieldOp>(ifOp.elseBlock()->getTerminator());
      for (auto [result, thenValue, elseValue] :
           llvm::zip_equal(ifOp.getResults(), thenYield.getOperands(),
                           elseYield.getOperands())) {
        addEdge(thenValue, result);
        addEdge(elseValue, result);
      }
    } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      for (auto [init, iterArgument, yielded, result] :
           llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs(),
                           forOp.getYieldedValues(), forOp.getResults())) {
        addEdge(init, iterArgument);
        addEdge(yielded, iterArgument);
        addEdge(init, result);
        addEdge(yielded, result);
      }
    } else if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(operation)) {
      auto condition = mlir::cast<mlir::scf::ConditionOp>(
          whileOp.getBefore().front().getTerminator());
      auto yield = mlir::cast<mlir::scf::YieldOp>(
          whileOp.getAfter().front().getTerminator());
      for (auto [init, beforeArgument, conditionValue, afterArgument, yielded,
                 result] :
           llvm::zip_equal(whileOp.getInits(), whileOp.getBeforeArguments(),
                           condition.getArgs(), whileOp.getAfterArguments(),
                           yield.getOperands(), whileOp.getResults())) {
        addEdge(init, beforeArgument);
        addEdge(yielded, beforeArgument);
        addEdge(conditionValue, afterArgument);
        addEdge(conditionValue, result);
      }
    }

    auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(operation);
    if (!branch)
      return;
    for (auto [successorIndex, successor] :
         llvm::enumerate(operation->getSuccessors())) {
      mlir::SuccessorOperands successorOperands =
          branch.getSuccessorOperands(successorIndex);
      for (unsigned argumentIndex =
               successorOperands.getProducedOperandCount();
           argumentIndex < successorOperands.size() &&
           argumentIndex < successor->getNumArguments();
           ++argumentIndex)
        addEdge(successorOperands[argumentIndex],
                successor->getArgument(argumentIndex));
    }
  });

  bool changed = true;
  while (changed) {
    changed = false;
    for (const ForwardingEdge &edge : edges) {
      const ResourceOrigins &source = provenance[edge.source];
      ResourceOrigins &destination = provenance[edge.destination];
      size_t previousSize = destination.size();
      destination.insert(source.begin(), source.end());
      changed |= destination.size() != previousSize;
    }
  }

  std::map<unsigned, std::set<SampledTextureBinding>> samplerBindings;
  std::map<SampledTextureBinding, std::set<unsigned>> bindingSamplers;
  bool invalid = false;
  function.walk([&](mlir::vernon::IntrinsicOp intrinsic) {
    if (intrinsic.getName() != "texture_sample")
      return;
    const ResourceOrigins &textureOrigins =
        provenance[intrinsic.getOperand(0)];
    const ResourceOrigins &samplerOrigins =
        provenance[intrinsic.getOperand(1)];
    if (textureOrigins.empty() || samplerOrigins.empty()) {
      intrinsic.emitError()
          << "cannot resolve texture_sample "
          << (textureOrigins.empty() && samplerOrigins.empty()
                  ? "texture and sampler"
                  : textureOrigins.empty() ? "texture" : "sampler")
          << " provenance to entry arguments";
      invalid = true;
      return;
    }

    for (unsigned textureOrigin : textureOrigins) {
      if (textureOrigin >= function.getNumArguments() ||
          resourceKind(function.getArgumentTypes()[textureOrigin]) !=
              ResourceKind::Texture) {
        intrinsic.emitError()
            << "texture_sample texture provenance includes non-texture entry "
               "argument #"
            << textureOrigin;
        invalid = true;
        continue;
      }
      auto attrs = function.getArgAttrDict(textureOrigin);
      auto descriptorSet = attrs.getAs<mlir::IntegerAttr>("vernon.set");
      auto binding = attrs.getAs<mlir::IntegerAttr>("vernon.binding");
      if (!descriptorSet || !binding) {
        intrinsic.emitError()
            << "texture_sample texture entry argument #" << textureOrigin
            << " has no descriptor set/binding";
        invalid = true;
        continue;
      }
      SampledTextureBinding sampledBinding{descriptorSet.getInt(),
                                           binding.getInt()};
      for (unsigned samplerOrigin : samplerOrigins) {
        if (samplerOrigin >= function.getNumArguments() ||
            resourceKind(function.getArgumentTypes()[samplerOrigin]) !=
                ResourceKind::Sampler) {
          intrinsic.emitError()
              << "texture_sample sampler provenance includes non-sampler "
                 "entry argument #"
              << samplerOrigin;
          invalid = true;
          continue;
        }
        samplerBindings[samplerOrigin].insert(sampledBinding);
        bindingSamplers[sampledBinding].insert(samplerOrigin);
      }
    }
  });
  if (invalid)
    return mlir::failure();

  for (const auto &[binding, samplers] : bindingSamplers) {
    if (samplers.size() <= 1)
      continue;
    std::string samplerList;
    llvm::raw_string_ostream stream(samplerList);
    llvm::interleaveComma(samplers, stream,
                          [&](unsigned sampler) { stream << '#' << sampler; });
    function.emitError()
        << "sampled texture at set " << binding.descriptorSet << ", binding "
        << binding.binding << " may use multiple sampler entry arguments ("
        << samplerList
        << "); the current combined-image SPIR-V model requires one sampler "
           "origin per sampled texture binding";
    return mlir::failure();
  }
  return samplerBindings;
}

mlir::FailureOr<std::string>
buildReflection(mlir::ModuleOp module, mlir::ModuleOp sourceModule) {
  auto scalarDtype = [](mlir::Type type) -> std::string {
    if (type.isF16())
      return "f16";
    if (type.isF32())
      return "f32";
    if (type.isF64())
      return "f64";
    if (type.isInteger(1))
      return "bool";
    if (type.isInteger(32))
      return "u32";
    if (type.isIndex())
      return "index";
    return "";
  };
  llvm::json::Array entries;
  std::set<std::string> requiredFeatures;
  bool invalid = false;
  module.walk([&](mlir::func::FuncOp function) {
    auto stage = function->getAttrOfType<mlir::StringAttr>("vernon.stage");
    if (!stage)
      return;
    if (stage.getValue() == "compute")
      requiredFeatures.insert("compute");

    mlir::FailureOr<std::map<unsigned, std::set<SampledTextureBinding>>>
        sampledBindings = analyzeSampledTextureBindings(function);
    if (mlir::failed(sampledBindings)) {
      invalid = true;
      return;
    }

    llvm::json::Array arguments;
    uint64_t argumentOffset = 0;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      llvm::json::Object argument;
      argument["index"] = static_cast<int64_t>(index);

      std::string type;
      llvm::raw_string_ostream typeStream(type);
      function.getArgumentTypes()[index].print(typeStream);
      argument["type"] = std::move(type);
      uint64_t size = cpuSourceTypeSize(function.getArgumentTypes()[index]);
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
      if (auto texture =
              mlir::dyn_cast<mlir::vernon::TextureType>(argumentType)) {
        argument["kind"] = "texture";
        argument["dtype"] = scalarDtype(texture.getElementType());
        argument["dimension"] = texture.getDimension().str();
        argument["access"] = "read";
      } else if (mlir::isa<mlir::vernon::SamplerType>(argumentType)) {
        argument["kind"] = "sampler";
        argument["access"] = "read";
        auto pairs = sampledBindings->find(index);
        if (pairs != sampledBindings->end()) {
          llvm::json::Array bindings;
          for (const SampledTextureBinding &pair : pairs->second) {
            llvm::json::Object reflectedBinding;
            reflectedBinding["set"] = pair.descriptorSet;
            reflectedBinding["binding"] = pair.binding;
            bindings.emplace_back(std::move(reflectedBinding));
          }
          argument["sampled_texture_bindings"] = std::move(bindings);
          if (pairs->second.size() == 1) {
            argument["sampled_texture_set"] =
                pairs->second.begin()->descriptorSet;
            argument["sampled_texture_binding"] =
                pairs->second.begin()->binding;
          }
        }
      }
      if (stage.getValue() == "compute") {
        auto attrs = function.getArgAttrDict(index);
        auto builtin = attrs.getAs<mlir::StringAttr>("vernon.builtin");
        auto sourceDtype = attrs.getAs<mlir::StringAttr>("vernon.dtype");
        if (builtin) {
          argument["kind"] = "builtin";
          argument["builtin"] = builtin.getValue().str();
          if (sourceDtype)
            argument["dtype"] = sourceDtype.getValue().str();
        } else if (mlir::isa<mlir::vernon::TextureType,
                             mlir::vernon::SamplerType>(argumentType)) {
          // Resource kind and binding metadata are emitted above.
        } else if (auto buffer =
                       mlir::dyn_cast<mlir::vernon::BufferType>(argumentType)) {
          argument["kind"] = "tensor";
          argument["cuda_abi"] = "strided_memref_1d";
          argument["dtype"] = sourceDtype
                                  ? sourceDtype.getValue().str()
                                  : scalarDtype(buffer.getElementType());
          argument["access"] = buffer.getAccess().str();
          uint64_t elementSize = std::max<uint64_t>(
              buffer.getElementType().getIntOrFloatBitWidth() / 8, 1);
          argument["alignment"] = static_cast<int64_t>(elementSize);
          if (auto shape =
                  attrs.getAs<mlir::DenseI64ArrayAttr>("vernon.tensor_shape")) {
            llvm::json::Array dimensions;
            llvm::json::Array strides;
            int64_t stride = 1;
            llvm::SmallVector<int64_t> reversedStrides(shape.size());
            for (int64_t dimensionIndex =
                     static_cast<int64_t>(shape.size()) - 1;
                 dimensionIndex >= 0; --dimensionIndex) {
              reversedStrides[dimensionIndex] = stride;
              stride *= shape[dimensionIndex];
            }
            for (auto [dimension, tensorStride] :
                 llvm::zip_equal(shape.asArrayRef(), reversedStrides)) {
              dimensions.emplace_back(dimension);
              strides.emplace_back(tensorStride);
            }
            argument["rank"] = static_cast<int64_t>(shape.size());
            argument["shape"] = std::move(dimensions);
            argument["strides"] = std::move(strides);
          }
        } else {
          argument["kind"] = "scalar";
          argument["dtype"] = sourceDtype ? sourceDtype.getValue().str()
                                          : scalarDtype(argumentType);
          argument["alignment"] = static_cast<int64_t>(
              std::max<uint64_t>(cpuSourceTypeSize(argumentType), 1));
        }
      }
      if (mlir::isa<mlir::vernon::BufferType>(argumentType))
        requiredFeatures.insert("buffers");
      if (mlir::isa<mlir::vernon::TextureType>(argumentType))
        requiredFeatures.insert("textures");
      if (mlir::isa<mlir::vernon::SamplerType>(argumentType))
        requiredFeatures.insert("samplers");
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
      resultSize = cpuSourceTypeSize(function.getResultTypes()[index]);
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
    entry["symbol"] = function.getSymName().str();
    entry["stage"] = stage.getValue().str();
    entry["arguments"] = std::move(arguments);
    entry["results"] = std::move(results);
    entry["cpu_arguments_size"] = static_cast<int64_t>(argumentOffset);
    entry["cpu_results_size"] = static_cast<int64_t>(resultSize);
    if (auto workgroup = function->getAttrOfType<mlir::DenseI32ArrayAttr>(
            "vernon.workgroup_size")) {
      llvm::json::Array dimensions;
      for (int32_t dimension : workgroup.asArrayRef())
        dimensions.emplace_back(static_cast<int64_t>(dimension));
      entry["workgroup_size"] = std::move(dimensions);
    }
    entries.emplace_back(std::move(entry));
  });
  if (invalid)
    return mlir::failure();

  llvm::json::Object root;
  root["schema_version"] = int64_t{2};
  root["gpu_launch_abi_version"] = int64_t{1};
  root["entries"] = std::move(entries);
  root["artifacts"] = llvm::json::Array();
  llvm::json::Array dependencies;
  if (auto encoded = sourceModule->getAttrOfType<mlir::ArrayAttr>(
          "vernon.source_dependencies")) {
    for (mlir::Attribute attribute : encoded) {
      auto value = mlir::dyn_cast<mlir::StringAttr>(attribute);
      if (!value)
        continue;
      auto [path, digest] = value.getValue().split('=');
      llvm::json::Object dependency;
      dependency["path"] = path.str();
      dependency["sha256"] = digest.str();
      dependencies.emplace_back(std::move(dependency));
    }
  }
  root["dependencies"] = std::move(dependencies);
  llvm::json::Array features;
  for (const std::string &feature : requiredFeatures)
    features.emplace_back(feature);
  root["required_features"] = std::move(features);
  std::string canonicalModule;
  llvm::raw_string_ostream moduleStream(canonicalModule);
  sourceModule.print(moduleStream,
                     mlir::OpPrintingFlags().enableDebugInfo(false));
  root["module_hash"] = llvm::utohexstr(llvm::xxHash64(canonicalModule));

  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << llvm::json::Value(std::move(root));
  return output;
}

bool setCpuReflectionSymbols(
    VernonCompileResult &result,
    const std::vector<vernon::CpuAbiWrapperMetadata> &metadata) {
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse(result.reflection);
  if (!parsed)
    return false;
  llvm::json::Object *root = parsed->getAsObject();
  llvm::json::Array *entries = root ? root->getArray("entries") : nullptr;
  if (!entries)
    return false;
  for (llvm::json::Value &entryValue : *entries) {
    llvm::json::Object *entry = entryValue.getAsObject();
    std::optional<llvm::StringRef> name =
        entry ? entry->getString("name") : std::nullopt;
    if (!name)
      return false;
    auto found = llvm::find_if(metadata, [&](const auto &candidate) {
      return candidate.internalFunctionSymbol == *name;
    });
    if (found == metadata.end())
      return false;
    (*entry)["symbol"] = found->exportedWrapperSymbol;
  }
  result.reflection.clear();
  llvm::raw_string_ostream stream(result.reflection);
  stream << llvm::json::Value(std::move(*root));
  return true;
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

  mlir::OwningOpRef<mlir::ModuleOp> reflectionModule(
      mlir::cast<mlir::ModuleOp>(module->clone()));
  mlir::PassManager reflectionPassManager(&context->context);
  reflectionPassManager.addPass(
      mlir::vernon::createVernonInlineHelpersPass());
  if (mlir::failed(reflectionPassManager.run(*reflectionModule))) {
    result->status = VERNON_STATUS_VERIFICATION_ERROR;
    return result;
  }
  mlir::FailureOr<std::string> reflection =
      buildReflection(*reflectionModule, *module);
  if (mlir::failed(reflection)) {
    result->status = VERNON_STATUS_VERIFICATION_ERROR;
    return result;
  }
  result->reflection = std::move(*reflection);
  result->status = VERNON_STATUS_OK;
  return result;
}

bool captureCpuAbiMetadata(mlir::ModuleOp module,
                           std::vector<vernon::CpuAbiWrapperMetadata> &entries,
                           llvm::StringRef moduleHash,
                           std::string &diagnostics) {
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (!function->hasAttr("vernon.entry"))
      continue;

    vernon::CpuAbiWrapperMetadata metadata;
    metadata.internalFunctionSymbol = function.getSymName().str();
    metadata.exportedWrapperSymbol =
        "__vernon_cpu_" + moduleHash.str() + "_" + function.getSymName().str();
    metadata.argumentsSize = 0;
    metadata.requiresTextureCallbacks = false;
    function.walk([&](mlir::vernon::IntrinsicOp intrinsic) {
      metadata.requiresTextureCallbacks |=
          intrinsic.getName() == "texture_sample";
    });

    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      mlir::Type type = function.getArgumentTypes()[index];
      uint64_t size = cpuSourceTypeSize(type);
      if (size == 0) {
        diagnostics = "unsupported CPU ABI argument type in entry '" +
                      function.getSymName().str() + "'";
        return false;
      }
      uint64_t alignment = size >= 16 ? 16 : size >= 8 ? 8 : 4;
      metadata.argumentsSize = llvm::alignTo(metadata.argumentsSize, alignment);

      vernon::CpuAbiArgumentPacking packing{
          metadata.argumentsSize, size,
          mlir::isa<mlir::vernon::BufferType>(type)
              ? vernon::CpuAbiArgumentKind::Buffer
              : vernon::CpuAbiArgumentKind::Direct,
          0};
      if (packing.kind == vernon::CpuAbiArgumentKind::Buffer) {
        auto attrs = function.getArgAttrDict(index);
        if (auto shape =
                attrs.getAs<mlir::DenseI64ArrayAttr>("vernon.tensor_shape")) {
          uint64_t extent = 1;
          for (int64_t dimension : shape.asArrayRef()) {
            if (dimension < 0 ||
                (dimension != 0 &&
                 extent > std::numeric_limits<uint64_t>::max() /
                              static_cast<uint64_t>(dimension))) {
              diagnostics = "invalid vernon.tensor_shape on CPU buffer in "
                            "entry '" +
                            function.getSymName().str() + "'";
              return false;
            }
            extent *= static_cast<uint64_t>(dimension);
          }
          packing.staticExtent = extent;
        }
      }
      metadata.sourceArguments.push_back(packing);
      metadata.argumentsSize += size;
    }

    if (function.getNumResults() > 1) {
      diagnostics = "CPU entry '" + function.getSymName().str() +
                    "' has more than one result";
      return false;
    }
    metadata.resultsSize =
        function.getNumResults() == 0
            ? 0
            : cpuSourceTypeSize(function.getResultTypes()[0]);
    if (function.getNumResults() != 0 && metadata.resultsSize == 0) {
      diagnostics = "unsupported CPU ABI result type in entry '" +
                    function.getSymName().str() + "'";
      return false;
    }
    entries.push_back(std::move(metadata));
  }
  if (entries.empty()) {
    diagnostics = "module has no CPU entry points";
    return false;
  }
  return true;
}

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
  // Backend lowerings intentionally only handle entry bodies. Inline shared
  // helpers while the module is still in common typed MLIR so every target
  // sees the same implementation.
  passManager.addPass(mlir::vernon::createVernonInlineHelpersPass());
  passManager.addPass(mlir::vernon::createVernonToGPUPass(true));
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::vernon::createVernonLowerGPUTensorsPass());
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertMathToSPIRVPass());
  passManager.addPass(mlir::createConvertGPUToSPIRVPass());
  passManager.addPass(mlir::vernon::createVernonToSPIRVPass());
  passManager.addNestedPass<mlir::spirv::ModuleOp>(
      mlir::spirv::createSPIRVLowerABIAttributesPass());
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
  passManager.addPass(mlir::vernon::createVernonInlineHelpersPass());
  passManager.addPass(mlir::vernon::createVernonToGPUPass());
  passManager.addPass(std::make_unique<KeepGpuModulesPass>());
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::vernon::createVernonLowerGPUTensorsPass());
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertElementwiseToLinalgPass());
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.allowUnknownOps = true;
  passManager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertLinalgToLoopsPass());
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::vernon::createVernonLowerCUDAMathPass());
  // VernonToGPU creates gpu.module directly, so the top-level SCF conversion
  // in MLIR's NVVM pipeline cannot enter that isolated symbol table.
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createSCFToControlFlowPass());
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

struct CpuCodegenOptions {
  std::string targetTriple;
  std::string cpu;
  std::string features;
};

std::string copyStringView(VernonStringView value) {
  return value.data && value.size ? std::string(value.data, value.size)
                                  : std::string();
}

std::string cpuObjectFilename(const llvm::Triple &triple) {
  return triple.isOSBinFormatCOFF() ? "module.obj" : "module.o";
}

bool emitCpuObject(llvm::Module &module, llvm::TargetMachine &targetMachine,
                   std::string &object, std::string &diagnostics) {
  std::unique_ptr<llvm::Module> codegenModule = llvm::CloneModule(module);
  llvm::SmallVector<char> bytes;
  llvm::raw_svector_ostream stream(bytes);
  llvm::legacy::PassManager passManager;
  if (targetMachine.addPassesToEmitFile(passManager, stream, nullptr,
                                        llvm::CodeGenFileType::ObjectFile)) {
    diagnostics = "selected CPU target cannot emit a relocatable object";
    return false;
  }
  passManager.run(*codegenModule);
  object.assign(bytes.begin(), bytes.end());
  if (object.empty()) {
    diagnostics = "CPU TargetMachine emitted an empty relocatable object";
    return false;
  }
  return true;
}

std::string hostNativeLibraryFilename() {
#if defined(_WIN32)
  return "module.dll";
#elif defined(__APPLE__)
  return "module.dylib";
#else
  return "module.so";
#endif
}

bool linkHostObject(llvm::StringRef object, std::string &library,
                    std::string &diagnostics) {
  llvm::SmallString<128> directory;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory("vernon_cpu_link", directory)) {
    diagnostics = "cannot create temporary LLD directory: " + error.message();
    return false;
  }
  const std::filesystem::path root(directory.str().str());
#if defined(_WIN32)
  const std::filesystem::path objectPath = root / "module.obj";
#else
  const std::filesystem::path objectPath = root / "module.o";
#endif
  const std::filesystem::path outputPath = root / hostNativeLibraryFilename();
  auto cleanup = [&] {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  };
  {
    std::ofstream output(objectPath, std::ios::binary);
    output.write(object.data(), static_cast<std::streamsize>(object.size()));
    if (!output) {
      diagnostics = "cannot write temporary CPU object";
      cleanup();
      return false;
    }
  }

  std::vector<std::string> storage;
#if defined(_WIN32)
  storage = {"lld-link", "/dll", "/noentry", "/out:" + outputPath.string(),
             objectPath.string()};
  const lld::DriverDef driver{lld::WinLink, &lld::coff::link};
#elif defined(__APPLE__)
#if defined(__aarch64__)
  constexpr const char *hostArch = "arm64";
#else
  constexpr const char *hostArch = "x86_64";
#endif
  storage = {"ld64.lld",
             "-dylib",
             "-arch",
             hostArch,
             "-platform_version",
             "macos",
             "11.0",
             "11.0",
             "-install_name",
             "@rpath/module.dylib",
             "-o",
             outputPath.string(),
             objectPath.string()};
  const lld::DriverDef driver{lld::Darwin, &lld::macho::link};
#else
  storage = {"ld.lld", "-shared", objectPath.string(), "-o",
             outputPath.string()};
  const lld::DriverDef driver{lld::Gnu, &lld::elf::link};
#endif
  std::vector<const char *> arguments;
  arguments.reserve(storage.size());
  for (const std::string &argument : storage)
    arguments.push_back(argument.c_str());

  std::string linkerOutput;
  llvm::raw_string_ostream linkerStream(linkerOutput);
  static std::mutex linkerMutex;
  lld::Result linkResult{1, false};
  {
    std::lock_guard<std::mutex> lock(linkerMutex);
    linkResult = lld::lldMain(arguments, linkerStream, linkerStream, {driver});
  }
  linkerStream.flush();
  if (linkResult.retCode != 0 || !linkResult.canRunAgain) {
    diagnostics =
        linkerOutput.empty() ? "embedded LLD failed" : std::move(linkerOutput);
    cleanup();
    return false;
  }
  std::ifstream input(outputPath, std::ios::binary);
  library.assign(std::istreambuf_iterator<char>(input),
                 std::istreambuf_iterator<char>());
  if (!input || library.empty()) {
    diagnostics = "embedded LLD produced no readable native library";
    cleanup();
    return false;
  }
  cleanup();
  return true;
}

bool compileCpu(VernonCompilerContext *compilerContext, const char *source,
                size_t sourceSize, const CpuCodegenOptions &options,
                VernonCompileResult &result) {
  mlir::ScopedDiagnosticHandler handler(
      &compilerContext->context, [&](mlir::Diagnostic &diagnostic) {
        appendDiagnostic(result.diagnostics, diagnostic);
      });
  llvm::StringRef text(source ? source : "", sourceSize);
  mlir::OwningOpRef<mlir::ModuleOp> sourceModule =
      mlir::parseSourceString<mlir::ModuleOp>(text, &compilerContext->context);
  if (!sourceModule)
    return false;

  mlir::PassManager metadataPassManager(&compilerContext->context);
  mlir::vernon::buildVernonCpuPreparationPipeline(metadataPassManager);
  if (mlir::failed(metadataPassManager.run(*sourceModule)))
    return false;

  llvm::Expected<llvm::json::Value> parsedReflection =
      llvm::json::parse(result.reflection);
  llvm::json::Object *reflectionRoot =
      parsedReflection ? parsedReflection->getAsObject() : nullptr;
  std::optional<llvm::StringRef> moduleHash =
      reflectionRoot ? reflectionRoot->getString("module_hash") : std::nullopt;
  if (!moduleHash || moduleHash->empty()) {
    result.diagnostics = "CPU reflection is missing its module hash";
    return false;
  }
  std::vector<vernon::CpuAbiWrapperMetadata> entries;
  if (!captureCpuAbiMetadata(*sourceModule, entries, *moduleHash,
                             result.diagnostics))
    return false;

  const bool hostTarget = options.targetTriple.empty();
  const std::string targetTriple = llvm::Triple::normalize(
      hostTarget ? llvm::sys::getDefaultTargetTriple() : options.targetTriple);
  const llvm::Triple parsedTriple(targetTriple);
  if (!parsedTriple.isArch64Bit()) {
    result.diagnostics =
        "CPU relocatable objects currently require a 64-bit target triple";
    return false;
  }
  std::string lookupError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(parsedTriple, lookupError);
  if (!target) {
    result.diagnostics =
        "CPU target '" + targetTriple + "' is unavailable: " + lookupError;
    return false;
  }
  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> objectTargetMachine(
      target->createTargetMachine(parsedTriple, options.cpu, options.features,
                                  targetOptions, llvm::Reloc::PIC_));
  if (!objectTargetMachine) {
    result.diagnostics =
        "failed to create TargetMachine for '" + targetTriple + "'";
    return false;
  }
  const llvm::DataLayout dataLayout = objectTargetMachine->createDataLayout();

  sourceModule->getOperation()->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      mlir::StringAttr::get(&compilerContext->context,
                            dataLayout.getStringRepresentation()));
  sourceModule->getOperation()->setAttr(
      mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
      mlir::StringAttr::get(&compilerContext->context, targetTriple));
  mlir::PassManager passManager(&compilerContext->context);
  mlir::vernon::buildVernonCpuLoweringPipeline(passManager);
  if (mlir::failed(passManager.run(*sourceModule)))
    return false;

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*sourceModule, *llvmContext);
  if (!llvmModule) {
    result.diagnostics = "failed to translate lowered CPU MLIR to LLVM IR";
    return false;
  }
  llvmModule->setDataLayout(dataLayout);
  llvmModule->setTargetTriple(llvm::Triple(targetTriple));
  if (llvm::Error error = vernon::defineCpuTextureSampleHelper(*llvmModule)) {
    result.diagnostics = llvm::toString(std::move(error));
    return false;
  }
  for (const vernon::CpuAbiWrapperMetadata &entry : entries) {
    if (llvm::Error error = vernon::emitCpuAbiWrapper(*llvmModule, entry)) {
      result.diagnostics = llvm::toString(std::move(error));
      return false;
    }
  }
  if (parsedTriple.isOSWindows() && !llvmModule->getNamedGlobal("_fltused"))
    new llvm::GlobalVariable(
        *llvmModule, llvm::Type::getInt32Ty(*llvmContext), true,
        llvm::GlobalValue::WeakAnyLinkage,
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(*llvmContext), 0),
        "_fltused");
  if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
    result.diagnostics = "generated CPU LLVM IR failed verification";
    return false;
  }
  std::string object;
  if (!emitCpuObject(*llvmModule, *objectTargetMachine, object,
                     result.diagnostics))
    return false;

  std::unique_ptr<llvm::orc::LLJIT> jit;
  if (hostTarget) {
    auto jitTargetMachine = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jitTargetMachine) {
      result.diagnostics = llvm::toString(jitTargetMachine.takeError());
      return false;
    }
    auto createdJit =
        llvm::orc::LLJITBuilder()
            .setJITTargetMachineBuilder(std::move(*jitTargetMachine))
            .setDataLayout(dataLayout)
            .create();
    if (!createdJit) {
      result.diagnostics = llvm::toString(createdJit.takeError());
      return false;
    }
    jit = std::move(*createdJit);
    if (llvm::Error error = jit->addIRModule(llvm::orc::ThreadSafeModule(
            std::move(llvmModule), std::move(llvmContext)))) {
      result.diagnostics = llvm::toString(std::move(error));
      return false;
    }
    for (const vernon::CpuAbiWrapperMetadata &entry : entries) {
      auto symbol = jit->lookup(entry.exportedWrapperSymbol);
      if (!symbol) {
        result.diagnostics = llvm::toString(symbol.takeError());
        return false;
      }
      result.cpuEntries.emplace(entry.internalFunctionSymbol,
                                symbol->toPtr<VernonCpuEntryPoint>());
    }
  }
  result.artifacts.clear();
  result.artifacts.push_back(VernonCompileResult::Artifact{
      cpuObjectFilename(parsedTriple), std::move(object)});
  if (!setCpuReflectionSymbols(result, entries)) {
    result.diagnostics = "compiler produced invalid CPU reflection metadata";
    return false;
  }
  result.cpuJit = std::move(jit);
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

bool crossCompile(VernonCompileResult &result, VernonTarget target,
                  uint32_t glslVersion) {
  std::vector<VernonCompileResult::Artifact> translated;
  try {
    for (const VernonCompileResult::Artifact &artifact : result.artifacts) {
      if (artifact.data.size() % sizeof(uint32_t) != 0)
        return false;
      std::vector<uint32_t> words(artifact.data.size() / sizeof(uint32_t));
      std::memcpy(words.data(), artifact.data.data(), artifact.data.size());
      spirv_cross::Compiler probe(words);
      const auto entryPoints = probe.get_entry_points_and_stages();
      std::map<uint32_t, std::string> varyingNames;
      if (target != VERNON_TARGET_METAL) {
        auto sourceName = [](std::string name) {
          for (llvm::StringRef suffix : {"_vertex", "_fragment"}) {
            if (llvm::StringRef(name).ends_with(suffix)) {
              name.resize(name.size() - suffix.size());
              break;
            }
          }
          return name;
        };
        for (const spirv_cross::EntryPoint &entry : entryPoints) {
          spirv_cross::CompilerGLSL interfaceCompiler(words);
          interfaceCompiler.set_entry_point(entry.name, entry.execution_model);
          const spirv_cross::ShaderResources resources =
              interfaceCompiler.get_shader_resources();
          const auto *variables =
              entry.execution_model == spv::ExecutionModelVertex
                  ? &resources.stage_outputs
              : entry.execution_model == spv::ExecutionModelFragment
                  ? &resources.stage_inputs
                  : nullptr;
          if (!variables)
            continue;
          for (const spirv_cross::Resource &variable : *variables) {
            if (!interfaceCompiler.has_decoration(variable.id,
                                                  spv::DecorationLocation))
              continue;
            const uint32_t location = interfaceCompiler.get_decoration(
                variable.id, spv::DecorationLocation);
            std::string name = sourceName(variable.name);
            if (entry.execution_model == spv::ExecutionModelVertex)
              varyingNames[location] = std::move(name);
            else
              varyingNames.try_emplace(location, std::move(name));
          }
        }
      }
      for (const spirv_cross::EntryPoint &entry : entryPoints) {
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
          const spirv_cross::ShaderResources resources =
              compiler.get_shader_resources();
          auto canonicalizeInterface = [&](const auto &variables) {
            for (const spirv_cross::Resource &variable : variables) {
              if (!compiler.has_decoration(variable.id,
                                           spv::DecorationLocation))
                continue;
              const uint32_t location =
                  compiler.get_decoration(variable.id, spv::DecorationLocation);
              const auto name = varyingNames.find(location);
              compiler.set_name(variable.id,
                                name != varyingNames.end()
                                    ? name->second
                                    : "vernon_location_" +
                                          std::to_string(location));
            }
          };
          // Canonical location names let GLSL 3.30 link vertex outputs to
          // fragment inputs. Prefer the source-derived vertex output name;
          // vertex inputs and fragment outputs retain their own source names.
          if (entry.execution_model == spv::ExecutionModelVertex)
            canonicalizeInterface(resources.stage_outputs);
          else if (entry.execution_model == spv::ExecutionModelFragment)
            canonicalizeInterface(resources.stage_inputs);
          spirv_cross::CompilerGLSL::Options options;
          options.es = target == VERNON_TARGET_OPENGL_ES;
          options.version =
              glslVersion != 0 ? glslVersion
              : options.es     ? 310
              : entry.execution_model == spv::ExecutionModelGLCompute ? 430
                                                                      : 330;
          options.enable_420pack_extension = options.es;
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
  return vernonCompilerCompileMlirWithOptions(context, source, sourceSize,
                                              target, nullptr);
}

VernonCompileResult *vernonCompilerCompileMlirWithOptions(
    VernonCompilerContext *context, const char *source, size_t sourceSize,
    VernonTarget target, const VernonCompileOptions *options) {
  auto result = validate(context, source, sourceSize);
  if (result->status != VERNON_STATUS_OK)
    return result.release();

  if (target < VERNON_TARGET_CPU || target > VERNON_TARGET_CUDA) {
    result->status = VERNON_STATUS_INVALID_ARGUMENT;
    result->diagnostics = "unknown compilation target";
    result->artifacts.clear();
    return result.release();
  }

  uint32_t glslVersion = 0;
  CpuCodegenOptions cpuOptions;
  if (options) {
    constexpr size_t requiredOptionsSize =
        offsetof(VernonCompileOptions, glsl_version) + sizeof(uint32_t);
    if (options->struct_size < requiredOptionsSize) {
      result->status = VERNON_STATUS_INVALID_ARGUMENT;
      result->diagnostics = "compile options structure is too small";
      result->artifacts.clear();
      return result.release();
    }
    glslVersion = options->glsl_version;
    if (glslVersion != 0 && target != VERNON_TARGET_OPENGL &&
        target != VERNON_TARGET_OPENGL_ES) {
      result->status = VERNON_STATUS_INVALID_ARGUMENT;
      result->diagnostics =
          "GLSL version is valid only for OpenGL and OpenGL ES targets";
      result->artifacts.clear();
      return result.release();
    }
    if (glslVersion != 0 && (glslVersion < 100 || glslVersion > 999)) {
      result->status = VERNON_STATUS_INVALID_ARGUMENT;
      result->diagnostics = "GLSL version must be a three-digit version number";
      result->artifacts.clear();
      return result.release();
    }
    auto readCpuOption = [&](size_t offset, std::string &destination) -> bool {
      if (options->struct_size < offset + sizeof(VernonStringView))
        return true;
      VernonStringView value{};
      std::memcpy(&value, reinterpret_cast<const char *>(options) + offset,
                  sizeof(value));
      if (value.size != 0 && !value.data) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "CPU compile option has null data";
        result->artifacts.clear();
        return false;
      }
      destination = copyStringView(value);
      return true;
    };
    if (!readCpuOption(offsetof(VernonCompileOptions, cpu_target_triple),
                       cpuOptions.targetTriple) ||
        !readCpuOption(offsetof(VernonCompileOptions, cpu_name),
                       cpuOptions.cpu) ||
        !readCpuOption(offsetof(VernonCompileOptions, cpu_features),
                       cpuOptions.features))
      return result.release();
    if (target != VERNON_TARGET_CPU &&
        (!cpuOptions.targetTriple.empty() || !cpuOptions.cpu.empty() ||
         !cpuOptions.features.empty())) {
      result->status = VERNON_STATUS_INVALID_ARGUMENT;
      result->diagnostics =
          "CPU code generation options are valid only for the CPU target";
      result->artifacts.clear();
      return result.release();
    }
  }

  if (target == VERNON_TARGET_CPU) {
    result->diagnostics.clear();
    if (compileCpu(context, source, sourceSize, cpuOptions, *result)) {
      addArtifactTable(*result, target, glslVersion, cpuOptions.targetTriple,
                       cpuOptions.cpu, cpuOptions.features);
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
      if (target != VERNON_TARGET_VULKAN &&
          !crossCompile(*result, target, glslVersion)) {
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
      addArtifactTable(*result, target, glslVersion);
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
      addArtifactTable(*result, target, glslVersion);
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

VernonCompileResult *
vernonCompilerLinkHostObject(VernonCompilerContext *context, const void *object,
                             size_t objectSize) {
  auto result = std::make_unique<VernonCompileResult>();
  if (!context || !object || objectSize == 0) {
    result->status = VERNON_STATUS_INVALID_ARGUMENT;
    result->diagnostics =
        "host object linking requires a compiler context and object bytes";
    return result.release();
  }
  std::string library;
  if (!linkHostObject(
          llvm::StringRef(static_cast<const char *>(object), objectSize),
          library, result->diagnostics)) {
    result->status = VERNON_STATUS_INTERNAL_ERROR;
    return result.release();
  }
  result->artifacts.push_back(VernonCompileResult::Artifact{
      hostNativeLibraryFilename(), std::move(library)});
  result->status = VERNON_STATUS_OK;
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
