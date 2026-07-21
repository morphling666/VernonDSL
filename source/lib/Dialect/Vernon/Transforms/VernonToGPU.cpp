#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::vernon {
namespace {

FailureOr<Type> convertKernelArgument(Type type, bool useSpirvStorage) {
  if (auto buffer = dyn_cast<BufferType>(type)) {
    if (!useSpirvStorage)
      return MemRefType::get({ShapedType::kDynamic}, buffer.getElementType());
    return MemRefType::get(
        {ShapedType::kDynamic}, buffer.getElementType(), AffineMap(),
        spirv::StorageClassAttr::get(type.getContext(),
                                     spirv::StorageClass::StorageBuffer));
  }
  return failure();
}

struct VernonToGPUPass
    : public PassWrapper<VernonToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonToGPUPass)

  VernonToGPUPass() = default;
  explicit VernonToGPUPass(bool useSpirvStorage)
      : useSpirvStorage(useSpirvStorage) {}
  VernonToGPUPass(const VernonToGPUPass &other)
      : PassWrapper(other), useSpirvStorage(other.useSpirvStorage) {}

  StringRef getArgument() const final { return "vernon-to-gpu"; }
  StringRef getDescription() const final {
    return "Outline Vernon compute entries as MLIR GPU kernels";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    spirv::SPIRVDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> computeEntries;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
      if (function->hasAttr(kEntryAttrName) && stage &&
          stage.getValue() == "compute")
        computeEntries.push_back(function);
    }
    if (computeEntries.empty())
      return;

    OpBuilder topBuilder(module.getContext());
    topBuilder.setInsertionPointToEnd(module.getBody());
    auto gpuModule =
        gpu::GPUModuleOp::create(topBuilder, module.getLoc(), "vernon_kernels");
    if (useSpirvStorage) {
      auto targetTriple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_3, {spirv::Capability::Shader},
          llvm::ArrayRef<spirv::Extension>(), module.getContext());
      gpuModule->setAttr(
          spirv::getTargetEnvAttrName(),
          spirv::TargetEnvAttr::get(
              targetTriple,
              spirv::getDefaultResourceLimits(module.getContext()),
              spirv::ClientAPI::Vulkan, spirv::Vendor::Unknown,
              spirv::DeviceType::Unknown,
              spirv::TargetEnvAttr::kUnknownDeviceID));
    }
    OpBuilder moduleBuilder = OpBuilder::atBlockBegin(gpuModule.getBody());

    for (func::FuncOp source : computeEntries) {
      SmallVector<Type> kernelArgumentTypes;
      SmallVector<unsigned> kernelSourceArgumentIndices;
      SmallVector<std::pair<unsigned, std::pair<unsigned, unsigned>>>
          resourceBindings;
      for (auto [index, type] : llvm::enumerate(source.getArgumentTypes())) {
        InterfaceAttrs attrs =
            parseInterfaceAttrs(source.getArgAttrDict(index));
        auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
        if (kind && kind.getValue() == "resource") {
          FailureOr<Type> converted =
              convertKernelArgument(type, useSpirvStorage);
          if (failed(converted)) {
            source.emitError() << "cannot lower compute resource argument #"
                               << index << " type " << type;
            return signalPassFailure();
          }
          unsigned kernelIndex = kernelArgumentTypes.size();
          kernelSourceArgumentIndices.push_back(index);
          kernelArgumentTypes.push_back(*converted);
          auto descriptorSet = cast<IntegerAttr>(attrs.descriptorSet);
          auto binding = cast<IntegerAttr>(attrs.binding);
          resourceBindings.emplace_back(
              kernelIndex,
              std::make_pair(descriptorSet.getInt(), binding.getInt()));
        } else if (!attrs.builtin &&
                   (type.isIntOrIndexOrFloat() || isa<VectorType>(type))) {
          kernelSourceArgumentIndices.push_back(index);
          kernelArgumentTypes.push_back(type);
        }
      }

      auto functionType =
          moduleBuilder.getFunctionType(kernelArgumentTypes, TypeRange{});
      auto kernel = gpu::GPUFuncOp::create(moduleBuilder, source.getLoc(),
                                           source.getSymName(), functionType);
      kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                      moduleBuilder.getUnitAttr());
      if (auto workgroup = source->getAttrOfType<DenseI32ArrayAttr>(
              kWorkgroupSizeAttrName)) {
        kernel.setKnownBlockSizeAttr(workgroup);
        kernel->setAttr(spirv::getEntryPointABIAttrName(),
                        spirv::getEntryPointABIAttr(source.getContext(),
                                                    workgroup.asArrayRef()));
      }
      for (auto [index, binding] : resourceBindings) {
        kernel.setArgAttr(
            index, spirv::getInterfaceVarABIAttrName(),
            spirv::getInterfaceVarABIAttr(binding.first, binding.second,
                                          std::nullopt, source.getContext()));
      }
      if (useSpirvStorage) {
        for (unsigned index = 0; index < kernel.getNumArguments(); ++index) {
          if (kernel.getArgAttr(index, spirv::getInterfaceVarABIAttrName()))
            continue;
          std::optional<spirv::StorageClass> storageClass;
          if (kernel.getArgument(index).getType().isIntOrIndexOrFloat())
            storageClass = spirv::StorageClass::StorageBuffer;
          kernel.setArgAttr(index, spirv::getInterfaceVarABIAttrName(),
                            spirv::getInterfaceVarABIAttr(
                                0, index, storageClass, source.getContext()));
        }
      }

      Block *entry = &kernel.front();
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
      IRMapping mapping;
      for (auto [sourceIndex, kernelArgument] :
           llvm::zip_equal(kernelSourceArgumentIndices, entry->getArguments()))
        mapping.map(source.getArgument(sourceIndex), kernelArgument);

      for (auto [index, argument] : llvm::enumerate(source.getArguments())) {
        if (mapping.contains(argument))
          continue;
        InterfaceAttrs attrs =
            parseInterfaceAttrs(source.getArgAttrDict(index));
        auto builtin = dyn_cast_if_present<StringAttr>(attrs.builtin);
        if (!builtin || builtin.getValue() != "global_invocation_id") {
          source.emitError()
              << "compute argument #" << index
              << " is neither a resource nor a supported builtin";
          return signalPassFailure();
        }
        if (auto tensorType = dyn_cast<RankedTensorType>(argument.getType())) {
          if (tensorType.getRank() != 1 || tensorType.getDimSize(0) != 3 ||
              !tensorType.getElementType().isInteger(32)) {
            source.emitError()
                << "global_invocation_id Tensor must have type tensor<3xi32>";
            return signalPassFailure();
          }
          SmallVector<Value> components;
          for (gpu::Dimension dimension :
               {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
            Value id = gpu::GlobalIdOp::create(bodyBuilder, source.getLoc(),
                                               dimension);
            components.push_back(arith::IndexCastUIOp::create(
                bodyBuilder, source.getLoc(), tensorType.getElementType(), id));
          }
          mapping.map(argument, tensor::FromElementsOp::create(
                                    bodyBuilder, source.getLoc(), tensorType,
                                    components));
        } else {
          Value globalId = gpu::GlobalIdOp::create(bodyBuilder, source.getLoc(),
                                                   gpu::Dimension::x);
          if (!argument.getType().isIndex())
            globalId = arith::IndexCastUIOp::create(
                bodyBuilder, source.getLoc(), argument.getType(), globalId);
          mapping.map(argument, globalId);
        }
      }

      for (Operation &operation : source.front()) {
        if (isa<func::ReturnOp>(operation)) {
          gpu::ReturnOp::create(bodyBuilder, operation.getLoc());
          continue;
        }
        if (auto intrinsic = dyn_cast<IntrinsicOp>(operation)) {
          if (intrinsic.getName() == "buffer_load") {
            Value loaded =
                memref::LoadOp::create(bodyBuilder, intrinsic.getLoc(),
                                       mapping.lookup(intrinsic.getOperand(0)),
                                       mapping.lookup(intrinsic.getOperand(1)));
            mapping.map(intrinsic.getResult(), loaded);
            continue;
          }
          if (intrinsic.getName() == "buffer_store") {
            memref::StoreOp::create(bodyBuilder, intrinsic.getLoc(),
                                    mapping.lookup(intrinsic.getOperand(2)),
                                    mapping.lookup(intrinsic.getOperand(0)),
                                    mapping.lookup(intrinsic.getOperand(1)));
            continue;
          }
        }
        bodyBuilder.clone(operation, mapping);
      }

      // Intrinsics nested under structured control flow are cloned recursively,
      // so lower them after the complete kernel body has been materialized.
      SmallVector<IntrinsicOp> nestedIntrinsics;
      kernel.walk([&](IntrinsicOp intrinsic) {
        nestedIntrinsics.push_back(intrinsic);
      });
      for (IntrinsicOp intrinsic : nestedIntrinsics) {
        OpBuilder builder(intrinsic);
        if (intrinsic.getName() == "buffer_load") {
          Value loaded = memref::LoadOp::create(builder, intrinsic.getLoc(),
                                                intrinsic.getOperand(0),
                                                intrinsic.getOperand(1));
          intrinsic.getResult().replaceAllUsesWith(loaded);
          intrinsic.erase();
          continue;
        }
        if (intrinsic.getName() == "buffer_store") {
          memref::StoreOp::create(
              builder, intrinsic.getLoc(), intrinsic.getOperand(2),
              intrinsic.getOperand(0), intrinsic.getOperand(1));
          intrinsic.erase();
        }
      }
    }
  }

  bool useSpirvStorage{false};
};

} // namespace

std::unique_ptr<Pass> createVernonToGPUPass(bool useSpirvStorage) {
  return std::make_unique<VernonToGPUPass>(useSpirvStorage);
}

void registerVernonToGPUPass() { PassRegistration<VernonToGPUPass>(); }

} // namespace mlir::vernon
