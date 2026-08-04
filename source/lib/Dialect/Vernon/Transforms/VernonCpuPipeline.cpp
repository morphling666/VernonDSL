#include "mlir/Dialect/Vernon/Transforms/VernonCpuPipeline.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUResources.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerSynchronization.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::vernon {
namespace {

struct MaterializeStorageProjectionPass final : PassWrapper<MaterializeStorageProjectionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeStorageProjectionPass)

    StringRef getArgument() const final { return "vernon-materialize-storage-projections"; }
    StringRef getDescription() const final { return "Materialize canonical TensorView physical indices"; }

    void runOnOperation() override {
        if (failed(materializeTensorViewProjections(getOperation())))
            signalPassFailure();
    }
};

struct VerifyVernonCpuLLVMConversionPass final
    : PassWrapper<VerifyVernonCpuLLVMConversionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyVernonCpuLLVMConversionPass)

    StringRef getArgument() const final { return "vernon-verify-cpu-llvm-conversion"; }
    StringRef getDescription() const final {
        return "Require a fully converted builtin module containing only LLVM IR";
    }

    void runOnOperation() override {
        WalkResult result = getOperation().walk([&](Operation *operation) {
            if (isa<ModuleOp>(operation) ||
                operation->getDialect() == getContext().getLoadedDialect<LLVM::LLVMDialect>())
                return WalkResult::advance();
            operation->emitError() << "CPU pipeline left illegal operation '" << operation->getName().getStringRef()
                                   << "' after LLVM conversion";
            return WalkResult::interrupt();
        });
        if (result.wasInterrupted())
            signalPassFailure();
    }
};

} // namespace

void registerVernonCpuPipelineDialects(DialectRegistry &registry) {
    registry
        .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect, index::IndexDialect, LLVM::LLVMDialect,
                math::MathDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
    arith::registerConvertArithToLLVMInterface(registry);
    cf::registerConvertControlFlowToLLVMInterface(registry);
    registerConvertFuncToLLVMInterface(registry);
    index::registerConvertIndexToLLVMInterface(registry);
    registerConvertMathToLLVMInterface(registry);
    registerConvertMemRefToLLVMInterface(registry);
    ub::registerConvertUBToLLVMInterface(registry);
    vector::registerConvertVectorToLLVMInterface(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
}

void buildVernonCpuPreparationPipeline(OpPassManager &passManager) {
    passManager.addPass(createVernonValidatePass());
    passManager.addPass(createVernonInlineHelpersPass());
    passManager.addPass(std::make_unique<MaterializeStorageProjectionPass>());
}

void buildVernonCpuLoweringPipeline(OpPassManager &passManager) {
    passManager.addPass(
        createVernonLowerAccumulationPass(AccumulationTargetCapabilities{/*supportsF32AtomicAdd=*/true,
                                                                         /*supportsF64AtomicAdd=*/true}));
    passManager.addPass(createVernonLowerCPUTensorsPass());
    passManager.addPass(createVernonLowerSynchronizationPass());
    passManager.addPass(createVernonLowerCPUResourcesPass());
    passManager.addPass(createSCFToControlFlowPass());
    passManager.addPass(createConvertToLLVMPass());
    passManager.addPass(std::make_unique<VerifyVernonCpuLLVMConversionPass>());
}

void buildVernonCpuPassPipeline(OpPassManager &passManager) {
    buildVernonCpuPreparationPipeline(passManager);
    buildVernonCpuLoweringPipeline(passManager);
}

void registerVernonCpuPassPipeline() {
    PassRegistration<MaterializeStorageProjectionPass>();
    PassPipelineRegistration<>("vernon-cpu-pipeline",
                               "Lower a validated Vernon CPU module through standard MLIR to LLVM",
                               [](OpPassManager &passManager) { buildVernonCpuPassPipeline(passManager); });
}

} // namespace mlir::vernon
