#include "compiler_cuda.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCUDAMath.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernon::compiler {
namespace {

struct KeepGpuModulesPass : public mlir::PassWrapper<KeepGpuModulesPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KeepGpuModulesPass)

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        for (mlir::Operation &operation : llvm::make_early_inc_range(module.getBody()->without_terminator()))
            if (!mlir::isa<mlir::gpu::GPUModuleOp>(operation))
                operation.erase();
    }
};

} // namespace

bool compileCuda(mlir::MLIRContext &context, const char *source, size_t sourceSize, std::vector<Artifact> &artifacts,
                 std::string &diagnostics) {
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    llvm::StringRef text(source ? source : "", sourceSize);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
        return false;

    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::vernon::createVernonValidatePass());
    passManager.addPass(mlir::vernon::createVernonInlineHelpersPass());
    passManager.addPass(mlir::vernon::createVernonToGPUPass());
    passManager.addPass(std::make_unique<KeepGpuModulesPass>());
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::vernon::createVernonLowerGPUTensorsPass());
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertElementwiseToLinalgPass());
    mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.allowUnknownOps = true;
    passManager.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertLinalgToLoopsPass());
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::vernon::createVernonLowerCUDAMathPass());
    // VernonToGPU creates gpu.module directly, so the top-level SCF conversion
    // in MLIR's NVVM pipeline cannot enter that isolated symbol table.
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createSCFToControlFlowPass());
    mlir::gpu::GPUToNVVMPipelineOptions options;
    options.cubinFormat = "isa";
    mlir::gpu::buildLowerToNVVMPassPipeline(passManager, options);
    if (mlir::failed(passManager.run(*module)))
        return false;

    artifacts.clear();
    module->walk([&](mlir::gpu::BinaryOp binary) {
        for (mlir::Attribute attribute : binary.getObjects()) {
            auto object = mlir::dyn_cast<mlir::gpu::ObjectAttr>(attribute);
            if (!object)
                continue;
            mlir::StringAttr data = object.getObject();
            artifacts.push_back(Artifact{binary.getSymName().str() + ".ptx", data.getValue().str()});
        }
    });
    if (artifacts.empty()) {
        diagnostics = "NVVM pipeline produced no PTX object";
        return false;
    }
    return true;
}

} // namespace vernon::compiler
