#include "compiler_cuda.h"

#include "VernonProgramCapabilities.h"
#include "compiler_dispatch.h"
#include "compiler_frontend.h"
#include "compiler_reflection.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCUDAMath.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerSynchronization.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/IR/Diagnostics.h"
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

bool compileCuda(PreparedModule &prepared, const TargetProfile &profile, std::vector<Artifact> &artifacts,
                 std::string &reflection, std::string &diagnostics) {
    mlir::MLIRContext &context = prepared.context();
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    mlir::FailureOr<TargetPreparationResult> preparedTarget =
        prepareTargetModule(prepared, preparePortableTargetModule);
    if (mlir::failed(preparedTarget))
        return false;
    mlir::OwningOpRef<mlir::ModuleOp> module = std::move(preparedTarget->module);
    if (moduleUsesF16(module.get())) {
        const program_capabilities::Entry &capability = program_capabilities::get(program_capabilities::Id::GpuF16);
        diagnostics = std::string(capability.diagnosticCode) + ": " + std::string(capability.diagnostic);
        return false;
    }
    {
        mlir::PassManager passManager(&context);
        passManager.addPass(mlir::vernon::createVernonLowerAccumulationPass(profile.accumulation));
        passManager.addPass(mlir::vernon::createVernonVerifyGeneratedAccumulationPass(profile.accumulation));
        if (mlir::failed(passManager.run(*module)))
            return false;
    }
    mlir::FailureOr<std::string> targetReflection =
        buildReflection(*module, prepared.logicalReflection(), preparedTarget->entries, preparedTarget->provenance);
    if (mlir::failed(targetReflection))
        return false;
    reflection = std::move(*targetReflection);

    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::vernon::createVernonToGPUPass());
    passManager.addPass(std::make_unique<KeepGpuModulesPass>());
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::vernon::createVernonLowerGPUSynchronizationPass());
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
