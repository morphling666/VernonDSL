#include "compiler_cuda.h"

#include "VernonProgramCapabilities.h"
#include "compiler_dispatch.h"
#include "compiler_frontend.h"
#include "compiler_reflection.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vernon/Transforms/VernonExpandFloorCeilDivision.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCUDAMath.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerSynchronization.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MathExtras.h"

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

// Gives every kernel a twin entry `<name>__vernon_indirect(ptr)` that loads the kernel's parameters from device memory
// and runs an always-inlined copy of its body. A parameter lies at the next offset its ABI alignment allows after the
// one before, which is how the runtime packs a launch's arguments into its record (specs/runtime/
// architecture_v2.md section 16.1), so a CUDA graph launches it with one pointer that never changes. The runtime
// checks that layout against the direct entry's PTX parameters before it uses the twin.
struct IndirectKernelEntriesPass
    : public mlir::PassWrapper<IndirectKernelEntriesPass, mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndirectKernelEntriesPass)

    void runOnOperation() override {
        mlir::gpu::GPUModuleOp module = getOperation();
        const llvm::StringRef kernelAttribute = mlir::NVVM::NVVMDialect::getKernelFuncAttrName();
        llvm::SmallVector<mlir::LLVM::LLVMFuncOp> kernels;
        for (auto function : module.getOps<mlir::LLVM::LLVMFuncOp>())
            if (function->hasAttr(kernelAttribute) && !function.isExternal())
                kernels.push_back(function);
        // Offsets follow LLVM's NVPTX layout, as the PTX parameters do: the module has no DLTI entries, and MLIR's
        // default layout aligns i64 to 4 bytes, which would put an aggregate after an f32 four bytes early.
        llvm::LLVMContext context;
        mlir::LLVM::TypeToLLVMIRTranslator types(context);
        const llvm::DataLayout nvptx("e-i64:64-i128:128-v16:16-v32:32-n16:32:64");
        mlir::OpBuilder builder(module.getContext());
        for (mlir::LLVM::LLVMFuncOp kernel : kernels) {
            mlir::Location location = kernel.getLoc();
            builder.setInsertionPointAfter(kernel);
            auto body = mlir::cast<mlir::LLVM::LLVMFuncOp>(builder.clone(*kernel));
            body.setSymName((kernel.getSymName() + "__vernon_body").str());
            for (const mlir::NamedAttribute &attribute : kernel->getDiscardableAttrs())
                body->removeAttr(attribute.getName());
            body.setLinkage(mlir::LLVM::Linkage::Internal);
            body.setAlwaysInline(true);

            // A kernel's pointer parameters are global addresses, which NVPTX infers for a kernel's own parameters
            // but not for values loaded from memory, so the body takes them in the global space and casts them back.
            auto pointer = mlir::LLVM::LLVMPointerType::get(module.getContext());
            auto global = mlir::LLVM::LLVMPointerType::get(module.getContext(),
                                                           static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Global));
            llvm::SmallVector<mlir::Type> bodyParameters(body.getFunctionType().getParams());
            mlir::Block &bodyEntry = body.getBody().front();
            builder.setInsertionPointToStart(&bodyEntry);
            for (auto [index, parameter] : llvm::enumerate(bodyParameters)) {
                if (parameter != pointer)
                    continue;
                parameter = global;
                mlir::BlockArgument argument = bodyEntry.getArgument(index);
                argument.setType(global);
                auto generic = mlir::LLVM::AddrSpaceCastOp::create(builder, location, pointer, argument);
                argument.replaceAllUsesExcept(generic, generic);
            }
            body.setFunctionType(
                mlir::LLVM::LLVMFunctionType::get(body.getFunctionType().getReturnType(), bodyParameters));
            builder.setInsertionPointAfter(body);
            auto type =
                mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(module.getContext()), {pointer});
            auto entry = mlir::LLVM::LLVMFuncOp::create(builder, location,
                                                        (kernel.getSymName() + "__vernon_indirect").str(), type);
            for (const mlir::NamedAttribute &attribute : kernel->getDiscardableAttrs())
                entry->setAttr(attribute.getName(), attribute.getValue());
            entry.setArgAttr(0, mlir::LLVM::LLVMDialect::getNoAliasAttrName(), builder.getUnitAttr());
            entry.setArgAttr(0, mlir::LLVM::LLVMDialect::getReadonlyAttrName(), builder.getUnitAttr());
            mlir::Block *block = entry.addEntryBlock(builder);
            builder.setInsertionPointToStart(block);
            llvm::SmallVector<mlir::Value> arguments;
            uint64_t end = 0;
            for (mlir::Type parameter : bodyParameters) {
                llvm::Type *translated = types.translateType(parameter);
                const uint64_t alignment = nvptx.getABITypeAlign(translated).value();
                const uint64_t offset = llvm::alignTo(end, alignment);
                end = offset + nvptx.getTypeAllocSize(translated).getFixedValue();
                auto address =
                    mlir::LLVM::GEPOp::create(builder, location, pointer, builder.getI8Type(), block->getArgument(0),
                                              llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(offset)},
                                              mlir::LLVM::GEPNoWrapFlags::inbounds);
                arguments.push_back(mlir::LLVM::LoadOp::create(builder, location, parameter, address, alignment,
                                                               /*isVolatile=*/false, /*isNonTemporal=*/false,
                                                               /*isInvariant=*/true));
            }
            mlir::LLVM::CallOp::create(builder, location, body, arguments);
            mlir::LLVM::ReturnOp::create(builder, location, mlir::ValueRange{});
        }
    }
};

// mlir::gpu::buildLowerToNVVMPassPipeline with IndirectKernelEntriesPass between the GPU module's lowering to LLVM
// and its serialization. VernonToGPU emits no host code, so the host half keeps only the serialization.
void buildLowerToNVVMWithIndirectEntries(mlir::OpPassManager &passManager,
                                         const mlir::gpu::GPUToNVVMPipelineOptions &options) {
    passManager.addPass(mlir::createConvertNVGPUToNVVMPass());
    passManager.addPass(mlir::createGpuKernelOutliningPass());
    passManager.addPass(mlir::createConvertVectorToSCFPass());
    passManager.addPass(mlir::createSCFToControlFlowPass());
    passManager.addPass(mlir::createConvertNVVMToLLVMPass());
    passManager.addPass(mlir::createConvertFuncToLLVMPass());
    passManager.addPass(mlir::memref::createExpandStridedMetadataPass());
    mlir::GpuNVVMAttachTargetOptions target;
    target.triple = options.cubinTriple;
    target.chip = options.cubinChip;
    target.features = options.cubinFeatures;
    target.optLevel = options.optLevel;
    target.cmdOptions = options.cmdOptions;
    passManager.addPass(mlir::createGpuNVVMAttachTarget(target));
    passManager.addPass(mlir::createLowerAffinePass());
    passManager.addPass(mlir::createArithToLLVMConversionPass());
    mlir::ConvertIndexToLLVMPassOptions index;
    index.indexBitwidth = options.indexBitWidth;
    passManager.addPass(mlir::createConvertIndexToLLVMPass(index));
    passManager.addPass(mlir::createCanonicalizerPass());
    passManager.addPass(mlir::createCSEPass());

    mlir::ConvertGpuOpsToNVVMOpsOptions gpu;
    gpu.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
    gpu.indexBitwidth = options.indexBitWidth;
    gpu.allowPatternRollback = options.allowPatternRollback;
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps(gpu));
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createReconcileUnrealizedCastsPass());
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(std::make_unique<IndirectKernelEntriesPass>());

    mlir::GpuToLLVMConversionPassOptions host;
    host.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
    host.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
    passManager.addPass(mlir::createGpuToLLVMConversionPass(host));
    mlir::GpuModuleToBinaryPassOptions binary;
    binary.compilationTarget = options.cubinFormat;
    passManager.addPass(mlir::createGpuModuleToBinaryPass(binary));
    passManager.addPass(mlir::createConvertMathToLLVMPass());
    passManager.addPass(mlir::createCanonicalizerPass());
    passManager.addPass(mlir::createCSEPass());
    passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
}

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
    if (moduleUsesFloat(module.get(), 16)) {
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
        buildReflection(*module, prepared.logicalReflection(), preparedTarget->entries, preparedTarget->provenance,
                        mlir::vernon::MetadataPhysicalProfile::CudaKernelMetadata);
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
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::vernon::createVernonExpandFloorCeilDivisionPass());
    // VernonToGPU creates gpu.module directly, so the top-level SCF conversion
    // in MLIR's NVVM pipeline cannot enter that isolated symbol table.
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createSCFToControlFlowPass());
    mlir::gpu::GPUToNVVMPipelineOptions options;
    options.cubinFormat = "isa";
    buildLowerToNVVMWithIndirectEntries(passManager, options);
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
