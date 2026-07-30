#include "compiler_frontend.h"

#include "compiler_reflection.h"

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonCpuPipeline.h"
#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
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
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <new>
#include <string>
#include <utility>

namespace vernon::compiler {

class CompilerFrontend {
public:
    CompilerFrontend() {
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

    mlir::MLIRContext context;
};

CompilerFrontend *createCompilerFrontend() { return new (std::nothrow) CompilerFrontend(); }

void destroyCompilerFrontend(CompilerFrontend *frontend) { delete frontend; }

PreparedModule::PreparedModule(mlir::OwningOpRef<mlir::ModuleOp> module) : module_(std::move(module)) {}

mlir::MLIRContext &PreparedModule::context() { return *module_->getContext(); }

mlir::OwningOpRef<mlir::ModuleOp> PreparedModule::clone() {
    return mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module_->clone()));
}

VernonStatus prepareMlir(CompilerFrontend &frontend, const char *source, size_t sourceSize, PreparedModulePtr &prepared,
                         std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics) {
    mlir::ScopedDiagnosticHandler handler(
        &frontend.context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });

    llvm::StringRef text(source ? source : "", sourceSize);
    mlir::ParserConfig parserConfig(&frontend.context, /*verifyAfterParse=*/false);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(text, parserConfig);
    if (!module)
        return VERNON_STATUS_PARSE_ERROR;
    if (mlir::failed(mlir::verify(*module)))
        return VERNON_STATUS_VERIFICATION_ERROR;

    mlir::PassManager passManager(&frontend.context);
    passManager.addPass(mlir::vernon::createVernonValidatePass());
    if (mlir::failed(passManager.run(*module)))
        return VERNON_STATUS_VERIFICATION_ERROR;

    std::string canonicalModule;
    llvm::raw_string_ostream artifactStream(canonicalModule);
    module->print(artifactStream, mlir::OpPrintingFlags().enableDebugInfo(false));
    artifacts.push_back(Artifact{"module.mlir", std::move(canonicalModule)});

    mlir::OwningOpRef<mlir::ModuleOp> reflectionModule(mlir::cast<mlir::ModuleOp>(module->clone()));
    mlir::PassManager reflectionPassManager(&frontend.context);
    reflectionPassManager.addPass(mlir::vernon::createVernonInlineHelpersPass());
    if (mlir::failed(reflectionPassManager.run(*reflectionModule)))
        return VERNON_STATUS_VERIFICATION_ERROR;
    mlir::FailureOr<std::string> builtReflection = buildReflection(*reflectionModule, *module);
    if (mlir::failed(builtReflection))
        return VERNON_STATUS_VERIFICATION_ERROR;
    reflection = std::move(*builtReflection);
    if (mlir::failed(mlir::vernon::materializeTensorViewProjections(*reflectionModule)))
        return VERNON_STATUS_VERIFICATION_ERROR;
    prepared.reset(new (std::nothrow) PreparedModule(std::move(reflectionModule)));
    if (!prepared) {
        diagnostics = "failed to allocate prepared compiler module";
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    return VERNON_STATUS_OK;
}

} // namespace vernon::compiler
