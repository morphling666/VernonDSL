#include "compiler_frontend.h"

#include "compiler_reflection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

#include <algorithm>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <utility>

namespace vernon::compiler {

namespace {

constexpr llvm::StringLiteral kLogicalArgumentOriginAttr = "vernon.internal.logical_argument_origin";
constexpr llvm::StringLiteral kLogicalResultOriginAttr = "vernon.internal.logical_result_origin";

const LogicalEntryModel *findLogicalEntry(const LogicalReflectionModel &logical, llvm::StringRef name) {
    for (const LogicalEntryModel &entry : logical.entries)
        if (entry.name == name)
            return &entry;
    return nullptr;
}

} // namespace

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

mlir::MLIRContext &compilerMlirContext(CompilerFrontend &frontend) { return frontend.context; }

PreparedModule::PreparedModule(mlir::OwningOpRef<mlir::ModuleOp> module, LogicalReflectionModel logicalReflection)
    : module_(std::move(module)), logicalReflection_(std::move(logicalReflection)) {}

mlir::MLIRContext &PreparedModule::context() { return *module_->getContext(); }

mlir::OwningOpRef<mlir::ModuleOp> PreparedModule::clone() {
    return mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module_->clone()));
}

mlir::ModuleOp PreparedModule::logicalModule() const { return *module_; }

const LogicalReflectionModel &PreparedModule::logicalReflection() const { return logicalReflection_; }

TargetPreparationProvenance::TargetPreparationProvenance(mlir::ModuleOp module, const LogicalReflectionModel &logical)
    : logical_(&logical) {
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        const LogicalEntryModel *entry = findLogicalEntry(logical, function.getSymName());
        if (!entry)
            continue;
        mlir::Builder builder(function.getContext());
        for (unsigned index = 0; index < std::min<unsigned>(function.getNumArguments(), entry->arguments.size());
             ++index)
            function.setArgAttr(index, kLogicalArgumentOriginAttr, builder.getI64IntegerAttr(index));
        for (unsigned index = 0; index < std::min<unsigned>(function.getNumResults(), entry->results.size()); ++index)
            function.setResultAttr(index, kLogicalResultOriginAttr, builder.getI64IntegerAttr(index));
    }
}

mlir::LogicalResult TargetPreparationProvenance::mapArgument(mlir::func::FuncOp function, unsigned physicalIndex,
                                                             unsigned logicalIndex) {
    const LogicalEntryModel *entry = logical_ ? findLogicalEntry(*logical_, function.getSymName()) : nullptr;
    if (!entry || physicalIndex >= function.getNumArguments() || logicalIndex >= entry->arguments.size())
        return function.emitError("cannot record logical origin for target-prepared argument");
    function.setArgAttr(physicalIndex, kLogicalArgumentOriginAttr,
                        mlir::Builder(function.getContext()).getI64IntegerAttr(logicalIndex));
    return mlir::success();
}

mlir::LogicalResult TargetPreparationProvenance::mapResult(mlir::func::FuncOp function, unsigned physicalIndex,
                                                           unsigned logicalIndex) {
    const LogicalEntryModel *entry = logical_ ? findLogicalEntry(*logical_, function.getSymName()) : nullptr;
    if (!entry || physicalIndex >= function.getNumResults() || logicalIndex >= entry->results.size())
        return function.emitError("cannot record logical origin for target-prepared result");
    function.setResultAttr(physicalIndex, kLogicalResultOriginAttr,
                           mlir::Builder(function.getContext()).getI64IntegerAttr(logicalIndex));
    return mlir::success();
}

mlir::FailureOr<std::vector<PhysicalEntryProvenance>> TargetPreparationProvenance::resolve(mlir::ModuleOp module) {
    std::vector<PhysicalEntryProvenance> resolved;
    bool invalid = false;
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        if (!function->getAttrOfType<mlir::StringAttr>("vernon.stage"))
            continue;
        PhysicalEntryProvenance entry;
        entry.name = function.getSymName().str();
        entry.arguments.resize(function.getNumArguments());
        entry.results.resize(function.getNumResults());
        const LogicalEntryModel *logicalEntry = logical_ ? findLogicalEntry(*logical_, function.getSymName()) : nullptr;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            mlir::IntegerAttr origin = function.getArgAttrOfType<mlir::IntegerAttr>(index, kLogicalArgumentOriginAttr);
            if (!origin)
                if (auto owner = function.getArgAttrOfType<mlir::IntegerAttr>(
                        index, mlir::vernon::kTensorDescriptorOwnerAttrName);
                    owner && owner.getInt() >= 0 && static_cast<uint64_t>(owner.getInt()) < function.getNumArguments())
                    origin = function.getArgAttrOfType<mlir::IntegerAttr>(static_cast<unsigned>(owner.getInt()),
                                                                          kLogicalArgumentOriginAttr);
            if (origin) {
                const int64_t logicalIndex = origin.getInt();
                if (!logicalEntry || logicalIndex < 0 ||
                    static_cast<uint64_t>(logicalIndex) >= logicalEntry->arguments.size()) {
                    function.emitError() << "argument #" << index << " has an invalid logical provenance index";
                    invalid = true;
                } else {
                    const LogicalValueModel &logicalValue = logicalEntry->arguments[logicalIndex];
                    entry.arguments[index] =
                        LogicalValueOrigin{static_cast<unsigned>(logicalIndex), logicalValue.sourcePath};
                }
            }
            function.removeArgAttr(index, kLogicalArgumentOriginAttr);
        }
        for (unsigned index = 0; index < function.getNumResults(); ++index) {
            if (auto origin = function.getResultAttrOfType<mlir::IntegerAttr>(index, kLogicalResultOriginAttr)) {
                const int64_t logicalIndex = origin.getInt();
                if (!logicalEntry || logicalIndex < 0 ||
                    static_cast<uint64_t>(logicalIndex) >= logicalEntry->results.size()) {
                    function.emitError() << "result #" << index << " has an invalid logical provenance index";
                    invalid = true;
                } else {
                    const LogicalValueModel &logicalValue = logicalEntry->results[logicalIndex];
                    entry.results[index] =
                        LogicalValueOrigin{static_cast<unsigned>(logicalIndex), logicalValue.sourcePath};
                }
            }
            function.removeResultAttr(index, kLogicalResultOriginAttr);
        }
        resolved.push_back(std::move(entry));
    }
    if (invalid)
        return mlir::failure();
    return resolved;
}

mlir::LogicalResult preparePortableTargetModule(mlir::ModuleOp module, TargetPreparationProvenance &) {
    mlir::PassManager passManager(module.getContext());
    passManager.addPass(mlir::vernon::createVernonInlineHelpersPass());
    if (mlir::failed(passManager.run(module)))
        return mlir::failure();
    return mlir::vernon::materializeTensorViewProjections(module);
}

mlir::FailureOr<TargetPreparationResult> prepareTargetModule(PreparedModule &prepared, const TargetPreparer &preparer) {
    TargetPreparationResult result;
    result.module = prepared.clone();
    TargetPreparationProvenance provenance(*result.module, prepared.logicalReflection());
    if (!preparer || mlir::failed(preparer(*result.module, provenance)))
        return mlir::failure();
    mlir::FailureOr<std::vector<PhysicalEntryProvenance>> resolved = provenance.resolve(*result.module);
    if (mlir::failed(resolved))
        return mlir::failure();
    result.provenance = std::move(*resolved);
    result.entries = buildPhysicalEntryModels(*result.module, result.provenance);
    return result;
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
    mlir::FailureOr<LogicalReflectionModel> logicalReflection = buildLogicalReflectionModel(*module);
    if (mlir::failed(logicalReflection))
        return VERNON_STATUS_VERIFICATION_ERROR;
    TargetPreparationProvenance provenance(*reflectionModule, *logicalReflection);
    mlir::FailureOr<std::vector<PhysicalEntryProvenance>> resolved = provenance.resolve(*reflectionModule);
    if (mlir::failed(resolved))
        return VERNON_STATUS_VERIFICATION_ERROR;
    std::vector<PhysicalEntryModel> physicalEntries = buildPhysicalEntryModels(*reflectionModule, *resolved);
    mlir::FailureOr<std::string> builtReflection =
        buildReflection(*reflectionModule, *logicalReflection, physicalEntries, *resolved);
    if (mlir::failed(builtReflection))
        return VERNON_STATUS_VERIFICATION_ERROR;
    reflection = std::move(*builtReflection);
    mlir::OwningOpRef<mlir::ModuleOp> targetNeutralModule(mlir::cast<mlir::ModuleOp>(module->clone()));
    prepared.reset(new (std::nothrow) PreparedModule(std::move(targetNeutralModule), std::move(*logicalReflection)));
    if (!prepared) {
        diagnostics = "failed to allocate prepared compiler module";
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    return VERNON_STATUS_OK;
}

} // namespace vernon::compiler
