#include "compiler_host.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <mutex>

namespace vernon::compiler {

namespace {

void initializeTargets() {
    static std::once_flag once;
    std::call_once(once, [] {
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
        llvm::InitializeAllAsmParsers();
    });
}

bool supportedTriple(const llvm::Triple &triple) {
    switch (triple.getArch()) {
    case llvm::Triple::x86_64:
    case llvm::Triple::aarch64:
    case llvm::Triple::wasm32:
        return true;
    default:
        return false;
    }
}

} // namespace

std::string defaultHostTriple() { return llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple()); }

bool compileHostObject(llvm::StringRef ir, llvm::StringRef triple, llvm::StringRef cpu, llvm::StringRef features,
                       HostObject &result, std::string &diagnostics) {
    initializeTargets();
    const std::string normalized = triple.empty() ? defaultHostTriple() : llvm::Triple::normalize(triple);
    const llvm::Triple parsedTriple(normalized);
    if (!supportedTriple(parsedTriple)) {
        diagnostics = "Host objects target x86_64, aarch64, or wasm32, not '" + normalized + "'";
        return false;
    }
    std::string lookupError;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(parsedTriple, lookupError);
    if (!target) {
        diagnostics = "Host target '" + normalized + "' is unavailable: " + lookupError;
        return false;
    }
    llvm::TargetOptions options;
    std::unique_ptr<llvm::TargetMachine> machine(
        target->createTargetMachine(parsedTriple, cpu, features, options, llvm::Reloc::PIC_));
    if (!machine) {
        diagnostics = "failed to create TargetMachine for '" + normalized + "'";
        return false;
    }
    const std::string dataLayout = machine->createDataLayout().getStringRepresentation();

    llvm::LLVMContext context;
    llvm::SMDiagnostic parseError;
    // Load and store alignments the IR omits are filled in from this layout while parsing, so it must be the
    // target's from the start rather than replaced afterwards.
    std::unique_ptr<llvm::Module> module =
        llvm::parseAssembly(llvm::MemoryBufferRef(ir, "host.ll"), parseError, context, nullptr,
                            [&](llvm::StringRef, llvm::StringRef) -> std::optional<std::string> { return dataLayout; });
    if (!module) {
        llvm::raw_string_ostream stream(diagnostics);
        parseError.print("host.ll", stream);
        return false;
    }
    module->setTargetTriple(parsedTriple);
    {
        llvm::raw_string_ostream stream(diagnostics);
        if (llvm::verifyModule(*module, &stream))
            return false;
    }
    {
        llvm::LoopAnalysisManager loops;
        llvm::FunctionAnalysisManager functions;
        llvm::CGSCCAnalysisManager cgscc;
        llvm::ModuleAnalysisManager modules;
        llvm::PassBuilder builder(machine.get());
        builder.registerModuleAnalyses(modules);
        builder.registerCGSCCAnalyses(cgscc);
        builder.registerFunctionAnalyses(functions);
        builder.registerLoopAnalyses(loops);
        builder.crossRegisterProxies(loops, functions, cgscc, modules);
        llvm::ModulePassManager pipeline = builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
        pipeline.run(*module, modules);
    }
    {
        llvm::raw_string_ostream stream(result.optimizedIr);
        module->print(stream, nullptr);
    }
    llvm::SmallVector<char> bytes;
    llvm::raw_svector_ostream stream(bytes);
    llvm::legacy::PassManager emission;
    if (machine->addPassesToEmitFile(emission, stream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        diagnostics = "Host target '" + normalized + "' cannot emit relocatable objects";
        return false;
    }
    emission.run(*module);
    result.object.assign(bytes.begin(), bytes.end());
    result.triple = normalized;
    return true;
}

} // namespace vernon::compiler
