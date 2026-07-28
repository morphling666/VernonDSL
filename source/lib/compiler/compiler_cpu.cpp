#include "compiler_cpu.h"

#include "VernonCpuAbiWrapper.h"
#include "compiler_frontend.h"
#include "compiler_reflection.h"

#include "lld/Common/Driver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonCpuPipeline.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValueAbi.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallString.h"
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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
LLD_HAS_DRIVER(coff)
#elif defined(__APPLE__)
LLD_HAS_DRIVER(macho)
#else
LLD_HAS_DRIVER(elf)
#endif

namespace vernon::compiler {

class CpuExecutionState {
public:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::map<std::string, VernonCpuEntryPoint, std::less<>> entries;
};

void CpuExecutionStateDeleter::operator()(CpuExecutionState *state) const { delete state; }

namespace {

bool captureCpuAbiMetadata(mlir::ModuleOp module, std::vector<vernon::CpuAbiWrapperMetadata> &entries,
                           llvm::StringRef moduleHash, std::string &diagnostics) {
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
        if (!function->hasAttr("vernon.entry"))
            continue;

        vernon::CpuAbiWrapperMetadata metadata;
        metadata.internalFunctionSymbol = function.getSymName().str();
        metadata.exportedWrapperSymbol = "__vernon_cpu_" + moduleHash.str() + "_" + function.getSymName().str();
        metadata.argumentsSize = 0;
        metadata.requiresTextureCallbacks = false;
        function.walk([&](mlir::vernon::IntrinsicOp intrinsic) {
            metadata.requiresTextureCallbacks |= intrinsic.getName() == "texture_sample";
        });

        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            mlir::Type type = function.getArgumentTypes()[index];
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiLayout> layout =
                mlir::vernon::getPhysicalValueAbiLayout(type, module, mlir::vernon::PhysicalAbiProfile::HostValue);
            if (mlir::failed(layout)) {
                diagnostics = "unsupported CPU ABI argument type in entry '" + function.getSymName().str() + "'";
                return false;
            }
            metadata.argumentsSize = llvm::alignTo(metadata.argumentsSize, layout->alignment);

            vernon::CpuAbiArgumentPacking packing{metadata.argumentsSize,
                                                  layout->size,
                                                  mlir::isa<mlir::vernon::TensorViewType>(type)
                                                      ? vernon::CpuAbiArgumentKind::TensorView
                                                      : vernon::CpuAbiArgumentKind::Direct,
                                                  0,
                                                  {}};
            if (packing.kind == vernon::CpuAbiArgumentKind::TensorView) {
                auto view = mlir::cast<mlir::vernon::TensorViewType>(type);
                mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout =
                    mlir::vernon::getValueAbiLayout(view.getElementType(), module);
                if (mlir::failed(layout)) {
                    diagnostics =
                        "invalid aggregate TensorView layout in CPU entry '" + function.getSymName().str() + "'";
                    return false;
                }
                for (const mlir::vernon::ValueAbiLeaf &leaf : layout->leaves)
                    packing.tensorLeafElementSizes.push_back(
                        std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1));
                if (auto shape = attrs.getAs<mlir::DenseI64ArrayAttr>("vernon.tensor_shape")) {
                    uint64_t extent = 1;
                    for (int64_t dimension : shape.asArrayRef()) {
                        if (dimension < 0 || (dimension != 0 && extent > std::numeric_limits<uint64_t>::max() /
                                                                             static_cast<uint64_t>(dimension))) {
                            diagnostics = "invalid vernon.tensor_shape on CPU TensorView in "
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
            metadata.argumentsSize += layout->size;
        }

        if (function.getNumResults() > 1) {
            diagnostics = "CPU entry '" + function.getSymName().str() + "' has more than one result";
            return false;
        }
        if (function.getNumResults() == 0) {
            metadata.resultsSize = 0;
        } else {
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiLayout> layout = mlir::vernon::getPhysicalValueAbiLayout(
                function.getResultTypes()[0], module, mlir::vernon::PhysicalAbiProfile::HostValue);
            metadata.resultsSize = mlir::succeeded(layout) ? layout->size : 0;
        }
        if (function.getNumResults() != 0 && metadata.resultsSize == 0) {
            diagnostics = "unsupported CPU ABI result type in entry '" + function.getSymName().str() + "'";
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

std::string cpuObjectFilename(const llvm::Triple &triple) {
    return triple.isOSBinFormatCOFF() ? "module.obj" : "module.o";
}

bool emitCpuObject(llvm::Module &module, llvm::TargetMachine &targetMachine, std::string &object,
                   std::string &diagnostics) {
    std::unique_ptr<llvm::Module> codegenModule = llvm::CloneModule(module);
    llvm::SmallVector<char> bytes;
    llvm::raw_svector_ostream stream(bytes);
    llvm::legacy::PassManager passManager;
    if (targetMachine.addPassesToEmitFile(passManager, stream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
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

bool linkHostObjectBytes(llvm::StringRef object, std::string &library, std::string &diagnostics) {
    llvm::SmallString<128> directory;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory("vernon_cpu_link", directory)) {
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
    storage = {"lld-link", "/dll", "/noentry", "/out:" + outputPath.string(), objectPath.string()};
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
    storage = {"ld.lld", "-shared", objectPath.string(), "-o", outputPath.string()};
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
        diagnostics = linkerOutput.empty() ? "embedded LLD failed" : std::move(linkerOutput);
        cleanup();
        return false;
    }
    std::ifstream input(outputPath, std::ios::binary);
    library.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (!input || library.empty()) {
        diagnostics = "embedded LLD produced no readable native library";
        cleanup();
        return false;
    }
    cleanup();
    return true;
}

} // namespace

bool compileCpu(PreparedModule &prepared, const CpuCodegenOptions &options, std::vector<Artifact> &artifacts,
                std::string &reflection, std::string &diagnostics, CpuExecutionStatePtr &execution) {
    mlir::MLIRContext &context = prepared.context();
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    mlir::OwningOpRef<mlir::ModuleOp> sourceModule = prepared.clone();

    llvm::Expected<llvm::json::Value> parsedReflection = llvm::json::parse(reflection);
    llvm::json::Object *reflectionRoot = parsedReflection ? parsedReflection->getAsObject() : nullptr;
    std::optional<llvm::StringRef> moduleHash =
        reflectionRoot ? reflectionRoot->getString("module_hash") : std::nullopt;
    if (!moduleHash || moduleHash->empty()) {
        diagnostics = "CPU reflection is missing its module hash";
        return false;
    }
    std::vector<vernon::CpuAbiWrapperMetadata> entries;
    if (!captureCpuAbiMetadata(*sourceModule, entries, *moduleHash, diagnostics))
        return false;

    const bool hostTarget = options.targetTriple.empty();
    const std::string targetTriple =
        llvm::Triple::normalize(hostTarget ? llvm::sys::getDefaultTargetTriple() : options.targetTriple);
    const llvm::Triple parsedTriple(targetTriple);
    if (!parsedTriple.isArch64Bit()) {
        diagnostics = "CPU relocatable objects currently require a 64-bit target triple";
        return false;
    }
    std::string lookupError;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(parsedTriple, lookupError);
    if (!target) {
        diagnostics = "CPU target '" + targetTriple + "' is unavailable: " + lookupError;
        return false;
    }
    llvm::TargetOptions targetOptions;
    std::unique_ptr<llvm::TargetMachine> objectTargetMachine(
        target->createTargetMachine(parsedTriple, options.cpu, options.features, targetOptions, llvm::Reloc::PIC_));
    if (!objectTargetMachine) {
        diagnostics = "failed to create TargetMachine for '" + targetTriple + "'";
        return false;
    }
    const llvm::DataLayout dataLayout = objectTargetMachine->createDataLayout();

    sourceModule->getOperation()->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                                          mlir::StringAttr::get(&context, dataLayout.getStringRepresentation()));
    sourceModule->getOperation()->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                                          mlir::StringAttr::get(&context, targetTriple));
    mlir::PassManager passManager(&context);
    mlir::vernon::buildVernonCpuLoweringPipeline(passManager);
    if (mlir::failed(passManager.run(*sourceModule)))
        return false;

    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*sourceModule, *llvmContext);
    if (!llvmModule) {
        diagnostics = "failed to translate lowered CPU MLIR to LLVM IR";
        return false;
    }
    llvmModule->setDataLayout(dataLayout);
    llvmModule->setTargetTriple(llvm::Triple(targetTriple));
    if (llvm::Error error = vernon::defineCpuTextureSampleHelper(*llvmModule)) {
        diagnostics = llvm::toString(std::move(error));
        return false;
    }
    for (const vernon::CpuAbiWrapperMetadata &entry : entries) {
        if (llvm::Error error = vernon::emitCpuAbiWrapper(*llvmModule, entry)) {
            diagnostics = llvm::toString(std::move(error));
            return false;
        }
    }
    if (parsedTriple.isOSWindows() && !llvmModule->getNamedGlobal("_fltused"))
        new llvm::GlobalVariable(*llvmModule, llvm::Type::getInt32Ty(*llvmContext), true,
                                 llvm::GlobalValue::WeakAnyLinkage,
                                 llvm::ConstantInt::get(llvm::Type::getInt32Ty(*llvmContext), 0), "_fltused");
    if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
        diagnostics = "generated CPU LLVM IR failed verification";
        return false;
    }
    std::string object;
    if (!emitCpuObject(*llvmModule, *objectTargetMachine, object, diagnostics))
        return false;

    CpuExecutionStatePtr nextExecution;
    if (hostTarget) {
        auto jitTargetMachine = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!jitTargetMachine) {
            diagnostics = llvm::toString(jitTargetMachine.takeError());
            return false;
        }
        auto createdJit = llvm::orc::LLJITBuilder()
                              .setJITTargetMachineBuilder(std::move(*jitTargetMachine))
                              .setDataLayout(dataLayout)
                              .create();
        if (!createdJit) {
            diagnostics = llvm::toString(createdJit.takeError());
            return false;
        }
        nextExecution.reset(new CpuExecutionState());
        nextExecution->jit = std::move(*createdJit);
        if (llvm::Error error = nextExecution->jit->addIRModule(
                llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext)))) {
            diagnostics = llvm::toString(std::move(error));
            return false;
        }
        for (const vernon::CpuAbiWrapperMetadata &entry : entries) {
            auto symbol = nextExecution->jit->lookup(entry.exportedWrapperSymbol);
            if (!symbol) {
                diagnostics = llvm::toString(symbol.takeError());
                return false;
            }
            nextExecution->entries.emplace(entry.internalFunctionSymbol, symbol->toPtr<VernonCpuEntryPoint>());
        }
    }
    artifacts.clear();
    artifacts.push_back(Artifact{cpuObjectFilename(parsedTriple), std::move(object)});
    if (!setCpuReflectionSymbols(reflection, entries)) {
        diagnostics = "compiler produced invalid CPU reflection metadata";
        return false;
    }
    execution = std::move(nextExecution);
    return true;
}

bool linkHostObject(const void *object, size_t objectSize, Artifact &artifact, std::string &diagnostics) {
    std::string library;
    if (!linkHostObjectBytes(llvm::StringRef(static_cast<const char *>(object), objectSize), library, diagnostics))
        return false;
    artifact = Artifact{hostNativeLibraryFilename(), std::move(library)};
    return true;
}

VernonCpuEntryPoint findCpuEntry(const CpuExecutionState *execution, std::string_view name) {
    if (!execution)
        return nullptr;
    auto entry = execution->entries.find(name);
    return entry == execution->entries.end() ? nullptr : entry->second;
}

} // namespace vernon::compiler
