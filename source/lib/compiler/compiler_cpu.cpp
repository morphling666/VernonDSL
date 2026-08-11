#include "compiler_cpu.h"

#include "VernonCpuAbiWrapper.h"
#include "VernonCpuHalfConversion.h"
#include "VernonCpuWorkgroupABI.h"
#include "compiler_frontend.h"
#include "compiler_reflection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonCpuPipeline.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

void *registeredCpuHelperAddress(const VernonCpuRuntimeHelpersV1 *helpers, llvm::StringRef name) {
    if (!helpers)
        return nullptr;
    if (name == VERNON_CPU_WORKGROUP_ADDRESS_V1_SYMBOL)
        return reinterpret_cast<void *>(helpers->workgroup_address);
    if (name == VERNON_CPU_LANE_ADDRESS_V1_SYMBOL)
        return reinterpret_cast<void *>(helpers->lane_address);
    if (name == VERNON_CPU_WORKGROUP_BARRIER_V1_SYMBOL)
        return reinterpret_cast<void *>(helpers->workgroup_barrier);
    if (name == VERNON_CPU_WORKGROUP_IS_LEADER_V1_SYMBOL)
        return reinterpret_cast<void *>(helpers->workgroup_is_leader);
    return nullptr;
}

} // namespace

namespace vernon::compiler {

class CpuExecutionState {
public:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::map<std::string, VernonCpuEntryPoint, std::less<>> entries;
};

void CpuExecutionStateDeleter::operator()(CpuExecutionState *state) const { delete state; }

namespace {

mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> logicalDtypes(mlir::DictionaryAttr attributes) {
    llvm::SmallVector<llvm::StringRef> result;
    auto dtypes = attributes.getAs<mlir::ArrayAttr>("vernon.abi_leaf_dtypes");
    if (!dtypes)
        return result;
    for (mlir::Attribute attribute : dtypes) {
        auto dtype = mlir::dyn_cast<mlir::StringAttr>(attribute);
        if (!dtype)
            return mlir::failure();
        result.push_back(dtype.getValue());
    }
    return result;
}

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
        metadata.requiresPhases = false;
        function.walk([&](mlir::vernon::IntrinsicOp intrinsic) {
            metadata.requiresTextureCallbacks |= intrinsic.getName() == "texture_sample";
        });
        function.walk([&](mlir::vernon::BarrierOp) { metadata.requiresPhases = true; });

        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            mlir::Type type = function.getArgumentTypes()[index];
            mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
            if (attrs.get(mlir::vernon::kTensorDescriptorComponentAttrName))
                continue;
            mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> dtypes = logicalDtypes(attrs);
            if (mlir::failed(dtypes)) {
                diagnostics = "malformed CPU ABI dtype metadata in entry '" + function.getSymName().str() + "'";
                return false;
            }
            const bool tensorView = mlir::isa<mlir::vernon::TensorViewType>(type);
            std::optional<mlir::vernon::CpuCallPlan> valuePlan;
            if (!tensorView)
                if (mlir::FailureOr<mlir::vernon::CpuCallPlan> plan =
                        mlir::vernon::getCpuCallPlan(type, module, *dtypes);
                    mlir::succeeded(plan))
                    valuePlan = std::move(*plan);
            const vernon::CpuAbiArgumentKind kind = tensorView  ? vernon::CpuAbiArgumentKind::TensorView
                                                    : valuePlan ? vernon::CpuAbiArgumentKind::CanonicalValue
                                                                : vernon::CpuAbiArgumentKind::OpaqueScalar;
            uint64_t size = 0;
            uint64_t alignment = 0;
            if (valuePlan) {
                size = valuePlan->layout.size;
                alignment = valuePlan->layout.alignment;
            } else if (tensorView || mlir::vernon::isCpuOpaqueAbiType(type)) {
                mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> interface =
                    mlir::vernon::getBackendInterfaceAbiPlan(type, module, mlir::vernon::PhysicalAbiProfile::HostValue);
                const auto *resource =
                    mlir::succeeded(interface) ? std::get_if<mlir::vernon::ResourceBindingPlan>(&*interface) : nullptr;
                const auto *bytes =
                    mlir::succeeded(interface) ? std::get_if<mlir::vernon::ByteTransportPlan>(&*interface) : nullptr;
                if ((!resource || resource->handleSize == 0) && (!bytes || !bytes->root)) {
                    diagnostics =
                        "unsupported explicit CPU ABI contract in entry '" + function.getSymName().str() + "'";
                    return false;
                }
                size = resource ? resource->handleSize : bytes->root->size;
                alignment = resource ? resource->handleAlignment : bytes->root->alignment;
            } else {
                std::string spelling;
                llvm::raw_string_ostream stream(spelling);
                type.print(stream);
                diagnostics = "CPU ABI argument #" + std::to_string(index) + " of type " + stream.str() +
                              " has neither a canonical Value plan nor an opaque scalar contract in entry '" +
                              function.getSymName().str() + "'";
                return false;
            }
            metadata.argumentsSize = llvm::alignTo(metadata.argumentsSize, alignment);
            const auto builtin = attrs.getAs<mlir::StringAttr>("vernon.builtin");
            vernon::CpuAbiArgumentPacking packing{
                metadata.argumentsSize, size, kind, 0, builtin ? builtin.getValue().str() : std::string{}, {}, {}};
            if (packing.kind == vernon::CpuAbiArgumentKind::TensorView) {
                auto view = mlir::cast<mlir::vernon::TensorViewType>(type);
                packing.tensorRank = static_cast<uint32_t>(view.getShape().size());
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
            } else if (packing.kind == vernon::CpuAbiArgumentKind::CanonicalValue) {
                for (const mlir::vernon::CpuCallLane &lane : valuePlan->lanes) {
                    const mlir::vernon::ValueAbiLeaf &leaf = valuePlan->layout.leaves[lane.leafIndex];
                    const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
                    packing.callLanes.push_back({leaf.byteOffset + lane.scalarIndex * scalarSize, scalarSize});
                }
            } else {
                packing.callLanes.push_back({0, packing.size});
            }
            metadata.sourceArguments.push_back(packing);
            metadata.argumentsSize += packing.size;
        }

        if (function.getNumResults() > 1) {
            diagnostics = "CPU entry '" + function.getSymName().str() + "' has more than one result";
            return false;
        }
        if (function.getNumResults() == 0) {
            metadata.resultsSize = 0;
        } else {
            mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> dtypes = logicalDtypes(function.getResultAttrDict(0));
            if (mlir::failed(dtypes)) {
                diagnostics = "malformed CPU ABI result dtype metadata in entry '" + function.getSymName().str() + "'";
                return false;
            }
            mlir::FailureOr<mlir::vernon::CpuCallPlan> plan =
                mlir::vernon::getCpuCallPlan(function.getResultTypes()[0], module, *dtypes);
            if (mlir::failed(plan)) {
                diagnostics =
                    "CPU ABI result has no canonical Value plan in entry '" + function.getSymName().str() + "'";
                return false;
            }
            metadata.resultsSize = plan->layout.size;
            for (const mlir::vernon::CpuCallLane &lane : plan->lanes) {
                const mlir::vernon::ValueAbiLeaf &leaf = plan->layout.leaves[lane.leafIndex];
                const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
                metadata.resultCallLanes.push_back({leaf.byteOffset + lane.scalarIndex * scalarSize, scalarSize});
            }
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
    if (triple.isWasm())
        return "module.wasm.o";
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

} // namespace

CpuCompileResult compileCpu(PreparedModule &prepared, const CpuCodegenOptions &options,
                            std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics,
                            const VernonCpuRuntimeHelpersV1 *runtimeHelpers, CpuExecutionStatePtr &execution) {
    mlir::MLIRContext &context = prepared.context();
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    mlir::FailureOr<TargetPreparationResult> preparedTarget =
        prepareTargetModule(prepared, [&](mlir::ModuleOp module, TargetPreparationProvenance &) {
            mlir::PassManager preparation(&context);
            mlir::vernon::buildVernonCpuPreparationPipeline(preparation);
            return preparation.run(module);
        });
    if (mlir::failed(preparedTarget))
        return CpuCompileResult::VerificationFailure;
    mlir::OwningOpRef<mlir::ModuleOp> sourceModule = std::move(preparedTarget->module);
    mlir::FailureOr<std::string> targetReflection = buildReflection(
        *sourceModule, prepared.logicalReflection(), preparedTarget->entries, preparedTarget->provenance);
    if (mlir::failed(targetReflection))
        return CpuCompileResult::CodegenFailure;
    reflection = std::move(*targetReflection);

    llvm::Expected<llvm::json::Value> parsedReflection = llvm::json::parse(reflection);
    llvm::json::Object *reflectionRoot = parsedReflection ? parsedReflection->getAsObject() : nullptr;
    std::optional<llvm::StringRef> moduleHash =
        reflectionRoot ? reflectionRoot->getString("module_hash") : std::nullopt;
    if (!moduleHash || moduleHash->empty()) {
        diagnostics = "CPU reflection is missing its module hash";
        return CpuCompileResult::CodegenFailure;
    }
    std::vector<vernon::CpuAbiWrapperMetadata> entries;
    if (!captureCpuAbiMetadata(*sourceModule, entries, *moduleHash, diagnostics))
        return CpuCompileResult::CodegenFailure;

    const bool hostTarget = options.targetTriple.empty();
    const std::string targetTriple =
        llvm::Triple::normalize(hostTarget ? llvm::sys::getDefaultTargetTriple() : options.targetTriple);
    const llvm::Triple parsedTriple(targetTriple);
    const bool supportedWasmTarget = parsedTriple.getArch() == llvm::Triple::wasm32 && parsedTriple.isOSEmscripten();
    if (!parsedTriple.isArch64Bit() && !supportedWasmTarget) {
        diagnostics = "CPU relocatable objects require a 64-bit native target or wasm32-unknown-emscripten";
        return CpuCompileResult::CodegenFailure;
    }
    std::string lookupError;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(parsedTriple, lookupError);
    if (!target) {
        diagnostics = "CPU target '" + targetTriple + "' is unavailable: " + lookupError;
        return CpuCompileResult::CodegenFailure;
    }
    llvm::TargetOptions targetOptions;
    std::unique_ptr<llvm::TargetMachine> objectTargetMachine(
        target->createTargetMachine(parsedTriple, options.cpu, options.features, targetOptions, llvm::Reloc::PIC_));
    if (!objectTargetMachine) {
        diagnostics = "failed to create TargetMachine for '" + targetTriple + "'";
        return CpuCompileResult::CodegenFailure;
    }
    const llvm::DataLayout dataLayout = objectTargetMachine->createDataLayout();

    sourceModule->getOperation()->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                                          mlir::StringAttr::get(&context, dataLayout.getStringRepresentation()));
    sourceModule->getOperation()->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                                          mlir::StringAttr::get(&context, targetTriple));
    mlir::PassManager passManager(&context);
    mlir::vernon::buildVernonCpuLoweringPipeline(passManager);
    if (mlir::failed(passManager.run(*sourceModule)))
        return CpuCompileResult::CodegenFailure;

    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(*sourceModule, *llvmContext);
    if (!llvmModule) {
        diagnostics = "failed to translate lowered CPU MLIR to LLVM IR";
        return CpuCompileResult::CodegenFailure;
    }
    llvmModule->setDataLayout(dataLayout);
    llvmModule->setTargetTriple(llvm::Triple(targetTriple));
    if (llvm::Error error = vernon::defineCpuTextureSampleHelper(*llvmModule)) {
        diagnostics = llvm::toString(std::move(error));
        return CpuCompileResult::CodegenFailure;
    }
    for (const vernon::CpuAbiWrapperMetadata &entry : entries) {
        if (llvm::Error error = vernon::emitCpuAbiWrapper(*llvmModule, entry)) {
            diagnostics = llvm::toString(std::move(error));
            return CpuCompileResult::CodegenFailure;
        }
    }
    {
        llvm::LoopAnalysisManager loopAnalyses;
        llvm::FunctionAnalysisManager functionAnalyses;
        llvm::CGSCCAnalysisManager cgsccAnalyses;
        llvm::ModuleAnalysisManager moduleAnalyses;
        llvm::PassBuilder passBuilder(objectTargetMachine.get());
        passBuilder.registerModuleAnalyses(moduleAnalyses);
        passBuilder.registerCGSCCAnalyses(cgsccAnalyses);
        passBuilder.registerFunctionAnalyses(functionAnalyses);
        passBuilder.registerLoopAnalyses(loopAnalyses);
        passBuilder.crossRegisterProxies(loopAnalyses, functionAnalyses, cgsccAnalyses, moduleAnalyses);
        llvm::ModulePassManager optimization = passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
        optimization.run(*llvmModule, moduleAnalyses);
    }
    if (parsedTriple.isOSWindows() && !llvmModule->getNamedGlobal("_fltused"))
        new llvm::GlobalVariable(*llvmModule, llvm::Type::getInt32Ty(*llvmContext), true,
                                 llvm::GlobalValue::WeakAnyLinkage,
                                 llvm::ConstantInt::get(llvm::Type::getInt32Ty(*llvmContext), 0), "_fltused");
    if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
        diagnostics = "generated CPU LLVM IR failed verification";
        return CpuCompileResult::CodegenFailure;
    }
    if (llvm::Error error = vernon::lowerCpuHalfConversions(*llvmModule, *objectTargetMachine)) {
        diagnostics = llvm::toString(std::move(error));
        return CpuCompileResult::CodegenFailure;
    }
    if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
        diagnostics = "CPU f16 legalization produced invalid LLVM IR";
        return CpuCompileResult::CodegenFailure;
    }
    std::string object;
    if (!emitCpuObject(*llvmModule, *objectTargetMachine, object, diagnostics))
        return CpuCompileResult::CodegenFailure;

    CpuExecutionStatePtr nextExecution;
    bool runtimeHelpersAvailable = true;
    if (hostTarget)
        for (const char *helper : {VERNON_CPU_WORKGROUP_ADDRESS_V1_SYMBOL, VERNON_CPU_LANE_ADDRESS_V1_SYMBOL,
                                   VERNON_CPU_WORKGROUP_BARRIER_V1_SYMBOL, VERNON_CPU_WORKGROUP_IS_LEADER_V1_SYMBOL}) {
            llvm::Function *declaration = llvmModule->getFunction(helper);
            if (declaration && !declaration->use_empty() && !registeredCpuHelperAddress(runtimeHelpers, helper)) {
                runtimeHelpersAvailable = false;
                break;
            }
        }
    if (hostTarget && runtimeHelpersAvailable) {
        auto jitTargetMachine = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!jitTargetMachine) {
            diagnostics = llvm::toString(jitTargetMachine.takeError());
            return CpuCompileResult::CodegenFailure;
        }
        auto createdJit = llvm::orc::LLJITBuilder()
                              .setJITTargetMachineBuilder(std::move(*jitTargetMachine))
                              .setDataLayout(dataLayout)
                              .create();
        if (!createdJit) {
            diagnostics = llvm::toString(createdJit.takeError());
            return CpuCompileResult::CodegenFailure;
        }
        nextExecution.reset(new CpuExecutionState());
        nextExecution->jit = std::move(*createdJit);
        llvm::orc::SymbolMap helperSymbols;
        for (const char *helper : {VERNON_CPU_WORKGROUP_ADDRESS_V1_SYMBOL, VERNON_CPU_LANE_ADDRESS_V1_SYMBOL,
                                   VERNON_CPU_WORKGROUP_BARRIER_V1_SYMBOL, VERNON_CPU_WORKGROUP_IS_LEADER_V1_SYMBOL}) {
            llvm::Function *declaration = llvmModule->getFunction(helper);
            if (!declaration || declaration->use_empty())
                continue;
            void *address = registeredCpuHelperAddress(runtimeHelpers, helper);
            if (!address) {
                diagnostics = std::string("CPU JIT requires runtime helper '") + helper +
                              "' but no explicit helper address is available";
                return CpuCompileResult::CodegenFailure;
            }
            helperSymbols[nextExecution->jit->mangleAndIntern(helper)] = {llvm::orc::ExecutorAddr::fromPtr(address),
                                                                          llvm::JITSymbolFlags::Exported};
        }
        if (!helperSymbols.empty())
            if (llvm::Error error = nextExecution->jit->getMainJITDylib().define(
                    llvm::orc::absoluteSymbols(std::move(helperSymbols)))) {
                diagnostics = llvm::toString(std::move(error));
                return CpuCompileResult::CodegenFailure;
            }
        auto processSymbols =
            llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(dataLayout.getGlobalPrefix());
        if (!processSymbols) {
            diagnostics = llvm::toString(processSymbols.takeError());
            return CpuCompileResult::CodegenFailure;
        }
        nextExecution->jit->getMainJITDylib().addGenerator(std::move(*processSymbols));
        if (llvm::Error error = nextExecution->jit->addObjectFile(
                llvm::MemoryBuffer::getMemBufferCopy(object, cpuObjectFilename(parsedTriple)))) {
            diagnostics = llvm::toString(std::move(error));
            return CpuCompileResult::CodegenFailure;
        }
        for (const vernon::CpuAbiWrapperMetadata &entry : entries) {
            auto symbol = nextExecution->jit->lookup(entry.exportedWrapperSymbol);
            if (!symbol) {
                diagnostics = llvm::toString(symbol.takeError());
                return CpuCompileResult::CodegenFailure;
            }
            VernonCpuEntryPoint entryPoint = symbol->toPtr<VernonCpuEntryPoint>();
            nextExecution->entries.emplace(entry.internalFunctionSymbol, entryPoint);
        }
    }
    artifacts.clear();
    artifacts.push_back(Artifact{cpuObjectFilename(parsedTriple), std::move(object)});
    if (!setCpuReflectionSymbols(reflection, entries)) {
        diagnostics = "compiler produced invalid CPU reflection metadata";
        return CpuCompileResult::CodegenFailure;
    }
    execution = std::move(nextExecution);
    return CpuCompileResult::Success;
}

VernonCpuEntryPoint findCpuEntry(const CpuExecutionState *execution, std::string_view name) {
    if (!execution)
        return nullptr;
    auto entry = execution->entries.find(name);
    return entry == execution->entries.end() ? nullptr : entry->second;
}

} // namespace vernon::compiler
