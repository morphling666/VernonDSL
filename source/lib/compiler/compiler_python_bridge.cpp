#include "compiler_python_bridge.h"

#include "compiler_frontend.h"
#include "compiler_internal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStructuredVjp.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace {

struct ValueAbiNode {
    uint64_t size{};
    uint64_t alignment{};
    std::vector<uint64_t> fieldOffsets;
    std::optional<uint64_t> elementStride;
};

struct FrontendDeleter {
    void operator()(vernon::compiler::CompilerFrontend *frontend) const {
        vernon::compiler::destroyCompilerFrontend(frontend);
    }
};

using FrontendPtr = std::unique_ptr<vernon::compiler::CompilerFrontend, FrontendDeleter>;

FrontendPtr &planningFrontend() {
    static thread_local FrontendPtr frontend(vernon::compiler::createCompilerFrontend());
    return frontend;
}

void appendLayoutTree(const mlir::vernon::CanonicalAbiNode &node, std::vector<ValueAbiNode> &nodes) {
    nodes.push_back(
        {node.size, node.alignment, {node.childOffsets.begin(), node.childOffsets.end()}, node.elementStride});
    for (const std::shared_ptr<const mlir::vernon::CanonicalAbiNode> &child : node.children)
        appendLayoutTree(*child, nodes);
}

VernonStringView viewOf(const std::string &value) { return {value.data(), value.size()}; }

} // namespace

struct VernonPythonValueAbiPlan {
    VernonStatus status{VERNON_STATUS_INTERNAL_ERROR};
    std::string diagnostics;
    std::vector<ValueAbiNode> nodes;
    std::vector<VernonPythonValueAbiNodeView> nodeViews;
};

struct VernonPythonStructuredVjp {
    VernonStatus status{VERNON_STATUS_INTERNAL_ERROR};
    std::string diagnostics;
    std::string forwardModule;
    std::string backwardModule;
    uint64_t tapeBytes{};
    uint64_t activeOperationCount{};
    uint64_t recomputationCost{};
    std::vector<std::string> derivativeRules;
    std::vector<VernonStringView> derivativeRuleViews;
};

extern "C" {

VernonPythonValueAbiPlan *vernonCompilerPlanPythonValueAbi(VernonStringView module,
                                                           const VernonStringView *logicalDtypes,
                                                           size_t logicalDtypeCount) {
    std::unique_ptr<VernonPythonValueAbiPlan> plan(new (std::nothrow) VernonPythonValueAbiPlan());
    if (!plan)
        return nullptr;
    if ((!module.data && module.size != 0) || (!logicalDtypes && logicalDtypeCount != 0)) {
        plan->status = VERNON_STATUS_INVALID_ARGUMENT;
        plan->diagnostics = "Value ABI planner input is invalid";
        return plan.release();
    }
    for (size_t index = 0; index < logicalDtypeCount; ++index) {
        if (!logicalDtypes[index].data && logicalDtypes[index].size != 0) {
            plan->status = VERNON_STATUS_INVALID_ARGUMENT;
            plan->diagnostics = "Value ABI planner dtype is invalid";
            return plan.release();
        }
    }

    FrontendPtr &frontend = planningFrontend();
    if (!frontend) {
        plan->diagnostics = "cannot create compiler frontend for Value ABI planning";
        return plan.release();
    }
    mlir::MLIRContext &context = vernon::compiler::compilerMlirContext(*frontend);
    mlir::ScopedDiagnosticHandler diagnostics(&context, [&](mlir::Diagnostic &diagnostic) {
        vernon::compiler::appendDiagnostic(plan->diagnostics, diagnostic);
        return mlir::success();
    });
    mlir::OwningOpRef<mlir::ModuleOp> parsed =
        mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(module.data, module.size), &context);
    if (!parsed) {
        plan->status = VERNON_STATUS_PARSE_ERROR;
        if (plan->diagnostics.empty())
            plan->diagnostics = "cannot parse Value ABI planning module";
        return plan.release();
    }
    mlir::func::FuncOp function = parsed->lookupSymbol<mlir::func::FuncOp>("__vernon_plan_value_abi");
    if (!function || function.getNumArguments() != 1) {
        plan->status = VERNON_STATUS_INVALID_ARGUMENT;
        plan->diagnostics = "Value ABI planning module must contain one planner argument";
        return plan.release();
    }

    std::vector<llvm::StringRef> dtypes;
    dtypes.reserve(logicalDtypeCount);
    for (size_t index = 0; index < logicalDtypeCount; ++index)
        dtypes.emplace_back(logicalDtypes[index].data, logicalDtypes[index].size);
    mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout =
        mlir::vernon::getValueAbiLayout(function.getArgumentTypes().front(), *parsed, dtypes);
    if (mlir::failed(layout) || !layout->tree.root) {
        plan->status = VERNON_STATUS_VERIFICATION_ERROR;
        plan->diagnostics = "type has no finite canonical Value ABI layout";
        plan->nodes.clear();
        return plan.release();
    }
    appendLayoutTree(*layout->tree.root, plan->nodes);
    plan->nodeViews.reserve(plan->nodes.size());
    for (const ValueAbiNode &node : plan->nodes) {
        plan->nodeViews.push_back({node.size, node.alignment, node.fieldOffsets.data(), node.fieldOffsets.size(),
                                   node.elementStride.value_or(0),
                                   static_cast<uint8_t>(node.elementStride.has_value())});
    }
    plan->status = VERNON_STATUS_OK;
    return plan.release();
}

void vernonCompilerDestroyPythonValueAbiPlan(VernonPythonValueAbiPlan *plan) { delete plan; }

VernonPythonValueAbiPlanView vernonCompilerGetPythonValueAbiPlanView(const VernonPythonValueAbiPlan *plan) {
    if (!plan)
        return {VERNON_STATUS_INVALID_ARGUMENT, {}, nullptr, 0};
    return {plan->status, viewOf(plan->diagnostics), plan->nodeViews.data(), plan->nodeViews.size()};
}

VernonPythonStructuredVjp *vernonCompilerBuildPythonStructuredVjp(VernonStringView module, VernonStringView entry,
                                                                  const VernonStringView *wrtPaths, size_t wrtPathCount,
                                                                  const VernonStringView *outputPaths,
                                                                  size_t outputPathCount,
                                                                  VernonStringView forwardSymbol,
                                                                  VernonStringView backwardSymbol) {
    std::unique_ptr<VernonPythonStructuredVjp> result(new (std::nothrow) VernonPythonStructuredVjp());
    if (!result)
        return nullptr;
    auto present = [](VernonStringView value) { return value.data && value.size; };
    if (!present(module) || !present(entry) || !present(forwardSymbol) || !present(backwardSymbol) ||
        (!wrtPaths && wrtPathCount != 0) || wrtPathCount == 0 || (!outputPaths && outputPathCount != 0) ||
        outputPathCount == 0) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "structured VJP bridge input is invalid";
        return result.release();
    }
    for (size_t index = 0; index < wrtPathCount; ++index) {
        if (!present(wrtPaths[index])) {
            result->status = VERNON_STATUS_INVALID_ARGUMENT;
            result->diagnostics = "structured VJP wrt path is invalid";
            return result.release();
        }
    }
    for (size_t index = 0; index < outputPathCount; ++index) {
        if (!present(outputPaths[index])) {
            result->status = VERNON_STATUS_INVALID_ARGUMENT;
            result->diagnostics = "structured VJP output path is invalid";
            return result.release();
        }
    }

    FrontendPtr &frontend = planningFrontend();
    if (!frontend) {
        result->diagnostics = "cannot create compiler frontend for structured VJP";
        return result.release();
    }
    mlir::MLIRContext &context = vernon::compiler::compilerMlirContext(*frontend);
    mlir::ScopedDiagnosticHandler diagnostics(&context, [&](mlir::Diagnostic &diagnostic) {
        vernon::compiler::appendDiagnostic(result->diagnostics, diagnostic);
        return mlir::success();
    });
    mlir::OwningOpRef<mlir::ModuleOp> parsed =
        mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(module.data, module.size), &context);
    if (!parsed) {
        result->status = VERNON_STATUS_PARSE_ERROR;
        return result.release();
    }
    if (mlir::failed(mlir::verify(*parsed))) {
        result->status = VERNON_STATUS_VERIFICATION_ERROR;
        return result.release();
    }
    mlir::PassManager normalization(&context);
    normalization.addPass(mlir::vernon::createVernonInlineHelpersPass());
    if (mlir::failed(normalization.run(*parsed))) {
        result->status = VERNON_STATUS_VERIFICATION_ERROR;
        if (result->diagnostics.empty())
            result->diagnostics = "structured VJP helper normalization failed";
        return result.release();
    }
    std::string entryName(entry.data, entry.size);
    mlir::func::FuncOp primal = parsed->lookupSymbol<mlir::func::FuncOp>(entryName);
    if (!primal) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "structured VJP entry function was not found";
        return result.release();
    }
    mlir::vernon::StructuredVjpOptions options;
    options.forwardSymbol.assign(forwardSymbol.data, forwardSymbol.size);
    options.backwardSymbol.assign(backwardSymbol.data, backwardSymbol.size);
    for (size_t index = 0; index < wrtPathCount; ++index)
        options.wrtPaths.emplace_back(wrtPaths[index].data, wrtPaths[index].size);
    for (size_t index = 0; index < outputPathCount; ++index)
        options.outputPaths.emplace_back(outputPaths[index].data, outputPaths[index].size);
    mlir::FailureOr<mlir::vernon::StructuredVjpResult> transformed = mlir::vernon::buildStructuredVjp(primal, options);
    if (mlir::failed(transformed)) {
        result->status = VERNON_STATUS_VERIFICATION_ERROR;
        return result.release();
    }
    result->tapeBytes = transformed->tapeBytes;
    const auto activeOperationCount =
        transformed->backward->getAttrOfType<mlir::IntegerAttr>("vernon.ad.active_operation_count");
    const auto recomputationCost =
        transformed->backward->getAttrOfType<mlir::IntegerAttr>("vernon.ad.recomputation_cost");
    if (!activeOperationCount || !recomputationCost) {
        result->status = VERNON_STATUS_INTERNAL_ERROR;
        result->diagnostics = "structured VJP result omitted native telemetry";
        return result.release();
    }
    result->activeOperationCount = activeOperationCount.getValue().getZExtValue();
    result->recomputationCost = recomputationCost.getValue().getZExtValue();
    result->derivativeRules.assign(transformed->derivativeRules.begin(), transformed->derivativeRules.end());
    result->derivativeRuleViews.reserve(result->derivativeRules.size());
    for (const std::string &rule : result->derivativeRules)
        result->derivativeRuleViews.push_back(viewOf(rule));
    auto printProfile = [&](llvm::StringRef keptSymbol, llvm::StringRef removedSymbol,
                            llvm::StringRef profileName) -> mlir::FailureOr<std::string> {
        mlir::OwningOpRef<mlir::ModuleOp> profile(mlir::cast<mlir::ModuleOp>(parsed->clone()));
        mlir::func::FuncOp removed = profile->lookupSymbol<mlir::func::FuncOp>(removedSymbol);
        mlir::func::FuncOp original = profile->lookupSymbol<mlir::func::FuncOp>(entryName);
        mlir::func::FuncOp kept = profile->lookupSymbol<mlir::func::FuncOp>(keptSymbol);
        if (!removed || !original || !kept)
            return mlir::failure();
        removed.erase();
        original.erase();
        profile->getOperation()->setAttr("vernon.ad_profile", mlir::StringAttr::get(&context, profileName));
        kept->setAttr("vernon.entry", mlir::UnitAttr::get(&context));
        if (mlir::failed(mlir::verify(*profile)))
            return mlir::failure();
        std::string text;
        llvm::raw_string_ostream stream(text);
        profile->print(stream, mlir::OpPrintingFlags().enableDebugInfo(false));
        stream << '\n';
        return text;
    };
    mlir::FailureOr<std::string> forward =
        printProfile(options.forwardSymbol, options.backwardSymbol, "forward_with_tape");
    mlir::FailureOr<std::string> backward = printProfile(options.backwardSymbol, options.forwardSymbol, "backward");
    if (mlir::failed(forward) || mlir::failed(backward)) {
        result->status = VERNON_STATUS_VERIFICATION_ERROR;
        result->diagnostics += "cannot isolate generated structured VJP profiles";
        return result.release();
    }
    result->forwardModule = std::move(*forward);
    result->backwardModule = std::move(*backward);
    result->status = VERNON_STATUS_OK;
    return result.release();
}

VernonStatus vernonCompilerFinalizePythonStructuredVjp(VernonPythonStructuredVjp *result,
                                                       VernonStringView profilesIdentity) {
    if (!result || !profilesIdentity.data || !profilesIdentity.size)
        return VERNON_STATUS_INVALID_ARGUMENT;
    FrontendPtr &frontend = planningFrontend();
    if (!frontend)
        return VERNON_STATUS_INTERNAL_ERROR;
    mlir::MLIRContext &context = vernon::compiler::compilerMlirContext(*frontend);
    result->diagnostics.clear();
    mlir::ScopedDiagnosticHandler diagnostics(&context, [&](mlir::Diagnostic &diagnostic) {
        vernon::compiler::appendDiagnostic(result->diagnostics, diagnostic);
        return mlir::success();
    });
    const std::string identity(profilesIdentity.data, profilesIdentity.size);
    auto finalize = [&](const std::string &text, std::string &finalized) -> mlir::LogicalResult {
        mlir::OwningOpRef<mlir::ModuleOp> profile = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        if (!profile)
            return mlir::failure();
        profile->getOperation()->setAttr("vernon.ad_profiles_identity", mlir::StringAttr::get(&context, identity));
        if (mlir::failed(mlir::verify(*profile)))
            return mlir::failure();
        llvm::raw_string_ostream stream(finalized);
        profile->print(stream, mlir::OpPrintingFlags().enableDebugInfo(false));
        stream << '\n';
        return mlir::success();
    };
    std::string forward;
    std::string backward;
    if (mlir::failed(finalize(result->forwardModule, forward)) ||
        mlir::failed(finalize(result->backwardModule, backward))) {
        result->status = VERNON_STATUS_VERIFICATION_ERROR;
        return result->status;
    }
    result->forwardModule = std::move(forward);
    result->backwardModule = std::move(backward);
    result->status = VERNON_STATUS_OK;
    return result->status;
}

void vernonCompilerDestroyPythonStructuredVjp(VernonPythonStructuredVjp *result) { delete result; }

VernonPythonStructuredVjpView vernonCompilerGetPythonStructuredVjpView(const VernonPythonStructuredVjp *result) {
    if (!result)
        return {VERNON_STATUS_INVALID_ARGUMENT, {}, {}, {}, 0, 0, 0, nullptr, 0};
    return {result->status,
            viewOf(result->diagnostics),
            viewOf(result->forwardModule),
            viewOf(result->backwardModule),
            result->tapeBytes,
            result->activeOperationCount,
            result->recomputationCost,
            result->derivativeRuleViews.data(),
            result->derivativeRuleViews.size()};
}

} // extern "C"
