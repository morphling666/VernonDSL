#include "compiler_python_bridge.h"

#include "compiler_frontend.h"
#include "compiler_internal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

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

mlir::LogicalResult appendLayoutTree(mlir::Type type, mlir::ModuleOp module, std::vector<ValueAbiNode> &nodes,
                                     llvm::ArrayRef<llvm::StringRef> logicalDtypes = {}) {
    mlir::FailureOr<mlir::vernon::ValueAbiLayout> layout = mlir::vernon::getValueAbiLayout(type, module, logicalDtypes);
    if (mlir::failed(layout))
        return mlir::failure();
    nodes.push_back({layout->size,
                     layout->alignment,
                     {layout->fieldOffsets.begin(), layout->fieldOffsets.end()},
                     layout->elementStride});

    if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
        return appendLayoutTree(tensor.getElementType(), module, nodes);
    if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(type))
        return appendLayoutTree(tensor.getElementType(), module, nodes);
    if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
        return appendLayoutTree(vector.getElementType(), module, nodes);
    if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type)) {
        for (mlir::Type element : tuple.getTypes())
            if (mlir::failed(appendLayoutTree(element, module, nodes)))
                return mlir::failure();
        return mlir::success();
    }
    if (auto structure = mlir::dyn_cast<mlir::vernon::StructType>(type)) {
        mlir::FailureOr<mlir::vernon::ResolvedStructFields> fields =
            mlir::vernon::resolveNamedStructFields(structure, module);
        if (mlir::failed(fields))
            return mlir::failure();
        for (const mlir::vernon::ResolvedStructField &field : fields->fields)
            if (mlir::failed(appendLayoutTree(field.type, module, nodes)))
                return mlir::failure();
    }
    return mlir::success();
}

VernonStringView viewOf(const std::string &value) { return {value.data(), value.size()}; }

} // namespace

struct VernonPythonValueAbiPlan {
    VernonStatus status{VERNON_STATUS_INTERNAL_ERROR};
    std::string diagnostics;
    std::vector<ValueAbiNode> nodes;
    std::vector<VernonPythonValueAbiNodeView> nodeViews;
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
    if (mlir::failed(appendLayoutTree(function.getArgumentTypes().front(), *parsed, plan->nodes, dtypes))) {
        plan->status = VERNON_STATUS_VERIFICATION_ERROR;
        plan->diagnostics = "type has no finite canonical Value ABI layout";
        plan->nodes.clear();
        return plan.release();
    }
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

} // extern "C"
