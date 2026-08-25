#include "mlir/Dialect/VernonProgram/Transforms/VernonProgramImplementation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/VernonProgram/IR/VernonProgram.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace mlir::vernon::program {
namespace {

bool isProgramGraph(Operation *operation) {
    auto function = operation->getParentOfType<func::FuncOp>();
    return function && function->hasAttr("vernon_program.graph");
}

FailureOr<int64_t> staticElementCount(Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape())
        return failure();
    int64_t count = 1;
    for (int64_t extent : tensor.getShape()) {
        if (extent < 0 || (extent != 0 && count > std::numeric_limits<int64_t>::max() / extent))
            return failure();
        count *= extent;
    }
    return std::max<int64_t>(count, 1);
}

FailureOr<SmallVector<int64_t, 3>> implementationGrid(StringRef implementation, Type resultType) {
    if (isa<TupleType>(resultType) && (implementation == "add" || implementation == "zeros"))
        return SmallVector<int64_t, 3>{1, 1, 1};
    auto tensor = dyn_cast<RankedTensorType>(resultType);
    if (!tensor || !tensor.hasStaticShape())
        return failure();
    if (implementation == "add" && tensor.getRank() <= 3) {
        SmallVector<int64_t, 3> grid{1, 1, 1};
        if (tensor.getRank() > 0)
            grid[0] = std::max<int64_t>(tensor.getShape().back(), 1);
        if (tensor.getRank() > 1)
            grid[1] = std::max<int64_t>(tensor.getShape()[tensor.getRank() - 2], 1);
        if (tensor.getRank() > 2)
            grid[2] = std::max<int64_t>(tensor.getShape()[tensor.getRank() - 3], 1);
        return grid;
    }
    FailureOr<int64_t> count = staticElementCount(resultType);
    if (failed(count))
        return failure();
    return SmallVector<int64_t, 3>{*count, 1, 1};
}

ArrayAttr names(OpBuilder &builder, StringRef prefix, unsigned count) {
    SmallVector<Attribute> values;
    values.reserve(count);
    for (unsigned index = 0; index < count; ++index)
        values.push_back(builder.getStringAttr((Twine(prefix) + Twine(index)).str()));
    return builder.getArrayAttr(values);
}

ArrayAttr operandNames(OpBuilder &builder, StringRef implementation, unsigned count) {
    if ((implementation == "add" || implementation == "matmul") && count == 2)
        return builder.getArrayAttr({builder.getStringAttr("left"), builder.getStringAttr("right")});
    if ((implementation == "transpose" || implementation == "cast" || implementation == "reduce_sum_to_shape") &&
        count == 1)
        return builder.getArrayAttr({builder.getStringAttr("source")});
    return names(builder, "operand", count);
}

ArrayAttr resultNames(OpBuilder &builder, unsigned count) {
    if (count == 1)
        return builder.getArrayAttr({builder.getStringAttr("output")});
    return names(builder, "result", count);
}

LogicalResult replaceWithBuiltin(Operation *operation, StringRef implementation, ValueRange operands = {},
                                 bool useSuppliedOperands = false) {
    if (operation->getNumResults() == 0)
        return operation->emitError("Program semantic implementation requires at least one result");
    FailureOr<SmallVector<int64_t, 3>> grid = implementationGrid(implementation, operation->getResult(0).getType());
    if (failed(grid))
        return operation->emitError("Program built-in implementation currently requires a static ranked Tensor result");

    OpBuilder builder(operation);
    OperationState state(operation->getLoc(), ComputeOp::getOperationName());
    state.addOperands(useSuppliedOperands ? operands : operation->getOperands());
    state.addTypes(operation->getResultTypes());
    state.addAttribute("callee", builder.getStringAttr((Twine("vernon.builtin.") + implementation).str()));
    state.addAttribute("grid", builder.getDenseI64ArrayAttr(*grid));
    state.addAttribute("features", builder.getArrayAttr({}));
    state.addAttribute(
        "operand_names",
        operandNames(builder, implementation, useSuppliedOperands ? operands.size() : operation->getNumOperands()));
    state.addAttribute("result_names", resultNames(builder, operation->getNumResults()));
    for (NamedAttribute attribute : operation->getAttrs())
        if (attribute.getName() != "name")
            state.addAttribute(attribute.getName(), attribute.getValue());
    Operation *replacement = builder.create(state);
    operation->replaceAllUsesWith(replacement->getResults());
    operation->erase();
    return success();
}

FailureOr<SmallVector<Value>> aggregateAddRoots(TupleCreateOp operation) {
    auto resultType = dyn_cast<TupleType>(operation.getResult().getType());
    if (!resultType || resultType.size() != operation.getElements().size())
        return failure();
    SmallVector<Value> roots;
    for (auto [index, element] : llvm::enumerate(operation.getElements())) {
        auto add = element.getDefiningOp<arith::AddFOp>();
        auto left = add ? add.getLhs().getDefiningOp<TupleGetOp>() : TupleGetOp{};
        auto right = add ? add.getRhs().getDefiningOp<TupleGetOp>() : TupleGetOp{};
        if (!left || !right || left.getIndex() != index || right.getIndex() != index)
            return failure();
        if (roots.empty())
            roots.assign({left.getInput(), right.getInput()});
        else if (roots[0] != left.getInput() || roots[1] != right.getInput())
            return failure();
    }
    return roots;
}

bool isAggregateZero(TupleCreateOp operation) {
    auto resultType = dyn_cast<TupleType>(operation.getResult().getType());
    return resultType && resultType.size() == operation.getElements().size() &&
           llvm::all_of(operation.getElements(), [](Value element) { return matchPattern(element, m_Zero()); });
}

void eraseDeadAggregateExpression(ValueRange values) {
    SmallVector<Operation *> worklist;
    for (Value value : values)
        if (Operation *operation = value.getDefiningOp())
            worklist.push_back(operation);
    while (!worklist.empty()) {
        Operation *operation = worklist.pop_back_val();
        if (!operation->use_empty() || !isa<TupleCreateOp, TupleGetOp, arith::AddFOp, arith::ConstantOp>(operation))
            continue;
        for (Value operand : operation->getOperands())
            if (Operation *producer = operand.getDefiningOp())
                worklist.push_back(producer);
        operation->erase();
    }
}

struct VernonProgramSelectImplementationsPass final
    : PassWrapper<VernonProgramSelectImplementationsPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonProgramSelectImplementationsPass)

    StringRef getArgument() const final { return "vernon-program-select-implementations"; }
    StringRef getDescription() const final {
        return "Map Program semantic tensor operators to concrete compute implementations";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, VernonDialect, VernonProgramDialect>();
    }

    void runOnOperation() override {
        SmallVector<Operation *> semanticOperations;
        getOperation().walk([&](Operation *operation) {
            if (!isProgramGraph(operation))
                return;
            if (auto intrinsic = dyn_cast<IntrinsicOp>(operation)) {
                static const llvm::StringSet<> supported = [] {
                    llvm::StringSet<> values;
                    for (StringRef name :
                         {"empty", "zeros", "add", "matmul", "transpose", "reduce_sum_to_shape", "cast", "construct"})
                        values.insert(name);
                    return values;
                }();
                if (isStorageAllocIntrinsic(operation))
                    return;
                if (supported.contains(intrinsic.getName()))
                    semanticOperations.push_back(operation);
                return;
            }
            if (auto tuple = dyn_cast<TupleCreateOp>(operation);
                tuple && (succeeded(aggregateAddRoots(tuple)) || isAggregateZero(tuple))) {
                semanticOperations.push_back(operation);
                return;
            }
            if (isa<arith::AddFOp, arith::ConstantOp>(operation) &&
                llvm::any_of(operation->getResultTypes(), [](Type type) { return isa<RankedTensorType>(type); }))
                semanticOperations.push_back(operation);
        });

        for (Operation *operation : semanticOperations) {
            StringRef name;
            SmallVector<Value> aggregateOperands;
            if (auto intrinsic = dyn_cast<IntrinsicOp>(operation))
                name = intrinsic.getName();
            else if (auto tuple = dyn_cast<TupleCreateOp>(operation)) {
                FailureOr<SmallVector<Value>> roots = aggregateAddRoots(tuple);
                if (succeeded(roots)) {
                    name = "add";
                    aggregateOperands = std::move(*roots);
                } else {
                    name = "zeros";
                }
            } else if (isa<arith::AddFOp>(operation))
                name = "add";
            else
                name = "constant";
            SmallVector<Value> aggregateExpressionOperands;
            if (isa<TupleCreateOp>(operation))
                llvm::append_range(aggregateExpressionOperands, operation->getOperands());
            if (failed(replaceWithBuiltin(operation, name, aggregateOperands, isa<TupleCreateOp>(operation)))) {
                signalPassFailure();
                return;
            }
            eraseDeadAggregateExpression(aggregateExpressionOperands);
        }

        bool unsupported = false;
        getOperation().walk([&](func::FuncOp function) {
            if (!function->hasAttr("vernon_program.graph"))
                return;
            function.walk([&](Operation *operation) {
                if (operation == function || isa<ComputeOp, GraphicsOp, func::ReturnOp>(operation) ||
                    isStorageAllocIntrinsic(operation))
                    return;
                operation->emitError("has no selected Program implementation");
                unsupported = true;
            });
        });
        if (unsupported)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonProgramSelectImplementationsPass() {
    return std::make_unique<VernonProgramSelectImplementationsPass>();
}

void registerVernonProgramSelectImplementationsPass() { PassRegistration<VernonProgramSelectImplementationsPass>(); }

} // namespace mlir::vernon::program
