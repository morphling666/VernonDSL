#include "mlir/Dialect/Vernon/Transforms/VernonStructuredVjp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffTapePlanning.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::vernon {
namespace {

constexpr StringLiteral kEntryAttr = "vernon.entry";
constexpr StringLiteral kStageAttr = "vernon.stage";

Value createTuple(OpBuilder &builder, Location location, TupleType type, ValueRange values) {
    OperationState state(location, TupleCreateOp::getOperationName());
    state.addOperands(values);
    state.addTypes(type);
    return builder.create(state)->getResult(0);
}

Value getTupleElement(OpBuilder &builder, Location location, Value tuple, unsigned index, Type type) {
    OperationState state(location, TupleGetOp::getOperationName());
    state.addOperands(tuple);
    state.addTypes(type);
    state.addAttribute("index", builder.getI64IntegerAttr(index));
    return builder.create(state)->getResult(0);
}

Value createZero(OpBuilder &builder, Location location, Type type) {
    auto floatType = dyn_cast<FloatType>(type);
    if (!floatType)
        return {};
    return arith::ConstantOp::create(builder, location, builder.getFloatAttr(floatType, 0.0));
}

void copyProfileFunctionAttrs(func::FuncOp source, func::FuncOp target) {
    for (NamedAttribute attribute : source->getAttrs()) {
        StringRef name = attribute.getName().strref();
        if (name == SymbolTable::getSymbolAttrName() || name == "function_type" ||
            name == source.getArgAttrsAttrName() || name == source.getResAttrsAttrName())
            continue;
        target->setAttr(attribute.getName(), attribute.getValue());
    }
    target->setAttr(kEntryAttr, UnitAttr::get(target.getContext()));
    target->setAttr(kStageAttr, StringAttr::get(target.getContext(), "compute"));
}

DictionaryAttr makeInterfaceAttrs(MLIRContext *context, StringRef interfaceName, StringRef sourceName,
                                  ArrayRef<StringRef> dtypes, int64_t location) {
    NamedAttrList attributes;
    attributes.set("vernon.interface", StringAttr::get(context, interfaceName));
    attributes.set("vernon.source_name", StringAttr::get(context, sourceName));
    SmallVector<Attribute> dtypeAttrs;
    for (StringRef dtype : dtypes)
        dtypeAttrs.push_back(StringAttr::get(context, dtype));
    attributes.set("vernon.abi_leaf_dtypes", ArrayAttr::get(context, dtypeAttrs));
    attributes.set("vernon.location", IntegerAttr::get(IntegerType::get(context, 64), location));
    return attributes.getDictionary(context);
}

StringRef scalarDtype(Type type) {
    if (type.isF16())
        return "f16";
    if (type.isF32())
        return "f32";
    if (type.isF64())
        return "f64";
    return {};
}

StringRef scalarDtype(Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp()))
            if (auto dtype = function.getArgAttrOfType<StringAttr>(argument.getArgNumber(), "vernon.dtype"))
                return dtype.getValue();
    }
    return scalarDtype(value.getType());
}

LogicalResult validateScalarPhase(func::FuncOp primal, const VernonAutodiffAnalysisResult &analysis) {
    if (!llvm::hasSingleElement(primal.getBody()))
        return primal.emitError("structured scalar VJP requires one straight-line block");
    if (primal.getNumResults() != 1 || !isa<FloatType>(primal.getResultTypes().front()))
        return primal.emitError("structured scalar VJP requires exactly one floating-point result");
    if (analysis.getActiveResultLeaves().size() != 1 || analysis.getWrtLeaves().empty())
        return primal.emitError("structured scalar VJP requires one active result and at least one wrt leaf");
    if (llvm::any_of(analysis.getWrtLeaves(), [](const AutodiffLeaf &leaf) {
            return leaf.abiLeafIndex != 0 || !isa<FloatType>(leaf.primalType);
        }))
        return primal.emitError("structured scalar VJP currently accepts only scalar wrt paths");
    WalkResult structured = primal.walk([&](Operation *operation) {
        if (isa<scf::IfOp, scf::WhileOp>(operation)) {
            operation->emitError("structured control-flow VJP belongs to Phase 7");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return structured.wasInterrupted() ? failure() : success();
}

FailureOr<func::FuncOp> createForward(func::FuncOp primal, StringRef symbol, const VernonAutodiffTapePlan &plan) {
    MLIRContext *context = primal.getContext();
    SmallVector<Type> tapeTypes;
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves)
        tapeTypes.push_back(leaf.scalarType);
    TupleType tapeType = TupleType::get(context, tapeTypes);
    TupleType resultType = TupleType::get(context, {primal.getResultTypes().front(), tapeType});
    OpBuilder moduleBuilder(primal);
    auto forward = func::FuncOp::create(moduleBuilder, primal.getLoc(), symbol,
                                        FunctionType::get(context, primal.getArgumentTypes(), {resultType}));
    bool committed = false;
    auto cleanup = llvm::make_scope_exit([&] {
        if (!committed)
            forward.erase();
    });
    copyProfileFunctionAttrs(primal, forward);
    forward.setAllArgAttrs(primal.getAllArgAttrs());
    SmallVector<StringRef> forwardDtypes = {scalarDtype(primal.getBody().front().getTerminator()->getOperand(0))};
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves)
        forwardDtypes.push_back(leaf.dtype);
    SmallVector<DictionaryAttr> forwardResultAttrs = {
        makeInterfaceAttrs(context, "output", "forward_result", forwardDtypes, 0)};
    forward.setAllResultAttrs(forwardResultAttrs);

    Block *block = forward.addEntryBlock();
    IRMapping mapping;
    DenseMap<Value, Value> forwardValues;
    for (auto [source, target] : llvm::zip_equal(primal.getArguments(), block->getArguments())) {
        mapping.map(source, target);
        forwardValues.try_emplace(source, target);
    }
    OpBuilder builder = OpBuilder::atBlockEnd(block);
    for (Operation &operation : primal.getBody().front().without_terminator()) {
        Operation *clone = builder.clone(operation, mapping);
        for (auto [source, target] : llvm::zip_equal(operation.getResults(), clone->getResults()))
            forwardValues.try_emplace(source, target);
    }
    auto primalReturn = cast<func::ReturnOp>(primal.getBody().front().getTerminator());
    SmallVector<Value> tapeValues;
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves) {
        Value mapped = forwardValues.lookup(leaf.value);
        if (!mapped) {
            if (Operation *definition = leaf.value.getDefiningOp())
                definition->emitError("planned tape value was not cloned into augmented forward");
            else
                primal.emitError("planned tape argument was not mapped into augmented forward");
            return failure();
        }
        tapeValues.push_back(mapped);
    }
    Value primalResult = mapping.lookupOrNull(primalReturn.getOperand(0));
    if (!primalResult)
        return primal.emitError("primal result was not cloned into augmented forward");
    Value tape = createTuple(builder, primal.getLoc(), tapeType, tapeValues);
    Value result = createTuple(builder, primal.getLoc(), resultType, {primalResult, tape});
    func::ReturnOp::create(builder, primal.getLoc(), result);
    committed = true;
    return forward;
}

FailureOr<func::FuncOp> createBackward(func::FuncOp primal, StringRef symbol,
                                       const VernonAutodiffAnalysisResult &analysis,
                                       const VernonAutodiffRuleRegistry &registry, const VernonAutodiffTapePlan &plan) {
    MLIRContext *context = primal.getContext();
    SmallVector<Type> tapeTypes;
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves)
        tapeTypes.push_back(leaf.scalarType);
    TupleType tapeType = TupleType::get(context, tapeTypes);
    Type cotangentType = analysis.getActiveResultLeaves().front().derivativeType;
    SmallVector<Type> gradientTypes;
    for (const AutodiffLeaf &leaf : analysis.getWrtLeaves())
        gradientTypes.push_back(leaf.derivativeType);
    Type resultType =
        gradientTypes.size() == 1 ? gradientTypes.front() : static_cast<Type>(TupleType::get(context, gradientTypes));

    OpBuilder moduleBuilder(primal);
    auto backward = func::FuncOp::create(moduleBuilder, primal.getLoc(), symbol,
                                         FunctionType::get(context, {tapeType, cotangentType}, {resultType}));
    bool committed = false;
    auto cleanup = llvm::make_scope_exit([&] {
        if (!committed)
            backward.erase();
    });
    copyProfileFunctionAttrs(primal, backward);
    SmallVector<StringRef> tapeDtypes;
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves)
        tapeDtypes.push_back(leaf.dtype);
    SmallVector<DictionaryAttr> backwardArgumentAttrs = {
        makeInterfaceAttrs(context, "input", "tape", tapeDtypes, 0),
        makeInterfaceAttrs(context, "input", "output", {scalarDtype(cotangentType)}, 1)};
    backward.setAllArgAttrs(backwardArgumentAttrs);
    SmallVector<StringRef> gradientDtypes;
    for (const AutodiffLeaf &leaf : analysis.getWrtLeaves())
        gradientDtypes.push_back(scalarDtype(leaf.derivativeType));
    SmallVector<DictionaryAttr> backwardResultAttrs = {
        makeInterfaceAttrs(context, "output", "gradients", gradientDtypes, 0)};
    backward.setAllResultAttrs(backwardResultAttrs);

    Block *block = backward.addEntryBlock();
    OpBuilder builder = OpBuilder::atBlockEnd(block);
    DenseMap<Value, Value> primals;
    for (auto [index, leaf] : llvm::enumerate(plan.getInvocationRecord().leaves))
        primals.try_emplace(leaf.value,
                            getTupleElement(builder, primal.getLoc(), block->getArgument(0), index, leaf.scalarType));

    DenseMap<Value, Value> adjoints;
    adjoints.try_emplace(cast<func::ReturnOp>(primal.getBody().front().getTerminator()).getOperand(0),
                         block->getArgument(1));
    for (Operation &operation : llvm::reverse(primal.getBody().front().without_terminator())) {
        if (operation.getNumResults() != 1) {
            if (llvm::any_of(operation.getResults(), [&](Value result) { return adjoints.contains(result); }))
                return operation.emitError("active structured scalar operation must have exactly one result");
            continue;
        }
        auto seed = adjoints.find(operation.getResult(0));
        if (seed == adjoints.end())
            continue;
        const DifferentiationRule *rule = registry.lookup(&operation);
        if (!rule)
            return operation.emitError("active scalar operation has no VJP rule");
        SmallVector<Value> operands;
        SmallVector<Value> results;
        for (Value operand : operation.getOperands())
            operands.push_back(primals.lookup(operand));
        results.push_back(primals.lookup(operation.getResult(0)));
        FailureOr<SmallVector<Value>> contributions =
            rule->buildVjp(&operation, AutodiffVjpBuildContext{builder, operation.getLoc(), operands, results,
                                                               ValueRange(seed->second)});
        if (failed(contributions))
            return failure();
        for (auto [operand, contribution] : llvm::zip_equal(operation.getOperands(), *contributions)) {
            if (!analysis.isActive(operand, 0))
                continue;
            auto previous = adjoints.find(operand);
            if (previous == adjoints.end())
                adjoints.try_emplace(operand, contribution);
            else
                previous->second = arith::AddFOp::create(builder, operation.getLoc(), previous->second, contribution);
        }
    }

    SmallVector<Value> gradients;
    for (const AutodiffLeaf &leaf : analysis.getWrtLeaves()) {
        Value gradient = adjoints.lookup(leaf.value);
        if (!gradient)
            gradient = createZero(builder, primal.getLoc(), leaf.derivativeType);
        gradients.push_back(gradient);
    }
    Value result = gradients.size() == 1
                       ? gradients.front()
                       : createTuple(builder, primal.getLoc(), cast<TupleType>(resultType), gradients);
    func::ReturnOp::create(builder, primal.getLoc(), result);
    committed = true;
    return backward;
}

struct VernonStructuredVjpPass final : PassWrapper<VernonStructuredVjpPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonStructuredVjpPass)

    VernonStructuredVjpPass() = default;
    VernonStructuredVjpPass(const VernonStructuredVjpPass &other) : PassWrapper(other), options(other.options) {}
    explicit VernonStructuredVjpPass(StructuredVjpOptions options) : options(std::move(options)) {}

    StringRef getArgument() const final { return "vernon-structured-vjp"; }
    StringRef getDescription() const final { return "Generate structured straight-line scalar VJP profiles"; }

    void runOnOperation() override {
        StructuredVjpOptions selected = options;
        if (selected.wrtPaths.empty())
            selected.wrtPaths.assign(wrt.begin(), wrt.end());
        if (selected.forwardSymbol.empty())
            selected.forwardSymbol = forward;
        if (selected.backwardSymbol.empty())
            selected.backwardSymbol = backward;
        if (selected.wrtPaths.empty() || selected.forwardSymbol.empty() || selected.backwardSymbol.empty()) {
            getOperation().emitError("structured VJP pass requires wrt paths and forward/backward symbols");
            return signalPassFailure();
        }
        SmallVector<func::FuncOp> entries;
        getOperation().walk([&](func::FuncOp function) {
            if (function->hasAttr(kEntryAttr))
                entries.push_back(function);
        });
        if (entries.size() != 1) {
            getOperation().emitError("structured VJP pass requires exactly one entry function");
            return signalPassFailure();
        }
        if (failed(buildStructuredScalarVjp(entries.front(), selected)))
            signalPassFailure();
    }

    StructuredVjpOptions options;
    ListOption<std::string> wrt{*this, "wrt", llvm::cl::desc("Canonical scalar wrt paths"), llvm::cl::ZeroOrMore};
    Option<std::string> forward{*this, "forward", llvm::cl::desc("Augmented forward symbol"), llvm::cl::init("")};
    Option<std::string> backward{*this, "backward", llvm::cl::desc("Reverse symbol"), llvm::cl::init("")};
};

} // namespace

FailureOr<StructuredVjpResult> buildStructuredScalarVjp(func::FuncOp primal, const StructuredVjpOptions &options) {
    if (!primal || !primal->getParentOfType<ModuleOp>())
        return failure();
    if (options.wrtPaths.empty() || options.forwardSymbol.empty() || options.backwardSymbol.empty())
        return primal.emitError("structured scalar VJP requires wrt paths and profile symbols");
    if (options.forwardSymbol == options.backwardSymbol)
        return primal.emitError("structured scalar VJP requires distinct profile symbols");
    llvm::StringSet<> uniqueWrtPaths;
    for (const std::string &path : options.wrtPaths)
        if (path.empty() || !uniqueWrtPaths.insert(path).second)
            return primal.emitError("structured scalar VJP requires unique non-empty wrt paths");
    ModuleOp module = primal->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(options.forwardSymbol) || module.lookupSymbol(options.backwardSymbol))
        return primal.emitError("structured VJP profile symbol already exists");

    SmallVector<StringRef> wrtPaths;
    for (const std::string &path : options.wrtPaths)
        wrtPaths.push_back(path);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(primal, wrtPaths, registry);
    if (failed(analysis) || failed(validateScalarPhase(primal, *analysis)))
        return failure();
    FailureOr<VernonAutodiffTapePlan> plan = planAutodiffTape(primal, *analysis, registry);
    if (failed(plan))
        return failure();
    if (!plan->getRegions().empty())
        return primal.emitError("structured scalar VJP cannot contain dynamic tape regions");
    if (llvm::any_of(plan->getInvocationRecord().leaves,
                     [](const AutodiffTapeLeaf &leaf) { return leaf.abiLeafIndex != 0 || leaf.scalarCount != 1; }))
        return primal.emitError("structured scalar VJP tape contains a non-scalar leaf");

    FailureOr<func::FuncOp> forward = createForward(primal, options.forwardSymbol, *plan);
    if (failed(forward))
        return failure();
    FailureOr<func::FuncOp> backward = createBackward(primal, options.backwardSymbol, *analysis, registry, *plan);
    if (failed(backward)) {
        forward->erase();
        return failure();
    }
    if (failed(verify(*forward)) || failed(verify(*backward))) {
        forward->erase();
        backward->erase();
        return primal.emitError("generated structured scalar VJP profile failed verification");
    }
    primal->removeAttr(kEntryAttr);
    SmallVector<std::string> derivativeRules;
    for (const AutodiffOperationActivity &activity : analysis->getOperations()) {
        if (!activity.active)
            continue;
        if (const DifferentiationRule *rule = registry.lookup(activity.operation))
            derivativeRules.push_back(rule->getRegistryKey().str());
    }
    llvm::sort(derivativeRules);
    derivativeRules.erase(std::unique(derivativeRules.begin(), derivativeRules.end()), derivativeRules.end());
    return StructuredVjpResult{*forward, *backward, plan->getInvocationRecord().stride, std::move(derivativeRules)};
}

std::unique_ptr<Pass> createVernonStructuredVjpPass() { return std::make_unique<VernonStructuredVjpPass>(); }

std::unique_ptr<Pass> createVernonStructuredVjpPass(StructuredVjpOptions options) {
    return std::make_unique<VernonStructuredVjpPass>(std::move(options));
}

void registerVernonStructuredVjpPass() { PassRegistration<VernonStructuredVjpPass>(); }

} // namespace mlir::vernon
