#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <set>

namespace mlir::vernon {
namespace {

constexpr unsigned kInvalidNode = std::numeric_limits<unsigned>::max();

FailureOr<unsigned> checkedUnsigned(size_t value) {
    if (value > std::numeric_limits<unsigned>::max())
        return failure();
    return static_cast<unsigned>(value);
}

bool samePathComponent(const ValueAbiPathComponent &left, const ValueAbiPathComponent &right) {
    return left.field == right.field && left.index == right.index;
}

bool hasPathPrefix(ArrayRef<ValueAbiPathComponent> path, ArrayRef<ValueAbiPathComponent> prefix) {
    return path.size() >= prefix.size() && llvm::equal(prefix, path.take_front(prefix.size()), samePathComponent);
}

std::string appendAbiPath(StringRef root, ArrayRef<ValueAbiPathComponent> path) {
    std::string result = root.str();
    for (const ValueAbiPathComponent &component : path) {
        result.push_back('.');
        if (component.field)
            result.append(*component.field);
        else
            result.append(std::to_string(component.index));
    }
    return result;
}

FailureOr<SmallVector<StringRef>> getDtypes(ArrayAttr attribute) {
    SmallVector<StringRef> result;
    if (!attribute)
        return result;
    result.reserve(attribute.size());
    for (Attribute item : attribute) {
        auto value = dyn_cast<StringAttr>(item);
        if (!value)
            return failure();
        result.push_back(value.getValue());
    }
    return result;
}

bool isDifferentiable(const ValueAbiLeaf &leaf) { return isDifferentiableAutodiffLeaf(leaf.scalarType, leaf.dtype); }

struct ValueNodes {
    Value value;
    ValueAbiLayout layout;
    SmallVector<unsigned> nodes;
};

SmallVector<std::string> copyDtypes(const ValueAbiLayout &layout) {
    SmallVector<std::string> result;
    result.reserve(layout.leaves.size());
    for (const ValueAbiLeaf &leaf : layout.leaves)
        result.push_back(leaf.dtype);
    return result;
}

class AutodiffValueAbiResolver {
public:
    AutodiffValueAbiResolver(func::FuncOp function, ModuleOp module) : function(function), module(module) {}

    FailureOr<ValueAbiLayout> resolve(Value value, ArrayRef<StringRef> explicitDtypes = {}) {
        if (auto found = cache.find(value); found != cache.end()) {
            if (!explicitDtypes.empty()) {
                FailureOr<ValueAbiLayout> explicitLayout =
                    mlir::vernon::getValueAbiLayout(value.getType(), module, explicitDtypes);
                if (failed(explicitLayout) || explicitLayout->layoutHash != found->second.layoutHash)
                    return failure();
            }
            return found->second;
        }

        SmallVector<std::string> ownedDtypes;
        if (!explicitDtypes.empty()) {
            for (StringRef dtype : explicitDtypes)
                ownedDtypes.push_back(dtype.str());
        } else {
            FailureOr<SmallVector<std::string>> inferred = inferDtypes(value);
            if (failed(inferred))
                return failure();
            ownedDtypes = std::move(*inferred);
        }
        SmallVector<StringRef> dtypeRefs;
        dtypeRefs.reserve(ownedDtypes.size());
        for (const std::string &dtype : ownedDtypes)
            dtypeRefs.push_back(dtype);
        FailureOr<ValueAbiLayout> layout = mlir::vernon::getValueAbiLayout(value.getType(), module, dtypeRefs);
        if (failed(layout))
            return failure();
        cache.try_emplace(value, *layout);
        return std::move(*layout);
    }

private:
    FailureOr<SmallVector<std::string>> inferDtypes(Value value) {
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            if (argument.getOwner() == &function.getBody().front()) {
                FailureOr<SmallVector<StringRef>> dtypes =
                    getDtypes(function.getArgAttrOfType<ArrayAttr>(argument.getArgNumber(), "vernon.abi_leaf_dtypes"));
                if (failed(dtypes))
                    return failure();
                SmallVector<std::string> result;
                for (StringRef dtype : *dtypes)
                    result.push_back(dtype.str());
                return result;
            }
            if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(argument.getOwner()->getParentOp())) {
                if (argument.getArgNumber() >= whileOp.getInits().size())
                    return failure();
                FailureOr<ValueAbiLayout> initial = resolve(whileOp.getInits()[argument.getArgNumber()]);
                return succeeded(initial) ? FailureOr<SmallVector<std::string>>(copyDtypes(*initial))
                                          : FailureOr<SmallVector<std::string>>(failure());
            }
            return failure();
        }

        auto result = dyn_cast<OpResult>(value);
        if (!result)
            return SmallVector<std::string>{};
        Operation *operation = result.getOwner();
        if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
            if (result.getResultNumber() >= whileOp.getInits().size())
                return failure();
            FailureOr<ValueAbiLayout> initial = resolve(whileOp.getInits()[result.getResultNumber()]);
            return succeeded(initial) ? FailureOr<SmallVector<std::string>>(copyDtypes(*initial))
                                      : FailureOr<SmallVector<std::string>>(failure());
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
            if (ifOp.getThenRegion().empty() || ifOp.getElseRegion().empty())
                return failure();
            FailureOr<ValueAbiLayout> thenLayout = resolve(ifOp.thenYield()->getOperand(result.getResultNumber()));
            FailureOr<ValueAbiLayout> elseLayout = resolve(ifOp.elseYield()->getOperand(result.getResultNumber()));
            if (failed(thenLayout) || failed(elseLayout) || thenLayout->layoutHash != elseLayout->layoutHash)
                return failure();
            return copyDtypes(*thenLayout);
        }
        if (auto get = dyn_cast<StructGetOp>(operation))
            return projectDtypes(get.getInput(), ValueAbiPathComponent::getField(get.getField()));
        if (auto get = dyn_cast<TupleGetOp>(operation))
            return projectDtypes(get.getInput(),
                                 ValueAbiPathComponent::getIndex(static_cast<uint64_t>(get.getIndex())));
        if (isa<TupleCreateOp, StructCreateOp>(operation)) {
            SmallVector<std::string> combined;
            for (Value operand : operation->getOperands()) {
                FailureOr<ValueAbiLayout> operandLayout = resolve(operand);
                if (failed(operandLayout))
                    return failure();
                llvm::append_range(combined, copyDtypes(*operandLayout));
            }
            return combined;
        }

        std::optional<SmallVector<std::string>> propagated;
        for (Value operand : operation->getOperands()) {
            if (operand.getType() != value.getType())
                continue;
            FailureOr<ValueAbiLayout> operandLayout = resolve(operand);
            if (failed(operandLayout))
                return failure();
            SmallVector<std::string> operandDtypes = copyDtypes(*operandLayout);
            if (propagated && *propagated != operandDtypes)
                return failure();
            propagated = std::move(operandDtypes);
        }
        return propagated ? std::move(*propagated) : SmallVector<std::string>{};
    }

    FailureOr<SmallVector<std::string>> projectDtypes(Value input, const ValueAbiPathComponent &prefix) {
        FailureOr<ValueAbiLayout> inputLayout = resolve(input);
        if (failed(inputLayout))
            return failure();
        SmallVector<std::string> projected;
        for (const ValueAbiLeaf &leaf : inputLayout->leaves) {
            if (hasPathPrefix(leaf.path, ArrayRef<ValueAbiPathComponent>(prefix)))
                projected.push_back(leaf.dtype);
        }
        return projected;
    }

    func::FuncOp function;
    ModuleOp module;
    DenseMap<Value, ValueAbiLayout> cache;
};

} // namespace

class AutodiffAnalysisBuilder {
public:
    AutodiffAnalysisBuilder(func::FuncOp function, ArrayRef<StringRef> wrtPaths,
                            const VernonAutodiffRuleRegistry &registry)
        : function(function), module(function->getParentOfType<ModuleOp>()), requestedWrt(wrtPaths), registry(registry),
          abiResolver(function, module) {}

    FailureOr<VernonAutodiffAnalysisResult> run();

private:
    FailureOr<unsigned> addValue(Value value, ArrayRef<StringRef> dtypes = {});
    std::optional<unsigned> getNode(Value value, unsigned leafIndex);
    void addDependency(unsigned result, unsigned operand);
    void verifySameAbi(Operation *operation, ValueRange values);
    void addAllDependencies(Value result, ValueRange operands);
    void addProjectedGetDependencies(Value result, Value input, const ValueAbiPathComponent &prefix);
    void addCreateDependencies(Value result, ValueRange operands);
    void buildDependencies(Operation *operation);
    LogicalResult resolveWrt();
    LogicalResult resolveResults();
    void computeActivity();
    LogicalResult discoverRegions();
    LogicalResult collectOperations();
    bool anyActiveLeaf(Value value) const;
    bool hasActiveDescendant(Operation *operation) const;
    void emitDependencyError(Operation *operation, const Twine &message);
    InFlightDiagnostic emitFunctionError(const Twine &message);

    func::FuncOp function;
    ModuleOp module;
    ArrayRef<StringRef> requestedWrt;
    const VernonAutodiffRuleRegistry &registry;
    AutodiffValueAbiResolver abiResolver;
    VernonAutodiffAnalysisResult result;
    DenseMap<Value, unsigned> valueIndices;
    SmallVector<ValueNodes, 0> values;
    SmallVector<std::pair<Value, unsigned>> nodeValues;
    SmallVector<SmallVector<unsigned>> dependencies;
    SmallVector<SmallVector<unsigned>> users;
    SmallVector<unsigned> wrtNodes;
    SmallVector<unsigned> resultNodes;
    llvm::BitVector needed;
    llvm::BitVector influenced;
    llvm::BitVector active;
    LogicalResult dependencyStatus = success();
};

FailureOr<unsigned> AutodiffAnalysisBuilder::addValue(Value value, ArrayRef<StringRef> dtypes) {
    auto existing = valueIndices.find(value);
    if (existing != valueIndices.end()) {
        if (!dtypes.empty() && failed(abiResolver.resolve(value, dtypes)))
            return failure();
        return existing->second;
    }
    FailureOr<ValueAbiLayout> layout = abiResolver.resolve(value, dtypes);
    if (failed(layout))
        return failure();

    FailureOr<unsigned> valueIndex = checkedUnsigned(values.size());
    if (failed(valueIndex) || layout->leaves.size() > std::numeric_limits<unsigned>::max())
        return failure();
    valueIndices.try_emplace(value, *valueIndex);
    ValueNodes &entry = values.emplace_back(ValueNodes{value, std::move(*layout), {}});
    entry.nodes.assign(entry.layout.leaves.size(), kInvalidNode);
    for (auto [leafIndex, leaf] : llvm::enumerate(entry.layout.leaves)) {
        if (!isDifferentiable(leaf))
            continue;
        FailureOr<unsigned> node = checkedUnsigned(nodeValues.size());
        if (failed(node))
            return failure();
        entry.nodes[leafIndex] = *node;
        nodeValues.emplace_back(value, static_cast<unsigned>(leafIndex));
        dependencies.emplace_back();
        users.emplace_back();
    }
    return *valueIndex;
}

std::optional<unsigned> AutodiffAnalysisBuilder::getNode(Value value, unsigned leafIndex) {
    FailureOr<unsigned> valueIndex = addValue(value);
    if (failed(valueIndex))
        return std::nullopt;
    ValueNodes &entry = values[*valueIndex];
    if (leafIndex >= entry.nodes.size() || entry.nodes[leafIndex] == kInvalidNode)
        return std::nullopt;
    return entry.nodes[leafIndex];
}

void AutodiffAnalysisBuilder::addDependency(unsigned resultNode, unsigned operandNode) {
    if (!llvm::is_contained(dependencies[resultNode], operandNode)) {
        dependencies[resultNode].push_back(operandNode);
        users[operandNode].push_back(resultNode);
    }
}

void AutodiffAnalysisBuilder::verifySameAbi(Operation *operation, ValueRange comparedValues) {
    const ValueAbiLayout *expected = nullptr;
    for (Value value : comparedValues) {
        FailureOr<unsigned> valueIndex = addValue(value);
        if (failed(valueIndex)) {
            emitDependencyError(operation, "cannot resolve structured control-flow Value ABI metadata");
            return;
        }
        const ValueAbiLayout &layout = values[*valueIndex].layout;
        if (expected && expected->layoutHash != layout.layoutHash) {
            emitDependencyError(operation, "structured control-flow values have incompatible canonical Value ABIs");
            return;
        }
        expected = &layout;
    }
}

void AutodiffAnalysisBuilder::addAllDependencies(Value resultValue, ValueRange operands) {
    FailureOr<unsigned> resultIndex = addValue(resultValue);
    if (failed(resultIndex)) {
        emitDependencyError(resultValue.getDefiningOp(), "cannot resolve the result's canonical Value ABI");
        return;
    }
    for (auto [resultLeafIndex, resultLeaf] : llvm::enumerate(values[*resultIndex].layout.leaves)) {
        if (!isDifferentiable(resultLeaf))
            continue;
        std::optional<unsigned> resultNode = getNode(resultValue, resultLeafIndex);
        if (!resultNode)
            continue;
        for (Value operand : operands) {
            FailureOr<unsigned> operandIndex = addValue(operand);
            if (failed(operandIndex)) {
                emitDependencyError(resultValue.getDefiningOp(), "cannot resolve an operand's canonical Value ABI");
                return;
            }
            for (auto [operandLeafIndex, operandLeaf] : llvm::enumerate(values[*operandIndex].layout.leaves)) {
                if (isDifferentiable(operandLeaf)) {
                    if (std::optional<unsigned> operandNode = getNode(operand, operandLeafIndex))
                        addDependency(*resultNode, *operandNode);
                }
            }
        }
    }
}

void AutodiffAnalysisBuilder::addProjectedGetDependencies(Value resultValue, Value input,
                                                          const ValueAbiPathComponent &prefix) {
    FailureOr<unsigned> resultIndex = addValue(resultValue);
    FailureOr<unsigned> inputIndex = addValue(input);
    if (failed(resultIndex) || failed(inputIndex)) {
        emitDependencyError(resultValue.getDefiningOp(),
                            "cannot resolve aggregate projection canonical Value ABI metadata");
        return;
    }
    for (auto [resultLeafIndex, resultLeaf] : llvm::enumerate(values[*resultIndex].layout.leaves)) {
        if (!isDifferentiable(resultLeaf))
            continue;
        bool matched = false;
        SmallVector<ValueAbiPathComponent> inputPath;
        inputPath.push_back(prefix);
        llvm::append_range(inputPath, resultLeaf.path);
        for (auto [inputLeafIndex, inputLeaf] : llvm::enumerate(values[*inputIndex].layout.leaves)) {
            if (!isDifferentiable(inputLeaf) || inputLeaf.path.size() != inputPath.size() ||
                !llvm::equal(inputLeaf.path, inputPath, samePathComponent))
                continue;
            addDependency(*getNode(resultValue, resultLeafIndex), *getNode(input, inputLeafIndex));
            matched = true;
            break;
        }
        if (!matched) {
            emitDependencyError(resultValue.getDefiningOp(),
                                "cannot project a differentiable aggregate leaf through this get operation");
            return;
        }
    }
}

void AutodiffAnalysisBuilder::addCreateDependencies(Value resultValue, ValueRange operands) {
    FailureOr<unsigned> resultIndex = addValue(resultValue);
    if (failed(resultIndex)) {
        emitDependencyError(resultValue.getDefiningOp(), "aggregate create result has no canonical Value ABI");
        return;
    }
    unsigned flattenedLeaf = 0;
    for (Value operand : operands) {
        FailureOr<unsigned> operandIndex = addValue(operand);
        if (failed(operandIndex)) {
            emitDependencyError(resultValue.getDefiningOp(), "aggregate create operand has no canonical Value ABI");
            return;
        }
        for (auto [operandLeafIndex, operandLeaf] : llvm::enumerate(values[*operandIndex].layout.leaves)) {
            if (flattenedLeaf >= values[*resultIndex].layout.leaves.size()) {
                emitDependencyError(resultValue.getDefiningOp(),
                                    "aggregate create operands exceed the result's canonical ABI leaves");
                return;
            }
            if (isDifferentiable(operandLeaf) && isDifferentiable(values[*resultIndex].layout.leaves[flattenedLeaf]))
                addDependency(*getNode(resultValue, flattenedLeaf), *getNode(operand, operandLeafIndex));
            ++flattenedLeaf;
        }
    }
    if (flattenedLeaf != values[*resultIndex].layout.leaves.size())
        emitDependencyError(resultValue.getDefiningOp(),
                            "aggregate create operands do not cover the result's canonical ABI leaves");
}

void AutodiffAnalysisBuilder::buildDependencies(Operation *operation) {
    if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
        for (auto [resultIndex, value] : llvm::enumerate(ifOp.getResults())) {
            SmallVector<Value> yielded;
            if (!ifOp.getThenRegion().empty())
                yielded.push_back(ifOp.thenYield()->getOperand(resultIndex));
            if (!ifOp.getElseRegion().empty())
                yielded.push_back(ifOp.elseYield()->getOperand(resultIndex));
            addAllDependencies(value, yielded);
            yielded.push_back(value);
            verifySameAbi(operation, yielded);
        }
        return;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
        auto condition = cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
        auto afterYield = cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
        for (auto [index, value] : llvm::enumerate(whileOp.getResults())) {
            addAllDependencies(value, ValueRange(condition.getArgs()[index]));
            SmallVector<Value> carried = {whileOp.getInits()[index],      whileOp.getBeforeArguments()[index],
                                          condition.getArgs()[index],     whileOp.getAfterArguments()[index],
                                          afterYield.getResults()[index], value};
            verifySameAbi(operation, carried);
        }
        for (auto [index, argument] : llvm::enumerate(whileOp.getBeforeArguments())) {
            addAllDependencies(argument, ValueRange(whileOp.getInits()[index]));
            addAllDependencies(argument, ValueRange(afterYield.getResults()[index]));
        }
        for (auto [index, argument] : llvm::enumerate(whileOp.getAfterArguments()))
            addAllDependencies(argument, ValueRange(condition.getArgs()[index]));
        return;
    }
    if (auto get = dyn_cast<StructGetOp>(operation)) {
        addProjectedGetDependencies(get.getResult(), get.getInput(), ValueAbiPathComponent::getField(get.getField()));
        return;
    }
    if (auto get = dyn_cast<TupleGetOp>(operation)) {
        addProjectedGetDependencies(get.getResult(), get.getInput(),
                                    ValueAbiPathComponent::getIndex(static_cast<uint64_t>(get.getIndex())));
        return;
    }
    if (auto create = dyn_cast<StructCreateOp>(operation)) {
        addCreateDependencies(create.getResult(), create.getFields());
        return;
    }
    if (auto create = dyn_cast<TupleCreateOp>(operation)) {
        addCreateDependencies(create.getResult(), create.getElements());
        return;
    }
    if (const DifferentiationRule *rule = registry.lookup(operation)) {
        FailureOr<AutodiffRuleActivity> activity = rule->classifyActivity(operation);
        if (failed(activity)) {
            // Preserve conservative dependency discovery so an incompatible
            // rule is diagnosed only if this operation is actually active.
            for (Value operationResult : operation->getResults())
                addAllDependencies(operationResult, operation->getOperands());
            return;
        }
        SmallVector<Value> operands;
        operands.reserve(activity->operandIndices.size());
        for (unsigned operandIndex : activity->operandIndices)
            operands.push_back(operation->getOperand(operandIndex));
        for (unsigned resultIndex : activity->resultIndices)
            addAllDependencies(operation->getResult(resultIndex), operands);
        return;
    }
    for (Value operationResult : operation->getResults())
        addAllDependencies(operationResult, operation->getOperands());
}

InFlightDiagnostic AutodiffAnalysisBuilder::emitFunctionError(const Twine &message) {
    return function.emitError() << "autodiff analysis: " << message;
}

void AutodiffAnalysisBuilder::emitDependencyError(Operation *operation, const Twine &message) {
    if (succeeded(dependencyStatus)) {
        if (operation)
            operation->emitError() << "autodiff analysis: " << message;
        else
            emitFunctionError(message);
    }
    dependencyStatus = failure();
}

LogicalResult AutodiffAnalysisBuilder::resolveWrt() {
    if (requestedWrt.empty())
        return emitFunctionError("requires at least one wrt path");
    std::set<std::string> canonicalPaths;
    for (StringRef requestedPath : requestedWrt) {
        if (requestedPath.empty())
            return emitFunctionError("empty wrt path");
        SmallVector<StringRef> components;
        requestedPath.split(components, '.', -1, false);
        if (components.empty() || llvm::any_of(components, [](StringRef item) { return item.empty(); }))
            return emitFunctionError(Twine("invalid wrt path '") + requestedPath + "'");

        std::optional<unsigned> argumentIndex;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            auto sourceName = function.getArgAttrOfType<StringAttr>(index, "vernon.source_name");
            if (sourceName && sourceName.getValue() == components.front()) {
                argumentIndex = index;
                break;
            }
        }
        if (!argumentIndex)
            return emitFunctionError(Twine("wrt path '") + requestedPath + "' names no entry parameter");

        Value argument = function.getArgument(*argumentIndex);
        Attribute dtypeMetadata = function.getArgAttr(*argumentIndex, "vernon.abi_leaf_dtypes");
        auto dtypeArray = dyn_cast_or_null<ArrayAttr>(dtypeMetadata);
        if (dtypeMetadata && !dtypeArray)
            return emitFunctionError(Twine("wrt path '") + requestedPath + "' has malformed ABI dtype metadata");
        FailureOr<SmallVector<StringRef>> dtypes = getDtypes(dtypeArray);
        if (failed(dtypes))
            return emitFunctionError(Twine("wrt path '") + requestedPath + "' has malformed ABI dtype metadata");
        FailureOr<unsigned> valueIndex = addValue(argument, *dtypes);
        if (failed(valueIndex))
            return emitFunctionError(Twine("wrt path '") + requestedPath + "' has no canonical Value ABI");
        if (dtypeArray && dtypeArray.size() != values[*valueIndex].layout.leaves.size())
            return emitFunctionError(Twine("wrt path '") + requestedPath + "' has incomplete ABI dtype metadata");

        SmallVector<ValueAbiPathComponent> prefix;
        Type current = argument.getType();
        for (StringRef component : ArrayRef<StringRef>(components).drop_front()) {
            if (auto structure = dyn_cast<StructType>(current)) {
                FailureOr<ResolvedStructFields> fields = resolveNamedStructFields(structure, module);
                if (failed(fields))
                    return emitFunctionError(Twine("cannot resolve Struct in wrt path '") + requestedPath + "'");
                auto field = llvm::find_if(fields->fields,
                                           [&](const ResolvedStructField &item) { return item.name == component; });
                if (field == fields->fields.end())
                    return emitFunctionError(Twine("wrt path '") + requestedPath + "' names no Struct field");
                prefix.push_back(ValueAbiPathComponent::getField(component));
                current = field->type;
            } else if (auto tuple = dyn_cast<TupleType>(current)) {
                uint64_t index = 0;
                if (component.getAsInteger(10, index) || index >= tuple.size())
                    return emitFunctionError(Twine("wrt path '") + requestedPath + "' has an invalid Tuple index");
                prefix.push_back(ValueAbiPathComponent::getIndex(index));
                current = tuple.getType(index);
            } else {
                return emitFunctionError(Twine("wrt path '") + requestedPath +
                                         "' does not resolve through an aggregate");
            }
        }

        unsigned matches = 0;
        for (auto [leafIndex, leaf] : llvm::enumerate(values[*valueIndex].layout.leaves)) {
            if (!hasPathPrefix(leaf.path, prefix) || !isDifferentiable(leaf))
                continue;
            const std::string path = appendAbiPath(components.front(), leaf.path);
            if (!canonicalPaths.insert(path).second)
                return emitFunctionError(Twine("duplicate canonical wrt leaf '") + path + "'");
            Type derivativeType = *getAutodiffDerivativeType(leaf.scalarType);
            result.wrtLeaves.push_back(AutodiffLeaf{argument, static_cast<unsigned>(leafIndex), path, leaf.scalarType,
                                                    derivativeType, leaf.dtype});
            wrtNodes.push_back(*getNode(argument, leafIndex));
            ++matches;
        }
        if (matches == 0)
            return emitFunctionError(Twine("wrt path '") + requestedPath + "' has no differentiable floating leaves");
    }
    llvm::sort(result.wrtLeaves,
               [](const AutodiffLeaf &left, const AutodiffLeaf &right) { return left.path < right.path; });
    llvm::sort(wrtNodes);
    wrtNodes.erase(std::unique(wrtNodes.begin(), wrtNodes.end()), wrtNodes.end());
    return success();
}

LogicalResult AutodiffAnalysisBuilder::resolveResults() {
    SmallVector<func::ReturnOp> returns;
    function.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    if (returns.empty())
        return emitFunctionError("function has no return operation");
    if (returns.size() != 1)
        return emitFunctionError("requires one canonical return operation");
    for (func::ReturnOp returnOp : returns) {
        if (returnOp.getNumOperands() != function.getNumResults())
            return emitFunctionError("return operand count does not match function results");
        for (auto [resultIndex, value] : llvm::enumerate(returnOp.getOperands())) {
            Attribute dtypeMetadata = function.getResultAttr(resultIndex, "vernon.abi_leaf_dtypes");
            auto dtypeArray = dyn_cast_or_null<ArrayAttr>(dtypeMetadata);
            if (dtypeMetadata && !dtypeArray)
                return emitFunctionError(Twine("result #") + Twine(resultIndex) + " has malformed ABI dtype metadata");
            FailureOr<SmallVector<StringRef>> dtypes = getDtypes(dtypeArray);
            if (failed(dtypes))
                return emitFunctionError(Twine("result #") + Twine(resultIndex) + " has malformed ABI dtype metadata");
            FailureOr<ValueAbiLayout> resultLayout =
                getValueAbiLayout(function.getResultTypes()[resultIndex], module, *dtypes);
            FailureOr<unsigned> valueIndex = addValue(value);
            if (failed(resultLayout) || failed(valueIndex) ||
                resultLayout->leaves.size() != values[*valueIndex].layout.leaves.size())
                return emitFunctionError(Twine("result #") + Twine(resultIndex) + " has no canonical Value ABI");
            if (dtypeArray && dtypeArray.size() != resultLayout->leaves.size())
                return emitFunctionError(Twine("result #") + Twine(resultIndex) + " has incomplete ABI dtype metadata");
            if (dtypeArray && failed(abiResolver.resolve(value, *dtypes)))
                return emitFunctionError(Twine("result #") + Twine(resultIndex) +
                                         " ABI dtype metadata disagrees with its SSA value");
            StringRef root = "output";
            if (function.getNumResults() > 1) {
                if (auto name = function.getResultAttrOfType<StringAttr>(resultIndex, "vernon.source_name"))
                    root = name.getValue();
            }
            for (auto [leafIndex, leaf] : llvm::enumerate(resultLayout->leaves)) {
                if (!isDifferentiable(leaf))
                    continue;
                std::optional<unsigned> node = getNode(value, leafIndex);
                if (!node)
                    return emitFunctionError("differentiable result leaf cannot be projected to returned SSA value");
                resultNodes.push_back(*node);
                std::string path = appendAbiPath(root, leaf.path);
                if (function.getNumResults() > 1 && root == "output")
                    path = appendAbiPath(("output." + std::to_string(resultIndex)), leaf.path);
                result.activeResultLeaves.push_back(
                    AutodiffLeaf{value, static_cast<unsigned>(leafIndex), std::move(path), leaf.scalarType,
                                 *getAutodiffDerivativeType(leaf.scalarType), leaf.dtype});
            }
        }
    }
    if (resultNodes.empty())
        return emitFunctionError("function results have no differentiable floating leaves");
    return success();
}

void AutodiffAnalysisBuilder::computeActivity() {
    needed.resize(nodeValues.size());
    influenced.resize(nodeValues.size());
    SmallVector<unsigned> worklist(resultNodes);
    while (!worklist.empty()) {
        unsigned node = worklist.pop_back_val();
        if (needed.test(node))
            continue;
        needed.set(node);
        llvm::append_range(worklist, dependencies[node]);
    }
    worklist = wrtNodes;
    while (!worklist.empty()) {
        unsigned node = worklist.pop_back_val();
        if (influenced.test(node))
            continue;
        influenced.set(node);
        llvm::append_range(worklist, users[node]);
    }
    active = needed;
    active &= influenced;

    llvm::erase_if(result.activeResultLeaves,
                   [&](const AutodiffLeaf &leaf) { return !active.test(*getNode(leaf.value, leaf.abiLeafIndex)); });
    for (const ValueNodes &entry : values) {
        result.valueAbis.push_back(AutodiffValueAbi{entry.value, entry.layout});
        AutodiffValueActivity activity;
        activity.value = entry.value;
        for (auto [leafIndex, node] : llvm::enumerate(entry.nodes)) {
            if (node != kInvalidNode && active.test(node))
                activity.activeAbiLeaves.push_back(leafIndex);
        }
        if (!activity.activeAbiLeaves.empty())
            result.activeValues.push_back(std::move(activity));
    }
}

bool AutodiffAnalysisBuilder::anyActiveLeaf(Value value) const {
    auto found = valueIndices.find(value);
    if (found == valueIndices.end())
        return false;
    for (unsigned node : values[found->second].nodes) {
        if (node != kInvalidNode && active.test(node))
            return true;
    }
    return false;
}

bool AutodiffAnalysisBuilder::hasActiveDescendant(Operation *operation) const {
    bool found = false;
    operation->walk([&](Operation *nested) {
        if (nested == operation)
            return;
        if (llvm::any_of(nested->getResults(), [&](Value value) { return anyActiveLeaf(value); }) ||
            classifyAutodiffEffect(nested) != AutodiffEffectKind::Pure)
            found = true;
    });
    return found;
}

LogicalResult AutodiffAnalysisBuilder::discoverRegions() {
    DenseMap<Operation *, unsigned> ordinals;
    WalkResult walkResult = function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
        if (!isa<scf::IfOp, scf::WhileOp>(operation))
            return WalkResult::advance();
        std::optional<unsigned> parent;
        for (Operation *ancestor = operation->getParentOp(); ancestor && ancestor != function.getOperation();
             ancestor = ancestor->getParentOp()) {
            auto found = ordinals.find(ancestor);
            if (found != ordinals.end()) {
                parent = found->second;
                break;
            }
        }
        FailureOr<unsigned> ordinal = checkedUnsigned(result.regions.size());
        if (failed(ordinal)) {
            operation->emitError("autodiff region count exceeds the analysis representation");
            return WalkResult::interrupt();
        }
        ordinals.try_emplace(operation, *ordinal);
        result.regions.push_back(AutodiffRegion{operation, *ordinal, parent, {}});
        if (parent)
            result.regions[*parent].childOrdinals.push_back(*ordinal);
        return WalkResult::advance();
    });
    return failure(walkResult.wasInterrupted());
}

LogicalResult AutodiffAnalysisBuilder::collectOperations() {
    LogicalResult status = success();
    function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
        if (operation == function.getOperation())
            return WalkResult::advance();
        AutodiffEffectKind effect = classifyAutodiffEffect(operation);
        // Known differentiable Vernon intrinsics are pure semantic operations.
        // Never let registration override a concrete Storage or visible effect.
        if (effect == AutodiffEffectKind::Unsupported && isa<IntrinsicOp>(operation) && registry.lookup(operation))
            effect = AutodiffEffectKind::Pure;
        bool operationActive = llvm::any_of(operation->getResults(), [&](Value value) { return anyActiveLeaf(value); });
        // Effectful operations are part of every differentiated execution even
        // when their operands are constants or their results are unused. They
        // cannot be pruned using differentiable SSA activity.
        if (effect != AutodiffEffectKind::Pure)
            operationActive = true;
        if (isa<scf::IfOp, scf::WhileOp>(operation))
            operationActive |= hasActiveDescendant(operation);
        if (operation->getNumRegions() != 0 && !isa<scf::IfOp, scf::WhileOp>(operation) && operationActive) {
            operation->emitError("autodiff analysis does not support an active unstructured region operation");
            status = failure();
            return WalkResult::interrupt();
        }
        result.operations.push_back(AutodiffOperationActivity{operation, effect, operationActive});
        if (operationActive &&
            (effect == AutodiffEffectKind::Unsupported || effect == AutodiffEffectKind::ExternallyVisible)) {
            operation->emitError() << "autodiff analysis does not support active " << stringifyAutodiffEffect(effect)
                                   << " operation";
            status = failure();
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return status;
}

FailureOr<VernonAutodiffAnalysisResult> AutodiffAnalysisBuilder::run() {
    if (!module) {
        emitFunctionError("function must be nested in a module");
        return failure();
    }
    if (failed(resolveWrt()) || failed(resolveResults()))
        return failure();
    function.walk([&](Operation *operation) { buildDependencies(operation); });
    if (failed(dependencyStatus))
        return failure();
    computeActivity();
    if (failed(discoverRegions()))
        return failure();
    if (failed(collectOperations()))
        return failure();
    if (failed(verifyAutodiffRuleCoverage(result, registry)))
        return failure();
    return std::move(result);
}

bool VernonAutodiffAnalysisResult::isActive(Value value, unsigned abiLeafIndex) const {
    auto activity = llvm::find_if(activeValues, [&](const AutodiffValueActivity &item) { return item.value == value; });
    return activity != activeValues.end() && llvm::is_contained(activity->activeAbiLeaves, abiLeafIndex);
}

const ValueAbiLayout *VernonAutodiffAnalysisResult::getValueAbi(Value value) const {
    auto found = llvm::find_if(valueAbis, [&](const AutodiffValueAbi &item) { return item.value == value; });
    return found == valueAbis.end() ? nullptr : &found->layout;
}

bool VernonAutodiffAnalysisResult::isActive(Operation *operation) const {
    auto activity =
        llvm::find_if(operations, [&](const AutodiffOperationActivity &item) { return item.operation == operation; });
    return activity != operations.end() && activity->active;
}

FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths) {
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    return analyzeAutodiffFunction(function, wrtPaths, registry);
}

FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths,
                                                                const VernonAutodiffRuleRegistry &registry) {
    return AutodiffAnalysisBuilder(function, wrtPaths, registry).run();
}

bool isDifferentiableAutodiffLeaf(Type scalarType, StringRef logicalDtype) {
    if (!(scalarType.isF16() || scalarType.isF32() || scalarType.isF64()))
        return false;
    return logicalDtype.empty() || logicalDtype == "f16" || logicalDtype == "f32" || logicalDtype == "f64";
}

FailureOr<Type> getAutodiffDerivativeType(Type scalarType) {
    if (scalarType.isF64())
        return scalarType;
    if (scalarType.isF16() || scalarType.isF32())
        return Float32Type::get(scalarType.getContext());
    return failure();
}

AutodiffEffectKind classifyAutodiffEffect(Operation *operation) {
    if (isa<LoadOp, PhysicalLoadOp>(operation))
        return AutodiffEffectKind::StorageRead;
    if (auto store = dyn_cast<StoreOp>(operation))
        return store.getStorage().getType().getAddressSpace() == "device" ? AutodiffEffectKind::ExternallyVisible
                                                                          : AutodiffEffectKind::StorageWrite;
    if (auto store = dyn_cast<PhysicalStoreOp>(operation))
        return store.getStorage().getType().getAddressSpace() == "device" ? AutodiffEffectKind::ExternallyVisible
                                                                          : AutodiffEffectKind::StorageWrite;
    if (auto atomic = dyn_cast<AtomicOp>(operation))
        return atomic.getStorage().getType().getAddressSpace() == "device" ? AutodiffEffectKind::ExternallyVisible
                                                                           : AutodiffEffectKind::Atomic;
    if (auto atomic = dyn_cast<PhysicalAtomicOp>(operation))
        return atomic.getStorage().getType().getAddressSpace() == "device" ? AutodiffEffectKind::ExternallyVisible
                                                                           : AutodiffEffectKind::Atomic;
    if (auto reduce = dyn_cast<ReduceSumOp>(operation))
        return reduce.getStorage().getType().getAddressSpace() == "device" ? AutodiffEffectKind::ExternallyVisible
                                                                           : AutodiffEffectKind::StorageWrite;
    if (auto scatter = dyn_cast<ScatterAddOp>(operation))
        return scatter.getStorage().getType().getAddressSpace() == "device" ? AutodiffEffectKind::ExternallyVisible
                                                                            : AutodiffEffectKind::StorageWrite;
    if (isa<WorkgroupAllocOp>(operation))
        return AutodiffEffectKind::StorageWrite;
    if (auto barrier = dyn_cast<BarrierOp>(operation))
        return barrier.getScope() == "workgroup" ? AutodiffEffectKind::Barrier : AutodiffEffectKind::ExternallyVisible;
    if (isa<IntrinsicOp>(operation))
        return AutodiffEffectKind::Unsupported;
    // Structured containers inherit execution activity from their regions;
    // their nested operations carry the actual effect classifications.
    if (isa<scf::IfOp, scf::WhileOp>(operation))
        return AutodiffEffectKind::Pure;
    if (isMemoryEffectFree(operation) || isa<func::ReturnOp, scf::YieldOp, scf::ConditionOp>(operation))
        return AutodiffEffectKind::Pure;

    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface)
        return AutodiffEffectKind::Unsupported;
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    if (effects.empty())
        return AutodiffEffectKind::Pure;
    if (llvm::all_of(effects, [](const MemoryEffects::EffectInstance &effect) {
            return isa<MemoryEffects::Read>(effect.getEffect());
        }))
        return AutodiffEffectKind::StorageRead;
    return AutodiffEffectKind::ExternallyVisible;
}

StringRef stringifyAutodiffEffect(AutodiffEffectKind effect) {
    switch (effect) {
    case AutodiffEffectKind::Pure:
        return "pure";
    case AutodiffEffectKind::StorageRead:
        return "Storage read";
    case AutodiffEffectKind::StorageWrite:
        return "Storage write";
    case AutodiffEffectKind::Atomic:
        return "atomic";
    case AutodiffEffectKind::Barrier:
        return "barrier";
    case AutodiffEffectKind::ExternallyVisible:
        return "externally visible";
    case AutodiffEffectKind::Unsupported:
        return "unsupported";
    }
    llvm_unreachable("unknown autodiff effect");
}

} // namespace mlir::vernon
