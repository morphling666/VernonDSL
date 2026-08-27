#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffUtils.h"
#include "mlir/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.h"
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
    if (value >= kInvalidNode)
        return failure();
    return static_cast<unsigned>(value);
}

ValueRange storageIndices(Operation *operation) {
    if (auto load = dyn_cast<LoadOp>(operation))
        return load.getIndices();
    if (auto store = dyn_cast<StoreOp>(operation))
        return store.getIndices();
    if (auto reduce = dyn_cast<ReduceSumOp>(operation))
        return reduce.getIndices();
    if (auto scatter = dyn_cast<ScatterAddOp>(operation))
        return scatter.getIndices();
    if (auto atomic = dyn_cast<AtomicOp>(operation))
        return atomic.getIndices();
    return {};
}

bool hasActiveStorageEffect(const VernonAutodiffAnalysisResult &result, const AutodiffStorageEffect &effect) {
    const AutodiffStorageIdentity &identity = result.getStorageIdentities()[effect.identity];
    const ValueAbiLayout *layout = result.getValueAbi(identity.binding);
    if (!layout)
        return false;
    return llvm::any_of(llvm::seq<unsigned>(0, layout->leaves.size()), [&](unsigned leafIndex) {
        return result.isActiveStorageVersion(effect.versionBefore, leafIndex) ||
               (effect.versionAfter && result.isActiveStorageVersion(*effect.versionAfter, leafIndex));
    });
}

bool needsInternalStorageAdjoint(const VernonAutodiffAnalysisResult &result, const AutodiffStorageIdentity &identity) {
    bool hasActiveLoad = false;
    bool hasActiveStore = false;
    for (const AutodiffStorageEffect &effect : result.getStorageEffects()) {
        if (effect.identity != identity.id || !hasActiveStorageEffect(result, effect))
            continue;
        hasActiveLoad |= isa<LoadOp>(effect.operation);
        if (auto atomic = dyn_cast<AtomicOp>(effect.operation))
            hasActiveLoad |= result.isActive(atomic.getResult(), 0);
        hasActiveStore |= isa<StoreOp, ReduceSumOp, ScatterAddOp, AtomicOp>(effect.operation);
    }
    return hasActiveLoad && hasActiveStore;
}

LogicalResult verifyLaneOwnedDeviceAdjoint(const VernonAutodiffAnalysisResult &result,
                                           const AutodiffStorageIdentity &identity) {
    // Same injectivity proof as ordinary stores:
    // specs/compiler/invocation_index_ownership.md
    std::optional<ConditionalIndexProof> owned;
    for (const AutodiffStorageEffect &effect : result.getStorageEffects()) {
        if (effect.identity != identity.id || !hasActiveStorageEffect(result, effect))
            continue;
        ValueRange indices = storageIndices(effect.operation);
        std::optional<ConditionalIndexProof> proof = proveInvocationOwnedIndex(indices);
        if (!proof || (owned && *owned != *proof))
            return failure();
        owned = std::move(proof);
    }
    if (!owned)
        return failure();
    return success();
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

AutodiffIndexProvenanceKind classifyIndexProvenanceImpl(Value value, func::FuncOp function, DenseSet<Value> &visiting) {
    if (Operation *defining = value.getDefiningOp()) {
        if (defining->hasTrait<OpTrait::ConstantLike>())
            return AutodiffIndexProvenanceKind::Constant;
        if (defining->getNumRegions() == 0 && classifyAutodiffEffect(defining) == AutodiffEffectKind::Pure &&
            visiting.insert(value).second && llvm::all_of(defining->getOperands(), [&](Value operand) {
                return classifyIndexProvenanceImpl(operand, function, visiting) != AutodiffIndexProvenanceKind::Dynamic;
            })) {
            visiting.erase(value);
            return AutodiffIndexProvenanceKind::PureExpression;
        }
        visiting.erase(value);
        return AutodiffIndexProvenanceKind::Dynamic;
    }
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.getBody().front())
        return AutodiffIndexProvenanceKind::Dynamic;
    if (function.getArgAttr(argument.getArgNumber(), kBuiltinAttrName))
        return AutodiffIndexProvenanceKind::Builtin;
    return AutodiffIndexProvenanceKind::PrimalArgument;
}

AutodiffIndexProvenanceKind classifyIndexProvenance(Value value, func::FuncOp function) {
    DenseSet<Value> visiting;
    return classifyIndexProvenanceImpl(value, function, visiting);
}

struct ValueNodes {
    Value value;
    ValueAbiLayout layout;
    SmallVector<unsigned> nodes;
};

struct StorageLeafSelection {
    Value binding;
    unsigned abiLeafIndex{};
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
                FailureOr<ValueAbiLayout> explicitLayout = planLayout(value.getType(), explicitDtypes);
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
        FailureOr<ValueAbiLayout> layout = planLayout(value.getType(), dtypeRefs);
        if (failed(layout))
            return failure();
        cache.try_emplace(value, *layout);
        return std::move(*layout);
    }

private:
    FailureOr<ValueAbiLayout> planLayout(Type type, ArrayRef<StringRef> dtypes) {
        auto view = dyn_cast<TensorViewType>(type);
        Type layoutType = view ? Type(view.getElementType()) : type;
        FailureOr<ValueAbiLayout> planned = dtypes.empty()
                                                ? mlir::vernon::getValueStorageLayout(layoutType, module)
                                                : mlir::vernon::getValueAbiLayout(layoutType, module, dtypes);
        if (failed(planned))
            return failure();
        if (!view)
            return planned;
        ValueAbiLayout layout = std::move(*planned);
        layout.elementStride = layout.size;
        layout.layoutHash += ":storage";
        for (int64_t extent : view.getShape())
            layout.layoutHash += ":" + std::to_string(extent);
        return layout;
    }

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
            if (auto forOp = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())) {
                if (argument == forOp.getInductionVar())
                    return SmallVector<std::string>{};
                if (argument.getArgNumber() == 0 || argument.getArgNumber() > forOp.getInitArgs().size())
                    return failure();
                FailureOr<ValueAbiLayout> initial = resolve(forOp.getInitArgs()[argument.getArgNumber() - 1]);
                return succeeded(initial) ? FailureOr<SmallVector<std::string>>(copyDtypes(*initial))
                                          : FailureOr<SmallVector<std::string>>(failure());
            }
            return failure();
        }

        auto result = dyn_cast<OpResult>(value);
        if (!result)
            return SmallVector<std::string>{};
        Operation *operation = result.getOwner();
        if (auto load = dyn_cast<LoadOp>(operation)) {
            FailureOr<ValueAbiLayout> storage = resolve(load.getStorage());
            return succeeded(storage) ? FailureOr<SmallVector<std::string>>(copyDtypes(*storage))
                                      : FailureOr<SmallVector<std::string>>(failure());
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
            if (result.getResultNumber() >= whileOp.getInits().size())
                return failure();
            FailureOr<ValueAbiLayout> initial = resolve(whileOp.getInits()[result.getResultNumber()]);
            return succeeded(initial) ? FailureOr<SmallVector<std::string>>(copyDtypes(*initial))
                                      : FailureOr<SmallVector<std::string>>(failure());
        }
        if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
            if (result.getResultNumber() >= forOp.getInitArgs().size())
                return failure();
            FailureOr<ValueAbiLayout> initial = resolve(forOp.getInitArgs()[result.getResultNumber()]);
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
    AutodiffAnalysisBuilder(func::FuncOp function, ArrayRef<StringRef> wrtPaths, ArrayRef<StringRef> outputPaths,
                            const VernonAutodiffRuleRegistry &registry)
        : function(function), module(function->getParentOfType<ModuleOp>()), requestedWrt(wrtPaths),
          requestedOutputs(outputPaths), registry(registry), abiResolver(function, module) {}

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
    LogicalResult resolveStorageOutputs();
    LogicalResult initializeStorageGraph();
    LogicalResult buildStructuredDependencies(Region &region, DenseMap<unsigned, unsigned> &versions);
    FailureOr<unsigned> createStorageVersion(unsigned identity, StorageVersionKind kind, Operation *operation,
                                             ArrayRef<unsigned> incomingVersions, Value storedValue = {});
    void addStorageLoadDependencies(LoadOp load, unsigned version);
    LogicalResult bindStorageRoots(const DenseMap<unsigned, unsigned> &finalVersions);
    void computeActivity();
    LogicalResult classifyStorageOwnership();
    LogicalResult discoverRegions();
    LogicalResult collectOperations();
    bool anyActiveLeaf(Value value) const;
    bool hasActiveDescendant(Operation *operation) const;
    void emitDependencyError(Operation *operation, const Twine &message);
    InFlightDiagnostic emitFunctionError(const Twine &message);

    func::FuncOp function;
    ModuleOp module;
    ArrayRef<StringRef> requestedWrt;
    ArrayRef<StringRef> requestedOutputs;
    const VernonAutodiffRuleRegistry &registry;
    AutodiffValueAbiResolver abiResolver;
    VernonAutodiffAnalysisResult result;
    DenseMap<Value, unsigned> valueIndices;
    SmallVector<ValueNodes, 0> values;
    size_t graphNodeCount{};
    SmallVector<SmallVector<unsigned>> dependencies;
    SmallVector<SmallVector<unsigned>> users;
    SmallVector<unsigned> wrtNodes;
    SmallVector<unsigned> resultNodes;
    SmallVector<StorageLeafSelection> wrtStorageLeaves;
    SmallVector<StorageLeafSelection> resultStorageLeaves;
    DenseMap<std::pair<Value, unsigned>, unsigned> selectedResultNodes;
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
        FailureOr<unsigned> node = checkedUnsigned(graphNodeCount);
        if (failed(node))
            return failure();
        entry.nodes[leafIndex] = *node;
        ++graphNodeCount;
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
    std::optional<std::string> expectedHash;
    for (Value value : comparedValues) {
        if (value.getType().isIntOrIndex())
            continue;
        FailureOr<unsigned> valueIndex = addValue(value);
        if (failed(valueIndex)) {
            emitDependencyError(operation, "cannot resolve structured control-flow Value ABI metadata");
            return;
        }
        const ValueAbiLayout &layout = values[*valueIndex].layout;
        if (llvm::none_of(layout.leaves, [&](const ValueAbiLeaf &leaf) { return isDifferentiable(leaf); }))
            continue;
        if (expectedHash && *expectedHash != layout.layoutHash) {
            emitDependencyError(operation, "structured control-flow values have incompatible canonical Value ABIs");
            return;
        }
        expectedHash = layout.layoutHash;
    }
}

void AutodiffAnalysisBuilder::addAllDependencies(Value resultValue, ValueRange operands) {
    if (resultValue.getType().isIntOrIndex())
        return;
    FailureOr<unsigned> resultIndex = addValue(resultValue);
    if (failed(resultIndex)) {
        Operation *defining = resultValue.getDefiningOp();
        emitDependencyError(defining,
                            Twine("cannot resolve the result's canonical Value ABI for ") +
                                (defining ? defining->getName().getStringRef() : StringRef("block argument")));
        return;
    }
    for (auto [resultLeafIndex, resultLeaf] : llvm::enumerate(values[*resultIndex].layout.leaves)) {
        if (!isDifferentiable(resultLeaf))
            continue;
        std::optional<unsigned> resultNode = getNode(resultValue, resultLeafIndex);
        if (!resultNode)
            continue;
        for (Value operand : operands) {
            if (operand.getType().isIntOrIndex())
                continue;
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
    if (isa<LoadOp, StoreOp, ReduceSumOp, ScatterAddOp, AtomicOp>(operation))
        return;
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
    if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
        auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        for (auto [index, value] : llvm::enumerate(forOp.getResults())) {
            addAllDependencies(value, ValueRange(yield.getResults()[index]));
            SmallVector<Value> carried = {forOp.getInitArgs()[index], forOp.getRegionIterArgs()[index],
                                          yield.getResults()[index], value};
            verifySameAbi(operation, carried);
        }
        for (auto [index, argument] : llvm::enumerate(forOp.getRegionIterArgs())) {
            addAllDependencies(argument, ValueRange(forOp.getInitArgs()[index]));
            addAllDependencies(argument, ValueRange(yield.getResults()[index]));
        }
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

FailureOr<unsigned> AutodiffAnalysisBuilder::createStorageVersion(unsigned identity, StorageVersionKind kind,
                                                                  Operation *operation,
                                                                  ArrayRef<unsigned> incomingVersions,
                                                                  Value storedValue) {
    if (identity >= result.storageIdentities.size())
        return failure();
    Value binding = result.storageIdentities[identity].binding;
    auto valueIndex = valueIndices.find(binding);
    if (valueIndex == valueIndices.end())
        return failure();
    const ValueAbiLayout &layout = values[valueIndex->second].layout;
    FailureOr<unsigned> versionId = checkedUnsigned(result.storageVersions.size());
    if (failed(versionId))
        return failure();
    result.storageVersions.push_back(
        AutodiffStorageVersion{*versionId, identity, kind, operation, SmallVector<unsigned>(incomingVersions)});
    SmallVector<unsigned> versionNodes(layout.leaves.size(), kInvalidNode);
    for (auto [leafIndex, leaf] : llvm::enumerate(layout.leaves)) {
        if (!isDifferentiable(leaf))
            continue;
        FailureOr<unsigned> node = checkedUnsigned(graphNodeCount);
        if (failed(node))
            return failure();
        versionNodes[leafIndex] = *node;
        ++graphNodeCount;
        dependencies.emplace_back();
        users.emplace_back();
        for (unsigned incoming : incomingVersions) {
            if (incoming >= result.storageVersionNodes.size() ||
                leafIndex >= result.storageVersionNodes[incoming].size())
                return failure();
            unsigned incomingNode = result.storageVersionNodes[incoming][leafIndex];
            if (incomingNode != kInvalidNode)
                addDependency(*node, incomingNode);
        }
        if (storedValue) {
            FailureOr<unsigned> storedIndex = addValue(storedValue);
            if (failed(storedIndex))
                return failure();
            for (auto [storedLeafIndex, storedLeaf] : llvm::enumerate(values[*storedIndex].layout.leaves))
                if (isDifferentiable(storedLeaf))
                    if (std::optional<unsigned> storedNode = getNode(storedValue, storedLeafIndex))
                        addDependency(*node, *storedNode);
        }
    }
    result.storageVersionNodes.push_back(std::move(versionNodes));
    return *versionId;
}

void AutodiffAnalysisBuilder::addStorageLoadDependencies(LoadOp load, unsigned version) {
    FailureOr<unsigned> resultIndex = addValue(load.getResult());
    if (failed(resultIndex) || version >= result.storageVersionNodes.size()) {
        emitDependencyError(load, "cannot resolve Storage load effect metadata");
        return;
    }
    for (auto [resultLeafIndex, resultLeaf] : llvm::enumerate(values[*resultIndex].layout.leaves)) {
        if (!isDifferentiable(resultLeaf))
            continue;
        std::optional<unsigned> resultNode = getNode(load.getResult(), resultLeafIndex);
        if (!resultNode)
            continue;
        for (unsigned versionNode : result.storageVersionNodes[version])
            if (versionNode != kInvalidNode)
                addDependency(*resultNode, versionNode);
    }
}

LogicalResult AutodiffAnalysisBuilder::buildStructuredDependencies(Region &region,
                                                                   DenseMap<unsigned, unsigned> &versions) {
    if (region.empty())
        return success();
    for (Operation &operation : region.front().without_terminator()) {
        if (auto load = dyn_cast<LoadOp>(operation)) {
            auto identity = result.storageIdentityIndices.find(load.getStorage());
            if (identity == result.storageIdentityIndices.end() || !versions.contains(identity->second))
                return load.emitError("autodiff Storage load has no effect identity");
            FailureOr<unsigned> effectId = checkedUnsigned(result.storageEffects.size());
            if (failed(effectId))
                return load.emitError("autodiff Storage effect count exceeds the analysis representation");
            result.storageEffects.push_back(
                AutodiffStorageEffect{*effectId, identity->second, load, versions[identity->second], std::nullopt});
            result.storageEffectIndices.try_emplace(load, *effectId);
            addStorageLoadDependencies(load, versions[identity->second]);
            continue;
        }
        if (auto store = dyn_cast<StoreOp>(operation)) {
            auto identity = result.storageIdentityIndices.find(store.getStorage());
            if (identity == result.storageIdentityIndices.end() || !versions.contains(identity->second))
                return store.emitError("autodiff Storage store has no effect identity");
            const unsigned before = versions[identity->second];
            FailureOr<unsigned> after =
                createStorageVersion(identity->second, StorageVersionKind::Write, store, {before}, store.getValue());
            if (failed(after))
                return store.emitError("cannot create autodiff Storage write version");
            FailureOr<unsigned> effectId = checkedUnsigned(result.storageEffects.size());
            if (failed(effectId))
                return store.emitError("autodiff Storage effect count exceeds the analysis representation");
            result.storageEffects.push_back(AutodiffStorageEffect{*effectId, identity->second, store, before, *after});
            result.storageEffectIndices.try_emplace(store, *effectId);
            versions[identity->second] = *after;
            continue;
        }
        if (isa<ReduceSumOp, ScatterAddOp, AtomicOp>(operation)) {
            Value storage;
            Value contribution;
            bool supportedAdd = true;
            if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
                storage = reduce.getStorage();
                contribution = reduce.getValue();
            } else if (auto scatter = dyn_cast<ScatterAddOp>(operation)) {
                storage = scatter.getStorage();
                contribution = scatter.getValue();
            } else {
                auto atomic = cast<AtomicOp>(operation);
                storage = atomic.getStorage();
                contribution = atomic.getValue();
                supportedAdd = atomic.getAtomicKind() == "add" &&
                               isa<FloatType>(cast<TensorViewType>(storage.getType()).getElementType());
            }
            auto identity = result.storageIdentityIndices.find(storage);
            if (identity == result.storageIdentityIndices.end() || !versions.contains(identity->second))
                return operation.emitError("autodiff additive Storage effect has no identity");
            const unsigned before = versions[identity->second];
            FailureOr<unsigned> after =
                createStorageVersion(identity->second, StorageVersionKind::AdditiveWrite, &operation, {before},
                                     supportedAdd ? contribution : Value{});
            if (failed(after))
                return operation.emitError("cannot create autodiff additive Storage version");
            FailureOr<unsigned> effectId = checkedUnsigned(result.storageEffects.size());
            if (failed(effectId))
                return operation.emitError("autodiff Storage effect count exceeds the analysis representation");
            result.storageEffects.push_back(
                AutodiffStorageEffect{*effectId, identity->second, &operation, before, *after});
            result.storageEffectIndices.try_emplace(&operation, *effectId);
            versions[identity->second] = *after;

            if (auto atomic = dyn_cast<AtomicOp>(operation)) {
                FailureOr<unsigned> resultIndex = addValue(atomic.getResult());
                if (failed(resultIndex))
                    return atomic.emitError("cannot resolve atomic old-value result ABI");
                for (auto [leafIndex, leaf] : llvm::enumerate(values[*resultIndex].layout.leaves)) {
                    if (!isDifferentiable(leaf))
                        continue;
                    std::optional<unsigned> resultNode = getNode(atomic.getResult(), leafIndex);
                    unsigned versionNode = result.storageVersionNodes[before][leafIndex];
                    if (resultNode && versionNode != kInvalidNode)
                        addDependency(*resultNode, versionNode);
                }
            }
            continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
            buildDependencies(ifOp);
            DenseMap<unsigned, unsigned> thenVersions = versions;
            DenseMap<unsigned, unsigned> elseVersions = versions;
            if (failed(buildStructuredDependencies(ifOp.getThenRegion(), thenVersions)) ||
                failed(buildStructuredDependencies(ifOp.getElseRegion(), elseVersions)))
                return failure();
            for (const AutodiffStorageIdentity &identity : result.storageIdentities) {
                unsigned thenVersion = thenVersions[identity.id];
                unsigned elseVersion = elseVersions[identity.id];
                if (thenVersion == elseVersion) {
                    versions[identity.id] = thenVersion;
                    continue;
                }
                FailureOr<unsigned> merged =
                    createStorageVersion(identity.id, StorageVersionKind::IfMerge, ifOp, {thenVersion, elseVersion});
                if (failed(merged))
                    return ifOp.emitError("cannot create autodiff Storage branch merge version");
                versions[identity.id] = *merged;
            }
            continue;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
            buildDependencies(forOp);
            DenseMap<unsigned, unsigned> loopVersions;
            for (const AutodiffStorageIdentity &identity : result.storageIdentities) {
                FailureOr<unsigned> phi =
                    createStorageVersion(identity.id, StorageVersionKind::ForPhi, forOp, {versions[identity.id]});
                if (failed(phi))
                    return forOp.emitError("cannot create autodiff Storage for-loop phi version");
                loopVersions[identity.id] = *phi;
            }
            DenseMap<unsigned, unsigned> bodyVersions = loopVersions;
            if (failed(buildStructuredDependencies(forOp.getRegion(), bodyVersions)))
                return failure();
            for (const AutodiffStorageIdentity &identity : result.storageIdentities) {
                const unsigned phi = loopVersions[identity.id];
                const unsigned backedge = bodyVersions[identity.id];
                result.storageVersions[phi].incomingVersions.push_back(backedge);
                for (auto [leafIndex, phiNode] : llvm::enumerate(result.storageVersionNodes[phi])) {
                    if (phiNode == kInvalidNode)
                        continue;
                    unsigned backedgeNode = result.storageVersionNodes[backedge][leafIndex];
                    if (backedgeNode != kInvalidNode)
                        addDependency(phiNode, backedgeNode);
                }
                FailureOr<unsigned> exit = createStorageVersion(identity.id, StorageVersionKind::ForExit, forOp, {phi});
                if (failed(exit))
                    return forOp.emitError("cannot create autodiff Storage for-loop exit version");
                versions[identity.id] = *exit;
            }
            continue;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
            buildDependencies(whileOp);
            DenseMap<unsigned, unsigned> phiVersions;
            DenseMap<unsigned, unsigned> loopVersions;
            for (const AutodiffStorageIdentity &identity : result.storageIdentities) {
                FailureOr<unsigned> phi =
                    createStorageVersion(identity.id, StorageVersionKind::WhilePhi, whileOp, {versions[identity.id]});
                if (failed(phi))
                    return whileOp.emitError("cannot create autodiff Storage loop phi version");
                phiVersions[identity.id] = *phi;
                loopVersions[identity.id] = *phi;
            }
            if (failed(buildStructuredDependencies(whileOp.getBefore(), loopVersions)))
                return failure();
            DenseMap<unsigned, unsigned> bodyVersions = loopVersions;
            if (failed(buildStructuredDependencies(whileOp.getAfter(), bodyVersions)))
                return failure();
            for (const AutodiffStorageIdentity &identity : result.storageIdentities) {
                const unsigned phi = phiVersions[identity.id];
                const unsigned backedge = bodyVersions[identity.id];
                result.storageVersions[phi].incomingVersions.push_back(backedge);
                for (auto [leafIndex, phiNode] : llvm::enumerate(result.storageVersionNodes[phi])) {
                    if (phiNode == kInvalidNode)
                        continue;
                    unsigned backedgeNode = result.storageVersionNodes[backedge][leafIndex];
                    if (backedgeNode != kInvalidNode)
                        addDependency(phiNode, backedgeNode);
                }
                FailureOr<unsigned> exit = createStorageVersion(identity.id, StorageVersionKind::WhileExit, whileOp,
                                                                {loopVersions[identity.id]});
                if (failed(exit))
                    return whileOp.emitError("cannot create autodiff Storage loop exit version");
                versions[identity.id] = *exit;
            }
            continue;
        }
        buildDependencies(&operation);
    }
    return dependencyStatus;
}

LogicalResult AutodiffAnalysisBuilder::initializeStorageGraph() {
    DenseMap<unsigned, unsigned> versions;
    for (Value argument : function.getArguments()) {
        if (!isa<TensorViewType>(argument.getType()))
            continue;
        FailureOr<unsigned> valueIndex = addValue(argument);
        if (failed(valueIndex))
            return emitFunctionError("cannot resolve TensorView Storage ABI");
        FailureOr<unsigned> identityId = checkedUnsigned(result.storageIdentities.size());
        if (failed(identityId))
            return emitFunctionError("Storage identity count exceeds the analysis representation");
        result.storageIdentities.push_back(AutodiffStorageIdentity{*identityId, argument});
        result.storageIdentityIndices.try_emplace(argument, *identityId);
        FailureOr<unsigned> entry = createStorageVersion(*identityId, StorageVersionKind::Entry, nullptr, {});
        if (failed(entry))
            return emitFunctionError("cannot create Storage entry version");
        versions[*identityId] = *entry;
    }
    WalkResult allocationResult = function.walk([&](WorkgroupAllocOp allocation) {
        Value binding = allocation.getResult();
        FailureOr<unsigned> valueIndex = addValue(binding);
        if (failed(valueIndex))
            return allocation.emitError("cannot resolve workgroup Storage ABI"), WalkResult::interrupt();
        FailureOr<unsigned> identityId = checkedUnsigned(result.storageIdentities.size());
        if (failed(identityId))
            return allocation.emitError("Storage identity count exceeds the analysis representation"),
                   WalkResult::interrupt();
        result.storageIdentities.push_back(AutodiffStorageIdentity{*identityId, binding});
        result.storageIdentityIndices.try_emplace(binding, *identityId);
        FailureOr<unsigned> entry = createStorageVersion(*identityId, StorageVersionKind::Entry, nullptr, {});
        if (failed(entry))
            return allocation.emitError("cannot create workgroup Storage entry version"), WalkResult::interrupt();
        versions[*identityId] = *entry;
        return WalkResult::advance();
    });
    if (allocationResult.wasInterrupted())
        return failure();
    if (failed(buildStructuredDependencies(function.getBody(), versions)))
        return failure();
    return bindStorageRoots(versions);
}

LogicalResult AutodiffAnalysisBuilder::bindStorageRoots(const DenseMap<unsigned, unsigned> &finalVersions) {
    for (const StorageLeafSelection &selection : wrtStorageLeaves) {
        auto identity = result.storageIdentityIndices.find(selection.binding);
        if (identity == result.storageIdentityIndices.end())
            return emitFunctionError("wrt Storage has no effect identity");
        unsigned entryVersion =
            llvm::find_if(result.storageVersions, [&](const AutodiffStorageVersion &version) {
                return version.identity == identity->second && version.kind == StorageVersionKind::Entry;
            })->id;
        wrtNodes.push_back(result.storageVersionNodes[entryVersion][selection.abiLeafIndex]);
    }
    for (const StorageLeafSelection &selection : resultStorageLeaves) {
        auto identity = result.storageIdentityIndices.find(selection.binding);
        if (identity == result.storageIdentityIndices.end() || !finalVersions.contains(identity->second))
            return emitFunctionError("Storage objective has no final effect version");
        unsigned node = result.storageVersionNodes[finalVersions.lookup(identity->second)][selection.abiLeafIndex];
        resultNodes.push_back(node);
        selectedResultNodes[{selection.binding, selection.abiLeafIndex}] = node;
    }
    return success();
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
        if (auto view = dyn_cast<TensorViewType>(current))
            current = view.getElementType();
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
            Type derivativeType = leaf.shape.empty() ? *getAutodiffDerivativeScalarType(leaf.scalarType)
                                                     : static_cast<Type>(RankedTensorType::get(
                                                           SmallVector<int64_t>(leaf.shape.begin(), leaf.shape.end()),
                                                           *getAutodiffDerivativeScalarType(leaf.scalarType)));
            result.wrtLeaves.push_back(AutodiffLeaf{argument, static_cast<unsigned>(leafIndex),
                                                    components.front().str(), path, leaf.scalarType, derivativeType,
                                                    leaf.dtype, leaf.shape});
            if (isa<TensorViewType>(argument.getType())) {
                wrtStorageLeaves.push_back(StorageLeafSelection{argument, static_cast<unsigned>(leafIndex)});
            } else {
                wrtNodes.push_back(*getNode(argument, leafIndex));
            }
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
                    AutodiffLeaf{value, static_cast<unsigned>(leafIndex), root.str(), std::move(path), leaf.scalarType,
                                 leaf.shape.empty() ? *getAutodiffDerivativeScalarType(leaf.scalarType)
                                                    : static_cast<Type>(RankedTensorType::get(
                                                          SmallVector<int64_t>(leaf.shape.begin(), leaf.shape.end()),
                                                          *getAutodiffDerivativeScalarType(leaf.scalarType))),
                                 leaf.dtype, leaf.shape});
            }
        }
    }
    if (resultNodes.empty())
        return emitFunctionError("function results have no differentiable floating leaves");
    return success();
}

LogicalResult AutodiffAnalysisBuilder::resolveStorageOutputs() {
    if (function.getNumResults() != 0)
        return emitFunctionError("Storage-objective VJP requires a void compute function");
    if (requestedOutputs.empty())
        return emitFunctionError("requires at least one Storage output path");
    std::set<std::string> canonicalPaths;
    for (StringRef path : requestedOutputs) {
        SmallVector<StringRef> components;
        path.split(components, '.', -1, false);
        if (components.empty() || llvm::any_of(components, [](StringRef item) { return item.empty(); }))
            return emitFunctionError(Twine("invalid Storage output path '") + path + "'");
        std::optional<unsigned> argumentIndex;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            auto sourceName = function.getArgAttrOfType<StringAttr>(index, "vernon.source_name");
            if (sourceName && sourceName.getValue() == components.front()) {
                argumentIndex = index;
                break;
            }
        }
        if (!argumentIndex)
            return emitFunctionError(Twine("Storage output path '") + path + "' names no entry parameter");
        Value argument = function.getArgument(*argumentIndex);
        auto view = dyn_cast<TensorViewType>(argument.getType());
        if (!view)
            return emitFunctionError(Twine("Storage output path '") + path + "' is not a TensorView");
        if (view.getAccess() == "read")
            return emitFunctionError(Twine("Storage output path '") + path + "' is not writable");
        Attribute dtypeMetadata = function.getArgAttr(*argumentIndex, "vernon.abi_leaf_dtypes");
        auto dtypeArray = dyn_cast_or_null<ArrayAttr>(dtypeMetadata);
        if (dtypeMetadata && !dtypeArray)
            return emitFunctionError(Twine("Storage output path '") + path + "' has malformed ABI dtype metadata");
        FailureOr<SmallVector<StringRef>> dtypes = getDtypes(dtypeArray);
        FailureOr<unsigned> valueIndex = failed(dtypes) ? FailureOr<unsigned>(failure()) : addValue(argument, *dtypes);
        if (failed(valueIndex))
            return emitFunctionError(Twine("Storage output path '") + path + "' has no canonical Value ABI");
        SmallVector<ValueAbiPathComponent> prefix;
        Type current = view.getElementType();
        for (StringRef component : ArrayRef<StringRef>(components).drop_front()) {
            if (auto structure = dyn_cast<StructType>(current)) {
                FailureOr<ResolvedStructFields> fields = resolveNamedStructFields(structure, module);
                if (failed(fields))
                    return emitFunctionError(Twine("cannot resolve Struct in Storage output path '") + path + "'");
                auto field = llvm::find_if(fields->fields,
                                           [&](const ResolvedStructField &item) { return item.name == component; });
                if (field == fields->fields.end())
                    return emitFunctionError(Twine("Storage output path '") + path + "' names no Struct field");
                prefix.push_back(ValueAbiPathComponent::getField(component));
                current = field->type;
            } else if (auto tuple = dyn_cast<TupleType>(current)) {
                uint64_t index = 0;
                if (component.getAsInteger(10, index) || index >= tuple.size())
                    return emitFunctionError(Twine("Storage output path '") + path + "' has an invalid Tuple index");
                prefix.push_back(ValueAbiPathComponent::getIndex(index));
                current = tuple.getType(index);
            } else {
                return emitFunctionError(Twine("Storage output path '") + path +
                                         "' does not resolve through an aggregate");
            }
        }
        unsigned matches = 0;
        for (auto [leafIndex, leaf] : llvm::enumerate(values[*valueIndex].layout.leaves)) {
            if (!hasPathPrefix(leaf.path, prefix) || !isDifferentiable(leaf))
                continue;
            std::string leafPath = appendAbiPath(components.front(), leaf.path);
            if (!canonicalPaths.insert(leafPath).second)
                return emitFunctionError(Twine("duplicate canonical Storage output leaf '") + leafPath + "'");
            Type derivativeType = leaf.shape.empty() ? *getAutodiffDerivativeScalarType(leaf.scalarType)
                                                     : static_cast<Type>(RankedTensorType::get(
                                                           SmallVector<int64_t>(leaf.shape.begin(), leaf.shape.end()),
                                                           *getAutodiffDerivativeScalarType(leaf.scalarType)));
            result.activeResultLeaves.push_back(AutodiffLeaf{argument, static_cast<unsigned>(leafIndex),
                                                             components.front().str(), std::move(leafPath),
                                                             leaf.scalarType, derivativeType, leaf.dtype, leaf.shape});
            resultStorageLeaves.push_back(StorageLeafSelection{argument, static_cast<unsigned>(leafIndex)});
            ++matches;
        }
        if (!matches)
            return emitFunctionError(Twine("Storage output path '") + path + "' has no differentiable floating leaves");
    }
    if (resultStorageLeaves.empty())
        return emitFunctionError("selected Storage outputs have no differentiable floating leaves");
    return success();
}

void AutodiffAnalysisBuilder::computeActivity() {
    needed.resize(graphNodeCount);
    influenced.resize(graphNodeCount);
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

    llvm::erase_if(result.activeResultLeaves, [&](const AutodiffLeaf &leaf) {
        unsigned node = kInvalidNode;
        if (auto found = selectedResultNodes.find({leaf.value, leaf.abiLeafIndex}); found != selectedResultNodes.end())
            node = found->second;
        else if (std::optional<unsigned> valueNode = getNode(leaf.value, leaf.abiLeafIndex))
            node = *valueNode;
        return node == kInvalidNode || !active.test(node);
    });
    for (const ValueNodes &entry : values) {
        result.valueAbiIndices.try_emplace(entry.value, static_cast<unsigned>(result.valueAbis.size()));
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
    result.activeStorageVersionLeaves.resize(result.storageVersions.size());
    for (const AutodiffStorageVersion &version : result.storageVersions)
        for (auto [leafIndex, node] : llvm::enumerate(result.storageVersionNodes[version.id]))
            if (node != kInvalidNode && active.test(node))
                result.activeStorageVersionLeaves[version.id].push_back(static_cast<unsigned>(leafIndex));
    for (const AutodiffStorageIdentity &identity : result.storageIdentities) {
        auto existing = llvm::find_if(
            result.activeValues, [&](const AutodiffValueActivity &item) { return item.value == identity.binding; });
        AutodiffValueActivity storageActivity;
        storageActivity.value = identity.binding;
        for (const AutodiffStorageVersion &version : result.storageVersions) {
            if (version.identity != identity.id)
                continue;
            for (auto [leafIndex, node] : llvm::enumerate(result.storageVersionNodes[version.id]))
                if (node != kInvalidNode && active.test(node) &&
                    !llvm::is_contained(storageActivity.activeAbiLeaves, static_cast<unsigned>(leafIndex)))
                    storageActivity.activeAbiLeaves.push_back(static_cast<unsigned>(leafIndex));
        }
        if (existing == result.activeValues.end()) {
            if (!storageActivity.activeAbiLeaves.empty())
                result.activeValues.push_back(std::move(storageActivity));
        } else {
            for (unsigned leaf : storageActivity.activeAbiLeaves)
                if (!llvm::is_contained(existing->activeAbiLeaves, leaf))
                    existing->activeAbiLeaves.push_back(leaf);
        }
    }
}

LogicalResult AutodiffAnalysisBuilder::classifyStorageOwnership() {
    for (AutodiffStorageIdentity &identity : result.storageIdentities) {
        identity.externalGradientDestination =
            llvm::any_of(result.wrtLeaves, [&](const AutodiffLeaf &leaf) { return leaf.value == identity.binding; });
        auto view = dyn_cast<TensorViewType>(identity.binding.getType());
        if (!view)
            return emitFunctionError("Storage identity has no TensorView type");
        if (identity.externalGradientDestination) {
            bool hasActiveLoad = false;
            bool invocationOwned = true;
            for (const AutodiffStorageEffect &effect : result.storageEffects) {
                if (effect.identity != identity.id || !isa<LoadOp>(effect.operation) ||
                    !hasActiveStorageEffect(result, effect))
                    continue;
                hasActiveLoad = true;
                invocationOwned &= static_cast<bool>(proveInvocationOwnedIndex(storageIndices(effect.operation)));
            }
            if (hasActiveLoad)
                identity.externalGradientOwnership =
                    view.getAddressSpace() == "workgroup" ? AutodiffExternalGradientOwnership::WorkgroupShared
                    : invocationOwned                     ? AutodiffExternalGradientOwnership::InvocationPrivate
                                                          : AutodiffExternalGradientOwnership::AtomicShared;
        }
        for (const AutodiffStorageEffect &effect : result.storageEffects) {
            if (effect.identity != identity.id)
                continue;
            auto atomic = dyn_cast<AtomicOp>(effect.operation);
            if (!atomic || !result.isActive(atomic.getResult(), 0))
                continue;
            if (view.getAddressSpace() != "device" || !proveInvocationOwnedIndex(atomic.getIndices()))
                return atomic.emitError(
                    "active atomic old-value result requires a proven lane-exclusive device index mapping");
        }
        if (view.getAccess() == "read")
            continue;
        if (!needsInternalStorageAdjoint(result, identity))
            continue;

        if (view.getAddressSpace() == "workgroup") {
            identity.internalAdjointOwnership = AutodiffInternalAdjointOwnership::WorkgroupCoupled;
            continue;
        }
        if (view.getAddressSpace() != "device" || failed(verifyLaneOwnedDeviceAdjoint(result, identity)))
            return emitFunctionError(
                "device Storage requiring an internal adjoint has no proven lane-owned injective index mapping");
        identity.internalAdjointOwnership = AutodiffInternalAdjointOwnership::LanePrivate;
    }
    return success();
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
            llvm::any_of(nested->getOperands(), [&](Value value) { return anyActiveLeaf(value); }))
            found = true;
    });
    return found;
}

LogicalResult AutodiffAnalysisBuilder::discoverRegions() {
    DenseMap<Operation *, unsigned> ordinals;
    WalkResult walkResult = function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
        if (!isa<scf::IfOp, scf::ForOp, scf::WhileOp>(operation))
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
        result.regions.push_back(AutodiffRegion{operation, *ordinal, parent});
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
        if (const AutodiffStorageEffect *storageEffect = result.getStorageEffect(operation)) {
            const ValueAbiLayout *layout =
                result.getValueAbi(result.storageIdentities[storageEffect->identity].binding);
            if (layout)
                for (unsigned leaf = 0; leaf < layout->leaves.size(); ++leaf)
                    operationActive |= result.isActiveStorageVersion(storageEffect->versionBefore, leaf) ||
                                       (storageEffect->versionAfter &&
                                        result.isActiveStorageVersion(*storageEffect->versionAfter, leaf));
        } else if (effect != AutodiffEffectKind::Pure) {
            operationActive |=
                llvm::any_of(operation->getOperands(), [&](Value value) { return anyActiveLeaf(value); });
        }
        if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(operation))
            operationActive |= hasActiveDescendant(operation);
        if (operation->getNumRegions() != 0 && !isa<scf::IfOp, scf::ForOp, scf::WhileOp>(operation) &&
            operationActive) {
            operation->emitError("autodiff analysis does not support an active unstructured region operation");
            status = failure();
            return WalkResult::interrupt();
        }
        result.operations.push_back(AutodiffOperationActivity{operation, effect, operationActive});
        if (operationActive)
            result.activeOperationSet.insert(operation);
        if (auto load = dyn_cast<LoadOp>(operation)) {
            const AutodiffStorageEffect *storageEffect = result.getStorageEffect(operation);
            if (!storageEffect) {
                load.emitError("Storage load has no exact-version effect metadata");
                status = failure();
                return WalkResult::interrupt();
            }
            AutodiffLoadInfo loadInfo;
            loadInfo.operation = operation;
            loadInfo.identity = storageEffect->identity;
            loadInfo.versionBefore = storageEffect->versionBefore;
            llvm::append_range(loadInfo.indices, load.getIndices());
            for (Value index : load.getIndices())
                loadInfo.indexProvenance.push_back(classifyIndexProvenance(index, function));
            if (auto view = dyn_cast<TensorViewType>(load.getStorage().getType());
                view && view.getAccess() == "read" && isa<BlockArgument>(load.getStorage()))
                loadInfo.stability = AutodiffStorageStabilityRequirement::RetainedExactVersion;
            FailureOr<unsigned> loadIndex = checkedUnsigned(result.loads.size());
            if (failed(loadIndex)) {
                load.emitError("Storage load count exceeds the analysis representation");
                status = failure();
                return WalkResult::interrupt();
            }
            result.loadInfoIndices.try_emplace(operation, *loadIndex);
            result.loads.push_back(std::move(loadInfo));
        }
        if (operationActive &&
            (effect == AutodiffEffectKind::Unsupported || effect == AutodiffEffectKind::ExternallyVisible)) {
            operation->emitError() << "autodiff analysis does not support active " << stringifyAutodiffEffect(effect)
                                   << " operation";
            status = failure();
            return WalkResult::interrupt();
        }
        if (operationActive) {
            if (auto atomic = dyn_cast<AtomicOp>(operation)) {
                auto view = cast<TensorViewType>(atomic.getStorage().getType());
                if (atomic.getAtomicKind() != "add" || !isa<FloatType>(view.getElementType())) {
                    atomic.emitError("autodiff supports only floating-point additive atomics");
                    status = failure();
                    return WalkResult::interrupt();
                }
                if (result.isActive(atomic.getResult(), 0) &&
                    (view.getAddressSpace() != "device" || !proveInvocationOwnedIndex(atomic.getIndices()))) {
                    atomic.emitError(
                        "active atomic old-value result requires a proven lane-exclusive device index mapping");
                    status = failure();
                    return WalkResult::interrupt();
                }
            }
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
    if (failed(resolveWrt()) || failed(requestedOutputs.empty() ? resolveResults() : resolveStorageOutputs()))
        return failure();
    if (failed(initializeStorageGraph()) || failed(dependencyStatus))
        return failure();
    computeActivity();
    if (failed(classifyStorageOwnership()))
        return failure();
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

bool VernonAutodiffAnalysisResult::hasAnyActiveLeaf(Value value) const {
    auto activity = llvm::find_if(activeValues, [&](const AutodiffValueActivity &item) { return item.value == value; });
    return activity != activeValues.end() && !activity->activeAbiLeaves.empty();
}

const ValueAbiLayout *VernonAutodiffAnalysisResult::getValueAbi(Value value) const {
    auto found = valueAbiIndices.find(value);
    return found == valueAbiIndices.end() ? nullptr : &valueAbis[found->second].layout;
}

const AutodiffStorageIdentity *VernonAutodiffAnalysisResult::getStorageIdentity(Value binding) const {
    auto found = storageIdentityIndices.find(binding);
    return found == storageIdentityIndices.end() ? nullptr : &storageIdentities[found->second];
}

const AutodiffStorageEffect *VernonAutodiffAnalysisResult::getStorageEffect(Operation *operation) const {
    auto found = storageEffectIndices.find(operation);
    return found == storageEffectIndices.end() ? nullptr : &storageEffects[found->second];
}

const AutodiffLoadInfo *VernonAutodiffAnalysisResult::getLoadInfo(Operation *operation) const {
    auto found = loadInfoIndices.find(operation);
    return found == loadInfoIndices.end() ? nullptr : &loads[found->second];
}

bool VernonAutodiffAnalysisResult::isActiveStorageVersion(unsigned version, unsigned abiLeafIndex) const {
    if (version >= activeStorageVersionLeaves.size())
        return false;
    return llvm::is_contained(activeStorageVersionLeaves[version], abiLeafIndex);
}

bool VernonAutodiffAnalysisResult::isActive(Operation *operation) const {
    return activeOperationSet.contains(operation);
}

FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths) {
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    return analyzeAutodiffFunction(function, wrtPaths, registry);
}

FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths,
                                                                const VernonAutodiffRuleRegistry &registry) {
    return AutodiffAnalysisBuilder(function, wrtPaths, {}, registry).run();
}

FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths,
                                                                ArrayRef<StringRef> outputPaths,
                                                                const VernonAutodiffRuleRegistry &registry) {
    return AutodiffAnalysisBuilder(function, wrtPaths, outputPaths, registry).run();
}

bool isDifferentiableAutodiffLeaf(Type scalarType, StringRef logicalDtype) {
    if (!(scalarType.isF16() || scalarType.isF32() || scalarType.isF64()))
        return false;
    return logicalDtype.empty() || logicalDtype == "f16" || logicalDtype == "f32" || logicalDtype == "f64";
}

AutodiffEffectKind classifyAutodiffEffect(Operation *operation) {
    if (isa<LoadOp, PhysicalLoadOp>(operation))
        return AutodiffEffectKind::StorageRead;
    if (isa<StoreOp, PhysicalStoreOp>(operation))
        return AutodiffEffectKind::StorageWrite;
    if (isa<AtomicOp>(operation))
        return AutodiffEffectKind::Atomic;
    if (auto atomic = dyn_cast<PhysicalAtomicOp>(operation))
        return atomic.getStorage().getType().getAddressSpace() == "device" ? AutodiffEffectKind::ExternallyVisible
                                                                           : AutodiffEffectKind::Atomic;
    if (isa<ReduceSumOp, ScatterAddOp>(operation))
        return AutodiffEffectKind::StorageWrite;
    if (isa<WorkgroupAllocOp>(operation))
        return AutodiffEffectKind::StorageWrite;
    if (auto barrier = dyn_cast<BarrierOp>(operation))
        return barrier.getScope() == "workgroup" ? AutodiffEffectKind::Barrier : AutodiffEffectKind::ExternallyVisible;
    if (isa<IntrinsicOp>(operation))
        return AutodiffEffectKind::Unsupported;
    // Structured containers inherit execution activity from their regions;
    // their nested operations carry the actual effect classifications.
    if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(operation))
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
