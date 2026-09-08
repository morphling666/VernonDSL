#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAttributeAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>
#include <set>
#include <string>

namespace mlir::vernon {
namespace {

struct LocationEndpoint {
    func::FuncOp function;
    unsigned index;
    Type type;
    int64_t location;
};

class WorkgroupUniformityAnalysis {
public:
    bool isUniform(Value value) {
        DenseMap<Value, bool> assumptions;
        DenseSet<Value> visiting;
        return isUniform(value, assumptions, visiting);
    }

    bool hasUniformControl(scf::WhileOp loop) {
        const size_t count = loop.getInits().size();
        SmallVector<bool> carried;
        carried.reserve(count);
        for (Value initial : loop.getInits())
            carried.push_back(isUniform(initial));

        bool conditionUniform = false;
        for (size_t iteration = 0; iteration <= count; ++iteration) {
            DenseMap<Value, bool> assumptions;
            for (auto [argument, uniform] : llvm::zip_equal(loop.getBeforeArguments(), carried))
                assumptions[argument] = uniform;
            DenseSet<Value> visiting;
            auto condition = dyn_cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
            if (!condition)
                return false;
            conditionUniform = isUniform(condition.getCondition(), assumptions, visiting);

            SmallVector<bool> afterUniform;
            afterUniform.reserve(condition.getArgs().size());
            for (Value argument : condition.getArgs()) {
                DenseSet<Value> argumentVisiting;
                afterUniform.push_back(isUniform(argument, assumptions, argumentVisiting));
            }
            assumptions.clear();
            for (auto [argument, uniform] : llvm::zip_equal(loop.getAfterArguments(), afterUniform))
                assumptions[argument] = uniform;
            auto yield = dyn_cast<scf::YieldOp>(loop.getAfter().front().getTerminator());
            if (!yield || yield.getNumOperands() != count)
                return false;
            SmallVector<bool> next;
            next.reserve(count);
            for (auto [previous, value] : llvm::zip_equal(carried, yield.getOperands())) {
                DenseSet<Value> yieldVisiting;
                next.push_back(previous && isUniform(value, assumptions, yieldVisiting));
            }
            if (next == carried)
                return conditionUniform;
            carried = std::move(next);
        }
        return false;
    }

private:
    static Value branchInvariantValue(Value value) {
        auto result = dyn_cast<OpResult>(value);
        if (!result)
            return value;
        auto ifOp = dyn_cast<scf::IfOp>(result.getOwner());
        if (!ifOp || ifOp.getElseRegion().empty())
            return value;
        auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
        auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
        Value thenValue = branchInvariantValue(thenYield.getOperand(result.getResultNumber()));
        Value elseValue = branchInvariantValue(elseYield.getOperand(result.getResultNumber()));
        return thenValue == elseValue ? thenValue : value;
    }

    bool isUniform(Value value, const DenseMap<Value, bool> &assumptions, DenseSet<Value> &visiting) {
        if (const auto assumed = assumptions.find(value); assumed != assumptions.end())
            return assumed->second;
        if (!visiting.insert(value).second)
            return false;
        const auto remove = llvm::make_scope_exit([&] { visiting.erase(value); });
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            auto function = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
            if (function) {
                auto builtin = function.getArgAttrOfType<StringAttr>(argument.getArgNumber(), kBuiltinAttrName);
                return !builtin || builtin.getValue() == "workgroup_id" || builtin.getValue() == "ad_tape_allocator" ||
                       builtin.getValue() == "ad_tape_root_region";
            }
            if (auto forOp = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())) {
                if (argument == forOp.getInductionVar())
                    return isUniform(forOp.getLowerBound(), assumptions, visiting) &&
                           isUniform(forOp.getUpperBound(), assumptions, visiting) &&
                           isUniform(forOp.getStep(), assumptions, visiting);
            }
            return false;
        }
        Operation *definition = value.getDefiningOp();
        if (!definition || isa<LoadOp, PhysicalLoadOp, func::CallOp>(definition))
            return false;
        if (auto result = dyn_cast<OpResult>(value)) {
            if (auto ifOp = dyn_cast<scf::IfOp>(definition)) {
                auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
                auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
                Value thenValue = branchInvariantValue(thenYield.getOperand(result.getResultNumber()));
                Value elseValue = branchInvariantValue(elseYield.getOperand(result.getResultNumber()));
                if (thenValue == elseValue)
                    return isUniform(thenValue, assumptions, visiting);
                if (!isUniform(ifOp.getCondition(), assumptions, visiting))
                    return false;
                DenseSet<Value> thenVisiting;
                DenseSet<Value> elseVisiting;
                return isUniform(thenValue, assumptions, thenVisiting) &&
                       isUniform(elseValue, assumptions, elseVisiting);
            }
        }
        if (definition->getNumRegions())
            return false;
        return llvm::all_of(definition->getOperands(),
                            [&](Value operand) { return isUniform(operand, assumptions, visiting); });
    }
};

LogicalResult verifyUniformControl(Operation *anchor, WorkgroupUniformityAnalysis &analysis) {
    for (Operation *parent = anchor->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
            if (!analysis.isUniform(ifOp.getCondition()))
                return anchor->emitError("workgroup barrier is nested under a lane-varying condition");
        } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            const bool replayingValidatedPrimalLoop = forOp.getUpperBound().getDefiningOp<AdReadExecutedCountOp>();
            if (!analysis.isUniform(forOp.getLowerBound()) ||
                (!replayingValidatedPrimalLoop && !analysis.isUniform(forOp.getUpperBound())) ||
                !analysis.isUniform(forOp.getStep()))
                return anchor->emitError("workgroup barrier is nested in a lane-varying loop");
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
            if (!whileOp.getBefore().hasOneBlock() || !whileOp.getAfter().hasOneBlock())
                return anchor->emitError("workgroup barrier requires a canonical scf.while");
            if (!analysis.hasUniformControl(whileOp))
                return anchor->emitError("workgroup barrier is nested in a lane-varying loop");
        } else if (isa<func::FuncOp>(parent)) {
            break;
        } else if (isa<AdCaptureOp>(parent)) {
            continue;
        } else if (parent->getNumRegions()) {
            return anchor->emitError("workgroup barrier is nested in unsupported region control flow");
        }
    }
    return success();
}

LogicalResult verifyUniformBarrier(BarrierOp barrier) {
    WorkgroupUniformityAnalysis analysis;
    return verifyUniformControl(barrier, analysis);
}

Operation *ancestorInBlock(Operation *operation, Block *block) {
    while (operation && operation->getBlock() != block)
        operation = operation->getParentOp();
    return operation;
}

std::optional<bool> structuredBefore(Operation *first, Operation *second) {
    for (Block *block = first->getBlock(); block;) {
        Operation *firstAnchor = ancestorInBlock(first, block);
        Operation *secondAnchor = ancestorInBlock(second, block);
        if (firstAnchor && secondAnchor && firstAnchor != secondAnchor)
            return firstAnchor->isBeforeInBlock(secondAnchor);
        Operation *parent = block->getParentOp();
        block = parent ? parent->getBlock() : nullptr;
    }
    return std::nullopt;
}

struct DeviceMemoryEffect {
    enum class Kind { Read, OrdinaryStore, AtomicRmw, ReduceSum, ScatterAdd };

    Operation *operation{};
    Value storage;
    Value owner;
    SmallVector<Value> indices;
    Kind kind{};
    bool reads{};
    bool writes{};
    bool singleInvocationConstrained{};
    std::optional<ConditionalIndexProof> ownershipProof;
};

class KernelMemoryPhaseAnalysis {
public:
    LogicalResult verify(func::FuncOp function) {
        function->removeAttr(kDispatchContractAttrName);
        SmallVector<BarrierOp> barriers;
        SmallVector<DeviceMemoryEffect, 8> effects;
        DenseSet<unsigned> unitGridAxes;
        auto workgroup = function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName);
        const bool unitWorkgroup = workgroup && workgroup.size() == 3 &&
                                   llvm::all_of(workgroup.asArrayRef(), [](int32_t value) { return value == 1; });
        function.walk([&](Operation *operation) {
            if (auto barrier = dyn_cast<BarrierOp>(operation)) {
                if (barrier.getScope() == "workgroup")
                    barriers.push_back(barrier);
                return;
            }
            DeviceMemoryEffect effect;
            effect.operation = operation;
            if (auto load = dyn_cast<LoadOp>(operation)) {
                effect.storage = load.getStorage();
                llvm::append_range(effect.indices, load.getIndices());
                effect.kind = DeviceMemoryEffect::Kind::Read;
                effect.reads = true;
            } else if (auto store = dyn_cast<StoreOp>(operation)) {
                effect.storage = store.getStorage();
                llvm::append_range(effect.indices, store.getIndices());
                effect.kind = DeviceMemoryEffect::Kind::OrdinaryStore;
                effect.writes = true;
            } else if (auto atomic = dyn_cast<AtomicOp>(operation)) {
                effect.storage = atomic.getStorage();
                llvm::append_range(effect.indices, atomic.getIndices());
                effect.kind = DeviceMemoryEffect::Kind::AtomicRmw;
                effect.reads = effect.writes = true;
            } else if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
                effect.storage = reduce.getStorage();
                llvm::append_range(effect.indices, reduce.getIndices());
                effect.kind = DeviceMemoryEffect::Kind::ReduceSum;
                effect.writes = true;
            } else if (auto scatter = dyn_cast<ScatterAddOp>(operation)) {
                effect.storage = scatter.getStorage();
                llvm::append_range(effect.indices, scatter.getIndices());
                effect.kind = DeviceMemoryEffect::Kind::ScatterAdd;
                effect.writes = true;
            } else {
                return;
            }
            auto view = dyn_cast<TensorViewType>(effect.storage.getType());
            if (!view || view.getAddressSpace() != "device")
                return;
            effect.owner = effect.storage;
            if (effect.kind == DeviceMemoryEffect::Kind::OrdinaryStore) {
                // Ordinary-write injectivity:
                // specs/compiler/invocation_index_ownership.md
                effect.ownershipProof = proveInvocationOwnedIndex(effect.indices);
                if (!effect.ownershipProof)
                    effect.ownershipProof = proveLeaderGuardedWorkgroupOwnedIndex(operation, effect.indices);
                if (!effect.ownershipProof && unitWorkgroup && !usesScalarGlobalInvocationId(effect.indices)) {
                    // Serialized launch |Inv|=1: specs/compiler/invocation_index_ownership.md §6
                    effect.singleInvocationConstrained = true;
                }
                if (!effect.ownershipProof && !effect.singleInvocationConstrained) {
                    operation->emitError(
                        "ordinary device TensorView store is not proven lane-exclusive; multi-invocation writes "
                        "must use accumulation or an atomic operation");
                    invalid = true;
                }
                if (effect.ownershipProof)
                    unitGridAxes.insert(effect.ownershipProof->unitGridAxes.begin(),
                                        effect.ownershipProof->unitGridAxes.end());
                else if (effect.singleInvocationConstrained)
                    for (unsigned axis = 0; axis < 3; ++axis)
                        unitGridAxes.insert(axis);
            } else if (effect.kind == DeviceMemoryEffect::Kind::ScatterAdd) {
                auto scatter = cast<ScatterAddOp>(operation);
                if (scatter.getDisjoint() && !proveStrictInvocationOwnedIndex(effect.indices)) {
                    operation->emitError(
                        "scatter_add 'disjoint' hint requires a lane-exclusive global invocation index proof");
                    invalid = true;
                }
            }
            effects.push_back(effect);
        });
        if (invalid)
            return failure();
        SmallVector<int32_t> axes;
        for (unsigned axis = 0; axis < 3; ++axis)
            if (unitGridAxes.contains(axis))
                axes.push_back(static_cast<int32_t>(axis));
        NamedAttrList contract;
        contract.append("unit_grid_axes", DenseI32ArrayAttr::get(function.getContext(), axes));
        contract.append(
            "requires_unit_workgroup",
            BoolAttr::get(function.getContext(), llvm::any_of(effects, [](const DeviceMemoryEffect &effect) {
                              return effect.singleInvocationConstrained;
                          })));
        function->setAttr(kDispatchContractAttrName, DictionaryAttr::get(function.getContext(), contract));

        for (const DeviceMemoryEffect &writer : effects) {
            if (!writer.writes)
                continue;
            for (const DeviceMemoryEffect &reader : effects) {
                if (!reader.reads || reader.owner != writer.owner)
                    continue;
                for (BarrierOp barrier : barriers) {
                    Operation *barrierOperation = barrier.getOperation();
                    std::optional<bool> writeBeforeBarrier = structuredBefore(writer.operation, barrierOperation);
                    std::optional<bool> barrierBeforeRead = structuredBefore(barrierOperation, reader.operation);
                    if ((writeBeforeBarrier && !*writeBeforeBarrier) || (barrierBeforeRead && !*barrierBeforeRead))
                        continue;
                    if (!writeBeforeBarrier || !barrierBeforeRead)
                        return reader.operation->emitError(
                            "device TensorView communication around a workgroup barrier has unsupported "
                            "structured placement; split this communication into a multi-kernel graph");
                    std::optional<GlobalIdAffineIndexTuple> readTuple;
                    GlobalIdAffineIndexTuple normalizedRead;
                    bool readable = true;
                    for (auto [dimension, index] : llvm::enumerate(reader.indices)) {
                        std::optional<GlobalIdAffineIndex> normalized = normalizeGlobalIdAffineIndex(index);
                        if (!normalized) {
                            readable = false;
                            break;
                        }
                        normalized->dimension = dimension;
                        normalizedRead.push_back(*normalized);
                    }
                    if (readable)
                        readTuple = std::move(normalizedRead);
                    if (writer.singleInvocationConstrained ||
                        (writer.ownershipProof && readTuple && writer.ownershipProof->normalizedIndices == *readTuple))
                        continue;
                    return reader.operation->emitError(
                        "device TensorView write followed by a workgroup barrier and cross-lane or unproven read "
                        "is not a global completion boundary; split this communication into a multi-kernel graph");
                }
            }
        }
        return success();
    }

private:
    bool invalid{};
};

struct VernonValidatePass : public PassWrapper<VernonValidatePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonValidatePass)

    StringRef getArgument() const final { return "vernon-validate"; }
    StringRef getDescription() const final { return "Validate Vernon shader stages and interface attributes"; }

    void runOnOperation() override {
        bool invalid = false;
        ModuleOp module = getOperation();
        std::map<ShaderStage, SmallVector<LocationEndpoint>> stageInputs;
        std::map<ShaderStage, SmallVector<LocationEndpoint>> stageOutputs;
        std::set<ShaderStage> entryStages;

        for (StructDeclOp declaration : module.getOps<StructDeclOp>()) {
            auto structure = StructType::get(module.getContext(), declaration.getSymName());
            FailureOr<ValueAbiLayout> layout = getValueStorageLayout(structure, module);
            if (failed(layout)) {
                declaration.emitError("does not define a finite canonical Value ABI layout");
                invalid = true;
                continue;
            }
            if (declaration->hasAttr("abi_size") || declaration->hasAttr("abi_alignment") ||
                declaration->hasAttr("abi_field_offsets") || declaration->hasAttr("abi_element_stride")) {
                declaration.emitError("contains obsolete frontend-authored Value ABI layout metadata");
                invalid = true;
            }
        }

        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            Attribute entryAttr = function->getAttr(kEntryAttrName);
            Attribute stageAttr = function->getAttr(kStageAttrName);
            Attribute workgroupAttr = function->getAttr(kWorkgroupSizeAttrName);

            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                if (function.getArgAttr(index, kTensorDescriptorOwnerAttrName) ||
                    function.getArgAttr(index, kTensorDescriptorComponentAttrName) ||
                    function.getArgAttr(index, kTensorDescriptorDimensionAttrName)) {
                    function.emitError() << "argument #" << index
                                         << " contains internal TensorView descriptor metadata";
                    invalid = true;
                }
            }

            if (entryAttr && !isa<UnitAttr>(entryAttr)) {
                function.emitError() << "'" << kEntryAttrName << "' must be a unit attribute";
                invalid = true;
            }

            if (stageAttr && !entryAttr) {
                function.emitError() << "'" << kStageAttrName << "' requires '" << kEntryAttrName << "'";
                invalid = true;
            }
            if (entryAttr && !stageAttr) {
                function.emitError() << "'" << kEntryAttrName << "' requires '" << kStageAttrName << "'";
                invalid = true;
            }

            std::optional<ShaderStage> stage;
            if (stageAttr) {
                FailureOr<ShaderStage> parsed = parseShaderStage(stageAttr);
                if (failed(parsed)) {
                    function.emitError() << "'" << kStageAttrName
                                         << "' must be one of \"vertex\", \"fragment\", or \"compute\"";
                    invalid = true;
                } else {
                    stage = *parsed;
                }
            }

            if (entryAttr && stage)
                entryStages.insert(*stage);

            if (stage == ShaderStage::Compute) {
                auto workgroup = dyn_cast_if_present<DenseI32ArrayAttr>(workgroupAttr);
                if (!workgroup || workgroup.size() != 3 ||
                    llvm::any_of(workgroup.asArrayRef(), [](int32_t value) { return value <= 0; })) {
                    function.emitError() << "compute entry requires '" << kWorkgroupSizeAttrName
                                         << "' as three positive i32 values";
                    invalid = true;
                }
            } else if (workgroupAttr) {
                function.emitError() << "'" << kWorkgroupSizeAttrName << "' is only valid on a compute entry";
                invalid = true;
            }

            std::map<std::string, unsigned> inputSlots;
            std::map<std::string, unsigned> outputSlots;
            std::set<std::pair<int64_t, int64_t>> bindings;

            auto validateInterface = [&](DictionaryAttr dictionary, Type type, unsigned index, bool isResult) {
                Type abiType = type;
                if (auto view = dyn_cast<TensorViewType>(type))
                    abiType = view.getElementType();
                const bool abiBearing = !containsLogicalAutodiffHandle(abiType) &&
                                        (abiType.isIntOrFloat() ||
                                         isa<TensorType, RankedTensorType, VectorType, StructType, TupleType>(abiType));
                if (abiBearing && failed(verifyValueAbiType(abiType, getOperation()))) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " has no finite canonical Value ABI layout";
                    invalid = true;
                }
                if (dictionary) {
                    constexpr StringLiteral retiredSuffixes[] = {
                        "size",           "alignment",    "layout_hash", "field_offsets",
                        "element_stride", "leaf_offsets", "leaf_counts", "leaf_paths",
                    };
                    constexpr StringLiteral abiPrefixes[] = {"vernon.abi_", "vernon.element_abi_"};
                    if (llvm::any_of(abiPrefixes, [&](StringLiteral prefix) {
                            return llvm::any_of(retiredSuffixes, [&](StringLiteral suffix) {
                                return dictionary.get((prefix + suffix).str()) != nullptr;
                            });
                        })) {
                        function.emitError()
                            << (isResult ? "result #" : "argument #") << index
                            << " uses retired duplicated Value ABI metadata; only logical leaf dtypes may be supplied";
                        invalid = true;
                    }
                }

                if (isa<TensorViewType>(type) && dictionary &&
                    (dictionary.get("vernon.tensor_shape") || dictionary.get("vernon.tensor_strides") ||
                     dictionary.get("vernon.tensor_offset"))) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uses retired TensorView layout attributes; shape, strides, and offset "
                                            "are supplied by the dispatch descriptor";
                    invalid = true;
                }

                InterfaceAttrs attrs = parseInterfaceAttrs(dictionary);
                if (attrs.empty()) {
                    if (entryAttr) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index
                                             << " of a Vernon entry must have '" << kInterfaceAttrName << "'";
                        invalid = true;
                    }
                    return;
                }

                if (!entryAttr || !stage) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uses Vernon interface attributes outside a valid entry";
                    invalid = true;
                    return;
                }

                FailureOr<InterfaceKind> parsedKind = parseInterfaceKind(attrs.kind);
                if (failed(parsedKind)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                         << kInterfaceAttrName
                                         << "' must be one of \"input\", \"output\", \"uniform\", "
                                            "or \"resource\"";
                    invalid = true;
                    return;
                }
                InterfaceKind kind = *parsedKind;

                if (isResult && kind != InterfaceKind::Output) {
                    function.emitError() << "result #" << index << " must use vernon.interface = \"output\"";
                    invalid = true;
                }
                if (!isResult && kind == InterfaceKind::Output) {
                    function.emitError() << "argument #" << index << " cannot use vernon.interface = \"output\"";
                    invalid = true;
                }

                std::optional<int64_t> location;
                if (attrs.location) {
                    auto integer = dyn_cast<IntegerAttr>(attrs.location);
                    if (!integer || integer.getInt() < 0) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kLocationAttrName << "' must be a non-negative integer";
                        invalid = true;
                    } else {
                        location = integer.getInt();
                    }
                }

                StringAttr builtin;
                if (attrs.builtin) {
                    builtin = dyn_cast<StringAttr>(attrs.builtin);
                    if (!builtin || builtin.getValue().empty()) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kBuiltinAttrName << "' must be a non-empty string";
                        invalid = true;
                    }
                }

                if (attrs.location && attrs.builtin) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " cannot have both '"
                                         << kLocationAttrName << "' and '" << kBuiltinAttrName << "'";
                    invalid = true;
                }

                bool isIo = kind == InterfaceKind::Input || kind == InterfaceKind::Output;
                bool isDescriptor = kind == InterfaceKind::Uniform || kind == InterfaceKind::Resource;
                if (isIo && !attrs.location && !attrs.builtin) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " "
                                         << stringifyInterfaceKind(kind) << " requires exactly one of '"
                                         << kLocationAttrName << "' or '" << kBuiltinAttrName << "'";
                    invalid = true;
                }

                auto parseNonNegative = [&](Attribute attr, StringRef name) -> std::optional<int64_t> {
                    if (!attr)
                        return std::nullopt;
                    auto integer = dyn_cast<IntegerAttr>(attr);
                    if (!integer || integer.getInt() < 0) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '" << name
                                             << "' must be a non-negative integer";
                        invalid = true;
                        return std::nullopt;
                    }
                    return integer.getInt();
                };

                std::optional<int64_t> descriptorSet = parseNonNegative(attrs.descriptorSet, kDescriptorSetAttrName);
                std::optional<int64_t> binding = parseNonNegative(attrs.binding, kBindingAttrName);

                if (kind == InterfaceKind::Resource && (!attrs.descriptorSet || !attrs.binding)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " "
                                         << stringifyInterfaceKind(kind) << " requires both '" << kDescriptorSetAttrName
                                         << "' and '" << kBindingAttrName << "'";
                    invalid = true;
                }
                if (kind == InterfaceKind::Uniform &&
                    static_cast<bool>(attrs.descriptorSet) != static_cast<bool>(attrs.binding)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uniform must provide both '" << kDescriptorSetAttrName << "' and '"
                                         << kBindingAttrName << "', or neither";
                    invalid = true;
                }
                if (isIo && (attrs.descriptorSet || attrs.binding)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " input/output cannot carry descriptor set or binding";
                    invalid = true;
                }
                if (isDescriptor && (attrs.location || attrs.builtin)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uniform/resource cannot carry location or builtin";
                    invalid = true;
                }

                if (descriptorSet && binding && !bindings.insert({*descriptorSet, *binding}).second) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " conflicts with another interface at descriptor set " << *descriptorSet
                                         << ", binding " << *binding;
                    invalid = true;
                }

                if (attrs.instanceDivisor) {
                    auto divisor = dyn_cast<IntegerAttr>(attrs.instanceDivisor);
                    if (!divisor || divisor.getInt() <= 0) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kInstanceDivisorAttrName << "' must be a positive integer";
                        invalid = true;
                    }
                    if (isResult || *stage != ShaderStage::Vertex || kind != InterfaceKind::Input) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kInstanceDivisorAttrName << "' is only valid on vertex inputs";
                        invalid = true;
                    }
                }

                if (!isIo || (!location && !builtin))
                    return;

                auto &slots = kind == InterfaceKind::Input ? inputSlots : outputSlots;
                int64_t locationSpan = 1;
                if (location && !isResult && *stage == ShaderStage::Vertex && kind == InterfaceKind::Input) {
                    SmallVector<StringRef> logicalDtypes;
                    if (auto dtypes = dictionary.getAs<ArrayAttr>("vernon.abi_leaf_dtypes"))
                        for (Attribute dtype : dtypes) {
                            auto value = dyn_cast<StringAttr>(dtype);
                            logicalDtypes.push_back(value ? value.getValue() : StringRef());
                        }
                    FailureOr<AttributeAbiLayout> plan = getAttributeAbiLayout(type, getOperation(), logicalDtypes);
                    if (failed(plan)) {
                        function.emitError() << "argument #" << index << " has no numeric vertex attribute layout";
                        invalid = true;
                        return;
                    }
                    locationSpan = static_cast<int64_t>(plan->getLocationSpan());
                }
                for (int64_t offset = 0; offset < locationSpan; ++offset) {
                    std::string slot = location ? ("location:" + std::to_string(*location + offset))
                                                : ("builtin:" + builtin.getValue().str());
                    auto [iterator, inserted] = slots.emplace(slot, index);
                    if (!inserted) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " conflicts with "
                                             << (kind == InterfaceKind::Input ? "input argument #" : "output result #")
                                             << iterator->second << " at " << slot;
                        invalid = true;
                    }
                }

                if (!location)
                    return;
                auto &endpoints = kind == InterfaceKind::Input ? stageInputs[*stage] : stageOutputs[*stage];
                endpoints.push_back({function, index, type, *location});
            };

            for (unsigned index = 0; index < function.getNumArguments(); ++index)
                validateInterface(function.getArgAttrDict(index), function.getArgumentTypes()[index], index,
                                  /*isResult=*/false);
            for (unsigned index = 0; index < function.getNumResults(); ++index)
                validateInterface(function.getResultAttrDict(index), function.getResultTypes()[index], index,
                                  /*isResult=*/true);
        }

        for (func::FuncOp function : module.getOps<func::FuncOp>()) {
            if (!function->hasAttr(kEntryAttrName))
                continue;
            auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
            if (!stage || stage.getValue() != "compute")
                continue;
            uint64_t totalPhysical = 0;
            function.walk([&](WorkgroupAllocOp allocation) {
                TensorViewType view = allocation.getResult().getType();
                FailureOr<WorkgroupPhysicalStoragePlan> plan = getWorkgroupPhysicalStoragePlan(view, module);
                if (failed(plan)) {
                    allocation.emitError("workgroup allocation has an invalid physical storage plan");
                    invalid = true;
                    return;
                }
                if (totalPhysical > kPortableWorkgroupStorageLimit ||
                    plan->totalPhysicalBytes > kPortableWorkgroupStorageLimit - totalPhysical) {
                    allocation.emitError(
                        "combined workgroup storage exceeds the portable 16 KiB workgroup storage limit");
                    invalid = true;
                    return;
                }
                totalPhysical += plan->totalPhysicalBytes;
            });
            KernelMemoryPhaseAnalysis memoryPhases;
            if (failed(memoryPhases.verify(function)))
                invalid = true;
        }

        getOperation().walk([&](Operation *operation) {
            if (operation->hasAttr("physical_index")) {
                operation->emitError("retired 'physical_index' attribute is not part of the Vernon IR contract");
                invalid = true;
            }
            if (isa<PhysicalLoadOp, PhysicalStoreOp, PhysicalAtomicOp>(operation)) {
                operation->emitError("physical TensorView operations are reserved for internal lowering");
                invalid = true;
            }
            bool workgroupStorage = isa<WorkgroupAllocOp>(operation);
            if (isa<LoadOp, StoreOp>(operation)) {
                auto view = dyn_cast<TensorViewType>(operation->getOperand(isa<StoreOp>(operation) ? 1 : 0).getType());
                workgroupStorage = view && view.getAddressSpace() == "workgroup";
            }
            if (!workgroupStorage && !isa<AtomicOp, BarrierOp, ReduceSumOp, ScatterAddOp>(operation))
                return;
            func::FuncOp function = operation->getParentOfType<func::FuncOp>();
            auto stage = function ? function->getAttrOfType<StringAttr>(kStageAttrName) : nullptr;
            if (!stage || stage.getValue() != "compute") {
                operation->emitError("workgroup storage, atomics, barriers, and accumulation require a compute entry");
                invalid = true;
            }
            if (auto barrier = dyn_cast<BarrierOp>(operation); barrier && failed(verifyUniformBarrier(barrier)))
                invalid = true;
            if (isa<WorkgroupAllocOp>(operation)) {
                bool entryAllocation = function && operation->getBlock() == &function.front();
                if (auto capture = operation->getParentOfType<AdCaptureOp>())
                    entryAllocation =
                        capture->getBlock() == &function.front() && operation->getBlock() == &capture.getBody().front();
                if (!entryAllocation) {
                    operation->emitError("workgroup storage must be allocated in the compute entry block");
                    invalid = true;
                }
            }
        });

        const auto validateStageInterface = [&](ShaderStage producer, ShaderStage consumer) {
            for (LocationEndpoint &input : stageInputs[consumer]) {
                auto matchingLocation = llvm::find_if(stageOutputs[producer], [&](const LocationEndpoint &output) {
                    return output.location == input.location;
                });
                if (matchingLocation == stageOutputs[producer].end()) {
                    input.function.emitError()
                        << stringifyShaderStage(consumer) << " input #" << input.index << " at location "
                        << input.location << " has no " << stringifyShaderStage(producer) << " output";
                    invalid = true;
                } else if (matchingLocation->type != input.type) {
                    input.function.emitError()
                        << stringifyShaderStage(consumer) << " input #" << input.index << " at location "
                        << input.location << " has type " << input.type << ", but the "
                        << stringifyShaderStage(producer) << " output has type " << matchingLocation->type;
                    invalid = true;
                }
            }
        };
        if (entryStages.count(ShaderStage::Vertex) && entryStages.count(ShaderStage::Fragment))
            validateStageInterface(ShaderStage::Vertex, ShaderStage::Fragment);

        if (invalid)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonValidatePass() { return std::make_unique<VernonValidatePass>(); }

void registerVernonValidatePass() { PassRegistration<VernonValidatePass>(); }

} // namespace mlir::vernon
