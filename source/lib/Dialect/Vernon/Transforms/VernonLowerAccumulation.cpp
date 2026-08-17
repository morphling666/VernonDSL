#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <limits>
#include <string>

namespace mlir::vernon {
namespace {

enum class LoweringForm {
    Store,
    WorkgroupReduce,
    AtomicAdd,
    LoadAddStore,
};

struct LoweringDecision {
    LoweringForm form;
    AtomicAddImplementation atomicImplementation{AtomicAddImplementation::Unsupported};
};

StringRef implementationName(AtomicAddImplementation implementation) {
    switch (implementation) {
    case AtomicAddImplementation::Native:
        return kNativeAtomicImplementation;
    case AtomicAddImplementation::IntegerCompareExchange:
        return kIntegerCasAtomicImplementation;
    case AtomicAddImplementation::Unsupported:
        return "unsupported";
    }
    llvm_unreachable("unknown atomic implementation");
}

FailureOr<AtomicAddImplementation> parseImplementation(StringRef value) {
    if (value == "unsupported")
        return AtomicAddImplementation::Unsupported;
    if (value == kNativeAtomicImplementation)
        return AtomicAddImplementation::Native;
    if (value == kIntegerCasAtomicImplementation)
        return AtomicAddImplementation::IntegerCompareExchange;
    return failure();
}

AtomicAddImplementation implementationFor(Type type, StringRef addressSpace,
                                          const AccumulationTargetCapabilities &capabilities) {
    if (isa<ShapedType>(type))
        return AtomicAddImplementation::Unsupported;
    const AtomicScopeCapabilities &scope = addressSpace == "workgroup" ? capabilities.workgroup : capabilities.device;
    if (type.isF32())
        return scope.f32;
    if (type.isF64())
        return scope.f64;
    return AtomicAddImplementation::Native;
}

bool isWorkgroupUniformIndex(Value value) {
    if (value.getDefiningOp<arith::ConstantOp>())
        return true;
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
        return isWorkgroupUniformIndex(cast.getIn());
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
        return isWorkgroupUniformIndex(cast.getIn());
    return false;
}

FailureOr<uint64_t> estimatedUniformContention(ReduceSumOp operation) {
    auto function = operation->getParentOfType<func::FuncOp>();
    if (!function || operation->getBlock() != &function.front() ||
        !llvm::all_of(operation.getIndices(), isWorkgroupUniformIndex))
        return failure();
    auto workgroup = function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName);
    if (!workgroup || workgroup.size() != 3)
        return failure();
    uint64_t lanes = 1;
    for (int32_t size : workgroup.asArrayRef()) {
        if (size <= 0 || lanes > std::numeric_limits<uint64_t>::max() / static_cast<uint32_t>(size))
            return failure();
        lanes *= static_cast<uint32_t>(size);
    }
    return lanes;
}

uint32_t reductionCrossover(AtomicAddImplementation implementation, const AccumulationCostModel &costModel) {
    switch (implementation) {
    case AtomicAddImplementation::Native:
        return std::max<uint32_t>(costModel.nativeAtomicReductionCrossover, 2);
    case AtomicAddImplementation::IntegerCompareExchange:
        return std::max<uint32_t>(costModel.integerCasReductionCrossover, 2);
    case AtomicAddImplementation::Unsupported:
        return std::numeric_limits<uint32_t>::max();
    }
    llvm_unreachable("unknown atomic implementation");
}

FailureOr<LoweringDecision> selectLowering(Operation *operation, const AccumulationTargetCapabilities &capabilities) {
    Value value;
    Value storage;
    bool deterministic = false;
    bool disjoint = false;
    if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
        value = reduce.getValue();
        storage = reduce.getStorage();
        deterministic = reduce.getDeterministic();
    } else {
        auto scatter = cast<ScatterAddOp>(operation);
        value = scatter.getValue();
        storage = scatter.getStorage();
        deterministic = scatter.getDeterministic();
        disjoint = static_cast<bool>(scatter.getDisjointAttr());
    }
    auto ownership = operation->getAttrOfType<StringAttr>(kAccumulationOwnershipAttrName);
    if (ownership && ownership.getValue() != kInvocationPrivateAccumulationOwnership)
        return operation->emitError("has unsupported accumulation ownership '") << ownership.getValue() << "'";
    if (ownership && !isa<ScatterAddOp>(operation))
        return operation->emitError("invocation-private accumulation ownership is only valid on scatter_add");
    if (ownership && capabilities.aggregateGradientStorage != AggregateGradientStorage::InvocationPrivateStaging)
        return operation->emitError(
            "invocation-private accumulation requires target support for invocation-private gradient staging");
    if (disjoint && !proveStrictInvocationOwnedIndex(cast<ScatterAddOp>(operation).getIndices()))
        return operation->emitError(
            "scatter_add 'disjoint' hint requires a lane-exclusive global invocation index proof");

    auto view = dyn_cast<TensorViewType>(storage.getType());
    if (!view)
        return operation->emitError("accumulation storage must be a TensorView");
    AtomicAddImplementation atomicImplementation =
        implementationFor(value.getType(), view.getAddressSpace(), capabilities);
    if (disjoint)
        return LoweringDecision{LoweringForm::Store};
    if (ownership)
        return LoweringDecision{LoweringForm::LoadAddStore};
    if (!deterministic && capabilities.supportsWorkgroupReduction)
        if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
            FailureOr<uint64_t> contention = estimatedUniformContention(reduce);
            if (succeeded(contention) && atomicImplementation != AtomicAddImplementation::Unsupported &&
                *contention >= reductionCrossover(atomicImplementation, capabilities.costModel))
                return LoweringDecision{LoweringForm::WorkgroupReduce, atomicImplementation};
        }
    if (!deterministic && atomicImplementation != AtomicAddImplementation::Unsupported)
        return LoweringDecision{LoweringForm::AtomicAdd, atomicImplementation};
    if (deterministic)
        return operation->emitError("deterministic shared accumulation requires a dedicated reduction kernel");
    if (isa<ShapedType>(value.getType()))
        return operation->emitError("shared shaped accumulation requires proven invocation-private gradient staging");
    return operation->emitError("shared scalar accumulation requires a supported atomic add");
}

Operation *createLoad(OpBuilder &builder, Location location, Value storage, ValueRange indices, Type type) {
    OperationState state(location, LoadOp::getOperationName());
    state.addOperands(storage);
    state.addOperands(indices);
    state.addTypes(type);
    return builder.create(state);
}

Operation *createStore(OpBuilder &builder, Location location, Value value, Value storage, ValueRange indices,
                       StringRef strategy, StringAttr ownership = {}) {
    OperationState state(location, StoreOp::getOperationName());
    state.addOperands(value);
    state.addOperands(storage);
    state.addOperands(indices);
    state.addAttribute(kAccumulationStrategyAttrName, builder.getStringAttr(strategy));
    if (ownership)
        state.addAttribute(kAccumulationOwnershipAttrName, ownership);
    return builder.create(state);
}

Operation *createAtomicAdd(OpBuilder &builder, Location location, Value value, Value storage, ValueRange indices,
                           AtomicAddImplementation implementation) {
    OperationState state(location, AtomicOp::getOperationName());
    state.addOperands(storage);
    state.addOperands(indices);
    state.addOperands(value);
    state.addTypes(value.getType());
    state.addAttribute("atomic_kind", builder.getStringAttr("add"));
    state.addAttribute("ordering", builder.getStringAttr("relaxed"));
    state.addAttribute(kAccumulationStrategyAttrName, builder.getStringAttr(kAtomicAccumulationStrategy));
    state.addAttribute(kAtomicImplementationAttrName, builder.getStringAttr(implementationName(implementation)));
    return builder.create(state);
}

struct VernonLowerAccumulationPass final : PassWrapper<VernonLowerAccumulationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerAccumulationPass)

    VernonLowerAccumulationPass() = default;
    explicit VernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities) {
        f32DeviceAtomic = implementationName(capabilities.device.f32).str();
        f64DeviceAtomic = implementationName(capabilities.device.f64).str();
        f32WorkgroupAtomic = implementationName(capabilities.workgroup.f32).str();
        f64WorkgroupAtomic = implementationName(capabilities.workgroup.f64).str();
        aggregateGradientStorage =
            capabilities.aggregateGradientStorage == AggregateGradientStorage::InvocationPrivateStaging
                ? "invocation-private-staging"
                : "shared";
        supportsWorkgroupReduction = capabilities.supportsWorkgroupReduction;
        nativeAtomicReductionCrossover = capabilities.costModel.nativeAtomicReductionCrossover;
        integerCasReductionCrossover = capabilities.costModel.integerCasReductionCrossover;
    }
    VernonLowerAccumulationPass(const VernonLowerAccumulationPass &other) : PassWrapper(other) {
        f32DeviceAtomic = other.f32DeviceAtomic;
        f64DeviceAtomic = other.f64DeviceAtomic;
        f32WorkgroupAtomic = other.f32WorkgroupAtomic;
        f64WorkgroupAtomic = other.f64WorkgroupAtomic;
        aggregateGradientStorage = other.aggregateGradientStorage;
        supportsWorkgroupReduction = other.supportsWorkgroupReduction;
        nativeAtomicReductionCrossover = other.nativeAtomicReductionCrossover;
        integerCasReductionCrossover = other.integerCasReductionCrossover;
    }

    StringRef getArgument() const final { return "vernon-lower-accumulation"; }
    StringRef getDescription() const final {
        return "Select and lower grid-independent autodiff accumulation semantics";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, VernonDialect>();
    }

    void runOnOperation() override {
        AccumulationTargetCapabilities capabilities;
        if (aggregateGradientStorage == "shared") {
            capabilities.aggregateGradientStorage = AggregateGradientStorage::Shared;
        } else if (aggregateGradientStorage == "invocation-private-staging") {
            capabilities.aggregateGradientStorage = AggregateGradientStorage::InvocationPrivateStaging;
        } else {
            getOperation()->emitError("unknown aggregate gradient storage model '") << aggregateGradientStorage << "'";
            return signalPassFailure();
        }
        auto parse = [&](StringRef value, AtomicAddImplementation &destination, StringRef scope,
                         StringRef type) -> LogicalResult {
            FailureOr<AtomicAddImplementation> implementation = parseImplementation(value);
            if (failed(implementation))
                return getOperation()->emitError("unknown ")
                       << scope << " " << type << " atomic implementation '" << value << "'";
            destination = *implementation;
            return success();
        };
        if (failed(parse(f32DeviceAtomic, capabilities.device.f32, "device", "f32")) ||
            failed(parse(f64DeviceAtomic, capabilities.device.f64, "device", "f64")) ||
            failed(parse(f32WorkgroupAtomic, capabilities.workgroup.f32, "workgroup", "f32")) ||
            failed(parse(f64WorkgroupAtomic, capabilities.workgroup.f64, "workgroup", "f64")))
            return signalPassFailure();
        capabilities.supportsWorkgroupReduction = supportsWorkgroupReduction;
        capabilities.costModel.nativeAtomicReductionCrossover = nativeAtomicReductionCrossover;
        capabilities.costModel.integerCasReductionCrossover = integerCasReductionCrossover;
        SmallVector<std::pair<Operation *, LoweringDecision>> decisions;
        WalkResult analysis = getOperation().walk([&](Operation *operation) {
            if (!isa<ReduceSumOp, ScatterAddOp>(operation))
                return WalkResult::advance();
            FailureOr<LoweringDecision> decision = selectLowering(operation, capabilities);
            if (failed(decision))
                return WalkResult::interrupt();
            decisions.emplace_back(operation, *decision);
            return WalkResult::advance();
        });
        if (analysis.wasInterrupted())
            return signalPassFailure();

        for (auto [operation, decision] : decisions) {
            Value value;
            Value storage;
            ValueRange indices;
            if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
                value = reduce.getValue();
                storage = reduce.getStorage();
                indices = reduce.getIndices();
            } else {
                auto scatter = cast<ScatterAddOp>(operation);
                value = scatter.getValue();
                storage = scatter.getStorage();
                indices = scatter.getIndices();
            }

            OpBuilder builder(operation);
            auto ownership = operation->getAttrOfType<StringAttr>(kAccumulationOwnershipAttrName);
            if (decision.form == LoweringForm::WorkgroupReduce) {
                operation->setAttr(kAccumulationStrategyAttrName,
                                   builder.getStringAttr(kWorkgroupReductionAccumulationStrategy));
                operation->setAttr(kAtomicImplementationAttrName,
                                   builder.getStringAttr(implementationName(decision.atomicImplementation)));
                continue;
            } else if (decision.form == LoweringForm::Store) {
                createStore(builder, operation->getLoc(), value, storage, indices, kExclusiveStoreAccumulationStrategy,
                            ownership);
            } else if (decision.form == LoweringForm::AtomicAdd) {
                createAtomicAdd(builder, operation->getLoc(), value, storage, indices, decision.atomicImplementation);
            } else {
                Operation *load = createLoad(builder, operation->getLoc(), storage, indices, value.getType());
                Value sum = arith::AddFOp::create(builder, operation->getLoc(), load->getResult(0), value);
                createStore(builder, operation->getLoc(), sum, storage, indices, kInvocationPrivateAccumulationStrategy,
                            ownership);
            }
            operation->erase();
        }

        WalkResult atomicPlanning = getOperation().walk([&](Operation *operation) {
            Value storage;
            Value value;
            StringRef kind;
            if (auto atomic = dyn_cast<AtomicOp>(operation)) {
                storage = atomic.getStorage();
                value = atomic.getValue();
                kind = atomic.getAtomicKind();
            } else if (auto atomic = dyn_cast<PhysicalAtomicOp>(operation)) {
                storage = atomic.getStorage();
                value = atomic.getValue();
                kind = atomic.getAtomicKind();
            } else {
                return WalkResult::advance();
            }
            if (kind != "add" || !isa<FloatType>(value.getType()))
                return WalkResult::advance();
            auto view = dyn_cast<TensorViewType>(storage.getType());
            if (!view) {
                operation->emitError("floating atomic storage must be a TensorView");
                return WalkResult::interrupt();
            }
            const AtomicAddImplementation implementation =
                implementationFor(value.getType(), view.getAddressSpace(), capabilities);
            if (implementation == AtomicAddImplementation::Unsupported) {
                operation->emitError("floating atomic add is unsupported for ") << view.getAddressSpace() << " scope";
                return WalkResult::interrupt();
            }
            auto planned = operation->getAttrOfType<StringAttr>(kAtomicImplementationAttrName);
            if (planned && planned.getValue() != implementationName(implementation)) {
                operation->emitError("floating atomic legalization '")
                    << planned.getValue() << "' conflicts with target implementation '"
                    << implementationName(implementation) << "'";
                return WalkResult::interrupt();
            }
            operation->setAttr(kAtomicImplementationAttrName,
                               StringAttr::get(operation->getContext(), implementationName(implementation)));
            return WalkResult::advance();
        });
        if (atomicPlanning.wasInterrupted())
            return signalPassFailure();

        if (failed(materializeTensorViewProjections(getOperation())))
            signalPassFailure();
    }

    Option<std::string> f32DeviceAtomic{
        *this, "f32-device-atomic",
        llvm::cl::desc("Device-scope f32 atomic implementation: native, integer_cas, or unsupported"),
        llvm::cl::init("unsupported")};
    Option<std::string> f64DeviceAtomic{
        *this, "f64-device-atomic",
        llvm::cl::desc("Device-scope f64 atomic implementation: native, integer_cas, or unsupported"),
        llvm::cl::init("unsupported")};
    Option<std::string> f32WorkgroupAtomic{
        *this, "f32-workgroup-atomic",
        llvm::cl::desc("Workgroup-scope f32 atomic implementation: native, integer_cas, or unsupported"),
        llvm::cl::init("unsupported")};
    Option<std::string> f64WorkgroupAtomic{
        *this, "f64-workgroup-atomic",
        llvm::cl::desc("Workgroup-scope f64 atomic implementation: native, integer_cas, or unsupported"),
        llvm::cl::init("unsupported")};
    Option<bool> supportsWorkgroupReduction{
        *this, "supports-workgroup-reduction",
        llvm::cl::desc("Target supports convergent workgroup reduction for shared scalar accumulation"),
        llvm::cl::init(false)};
    Option<std::string> aggregateGradientStorage{
        *this, "aggregate-gradient-storage",
        llvm::cl::desc("Aggregate gradient storage model: shared or invocation-private-staging"),
        llvm::cl::init("shared")};
    Option<uint32_t> nativeAtomicReductionCrossover{
        *this, "native-atomic-reduction-crossover",
        llvm::cl::desc("Measured contenders per destination where workgroup reduction beats native atomic add"),
        llvm::cl::init(64)};
    Option<uint32_t> integerCasReductionCrossover{
        *this, "integer-cas-reduction-crossover",
        llvm::cl::desc("Measured contenders per destination where workgroup reduction beats integer CAS"),
        llvm::cl::init(64)};
};

struct VernonVerifyGeneratedAccumulationPass final
    : PassWrapper<VernonVerifyGeneratedAccumulationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonVerifyGeneratedAccumulationPass)

    VernonVerifyGeneratedAccumulationPass() = default;
    explicit VernonVerifyGeneratedAccumulationPass(AccumulationTargetCapabilities capabilities)
        : capabilities(capabilities) {}
    VernonVerifyGeneratedAccumulationPass(const VernonVerifyGeneratedAccumulationPass &other)
        : PassWrapper(other), capabilities(other.capabilities) {}

    StringRef getArgument() const final { return "vernon-verify-generated-accumulation"; }
    StringRef getDescription() const final {
        return "Verify generated atomic requirements against the selected target profile";
    }

    void runOnOperation() override {
        WalkResult result = getOperation().walk([&](Operation *operation) {
            Value storage;
            Value value;
            StringRef kind;
            if (auto atomic = dyn_cast<AtomicOp>(operation)) {
                storage = atomic.getStorage();
                value = atomic.getValue();
                kind = atomic.getAtomicKind();
            } else if (auto atomic = dyn_cast<PhysicalAtomicOp>(operation)) {
                storage = atomic.getStorage();
                value = atomic.getValue();
                kind = atomic.getAtomicKind();
            } else {
                return WalkResult::advance();
            }
            if (kind != "add" || !isa<FloatType>(value.getType()))
                return WalkResult::advance();
            auto view = dyn_cast<TensorViewType>(storage.getType());
            if (!view) {
                operation->emitError("generated floating atomic has no TensorView scope");
                return WalkResult::interrupt();
            }
            AtomicAddImplementation supported =
                implementationFor(value.getType(), view.getAddressSpace(), capabilities);
            auto selected = operation->getAttrOfType<StringAttr>(kAtomicImplementationAttrName);
            if (!selected) {
                operation->emitError("generated floating atomic has no selected legalization");
                return WalkResult::interrupt();
            }
            FailureOr<AtomicAddImplementation> implementation = parseImplementation(selected.getValue());
            if (failed(implementation) || *implementation == AtomicAddImplementation::Unsupported) {
                operation->emitError("generated floating atomic has an invalid legalization '")
                    << selected.getValue() << "'";
                return WalkResult::interrupt();
            }
            if (*implementation != supported) {
                operation->emitError("generated floating atomic requires '")
                    << selected.getValue() << "' but the target profile provides '" << implementationName(supported)
                    << "' for " << view.getAddressSpace() << " scope";
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (result.wasInterrupted())
            signalPassFailure();
    }

    AccumulationTargetCapabilities capabilities;
};

} // namespace

std::unique_ptr<Pass> createVernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities) {
    return std::make_unique<VernonLowerAccumulationPass>(capabilities);
}

std::unique_ptr<Pass> createVernonVerifyGeneratedAccumulationPass(AccumulationTargetCapabilities capabilities) {
    return std::make_unique<VernonVerifyGeneratedAccumulationPass>(capabilities);
}

void registerVernonLowerAccumulationPass() {
    PassRegistration<VernonLowerAccumulationPass>();
    PassRegistration<VernonVerifyGeneratedAccumulationPass>();
}

} // namespace mlir::vernon
