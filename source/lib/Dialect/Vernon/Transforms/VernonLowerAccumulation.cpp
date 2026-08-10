#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir::vernon {
namespace {

enum class LoweringForm {
    Store,
    AtomicAdd,
    LoadAddStore,
};

struct LoweringDecision {
    LoweringForm form;
};

FailureOr<LoweringDecision> selectLowering(Operation *operation, bool supportsF32AtomicAdd, bool supportsF64AtomicAdd,
                                           AggregateGradientStorage aggregateGradientStorage) {
    Value value;
    bool deterministic = false;
    bool disjoint = false;
    if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
        value = reduce.getValue();
        deterministic = reduce.getDeterministic();
    } else {
        auto scatter = cast<ScatterAddOp>(operation);
        value = scatter.getValue();
        deterministic = scatter.getDeterministic();
        disjoint = static_cast<bool>(scatter.getDisjointAttr());
    }
    auto ownership = operation->getAttrOfType<StringAttr>(kAccumulationOwnershipAttrName);
    if (ownership && ownership.getValue() != kInvocationPrivateAccumulationOwnership)
        return operation->emitError("has unsupported accumulation ownership '") << ownership.getValue() << "'";
    if (ownership && !isa<ScatterAddOp>(operation))
        return operation->emitError("invocation-private accumulation ownership is only valid on scatter_add");
    if (ownership && aggregateGradientStorage != AggregateGradientStorage::InvocationPrivateStaging)
        return operation->emitError(
            "invocation-private accumulation requires target support for invocation-private gradient staging");
    if (disjoint && !proveStrictInvocationOwnedIndex(cast<ScatterAddOp>(operation).getIndices()))
        return operation->emitError(
            "scatter_add 'disjoint' hint requires a lane-exclusive global invocation index proof");

    const bool atomicSupported =
        (value.getType().isF32() && supportsF32AtomicAdd) || (value.getType().isF64() && supportsF64AtomicAdd);
    if (disjoint)
        return LoweringDecision{LoweringForm::Store};
    if (ownership)
        return LoweringDecision{LoweringForm::LoadAddStore};
    if (!deterministic && atomicSupported)
        return LoweringDecision{LoweringForm::AtomicAdd};
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
                       StringAttr ownership = {}) {
    OperationState state(location, StoreOp::getOperationName());
    state.addOperands(value);
    state.addOperands(storage);
    state.addOperands(indices);
    if (ownership)
        state.addAttribute(kAccumulationOwnershipAttrName, ownership);
    return builder.create(state);
}

Operation *createAtomicAdd(OpBuilder &builder, Location location, Value value, Value storage, ValueRange indices) {
    OperationState state(location, AtomicOp::getOperationName());
    state.addOperands(storage);
    state.addOperands(indices);
    state.addOperands(value);
    state.addTypes(value.getType());
    state.addAttribute("atomic_kind", builder.getStringAttr("add"));
    state.addAttribute("ordering", builder.getStringAttr("relaxed"));
    return builder.create(state);
}

struct VernonLowerAccumulationPass final : PassWrapper<VernonLowerAccumulationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerAccumulationPass)

    VernonLowerAccumulationPass() = default;
    explicit VernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities) {
        supportsF32AtomicAdd = capabilities.supportsF32AtomicAdd;
        supportsF64AtomicAdd = capabilities.supportsF64AtomicAdd;
        aggregateGradientStorage =
            capabilities.aggregateGradientStorage == AggregateGradientStorage::InvocationPrivateStaging
                ? "invocation-private-staging"
                : "shared";
    }
    VernonLowerAccumulationPass(const VernonLowerAccumulationPass &other) : PassWrapper(other) {
        supportsF32AtomicAdd = other.supportsF32AtomicAdd;
        supportsF64AtomicAdd = other.supportsF64AtomicAdd;
        aggregateGradientStorage = other.aggregateGradientStorage;
    }

    StringRef getArgument() const final { return "vernon-lower-accumulation"; }
    StringRef getDescription() const final {
        return "Select and lower grid-independent autodiff accumulation semantics";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, VernonDialect>();
    }

    void runOnOperation() override {
        AggregateGradientStorage storageModel;
        if (aggregateGradientStorage == "shared") {
            storageModel = AggregateGradientStorage::Shared;
        } else if (aggregateGradientStorage == "invocation-private-staging") {
            storageModel = AggregateGradientStorage::InvocationPrivateStaging;
        } else {
            getOperation()->emitError("unknown aggregate gradient storage model '") << aggregateGradientStorage << "'";
            return signalPassFailure();
        }
        SmallVector<std::pair<Operation *, LoweringDecision>> decisions;
        WalkResult analysis = getOperation().walk([&](Operation *operation) {
            if (!isa<ReduceSumOp, ScatterAddOp>(operation))
                return WalkResult::advance();
            FailureOr<LoweringDecision> decision =
                selectLowering(operation, supportsF32AtomicAdd, supportsF64AtomicAdd, storageModel);
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
            if (decision.form == LoweringForm::Store) {
                createStore(builder, operation->getLoc(), value, storage, indices, ownership);
            } else if (decision.form == LoweringForm::AtomicAdd) {
                createAtomicAdd(builder, operation->getLoc(), value, storage, indices);
            } else {
                Operation *load = createLoad(builder, operation->getLoc(), storage, indices, value.getType());
                Value sum = arith::AddFOp::create(builder, operation->getLoc(), load->getResult(0), value);
                createStore(builder, operation->getLoc(), sum, storage, indices, ownership);
            }
            operation->erase();
        }

        if (failed(materializeTensorViewProjections(getOperation())))
            signalPassFailure();
    }

    Option<bool> supportsF32AtomicAdd{*this, "supports-atomic-f32",
                                      llvm::cl::desc("Target supports device-scope f32 atomic add"),
                                      llvm::cl::init(false)};
    Option<bool> supportsF64AtomicAdd{*this, "supports-atomic-f64",
                                      llvm::cl::desc("Target supports device-scope f64 atomic add"),
                                      llvm::cl::init(false)};
    Option<std::string> aggregateGradientStorage{
        *this, "aggregate-gradient-storage",
        llvm::cl::desc("Aggregate gradient storage model: shared or invocation-private-staging"),
        llvm::cl::init("shared")};
};

} // namespace

std::unique_ptr<Pass> createVernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities) {
    return std::make_unique<VernonLowerAccumulationPass>(capabilities);
}

void registerVernonLowerAccumulationPass() { PassRegistration<VernonLowerAccumulationPass>(); }

} // namespace mlir::vernon
