#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::vernon {
namespace {

enum class SelectedStrategy {
    Direct,
    Atomic,
    LoadAddStore,
};

Operation *createLoad(OpBuilder &builder, Location location, Value storage, ValueRange indices, Type type) {
    OperationState state(location, LoadOp::getOperationName());
    state.addOperands(storage);
    state.addOperands(indices);
    state.addTypes(type);
    return builder.create(state);
}

Operation *createStore(OpBuilder &builder, Location location, Value value, Value storage, ValueRange indices) {
    OperationState state(location, StoreOp::getOperationName());
    state.addOperands(value);
    state.addOperands(storage);
    state.addOperands(indices);
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
    }
    VernonLowerAccumulationPass(const VernonLowerAccumulationPass &other) : PassWrapper(other) {
        supportsF32AtomicAdd = other.supportsF32AtomicAdd;
        supportsF64AtomicAdd = other.supportsF64AtomicAdd;
    }

    StringRef getArgument() const final { return "vernon-lower-accumulation"; }
    StringRef getDescription() const final {
        return "Select and lower grid-independent autodiff accumulation semantics";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, VernonDialect>();
    }

    void runOnOperation() override {
        SmallVector<Operation *> operations;
        getOperation().walk([&](Operation *operation) {
            if (isa<ReduceSumOp, ScatterAddOp>(operation))
                operations.push_back(operation);
        });

        for (Operation *operation : operations) {
            Value value;
            Value storage;
            ValueRange indices;
            bool deterministic = false;
            bool disjoint = false;
            if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
                value = reduce.getValue();
                storage = reduce.getStorage();
                indices = reduce.getIndices();
                deterministic = reduce.getDeterministic();
            } else {
                auto scatter = cast<ScatterAddOp>(operation);
                value = scatter.getValue();
                storage = scatter.getStorage();
                indices = scatter.getIndices();
                deterministic = scatter.getDeterministic();
                disjoint = static_cast<bool>(scatter.getDisjointAttr());
            }

            const bool atomicSupported =
                (value.getType().isF32() && supportsF32AtomicAdd) || (value.getType().isF64() && supportsF64AtomicAdd);
            const SelectedStrategy strategy =
                disjoint
                    ? SelectedStrategy::Direct
                    : (!deterministic && atomicSupported ? SelectedStrategy::Atomic : SelectedStrategy::LoadAddStore);

            OpBuilder builder(operation);
            if (strategy == SelectedStrategy::Direct) {
                createStore(builder, operation->getLoc(), value, storage, indices);
            } else if (strategy == SelectedStrategy::Atomic) {
                createAtomicAdd(builder, operation->getLoc(), value, storage, indices);
            } else {
                Operation *load = createLoad(builder, operation->getLoc(), storage, indices, value.getType());
                Value sum = arith::AddFOp::create(builder, operation->getLoc(), load->getResult(0), value);
                createStore(builder, operation->getLoc(), sum, storage, indices);
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
};

} // namespace

std::unique_ptr<Pass> createVernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities) {
    return std::make_unique<VernonLowerAccumulationPass>(capabilities);
}

void registerVernonLowerAccumulationPass() { PassRegistration<VernonLowerAccumulationPass>(); }

} // namespace mlir::vernon
