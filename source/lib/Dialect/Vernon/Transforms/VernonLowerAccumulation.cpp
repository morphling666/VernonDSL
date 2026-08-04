#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::vernon {
namespace {

enum class SelectedStrategy {
    Direct,
    Atomic,
    LoadAddStore,
};

SelectedStrategy selectStrategy(Operation *operation, bool supportsF32AtomicAdd, bool supportsF64AtomicAdd) {
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
    const bool atomicSupported =
        (value.getType().isF32() && supportsF32AtomicAdd) || (value.getType().isF64() && supportsF64AtomicAdd);
    return disjoint ? SelectedStrategy::Direct
                    : (!deterministic && atomicSupported ? SelectedStrategy::Atomic : SelectedStrategy::LoadAddStore);
}

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

FailureOr<std::pair<BlockArgument, BlockArgument>> findSerialDispatchArguments(func::FuncOp function) {
    BlockArgument launch;
    BlockArgument globalId;
    for (BlockArgument argument : function.getArguments()) {
        const unsigned index = argument.getArgNumber();
        auto sourceName = function.getArgAttrOfType<StringAttr>(index, "vernon.source_name");
        auto builtin = function.getArgAttrOfType<StringAttr>(index, "vernon.builtin");
        if (sourceName && sourceName.getValue() == "__vernon_launch")
            launch = argument;
        if (builtin && builtin.getValue() == "global_invocation_id")
            globalId = argument;
    }
    if (!launch || !globalId || !isa<RankedTensorType>(globalId.getType()))
        return failure();
    return std::make_pair(launch, globalId);
}

LogicalResult makeSerialDispatch(func::FuncOp function) {
    FailureOr<std::pair<BlockArgument, BlockArgument>> arguments = findSerialDispatchArguments(function);
    if (failed(arguments))
        return function.emitError("deterministic accumulation requires launch and global-invocation arguments");

    Block &entry = function.front();
    Operation *terminator = entry.getTerminator();
    SmallVector<Operation *> originalOperations;
    for (Operation &operation : entry.without_terminator())
        originalOperations.push_back(&operation);
    OpBuilder builder(terminator);
    Location location = function.getLoc();
    Value zero = arith::ConstantIndexOp::create(builder, location, 0);
    Value one = arith::ConstantIndexOp::create(builder, location, 1);
    auto launchType = dyn_cast<TensorViewType>(arguments->first.getType());
    auto extentTensorType = launchType ? dyn_cast<RankedTensorType>(launchType.getElementType()) : nullptr;
    auto globalIdType = dyn_cast<RankedTensorType>(arguments->second.getType());
    if (!extentTensorType || extentTensorType.getShape() != ArrayRef<int64_t>{3} ||
        !extentTensorType.getElementType().isInteger(32) || globalIdType != extentTensorType)
        return function.emitError("deterministic accumulation has an invalid launch ABI");

    Operation *load = createLoad(builder, location, arguments->first, ValueRange{zero}, extentTensorType);
    SmallVector<Value> extents;
    for (int64_t axis = 0; axis != 3; ++axis) {
        Value index = arith::ConstantIndexOp::create(builder, location, axis);
        Value extent = tensor::ExtractOp::create(builder, location, load->getResult(0), ValueRange{index}).getResult();
        extents.push_back(arith::IndexCastUIOp::create(builder, location, builder.getIndexType(), extent));
    }

    scf::ForOp zLoop = scf::ForOp::create(builder, location, zero, extents[2], one);
    builder.setInsertionPointToStart(zLoop.getBody());
    scf::ForOp yLoop = scf::ForOp::create(builder, location, zero, extents[1], one);
    builder.setInsertionPointToStart(yLoop.getBody());
    scf::ForOp xLoop = scf::ForOp::create(builder, location, zero, extents[0], one);
    builder.setInsertionPointToStart(xLoop.getBody());

    SmallVector<Value> coordinates;
    for (Value index : {xLoop.getInductionVar(), yLoop.getInductionVar(), zLoop.getInductionVar()})
        coordinates.push_back(
            arith::IndexCastUIOp::create(builder, location, extentTensorType.getElementType(), index));
    Value globalId = tensor::FromElementsOp::create(builder, location, globalIdType, coordinates);

    IRMapping mapping;
    for (BlockArgument argument : function.getArguments())
        mapping.map(argument, argument);
    mapping.map(arguments->second, globalId);
    for (Operation *operation : originalOperations)
        builder.clone(*operation, mapping);
    for (Operation *operation : llvm::reverse(originalOperations))
        operation->erase();

    function->setAttr("vernon.workgroup_size", builder.getDenseI32ArrayAttr({1, 1, 1}));
    function->setAttr("vernon.serial_dispatch", builder.getUnitAttr());
    return success();
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
        registry
            .insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect, tensor::TensorDialect, VernonDialect>();
    }

    void runOnOperation() override {
        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            bool requiresSerialDispatch = false;
            function.walk([&](Operation *operation) {
                if (isa<ReduceSumOp, ScatterAddOp>(operation) &&
                    selectStrategy(operation, supportsF32AtomicAdd, supportsF64AtomicAdd) ==
                        SelectedStrategy::LoadAddStore)
                    requiresSerialDispatch = true;
            });
            if (requiresSerialDispatch && function->hasAttr("vernon.entry") && failed(makeSerialDispatch(function)))
                return signalPassFailure();
        }

        SmallVector<Operation *> operations;
        getOperation().walk([&](Operation *operation) {
            if (isa<ReduceSumOp, ScatterAddOp>(operation))
                operations.push_back(operation);
        });

        for (Operation *operation : operations) {
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

            const SelectedStrategy strategy = selectStrategy(operation, supportsF32AtomicAdd, supportsF64AtomicAdd);

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
