#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAggregateStorage.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"

#include <vector>

namespace mlir::vernon {
namespace {

struct ArgumentPlan {
    Type sourceType;
    std::optional<CpuCallPlan> value;
    unsigned firstBridgeArgument{};
    unsigned bridgeArgumentCount{};
    bool metadataCarrier{};
};

FailureOr<SmallVector<StringRef>> logicalDtypes(DictionaryAttr attributes) {
    SmallVector<StringRef> result;
    auto dtypes = attributes.getAs<ArrayAttr>("vernon.abi_leaf_dtypes");
    if (!dtypes)
        return result;
    for (Attribute attribute : dtypes) {
        auto dtype = dyn_cast<StringAttr>(attribute);
        if (!dtype)
            return failure();
        result.push_back(dtype.getValue());
    }
    return result;
}

struct VernonLowerCPUABIPass final : PassWrapper<VernonLowerCPUABIPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerCPUABIPass)

    StringRef getArgument() const final { return "vernon-lower-cpu-abi"; }
    StringRef getDescription() const final {
        return "Materialize canonical CPU Value ABI boundaries before target type lowering";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect, VernonDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SmallVector<func::FuncOp> entries;
        for (func::FuncOp function : module.getOps<func::FuncOp>())
            if (function->hasAttr("vernon.entry"))
                entries.push_back(function);

        for (func::FuncOp function : entries) {
            if (failed(lowerEntry(function, module))) {
                signalPassFailure();
                return;
            }
        }
    }

    LogicalResult lowerEntry(func::FuncOp function, ModuleOp module) {
        if (function.isDeclaration()) {
            function.emitError("CPU entry must have a body before ABI lowering");
            return failure();
        }
        if (function.getNumResults() > 1) {
            function.emitError("CPU ABI lowering supports at most one source result");
            return failure();
        }

        std::vector<ArgumentPlan> argumentPlans;
        SmallVector<Type> bridgeArgumentTypes;
        argumentPlans.reserve(function.getNumArguments());
        for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
            ArgumentPlan argument{type, std::nullopt, static_cast<unsigned>(bridgeArgumentTypes.size()), 1, false};
            FailureOr<SmallVector<StringRef>> dtypes = logicalDtypes(function.getArgAttrDict(index));
            if (failed(dtypes)) {
                function.emitError("CPU entry argument has malformed logical dtype metadata");
                return failure();
            }
            if (FailureOr<CpuCallPlan> plan = getCpuCallPlan(type, module, *dtypes); succeeded(plan)) {
                argument.value = std::move(*plan);
                argument.bridgeArgumentCount = argument.value->lanes.size();
                for (const CpuCallLane &lane : argument.value->lanes)
                    bridgeArgumentTypes.push_back(argument.value->layout.leaves[lane.leafIndex].scalarType);
            } else if (function.getArgAttr(index, kTensorMetadataCarrierAttrName)) {
                auto tuple = dyn_cast<TupleType>(type);
                if (!tuple || !llvm::all_of(tuple.getTypes(), [](Type member) { return member.isIndex(); })) {
                    function.emitError() << "CPU metadata carrier argument #" << index
                                         << " is not a tuple of logical index fields";
                    return failure();
                }
                auto dataLayout = module->getAttrOfType<StringAttr>("llvm.data_layout");
                const unsigned indexBits =
                    dataLayout ? llvm::DataLayout(dataLayout.getValue()).getPointerSizeInBits() : 0;
                if (indexBits != 32 && indexBits != 64) {
                    function.emitError("CPU metadata carrier requires a 32-bit or 64-bit target index");
                    return failure();
                }
                argument.metadataCarrier = true;
                argument.bridgeArgumentCount = tuple.size();
                bridgeArgumentTypes.append(tuple.size(), IntegerType::get(function.getContext(), indexBits));
            } else if (isa<TensorViewType>(type) || isCpuOpaqueAbiType(type)) {
                bridgeArgumentTypes.push_back(type);
            } else {
                function.emitError() << "CPU entry argument #" << index << " of type " << type
                                     << " has no canonical Value ABI plan or explicit opaque ABI contract";
                return failure();
            }
            argumentPlans.push_back(std::move(argument));
        }

        std::optional<CpuCallPlan> resultPlan;
        SmallVector<Type> bridgeResultTypes;
        if (function.getNumResults() == 1) {
            FailureOr<SmallVector<StringRef>> dtypes = logicalDtypes(function.getResultAttrDict(0));
            if (failed(dtypes)) {
                function.emitError("CPU entry result has malformed logical dtype metadata");
                return failure();
            }
            FailureOr<CpuCallPlan> plan = getCpuCallPlan(function.getResultTypes().front(), module, *dtypes);
            if (failed(plan)) {
                function.emitError("CPU entry result has no canonical Value ABI plan");
                return failure();
            }
            resultPlan = std::move(*plan);
            for (const CpuCallLane &lane : resultPlan->lanes)
                bridgeResultTypes.push_back(resultPlan->layout.leaves[lane.leafIndex].scalarType);
        }

        const std::string publicName = function.getSymName().str();
        Type sourceResultType = function.getNumResults() == 1 ? function.getResultTypes().front() : Type();
        OpBuilder builder(module.getBodyRegion());
        builder.setInsertionPoint(function);
        auto bridgeType = builder.getFunctionType(bridgeArgumentTypes, bridgeResultTypes);
        func::FuncOp bridge = func::FuncOp::create(builder, function.getLoc(), publicName, bridgeType);
        for (NamedAttribute attribute : function->getAttrs()) {
            StringRef name = attribute.getName().strref();
            if (name == SymbolTable::getSymbolAttrName() || name == function.getFunctionTypeAttrName() ||
                name == function.getArgAttrsAttrName() || name == function.getResAttrsAttrName())
                continue;
            bridge->setAttr(attribute.getName(), attribute.getValue());
        }
        bridge->setAttr("vernon.entry", builder.getUnitAttr());
        for (auto [sourceIndex, plan] : llvm::enumerate(argumentPlans)) {
            DictionaryAttr attributes = function.getArgAttrDict(sourceIndex);
            if (plan.metadataCarrier)
                continue;
            for (unsigned lane = 0; lane < plan.bridgeArgumentCount; ++lane)
                bridge.setArgAttrs(plan.firstBridgeArgument + lane, attributes);
        }

        bridge.getBody().takeBody(function.getBody());
        Block *entry = &bridge.getBody().front();
        SmallVector<BlockArgument> sourceArguments(entry->getArguments());
        SmallVector<Location> argumentLocations(bridgeArgumentTypes.size(), function.getLoc());
        entry->addArguments(bridgeArgumentTypes, argumentLocations);
        builder.setInsertionPointToStart(entry);
        const unsigned bridgeBase = sourceArguments.size();
        for (auto [sourceIndex, plan] : llvm::enumerate(argumentPlans)) {
            ValueRange bridgeValues =
                entry->getArguments().slice(bridgeBase + plan.firstBridgeArgument, plan.bridgeArgumentCount);
            if (plan.metadataCarrier) {
                SmallVector<TupleGetOp> reads;
                for (Operation *user : sourceArguments[sourceIndex].getUsers()) {
                    auto read = dyn_cast<TupleGetOp>(user);
                    if (!read || read.getInput() != sourceArguments[sourceIndex] ||
                        read.getIndex() >= bridgeValues.size()) {
                        bridge.emitError("CPU metadata carrier has a non-canonical use");
                        bridge.erase();
                        return failure();
                    }
                    reads.push_back(read);
                }
                for (TupleGetOp read : reads) {
                    builder.setInsertionPoint(read);
                    Value logical = arith::IndexCastOp::create(builder, read.getLoc(), builder.getIndexType(),
                                                               bridgeValues[read.getIndex()]);
                    read.getResult().replaceAllUsesWith(logical);
                    read.erase();
                }
                continue;
            }
            if (!plan.value) {
                sourceArguments[sourceIndex].replaceAllUsesWith(bridgeValues.front());
                continue;
            }
            FailureOr<Value> rebuilt =
                buildAggregateValueFromScalars(plan.sourceType, bridgeValues, module, builder, function.getLoc());
            if (failed(rebuilt)) {
                bridge.emitError("cannot rebuild canonical CPU ABI argument");
                bridge.erase();
                return failure();
            }
            sourceArguments[sourceIndex].replaceAllUsesWith(*rebuilt);
        }
        llvm::BitVector eraseArguments(entry->getNumArguments(), false);
        eraseArguments.set(0, sourceArguments.size());
        entry->eraseArguments(eraseArguments);

        if (resultPlan) {
            SmallVector<func::ReturnOp> returns;
            bridge.walk([&](func::ReturnOp op) { returns.push_back(op); });
            for (func::ReturnOp op : returns) {
                if (op.getNumOperands() != 1) {
                    bridge.emitError("CPU ABI source return does not match its result plan");
                    bridge.erase();
                    return failure();
                }
                builder.setInsertionPoint(op);
                FailureOr<SmallVector<Value>> scalars =
                    decomposeAggregateValueToScalars(sourceResultType, op.getOperand(0), module, builder, op.getLoc());
                if (failed(scalars) || scalars->size() != resultPlan->lanes.size()) {
                    bridge.emitError("cannot decompose canonical CPU ABI result");
                    bridge.erase();
                    return failure();
                }
                func::ReturnOp::create(builder, op.getLoc(), *scalars);
                op.erase();
            }
        }
        function.erase();
        return success();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerCPUABIPass() { return std::make_unique<VernonLowerCPUABIPass>(); }

void registerVernonLowerCPUABIPass() { PassRegistration<VernonLowerCPUABIPass>(); }

} // namespace mlir::vernon
