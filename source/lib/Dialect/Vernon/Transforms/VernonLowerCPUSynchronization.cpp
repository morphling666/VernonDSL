#include "mlir/Dialect/Vernon/Transforms/VernonLowerSynchronization.h"

#include "VernonCpuWorkgroupABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlir::vernon {
namespace {

struct CpuAllocation {
    uint64_t site{};
    uint64_t elementBytes{};
    uint64_t totalBytes{};
    uint64_t alignment{};
};

struct CpuSynchronizationOps {
    SmallVector<WorkgroupAllocOp> allocations;
    SmallVector<PhysicalLoadOp> loads;
    SmallVector<PhysicalStoreOp> stores;
    SmallVector<PhysicalAtomicOp> atomics;
    SmallVector<BarrierOp> barriers;
};

LogicalResult collectCpuSynchronizationOps(Operation *root, CpuSynchronizationOps &ops) {
    WalkResult projectionCheck = root->walk([&](Operation *operation) {
        if (!isa<LoadOp, StoreOp, AtomicOp>(operation))
            return WalkResult::advance();
        Value storage = isa<StoreOp>(operation) ? operation->getOperand(1) : operation->getOperand(0);
        auto view = dyn_cast<TensorViewType>(storage.getType());
        if (!view || view.getAddressSpace() != "workgroup")
            return WalkResult::advance();
        operation->emitError("workgroup storage operation reached CPU synchronization lowering before projection");
        return WalkResult::interrupt();
    });
    if (projectionCheck.wasInterrupted())
        return failure();

    root->walk([&](WorkgroupAllocOp op) { ops.allocations.push_back(op); });
    root->walk([&](PhysicalLoadOp op) {
        if (cast<TensorViewType>(op.getStorage().getType()).getAddressSpace() == "workgroup")
            ops.loads.push_back(op);
    });
    root->walk([&](PhysicalStoreOp op) {
        if (cast<TensorViewType>(op.getStorage().getType()).getAddressSpace() == "workgroup")
            ops.stores.push_back(op);
    });
    root->walk([&](PhysicalAtomicOp op) {
        if (cast<TensorViewType>(op.getStorage().getType()).getAddressSpace() == "workgroup")
            ops.atomics.push_back(op);
    });
    root->walk([&](BarrierOp op) { ops.barriers.push_back(op); });
    return success();
}

func::FuncOp declareCpuHelper(ModuleOp module, StringRef name, FunctionType type) {
    if (Operation *existing = SymbolTable::lookupSymbolIn(module, name))
        return dyn_cast<func::FuncOp>(existing);
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    func::FuncOp function = func::FuncOp::create(builder, module.getLoc(), name, type);
    function.setPrivate();
    function->setAttr("llvm.linkage", LLVM::LinkageAttr::get(module.getContext(), LLVM::Linkage::ExternWeak));
    return function;
}

FailureOr<DenseMap<Value, CpuAllocation>> planCpuAllocations(SmallVectorImpl<WorkgroupAllocOp> &allocations) {
    constexpr uint64_t allocationAlignment = 16;
    constexpr uint64_t storageLimit = 16 * 1024;
    DenseMap<Value, CpuAllocation> result;
    uint64_t nextOffset = 0;
    for (auto [site, op] : llvm::enumerate(allocations)) {
        TensorViewType type = op.getResult().getType();
        Type elementType = type.getElementType();
        if (!elementType.isIntOrFloat()) {
            op.emitError("CPU workgroup storage must be projected to scalar integer or floating-point leaves");
            return failure();
        }
        const uint64_t elementBytes = std::max<uint64_t>(1, (elementType.getIntOrFloatBitWidth() + 7) / 8);
        uint64_t elementCount = 1;
        for (int64_t extent : type.getShape()) {
            if (extent <= 0 || elementCount > UINT64_MAX / static_cast<uint64_t>(extent)) {
                op.emitError("CPU workgroup storage size overflows");
                return failure();
            }
            elementCount *= static_cast<uint64_t>(extent);
        }
        if (elementCount > UINT64_MAX / elementBytes) {
            op.emitError("CPU workgroup storage byte size overflows");
            return failure();
        }
        const uint64_t totalBytes = elementCount * elementBytes;
        if (nextOffset > UINT64_MAX - (allocationAlignment - 1)) {
            op.emitError("CPU workgroup storage alignment overflows");
            return failure();
        }
        nextOffset = (nextOffset + allocationAlignment - 1) & ~(allocationAlignment - 1);
        if (nextOffset > storageLimit || totalBytes > storageLimit - nextOffset) {
            op.emitError("combined CPU workgroup storage exceeds the portable 16 KiB limit");
            return failure();
        }
        nextOffset += totalBytes;
        result.insert({op.getResult(), CpuAllocation{site, elementBytes, totalBytes, allocationAlignment}});
    }
    return result;
}

class CpuSynchronizationLowering {
public:
    CpuSynchronizationLowering(Operation *root, IRRewriter &rewriter, func::FuncOp addressHelper)
        : root(root), rewriter(rewriter), addressHelper(addressHelper) {}

    LogicalResult preflight(const DenseMap<Value, CpuAllocation> &allocations, func::FuncOp barrierHelper) {
        if (allocations.empty())
            return success();
        auto function = dyn_cast<func::FuncOp>(root);
        if (!function || !function.getBody().hasOneBlock() || function.getNumResults() != 0)
            return root->emitError("CPU workgroup allocation preflight requires a single-block void compute entry");

        Block &entry = function.getBody().front();
        SmallVector<Operation *> bodyOperations;
        for (Operation &operation : entry.without_terminator())
            bodyOperations.push_back(&operation);
        rewriter.setInsertionPointToStart(&entry);
        Location location = function.getLoc();
        Type i64 = rewriter.getI64Type();
        auto constant = [&](uint64_t value) {
            return arith::ConstantOp::create(rewriter, location, i64,
                                             rewriter.getI64IntegerAttr(static_cast<int64_t>(value)));
        };
        Value ready = arith::ConstantOp::create(rewriter, location, rewriter.getI1Type(), rewriter.getBoolAttr(true));
        SmallVector<CpuAllocation> plan;
        plan.reserve(allocations.size());
        for (const auto &entry : allocations)
            plan.push_back(entry.second);
        llvm::sort(plan, [](const CpuAllocation &left, const CpuAllocation &right) { return left.site < right.site; });
        for (const CpuAllocation &allocation : plan) {
            Value raw = func::CallOp::create(rewriter, location, addressHelper,
                                             ValueRange{constant(allocation.site), constant(allocation.totalBytes),
                                                        constant(allocation.alignment), constant(0)})
                            .getResult(0);
            Value allocated = arith::CmpIOp::create(rewriter, location, arith::CmpIPredicate::ne, raw, constant(0));
            ready = arith::AndIOp::create(rewriter, location, ready, allocated);
        }
        func::CallOp::create(rewriter, location, barrierHelper,
                             ValueRange{constant(std::numeric_limits<uint64_t>::max())});
        scf::IfOp execute = scf::IfOp::create(rewriter, location, ready, false);
        Operation *yield = execute.getThenRegion().front().getTerminator();
        for (Operation *operation : bodyOperations)
            operation->moveBefore(yield);
        return success();
    }

    LogicalResult lower(const CpuSynchronizationOps &ops, const DenseMap<Value, CpuAllocation> &allocations,
                        func::FuncOp barrierHelper) {
        this->allocations = &allocations;
        for (PhysicalLoadOp op : ops.loads) {
            FailureOr<Value> pointer = address(op, op.getStorage(), op.getIndex());
            if (failed(pointer))
                return failure();
            rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getResult().getType(), *pointer);
        }
        for (PhysicalStoreOp op : ops.stores) {
            FailureOr<Value> pointer = address(op, op.getStorage(), op.getIndex());
            if (failed(pointer))
                return failure();
            rewriter.setInsertionPoint(op);
            LLVM::StoreOp::create(rewriter, op.getLoc(), op.getValue(), *pointer);
            rewriter.eraseOp(op);
        }
        for (PhysicalAtomicOp op : ops.atomics) {
            FailureOr<Value> pointer = address(op, op.getStorage(), op.getIndex());
            if (failed(pointer))
                return failure();
            const bool floating = isa<FloatType>(op.getValue().getType());
            LLVM::AtomicBinOp kind = op.getAtomicKind() == "add"
                                         ? (floating ? LLVM::AtomicBinOp::fadd : LLVM::AtomicBinOp::add)
                                     : op.getAtomicKind() == "min"  ? LLVM::AtomicBinOp::min
                                     : op.getAtomicKind() == "max"  ? LLVM::AtomicBinOp::max
                                     : op.getAtomicKind() == "umin" ? LLVM::AtomicBinOp::umin
                                     : op.getAtomicKind() == "umax" ? LLVM::AtomicBinOp::umax
                                                                    : LLVM::AtomicBinOp::xchg;
            rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(op, kind, *pointer, op.getValue(),
                                                           LLVM::AtomicOrdering::monotonic);
        }
        for (WorkgroupAllocOp op : ops.allocations)
            rewriter.eraseOp(op);
        for (size_t site = 0; site < ops.barriers.size(); ++site) {
            BarrierOp op = ops.barriers[site];
            rewriter.setInsertionPoint(op);
            auto siteValue = arith::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI64Type(),
                                                       rewriter.getI64IntegerAttr(static_cast<int64_t>(site)));
            func::CallOp::create(rewriter, op.getLoc(), barrierHelper, ValueRange{siteValue});
            rewriter.eraseOp(op);
        }
        return success();
    }

private:
    FailureOr<Value> address(Operation *operation, Value storage, Value index) {
        const auto found = allocations->find(storage);
        if (found == allocations->end()) {
            operation->emitError("workgroup storage must originate from vernon.workgroup_alloc");
            return failure();
        }
        const CpuAllocation allocation = found->second;
        rewriter.setInsertionPoint(operation);
        Type i64 = rewriter.getI64Type();
        auto constant = [&](uint64_t value) {
            return arith::ConstantOp::create(rewriter, operation->getLoc(), i64,
                                             rewriter.getI64IntegerAttr(static_cast<int64_t>(value)));
        };
        Value physicalIndex = arith::IndexCastOp::create(rewriter, operation->getLoc(), i64, index);
        Value offset =
            arith::MulIOp::create(rewriter, operation->getLoc(), physicalIndex, constant(allocation.elementBytes));
        Value raw = func::CallOp::create(rewriter, operation->getLoc(), addressHelper,
                                         ValueRange{constant(allocation.site), constant(allocation.totalBytes),
                                                    constant(allocation.alignment), offset})
                        .getResult(0);
        return LLVM::IntToPtrOp::create(rewriter, operation->getLoc(), LLVM::LLVMPointerType::get(root->getContext()),
                                        raw)
            .getResult();
    }

    Operation *root;
    IRRewriter &rewriter;
    func::FuncOp addressHelper;
    const DenseMap<Value, CpuAllocation> *allocations{};
};

struct LowerCpuSynchronizationPass final : PassWrapper<LowerCpuSynchronizationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCpuSynchronizationPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect, scf::SCFDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        IRRewriter rewriter(module.getContext());
        Type i64 = rewriter.getI64Type();
        func::FuncOp addressHelper = declareCpuHelper(module, VERNON_CPU_WORKGROUP_ADDRESS_V1_SYMBOL,
                                                      rewriter.getFunctionType({i64, i64, i64, i64}, {i64}));
        func::FuncOp barrierHelper =
            declareCpuHelper(module, VERNON_CPU_WORKGROUP_BARRIER_V1_SYMBOL, rewriter.getFunctionType({i64}, {}));
        if (!addressHelper || !barrierHelper) {
            module.emitError("CPU workgroup helper declaration conflicts with an existing symbol");
            return signalPassFailure();
        }
        for (func::FuncOp function : module.getOps<func::FuncOp>()) {
            if (function.isDeclaration())
                continue;
            CpuSynchronizationOps ops;
            if (failed(collectCpuSynchronizationOps(function, ops)))
                return signalPassFailure();
            FailureOr<DenseMap<Value, CpuAllocation>> allocations = planCpuAllocations(ops.allocations);
            if (failed(allocations))
                return signalPassFailure();
            CpuSynchronizationLowering lowering(function, rewriter, addressHelper);
            if (failed(lowering.preflight(*allocations, barrierHelper)))
                return signalPassFailure();
            if (failed(lowering.lower(ops, *allocations, barrierHelper)))
                return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerCPUSynchronizationPass() {
    return std::make_unique<LowerCpuSynchronizationPass>();
}

} // namespace mlir::vernon
