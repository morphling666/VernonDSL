#include "mlir/Dialect/Vernon/Transforms/VernonLowerSynchronization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vernon {
namespace {

struct LowerSynchronizationPass final : PassWrapper<LowerSynchronizationPass, OperationPass<>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSynchronizationPass)

    LowerSynchronizationPass() = default;
    LowerSynchronizationPass(bool gpu, bool spirv) : gpuTarget(gpu), spirvTarget(spirv) {}

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<gpu::GPUDialect, memref::MemRefDialect, spirv::SPIRVDialect>();
    }

    void runOnOperation() override {
        Operation *root = getOperation();
        IRRewriter rewriter(root->getContext());
        SmallVector<WorkgroupAllocOp> allocations;
        SmallVector<LoadOp> loads;
        SmallVector<StoreOp> stores;
        SmallVector<AtomicOp> atomics;
        SmallVector<BarrierOp> barriers;
        root->walk([&](WorkgroupAllocOp op) { allocations.push_back(op); });
        root->walk([&](LoadOp op) {
            auto view = dyn_cast<TensorViewType>(op.getStorage().getType());
            if (view && view.getAddressSpace() == "workgroup")
                loads.push_back(op);
        });
        root->walk([&](StoreOp op) {
            auto view = dyn_cast<TensorViewType>(op.getStorage().getType());
            if (view && view.getAddressSpace() == "workgroup")
                stores.push_back(op);
        });
        root->walk([&](AtomicOp op) { atomics.push_back(op); });
        root->walk([&](BarrierOp op) { barriers.push_back(op); });

        DenseMap<Operation *, Value> sourceStorage;
        for (LoadOp op : loads)
            sourceStorage.insert({op, op.getStorage()});
        for (StoreOp op : stores)
            sourceStorage.insert({op, op.getStorage()});
        for (AtomicOp op : atomics)
            sourceStorage.insert({op, op.getStorage()});
        DenseMap<Value, Value> loweredStorage;
        for (WorkgroupAllocOp op : allocations) {
            TensorViewType type = op.getResult().getType();
            if (type.getAddressSpace() != "workgroup") {
                op.emitError("expected a workgroup TensorView result type");
                return signalPassFailure();
            }
            Attribute memorySpace;
            if (spirvTarget)
                memorySpace = spirv::StorageClassAttr::get(root->getContext(), spirv::StorageClass::Workgroup);
            else if (gpuTarget)
                memorySpace = gpu::AddressSpaceAttr::get(root->getContext(), gpu::AddressSpace::Workgroup);
            MemRefType memref =
                MemRefType::get(type.getShape(), type.getElementType(), MemRefLayoutAttrInterface{}, memorySpace);
            Value replacement;
            if (spirvTarget) {
                rewriter.setInsertionPoint(op);
                replacement = memref::AllocOp::create(rewriter, op.getLoc(), memref);
            } else if (gpuTarget) {
                gpu::GPUFuncOp function = op->getParentOfType<gpu::GPUFuncOp>();
                if (!function) {
                    op.emitError("workgroup allocation must be inside a GPU function");
                    return signalPassFailure();
                }
                replacement = function.addWorkgroupAttribution(memref, op.getLoc());
            } else {
                rewriter.setInsertionPoint(op);
                replacement = memref::AllocaOp::create(rewriter, op.getLoc(), memref);
            }
            loweredStorage.insert({op.getResult(), replacement});
        }
        auto requireAllocation = [&](Operation *operation, Value storage) {
            if (loweredStorage.find(storage) != loweredStorage.end())
                return true;
            operation->emitError("workgroup storage must originate from vernon.workgroup_alloc");
            signalPassFailure();
            return false;
        };
        for (LoadOp op : loads)
            if (!requireAllocation(op, sourceStorage.lookup(op)))
                return;
        for (StoreOp op : stores)
            if (!requireAllocation(op, sourceStorage.lookup(op)))
                return;
        for (AtomicOp op : atomics)
            if (auto view = dyn_cast<TensorViewType>(sourceStorage.lookup(op).getType());
                view && view.getAddressSpace() == "workgroup" && !requireAllocation(op, sourceStorage.lookup(op)))
                return;
        for (LoadOp op : loads) {
            rewriter.setInsertionPoint(op);
            rewriter.replaceOpWithNewOp<memref::LoadOp>(op, loweredStorage.lookup(sourceStorage.lookup(op)),
                                                        op.getIndices());
        }
        for (StoreOp op : stores) {
            rewriter.setInsertionPoint(op);
            memref::StoreOp::create(rewriter, op.getLoc(), op.getValue(),
                                    loweredStorage.lookup(sourceStorage.lookup(op)), op.getIndices());
            rewriter.eraseOp(op);
        }
        for (AtomicOp op : atomics) {
            Value source = sourceStorage.lookup(op);
            auto sourceView = dyn_cast<TensorViewType>(source.getType());
            if (!sourceView || sourceView.getAddressSpace() != "workgroup")
                continue;
            Value storage = loweredStorage.lookup(source);
            if (!isa<MemRefType>(storage.getType())) {
                op.emitError("device atomic TensorView storage was not lowered to an addressable buffer");
                return signalPassFailure();
            }
            arith::AtomicRMWKind kind = op.getAtomicKind() == "add"    ? arith::AtomicRMWKind::addi
                                        : op.getAtomicKind() == "min"  ? arith::AtomicRMWKind::mins
                                        : op.getAtomicKind() == "max"  ? arith::AtomicRMWKind::maxs
                                        : op.getAtomicKind() == "umin" ? arith::AtomicRMWKind::minu
                                        : op.getAtomicKind() == "umax" ? arith::AtomicRMWKind::maxu
                                                                       : arith::AtomicRMWKind::assign;
            rewriter.setInsertionPoint(op);
            rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(op, kind, op.getValue(), storage, op.getIndices());
        }
        for (WorkgroupAllocOp op : allocations)
            rewriter.eraseOp(op);
        for (BarrierOp op : barriers) {
            rewriter.setInsertionPoint(op);
            if (!gpuTarget) {
                rewriter.eraseOp(op);
            } else if (spirvTarget && op.getScope() == "device") {
                auto scope = spirv::ScopeAttr::get(root->getContext(), spirv::Scope::Device);
                auto semantics = spirv::MemorySemanticsAttr::get(
                    root->getContext(), spirv::MemorySemantics::UniformMemory | spirv::MemorySemantics::AcquireRelease);
                spirv::MemoryBarrierOp::create(rewriter, op.getLoc(), scope, semantics);
                rewriter.eraseOp(op);
            } else {
                rewriter.replaceOpWithNewOp<gpu::BarrierOp>(op);
            }
        }
    }

    bool gpuTarget{};
    bool spirvTarget{};
};

} // namespace

std::unique_ptr<Pass> createVernonLowerSynchronizationPass(bool gpu, bool spirv) {
    return std::make_unique<LowerSynchronizationPass>(gpu, spirv);
}

} // namespace mlir::vernon
