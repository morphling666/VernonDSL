#include "mlir/Dialect/Vernon/Transforms/VernonLowerSynchronization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
        WalkResult projectionCheck = root->walk([&](Operation *operation) {
            if (!isa<LoadOp, StoreOp, AtomicOp>(operation))
                return WalkResult::advance();
            Value storage = isa<StoreOp>(operation) ? operation->getOperand(1) : operation->getOperand(0);
            auto view = dyn_cast<TensorViewType>(storage.getType());
            if (!view || view.getAddressSpace() != "workgroup")
                return WalkResult::advance();
            operation->emitError("workgroup storage operation reached synchronization lowering before projection");
            return WalkResult::interrupt();
        });
        if (projectionCheck.wasInterrupted())
            return signalPassFailure();

        SmallVector<WorkgroupAllocOp> allocations;
        SmallVector<PhysicalLoadOp> physicalLoads;
        SmallVector<PhysicalStoreOp> physicalStores;
        SmallVector<PhysicalAtomicOp> physicalAtomics;
        SmallVector<BarrierOp> barriers;
        root->walk([&](WorkgroupAllocOp op) { allocations.push_back(op); });
        root->walk([&](PhysicalLoadOp op) {
            if (cast<TensorViewType>(op.getStorage().getType()).getAddressSpace() == "workgroup")
                physicalLoads.push_back(op);
        });
        root->walk([&](PhysicalStoreOp op) {
            if (cast<TensorViewType>(op.getStorage().getType()).getAddressSpace() == "workgroup")
                physicalStores.push_back(op);
        });
        root->walk([&](PhysicalAtomicOp op) {
            if (cast<TensorViewType>(op.getStorage().getType()).getAddressSpace() == "workgroup")
                physicalAtomics.push_back(op);
        });
        root->walk([&](BarrierOp op) { barriers.push_back(op); });

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
            int64_t elementCount = 1;
            for (int64_t extent : type.getShape())
                elementCount *= extent;
            MemRefType memref =
                MemRefType::get({elementCount}, type.getElementType(), MemRefLayoutAttrInterface{}, memorySpace);
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
                const unsigned attributionIndex = function.getNumWorkgroupAttributions();
                replacement = function.addWorkgroupAttribution(memref, op.getLoc());
                // Aggregate leaves can be vectorized after this pass.  Give
                // shared globals the maximum portable vector alignment so an
                // adjacent byte-sized leaf cannot leave a vector load based
                // at only scalar alignment on NVPTX.
                function.setWorkgroupAttributionAttr(attributionIndex, LLVM::LLVMDialect::getAlignAttrName(),
                                                     rewriter.getI64IntegerAttr(16));
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
        for (PhysicalLoadOp op : physicalLoads)
            if (!requireAllocation(op, op.getStorage()))
                return;
        for (PhysicalStoreOp op : physicalStores)
            if (!requireAllocation(op, op.getStorage()))
                return;
        for (PhysicalAtomicOp op : physicalAtomics)
            if (!requireAllocation(op, op.getStorage()))
                return;
        for (PhysicalLoadOp op : physicalLoads) {
            rewriter.setInsertionPoint(op);
            rewriter.replaceOpWithNewOp<memref::LoadOp>(op, loweredStorage.lookup(op.getStorage()), op.getIndex());
        }
        for (PhysicalStoreOp op : physicalStores) {
            rewriter.setInsertionPoint(op);
            memref::StoreOp::create(rewriter, op.getLoc(), op.getValue(), loweredStorage.lookup(op.getStorage()),
                                    op.getIndex());
            rewriter.eraseOp(op);
        }
        for (PhysicalAtomicOp op : physicalAtomics) {
            arith::AtomicRMWKind kind = op.getAtomicKind() == "add"
                                            ? (isa<FloatType>(op.getValue().getType()) ? arith::AtomicRMWKind::addf
                                                                                       : arith::AtomicRMWKind::addi)
                                        : op.getAtomicKind() == "min"  ? arith::AtomicRMWKind::mins
                                        : op.getAtomicKind() == "max"  ? arith::AtomicRMWKind::maxs
                                        : op.getAtomicKind() == "umin" ? arith::AtomicRMWKind::minu
                                        : op.getAtomicKind() == "umax" ? arith::AtomicRMWKind::maxu
                                                                       : arith::AtomicRMWKind::assign;
            rewriter.setInsertionPoint(op);
            rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(op, kind, op.getValue(),
                                                             loweredStorage.lookup(op.getStorage()), op.getIndex());
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
