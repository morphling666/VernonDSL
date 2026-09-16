#include "mlir/Dialect/Vernon/Transforms/VernonVerifyCPUAutodiffABI.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::vernon {
namespace {

constexpr StringLiteral kTapeAllocatorBuiltin = "ad_tape_allocator";

struct VernonVerifyCPUAutodiffABIPass final : PassWrapper<VernonVerifyCPUAutodiffABIPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonVerifyCPUAutodiffABIPass)

    StringRef getArgument() const final { return "vernon-verify-cpu-autodiff-abi"; }
    StringRef getDescription() const final { return "Verify hidden CPU autodiff Compiler/Runtime arguments"; }

    void runOnOperation() override {
        bool invalid = false;
        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            unsigned allocatorCount = 0;
            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                auto builtin = function.getArgAttrOfType<StringAttr>(index, kBuiltinAttrName);
                if (!builtin || builtin.getValue() != kTapeAllocatorBuiltin)
                    continue;
                ++allocatorCount;
                const auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
                const auto interface = function.getArgAttrOfType<StringAttr>(index, kInterfaceAttrName);
                if (!function->hasAttr(kEntryAttrName) || !stage || stage.getValue() != "compute") {
                    function.emitError() << "argument #" << index
                                         << " ad_tape_allocator is only valid on a compute entry";
                    invalid = true;
                }
                if (!function.getArgument(index).getType().isIndex()) {
                    function.emitError() << "argument #" << index << " ad_tape_allocator must have index type";
                    invalid = true;
                }
                if (!interface || interface.getValue() != "input") {
                    function.emitError() << "argument #" << index << " ad_tape_allocator must use the input interface";
                    invalid = true;
                }
                if (function.getArgAttr(index, kLocationAttrName) ||
                    function.getArgAttr(index, kDescriptorSetAttrName) ||
                    function.getArgAttr(index, kBindingAttrName) || function.getArgAttr(index, "vernon.source_name")) {
                    function.emitError() << "argument #" << index
                                         << " ad_tape_allocator cannot carry a user interface location or name";
                    invalid = true;
                }
            }
            if (allocatorCount > 1) {
                function.emitError("contains more than one ad_tape_allocator argument");
                invalid = true;
            }
        }
        if (invalid)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonVerifyCPUAutodiffABIPass() {
    return std::make_unique<VernonVerifyCPUAutodiffABIPass>();
}

void registerVernonVerifyCPUAutodiffABIPass() { PassRegistration<VernonVerifyCPUAutodiffABIPass>(); }

} // namespace mlir::vernon
