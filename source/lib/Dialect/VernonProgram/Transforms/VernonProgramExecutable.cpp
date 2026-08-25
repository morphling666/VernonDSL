#include "mlir/Dialect/VernonProgram/Transforms/VernonProgramExecutable.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/VernonProgram/IR/VernonProgram.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

namespace mlir::vernon::program {
namespace {

struct VernonProgramBuildExecutablePass final : PassWrapper<VernonProgramBuildExecutablePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonProgramBuildExecutablePass)

    StringRef getArgument() const final { return "vernon-program-build-executable"; }
    StringRef getDescription() const final { return "Derive executable value, dependency, and resource metadata"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect, VernonProgramDialect>();
    }

    void runOnOperation() override {
        Builder builder(&getContext());
        uint32_t nextValue = 0;
        bool invalid = false;
        func::FuncOp forward;
        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            auto direction = function->getAttrOfType<StringAttr>("vernon_program.graph");
            if (!direction || direction.getValue() != "forward")
                continue;
            if (forward) {
                function.emitError("executable Program has more than one forward graph");
                invalid = true;
                continue;
            }
            forward = function;
        }
        SmallVector<uint32_t> forwardValueIds;
        DenseMap<Value, uint32_t> forwardIds;
        if (forward && llvm::hasSingleElement(forward.getBody())) {
            for (Value argument : forward.getArguments()) {
                forwardIds[argument] = nextValue;
                forwardValueIds.push_back(nextValue++);
            }
            for (Operation &operation : forward.getBody().front().without_terminator())
                for (Value result : operation.getResults()) {
                    forwardIds[result] = nextValue;
                    forwardValueIds.push_back(nextValue++);
                }
        }

        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            if (!function->hasAttr("vernon_program.graph"))
                continue;
            if (!llvm::hasSingleElement(function.getBody())) {
                function.emitError("executable Program graph requires one straight-line block");
                invalid = true;
                continue;
            }
            DenseMap<Value, uint32_t> values;
            for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
                uint32_t valueId = nextValue;
                bool assigned = false;
                if (function == forward) {
                    valueId = forwardIds.lookup(argument);
                    assigned = true;
                } else if (auto capture = function.getArgAttrOfType<IntegerAttr>(index, kCaptureForwardValueAttr)) {
                    const int64_t source = capture.getInt();
                    if (!forward || source < 0 || static_cast<uint64_t>(source) >= forwardValueIds.size()) {
                        function.emitError("capture argument references an unknown forward value");
                        invalid = true;
                    } else {
                        valueId = forwardValueIds[static_cast<size_t>(source)];
                        assigned = true;
                    }
                }
                if (!assigned)
                    ++nextValue;
                values[argument] = valueId;
                function.setArgAttr(index, "vernon_program.value_id", builder.getI32IntegerAttr(valueId));
            }
            uint32_t nextNode = 0;
            for (Operation &operation : function.getBody().front().without_terminator()) {
                if (isStorageAllocIntrinsic(&operation)) {
                    for (Value result : operation.getResults()) {
                        const uint32_t valueId = function == forward ? forwardIds.lookup(result) : nextValue++;
                        values[result] = valueId;
                    }
                    continue;
                }
                if (!isa<ComputeOp, GraphicsOp>(operation)) {
                    operation.emitError(
                        "executable Program graph contains an operation without a selected implementation");
                    invalid = true;
                    continue;
                }
                SmallVector<int64_t> operandIds;
                llvm::SmallSet<int64_t, 4> dependencySet;
                for (Value operand : operation.getOperands()) {
                    auto found = values.find(operand);
                    if (found == values.end()) {
                        operation.emitError("executable Program operand has no stable value id");
                        invalid = true;
                        continue;
                    }
                    operandIds.push_back(found->second);
                    if (Operation *producer = operand.getDefiningOp())
                        if (auto node = producer->getAttrOfType<IntegerAttr>("vernon_program.node_id"))
                            dependencySet.insert(node.getInt());
                }
                SmallVector<int64_t> resultIds;
                for (Value result : operation.getResults()) {
                    const uint32_t valueId = function == forward ? forwardIds.lookup(result) : nextValue++;
                    values[result] = valueId;
                    resultIds.push_back(valueId);
                }
                SmallVector<int64_t> dependencies(dependencySet.begin(), dependencySet.end());
                llvm::sort(dependencies);
                operation.setAttr("vernon_program.node_id", builder.getI32IntegerAttr(nextNode++));
                operation.setAttr("vernon_program.operand_value_ids", builder.getDenseI64ArrayAttr(operandIds));
                operation.setAttr("vernon_program.result_value_ids", builder.getDenseI64ArrayAttr(resultIds));
                operation.setAttr("vernon_program.dependencies", builder.getDenseI64ArrayAttr(dependencies));
                if (auto graphics = dyn_cast<GraphicsOp>(operation)) {
                    SmallVector<int64_t> resourceOperands;
                    for (size_t index = 0; index < graphics.getNumResults(); ++index)
                        resourceOperands.push_back(static_cast<int64_t>(index));
                    for (auto [index, argument] : llvm::enumerate(graphics.getLogicalArguments()))
                        if (isa<vernon::TensorViewType, vernon::TextureType, vernon::SamplerType>(argument.getType()))
                            resourceOperands.push_back(static_cast<int64_t>(index + graphics.getNumResults()));
                    SmallVector<int64_t> resultSources;
                    for (size_t index = 0; index < graphics.getNumResults(); ++index)
                        resultSources.push_back(static_cast<int64_t>(index));
                    operation.setAttr("vernon_program.resource_operand_indices",
                                      builder.getDenseI64ArrayAttr(resourceOperands));
                    operation.setAttr("vernon_program.result_resource_sources",
                                      builder.getDenseI64ArrayAttr(resultSources));
                    operation.setAttr("vernon_program.color_count",
                                      builder.getI32IntegerAttr(graphics.getColorCount()));
                }
                if (!operation.hasAttr("vernon_program.operand_accesses"))
                    operation.setAttr(
                        "vernon_program.operand_accesses",
                        builder.getArrayAttr(llvm::map_to_vector(
                            llvm::seq<size_t>(0, operation.getNumOperands()), [&](size_t index) -> Attribute {
                                const bool attachment = isa<GraphicsOp>(operation) && index < operation.getNumResults();
                                return builder.getStringAttr(attachment ? "read_write" : "read");
                            })));
                if (!operation.hasAttr("vernon_program.result_accesses"))
                    operation.setAttr("vernon_program.result_accesses",
                                      builder.getArrayAttr(SmallVector<Attribute>(operation.getNumResults(),
                                                                                  builder.getStringAttr("write"))));
                StringRef callee = isa<ComputeOp>(operation) ? cast<ComputeOp>(operation).getCallee()
                                                             : cast<GraphicsOp>(operation).getCallee();
                operation.setAttr("vernon_program.stage", builder.getStringAttr(callee));
            }
            auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
            SmallVector<int64_t> resultIds;
            if (!returnOp) {
                function.emitError("executable Program graph requires func.return");
                invalid = true;
            } else {
                for (Value result : returnOp.getOperands()) {
                    auto found = values.find(result);
                    if (found == values.end()) {
                        returnOp.emitError("returns a value without a stable executable id");
                        invalid = true;
                        continue;
                    }
                    resultIds.push_back(found->second);
                }
            }
            function->setAttr("vernon_program.result_value_ids", builder.getDenseI64ArrayAttr(resultIds));
        }
        getOperation()->setAttr("vernon_program.value_count", builder.getI32IntegerAttr(nextValue));
        if (invalid)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonProgramBuildExecutablePass() {
    return std::make_unique<VernonProgramBuildExecutablePass>();
}

void registerVernonProgramBuildExecutablePass() { PassRegistration<VernonProgramBuildExecutablePass>(); }

} // namespace mlir::vernon::program
