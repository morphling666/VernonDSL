#include "mlir/Dialect/Vernon/Transforms/VernonToSPIRV.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::vernon {

namespace {

struct VernonToSPIRVPass
    : public PassWrapper<VernonToSPIRVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonToSPIRVPass)

  StringRef getArgument() const final { return "vernon-to-spirv"; }

  StringRef getDescription() const final {
    return "Lower Vernon shader semantics to SPIR-V interface variables";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // TODO:
    // 1. Find funcs with vernon.semantic attrs
    // 2. Create spirv.module
    // 3. Convert func -> spirv.func
    // 4. Replace tensors/vectors with spirv types
    func::FuncOp vertex_main = module.lookupSymbol<func::FuncOp>("vertex_main");
    // builder.create<spirv::module>(module.getLoc());
    for (auto arg : vertex_main.getArguments()) {
      unsigned idx = arg.getArgNumber();
      auto semantic =
          vertex_main.getArgAttrOfType<StringAttr>(idx, "vernon.semantic");
      if (!semantic) {
        continue;
      }
      spirv::StorageClass sc;
      if (semantic == "input") {
        sc = spirv::StorageClass::Input;
      } else if (semantic == "output") {
        sc = spirv::StorageClass::Output;
      } else if (semantic == "uniform") {
        sc = spirv::StorageClass::Uniform;
      } else if (semantic == "uniform_constant") {
        sc = spirv::StorageClass::UniformConstant;
      } else if (semantic == "storage_buffer") {
        sc = spirv::StorageClass::StorageBuffer;
      } else if (semantic == "image") {
        sc = spirv::StorageClass::Input;
      }
      llvm::errs() << "semantic: " << semantic << " sc: " << sc << "\n";
    }
    // vertex_main->setAttr("stage",
    // spirv::ExecutionStageAttr::get(module.getContext(),
    // spirv::ExecutionStage::Vertex));
    func::FuncOp frag_main = module.lookupSymbol<func::FuncOp>("frag_main");
    // frag_main->setAttr("stage",
    // spirv::ExecutionStageAttr::get(module.getContext(),
    // spirv::ExecutionStage::Fragment));
  }
};

} // namespace

std::unique_ptr<Pass> createVernonToSPIRVPass() {
  return std::make_unique<VernonToSPIRVPass>();
}

void registerVernonToSPIRVPass() { PassRegistration<VernonToSPIRVPass>(); }

} // namespace mlir::vernon
