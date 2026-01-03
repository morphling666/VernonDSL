#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToSPIRV.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Core
  registry.insert<func::FuncDialect>();
  // Dependents (optional, safer)
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<linalg::LinalgDialect>();

  // Your dialect
  registry.insert<vernon::VernonDialect>();

  // Target dialect
  registry.insert<spirv::SPIRVDialect>();

  // Register passes
  vernon::registerVernonToSPIRVPass();

  return failed(MlirOptMain(argc, argv, "Vernon optimizer\n", registry));
}
