#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonConvertGPUToSPIRV.h"
#include "mlir/Dialect/Vernon/Transforms/VernonCpuPipeline.h"
#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUResources.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCUDAMath.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
    DialectRegistry registry;

    // Core
    registry.insert<func::FuncDialect>();
    registry.insert<gpu::GPUDialect>();
    // Standard computation and bufferization dialects used by Vernon programs.
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<bufferization::BufferizationDialect>();

    // Your dialect
    registry.insert<vernon::VernonDialect>();
    registerAllExtensions(registry);
    vernon::registerVernonCpuPipelineDialects(registry);

    // Target dialect
    registry.insert<spirv::SPIRVDialect>();
    func::registerInlinerExtension(registry);

    // mlir::registerAllPasses();
    registerTransformsPasses();
    mlir::bufferization::registerBufferizationPasses();
    registerLinalgPasses();
    memref::registerMemRefPasses();
    spirv::registerSPIRVPasses();
    // Register passes
    vernon::registerVernonValidatePass();
    vernon::registerVernonInlineHelpersPass();
    vernon::registerVernonLowerCPUTensorsPass();
    vernon::registerVernonLowerCPUResourcesPass();
    vernon::registerVernonCpuPassPipeline();
    vernon::registerVernonConvertGPUToSPIRVPass();
    vernon::registerVernonLowerCUDAMathPass();
    vernon::registerVernonLowerGPUTensorsPass();
    vernon::registerVernonToGPUPass();
    vernon::registerVernonToSPIRVPass();

    return failed(MlirOptMain(argc, argv, "Vernon optimizer\n", registry));
}
