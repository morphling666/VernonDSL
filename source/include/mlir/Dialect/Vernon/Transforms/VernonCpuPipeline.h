#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONCPUPIPELINE_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONCPUPIPELINE_H

namespace mlir {
class DialectRegistry;
class OpPassManager;

namespace vernon {

void registerVernonCpuPipelineDialects(DialectRegistry &registry);
void buildVernonCpuPreparationPipeline(OpPassManager &passManager);
void buildVernonCpuLoweringPipeline(OpPassManager &passManager);
void buildVernonCpuPassPipeline(OpPassManager &passManager);
void registerVernonCpuPassPipeline();

} // namespace vernon
} // namespace mlir

#endif
