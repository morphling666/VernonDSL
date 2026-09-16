#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONVERIFYCPUAUTODIFFABI_H_
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONVERIFYCPUAUTODIFFABI_H_

#include "mlir/Pass/Pass.h"

namespace mlir::vernon {

std::unique_ptr<Pass> createVernonVerifyCPUAutodiffABIPass();
void registerVernonVerifyCPUAutodiffABIPass();

} // namespace mlir::vernon

#endif // MLIR_DIALECT_VERNON_TRANSFORMS_VERNONVERIFYCPUAUTODIFFABI_H_
