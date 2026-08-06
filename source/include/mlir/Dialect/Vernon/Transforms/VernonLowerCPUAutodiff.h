#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::vernon {

/// Materializes the hidden CPU allocator/root argument annotations while the
/// function still has its logical autodiff types.
std::unique_ptr<Pass> createVernonPrepareCPUAutodiffSignaturesPass();

/// Materializes the hidden host allocator ABI for logical autodiff tape
/// operations. The resulting entry signatures contain only canonical values
/// and internal index-sized allocator/region handles.
std::unique_ptr<Pass> createVernonLowerCPUAutodiffPass();
std::unique_ptr<Pass> createVernonCPUAutodiffToLLVMPass();
void registerVernonPrepareCPUAutodiffSignaturesPass();
void registerVernonLowerCPUAutodiffPass();
void registerVernonCPUAutodiffToLLVMPass();

} // namespace mlir::vernon
