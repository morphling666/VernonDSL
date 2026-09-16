#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonLowerCPUABIPass();
void registerVernonLowerCPUABIPass();

} // namespace vernon
} // namespace mlir
