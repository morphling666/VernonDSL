#pragma once

#include "mlir/IR/Builders.h"

namespace mlir::vernon {

FailureOr<Value> lowerAtan2ToSpirv(Location location, Type resultType, Value y, Value x, OpBuilder &builder);
Value lowerRemFToSpirv(Location location, Value dividend, Value divisor, OpBuilder &builder);

} // namespace mlir::vernon
