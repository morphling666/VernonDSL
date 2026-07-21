#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <optional>

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

namespace vernon {

// A missing element limit converts every non-empty static Tensor.
FailureOr<VectorType>
convertStaticTensorToVector(Type type,
                            std::optional<int64_t> elementLimit = std::nullopt);

void addVernonSharedValueTypeConversions(
    TypeConverter &converter,
    std::optional<int64_t> staticTensorElementLimit = std::nullopt);

void populateVernonSharedValuePatterns(TypeConverter &converter,
                                       RewritePatternSet &patterns);

void populateVernonSharedValueStructuralTypeConversions(
    TypeConverter &converter, RewritePatternSet &patterns,
    ConversionTarget &target);

} // namespace vernon
} // namespace mlir
