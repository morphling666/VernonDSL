#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir {
class ConversionTarget;
class TypeConverter;

namespace vernon {

// Adds Pattern<Op> for every op the frontend emits on Tensor values that applies to each element independently.
template <template <typename> class Pattern, typename... Arguments>
void addVernonElementwisePatterns(RewritePatternSet &patterns, Arguments &&...arguments) {
    patterns.add<Pattern<arith::AddFOp>, Pattern<arith::SubFOp>, Pattern<arith::MulFOp>, Pattern<arith::DivFOp>,
                 Pattern<arith::RemFOp>, Pattern<arith::AddIOp>, Pattern<arith::SubIOp>, Pattern<arith::MulIOp>,
                 Pattern<arith::DivSIOp>, Pattern<arith::DivUIOp>, Pattern<arith::FloorDivSIOp>,
                 Pattern<arith::RemSIOp>, Pattern<arith::RemUIOp>, Pattern<arith::AndIOp>, Pattern<arith::OrIOp>,
                 Pattern<arith::XOrIOp>, Pattern<arith::ShLIOp>, Pattern<arith::ShRSIOp>, Pattern<arith::ShRUIOp>,
                 Pattern<arith::CmpFOp>, Pattern<arith::CmpIOp>, Pattern<arith::SelectOp>, Pattern<arith::BitcastOp>,
                 Pattern<arith::SIToFPOp>, Pattern<arith::UIToFPOp>, Pattern<arith::FPToSIOp>, Pattern<arith::FPToUIOp>,
                 Pattern<arith::ExtFOp>, Pattern<arith::TruncFOp>, Pattern<math::AbsFOp>, Pattern<math::AbsIOp>,
                 Pattern<math::AcosOp>, Pattern<math::Atan2Op>, Pattern<math::CosOp>, Pattern<math::ExpOp>,
                 Pattern<math::FloorOp>, Pattern<math::LogOp>, Pattern<math::SinOp>, Pattern<math::SqrtOp>>(
        std::forward<Arguments>(arguments)...);
}

// A missing element limit converts every non-empty static Tensor.
FailureOr<VectorType> convertStaticTensorToVector(Type type, std::optional<int64_t> elementLimit = std::nullopt);

void addVernonSharedValueTypeConversions(TypeConverter &converter,
                                         std::optional<int64_t> staticTensorElementLimit = std::nullopt);

void populateVernonSharedValuePatterns(TypeConverter &converter, RewritePatternSet &patterns);

void populateVernonSharedValueStructuralTypeConversions(TypeConverter &converter, RewritePatternSet &patterns,
                                                        ConversionTarget &target);

} // namespace vernon
} // namespace mlir
