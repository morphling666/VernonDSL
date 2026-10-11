#include "mlir/Dialect/Vernon/Transforms/VernonLowerMulOverflowChecks.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

namespace mlir::vernon {
namespace {

// `(a * b) / a == b` (unsigned, signed truncating or signed floor) and `b <= UMAX / a` all mean "a * b does not
// overflow" for a != 0: a non-overflowing product divides exactly, and a wrapped one is off by a multiple of 2^n
// that no rounding mode can absorb. a == 0 and signed INT_MIN / -1 are undefined, so any result refines them. NVIDIA's
// D3D12 and Vulkan shader compilers fail pipeline creation on these idioms, so the replacement must not divide.
struct OverflowCheck {
    Value a;
    Value b;
    bool fits;
    bool isSigned;
};

arith::CmpIPredicate swapOperands(arith::CmpIPredicate predicate) {
    switch (predicate) {
    case arith::CmpIPredicate::ult:
        return arith::CmpIPredicate::ugt;
    case arith::CmpIPredicate::ule:
        return arith::CmpIPredicate::uge;
    case arith::CmpIPredicate::ugt:
        return arith::CmpIPredicate::ult;
    case arith::CmpIPredicate::uge:
        return arith::CmpIPredicate::ule;
    default:
        return predicate;
    }
}

std::optional<OverflowCheck> matchQuotientEqualsCofactor(Value dividend, Value divisor, Value other,
                                                         arith::CmpIPredicate predicate, bool isSigned) {
    if (predicate != arith::CmpIPredicate::eq && predicate != arith::CmpIPredicate::ne)
        return std::nullopt;
    auto product = dividend.getDefiningOp<arith::MulIOp>();
    if (!product)
        return std::nullopt;
    Value cofactor = product.getLhs() == divisor   ? product.getRhs()
                     : product.getRhs() == divisor ? product.getLhs()
                                                   : Value();
    if (!cofactor || cofactor != other)
        return std::nullopt;
    return OverflowCheck{divisor, other, predicate == arith::CmpIPredicate::eq, isSigned};
}

std::optional<OverflowCheck> matchOriented(Value quotient, Value other, arith::CmpIPredicate predicate) {
    if (auto division = quotient.getDefiningOp<arith::DivSIOp>())
        return matchQuotientEqualsCofactor(division.getLhs(), division.getRhs(), other, predicate, true);
    if (auto division = quotient.getDefiningOp<arith::FloorDivSIOp>())
        return matchQuotientEqualsCofactor(division.getLhs(), division.getRhs(), other, predicate, true);
    auto division = quotient.getDefiningOp<arith::DivUIOp>();
    if (!division)
        return std::nullopt;
    if (std::optional<OverflowCheck> check =
            matchQuotientEqualsCofactor(division.getLhs(), division.getRhs(), other, predicate, false))
        return check;
    if (predicate != arith::CmpIPredicate::uge && predicate != arith::CmpIPredicate::ult)
        return std::nullopt;
    APInt dividend;
    if (!matchPattern(division.getLhs(), m_ConstantInt(&dividend)) || !dividend.isAllOnes())
        return std::nullopt;
    return OverflowCheck{division.getRhs(), other, predicate == arith::CmpIPredicate::uge, false};
}

std::optional<OverflowCheck> matchOverflowCheck(arith::CmpIOp compare) {
    if (std::optional<OverflowCheck> check = matchOriented(compare.getLhs(), compare.getRhs(), compare.getPredicate()))
        return check;
    return matchOriented(compare.getRhs(), compare.getLhs(), swapOperands(compare.getPredicate()));
}

Value constantLike(PatternRewriter &rewriter, Location location, Type type, const APInt &value) {
    TypedAttr element = rewriter.getIntegerAttr(getElementTypeOrSelf(type), value);
    if (auto shaped = dyn_cast<ShapedType>(type))
        element = DenseElementsAttr::get(shaped, ArrayRef<Attribute>{element});
    return arith::ConstantOp::create(rewriter, location, element);
}

class OverflowBuilder {
public:
    OverflowBuilder(PatternRewriter &rewriter, Location location, Type type, unsigned width)
        : rewriter(rewriter), location(location), type(type), width(width) {}

    Value constant(const APInt &value) const { return constantLike(rewriter, location, type, value); }

    Value compare(arith::CmpIPredicate predicate, Value left, Value right) const {
        return arith::CmpIOp::create(rewriter, location, predicate, left, right);
    }

    // True when the unsigned product a * b is representable in `width` bits.
    Value unsignedFits(Value a, Value b) const {
        unsigned half = width / 2;
        Value shift = constant(APInt(width, half));
        Value mask = constant(APInt::getLowBitsSet(width, half));
        Value zero = constant(APInt::getZero(width));
        auto high = [&](Value value) { return arith::ShRUIOp::create(rewriter, location, value, shift).getResult(); };
        auto low = [&](Value value) { return arith::AndIOp::create(rewriter, location, value, mask).getResult(); };
        auto multiply = [&](Value left, Value right) {
            return arith::MulIOp::create(rewriter, location, left, right).getResult();
        };
        auto add = [&](Value left, Value right) {
            return arith::AddIOp::create(rewriter, location, left, right).getResult();
        };
        Value aHigh = high(a);
        Value aLow = low(a);
        Value bHigh = high(b);
        Value bLow = low(b);
        // At most one cross term is nonzero whenever aHigh or bHigh is zero, so `cross` cannot wrap on the
        // path where its high half decides the result.
        Value cross = add(add(multiply(aLow, bHigh), multiply(aHigh, bLow)), high(multiply(aLow, bLow)));
        Value highHalfZero = arith::OrIOp::create(rewriter, location, compare(arith::CmpIPredicate::eq, aHigh, zero),
                                                  compare(arith::CmpIPredicate::eq, bHigh, zero));
        return arith::AndIOp::create(rewriter, location, highHalfZero,
                                     compare(arith::CmpIPredicate::eq, high(cross), zero));
    }

    // True when the signed product a * b is representable: |a| * |b| must fit unsigned and stay within
    // SMAX, or SMAX + 1 when the signs differ.
    Value signedFits(Value a, Value b) const {
        Value zero = constant(APInt::getZero(width));
        Value aNegative = compare(arith::CmpIPredicate::slt, a, zero);
        Value bNegative = compare(arith::CmpIPredicate::slt, b, zero);
        auto magnitude = [&](Value value, Value negative) {
            return arith::SelectOp::create(rewriter, location, negative,
                                           arith::SubIOp::create(rewriter, location, zero, value), value)
                .getResult();
        };
        Value aMagnitude = magnitude(a, aNegative);
        Value bMagnitude = magnitude(b, bNegative);
        Value limit = arith::SelectOp::create(
            rewriter, location, arith::XOrIOp::create(rewriter, location, aNegative, bNegative),
            constant(APInt::getSignedMinValue(width)), constant(APInt::getSignedMaxValue(width)));
        Value product = arith::MulIOp::create(rewriter, location, aMagnitude, bMagnitude);
        return arith::AndIOp::create(rewriter, location, unsignedFits(aMagnitude, bMagnitude),
                                     compare(arith::CmpIPredicate::ule, product, limit));
    }

private:
    PatternRewriter &rewriter;
    Location location;
    Type type;
    unsigned width;
};

struct MulOverflowCheckPattern final : OpRewritePattern<arith::CmpIOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::CmpIOp compare, PatternRewriter &rewriter) const override {
        std::optional<OverflowCheck> check = matchOverflowCheck(compare);
        if (!check)
            return failure();
        auto element = dyn_cast<IntegerType>(getElementTypeOrSelf(check->a.getType()));
        if (!element || element.getWidth() < 2 || element.getWidth() % 2 != 0)
            return failure();
        Location location = compare.getLoc();
        OverflowBuilder builder(rewriter, location, check->a.getType(), element.getWidth());
        Value fits =
            check->isSigned ? builder.signedFits(check->a, check->b) : builder.unsignedFits(check->a, check->b);
        if (!check->fits)
            fits = arith::XOrIOp::create(rewriter, location, fits,
                                         constantLike(rewriter, location, compare.getType(), APInt::getAllOnes(1)));
        rewriter.replaceOp(compare, fits);
        return success();
    }
};

struct VernonLowerMulOverflowChecksPass final : PassWrapper<VernonLowerMulOverflowChecksPass, OperationPass<>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerMulOverflowChecksPass)

    StringRef getArgument() const final { return "vernon-lower-mul-overflow-checks"; }
    StringRef getDescription() const final {
        return "Rewrite division-based multiply overflow checks into division-free arithmetic";
    }

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<arith::ArithDialect>(); }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<MulOverflowCheckPattern>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerMulOverflowChecksPass() {
    return std::make_unique<VernonLowerMulOverflowChecksPass>();
}

void registerVernonLowerMulOverflowChecksPass() { PassRegistration<VernonLowerMulOverflowChecksPass>(); }

} // namespace mlir::vernon
