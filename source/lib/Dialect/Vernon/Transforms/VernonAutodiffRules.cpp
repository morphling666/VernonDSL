#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffUtils.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace mlir::vernon {
namespace {

using Requirement = AutodiffPrimalRequirement;

std::string getRuleKey(Operation *operation) {
    if (auto intrinsic = dyn_cast<IntrinsicOp>(operation))
        return (IntrinsicOp::getOperationName() + Twine(".") + intrinsic.getName()).str();
    return operation->getName().getStringRef().str();
}

Value castPrimal(OpBuilder &builder, Location location, Value value, Type targetType) {
    if (!value)
        return {};
    if (value.getType() == targetType)
        return value;
    Type sourceElement = getElementTypeOrSelf(value.getType());
    Type targetElement = getElementTypeOrSelf(targetType);
    if (sourceElement.isF16() && targetElement.isF32())
        return arith::ExtFOp::create(builder, location, targetType, value);
    return {};
}

Value constant(OpBuilder &builder, Location location, Type type, double value) {
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        auto element = cast<FloatType>(tensor.getElementType());
        return arith::ConstantOp::create(builder, location,
                                         DenseElementsAttr::get(tensor, builder.getFloatAttr(element, value)));
    }
    return arith::ConstantOp::create(builder, location, builder.getFloatAttr(type, value));
}

Value negative(OpBuilder &builder, Location location, Value value) {
    Value zero = constant(builder, location, value.getType(), 0.0);
    return arith::SubFOp::create(builder, location, zero, value);
}

LogicalResult requireValues(Operation *operation, ArrayRef<Value> values) {
    if (llvm::all_of(values, [](Value value) { return static_cast<bool>(value); }))
        return success();
    return operation->emitOpError("autodiff rule cannot promote a required primal to the cotangent type");
}

DifferentiationRule makeRule(StringRef name, unsigned operands, SmallVector<Requirement> requirements,
                             AutodiffVjpBuilder builder) {
    return DifferentiationRule(name.str(), operands, 1, std::move(requirements), std::move(builder));
}

std::string intrinsicRuleKey(StringRef name) { return (Twine(IntrinsicOp::getOperationName()) + "." + name).str(); }

Value createIntrinsic(OpBuilder &builder, Location location, StringRef name, ValueRange operands, Type resultType) {
    OperationState state(location, IntrinsicOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(resultType);
    state.addAttribute("name", builder.getStringAttr(name));
    return builder.create(state)->getResult(0);
}

Value splat(OpBuilder &builder, Location location, Value scalar, RankedTensorType tensorType) {
    return tensor::SplatOp::create(builder, location, tensorType, scalar);
}

Value scale(OpBuilder &builder, Location location, Value tensorValue, Value scalar) {
    auto type = cast<RankedTensorType>(tensorValue.getType());
    return arith::MulFOp::create(builder, location, tensorValue, splat(builder, location, scalar, type));
}

bool hasDynamicRankedTensorShape(Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && !tensor.hasStaticShape();
}

LogicalResult verifyFloatingIntrinsic(Operation *operation) {
    if (!isa<IntrinsicOp>(operation) || operation->getNumResults() != 1)
        return operation->emitOpError("autodiff intrinsic rule requires one result");
    // Static-shape rejection must win over "not a differentiable value": dynamic
    // ranked tensors fail getAutodiffDerivativeValueType, but the rule diagnostic
    // is that autodiff needs static Tensor shapes.
    if (llvm::any_of(operation->getOperandTypes(), hasDynamicRankedTensorShape) ||
        hasDynamicRankedTensorShape(operation->getResult(0).getType()))
        return operation->emitOpError("autodiff intrinsic rule requires static Tensor shapes");
    if (failed(getAutodiffDerivativeValueType(operation->getResult(0).getType())) ||
        llvm::any_of(operation->getOperandTypes(),
                     [](Type type) { return failed(getAutodiffDerivativeValueType(type)); }))
        return operation->emitOpError("autodiff intrinsic rule requires floating-point scalar or ranked Tensor values");
    return success();
}

Value extract(OpBuilder &builder, Location location, Value value, ArrayRef<int64_t> coordinate) {
    if (!isa<RankedTensorType>(value.getType())) {
        assert(coordinate.empty());
        return value;
    }
    SmallVector<Value> indices;
    for (int64_t index : coordinate)
        indices.push_back(arith::ConstantIndexOp::create(builder, location, index));
    return tensor::ExtractOp::create(builder, location, value, indices);
}

FailureOr<SmallVector<int64_t>> broadcastBatchShape(ArrayRef<int64_t> left, ArrayRef<int64_t> right) {
    unsigned rank = std::max(left.size(), right.size());
    SmallVector<int64_t> result(rank, 1);
    for (unsigned offset = 0; offset < rank; ++offset) {
        int64_t leftExtent = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        int64_t rightExtent = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (leftExtent != rightExtent && leftExtent != 1 && rightExtent != 1)
            return failure();
        result[rank - 1 - offset] = std::max(leftExtent, rightExtent);
    }
    return result;
}

FailureOr<size_t> checkedElementCount(ArrayRef<int64_t> shape) {
    size_t count = 1;
    for (int64_t extent : shape) {
        if (extent < 0)
            return failure();
        size_t unsignedExtent = static_cast<size_t>(extent);
        if (unsignedExtent != 0 && count > std::numeric_limits<size_t>::max() / unsignedExtent)
            return failure();
        count *= unsignedExtent;
    }
    return count;
}

SmallVector<int64_t> broadcastCoordinate(ArrayRef<int64_t> coordinate, ArrayRef<int64_t> sourceShape) {
    SmallVector<int64_t> result;
    unsigned offset = coordinate.size() - sourceShape.size();
    for (auto [index, extent] : llvm::enumerate(sourceShape))
        result.push_back(extent == 1 ? 0 : coordinate[offset + index]);
    return result;
}

LogicalResult buildMatmulVjp(Operation *operation, const AutodiffVjpBuildContext &context,
                             SmallVectorImpl<Value> &results) {
    Value seed = context.resultCotangents[0];
    FailureOr<Type> leftDerivativeType = getAutodiffDerivativeValueType(operation->getOperand(0).getType());
    FailureOr<Type> rightDerivativeType = getAutodiffDerivativeValueType(operation->getOperand(1).getType());
    if (failed(leftDerivativeType) || failed(rightDerivativeType))
        return operation->emitOpError("matmul VJP requires floating-point ranked Tensor operands");
    auto leftType = dyn_cast<RankedTensorType>(*leftDerivativeType);
    auto rightType = dyn_cast<RankedTensorType>(*rightDerivativeType);
    if (!leftType || !rightType || !leftType.hasStaticShape() || !rightType.hasStaticShape() ||
        leftType.getRank() < 1 || rightType.getRank() < 1)
        return operation->emitOpError("matmul VJP requires static ranked Tensor operands");
    Value left = castPrimal(context.builder, context.location, context.getPrimalOperand(0), leftType);
    Value right = castPrimal(context.builder, context.location, context.getPrimalOperand(1), rightType);
    if (failed(requireValues(operation, {left, right})))
        return failure();

    ArrayRef<int64_t> leftShape = leftType.getShape();
    ArrayRef<int64_t> rightShape = rightType.getShape();
    bool leftVector = leftType.getRank() == 1;
    bool rightVector = rightType.getRank() == 1;
    ArrayRef<int64_t> leftBatch = leftVector ? ArrayRef<int64_t>() : leftShape.drop_back(2);
    ArrayRef<int64_t> rightBatch = rightVector ? ArrayRef<int64_t>() : rightShape.drop_back(2);
    FailureOr<SmallVector<int64_t>> batchShape = broadcastBatchShape(leftBatch, rightBatch);
    int64_t reduction = leftShape.back();
    int64_t rightReduction = rightVector ? rightShape.front() : rightShape[rightShape.size() - 2];
    FailureOr<size_t> leftElementCount = checkedElementCount(leftShape);
    FailureOr<size_t> rightElementCount = checkedElementCount(rightShape);
    if (failed(batchShape) || failed(leftElementCount) || failed(rightElementCount) || reduction != rightReduction)
        return operation->emitOpError("matmul VJP received incompatible operand shapes");
    FailureOr<size_t> batchElementCount = checkedElementCount(*batchShape);
    if (failed(batchElementCount))
        return operation->emitOpError("matmul VJP batch shape is too large");
    int64_t rows = leftVector ? 1 : leftShape[leftShape.size() - 2];
    int64_t columns = rightVector ? 1 : rightShape.back();
    SmallVector<int64_t> expectedResultShape(*batchShape);
    if (!leftVector)
        expectedResultShape.push_back(rows);
    if (!rightVector)
        expectedResultShape.push_back(columns);
    if (failed(checkedElementCount(expectedResultShape)))
        return operation->emitOpError("matmul VJP result shape is too large");
    FailureOr<Type> resultDerivativeType = getAutodiffDerivativeValueType(operation->getResult(0).getType());
    if (failed(resultDerivativeType))
        return operation->emitOpError("matmul VJP requires a differentiable result");
    Type expectedResultType =
        expectedResultShape.empty()
            ? leftType.getElementType()
            : static_cast<Type>(RankedTensorType::get(expectedResultShape, leftType.getElementType()));
    if (*resultDerivativeType != expectedResultType || seed.getType() != expectedResultType ||
        leftType.getElementType() != rightType.getElementType())
        return operation->emitOpError("matmul VJP result type does not match the operand shapes");

    SmallVector<SmallVector<Value>> leftTerms(*leftElementCount);
    SmallVector<SmallVector<Value>> rightTerms(*rightElementCount);
    auto flatIndex = [](ArrayRef<int64_t> coordinate, ArrayRef<int64_t> shape) {
        size_t result = 0;
        for (auto [index, value] : llvm::enumerate(coordinate))
            result = result * static_cast<size_t>(shape[index]) + static_cast<size_t>(value);
        return result;
    };
    SmallVector<SmallVector<int64_t>> batches;
    if (rows != 0 && columns != 0)
        batches = enumerateStaticCoordinates(*batchShape);
    for (const SmallVector<int64_t> &batch : batches) {
        SmallVector<int64_t> leftPrefix = broadcastCoordinate(batch, leftBatch);
        SmallVector<int64_t> rightPrefix = broadcastCoordinate(batch, rightBatch);
        for (int64_t row = 0; row < rows; ++row)
            for (int64_t column = 0; column < columns; ++column) {
                SmallVector<int64_t> outputCoordinate(batch);
                if (!leftVector)
                    outputCoordinate.push_back(row);
                if (!rightVector)
                    outputCoordinate.push_back(column);
                Value outputSeed = extract(context.builder, context.location, seed, outputCoordinate);
                for (int64_t inner = 0; inner < reduction; ++inner) {
                    SmallVector<int64_t> leftCoordinate = leftVector ? SmallVector<int64_t>{inner} : leftPrefix;
                    if (!leftVector)
                        leftCoordinate.append({row, inner});
                    SmallVector<int64_t> rightCoordinate = rightVector ? SmallVector<int64_t>{inner} : rightPrefix;
                    if (!rightVector)
                        rightCoordinate.append({inner, column});
                    Value leftValue = extract(context.builder, context.location, left, leftCoordinate);
                    Value rightValue = extract(context.builder, context.location, right, rightCoordinate);
                    leftTerms[flatIndex(leftCoordinate, leftShape)].push_back(
                        arith::MulFOp::create(context.builder, context.location, outputSeed, rightValue));
                    rightTerms[flatIndex(rightCoordinate, rightShape)].push_back(
                        arith::MulFOp::create(context.builder, context.location, outputSeed, leftValue));
                }
            }
    }
    auto constructGradient = [&](ArrayRef<SmallVector<Value>> terms, RankedTensorType type) {
        SmallVector<Value> elements;
        for (const SmallVector<Value> &contributions : terms) {
            Value sum = contributions.empty() ? constant(context.builder, context.location, type.getElementType(), 0.0)
                                              : contributions.front();
            for (Value contribution : ArrayRef<Value>(contributions).drop_front())
                sum = arith::AddFOp::create(context.builder, context.location, sum, contribution);
            elements.push_back(sum);
        }
        return createIntrinsic(context.builder, context.location, "construct", elements, type);
    };
    results.push_back(constructGradient(leftTerms, leftType));
    results.push_back(constructGradient(rightTerms, rightType));
    return success();
}

LogicalResult buildPowVjp(Operation *operation, const AutodiffVjpBuildContext &context,
                          SmallVectorImpl<Value> &results) {
    Value seed = context.resultCotangents[0];
    Value base = castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
    Value exponent = castPrimal(context.builder, context.location, context.getPrimalOperand(1), seed.getType());
    Value output = castPrimal(context.builder, context.location, context.getPrimalResult(0), seed.getType());
    if (failed(requireValues(operation, {base, exponent, output})))
        return failure();
    Value one = constant(context.builder, context.location, seed.getType(), 1.0);
    Value exponentMinusOne = arith::SubFOp::create(context.builder, context.location, exponent, one);
    Value power = createIntrinsic(context.builder, context.location, "pow", {base, exponentMinusOne}, seed.getType());
    Value baseContribution =
        arith::MulFOp::create(context.builder, context.location,
                              arith::MulFOp::create(context.builder, context.location, seed, exponent), power);
    Value exponentContribution = arith::MulFOp::create(
        context.builder, context.location, arith::MulFOp::create(context.builder, context.location, seed, output),
        math::LogOp::create(context.builder, context.location, base));
    results.append({baseContribution, exponentContribution});
    return success();
}

LogicalResult buildConstructVjp(Operation *operation, const AutodiffVjpBuildContext &context,
                                SmallVectorImpl<Value> &results) {
    auto resultType = cast<RankedTensorType>(operation->getResult(0).getType());
    for (const SmallVector<int64_t> &coordinate : enumerateStaticCoordinates(resultType.getShape()))
        results.push_back(extract(context.builder, context.location, context.resultCotangents[0], coordinate));
    return success();
}

LogicalResult verifyElementwiseOperation(Operation *operation, unsigned operandCount, unsigned resultCount) {
    if (operation->getNumOperands() != operandCount || operation->getNumResults() != resultCount)
        return operation->emitOpError() << "autodiff rule expects " << operandCount << " operand(s) and " << resultCount
                                        << " result(s)";
    if (resultCount != 1 || failed(getAutodiffDerivativeValueType(operation->getResult(0).getType())))
        return operation->emitOpError("autodiff rule requires one floating-point scalar or ranked Tensor result");
    Type resultType = operation->getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(resultType); tensor && !tensor.hasStaticShape())
        return operation->emitOpError("autodiff elementwise rule requires a static Tensor shape");
    if (llvm::any_of(operation->getOperandTypes(), [&](Type type) { return type != resultType; }))
        return operation->emitOpError("autodiff rule requires operands matching the result type");
    return success();
}

bool isStructuralAutodiffOperation(Operation *operation) {
    return isa<scf::IfOp, scf::ForOp, scf::WhileOp, StructGetOp, TupleGetOp, StructCreateOp, TupleCreateOp, LoadOp,
               StoreOp, ReduceSumOp, ScatterAddOp, AtomicOp>(operation);
}

} // namespace

DifferentiationRule::DifferentiationRule(std::string registryKey, std::optional<unsigned> operandCount,
                                         unsigned resultCount,
                                         SmallVector<AutodiffPrimalRequirement> vjpPrimalRequirements,
                                         AutodiffVjpBuilder vjpBuilder, AutodiffJvpBuilder jvpBuilder,
                                         AutodiffRuleVerifier verifier)
    : registryKey(std::move(registryKey)), operandCount(operandCount), resultCount(resultCount),
      vjpPrimalRequirements(std::move(vjpPrimalRequirements)), vjpBuilder(std::move(vjpBuilder)),
      jvpBuilder(std::move(jvpBuilder)), verifier(std::move(verifier)) {}

bool AutodiffPrimalRequirement::isRequiredFor(ArrayRef<unsigned> activeOperands) const {
    if (contributionOperands.empty() || activeOperands.empty())
        return true;
    return llvm::any_of(contributionOperands,
                        [&](unsigned operand) { return llvm::is_contained(activeOperands, operand); });
}

LogicalResult DifferentiationRule::verifyCompatibility(Operation *operation) const {
    if (getRuleKey(operation) != registryKey)
        return operation->emitOpError() << "was passed to the autodiff rule for '" << registryKey << "'";
    if ((operandCount && operation->getNumOperands() != *operandCount) || operation->getNumResults() != resultCount)
        return operation->emitOpError("does not match the registered autodiff rule signature");
    if (!verifier && !operandCount)
        return operation->emitOpError("variadic autodiff rule requires an explicit verifier");
    if (verifier ? failed(verifier(operation))
                 : failed(verifyElementwiseOperation(operation, *operandCount, resultCount)))
        return failure();
    for (const AutodiffPrimalRequirement &requirement : vjpPrimalRequirements) {
        unsigned extent = requirement.kind == AutodiffPrimalKind::Operand ? operation->getNumOperands() : resultCount;
        if (requirement.index >= extent)
            return operation->emitOpError("autodiff rule contains an out-of-range primal requirement");
    }
    return success();
}

FailureOr<AutodiffRuleActivity> DifferentiationRule::classifyActivity(Operation *operation) const {
    if (getRuleKey(operation) != registryKey || (operandCount && operation->getNumOperands() != *operandCount) ||
        operation->getNumResults() != resultCount)
        return failure();
    AutodiffRuleActivity activity;
    activity.operandIndices.resize(operation->getNumOperands());
    std::iota(activity.operandIndices.begin(), activity.operandIndices.end(), 0u);
    activity.resultIndices.resize(resultCount);
    std::iota(activity.resultIndices.begin(), activity.resultIndices.end(), 0u);
    return activity;
}

FailureOr<SmallVector<Value>> DifferentiationRule::buildVjp(Operation *operation,
                                                            const AutodiffVjpBuildContext &context) const {
    if (failed(verifyCompatibility(operation)))
        return failure();
    if (!vjpBuilder)
        return operation->emitOpError("has no VJP builder");
    if (resultCount != 1)
        return operation->emitOpError("VJP rules currently require exactly one result");
    if (context.primalOperands.size() != operation->getNumOperands() || context.primalResults.size() != resultCount ||
        context.resultCotangents.size() != resultCount)
        return operation->emitOpError("autodiff VJP provider does not match the rule signature");
    FailureOr<Type> derivativeType = getAutodiffDerivativeValueType(operation->getResult(0).getType());
    if (failed(derivativeType) || context.resultCotangents.front().getType() != *derivativeType)
        return operation->emitOpError("autodiff VJP cotangent has an incompatible derivative type");
    for (const AutodiffPrimalRequirement &requirement : vjpPrimalRequirements) {
        if (!requirement.isRequiredFor(context.activeOperandIndices))
            continue;
        Value primal = requirement.kind == AutodiffPrimalKind::Operand ? context.getPrimalOperand(requirement.index)
                                                                       : context.getPrimalResult(requirement.index);
        if (!primal)
            return operation->emitOpError("autodiff VJP provider omitted a declared primal requirement");
    }
    SmallVector<Value> contributions;
    if (failed(vjpBuilder(operation, context, contributions)))
        return failure();
    if (contributions.size() != operation->getNumOperands())
        return operation->emitOpError("autodiff VJP builder returned the wrong number of operand contributions");
    for (unsigned operandIndex = 0; operandIndex < contributions.size(); ++operandIndex) {
        Value contribution = contributions[operandIndex];
        Value operand = operation->getOperand(operandIndex);
        FailureOr<Type> expectedType = getAutodiffDerivativeValueType(operand.getType());
        if (failed(expectedType)) {
            if (contribution)
                return operation->emitOpError(
                    "autodiff VJP builder returned a contribution for a non-differentiable operand");
            continue;
        }
        if ((!contribution && context.isOperandActive(operandIndex)) ||
            (contribution && contribution.getType() != *expectedType))
            return operation->emitOpError("autodiff VJP builder returned an incompatible contribution type");
    }
    return contributions;
}

FailureOr<SmallVector<Value>> DifferentiationRule::buildJvp(Operation *operation,
                                                            const AutodiffJvpBuildContext &context) const {
    if (failed(verifyCompatibility(operation)))
        return failure();
    if (!jvpBuilder)
        return operation->emitOpError("has no JVP builder yet");
    if (context.primalOperands.size() != operation->getNumOperands() ||
        context.primalResults.size() != operation->getNumResults() ||
        context.tangentOperands.size() != operation->getNumOperands())
        return operation->emitOpError("autodiff JVP provider does not match the rule signature");
    for (auto [tangent, operand] : llvm::zip_equal(context.tangentOperands, operation->getOperands())) {
        FailureOr<Type> expectedType = getAutodiffDerivativeValueType(operand.getType());
        if (failed(expectedType) || tangent.getType() != *expectedType)
            return operation->emitOpError("autodiff JVP operand tangent has an incompatible type");
    }
    SmallVector<Value> tangents;
    if (failed(jvpBuilder(operation, context, tangents)))
        return failure();
    if (tangents.size() != resultCount)
        return operation->emitOpError("autodiff JVP builder returned the wrong number of result tangents");
    for (auto [tangent, result] : llvm::zip_equal(tangents, operation->getResults())) {
        FailureOr<Type> expectedType = getAutodiffDerivativeValueType(result.getType());
        if (failed(expectedType) || !tangent || tangent.getType() != *expectedType)
            return operation->emitOpError("autodiff JVP builder returned an incompatible tangent type");
    }
    return tangents;
}

LogicalResult VernonAutodiffRuleRegistry::registerRule(DifferentiationRule rule) {
    StringRef key = rule.getRegistryKey();
    if (key.empty() || rules.contains(key))
        return failure();
    rules.try_emplace(key, std::move(rule));
    return success();
}

const DifferentiationRule *VernonAutodiffRuleRegistry::lookup(StringRef registryKey) const {
    auto found = rules.find(registryKey);
    return found == rules.end() ? nullptr : &found->second;
}

const DifferentiationRule *VernonAutodiffRuleRegistry::lookup(Operation *operation) const {
    return lookup(getRuleKey(operation));
}

SmallVector<StringRef> VernonAutodiffRuleRegistry::getRegisteredKeys() const {
    SmallVector<StringRef> names;
    names.reserve(rules.size());
    for (const auto &entry : rules)
        names.push_back(entry.getKey());
    llvm::sort(names);
    return names;
}

VernonAutodiffRuleRegistry createDefaultAutodiffRuleRegistry() {
    VernonAutodiffRuleRegistry registry;
    auto add = [&](DifferentiationRule rule) {
        LogicalResult status = registry.registerRule(std::move(rule));
        assert(succeeded(status) && "duplicate default autodiff rule");
        (void)status;
    };

    add(makeRule(arith::AddFOp::getOperationName(), 2, {},
                 [](Operation *, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     results.append(2, context.resultCotangents[0]);
                     return success();
                 }));
    add(makeRule(arith::SubFOp::getOperationName(), 2, {},
                 [](Operation *, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     results.append({seed, negative(context.builder, context.location, seed)});
                     return success();
                 }));
    add(makeRule(arith::MulFOp::getOperationName(), 2, {Requirement::operand(0, {1}), Requirement::operand(1, {0})},
                 [](Operation *operation, const AutodiffVjpBuildContext &context,
                    SmallVectorImpl<Value> &results) -> LogicalResult {
                     Value seed = context.resultCotangents[0];
                     Value left;
                     Value right;
                     if (context.isOperandActive(1))
                         left =
                             castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
                     if (context.isOperandActive(0))
                         right =
                             castPrimal(context.builder, context.location, context.getPrimalOperand(1), seed.getType());
                     if ((context.isOperandActive(1) && !left) || (context.isOperandActive(0) && !right))
                         return operation->emitOpError("autodiff multiplication rule omitted a required primal");
                     results.push_back(context.isOperandActive(0)
                                           ? arith::MulFOp::create(context.builder, context.location, seed, right)
                                           : Value{});
                     results.push_back(context.isOperandActive(1)
                                           ? arith::MulFOp::create(context.builder, context.location, seed, left)
                                           : Value{});
                     return success();
                 }));
    add(makeRule(arith::DivFOp::getOperationName(), 2, {Requirement::operand(0), Requirement::operand(1)},
                 [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     Value numerator =
                         castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
                     Value denominator =
                         castPrimal(context.builder, context.location, context.getPrimalOperand(1), seed.getType());
                     if (failed(requireValues(operation, {numerator, denominator})))
                         return failure();
                     results.push_back(arith::DivFOp::create(context.builder, context.location, seed, denominator));
                     Value square = arith::MulFOp::create(context.builder, context.location, denominator, denominator);
                     Value quotient = arith::DivFOp::create(
                         context.builder, context.location,
                         arith::MulFOp::create(context.builder, context.location, seed, numerator), square);
                     results.push_back(negative(context.builder, context.location, quotient));
                     return success();
                 }));
    add(makeRule(arith::NegFOp::getOperationName(), 1, {},
                 [](Operation *, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     results.push_back(negative(context.builder, context.location, context.resultCotangents[0]));
                     return success();
                 }));
    add(DifferentiationRule(
        arith::ExtFOp::getOperationName().str(), 1, 1, {},
        [](Operation *, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
            results.push_back(context.resultCotangents[0]);
            return success();
        },
        {},
        [](Operation *operation) -> LogicalResult {
            auto extension = dyn_cast<arith::ExtFOp>(operation);
            if (!extension)
                return operation->emitOpError("autodiff extension rule requires arith.extf");
            Type source = extension.getIn().getType();
            Type result = extension.getOut().getType();
            FailureOr<Type> sourceDerivative = getAutodiffDerivativeValueType(source);
            FailureOr<Type> resultDerivative = getAutodiffDerivativeValueType(result);
            if (failed(sourceDerivative) || failed(resultDerivative) || *sourceDerivative != *resultDerivative ||
                !getElementTypeOrSelf(source).isF16() || !getElementTypeOrSelf(result).isF32())
                return operation->emitOpError("autodiff extension rule supports only f16-to-f32 promotion");
            return success();
        }));
    add(DifferentiationRule(
        intrinsicRuleKey("clamp"), 3, 1, {Requirement::operand(0), Requirement::operand(1), Requirement::operand(2)},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            Value seed = context.resultCotangents[0];
            Value value = context.getPrimalOperand(0);
            Value lower = context.getPrimalOperand(1);
            Value upper = context.getPrimalOperand(2);
            if (failed(requireValues(operation, {value, lower, upper})))
                return failure();
            Value aboveLower =
                arith::CmpFOp::create(context.builder, context.location, arith::CmpFPredicate::OGT, value, lower);
            Value belowUpper =
                arith::CmpFOp::create(context.builder, context.location, arith::CmpFPredicate::OLT, value, upper);
            Value interior = arith::AndIOp::create(context.builder, context.location, aboveLower, belowUpper);
            Value belowLower =
                arith::CmpFOp::create(context.builder, context.location, arith::CmpFPredicate::OLT, value, lower);
            Value aboveUpper =
                arith::CmpFOp::create(context.builder, context.location, arith::CmpFPredicate::OGT, value, upper);
            Value zero = constant(context.builder, context.location, seed.getType(), 0.0);
            results.append({arith::SelectOp::create(context.builder, context.location, interior, seed, zero),
                            arith::SelectOp::create(context.builder, context.location, belowLower, seed, zero),
                            arith::SelectOp::create(context.builder, context.location, aboveUpper, seed, zero)});
            return success();
        },
        {},
        [](Operation *operation) -> LogicalResult {
            if (!isa<IntrinsicOp>(operation) || operation->getNumOperands() != 3 || operation->getNumResults() != 1 ||
                !isa<FloatType>(operation->getResult(0).getType()) ||
                llvm::any_of(operation->getOperandTypes(),
                             [&](Type type) { return type != operation->getResult(0).getType(); }))
                return operation->emitOpError("clamp VJP requires three matching floating scalar operands");
            return success();
        }));
    add(makeRule(math::FloorOp::getOperationName(), 1, {},
                 [](Operation *, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     results.push_back(
                         constant(context.builder, context.location, context.resultCotangents[0].getType(), 0.0));
                     return success();
                 }));
    add(makeRule(math::SinOp::getOperationName(), 1, {Requirement::operand(0)},
                 [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     Value input =
                         castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
                     if (failed(requireValues(operation, {input})))
                         return failure();
                     Value factor = math::CosOp::create(context.builder, context.location, input);
                     results.push_back(arith::MulFOp::create(context.builder, context.location, seed, factor));
                     return success();
                 }));
    add(makeRule(math::CosOp::getOperationName(), 1, {Requirement::operand(0)},
                 [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     Value input =
                         castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
                     if (failed(requireValues(operation, {input})))
                         return failure();
                     Value sine = math::SinOp::create(context.builder, context.location, input);
                     Value contribution = arith::MulFOp::create(context.builder, context.location, seed, sine);
                     results.push_back(negative(context.builder, context.location, contribution));
                     return success();
                 }));
    add(makeRule(math::ExpOp::getOperationName(), 1, {Requirement::result(0)},
                 [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     Value output =
                         castPrimal(context.builder, context.location, context.getPrimalResult(0), seed.getType());
                     if (failed(requireValues(operation, {output})))
                         return failure();
                     results.push_back(arith::MulFOp::create(context.builder, context.location, seed, output));
                     return success();
                 }));
    add(makeRule(math::LogOp::getOperationName(), 1, {Requirement::operand(0)},
                 [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     Value input =
                         castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
                     if (failed(requireValues(operation, {input})))
                         return failure();
                     results.push_back(arith::DivFOp::create(context.builder, context.location, seed, input));
                     return success();
                 }));
    add(makeRule(math::SqrtOp::getOperationName(), 1, {Requirement::result(0)},
                 [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     Value output =
                         castPrimal(context.builder, context.location, context.getPrimalResult(0), seed.getType());
                     if (failed(requireValues(operation, {output})))
                         return failure();
                     Value half = constant(context.builder, context.location, seed.getType(), 0.5);
                     Value factor = arith::DivFOp::create(context.builder, context.location, half, output);
                     results.push_back(arith::MulFOp::create(context.builder, context.location, seed, factor));
                     return success();
                 }));
    add(makeRule(math::AcosOp::getOperationName(), 1, {Requirement::operand(0)},
                 [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
                     Value seed = context.resultCotangents[0];
                     Value input =
                         castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
                     if (failed(requireValues(operation, {input})))
                         return failure();
                     Value one = constant(context.builder, context.location, seed.getType(), 1.0);
                     Value square = arith::MulFOp::create(context.builder, context.location, input, input);
                     Value radicand = arith::SubFOp::create(context.builder, context.location, one, square);
                     Value root = math::SqrtOp::create(context.builder, context.location, radicand);
                     Value quotient = arith::DivFOp::create(context.builder, context.location, seed, root);
                     results.push_back(negative(context.builder, context.location, quotient));
                     return success();
                 }));
    add(makeRule(
        math::Atan2Op::getOperationName(), 2, {Requirement::operand(0), Requirement::operand(1)},
        [](Operation *operation, const AutodiffVjpBuildContext &context, SmallVectorImpl<Value> &results) {
            Value seed = context.resultCotangents[0];
            Value y = castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
            Value x = castPrimal(context.builder, context.location, context.getPrimalOperand(1), seed.getType());
            if (failed(requireValues(operation, {y, x})))
                return failure();
            Value denominator = arith::AddFOp::create(context.builder, context.location,
                                                      arith::MulFOp::create(context.builder, context.location, y, y),
                                                      arith::MulFOp::create(context.builder, context.location, x, x));
            Value dy =
                arith::DivFOp::create(context.builder, context.location,
                                      arith::MulFOp::create(context.builder, context.location, seed, x), denominator);
            Value dx =
                arith::DivFOp::create(context.builder, context.location,
                                      arith::MulFOp::create(context.builder, context.location, seed, y), denominator);
            results.append({dy, negative(context.builder, context.location, dx)});
            return success();
        }));
    add(makeRule(
        math::AbsFOp::getOperationName(), 1, {Requirement::operand(0)},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            Value seed = context.resultCotangents[0];
            Value input = castPrimal(context.builder, context.location, context.getPrimalOperand(0), seed.getType());
            if (failed(requireValues(operation, {input})))
                return failure();
            Value zero = constant(context.builder, context.location, seed.getType(), 0.0);
            Value one = constant(context.builder, context.location, seed.getType(), 1.0);
            Value minusOne = constant(context.builder, context.location, seed.getType(), -1.0);
            Value positive =
                arith::CmpFOp::create(context.builder, context.location, arith::CmpFPredicate::OGT, input, zero);
            Value negativeInput =
                arith::CmpFOp::create(context.builder, context.location, arith::CmpFPredicate::OLT, input, zero);
            Value nonPositive =
                arith::SelectOp::create(context.builder, context.location, negativeInput, minusOne, zero);
            Value sign = arith::SelectOp::create(context.builder, context.location, positive, one, nonPositive);
            results.push_back(arith::MulFOp::create(context.builder, context.location, seed, sign));
            return success();
        }));
    add(DifferentiationRule(intrinsicRuleKey("pow"), 2, 1,
                            {Requirement::operand(0), Requirement::operand(1), Requirement::result(0)}, buildPowVjp, {},
                            verifyFloatingIntrinsic));
    add(DifferentiationRule(
        intrinsicRuleKey("dot"), 2, 1, {Requirement::operand(0), Requirement::operand(1)},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            Value seed = context.resultCotangents[0];
            FailureOr<Type> leftType = getAutodiffDerivativeValueType(operation->getOperand(0).getType());
            FailureOr<Type> rightType = getAutodiffDerivativeValueType(operation->getOperand(1).getType());
            if (failed(leftType) || failed(rightType) || !isa<RankedTensorType>(*leftType) ||
                !isa<RankedTensorType>(*rightType))
                return operation->emitOpError("dot VJP requires floating-point ranked Tensor operands");
            auto leftTensor = cast<RankedTensorType>(*leftType);
            auto rightTensor = cast<RankedTensorType>(*rightType);
            if (leftTensor.getRank() == 0 || leftTensor != rightTensor ||
                operation->getResult(0).getType() != getElementTypeOrSelf(operation->getOperand(0).getType()))
                return operation->emitOpError("dot VJP requires equal ranked Tensors and a matching scalar result");
            Value left = castPrimal(context.builder, context.location, context.getPrimalOperand(0), *leftType);
            Value right = castPrimal(context.builder, context.location, context.getPrimalOperand(1), *rightType);
            if (failed(requireValues(operation, {left, right})))
                return failure();
            results.append({scale(context.builder, context.location, right, seed),
                            scale(context.builder, context.location, left, seed)});
            return success();
        },
        {}, verifyFloatingIntrinsic));
    add(DifferentiationRule(
        intrinsicRuleKey("cross"), 2, 1, {Requirement::operand(0), Requirement::operand(1)},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            Value seed = context.resultCotangents[0];
            FailureOr<Type> leftType = getAutodiffDerivativeValueType(operation->getOperand(0).getType());
            FailureOr<Type> rightType = getAutodiffDerivativeValueType(operation->getOperand(1).getType());
            if (failed(leftType) || failed(rightType) || !isa<RankedTensorType>(*leftType) ||
                cast<RankedTensorType>(*leftType).getRank() != 1 ||
                cast<RankedTensorType>(*leftType).getDimSize(0) != 3 || *leftType != *rightType ||
                operation->getResult(0).getType() != operation->getOperand(0).getType())
                return operation->emitOpError("cross VJP requires matching three-component vectors");
            Value left = castPrimal(context.builder, context.location, context.getPrimalOperand(0), *leftType);
            Value right = castPrimal(context.builder, context.location, context.getPrimalOperand(1), *rightType);
            if (failed(requireValues(operation, {left, right})))
                return failure();
            results.append({createIntrinsic(context.builder, context.location, "cross", {right, seed}, *leftType),
                            createIntrinsic(context.builder, context.location, "cross", {seed, left}, *rightType)});
            return success();
        },
        {}, verifyFloatingIntrinsic));
    add(DifferentiationRule(intrinsicRuleKey("matmul"), 2, 1, {Requirement::operand(0), Requirement::operand(1)},
                            buildMatmulVjp, {}, verifyFloatingIntrinsic));
    add(DifferentiationRule(
        intrinsicRuleKey("construct"), std::nullopt, 1, {}, buildConstructVjp, {},
        [](Operation *operation) -> LogicalResult {
            auto resultType = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
            if (!resultType || !resultType.hasStaticShape() || failed(getAutodiffDerivativeValueType(resultType)))
                return operation->emitOpError("construct VJP requires a static floating-point Tensor result");
            FailureOr<size_t> elementCount = checkedElementCount(resultType.getShape());
            if (failed(elementCount) || *elementCount != operation->getNumOperands() ||
                llvm::any_of(operation->getOperandTypes(),
                             [&](Type type) { return type != resultType.getElementType(); }))
                return operation->emitOpError(
                    "construct VJP requires one matching floating scalar operand per static Tensor element");
            return success();
        }));
    add(DifferentiationRule(
        SwizzleOp::getOperationName().str(), 1, 1, {},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            auto swizzle = cast<SwizzleOp>(operation);
            auto inputType = cast<RankedTensorType>(*getAutodiffDerivativeValueType(swizzle.getInput().getType()));
            StringRef mask = swizzle.getMask();
            Value seed = context.resultCotangents[0];
            Value zero = constant(context.builder, context.location, inputType.getElementType(), 0.0);
            SmallVector<Value> elements(inputType.getDimSize(0), zero);
            auto component = [](char value) -> int {
                StringRef xyzw = "xyzw";
                StringRef rgba = "rgba";
                size_t index = xyzw.find(value);
                return static_cast<int>(index == StringRef::npos ? rgba.find(value) : index);
            };
            for (auto [outputIndex, spelling] : llvm::enumerate(mask)) {
                const int inputIndex = component(spelling);
                Value contribution = mask.size() == 1 ? seed
                                                      : extract(context.builder, context.location, seed,
                                                                {static_cast<int64_t>(outputIndex)});
                Value &target = elements[static_cast<size_t>(inputIndex)];
                target = target == zero
                             ? contribution
                             : arith::AddFOp::create(context.builder, context.location, target, contribution);
            }
            results.push_back(tensor::FromElementsOp::create(context.builder, context.location, inputType, elements));
            return success();
        },
        {},
        [](Operation *operation) -> LogicalResult {
            auto swizzle = dyn_cast<SwizzleOp>(operation);
            if (!swizzle)
                return failure();
            FailureOr<Type> derivative = getAutodiffDerivativeValueType(swizzle.getInput().getType());
            auto input = succeeded(derivative) ? dyn_cast<RankedTensorType>(*derivative) : RankedTensorType{};
            if (!input || !input.hasStaticShape() || input.getRank() != 1 || input.getDimSize(0) > 4)
                return operation->emitOpError("swizzle VJP requires a static floating rank-one input");
            return success();
        }));
    add(DifferentiationRule(
        intrinsicRuleKey("normalize"), 1, 1, {Requirement::operand(0)},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            Value seed = context.resultCotangents[0];
            FailureOr<Type> inputType = getAutodiffDerivativeValueType(operation->getOperand(0).getType());
            if (failed(inputType) || !isa<RankedTensorType>(*inputType))
                return operation->emitOpError("normalize VJP requires a floating-point ranked Tensor operand");
            if (cast<RankedTensorType>(*inputType).getRank() != 1 ||
                operation->getResult(0).getType() != operation->getOperand(0).getType())
                return operation->emitOpError("normalize VJP requires matching vector operand and result types");
            Value input = castPrimal(context.builder, context.location, context.getPrimalOperand(0), *inputType);
            if (failed(requireValues(operation, {input})))
                return failure();
            auto tensorType = cast<RankedTensorType>(*inputType);
            Type elementType = tensorType.getElementType();
            Value squared = createIntrinsic(context.builder, context.location, "dot", {input, input}, elementType);
            Value norm = math::SqrtOp::create(context.builder, context.location, squared);
            Value normTensor = splat(context.builder, context.location, norm, tensorType);
            Value first = arith::DivFOp::create(context.builder, context.location, seed, normTensor);
            Value projection = createIntrinsic(context.builder, context.location, "dot", {seed, input}, elementType);
            Value normCubed =
                arith::MulFOp::create(context.builder, context.location,
                                      arith::MulFOp::create(context.builder, context.location, norm, norm), norm);
            Value factor = arith::DivFOp::create(context.builder, context.location, projection, normCubed);
            Value second = scale(context.builder, context.location, input, factor);
            results.push_back(arith::SubFOp::create(context.builder, context.location, first, second));
            return success();
        },
        {}, verifyFloatingIntrinsic));
    add(DifferentiationRule(
        intrinsicRuleKey("reflect"), 2, 1, {Requirement::operand(0), Requirement::operand(1)},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            Value seed = context.resultCotangents[0];
            FailureOr<Type> directionType = getAutodiffDerivativeValueType(operation->getOperand(0).getType());
            FailureOr<Type> normalType = getAutodiffDerivativeValueType(operation->getOperand(1).getType());
            if (failed(directionType) || failed(normalType) || !isa<RankedTensorType>(*directionType) ||
                cast<RankedTensorType>(*directionType).getRank() != 1 || *directionType != *normalType ||
                operation->getResult(0).getType() != operation->getOperand(0).getType())
                return operation->emitOpError("reflect VJP requires floating-point ranked Tensor operands");
            Value direction =
                castPrimal(context.builder, context.location, context.getPrimalOperand(0), *directionType);
            Value normal = castPrimal(context.builder, context.location, context.getPrimalOperand(1), *normalType);
            if (failed(requireValues(operation, {direction, normal})))
                return failure();
            Type elementType = cast<RankedTensorType>(*directionType).getElementType();
            Value two = constant(context.builder, context.location, elementType, 2.0);
            Value seedDotNormal =
                createIntrinsic(context.builder, context.location, "dot", {seed, normal}, elementType);
            Value directionGradient = arith::SubFOp::create(
                context.builder, context.location, seed,
                scale(context.builder, context.location, normal,
                      arith::MulFOp::create(context.builder, context.location, two, seedDotNormal)));
            Value directionDotNormal =
                createIntrinsic(context.builder, context.location, "dot", {direction, normal}, elementType);
            Value normalGradient = arith::AddFOp::create(
                context.builder, context.location, scale(context.builder, context.location, direction, seedDotNormal),
                scale(context.builder, context.location, seed, directionDotNormal));
            results.append({directionGradient, scale(context.builder, context.location, normalGradient,
                                                     constant(context.builder, context.location, elementType, -2.0))});
            return success();
        },
        {}, verifyFloatingIntrinsic));
    auto buildBroadcastVjp = [](Operation *operation, const AutodiffVjpBuildContext &context,
                                SmallVectorImpl<Value> &results) -> LogicalResult {
        FailureOr<Type> inputType = getAutodiffDerivativeValueType(operation->getOperand(0).getType());
        if (failed(inputType))
            return operation->emitOpError("broadcast VJP requires a differentiable input");
        results.push_back(createIntrinsic(context.builder, context.location, "reduce_sum_to_shape",
                                          context.resultCotangents, *inputType));
        return success();
    };
    add(DifferentiationRule(tensor::SplatOp::getOperationName().str(), 1, 1, {}, buildBroadcastVjp, {},
                            [](Operation *operation) -> LogicalResult {
                                auto resultType = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
                                if (!isa<FloatType>(operation->getOperand(0).getType()) || !resultType ||
                                    !resultType.hasStaticShape() ||
                                    resultType.getElementType() != operation->getOperand(0).getType())
                                    return operation->emitOpError(
                                        "tensor.splat autodiff rule requires matching static floating Tensor values");
                                return success();
                            }));
    add(DifferentiationRule(intrinsicRuleKey("broadcast"), 1, 1, {}, buildBroadcastVjp, {}, verifyFloatingIntrinsic));
    add(DifferentiationRule(
        tensor::ExtractOp::getOperationName().str(), std::nullopt, 1, {},
        [](Operation *operation, const AutodiffVjpBuildContext &context,
           SmallVectorImpl<Value> &results) -> LogicalResult {
            auto extract = cast<tensor::ExtractOp>(operation);
            auto sourceType = dyn_cast<RankedTensorType>(extract.getTensor().getType());
            FailureOr<Type> derivative = getAutodiffDerivativeValueType(extract.getTensor().getType());
            if (!sourceType || !sourceType.hasStaticShape() || failed(derivative))
                return operation->emitOpError("tensor.extract VJP requires a static floating-point Tensor");
            auto derivativeType = cast<RankedTensorType>(*derivative);
            Value seed = context.resultCotangents.front();
            Value zero = constant(context.builder, context.location, derivativeType.getElementType(), 0.0);
            SmallVector<Value> elements;
            for (const SmallVector<int64_t> &coordinate : enumerateStaticCoordinates(sourceType.getShape())) {
                Value selected = seed;
                for (auto [dimension, expected] : llvm::enumerate(coordinate)) {
                    Value expectedValue = arith::ConstantIndexOp::create(context.builder, context.location, expected);
                    Value matches = arith::CmpIOp::create(context.builder, context.location, arith::CmpIPredicate::eq,
                                                          context.primalOperands[dimension + 1], expectedValue);
                    selected = arith::SelectOp::create(context.builder, context.location, matches, selected, zero);
                }
                elements.push_back(selected);
            }
            results.push_back(
                tensor::FromElementsOp::create(context.builder, context.location, derivativeType, elements));
            results.append(extract.getIndices().size(), Value{});
            return success();
        },
        {},
        [](Operation *operation) -> LogicalResult {
            auto extract = dyn_cast<tensor::ExtractOp>(operation);
            if (!extract)
                return failure();
            auto source = dyn_cast<RankedTensorType>(extract.getTensor().getType());
            if (!source || !source.hasStaticShape() || !source.getElementType().isIntOrFloat() ||
                !extract.getResult().getType().isIntOrFloat())
                return operation->emitOpError("tensor.extract VJP requires a static scalar-element Tensor");
            return success();
        }));
    return registry;
}

LogicalResult verifyAutodiffRuleCoverage(const VernonAutodiffAnalysisResult &analysis,
                                         const VernonAutodiffRuleRegistry &registry) {
    for (const AutodiffOperationActivity &activity : analysis.getOperations()) {
        Operation *operation = activity.operation;
        if (!activity.active || isStructuralAutodiffOperation(operation))
            continue;
        if (operation->getNumResults() == 0)
            continue;
        if (operation->getNumResults() > 1)
            return operation->emitOpError("active autodiff operation must have exactly one result");
        if (failed(getAutodiffDerivativeValueType(operation->getResult(0).getType())))
            continue;
        const DifferentiationRule *rule = registry.lookup(operation);
        if (!rule)
            return operation->emitOpError("is active but has no registered differentiation rule");
        if (failed(rule->verifyCompatibility(operation)))
            return failure();
    }
    return success();
}

} // namespace mlir::vernon
