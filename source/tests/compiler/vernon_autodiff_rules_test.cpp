#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace mlir::vernon {
namespace {

using Requirement = AutodiffPrimalRequirement;

struct ScalarRuleCase {
    const char *function;
    const char *operation;
    SmallVector<Requirement> requirements;
    SmallVector<double> inputs;
    SmallVector<double> cpuOracle;
};

struct TensorValue {
    SmallVector<int64_t> shape;
    SmallVector<double> elements;

    static TensorValue scalar(double value) { return {{}, {value}}; }
};

class VernonAutodiffRulesTest : public testing::Test {
protected:
    VernonAutodiffRulesTest() {
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<math::MathDialect>();
        context.getOrLoadDialect<tensor::TensorDialect>();
        context.getOrLoadDialect<VernonDialect>();
    }

    OwningOpRef<ModuleOp> parseModule() {
        return parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @add(%x: f32, %y: f32) -> f32 {
    %r = arith.addf %x, %y : f32
    func.return %r : f32
  }
  func.func @sub(%x: f32, %y: f32) -> f32 {
    %r = arith.subf %x, %y : f32
    func.return %r : f32
  }
  func.func @mul(%x: f32, %y: f32) -> f32 {
    %r = arith.mulf %x, %y : f32
    func.return %r : f32
  }
  func.func @div(%x: f32, %y: f32) -> f32 {
    %r = arith.divf %x, %y : f32
    func.return %r : f32
  }
  func.func @neg(%x: f32) -> f32 {
    %r = arith.negf %x : f32
    func.return %r : f32
  }
  func.func @sin(%x: f32) -> f32 {
    %r = math.sin %x : f32
    func.return %r : f32
  }
  func.func @cos(%x: f32) -> f32 {
    %r = math.cos %x : f32
    func.return %r : f32
  }
  func.func @acos(%x: f32) -> f32 {
    %r = math.acos %x : f32
    func.return %r : f32
  }
  func.func @atan2(%y: f32, %x: f32) -> f32 {
    %r = math.atan2 %y, %x : f32
    func.return %r : f32
  }
  func.func @exp(%x: f32) -> f32 {
    %r = math.exp %x : f32
    func.return %r : f32
  }
  func.func @log(%x: f32) -> f32 {
    %r = math.log %x : f32
    func.return %r : f32
  }
  func.func @sqrt(%x: f32) -> f32 {
    %r = math.sqrt %x : f32
    func.return %r : f32
  }
  func.func @abs(%x: f32) -> f32 {
    %r = math.absf %x : f32
    func.return %r : f32
  }
  func.func @pow(%x: f32, %y: f32) -> f32 {
    %r = "vernon.intrinsic"(%x, %y) {name = "pow"} : (f32, f32) -> f32
    func.return %r : f32
  }
}
)mlir",
                                           ParserConfig(&context));
    }

    static SmallVector<ScalarRuleCase> cases() {
        return {
            {"add", "arith.addf", {}, {1.3, 0.7}, {1.0, 1.0}},
            {"sub", "arith.subf", {}, {1.3, 0.7}, {1.0, -1.0}},
            {"mul", "arith.mulf", {Requirement::operand(0), Requirement::operand(1)}, {1.3, 0.7}, {0.7, 1.3}},
            {"div",
             "arith.divf",
             {Requirement::operand(0), Requirement::operand(1)},
             {1.3, 0.7},
             {1.0 / 0.7, -1.3 / (0.7 * 0.7)}},
            {"neg", "arith.negf", {}, {1.3}, {-1.0}},
            {"sin", "math.sin", {Requirement::operand(0)}, {0.4}, {std::cos(0.4)}},
            {"cos", "math.cos", {Requirement::operand(0)}, {0.4}, {-std::sin(0.4)}},
            {"acos", "math.acos", {Requirement::operand(0)}, {0.4}, {-1.0 / std::sqrt(1.0 - 0.4 * 0.4)}},
            {"atan2",
             "math.atan2",
             {Requirement::operand(0), Requirement::operand(1)},
             {0.4, 1.2},
             {1.2 / (0.4 * 0.4 + 1.2 * 1.2), -0.4 / (0.4 * 0.4 + 1.2 * 1.2)}},
            {"exp", "math.exp", {Requirement::result(0)}, {0.4}, {std::exp(0.4)}},
            {"log", "math.log", {Requirement::operand(0)}, {1.4}, {1.0 / 1.4}},
            {"sqrt", "math.sqrt", {Requirement::result(0)}, {1.4}, {0.5 / std::sqrt(1.4)}},
            {"abs", "math.absf", {Requirement::operand(0)}, {-1.4}, {-1.0}},
            {"pow",
             "vernon.intrinsic.pow",
             {Requirement::operand(0), Requirement::operand(1), Requirement::result(0)},
             {1.4, 1.7},
             {1.7 * std::pow(1.4, 0.7), std::pow(1.4, 1.7) * std::log(1.4)}},
        };
    }

    double evaluate(Value value, ArrayRef<double> arguments, DenseMap<Value, double> &cache) {
        if (auto found = cache.find(value); found != cache.end())
            return found->second;
        if (auto argument = dyn_cast<BlockArgument>(value))
            return arguments[argument.getArgNumber()];
        Operation *operation = value.getDefiningOp();
        auto operand = [&](unsigned index) { return evaluate(operation->getOperand(index), arguments, cache); };
        double result = 0.0;
        if (auto constant = dyn_cast<arith::ConstantOp>(operation))
            result = cast<FloatAttr>(constant.getValue()).getValueAsDouble();
        else if (isa<arith::AddFOp>(operation))
            result = operand(0) + operand(1);
        else if (isa<arith::SubFOp>(operation))
            result = operand(0) - operand(1);
        else if (isa<arith::MulFOp>(operation))
            result = operand(0) * operand(1);
        else if (isa<arith::DivFOp>(operation))
            result = operand(0) / operand(1);
        else if (isa<arith::NegFOp>(operation))
            result = -operand(0);
        else if (isa<math::SinOp>(operation))
            result = std::sin(operand(0));
        else if (isa<math::CosOp>(operation))
            result = std::cos(operand(0));
        else if (isa<math::AcosOp>(operation))
            result = std::acos(operand(0));
        else if (isa<math::Atan2Op>(operation))
            result = std::atan2(operand(0), operand(1));
        else if (isa<math::ExpOp>(operation))
            result = std::exp(operand(0));
        else if (isa<math::LogOp>(operation))
            result = std::log(operand(0));
        else if (isa<math::SqrtOp>(operation))
            result = std::sqrt(operand(0));
        else if (isa<math::AbsFOp>(operation))
            result = std::abs(operand(0));
        else if (isa<math::PowFOp>(operation) ||
                 (isa<IntrinsicOp>(operation) && cast<IntrinsicOp>(operation).getName() == "pow"))
            result = std::pow(operand(0), operand(1));
        else if (auto compare = dyn_cast<arith::CmpFOp>(operation)) {
            if (compare.getPredicate() == arith::CmpFPredicate::OGT)
                result = operand(0) > operand(1);
            else if (compare.getPredicate() == arith::CmpFPredicate::OLT)
                result = operand(0) < operand(1);
            else
                ADD_FAILURE() << "unsupported comparison in emitted VJP";
        } else if (isa<arith::SelectOp>(operation))
            result = operand(0) != 0.0 ? operand(1) : operand(2);
        else
            ADD_FAILURE() << "unsupported operation in emitted VJP: " << operation->getName().getStringRef().str();
        cache.try_emplace(value, result);
        return result;
    }

    static size_t elementCount(ArrayRef<int64_t> shape) {
        size_t count = 1;
        for (int64_t extent : shape)
            count *= static_cast<size_t>(extent);
        return count;
    }

    static SmallVector<SmallVector<int64_t>> coordinates(ArrayRef<int64_t> shape) {
        SmallVector<SmallVector<int64_t>> result(1);
        for (int64_t extent : shape) {
            SmallVector<SmallVector<int64_t>> expanded;
            for (const auto &prefix : result)
                for (int64_t index = 0; index < extent; ++index) {
                    SmallVector<int64_t> coordinate(prefix);
                    coordinate.push_back(index);
                    expanded.push_back(std::move(coordinate));
                }
            result = std::move(expanded);
        }
        return result;
    }

    static size_t flatIndex(ArrayRef<int64_t> coordinate, ArrayRef<int64_t> shape) {
        size_t result = 0;
        for (auto [index, value] : llvm::enumerate(coordinate))
            result = result * static_cast<size_t>(shape[index]) + static_cast<size_t>(value);
        return result;
    }

    static SmallVector<int64_t> broadcastCoordinate(ArrayRef<int64_t> coordinate, ArrayRef<int64_t> shape) {
        SmallVector<int64_t> result;
        unsigned offset = coordinate.size() - shape.size();
        for (auto [index, extent] : llvm::enumerate(shape))
            result.push_back(extent == 1 ? 0 : coordinate[offset + index]);
        return result;
    }

    static SmallVector<int64_t> valueShape(Type type) {
        if (auto tensorType = dyn_cast<RankedTensorType>(type))
            return SmallVector<int64_t>(tensorType.getShape());
        return {};
    }

    TensorValue evaluateTensor(Value value, ArrayRef<TensorValue> arguments, DenseMap<Value, TensorValue> &cache) {
        if (auto found = cache.find(value); found != cache.end())
            return found->second;
        if (auto argument = dyn_cast<BlockArgument>(value))
            return arguments[argument.getArgNumber()];

        Operation *operation = value.getDefiningOp();
        auto operand = [&](unsigned index) { return evaluateTensor(operation->getOperand(index), arguments, cache); };
        TensorValue result;
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
            result.shape = valueShape(value.getType());
            if (auto scalar = dyn_cast<FloatAttr>(constant.getValue()))
                result.elements.push_back(scalar.getValueAsDouble());
            else if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
                result.elements.push_back(integer.getValue().getSExtValue());
            else if (auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue()))
                for (const APFloat &element : dense.getValues<APFloat>())
                    result.elements.push_back(element.convertToDouble());
            else
                ADD_FAILURE() << "unsupported tensor constant";
        } else if (isa<arith::ExtFOp>(operation)) {
            result = operand(0);
        } else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(operation)) {
            TensorValue left = operand(0);
            TensorValue right = operand(1);
            result.shape = left.shape;
            EXPECT_EQ(left.shape, right.shape);
            EXPECT_EQ(left.elements.size(), right.elements.size());
            for (auto [leftElement, rightElement] : llvm::zip_equal(left.elements, right.elements)) {
                if (isa<arith::AddFOp>(operation))
                    result.elements.push_back(leftElement + rightElement);
                else if (isa<arith::SubFOp>(operation))
                    result.elements.push_back(leftElement - rightElement);
                else if (isa<arith::MulFOp>(operation))
                    result.elements.push_back(leftElement * rightElement);
                else
                    result.elements.push_back(leftElement / rightElement);
            }
        } else if (isa<math::SqrtOp>(operation)) {
            result = operand(0);
            for (double &element : result.elements)
                element = std::sqrt(element);
        } else if (auto splat = dyn_cast<tensor::SplatOp>(operation)) {
            TensorValue scalar = operand(0);
            result.shape = valueShape(splat.getType());
            result.elements.assign(elementCount(result.shape), scalar.elements.front());
        } else if (auto extract = dyn_cast<tensor::ExtractOp>(operation)) {
            TensorValue tensorValue = operand(0);
            SmallVector<int64_t> coordinate;
            for (unsigned index = 1; index < operation->getNumOperands(); ++index)
                coordinate.push_back(static_cast<int64_t>(operand(index).elements.front()));
            result = TensorValue::scalar(tensorValue.elements[flatIndex(coordinate, tensorValue.shape)]);
        } else if (auto intrinsic = dyn_cast<IntrinsicOp>(operation)) {
            StringRef name = intrinsic.getName();
            if (name == "construct") {
                result.shape = valueShape(value.getType());
                for (unsigned index = 0; index < operation->getNumOperands(); ++index) {
                    TensorValue element = operand(index);
                    EXPECT_EQ(element.elements.size(), 1u);
                    result.elements.push_back(element.elements.front());
                }
            } else if (name == "dot") {
                TensorValue left = operand(0);
                TensorValue right = operand(1);
                EXPECT_EQ(left.shape, right.shape);
                double sum = 0.0;
                for (auto [leftElement, rightElement] : llvm::zip_equal(left.elements, right.elements))
                    sum += leftElement * rightElement;
                result = TensorValue::scalar(sum);
            } else if (name == "cross") {
                TensorValue left = operand(0);
                TensorValue right = operand(1);
                EXPECT_EQ(left.elements.size(), 3u);
                EXPECT_EQ(right.elements.size(), 3u);
                result = {{3},
                          {left.elements[1] * right.elements[2] - left.elements[2] * right.elements[1],
                           left.elements[2] * right.elements[0] - left.elements[0] * right.elements[2],
                           left.elements[0] * right.elements[1] - left.elements[1] * right.elements[0]}};
            } else if (name == "normalize") {
                result = operand(0);
                double squared = 0.0;
                for (double element : result.elements)
                    squared += element * element;
                double length = std::sqrt(squared);
                for (double &element : result.elements)
                    element /= length;
            } else if (name == "reflect") {
                TensorValue direction = operand(0);
                TensorValue normal = operand(1);
                double projection = 0.0;
                for (auto [directionElement, normalElement] : llvm::zip_equal(direction.elements, normal.elements))
                    projection += directionElement * normalElement;
                result = direction;
                for (auto [output, normalElement] : llvm::zip_equal(result.elements, normal.elements))
                    output -= 2.0 * projection * normalElement;
            } else if (name == "broadcast") {
                TensorValue source = operand(0);
                result.shape = valueShape(value.getType());
                for (const auto &coordinate : coordinates(result.shape))
                    result.elements.push_back(
                        source.elements[flatIndex(broadcastCoordinate(coordinate, source.shape), source.shape)]);
            } else if (name == "reduce_sum_to_shape") {
                TensorValue source = operand(0);
                result.shape = valueShape(value.getType());
                result.elements.assign(elementCount(result.shape), 0.0);
                for (const auto &coordinate : coordinates(source.shape)) {
                    SmallVector<int64_t> target =
                        result.shape.empty() ? SmallVector<int64_t>() : broadcastCoordinate(coordinate, result.shape);
                    result.elements[flatIndex(target, result.shape)] +=
                        source.elements[flatIndex(coordinate, source.shape)];
                }
            } else if (name == "matmul") {
                TensorValue left = operand(0);
                TensorValue right = operand(1);
                bool leftVector = left.shape.size() == 1;
                bool rightVector = right.shape.size() == 1;
                ArrayRef<int64_t> leftBatch =
                    leftVector ? ArrayRef<int64_t>() : ArrayRef<int64_t>(left.shape).drop_back(2);
                ArrayRef<int64_t> rightBatch =
                    rightVector ? ArrayRef<int64_t>() : ArrayRef<int64_t>(right.shape).drop_back(2);
                result.shape = valueShape(value.getType());
                result.elements.assign(elementCount(result.shape), 0.0);
                unsigned batchRank = std::max(leftBatch.size(), rightBatch.size());
                ArrayRef<int64_t> outputBatch =
                    ArrayRef<int64_t>(result.shape).take_front(std::min<unsigned>(batchRank, result.shape.size()));
                int64_t rows = leftVector ? 1 : left.shape[left.shape.size() - 2];
                int64_t reduction = left.shape.back();
                int64_t columns = rightVector ? 1 : right.shape.back();
                for (const auto &batch : coordinates(outputBatch))
                    for (int64_t row = 0; row < rows; ++row)
                        for (int64_t column = 0; column < columns; ++column) {
                            SmallVector<int64_t> leftCoordinate =
                                leftVector ? SmallVector<int64_t>() : broadcastCoordinate(batch, leftBatch);
                            SmallVector<int64_t> rightCoordinate =
                                rightVector ? SmallVector<int64_t>() : broadcastCoordinate(batch, rightBatch);
                            if (!leftVector)
                                leftCoordinate.push_back(row);
                            if (!rightVector)
                                rightCoordinate.push_back(0);
                            SmallVector<int64_t> outputCoordinate(batch);
                            if (!leftVector)
                                outputCoordinate.push_back(row);
                            if (!rightVector)
                                outputCoordinate.push_back(column);
                            double sum = 0.0;
                            for (int64_t inner = 0; inner < reduction; ++inner) {
                                SmallVector<int64_t> currentLeft(leftCoordinate);
                                SmallVector<int64_t> currentRight(rightCoordinate);
                                if (leftVector)
                                    currentLeft.push_back(inner);
                                else
                                    currentLeft.push_back(inner);
                                if (rightVector)
                                    currentRight.push_back(inner);
                                else {
                                    currentRight[currentRight.size() - 1] = inner;
                                    currentRight.push_back(column);
                                }
                                sum += left.elements[flatIndex(currentLeft, left.shape)] *
                                       right.elements[flatIndex(currentRight, right.shape)];
                            }
                            result.elements[flatIndex(outputCoordinate, result.shape)] = sum;
                        }
            } else {
                ADD_FAILURE() << "unsupported intrinsic in tensor evaluator: " << name.str();
            }
        } else {
            ADD_FAILURE() << "unsupported operation in tensor VJP: " << operation->getName().getStringRef().str();
        }
        cache.try_emplace(value, result);
        return result;
    }

    Value makeTensorConstant(OpBuilder &builder, Location location, Type type, const TensorValue &value) {
        if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
            SmallVector<Attribute> elements;
            auto elementType = cast<FloatType>(tensorType.getElementType());
            for (double element : value.elements)
                elements.push_back(builder.getFloatAttr(elementType, element));
            return arith::ConstantOp::create(builder, location, DenseElementsAttr::get(tensorType, elements));
        }
        return arith::ConstantOp::create(builder, location,
                                         builder.getFloatAttr(cast<FloatType>(type), value.elements.front()));
    }

    void expectTensorVjpMatchesFiniteDifference(func::FuncOp function, ArrayRef<TensorValue> arguments,
                                                const TensorValue &seed, double tolerance = 2.0e-4) {
        IntrinsicOp intrinsic;
        tensor::SplatOp splat;
        function.walk([&](IntrinsicOp operation) { intrinsic = operation; });
        function.walk([&](tensor::SplatOp operation) { splat = operation; });
        Operation *operation = intrinsic ? intrinsic.getOperation() : splat.getOperation();
        ASSERT_NE(operation, nullptr);
        VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
        const DifferentiationRule *rule = registry.lookup(operation);
        ASSERT_NE(rule, nullptr);
        auto returnOp = cast<func::ReturnOp>(function.front().getTerminator());
        OpBuilder builder(returnOp);
        Value seedValue = makeTensorConstant(builder, operation->getLoc(), operation->getResult(0).getType(), seed);
        FailureOr<SmallVector<Value>> contributions =
            rule->buildVjp(operation, AutodiffVjpBuildContext{builder, operation->getLoc(), operation->getOperands(),
                                                              operation->getResults(), ValueRange(seedValue)});
        ASSERT_TRUE(succeeded(contributions));
        ASSERT_EQ(contributions->size(), arguments.size());

        constexpr double epsilon = 1.0e-4;
        for (unsigned operandIndex = 0; operandIndex < arguments.size(); ++operandIndex) {
            DenseMap<Value, TensorValue> emittedCache;
            TensorValue emitted = evaluateTensor((*contributions)[operandIndex], arguments, emittedCache);
            ASSERT_EQ(emitted.shape, arguments[operandIndex].shape);
            ASSERT_EQ(emitted.elements.size(), arguments[operandIndex].elements.size());
            for (unsigned elementIndex = 0; elementIndex < emitted.elements.size(); ++elementIndex) {
                SmallVector<TensorValue> lower(arguments);
                SmallVector<TensorValue> upper(arguments);
                lower[operandIndex].elements[elementIndex] -= epsilon;
                upper[operandIndex].elements[elementIndex] += epsilon;
                DenseMap<Value, TensorValue> lowerCache;
                DenseMap<Value, TensorValue> upperCache;
                TensorValue lowerOutput = evaluateTensor(operation->getResult(0), lower, lowerCache);
                TensorValue upperOutput = evaluateTensor(operation->getResult(0), upper, upperCache);
                ASSERT_EQ(lowerOutput.elements.size(), seed.elements.size());
                double lowerObjective = 0.0;
                double upperObjective = 0.0;
                for (unsigned outputIndex = 0; outputIndex < seed.elements.size(); ++outputIndex) {
                    lowerObjective += lowerOutput.elements[outputIndex] * seed.elements[outputIndex];
                    upperObjective += upperOutput.elements[outputIndex] * seed.elements[outputIndex];
                }
                double finiteDifference = (upperObjective - lowerObjective) / (2.0 * epsilon);
                EXPECT_NEAR(emitted.elements[elementIndex], finiteDifference, tolerance)
                    << function.getName().str() << " operand " << operandIndex << " element " << elementIndex;
            }
        }
    }

    MLIRContext context;
};

TEST_F(VernonAutodiffRulesTest, RegistryMatchesFrontendPureRuleInventoryAndPrimalRequirements) {
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    SmallVector<StringRef> expectedNames;
    for (const ScalarRuleCase &testCase : cases()) {
        expectedNames.push_back(testCase.operation);
        const DifferentiationRule *rule = registry.lookup(testCase.operation);
        ASSERT_NE(rule, nullptr) << testCase.operation;
        EXPECT_EQ(rule->getVjpPrimalRequirements(), ArrayRef<Requirement>(testCase.requirements)) << testCase.operation;
        EXPECT_FALSE(rule->hasJvpBuilder());
    }
    const SmallVector<std::pair<StringRef, SmallVector<Requirement>>> tensorRules = {
        {"tensor.extract", {}},
        {"tensor.splat", {}},
        {"vernon.intrinsic.broadcast", {}},
        {"vernon.intrinsic.construct", {}},
        {"vernon.intrinsic.cross", {Requirement::operand(0), Requirement::operand(1)}},
        {"vernon.intrinsic.dot", {Requirement::operand(0), Requirement::operand(1)}},
        {"vernon.intrinsic.matmul", {Requirement::operand(0), Requirement::operand(1)}},
        {"vernon.intrinsic.normalize", {Requirement::operand(0)}},
        {"vernon.intrinsic.reflect", {Requirement::operand(0), Requirement::operand(1)}},
    };
    for (const auto &[name, requirements] : tensorRules) {
        expectedNames.push_back(name);
        const DifferentiationRule *rule = registry.lookup(name);
        ASSERT_NE(rule, nullptr) << name.str();
        EXPECT_EQ(rule->getVjpPrimalRequirements(), ArrayRef<Requirement>(requirements)) << name.str();
    }
    llvm::sort(expectedNames);
    EXPECT_EQ(registry.getRegisteredKeys(), expectedNames);
}

TEST_F(VernonAutodiffRulesTest, EmittedVjpsMatchFiniteDifferencesAndCpuOracle) {
    OwningOpRef<ModuleOp> module = parseModule();
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    constexpr double epsilon = 1.0e-3;

    for (const ScalarRuleCase &testCase : cases()) {
        func::FuncOp function = module->lookupSymbol<func::FuncOp>(testCase.function);
        ASSERT_TRUE(function);
        const DifferentiationRule *rule = registry.lookup(testCase.operation);
        ASSERT_NE(rule, nullptr);
        Operation *operation = nullptr;
        function.walk([&](Operation *candidate) {
            if (registry.lookup(candidate) == rule)
                operation = candidate;
        });
        ASSERT_NE(operation, nullptr) << testCase.function;
        auto returnOp = cast<func::ReturnOp>(function.front().getTerminator());
        OpBuilder builder(returnOp);
        Value seed = arith::ConstantOp::create(builder, operation->getLoc(), builder.getF32FloatAttr(1.0));
        FailureOr<SmallVector<Value>> contributions = rule->buildVjp(operation, AutodiffVjpBuildContext{
                                                                                    builder,
                                                                                    operation->getLoc(),
                                                                                    operation->getOperands(),
                                                                                    operation->getResults(),
                                                                                    ValueRange(seed),
                                                                                });
        ASSERT_TRUE(succeeded(contributions)) << testCase.function;
        ASSERT_EQ(contributions->size(), testCase.inputs.size());

        for (unsigned index = 0; index < testCase.inputs.size(); ++index) {
            DenseMap<Value, double> cache;
            double emitted = evaluate((*contributions)[index], testCase.inputs, cache);
            EXPECT_NEAR(emitted, testCase.cpuOracle[index], 2.0e-5) << testCase.function << " operand " << index;

            SmallVector<double> lower(testCase.inputs);
            SmallVector<double> upper(testCase.inputs);
            lower[index] -= epsilon;
            upper[index] += epsilon;
            DenseMap<Value, double> lowerCache;
            DenseMap<Value, double> upperCache;
            double finiteDifference = (evaluate(operation->getResult(0), upper, upperCache) -
                                       evaluate(operation->getResult(0), lower, lowerCache)) /
                                      (2.0 * epsilon);
            EXPECT_NEAR(emitted, finiteDifference, 3.0e-3) << testCase.function << " operand " << index;
        }
    }
}

TEST_F(VernonAutodiffRulesTest, PromotesF16PrimalsToF32DerivativeArithmetic) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @multiply(%x: f16, %y: f16) -> f16 {
    %result = arith.mulf %x, %y : f16
    func.return %result : f16
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("multiply");
    arith::MulFOp multiply;
    function.walk([&](arith::MulFOp operation) { multiply = operation; });
    ASSERT_TRUE(multiply);
    auto returnOp = cast<func::ReturnOp>(function.front().getTerminator());
    OpBuilder builder(returnOp);
    Value seed = arith::ConstantOp::create(builder, multiply.getLoc(), builder.getF32FloatAttr(1.0));
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<SmallVector<Value>> contributions =
        registry.lookup(multiply)->buildVjp(multiply, AutodiffVjpBuildContext{
                                                          builder,
                                                          multiply.getLoc(),
                                                          multiply->getOperands(),
                                                          multiply->getResults(),
                                                          ValueRange(seed),
                                                      });
    ASSERT_TRUE(succeeded(contributions));
    ASSERT_EQ(contributions->size(), 2u);
    EXPECT_TRUE((*contributions)[0].getType().isF32());
    EXPECT_TRUE((*contributions)[1].getType().isF32());
    unsigned extensionCount = 0;
    function.walk([&](arith::ExtFOp) { ++extensionCount; });
    EXPECT_EQ(extensionCount, 2u);

    ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
    SmallVector<Value> unavailableOperands(2);
    SmallVector<Value> unavailableResults(1);
    EXPECT_TRUE(failed(registry.lookup(multiply)->buildVjp(
        multiply, AutodiffVjpBuildContext{builder, multiply.getLoc(), unavailableOperands, unavailableResults,
                                          ValueRange(seed)})));
    DifferentiationRule invalidRule(
        arith::MulFOp::getOperationName().str(), 2, 1, {},
        [](Operation *operation, const AutodiffVjpBuildContext &, SmallVectorImpl<Value> &results) {
            results.append(operation->operand_begin(), operation->operand_end());
            return success();
        });
    EXPECT_TRUE(failed(invalidRule.buildVjp(multiply, AutodiffVjpBuildContext{
                                                          builder,
                                                          multiply.getLoc(),
                                                          multiply->getOperands(),
                                                          multiply->getResults(),
                                                          ValueRange(seed),
                                                      })));
}

TEST_F(VernonAutodiffRulesTest, EmitsReflectAndStaticMatmulVjpsFromIntrinsicRules) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @reflect(%direction: tensor<3xf32>, %normal: tensor<3xf32>) -> tensor<3xf32> {
    %result = "vernon.intrinsic"(%direction, %normal) {name = "reflect"}
        : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %result : tensor<3xf32>
  }
  func.func @matmul(%left: tensor<2x3xf32>, %right: tensor<3x2xf32>) -> tensor<2x2xf32> {
    %result = "vernon.intrinsic"(%left, %right) {name = "matmul"}
        : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
  func.func @zero_reduction(%left: tensor<2x0xf32>, %right: tensor<0x2xf32>) -> tensor<2x2xf32> {
    %result = "vernon.intrinsic"(%left, %right) {name = "matmul"}
        : (tensor<2x0xf32>, tensor<0x2xf32>) -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
  func.func @construct(%x: f32, %y: f32, %z: f32) -> tensor<3xf32> {
    %result = "vernon.intrinsic"(%x, %y, %z) {name = "construct"}
        : (f32, f32, f32) -> tensor<3xf32>
    func.return %result : tensor<3xf32>
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();

    for (StringRef functionName : {"reflect", "matmul", "zero_reduction"}) {
        func::FuncOp function = module->lookupSymbol<func::FuncOp>(functionName);
        IntrinsicOp intrinsic;
        function.walk([&](IntrinsicOp operation) {
            if (operation.getName() == functionName ||
                (functionName == "zero_reduction" && operation.getName() == "matmul"))
                intrinsic = operation;
        });
        ASSERT_TRUE(intrinsic);
        const DifferentiationRule *rule = registry.lookup(intrinsic);
        ASSERT_NE(rule, nullptr);
        SmallVector<Requirement> expectedRequirements{Requirement::operand(0), Requirement::operand(1)};
        EXPECT_EQ(rule->getVjpPrimalRequirements(), ArrayRef<Requirement>(expectedRequirements));

        auto returnOp = cast<func::ReturnOp>(function.front().getTerminator());
        OpBuilder builder(returnOp);
        auto resultType = cast<RankedTensorType>(intrinsic.getResult().getType());
        Value seed = arith::ConstantOp::create(builder, intrinsic.getLoc(),
                                               DenseElementsAttr::get(resultType, builder.getF32FloatAttr(1.0)));
        FailureOr<SmallVector<Value>> contributions = rule->buildVjp(intrinsic, AutodiffVjpBuildContext{
                                                                                    builder,
                                                                                    intrinsic.getLoc(),
                                                                                    intrinsic->getOperands(),
                                                                                    intrinsic->getResults(),
                                                                                    ValueRange(seed),
                                                                                });
        ASSERT_TRUE(succeeded(contributions)) << functionName.str();
        ASSERT_EQ(contributions->size(), 2u);
        EXPECT_EQ((*contributions)[0].getType(), intrinsic->getOperand(0).getType());
        EXPECT_EQ((*contributions)[1].getType(), intrinsic->getOperand(1).getType());
    }

    unsigned dotCount = 0;
    module->lookupSymbol<func::FuncOp>("reflect").walk([&](IntrinsicOp operation) {
        if (operation.getName() == "dot")
            ++dotCount;
    });
    EXPECT_EQ(dotCount, 2u);
    unsigned constructCount = 0;
    module->lookupSymbol<func::FuncOp>("matmul").walk([&](IntrinsicOp operation) {
        if (operation.getName() == "construct")
            ++constructCount;
    });
    EXPECT_EQ(constructCount, 2u);

    func::FuncOp constructFunction = module->lookupSymbol<func::FuncOp>("construct");
    IntrinsicOp construct;
    constructFunction.walk([&](IntrinsicOp operation) { construct = operation; });
    ASSERT_TRUE(construct);
    auto constructReturn = cast<func::ReturnOp>(constructFunction.front().getTerminator());
    OpBuilder constructBuilder(constructReturn);
    auto constructType = cast<RankedTensorType>(construct.getResult().getType());
    Value constructSeed =
        arith::ConstantOp::create(constructBuilder, construct.getLoc(),
                                  DenseElementsAttr::get(constructType, constructBuilder.getF32FloatAttr(1.0)));
    FailureOr<SmallVector<Value>> constructContributions =
        registry.lookup(construct)->buildVjp(construct, AutodiffVjpBuildContext{
                                                            constructBuilder,
                                                            construct.getLoc(),
                                                            construct->getOperands(),
                                                            construct->getResults(),
                                                            ValueRange(constructSeed),
                                                        });
    ASSERT_TRUE(succeeded(constructContributions));
    EXPECT_EQ(constructContributions->size(), 3u);
    EXPECT_TRUE(llvm::all_of(*constructContributions,
                             [](Value value) { return isa<tensor::ExtractOp>(value.getDefiningOp()); }));
    EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(VernonAutodiffRulesTest, TensorRuleVjpsMatchFiniteDifferences) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @dot(%x: tensor<3xf32>, %y: tensor<3xf32>) -> f32 {
    %result = "vernon.intrinsic"(%x, %y) {name = "dot"}
        : (tensor<3xf32>, tensor<3xf32>) -> f32
    func.return %result : f32
  }
  func.func @cross(%x: tensor<3xf32>, %y: tensor<3xf32>) -> tensor<3xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "cross"}
        : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %result : tensor<3xf32>
  }
  func.func @normalize(%x: tensor<3xf32>) -> tensor<3xf32> {
    %result = "vernon.intrinsic"(%x) {name = "normalize"}
        : (tensor<3xf32>) -> tensor<3xf32>
    func.return %result : tensor<3xf32>
  }
  func.func @reflect(%x: tensor<3xf32>, %n: tensor<3xf32>) -> tensor<3xf32> {
    %result = "vernon.intrinsic"(%x, %n) {name = "reflect"}
        : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %result : tensor<3xf32>
  }
  func.func @construct(%x: f32, %y: f32, %z: f32, %w: f32) -> tensor<2x2xf32> {
    %result = "vernon.intrinsic"(%x, %y, %z, %w) {name = "construct"}
        : (f32, f32, f32, f32) -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
  func.func @splat(%x: f32) -> tensor<2x3xf32> {
    %result = tensor.splat %x : tensor<2x3xf32>
    func.return %result : tensor<2x3xf32>
  }
  func.func @broadcast(%x: tensor<1x3xf32>) -> tensor<2x3xf32> {
    %result = "vernon.intrinsic"(%x) {name = "broadcast"}
        : (tensor<1x3xf32>) -> tensor<2x3xf32>
    func.return %result : tensor<2x3xf32>
  }
  func.func @matmul_vv(%x: tensor<3xf32>, %y: tensor<3xf32>) -> f32 {
    %result = "vernon.intrinsic"(%x, %y) {name = "matmul"}
        : (tensor<3xf32>, tensor<3xf32>) -> f32
    func.return %result : f32
  }
  func.func @matmul_vm(%x: tensor<3xf32>, %y: tensor<3x2xf32>) -> tensor<2xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "matmul"}
        : (tensor<3xf32>, tensor<3x2xf32>) -> tensor<2xf32>
    func.return %result : tensor<2xf32>
  }
  func.func @matmul_mv(%x: tensor<2x3xf32>, %y: tensor<3xf32>) -> tensor<2xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "matmul"}
        : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
    func.return %result : tensor<2xf32>
  }
  func.func @matmul_mm(%x: tensor<2x3xf32>, %y: tensor<3x2xf32>) -> tensor<2x2xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "matmul"}
        : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
  func.func @matmul_batched(%x: tensor<2x2x3xf32>, %y: tensor<1x3x2xf32>) -> tensor<2x2x2xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "matmul"}
        : (tensor<2x2x3xf32>, tensor<1x3x2xf32>) -> tensor<2x2x2xf32>
    func.return %result : tensor<2x2x2xf32>
  }
  func.func @matmul_zero(%x: tensor<2x0xf32>, %y: tensor<0x2xf32>) -> tensor<2x2xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "matmul"}
        : (tensor<2x0xf32>, tensor<0x2xf32>) -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);

    struct Case {
        const char *function;
        SmallVector<TensorValue> arguments;
        TensorValue seed;
    };
    const SmallVector<Case, 0> cases = {
        {"dot", {{{3}, {0.3, -0.7, 1.2}}, {{3}, {-0.4, 0.8, 0.5}}}, TensorValue::scalar(1.7)},
        {"cross", {{{3}, {0.3, -0.7, 1.2}}, {{3}, {-0.4, 0.8, 0.5}}}, {{3}, {0.6, -1.1, 0.4}}},
        {"normalize", {{{3}, {0.3, -0.7, 1.2}}}, {{3}, {0.6, -1.1, 0.4}}},
        {"reflect", {{{3}, {0.3, -0.7, 1.2}}, {{3}, {-0.4, 0.8, 0.5}}}, {{3}, {0.6, -1.1, 0.4}}},
        {"construct",
         {TensorValue::scalar(0.2), TensorValue::scalar(-0.5), TensorValue::scalar(1.3), TensorValue::scalar(0.7)},
         {{2, 2}, {0.6, -1.1, 0.4, 0.9}}},
        {"splat", {TensorValue::scalar(0.2)}, {{2, 3}, {0.6, -1.1, 0.4, 0.9, -0.3, 0.2}}},
        {"broadcast", {{{1, 3}, {0.2, -0.5, 1.3}}}, {{2, 3}, {0.6, -1.1, 0.4, 0.9, -0.3, 0.2}}},
        {"matmul_vv", {{{3}, {0.2, -0.5, 1.3}}, {{3}, {-0.4, 0.8, 0.5}}}, TensorValue::scalar(0.7)},
        {"matmul_vm", {{{3}, {0.2, -0.5, 1.3}}, {{3, 2}, {-0.4, 0.8, 0.5, -0.2, 1.1, 0.3}}}, {{2}, {0.7, -0.6}}},
        {"matmul_mv", {{{2, 3}, {0.2, -0.5, 1.3, 0.6, -0.9, 0.4}}, {{3}, {-0.4, 0.8, 0.5}}}, {{2}, {0.7, -0.6}}},
        {"matmul_mm",
         {{{2, 3}, {0.2, -0.5, 1.3, 0.6, -0.9, 0.4}}, {{3, 2}, {-0.4, 0.8, 0.5, -0.2, 1.1, 0.3}}},
         {{2, 2}, {0.7, -0.6, 0.2, 1.4}}},
        {"matmul_batched",
         {{{2, 2, 3}, {0.2, -0.5, 1.3, 0.6, -0.9, 0.4, -0.7, 0.1, 0.8, 1.2, -0.3, 0.5}},
          {{1, 3, 2}, {-0.4, 0.8, 0.5, -0.2, 1.1, 0.3}}},
         {{2, 2, 2}, {0.7, -0.6, 0.2, 1.4, -0.3, 0.9, 0.5, -0.8}}},
        {"matmul_zero", {{{2, 0}, {}}, {{0, 2}, {}}}, {{2, 2}, {0.7, -0.6, 0.2, 1.4}}},
    };
    for (const Case &testCase : cases) {
        func::FuncOp function = module->lookupSymbol<func::FuncOp>(testCase.function);
        ASSERT_TRUE(function) << testCase.function;
        expectTensorVjpMatchesFiniteDifference(function, testCase.arguments, testCase.seed);
    }
    EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(VernonAutodiffRulesTest, NormDecompositionVjpMatchesFiniteDifference) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @norm(%x: tensor<3xf32>) -> f32 {
    %squared = "vernon.intrinsic"(%x, %x) {name = "dot"}
        : (tensor<3xf32>, tensor<3xf32>) -> f32
    %result = math.sqrt %squared : f32
    func.return %result : f32
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("norm");
    IntrinsicOp dot;
    math::SqrtOp squareRoot;
    function.walk([&](IntrinsicOp operation) { dot = operation; });
    function.walk([&](math::SqrtOp operation) { squareRoot = operation; });
    ASSERT_TRUE(dot);
    ASSERT_TRUE(squareRoot);

    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    auto returnOp = cast<func::ReturnOp>(function.front().getTerminator());
    OpBuilder builder(returnOp);
    Value seed = arith::ConstantOp::create(builder, squareRoot.getLoc(), builder.getF32FloatAttr(1.0));
    FailureOr<SmallVector<Value>> squareRootContribution = registry.lookup(squareRoot)
                                                               ->buildVjp(squareRoot, AutodiffVjpBuildContext{
                                                                                          builder,
                                                                                          squareRoot.getLoc(),
                                                                                          squareRoot->getOperands(),
                                                                                          squareRoot->getResults(),
                                                                                          ValueRange(seed),
                                                                                      });
    ASSERT_TRUE(succeeded(squareRootContribution));
    ASSERT_EQ(squareRootContribution->size(), 1u);
    FailureOr<SmallVector<Value>> dotContributions =
        registry.lookup(dot)->buildVjp(dot, AutodiffVjpBuildContext{
                                                builder,
                                                dot.getLoc(),
                                                dot->getOperands(),
                                                dot->getResults(),
                                                ValueRange(squareRootContribution->front()),
                                            });
    ASSERT_TRUE(succeeded(dotContributions));
    ASSERT_EQ(dotContributions->size(), 2u);
    Value gradient = arith::AddFOp::create(builder, dot.getLoc(), dotContributions->front(), dotContributions->back());

    TensorValue input{{3}, {0.3, -0.7, 1.2}};
    DenseMap<Value, TensorValue> cache;
    TensorValue actual = evaluateTensor(gradient, ArrayRef<TensorValue>(input), cache);
    double norm = std::sqrt(0.3 * 0.3 + 0.7 * 0.7 + 1.2 * 1.2);
    ASSERT_EQ(actual.elements.size(), input.elements.size());
    for (auto [emitted, primal] : llvm::zip_equal(actual.elements, input.elements))
        EXPECT_NEAR(emitted, primal / norm, 2.0e-5);
    EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(VernonAutodiffRulesTest, TensorRulesPromoteF16AndRejectInvalidShapes) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @dot_f16(%x: tensor<3xf16>, %y: tensor<3xf16>) -> f16 {
    %result = "vernon.intrinsic"(%x, %y) {name = "dot"}
        : (tensor<3xf16>, tensor<3xf16>) -> f16
    func.return %result : f16
  }
  func.func @bad_dot(%x: tensor<2xf32>, %y: tensor<3xf32>) -> f32 {
    %result = "vernon.intrinsic"(%x, %y) {name = "dot"}
        : (tensor<2xf32>, tensor<3xf32>) -> f32
    func.return %result : f32
  }
  func.func @bad_cross(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "cross"}
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    func.return %result : tensor<4xf32>
  }
  func.func @bad_matmul(%x: tensor<2x3xf32>, %y: tensor<4x2xf32>) -> tensor<2x2xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "matmul"}
        : (tensor<2x3xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
  func.func @bad_normalize(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %result = "vernon.intrinsic"(%x) {name = "normalize"}
        : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
  func.func @bad_reflect(%x: tensor<2xf32>, %n: tensor<3xf32>) -> tensor<2xf32> {
    %result = "vernon.intrinsic"(%x, %n) {name = "reflect"}
        : (tensor<2xf32>, tensor<3xf32>) -> tensor<2xf32>
    func.return %result : tensor<2xf32>
  }
  func.func @bad_construct(%x: f32, %y: f32) -> tensor<3xf32> {
    %result = "vernon.intrinsic"(%x, %y) {name = "construct"}
        : (f32, f32) -> tensor<3xf32>
    func.return %result : tensor<3xf32>
  }
  func.func @dynamic_broadcast(%x: tensor<?xf32>) -> tensor<?xf32> {
    %result = "vernon.intrinsic"(%x) {name = "broadcast"}
        : (tensor<?xf32>) -> tensor<?xf32>
    func.return %result : tensor<?xf32>
  }
  func.func @bad_type(%x: tensor<3xi32>, %y: tensor<3xi32>) -> i32 {
    %result = "vernon.intrinsic"(%x, %y) {name = "dot"}
        : (tensor<3xi32>, tensor<3xi32>) -> i32
    func.return %result : i32
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();

    func::FuncOp promotedFunction = module->lookupSymbol<func::FuncOp>("dot_f16");
    IntrinsicOp promoted;
    promotedFunction.walk([&](IntrinsicOp operation) { promoted = operation; });
    ASSERT_TRUE(promoted);
    auto returnOp = cast<func::ReturnOp>(promotedFunction.front().getTerminator());
    OpBuilder builder(returnOp);
    Value seed = arith::ConstantOp::create(builder, promoted.getLoc(), builder.getF32FloatAttr(0.7));
    FailureOr<SmallVector<Value>> contributions = registry.lookup(promoted)->buildVjp(
        promoted, AutodiffVjpBuildContext{builder, promoted.getLoc(), promoted->getOperands(), promoted->getResults(),
                                          ValueRange(seed)});
    ASSERT_TRUE(succeeded(contributions));
    ASSERT_EQ(contributions->size(), 2u);
    EXPECT_EQ((*contributions)[0].getType(), RankedTensorType::get({3}, builder.getF32Type()));
    EXPECT_EQ((*contributions)[1].getType(), RankedTensorType::get({3}, builder.getF32Type()));
    unsigned extensionCount = 0;
    promotedFunction.walk([&](arith::ExtFOp) { ++extensionCount; });
    EXPECT_EQ(extensionCount, 2u);

    std::string diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        diagnostics.push_back('\n');
        return success();
    });
    for (StringRef functionName : {"bad_dot", "bad_cross", "bad_matmul", "bad_normalize", "bad_reflect",
                                   "bad_construct", "dynamic_broadcast", "bad_type"}) {
        func::FuncOp function = module->lookupSymbol<func::FuncOp>(functionName);
        IntrinsicOp intrinsic;
        function.walk([&](IntrinsicOp operation) { intrinsic = operation; });
        ASSERT_TRUE(intrinsic) << functionName.str();
        const DifferentiationRule *rule = registry.lookup(intrinsic);
        ASSERT_NE(rule, nullptr);
        if (failed(rule->verifyCompatibility(intrinsic)))
            continue;
        auto invalidReturn = cast<func::ReturnOp>(function.front().getTerminator());
        OpBuilder invalidBuilder(invalidReturn);
        Type resultType = intrinsic.getResult().getType();
        SmallVector<int64_t> shape = valueShape(resultType);
        TensorValue zeroSeed{shape, SmallVector<double>(elementCount(shape), 0.0)};
        Value invalidSeed = makeTensorConstant(invalidBuilder, intrinsic.getLoc(), resultType, zeroSeed);
        EXPECT_TRUE(failed(rule->buildVjp(
            intrinsic, AutodiffVjpBuildContext{invalidBuilder, intrinsic.getLoc(), intrinsic->getOperands(),
                                               intrinsic->getResults(), ValueRange(invalidSeed)})))
            << functionName.str();
    }
    EXPECT_NE(diagnostics.find("incompatible operand shapes"), std::string::npos);
    EXPECT_NE(diagnostics.find("static Tensor shapes"), std::string::npos);
    EXPECT_NE(diagnostics.find("floating-point scalar or ranked Tensor values"), std::string::npos);
}

TEST_F(VernonAutodiffRulesTest, RejectsMissingActiveRuleAndDuplicateRegistration) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @missing(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %one = arith.constant 1.0 : f32
    %result = arith.addf %x, %one : f32
    func.return %result : f32
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry emptyRegistry;
    std::string message;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(message);
        diagnostic.print(stream);
        return success();
    });
    EXPECT_TRUE(failed(analyzeAutodiffFunction(module->lookupSymbol<func::FuncOp>("missing"), {"x"}, emptyRegistry)));
    EXPECT_NE(message.find("no registered differentiation rule"), std::string::npos);

    VernonAutodiffRuleRegistry extensionRegistry;
    EXPECT_TRUE(succeeded(extensionRegistry.registerRule(DifferentiationRule(
        arith::AddFOp::getOperationName().str(), 2, 1, {},
        [](Operation *, const AutodiffVjpBuildContext &buildContext, SmallVectorImpl<Value> &results) {
            results.append(2, buildContext.resultCotangents[0]);
            return success();
        }))));
    EXPECT_TRUE(
        succeeded(analyzeAutodiffFunction(module->lookupSymbol<func::FuncOp>("missing"), {"x"}, extensionRegistry)));

    VernonAutodiffRuleRegistry registry;
    DifferentiationRule first("test.rule", 1, 1, {}, {});
    DifferentiationRule duplicate("test.rule", 1, 1, {}, {});
    EXPECT_TRUE(succeeded(registry.registerRule(std::move(first))));
    EXPECT_TRUE(failed(registry.registerRule(std::move(duplicate))));
}

TEST_F(VernonAutodiffRulesTest, DefersCompatibilityDiagnosticsUntilOperationIsActive) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @inactive(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %one = arith.constant 1.0 : f32
    %unused = arith.addf %one, %one : f32
    func.return %x : f32
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry;
    ASSERT_TRUE(
        succeeded(registry.registerRule(DifferentiationRule(arith::AddFOp::getOperationName().str(), 1, 1, {}, {}))));
    EXPECT_TRUE(succeeded(analyzeAutodiffFunction(module->lookupSymbol<func::FuncOp>("inactive"), {"x"}, registry)));
}

TEST_F(VernonAutodiffRulesTest, RejectsDynamicTensorRulesBeforeVjpConstruction) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @dynamic(%x: tensor<?xf32>, %y: tensor<?xf32>) -> tensor<?xf32> {
    %result = arith.addf %x, %y : tensor<?xf32>
    func.return %result : tensor<?xf32>
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    arith::AddFOp add;
    module->walk([&](arith::AddFOp operation) { add = operation; });
    ASSERT_TRUE(add);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    ASSERT_NE(registry.lookup(add), nullptr);
    ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(registry.lookup(add)->verifyCompatibility(add)));
}

TEST_F(VernonAutodiffRulesTest, RegistrationCannotHideStorageEffects) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @storage(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %buffer: !vernon.tensor_view<f32, [1], "write", "workgroup">,
      %index: index) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    "vernon.store"(%x, %buffer, %index)
        : (f32, !vernon.tensor_view<f32, [1], "write", "workgroup">, index) -> ()
    func.return %x : f32
  }
}
)mlir",
                                                               ParserConfig(&context));
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry;
    ASSERT_TRUE(succeeded(registry.registerRule(DifferentiationRule(StoreOp::getOperationName().str(), 3, 0, {}, {}, {},
                                                                    [](Operation *) { return success(); }))));
    FailureOr<VernonAutodiffAnalysisResult> analysis =
        analyzeAutodiffFunction(module->lookupSymbol<func::FuncOp>("storage"), {"x"}, registry);
    ASSERT_TRUE(succeeded(analysis));
    auto store = llvm::find_if(analysis->getOperations(),
                               [](const AutodiffOperationActivity &item) { return isa<StoreOp>(item.operation); });
    ASSERT_NE(store, analysis->getOperations().end());
    EXPECT_EQ(store->effect, AutodiffEffectKind::StorageWrite);
}

} // namespace
} // namespace mlir::vernon
