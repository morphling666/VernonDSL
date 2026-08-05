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
