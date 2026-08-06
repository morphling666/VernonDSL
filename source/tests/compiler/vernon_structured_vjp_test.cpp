#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStructuredVjp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

namespace mlir::vernon {
namespace {

class VernonStructuredVjpTest : public testing::Test {
protected:
    VernonStructuredVjpTest() {
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<math::MathDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        context.getOrLoadDialect<VernonDialect>();
    }

    OwningOpRef<ModuleOp> parse(StringRef operation, StringRef arguments = "%x, %y") {
        std::string source = (Twine(R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %y: f32 {vernon.source_name = "y", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %result = )mlir") + operation +
                              " " + arguments + R"mlir( : f32
    func.return %result : f32
  }
}
)mlir")
                                 .str();
        return parseSourceString<ModuleOp>(source, ParserConfig(&context));
    }

    MLIRContext context;
};

TEST_F(VernonStructuredVjpTest, GeneratesProfilesForEveryScalarRule) {
    struct Case {
        const char *operation;
        const char *arguments;
        unsigned tapeLeaves;
    };
    const Case cases[] = {
        {"arith.addf", "%x, %y", 0}, {"arith.subf", "%x, %y", 0}, {"arith.mulf", "%x, %y", 2},
        {"arith.divf", "%x, %y", 2}, {"arith.negf", "%x", 0},     {"math.sin", "%x", 1},
        {"math.cos", "%x", 1},       {"math.exp", "%x", 1},       {"math.log", "%x", 1},
        {"math.sqrt", "%x", 1},      {"math.acos", "%x", 1},      {"math.atan2", "%x, %y", 2},
        {"math.absf", "%x", 1},
    };
    for (auto [index, testCase] : llvm::enumerate(cases)) {
        OwningOpRef<ModuleOp> module = parse(testCase.operation, testCase.arguments);
        ASSERT_TRUE(module) << testCase.operation;
        std::string suffix = std::to_string(index);
        FailureOr<StructuredVjpResult> result =
            buildStructuredScalarVjp(module->lookupSymbol<func::FuncOp>("primal"),
                                     StructuredVjpOptions{{"x", "y"}, "forward" + suffix, "backward" + suffix});
        ASSERT_TRUE(succeeded(result)) << testCase.operation;
        EXPECT_TRUE(succeeded(verify(*module))) << testCase.operation;
        EXPECT_EQ(result->forward.getNumArguments(), 2u);
        EXPECT_EQ(result->forward.getNumResults(), 1u);
        EXPECT_TRUE(isa<TupleType>(result->forward.getResultTypes().front()));
        EXPECT_EQ(result->backward.getNumArguments(), 2u);
        EXPECT_EQ(result->backward.getNumResults(), 1u);
        EXPECT_TRUE(isa<TupleType>(result->backward.getResultTypes().front()));
        ASSERT_EQ(result->derivativeRules.size(), 1u);
        EXPECT_EQ(result->derivativeRules.front(), testCase.operation);
        auto forwardResult = cast<TupleType>(result->forward.getResultTypes().front());
        EXPECT_EQ(cast<TupleType>(forwardResult.getType(1)).size(), testCase.tapeLeaves);
    }

    OwningOpRef<ModuleOp> powerModule = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %y: f32 {vernon.source_name = "y", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %result = "vernon.intrinsic"(%x, %y) {name = "pow"} : (f32, f32) -> f32
    func.return %result : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(powerModule);
    FailureOr<StructuredVjpResult> power =
        buildStructuredScalarVjp(powerModule->lookupSymbol<func::FuncOp>("primal"),
                                 StructuredVjpOptions{{"x", "y"}, "power_forward", "power_backward"});
    ASSERT_TRUE(succeeded(power));
    EXPECT_EQ(power->derivativeRules, SmallVector<std::string>({"vernon.intrinsic.pow"}));
    auto powerForwardResult = cast<TupleType>(power->forward.getResultTypes().front());
    EXPECT_EQ(cast<TupleType>(powerForwardResult.getType(1)).size(), 3u);
}

TEST_F(VernonStructuredVjpTest, DeduplicatesTapeAndAccumulatesSharedUseAdjoints) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %left = arith.mulf %x, %x : f32
    %right = arith.mulf %x, %x : f32
    %result = arith.addf %left, %right : f32
    func.return %result : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredScalarVjp(
        module->lookupSymbol<func::FuncOp>("primal"), StructuredVjpOptions{{"x"}, "shared_forward", "shared_backward"});
    ASSERT_TRUE(succeeded(result));
    auto forwardResult = cast<TupleType>(result->forward.getResultTypes().front());
    EXPECT_EQ(cast<TupleType>(forwardResult.getType(1)).size(), 1u);
    EXPECT_EQ(result->derivativeRules, SmallVector<std::string>({"arith.addf", "arith.mulf"}));
    unsigned additions = 0;
    result->backward.walk([&](arith::AddFOp) { ++additions; });
    EXPECT_GE(additions, 3u);
}

TEST_F(VernonStructuredVjpTest, RejectsAmbiguousOptionsWithoutMutatingPrimal) {
    for (const StructuredVjpOptions &options :
         {StructuredVjpOptions{{"x"}, "derivative", "derivative"},
          StructuredVjpOptions{{"x", "x"}, "forward", "backward"}, StructuredVjpOptions{{""}, "forward", "backward"}}) {
        OwningOpRef<ModuleOp> module = parse("arith.addf");
        ASSERT_TRUE(module);
        func::FuncOp primal = module->lookupSymbol<func::FuncOp>("primal");
        EXPECT_TRUE(failed(buildStructuredScalarVjp(primal, options)));
        EXPECT_TRUE(primal->hasAttr("vernon.entry"));
        EXPECT_EQ(llvm::range_size(module->getOps<func::FuncOp>()), 1u);
    }
}

TEST_F(VernonStructuredVjpTest, UsesPlannedTapeAndPromotedAdjoints) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f16 {vernon.source_name = "x", vernon.dtype = "f16", vernon.abi_leaf_dtypes = ["f16"]},
      %y: f16 {vernon.source_name = "y", vernon.dtype = "f16", vernon.abi_leaf_dtypes = ["f16"]})
      -> (f16 {vernon.dtype = "f16", vernon.abi_leaf_dtypes = ["f16"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %result = arith.mulf %x, %y : f16
    func.return %result : f16
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredScalarVjp(module->lookupSymbol<func::FuncOp>("primal"),
                                 StructuredVjpOptions{{"x"}, "structured_forward", "structured_backward"});
    ASSERT_TRUE(succeeded(result));
    auto forwardResult = cast<TupleType>(result->forward.getResultTypes().front());
    auto tape = cast<TupleType>(forwardResult.getType(1));
    EXPECT_EQ(tape.size(), 2u);
    EXPECT_GT(result->tapeBytes, 0u);
    EXPECT_TRUE(result->backward.getResultTypes().front().isF32());
    auto cotangentDtypes = cast<ArrayAttr>(result->backward.getArgAttr(1, "vernon.abi_leaf_dtypes"));
    ASSERT_EQ(cotangentDtypes.size(), 1u);
    EXPECT_EQ(cast<StringAttr>(cotangentDtypes[0]).getValue(), "f32");
    auto gradientDtypes = cast<ArrayAttr>(result->backward.getResultAttr(0, "vernon.abi_leaf_dtypes"));
    ASSERT_EQ(gradientDtypes.size(), 1u);
    EXPECT_EQ(cast<StringAttr>(gradientDtypes[0]).getValue(), "f32");
}

TEST_F(VernonStructuredVjpTest, RejectsStructuredControlFlowWithoutFallback) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(%x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
                    %condition: i1 {vernon.source_name = "condition", vernon.abi_leaf_dtypes = ["bool"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %result = scf.if %condition -> (f32) {
      scf.yield %x : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    func.return %result : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    EXPECT_TRUE(
        failed(buildStructuredScalarVjp(module->lookupSymbol<func::FuncOp>("primal"),
                                        StructuredVjpOptions{{"x"}, "structured_forward", "structured_backward"})));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("structured_forward"));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("structured_backward"));
}

TEST_F(VernonStructuredVjpTest, RejectsActiveMultiResultOperationWithoutMutation) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func private @pair(f32) -> (f32, f32)
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %first, %second = func.call @pair(%x) : (f32) -> (f32, f32)
    func.return %first : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    func::FuncOp primal = module->lookupSymbol<func::FuncOp>("primal");
    EXPECT_TRUE(failed(
        buildStructuredScalarVjp(primal, StructuredVjpOptions{{"x"}, "structured_forward", "structured_backward"})));
    EXPECT_TRUE(primal->hasAttr("vernon.entry"));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("structured_forward"));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("structured_backward"));
}

} // namespace
} // namespace mlir::vernon
