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

    OwningOpRef<ModuleOp> parseStorageObjective(StringRef operation, StringRef arguments = "%x, %y") {
        std::string source = (Twine(R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %y: f32 {vernon.source_name = "y", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %index = arith.constant 0 : index
    %result = )mlir") + operation +
                              " " + arguments + R"mlir( : f32
    "vernon.store"(%result, %loss, %index)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir")
                                 .str();
        return parseSourceString<ModuleOp>(source, ParserConfig(&context));
    }

    MLIRContext context;
};

TEST_F(VernonStructuredVjpTest, GeneratesStorageObjectiveProfilesForScalarRules) {
    const std::pair<StringRef, StringRef> cases[] = {
        {"arith.addf", "%x, %y"}, {"arith.subf", "%x, %y"}, {"arith.mulf", "%x, %y"}, {"arith.divf", "%x, %y"},
        {"arith.negf", "%x"},     {"math.sin", "%x"},       {"math.cos", "%x"},       {"math.exp", "%x"},
        {"math.log", "%x"},       {"math.sqrt", "%x"},      {"math.acos", "%x"},      {"math.atan2", "%x, %y"},
        {"math.absf", "%x"},
    };
    for (auto [index, testCase] : llvm::enumerate(cases)) {
        OwningOpRef<ModuleOp> module = parseStorageObjective(testCase.first, testCase.second);
        ASSERT_TRUE(module) << testCase.first.str();
        std::string suffix = std::to_string(index);
        FailureOr<StructuredVjpResult> result =
            buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                               StructuredVjpOptions{{"x", "y"}, "forward" + suffix, "backward" + suffix, {"loss"}});
        ASSERT_TRUE(succeeded(result)) << testCase.first.str();
        EXPECT_TRUE(succeeded(verify(*module))) << testCase.first.str();
        EXPECT_EQ(result->forward.getNumResults(), 1u);
        EXPECT_EQ(result->backward.getNumResults(), 1u);
        ASSERT_EQ(result->derivativeRules.size(), 1u);
        EXPECT_EQ(result->derivativeRules.front(), testCase.first.str());
    }
}

TEST_F(VernonStructuredVjpTest, ReversesStorageScratchWritesIntoSelectedInput) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [1], "read_write", "device">
          {vernon.source_name = "scratch", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %index = arith.constant 0 : index
    %twice = arith.addf %x, %x : f32
    "vernon.store"(%twice, %scratch, %index)
        : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    %stored = "vernon.load"(%scratch, %index)
        : (!vernon.tensor_view<f32, [1], "read_write", "device">, index) -> f32
    %squared = arith.mulf %stored, %stored : f32
    "vernon.store"(%squared, %loss, %index)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x"}, "storage_forward", "storage_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned scatterAdds = 0;
    unsigned takeAndClears = 0;
    result->backward.walk([&](Operation *operation) {
        StringRef name = operation->getName().getStringRef();
        scatterAdds += name == "vernon.ad.adjoint_buffer.scatter_add";
        takeAndClears += name == "vernon.ad.adjoint_buffer.take_and_clear";
    });
    EXPECT_GE(scatterAdds, 1u);
    EXPECT_GE(takeAndClears, 1u);
}

TEST_F(VernonStructuredVjpTest, ReversesEveryAggregateStorageAbiLeaf) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:f32", "right:f32"],
    abi_leaf_dtypes = ["f32", "f32"]
  } : () -> ()
  func.func @primal(
      %values: !vernon.tensor_view<!vernon.struct<"Pair">, [1], "read", "device">
          {vernon.source_name = "values", vernon.abi_leaf_dtypes = ["f32", "f32"]},
      %scratch: !vernon.tensor_view<!vernon.struct<"Pair">, [1], "read_write", "device">
          {vernon.source_name = "scratch", vernon.abi_leaf_dtypes = ["f32", "f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %index = arith.constant 0 : index
    %input = "vernon.load"(%values, %index)
        : (!vernon.tensor_view<!vernon.struct<"Pair">, [1], "read", "device">, index) ->
          !vernon.struct<"Pair">
    "vernon.store"(%input, %scratch, %index)
        : (!vernon.struct<"Pair">,
           !vernon.tensor_view<!vernon.struct<"Pair">, [1], "read_write", "device">,
           index) -> ()
    %stored = "vernon.load"(%scratch, %index)
        : (!vernon.tensor_view<!vernon.struct<"Pair">, [1], "read_write", "device">, index) ->
          !vernon.struct<"Pair">
    %left = "vernon.struct_get"(%stored) {field = "left", index = 0 : i64}
        : (!vernon.struct<"Pair">) -> f32
    %right = "vernon.struct_get"(%stored) {field = "right", index = 1 : i64}
        : (!vernon.struct<"Pair">) -> f32
    %left_squared = arith.mulf %left, %left : f32
    %right_squared = arith.mulf %right, %right : f32
    %sum = arith.addf %left_squared, %right_squared : f32
    "vernon.store"(%sum, %loss, %index)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"values"}, "aggregate_forward", "aggregate_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned scatterAdds = 0;
    unsigned takeAndClears = 0;
    result->backward.walk([&](Operation *operation) {
        StringRef name = operation->getName().getStringRef();
        scatterAdds += name == "vernon.ad.adjoint_buffer.scatter_add";
        takeAndClears += name == "vernon.ad.adjoint_buffer.take_and_clear";
    });
    EXPECT_GE(scatterAdds, 4u);
    EXPECT_GE(takeAndClears, 3u);
}

TEST_F(VernonStructuredVjpTest, BuildsDynamicForStorageObjective) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %count: i32 {vernon.source_name = "count", vernon.dtype = "i32", vernon.abi_leaf_dtypes = ["i32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %upper = arith.index_cast %count : i32 to index
    %result = scf.for %index = %zero to %upper step %one
        iter_args(%value = %x) -> (f32) {
      %next = arith.addf %value, %x : f32
      scf.yield %next : f32
    }
    %loss_index = arith.constant 0 : index
    "vernon.store"(%result, %loss, %loss_index)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x"}, "for_forward", "for_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_GT(result->tapeBytes, 0u);
    unsigned dynamicRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++dynamicRegions; });
    EXPECT_GE(dynamicRegions, 1u);
    unsigned sourceFors = 0;
    module->lookupSymbol<func::FuncOp>("primal").walk([&](scf::ForOp) { ++sourceFors; });
    EXPECT_EQ(sourceFors, 1u);
    EXPECT_EQ(llvm::range_size(module->getOps<func::FuncOp>()), 3u);
}

TEST_F(VernonStructuredVjpTest, BuildsDynamicWhileStorageObjective) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %count: i32 {vernon.source_name = "count", vernon.dtype = "i32", vernon.abi_leaf_dtypes = ["i32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %result, %final_index = scf.while (%value = %x, %index = %zero) : (f32, i32) -> (f32, i32) {
      %continue = arith.cmpi slt, %index, %count : i32
      scf.condition(%continue) %value, %index : f32, i32
    } do {
    ^bb0(%value: f32, %index: i32):
      %next = arith.addf %value, %x : f32
      %next_index = arith.addi %index, %one : i32
      scf.yield %next, %next_index : f32, i32
    }
    %loss_index = arith.constant 0 : index
    "vernon.store"(%result, %loss, %loss_index)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x"}, "loop_forward", "loop_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_GT(result->tapeBytes, 0u);
    unsigned dynamicRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++dynamicRegions; });
    EXPECT_GE(dynamicRegions, 1u);
}

TEST_F(VernonStructuredVjpTest, RejectsFunctionReturnComputeObjective) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    func.return %x : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    EXPECT_TRUE(failed(buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                                          StructuredVjpOptions{{"x"}, "forward", "backward", {"loss"}})));
}

} // namespace
} // namespace mlir::vernon
