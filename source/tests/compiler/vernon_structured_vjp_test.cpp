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
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %result = )mlir") + operation +
                              " " + arguments + R"mlir( : f32
    "vernon.store"(%result, %loss, %gid)
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
        unsigned backwardBarriers = 0;
        result->backward.walk([&](BarrierOp) { ++backwardBarriers; });
        EXPECT_EQ(backwardBarriers, 0u);
    }
}

TEST_F(VernonStructuredVjpTest, ReversesStorageScratchWritesIntoSelectedInput) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">
          {vernon.source_name = "scratch", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1, 1, 1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %twice = arith.addf %x, %x : f32
    "vernon.store"(%twice, %scratch, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
           index, index, index) -> ()
    %stored = "vernon.load"(%scratch, %gx, %gy, %gz)
        : (!vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
           index, index, index) -> f32
    %squared = arith.mulf %stored, %stored : f32
    "vernon.store"(%squared, %loss, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [1, 1, 1], "write", "device">,
           index, index, index) -> ()
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
    unsigned lanePrivateBuffers = 0;
    result->backward.walk([&](Operation *operation) {
        StringRef name = operation->getName().getStringRef();
        scatterAdds += name == "vernon.ad.adjoint_buffer.scatter_add";
        takeAndClears += name == "vernon.ad.adjoint_buffer.take_and_clear";
        if (auto create = dyn_cast<AdAdjointBufferCreateOp>(operation))
            lanePrivateBuffers += create.getOwnershipAttr() && create.getOwnershipAttr().getValue() == "lane_private";
    });
    EXPECT_GE(scatterAdds, 1u);
    EXPECT_GE(takeAndClears, 1u);
    EXPECT_GE(lanePrivateBuffers, 1u);
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
      %values: !vernon.tensor_view<!vernon.struct<"Pair">, [1, 1, 1], "read", "device">
          {vernon.source_name = "values", vernon.abi_leaf_dtypes = ["f32", "f32"]},
      %scratch: !vernon.tensor_view<!vernon.struct<"Pair">, [1, 1, 1], "read_write", "device">
          {vernon.source_name = "scratch", vernon.abi_leaf_dtypes = ["f32", "f32"]},
      %loss: !vernon.tensor_view<f32, [1, 1, 1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %input = "vernon.load"(%values, %gx, %gy, %gz)
        : (!vernon.tensor_view<!vernon.struct<"Pair">, [1, 1, 1], "read", "device">,
           index, index, index) ->
          !vernon.struct<"Pair">
    "vernon.store"(%input, %scratch, %gx, %gy, %gz)
        : (!vernon.struct<"Pair">,
           !vernon.tensor_view<!vernon.struct<"Pair">, [1, 1, 1], "read_write", "device">,
           index, index, index) -> ()
    %stored = "vernon.load"(%scratch, %gx, %gy, %gz)
        : (!vernon.tensor_view<!vernon.struct<"Pair">, [1, 1, 1], "read_write", "device">,
           index, index, index) ->
          !vernon.struct<"Pair">
    %left = "vernon.struct_get"(%stored) {field = "left", index = 0 : i64}
        : (!vernon.struct<"Pair">) -> f32
    %right = "vernon.struct_get"(%stored) {field = "right", index = 1 : i64}
        : (!vernon.struct<"Pair">) -> f32
    %left_squared = arith.mulf %left, %left : f32
    %right_squared = arith.mulf %right, %right : f32
    %sum = arith.addf %left_squared, %right_squared : f32
    "vernon.store"(%sum, %loss, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [1, 1, 1], "write", "device">,
           index, index, index) -> ()
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
    EXPECT_EQ(result->backward.getNumResults(), 0u);
    unsigned internalScatterAdds = 0;
    unsigned externalScatterAdds = 0;
    unsigned takeAndClears = 0;
    result->backward.walk([&](Operation *operation) {
        StringRef name = operation->getName().getStringRef();
        internalScatterAdds += name == "vernon.ad.adjoint_buffer.scatter_add";
        externalScatterAdds += name == "vernon.scatter_add";
        takeAndClears += name == "vernon.ad.adjoint_buffer.take_and_clear";
    });
    EXPECT_GE(internalScatterAdds, 2u);
    EXPECT_EQ(externalScatterAdds, 2u);
    EXPECT_GE(takeAndClears, 2u);
}

TEST_F(VernonStructuredVjpTest, BuildsDynamicStorageGradientAsWritableTensorViewArgument) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %values: !vernon.tensor_view<f32, [-1], "read", "device">
          {vernon.source_name = "values", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %value = "vernon.load"(%values, %gid)
        : (!vernon.tensor_view<f32, [-1], "read", "device">, index) -> f32
    %squared = arith.mulf %value, %value : f32
    "vernon.store"(%squared, %loss, %gid)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"values"}, "dynamic_forward", "dynamic_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(result->backward.getNumResults(), 0u);
    ASSERT_EQ(result->backward.getNumArguments(), 5u);
    auto gradient = dyn_cast<TensorViewType>(result->backward.getArgumentTypes().back());
    ASSERT_TRUE(gradient);
    EXPECT_EQ(gradient.getShape(), ArrayRef<int64_t>({-1}));
    EXPECT_EQ(gradient.getAccess(), "write");
    unsigned dynamicBuffers = 0;
    result->backward.walk([&](AdAdjointBufferCreateOp create) {
        dynamicBuffers += llvm::is_contained(create.getBuffer().getType().getShape(), int64_t{-1});
    });
    EXPECT_EQ(dynamicBuffers, 0u);
}

TEST_F(VernonStructuredVjpTest, BuildsDynamicForStorageObjective) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %count: i32 {vernon.source_name = "count", vernon.dtype = "i32", vernon.abi_leaf_dtypes = ["i32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %upper = arith.index_cast %count : i32 to index
    %result = scf.for %index = %zero to %upper step %one
        iter_args(%value = %x) -> (f32) {
      %next = arith.addf %value, %x : f32
      scf.yield %next : f32
    }
    "vernon.store"(%result, %loss, %gid)
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
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
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
    "vernon.store"(%result, %loss, %gid)
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

TEST_F(VernonStructuredVjpTest, ReversesOutputOnlyAdditiveRmwFromExternalCotangent) {
    const StringRef effects[] = {
        R"mlir("vernon.reduce_sum"(%x, %loss, %gid) {deterministic = false} : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ())mlir",
        R"mlir("vernon.scatter_add"(%x, %loss, %gid) {deterministic = false} : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ())mlir",
        R"mlir(%old = "vernon.atomic"(%loss, %gid, %x) {atomic_kind = "add", ordering = "relaxed"} : (!vernon.tensor_view<f32, [1], "write", "device">, index, f32) -> f32)mlir",
    };
    for (auto [index, effect] : llvm::enumerate(effects)) {
        std::string source = (Twine(R"mlir(module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    )mlir") + effect + R"mlir(
    func.return
  }
})mlir")
                                 .str();
        OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, ParserConfig(&context));
        ASSERT_TRUE(module);
        std::string suffix = std::to_string(index);
        FailureOr<StructuredVjpResult> result =
            buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                               StructuredVjpOptions{{"x"}, "rmw_forward" + suffix, "rmw_backward" + suffix, {"loss"}});
        ASSERT_TRUE(succeeded(result));
        EXPECT_TRUE(succeeded(verify(*module)));
        unsigned buffers = 0;
        unsigned peeks = 0;
        unsigned externalLoads = 0;
        result->backward.walk([&](Operation *operation) {
            buffers += isa<AdAdjointBufferCreateOp>(operation);
            peeks += isa<AdAdjointPeekOp>(operation);
            externalLoads += isa<LoadOp>(operation);
        });
        EXPECT_EQ(buffers, 0u);
        EXPECT_EQ(peeks, 0u);
        EXPECT_GE(externalLoads, 1u);
    }
}

TEST_F(VernonStructuredVjpTest, PeeksInternalAdjointForMixedAdditiveStorage) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">
          {vernon.source_name = "scratch", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1, 1, 1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    "vernon.scatter_add"(%x, %scratch, %gx, %gy, %gz) {deterministic = false}
        : (f32, !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
           index, index, index) -> ()
    %value = "vernon.load"(%scratch, %gx, %gy, %gz)
        : (!vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
           index, index, index) -> f32
    "vernon.store"(%value, %loss, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [1, 1, 1], "write", "device">,
           index, index, index) -> ()
    func.return
  }
})mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x"}, "mixed_forward", "mixed_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned peeks = 0;
    unsigned takeAndClears = 0;
    result->backward.walk([&](Operation *operation) {
        peeks += isa<AdAdjointPeekOp>(operation);
        takeAndClears += isa<AdAdjointTakeAndClearOp>(operation);
    });
    EXPECT_EQ(peeks, 1u);
    EXPECT_EQ(takeAndClears, 0u);
}

TEST_F(VernonStructuredVjpTest, ReversesLaneExclusiveAtomicOldValue) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(module {
  func.func @primal(
      %state: !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">
          {vernon.source_name = "state", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1, 1, 1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx32 : i32 to index
    %gy = arith.index_cast %gy32 : i32 to index
    %gz = arith.index_cast %gz32 : i32 to index
    %value = arith.constant 2.0 : f32
    %old = "vernon.atomic"(%state, %gx, %gy, %gz, %value)
        {atomic_kind = "add", ordering = "relaxed"}
        : (!vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
           index, index, index, f32) -> f32
    "vernon.store"(%old, %loss, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [1, 1, 1], "write", "device">,
           index, index, index) -> ()
    func.return
  }
})mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"state"}, "atomic_forward", "atomic_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned internalScatterAdds = 0;
    result->backward.walk([&](AdAdjointScatterAddOp) { ++internalScatterAdds; });
    EXPECT_GE(internalScatterAdds, 1u);
}

TEST_F(VernonStructuredVjpTest, ReversesWorkgroupAtomicAddWithSharedAdjointSynchronization) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [4], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %shared = "vernon.workgroup_alloc"()
        : () -> !vernon.tensor_view<f32, [1], "read_write", "workgroup">
    %old = "vernon.atomic"(%shared, %zero, %x)
        {atomic_kind = "add", ordering = "relaxed"}
        : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index, f32) -> f32
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %sum = "vernon.load"(%shared, %zero)
        : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> f32
    "vernon.store"(%sum, %loss, %gid)
        : (f32, !vernon.tensor_view<f32, [4], "write", "device">, index) -> ()
    func.return
  }
})mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"),
        StructuredVjpOptions{{"x"}, "workgroup_atomic_forward", "workgroup_atomic_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned sharedBuffers = 0;
    unsigned peeks = 0;
    unsigned barriers = 0;
    result->backward.walk([&](Operation *operation) {
        if (auto create = dyn_cast<AdAdjointBufferCreateOp>(operation))
            sharedBuffers += create.getOwnershipAttr() && create.getOwnershipAttr().getValue() == "workgroup_shared";
        peeks += isa<AdAdjointPeekOp>(operation);
        barriers += isa<BarrierOp>(operation);
    });
    EXPECT_GE(sharedBuffers, 1u);
    EXPECT_GE(peeks, 1u);
    EXPECT_EQ(barriers, 3u);
}

} // namespace
} // namespace mlir::vernon
